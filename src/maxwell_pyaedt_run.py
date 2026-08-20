import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

from pyaedt import Desktop, Maxwell3d


AEDT_VERSION = "2023.1"
MM_TO_M = 1e-3
KG_PER_M3_CORE = 6400.0
KG_PER_M3_MAGNET = 7400.0
KG_PER_M3_COPPER = 8700.0
PENALTY_FITNESS = 1_000_000.0
DEFAULT_WCOIL_MM = float(os.getenv("WIRE_DIAMETER_MM", "0.05"))  # 线径，可通过环境变量配置
WA_FIXED = 2.0
MAX_GENOME_ATTEMPTS = 100_000
DECIMAL_PLACES = 2


def _env_bool(name: str, default: bool) -> bool:
    """读取环境变量布尔值：1/true/yes/on 为 True；0/false/no/off 为 False。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


TURNS_FLOOR_MODE = _env_bool("TURNS_FLOOR_MODE", False)  # n1 取整方式：True=向下取整，False=四舍五入（默认）


def _quantize(value: float, places: int = DECIMAL_PLACES) -> float:
    """将浮点数统一量化到指定小数位，默认两位。"""
    return round(float(value), places)


@dataclass
class MaxwellSimulationConfig:
    output_dir: str = "."
    calc_fld_path: str = "11.fld"
    bsat_fld_path: str = "Bsat_max.fld"
    bsat_ta_fld_path: str = "Bsat_ta.fld"
    bsat_tb_fld_path: str = "Bsat_tb.fld"
    lm: float = 5.0  # 永磁体长度
    ta: float = 0.5  # 上下轭厚度
    tb: float = 0.8  # 中间块高度
    tm: float = 0.5  # 中段厚度
    dg: float = 0.4  # 气隙高度
    wa: float = 2.0  # X 向扫掠长度

    @staticmethod
    def _mm(value: float) -> str:
        return f"{value:g}mm"

    @property
    def la(self) -> float:
        return self.lm + 2.0 * self.ta

    @property
    def ha(self) -> float:
        # 与 ActuatorDesignVariables.ha 保持一致：3*ta + 2*tm + 2*dg
        return 3 * self.ta + 2 * self.tm + 2 * self.dg

    @property
    def _dims(self):
        ta, tm, dg, tb, la = self.ta, self.tm, self.dg, self.tb, self.la
        y_outer = la
        y_inner = ta
        z_ta = ta
        z_ta_tm_dg = ta + tm + dg
        z_ta_tm_dg_tb = z_ta_tm_dg + tb
        z_ta_tm_dg_tb_tm_dg = z_ta_tm_dg_tb + tm + dg
        z_total = z_ta_tm_dg_tb_tm_dg + ta
        return {
            "y_outer": y_outer,
            "y_inner": y_inner,
            "z_ta": z_ta,
            "z_ta_tm_dg": z_ta_tm_dg,
            "z_ta_tm_dg_tb": z_ta_tm_dg_tb,
            "z_ta_tm_dg_tb_tm_dg": z_ta_tm_dg_tb_tm_dg,
            "z_total": z_total,
        }

    @property
    def polyline1_points(self):
        mm = self._mm
        d = self._dims
        return [
            [mm(0), mm(0), mm(0)],
            [mm(0), mm(d["y_outer"]), mm(0)],
            [mm(0), mm(d["y_outer"]), mm(self.ta)],
            [mm(0), mm(d["y_inner"]), mm(self.ta)],
            [mm(0), mm(d["y_inner"]), mm(d["z_ta_tm_dg"])],
            [mm(0), mm(d["y_outer"]), mm(d["z_ta_tm_dg"])],
            [mm(0), mm(d["y_outer"]), mm(d["z_ta_tm_dg_tb"])],
            [mm(0), mm(d["y_inner"]), mm(d["z_ta_tm_dg_tb"])],
            [mm(0), mm(d["y_inner"]), mm(d["z_ta_tm_dg_tb_tm_dg"])],
            [mm(0), mm(d["y_outer"]), mm(d["z_ta_tm_dg_tb_tm_dg"])],
            [mm(0), mm(d["y_outer"]), mm(d["z_total"])],
            [mm(0), mm(0), mm(d["z_total"])],
        ]

    @property
    def rect1_origin(self):
        mm = self._mm
        d = self._dims
        return ["0mm", mm(d["y_outer"]), mm(d["z_total"])]

    @property
    def rect1_sizes(self):
        mm = self._mm
        height = 2 * (self.tm + self.dg) + self.tb + 2 * self.ta
        return [mm(self.ta), mm(-height)]

    @property
    def rect2_origin(self):
        mm = self._mm
        d = self._dims
        return ["0mm", mm(self.ta), mm(d["z_ta_tm_dg_tb_tm_dg"])]

    @property
    def rect2_sizes(self):
        mm = self._mm
        return [mm(self.la - self.ta), mm(-self.tm)]

    @property
    def rect3_origin(self):
        mm = self._mm
        return ["0mm", mm(self.ta), mm(self.ta)]

    @property
    def rect3_sizes(self):
        mm = self._mm
        return [mm(self.la - self.ta), mm(self.tm)]

    @property
    def sweep_vector(self):
        return [self._mm(self.wa), "0mm", "0mm"]
        
    @property
    def box1_position(self):
        mm = self._mm
        z_origin = self.ta + self.tm
        return ["0mm", mm(self.ta), mm(z_origin)]

    @property
    def box1_size(self):
        mm = self._mm
        span_y = max(self.la - self.ta, 0.01)
        span_z = max(self.dg, 0.01)
        return [mm(self.wa), mm(span_y), mm(span_z)]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


@dataclass
class ActuatorDesignVariables:
    """封装自由变量，并提供派生参数、约束检测与性能计算。尺寸单位统一为 mm。"""

    FLOAT_FIELDS: ClassVar[Tuple[str, ...]] = (
        "lm",
        "tm",
        "ta",
        "dg",
        "hs",
        "wslot",
        "hslot",
        "s",
        "wa",
        "wcoil",
        "tb_ratio",
    )

    lm: float
    tm: float
    ta: float
    dg: float
    hs: float
    wslot: float
    hslot: float
    s: float
    tb_ratio: float  # 自由变量，范围 [1.6, 2.0]，用于计算 tb（必须在有默认值的字段之前）
    wa: float = WA_FIXED  # 磁轭宽度（固定为 2.0）
    wcoil: float = DEFAULT_WCOIL_MM  # 线径固定 0.05mm

    def __post_init__(self):
        for name in self.FLOAT_FIELDS:
            setattr(self, name, _quantize(getattr(self, name)))
        # tm 现在是连续变量 [0.3, 0.5]，不再离散化
        self.wcoil = DEFAULT_WCOIL_MM
        self.wa = WA_FIXED

    @property
    def la(self) -> float:
        return self.lm + 2.0 * self.ta

    @property
    def ha(self) -> float:
        return 2 * self.ta + self.tb + 2 * self.tm + 2 * self.dg

    @property
    def ws(self) -> float:
        return self.wslot + 2 * self.twall

    @property
    def ls(self) -> float:
        return self.lm - self.s

    @property
    def tb(self) -> float:
        # tb_ratio 范围限制为 [1.6, 2.0]
        return _clamp(self.tb_ratio * self.ta, 1.6 * self.ta, 2.0 * self.ta)

    @property
    def twall(self) -> float:
        return 0.5 * max(self.hs - self.hslot, 0.0)

    def _derived_turns(self, numerator: float) -> int:
        if self.wcoil <= 0:
            return 0
        raw = max(numerator / self.wcoil, 0.0)
        return max(1, int(round(raw)))

    @property
    def n1(self) -> int:
        if self.wcoil <= 0:
            return 0
        ls_eff = max(self.ls, 0.0)
        raw = 0.8 * ls_eff / self.wcoil
        return max(1, int(raw) if TURNS_FLOOR_MODE else int(round(raw)))

    @property
    def n2(self) -> int:
        if self.wcoil <= 0:
            return 0
        twall_eff = max(self.twall, 0.0)
        raw = 0.9 * twall_eff / self.wcoil
        # n2 is always downward-truncated to preserve a conservative,
        # physically packable winding-layer count in the released workflow.
        return max(1, int(raw))

    @property
    def total_turns(self) -> int:
        return self.n1 * self.n2

    def to_sim_config(
        self,
        output_dir: str,
    ) -> MaxwellSimulationConfig:
        abs_output = os.path.abspath(output_dir)
        fld_target = os.path.join(abs_output, "11.fld")
        bsat_target = os.path.join(abs_output, "Bsat_max.fld")  # 保留兼容
        bsat_ta_target = os.path.join(abs_output, "Bsat_ta.fld")
        bsat_tb_target = os.path.join(abs_output, "Bsat_tb.fld")
        return MaxwellSimulationConfig(
            output_dir=abs_output,
            calc_fld_path=fld_target,
            bsat_fld_path=bsat_target,
            bsat_ta_fld_path=bsat_ta_target,
            bsat_tb_fld_path=bsat_tb_target,
            lm=self.lm,
            ta=self.ta,
            tb=self.tb,
            tm=self.tm,
            dg=self.dg,
            wa=self.wa,
        )

    def _mm_to_m(self, value: float) -> float:
        return value * MM_TO_M

    def compute_performance_metrics(self, avg_B: float) -> Dict[str, float]:
        # 仿真可能得到负值，这里统一取绝对值，保证输出非负
        avg_B = abs(avg_B)

        la_m = self._mm_to_m(self.la)
        ha_m = self._mm_to_m(self.ha)
        ws_m = self._mm_to_m(self.ws)
        lm_m = self._mm_to_m(self.lm)
        tm_m = self._mm_to_m(self.tm)
        dg_m = self._mm_to_m(self.dg)
        tb_m = self._mm_to_m(self.tb)
        ls_m = self._mm_to_m(self.ls)
        hs_m = self._mm_to_m(self.hs)
        wslot_m = self._mm_to_m(self.wslot)
        twall_m = self._mm_to_m(self.twall)
        hslot_m = self._mm_to_m(self.hslot)
        s_m = self._mm_to_m(self.s)
        wa_m = self._mm_to_m(self.wa)

        volume = la_m * ha_m * ws_m

        core_cross_section = max(la_m * ha_m - 2.0 * lm_m * (tm_m + dg_m), 0.0)
        core_volume = wa_m * core_cross_section
        magnet_volume = 2.0 * lm_m * tm_m * wa_m
        ma = KG_PER_M3_CORE * core_volume + KG_PER_M3_MAGNET * magnet_volume

        coil_area = max((hs_m * ws_m) - (hslot_m * wslot_m), 0.0)
        ms = 0.8 * KG_PER_M3_COPPER * max(ls_m, 0.0) * coil_area
        mass_total = ma + ms

        n_turns = self.total_turns
        kb = 2.0 * n_turns * avg_B * wa_m

        pb = 0.0
        if ma > 0 and s_m > 0 and kb > 0:
            pb = 4.0 * (kb ** 1.5) * (s_m ** 0.5) * (ma ** -0.5)

        return {
            "volume": volume,
            "mass_mover": ma,
            "mass_stator": ms,
            "mass_total": mass_total,
            "kb": kb,
            "pb": pb,
            "ls_m": ls_m,
            "tb_m": tb_m,
        }

    def validate_without_B(self) -> List[str]:
        errors: List[str] = []
        clearance = 0.02  # 20 μm -> 0.02 mm

        if self.hs <= 0 or self.wslot <= 0 or self.twall <= 0:
            errors.append("线圈外形尺寸必须为正值。")

        if self.twall <= 0:
            errors.append("线圈壁厚 (twall) 必须为正值。")

        if (2 * self.dg + self.tb - self.hs) < 0.1:
            errors.append("约束(11) 不满足：2dg + tb - hs ≥ 0.1mm。")

        if self.wcoil < 0.05:
            errors.append("约束(12) 不满足：线径需 ≥ 0.05mm。")

        # 线圈第二方向匝数至少为 3 匝
        if self.n2 < 3:
            errors.append(f"约束(28) 不满足：n2={self.n2} < 3 匝。")

        if abs(self.ws - (self.wslot + 2 * self.twall)) > 1e-6:
            errors.append("约束(23) 不满足：ws ≠ wslot + 2*twall。")

        if abs(self.hs - (self.hslot + 2 * self.twall)) > 1e-6:
            errors.append("约束(24) 不满足：hs ≠ hslot + 2*twall。")

        if (self.hslot - self.tb) < 10*clearance:
            errors.append("约束(25) 不满足：hslot - tb ≥ 0.2mm。")

        if (self.wslot - self.wa) < 10*clearance:
            errors.append("约束(26) 不满足：wslot - wa ≥ 0.2mm。")

        if (self.dg - self.twall) < clearance:
            errors.append("约束(27) 不满足：dg - twall ≥ 0.02mm。")

        if self.s < 1.0:
            errors.append("约束(13) 不满足：行程 s >= 1mm。")

        if self.ws >= 4.0:
            errors.append("约束(14) 不满足：ws < 4mm。")

        if self.ha >= 5.0:
            errors.append("约束(15) 不满足：ha < 5mm。")

        if self.la > 6.0:
            errors.append("约束(16) 不满足：la ≤ 6mm。")

        if not (0.3 < self.ta < 1.0):
            errors.append("约束(21) 不满足：0.3mm < ta < 1mm。")

        if self.ls <= 0:
            errors.append("约束(18) 不满足：ls = lm - s > 0。")

        if self.tb < 1.5 * self.ta or self.tb > 2.5 * self.ta:
            errors.append("约束(22) 不满足：tb 未处于 [1.5ta, 2.5ta]。")

        if (self.hslot - self.tb) < 0.1:
            errors.append("约束(29) 不满足：hslot - tb ≥ 0.1mm。")

        if self.wslot <= self.wa:
            errors.append("wslot 必须大于 wa 以保证装配间隙。")

        return errors

    def validate_post_sim(self, b_max: float) -> Tuple[List[str], float, bool]:
        """
        仿真后验证
        
        Returns:
            (errors, b_max, is_saturated)
            - errors: 严重错误列表
            - b_max: 最大磁密值
            - is_saturated: 是否发生磁饱和 (B_max >= 2.0T)
        """
        errors: List[str] = []
        is_saturated = False
        
        if b_max is None:
            errors.append("无法计算最大磁密，B_max 为 None。")
            return errors, float("inf"), False

        # 磁饱和检测：B_max >= 2.0T 时标记为饱和，但不作为错误
        if b_max >= 2.0:
            is_saturated = True
            # 不再添加到 errors 中，改为返回标志
        
        return errors, b_max, is_saturated


DESIGN_VARIABLE_NAMES = [
    "lm",
    "tm",
    "ta",
    "dg",
    "hs",
    "wslot",
    "hslot",
    "s",
    "wa",
    "tb_ratio",
]
INT_VARIABLES: set = set()


@dataclass
class DesignVariableBounds:
    """更宽泛的可行域，用于早期探索而不过度限制设计空间。"""

    lm: Tuple[float, float] = (0, 6)
    tm: Tuple[float, float] = (0.3, 0.5)  # tm 连续变量，范围扩大到 [0.3, 0.5]
    ta: Tuple[float, float] = (0.35, 0.75)
    dg: Tuple[float, float] = (0.25, 0.65)
    hs: Tuple[float, float] = (1.2, 2.2)
    wslot: Tuple[float, float] = (2.0, 2.8)
    hslot: Tuple[float, float] = (0.8, 1.3)
    s: Tuple[float, float] = (0.8, 1.2)
    wa: Tuple[float, float] = (WA_FIXED, WA_FIXED)
    tb_ratio: Tuple[float, float] = (1.6, 2.0)  # tb_ratio 自由变量范围

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        return asdict(self)


@dataclass
class GAConfig:
    population_size: int = 6
    generations: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    mutation_sigma: float = 0.1
    tournament_k: int = 3
    minimize: bool = True
    seed: Optional[int] = None
    min_valid_evals: int = 60


def _clamp_gene(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _individual_to_design(
    individual: List[float],
    bounds_map: Dict[str, Tuple[float, float]],
) -> ActuatorDesignVariables:
    kwargs = {}
    for val, name in zip(individual, DESIGN_VARIABLE_NAMES):
        low, high = bounds_map[name]
        clamped = _clamp_gene(val, low, high)
        if name in INT_VARIABLES:
            clamped = int(round(clamped))
        if name == "tm":
            clamped = 0.4 if clamped < 0.45 else 0.5
        kwargs[name] = clamped
    return ActuatorDesignVariables(**kwargs)


def _genome_to_design(
    genome: List[float], bounds_map: Dict[str, Tuple[float, float]]
) -> ActuatorDesignVariables:
    return _individual_to_design(genome, bounds_map)


def _is_genome_valid(
    genome: List[float], bounds_map: Dict[str, Tuple[float, float]]
) -> Tuple[bool, ActuatorDesignVariables, List[str]]:
    design = _genome_to_design(genome, bounds_map)
    errors = design.validate_without_B()
    return not errors, design, errors


def _resample_genome(
    generator_fn,
    bounds_map: Dict[str, Tuple[float, float]],
    description: str,
) -> Tuple[List[float], ActuatorDesignVariables]:
    last_errors: List[str] = []
    last_design: Optional[ActuatorDesignVariables] = None
    for _ in range(MAX_GENOME_ATTEMPTS):
        genome = generator_fn()
        is_valid, design, errors = _is_genome_valid(genome, bounds_map)
        if is_valid:
            return genome, design
        last_errors = errors
        last_design = design
    raise RuntimeError(
        f"{description} 在 {MAX_GENOME_ATTEMPTS} 次尝试后仍未满足约束。"
        f"最后一次设计: {last_design}，错误: {last_errors}"
    )


def _random_individual(bounds_map: Dict[str, Tuple[float, float]]) -> List[float]:
    def _generator() -> List[float]:
        genome: List[float] = []
        for name in DESIGN_VARIABLE_NAMES:
            low, high = bounds_map[name]
            genome.append(random.uniform(low, high))
        return genome

    genome, _ = _resample_genome(_generator, bounds_map, "随机采样")
    return genome


def _sigma_map(
    bounds_map: Dict[str, Tuple[float, float]],
    cfg: GAConfig,
) -> Dict[str, float]:
    sigmas: Dict[str, float] = {}
    for name in DESIGN_VARIABLE_NAMES:
        low, high = bounds_map[name]
        span = high - low
        base = max(span * cfg.mutation_sigma, 1.0 if name in INT_VARIABLES else 1e-3)
        sigmas[name] = base
    return sigmas


def _mutate(
    genome: List[float],
    bounds_map: Dict[str, Tuple[float, float]],
    sigma_map: Dict[str, float],
    cfg: GAConfig,
) -> List[float]:
    def _generator() -> List[float]:
        mutated = genome[:]
        for idx, name in enumerate(DESIGN_VARIABLE_NAMES):
            if random.random() < cfg.mutation_rate:
                low, high = bounds_map[name]
                sigma = sigma_map[name]
                mutated[idx] = _clamp_gene(
                    mutated[idx] + random.gauss(0, sigma), low, high
                )
        return mutated

    mutated, _ = _resample_genome(_generator, bounds_map, "突变")
    return mutated


def _history_entry(label: object, record: Dict[str, object]) -> Dict[str, object]:
    result = record.get("result", {}) or {}
    return {
        "generation": label,
        "best_fitness": record.get("fitness"),
        "best_avg_B": result.get("avg_B"),
        "best_kb": result.get("kb"),
        "best_pb": result.get("pb"),
        "best_mass_total": result.get("mass_total"),
    }


def _blend_crossover(
    parent1: List[float],
    parent2: List[float],
    bounds_map: Dict[str, Tuple[float, float]],
) -> Tuple[List[float], List[float]]:
    alpha = random.random()
    child1: List[float] = []
    child2: List[float] = []
    for idx, name in enumerate(DESIGN_VARIABLE_NAMES):
        low, high = bounds_map[name]
        v1 = parent1[idx]
        v2 = parent2[idx]
        c1 = _clamp_gene(alpha * v1 + (1 - alpha) * v2, low, high)
        c2 = _clamp_gene(alpha * v2 + (1 - alpha) * v1, low, high)
        child1.append(c1)
        child2.append(c2)
    return child1, child2


def _tournament_select(
    population: List[List[float]],
    records: List[Dict[str, object]],
    cfg: GAConfig,
) -> List[float]:
    best_idx = None
    for _ in range(cfg.tournament_k):
        idx = random.randrange(len(population))
        if best_idx is None or records[idx]["score"] > records[best_idx]["score"]:
            best_idx = idx
    return population[best_idx][:]


def run_genetic_optimization(
    project_name: str,
    design_name: str,
    setup_name: str,
    weight_factors: Sequence[float],
    bounds: DesignVariableBounds,
    ga_config: GAConfig,
    output_root: str = "ga_runs",
) -> Dict[str, object]:
    """基于 evaluate_design_fitness 的简单遗传算法封装。"""

    if ga_config.seed is not None:
        random.seed(ga_config.seed)

    os.makedirs(output_root, exist_ok=True)
    bounds_map = bounds.as_dict()
    sigma_map = _sigma_map(bounds_map, ga_config)
    cache: Dict[Tuple[float, ...], Dict[str, object]] = {}
    valid_evaluations = 0
    feasible_records: List[Dict[str, object]] = []

    def evaluate(individual: List[float]) -> Dict[str, object]:
        nonlocal valid_evaluations
        key = tuple(_quantize(v) for v in individual)
        if key in cache:
            return cache[key]

        eval_id = len(cache)
        eval_dir = os.path.join(output_root, f"eval_{eval_id:05d}")
        os.makedirs(eval_dir, exist_ok=True)

        design = _individual_to_design(individual, bounds_map)
        result = evaluate_design_fitness(
            design,
            weight_factors=weight_factors,
            project_name=project_name,
            design_name=design_name,
            setup_name=setup_name,
            output_dir=eval_dir,
        )
        fitness = result.get("fitness", PENALTY_FITNESS)
        score = -fitness if ga_config.minimize else fitness

        record = {
            "genome": individual[:],
            "design": design,
            "result": result,
            "fitness": fitness,
            "score": score,
        }
        cache[key] = record
        if result.get("status") == "ok":
            valid_evaluations += 1
            feasible_records.append(record)
        return record

    population = [_random_individual(bounds_map) for _ in range(ga_config.population_size)]
    records = [evaluate(ind) for ind in population]

    best_record = min(records, key=lambda r: r["fitness"])
    history = [_history_entry(0, best_record)]

    for gen in range(1, ga_config.generations + 1):
        new_population: List[List[float]] = []
        while len(new_population) < ga_config.population_size:
            parent1 = _tournament_select(population, records, ga_config)
            parent2 = _tournament_select(population, records, ga_config)

            if random.random() < ga_config.crossover_rate:
                child1, child2 = _blend_crossover(parent1, parent2, bounds_map)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = _mutate(child1, bounds_map, sigma_map, ga_config)
            child2 = _mutate(child2, bounds_map, sigma_map, ga_config)
            new_population.extend([child1, child2])

        population = new_population[: ga_config.population_size]
        records = [evaluate(ind) for ind in population]

        gen_best = min(records, key=lambda r: r["fitness"])
        if gen_best["fitness"] < best_record["fitness"]:
            best_record = gen_best
        history.append(_history_entry(gen, gen_best))

    extras_added = 0
    while valid_evaluations < ga_config.min_valid_evals:
        extra_genome = _random_individual(bounds_map)
        extra_key = tuple(_quantize(v) for v in extra_genome)
        if extra_key in cache:
            continue
        record = evaluate(extra_genome)
        extras_added += 1
        if record["fitness"] < best_record["fitness"]:
            best_record = record
        history.append(_history_entry(f"extra_{extras_added}", best_record))

    return {
        "best_record": best_record,
        "history": history,
        "evaluations": len(cache),
        "valid_evaluations": valid_evaluations,
        "population": population,
        "feasible_records": feasible_records,
    }


def save_fitness_convergence_plot(
    history: Sequence[Dict[str, object]], output_path: str
) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    xs = list(range(len(history)))
    ys = [entry.get("best_fitness") for entry in history]
    labels = [str(entry.get("generation")) for entry in history]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Generation / Extra Samples")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Convergence")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(xs, labels, rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_maxwell_with_pyaedt(
    config: MaxwellSimulationConfig,
    project_name: str = "PyAEDT_Project",
    design_name: str = "Maxwell3DDesign_PyAEDT",
    setup_name: str = "Setup1",
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    使用 PyAEDT 自动完成：
    1. 启动 AEDT / Maxwell
    2. 创建 Maxwell 3D 工程与设计（磁静态）
    3. 建一个简单的几何模型并赋材料
    4. 创建求解设置并运行仿真
    5. 使用 FieldsReporter 将 Box1 平均 Bz 写入 .fld
    6. 分别计算上轭和中间块区域的平均|B|（用于磁饱和判断）

    返回 (avg_B, b_mean_ta, b_mean_tb, result_source, result_desc)。
    - b_mean_ta: 上轭区域平均|B| (T)
    - b_mean_tb: 中间块区域平均|B| (T)
    """

    output_dir = os.path.abspath(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # non_graphical=True 静默运行，减少控制台输出；
    # version/new_desktop 为新接口，close_on_exit=True 在每次仿真后关闭 AEDT。
    avg_B = None
    b_mean_ta = None  # 上轭区域平均|B|
    b_mean_tb = None  # 中间块区域平均|B|
    result_source = None
    result_desc = None

    # 默认保持“静默+运行完自动关闭”，避免弹出 GUI 干扰。
    # 如需打开 GUI 或保持不关闭，请设置：
    # - AEDT_NON_GRAPHICAL=0
    # - AEDT_CLOSE_ON_EXIT=0
    aedt_non_graphical = _env_bool("AEDT_NON_GRAPHICAL", True)
    aedt_close_on_exit = _env_bool("AEDT_CLOSE_ON_EXIT", True)
    aedt_new_desktop = _env_bool("AEDT_NEW_DESKTOP", True)

    with Desktop(
        version=AEDT_VERSION,
        non_graphical=aedt_non_graphical,
        new_desktop=aedt_new_desktop,
        close_on_exit=aedt_close_on_exit,
    ):
        app = Maxwell3d(
            projectname=project_name,
            designname=design_name,
            solution_type="Magnetostatic",
        )

        # 与 12.py 一致，使用 mm 作为建模单位
        app.modeler.model_units = "mm"

        # 0. 确保 NdFe35 和 NdFe35 - Copy 材料存在，且矫顽力方向分别为 +z、-z ----------
        oDefinitionManager = app.oproject.GetDefinitionManager()

        # 修改 NdFe35，使其矫顽力方向为 (0, 0, 1)
        oDefinitionManager.EditMaterial(
            "NdFe35",
            [
                "NAME:NdFe35",
                "CoordinateSystemType:=",
                "Cartesian",
                "BulkOrSurfaceType:=",
                1,
                [
                    "NAME:PhysicsTypes",
                    "set:=",
                    ["Electromagnetic", "Thermal", "Structural"],
                ],
                [
                    "NAME:AttachedData",
                    [
                        "NAME:MatAppearanceData",
                        "property_data:=",
                        "appearance_data",
                        "Red:=",
                        204,
                        "Green:=",
                        204,
                        "Blue:=",
                        204,
                    ],
                ],
                "permittivity:=",
                "1",
                "permeability:=",
                "1.0997785406",
                "conductivity:=",
                "625000",
                "dielectric_loss_tangent:=",
                "0",
                "magnetic_loss_tangent:=",
                "0",
                [
                    "NAME:magnetic_coercivity",
                    "property_type:=",
                    "VectorProperty",
                    "Magnitude:=",
                    "-890000A_per_meter",
                    "DirComp1:=",
                    "0",
                    "DirComp2:=",
                    "0",
                    "DirComp3:=",
                    "1",
                ],
                "thermal_conductivity:=",
                "0",
                "saturation_mag:=",
                "0gauss",
                "lande_g_factor:=",
                "2",
                "delta_H:=",
                "0Oe",
                "mass_density:=",
                "7400",
                "youngs_modulus:=",
                "147000000000",
                [
                    "NAME:thermal_expansion_coefficient",
                    "property_type:=",
                    "AnisoProperty",
                    "unit:=",
                    "",
                    "component1:=",
                    "3e-06",
                    "component2:=",
                    "-5e-06",
                    "component3:=",
                    "-5e-06",
                ],
            ],
        )

        # 克隆 NdFe35 为 NdFe35 - Copy，并将矫顽力方向改为 (0, 0, -1)
        oDefinitionManager.AddMaterial(
            [
                "NAME:NdFe35 - Copy",
                "CoordinateSystemType:=",
                "Cartesian",
                "BulkOrSurfaceType:=",
                1,
                [
                    "NAME:PhysicsTypes",
                    "set:=",
                    ["Electromagnetic", "Thermal", "Structural"],
                ],
                [
                    "NAME:AttachedData",
                    [
                        "NAME:MatAppearanceData",
                        "property_data:=",
                        "appearance_data",
                        "Red:=",
                        204,
                        "Green:=",
                        204,
                        "Blue:=",
                        204,
                    ],
                ],
                "permittivity:=",
                "1",
                "permeability:=",
                "1.0997785406",
                "conductivity:=",
                "625000",
                "dielectric_loss_tangent:=",
                "0",
                "magnetic_loss_tangent:=",
                "0",
                [
                    "NAME:magnetic_coercivity",
                    "property_type:=",
                    "VectorProperty",
                    "Magnitude:=",
                    "-890000A_per_meter",
                    "DirComp1:=",
                    "0",
                    "DirComp2:=",
                    "0",
                    "DirComp3:=",
                    "-1",
                ],
                "thermal_conductivity:=",
                "0",
                "saturation_mag:=",
                "0gauss",
                "lande_g_factor:=",
                "2",
                "delta_H:=",
                "0Oe",
                "mass_density:=",
                "7400",
                "youngs_modulus:=",
                "147000000000",
                [
                    "NAME:thermal_expansion_coefficient",
                    "property_type:=",
                    "AnisoProperty",
                    "unit:=",
                    "",
                    "component1:=",
                    "3e-06",
                    "component2:=",
                    "-5e-06",
                    "component3:=",
                    "-5e-06",
                ],
            ],
        )

        # 1. 按 12.py 逐点重建几何 -------------------------------------------------
        # 1.1 折线 Polyline1（Y-Z 截面轮廓），然后沿 X 方向扫掠 2mm
        # PyAEDT 0.22.2 中，多段线的 segment_type 用列表更稳妥
        # 在你当前的 PyAEDT 版本中，segment_type 使用简单字符串列表即可
        segment_types = ["Line"] * (len(config.polyline1_points) - 1)
        polyline1 = app.modeler.create_polyline(
            points=config.polyline1_points,
            segment_type=segment_types,
            cover_surface=True,
            close_surface=True,
            name="Polyline1",
            material="vacuum",
        )

        # 1.2 Rectangle1（与 12.py 一致）
        rect1 = app.modeler.create_rectangle(
            origin=config.rect1_origin,
            sizes=config.rect1_sizes,
            orientation="YZ",  # 与 12.py 中 WhichAxis='X' 一致，在 Y-Z 平面上
            name="Rectangle1",
            material="vacuum",
        )

        # 1.3 Rectangle2（上永磁体投影，稍后再通过厚度和材料区分）
        rect2 = app.modeler.create_rectangle(
            origin=config.rect2_origin,
            sizes=config.rect2_sizes,  # 初始为 -0.6mm，后续等效 0.5mm 厚
            orientation="YZ",
            name="Rectangle2",
            material="vacuum",
        )

        # 1.4 Rectangle3（下永磁体投影）
        rect3 = app.modeler.create_rectangle(
            origin=config.rect3_origin,
            sizes=config.rect3_sizes,
            orientation="YZ",
            name="Rectangle3",
            material="vacuum",
        )

        # 1.5 沿 X 向正方向扫掠（与 12.py 中最终 SweepVectorX 一致）
        # 为避免某个对象创建失败导致整组 sweep 报错，这里逐个对象检查并分别扫掠。
        for obj_name in ["Polyline1", "Rectangle1", "Rectangle2", "Rectangle3"]:
            if obj_name in app.modeler.object_names:
                app.modeler.sweep_along_vector(
                    assignment=obj_name,
                    sweep_vector=config.sweep_vector,
                )

        # 1.6 按 12.py 赋材料：
        # - Polyline1 + Rectangle1  → DW310_35（磁芯）
        # - Rectangle2             → NdFe35      （上永磁体，+z 矫顽力）
        # - Rectangle3             → NdFe35 - Copy（下永磁体，-z 矫顽力）
        # PyAEDT 中赋材料接口在 app 上，而不是 modeler
        # 赋材料时按对象名称字符串传入
        app.assign_material("Polyline1", "DW310_35")
        app.assign_material("Rectangle1", "DW310_35")
        app.assign_material("Rectangle2", "NdFe35")
        app.assign_material("Rectangle3", "NdFe35 - Copy")

        # 1.7 观察体积 Box1，用于 FieldsReporter 计算平均磁感应强度
        box1 = app.modeler.create_box(
            position=config.box1_position,
            dimensions_list=config.box1_size,
            name="Box1",
            material="vacuum",
        )
        try:
            box1.solve_inside = False
            box1.model = False
        except Exception:
            pass

        # 1.8 磁饱和检测面（两个面分别测量上轭和中间块的平均|B|）
        # ════════════════════════════════════════════════════════════════
        # 代码坐标系 vs 用户/AEDT 显示坐标系对应关系：
        #   代码 X（扫掠方向，尺寸 wa）  ←→  用户 Z
        #   代码 Y（长度方向，尺寸 la）  ←→  用户 Y
        #   代码 Z（高度方向，层叠 ta/tb）←→  用户 X
        #
        # 几何分布（沿代码 Z / 用户 X 方向，从下到上）：
        #   下轭 [0, ta] → 永磁体+气隙 → 中间块 [z_tb_start, +tb]
        #   → 永磁体+气隙 → 上轭 [z_ta_start, +ta]
        #
        # 检测面放置：
        #   - 平面类型：代码 XZ 平面（法向量指向代码 +Y）
        #   - Y 坐标 = ta（铁芯内侧位置）
        #   - BsatPlane_ta：覆盖上轭区域，高度方向尺寸 = ta
        #   - BsatPlane_tb：覆盖中间块区域，高度方向尺寸 = tb
        #   - 扫掠方向尺寸 = wa
        # ════════════════════════════════════════════════════════════════
        z_tb_start = config.ta + config.tm + config.dg  # 中间块底部（代码 Z）
        z_ta_start = z_tb_start + config.tb + config.tm + config.dg  # 上轭底部（代码 Z）
        
        # 1.8.1 上轭区域检测面 BsatPlane_ta
        # 位置：代码坐标 (X=0, Y=ta, Z=z_ta_start)
        # 尺寸：扫掠方向（代码 X）= wa，高度方向（代码 Z / 用户 X）= ta
        try:
            if "BsatPlane_ta" in app.modeler.object_names:
                app.modeler.delete("BsatPlane_ta")
            bsat_plane_ta = app.modeler.create_rectangle(
                origin=[config._mm(0), config._mm(config.ta), config._mm(z_ta_start)],
                sizes=[config._mm(config.ta), config._mm(config.wa)],
                orientation="XZ",  # 代码 XZ 平面，法向量指向代码 +Y
                name="BsatPlane_ta",
                material="vacuum",
            )
            # 2D 平面对象没有 solve_inside 属性，只设置 model=False
            try:
                bsat_plane_ta.model = False
            except Exception:
                pass
        except Exception as e:
            print("创建 BsatPlane_ta（上轭）失败：", repr(e))
        
        # 1.8.2 中间块区域检测面 BsatPlane_tb
        # 位置：代码坐标 (X=0, Y=ta, Z=z_tb_start)
        # 尺寸：扫掠方向（代码 X）= wa，高度方向（代码 Z / 用户 X）= tb
        try:
            if "BsatPlane_tb" in app.modeler.object_names:
                app.modeler.delete("BsatPlane_tb")
            bsat_plane_tb = app.modeler.create_rectangle(
                origin=[config._mm(0), config._mm(config.ta), config._mm(z_tb_start)],
                sizes=[config._mm(config.tb), config._mm(config.wa)],
                orientation="XZ",  # 代码 XZ 平面，法向量指向代码 +Y
                name="BsatPlane_tb",
                material="vacuum",
            )
            # 2D 平面对象没有 solve_inside 属性，只设置 model=False
            try:
                bsat_plane_tb.model = False
            except Exception:
                pass
        except Exception as e:
            print("创建 BsatPlane_tb（中间块）失败：", repr(e))

        # 1.9 外部 Region（与 12.py 类似，使用 40% padding）
        region = app.modeler.create_region(pad_percent=[40, 40, 40, 40, 40, 40])

        # 1.10 网格长度控制（直接调用底层 MeshSetup.AssignLengthOp，与 12.py 完全一致：onlength = inlength = 0.3mm）
        try:
            if isinstance(region, str):
                region_name = region
            else:
                region_name = getattr(region, "name", "Region")

            omesh = app.odesign.GetModule("MeshSetup")

            # Length1：只对 Region，RefineInside=False，MaxLength=0.3mm
            omesh.AssignLengthOp(
                [
                    "NAME:Length1",
                    "RefineInside:=",
                    False,
                    "Enabled:=",
                    True,
                    "Objects:=",
                    [region_name],
                    "RestrictElem:=",
                    False,
                    "NumMaxElem:=",
                    "1000",
                    "RestrictLength:=",
                    True,
                    "MaxLength:=",
                    "1mm",
                ]
            )

            # Length2：对核心铁心、永磁体及 Region 进行细化
            mesh_targets = []
            for name in ["Polyline1", "Rectangle1", "Rectangle2", "Rectangle3"]:
                if name in app.modeler.object_names:
                    mesh_targets.append(name)
            mesh_targets.append(region_name)
            omesh.AssignLengthOp(
                [
                    "NAME:Length2",
                    "RefineInside:=",
                    True,
                    "Enabled:=",
                    True,
                    "Objects:=",
                    mesh_targets,
                    "RestrictElem:=",
                    False,
                    "NumMaxElem:=",
                    "1000",
                    "RestrictLength:=",
                    True,
                    "MaxLength:=",
                    "1mm",
                ]
            )
        except Exception as e:
            # 如果网格控制失败，则回退到默认网格，但不中断整个流程
            print("对 Region 及关键部件进行网格长度控制时出错，将使用默认网格。具体信息：", repr(e))

        # 2. 创建求解设置 --------------------------------------------------------
        setup = app.create_setup(setup_name)
        setup.props["MaximumPasses"] = 10
        setup.props["MinimumPasses"] = 2
        setup.props["MinimumConvergedPasses"] = 1
        setup.props["PercentRefinement"] = 30
        setup.props["PercentError"] = 1
        setup.props["SolveFieldOnly"] = False
        setup.props["UseIterativeSolver"] = False
        setup.update()

        # 3. 运行仿真 ------------------------------------------------------------
        # 与 12.py 一致，先按设置生成/细化网格，再对整个设计做一次全局 AnalyzeAllNominal。
        try:
            app.mesh.generate_mesh(setup_name)
        except Exception as e:
            # generate_mesh 在部分版本中不是必需，如失败则直接进入求解
            print("显式生成网格失败，将在求解时自动生成网格。具体信息：", repr(e))

        # 按 12.py 的思路做全局求解：先保存，再执行 AnalyzeAllNominal
        oproject = app.oproject
        odesign = app.odesign

        oproject.Save()
        odesign.AnalyzeAllNominal()
        oproject.Save()

        # 保存工程（可选）
        app.save_project()
        try:
            avg_B = compute_average_B(
                app,
                config.calc_fld_path,
                setup_name=setup_name,
                volume_name="Box1",
            )
            # ★新增：分别计算上轭和中间块区域的平均|B|
            b_mean_ta = compute_mean_B_on_plane(
                app,
                config.bsat_ta_fld_path,
                setup_name=setup_name,
                plane_name="BsatPlane_ta",
            )
            b_mean_tb = compute_mean_B_on_plane(
                app,
                config.bsat_tb_fld_path,
                setup_name=setup_name,
                plane_name="BsatPlane_tb",
            )
            result_source = "FieldsReporter"
            result_desc = (
                f"Box1 体积平均 Bz（写入 {config.calc_fld_path}）; "
                f"上轭平均|B|={b_mean_ta:.3f}T; 中间块平均|B|={b_mean_tb:.3f}T"
            )
        except Exception as e:
            print("使用 FieldsReporter 计算 Box1 平均 B 或饱和面平均 B 失败，请在 Maxwell GUI 中验证相同步骤。")
            print("具体异常信息：", repr(e))
            avg_B = None
            b_mean_ta = None
            b_mean_tb = None
            result_source = None
            result_desc = None

    if avg_B is None or b_mean_ta is None or b_mean_tb is None:
        return None, None, None, None, None

    return avg_B, b_mean_ta, b_mean_tb, result_source, result_desc


def evaluate_design_fitness(
    design: ActuatorDesignVariables,
    weight_factors: Sequence[float] = (0.5, 0.5, 4.0, 1.0),
    penalty_value: float = PENALTY_FITNESS,
    project_name: str = "PyAEDT_Project",
    design_name: str = "Maxwell3DDesign_PyAEDT",
    setup_name: str = "Setup1",
    output_dir: str = ".",
) -> Dict[str, object]:
    """整体流程：约束校验 → Maxwell 仿真（得到 B）→ 目标函数计算，返回 Fitness 及全部指标。"""

    pre_errors = design.validate_without_B()
    sim_config = design.to_sim_config(output_dir=output_dir)
    avg_B, b_mean_ta, b_mean_tb, result_source, result_desc = run_maxwell_with_pyaedt(
        sim_config,
        project_name=project_name,
        design_name=design_name,
        setup_name=setup_name,
    )

    if avg_B is None or b_mean_ta is None or b_mean_tb is None:
        return {
            "status": "simulation_failed",
            "fitness": penalty_value,
            "errors": ["Maxwell 仿真或 CSV 处理失败，无法得到 B 或饱和面平均B。"],
        }

    # 输出不允许为负，统一取绝对值
    avg_B = abs(avg_B)
    b_mean_ta = abs(b_mean_ta)
    b_mean_tb = abs(b_mean_tb)

    # ★新的饱和判断逻辑：分别判断上轭(ta)和中间块(tb)区域
    SATURATION_THRESHOLD = 2.0  # T
    is_saturated_ta = b_mean_ta >= SATURATION_THRESHOLD
    is_saturated_tb = b_mean_tb >= SATURATION_THRESHOLD
    is_saturated = is_saturated_ta or is_saturated_tb
    
    # 构建饱和区域描述
    saturation_region = None
    saturation_suggestion = None
    if is_saturated_ta and is_saturated_tb:
        saturation_region = "ta_and_tb"
        saturation_suggestion = "上轭和中间块均出现磁饱和，建议整体增大 ta"
    elif is_saturated_ta:
        saturation_region = "ta_region_upper"
        saturation_suggestion = "上轭区域出现磁饱和，建议增大 ta"
    elif is_saturated_tb:
        saturation_region = "tb_region_middle"
        saturation_suggestion = "中间块区域出现磁饱和，建议增大 tb 或调整 dg"

    # 兼容旧接口：B_sat 取两者最大值
    b_sat = max(b_mean_ta, b_mean_tb)
    
    # validate_post_sim 使用最大值（但不再依赖其判断饱和）
    post_errors, _, _ = design.validate_post_sim(b_sat)
    
    # 只有 pre_errors 或 post_errors 才返回约束违规（磁饱和不再是错误）
    if pre_errors or post_errors:
        result = {
            "status": "constraint_violation",
            "fitness": penalty_value,
            "avg_B": avg_B,
            "B_sat": b_sat,
            "B_mean_ta": b_mean_ta,
            "B_mean_tb": b_mean_tb,
            "errors": pre_errors + post_errors,
            "is_saturated": is_saturated,
            "is_saturated_ta": is_saturated_ta,
            "is_saturated_tb": is_saturated_tb,
            "saturation_region": saturation_region,
            "saturation_suggestion": saturation_suggestion,
        }
        return result

    metrics = design.compute_performance_metrics(avg_B)
    if len(weight_factors) != 4:
        raise ValueError("weight_factors 需要给出 4 个正数，对应 v1~v4。")
    v1, v2, v3, v4 = weight_factors

    # 参考值与归一化
    volume_refer = 6.7e-8
    mass_refer = 3.79e-4
    kb_refer = 0.239
    pb_refer = 0.819

    volume_r = metrics["volume"] / volume_refer
    mass_r = metrics["mass_total"] / mass_refer
    kb_r = metrics["kb"] / kb_refer if kb_refer else 0.0
    pb_r = metrics["pb"] / pb_refer if pb_refer else 0.0

    # 目标改为“越小越好”：几何/质量直接累加，性能用倒数降低权重
    eps = 1e-12
    fitness = (
        v1 * volume_r
        + v2 * mass_r
        - v3 * max(kb_r, eps)
        - v4 * max(pb_r, eps)
    )

    result = {
        "status": "ok",
        "fitness": fitness,
        "weights": weight_factors,
        "avg_B": avg_B,
        "B_sat": b_sat,  # 兼容：取两面平均的最大值
        "B_mean_ta": b_mean_ta,  # 上轭区域平均|B|
        "B_mean_tb": b_mean_tb,  # 中间块区域平均|B|
        "is_saturated": is_saturated,
        "is_saturated_ta": is_saturated_ta,  # 上轭是否饱和
        "is_saturated_tb": is_saturated_tb,  # 中间块是否饱和
        "saturation_region": saturation_region,
        "saturation_suggestion": saturation_suggestion,
        "result_source": result_source,
        "result_description": result_desc,
        "fld_file": sim_config.calc_fld_path,
        "fld_bsat_ta_file": sim_config.bsat_ta_fld_path,
        "fld_bsat_tb_file": sim_config.bsat_tb_fld_path,
        "volume": metrics["volume"],
        "mass_total": metrics["mass_total"],
        "mass_mover": metrics["mass_mover"],
        "mass_stator": metrics["mass_stator"],
        "kb": metrics["kb"],
        "pb": metrics["pb"],
        "derived_dimensions": {
            "la": design.la,
            "ha": design.ha,
            "ws": design.ws,
            "ls": design.ls,
            "tb": design.tb,
            "twall": design.twall,
        },
        "turns": {
            "n1": design.n1,
            "n2": design.n2,
            "total": design.total_turns,
        },
    }
    
    # 添加磁饱和警告（基于两个区域的平均|B|）
    if is_saturated:
        warning_parts = []
        if is_saturated_ta:
            warning_parts.append(f"上轭区域平均|B|={b_mean_ta:.3f}T≥2.0T")
        if is_saturated_tb:
            warning_parts.append(f"中间块区域平均|B|={b_mean_tb:.3f}T≥2.0T")
        result["saturation_warning"] = f"📌 磁饱和提示：{'; '.join(warning_parts)}。{saturation_suggestion or ''}"
    
    return result


def compute_average_B(
    app: Maxwell3d,
    fld_path: str,
    setup_name: str,
    volume_name: str = "Box1",
) -> float:
    """调用 FieldsReporter 栈操作计算指定体积（默认 Box1）内 Bz 的平均值，并写入 .fld 文件。"""
    if not fld_path:
        raise ValueError("fld_path 不能为空。")

    fld_full_path = os.path.abspath(fld_path)
    os.makedirs(os.path.dirname(fld_full_path), exist_ok=True)

    ofields = app.odesign.GetModule("FieldsReporter")
    ofields.CalcStack("clear")
    ofields.EnterQty("B")
    ofields.CalcOp("ScalarZ")
    ofields.EnterVol(volume_name)
    ofields.CalcOp("Integrate")
    ofields.EnterScalar(1)
    ofields.EnterVol(volume_name)
    ofields.CalcOp("Integrate")
    ofields.CalcOp("/")
    ofields.ClcEval(f"{setup_name} : LastAdaptive", [], "Fields")
    ofields.CalculatorWrite(
        fld_full_path,
        [
            "Solution:=",
            f"{setup_name} : LastAdaptive",
        ],
        [],
    )

    return _read_scalar_from_fld(fld_full_path)


def compute_mean_B_on_plane(
    app: Maxwell3d,
    fld_path: str,
    setup_name: str,
    plane_name: str,
) -> float:
    """
    使用 FieldsReporter 计算指定面的 |B| 平均值，并写入 .fld。
    
    操作步骤（参考用户脚本 1111111111.py）：
    1. EnterQty("B")
    2. CalcOp("Mag")
    3. EnterSurf(plane_name)
    4. CalcOp("Mean")
    
    Args:
        app: Maxwell3d 应用实例
        fld_path: 导出文件路径
        setup_name: setup 名称
        plane_name: 平面名称
        
    Returns:
        b_mean: 面上的平均 |B| 值 (T)
    """
    if not fld_path:
        raise ValueError("fld_path 不能为空。")

    fld_full_path = os.path.abspath(fld_path)
    os.makedirs(os.path.dirname(fld_full_path), exist_ok=True)

    ofields = app.odesign.GetModule("FieldsReporter")
    
    # 计算面上 |B| 平均值
    ofields.CalcStack("clear")
    ofields.EnterQty("B")
    ofields.CalcOp("Mag")
    ofields.EnterSurf(plane_name)
    ofields.CalcOp("Mean")
    ofields.ClcEval(f"{setup_name} : LastAdaptive", [], "Fields")
    ofields.CalculatorWrite(
        fld_full_path,
        [
            "Solution:=",
            f"{setup_name} : LastAdaptive",
        ],
        [],
    )
    
    return _read_scalar_from_fld(fld_full_path)


def compute_max_B_on_plane(
    app: Maxwell3d,
    fld_path: str,
    setup_name: str,
    plane_name: str = "BsatPlane",
    enable_position: bool = False,  # 开关：是否获取最大值位置
) -> Tuple[float, Optional[Tuple[float, float, float]]]:
    """
    使用 FieldsReporter 计算指定面的 |B| 最大值，并写入 .fld。
    
    Args:
        app: Maxwell3d 应用实例
        fld_path: 导出文件路径
        setup_name: setup 名称
        plane_name: 平面名称
        enable_position: 是否同时获取 B_max 位置坐标
        
    Returns:
        (b_max, b_max_pos): B_max 值和位置坐标 (x, y, z)，单位 mm
                           如果 enable_position=False，b_max_pos 为 None
    """
    if not fld_path:
        raise ValueError("fld_path 不能为空。")

    fld_full_path = os.path.abspath(fld_path)
    pos_fld_path = fld_full_path.replace(".fld", "_pos.fld")
    os.makedirs(os.path.dirname(fld_full_path), exist_ok=True)

    ofields = app.odesign.GetModule("FieldsReporter")
    
    # 步骤 1：计算 B_max 值
    ofields.CalcStack("clear")
    ofields.EnterQty("B")
    ofields.CalcOp("Mag")
    ofields.EnterSurf(plane_name)
    ofields.CalcOp("Maximum")
    ofields.ClcEval(f"{setup_name} : LastAdaptive", [], "Fields")
    ofields.CalculatorWrite(
        fld_full_path,
        [
            "Solution:=",
            f"{setup_name} : LastAdaptive",
        ],
        [],
    )
    b_max = _read_scalar_from_fld(fld_full_path)
    
    # 步骤 2：获取 B_max 位置（如果开启）
    b_max_pos = None
    if enable_position:
        try:
            ofields.CalcStack("clear")
            ofields.EnterQty("B")
            ofields.CalcOp("Mag")
            ofields.EnterSurf(plane_name)
            ofields.CalcOp("MaxPos")  # 获取最大值位置向量
            ofields.ClcEval(f"{setup_name} : LastAdaptive", [], "Fields")
            ofields.CalculatorWrite(
                pos_fld_path,
                [
                    "Solution:=",
                    f"{setup_name} : LastAdaptive",
                ],
                [],
            )
            b_max_pos = _read_vector_position_from_fld(pos_fld_path)
        except Exception as e:
            print(f"[WARNING] 获取 B_max 位置失败: {e}", file=sys.stderr)
            b_max_pos = None

    return b_max, b_max_pos


def _read_vector_position_from_fld(fld_path: str) -> Optional[Tuple[float, float, float]]:
    """
    从 FieldsReporter 导出的 .fld 文件中提取位置向量 (x, y, z)。
    
    fld 文件格式示例：
    $begin 'Named'
    Vector data "<0.00147, 0.0005, 0.0005>"
    x 1.47e-03
    y 5.0e-04
    z 5.0e-04
    $end 'Named'
    
    输出单位：mm（从 m 转换）
    """
    if not os.path.isfile(fld_path):
        return None
    
    x, y, z = None, None, None
    with open(fld_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 尝试解析 "x 1.47e-03" 格式
            if line.startswith("x "):
                try:
                    x = float(line.split()[1]) * 1000  # m -> mm
                except (IndexError, ValueError):
                    pass
            elif line.startswith("y "):
                try:
                    y = float(line.split()[1]) * 1000  # m -> mm
                except (IndexError, ValueError):
                    pass
            elif line.startswith("z "):
                try:
                    z = float(line.split()[1]) * 1000  # m -> mm
                except (IndexError, ValueError):
                    pass
            # 尝试解析 Vector data "<0.00147, 0.0005, 0.0005>" 格式
            elif "Vector data" in line:
                import re
                match = re.search(r'<([^>]+)>', line)
                if match:
                    parts = match.group(1).split(',')
                    if len(parts) == 3:
                        try:
                            x = float(parts[0].strip()) * 1000  # m -> mm
                            y = float(parts[1].strip()) * 1000  # m -> mm
                            z = float(parts[2].strip()) * 1000  # m -> mm
                        except ValueError:
                            pass
    
    if x is not None and y is not None and z is not None:
        return (x, y, z)
    return None


def _read_scalar_from_fld(fld_path: str) -> float:
    """从 FieldsReporter 导出的 .fld 文件中提取最终标量数值。"""
    if not os.path.isfile(fld_path):
        raise FileNotFoundError(f"找不到 .fld 文件: {fld_path}")

    value: Optional[float] = None
    with open(fld_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            candidate = token
            if '"' in candidate:
                parts = candidate.split('"')
                if len(parts) >= 2 and parts[1]:
                    candidate = parts[1]
            try:
                value = float(candidate)
            except ValueError:
                continue

    if value is None:
        raise ValueError(f"{fld_path} 中未找到有效的标量数值。")

    return value


def analyze_saturation_region(
    b_max_pos: Optional[Tuple[float, float, float]],
    design: ActuatorDesignVariables,
) -> Tuple[str, str]:
    """
    根据 B_max 位置坐标判断饱和发生的区域。
    
    BsatPlane 位于 y = ta 处的 XZ 平面。
    
    结构说明（从下到上，z 方向）：
    - 下轭 (ta_region): z ∈ [0, ta]
    - 气隙 (gap): z ∈ [ta, ta + dg]
    - 中间块 (tb_region): z ∈ [ta + dg, ta + dg + tb]
    - 上轭 (ta_region): z ∈ [ha - ta, ha]
    
    Args:
        b_max_pos: B_max 位置坐标 (x, y, z)，单位 mm
        design: 设计参数对象
        
    Returns:
        (region, suggestion): 区域名称和优化建议
    """
    if b_max_pos is None:
        return "unknown", "无法获取饱和位置"
    
    x, y, z = b_max_pos
    ta = design.ta
    dg = design.dg
    tb = design.tb
    ha = design.ha
    
    # 定义区域边界（带容差）
    tolerance = 0.1  # mm
    
    # 下轭区域
    lower_yoke_top = ta * 1.2
    # 上轭区域
    upper_yoke_bottom = ha - ta * 1.2
    # 中间块区域
    middle_bottom = ta + dg - tolerance
    middle_top = ta + dg + tb + tolerance
    
    region = "unknown"
    suggestion = ""
    
    if z <= lower_yoke_top:
        # 下轭区域饱和
        region = "ta_region_lower"
        suggestion = "磁饱和发生在下轭区域"
    elif z >= upper_yoke_bottom:
        # 上轭区域饱和
        region = "ta_region_upper"
        suggestion = "磁饱和发生在上轭区域"
    elif middle_bottom <= z <= middle_top:
        # 中间块区域饱和
        region = "tb_region"
        suggestion = "磁饱和发生在中间块区域"
    else:
        # 过渡区域（气隙附近或其他位置）
        region = "transition"
        suggestion = "磁饱和发生在过渡区域"
    
    return region, suggestion


if __name__ == "__main__":
    import argparse

    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="运行单次 Maxwell 仿真或 GA 优化。")
    parser.add_argument(
        "--mode",
        choices=["demo", "ga"],
        default="demo",
        help="demo=单次评估，ga=运行遗传算法优化",
    )
    parser.add_argument("--project", default="PyAEDT_Project")
    parser.add_argument("--design", default="Maxwell3DDesign_PyAEDT")
    parser.add_argument("--setup", default="Setup1")
    parser.add_argument(
        "--weights",
        nargs=4,
        type=float,
        default=[0.5, 0.5, 4.0, 1.0],
        metavar=("v1", "v2", "v3", "v4"),
        help="对应公式中 v1~v4 的权重",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--mutation-sigma", type=float, default=0.1)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="若 Fitness 需最大化，添加该参数；默认最小化",
    )
    parser.add_argument(
        "--min-valid-evals",
        type=int,
        default=60,
        help="至少生成多少个通过前置约束并进入 Maxwell 的个体",
    )
    parser.add_argument(
        "--ga-output-root",
        default="ga_runs",
        help="GA 运行时用于存放各次仿真输出的根目录",
    )
    args = parser.parse_args()

    weights = tuple(args.weights)

    design_to_report: Optional[ActuatorDesignVariables] = None

    if args.mode == "demo":
        sample_design = ActuatorDesignVariables(
            lm=5.0,
            tm=0.45,
            ta=0.5,
            dg=0.45,
            hs=1.6,
            wslot=2.4,
            hslot=1.1,
            s=1.2,
            wa=2.2,
        )
        result = evaluate_design_fitness(
            sample_design,
            weight_factors=weights,
            project_name=args.project,
            design_name=args.design,
            setup_name=args.setup,
            output_dir=args.output_dir,
        )
        if result["status"] == "ok":
            design_to_report = sample_design
        else:
            for item in result.get("errors", []):
                print(item)
    else:
        ga_cfg = GAConfig(
            population_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            mutation_sigma=args.mutation_sigma,
            crossover_rate=args.crossover_rate,
            tournament_k=args.tournament_k,
            minimize=not args.maximize,
            seed=args.seed,
            min_valid_evals=args.min_valid_evals,
        )
        bounds = DesignVariableBounds()
        ga_result = run_genetic_optimization(
            project_name=args.project,
            design_name=args.design,
            setup_name=args.setup,
            weight_factors=weights,
            bounds=bounds,
            ga_config=ga_cfg,
            output_root=args.ga_output_root,
        )
        best = ga_result["best_record"]
        design_to_report = best["design"]
        if best["result"].get("status") != "ok":
            for item in best["result"].get("errors", []):
                print(item)

        plots_dir = os.path.join(args.ga_output_root, "plots")
        fitness_plot_path = os.path.join(plots_dir, "fitness_convergence.png")
        save_fitness_convergence_plot(ga_result["history"], fitness_plot_path)

    if design_to_report is not None:
        print(f"设计变量: {design_to_report}")

    elapsed = time.perf_counter() - start_time
    print(f"脚本总运行时间: {elapsed:.2f} s")


# python maxwell_pyaedt_run.py --mode ga --population 10 --generations 5 --ga-output-root ga_runs --project PyAEDT_Project --design Maxwell3DDesign_PyAEDT --setup Setup1