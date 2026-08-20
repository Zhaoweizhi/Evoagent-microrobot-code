"""
E型磁芯仿真脚本（无I片）
基于 maxwell_pyaedt_run.py，删除了 Rectangle1 (I片) 相关代码
用于生成训练集数据
"""
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
    """E型磁芯仿真配置（无I片）"""
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

    # ========== 已删除 rect1_origin 和 rect1_sizes（I片相关） ==========

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
        # Keep n2 conservative and consistent with the EI-core workflow.
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


def run_maxwell_with_pyaedt_E_only(
    config: MaxwellSimulationConfig,
    project_name: str = "PyAEDT_E_Only_Project",
    design_name: str = "Maxwell3DDesign_E_Only",
    setup_name: str = "Setup1",
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    E型磁芯仿真（无I片版本）
    
    使用 PyAEDT 自动完成：
    1. 启动 AEDT / Maxwell
    2. 创建 Maxwell 3D 工程与设计（磁静态）
    3. 建一个简单的几何模型并赋材料（只有 E型磁芯 + 永磁体，无I片）
    4. 创建求解设置并运行仿真
    5. 使用 FieldsReporter 将 Box1 平均 Bz 写入 .fld
    6. 分别计算上轭和中间块区域的平均|B|（用于磁饱和判断）

    返回 (avg_B, b_mean_ta, b_mean_tb, result_source, result_desc)。
    - b_mean_ta: 上轭区域平均|B| (T)
    - b_mean_tb: 中间块区域平均|B| (T)
    """

    output_dir = os.path.abspath(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    avg_B = None
    b_mean_ta = None
    b_mean_tb = None
    result_source = None
    result_desc = None

    # 默认保持"静默+运行完自动关闭"，避免弹出 GUI 干扰。
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
        segment_types = ["Line"] * (len(config.polyline1_points) - 1)
        polyline1 = app.modeler.create_polyline(
            points=config.polyline1_points,
            segment_type=segment_types,
            cover_surface=True,
            close_surface=True,
            name="Polyline1",
            material="vacuum",
        )

        # ========== 已删除 Rectangle1（I片）创建代码 ==========

        # 1.3 Rectangle2（上永磁体投影，稍后再通过厚度和材料区分）
        rect2 = app.modeler.create_rectangle(
            origin=config.rect2_origin,
            sizes=config.rect2_sizes,
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

        # 1.5 沿 X 向正方向扫掠（已删除 Rectangle1）
        for obj_name in ["Polyline1", "Rectangle2", "Rectangle3"]:  # 删除了 "Rectangle1"
            if obj_name in app.modeler.object_names:
                app.modeler.sweep_along_vector(
                    assignment=obj_name,
                    sweep_vector=config.sweep_vector,
                )

        # 1.6 按 12.py 赋材料（已删除 Rectangle1 的材料赋予）
        # - Polyline1             → DW310_35（E型磁芯）
        # - Rectangle2            → NdFe35      （上永磁体，+z 矫顽力）
        # - Rectangle3            → NdFe35 - Copy（下永磁体，-z 矫顽力）
        app.assign_material("Polyline1", "DW310_35")
        # ========== 已删除 Rectangle1 材料赋予 ==========
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
        z_tb_start = config.ta + config.tm + config.dg
        z_ta_start = z_tb_start + config.tb + config.tm + config.dg
        
        # 1.8.1 上轭区域检测面 BsatPlane_ta
        try:
            if "BsatPlane_ta" in app.modeler.object_names:
                app.modeler.delete("BsatPlane_ta")
            bsat_plane_ta = app.modeler.create_rectangle(
                origin=[config._mm(0), config._mm(config.ta), config._mm(z_ta_start)],
                sizes=[config._mm(config.ta), config._mm(config.wa)],
                orientation="XZ",
                name="BsatPlane_ta",
                material="vacuum",
            )
            try:
                bsat_plane_ta.model = False
            except Exception:
                pass
        except Exception as e:
            print("创建 BsatPlane_ta（上轭）失败：", repr(e))
        
        # 1.8.2 中间块区域检测面 BsatPlane_tb
        try:
            if "BsatPlane_tb" in app.modeler.object_names:
                app.modeler.delete("BsatPlane_tb")
            bsat_plane_tb = app.modeler.create_rectangle(
                origin=[config._mm(0), config._mm(config.ta), config._mm(z_tb_start)],
                sizes=[config._mm(config.tb), config._mm(config.wa)],
                orientation="XZ",
                name="BsatPlane_tb",
                material="vacuum",
            )
            try:
                bsat_plane_tb.model = False
            except Exception:
                pass
        except Exception as e:
            print("创建 BsatPlane_tb（中间块）失败：", repr(e))

        # 1.9 外部 Region（与 12.py 类似，使用 40% padding）
        region = app.modeler.create_region(pad_percent=[40, 40, 40, 40, 40, 40])

        # 1.10 网格长度控制（已删除 Rectangle1 的网格设置）
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

            # Length2：对核心铁心、永磁体及 Region 进行细化（已删除 Rectangle1）
            mesh_targets = []
            for name in ["Polyline1", "Rectangle2", "Rectangle3"]:  # 删除了 "Rectangle1"
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
        try:
            app.mesh.generate_mesh(setup_name)
        except Exception as e:
            print("显式生成网格失败，将在求解时自动生成网格。具体信息：", repr(e))

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
            # 分别计算上轭和中间块区域的平均|B|
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
    """
    if not fld_path:
        raise ValueError("fld_path 不能为空。")

    fld_full_path = os.path.abspath(fld_path)
    os.makedirs(os.path.dirname(fld_full_path), exist_ok=True)

    ofields = app.odesign.GetModule("FieldsReporter")
    
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


def evaluate_design_fitness_E_only(
    design: ActuatorDesignVariables,
    weight_factors: Sequence[float] = (0.5, 0.5, 4.0, 1.0),
    penalty_value: float = PENALTY_FITNESS,
    project_name: str = "PyAEDT_E_Only_Project",
    design_name: str = "Maxwell3DDesign_E_Only",
    setup_name: str = "Setup1",
    output_dir: str = ".",
) -> Dict[str, object]:
    """E型磁芯评估函数（无I片版本）"""

    pre_errors = design.validate_without_B()
    sim_config = design.to_sim_config(output_dir=output_dir)
    avg_B, b_mean_ta, b_mean_tb, result_source, result_desc = run_maxwell_with_pyaedt_E_only(
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

    avg_B = abs(avg_B)
    b_mean_ta = abs(b_mean_ta)
    b_mean_tb = abs(b_mean_tb)

    SATURATION_THRESHOLD = 2.0
    is_saturated_ta = b_mean_ta >= SATURATION_THRESHOLD
    is_saturated_tb = b_mean_tb >= SATURATION_THRESHOLD
    is_saturated = is_saturated_ta or is_saturated_tb
    
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

    b_sat = max(b_mean_ta, b_mean_tb)
    
    post_errors, _, _ = design.validate_post_sim(b_sat)
    
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

    volume_refer = 6.7e-8
    mass_refer = 3.79e-4
    kb_refer = 0.239
    pb_refer = 0.819

    volume_r = metrics["volume"] / volume_refer
    mass_r = metrics["mass_total"] / mass_refer
    kb_r = metrics["kb"] / kb_refer if kb_refer else 0.0
    pb_r = metrics["pb"] / pb_refer if pb_refer else 0.0

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
        "B_sat": b_sat,
        "B_mean_ta": b_mean_ta,
        "B_mean_tb": b_mean_tb,
        "is_saturated": is_saturated,
        "is_saturated_ta": is_saturated_ta,
        "is_saturated_tb": is_saturated_tb,
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
    
    if is_saturated:
        warning_parts = []
        if is_saturated_ta:
            warning_parts.append(f"上轭区域平均|B|={b_mean_ta:.3f}T≥2.0T")
        if is_saturated_tb:
            warning_parts.append(f"中间块区域平均|B|={b_mean_tb:.3f}T≥2.0T")
        result["saturation_warning"] = f"磁饱和提示：{'; '.join(warning_parts)}。{saturation_suggestion or ''}"
    
    return result


if __name__ == "__main__":
    """测试 E型磁芯仿真（无I片）"""
    import argparse

    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="运行 E型磁芯 Maxwell 仿真（无I片）")
    parser.add_argument("--project", default="PyAEDT_E_Only_Project")
    parser.add_argument("--design", default="Maxwell3DDesign_E_Only")
    parser.add_argument("--setup", default="Setup1")
    parser.add_argument(
        "--weights",
        nargs=4,
        type=float,
        default=[0.5, 0.5, 4.0, 1.0],
        metavar=("v1", "v2", "v3", "v4"),
        help="对应公式中 v1~v4 的权重",
    )
    parser.add_argument("--output-dir", default="maxwell_runs/e_only_test")
    args = parser.parse_args()

    weights = tuple(args.weights)

    # 使用一个示例设计参数进行测试
    sample_design = ActuatorDesignVariables(
        lm=5.0,
        tm=0.45,
        ta=0.5,
        dg=0.45,
        hs=1.6,
        wslot=2.4,
        hslot=1.1,
        s=1.2,
        tb_ratio=1.8,
    )
    
    print("=" * 60)
    print("E型磁芯仿真测试（无I片）")
    print("=" * 60)
    print(f"设计参数: {sample_design}")
    print(f"派生参数: la={sample_design.la:.2f}, ha={sample_design.ha:.2f}, tb={sample_design.tb:.2f}")
    print("=" * 60)
    
    result = evaluate_design_fitness_E_only(
        sample_design,
        weight_factors=weights,
        project_name=args.project,
        design_name=args.design,
        setup_name=args.setup,
        output_dir=args.output_dir,
    )
    
    print("\n仿真结果:")
    print("-" * 40)
    if result["status"] == "ok":
        print(f"状态: 成功")
        print(f"Fitness: {result['fitness']:.6f}")
        print(f"avg_B: {result['avg_B']:.6f} T")
        print(f"B_mean_ta (上轭): {result['B_mean_ta']:.6f} T")
        print(f"B_mean_tb (中间块): {result['B_mean_tb']:.6f} T")
        print(f"kb: {result['kb']:.6f}")
        print(f"pb: {result['pb']:.6f}")
        if result.get("is_saturated"):
            print(f"饱和警告: {result.get('saturation_warning')}")
    else:
        print(f"状态: {result['status']}")
        for err in result.get("errors", []):
            print(f"  - {err}")

    elapsed = time.perf_counter() - start_time
    print(f"\n脚本总运行时间: {elapsed:.2f} s")
