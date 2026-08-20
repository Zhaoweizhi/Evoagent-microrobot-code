# -*- coding: utf-8 -*-
"""
基线优化器基类
提供参数边界、CSV记录、仿真调用等公共功能
"""
import csv
import json
import os
import sys
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Resolve the project root (paper_code_release/) regardless of nesting depth.
# This file lives at:  <project_root>/case_actuator/baselines/base_optimizer.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loguru import logger
import random

# 参数边界（用户指定的范围）
# 注意：部分约束会进一步限制有效范围
# - 约束(13): s >= 1.0 → s 实际有效范围 [1.0, 1.2]
# - 约束(26): wslot > wa + 0.2 = 2.2 → wslot 实际有效范围 (2.2, 2.8]
# - twall = (hs - hslot) / 2，范围 [0.12, 0.35]
PARAM_BOUNDS = {
    "lm": (0.0, 6.0),       # 用户指定 [0, 6]
    "tm": (0.3, 0.5),       # 用户指定 [0.3, 0.5]
    "ta": (0.35, 0.65),     # 用户指定 [0.35, 0.65]
    "dg": (0.3, 0.65),      # 用户指定 [0.3, 0.65]
    "hs": (1.2, 2.2),       # 用户指定 [1.2, 2.2]
    "wslot": (2.0, 2.8),    # 用户指定 [2.0, 2.8]
    "hslot": (0.8, 1.3),    # 用户指定 [0.8, 1.3]
    "s": (0.8, 1.2),        # 用户指定 [0.8, 1.2]
    "tb_ratio": (1.5, 2.0), # 用户指定 [1.5, 2]
}

# twall 目标范围 [0.12, 0.35]（用于约束感知采样）
TWALL_BOUNDS = (0.12, 0.35)

# 重采样最大尝试次数（用户指定）
MAX_RESAMPLE_ATTEMPTS = 50_000

# 参数名称顺序
PARAM_NAMES = ["lm", "tm", "ta", "dg", "hs", "wslot", "hslot", "s", "tb_ratio"]

# 违反约束时的惩罚 fitness
PENALTY_FITNESS = 1e6

# 默认收敛参数（与 LLM 优化一致/可按实验需要调整）
# 注意：min_iterations 语义为“总评估轮次（含约束违规）达到该值后才允许触发早停/收敛”
DEFAULT_MIN_ITERATIONS = 200
DEFAULT_CONVERGENCE_WINDOW = 40
DEFAULT_AVG_WINDOW = 10
DEFAULT_CONVERGENCE_THRESHOLD = 0.01

# AEDT 周期性清理（解决长跑卡顿/端口不释放）
# - 默认每 50 次“通过约束检查并尝试启动仿真”的评估后清理一次
# - 可通过环境变量覆盖：AEDT_CLEANUP_INTERVAL / AEDT_CLEANUP_COOLDOWN
# - AEDT_CLEANUP_INTERVAL <= 0 表示禁用
DEFAULT_AEDT_CLEANUP_INTERVAL = int(os.getenv("AEDT_CLEANUP_INTERVAL", "10"))
DEFAULT_AEDT_CLEANUP_COOLDOWN = float(os.getenv("AEDT_CLEANUP_COOLDOWN", "5"))

# CSV 字段（与 LLM 优化结果一致）
CSV_FIELDS = [
    "iteration", "status", "fitness", "avg_B", "B_sat", "kb", "pb",
    "volume_r", "mass_r", "kb_r", "pb_r", "volume", "mass_total",
    "mass_mover", "mass_stator", "la", "ha", "ws", "ls", "tb", "twall",
    "lm", "tm", "ta", "dg", "hs", "wslot", "hslot", "s", "wa", "tb_ratio",
    "n1", "n2", "total_turns", "result_source", "result_description",
    "fld_file", "fld_bsat_file", "errors",
    "B_mean_ta", "B_mean_tb", "is_saturated_ta", "is_saturated_tb",
    "saturation_region", "saturation_suggestion",
    "algorithm", "eval_count"  # 额外字段：算法名称、评估次数
]


@dataclass
class EvalResult:
    """单次评估结果"""
    params: Dict[str, float]
    status: str  # "ok" or "constraint_violation"
    fitness: float
    raw_result: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    eval_count: int = 0
    
    def to_csv_row(self, iteration: int, algorithm: str) -> Dict[str, Any]:
        """转换为 CSV 行"""
        row = {"iteration": iteration, "algorithm": algorithm, "eval_count": self.eval_count}
        row["status"] = self.status
        row["fitness"] = self.fitness
        row["errors"] = json.dumps(self.errors, ensure_ascii=False) if self.errors else ""
        
        # 设计参数
        for key in PARAM_NAMES:
            row[key] = self.params.get(key, "")
        row["wa"] = self.params.get("wa", 2.0)
        
        # 仿真结果
        if self.raw_result:
            # 直接字段
            for key in ["avg_B", "B_sat", "kb", "pb", "volume", "mass_total", "mass_mover", 
                       "mass_stator", "result_source", "result_description",
                       "fld_file", "B_mean_ta", "B_mean_tb",
                       "is_saturated_ta", "is_saturated_tb", "saturation_region",
                       "saturation_suggestion"]:
                row[key] = self.raw_result.get(key, "")
            
            # fld_bsat_file: Maxwell 返回的是 fld_bsat_ta_file，映射到 CSV 的 fld_bsat_file
            row["fld_bsat_file"] = self.raw_result.get("fld_bsat_ta_file", "")
            
            # 嵌套字段：derived_dimensions
            derived = self.raw_result.get("derived_dimensions", {})
            for key in ["la", "ha", "ws", "ls", "tb", "twall"]:
                row[key] = derived.get(key, "")
            
            # 嵌套字段：turns
            turns = self.raw_result.get("turns", {})
            row["n1"] = turns.get("n1", "")
            row["n2"] = turns.get("n2", "")
            row["total_turns"] = turns.get("total", "")
            
            # 计算相对值（参考值来自优化目标）
            # 参考值：volume_ref=5e-8, mass_ref=3e-4, kb_ref=0.3, pb_ref=1.5
            volume = self.raw_result.get("volume")
            mass = self.raw_result.get("mass_total")
            kb = self.raw_result.get("kb")
            pb = self.raw_result.get("pb")
            
            row["volume_r"] = volume / 5e-8 if volume else ""
            row["mass_r"] = mass / 3e-4 if mass else ""
            row["kb_r"] = kb / 0.3 if kb else ""
            row["pb_r"] = pb / 1.5 if pb else ""
        
        return row


def _quick_validate_params(params: Dict[str, float]) -> List[str]:
    """
    快速验证参数是否满足主要约束（不调用 Maxwell）
    返回错误列表，空列表表示通过
    
    参考约束（来自 maxwell_pyaedt_run.py）：
    - 约束(11): 2dg + tb - hs >= 0.1
    - 约束(13): s >= 1.0
    - 约束(14): ws < 4
    - 约束(15): ha < 5
    - 约束(16): la <= 6
    - 约束(18): ls = lm - s > 0
    - 约束(21): 0.3 < ta < 1.0
    - 约束(22): tb in [1.5*ta, 2.5*ta]（由 tb_ratio 保证）
    - 约束(25): hslot - tb >= 0.2
    - 约束(26): wslot - wa >= 0.2
    - 约束(27): dg - twall >= 0.02
    - 约束(28): n2 >= 3
    - 约束(29): hslot - tb >= 0.1（已包含在约束25中）
    """
    errors = []
    
    lm = params.get("lm", 0)
    tm = params.get("tm", 0)
    ta = params.get("ta", 0)
    dg = params.get("dg", 0)
    hs = params.get("hs", 0)
    wslot = params.get("wslot", 0)
    hslot = params.get("hslot", 0)
    s = params.get("s", 0)
    tb_ratio = params.get("tb_ratio", 1.75)
    wa = 2.0  # 固定值
    wcoil = 0.05  # 固定值
    
    # 派生参数（与 maxwell_pyaedt_run.py 一致）
    # tb = _clamp(tb_ratio * ta, 1.6 * ta, 2.0 * ta)，但用户范围是 [1.5, 2]
    tb = max(1.5 * ta, min(tb_ratio * ta, 2.0 * ta))
    twall = 0.5 * max(hs - hslot, 0.0)
    ls = lm - s
    la = lm + 2 * ta
    ha = 2 * ta + tb + 2 * tm + 2 * dg
    ws = wslot + 2 * twall
    
    # n2 计算（与 maxwell_pyaedt_run.py 一致：向下取整）
    n2 = max(1, int(0.9 * twall / wcoil)) if wcoil > 0 and twall > 0 else 0
    
    clearance = 0.02  # 20 μm
    
    # 基本正值检查
    if hs <= 0 or wslot <= 0 or twall <= 0:
        errors.append("线圈外形尺寸必须为正值。")
    
    if twall <= 0:
        errors.append("线圈壁厚 (twall) 必须为正值。")
    
    # twall 范围检查 [0.12, 0.35]
    if twall < 0.12 or twall > 0.35:
        errors.append(f"twall={twall:.3f} 超出范围 [0.12, 0.35]。")
    
    # 约束(11): 2dg + tb - hs >= 0.1
    if (2 * dg + tb - hs) < 0.1:
        errors.append(f"约束(11) 不满足：2dg + tb - hs ≥ 0.1mm。")
    
    # 约束(28): n2 >= 3
    if n2 < 3:
        errors.append(f"约束(28) 不满足：n2={n2} < 3 匝。")
    
    # 约束(25): hslot - tb >= 0.2
    if (hslot - tb) < 0.2:
        errors.append(f"约束(25) 不满足：hslot - tb ≥ 0.2mm。")
    
    # 约束(26): wslot - wa >= 0.2
    if (wslot - wa) < 0.2:
        errors.append(f"约束(26) 不满足：wslot - wa ≥ 0.2mm。")
    
    # 约束(27): dg - twall >= 0.02
    if (dg - twall) < clearance:
        errors.append(f"约束(27) 不满足：dg - twall ≥ 0.02mm。")
    
    # 约束(13): s >= 1
    if s < 1.0:
        errors.append(f"约束(13) 不满足：行程 s >= 1mm。")
    
    # 约束(14): ws < 4
    if ws >= 4.0:
        errors.append(f"约束(14) 不满足：ws < 4mm。")
    
    # 约束(15): ha < 5
    if ha >= 5.0:
        errors.append(f"约束(15) 不满足：ha < 5mm。")
    
    # 约束(16): la <= 6
    if la > 6.0:
        errors.append(f"约束(16) 不满足：la ≤ 6mm。")
    
    # 约束(21): 0.3 < ta < 1.0（用户范围 [0.35, 0.65] 已满足）
    if not (0.3 < ta < 1.0):
        errors.append(f"约束(21) 不满足：0.3mm < ta < 1mm。")
    
    # 约束(18): ls > 0
    if ls <= 0:
        errors.append(f"约束(18) 不满足：ls = lm - s > 0。")
    
    # 约束(22): tb in [1.5*ta, 2.5*ta]
    if tb < 1.5 * ta or tb > 2.5 * ta:
        errors.append(f"约束(22) 不满足：tb 未处于 [1.5ta, 2.5ta]。")
    
    # wslot > wa
    if wslot <= wa:
        errors.append("wslot 必须大于 wa 以保证装配间隙。")
    
    return errors


def generate_valid_params(seed: Optional[int] = None) -> Tuple[Dict[str, float], int]:
    """
    生成满足约束的参数组合（使用重采样机制）
    
    参数范围（用户指定）：
    - lm: [0, 6], tm: [0.3, 0.5], ta: [0.35, 0.65], dg: [0.3, 0.65]
    - hs: [1.2, 2.2], wslot: [2.0, 2.8], hslot: [0.8, 1.3], s: [0.8, 1.2]
    - tb_ratio: [1.5, 2], twall: [0.12, 0.35]（派生参数）
    
    Returns:
        (params, attempts): 参数字典和尝试次数
    """
    rng = random.Random(seed)
    
    for attempt in range(MAX_RESAMPLE_ATTEMPTS):
        # 1. 先采样 ta 和 tb_ratio
        ta = rng.uniform(PARAM_BOUNDS["ta"][0], PARAM_BOUNDS["ta"][1])
        tb_ratio = rng.uniform(PARAM_BOUNDS["tb_ratio"][0], PARAM_BOUNDS["tb_ratio"][1])
        tb = tb_ratio * ta
        
        # 2. lm 需要满足约束(16): la = lm + 2*ta <= 6
        lm_max = min(PARAM_BOUNDS["lm"][1], 6.0 - 2 * ta)
        lm_min = max(PARAM_BOUNDS["lm"][0], 1.0)  # lm 需要足够大以满足 ls > 0
        if lm_max < lm_min:
            continue
        lm = rng.uniform(lm_min, lm_max)
        
        # 3. s 需要满足约束(13): s >= 1.0 和约束(18): ls = lm - s > 0
        s_min = max(PARAM_BOUNDS["s"][0], 1.0)  # 约束(13)
        s_max = min(PARAM_BOUNDS["s"][1], lm - 0.1)  # 约束(18)
        if s_max < s_min:
            continue
        s = rng.uniform(s_min, s_max)
        
        # 4. 采样 twall（目标范围 [0.12, 0.35]），然后推导 hs 和 hslot
        twall = rng.uniform(TWALL_BOUNDS[0], TWALL_BOUNDS[1])
        
        # 5. hslot 需要满足约束(25): hslot - tb >= 0.2
        hslot_min = max(PARAM_BOUNDS["hslot"][0], tb + 0.2)
        if hslot_min > PARAM_BOUNDS["hslot"][1]:
            continue
        hslot = rng.uniform(hslot_min, PARAM_BOUNDS["hslot"][1])
        
        # 6. hs = hslot + 2 * twall（由约束(24)决定）
        hs = hslot + 2 * twall
        if hs < PARAM_BOUNDS["hs"][0] or hs > PARAM_BOUNDS["hs"][1]:
            continue
        
        # 7. dg 需要满足：
        # - 约束(27): dg - twall >= 0.02
        # - 约束(11): 2*dg + tb - hs >= 0.1 → dg >= (hs - tb + 0.1) / 2
        dg_min_27 = twall + 0.02
        dg_min_11 = (hs - tb + 0.1) / 2
        dg_min = max(PARAM_BOUNDS["dg"][0], dg_min_27, dg_min_11)
        if dg_min > PARAM_BOUNDS["dg"][1]:
            continue
        dg = rng.uniform(dg_min, PARAM_BOUNDS["dg"][1])
        
        # 8. wslot 需要满足约束(26): wslot - wa >= 0.2 (wa=2.0)
        wslot_min = max(PARAM_BOUNDS["wslot"][0], 2.0 + 0.2)
        if wslot_min > PARAM_BOUNDS["wslot"][1]:
            continue
        wslot = rng.uniform(wslot_min, PARAM_BOUNDS["wslot"][1])
        
        # 9. tm 没有强耦合约束
        tm = rng.uniform(PARAM_BOUNDS["tm"][0], PARAM_BOUNDS["tm"][1])
        
        params = {
            "lm": round(lm, 2),
            "tm": round(tm, 2),
            "ta": round(ta, 2),
            "dg": round(dg, 2),
            "hs": round(hs, 2),
            "wslot": round(wslot, 2),
            "hslot": round(hslot, 2),
            "s": round(s, 2),
            "tb_ratio": round(tb_ratio, 2),
        }
        
        # 快速验证
        errors = _quick_validate_params(params)
        if not errors:
            return params, attempt + 1
    
    # 如果多次尝试后仍然失败，抛出异常
    raise RuntimeError(f"在 {MAX_RESAMPLE_ATTEMPTS} 次尝试后仍未找到满足约束的参数组合")


class BaseOptimizer(ABC):
    """基线优化器基类"""
    
    def __init__(
        self,
        max_evals: int = 200,
        min_iterations: int = DEFAULT_MIN_ITERATIONS,
        convergence_window: int = DEFAULT_CONVERGENCE_WINDOW,
        avg_window: int = DEFAULT_AVG_WINDOW,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        output_dir: str = ".",
        algorithm_name: str = "baseline",
        seed: int = 42,
    ):
        """
        Args:
            max_evals: 最大仿真评估次数
            min_iterations: 最小迭代次数（收敛判断前至少运行这么多轮）
            convergence_window: 连续无改进窗口（连续多少轮无改进视为收敛）
            avg_window: 早停平均窗口
            convergence_threshold: 收敛阈值（改进比例小于此值视为无改进）
            output_dir: 输出目录
            algorithm_name: 算法名称
            seed: 随机种子
        """
        self.max_evals = max_evals
        self.min_iterations = min_iterations
        self.convergence_window = convergence_window
        self.avg_window = avg_window
        self.convergence_threshold = convergence_threshold
        self.output_dir = Path(output_dir)
        self.algorithm_name = algorithm_name
        self.seed = seed
        
        # 评估计数器
        self.eval_count = 0
        
        # 结果记录
        self.results: List[EvalResult] = []
        self.best_result: Optional[EvalResult] = None
        self.best_fitness = float("inf")
        
        # 收敛检测
        self.valid_fitness_history: List[float] = []  # 有效 fitness 历史
        self.no_improvement_count = 0  # 连续无改进计数
        self.converged = False  # 是否已收敛
        self.convergence_reason = ""  # 收敛原因

        # 周期性清理 AEDT 进程，缓解长跑卡顿/端口占用
        self.aedt_cleanup_interval = DEFAULT_AEDT_CLEANUP_INTERVAL
        self.aedt_cleanup_cooldown = DEFAULT_AEDT_CLEANUP_COOLDOWN
        
        # ★邮件通知配置（由 run_baselines.py 注入）
        self._notify_email: Optional[str] = None
        self._smtp_server: str = "smtp.qq.com"
        self._smtp_port: int = 587
        self._smtp_password: Optional[str] = None
        
        # ★超时/卡住检测
        self._last_progress_time: float = 0.0  # 上次有进展的时间戳
        self._stall_warning_sent: bool = False  # 是否已发送卡住警告
        self._stall_threshold_seconds: float = 600.0  # 10 分钟无进展视为卡住
        
        # CSV 文件
        self.csv_path: Optional[Path] = None
        self.csv_writer = None
        self.csv_file = None
        self.resume_csv = False
        
        # 导入仿真函数（延迟导入）
        self._validate_func = None
        self._simulate_func = None
    
    def _lazy_import(self):
        """延迟导入仿真函数"""
        if self._validate_func is None:
            from mymcp.tool import validate_maxwell_design, run_maxwell_simulation
            self._validate_func = validate_maxwell_design
            self._simulate_func = run_maxwell_simulation
    
    def _send_notification_email(self, subject: str, body: str) -> bool:
        """发送邮件通知（失败不抛异常）"""
        if not self._notify_email or not self._smtp_password:
            return False
        
        import smtplib
        from email.mime.text import MIMEText
        from email.header import Header
        
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = Header(subject, "utf-8")
            msg["From"] = self._notify_email
            msg["To"] = self._notify_email
            
            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                server.starttls()
                server.login(self._notify_email, self._smtp_password)
                server.sendmail(self._notify_email, [self._notify_email], msg.as_string())
            
            logger.info(f"[{self.algorithm_name}] 邮件通知已发送: {subject}")
            return True
        except Exception as e:
            logger.warning(f"[{self.algorithm_name}] 邮件发送失败: {e}")
            return False
    
    def _check_stall_and_notify(self) -> None:
        """检查是否长时间无进展，如果是则发送警告邮件（只发一次）"""
        import time
        
        if not self._notify_email or not self._smtp_password:
            return
        
        if self._stall_warning_sent:
            return
        
        current_time = time.time()
        if self._last_progress_time <= 0:
            self._last_progress_time = current_time
            return
        
        stall_duration = current_time - self._last_progress_time
        if stall_duration >= self._stall_threshold_seconds:
            self._stall_warning_sent = True
            stall_minutes = stall_duration / 60
            self._send_notification_email(
                subject=f"Baseline WARNING - {self.algorithm_name} Stalled ({stall_minutes:.1f} min)",
                body=(
                    f"Baseline optimization appears to be stalled.\n\n"
                    f"Algorithm: {self.algorithm_name}\n"
                    f"Eval count: {self.eval_count}\n"
                    f"Best fitness: {self.best_fitness:.6f}\n"
                    f"No progress for: {stall_minutes:.1f} minutes\n"
                    f"CSV: {self.csv_path}\n\n"
                    f"This may indicate Maxwell simulation is hung."
                ),
            )
    
    def _update_progress_time(self) -> None:
        """更新最后进展时间戳，并重置卡住警告标志"""
        import time
        self._last_progress_time = time.time()
        self._stall_warning_sent = False
    
    def _init_csv(self):
        """初始化 CSV 文件"""
        if self.csv_writer is not None:
            return

        if self.csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.csv_path = self.output_dir / f"{self.algorithm_name}_results_{timestamp}.csv"
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
            self.csv_writer.writeheader()
            self.csv_file.flush()
            logger.info(f"CSV 文件已创建: {self.csv_path}")
            return

        # 断点续跑：复用已有 CSV
        if self.resume_csv and self.csv_path.exists():
            self.csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
            self.csv_file.flush()
            logger.info(f"CSV 文件续写: {self.csv_path}")
            return

        # 指定了路径但不是续写时，仍写入新文件
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        self.csv_writer.writeheader()
        self.csv_file.flush()
        logger.info(f"CSV 文件已创建: {self.csv_path}")
    
    def _write_csv_row(self, result: EvalResult, iteration: int):
        """写入一行 CSV"""
        if self.csv_writer is None:
            self._init_csv()
        row = result.to_csv_row(iteration, self.algorithm_name)
        self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def _close_csv(self):
        """关闭 CSV 文件"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
    
    def evaluate(self, params: Dict[str, float]) -> EvalResult:
        """
        评估一组参数（同步版本）
        
        健壮模式：任何异常都会被捕获并返回惩罚 fitness，保证程序能继续运行。
        
        Args:
            params: 设计参数字典
        
        Returns:
            EvalResult: 评估结果（即使出错也会返回带惩罚值的结果）
        """
        try:
            return asyncio.run(self.evaluate_async(params))
        except Exception as e:
            # asyncio.run 本身也可能抛异常（如事件循环问题）
            self.eval_count += 1
            logger.error(f"[{self.algorithm_name}] #{self.eval_count} 评估执行异常: {e}，跳过此轮")
            self._cleanup_aedt_processes()
            return EvalResult(
                params=params.copy(),
                status="execution_error",
                fitness=PENALTY_FITNESS,
                errors=[f"评估执行异常: {str(e)}"],
                eval_count=self.eval_count,
            )
    
    async def evaluate_async(self, params: Dict[str, float]) -> EvalResult:
        """
        评估一组参数（异步版本）
        
        健壮模式：任何异常都会被捕获并返回惩罚 fitness，保证程序能继续运行。
        
        Args:
            params: 设计参数字典
        
        Returns:
            EvalResult: 评估结果（即使出错也会返回带惩罚值的结果）
        """
        self._lazy_import()
        self.eval_count += 1
        current_eval = self.eval_count
        
        # 整个评估过程都包在 try-except 中，确保任何异常都能被捕获
        validate_ok = False
        simulation_started = False
        cleanup_done = False
        try:
            # 1. 约束检查
            try:
                validate_result_str = await self._validate_func(
                    lm=params["lm"],
                    tm=params["tm"],
                    ta=params["ta"],
                    dg=params["dg"],
                    hs=params["hs"],
                    wslot=params["wslot"],
                    hslot=params["hslot"],
                    s=params["s"],
                    tb_ratio=params["tb_ratio"],
                )
                validate_result = json.loads(validate_result_str)
            except Exception as e:
                logger.error(f"[{self.algorithm_name}] #{current_eval} 约束检查异常: {e}")
                return EvalResult(
                    params=params.copy(),
                    status="validate_error",
                    fitness=PENALTY_FITNESS,
                    errors=[f"约束检查异常: {str(e)}"],
                    eval_count=current_eval,
                )
            
            if validate_result.get("status") != "ok":
                # 违反约束
                errors = validate_result.get("errors", [])
                result = EvalResult(
                    params=params.copy(),
                    status="constraint_violation",
                    fitness=PENALTY_FITNESS,
                    errors=errors,
                    eval_count=current_eval,
                )
                logger.warning(f"[{self.algorithm_name}] #{current_eval} 违反约束: {errors[:2]}...")
                return result

            validate_ok = True

            # 周期性清理（在仿真启动前，避免长期残留导致卡顿）
            if (
                isinstance(self.aedt_cleanup_interval, int)
                and self.aedt_cleanup_interval > 0
                and (current_eval % self.aedt_cleanup_interval == 0)
            ):
                logger.warning(
                    f"[{self.algorithm_name}] #{current_eval} 触发预清理 AEDT "
                    f"(interval={self.aedt_cleanup_interval})"
                )
                self._cleanup_aedt_processes()
                cleanup_done = True
                # 给系统/端口一些释放时间
                cooldown = 0.0
                try:
                    cooldown = float(self.aedt_cleanup_cooldown) if self.aedt_cleanup_cooldown else 0.0
                except Exception:
                    cooldown = 0.0
                if cooldown > 0:
                    await asyncio.sleep(cooldown)
            
            # ★检查是否长时间无进展（在开始新仿真前检查）
            self._check_stall_and_notify()
            
            # 2. 运行仿真（带硬超时和异常处理）
            SIMULATION_TIMEOUT = 300  # 5 分钟硬超时
            INTER_SIMULATION_DELAY = 3  # 仿真间隔延迟（秒），让 gRPC 端口有时间释放
            
            # 注意：不要在仿真前清理进程，否则会杀掉正在启动的 AEDT
            # 只在超时/异常后清理
            
            # 等待端口释放，避免 gRPC 端口冲突
            await asyncio.sleep(INTER_SIMULATION_DELAY)
            
            try:
                simulation_started = True
                sim_result_str = await asyncio.wait_for(
                    self._simulate_func(
                        lm=params["lm"],
                        tm=params["tm"],
                        ta=params["ta"],
                        dg=params["dg"],
                        hs=params["hs"],
                        wslot=params["wslot"],
                        hslot=params["hslot"],
                        s=params["s"],
                        tb_ratio=params["tb_ratio"],
                    ),
                    timeout=SIMULATION_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"[{self.algorithm_name}] #{current_eval} 仿真超时，跳过此轮")
                # ★先发送超时邮件通知（在清理前发送，避免清理耗时导致通知延迟）
                self._send_notification_email(
                    subject=f"Baseline WARNING - {self.algorithm_name} Simulation Timeout",
                    body=(
                        f"Simulation timed out.\n\n"
                        f"Algorithm: {self.algorithm_name}\n"
                        f"Eval: #{current_eval}\n"
                        f"Best fitness so far: {self.best_fitness:.6f}\n"
                        f"CSV: {self.csv_path}"
                    ),
                )
                # 再清理 AEDT 进程
                self._cleanup_aedt_processes()
                cleanup_done = True
                return EvalResult(
                    params=params.copy(),
                    status="timeout",
                    fitness=PENALTY_FITNESS,
                    errors=["仿真超时"],
                    eval_count=current_eval,
                )
            except Exception as e:
                logger.error(f"[{self.algorithm_name}] #{current_eval} 仿真调用异常: {e}，跳过此轮")
                self._cleanup_aedt_processes()
                cleanup_done = True
                return EvalResult(
                    params=params.copy(),
                    status="simulation_exception",
                    fitness=PENALTY_FITNESS,
                    errors=[f"仿真调用异常: {str(e)}"],
                    eval_count=current_eval,
                )
            
            # 解析仿真结果
            try:
                sim_response = json.loads(sim_result_str)
                # 仿真返回格式: {"result": {...}, "logs": [...]}
                sim_result = sim_response.get("result", sim_response)  # 兼容两种格式
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"[{self.algorithm_name}] #{current_eval} 仿真结果解析失败: {e}")
                return EvalResult(
                    params=params.copy(),
                    status="parse_error",
                    fitness=PENALTY_FITNESS,
                    errors=[f"仿真结果解析失败: {str(e)}"],
                    eval_count=current_eval,
                )
            
            if sim_result.get("status") != "ok":
                # 仿真失败
                errors = sim_result.get("errors", ["simulation_failed"])
                result = EvalResult(
                    params=params.copy(),
                    status="simulation_error",
                    fitness=PENALTY_FITNESS,
                    errors=errors,
                    raw_result=sim_result,
                    eval_count=current_eval,
                )
                logger.error(f"[{self.algorithm_name}] #{current_eval} 仿真失败: {errors}")
                return result
            
            # 3. 提取 fitness
            fitness = sim_result.get("fitness", PENALTY_FITNESS)
            result = EvalResult(
                params=params.copy(),
                status="ok",
                fitness=fitness,
                raw_result=sim_result,
                eval_count=current_eval,
            )
            
            # ★更新进展时间（成功完成一次仿真）
            self._update_progress_time()
            
            # 更新最优（注意：收敛判断需要用"更新前"的 best_fitness）
            prev_best = self.best_fitness
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_result = result
                logger.info(f"[{self.algorithm_name}] #{current_eval} 🎯 新最优 fitness={fitness:.6f}")
            else:
                logger.info(f"[{self.algorithm_name}] #{current_eval} fitness={fitness:.6f}")
            
            # 检查收敛
            self.check_convergence(fitness, prev_best=prev_best)
            
            return result
            
        except Exception as e:
            # 最外层兜底：捕获所有未预料的异常
            logger.error(f"[{self.algorithm_name}] #{current_eval} 未知异常: {e}，跳过此轮继续运行")
            self._cleanup_aedt_processes()
            cleanup_done = True
            return EvalResult(
                params=params.copy(),
                status="unknown_error",
                fitness=PENALTY_FITNESS,
                errors=[f"未知异常: {str(e)}"],
                eval_count=current_eval,
            )
        finally:
            # 周期性清理（只在“启动过仿真”的评估上计数）
            if (
                simulation_started
                and not cleanup_done
                and isinstance(self.aedt_cleanup_interval, int)
                and self.aedt_cleanup_interval > 0
                and (current_eval % self.aedt_cleanup_interval == 0)
            ):
                logger.warning(
                    f"[{self.algorithm_name}] #{current_eval} 触发周期性 AEDT 清理 "
                    f"(interval={self.aedt_cleanup_interval})"
                )
                self._cleanup_aedt_processes()
                # 给系统/端口一些释放时间，降低下一轮 gRPC 10048 概率
                cooldown = 0.0
                try:
                    cooldown = float(self.aedt_cleanup_cooldown) if self.aedt_cleanup_cooldown else 0.0
                except Exception:
                    cooldown = 0.0
                if cooldown > 0:
                    await asyncio.sleep(cooldown)
    
    def _cleanup_aedt_processes(self):
        """清理残留的 AEDT 进程，防止影响后续仿真"""
        try:
            import subprocess
            # 尝试清理 ansysedt 相关进程
            subprocess.run(
                ["taskkill", "/F", "/IM", "ansysedt.exe"],
                capture_output=True,
                timeout=30
            )
            subprocess.run(
                ["taskkill", "/F", "/IM", "ansysedtsv.exe"],
                capture_output=True,
                timeout=30
            )
            logger.info(f"[{self.algorithm_name}] 已清理残留 AEDT 进程")
        except Exception as e:
            logger.warning(f"[{self.algorithm_name}] 清理 AEDT 进程时出错: {e}")
    
    def check_convergence(self, fitness: float, prev_best: Optional[float] = None) -> bool:
        """
        检查是否收敛
        
        Args:
            fitness: 当前评估的 fitness
            prev_best: 本轮评估前的 best_fitness（用于判断是否“有改进”）
        
        Returns:
            bool: 是否已收敛
        """
        if self.converged:
            return True

        # 在达到最小总评估轮次前，不允许触发收敛/早停
        # 这里使用总评估次数 self.eval_count（包含约束违规与仿真失败），保证“至少跑满 N 轮”
        if self.eval_count < self.min_iterations:
            return False
        
        # 只记录有效 fitness（非惩罚值）
        is_valid = fitness < PENALTY_FITNESS * 0.1
        
        if is_valid:
            self.valid_fitness_history.append(fitness)

            # 使用“更新前”的 best_fitness 来判断是否改进，避免 evaluate_async 先更新 best 导致误判
            if prev_best is None:
                prev_best = self.best_fitness

            # prev_best 可能为 inf（第一次有效解），视为显著改进
            if prev_best == float("inf"):
                self.no_improvement_count = 0
            else:
                threshold_abs = (
                    abs(prev_best) * self.convergence_threshold
                    if prev_best != 0
                    else self.convergence_threshold
                )
                delta = prev_best - fitness
                if delta > threshold_abs:
                    # 有显著改进
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
        
        valid_count = len(self.valid_fitness_history)
        
        # 条件1：连续无改进收敛（总评估轮次达到门槛）
        if self.eval_count >= self.min_iterations and self.no_improvement_count >= self.convergence_window:
            self.converged = True
            self.convergence_reason = f"连续 {self.convergence_window} 轮无显著改进"
            logger.info(f"[{self.algorithm_name}] ✅ 收敛：{self.convergence_reason}")
            logger.info(f"[{self.algorithm_name}] 有效迭代: {valid_count}, 最佳 fitness: {self.best_fitness:.6f}")
            return True
        
        # 条件2：早停（平均窗口无改进）
        if valid_count >= self.avg_window * 2 and self.eval_count >= max(self.min_iterations, self.avg_window * 2):
            recent_avg = sum(self.valid_fitness_history[-self.avg_window:]) / self.avg_window
            prev_avg = sum(self.valid_fitness_history[-2 * self.avg_window:-self.avg_window]) / self.avg_window
            improvement = (prev_avg - recent_avg) / abs(prev_avg) if prev_avg != 0 else 0
            
            if improvement < self.convergence_threshold:
                self.converged = True
                self.convergence_reason = f"早停：近 {self.avg_window} 轮平均改进 {improvement:.4%} < {self.convergence_threshold:.4%}"
                logger.info(f"[{self.algorithm_name}] ✅ {self.convergence_reason}")
                logger.info(f"[{self.algorithm_name}] 有效迭代: {valid_count}, 最佳 fitness: {self.best_fitness:.6f}")
                return True
        
        return False
    
    def params_to_vector(self, params: Dict[str, float]) -> List[float]:
        """参数字典转向量"""
        return [params[name] for name in PARAM_NAMES]
    
    def vector_to_params(self, vector: List[float]) -> Dict[str, float]:
        """向量转参数字典"""
        return {name: vector[i] for i, name in enumerate(PARAM_NAMES)}
    
    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """获取参数边界 (lower, upper)"""
        lower = [PARAM_BOUNDS[name][0] for name in PARAM_NAMES]
        upper = [PARAM_BOUNDS[name][1] for name in PARAM_NAMES]
        return lower, upper
    
    @abstractmethod
    def optimize(self) -> EvalResult:
        """
        执行优化（子类实现）
        
        Returns:
            EvalResult: 最优结果
        """
        pass
    
    def run(self) -> EvalResult:
        """
        运行优化流程
        
        Returns:
            EvalResult: 最优结果
        """
        logger.info(f"=" * 60)
        logger.info(f"[{self.algorithm_name}] 开始优化 | max_evals={self.max_evals} | seed={self.seed}")
        logger.info(f"=" * 60)
        
        try:
            self._init_csv()
            result = self.optimize()
            return result
        finally:
            self._close_csv()
            logger.info(f"[{self.algorithm_name}] 优化完成 | 最优 fitness={self.best_fitness:.6f}")
            logger.info(f"[{self.algorithm_name}] 结果已保存: {self.csv_path}")
