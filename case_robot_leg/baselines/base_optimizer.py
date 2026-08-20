# -*- coding: utf-8 -*-
"""Shared optimizer infrastructure for the robot-leg task.

This module defines the design space, geometry checks, simulation interface,
CSV output and checkpoint handling used by all baseline methods.
"""
import asyncio
import csv
import json
import os
import pickle
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ============================================================
# Design variables and bounds
# ============================================================
PARAM_NAMES = ["m", "n", "alpha", "beta", "DIST_BETTERY"]
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "m":             (0.0, 0.006),
    "n":             (0.0, 0.005),
    "alpha":         (80.0, 130.0),
    "beta":          (50.0, 120.0),
    "DIST_BETTERY":  (0.0, 0.009),
}

PARAM_DECIMALS = {"m": 5, "n": 5, "alpha": 1, "beta": 1, "DIST_BETTERY": 5}

PENALTY_FITNESS = 0.0
CONSTRAINT_VIOL_FITNESS_MARKER = -1e6

# ============================================================
# 7 类几何硬约束分类
# ============================================================
CONSTRAINT_PATTERNS: List[Tuple[str, str]] = [
    ("C1_步幅极限",  r"步幅极限"),
    ("C2_机身触地",  r"机身触地"),
    ("C3_构型奇异",  r"构型奇异"),
    ("C4a_G穿模",    r"G\(后膝关节\)"),
    ("C4b_D穿模",    r"D\(后脚垫\)"),
    ("C4c_F穿模",    r"F\(前脚垫\)"),
    ("C5_高度超限",  r"高度超限"),
    ("C6_D12关节角", r"前腿关节角超限"),
    ("C7_DF跨度",    r"DF跨度超限"),
]

# CSV fields shared by all baseline methods
EVAL_CSV_FIELDS = [
    "iteration", "algorithm", "status", "fitness",
    "Net_Displacement_X", "Displacement_Range_Y", "Net_Displacement_Y",
    "Average_Velocity_X",
    "m", "n", "alpha", "beta", "DIST_BETTERY",
    "D6_rear_angle", "D12_front_angle", "D2_body_pitch",
    "C_y", "H_max", "DF_span",
    "errors", "eval_count", "eval_time_s",
]

PROPOSAL_FIELDS = [
    "proposal_idx", "generation", "source", "status", "fitness", "sim_status",
    "Net_Displacement_X", "Displacement_Range_Y", "Net_Displacement_Y",
    "Average_Velocity_X",
    "m", "n", "alpha", "beta", "DIST_BETTERY",
    "violated_constraints", "errors_raw", "eval_count", "eval_time_s",
    "timestamp",
]


def snap(params: Dict[str, float]) -> Dict[str, float]:
    return {k: round(v, PARAM_DECIMALS.get(k, 5)) for k, v in params.items()}


def clamp(name: str, val: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    return max(lo, min(hi, val))


def random_params(rng: random.Random) -> Dict[str, float]:
    p = {}
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        p[name] = rng.uniform(lo, hi)
    return snap(p)


def classify_errors(errors: List[str]) -> List[str]:
    violated: List[str] = []
    for cname, pat in CONSTRAINT_PATTERNS:
        for err in errors:
            if re.search(pat, err):
                violated.append(cname)
                break
    if errors and not violated:
        violated = ["UNCATEGORIZED"]
    return violated


def validate_quick(params: Dict[str, float]) -> Tuple[bool, List[str]]:
    """调用 _validate_design 检查 7 类几何约束。
    返回 (passed, errors)。
    """
    from mymcp.tool.robot_leg import _validate_design
    result = _validate_design(
        params["m"], params["n"], params["alpha"],
        params["beta"], params["DIST_BETTERY"]
    )
    status = result["status"]
    errors = result.get("errors", [])
    return (status == "ok"), errors


def generate_valid_params(rng: random.Random,
                          max_attempts: int = 5000) -> Tuple[Dict[str, float], int]:
    """生成满足 7 类约束的参数，返回 (params, attempts)"""
    for i in range(max_attempts):
        p = random_params(rng)
        ok, _ = validate_quick(p)
        if ok:
            return p, i + 1
    return random_params(rng), max_attempts


class BaseOptimizer(ABC):
    """机器人腿优化基线基类

    核心职责:
    - 管理 proposals CSV（全量记录，含约束违反）
    - 管理 eval CSV（仿真结果 + 约束违反行穿插）
    - 调用 validate_quick → run_robot_simulation
    - 约束违反计数与分类
    - 断点续跑
    - 收敛判断
    """

    def __init__(
        self,
        max_evals: int = 200,
        output_dir: str = ".",
        algorithm_name: str = "baseline",
        seed: int = 42,
    ):
        self.max_evals = max_evals
        self.output_dir = Path(output_dir)
        self.algorithm_name = algorithm_name
        self.seed = seed

        # 计数器
        self.eval_count = 0         # 成功进入仿真的次数
        self.proposal_counter = 0   # validate_quick 总调用次数（含失败）
        self.current_generation = 0

        # 最优记录
        self.best_fitness = PENALTY_FITNESS
        self.best_params: Optional[Dict[str, float]] = None
        self.min_evals_before_stop = 100
        self.patience_evals = 40
        self.significant_improvement_ratio = 0.01
        self._last_improvement_eval = 0
        self._last_significant_improvement_eval = 0
        self._last_significant_best = PENALTY_FITNESS
        self._early_stop_triggered = False

        # Proposals 日志
        self._proposal_log: List[Dict] = []
        self._pending_ok_proposals: List[Dict] = []

        # CSV 文件句柄
        self._eval_file = None
        self._eval_writer = None
        self._proposal_file = None
        self._proposal_writer = None
        self._eval_csv_path: Optional[Path] = None
        self._proposal_csv_path: Optional[Path] = None
        self._summary_csv_path: Optional[Path] = None

        # 断点保存路径（run() 中可被覆盖）
        self._checkpoint_save_path = self.output_dir / f"checkpoint_{algorithm_name.lower()}.pkl"

    # ============================================================
    # CSV 初始化
    # ============================================================

    def _init_csv(self, timestamp: Optional[str] = None, resume_paths: Optional[Dict] = None):
        """初始化或续写 CSV。
        resume_paths: {"eval_csv": path, "proposal_csv": path, "summary_csv": path}
        """
        if resume_paths:
            self._eval_csv_path = Path(resume_paths["eval_csv"])
            self._proposal_csv_path = Path(resume_paths["proposal_csv"])
            self._summary_csv_path = Path(resume_paths["summary_csv"])

            self._eval_file = open(self._eval_csv_path, "a", encoding="utf-8-sig", newline="")
            self._eval_writer = csv.DictWriter(
                self._eval_file, fieldnames=EVAL_CSV_FIELDS, extrasaction="ignore"
            )
            self._eval_file.flush()

            self._proposal_file = open(self._proposal_csv_path, "a", encoding="utf-8-sig", newline="")
            self._proposal_writer = csv.DictWriter(
                self._proposal_file, fieldnames=PROPOSAL_FIELDS, extrasaction="ignore"
            )
            self._proposal_file.flush()

            print(f"[{self.algorithm_name}] 续写 Eval CSV    : {self._eval_csv_path}")
            print(f"[{self.algorithm_name}] 续写 Proposal CSV: {self._proposal_csv_path}")
            return

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"RobotLeg_{self.algorithm_name.lower()}"
        self._eval_csv_path = self.output_dir / f"{prefix}_{timestamp}.csv"
        self._proposal_csv_path = self.output_dir / f"{prefix}_proposals_{timestamp}.csv"
        self._summary_csv_path = self.output_dir / f"{prefix}_summary_{timestamp}.csv"

        self._eval_file = open(self._eval_csv_path, "w", encoding="utf-8-sig", newline="")
        self._eval_writer = csv.DictWriter(
            self._eval_file, fieldnames=EVAL_CSV_FIELDS, extrasaction="ignore"
        )
        self._eval_writer.writeheader()
        self._eval_file.flush()

        self._proposal_file = open(self._proposal_csv_path, "w", encoding="utf-8-sig", newline="")
        self._proposal_writer = csv.DictWriter(
            self._proposal_file, fieldnames=PROPOSAL_FIELDS, extrasaction="ignore"
        )
        self._proposal_writer.writeheader()
        self._proposal_file.flush()

        print(f"[{self.algorithm_name}] Eval CSV    : {self._eval_csv_path}")
        print(f"[{self.algorithm_name}] Proposal CSV: {self._proposal_csv_path}")

    def _close_csv(self):
        if self._eval_file:
            self._eval_file.close()
            self._eval_file = None
        if self._proposal_file:
            self._proposal_file.close()
            self._proposal_file = None

    # ============================================================
    # Proposal 记录（核心：与 GA instrumented 完全一致的记录方式）
    # ============================================================

    def log_proposal(self, params: Dict[str, float], source: str,
                     status: str, errors: List[str]) -> None:
        """记录一次 validate_quick 调用"""
        self.proposal_counter += 1
        classified = classify_errors(errors)
        rec = {
            "proposal_idx": self.proposal_counter,
            "generation": self.current_generation,
            "source": source,
            "status": status,
            "fitness": CONSTRAINT_VIOL_FITNESS_MARKER if status != "ok" else "",
            "sim_status": "constraint_violation" if status != "ok" else "",
            "Net_Displacement_X": "",
            "Displacement_Range_Y": "",
            "Net_Displacement_Y": "",
            "Average_Velocity_X": "",
            "m": params["m"],
            "n": params["n"],
            "alpha": params["alpha"],
            "beta": params["beta"],
            "DIST_BETTERY": params["DIST_BETTERY"],
            "violated_constraints": ";".join(classified),
            "errors_raw": json.dumps(errors, ensure_ascii=False) if errors else "",
            "eval_count": "",
            "eval_time_s": 0 if status != "ok" else "",
            "timestamp": round(time.time(), 3),
        }
        self._proposal_log.append(rec)

        if status == "ok":
            self._pending_ok_proposals.append(rec)
        elif self._proposal_writer is not None:
            self._proposal_writer.writerow(rec)
            self._proposal_file.flush()

        # 约束违反行也写入 eval CSV
        if status != "ok" and self._eval_writer is not None:
            viol_row = {
                "iteration": self.current_generation,
                "algorithm": self.algorithm_name,
                "status": "constraint_violation",
                "fitness": CONSTRAINT_VIOL_FITNESS_MARKER,
                "Net_Displacement_X": "",
                "Displacement_Range_Y": "",
                "Net_Displacement_Y": "",
                "Average_Velocity_X": "",
                "m": params["m"],
                "n": params["n"],
                "alpha": params["alpha"],
                "beta": params["beta"],
                "DIST_BETTERY": params["DIST_BETTERY"],
                "D6_rear_angle": "",
                "D12_front_angle": "",
                "D2_body_pitch": "",
                "C_y": "",
                "H_max": "",
                "DF_span": "",
                "errors": json.dumps(errors, ensure_ascii=False) if errors else "",
                "eval_count": "",
                "eval_time_s": 0,
            }
            self._eval_writer.writerow(viol_row)
            self._eval_file.flush()

    def finalize_ok_proposal(self, sim_rec: Dict) -> None:
        """仿真完成后，回填 proposal 记录"""
        if not self._pending_ok_proposals:
            return
        rec = self._pending_ok_proposals.pop(0)
        result = sim_rec.get("result", {})
        errors = sim_rec.get("errors", [])
        rec.update({
            "fitness": sim_rec.get("fitness", PENALTY_FITNESS),
            "sim_status": sim_rec.get("status", ""),
            "Net_Displacement_X": result.get("Net_Displacement_X", ""),
            "Displacement_Range_Y": result.get("Displacement_Range_Y", ""),
            "Net_Displacement_Y": result.get("Net_Displacement_Y", ""),
            "Average_Velocity_X": result.get("Average_Velocity_X", ""),
            "errors_raw": json.dumps(errors, ensure_ascii=False) if errors else rec.get("errors_raw", ""),
            "eval_count": sim_rec.get("eval_count", ""),
            "eval_time_s": round(sim_rec.get("eval_time_s", 0), 1),
        })
        if self._proposal_writer is not None:
            self._proposal_writer.writerow(rec)
            self._proposal_file.flush()

    # ============================================================
    # 约束检查（带日志记录）
    # ============================================================

    def check_constraints(self, params: Dict[str, float], source: str = "unknown") -> bool:
        """检查约束并记录到 proposal log。返回 True 表示通过。"""
        ok, errors = validate_quick(params)
        status = "ok" if ok else "constraint_violation"
        self.log_proposal(params, source, status, errors)
        return ok

    # ============================================================
    # SolidWorks / Adams 进程清理（防止残留进程卡死后续仿真）
    # ============================================================

    # 周期性清理：每 N 次成功进入仿真后清理一次残留进程
    CLEANUP_INTERVAL = int(os.getenv("SW_CLEANUP_INTERVAL", "10"))
    CLEANUP_COOLDOWN = float(os.getenv("SW_CLEANUP_COOLDOWN", "3"))

    def _cleanup_sim_processes(self):
        """杀掉残留的 SolidWorks / Adams 进程"""
        import subprocess
        targets = ["SLDWORKS.exe", "sldworks.exe", "aview.exe",
                   "adams_main.exe", "adamssolve.exe"]
        for exe in targets:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", exe],
                    capture_output=True, timeout=15
                )
            except Exception:
                pass
        print(f"[{self.algorithm_name}] 已清理残留 SolidWorks/Adams 进程")

    # ============================================================
    # 仿真评估
    # ============================================================

    async def evaluate_one(self, params: Dict[str, float], dry_run: bool = False) -> Dict:
        """一次完整评估: validate → SolidWorks + Adams → 提取 fitness"""
        from mymcp.tool.robot_leg import validate_robot_design, run_robot_simulation

        self.eval_count += 1
        t0 = time.time()
        params = snap(params)

        if dry_run:
            return {
                "status": "ok", "fitness": float(self.eval_count),
                "params": params,
                "result": {"Net_Displacement_X": float(self.eval_count),
                           "Displacement_Range_Y": 0.1,
                           "Net_Displacement_Y": 0.0,
                           "Average_Velocity_X": 0.0},
                "derived": {"D6_rear_angle": 50.0, "D12_front_angle": 70.0,
                            "D2_body_pitch": 90.0, "C_y": 1.0,
                            "H_max": 14.0, "DF_span": 10.0},
                "errors": [], "eval_count": self.eval_count,
                "eval_time_s": time.time() - t0,
            }

        # 周期性清理（仿真前），防止残留进程卡端口
        if (self.CLEANUP_INTERVAL > 0 and
                self.eval_count > 1 and
                self.eval_count % self.CLEANUP_INTERVAL == 0):
            print(f"[{self.algorithm_name}] #{self.eval_count} 触发预清理")
            self._cleanup_sim_processes()
            await asyncio.sleep(self.CLEANUP_COOLDOWN)

        try:
            val_str = await validate_robot_design(**params)
            val = json.loads(val_str)
        except Exception as e:
            return {"status": "validate_error", "fitness": PENALTY_FITNESS,
                    "params": params, "errors": [str(e)],
                    "eval_count": self.eval_count, "eval_time_s": time.time() - t0}

        if val.get("status") != "ok":
            return {"status": "constraint_violation", "fitness": PENALTY_FITNESS,
                    "params": params, "errors": val.get("errors", []),
                    "derived": val.get("derived", {}),
                    "eval_count": self.eval_count, "eval_time_s": time.time() - t0}

        try:
            sim_str = await asyncio.wait_for(
                run_robot_simulation(**params), timeout=600
            )
            sim = json.loads(sim_str)
            result = sim.get("result", sim)
        except asyncio.TimeoutError:
            # 超时 → 清理进程
            self._cleanup_sim_processes()
            return {"status": "timeout", "fitness": PENALTY_FITNESS,
                    "params": params, "errors": ["simulation timeout 600s"],
                    "eval_count": self.eval_count, "eval_time_s": time.time() - t0}
        except Exception as e:
            self._cleanup_sim_processes()
            return {"status": "sim_error", "fitness": PENALTY_FITNESS,
                    "params": params, "errors": [str(e)],
                    "eval_count": self.eval_count, "eval_time_s": time.time() - t0}

        fitness = result.get("fitness", PENALTY_FITNESS)
        if fitness is None:
            fitness = PENALTY_FITNESS

        return {
            "status": result.get("status", "ok"),
            "fitness": fitness,
            "params": params,
            "result": result,
            "derived": val.get("derived", {}),
            "errors": result.get("errors", []),
            "eval_count": self.eval_count,
            "eval_time_s": time.time() - t0,
        }

    def write_eval_row(self, rec: Dict, generation: int):
        """写入 eval CSV 一行"""
        result = rec.get("result", {})
        derived = rec.get("derived", {})
        params = rec.get("params", {})
        errors = rec.get("errors", [])
        row = {
            "iteration": generation,
            "algorithm": self.algorithm_name,
            "status": rec.get("status", ""),
            "fitness": rec.get("fitness", 0),
            "Net_Displacement_X": result.get("Net_Displacement_X", ""),
            "Displacement_Range_Y": result.get("Displacement_Range_Y", ""),
            "Net_Displacement_Y": result.get("Net_Displacement_Y", ""),
            "Average_Velocity_X": result.get("Average_Velocity_X", ""),
            "m": params.get("m", ""),
            "n": params.get("n", ""),
            "alpha": params.get("alpha", ""),
            "beta": params.get("beta", ""),
            "DIST_BETTERY": params.get("DIST_BETTERY", ""),
            "D6_rear_angle": derived.get("D6_rear_angle", ""),
            "D12_front_angle": derived.get("D12_front_angle", ""),
            "D2_body_pitch": derived.get("D2_body_pitch", ""),
            "C_y": derived.get("C_y", ""),
            "H_max": derived.get("H_max", ""),
            "DF_span": derived.get("DF_span", ""),
            "errors": json.dumps(errors, ensure_ascii=False) if errors else "",
            "eval_count": rec.get("eval_count", ""),
            "eval_time_s": round(rec.get("eval_time_s", 0), 1),
        }
        if self._eval_writer:
            self._eval_writer.writerow(row)
            self._eval_file.flush()

    # ============================================================
    # Summary 输出
    # ============================================================

    def write_summary(self):
        """输出 proposal summary CSV"""
        from collections import Counter
        total = len(self._proposal_log)
        ok = sum(1 for r in self._proposal_log if r["status"] == "ok")
        viol = total - ok

        constraint_count: Counter = Counter()
        for r in self._proposal_log:
            if r["status"] != "ok":
                for c in r["violated_constraints"].split(";"):
                    if c:
                        constraint_count[c] += 1

        print(f"\n{'='*70}")
        print(f"[{self.algorithm_name}] Proposal Summary")
        print(f"{'='*70}")
        print(f"总 validate 调用  : {total}")
        print(f"  通过(ok)        : {ok}  ({ok/max(1,total)*100:.2f}%)")
        print(f"  约束违反        : {viol}  ({viol/max(1,total)*100:.2f}%)")
        print(f"成功仿真次数      : {self.eval_count}")
        print(f"最优 fitness      : {self.best_fitness:.4f}")
        print(f"\n约束违反分布:")
        for cname, _ in CONSTRAINT_PATTERNS:
            n = constraint_count.get(cname, 0)
            pct = n / max(1, viol) * 100
            print(f"  {cname:<18s} {n:5d}  ({pct:5.1f}%)")

        if self._summary_csv_path:
            with open(self._summary_csv_path, "w", encoding="utf-8-sig", newline="") as f:
                fields = ["algorithm", "total_proposals", "n_ok", "n_violation",
                          "feasibility_rate", "total_evals", "best_fitness"] \
                         + [c for c, _ in CONSTRAINT_PATTERNS]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                row = {
                    "algorithm": self.algorithm_name,
                    "total_proposals": total,
                    "n_ok": ok,
                    "n_violation": viol,
                    "feasibility_rate": round(ok / max(1, total), 4),
                    "total_evals": self.eval_count,
                    "best_fitness": self.best_fitness,
                }
                for cname, _ in CONSTRAINT_PATTERNS:
                    row[cname] = constraint_count.get(cname, 0)
                writer.writerow(row)
            print(f"\n[OK] Summary: {self._summary_csv_path}")

    # ============================================================
    # 断点续跑
    # ============================================================

    def save_checkpoint(self, checkpoint_path: Path, extra_state: Dict = None):
        state = {
            "algorithm_name": self.algorithm_name,
            "eval_count": self.eval_count,
            "proposal_counter": self.proposal_counter,
            "current_generation": self.current_generation,
            "best_fitness": self.best_fitness,
            "best_params": self.best_params,
            "last_improvement_eval": self._last_improvement_eval,
            "last_significant_improvement_eval": self._last_significant_improvement_eval,
            "last_significant_best": self._last_significant_best,
            "min_evals_before_stop": self.min_evals_before_stop,
            "patience_evals": self.patience_evals,
            "significant_improvement_ratio": self.significant_improvement_ratio,
            "eval_csv_path": str(self._eval_csv_path) if self._eval_csv_path else None,
            "proposal_csv_path": str(self._proposal_csv_path) if self._proposal_csv_path else None,
            "summary_csv_path": str(self._summary_csv_path) if self._summary_csv_path else None,
        }
        if extra_state:
            state.update(extra_state)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(state, f)

    def load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict]:
        if not checkpoint_path.exists():
            return None
        try:
            with open(checkpoint_path, "rb") as f:
                state = pickle.load(f)
            self.eval_count = state.get("eval_count", 0)
            self.proposal_counter = state.get("proposal_counter", 0)
            self.current_generation = state.get("current_generation", 0)
            self.best_fitness = state.get("best_fitness", PENALTY_FITNESS)
            self.best_params = state.get("best_params")
            self._last_improvement_eval = state.get(
                "last_improvement_eval",
                self.eval_count if self.best_params else 0,
            )
            self._last_significant_improvement_eval = state.get(
                "last_significant_improvement_eval",
                self._last_improvement_eval,
            )
            self._last_significant_best = state.get(
                "last_significant_best",
                self.best_fitness,
            )
            self.min_evals_before_stop = state.get(
                "min_evals_before_stop", self.min_evals_before_stop
            )
            self.patience_evals = state.get("patience_evals", self.patience_evals)
            self.significant_improvement_ratio = state.get(
                "significant_improvement_ratio", self.significant_improvement_ratio
            )
            return state
        except Exception as e:
            print(f"[{self.algorithm_name}] 断点加载失败: {e}")
            return None

    def record_evaluation(self, fit: float, params: Dict[str, float]) -> bool:
        """更新全局最优，并记录最后一次提升所在真实 eval 轮次。"""
        if fit > self.best_fitness:
            self.best_fitness = fit
            self.best_params = params.copy()
            self._last_improvement_eval = self.eval_count
            if self._is_significant_improvement(fit):
                self._last_significant_improvement_eval = self.eval_count
                self._last_significant_best = fit
            return True
        return False

    def _is_significant_improvement(self, fit: float) -> bool:
        """判断当前 best 刷新是否达到显著提升阈值。"""
        baseline = self._last_significant_best
        if self._last_significant_improvement_eval <= 0:
            return True
        if baseline <= 0:
            return fit > baseline and fit > 0
        return fit >= baseline * (1.0 + self.significant_improvement_ratio)

    def should_stop_early(self) -> bool:
        """至少跑满 min_evals 后，连续 patience_evals 无显著提升则提前停止。"""
        if self.eval_count < self.min_evals_before_stop:
            return False
        if self._last_significant_improvement_eval <= 0:
            return False
        stagnant = self.eval_count - self._last_significant_improvement_eval
        if stagnant < self.patience_evals:
            return False
        if not self._early_stop_triggered:
            print(
                f"[{self.algorithm_name}] 提前收敛: eval={self.eval_count}, "
                f"last_significant_improve={self._last_significant_improvement_eval}, "
                f"连续 {stagnant} 轮未超过 "
                f"{self.significant_improvement_ratio:.1%} 显著提升, "
                f"best={self.best_fitness:.4f}"
            )
            self._early_stop_triggered = True
        return True

    # ============================================================
    # 主运行流程
    # ============================================================

    @abstractmethod
    async def optimize(self, dry_run: bool = False):
        """子类实现的优化主循环"""
        pass

    async def run(self, dry_run: bool = False, resume_checkpoint: Optional[str] = None,
                  checkpoint_path: Optional[str] = None):
        """统一运行入口

        Args:
            dry_run: 是否跳过真实仿真
            resume_checkpoint: 断点文件路径，如有则继续
            checkpoint_path: 保存断点的路径（每代/每轮保存）
        """
        self._checkpoint_save_path = Path(checkpoint_path) if checkpoint_path else (
            self.output_dir / f"checkpoint_{self.algorithm_name.lower()}.pkl"
        )

        resumed = False
        resume_paths = None
        if resume_checkpoint:
            ckpt = Path(resume_checkpoint)
            state = self.load_checkpoint(ckpt)
            if state:
                resumed = True
                resume_paths = {
                    "eval_csv": state.get("eval_csv_path", ""),
                    "proposal_csv": state.get("proposal_csv_path", ""),
                    "summary_csv": state.get("summary_csv_path", ""),
                }
                print(f"[{self.algorithm_name}] 断点恢复: eval_count={self.eval_count}, "
                      f"gen={self.current_generation}, best={self.best_fitness:.4f}")
            else:
                print(f"[{self.algorithm_name}] 断点加载失败，从头开始")

        print(f"{'='*70}")
        print(f"[{self.algorithm_name}] {'恢复' if resumed else '开始'}优化 | "
              f"max_evals={self.max_evals} | seed={self.seed}")
        print(f"[{self.algorithm_name}] early_stop: min_evals={self.min_evals_before_stop}, "
              f"patience={self.patience_evals}, "
              f"significant_improvement={self.significant_improvement_ratio:.1%}")
        print(f"{'='*70}")

        try:
            if resumed and resume_paths and Path(resume_paths["eval_csv"]).exists():
                self._init_csv(resume_paths=resume_paths)
            else:
                self._init_csv()
            await self.optimize(dry_run=dry_run)
        finally:
            self._close_csv()
            self.write_summary()
            # 优化结束 → 最终清理残留进程
            if not dry_run:
                self._cleanup_sim_processes()
            print(f"\n[{self.algorithm_name}] 完成! eval={self.eval_count}, best={self.best_fitness:.4f}")
