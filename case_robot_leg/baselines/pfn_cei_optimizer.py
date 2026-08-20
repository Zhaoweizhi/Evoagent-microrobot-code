# -*- coding: utf-8 -*-
"""PFN-CEI optimization for the robot-leg task.

The optimizer uses a pretrained transformer surrogate and constrained
expected improvement. It requires the official BOEngineeringBenchmark
repository and its pfns4bo model files.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# IMPORTANT: import torch BEFORE numpy / scipy / sklearn / pyaedt.
# On Windows, if numpy's Intel OpenMP (libiomp5md.dll) loads first, then
# torch's shm.dll (which depends on libomp.dll) fails with WinError 127.
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"[PFN-CEI] PyTorch 不可用，将使用 GP+CEI fallback: {e}")
    torch = None
    TORCH_AVAILABLE = False

import random
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from .base_optimizer import (
    BaseOptimizer, PARAM_NAMES, PARAM_BOUNDS,
    PENALTY_FITNESS, snap, clamp, random_params,
    generate_valid_params, validate_quick,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_boeng_env = os.getenv("BOENGINEERINGBENCHMARK_DIR")
local_boeng = Path(__file__).resolve().parent / "BOEngineeringBenchmark"

def _is_complete_boeng(path: Path) -> bool:
    return (
        (path / "PFN_CEI.py").exists()
        and (path / "pfns4bo" / "layer.py").exists()
        and (path / "pfns4bo" / "transformer.py").exists()
    )

BOENG_DIR = Path(_boeng_env) if _boeng_env else local_boeng
if str(BOENG_DIR) not in sys.path:
    sys.path.insert(0, str(BOENG_DIR))


class PFNCEIOptimizer(BaseOptimizer):
    """PFN-CEI 优化器（机器人腿）

    仅作为 BOEngineeringBenchmark 官方 PFN-CEI 的 adapter。
    """

    def __init__(
        self,
        max_evals: int = 200,
        n_initial: int = 10,
        n_candidates: int = 5000,
        device: str = "cpu",
        model_path: Optional[str] = None,
        output_dir: str = ".",
        seed: int = 42,
    ):
        super().__init__(
            max_evals=max_evals,
            output_dir=output_dir,
            algorithm_name="PFN_CEI",
            seed=seed,
        )
        self.n_initial = min(n_initial, max_evals)
        self.n_candidates = n_candidates
        self.device = device
        self.model_path = model_path

        self._X_observed: List[List[float]] = []
        self._y_observed: List[float] = []
        self._feasible_mask: List[bool] = []

        self._pfn_model = None
        self._pfn_method = None
        self._official_repo_dir = BOENG_DIR

    def _params_to_vector(self, params: Dict[str, float]) -> List[float]:
        return [params[name] for name in PARAM_NAMES]

    def _vector_to_params(self, x: List[float]) -> Dict[str, float]:
        return snap({name: x[i] for i, name in enumerate(PARAM_NAMES)})

    def _normalize_x(self, X: np.ndarray) -> np.ndarray:
        lower = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES])
        upper = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES])
        return (X - lower) / (upper - lower + 1e-8)

    def _generate_valid_sample(self, rng: random.Random) -> Dict[str, float]:
        for _ in range(5000):
            p = random_params(rng)
            if self.check_constraints(p, source="pfncei_random_sample"):
                return p
        return random_params(rng)

    def _load_pfn(self) -> None:
        """加载官方 BOEngineeringBenchmark PFN-CEI 模型。"""
        if self._pfn_model is not None and self._pfn_method is not None:
            return
        if not TORCH_AVAILABLE:
            raise RuntimeError("PFN-CEI official adapter 需要 PyTorch")
        if self.model_path is None:
            raise RuntimeError("PFN-CEI official adapter 需要 --pfn-model 指向官方模型文件")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"PFN-CEI 官方模型文件不存在: {self.model_path}")
        if not self._official_repo_dir.exists():
            raise FileNotFoundError(
                f"BOEngineeringBenchmark 官方仓库不存在: {self._official_repo_dir}. "
                "请 clone 官方仓库或设置 BOENGINEERINGBENCHMARK_DIR。"
            )

        try:
            from pfns4bo import transformer
            from PFN_CEI import PFNCEI_TransformerBOMethod

            self._pfn_model = torch.load(
                self.model_path, map_location=self.device, weights_only=False
            )
            self._pfn_model.eval()
            self._pfn_model.to(self.device)

            self._pfn_method = PFNCEI_TransformerBOMethod(
                model=self._pfn_model,
                device=self.device,
                apply_power_transform=True,
            )
            print(f"[PFN-CEI] PFN 模型加载成功")
        except Exception as e:
            raise RuntimeError(f"PFN-CEI 官方模型/代码加载失败: {e}") from e

    def _suggest_with_pfn(self, rng: random.Random) -> Dict[str, float]:
        """使用 PFN-CEI 建议下一个点"""
        X_obs = torch.tensor(
            self._normalize_x(np.array(self._X_observed)),
            dtype=torch.float32
        ).to(self.device)

        y_array = np.array(self._y_observed)
        y_obs = torch.tensor(y_array, dtype=torch.float32).to(self.device)

        feasible_array = np.array(self._feasible_mask, dtype=float)
        g_obs = torch.tensor(
            np.where(feasible_array > 0.5, -1.0, 1.0).reshape(-1, 1),
            dtype=torch.float32
        ).to(self.device)

        candidates = []
        for _ in range(self.n_candidates):
            p = random_params(rng)
            ok, _ = validate_quick(p)
            if ok:
                candidates.append(p)
            if len(candidates) >= self.n_candidates:
                break

        if not candidates:
            return self._generate_valid_sample(rng)

        X_pen = torch.tensor(
            self._normalize_x(np.array([self._params_to_vector(p) for p in candidates])),
            dtype=torch.float32
        ).to(self.device)

        try:
            obj_acq, constraint_acq = self._pfn_method.observe_and_suggest(
                X_obs=X_obs, y_obs=y_obs, X_pen=X_pen, GX=g_obs,
            )
            if constraint_acq.dim() > 1:
                pof = constraint_acq.prod(dim=1)
            else:
                pof = constraint_acq
            cei = obj_acq * pof
            best_idx = cei.argmax().item()
            chosen = candidates[best_idx]
        except Exception as e:
            raise RuntimeError(f"官方 PFN-CEI observe_and_suggest 失败: {e}") from e

        self.log_proposal(chosen, "pfncei_suggest", "ok", [])
        return chosen

    async def optimize(self, dry_run: bool = False):
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        print(f"[PFN-CEI] n_initial={self.n_initial}, n_candidates={self.n_candidates}, "
              f"max_evals={self.max_evals}")
        print(f"[PFN-CEI] BOEngineeringBenchmark: {self._official_repo_dir}")
        self._load_pfn()
        print(f"[PFN-CEI] PFN 官方模型: 可用")
        print("=" * 60)

        # 断点恢复
        if hasattr(self, '_checkpoint_save_path') and self._checkpoint_save_path.exists():
            state = self.load_checkpoint(self._checkpoint_save_path)
            if state and "X_observed" in state:
                self._X_observed = state["X_observed"]
                self._y_observed = state["y_observed"]
                self._feasible_mask = state.get("feasible_mask", [])
                print(f"[PFN-CEI] 从断点恢复: {len(self._X_observed)} 已观测点, "
                      f"eval_count={self.eval_count}, best={self.best_fitness:.4f}")

        # 阶段1: 初始随机采样（如果还没够初始点）
        if len(self._X_observed) < self.n_initial:
            self.current_generation = 0
            print(f"[PFN-CEI] 阶段1: 生成 {self.n_initial} 个初始点...")

            while len(self._X_observed) < self.n_initial and self.eval_count < self.max_evals:
                params = self._generate_valid_sample(rng)

                rec = await self.evaluate_one(params, dry_run=dry_run)
                fit = rec["fitness"]
                self.finalize_ok_proposal(rec)
                self.write_eval_row(rec, generation=0)

                self._X_observed.append(self._params_to_vector(params))
                self._y_observed.append(fit)
                self._feasible_mask.append(rec["status"] == "ok")

                self.record_evaluation(fit, params)

            print(f"[PFN-CEI] 阶段1完成: {len(self._X_observed)} 初始点, best={self.best_fitness:.4f}")

        # 阶段2: 序贯优化
        self.current_generation = 1
        print(f"[PFN-CEI] 阶段2: CEI 序贯优化...")
        iteration = 0

        while self.eval_count < self.max_evals and not self.should_stop_early():
            params = self._suggest_with_pfn(rng)

            rec = await self.evaluate_one(params, dry_run=dry_run)
            fit = rec["fitness"]
            self.finalize_ok_proposal(rec)
            self.write_eval_row(rec, generation=1)

            self._X_observed.append(self._params_to_vector(params))
            self._y_observed.append(fit)
            self._feasible_mask.append(rec["status"] == "ok")

            if self.record_evaluation(fit, params):
                print(f"  [{self.eval_count:4d}] 🎯 NEW BEST fitness={fit:8.4f}")
            elif self.eval_count % 10 == 0:
                print(f"  [{self.eval_count:4d}] best={self.best_fitness:.4f}")

            iteration += 1

            # 每 10 次保存断点
            if self.eval_count % 10 == 0:
                self.save_checkpoint(self._checkpoint_save_path, extra_state={
                    "X_observed": self._X_observed,
                    "y_observed": self._y_observed,
                    "feasible_mask": self._feasible_mask,
                })

            if self.should_stop_early():
                break
