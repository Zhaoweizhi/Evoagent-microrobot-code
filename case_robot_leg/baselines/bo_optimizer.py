# -*- coding: utf-8 -*-
"""Gaussian-process Bayesian optimization for the robot-leg task.

The implementation uses constraint-aware initial sampling, an expected-
improvement acquisition function and the common evaluation logger.
"""
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_optimizer import (
    BaseOptimizer, PARAM_NAMES, PARAM_BOUNDS,
    PENALTY_FITNESS, snap, clamp, random_params,
    generate_valid_params, validate_quick,
)

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


class BOOptimizer(BaseOptimizer):
    """贝叶斯优化器（GP+EI，机器人腿）"""

    def __init__(
        self,
        max_evals: int = 200,
        n_initial: int = 20,
        acq_func: str = "EI",
        xi: float = 0.01,
        output_dir: str = ".",
        seed: int = 42,
    ):
        super().__init__(
            max_evals=max_evals,
            output_dir=output_dir,
            algorithm_name="BO",
            seed=seed,
        )
        self.n_initial = min(n_initial, max_evals)
        self.acq_func = acq_func
        self.xi = xi

        self._X_observed: List[List[float]] = []
        self._y_observed: List[float] = []

    def _save_bo_checkpoint(self) -> None:
        """Persist BO state after each real simulation so interrupted runs can resume."""
        if hasattr(self, "_checkpoint_save_path") and self._checkpoint_save_path:
            self.save_checkpoint(self._checkpoint_save_path, extra_state={
                "X_observed": self._X_observed,
                "y_observed": self._y_observed,
            })

    def _params_to_vector(self, params: Dict[str, float]) -> List[float]:
        return [params[name] for name in PARAM_NAMES]

    def _vector_to_params(self, x: List[float]) -> Dict[str, float]:
        return snap({name: x[i] for i, name in enumerate(PARAM_NAMES)})

    def _generate_valid_sample(self, rng: random.Random,
                               base_params: Optional[Dict[str, float]] = None,
                               scale: float = 0.2) -> Dict[str, float]:
        """生成满足约束的采样点（失败记录到 proposal log）"""
        if base_params is None:
            for _ in range(5000):
                p = random_params(rng)
                if self.check_constraints(p, source="bo_random_sample"):
                    return p
            return random_params(rng)

        for _ in range(1000):
            p = {}
            for name in PARAM_NAMES:
                lo, hi = PARAM_BOUNDS[name]
                base_val = base_params[name]
                delta = (hi - lo) * scale
                val = base_val + rng.gauss(0, delta)
                p[name] = clamp(name, val)
            p = snap(p)
            if self.check_constraints(p, source="bo_local_sample"):
                return p

        # 兜底全局采样
        for _ in range(5000):
            p = random_params(rng)
            if self.check_constraints(p, source="bo_fallback_sample"):
                return p
        return random_params(rng)

    async def optimize(self, dry_run: bool = False):
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        print(f"[BO] n_initial={self.n_initial}, acq_func={self.acq_func}, "
              f"max_evals={self.max_evals}")
        print(f"[BO] scikit-optimize 可用: {SKOPT_AVAILABLE}")
        print("=" * 60)

        # 断点恢复
        if hasattr(self, '_checkpoint_save_path') and self._checkpoint_save_path.exists():
            state = self.load_checkpoint(self._checkpoint_save_path)
            if state and "X_observed" in state:
                self._X_observed = state["X_observed"]
                self._y_observed = state["y_observed"]
                print(f"[BO] 从断点恢复: {len(self._X_observed)} 已观测点, "
                      f"eval_count={self.eval_count}, best={self.best_fitness:.4f}")

        if SKOPT_AVAILABLE:
            await self._optimize_skopt(rng, dry_run)
        else:
            await self._optimize_simple(rng, dry_run)

    async def _optimize_skopt(self, rng: random.Random, dry_run: bool):
        """使用 scikit-optimize 的完整 BO"""
        # 阶段1: 约束感知初始采样
        print(f"[BO] 阶段1: 生成 {self.n_initial} 个约束感知初始点...")
        self.current_generation = 0

        for i in range(self.n_initial):
            if self.eval_count >= self.max_evals:
                break
            params = self._generate_valid_sample(rng)

            rec = await self.evaluate_one(params, dry_run=dry_run)
            fit = rec["fitness"]
            self.finalize_ok_proposal(rec)
            self.write_eval_row(rec, generation=0)

            x = self._params_to_vector(params)
            # skopt 最小化，我们的 fitness 越大越好 → 传入 -fitness
            self._X_observed.append(x)
            self._y_observed.append(-fit)

            self.record_evaluation(fit, params)
            self._save_bo_checkpoint()

            if (i + 1) % 5 == 0:
                print(f"[BO] 初始采样: {i+1}/{self.n_initial} | best={self.best_fitness:.4f}")

        # 阶段2: GP+EI 序贯优化
        remaining = self.max_evals - self.eval_count
        if remaining <= 0:
            return

        print(f"[BO] 阶段2: GP+EI 优化，剩余 {remaining} 次...")
        self.current_generation = 1

        space = [
            Real(PARAM_BOUNDS[name][0], PARAM_BOUNDS[name][1], name=name)
            for name in PARAM_NAMES
        ]

        eval_idx = [0]

        def objective(x):
            # 注意: gp_minimize 是同步的，我们需要在里面跑 asyncio
            import asyncio
            if self.should_stop_early():
                raise StopIteration("BO early convergence")
            params = self._vector_to_params(x)
            eval_idx[0] += 1

            # 约束检查（记录失败）
            if not self.check_constraints(params, source="bo_gp_suggest"):
                # 被 GP 建议的点不满足约束，返回惩罚值
                return -PENALTY_FITNESS  # skopt 最小化

            loop = asyncio.new_event_loop()
            try:
                rec = loop.run_until_complete(self.evaluate_one(params, dry_run=dry_run))
            finally:
                loop.close()

            fit = rec["fitness"]
            self.finalize_ok_proposal(rec)
            self.write_eval_row(rec, generation=self.current_generation)

            self._X_observed.append(list(x))
            self._y_observed.append(-fit)

            if self.record_evaluation(fit, params):
                print(f"  [{self.eval_count:4d}] 🎯 NEW BEST fitness={fit:8.4f}")
            elif self.eval_count % 10 == 0:
                print(f"  [{self.eval_count:4d}] fitness={fit:8.4f}  best={self.best_fitness:.4f}")
            self._save_bo_checkpoint()

            return -fit  # skopt 最小化

        try:
            gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=remaining,
                n_initial_points=0,
                acq_func=self.acq_func,
                xi=self.xi,
                x0=self._X_observed,
                y0=self._y_observed,
                random_state=self.seed,
            )
        except StopIteration:
            print(f"[BO] 按早停规则收敛，停止 GP+EI 优化")
        except Exception as e:
            print(f"[BO] skopt 异常: {e}，切换到简化版")
            await self._optimize_simple(rng, dry_run)

    async def _optimize_simple(self, rng: random.Random, dry_run: bool):
        """简化版 BO（约束感知随机搜索 + 局部优化）"""
        print(f"[BO] 使用简化版 BO（约束感知局部搜索）")

        # 阶段1: 随机采样
        self.current_generation = 0
        while len(self._X_observed) < self.n_initial and self.eval_count < self.max_evals:
            params = self._generate_valid_sample(rng)
            rec = await self.evaluate_one(params, dry_run=dry_run)
            fit = rec["fitness"]
            self.finalize_ok_proposal(rec)
            self.write_eval_row(rec, generation=0)

            self._X_observed.append(self._params_to_vector(params))
            self._y_observed.append(-fit)

            self.record_evaluation(fit, params)

        print(f"[BO] 阶段1完成: {len(self._X_observed)} 初始点, best={self.best_fitness:.4f}")

        # 阶段2: 基于最优的局部搜索
        self.current_generation = 1
        while self.eval_count < self.max_evals and not self.should_stop_early():
            if self.best_params is None:
                params = self._generate_valid_sample(rng)
            else:
                progress = self.eval_count / self.max_evals
                scale = 0.3 * (1 - 0.8 * progress)
                params = self._generate_valid_sample(rng, self.best_params, scale)

            rec = await self.evaluate_one(params, dry_run=dry_run)
            fit = rec["fitness"]
            self.finalize_ok_proposal(rec)
            self.write_eval_row(rec, generation=1)

            self._X_observed.append(self._params_to_vector(params))
            self._y_observed.append(-fit)

            if self.record_evaluation(fit, params):
                print(f"  [{self.eval_count:4d}] 🎯 NEW BEST fitness={fit:8.4f}")
            elif self.eval_count % 20 == 0:
                print(f"  [{self.eval_count:4d}] best={self.best_fitness:.4f}")

            # 每 10 次保存断点
            self._save_bo_checkpoint()

            if self.should_stop_early():
                break
