# -*- coding: utf-8 -*-
"""
POM (Portfolio of Methods) — 机器人腿任务
组合多种采集策略的元优化方法，类似 HEBO 的 portfolio 思路:
  - 同时维护多个 surrogate/采集策略
  - 根据历史表现动态分配采样预算
  - 全量记录 validate_quick 调用（含失败）

策略组合:
  1. Random exploration（全局随机）
  2. Local exploitation（最优附近局部搜索）
  3. GP-EI（高斯过程 + EI 采集）
  4. RF-EI（随机森林 + EI 采集）

选择机制: Thompson Sampling (基于各策略历史改进率)
"""
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from .base_optimizer import (
    BaseOptimizer, PARAM_NAMES, PARAM_BOUNDS,
    PENALTY_FITNESS, snap, clamp, random_params,
    generate_valid_params, validate_quick,
)

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import norm
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class POMOptimizer(BaseOptimizer):
    """POM 优化器（Portfolio of Methods，机器人腿）"""

    STRATEGY_NAMES = ["random", "local", "gp_ei", "rf_ei"]

    def __init__(
        self,
        max_evals: int = 200,
        n_initial: int = 20,
        n_candidates: int = 1000,
        output_dir: str = ".",
        seed: int = 42,
    ):
        super().__init__(
            max_evals=max_evals,
            output_dir=output_dir,
            algorithm_name="POM",
            seed=seed,
        )
        self.n_initial = min(n_initial, max_evals)
        self.n_candidates = n_candidates

        self._X_observed: List[List[float]] = []
        self._y_observed: List[float] = []

        # Thompson Sampling 参数（Beta 分布: alpha=成功, beta=失败）
        self._strategy_alpha = {s: 1.0 for s in self.STRATEGY_NAMES}
        self._strategy_beta = {s: 1.0 for s in self.STRATEGY_NAMES}
        self._strategy_counts = {s: 0 for s in self.STRATEGY_NAMES}

        self._gp_model = None
        self._rf_model = None

    def _params_to_vector(self, params: Dict[str, float]) -> List[float]:
        return [params[name] for name in PARAM_NAMES]

    def _vector_to_params(self, x: List[float]) -> Dict[str, float]:
        return snap({name: x[i] for i, name in enumerate(PARAM_NAMES)})

    def _normalize_x(self, X: np.ndarray) -> np.ndarray:
        lower = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES])
        upper = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES])
        return (X - lower) / (upper - lower + 1e-8)

    def _generate_valid_sample(self, rng: random.Random,
                               base_params: Optional[Dict[str, float]] = None,
                               scale: float = 0.2) -> Dict[str, float]:
        if base_params is None:
            for _ in range(5000):
                p = random_params(rng)
                if self.check_constraints(p, source="pom_random"):
                    return p
            return random_params(rng)

        for _ in range(1000):
            p = {}
            for name in PARAM_NAMES:
                lo, hi = PARAM_BOUNDS[name]
                val = base_params[name] + rng.gauss(0, (hi - lo) * scale)
                p[name] = clamp(name, val)
            p = snap(p)
            if self.check_constraints(p, source="pom_local"):
                return p

        for _ in range(5000):
            p = random_params(rng)
            if self.check_constraints(p, source="pom_fallback"):
                return p
        return random_params(rng)

    def _fit_surrogates(self):
        """训练 GP 和 RF 代理模型"""
        if not SKLEARN_AVAILABLE or len(self._X_observed) < 5:
            return

        X = np.array(self._X_observed)
        y = np.array(self._y_observed)

        valid_mask = y > PENALTY_FITNESS
        if valid_mask.sum() < 5:
            valid_mask = np.ones(len(y), dtype=bool)

        X_norm = self._normalize_x(X[valid_mask])
        y_valid = y[valid_mask]

        # GP
        try:
            self._gp_model = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                random_state=self.seed,
            )
            self._gp_model.fit(X_norm, y_valid)
        except Exception:
            self._gp_model = None

        # RF
        try:
            self._rf_model = RandomForestRegressor(
                n_estimators=10,
                max_depth=10,
                min_samples_split=3,
                random_state=self.seed,
                n_jobs=-1,
            )
            self._rf_model.fit(X_norm, y_valid)
        except Exception:
            self._rf_model = None

    def _select_strategy(self, rng: random.Random) -> str:
        """Thompson Sampling 选择策略"""
        scores = {}
        for s in self.STRATEGY_NAMES:
            scores[s] = rng.betavariate(self._strategy_alpha[s], self._strategy_beta[s])
        return max(scores, key=scores.get)

    def _update_strategy(self, strategy: str, improved: bool):
        """更新策略的 Thompson Sampling 参数"""
        self._strategy_counts[strategy] += 1
        if improved:
            self._strategy_alpha[strategy] += 1.0
        else:
            self._strategy_beta[strategy] += 0.5

    def _suggest_random(self, rng: random.Random) -> Dict[str, float]:
        return self._generate_valid_sample(rng)

    def _suggest_local(self, rng: random.Random) -> Dict[str, float]:
        if self.best_params is None:
            return self._generate_valid_sample(rng)
        progress = self.eval_count / max(1, self.max_evals)
        scale = 0.25 * (1.0 - 0.7 * progress)
        return self._generate_valid_sample(rng, self.best_params, scale)

    def _suggest_gp_ei(self, rng: random.Random) -> Dict[str, float]:
        if self._gp_model is None:
            return self._suggest_local(rng)

        candidates = []
        for _ in range(self.n_candidates * 3):
            p = random_params(rng)
            ok, _ = validate_quick(p)
            if ok:
                candidates.append(p)
            if len(candidates) >= self.n_candidates:
                break

        if not candidates:
            return self._generate_valid_sample(rng)

        X = np.array([self._params_to_vector(p) for p in candidates])
        X_norm = self._normalize_x(X)
        mean, std = self._gp_model.predict(X_norm, return_std=True)
        std = np.maximum(std, 1e-8)

        valid_y = [y for y in self._y_observed if y > PENALTY_FITNESS]
        y_best = max(valid_y) if valid_y else 0.0
        z = (mean - y_best) / std
        ei = (mean - y_best) * norm.cdf(z) + std * norm.pdf(z)

        best_idx = np.argmax(ei)
        chosen = candidates[best_idx]
        self.log_proposal(chosen, "pom_gp_ei", "ok", [])
        return chosen

    def _suggest_rf_ei(self, rng: random.Random) -> Dict[str, float]:
        if self._rf_model is None:
            return self._suggest_local(rng)

        candidates = []
        for _ in range(self.n_candidates * 3):
            p = random_params(rng)
            ok, _ = validate_quick(p)
            if ok:
                candidates.append(p)
            if len(candidates) >= self.n_candidates:
                break

        if not candidates:
            return self._generate_valid_sample(rng)

        X = np.array([self._params_to_vector(p) for p in candidates])
        X_norm = self._normalize_x(X)

        predictions = np.array([tree.predict(X_norm)
                                for tree in self._rf_model.estimators_])
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        std = np.maximum(std, 1e-8)

        valid_y = [y for y in self._y_observed if y > PENALTY_FITNESS]
        y_best = max(valid_y) if valid_y else 0.0
        z = (mean - y_best) / std
        ei = (mean - y_best) * norm.cdf(z) + std * norm.pdf(z)

        best_idx = np.argmax(ei)
        chosen = candidates[best_idx]
        self.log_proposal(chosen, "pom_rf_ei", "ok", [])
        return chosen

    def _suggest_by_strategy(self, strategy: str, rng: random.Random) -> Dict[str, float]:
        if strategy == "random":
            return self._suggest_random(rng)
        elif strategy == "local":
            return self._suggest_local(rng)
        elif strategy == "gp_ei":
            return self._suggest_gp_ei(rng)
        elif strategy == "rf_ei":
            return self._suggest_rf_ei(rng)
        return self._suggest_random(rng)

    async def optimize(self, dry_run: bool = False):
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        print(f"[POM] n_initial={self.n_initial}, n_candidates={self.n_candidates}, "
              f"max_evals={self.max_evals}")
        print(f"[POM] strategies={self.STRATEGY_NAMES}")
        print(f"[POM] sklearn: {SKLEARN_AVAILABLE}")
        print("=" * 60)

        # 断点恢复
        if hasattr(self, '_checkpoint_save_path') and self._checkpoint_save_path.exists():
            state = self.load_checkpoint(self._checkpoint_save_path)
            if state and "X_observed" in state:
                self._X_observed = state["X_observed"]
                self._y_observed = state["y_observed"]
                self._strategy_alpha = state.get("strategy_alpha", self._strategy_alpha)
                self._strategy_beta = state.get("strategy_beta", self._strategy_beta)
                self._strategy_counts = state.get("strategy_counts", self._strategy_counts)
                print(f"[POM] 从断点恢复: {len(self._X_observed)} 已观测点, "
                      f"eval_count={self.eval_count}, best={self.best_fitness:.4f}")

        # 阶段1: 初始随机采样（如果还没够初始点）
        if len(self._X_observed) < self.n_initial:
            self.current_generation = 0
            print(f"[POM] 阶段1: 生成 {self.n_initial} 个初始点...")

            while len(self._X_observed) < self.n_initial and self.eval_count < self.max_evals:
                params = self._generate_valid_sample(rng)

                rec = await self.evaluate_one(params, dry_run=dry_run)
                fit = rec["fitness"]
                self.finalize_ok_proposal(rec)
                self.write_eval_row(rec, generation=0)

                self._X_observed.append(self._params_to_vector(params))
                self._y_observed.append(fit)

                self.record_evaluation(fit, params)

                if len(self._X_observed) % 5 == 0:
                    print(f"[POM] 初始采样: {len(self._X_observed)}/{self.n_initial} | "
                          f"best={self.best_fitness:.4f}")

            print(f"[POM] 阶段1完成: {len(self._X_observed)} 初始点, best={self.best_fitness:.4f}")

        # 阶段2: Portfolio 优化
        self.current_generation = 1
        print(f"[POM] 阶段2: Portfolio 序贯优化...")
        surrogate_update_interval = 5
        iteration = 0

        while self.eval_count < self.max_evals and not self.should_stop_early():
            if iteration % surrogate_update_interval == 0:
                self._fit_surrogates()

            strategy = self._select_strategy(rng)
            prev_best = self.best_fitness

            params = self._suggest_by_strategy(strategy, rng)

            rec = await self.evaluate_one(params, dry_run=dry_run)
            fit = rec["fitness"]
            self.finalize_ok_proposal(rec)
            self.write_eval_row(rec, generation=1)

            self._X_observed.append(self._params_to_vector(params))
            self._y_observed.append(fit)

            improved = fit > prev_best
            self._update_strategy(strategy, improved)

            if self.record_evaluation(fit, params):
                print(f"  [{self.eval_count:4d}] 🎯 NEW BEST fitness={fit:8.4f} "
                      f"(strategy={strategy})")
            elif self.eval_count % 10 == 0:
                print(f"  [{self.eval_count:4d}] best={self.best_fitness:.4f} "
                      f"(strategy={strategy})")

            iteration += 1

            # 每 10 次保存断点
            if self.eval_count % 10 == 0:
                self.save_checkpoint(self._checkpoint_save_path, extra_state={
                    "X_observed": self._X_observed,
                    "y_observed": self._y_observed,
                    "strategy_alpha": self._strategy_alpha,
                    "strategy_beta": self._strategy_beta,
                    "strategy_counts": self._strategy_counts,
                })

            if self.should_stop_early():
                break

        # 打印策略统计
        print(f"\n[POM] 策略使用统计:")
        for s in self.STRATEGY_NAMES:
            n = self._strategy_counts[s]
            alpha = self._strategy_alpha[s]
            rate = (alpha - 1) / max(1, n)
            print(f"  {s:<12s}: {n:4d} 次, 改进率={rate:.3f}")
