# -*- coding: utf-8 -*-
"""
SMAC (Sequential Model-based Algorithm Configuration) — 机器人腿任务
默认使用官方 smac 包；本文件只把机器人腿问题包装成 SMAC objective。
本地 Random Forest 仅保留为显式 --smac-backend rf 的 legacy 补充模式。
"""
from __future__ import annotations

import asyncio
import random
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Literal

from .base_optimizer import (
    BaseOptimizer, PARAM_NAMES, PARAM_BOUNDS,
    PENALTY_FITNESS, snap, clamp, random_params,
    generate_valid_params, validate_quick,
)

try:
    from smac import HyperparameterOptimizationFacade, Scenario
    from ConfigSpace import ConfigurationSpace, Float
    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SMACOptimizer(BaseOptimizer):
    """SMAC 优化器（Random Forest surrogate + EI，机器人腿）"""

    def __init__(
        self,
        max_evals: int = 200,
        n_initial: int = 20,
        n_trees: int = 10,
        n_candidates: int = 1000,
        backend: Literal["official", "rf"] = "official",
        output_dir: str = ".",
        seed: int = 42,
    ):
        super().__init__(
            max_evals=max_evals,
            output_dir=output_dir,
            algorithm_name="SMAC" if backend == "official" else "SMAC_rf",
            seed=seed,
        )
        self.n_initial = min(n_initial, max_evals)
        self.n_trees = n_trees
        self.n_candidates = n_candidates
        self.backend = backend

        self._X_observed: List[List[float]] = []
        self._y_observed: List[float] = []
        self._rf_model: Optional[RandomForestRegressor] = None

    def _configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)
        for name in PARAM_NAMES:
            lo, hi = PARAM_BOUNDS[name]
            cs.add_hyperparameter(Float(name, (lo, hi), default=(lo + hi) / 2))
        return cs

    def _evaluate_one_sync(self, params: Dict[str, float], dry_run: bool = False) -> Dict:
        holder: Dict[str, object] = {}

        def runner():
            try:
                holder["rec"] = asyncio.run(self.evaluate_one(params, dry_run=dry_run))
            except Exception as exc:
                holder["exc"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        if "exc" in holder:
            raise holder["exc"]
        return holder["rec"]

    async def _optimize_official(self, dry_run: bool = False):
        if not SMAC_AVAILABLE:
            raise RuntimeError("SMAC official backend 需要安装官方 smac 和 ConfigSpace 包")

        scenario = Scenario(
            self._configspace(),
            n_trials=self.max_evals,
            deterministic=True,
            seed=self.seed,
            output_directory=str(self.output_dir / "smac3_output"),
        )

        def objective(config, seed: int = 0) -> float:
            params = snap({name: float(config[name]) for name in PARAM_NAMES})
            rec = self._evaluate_one_sync(params, dry_run=dry_run)
            self.finalize_ok_proposal(rec)
            self.write_eval_row(rec, generation=self.current_generation)
            fit = float(rec.get("fitness", PENALTY_FITNESS))
            self._X_observed.append(self._params_to_vector(params))
            self._y_observed.append(fit)
            if rec.get("status") == "ok":
                self.record_evaluation(fit, params)
            return -fit if rec.get("status") == "ok" else 1e6

        print(f"[SMAC] official smac backend | max_evals={self.max_evals}")
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=objective,
            overwrite=True,
        )
        incumbent = smac.optimize()
        print(f"[SMAC] official incumbent: {dict(incumbent)}")
        return incumbent

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
                if self.check_constraints(p, source="smac_random_sample"):
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
            if self.check_constraints(p, source="smac_local_sample"):
                return p

        for _ in range(5000):
            p = random_params(rng)
            if self.check_constraints(p, source="smac_fallback_sample"):
                return p
        return random_params(rng)

    def _fit_rf_model(self):
        """训练 Random Forest surrogate"""
        if not SKLEARN_AVAILABLE or len(self._X_observed) < 5:
            return

        X = np.array(self._X_observed)
        y = np.array(self._y_observed)

        # 过滤掉无效值（fitness <= 0 的失败点不参与建模）
        valid_mask = y > PENALTY_FITNESS
        if valid_mask.sum() < 5:
            valid_mask = np.ones(len(y), dtype=bool)

        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        self._rf_model = RandomForestRegressor(
            n_estimators=self.n_trees,
            max_depth=10,
            min_samples_split=3,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._rf_model.fit(X_valid, y_valid)

    def _acquisition_ei(self, X: np.ndarray) -> np.ndarray:
        """Expected Improvement 采集函数（最大化版本）"""
        if self._rf_model is None:
            return np.zeros(len(X))

        predictions = np.array([tree.predict(X) for tree in self._rf_model.estimators_])
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        # 当前最优
        valid_y = [y for y in self._y_observed if y > PENALTY_FITNESS]
        if not valid_y:
            return mean
        y_best = max(valid_y)

        std = np.maximum(std, 1e-8)

        from scipy.stats import norm
        z = (mean - y_best) / std
        ei = (mean - y_best) * norm.cdf(z) + std * norm.pdf(z)
        return ei

    def _select_next_point(self, rng: random.Random) -> Dict[str, float]:
        """使用 RF + EI 选择下一个评估点"""
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

        X_candidates = np.array([self._params_to_vector(p) for p in candidates])
        ei_values = self._acquisition_ei(X_candidates)
        best_idx = np.argmax(ei_values)

        chosen = candidates[best_idx]
        self.log_proposal(chosen, "smac_ei_select", "ok", [])
        return chosen

    async def optimize(self, dry_run: bool = False):
        if self.backend == "official":
            await self._optimize_official(dry_run=dry_run)
            return

        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        print(f"[SMAC] n_initial={self.n_initial}, n_trees={self.n_trees}, "
              f"max_evals={self.max_evals}")
        print(f"[SMAC] sklearn 可用: {SKLEARN_AVAILABLE}")
        print("=" * 60)

        # 断点恢复
        if hasattr(self, '_checkpoint_save_path') and self._checkpoint_save_path.exists():
            state = self.load_checkpoint(self._checkpoint_save_path)
            if state and "X_observed" in state:
                self._X_observed = state["X_observed"]
                self._y_observed = state["y_observed"]
                print(f"[SMAC] 从断点恢复: {len(self._X_observed)} 已观测点, "
                      f"eval_count={self.eval_count}, best={self.best_fitness:.4f}")

        # 阶段1: 初始随机采样（如果还没够初始点）
        if len(self._X_observed) < self.n_initial:
            self.current_generation = 0
            print(f"[SMAC] 阶段1: 生成 {self.n_initial} 个初始点...")

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
                    print(f"[SMAC] 初始采样: {len(self._X_observed)}/{self.n_initial} | "
                          f"best={self.best_fitness:.4f}")

            print(f"[SMAC] 阶段1完成: {len(self._X_observed)} 初始点")

        # 阶段2: RF-based 序贯优化
        self.current_generation = 1
        print(f"[SMAC] 阶段2: Random Forest 序贯优化...")

        iteration = 0
        while self.eval_count < self.max_evals and not self.should_stop_early():
            if iteration % 5 == 0:
                self._fit_rf_model()

            if self._rf_model is not None and rng.random() > 0.1:
                params = self._select_next_point(rng)
            else:
                params = self._generate_valid_sample(rng)

            rec = await self.evaluate_one(params, dry_run=dry_run)
            fit = rec["fitness"]
            self.finalize_ok_proposal(rec)
            self.write_eval_row(rec, generation=1)

            self._X_observed.append(self._params_to_vector(params))
            self._y_observed.append(fit)

            if self.record_evaluation(fit, params):
                print(f"  [{self.eval_count:4d}] 🎯 NEW BEST fitness={fit:8.4f}")
            elif self.eval_count % 20 == 0:
                print(f"  [{self.eval_count:4d}] best={self.best_fitness:.4f}")

            iteration += 1

            # 每 10 次保存断点
            if self.eval_count % 10 == 0:
                self.save_checkpoint(self._checkpoint_save_path, extra_state={
                    "X_observed": self._X_observed,
                    "y_observed": self._y_observed,
                })

            if self.should_stop_early():
                break
