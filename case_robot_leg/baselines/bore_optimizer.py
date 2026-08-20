# -*- coding: utf-8 -*-
"""BORE optimization for the robot-leg task.

The optimizer casts the objective as density-ratio estimation and trains a
classifier to distinguish high-performing candidates.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# agent_maxwell 环境里 torch 会触发 Windows 原生 DLL 崩溃；
# robot 任务默认用 sklearn 训练分类器，仍强制使用官方 bore 优化候选点。
if os.getenv("BORE_USE_TORCH", "0") == "1":
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        TORCH_AVAILABLE = True
    except Exception as e:
        print(f"[BORE] PyTorch 不可用，将使用 sklearn fallback: {e}")
        torch = None
        nn = None
        optim = None
        TORCH_AVAILABLE = False
else:
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False

import random
import importlib.util
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

from .base_optimizer import (
    BaseOptimizer, PARAM_NAMES, PARAM_BOUNDS,
    PENALTY_FITNESS, snap, clamp, random_params,
)

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
OFFICIAL_BORE_AVAILABLE = importlib.util.find_spec("bore") is not None

class BOREOptimizer(BaseOptimizer):
    """BORE 优化器（密度比估计，机器人腿）"""

    def __init__(
        self,
        max_evals: int = 200,
        n_initial: int = 20,
        gamma: float = 0.25,
        hidden_dims: List[int] = None,
        n_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.01,
        n_candidates: int = 1000,
        backend: Literal["auto", "official", "torch", "sklearn"] = "official",
        output_dir: str = ".",
        seed: int = 42,
    ):
        algorithm_name = "BORE_official" if backend == "official" else f"BORE_{backend}"
        super().__init__(
            max_evals=max_evals,
            output_dir=output_dir,
            algorithm_name=algorithm_name,
            seed=seed,
        )
        self.n_initial = min(n_initial, max_evals)
        self.gamma = gamma
        self.hidden_dims = hidden_dims or [64, 32]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_candidates = n_candidates
        self.backend = backend

        self._X_observed: List[List[float]] = []
        self._y_observed: List[float] = []
        self._torch_model = None
        self._sklearn_model = None
        self._model_backend: Optional[str] = None
        self._official_bore_available = OFFICIAL_BORE_AVAILABLE
        self._candidate_optimizer = (
            "bore.optimizers.base.minimize_multi_start"
            if self.backend in ("official", "auto")
            else f"local_{self.backend}_candidate_screen"
        )
        self._seen_param_keys = set()
        self._reject_log_budget = 0
        self.max_reject_logs_per_suggestion = 3

    def _save_bore_checkpoint(self):
        """保存 BORE 状态；每次真实评估后调用，防止长仿真中断丢进度。"""
        self.save_checkpoint(self._checkpoint_save_path, extra_state={
            "X_observed": self._X_observed,
            "y_observed": self._y_observed,
            "seen_param_keys": list(self._seen_param_keys),
            "backend": self.backend,
            "model_backend": self._model_backend,
            "official_bore_available": OFFICIAL_BORE_AVAILABLE,
            "candidate_optimizer": self._candidate_optimizer,
        })

    def _params_to_vector(self, params: Dict[str, float]) -> List[float]:
        return [params[name] for name in PARAM_NAMES]

    def _vector_to_params(self, x: List[float]) -> Dict[str, float]:
        return snap({name: x[i] for i, name in enumerate(PARAM_NAMES)})

    def _params_key(self, params: Dict[str, float]) -> Tuple[float, ...]:
        return tuple(params[name] for name in PARAM_NAMES)

    def _normalize_x(self, X: np.ndarray) -> np.ndarray:
        lower = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES])
        upper = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES])
        return (X - lower) / (upper - lower + 1e-8)

    def _denormalize_x(self, X_norm: np.ndarray) -> np.ndarray:
        lower = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES], dtype=float)
        upper = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES], dtype=float)
        return lower + np.clip(X_norm, 0.0, 1.0) * (upper - lower)

    def _is_valid(self, params: Dict[str, float],
                  source: str = "bore_candidate_filter") -> bool:
        from mymcp.tool.robot_leg import _validate_design

        result = _validate_design(
            params["m"], params["n"], params["alpha"],
            params["beta"], params["DIST_BETTERY"]
        )
        errors = list(result.get("errors", []))

        ok = result.get("status") == "ok" and not errors
        if (not ok and self._reject_log_budget > 0 and
                self._eval_writer is not None):
            self.log_proposal(params, source, "constraint_violation", errors)
            self._reject_log_budget -= 1
        return ok

    def _sample_local_param(self, rng: random.Random, name: str,
                            center: float, scale: float) -> float:
        lo, hi = PARAM_BOUNDS[name]
        sigma = (hi - lo) * scale
        for _ in range(30):
            val = center + rng.gauss(0, sigma)
            if lo <= val <= hi:
                return val
        return rng.uniform(lo, hi)

    def _generate_bore_valid_random(self, rng: random.Random,
                                    max_attempts: int = 20000) -> Dict[str, float]:
        for _ in range(max_attempts):
            p = random_params(rng)
            if self._is_valid(p, "bore_random_reject"):
                return p
        return random_params(rng)

    def _generate_valid_sample(self, rng: random.Random,
                               base_params: Optional[Dict[str, float]] = None,
                               scale: float = 0.2,
                               avoid_seen: bool = True) -> Dict[str, float]:
        """Generate a feasible sample without logging failed internal trials."""
        if base_params is None:
            for _ in range(5000):
                p = random_params(rng)
                if self._is_valid(p, "bore_random_reject") and (not avoid_seen or self._params_key(p) not in self._seen_param_keys):
                    return p
            return self._generate_bore_valid_random(rng)

        for _ in range(2000):
            p = {}
            for name in PARAM_NAMES:
                p[name] = self._sample_local_param(rng, name, base_params[name], scale)
            p = snap(p)
            if self._is_valid(p, "bore_local_reject") and (not avoid_seen or self._params_key(p) not in self._seen_param_keys):
                return p

        for _ in range(5000):
            p = random_params(rng)
            if self._is_valid(p, "bore_fallback_reject") and (not avoid_seen or self._params_key(p) not in self._seen_param_keys):
                return p
        return self._generate_bore_valid_random(rng)

    def _prepare_training_data(self, X: np.ndarray, y: np.ndarray):
        """筛选可用于密度比训练的有效样本。"""
        y = np.asarray(y, dtype=float)
        valid_mask = y > PENALTY_FITNESS
        if valid_mask.sum() >= 3:
            return X[valid_mask], y[valid_mask]
        pos_mask = y > 0
        if pos_mask.sum() >= 2:
            return X[pos_mask], y[pos_mask]
        return None, None

    def _density_labels(self, y: np.ndarray) -> np.ndarray:
        """密度比二分类标签；样本极少或分位数退化时回退到 top-gamma。"""
        y = np.asarray(y, dtype=float)
        tau = np.quantile(y, 1 - self.gamma)
        labels = (y >= tau).astype(int)
        if len(np.unique(labels)) < 2 and len(y) >= 2:
            k = max(1, int(np.ceil(len(y) * self.gamma)))
            order = np.argsort(y)
            labels = np.zeros(len(y), dtype=int)
            labels[order[-k:]] = 1
        return labels

    def _train_model_torch(self, X: np.ndarray, y: np.ndarray) -> bool:
        """使用 PyTorch 训练密度比估计分类器"""
        if len(y) < 2 or len(np.unique(self._density_labels(y))) < 2:
            return False
        tau = np.quantile(y, 1 - self.gamma)  # 注意：我们是最大化，所以取 top gamma 是大值
        labels = self._density_labels(y).astype(np.float32)
        X_norm = self._normalize_x(X).astype(np.float32)

        X_tensor = torch.from_numpy(X_norm)
        y_tensor = torch.from_numpy(labels).unsqueeze(1)

        layers = []
        prev_dim = len(PARAM_NAMES)
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self._torch_model = nn.Sequential(*layers)
        optimizer_nn = optim.Adam(self._torch_model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=min(self.batch_size, len(X)), shuffle=True
        )

        self._torch_model.train()
        for epoch in range(self.n_epochs):
            for batch_x, batch_y in dataloader:
                optimizer_nn.zero_grad()
                pred = self._torch_model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer_nn.step()

        self._torch_model.eval()
        self._model_backend = "torch"
        return True

    def _train_model_sklearn(self, X: np.ndarray, y: np.ndarray) -> bool:
        """使用 sklearn 训练密度比估计分类器"""
        from sklearn.neural_network import MLPClassifier

        labels = self._density_labels(y)
        X_norm = self._normalize_x(X)

        if len(y) < 2 or len(np.unique(labels)) < 2:
            return False

        self._sklearn_model = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_dims),
            activation='relu',
            solver='adam',
            max_iter=self.n_epochs,
            random_state=self.seed,
            # BORE 的样本量很小且类别经常不均衡；sklearn 的 early_stopping
            # 会内部做分层验证集切分，容易在正式仿真中途因 test_size 太小失败。
            early_stopping=False,
        )
        self._sklearn_model.fit(X_norm, labels)
        self._model_backend = "sklearn"
        return True

    def _train_density_ratio_model(self, X: np.ndarray, y: np.ndarray):
        """训练密度比模型（自动选择后端）"""
        X_fit, y_fit = self._prepare_training_data(np.asarray(X), np.asarray(y))
        if X_fit is None:
            print(f"[{self.algorithm_name}] 训练样本不足，跳过分类器训练")
            self._torch_model = None
            self._sklearn_model = None
            self._model_backend = None
            return

        preferred = self.backend
        if preferred == "official":
            # Train the classifier locally and use the official BORE
            # multi-start optimizer to generate the next candidate.
            preferred = "torch" if TORCH_AVAILABLE else "sklearn"

        if preferred in ("auto", "torch") and TORCH_AVAILABLE:
            try:
                if self._train_model_torch(X_fit, y_fit):
                    return
            except Exception as e:
                if self.backend == "torch":
                    raise
                print(f"[{self.algorithm_name}] PyTorch 训练失败，尝试 sklearn: {e}")

        if preferred in ("auto", "sklearn", "torch") and SKLEARN_AVAILABLE:
            try:
                if self._train_model_sklearn(X_fit, y_fit):
                    return
            except Exception as e:
                if self.backend == "sklearn":
                    raise
                print(f"[{self.algorithm_name}] sklearn 训练失败: {e}")

        print(
            f"[{self.algorithm_name}] 分类器暂不可训练（样本={len(y_fit)}，"
            f"正样本={int(self._density_labels(y_fit).sum())}），"
            f"阶段2 将回退随机/精英邻域采样"
        )
        self._torch_model = None
        self._sklearn_model = None
        self._model_backend = None

    def _predict_scores(self, candidates: List[Dict[str, float]]) -> np.ndarray:
        """预测候选点的"好点"概率"""
        X = np.array([self._params_to_vector(p) for p in candidates], dtype=np.float32)
        X_norm = self._normalize_x(X)

        return self._predict_scores_norm(X_norm)

    def _predict_scores_norm(self, X_norm: np.ndarray) -> np.ndarray:
        """预测归一化候选点的"好点"概率"""
        X_norm = np.asarray(X_norm, dtype=np.float32)

        if self._torch_model is not None and self._model_backend == "torch" and TORCH_AVAILABLE:
            with torch.no_grad():
                scores = self._torch_model(
                    torch.from_numpy(X_norm.astype(np.float32))
                ).numpy().flatten()
            return scores

        if self._sklearn_model is not None and self._model_backend == "sklearn":
            scores = self._sklearn_model.predict_proba(X_norm)[:, 1]
            return scores

        return np.random.rand(len(X_norm))

    def _official_candidate_to_params(self, x_norm: np.ndarray) -> Optional[Dict[str, float]]:
        x = self._denormalize_x(np.asarray(x_norm, dtype=float))
        params = self._vector_to_params([float(v) for v in x])
        key = self._params_key(params)
        if key in self._seen_param_keys:
            return None
        if not self._is_valid(params, "bore_official_reject"):
            return None
        return params

    def _suggest_next_official_bore(self) -> Optional[Dict[str, float]]:
        """使用官方 bore 多起点优化器在归一化空间中最大化分类器分数。"""
        if not OFFICIAL_BORE_AVAILABLE:
            if self.backend == "official":
                raise RuntimeError("已指定 --bore-backend official，但官方 bore 包不可用")
            return None

        from bore.optimizers import base as bore_base

        def objective(x_norm: np.ndarray):
            x_norm = np.asarray(x_norm, dtype=np.float32)
            is_single = x_norm.ndim == 1
            x_batch = x_norm.reshape(1, -1) if is_single else x_norm
            values = -self._predict_scores_norm(x_batch)

            penalties = []
            for row in x_batch:
                penalties.append(0.0 if self._official_candidate_to_params(row) is not None else 1000.0)
            values = values + np.asarray(penalties, dtype=np.float32)

            if is_single:
                grad = np.zeros_like(x_norm, dtype=np.float64)
                return float(values[0]), grad
            return values.astype(np.float64), np.zeros_like(x_batch, dtype=np.float64)

        try:
            results = bore_base.minimize_multi_start(
                objective,
                bounds=[(0.0, 1.0)] * len(PARAM_NAMES),
                num_starts=max(5, min(20, self.n_candidates // 50)),
                num_samples=max(20, min(self.n_candidates, 500)),
                random_state=self.seed + self.eval_count,
                method="L-BFGS-B",
                jac=True,
            )
        except Exception as e:
            if self.backend == "official":
                raise RuntimeError(f"官方 bore minimize_multi_start 失败: {e}") from e
            print(f"[{self.algorithm_name}] 官方 bore 候选优化失败，回退本地候选筛选: {e}")
            return None

        ranked = sorted(
            [r for r in results if getattr(r, "x", None) is not None],
            key=lambda r: float(getattr(r, "fun", float("inf"))),
        )
        for result in ranked:
            params = self._official_candidate_to_params(result.x)
            if params is not None:
                self.log_proposal(params, "bore_official_minimize_multi_start", "ok", [])
                print(f"[{self.algorithm_name}] 使用官方 bore 多起点优化建议候选点")
                return params

        if self.backend == "official":
            raise RuntimeError("BORE official backend 未找到满足约束的官方候选点")
        return None

    def _elite_params(self, max_elites: int = 10) -> List[Dict[str, float]]:
        if not self._X_observed:
            return []
        ranked = sorted(
            zip(self._y_observed, self._X_observed),
            key=lambda item: item[0],
            reverse=True,
        )
        elites = []
        for fit, x in ranked:
            if fit <= PENALTY_FITNESS:
                continue
            elites.append(self._vector_to_params(x))
            if len(elites) >= max_elites:
                break
        return elites

    def _suggest_next(self, rng: random.Random) -> Dict[str, float]:
        """使用训练好的模型建议下一个评估点"""
        has_model = (
            self._torch_model is not None
            or self._sklearn_model is not None
        )
        self._reject_log_budget = self.max_reject_logs_per_suggestion

        if not has_model:
            print(
                f"[{self.algorithm_name}] 分类器未就绪，回退随机/精英邻域采样"
            )
            elites = self._elite_params()
            if elites:
                return self._generate_valid_sample(rng, rng.choice(elites), scale=0.15)
            if self.best_params:
                return self._generate_valid_sample(rng, self.best_params, scale=0.2)
            return self._generate_valid_sample(rng)

        if self.backend in ("auto", "official"):
            params = self._suggest_next_official_bore()
            if params is not None:
                self._reject_log_budget = 0
                return params
            if self.backend == "official":
                raise RuntimeError("BORE official backend 未能产生官方候选点")

        # 生成可行候选点。内部筛选不写 CSV，避免把候选失败误当作真实评估。
        candidates = []
        candidate_keys = set()
        elites = self._elite_params()
        max_attempts = max(self.n_candidates * 80, 5000)
        for _ in range(max_attempts):
            use_global = rng.random() < 0.7 or not elites
            if use_global:
                p = random_params(rng)
                if not self._is_valid(p, "bore_candidate_reject"):
                    continue
            else:
                base = rng.choice(elites)
                p = self._generate_valid_sample(rng, base, scale=0.08)

            key = self._params_key(p)
            if key in self._seen_param_keys or key in candidate_keys:
                continue
            candidates.append(p)
            candidate_keys.add(key)

            if len(candidates) >= self.n_candidates:
                break

        if not candidates:
            return self._generate_valid_sample(rng)

        scores = self._predict_scores(candidates)
        best_idx = np.argmax(scores)
        chosen = candidates[best_idx]
        self._reject_log_budget = 0
        self.log_proposal(chosen, "bore_nn_select", "ok", [])
        return chosen

    async def optimize(self, dry_run: bool = False):
        rng = random.Random(self.seed)
        np.random.seed(self.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)

        print(f"[{self.algorithm_name}] n_initial={self.n_initial}, gamma={self.gamma}, "
              f"hidden={self.hidden_dims}, max_evals={self.max_evals}")
        print(f"[{self.algorithm_name}] backend={self.backend}, candidate_optimizer={self._candidate_optimizer}")
        print(f"[{self.algorithm_name}] official_bore={OFFICIAL_BORE_AVAILABLE}, PyTorch={TORCH_AVAILABLE}, sklearn={SKLEARN_AVAILABLE}")
        print(f"[{self.algorithm_name}] Checkpoint : {self._checkpoint_save_path}")
        print("=" * 60)

        if self.backend == "official" and not OFFICIAL_BORE_AVAILABLE:
            raise RuntimeError("已指定 --bore-backend official，但官方 bore 包不可用")
        if self.backend == "torch" and not TORCH_AVAILABLE:
            raise RuntimeError("已指定 --bore-backend torch，但 PyTorch 不可用")
        if self.backend == "sklearn" and not SKLEARN_AVAILABLE:
            raise RuntimeError("已指定 --bore-backend sklearn，但 scikit-learn 不可用")
        if not TORCH_AVAILABLE and not SKLEARN_AVAILABLE:
            raise RuntimeError("BORE 需要 PyTorch 或 scikit-learn")

        # 断点恢复
        if hasattr(self, '_checkpoint_save_path') and self._checkpoint_save_path.exists():
            state = self.load_checkpoint(self._checkpoint_save_path)
            if state and "X_observed" in state:
                self._X_observed = state["X_observed"]
                self._y_observed = state["y_observed"]
                self._seen_param_keys = {
                    self._params_key(self._vector_to_params(x))
                    for x in self._X_observed
                }
                print(f"[BORE] 从断点恢复: {len(self._X_observed)} 已观测点, "
                      f"eval_count={self.eval_count}, best={self.best_fitness:.4f}")

        # 阶段1: 初始随机采样（如果还没够初始点）
        if len(self._X_observed) < self.n_initial:
            self.current_generation = 0
            print(f"[BORE] 阶段1: 生成 {self.n_initial} 个初始点...")

            while len(self._X_observed) < self.n_initial and self.eval_count < self.max_evals:
                params = self._generate_valid_sample(rng)

                rec = await self.evaluate_one(params, dry_run=dry_run)
                fit = rec["fitness"]
                self.finalize_ok_proposal(rec)
                self.write_eval_row(rec, generation=0)

                self._X_observed.append(self._params_to_vector(params))
                self._y_observed.append(fit)
                self._seen_param_keys.add(self._params_key(params))

                self.record_evaluation(fit, params)

                self._save_bore_checkpoint()

                if len(self._X_observed) % 5 == 0:
                    print(f"[BORE] 初始采样: {len(self._X_observed)}/{self.n_initial} | "
                          f"best={self.best_fitness:.4f}")

            print(f"[BORE] 阶段1完成: {len(self._X_observed)} 初始点")

        # 阶段2: 神经网络引导优化
        self.current_generation = 1
        print(f"[BORE] 阶段2: 神经网络引导优化...")
        retrain_interval = 5
        iteration = 0

        while self.eval_count < self.max_evals and not self.should_stop_early():
            if iteration % retrain_interval == 0 and len(self._X_observed) >= self.n_initial:
                X = np.array(self._X_observed)
                y = np.array(self._y_observed)
                self._train_density_ratio_model(X, y)

            params = self._suggest_next(rng)

            rec = await self.evaluate_one(params, dry_run=dry_run)
            fit = rec["fitness"]
            self.finalize_ok_proposal(rec)
            self.write_eval_row(rec, generation=1)

            self._X_observed.append(self._params_to_vector(params))
            self._y_observed.append(fit)
            self._seen_param_keys.add(self._params_key(params))

            if self.record_evaluation(fit, params):
                print(f"  [{self.eval_count:4d}] 🎯 NEW BEST fitness={fit:8.4f}")
            elif self.eval_count % 10 == 0:
                print(f"  [{self.eval_count:4d}] best={self.best_fitness:.4f}")

            iteration += 1

            self._save_bore_checkpoint()

            if self.should_stop_early():
                break
