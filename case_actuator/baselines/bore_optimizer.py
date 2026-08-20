# -*- coding: utf-8 -*-
"""
BORE (Bayesian Optimization by Density-Ratio Estimation) 优化器
使用神经网络分类器作为代理模型，通过密度比估计进行贝叶斯优化

论文: BORE: Bayesian Optimization by Density-Ratio Estimation
DOI: 10.48550/arXiv.2102.09009
GitHub: https://github.com/ltiao/bore

核心思想：
- 传统 BO 用 GP 建模 f(x)，然后用 EI/UCB 采集
- BORE 把"找最优点"转化为分类问题：学习 p(x|y < τ) / p(x) 的密度比
- 用神经网络分类器直接估计这个密度比，然后采样
"""
import os
# 解决 OpenMP 冲突问题（Anaconda + PyTorch 常见问题）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import random
import numpy as np
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger

from .base_optimizer import (
    BaseOptimizer, EvalResult, PARAM_BOUNDS, PARAM_NAMES, PENALTY_FITNESS,
    generate_valid_params, _quick_validate_params
)


class BOREOptimizer(BaseOptimizer):
    """
    BORE 优化器（基于神经网络的密度比估计）
    
    支持三种后端（按优先级）：
    1. bore-pytorch 库（如果已安装）
    2. PyTorch 自己实现密度比估计
    3. scikit-learn MLPClassifier（fallback）
    """
    
    # 类级别的可用性缓存
    _bore_available: Optional[bool] = None
    _torch_available: Optional[bool] = None
    _sklearn_available: Optional[bool] = None
    
    @classmethod
    def _check_bore_available(cls) -> bool:
        """延迟检测 bore-pytorch 是否可用"""
        if cls._bore_available is None:
            try:
                from bore.optimizers import BoREOptimizer as _BoREOptimizer
                from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
                cls._bore_available = True
                logger.info("[BORE] bore-pytorch 可用")
            except ImportError:
                cls._bore_available = False
                logger.debug("[BORE] bore-pytorch 不可用")
        return cls._bore_available
    
    @classmethod
    def _check_torch_available(cls) -> bool:
        """延迟检测 PyTorch 是否可用"""
        if cls._torch_available is None:
            try:
                import torch
                import torch.nn as nn
                _ = torch.zeros(1)
                cls._torch_available = True
                logger.info(f"[BORE] PyTorch 可用 (version={torch.__version__})")
            except Exception as e:
                cls._torch_available = False
                logger.warning(f"[BORE] PyTorch 不可用: {e}")
        return cls._torch_available
    
    @classmethod
    def _check_sklearn_available(cls) -> bool:
        """延迟检测 scikit-learn 是否可用"""
        if cls._sklearn_available is None:
            try:
                from sklearn.neural_network import MLPClassifier
                cls._sklearn_available = True
                logger.info("[BORE] scikit-learn MLPClassifier 可用")
            except ImportError:
                cls._sklearn_available = False
                logger.debug("[BORE] scikit-learn 不可用")
        return cls._sklearn_available
    
    def __init__(
        self,
        max_evals: int = 200,
        min_iterations: int = 200,
        convergence_window: int = 40,
        avg_window: int = 10,
        convergence_threshold: float = 0.01,
        n_initial: int = 20,
        gamma: float = 0.25,
        hidden_dims: List[int] = [64, 32],
        n_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.01,
        n_candidates: int = 1000,
        output_dir: str = ".",
        seed: int = 42,
        resume: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(
            max_evals=max_evals,
            min_iterations=min_iterations,
            convergence_window=convergence_window,
            avg_window=avg_window,
            convergence_threshold=convergence_threshold,
            output_dir=output_dir,
            algorithm_name="BORE",
            seed=seed,
        )
        self.n_initial = min(n_initial, max_evals)
        self.gamma = gamma
        self.hidden_dims = hidden_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_candidates = n_candidates
        
        self.resume = resume
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (self.output_dir / "BORE_checkpoint.pkl")
        self._resume_loaded = False
        self._resume_X_observed: List[List[float]] = []
        self._resume_y_observed: List[float] = []
        
        self._X_observed: List[List[float]] = []
        self._y_observed: List[float] = []
        
        # 模型（PyTorch 或 sklearn）
        self._torch_model = None
        self._sklearn_model = None
        
        logger.info(f"[BORE] 配置: n_initial={self.n_initial}, gamma={gamma}, max_evals={max_evals}")
        logger.info(f"[BORE] 网络结构: hidden_dims={hidden_dims}, n_epochs={n_epochs}, lr={lr}")
        logger.info(f"[BORE] 收敛条件: min_iter={min_iterations}, conv_window={convergence_window}")
        if self.resume:
            logger.info(f"[BORE] 断点续跑启用: checkpoint={self.checkpoint_path}")
    
    def _eval_result_to_dict(self, result: Optional[EvalResult]) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        return asdict(result)

    def _eval_result_from_dict(self, data: Optional[Dict[str, Any]]) -> Optional[EvalResult]:
        if not data:
            return None
        return EvalResult(**data)

    def _save_checkpoint(self):
        """保存断点"""
        state = {
            "X_observed": self._X_observed,
            "y_observed": self._y_observed,
            "eval_count": self.eval_count,
            "best_fitness": self.best_fitness,
            "best_result": self._eval_result_to_dict(self.best_result),
            "valid_fitness_history": self.valid_fitness_history,
            "no_improvement_count": self.no_improvement_count,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
        }
        
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(state, f)
        logger.debug(f"[BORE] 已保存断点: eval={self.eval_count}, best={self.best_fitness:.6f}")

    def _load_checkpoint(self) -> bool:
        """加载断点"""
        if not self.checkpoint_path.exists():
            logger.warning(f"[BORE] 未找到断点文件: {self.checkpoint_path}")
            return False
        try:
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            logger.error(f"[BORE] 断点文件读取失败: {e}")
            return False

        self.eval_count = state.get("eval_count", 0)
        self.best_fitness = state.get("best_fitness", float("inf"))
        self.best_result = self._eval_result_from_dict(state.get("best_result"))
        self.valid_fitness_history = state.get("valid_fitness_history", [])
        self.no_improvement_count = state.get("no_improvement_count", 0)
        self.converged = state.get("converged", False)
        self.convergence_reason = state.get("convergence_reason", "")
        self._resume_X_observed = state.get("X_observed", [])
        self._resume_y_observed = state.get("y_observed", [])

        csv_path = state.get("csv_path")
        if csv_path:
            self.csv_path = Path(csv_path)
            self.resume_csv = True

        rand_state = state.get("random_state")
        np_state = state.get("numpy_state")
        if rand_state:
            random.setstate(rand_state)
        if np_state is not None:
            np.random.set_state(np_state)

        self._resume_loaded = True
        logger.info(f"[BORE] 已加载断点: eval_count={self.eval_count}, best={self.best_fitness:.6f}, "
                   f"已有观测点={len(self._resume_X_observed)}")
        return True

    def run(self) -> EvalResult:
        """运行优化流程"""
        logger.info(f"=" * 60)
        logger.info(f"[BORE] 开始优化 | max_evals={self.max_evals} | seed={self.seed}")
        logger.info(f"=" * 60)
        
        # 检测可用后端
        bore_ok = self._check_bore_available()
        torch_ok = self._check_torch_available()
        sklearn_ok = self._check_sklearn_available()
        logger.info(f"[BORE] 后端检测: bore={bore_ok}, torch={torch_ok}, sklearn={sklearn_ok}")

        try:
            if self.resume:
                self._load_checkpoint()
            self._init_csv()
            result = self.optimize()
            return result
        finally:
            self._close_csv()
            logger.info(f"[BORE] 优化完成 | 最优 fitness={self.best_fitness:.6f}")
            logger.info(f"[BORE] 结果已保存: {self.csv_path}")
    
    def _generate_valid_sample(self, base_params: Optional[Dict[str, float]] = None, scale: float = 0.2) -> List[float]:
        """生成满足约束的采样点"""
        lower, upper = self.get_bounds()
        max_attempts = 1000
        
        if base_params is None:
            try:
                params, _ = generate_valid_params(seed=random.randint(0, 1000000))
                return self.params_to_vector(params)
            except RuntimeError:
                return [random.uniform(lower[j], upper[j]) for j in range(len(PARAM_NAMES))]
        
        for _ in range(max_attempts):
            x_new = []
            for j, name in enumerate(PARAM_NAMES):
                base_val = base_params.get(name, (lower[j] + upper[j]) / 2)
                delta = (upper[j] - lower[j]) * scale
                val = base_val + random.gauss(0, delta)
                val = max(lower[j], min(upper[j], val))
                x_new.append(round(val, 2))
            
            params = self.vector_to_params(x_new)
            errors = _quick_validate_params(params)
            if not errors:
                return x_new
        
        try:
            params, _ = generate_valid_params(seed=random.randint(0, 1000000))
            return self.params_to_vector(params)
        except RuntimeError:
            return self.params_to_vector(base_params)
    
    def _normalize_x(self, X: np.ndarray) -> np.ndarray:
        """将参数归一化到 [0, 1]"""
        lower, upper = self.get_bounds()
        lower = np.array(lower)
        upper = np.array(upper)
        return (X - lower) / (upper - lower + 1e-8)
    
    def _train_model_torch(self, X: np.ndarray, y: np.ndarray) -> bool:
        """使用 PyTorch 训练密度比估计模型"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        tau = np.quantile(y, self.gamma)
        labels = (y < tau).astype(np.float32)
        X_norm = self._normalize_x(X).astype(np.float32)
        
        X_tensor = torch.from_numpy(X_norm)
        y_tensor = torch.from_numpy(labels).unsqueeze(1)
        
        # 构建模型
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
        optimizer = optim.Adam(self._torch_model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=min(self.batch_size, len(X)), shuffle=True
        )
        
        self._torch_model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self._torch_model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 50 == 0:
                logger.debug(f"[BORE] PyTorch 训练 epoch {epoch+1}/{self.n_epochs}, loss={total_loss/len(dataloader):.4f}")
        
        self._torch_model.eval()
        return True
    
    def _train_model_sklearn(self, X: np.ndarray, y: np.ndarray) -> bool:
        """使用 scikit-learn 训练密度比估计模型"""
        from sklearn.neural_network import MLPClassifier
        
        tau = np.quantile(y, self.gamma)
        labels = (y < tau).astype(int)
        X_norm = self._normalize_x(X)
        
        self._sklearn_model = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_dims),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=self.n_epochs,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        
        # 需要至少两个类别
        if len(np.unique(labels)) < 2:
            logger.warning("[BORE] 只有一个类别，跳过训练")
            return False
        
        self._sklearn_model.fit(X_norm, labels)
        logger.debug(f"[BORE] sklearn 训练完成, iterations={self._sklearn_model.n_iter_}")
        return True
    
    def _train_density_ratio_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练密度比估计模型（自动选择后端）"""
        if self._check_torch_available():
            try:
                self._train_model_torch(X, y)
                return
            except Exception as e:
                logger.warning(f"[BORE] PyTorch 训练失败: {e}，尝试 sklearn")
        
        if self._check_sklearn_available():
            try:
                self._train_model_sklearn(X, y)
                return
            except Exception as e:
                logger.warning(f"[BORE] sklearn 训练失败: {e}")
        
        logger.warning("[BORE] 无可用后端，跳过模型训练")
    
    def _predict_scores(self, candidates: List[List[float]]) -> np.ndarray:
        """预测候选点的"好点"概率"""
        candidates_np = np.array(candidates, dtype=np.float32)
        candidates_norm = self._normalize_x(candidates_np)
        
        if self._torch_model is not None and self._check_torch_available():
            import torch
            with torch.no_grad():
                scores = self._torch_model(torch.from_numpy(candidates_norm.astype(np.float32))).numpy().flatten()
            return scores
        
        if self._sklearn_model is not None:
            scores = self._sklearn_model.predict_proba(candidates_norm)[:, 1]
            return scores
        
        return np.random.rand(len(candidates))
    
    def _suggest_next(self) -> List[float]:
        """使用训练好的模型建议下一个评估点"""
        has_model = (self._torch_model is not None) or (self._sklearn_model is not None)
        
        if not has_model:
            if self._y_observed:
                best_idx = np.argmin(self._y_observed)
                best_params = self.vector_to_params(self._X_observed[best_idx])
                return self._generate_valid_sample(best_params, scale=0.2)
            return self._generate_valid_sample()
        
        candidates = []
        for _ in range(self.n_candidates):
            if random.random() < 0.8 or not self._y_observed:
                x = self._generate_valid_sample()
            else:
                best_idx = np.argmin(self._y_observed)
                best_params = self.vector_to_params(self._X_observed[best_idx])
                x = self._generate_valid_sample(best_params, scale=0.15)
            candidates.append(x)
        
        scores = self._predict_scores(candidates)
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def optimize(self) -> EvalResult:
        """执行 BORE 优化"""
        if not self._resume_loaded:
            random.seed(self.seed)
            np.random.seed(self.seed)
            if self._check_torch_available():
                import torch
                torch.manual_seed(self.seed)
        
        # 检查是否有可用后端
        has_backend = (
            self._check_bore_available() or 
            self._check_torch_available() or 
            self._check_sklearn_available()
        )
        
        if not has_backend:
            logger.error("[BORE] 没有可用的后端！")
            logger.error("[BORE] 请安装以下任一依赖:")
            logger.error("[BORE]   pip install torch  (推荐)")
            logger.error("[BORE]   pip install scikit-learn")
            raise RuntimeError("BORE 需要 PyTorch 或 scikit-learn")
        
        # 使用简化版实现（兼容 PyTorch 和 sklearn）
        return self._optimize_simple()
    
    def _optimize_simple(self) -> EvalResult:
        """简化版 BORE 实现"""
        logger.info(f"[BORE] 使用简化版神经网络密度比估计")
        
        if self._resume_loaded:
            self._X_observed = list(self._resume_X_observed)
            self._y_observed = list(self._resume_y_observed)
            logger.info(f"[BORE] 从断点恢复: eval={self.eval_count}, 已有观测点={len(self._X_observed)}")
        
        # 阶段1: 初始随机采样
        logger.info(f"[BORE] 阶段1: 生成 {self.n_initial} 个初始点...")
        while len(self._X_observed) < self.n_initial and self.eval_count < self.max_evals:
            if self.converged:
                logger.info(f"[BORE] 已收敛，停止迭代")
                return self.best_result
            
            x = self._generate_valid_sample()
            params = self.vector_to_params(x)
            
            try:
                result = self.evaluate(params)
                self.results.append(result)
                self._write_csv_row(result, iteration=self.eval_count)
                
                self._X_observed.append(x)
                self._y_observed.append(result.fitness)
            except Exception as e:
                logger.error(f"[BORE] 初始采样评估异常: {e}，使用惩罚值")
                self._X_observed.append(x)
                self._y_observed.append(PENALTY_FITNESS)
            
            self._save_checkpoint()
            
            if len(self._X_observed) % 5 == 0:
                best_so_far = min(self._y_observed) if self._y_observed else float('inf')
                logger.info(f"[BORE] 初始采样进度: {len(self._X_observed)}/{self.n_initial} | 当前最优={best_so_far:.6f}")
        
        logger.info(f"[BORE] 阶段1完成: {len(self._X_observed)} 个初始点")
        
        # 阶段2: 神经网络引导的优化
        logger.info(f"[BORE] 阶段2: 神经网络引导优化...")
        iteration = 0
        retrain_interval = 5
        
        while self.eval_count < self.max_evals:
            if self.converged:
                logger.info(f"[BORE] 已收敛，停止迭代")
                break
            
            if iteration % retrain_interval == 0 and len(self._X_observed) >= self.n_initial:
                X = np.array(self._X_observed)
                y = np.array(self._y_observed)
                valid_mask = y < PENALTY_FITNESS * 0.1
                if valid_mask.sum() >= 5:
                    self._train_density_ratio_model(X[valid_mask], y[valid_mask])
                else:
                    self._train_density_ratio_model(X, y)
            
            x_new = self._suggest_next()
            params = self.vector_to_params(x_new)
            
            try:
                result = self.evaluate(params)
                self.results.append(result)
                self._write_csv_row(result, iteration=self.eval_count)
                
                self._X_observed.append(x_new)
                self._y_observed.append(result.fitness)
            except Exception as e:
                logger.error(f"[BORE] 阶段2评估异常: {e}，使用惩罚值继续")
                self._X_observed.append(x_new)
                self._y_observed.append(PENALTY_FITNESS)
            
            self._save_checkpoint()
            
            iteration += 1
            if iteration % 10 == 0:
                valid_y = [y for y in self._y_observed if y < PENALTY_FITNESS * 0.1]
                best_so_far = min(valid_y) if valid_y else float('inf')
                logger.info(f"[BORE] 进度: {self.eval_count}/{self.max_evals} | 当前最优={best_so_far:.6f}")
        
        logger.info(f"[BORE] 优化结束 | 总评估次数={self.eval_count} | 最优={self.best_fitness:.6f}")
        return self.best_result
