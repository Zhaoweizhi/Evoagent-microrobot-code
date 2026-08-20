# -*- coding: utf-8 -*-
"""
SMAC (Sequential Model-based Algorithm Configuration) 优化器
使用 Random Forest 作为 surrogate model，代表 Learned Black-box Optimization

SMAC 是 AutoML 领域的标准基线，其核心特点：
1. 使用 Random Forest 替代 GP 作为 surrogate（可扩展到高维）
2. 学习式的 surrogate 建模（从数据中学习）
3. 支持条件参数和混合参数空间

这是文献综述中"Learned black-box optimization"类别的代表方法。
"""
import pickle
import random
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import asdict
from loguru import logger

from .base_optimizer import (
    BaseOptimizer, EvalResult, PARAM_BOUNDS, PARAM_NAMES, PENALTY_FITNESS,
    generate_valid_params, _quick_validate_params
)

# 尝试导入 SMAC
try:
    from smac import HyperparameterOptimizationFacade, Scenario
    from ConfigSpace import ConfigurationSpace, Float
    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False
    logger.warning("SMAC3 未安装，将使用 Random Forest 替代实现")

# 尝试导入 sklearn 的 Random Forest（作为 fallback）
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SMACOptimizer(BaseOptimizer):
    """SMAC 优化器（使用 Random Forest surrogate）"""
    
    def __init__(
        self,
        max_evals: int = 200,
        min_iterations: int = 200,
        convergence_window: int = 40,
        avg_window: int = 10,
        convergence_threshold: float = 0.01,
        n_initial: int = 20,
        n_trees: int = 10,
        output_dir: str = ".",
        seed: int = 42,
        resume: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Args:
            max_evals: 最大评估次数
            min_iterations: 最小迭代次数
            convergence_window: 收敛窗口
            avg_window: 早停窗口
            convergence_threshold: 收敛阈值
            n_initial: 初始随机采样数量
            n_trees: Random Forest 树的数量
            output_dir: 输出目录
            seed: 随机种子
            resume: 是否断点续跑
            checkpoint_path: 断点文件路径
        """
        super().__init__(
            max_evals=max_evals,
            min_iterations=min_iterations,
            convergence_window=convergence_window,
            avg_window=avg_window,
            convergence_threshold=convergence_threshold,
            output_dir=output_dir,
            algorithm_name="SMAC",
            seed=seed,
        )
        self.n_initial = min(n_initial, max_evals)
        self.n_trees = n_trees
        
        # 断点续跑相关
        self.resume = resume
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (self.output_dir / "SMAC_checkpoint.pkl")
        self._resume_loaded = False
        
        # 观测数据
        self._X_observed: List[List[float]] = []
        self._y_observed: List[float] = []
        
        # Random Forest surrogate
        self._rf_model: Optional[RandomForestRegressor] = None
        
        logger.info(f"[SMAC] 配置: n_initial={self.n_initial}, n_trees={n_trees}, max_evals={max_evals}")
        logger.info(f"[SMAC] 收敛条件: min_iter={min_iterations}, conv_window={convergence_window}")
        logger.info(f"[SMAC] SMAC3 可用: {SMAC_AVAILABLE}, sklearn 可用: {SKLEARN_AVAILABLE}")
        if self.resume:
            logger.info(f"[SMAC] 断点续跑启用: checkpoint={self.checkpoint_path}")
    
    # ========== 断点续跑 ==========
    
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
        logger.debug(f"[SMAC] 已保存断点: eval={self.eval_count}, best={self.best_fitness:.6f}")

    def _load_checkpoint(self) -> bool:
        """加载断点"""
        if not self.checkpoint_path.exists():
            logger.warning(f"[SMAC] 未找到断点文件: {self.checkpoint_path}")
            return False
        try:
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            logger.error(f"[SMAC] 断点文件读取失败: {e}")
            return False

        self.eval_count = state.get("eval_count", 0)
        self.best_fitness = state.get("best_fitness", float("inf"))
        self.best_result = self._eval_result_from_dict(state.get("best_result"))
        self.valid_fitness_history = state.get("valid_fitness_history", [])
        self.no_improvement_count = state.get("no_improvement_count", 0)
        self.converged = state.get("converged", False)
        self.convergence_reason = state.get("convergence_reason", "")
        self._X_observed = state.get("X_observed", [])
        self._y_observed = state.get("y_observed", [])

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
        logger.info(f"[SMAC] 已加载断点: eval_count={self.eval_count}, best={self.best_fitness:.6f}")
        return True

    def run(self) -> EvalResult:
        """运行优化流程"""
        logger.info(f"=" * 60)
        logger.info(f"[SMAC] 开始优化 | max_evals={self.max_evals} | seed={self.seed}")
        logger.info(f"=" * 60)

        try:
            if self.resume:
                self._load_checkpoint()
            self._init_csv()
            result = self.optimize()
            return result
        finally:
            self._close_csv()
            logger.info(f"[SMAC] 优化完成 | 最优 fitness={self.best_fitness:.6f}")
            logger.info(f"[SMAC] 结果已保存: {self.csv_path}")

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

    def _fit_rf_model(self):
        """训练 Random Forest surrogate"""
        if not SKLEARN_AVAILABLE:
            return
        
        if len(self._X_observed) < 5:
            return
        
        X = np.array(self._X_observed)
        y = np.array(self._y_observed)
        
        # 过滤掉惩罚值（避免影响模型）
        valid_mask = y < PENALTY_FITNESS * 0.9
        if valid_mask.sum() < 5:
            return
        
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
        logger.debug(f"[SMAC] RF 模型已更新: {len(X_valid)} 个训练样本")

    def _predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """使用 RF 预测均值和不确定度"""
        if self._rf_model is None:
            return None, None
        
        # 获取每棵树的预测
        predictions = np.array([tree.predict(X) for tree in self._rf_model.estimators_])
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std

    def _acquisition_ei(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement 采集函数"""
        mean, std = self._predict_with_uncertainty(X)
        if mean is None:
            return np.zeros(len(X))
        
        # 当前最优（注意：我们是最小化）
        y_best = min(self._y_observed)
        
        # 避免除零
        std = np.maximum(std, 1e-8)
        
        # EI 计算（最小化版本）
        from scipy.stats import norm
        z = (y_best - mean - xi) / std
        ei = (y_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
        
        return ei

    def _select_next_point_rf(self) -> List[float]:
        """使用 RF surrogate 选择下一个评估点"""
        lower, upper = self.get_bounds()
        
        # 生成候选点（约束感知）
        n_candidates = 1000
        candidates = []
        for _ in range(n_candidates * 3):
            x = self._generate_valid_sample()
            candidates.append(x)
            if len(candidates) >= n_candidates:
                break
        
        if len(candidates) == 0:
            return self._generate_valid_sample()
        
        X_candidates = np.array(candidates)
        
        # 计算 EI
        ei_values = self._acquisition_ei(X_candidates)
        
        # 选择 EI 最大的点
        best_idx = np.argmax(ei_values)
        return candidates[best_idx]

    def optimize(self) -> EvalResult:
        """执行 SMAC 优化（使用 Random Forest surrogate）"""
        if not self._resume_loaded:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        if SMAC_AVAILABLE:
            return self._optimize_smac()
        elif SKLEARN_AVAILABLE:
            return self._optimize_rf_fallback()
        else:
            logger.error("[SMAC] 无可用的 surrogate 实现，请安装 smac 或 scikit-learn")
            return self._optimize_random_fallback()

    def _optimize_smac(self) -> EvalResult:
        """使用 SMAC3 库进行优化"""
        logger.info(f"[SMAC] 使用 SMAC3 进行优化")
        
        # 定义配置空间
        cs = ConfigurationSpace(seed=self.seed)
        for name in PARAM_NAMES:
            low, high = PARAM_BOUNDS[name]
            cs.add_hyperparameter(Float(name, (low, high)))
        
        # 定义场景
        scenario = Scenario(
            configspace=cs,
            deterministic=True,
            n_trials=self.max_evals,
            seed=self.seed,
        )
        
        # 目标函数
        def target_function(config, seed: int = 0):
            params = {name: config[name] for name in PARAM_NAMES}
            
            # 约束检查
            errors = _quick_validate_params(params)
            if errors:
                return PENALTY_FITNESS
            
            result = self.evaluate(params)
            self.results.append(result)
            self._write_csv_row(result, iteration=self.eval_count)
            
            x = self.params_to_vector(params)
            self._X_observed.append(x)
            self._y_observed.append(result.fitness)
            self._save_checkpoint()
            
            return result.fitness
        
        # 运行 SMAC
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=target_function,
        )
        
        incumbent = smac.optimize()
        logger.info(f"[SMAC] 优化完成 | 最优配置: {incumbent}")
        
        return self.best_result

    def _optimize_rf_fallback(self) -> EvalResult:
        """使用 sklearn Random Forest 的 fallback 实现"""
        logger.info(f"[SMAC] 使用 sklearn Random Forest surrogate")
        
        # 阶段1: 初始随机采样
        logger.info(f"[SMAC] 阶段1: 生成 {self.n_initial} 个初始点...")
        
        while len(self._X_observed) < self.n_initial and self.eval_count < self.max_evals:
            if self.converged:
                logger.info(f"[SMAC] 已收敛，停止迭代")
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
                logger.error(f"[SMAC] 初始采样异常: {e}")
                self._X_observed.append(x)
                self._y_observed.append(PENALTY_FITNESS)
            
            self._save_checkpoint()
            
            if len(self._X_observed) % 5 == 0:
                best_so_far = min(self._y_observed) if self._y_observed else float('inf')
                logger.info(f"[SMAC] 初始采样: {len(self._X_observed)}/{self.n_initial} | 最优={best_so_far:.6f}")
        
        logger.info(f"[SMAC] 阶段1完成: {len(self._X_observed)} 个初始点")
        
        # 阶段2: RF-based 序贯优化
        logger.info(f"[SMAC] 阶段2: Random Forest 序贯优化...")
        
        iteration = 0
        while self.eval_count < self.max_evals:
            if self.converged:
                logger.info(f"[SMAC] 已收敛，停止迭代")
                break
            
            # 每 5 轮更新一次 RF 模型
            if iteration % 5 == 0:
                self._fit_rf_model()
            
            # 使用 RF 选择下一个点
            if self._rf_model is not None and random.random() > 0.1:
                x_new = self._select_next_point_rf()
            else:
                # 10% 概率随机探索
                x_new = self._generate_valid_sample()
            
            params = self.vector_to_params(x_new)
            
            try:
                result = self.evaluate(params)
                self.results.append(result)
                self._write_csv_row(result, iteration=self.eval_count)
                
                self._X_observed.append(x_new)
                self._y_observed.append(result.fitness)
            except Exception as e:
                logger.error(f"[SMAC] 评估异常: {e}")
                self._X_observed.append(x_new)
                self._y_observed.append(PENALTY_FITNESS)
            
            self._save_checkpoint()
            
            iteration += 1
            if iteration % 20 == 0:
                logger.info(f"[SMAC] 进度: {self.eval_count}/{self.max_evals} | 最优={min(self._y_observed):.6f}")
        
        logger.info(f"[SMAC] 优化结束 | 总评估={self.eval_count} | 最优={self.best_fitness:.6f}")
        return self.best_result

    def _optimize_random_fallback(self) -> EvalResult:
        """纯随机搜索 fallback"""
        logger.warning(f"[SMAC] 使用纯随机搜索（无 surrogate）")
        
        while self.eval_count < self.max_evals:
            if self.converged:
                break
            
            x = self._generate_valid_sample()
            params = self.vector_to_params(x)
            
            try:
                result = self.evaluate(params)
                self.results.append(result)
                self._write_csv_row(result, iteration=self.eval_count)
                
                self._X_observed.append(x)
                self._y_observed.append(result.fitness)
            except Exception as e:
                logger.error(f"[SMAC] 评估异常: {e}")
            
            self._save_checkpoint()
            
            if self.eval_count % 20 == 0:
                logger.info(f"[SMAC] 进度: {self.eval_count}/{self.max_evals} | 最优={self.best_fitness:.6f}")
        
        return self.best_result
