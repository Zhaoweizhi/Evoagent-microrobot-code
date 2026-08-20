# -*- coding: utf-8 -*-
"""
贝叶斯优化 (Bayesian Optimization) 优化器
使用 scikit-optimize 的高斯过程代理模型 + EI 采集函数
支持断点续跑（通过 x0/y0 传入已有观测点）
"""
import pickle
import random
import numpy as np
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

from .base_optimizer import (
    BaseOptimizer, EvalResult, PARAM_BOUNDS, PARAM_NAMES, PENALTY_FITNESS,
    generate_valid_params, _quick_validate_params
)

# 尝试导入 scikit-optimize
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.callbacks import CheckpointSaver
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize 未安装，将使用简化版贝叶斯优化")


class BOOptimizer(BaseOptimizer):
    """贝叶斯优化器（使用 scikit-optimize，支持断点续跑）"""
    
    def __init__(
        self,
        max_evals: int = 200,
        min_iterations: int = 200,
        convergence_window: int = 40,
        avg_window: int = 10,
        convergence_threshold: float = 0.01,
        n_initial: int = 20,
        acq_func: str = "EI",
        xi: float = 0.01,
        kappa: float = 1.96,
        output_dir: str = ".",
        seed: int = 42,
        resume: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Args:
            max_evals: 最大评估次数
            min_iterations: 最小迭代次数（用于收敛判断）
            convergence_window: 收敛窗口
            avg_window: 早停窗口
            convergence_threshold: 收敛阈值
            n_initial: 初始随机采样数量
            acq_func: 采集函数 ("EI", "PI", "LCB")
            xi: EI/PI 的探索参数
            kappa: LCB 的探索参数
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
            algorithm_name="BO",
            seed=seed,
        )
        self.n_initial = min(n_initial, max_evals)
        self.acq_func = acq_func
        self.xi = xi
        self.kappa = kappa
        
        # 断点续跑相关
        self.resume = resume
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (self.output_dir / "BO_checkpoint.pkl")
        self._resume_loaded = False
        self._resume_X_observed: List[List[float]] = []
        self._resume_y_observed: List[float] = []
        
        # 当前观测数据（用于 checkpoint 保存）
        self._X_observed: List[List[float]] = []
        self._y_observed: List[float] = []
        
        logger.info(f"[BO] 配置: n_initial={self.n_initial}, acq_func={acq_func}, max_evals={max_evals}")
        logger.info(f"[BO] 收敛条件: min_iter={min_iterations}, conv_window={convergence_window}, "
                   f"threshold={convergence_threshold}")
        logger.info(f"[BO] scikit-optimize 可用: {SKOPT_AVAILABLE}")
        if self.resume:
            logger.info(f"[BO] 断点续跑启用: checkpoint={self.checkpoint_path}")
    
    # ========== 断点续跑相关方法 ==========
    
    def _eval_result_to_dict(self, result: Optional[EvalResult]) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        return asdict(result)

    def _eval_result_from_dict(self, data: Optional[Dict[str, Any]]) -> Optional[EvalResult]:
        if not data:
            return None
        return EvalResult(**data)

    def _save_checkpoint(self):
        """保存断点（每次评估后调用）"""
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
        logger.debug(f"[BO] 已保存断点: eval={self.eval_count}, best={self.best_fitness:.6f}")

    def _load_checkpoint(self) -> bool:
        """加载断点"""
        if not self.checkpoint_path.exists():
            logger.warning(f"[BO] 未找到断点文件: {self.checkpoint_path}")
            return False
        try:
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            logger.error(f"[BO] 断点文件读取失败: {e}")
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
        logger.info(f"[BO] 已加载断点: eval_count={self.eval_count}, best={self.best_fitness:.6f}, "
                   f"已有观测点={len(self._resume_X_observed)}")
        return True

    def run(self) -> EvalResult:
        """运行优化流程"""
        logger.info(f"=" * 60)
        logger.info(f"[BO] 开始优化 | max_evals={self.max_evals} | seed={self.seed}")
        logger.info(f"=" * 60)

        try:
            # 断点续跑：先加载断点以恢复 CSV 路径与随机状态
            if self.resume:
                self._load_checkpoint()
            self._init_csv()
            result = self.optimize()
            return result
        finally:
            self._close_csv()
            logger.info(f"[BO] 优化完成 | 最优 fitness={self.best_fitness:.6f}")
            logger.info(f"[BO] 结果已保存: {self.csv_path}")
    
    def _objective_with_checkpoint(self, x: List[float]) -> float:
        """目标函数（带 checkpoint 保存）"""
        params = self.vector_to_params(x)
        result = self.evaluate(params)
        self.results.append(result)
        self._write_csv_row(result, iteration=self.eval_count)
        
        # 记录观测数据并保存断点
        self._X_observed.append(list(x))
        self._y_observed.append(result.fitness)
        self._save_checkpoint()
        
        return result.fitness
    
    def optimize(self) -> EvalResult:
        """执行贝叶斯优化"""
        if not self._resume_loaded:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        if SKOPT_AVAILABLE:
            return self._optimize_skopt()
        else:
            return self._optimize_simple()
    
    def _optimize_skopt(self) -> EvalResult:
        """使用 scikit-optimize 的贝叶斯优化（支持断点续跑）
        
        关键：先用约束感知采样生成初始点，再让 skopt 基于这些点建模优化
        """
        logger.info(f"[BO] 使用 scikit-optimize 进行贝叶斯优化")
        
        # 定义搜索空间
        space = [
            Real(PARAM_BOUNDS[name][0], PARAM_BOUNDS[name][1], name=name)
            for name in PARAM_NAMES
        ]
        
        # 从断点恢复或生成约束感知的初始点
        if self._resume_loaded and len(self._resume_X_observed) > 0:
            # 从断点恢复
            self._X_observed = list(self._resume_X_observed)
            self._y_observed = list(self._resume_y_observed)
            logger.info(f"[BO] 从断点恢复: 已有 {len(self._X_observed)} 个观测点")
        else:
            # 生成约束感知的初始点（关键修复！）
            logger.info(f"[BO] 阶段1: 生成 {self.n_initial} 个约束感知的初始点...")
            self._X_observed = []
            self._y_observed = []
            
            for i in range(self.n_initial):
                if self.eval_count >= self.max_evals:
                    logger.info(f"[BO] 达到最大评估次数 {self.max_evals}，停止初始采样")
                    break
                
                # 使用约束感知采样生成合法的初始点
                x = self._generate_valid_sample()
                params = self.vector_to_params(x)
                
                try:
                    result = self.evaluate(params)
                    self.results.append(result)
                    self._write_csv_row(result, iteration=self.eval_count)
                    
                    self._X_observed.append(x)
                    self._y_observed.append(result.fitness)
                except Exception as e:
                    logger.error(f"[BO] 初始采样评估异常: {e}，使用惩罚值")
                    self._X_observed.append(x)
                    self._y_observed.append(PENALTY_FITNESS)
                
                # 每次评估后保存断点
                self._save_checkpoint()
                
                if (i + 1) % 5 == 0:
                    best_so_far = min(self._y_observed) if self._y_observed else float('inf')
                    logger.info(f"[BO] 初始采样进度: {i+1}/{self.n_initial} | 当前最优={best_so_far:.6f}")
            
            logger.info(f"[BO] 阶段1完成: 生成了 {len(self._X_observed)} 个初始点")
        
        # 计算剩余评估次数
        remaining_evals = self.max_evals - self.eval_count
        if remaining_evals <= 0:
            logger.info(f"[BO] 已达到最大评估次数 {self.max_evals}，无需继续")
            return self.best_result
        
        # 阶段2: 使用 skopt 进行贝叶斯优化
        logger.info(f"[BO] 阶段2: skopt 贝叶斯优化，剩余 {remaining_evals} 次评估...")
        
        try:
            result = gp_minimize(
                func=self._objective_with_checkpoint,
                dimensions=space,
                n_calls=remaining_evals,
                n_initial_points=0,  # 关键：不让 skopt 再随机采样，直接用我们的初始点
                acq_func=self.acq_func,
                xi=self.xi,
                kappa=self.kappa,
                x0=self._X_observed,  # 传入约束感知的初始点
                y0=self._y_observed,
                random_state=self.seed,
                verbose=True,
            )
            
            logger.info(f"[BO] skopt 优化完成 | 最优 fitness={result.fun:.6f}")
            logger.info(f"[BO] 总评估次数: {self.eval_count}")
            
        except Exception as e:
            logger.error(f"[BO] skopt 优化过程中出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return self.best_result
    
    # ========== 简化版 BO（fallback）==========
    
    def _generate_valid_sample(self, base_params: Optional[Dict[str, float]] = None, scale: float = 0.2) -> List[float]:
        """
        生成满足约束的采样点
        如果提供 base_params，则在其附近进行约束感知的局部搜索
        """
        lower, upper = self.get_bounds()
        max_attempts = 1000
        
        if base_params is None:
            # 全局随机采样
            try:
                params, _ = generate_valid_params(seed=random.randint(0, 1000000))
                return self.params_to_vector(params)
            except RuntimeError:
                # 降级到简单随机采样
                return [random.uniform(lower[j], upper[j]) for j in range(len(PARAM_NAMES))]
        
        # 在 base_params 附近采样
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
        
        # 如果局部搜索失败，使用全局重采样
        try:
            params, _ = generate_valid_params(seed=random.randint(0, 1000000))
            return self.params_to_vector(params)
        except RuntimeError:
            return self.params_to_vector(base_params)
    
    def _optimize_simple(self) -> EvalResult:
        """简化版贝叶斯优化（约束感知的随机搜索 + 局部优化），作为 fallback"""
        logger.info(f"[BO] 使用简化版贝叶斯优化（约束感知的随机搜索 + 局部优化）")
        logger.warning(f"[BO] 请安装 scikit-optimize 以使用标准贝叶斯优化: pip install scikit-optimize")
        
        lower, upper = self.get_bounds()
        
        # 从断点恢复或初始化
        if self._resume_loaded:
            self._X_observed = list(self._resume_X_observed)
            self._y_observed = list(self._resume_y_observed)
            logger.info(f"[BO] 从断点恢复: eval={self.eval_count}, 已有观测点={len(self._X_observed)}")
        
        # 阶段1: 约束感知的随机采样（如果还需要）
        while len(self._X_observed) < self.n_initial and self.eval_count < self.max_evals:
            if self.converged:
                logger.info(f"[BO] 已收敛，停止迭代")
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
                logger.error(f"[BO] 阶段1评估异常: {e}，使用惩罚值继续")
                self._X_observed.append(x)
                self._y_observed.append(PENALTY_FITNESS)
            
            self._save_checkpoint()
        
        logger.info(f"[BO] 阶段1完成: {len(self._X_observed)} 个初始点")
        
        # 阶段2: 基于当前最优的约束感知局部搜索
        iteration = 0
        while self.eval_count < self.max_evals:
            if self.converged:
                logger.info(f"[BO] 已收敛，停止迭代")
                break
            
            # 找到当前最优点
            best_idx = np.argmin(self._y_observed)
            best_x = self._X_observed[best_idx]
            best_params = self.vector_to_params(best_x)
            
            # 在最优点附近采样（自适应探索）
            remaining = self.max_evals - self.eval_count
            progress = 1.0 - (remaining / self.max_evals)
            scale = 0.3 * (1 - 0.8 * progress)  # 从 0.3 衰减到 0.06
            
            x_new = self._generate_valid_sample(best_params, scale)
            params = self.vector_to_params(x_new)
            
            try:
                result = self.evaluate(params)
                self.results.append(result)
                self._write_csv_row(result, iteration=self.eval_count)
                
                self._X_observed.append(x_new)
                self._y_observed.append(result.fitness)
            except Exception as e:
                logger.error(f"[BO] 阶段2评估异常: {e}，使用惩罚值继续")
                self._X_observed.append(x_new)
                self._y_observed.append(PENALTY_FITNESS)
            
            self._save_checkpoint()
            
            iteration += 1
            if iteration % 20 == 0:
                logger.info(f"[BO] 进度: {self.eval_count}/{self.max_evals} | 当前最优={min(self._y_observed):.6f}")
        
        logger.info(f"[BO] 优化结束 | 总评估次数={self.eval_count} | 最优={self.best_fitness:.6f}")
        return self.best_result
