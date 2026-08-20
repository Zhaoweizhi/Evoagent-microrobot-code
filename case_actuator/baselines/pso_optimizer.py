# -*- coding: utf-8 -*-
"""
粒子群优化 (Particle Swarm Optimization) 优化器
带约束感知的重采样机制，支持断点续跑
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


class Particle:
    """粒子"""
    
    def __init__(self, position: List[float], velocity: List[float]):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_fitness = float("inf")
        self.fitness = float("inf")
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "position": self.position,
            "velocity": self.velocity,
            "best_position": self.best_position,
            "best_fitness": self.best_fitness,
            "fitness": self.fitness,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Particle":
        """从字典反序列化"""
        p = cls(data["position"], data["velocity"])
        p.best_position = data["best_position"]
        p.best_fitness = data["best_fitness"]
        p.fitness = data["fitness"]
        return p


class PSOOptimizer(BaseOptimizer):
    """粒子群优化器（支持断点续跑）"""
    
    def __init__(
        self,
        max_evals: int = 200,
        min_iterations: int = 200,
        convergence_window: int = 40,
        avg_window: int = 10,
        convergence_threshold: float = 0.01,
        swarm_size: int = 20,
        w: float = 0.7,  # 惯性权重
        c1: float = 1.5,  # 认知因子（个体学习）
        c2: float = 1.5,  # 社会因子（群体学习）
        w_decay: float = 0.99,  # 惯性权重衰减
        v_max_ratio: float = 0.2,  # 最大速度比例
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
            swarm_size: 粒子群大小
            w: 惯性权重
            c1: 认知因子
            c2: 社会因子
            w_decay: 惯性权重衰减系数
            v_max_ratio: 最大速度相对于参数范围的比例
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
            algorithm_name="PSO",
            seed=seed,
        )
        self.swarm_size = swarm_size
        self.w = w
        self.w_init = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        self.v_max_ratio = v_max_ratio
        
        # 计算迭代数
        self.n_iterations = max(1, max_evals // swarm_size)
        
        # 全局最优
        self.global_best_position: Optional[List[float]] = None
        self.global_best_fitness = float("inf")
        
        # 断点续跑相关
        self.resume = resume
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (self.output_dir / "PSO_checkpoint.pkl")
        self._resume_loaded = False
        self._resume_swarm: List[Dict[str, Any]] = []
        self._resume_iteration = 0
        
        logger.info(f"[PSO] 配置: swarm_size={swarm_size}, max_iter={self.n_iterations}, "
                   f"w={w}, c1={c1}, c2={c2}")
        logger.info(f"[PSO] 收敛条件: min_iter={min_iterations}, conv_window={convergence_window}, "
                   f"threshold={convergence_threshold}")
        if self.resume:
            logger.info(f"[PSO] 断点续跑启用: checkpoint={self.checkpoint_path}")
    
    # ========== 断点续跑相关方法 ==========
    
    def _eval_result_to_dict(self, result: Optional[EvalResult]) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        return asdict(result)

    def _eval_result_from_dict(self, data: Optional[Dict[str, Any]]) -> Optional[EvalResult]:
        if not data:
            return None
        return EvalResult(**data)

    def _save_checkpoint(self, iteration: int, swarm: List[Particle]):
        """保存断点（每次评估后调用）"""
        state = {
            "iteration": iteration,
            "swarm": [p.to_dict() for p in swarm],
            "global_best_position": self.global_best_position,
            "global_best_fitness": self.global_best_fitness,
            "w": self.w,
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
        logger.debug(f"[PSO] 已保存断点: iter={iteration}, eval={self.eval_count}")

    def _load_checkpoint(self) -> bool:
        """加载断点"""
        if not self.checkpoint_path.exists():
            logger.warning(f"[PSO] 未找到断点文件: {self.checkpoint_path}")
            return False
        try:
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            logger.error(f"[PSO] 断点文件读取失败: {e}")
            return False

        self.eval_count = state.get("eval_count", 0)
        self.best_fitness = state.get("best_fitness", float("inf"))
        self.best_result = self._eval_result_from_dict(state.get("best_result"))
        self.valid_fitness_history = state.get("valid_fitness_history", [])
        self.no_improvement_count = state.get("no_improvement_count", 0)
        self.converged = state.get("converged", False)
        self.convergence_reason = state.get("convergence_reason", "")
        self._resume_swarm = state.get("swarm", [])
        self._resume_iteration = state.get("iteration", 0)
        self.global_best_position = state.get("global_best_position")
        self.global_best_fitness = state.get("global_best_fitness", float("inf"))
        self.w = state.get("w", self.w_init)

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
        logger.info(f"[PSO] 已加载断点: {self.checkpoint_path}")
        return True

    def run(self) -> EvalResult:
        """运行优化流程"""
        logger.info(f"=" * 60)
        logger.info(f"[PSO] 开始优化 | max_evals={self.max_evals} | seed={self.seed}")
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
            logger.info(f"[PSO] 优化完成 | 最优 fitness={self.best_fitness:.6f}")
            logger.info(f"[PSO] 结果已保存: {self.csv_path}")
    
    def _init_swarm(self) -> List[Particle]:
        """初始化粒子群 - 使用约束感知的采样"""
        lower, upper = self.get_bounds()
        swarm = []
        
        logger.info(f"[PSO] 生成约束满足的初始粒子群 ({self.swarm_size} 粒子)...")
        
        for i in range(self.swarm_size):
            try:
                # 使用约束感知的采样生成初始位置
                params, attempts = generate_valid_params(seed=self.seed + i)
                position = self.params_to_vector(params)
                if attempts > 100:
                    logger.debug(f"[PSO] 粒子 {i+1} 生成，尝试 {attempts} 次")
            except RuntimeError:
                # 降级到随机采样
                position = [random.uniform(lower[j], upper[j]) for j in range(len(PARAM_NAMES))]
            
            # 随机速度（初始较小）
            velocity = [
                random.uniform(-1, 1) * (upper[j] - lower[j]) * 0.1
                for j in range(len(PARAM_NAMES))
            ]
            
            swarm.append(Particle(position, velocity))
        
        logger.info(f"[PSO] 初始粒子群生成完成")
        return swarm
    
    def _repair_position(self, position: List[float], base_position: Optional[List[float]] = None) -> List[float]:
        """
        修复不满足约束的位置
        如果简单修复失败，则在 base_position 附近重采样
        """
        lower, upper = self.get_bounds()
        
        # 先尝试边界约束
        repaired = [max(lower[i], min(upper[i], position[i])) for i in range(len(PARAM_NAMES))]
        repaired = [round(v, 2) for v in repaired]
        
        # 检查约束
        params = self.vector_to_params(repaired)
        errors = _quick_validate_params(params)
        
        if not errors:
            return repaired
        
        # 如果仍然违反约束，尝试局部修复
        if base_position is not None:
            base_params = self.vector_to_params(base_position)
            for _ in range(100):
                # 在原位置和新位置之间插值
                alpha = random.uniform(0.1, 0.9)
                mixed = [
                    round(alpha * base_position[i] + (1 - alpha) * repaired[i], 2)
                    for i in range(len(PARAM_NAMES))
                ]
                mixed = [max(lower[i], min(upper[i], mixed[i])) for i in range(len(PARAM_NAMES))]
                
                params = self.vector_to_params(mixed)
                errors = _quick_validate_params(params)
                if not errors:
                    return mixed
        
        # 如果仍然失败，重采样
        try:
            params, _ = generate_valid_params(seed=random.randint(0, 1000000))
            return self.params_to_vector(params)
        except RuntimeError:
            # 最后手段：返回原位置（如果有）
            if base_position is not None:
                return base_position.copy()
            return repaired
    
    def _update_velocity(self, particle: Particle) -> None:
        """更新粒子速度"""
        lower, upper = self.get_bounds()
        
        for i in range(len(PARAM_NAMES)):
            r1 = random.random()
            r2 = random.random()
            
            # 认知项（向个体最优学习）
            cognitive = self.c1 * r1 * (particle.best_position[i] - particle.position[i])
            
            # 社会项（向全局最优学习）
            social = self.c2 * r2 * (self.global_best_position[i] - particle.position[i])
            
            # 更新速度
            particle.velocity[i] = self.w * particle.velocity[i] + cognitive + social
            
            # 速度限制
            v_max = (upper[i] - lower[i]) * self.v_max_ratio
            particle.velocity[i] = max(-v_max, min(v_max, particle.velocity[i]))
    
    def _update_position(self, particle: Particle) -> None:
        """更新粒子位置 - 带约束修复"""
        lower, upper = self.get_bounds()
        
        # 保存原位置
        old_position = particle.position.copy()
        
        # 应用速度更新
        new_position = []
        for i in range(len(PARAM_NAMES)):
            new_val = particle.position[i] + particle.velocity[i]
            
            # 边界处理（反弹）
            if new_val < lower[i]:
                new_val = lower[i]
                particle.velocity[i] *= -0.5
            elif new_val > upper[i]:
                new_val = upper[i]
                particle.velocity[i] *= -0.5
            
            new_position.append(new_val)
        
        # 检查约束并修复
        params = self.vector_to_params(new_position)
        errors = _quick_validate_params(params)
        
        if errors:
            # 需要修复
            new_position = self._repair_position(new_position, old_position)
        
        particle.position = new_position
    
    def optimize(self) -> EvalResult:
        """执行粒子群优化"""
        if not self._resume_loaded:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # 初始化粒子群或从断点恢复
        swarm: List[Particle]
        start_iter = 1
        
        if self._resume_loaded and len(self._resume_swarm) > 0:
            # 从断点恢复粒子群
            swarm = [Particle.from_dict(p) for p in self._resume_swarm]
            start_iter = self._resume_iteration + 1
            logger.info(f"[PSO] 从断点恢复: iter={self._resume_iteration}, eval={self.eval_count}, "
                       f"global_best={self.global_best_fitness:.6f}")
        else:
            # 初始化新粒子群
            swarm = self._init_swarm()
            
            logger.info(f"[PSO] 评估初始粒子群 ({self.swarm_size} 粒子)...")
            
            # 评估初始粒子群
            for i, particle in enumerate(swarm):
                if self.eval_count >= self.max_evals:
                    break
                
                try:
                    params = self.vector_to_params(particle.position)
                    result = self.evaluate(params)
                    particle.fitness = result.fitness
                    
                    # 更新个体最优
                    if result.fitness < particle.best_fitness:
                        particle.best_fitness = result.fitness
                        particle.best_position = particle.position.copy()
                    
                    # 更新全局最优
                    if result.fitness < self.global_best_fitness:
                        self.global_best_fitness = result.fitness
                        self.global_best_position = particle.position.copy()
                    
                    self.results.append(result)
                    self._write_csv_row(result, iteration=0)
                except Exception as e:
                    logger.error(f"[PSO] 初始粒子群评估异常: {e}，使用惩罚值继续")
                    particle.fitness = PENALTY_FITNESS
                
                # 每次评估后保存断点
                self._save_checkpoint(0, swarm)
        
        # 迭代优化
        for iteration in range(start_iter, self.n_iterations + 1):
            if self.eval_count >= self.max_evals:
                logger.info(f"[PSO] 达到最大评估次数 {self.max_evals}，停止")
                break
            
            if self.converged:
                logger.info(f"[PSO] 已收敛，停止迭代")
                break
            
            logger.info(f"[PSO] === 第 {iteration}/{self.n_iterations} 轮 ===")
            
            # 惯性权重衰减
            self.w = self.w_init * (self.w_decay ** iteration)
            
            # 更新每个粒子
            for particle in swarm:
                if self.eval_count >= self.max_evals:
                    break
                
                # 更新速度和位置
                self._update_velocity(particle)
                self._update_position(particle)
                
                try:
                    # 评估新位置
                    params = self.vector_to_params(particle.position)
                    result = self.evaluate(params)
                    particle.fitness = result.fitness
                    
                    # 更新个体最优
                    if result.fitness < particle.best_fitness:
                        particle.best_fitness = result.fitness
                        particle.best_position = particle.position.copy()
                    
                    # 更新全局最优
                    if result.fitness < self.global_best_fitness:
                        self.global_best_fitness = result.fitness
                        self.global_best_position = particle.position.copy()
                    
                    self.results.append(result)
                    self._write_csv_row(result, iteration=iteration)
                except Exception as e:
                    logger.error(f"[PSO] 粒子评估异常: {e}，使用惩罚值继续")
                    particle.fitness = PENALTY_FITNESS
                
                # 每次评估后保存断点
                self._save_checkpoint(iteration, swarm)
            
            # 日志
            logger.info(f"[PSO] 第 {iteration} 轮完成 | w={self.w:.3f} | 全局最优={self.global_best_fitness:.6f}")
        
        logger.info(f"[PSO] 优化结束 | 总评估次数={self.eval_count} | 最优={self.best_fitness:.6f}")
        return self.best_result
