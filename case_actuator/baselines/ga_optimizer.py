# -*- coding: utf-8 -*-
"""
遗传算法 (Genetic Algorithm) 优化器
使用约束感知的重采样机制，确保生成满足约束的个体
"""
import pickle
import random
import numpy as np
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from loguru import logger

from .base_optimizer import (
    BaseOptimizer, EvalResult, PARAM_BOUNDS, PARAM_NAMES, PENALTY_FITNESS,
    generate_valid_params, _quick_validate_params, MAX_RESAMPLE_ATTEMPTS
)


class GAOptimizer(BaseOptimizer):
    """遗传算法优化器（约束感知版）"""
    
    def __init__(
        self,
        max_evals: int = 200,
        min_iterations: int = 200,
        convergence_window: int = 40,
        avg_window: int = 10,
        convergence_threshold: float = 0.01,
        pop_size: int = 20,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        tournament_size: int = 3,
        elite_size: int = 2,
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
            pop_size: 种群大小
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            tournament_size: 锦标赛选择大小
            elite_size: 精英保留数量
            output_dir: 输出目录
            seed: 随机种子
        """
        super().__init__(
            max_evals=max_evals,
            min_iterations=min_iterations,
            convergence_window=convergence_window,
            avg_window=avg_window,
            convergence_threshold=convergence_threshold,
            output_dir=output_dir,
            algorithm_name="GA",
            seed=seed,
        )
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.resume = resume
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (self.output_dir / "GA_checkpoint.pkl")
        self._resume_loaded = False
        self._resume_population: List[List[float]] = []
        self._resume_fitnesses: List[float] = []
        self._resume_generation = 0
        
        # 计算代数
        self.n_generations = max(1, max_evals // pop_size)
        
        logger.info(f"[GA] 配置: pop_size={pop_size}, max_gen={self.n_generations}, "
                   f"crossover={crossover_prob}, mutation={mutation_prob}")
        logger.info(f"[GA] 收敛条件: min_iter={min_iterations}, conv_window={convergence_window}, "
                   f"threshold={convergence_threshold}")
        if self.resume:
            logger.info(f"[GA] 断点续跑启用: checkpoint={self.checkpoint_path}")

    def _eval_result_to_dict(self, result: Optional[EvalResult]) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        return asdict(result)

    def _eval_result_from_dict(self, data: Optional[Dict[str, Any]]) -> Optional[EvalResult]:
        if not data:
            return None
        return EvalResult(**data)

    def _save_checkpoint(self, generation: int, population: List[List[float]], fitnesses: List[float]):
        """保存断点（每代一次）"""
        state = {
            "generation": generation,
            "population": population,
            "fitnesses": fitnesses,
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
        logger.info(f"[GA] 已保存断点: {self.checkpoint_path} (gen={generation}, eval={self.eval_count})")

    def _load_checkpoint(self) -> bool:
        """加载断点"""
        if not self.checkpoint_path.exists():
            logger.warning(f"[GA] 未找到断点文件: {self.checkpoint_path}")
            return False
        try:
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            logger.error(f"[GA] 断点文件读取失败: {e}")
            return False

        self.eval_count = state.get("eval_count", 0)
        self.best_fitness = state.get("best_fitness", float("inf"))
        self.best_result = self._eval_result_from_dict(state.get("best_result"))
        self.valid_fitness_history = state.get("valid_fitness_history", [])
        self.no_improvement_count = state.get("no_improvement_count", 0)
        self.converged = state.get("converged", False)
        self.convergence_reason = state.get("convergence_reason", "")
        self._resume_population = state.get("population", [])
        self._resume_fitnesses = state.get("fitnesses", [])
        self._resume_generation = state.get("generation", 0)

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
        logger.info(f"[GA] 已加载断点: {self.checkpoint_path}")
        return True

    def run(self) -> EvalResult:
        logger.info(f"=" * 60)
        logger.info(f"[GA] 开始优化 | max_evals={self.max_evals} | seed={self.seed}")
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
            logger.info(f"[GA] 优化完成 | 最优 fitness={self.best_fitness:.6f}")
            logger.info(f"[GA] 结果已保存: {self.csv_path}")
    
    def _init_population(self) -> List[List[float]]:
        """初始化种群 - 使用约束感知的重采样"""
        population = []
        logger.info(f"[GA] 生成约束满足的初始种群 ({self.pop_size} 个体)...")
        
        for i in range(self.pop_size):
            try:
                # 使用不同的种子生成每个个体
                params, attempts = generate_valid_params(seed=self.seed + i)
                individual = self.params_to_vector(params)
                population.append(individual)
                if attempts > 100:
                    logger.debug(f"[GA] 个体 {i+1} 生成，尝试 {attempts} 次")
            except RuntimeError as e:
                logger.error(f"[GA] 无法生成有效个体 {i+1}: {e}")
                # 降级到随机采样
                lower, upper = self.get_bounds()
                individual = [random.uniform(lower[j], upper[j]) for j in range(len(PARAM_NAMES))]
                population.append(individual)
        
        logger.info(f"[GA] 初始种群生成完成")
        return population
    
    def _resample_individual(self, base_params: Optional[Dict[str, float]] = None) -> List[float]:
        """
        重采样一个满足约束的个体
        如果提供了 base_params，则在其基础上小幅变异
        """
        max_local_attempts = 1000
        
        if base_params is None:
            # 完全随机生成
            try:
                params, _ = generate_valid_params(seed=random.randint(0, 1000000))
                return self.params_to_vector(params)
            except RuntimeError:
                lower, upper = self.get_bounds()
                return [random.uniform(lower[i], upper[i]) for i in range(len(PARAM_NAMES))]
        
        # 在 base_params 基础上小幅变异并检查约束
        lower, upper = self.get_bounds()
        for _ in range(max_local_attempts):
            new_params = {}
            for i, name in enumerate(PARAM_NAMES):
                base_val = base_params.get(name, (lower[i] + upper[i]) / 2)
                # 小幅高斯扰动
                sigma = (upper[i] - lower[i]) * 0.1
                new_val = base_val + random.gauss(0, sigma)
                new_val = max(lower[i], min(upper[i], new_val))
                new_params[name] = round(new_val, 2)
            
            errors = _quick_validate_params(new_params)
            if not errors:
                return self.params_to_vector(new_params)
        
        # 如果局部变异失败，尝试全局重采样
        try:
            params, _ = generate_valid_params(seed=random.randint(0, 1000000))
            return self.params_to_vector(params)
        except RuntimeError:
            return self.params_to_vector(base_params)
    
    def _tournament_select(
        self, 
        population: List[List[float]], 
        fitnesses: List[float]
    ) -> List[float]:
        """锦标赛选择"""
        indices = random.sample(range(len(population)), self.tournament_size)
        best_idx = min(indices, key=lambda i: fitnesses[i])
        return population[best_idx].copy()
    
    def _crossover(
        self, 
        parent1: List[float], 
        parent2: List[float]
    ) -> Tuple[List[float], List[float]]:
        """SBX 交叉 - 带约束验证和重采样"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        lower, upper = self.get_bounds()
        eta = 20  # 分布指数
        
        # 尝试多次生成满足约束的子代
        for attempt in range(100):
            child1, child2 = [], []
            
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    # 执行 SBX
                    if abs(parent1[i] - parent2[i]) > 1e-10:
                        if parent1[i] < parent2[i]:
                            y1, y2 = parent1[i], parent2[i]
                        else:
                            y1, y2 = parent2[i], parent1[i]
                        
                        yl, yu = lower[i], upper[i]
                        
                        rand = random.random()
                        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1 + 1e-10))
                        alpha = 2.0 - beta ** (-(eta + 1))
                        if rand <= 1.0 / alpha:
                            betaq = (rand * alpha) ** (1.0 / (eta + 1))
                        else:
                            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                        
                        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                        
                        c1 = max(yl, min(yu, c1))
                        c2 = max(yl, min(yu, c2))
                        
                        if random.random() < 0.5:
                            child1.append(c1)
                            child2.append(c2)
                        else:
                            child1.append(c2)
                            child2.append(c1)
                    else:
                        child1.append(parent1[i])
                        child2.append(parent2[i])
                else:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
            
            # 检查约束
            params1 = self.vector_to_params(child1)
            params2 = self.vector_to_params(child2)
            errors1 = _quick_validate_params(params1)
            errors2 = _quick_validate_params(params2)
            
            if not errors1 and not errors2:
                return child1, child2
        
        # 如果多次尝试后仍然失败，使用重采样生成新个体
        base_params1 = self.vector_to_params(parent1)
        base_params2 = self.vector_to_params(parent2)
        new_child1 = self._resample_individual(base_params1)
        new_child2 = self._resample_individual(base_params2)
        return new_child1, new_child2
    
    def _mutate(self, individual: List[float]) -> List[float]:
        """多项式变异 - 带约束验证和重采样"""
        lower, upper = self.get_bounds()
        eta = 20  # 变异分布指数
        
        # 先尝试标准变异
        for attempt in range(100):  # 最多尝试100次
            mutant = individual.copy()
            for i in range(len(mutant)):
                if random.random() < self.mutation_prob:
                    y = mutant[i]
                    yl, yu = lower[i], upper[i]
                    delta1 = (y - yl) / (yu - yl + 1e-10)
                    delta2 = (yu - y) / (yu - yl + 1e-10)
                    
                    rand = random.random()
                    mut_pow = 1.0 / (eta + 1)
                    
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                        deltaq = 1.0 - val ** mut_pow
                    
                    y = y + deltaq * (yu - yl)
                    y = max(yl, min(yu, y))
                    mutant[i] = y
            
            # 检查约束
            params = self.vector_to_params(mutant)
            errors = _quick_validate_params(params)
            if not errors:
                return mutant
        
        # 如果多次尝试后仍然失败，使用重采样
        base_params = self.vector_to_params(individual)
        return self._resample_individual(base_params)
    
    def optimize(self) -> EvalResult:
        """执行遗传算法优化"""
        if not self._resume_loaded:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # 初始化种群或从断点恢复
        population: List[List[float]]
        fitnesses: List[float]
        start_gen = 1
        if self._resume_loaded:
            population = self._resume_population
            fitnesses = self._resume_fitnesses
            last_gen = self._resume_generation
            start_gen = last_gen + 1
            if len(population) != self.pop_size:
                logger.warning(f"[GA] 断点种群大小={len(population)} 与当前 pop_size={self.pop_size} 不一致，可能影响结果")
            if len(fitnesses) != len(population):
                logger.warning(f"[GA] 断点 fitness 数量={len(fitnesses)} 与种群数量不一致，将继续运行")
            logger.info(f"[GA] 从断点恢复: gen={last_gen}, eval={self.eval_count}")
        else:
            population = self._init_population()
            fitnesses = []
        
        logger.info(f"[GA] 评估初始种群 ({self.pop_size} 个体)...")
        
        # 评估初始种群（如果不是断点恢复）
        if not self._resume_loaded:
            for i, individual in enumerate(population):
                if self.eval_count >= self.max_evals:
                    break
                try:
                    params = self.vector_to_params(individual)
                    result = self.evaluate(params)
                    fitnesses.append(result.fitness)
                    self.results.append(result)
                    self._write_csv_row(result, iteration=0)
                except Exception as e:
                    # 极端情况：evaluate 本身抛异常（正常不会发生，因为 evaluate 已经兜底）
                    logger.error(f"[GA] 初始种群评估异常: {e}，使用惩罚值继续")
                    fitnesses.append(PENALTY_FITNESS)
                
                # 每次评估后都保存断点（防止中途崩溃丢失进度）
                self._save_checkpoint(0, population[:len(fitnesses)], fitnesses)
            # 保存第 0 代断点（完整）
            self._save_checkpoint(0, population, fitnesses)
        
        # 迭代进化
        for gen in range(start_gen, self.n_generations + 1):
            if self.eval_count >= self.max_evals:
                logger.info(f"[GA] 达到最大评估次数 {self.max_evals}，停止")
                break
            
            if self.converged:
                logger.info(f"[GA] 已收敛，停止迭代")
                break
            
            logger.info(f"[GA] === 第 {gen}/{self.n_generations} 代 ===")
            
            # 精英保留
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:self.elite_size]
            new_population = [population[i].copy() for i in elite_indices]
            new_fitnesses = [fitnesses[i] for i in elite_indices]
            
            # 生成新个体
            while len(new_population) < self.pop_size:
                if self.eval_count >= self.max_evals:
                    break
                
                # 选择
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # 评估
                for child in [child1, child2]:
                    if len(new_population) >= self.pop_size or self.eval_count >= self.max_evals:
                        break
                    
                    try:
                        params = self.vector_to_params(child)
                        result = self.evaluate(params)
                        
                        new_population.append(child)
                        new_fitnesses.append(result.fitness)
                        self.results.append(result)
                        self._write_csv_row(result, iteration=gen)
                    except Exception as e:
                        # 极端情况：evaluate 本身抛异常（正常不会发生）
                        logger.error(f"[GA] 个体评估异常: {e}，使用惩罚值继续")
                        new_population.append(child)
                        new_fitnesses.append(PENALTY_FITNESS)
                    
                    # 每次评估后都保存断点（防止中途崩溃丢失进度）
                    self._save_checkpoint(gen, new_population, new_fitnesses)
            
            # 更新种群
            population = new_population
            fitnesses = new_fitnesses
            
            # 日志
            best_gen_fitness = min(fitnesses) if fitnesses else float("inf")
            logger.info(f"[GA] 第 {gen} 代完成 | 本代最优={best_gen_fitness:.6f} | 全局最优={self.best_fitness:.6f}")
            # 每代保存断点
            self._save_checkpoint(gen, population, fitnesses)
        
        return self.best_result
