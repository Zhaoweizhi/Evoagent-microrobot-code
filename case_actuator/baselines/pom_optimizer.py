# -*- coding: utf-8 -*-
"""
POM (Pretrained Optimization Model) 优化器
基于预训练的元黑箱优化器

论文: Pretrained Optimization Model for Zero-Shot Black Box Optimization (NeurIPS 2024)
代码: https://github.com/ninja-wm/POM
"""

# 解决 Windows 上 PyTorch 与 NumPy 的 OpenMP 冲突
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 预加载 PyTorch
try:
    import torch
    _TORCH_PRELOADED = True
except Exception:
    _TORCH_PRELOADED = False

import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from loguru import logger

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 添加 POM 路径
POM_DIR = Path(__file__).resolve().parent / "POM"
if str(POM_DIR) not in sys.path:
    sys.path.insert(0, str(POM_DIR))

# 添加 GLHF 包路径
GLHF_DIR = POM_DIR / "GLHF_pkg"
if str(GLHF_DIR) not in sys.path:
    sys.path.insert(0, str(GLHF_DIR))

# 添加 BBOB 包路径 (POM 依赖)
BBOB_DIR = POM_DIR / "BBOB_pkg"
if str(BBOB_DIR) not in sys.path:
    sys.path.insert(0, str(BBOB_DIR))

from .base_optimizer import (
    BaseOptimizer, EvalResult, PARAM_BOUNDS, PARAM_NAMES,
    PENALTY_FITNESS, generate_valid_params, _quick_validate_params
)


# POM 使用的设备
DEVICE = "cuda" if _TORCH_PRELOADED and torch.cuda.is_available() else "cpu"


class POMOptimizer(BaseOptimizer):
    """
    POM (Pretrained Optimization Model) 优化器
    
    使用预训练的神经网络模型来学习优化策略，实现零样本黑箱优化。
    POM 是一个基于种群的元优化器，通过在多种优化任务上预训练来学习通用的优化策略。
    """
    
    def __init__(
        self,
        max_evals: int = 200,
        pop_size: int = 50,
        device: str = "cpu",
        model_path: Optional[str] = None,
        output_dir: str = ".",
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            max_evals: 最大评估次数
            pop_size: 种群大小
            device: 计算设备 ("cpu" 或 "cuda")
            model_path: 预训练模型路径
            output_dir: 输出目录
            seed: 随机种子
        """
        super().__init__(
            max_evals=max_evals,
            output_dir=output_dir,
            algorithm_name="POM",
            seed=seed,
            **kwargs
        )
        
        self.pop_size = pop_size
        self.device = device
        
        # 模型路径
        if model_path is None:
            model_path = str(POM_DIR / "ckpt" / "pom_m_release.pth")
        self.model_path = model_path
        
        # POM 模型
        self.model = None
        
        # 种群数据
        self.population: List[Dict[str, float]] = []
        self.fitness_values: List[float] = []
        
        # 检查点
        self.checkpoint_path: Optional[Path] = None
        
        # 参数维度
        self.dim = len(PARAM_NAMES)
        
    def _load_model(self):
        """加载预训练的 POM 模型"""
        if self.model is not None:
            return
            
        logger.info(f"[{self.algorithm_name}] 加载 POM 模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"POM 模型文件不存在: {self.model_path}")
        
        try:
            # 设置 GLHF 的设备
            import GLHF.imports as glhf_imports
            glhf_imports.DEVICE = self.device
            
            from GLHF.GLHFMODEL import GB_GLHF as GLHF
            
            # 创建模型 (m 版本: muthdim=1000, crhdim=4)
            self.model = GLHF(
                popsize=self.pop_size,
                selmod='1-to-1',
                cr_policy='learned',
                muthdim=1000,
                crhdim=4
            ).to(self.device)
            
            # 加载权重
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            logger.info(f"[{self.algorithm_name}] POM 模型加载成功")
            
        except Exception as e:
            logger.error(f"[{self.algorithm_name}] 加载 POM 模型失败: {e}")
            raise
    
    def _params_to_normalized(self, params: Dict[str, float]) -> np.ndarray:
        """将参数字典转换为归一化向量 [-10, 10] (POM 的默认范围)"""
        x = []
        for name in PARAM_NAMES:
            low, high = PARAM_BOUNDS[name]
            # 先归一化到 [0, 1]，再映射到 [-10, 10]
            val = (params[name] - low) / (high - low)
            val = val * 20 - 10  # 映射到 [-10, 10]
            x.append(val)
        return np.array(x)
    
    def _normalized_to_params(self, x: np.ndarray) -> Dict[str, float]:
        """将归一化向量转换为参数字典"""
        params = {}
        for i, name in enumerate(PARAM_NAMES):
            low, high = PARAM_BOUNDS[name]
            # 从 [-10, 10] 映射回 [0, 1]，再映射到实际范围
            val = (x[i] + 10) / 20  # 映射到 [0, 1]
            val = np.clip(val, 0, 1)
            params[name] = low + val * (high - low)
            # 转换为 Python 原生 float，避免 JSON 序列化问题
            params[name] = round(float(params[name]), 3)
        return params
    
    def _init_population(self) -> List[Dict[str, float]]:
        """初始化种群"""
        population = []
        for i in range(self.pop_size):
            try:
                params, _ = generate_valid_params(seed=self.seed + i)
                population.append(params)
            except RuntimeError:
                # 如果生成失败，使用随机参数
                params = {}
                for name in PARAM_NAMES:
                    low, high = PARAM_BOUNDS[name]
                    params[name] = round(np.random.uniform(low, high), 3)
                population.append(params)
        return population
    
    def _evaluate_population(self, population: List[Dict[str, float]]) -> List[float]:
        """评估种群中所有个体"""
        fitness_values = []
        for params in population:
            result = self.evaluate(params)
            self.results.append(result)
            self._write_csv_row(result, self.eval_count)
            
            if result.fitness < PENALTY_FITNESS * 0.1:
                fitness_values.append(result.fitness)
            else:
                fitness_values.append(1e6)  # 惩罚值
                
        return fitness_values
    
    def _save_checkpoint(self):
        """保存检查点"""
        if self.checkpoint_path is None:
            self.checkpoint_path = self.output_dir / f"{self.algorithm_name}_checkpoint.pkl"
        
        checkpoint = {
            "eval_count": self.eval_count,
            "population": self.population,
            "fitness_values": self.fitness_values,
            "best_fitness": self.best_fitness,
            "best_result": self.best_result,
            "results": self.results,
            "seed": self.seed,
        }
        
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        logger.debug(f"[{self.algorithm_name}] 检查点已保存: {self.checkpoint_path}")
    
    def _load_checkpoint(self) -> bool:
        """加载检查点"""
        if self.checkpoint_path is None:
            self.checkpoint_path = self.output_dir / f"{self.algorithm_name}_checkpoint.pkl"
        
        if not self.checkpoint_path.exists():
            return False
        
        try:
            with open(self.checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            
            self.eval_count = checkpoint["eval_count"]
            self.population = checkpoint["population"]
            self.fitness_values = checkpoint["fitness_values"]
            self.best_fitness = checkpoint["best_fitness"]
            self.best_result = checkpoint["best_result"]
            self.results = checkpoint["results"]
            
            logger.info(f"[{self.algorithm_name}] 从检查点恢复: eval_count={self.eval_count}, "
                       f"best_fitness={self.best_fitness:.6f}")
            return True
            
        except Exception as e:
            logger.warning(f"[{self.algorithm_name}] 加载检查点失败: {e}")
            return False
    
    def optimize(self) -> EvalResult:
        """执行 POM 优化"""
        # 加载模型
        self._load_model()
        
        # 尝试从检查点恢复
        resumed = self._load_checkpoint()
        
        if not resumed:
            # 初始化种群
            logger.info(f"[{self.algorithm_name}] 初始化种群 (size={self.pop_size})")
            self.population = self._init_population()
            
            # 评估初始种群
            logger.info(f"[{self.algorithm_name}] 评估初始种群...")
            self.fitness_values = self._evaluate_population(self.population)
        
        # 主优化循环
        generation = 0
        while self.eval_count < self.max_evals and not self.converged:
            generation += 1
            
            # 构建 POM 输入张量
            # POM 期望输入: (batch, pop_size, dim+1)，其中第一列是 fitness
            pop_tensor = []
            for i, params in enumerate(self.population):
                x = self._params_to_normalized(params)
                fitness = self.fitness_values[i]
                # POM 格式: [fitness, x1, x2, ..., xd]
                individual = np.concatenate([[fitness], x])
                pop_tensor.append(individual)
            
            pop_tensor = torch.tensor(
                np.array([pop_tensor]),  # batch=1
                dtype=torch.float32,
                device=self.device
            )
            
            # 使用 POM 生成新种群
            with torch.no_grad():
                # POM 需要一个 problem 对象来计算 fitness
                # 我们创建一个简单的包装器
                class ProblemWrapper:
                    def __init__(self, optimizer):
                        self.optimizer = optimizer
                        self.pending_fitness = []
                    
                    def calfitness(self, x):
                        """
                        x: (batch, pop_size, dim)
                        返回: (batch, pop_size, dim+1), fitness
                        """
                        b, n, d = x.shape
                        x_np = x.cpu().numpy()
                        
                        fitness_list = []
                        for i in range(n):
                            params = self.optimizer._normalized_to_params(x_np[0, i])
                            
                            # 快速约束检查
                            errors = _quick_validate_params(params)
                            if errors:
                                fitness_list.append(1e6)
                            else:
                                # 实际评估
                                result = self.optimizer.evaluate(params)
                                self.optimizer.results.append(result)
                                self.optimizer._write_csv_row(result, self.optimizer.eval_count)
                                
                                if result.fitness < PENALTY_FITNESS * 0.1:
                                    fitness_list.append(result.fitness)
                                else:
                                    fitness_list.append(1e6)
                            
                            # 检查是否超过最大评估次数
                            if self.optimizer.eval_count >= self.optimizer.max_evals:
                                # 填充剩余的 fitness
                                while len(fitness_list) < n:
                                    fitness_list.append(1e6)
                                break
                        
                        fitness_tensor = torch.tensor(
                            [[f] for f in fitness_list],
                            dtype=torch.float32,
                            device=x.device
                        ).unsqueeze(0)  # (1, n, 1)
                        
                        # 构建返回值: (batch, pop_size, dim+1)
                        pop_with_fitness = torch.cat([fitness_tensor, x], dim=-1)
                        
                        return pop_with_fitness, fitness_tensor
                
                problem = ProblemWrapper(self)
                
                # 调用 POM 模型
                try:
                    new_pop, _, _ = self.model(pop_tensor, problem)
                    
                    # 更新种群
                    new_pop_np = new_pop.cpu().numpy()[0]  # (pop_size, dim+1)
                    
                    self.population = []
                    self.fitness_values = []
                    for i in range(len(new_pop_np)):
                        fitness = new_pop_np[i, 0]
                        x = new_pop_np[i, 1:]
                        params = self._normalized_to_params(x)
                        self.population.append(params)
                        self.fitness_values.append(fitness)
                        
                except Exception as e:
                    logger.error(f"[{self.algorithm_name}] POM 模型调用失败: {e}")
                    break
            
            # 保存检查点
            if generation % 5 == 0:
                self._save_checkpoint()
            
            logger.info(f"[{self.algorithm_name}] Generation {generation}: "
                       f"best_fitness={self.best_fitness:.6f}, eval_count={self.eval_count}")
        
        # 最终保存
        self._save_checkpoint()
        
        return self.best_result if self.best_result else EvalResult(
            params={},
            status="no_valid_result",
            fitness=PENALTY_FITNESS,
            eval_count=self.eval_count,
        )


def run_pom(
    max_evals: int = 200,
    pop_size: int = 50,
    device: str = "cpu",
    output_dir: str = ".",
    seed: int = 42,
) -> EvalResult:
    """
    运行 POM 优化
    """
    optimizer = POMOptimizer(
        max_evals=max_evals,
        pop_size=pop_size,
        device=device,
        output_dir=output_dir,
        seed=seed,
    )
    
    return optimizer.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="POM 优化器")
    parser.add_argument("--max-evals", type=int, default=200, help="最大评估次数")
    parser.add_argument("--pop-size", type=int, default=50, help="种群大小")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备")
    parser.add_argument("--output-dir", type=str, default=".", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    result = run_pom(
        max_evals=args.max_evals,
        pop_size=args.pop_size,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    print(f"\n最优结果:")
    print(f"  Fitness: {result.fitness:.6f}")
    print(f"  参数: {result.params}")
