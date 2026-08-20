# -*- coding: utf-8 -*-
"""
PFN-CEI 优化器
基于预训练 Transformer 的约束贝叶斯优化

论文: Fast and accurate Bayesian optimization with pre-trained transformers 
      for constrained engineering problems
DOI: 10.1007/s00158-025-03987-z
代码: https://github.com/rosenyu304/BOEngineeringBenchmark
"""

# 解决 Windows 上 PyTorch 与 NumPy 的 OpenMP 冲突
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 预加载 PyTorch（必须在 numpy 之前）
try:
    import torch
    _TORCH_PRELOADED = True
except Exception:
    _TORCH_PRELOADED = False

import sys
import json
import csv
import pickle
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from scipy.optimize import minimize
from loguru import logger

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 添加 BOEngineeringBenchmark 路径
BOENG_DIR = Path(__file__).resolve().parent / "BOEngineeringBenchmark"
if str(BOENG_DIR) not in sys.path:
    sys.path.insert(0, str(BOENG_DIR))

from base_optimizer import (
    BaseOptimizer, EvalResult, PARAM_BOUNDS, PARAM_NAMES, 
    PENALTY_FITNESS, generate_valid_params, _quick_validate_params
)


class PFNCEIOptimizer(BaseOptimizer):
    """
    PFN-CEI (Prior-data Fitted Network with Constrained Expected Improvement) 优化器
    
    使用预训练的 Transformer 模型作为代理模型，计算约束期望改进作为采集函数。
    """
    
    def __init__(
        self,
        max_evals: int = 200,
        initial_samples: int = 10,
        n_candidates: int = 5000,
        device: str = "cpu",
        model_path: Optional[str] = None,
        output_dir: str = ".",
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            max_evals: 最大评估次数
            initial_samples: 初始随机采样数
            n_candidates: 每轮候选点数量（用于优化采集函数）
            device: 计算设备 ("cpu" 或 "cuda:0")
            model_path: 预训练模型路径
            output_dir: 输出目录
            seed: 随机种子
        """
        super().__init__(
            max_evals=max_evals,
            output_dir=output_dir,
            algorithm_name="PFN_CEI",
            seed=seed,
            **kwargs
        )
        
        self.initial_samples = initial_samples
        self.n_candidates = n_candidates
        self.device = device
        
        # 模型路径
        if model_path is None:
            model_path = os.environ.get(
                "PFN_CEI_MODEL_PATH",
                str(BOENG_DIR / "pfns4bo" / "final_models" / "pfn_cei_model.pt"),
            )
        self.model_path = model_path
        
        # 历史数据
        self.X_obs: List[np.ndarray] = []  # 观测点 (归一化到 [0,1])
        self.y_obs: List[float] = []        # 目标值 (取负，因为 PFN 假设最大化)
        self.g_obs: List[np.ndarray] = []   # 约束值 (g(x) <= 0 表示可行)
        
        # PFN 模型
        self.model = None
        self.pfn_method = None
        
        # 检查点
        self.checkpoint_path: Optional[Path] = None
        
    def _load_model(self):
        """加载预训练的 PFN 模型"""
        if self.model is not None:
            return
            
        logger.info(f"[{self.algorithm_name}] 加载 PFN 模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"PFN 模型文件不存在: {self.model_path}")
        
        # 导入 PFN 模块
        try:
            from pfns4bo import transformer
            from PFN_CEI import PFNCEI_TransformerBOMethod
        except ImportError as e:
            logger.error(f"导入 PFN 模块失败: {e}")
            logger.error("请确保 BOEngineeringBenchmark 目录在 Python 路径中")
            raise
        
        # 加载模型 (weights_only=False 因为模型包含自定义类)
        self.model = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.eval()
        self.model.to(self.device)
        
        # 创建 PFN-CEI 方法
        self.pfn_method = PFNCEI_TransformerBOMethod(
            model=self.model,
            device=self.device,
            apply_power_transform=True,
        )
        
        logger.info(f"[{self.algorithm_name}] PFN 模型加载成功")
    
    def _params_to_normalized(self, params: Dict[str, float]) -> np.ndarray:
        """将参数字典转换为归一化向量 [0, 1]"""
        x = []
        for name in PARAM_NAMES:
            low, high = PARAM_BOUNDS[name]
            val = (params[name] - low) / (high - low)
            x.append(np.clip(val, 0.0, 1.0))
        return np.array(x)
    
    def _normalized_to_params(self, x: np.ndarray) -> Dict[str, float]:
        """将归一化向量转换为参数字典"""
        params = {}
        for i, name in enumerate(PARAM_NAMES):
            low, high = PARAM_BOUNDS[name]
            params[name] = low + x[i] * (high - low)
            params[name] = round(params[name], 3)
        return params
    
    def _compute_constraints(self, params: Dict[str, float]) -> np.ndarray:
        """
        计算约束值 (g(x) <= 0 表示可行)
        
        返回约束向量，每个元素 <= 0 表示该约束满足
        """
        errors = _quick_validate_params(params)
        
        # 将错误转换为约束值
        # 如果有错误，返回正值；否则返回负值
        if errors:
            # 简化：有任何约束违反就返回 [1.0]
            return np.array([1.0])
        else:
            return np.array([-1.0])
    
    def _suggest_next(self) -> Dict[str, float]:
        """使用 PFN-CEI 建议下一个采样点"""
        if len(self.X_obs) < self.initial_samples:
            # 初始阶段：随机采样
            params, _ = generate_valid_params(seed=self.seed + len(self.X_obs))
            return params
        
        # 准备数据
        X_obs = torch.tensor(np.array(self.X_obs), dtype=torch.float32).to(self.device)
        y_obs = torch.tensor(np.array(self.y_obs), dtype=torch.float32).to(self.device)
        
        # 约束数据 (转换为 PFN 期望的格式)
        # PFN-CEI 期望 g(x) 形式，其中 g(x) <= 0 表示可行
        g_obs = torch.tensor(np.array(self.g_obs), dtype=torch.float32).to(self.device)
        
        # 生成候选点
        best_acq = -float('inf')
        best_params = None
        
        # 随机生成候选点
        candidates = []
        for _ in range(self.n_candidates):
            try:
                params, _ = generate_valid_params(
                    seed=self.seed + self.eval_count + random.randint(0, 100000)
                )
                candidates.append(params)
            except RuntimeError:
                continue
        
        if not candidates:
            # 如果生成失败，回退到随机采样
            params, _ = generate_valid_params(seed=self.seed + self.eval_count)
            return params
        
        # 转换为张量
        X_candidates = []
        for params in candidates:
            X_candidates.append(self._params_to_normalized(params))
        X_pen = torch.tensor(np.array(X_candidates), dtype=torch.float32).to(self.device)
        
        # 计算采集函数值
        try:
            obj_acq, constraint_acq = self.pfn_method.observe_and_suggest(
                X_obs=X_obs,
                y_obs=y_obs,
                X_pen=X_pen,
                GX=g_obs,
            )
            
            # CEI = EI * P(feasible)
            # obj_acq: (n_candidates,) - 期望改进
            # constraint_acq: (n_candidates, n_constraints) - 可行概率
            
            # 计算约束期望改进
            if constraint_acq.dim() > 1:
                pof = constraint_acq.prod(dim=1)  # 所有约束同时满足的概率
            else:
                pof = constraint_acq
            
            cei = obj_acq * pof
            
            # 选择最佳候选点
            best_idx = cei.argmax().item()
            best_params = candidates[best_idx]
            
        except Exception as e:
            logger.warning(f"[{self.algorithm_name}] PFN-CEI 计算失败: {e}，回退到随机采样")
            best_params = candidates[0]
        
        return best_params
    
    def _save_checkpoint(self):
        """保存检查点"""
        if self.checkpoint_path is None:
            self.checkpoint_path = self.output_dir / f"{self.algorithm_name}_checkpoint.pkl"
        
        checkpoint = {
            "eval_count": self.eval_count,
            "X_obs": self.X_obs,
            "y_obs": self.y_obs,
            "g_obs": self.g_obs,
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
            self.X_obs = checkpoint["X_obs"]
            self.y_obs = checkpoint["y_obs"]
            self.g_obs = checkpoint["g_obs"]
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
        """执行 PFN-CEI 优化"""
        # 加载模型
        self._load_model()
        
        # 尝试从检查点恢复
        resumed = self._load_checkpoint()
        
        while self.eval_count < self.max_evals and not self.converged:
            # 建议下一个采样点
            params = self._suggest_next()
            
            # 评估
            result = self.evaluate(params)
            self.results.append(result)
            
            # 记录到 CSV
            self._write_csv_row(result, self.eval_count)
            
            # 更新历史数据
            x_normalized = self._params_to_normalized(params)
            self.X_obs.append(x_normalized)
            
            # 目标值 (取负，因为 PFN 假设最大化)
            # 对于惩罚值，使用一个较大的负值
            if result.fitness < PENALTY_FITNESS * 0.1:
                self.y_obs.append(-result.fitness)  # 取负
            else:
                self.y_obs.append(-1e6)  # 惩罚
            
            # 约束值
            g = self._compute_constraints(params)
            if result.status == "ok":
                g = np.array([-1.0])  # 可行
            else:
                g = np.array([1.0])   # 不可行
            self.g_obs.append(g)
            
            # 保存检查点
            if self.eval_count % 10 == 0:
                self._save_checkpoint()
        
        # 最终保存
        self._save_checkpoint()
        
        return self.best_result if self.best_result else EvalResult(
            params={},
            status="no_valid_result",
            fitness=PENALTY_FITNESS,
            eval_count=self.eval_count,
        )


def run_pfn_cei(
    max_evals: int = 200,
    initial_samples: int = 10,
    n_candidates: int = 5000,
    device: str = "cpu",
    output_dir: str = ".",
    seed: int = 42,
) -> EvalResult:
    """
    运行 PFN-CEI 优化
    
    Args:
        max_evals: 最大评估次数
        initial_samples: 初始随机采样数
        n_candidates: 候选点数量
        device: 计算设备
        output_dir: 输出目录
        seed: 随机种子
    
    Returns:
        最优结果
    """
    optimizer = PFNCEIOptimizer(
        max_evals=max_evals,
        initial_samples=initial_samples,
        n_candidates=n_candidates,
        device=device,
        output_dir=output_dir,
        seed=seed,
    )
    
    return optimizer.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PFN-CEI 优化器")
    parser.add_argument("--max-evals", type=int, default=200, help="最大评估次数")
    parser.add_argument("--initial-samples", type=int, default=10, help="初始采样数")
    parser.add_argument("--n-candidates", type=int, default=5000, help="候选点数量")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备")
    parser.add_argument("--output-dir", type=str, default=".", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    result = run_pfn_cei(
        max_evals=args.max_evals,
        initial_samples=args.initial_samples,
        n_candidates=args.n_candidates,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    print(f"\n最优结果:")
    print(f"  Fitness: {result.fitness:.6f}")
    print(f"  参数: {result.params}")
