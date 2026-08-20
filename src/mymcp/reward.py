"""
奖励计算模块

功能：
1. 多目标奖励计算
2. 约束惩罚
3. 探索奖励
4. 人类反馈奖励
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from loguru import logger


@dataclass
class RewardConfig:
    """奖励配置"""
    # 性能奖励权重
    fitness_weight: float = 1.0
    kb_weight: float = 0.5
    pb_weight: float = 0.3
    
    # 约束惩罚
    constraint_penalty: float = -10.0
    simulation_failure_penalty: float = -20.0
    
    # 改进奖励
    improvement_bonus: float = 5.0  # 比历史最佳更好时的额外奖励
    
    # 探索奖励
    exploration_bonus: float = 1.0  # 尝试新区域的奖励
    novelty_threshold: float = 0.2  # 判定为新颖的距离阈值
    
    # 人类反馈奖励
    feedback_compliance_bonus: float = 2.0  # 遵循反馈的奖励
    feedback_violation_penalty: float = -3.0  # 违反反馈的惩罚
    
    # 归一化参考值
    fitness_ref: float = -10.0  # 参考 fitness（用于归一化）
    kb_ref: float = 0.3
    pb_ref: float = 1.0


class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.best_fitness: Optional[float] = None
        self.visited_states: List[Dict[str, float]] = []
    
    def calculate(
        self,
        result: Dict[str, Any],
        old_state: Optional[Dict[str, float]] = None,
        new_state: Optional[Dict[str, float]] = None,
        is_exploration: bool = False,
        feedback_compliance: Optional[bool] = None
    ) -> float:
        """计算综合奖励"""
        reward = 0.0
        
        status = result.get("status", "unknown")
        
        # 1. 基础奖励/惩罚
        if status == "ok":
            reward += self._performance_reward(result)
            reward += self._improvement_reward(result)
        elif status == "constraint_violation":
            reward += self.config.constraint_penalty
            # 部分奖励：如果仿真成功但约束违规，给予小的负奖励而非最大惩罚
            if result.get("avg_B") is not None:
                reward += 2.0  # 至少仿真成功了
        elif status == "simulation_failed":
            reward += self.config.simulation_failure_penalty
        
        # 2. 探索奖励
        if is_exploration and new_state:
            reward += self._exploration_reward(new_state)
        
        # 3. 人类反馈奖励
        if feedback_compliance is not None:
            if feedback_compliance:
                reward += self.config.feedback_compliance_bonus
            else:
                reward += self.config.feedback_violation_penalty
        
        # 4. 更新历史最佳
        fitness = result.get("fitness")
        if status == "ok" and fitness is not None:
            if self.best_fitness is None or fitness < self.best_fitness:
                self.best_fitness = fitness
        
        # 5. 记录访问状态
        if new_state:
            self.visited_states.append(new_state.copy())
            if len(self.visited_states) > 100:
                self.visited_states.pop(0)
        
        return reward
    
    def _performance_reward(self, result: Dict[str, Any]) -> float:
        """计算性能奖励"""
        reward = 0.0
        
        # fitness 奖励（越小越好，转换为正奖励）
        fitness = result.get("fitness")
        if fitness is not None:
            # 归一化：fitness 接近或低于参考值时给正奖励
            normalized = (self.config.fitness_ref - fitness) / abs(self.config.fitness_ref)
            reward += self.config.fitness_weight * normalized * 5
        
        # kb 奖励（越大越好）
        kb = result.get("kb")
        if kb is not None:
            normalized = kb / self.config.kb_ref
            reward += self.config.kb_weight * min(normalized, 2.0)
        
        # pb 奖励（越大越好）
        pb = result.get("pb")
        if pb is not None:
            normalized = pb / self.config.pb_ref
            reward += self.config.pb_weight * min(normalized, 2.0)
        
        return reward
    
    def _improvement_reward(self, result: Dict[str, Any]) -> float:
        """计算改进奖励"""
        fitness = result.get("fitness")
        if fitness is None or self.best_fitness is None:
            return 0.0
        
        # 如果比历史最佳更好，给予额外奖励
        if fitness < self.best_fitness:
            improvement_ratio = (self.best_fitness - fitness) / abs(self.best_fitness)
            return self.config.improvement_bonus * min(improvement_ratio * 10, 2.0)
        
        return 0.0
    
    def _exploration_reward(self, new_state: Dict[str, float]) -> float:
        """计算探索奖励"""
        if not self.visited_states:
            return self.config.exploration_bonus
        
        # 计算到最近访问状态的距离
        min_distance = float('inf')
        for visited in self.visited_states:
            dist = self._state_distance(new_state, visited)
            min_distance = min(min_distance, dist)
        
        # 如果足够新颖，给予探索奖励
        if min_distance > self.config.novelty_threshold:
            return self.config.exploration_bonus * min(min_distance / self.config.novelty_threshold, 2.0)
        
        return 0.0
    
    def _state_distance(self, s1: Dict[str, float], s2: Dict[str, float]) -> float:
        """计算状态距离"""
        ranges = {
            "lm": 5.0, "tm": 0.1, "ta": 0.4, "dg": 0.35,
            "hs": 1.0, "wslot": 0.8, "hslot": 0.5, "s": 0.4, "wa": 0.6
        }
        
        dist_sq = 0.0
        for param, r in ranges.items():
            v1 = s1.get(param, 0)
            v2 = s2.get(param, 0)
            dist_sq += ((v1 - v2) / r) ** 2
        
        return math.sqrt(dist_sq)
    
    def get_reward_breakdown(
        self,
        result: Dict[str, Any],
        is_exploration: bool = False
    ) -> Dict[str, float]:
        """获取奖励分解（用于调试和可解释性）"""
        breakdown = {
            "status": result.get("status", "unknown"),
            "performance_reward": 0.0,
            "improvement_reward": 0.0,
            "constraint_penalty": 0.0,
            "exploration_bonus": 0.0,
            "total": 0.0
        }
        
        status = result.get("status", "unknown")
        
        if status == "ok":
            breakdown["performance_reward"] = self._performance_reward(result)
            breakdown["improvement_reward"] = self._improvement_reward(result)
        elif status == "constraint_violation":
            breakdown["constraint_penalty"] = self.config.constraint_penalty
        elif status == "simulation_failed":
            breakdown["constraint_penalty"] = self.config.simulation_failure_penalty
        
        if is_exploration:
            breakdown["exploration_bonus"] = self.config.exploration_bonus
        
        breakdown["total"] = sum([
            breakdown["performance_reward"],
            breakdown["improvement_reward"],
            breakdown["constraint_penalty"],
            breakdown["exploration_bonus"]
        ])
        
        return breakdown
    
    def update_best_fitness(self, fitness: float):
        """手动更新最佳 fitness"""
        if self.best_fitness is None or fitness < self.best_fitness:
            self.best_fitness = fitness
    
    def reset(self):
        """重置状态"""
        self.best_fitness = None
        self.visited_states = []

