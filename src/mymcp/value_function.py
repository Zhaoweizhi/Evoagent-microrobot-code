# -*- coding: utf-8 -*-
"""
状态价值函数 V(s) 模块

实现正统 Actor-Critic 架构中的价值函数部分。
V(s) 评估当前参数状态的"潜力"——从该状态出发能获得的期望累积奖励。

实现方式（不微调）：
1. 基于历史数据的 k-近邻估计
2. 可选：基于 LLM 的价值预测

作用：
- 评估当前状态的基线价值
- 计算 TD 误差: δ = r + γ·V(s') - V(s)
- 指导探索：优先探索 V(s) 低但潜力大的区域
"""

import json
import os
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import numpy as np


# ============== 配置常量 ==============

# 设计参数边界（用于归一化）
PARAM_BOUNDS = {
    "lm": (3.0, 6.5),
    "tm": (0.3, 0.8),
    "ta": (0.4, 1.2),
    "dg": (0.2, 1.0),
    "hs": (1.0, 4.0),
    "wslot": (1.5, 4.0),
    "hslot": (0.8, 2.5),
    "s": (0.5, 2.0),
    "tb_ratio": (1.6, 2.0),
    "wa": (1.5, 2.5),
}

# fitness 参考值（用于归一化）
FITNESS_REF_MIN = 3.5      # 历史最优 fitness 约在 3.5-4.0
FITNESS_REF_MAX = 100.0    # 较差的 fitness
SATURATION_PENALTY = 1e5   # 磁饱和惩罚值

# 折扣因子
GAMMA = 0.95


# ============== 数据结构 ==============

@dataclass
class StateRecord:
    """状态记录"""
    state: Dict[str, float]           # 参数状态
    fitness: float                    # 该状态的 fitness
    value: float                      # 估计的状态价值
    visit_count: int = 1              # 访问次数
    cumulative_reward: float = 0.0    # 累积奖励
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "StateRecord":
        return cls(**d)


@dataclass
class ValueEstimate:
    """价值估计结果"""
    value: float                      # 估计的状态价值 V(s)
    confidence: float                 # 置信度 (0-1)
    similar_states_count: int         # 使用的相似状态数量
    method: str                       # 估计方法 ("knn", "llm", "default")
    reasoning: str = ""               # 估计理由
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============== 状态价值函数 ==============

class StateValueFunction:
    """
    状态价值函数 V(s)
    
    使用 k-近邻方法基于历史数据估计状态价值。
    价值 = 负的归一化 fitness（fitness 越小，价值越高）
    """
    
    def __init__(
        self,
        storage_path: str = "state_value_history.json",
        k_neighbors: int = 5,
        distance_threshold: float = 0.3,
        gamma: float = GAMMA
    ):
        """
        Args:
            storage_path: 状态历史存储路径
            k_neighbors: k-近邻中的 k 值
            distance_threshold: 距离阈值，超过此值的邻居权重降低
            gamma: 折扣因子
        """
        self.storage_path = storage_path
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.gamma = gamma
        
        # 状态历史
        self.state_history: List[StateRecord] = []
        
        # 最佳状态缓存
        self.best_state: Optional[StateRecord] = None
        self.best_fitness: float = float('inf')
        
        # 加载历史
        self._load_history()
        
        logger.info(f"[V(s)] 状态价值函数初始化 | 历史记录: {len(self.state_history)} | k={k_neighbors}")
    
    def _load_history(self):
        """加载状态历史"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.state_history = [StateRecord.from_dict(d) for d in data.get("history", [])]
                    
                    # 恢复最佳状态
                    if self.state_history:
                        valid_records = [r for r in self.state_history if r.fitness < SATURATION_PENALTY]
                        if valid_records:
                            self.best_state = min(valid_records, key=lambda r: r.fitness)
                            self.best_fitness = self.best_state.fitness
                            
                logger.info(f"[V(s)] 加载 {len(self.state_history)} 条状态历史")
            except Exception as e:
                logger.warning(f"[V(s)] 加载状态历史失败: {e}")
    
    def _save_history(self):
        """保存状态历史"""
        try:
            os.makedirs(os.path.dirname(self.storage_path) if os.path.dirname(self.storage_path) else ".", exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({
                    "history": [r.to_dict() for r in self.state_history[-1000:]],  # 只保留最近 1000 条
                    "best_fitness": self.best_fitness,
                    "updated_at": time.time()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[V(s)] 保存状态历史失败: {e}")
    
    def _normalize_state(self, state: Dict[str, float]) -> Dict[str, float]:
        """将状态参数归一化到 [0, 1]"""
        normalized = {}
        for k, v in state.items():
            if k in PARAM_BOUNDS:
                low, high = PARAM_BOUNDS[k]
                normalized[k] = (v - low) / (high - low) if high > low else 0.5
                normalized[k] = max(0.0, min(1.0, normalized[k]))
            else:
                normalized[k] = v
        return normalized
    
    def _compute_distance(self, state1: Dict[str, float], state2: Dict[str, float]) -> float:
        """计算两个状态之间的欧氏距离（归一化空间）"""
        norm1 = self._normalize_state(state1)
        norm2 = self._normalize_state(state2)
        
        # 只比较共有的参数
        common_keys = set(norm1.keys()) & set(norm2.keys()) & set(PARAM_BOUNDS.keys())
        if not common_keys:
            return float('inf')
        
        squared_sum = sum((norm1.get(k, 0) - norm2.get(k, 0)) ** 2 for k in common_keys)
        return math.sqrt(squared_sum / len(common_keys))
    
    def _fitness_to_value(self, fitness: float) -> float:
        """
        将 fitness 转换为价值
        fitness 越小越好 → 价值越高
        使用负的归一化 fitness 作为价值
        """
        if fitness >= SATURATION_PENALTY:
            return -1.0  # 磁饱和状态价值最低
        
        # 归一化到 [0, 1]，然后取负
        # fitness 在 [FITNESS_REF_MIN, FITNESS_REF_MAX] 范围内
        normalized = (fitness - FITNESS_REF_MIN) / (FITNESS_REF_MAX - FITNESS_REF_MIN)
        normalized = max(0.0, min(1.0, normalized))
        
        # 转换为价值：fitness 越小，价值越高
        # value 范围: [-1, 1]，fitness=3.5 → value≈1.0，fitness=100 → value≈-1.0
        value = 1.0 - 2.0 * normalized
        
        return value
    
    def estimate(self, state: Dict[str, float]) -> ValueEstimate:
        """
        估计状态价值 V(s)
        
        使用 k-近邻方法：
        1. 找到历史中最相似的 k 个状态
        2. 根据距离加权平均它们的价值
        """
        if not state:
            return ValueEstimate(
                value=0.0,
                confidence=0.0,
                similar_states_count=0,
                method="default",
                reasoning="空状态"
            )
        
        # 如果没有历史，返回中性估计
        if not self.state_history:
            return ValueEstimate(
                value=0.0,
                confidence=0.1,
                similar_states_count=0,
                method="default",
                reasoning="无历史数据，返回中性估计"
            )
        
        # 计算与所有历史状态的距离
        distances = []
        for record in self.state_history:
            if record.fitness >= SATURATION_PENALTY:
                continue  # 跳过磁饱和状态
            dist = self._compute_distance(state, record.state)
            if dist < float('inf'):
                distances.append((dist, record))
        
        if not distances:
            return ValueEstimate(
                value=0.0,
                confidence=0.1,
                similar_states_count=0,
                method="default",
                reasoning="未找到有效的相似状态"
            )
        
        # 排序并取 k 个最近邻
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k_neighbors]
        
        # 计算加权平均价值
        total_weight = 0.0
        weighted_value = 0.0
        
        for dist, record in k_nearest:
            # 距离越近权重越大，使用高斯核
            if dist < 0.001:
                weight = 10.0  # 几乎完全相同的状态
            else:
                weight = math.exp(-dist ** 2 / (2 * self.distance_threshold ** 2))
            
            value = self._fitness_to_value(record.fitness)
            weighted_value += weight * value
            total_weight += weight
        
        if total_weight > 0:
            estimated_value = weighted_value / total_weight
        else:
            estimated_value = 0.0
        
        # 计算置信度（基于邻居数量和距离）
        avg_distance = sum(d for d, _ in k_nearest) / len(k_nearest) if k_nearest else float('inf')
        confidence = min(1.0, len(k_nearest) / self.k_neighbors) * math.exp(-avg_distance / self.distance_threshold)
        
        # 构建理由
        nearest_dist, nearest_record = k_nearest[0]
        reasoning = (
            f"基于 {len(k_nearest)} 个相似状态估计 | "
            f"最近邻距离: {nearest_dist:.3f} | "
            f"最近邻 fitness: {nearest_record.fitness:.4f}"
        )
        
        return ValueEstimate(
            value=estimated_value,
            confidence=confidence,
            similar_states_count=len(k_nearest),
            method="knn",
            reasoning=reasoning
        )
    
    def update(self, state: Dict[str, float], fitness: float, reward: float = 0.0):
        """
        更新状态价值估计
        
        Args:
            state: 参数状态
            fitness: 该状态的 fitness
            reward: 获得的奖励
        """
        if not state:
            return
        
        value = self._fitness_to_value(fitness)
        
        # 检查是否是已存在的状态（近似匹配）
        existing_record = None
        for record in self.state_history:
            dist = self._compute_distance(state, record.state)
            if dist < 0.01:  # 非常接近的状态视为同一状态
                existing_record = record
                break
        
        if existing_record:
            # 更新已存在的记录
            existing_record.visit_count += 1
            existing_record.cumulative_reward += reward
            # 使用增量更新价值估计
            alpha = 1.0 / existing_record.visit_count
            existing_record.value = (1 - alpha) * existing_record.value + alpha * value
            existing_record.fitness = min(existing_record.fitness, fitness)  # 保留最好的 fitness
        else:
            # 添加新记录
            record = StateRecord(
                state=state.copy(),
                fitness=fitness,
                value=value,
                visit_count=1,
                cumulative_reward=reward
            )
            self.state_history.append(record)
        
        # 更新最佳状态
        if fitness < self.best_fitness and fitness < SATURATION_PENALTY:
            self.best_fitness = fitness
            self.best_state = StateRecord(
                state=state.copy(),
                fitness=fitness,
                value=value
            )
            logger.info(f"[V(s)] 🎯 发现新的最佳状态 | fitness={fitness:.4f} | V(s)={value:.3f}")
        
        # 定期保存
        if len(self.state_history) % 10 == 0:
            self._save_history()
    
    def compute_td_error(
        self,
        state: Dict[str, float],
        next_state: Dict[str, float],
        reward: float
    ) -> float:
        """
        计算时序差分误差 TD Error
        
        δ = r + γ·V(s') - V(s)
        
        TD 误差用于：
        - 正值：实际比预期好，应该增强这个方向
        - 负值：实际比预期差，应该避免这个方向
        """
        v_current = self.estimate(state)
        v_next = self.estimate(next_state)
        
        td_error = reward + self.gamma * v_next.value - v_current.value
        
        return td_error
    
    def get_exploration_priority(self, state: Dict[str, float]) -> float:
        """
        计算状态的探索优先级
        
        优先探索：
        1. 访问次数少的区域
        2. 价值估计不确定的区域
        3. 潜在价值高的区域
        """
        estimate = self.estimate(state)
        
        # 计算该区域的访问密度
        visit_density = 0
        for record in self.state_history:
            dist = self._compute_distance(state, record.state)
            if dist < self.distance_threshold:
                visit_density += record.visit_count
        
        # 探索优先级 = 不确定性 + 潜在价值 - 访问密度
        uncertainty_bonus = 1.0 - estimate.confidence
        potential_value = max(0, estimate.value + 0.5)  # 偏向价值较高的区域
        visit_penalty = min(1.0, visit_density / 10.0)
        
        priority = uncertainty_bonus + 0.5 * potential_value - 0.3 * visit_penalty
        
        return priority
    
    def get_best_known_state(self) -> Optional[Dict[str, float]]:
        """获取已知的最佳状态"""
        if self.best_state:
            return self.best_state.state.copy()
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """获取价值函数摘要"""
        valid_records = [r for r in self.state_history if r.fitness < SATURATION_PENALTY]
        
        summary = {
            "total_states": len(self.state_history),
            "valid_states": len(valid_records),
            "best_fitness": self.best_fitness if self.best_fitness < float('inf') else None,
            "k_neighbors": self.k_neighbors,
        }
        
        if valid_records:
            values = [r.value for r in valid_records]
            summary["avg_value"] = sum(values) / len(values)
            summary["max_value"] = max(values)
            summary["min_value"] = min(values)
        
        return summary
    
    def build_value_context(self, current_state: Dict[str, float]) -> str:
        """
        构建价值函数上下文，用于注入到 LLM 提示中
        """
        estimate = self.estimate(current_state)
        
        parts = [
            f"### 状态价值评估 V(s)",
            f"- 当前状态价值: {estimate.value:+.3f} (置信度 {estimate.confidence:.0%})",
            f"- 估计方法: {estimate.method} | {estimate.reasoning}"
        ]
        
        if self.best_state:
            parts.append(f"- 历史最佳 fitness: {self.best_fitness:.4f}")
            
            # 计算与最佳状态的距离
            dist_to_best = self._compute_distance(current_state, self.best_state.state)
            parts.append(f"- 与最佳状态距离: {dist_to_best:.3f}")
        
        # 探索优先级
        priority = self.get_exploration_priority(current_state)
        if priority > 0.7:
            parts.append(f"- 💡 探索建议: 该区域探索不足，建议深入探索")
        elif priority < 0.3:
            parts.append(f"- ⚠️ 该区域已充分探索，建议尝试新方向")
        
        return "\n".join(parts)


# ============== 集成的 Actor-Critic 系统 ==============

class ActorCriticSystem:
    """
    完整的 Actor-Critic 系统
    
    整合：
    - 状态价值函数 V(s)
    - 动作评论家集群（Advantage 估计）
    - TD 误差计算
    - 稠密奖励生成
    """
    
    def __init__(
        self,
        critic_ensemble: Optional[Any] = None,
        value_function: Optional[StateValueFunction] = None,
        alpha_dense: float = 0.3,
        beta_real: float = 0.7
    ):
        """
        Args:
            critic_ensemble: 评论家集群（评估动作）
            value_function: 状态价值函数 V(s)
            alpha_dense: 稠密奖励权重
            beta_real: 真实奖励权重
        """
        self.critic_ensemble = critic_ensemble
        self.value_function = value_function or StateValueFunction()
        self.alpha_dense = alpha_dense
        self.beta_real = beta_real
        
        # 缓存
        self._last_state: Optional[Dict[str, float]] = None
        self._last_value_estimate: Optional[ValueEstimate] = None
        self._last_critic_scores: Optional[Dict] = None
        self._last_dense_reward: float = 0.0
    
    async def evaluate_before_action(
        self,
        current_state: Dict[str, float],
        proposed_state: Dict[str, float],
        additional_context: Optional[str] = None
    ) -> Tuple[float, Dict[str, Any], List[str]]:
        """
        在执行动作（validate/simulate）之前评估
        
        Returns:
            (dense_reward, evaluation_result, suggestions)
        """
        evaluation = {
            "v_current": None,
            "v_proposed": None,
            "advantage": None,
            "critic_scores": {},
            "td_estimate": None
        }
        suggestions = []
        
        # 1. 估计当前状态价值 V(s)
        v_current = self.value_function.estimate(current_state)
        evaluation["v_current"] = v_current.to_dict()
        self._last_value_estimate = v_current
        
        # 2. 估计提议状态价值 V(s')
        v_proposed = self.value_function.estimate(proposed_state)
        evaluation["v_proposed"] = v_proposed.to_dict()
        
        # 3. 计算预估优势 A(s,a) ≈ V(s') - V(s)
        advantage = v_proposed.value - v_current.value
        evaluation["advantage"] = advantage
        
        # 4. 调用动作评论家集群
        dense_reward = advantage  # 基础稠密奖励来自价值差
        
        if self.critic_ensemble:
            try:
                critic_dense, critic_scores, critic_suggestions = await self.critic_ensemble.evaluate(
                    params_before=current_state,
                    params_after=proposed_state,
                    additional_context=additional_context
                )
                evaluation["critic_scores"] = {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                                               for k, v in critic_scores.items()}
                suggestions.extend(critic_suggestions)
                self._last_critic_scores = critic_scores
                
                # 融合价值函数估计和评论家评估
                # dense_reward = 0.5 * advantage + 0.5 * critic_dense
                dense_reward = 0.4 * advantage + 0.6 * critic_dense
                
            except Exception as e:
                logger.warning(f"[AC] 评论家评估异常: {e}")
        
        self._last_dense_reward = dense_reward
        self._last_state = current_state.copy()
        
        # 生成建议
        if advantage < -0.3:
            suggestions.append(f"⚠️ 价值函数预测: 提议状态价值较低 (ΔV={advantage:.2f})")
        elif advantage > 0.3:
            suggestions.append(f"✓ 价值函数预测: 提议状态价值较高 (ΔV={advantage:.2f})")
        
        return dense_reward, evaluation, suggestions
    
    def update_after_result(
        self,
        state: Dict[str, float],
        next_state: Dict[str, float],
        fitness: float,
        reward: float
    ) -> Dict[str, float]:
        """
        在获得真实结果后更新价值函数
        
        Returns:
            包含 TD 误差等信息的字典
        """
        result = {}
        
        # 更新价值函数
        self.value_function.update(next_state, fitness, reward)
        
        # 计算 TD 误差
        td_error = self.value_function.compute_td_error(state, next_state, reward)
        result["td_error"] = td_error
        
        # 计算预测准确性（稠密奖励与真实奖励的一致性）
        if self._last_dense_reward != 0:
            direction_match = (self._last_dense_reward > 0 and reward > 0) or \
                              (self._last_dense_reward < 0 and reward < 0) or \
                              (abs(self._last_dense_reward) < 0.1 and abs(reward) < 0.5)
            result["prediction_accuracy"] = 1.0 if direction_match else 0.0
        
        # 更新评论家的实际结果
        if self.critic_ensemble and self._last_state:
            try:
                self.critic_ensemble.record_actual_result(
                    params_before=self._last_state,
                    params_after=next_state,
                    actual_result={"fitness": fitness},
                    actual_reward=reward
                )
            except Exception as e:
                logger.warning(f"[AC] 更新评论家失败: {e}")
        
        return result
    
    def compute_combined_reward(self, real_reward: float) -> float:
        """
        计算融合奖励
        
        r_combined = β·r_real + α·r_dense
        """
        combined = self.beta_real * real_reward + self.alpha_dense * self._last_dense_reward * 10
        return combined
    
    def build_context_for_llm(self, current_state: Dict[str, float]) -> str:
        """
        构建完整的 Actor-Critic 上下文，用于注入 LLM 提示
        """
        parts = []
        
        # 价值函数上下文
        value_context = self.value_function.build_value_context(current_state)
        parts.append(value_context)
        
        # 上一轮的稠密奖励
        if self._last_dense_reward != 0:
            emoji = "✓" if self._last_dense_reward > 0 else "✗" if self._last_dense_reward < 0 else "→"
            parts.append(f"\n### 上轮评论家反馈")
            parts.append(f"{emoji} 稠密奖励: {self._last_dense_reward:+.3f}")
        
        return "\n".join(parts)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取 Actor-Critic 系统摘要"""
        summary = {
            "value_function": self.value_function.get_summary(),
            "alpha_dense": self.alpha_dense,
            "beta_real": self.beta_real
        }
        
        if self.critic_ensemble:
            summary["critic_ensemble"] = self.critic_ensemble.get_summary()
        
        return summary

