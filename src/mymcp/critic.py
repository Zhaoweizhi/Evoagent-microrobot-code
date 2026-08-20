# -*- coding: utf-8 -*-
"""
评论家Agent模块 (Critic Agent Module)

实现Actor-Critic架构中的Critic部分，提供稠密奖励（Dense Reward）
每个评论家Agent使用与主Agent相同的架构：LLM + Prompt + 经验库/策略库

评论家类型：
1. 磁路评论家 (MagneticCritic): 评估ta, tb_ratio, dg改动对磁饱和的影响
2. 性能评论家 (PerformanceCritic): 预估整体参数改动对fitness的影响
3. 约束评论家 (ConstraintCritic): 快速约束预检，评估违规风险
4. 幅度评论家 (MagnitudeCritic): 评估参数改动幅度的合理性
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import asyncio
from loguru import logger

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


# ============== 评论家配置 ==============

# 默认评论家模型配置
DEFAULT_CRITIC_MODELS = {
    "magnetic": "openai/gpt-4o",      # 磁路评论家 - 物理推理
    "performance": "openai/gpt-4o",   # 性能评论家
    "constraint": "openai/gpt-4o",    # 约束评论家
    "magnitude": "openai/gpt-4o",     # 幅度评论家
}

# 评论家权重配置
DEFAULT_CRITIC_WEIGHTS = {
    "magnetic": 0.35,      # 磁路评论家权重（磁饱和是核心问题）
    "performance": 0.30,   # 性能评论家权重
    "constraint": 0.20,    # 约束评论家权重
    "magnitude": 0.15,     # 幅度评论家权重
}

# 设计参数边界（用于约束评论家和幅度评论家）
DESIGN_BOUNDS = {
    "lm": (3.0, 6.5),
    "tm": (0.3, 0.8),
    "ta": (0.4, 1.2),
    "dg": (0.2, 1.0),
    "hs": (1.0, 4.0),
    "wslot": (1.5, 4.0),
    "hslot": (0.8, 2.5),
    "s": (0.5, 2.0),
    "tb_ratio": (1.6, 2.0),
}

# 磁饱和敏感参数
MAGNETIC_SENSITIVE_PARAMS = {"ta", "tb_ratio", "dg", "tm"}

# 约束敏感参数
CONSTRAINT_SENSITIVE_PARAMS = {"wslot", "hslot", "s", "n1", "n2"}


# ============== 数据结构 ==============

class CriticType(Enum):
    """评论家类型"""
    MAGNETIC = "magnetic"
    PERFORMANCE = "performance"
    CONSTRAINT = "constraint"
    MAGNITUDE = "magnitude"


@dataclass
class CriticScore:
    """评论家评分结果"""
    critic_type: str                    # 评论家类型
    score: float                        # 评分 (-1.0 ~ +1.0)
    confidence: float                   # 置信度 (0.0 ~ 1.0)
    direction: str                      # 方向建议 ("positive", "negative", "neutral")
    reasoning: str                      # 评分理由
    suggestions: List[str] = field(default_factory=list)  # 改进建议
    timestamp: float = field(default_factory=time.time)
    
    # TD 学习相关（方案A）
    td_adjusted_confidence: Optional[float] = None  # TD误差调整后的置信度
    prediction_accuracy: Optional[float] = None      # 历史预测准确率
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TDStatistics:
    """TD 误差统计（方案B）"""
    param_name: str                     # 参数名
    direction: str                      # 改动方向 ("increase", "decrease")
    count: int = 0                      # 样本数
    avg_td_error: float = 0.0           # 平均 TD 误差
    avg_prediction_error: float = 0.0  # 平均预测偏差
    accuracy_rate: float = 0.5          # 预测准确率
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CriticExperience:
    """评论家经验记录"""
    iteration: int                      # 迭代轮次
    params_before: Dict[str, float]     # 改动前参数
    params_after: Dict[str, float]      # 改动后参数
    param_deltas: Dict[str, float]      # 参数变化量
    critic_scores: Dict[str, CriticScore]  # 各评论家评分
    dense_reward: float                 # 稠密奖励
    actual_result: Optional[Dict] = None  # 实际仿真结果（事后填充）
    actual_reward: Optional[float] = None  # 实际奖励（事后填充）
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["critic_scores"] = {k: v.to_dict() if isinstance(v, CriticScore) else v 
                              for k, v in self.critic_scores.items()}
        return d


@dataclass
class CriticConfig:
    """评论家配置"""
    critic_type: CriticType
    model: str
    base_url: str
    api_key: Optional[str]
    weight: float = 0.25
    timeout: int = 30
    enabled: bool = True
    
    # 经验库路径
    experience_path: Optional[str] = None
    # 策略库路径
    strategy_path: Optional[str] = None
    
    # ★批处理模式配置（TD(n)）
    enable_batch_mode: bool = False       # 是否启用批处理模式
    batch_interval: int = 3               # 批处理间隔（每 n 轮生成一次规则）


# ============== 评论家基类 ==============

class BaseCritic(ABC):
    """评论家基类"""
    
    # 置信度调整参数（方案A）
    CONFIDENCE_INCREASE_RATE = 1.05   # 预测正确时置信度增加率
    CONFIDENCE_DECREASE_RATE = 0.90   # 预测错误时置信度降低率
    MIN_CONFIDENCE = 0.2              # 最小置信度
    MAX_CONFIDENCE = 0.95             # 最大置信度
    
    def __init__(self, config: CriticConfig):
        self.config = config
        self.critic_type = config.critic_type
        self.weight = config.weight
        self.enabled = config.enabled
        
        # LLM 客户端
        if AsyncOpenAI and config.api_key:
            self.llm = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
        else:
            self.llm = None
        
        # 经验库
        self.experiences: List[Dict] = []
        self._load_experiences()
        
        # 策略库
        self.strategies: List[str] = []
        self._load_strategies()
        
        # ========== TD 学习相关（方案A + B）==========
        # 动态置信度（方案A）
        self.dynamic_confidence: float = 0.5  # 初始置信度
        self.prediction_history: List[bool] = []  # 预测正确/错误历史
        
        # TD 误差统计（方案B）
        self.td_statistics: Dict[str, TDStatistics] = {}  # key: "param_direction"
        self._load_td_statistics()
        
        # 统计计数
        self.total_predictions: int = 0
        self.correct_predictions: int = 0
        
        # ★批处理模式（TD(n)）
        self.enable_batch_mode: bool = config.enable_batch_mode
        self.batch_interval: int = config.batch_interval
        self._batch_update_counter: int = 0  # 批处理计数器
        self._pending_td_updates: List[Dict] = []  # 待处理的 TD 更新
    
    def _load_experiences(self):
        """加载评论家经验库"""
        if self.config.experience_path and os.path.exists(self.config.experience_path):
            try:
                with open(self.config.experience_path, "r", encoding="utf-8") as f:
                    self.experiences = json.load(f)
                logger.info(f"[{self.critic_type.value}] 加载 {len(self.experiences)} 条经验")
            except Exception as e:
                logger.warning(f"[{self.critic_type.value}] 加载经验失败: {e}")
    
    def _save_experiences(self):
        """保存评论家经验库"""
        if self.config.experience_path:
            try:
                os.makedirs(os.path.dirname(self.config.experience_path), exist_ok=True)
                with open(self.config.experience_path, "w", encoding="utf-8") as f:
                    json.dump(self.experiences, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"[{self.critic_type.value}] 保存经验失败: {e}")
    
    def _load_strategies(self):
        """加载评论家策略库"""
        if self.config.strategy_path and os.path.exists(self.config.strategy_path):
            try:
                with open(self.config.strategy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.strategies = data.get("strategies", [])
                logger.info(f"[{self.critic_type.value}] 加载 {len(self.strategies)} 条策略")
            except Exception as e:
                logger.warning(f"[{self.critic_type.value}] 加载策略失败: {e}")
    
    def _save_strategies(self):
        """保存评论家策略库"""
        if self.config.strategy_path:
            try:
                os.makedirs(os.path.dirname(self.config.strategy_path), exist_ok=True)
                with open(self.config.strategy_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "strategies": self.strategies,
                        "dynamic_confidence": self.dynamic_confidence,
                        "total_predictions": self.total_predictions,
                        "correct_predictions": self.correct_predictions
                    }, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"[{self.critic_type.value}] 保存策略失败: {e}")
    
    def _load_td_statistics(self):
        """加载 TD 误差统计（方案B）"""
        td_stats_path = self.config.strategy_path.replace("_strategy.json", "_td_stats.json") if self.config.strategy_path else None
        if td_stats_path and os.path.exists(td_stats_path):
            try:
                with open(td_stats_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, stat_dict in data.get("statistics", {}).items():
                        self.td_statistics[key] = TDStatistics(**stat_dict)
                    # 恢复动态置信度
                    self.dynamic_confidence = data.get("dynamic_confidence", 0.5)
                    self.total_predictions = data.get("total_predictions", 0)
                    self.correct_predictions = data.get("correct_predictions", 0)
                logger.info(f"[{self.critic_type.value}] 加载 TD 统计: {len(self.td_statistics)} 条 | 置信度: {self.dynamic_confidence:.2f}")
            except Exception as e:
                logger.warning(f"[{self.critic_type.value}] 加载 TD 统计失败: {e}")
    
    def _save_td_statistics(self):
        """保存 TD 误差统计"""
        td_stats_path = self.config.strategy_path.replace("_strategy.json", "_td_stats.json") if self.config.strategy_path else None
        if td_stats_path:
            try:
                os.makedirs(os.path.dirname(td_stats_path), exist_ok=True)
                with open(td_stats_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "statistics": {k: v.to_dict() for k, v in self.td_statistics.items()},
                        "dynamic_confidence": self.dynamic_confidence,
                        "total_predictions": self.total_predictions,
                        "correct_predictions": self.correct_predictions
                    }, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"[{self.critic_type.value}] 保存 TD 统计失败: {e}")
    
    def update_with_td_error(
        self,
        td_error: float,
        prediction_score: float,
        param_deltas: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        基于 TD 误差更新评论家（方案A + B）
        
        支持两种模式：
        - 即时模式（enable_batch_mode=False）：每轮更新统计并检查规则生成
        - 批处理模式（enable_batch_mode=True）：每轮更新统计，但每 batch_interval 轮才生成规则
        
        Args:
            td_error: TD 误差 δ = r + γV(s') - V(s)
            prediction_score: 评论家的预测评分
            param_deltas: 参数改动量
        
        Returns:
            更新结果字典
        """
        result = {
            "old_confidence": self.dynamic_confidence,
            "new_confidence": self.dynamic_confidence,
            "prediction_correct": False,
            "new_rules": [],
            "batch_mode": self.enable_batch_mode,
            "batch_counter": self._batch_update_counter
        }
        
        self.total_predictions += 1
        self._batch_update_counter += 1
        
        # ========== 方案A: 动态调整置信度（每轮都执行）==========
        # 判断预测是否正确（方向一致或幅度匹配）
        # ★ 放宽条件：方向一致即可，不强求幅度匹配
        direction_match = (
            (td_error > 0 and prediction_score > 0) or   # 方向都是正（会变好）
            (td_error < 0 and prediction_score < 0) or   # 方向都是负（会变差）
            (abs(td_error) < 0.05 and abs(prediction_score) < 0.2)  # 都认为变化不大（放宽阈值）
        )
        
        # 幅度严格匹配（原条件）
        magnitude_match = (
            (td_error > 0.1 and prediction_score > 0.1) or
            (td_error < -0.1 and prediction_score < -0.1) or
            (abs(td_error) <= 0.1 and abs(prediction_score) <= 0.1)
        )
        
        # ★ 只要方向对了就算正确，幅度匹配额外加分
        prediction_correct = direction_match
        
        result["prediction_correct"] = prediction_correct
        result["direction_match"] = direction_match
        result["magnitude_match"] = magnitude_match
        self.prediction_history.append(prediction_correct)
        
        if prediction_correct:
            self.correct_predictions += 1
            # 预测正确，增加置信度
            self.dynamic_confidence = min(
                self.MAX_CONFIDENCE,
                self.dynamic_confidence * self.CONFIDENCE_INCREASE_RATE
            )
        else:
            # 预测错误，降低置信度
            self.dynamic_confidence = max(
                self.MIN_CONFIDENCE,
                self.dynamic_confidence * self.CONFIDENCE_DECREASE_RATE
            )
        
        result["new_confidence"] = self.dynamic_confidence
        
        # ========== 方案B: 更新 TD 误差统计（每轮都执行）==========
        for param, delta in param_deltas.items():
            if abs(delta) < 0.01:  # 忽略微小变化
                continue
            
            direction = "increase" if delta > 0 else "decrease"
            key = f"{param}_{direction}"
            
            # 更新或创建统计
            if key not in self.td_statistics:
                self.td_statistics[key] = TDStatistics(
                    param_name=param,
                    direction=direction
                )
            
            stat = self.td_statistics[key]
            stat.count += 1
            
            # 增量更新平均值
            alpha = 1.0 / stat.count
            stat.avg_td_error = (1 - alpha) * stat.avg_td_error + alpha * td_error
            prediction_error = td_error - prediction_score
            stat.avg_prediction_error = (1 - alpha) * stat.avg_prediction_error + alpha * prediction_error
            
            # 更新准确率
            if prediction_correct:
                stat.accuracy_rate = (1 - alpha) * stat.accuracy_rate + alpha * 1.0
            else:
                stat.accuracy_rate = (1 - alpha) * stat.accuracy_rate + alpha * 0.0
            
            stat.last_updated = time.time()
        
        # ========== 规则生成：根据批处理模式决定执行时机 ==========
        should_generate_rules = True
        if self.enable_batch_mode:
            # 批处理模式：每 batch_interval 轮执行一次规则生成
            should_generate_rules = (self._batch_update_counter >= self.batch_interval)
            if should_generate_rules:
                logger.info(f"[{self.critic_type.value}] 📦 批处理模式：第 {self._batch_update_counter} 轮，触发规则生成")
                self._batch_update_counter = 0  # 重置计数器
        
        if should_generate_rules:
            result["new_rules"] = self._batch_generate_rules()
        
        # 保存更新
        self._save_td_statistics()
        if result["new_rules"]:
            self._save_strategies()
        
        return result
    
    def _batch_generate_rules(self) -> List[str]:
        """批量生成规则（遍历所有 TD 统计，检查是否满足规则生成条件）"""
        new_rules = []
        
        for key, stat in self.td_statistics.items():
            # ★门槛提高：样本数 >= 10 且 准确率 > 60% 且存在显著偏差
            should_generate = (
                stat.count >= 10 and 
                stat.accuracy_rate > 0.6 and 
                abs(stat.avg_prediction_error) > 0.2
            )
            
            if should_generate:
                new_rule = self._generate_td_rule(stat)
                if new_rule:
                    # ★冲突检测：移除同一参数的所有旧 TD 规则和"注意"规则
                    conflict_removed = self._remove_conflicting_rules(stat.param_name)
                    if conflict_removed:
                        logger.info(f"[{self.critic_type.value}] 🔄 移除 {stat.param_name} 的 {conflict_removed} 条旧规则")
                    
                    if new_rule not in self.strategies:
                        self.strategies.append(new_rule)
                        new_rules.append(new_rule)
                        logger.info(f"[{self.critic_type.value}] 📚 学习新规则（样本={stat.count}, 准确率={stat.accuracy_rate:.0%}）: {new_rule}")
        
        return new_rules
    
    def _remove_conflicting_rules(self, param_name: str) -> int:
        """移除与指定参数相关的所有旧规则（TD规则 + 注意规则）"""
        removed_count = 0
        rules_to_remove = []
        
        for s in self.strategies:
            # 检查是否是该参数的 TD 学习规则
            is_td_rule = param_name in s and ("TD误差" in s or "效果通常" in s or "评估整体偏" in s)
            # 检查是否是该参数的"注意"规则
            is_notice_rule = f"'{param_name}'" in s and "注意" in s and "改动预测需要更谨慎" in s
            
            if is_td_rule or is_notice_rule:
                rules_to_remove.append(s)
        
        for rule in rules_to_remove:
            self.strategies.remove(rule)
            removed_count += 1
        
        return removed_count
    
    def _generate_td_rule(self, stat: TDStatistics) -> Optional[str]:
        """根据 TD 统计生成策略规则（改进版：检测双向冲突）"""
        param = stat.param_name
        direction_cn = "增大" if stat.direction == "increase" else "减小"
        opposite_direction = "decrease" if stat.direction == "increase" else "increase"
        
        # 检查相反方向的统计
        opposite_key = f"{param}_{opposite_direction}"
        opposite_stat = self.td_statistics.get(opposite_key)
        
        # 如果两个方向都存在且都偏保守（或都偏乐观），说明评论家整体有偏差
        if opposite_stat and opposite_stat.count >= 3:
            both_conservative = stat.avg_prediction_error > 0.15 and opposite_stat.avg_prediction_error > 0.15
            both_optimistic = stat.avg_prediction_error < -0.15 and opposite_stat.avg_prediction_error < -0.15
            
            if both_conservative:
                # 双向都偏保守：评论家对该参数整体过于谨慎
                return f"对 {param} 的评估整体偏保守，无论增大还是减小，实际效果通常都比预测更好，建议提高对 {param} 改动的评分"
            elif both_optimistic:
                # 双向都偏乐观：评论家对该参数整体过于乐观
                return f"对 {param} 的评估整体偏乐观，无论增大还是减小，实际效果通常都比预测更差，建议降低对 {param} 改动的评分"
            else:
                # 一个方向偏保守，另一个偏乐观：说明参数有单调性影响
                if stat.avg_prediction_error > 0.2:
                    return f"{param} {direction_cn}时效果通常优于预测（TD误差={stat.avg_td_error:+.2f}），而{('减小' if stat.direction == 'increase' else '增大')}时效果较差"
                elif stat.avg_prediction_error < -0.2:
                    return f"{param} {direction_cn}时效果通常差于预测（TD误差={stat.avg_td_error:+.2f}），而{('减小' if stat.direction == 'increase' else '增大')}时效果较好"
        else:
            # 只有单方向数据
            if stat.avg_prediction_error > 0.2:
                return f"当 {param} {direction_cn}时，实际效果通常比预测更好（TD误差={stat.avg_td_error:+.2f}），可以更积极评分"
            elif stat.avg_prediction_error < -0.2:
                return f"当 {param} {direction_cn}时，实际效果通常比预测更差（TD误差={stat.avg_td_error:+.2f}），需要更保守评分"
        
        return None
    
    def get_adjusted_confidence(self) -> float:
        """获取 TD 调整后的置信度（方案A）"""
        return self.dynamic_confidence
    
    def get_prediction_accuracy(self) -> float:
        """获取历史预测准确率"""
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions
    
    def get_td_summary(self) -> Dict[str, Any]:
        """获取 TD 学习摘要（用于前端显示）"""
        return {
            "critic_type": self.critic_type.value,
            "dynamic_confidence": self.dynamic_confidence,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy_rate": self.get_prediction_accuracy(),
            "td_statistics_count": len(self.td_statistics),
            "learned_rules_from_td": len([s for s in self.strategies if "TD误差" in s])
        }
    
    def build_td_context_for_prompt(self) -> str:
        """构建 TD 学习上下文，注入到评论家 Prompt（方案B）"""
        if not self.td_statistics:
            return ""
        
        parts = ["### TD 学习历史规律："]
        
        for key, stat in sorted(self.td_statistics.items(), key=lambda x: -x[1].count)[:5]:
            if stat.count < 3:
                continue
            
            direction = "增大" if stat.direction == "increase" else "减小"
            bias = "偏保守" if stat.avg_prediction_error > 0.1 else "偏乐观" if stat.avg_prediction_error < -0.1 else "准确"
            
            parts.append(
                f"- {stat.param_name} {direction}: "
                f"TD误差={stat.avg_td_error:+.2f}, 预测{bias}, 准确率={stat.accuracy_rate:.0%} (n={stat.count})"
            )
        
        return "\n".join(parts) if len(parts) > 1 else ""
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """获取评论家系统提示词"""
        pass
    
    @abstractmethod
    def build_evaluation_prompt(
        self, 
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        param_deltas: Dict[str, float]
    ) -> str:
        """构建评估提示词"""
        pass
    
    def _build_experience_context(self, params_after: Dict[str, float], limit: int = 3) -> str:
        """构建经验上下文"""
        if not self.experiences:
            return ""
        
        # 找到最相似的历史经验
        similar = []
        for exp in self.experiences[-50:]:  # 只看最近50条
            if "params_after" not in exp:
                continue
            # 计算相似度（简单的参数距离）
            dist = sum(
                abs(exp["params_after"].get(k, 0) - params_after.get(k, 0))
                for k in params_after.keys()
            )
            similar.append((dist, exp))
        
        similar.sort(key=lambda x: x[0])
        top_experiences = similar[:limit]
        
        if not top_experiences:
            return ""
        
        context_parts = ["### 相似历史经验参考："]
        for i, (dist, exp) in enumerate(top_experiences):
            score = exp.get("critic_score", {}).get("score", 0)
            actual = exp.get("actual_reward")
            accuracy = ""
            if actual is not None and score != 0:
                # 评估准确性
                if (score > 0 and actual > 0) or (score < 0 and actual < 0):
                    accuracy = "✓方向正确"
                else:
                    accuracy = "✗方向错误"
            
            deltas = exp.get("param_deltas", {})
            delta_str = ", ".join(f"{k}={v:+.3f}" for k, v in deltas.items() if abs(v) > 0.001)
            context_parts.append(
                f"{i+1}. 改动: {delta_str}\n"
                f"   评分: {score:+.2f} | 实际: {(f'{actual:.2f}' if actual is not None else 'N/A')} {accuracy}"
            )
        
        return "\n".join(context_parts)
    
    def _build_strategy_context(self) -> str:
        """构建策略上下文"""
        if not self.strategies:
            return ""
        
        return "### 已学习的评估策略：\n" + "\n".join(f"- {s}" for s in self.strategies[:10])
    
    async def evaluate(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        additional_context: Optional[str] = None
    ) -> CriticScore:
        """评估参数改动
        
        Args:
            params_before: 改动前参数
            params_after: 改动后参数
            additional_context: 额外上下文（如当前最佳fitness）
        
        Returns:
            CriticScore: 评分结果
        """
        if not self.enabled:
            return CriticScore(
                critic_type=self.critic_type.value,
                score=0.0,
                confidence=0.0,
                direction="neutral",
                reasoning="评论家已禁用"
            )
        
        # 计算参数变化
        param_deltas = {}
        for k in set(params_before.keys()) | set(params_after.keys()):
            before = params_before.get(k, 0)
            after = params_after.get(k, 0)
            if before != 0 or after != 0:
                param_deltas[k] = after - before
        
        # 如果没有LLM，使用规则评估
        if self.llm is None:
            return self._rule_based_evaluate(params_before, params_after, param_deltas)
        
        # 构建提示
        system_prompt = self.get_system_prompt()
        experience_context = self._build_experience_context(params_after)
        strategy_context = self._build_strategy_context()
        td_context = self.build_td_context_for_prompt()  # TD 学习上下文（方案B）
        eval_prompt = self.build_evaluation_prompt(params_before, params_after, param_deltas)
        
        full_prompt = eval_prompt
        if experience_context:
            full_prompt = experience_context + "\n\n" + full_prompt
        if strategy_context:
            full_prompt = strategy_context + "\n\n" + full_prompt
        if td_context:
            full_prompt = td_context + "\n\n" + full_prompt
        if additional_context:
            full_prompt += f"\n\n### 额外信息：\n{additional_context}"
        
        try:
            response = await self.llm.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            score = self._parse_response(result_text, param_deltas)
            
            # 添加 TD 调整后的置信度（方案A）
            score.td_adjusted_confidence = self.dynamic_confidence
            score.prediction_accuracy = self.get_prediction_accuracy()
            
            return score
            
        except Exception as e:
            logger.warning(f"[{self.critic_type.value}] LLM评估失败: {e}，使用规则评估")
            score = self._rule_based_evaluate(params_before, params_after, param_deltas)
            score.td_adjusted_confidence = self.dynamic_confidence
            score.prediction_accuracy = self.get_prediction_accuracy()
            return score
    
    def _parse_response(self, response_text: str, param_deltas: Dict[str, float]) -> CriticScore:
        """解析LLM响应"""
        import re
        
        # 尝试解析JSON格式
        try:
            # 查找JSON块
            json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return CriticScore(
                    critic_type=self.critic_type.value,
                    score=float(data.get("score", 0)),
                    confidence=float(data.get("confidence", 0.5)),
                    direction=data.get("direction", "neutral"),
                    reasoning=data.get("reasoning", response_text[:200]),
                    suggestions=data.get("suggestions", [])
                )
        except Exception:
            pass
        
        # 解析纯文本格式
        score = 0.0
        confidence = 0.5
        direction = "neutral"
        
        # 查找评分
        score_match = re.search(r'(?:score|评分)[：:]\s*([-+]?\d*\.?\d+)', response_text, re.IGNORECASE)
        if score_match:
            score = max(-1.0, min(1.0, float(score_match.group(1))))
        
        # 查找置信度
        conf_match = re.search(r'(?:confidence|置信度)[：:]\s*(\d*\.?\d+)', response_text, re.IGNORECASE)
        if conf_match:
            confidence = max(0.0, min(1.0, float(conf_match.group(1))))
        
        # 判断方向
        if score > 0.1:
            direction = "positive"
        elif score < -0.1:
            direction = "negative"
        
        return CriticScore(
            critic_type=self.critic_type.value,
            score=score,
            confidence=confidence,
            direction=direction,
            reasoning=response_text[:300]
        )
    
    @abstractmethod
    def _rule_based_evaluate(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        param_deltas: Dict[str, float]
    ) -> CriticScore:
        """基于规则的评估（LLM不可用时的备选方案）"""
        pass
    
    def record_actual_result(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        critic_score: CriticScore,
        actual_result: Dict,
        actual_reward: float
    ):
        """记录实际结果，用于校准评论家"""
        param_deltas = {
            k: params_after.get(k, 0) - params_before.get(k, 0)
            for k in set(params_before.keys()) | set(params_after.keys())
        }
        
        experience = {
            "timestamp": time.time(),
            "params_before": params_before,
            "params_after": params_after,
            "param_deltas": param_deltas,
            "critic_score": critic_score.to_dict(),
            "actual_result": actual_result,
            "actual_reward": actual_reward
        }
        
        self.experiences.append(experience)
        self._save_experiences()
        
        # 学习新策略
        self._learn_from_experience(experience)
    
    def _learn_from_experience(self, experience: Dict):
        """从经验中学习新策略（改进版：避免重复添加"注意"规则）"""
        score = experience.get("critic_score", {}).get("score", 0)
        actual = experience.get("actual_reward", 0)
        
        if actual is None:
            return
        
        # 如果预测方向错误，记录为需要调整的策略
        if (score > 0.2 and actual < -0.2) or (score < -0.2 and actual > 0.2):
            deltas = experience.get("param_deltas", {})
            significant_changes = [k for k, v in deltas.items() if abs(v) > 0.05]
            if significant_changes:
                # ★改进：检查是否已存在相同参数的"注意"规则
                # 同一组参数只保留一条"注意"规则
                param_key = str(sorted(significant_changes))  # 标准化参数列表
                
                # 检查是否已有相同参数组合的规则
                existing_notice = any(
                    param_key in s and "注意" in s and "改动预测需要更谨慎" in s
                    for s in self.strategies
                )
                
                if not existing_notice:
                    strategy = f"注意: 参数 {significant_changes} 的改动预测需要更谨慎（历史预测偏差）"
                    if strategy not in self.strategies:
                        self.strategies.append(strategy)
                        self._save_strategies()
                        logger.info(f"[{self.critic_type.value}] 📝 记录注意事项: {strategy}")


# ============== 具体评论家实现 ==============

class MagneticCritic(BaseCritic):
    """磁路评论家 - 评估磁饱和风险"""
    
    def get_system_prompt(self) -> str:
        return """你是一个电磁执行器磁路分析专家评论家。
你的任务是评估参数改动对磁饱和（B_max）的影响。

关键知识：
1. ta（铁芯厚度）增大 → 磁路截面增大 → B减小（降低饱和风险）
2. tb_ratio 增大 → tb增大 → 漏磁路径增加 → B可能变化复杂
3. dg（气隙）增大 → 磁阻增大 → B减小
4. tm（磁铁厚度）增大 → 磁动势增大 → B增大

约束：B_max 必须 < 2.0T，否则磁饱和会导致性能下降。

请根据参数改动，评估对磁饱和的影响：
- score: -1.0（会加剧饱和）到 +1.0（会减轻饱和）
- confidence: 0.0-1.0 置信度
- direction: positive/negative/neutral

输出JSON格式：
{"score": 0.5, "confidence": 0.8, "direction": "positive", "reasoning": "...", "suggestions": [...]}"""
    
    def build_evaluation_prompt(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        param_deltas: Dict[str, float]
    ) -> str:
        # 只关注磁路相关参数
        relevant_params = {"ta", "tb_ratio", "dg", "tm", "lm"}
        
        before_text = ", ".join(
            f"{k}={params_before.get(k, 0):.3f}" 
            for k in relevant_params if k in params_before
        )
        after_text = ", ".join(
            f"{k}={params_after.get(k, 0):.3f}" 
            for k in relevant_params if k in params_after
        )
        delta_text = ", ".join(
            f"Δ{k}={v:+.3f}" 
            for k, v in param_deltas.items() 
            if k in relevant_params and abs(v) > 0.001
        )
        
        return f"""### 磁路参数改动评估

**改动前**: {before_text}
**改动后**: {after_text}
**变化量**: {delta_text if delta_text else "无显著变化"}

请评估这次改动对磁饱和的影响，输出JSON格式评分。"""
    
    def _rule_based_evaluate(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        param_deltas: Dict[str, float]
    ) -> CriticScore:
        """基于规则的磁路评估"""
        score = 0.0
        reasons = []
        
        # ta增大 → 降低饱和风险 → 正分
        delta_ta = param_deltas.get("ta", 0)
        if delta_ta > 0.01:
            score += 0.3
            reasons.append(f"ta增大{delta_ta:.3f}mm，有助降低饱和")
        elif delta_ta < -0.01:
            score -= 0.3
            reasons.append(f"ta减小{abs(delta_ta):.3f}mm，可能加剧饱和")
        
        # dg增大 → 磁阻增大 → B减小 → 正分（但推力也会减小）
        delta_dg = param_deltas.get("dg", 0)
        if delta_dg > 0.01:
            score += 0.2
            reasons.append(f"dg增大{delta_dg:.3f}mm，降低B但可能降推力")
        elif delta_dg < -0.01:
            score -= 0.2
            reasons.append(f"dg减小{abs(delta_dg):.3f}mm，B可能增大")
        
        # tm增大 → B增大 → 负分（饱和角度）
        delta_tm = param_deltas.get("tm", 0)
        if delta_tm > 0.01:
            score -= 0.15
            reasons.append(f"tm增大{delta_tm:.3f}mm，磁动势增大可能加剧饱和")
        
        # tb_ratio变化 → 复杂影响
        delta_tb_ratio = param_deltas.get("tb_ratio", 0)
        if abs(delta_tb_ratio) > 0.05:
            reasons.append(f"tb_ratio变化{delta_tb_ratio:+.2f}，影响需观察")
        
        score = max(-1.0, min(1.0, score))
        direction = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        
        return CriticScore(
            critic_type=self.critic_type.value,
            score=score,
            confidence=0.6,  # 规则评估置信度较低
            direction=direction,
            reasoning="; ".join(reasons) if reasons else "无显著磁路参数变化"
        )


class PerformanceCritic(BaseCritic):
    """性能评论家 - 预估fitness变化趋势"""
    
    def get_system_prompt(self) -> str:
        return """你是一个电磁执行器性能优化专家评论家。
你的任务是预估参数改动对整体性能（fitness）的影响。

fitness 计算公式（越小越好）：
fitness = (volume/V_ref) + (mass/M_ref) + (1/kb)·(kb_ref) + (1/pb)·(pb_ref)

其中：
- volume: 体积（与尺寸相关）
- mass: 质量（与尺寸和材料相关）
- kb: 推力线性度（越大越好）
- pb: 功率系数（越大越好）

关键关系：
1. 减小尺寸（la, ha等）→ volume↓, mass↓ → fitness可能改善
2. 但尺寸过小 → kb, pb下降 → fitness变差
3. 需要在体积/质量和性能之间权衡

请评估参数改动对fitness的预期影响：
- score: -1.0（预计fitness变差）到 +1.0（预计fitness改善）
- confidence: 0.0-1.0 置信度

输出JSON格式：
{"score": 0.5, "confidence": 0.7, "direction": "positive", "reasoning": "...", "suggestions": [...]}"""
    
    def build_evaluation_prompt(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        param_deltas: Dict[str, float]
    ) -> str:
        before_text = ", ".join(f"{k}={v:.3f}" for k, v in params_before.items())
        after_text = ", ".join(f"{k}={v:.3f}" for k, v in params_after.items())
        delta_text = ", ".join(
            f"Δ{k}={v:+.3f}" 
            for k, v in param_deltas.items() 
            if abs(v) > 0.001
        )
        
        return f"""### 整体性能评估

**改动前**: {before_text}
**改动后**: {after_text}
**变化量**: {delta_text if delta_text else "无显著变化"}

请预估这次改动对fitness的影响，输出JSON格式评分。"""
    
    def _rule_based_evaluate(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        param_deltas: Dict[str, float]
    ) -> CriticScore:
        """基于规则的性能评估"""
        score = 0.0
        reasons = []
        
        # 评估体积变化趋势
        size_params = {"lm", "tm", "hs", "wslot", "hslot"}
        size_change = sum(param_deltas.get(k, 0) for k in size_params)
        
        if size_change < -0.1:
            score += 0.2
            reasons.append("整体尺寸减小，体积/质量可能改善")
        elif size_change > 0.1:
            score -= 0.1
            reasons.append("整体尺寸增大，体积/质量可能变差")
        
        # 评估关键性能参数
        # ta太小会影响磁路，太大浪费空间
        ta = params_after.get("ta", 0.6)
        if 0.5 <= ta <= 0.8:
            score += 0.1
            reasons.append(f"ta={ta:.2f}在合理范围")
        
        # dg过大会显著降低推力
        dg = params_after.get("dg", 0.5)
        if dg > 0.7:
            score -= 0.2
            reasons.append(f"dg={dg:.2f}偏大，可能降低推力")
        elif 0.3 <= dg <= 0.5:
            score += 0.1
            reasons.append(f"dg={dg:.2f}在较优范围")
        
        score = max(-1.0, min(1.0, score))
        direction = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        
        return CriticScore(
            critic_type=self.critic_type.value,
            score=score,
            confidence=0.5,
            direction=direction,
            reasoning="; ".join(reasons) if reasons else "无法从规则判断性能变化"
        )


class ConstraintCritic(BaseCritic):
    """约束评论家 - 快速约束预检"""
    
    def get_system_prompt(self) -> str:
        return """你是一个电磁执行器约束检查专家评论家。
你的任务是快速预检参数是否可能违反设计约束。

主要约束（从代码提取）：
1. 约束(11)：2*dg + tb - hs ≥ 0.1mm
2. 约束(21)：0.3mm < ta < 1.0mm
3. 约束(22)：tb ∈ [1.5*ta, 2.5*ta]，即tb_ratio在[1.5, 2.5]
4. 约束(25)(29)：hslot - tb ≥ 0.2mm
5. 约束(26)：wslot - wa ≥ 0.2mm（wa固定=2.0mm，所以wslot≥2.2mm）
6. 约束(27)：dg - twall ≥ 0.02mm
7. 约束(13)：s ≥ 1mm
8. 约束(14)(15)(16)：ws<4mm, ha<5mm, la≤6mm
9. 约束(18)：ls = lm - s > 0，即lm > s

请评估参数改动违反约束的风险：
- score: -1.0（很可能违规）到 +1.0（很安全）
- confidence: 0.0-1.0 置信度

输出JSON格式：
{"score": 0.5, "confidence": 0.8, "direction": "positive", "reasoning": "...", "suggestions": [...]}"""
    
    def build_evaluation_prompt(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        param_deltas: Dict[str, float]
    ) -> str:
        after_text = ", ".join(f"{k}={v:.3f}" for k, v in params_after.items())
        
        # 检查边界
        boundary_warnings = []
        for k, v in params_after.items():
            if k in DESIGN_BOUNDS:
                low, high = DESIGN_BOUNDS[k]
                if v < low * 1.05:
                    boundary_warnings.append(f"{k}={v:.3f} 接近下界 {low}")
                elif v > high * 0.95:
                    boundary_warnings.append(f"{k}={v:.3f} 接近上界 {high}")
        
        boundary_text = "\n".join(boundary_warnings) if boundary_warnings else "无边界警告"
        
        return f"""### 约束预检

**当前参数**: {after_text}

**边界检查**:
{boundary_text}

请评估约束违规风险，输出JSON格式评分。"""
    
    def _rule_based_evaluate(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        param_deltas: Dict[str, float]
    ) -> CriticScore:
        """基于规则的约束评估"""
        score = 1.0  # 从满分开始扣
        reasons = []
        suggestions = []
        
        # 检查边界
        for k, v in params_after.items():
            if k in DESIGN_BOUNDS:
                low, high = DESIGN_BOUNDS[k]
                if v < low:
                    score -= 0.5
                    reasons.append(f"{k}={v:.3f} < 下界{low}")
                    suggestions.append(f"增大 {k}")
                elif v > high:
                    score -= 0.5
                    reasons.append(f"{k}={v:.3f} > 上界{high}")
                    suggestions.append(f"减小 {k}")
                elif v < low * 1.1:
                    score -= 0.1
                    reasons.append(f"{k}接近下界")
                elif v > high * 0.9:
                    score -= 0.1
                    reasons.append(f"{k}接近上界")
        
        # 检查真实约束
        wslot = params_after.get("wslot", 2.2)
        wa = 2.0  # 固定值
        ta = params_after.get("ta", 0.5)
        tb_ratio = params_after.get("tb_ratio", 1.6)
        tb = tb_ratio * ta
        hslot = params_after.get("hslot", 1.0)
        dg = params_after.get("dg", 0.4)
        lm = params_after.get("lm", 3.0)
        s = params_after.get("s", 1.0)
        
        # 约束(26)：wslot - wa ≥ 0.2mm
        if wslot - wa < 0.2:
            score -= 0.4
            reasons.append(f"约束(26)风险：wslot-wa={wslot-wa:.2f}mm < 0.2mm")
            suggestions.append("增大 wslot（需 ≥ 2.2mm）")
        
        # 约束(22)：tb ∈ [1.5*ta, 2.5*ta]
        if tb < 1.5 * ta or tb > 2.5 * ta:
            score -= 0.4
            reasons.append(f"约束(22)风险：tb={tb:.2f}不在[{1.5*ta:.2f}, {2.5*ta:.2f}]")
            suggestions.append("调整 tb_ratio 到 [1.5, 2.5] 范围")
        
        # 约束(25)(29)：hslot - tb ≥ 0.2mm
        if hslot - tb < 0.2:
            score -= 0.3
            reasons.append(f"约束(25)风险：hslot-tb={hslot-tb:.2f}mm < 0.2mm")
            suggestions.append("增大 hslot 或减小 tb_ratio")
        
        # 约束(18)：lm > s
        if lm <= s:
            score -= 0.5
            reasons.append(f"约束(18)风险：lm={lm:.2f} ≤ s={s:.2f}")
            suggestions.append("增大 lm 或减小 s")
        
        score = max(-1.0, min(1.0, score))
        direction = "positive" if score > 0.5 else "negative" if score < 0 else "neutral"
        
        return CriticScore(
            critic_type=self.critic_type.value,
            score=score,
            confidence=0.8,  # 规则约束检查置信度较高
            direction=direction,
            reasoning="; ".join(reasons) if reasons else "约束检查通过",
            suggestions=suggestions
        )


class MagnitudeCritic(BaseCritic):
    """幅度评论家 - 评估参数改动幅度的合理性"""
    
    def get_system_prompt(self) -> str:
        return """你是一个优化步长专家评论家。
你的任务是评估参数改动幅度是否合理。

原则：
1. 改动过小（<1%）→ 收敛太慢，可能陷入局部最优
2. 改动过大（>20%）→ 跳跃太大，可能错过最优解
3. 理想改动幅度：3%-15%
4. 多个参数同时大幅改动 → 风险较高，难以判断因果

请评估改动幅度的合理性：
- score: -1.0（幅度不合理）到 +1.0（幅度合理）
- confidence: 0.0-1.0 置信度

输出JSON格式：
{"score": 0.5, "confidence": 0.8, "direction": "positive", "reasoning": "...", "suggestions": [...]}"""
    
    def build_evaluation_prompt(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        param_deltas: Dict[str, float]
    ) -> str:
        # 计算相对变化率
        relative_changes = []
        for k, delta in param_deltas.items():
            before = params_before.get(k, 0)
            if before != 0 and abs(delta) > 0.001:
                pct = abs(delta / before) * 100
                relative_changes.append(f"{k}: {delta:+.3f} ({pct:.1f}%)")
        
        change_text = "\n".join(relative_changes) if relative_changes else "无显著变化"
        
        return f"""### 改动幅度评估

**参数变化（绝对值及百分比）**:
{change_text}

请评估改动幅度的合理性，输出JSON格式评分。"""
    
    def _rule_based_evaluate(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        param_deltas: Dict[str, float]
    ) -> CriticScore:
        """基于规则的幅度评估"""
        score = 1.0
        reasons = []
        suggestions = []
        
        large_changes = 0
        tiny_changes = 0
        reasonable_changes = 0
        
        for k, delta in param_deltas.items():
            before = params_before.get(k, 0)
            if before == 0 or abs(delta) < 0.001:
                continue
            
            pct = abs(delta / before)
            
            if pct > 0.25:  # >25%
                large_changes += 1
                reasons.append(f"{k}改动过大({pct*100:.1f}%)")
                score -= 0.2
            elif pct < 0.01:  # <1%
                tiny_changes += 1
            elif 0.03 <= pct <= 0.15:  # 3%-15%
                reasonable_changes += 1
        
        if large_changes > 2:
            score -= 0.3
            suggestions.append("减小单步改动幅度，避免同时大幅调整多个参数")
        
        if reasonable_changes > 0:
            score = min(1.0, score + 0.1 * reasonable_changes)
        
        if tiny_changes > 3 and large_changes == 0:
            score -= 0.2
            reasons.append("改动过于保守")
            suggestions.append("可以适当增大探索步长")
        
        score = max(-1.0, min(1.0, score))
        direction = "positive" if score > 0.5 else "negative" if score < 0 else "neutral"
        
        return CriticScore(
            critic_type=self.critic_type.value,
            score=score,
            confidence=0.7,
            direction=direction,
            reasoning="; ".join(reasons) if reasons else "改动幅度合理",
            suggestions=suggestions
        )


# ============== 评论家集群管理器 ==============

class CriticEnsemble:
    """评论家集群管理器"""
    
    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None,
        models: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None,
        experience_dir: str = "critic_experience",
        enabled_critics: Optional[List[str]] = None,
        # ★批处理模式配置（TD(n)）
        enable_batch_mode: bool = False,
        batch_interval: int = 3
    ):
        """
        Args:
            base_url: LLM API base URL
            api_key: API key
            models: 各评论家使用的模型，格式如 {"magnetic": "claude-sonnet-4", ...}
            weights: 各评论家权重，格式如 {"magnetic": 0.35, ...}
            experience_dir: 经验库目录
            enabled_critics: 启用的评论家列表，None表示全部启用
            enable_batch_mode: 是否启用批处理模式（每 n 轮生成一次规则）
            batch_interval: 批处理间隔（默认 3 轮）
        """
        self.base_url = base_url
        self.api_key = api_key
        self.models = models or DEFAULT_CRITIC_MODELS
        self.weights = weights or DEFAULT_CRITIC_WEIGHTS
        self.experience_dir = experience_dir
        self.enabled_critics = enabled_critics or ["magnetic", "performance", "constraint", "magnitude"]
        
        # ★从稠密奖励聚合中排除的评论家（仍然会运行并提供建议/评分，但不计入 dense_reward）
        # 典型用法：将磁饱和作为“约束”处理（拉格朗日乘子），而不是作为稠密奖励惩罚项
        self.exclude_from_dense_reward: set[str] = set()
        
        # ★批处理模式配置
        self.enable_batch_mode = enable_batch_mode
        self.batch_interval = batch_interval
        
        # 确保经验库目录存在
        os.makedirs(experience_dir, exist_ok=True)
        
        # 初始化评论家
        self.critics: Dict[str, BaseCritic] = {}
        self._init_critics()
        
        # 集群经验记录
        self.ensemble_experiences: List[CriticExperience] = []
    
    def _init_critics(self):
        """初始化各评论家"""
        critic_classes = {
            "magnetic": MagneticCritic,
            "performance": PerformanceCritic,
            "constraint": ConstraintCritic,
            "magnitude": MagnitudeCritic,
        }
        
        for critic_name, critic_class in critic_classes.items():
            enabled = critic_name in self.enabled_critics
            config = CriticConfig(
                critic_type=CriticType(critic_name),
                model=self.models.get(critic_name, "gpt-4o-mini"),
                base_url=self.base_url,
                api_key=self.api_key,
                weight=self.weights.get(critic_name, 0.25),
                enabled=enabled,
                experience_path=os.path.join(self.experience_dir, f"{critic_name}_experience.json"),
                strategy_path=os.path.join(self.experience_dir, f"{critic_name}_strategy.json"),
                # ★批处理模式配置
                enable_batch_mode=self.enable_batch_mode,
                batch_interval=self.batch_interval,
            )
            self.critics[critic_name] = critic_class(config)
            
            status = "✓" if enabled else "✗"
            logger.info(f"[Critic] {status} {critic_name}: model={config.model}, weight={config.weight}")
    
    async def evaluate(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        additional_context: Optional[str] = None
    ) -> Tuple[float, Dict[str, CriticScore], List[str]]:
        """并行评估所有评论家
        
        Args:
            params_before: 改动前参数
            params_after: 改动后参数
            additional_context: 额外上下文
        
        Returns:
            (dense_reward, critic_scores, suggestions)
            - dense_reward: 加权稠密奖励 (-1.0 ~ +1.0)
            - critic_scores: 各评论家评分
            - suggestions: 汇总的改进建议
        """
        # 并行调用所有评论家
        tasks = []
        critic_names = []
        for name, critic in self.critics.items():
            if critic.enabled:
                tasks.append(critic.evaluate(params_before, params_after, additional_context))
                critic_names.append(name)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 汇总结果
        critic_scores: Dict[str, CriticScore] = {}
        weighted_sum = 0.0
        total_weight = 0.0
        all_suggestions = []
        
        for name, result in zip(critic_names, results):
            if isinstance(result, Exception):
                logger.warning(f"[Critic] {name} 评估异常: {result}")
                continue
            
            critic_scores[name] = result
            weight = self.weights.get(name, 0.25)
            
            # ★使用 TD 调整后的置信度（方案A）
            # 如果有 TD 调整后的置信度，使用它；否则用原始置信度
            confidence = result.td_adjusted_confidence if result.td_adjusted_confidence else result.confidence
            
            # ★可选：从 dense_reward 聚合中排除某些评论家（例如 magnetic）
            if name in self.exclude_from_dense_reward:
                effective_weight = 0.0
            else:
                effective_weight = weight * confidence
            weighted_sum += result.score * effective_weight
            total_weight += effective_weight
            
            # 收集建议
            if result.suggestions:
                all_suggestions.extend(result.suggestions)
        
        # 计算最终稠密奖励
        dense_reward = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # 记录到集群经验
        param_deltas = {
            k: params_after.get(k, 0) - params_before.get(k, 0)
            for k in set(params_before.keys()) | set(params_after.keys())
        }
        
        experience = CriticExperience(
            iteration=len(self.ensemble_experiences) + 1,
            params_before=params_before,
            params_after=params_after,
            param_deltas=param_deltas,
            critic_scores=critic_scores,
            dense_reward=dense_reward
        )
        self.ensemble_experiences.append(experience)
        
        return dense_reward, critic_scores, list(set(all_suggestions))
    
    def record_actual_result(
        self,
        params_before: Dict[str, float],
        params_after: Dict[str, float],
        actual_result: Dict,
        actual_reward: float
    ):
        """记录实际结果，用于校准所有评论家"""
        # 更新最近一条集群经验
        if self.ensemble_experiences:
            last_exp = self.ensemble_experiences[-1]
            last_exp.actual_result = actual_result
            last_exp.actual_reward = actual_reward
        
        # 通知各评论家记录
        for name, critic in self.critics.items():
            if name in (self.ensemble_experiences[-1].critic_scores if self.ensemble_experiences else {}):
                score = self.ensemble_experiences[-1].critic_scores.get(name)
                if score:
                    critic.record_actual_result(
                        params_before, params_after, score, actual_result, actual_reward
                    )
    
    def update_with_td_error(
        self,
        td_error: float,
        params_before: Dict[str, float],
        params_after: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        基于 TD 误差更新所有评论家（方案A + B）
        
        Args:
            td_error: TD 误差 δ = r + γV(s') - V(s)
            params_before: 改动前参数
            params_after: 改动后参数
        
        Returns:
            各评论家的更新结果
        """
        results = {}
        
        # 计算参数改动量
        param_deltas = {
            k: params_after.get(k, 0) - params_before.get(k, 0)
            for k in set(params_before.keys()) | set(params_after.keys())
        }
        
        # 获取最近一轮的预测评分
        if not self.ensemble_experiences:
            return results
        
        last_exp = self.ensemble_experiences[-1]
        
        # 更新每个评论家
        for name, critic in self.critics.items():
            if not critic.enabled:
                continue
            
            score = last_exp.critic_scores.get(name)
            if not score:
                continue
            
            prediction_score = score.score if isinstance(score, CriticScore) else score.get("score", 0)
            
            update_result = critic.update_with_td_error(
                td_error=td_error,
                prediction_score=prediction_score,
                param_deltas=param_deltas
            )
            
            results[name] = update_result
        
        return results
    
    def get_td_learning_status(self) -> Dict[str, Any]:
        """获取 TD 学习状态（用于前端日志输出）"""
        status = {
            "critics": {},
            "ensemble_accuracy": 0.0,
            "total_td_rules": 0
        }
        
        total_correct = 0
        total_predictions = 0
        
        for name, critic in self.critics.items():
            td_summary = critic.get_td_summary()
            status["critics"][name] = td_summary
            
            total_correct += td_summary.get("correct_predictions", 0)
            total_predictions += td_summary.get("total_predictions", 0)
            status["total_td_rules"] += td_summary.get("learned_rules_from_td", 0)
        
        if total_predictions > 0:
            status["ensemble_accuracy"] = total_correct / total_predictions
        
        return status
    
    def build_td_learning_log(self) -> str:
        """构建 TD 学习日志（用于前端显示）"""
        status = self.get_td_learning_status()
        
        lines = [
            "=" * 60,
            "[Critic] 📊 【评论家 TD 学习状态】",
            f"[Critic] 集群整体准确率: {status['ensemble_accuracy']:.1%}",
            f"[Critic] TD 学习规则总数: {status['total_td_rules']}",
            "[Critic] --- 各评论家详情 ---"
        ]
        
        for name, info in status["critics"].items():
            conf = info.get("dynamic_confidence", 0.5)
            acc = info.get("accuracy_rate", 0.5)
            total = info.get("total_predictions", 0)
            correct = info.get("correct_predictions", 0)
            td_stats = info.get("td_statistics_count", 0)
            
            # 置信度变化趋势
            if conf > 0.6:
                conf_emoji = "📈"
            elif conf < 0.4:
                conf_emoji = "📉"
            else:
                conf_emoji = "➡️"
            
            lines.append(
                f"[Critic] {conf_emoji} {name}: "
                f"置信度={conf:.2f} | 准确率={acc:.1%} ({correct}/{total}) | TD统计={td_stats}条"
            )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取评论家集群摘要"""
        summary = {
            "total_evaluations": len(self.ensemble_experiences),
            "critics": {}
        }
        
        for name, critic in self.critics.items():
            td_summary = critic.get_td_summary()
            summary["critics"][name] = {
                "model": critic.config.model,
                "weight": critic.weight,
                "enabled": critic.enabled,
                "experience_count": len(critic.experiences),
                "strategy_count": len(critic.strategies),
                # TD 学习信息
                "dynamic_confidence": td_summary.get("dynamic_confidence", 0.5),
                "prediction_accuracy": td_summary.get("accuracy_rate", 0.5),
                "td_statistics_count": td_summary.get("td_statistics_count", 0),
                "td_rules_count": td_summary.get("learned_rules_from_td", 0)
            }
        
        # 统计预测准确率（如果有实际结果）
        if self.ensemble_experiences:
            correct = 0
            total = 0
            for exp in self.ensemble_experiences:
                if exp.actual_reward is not None:
                    total += 1
                    # 方向正确即视为准确
                    if (exp.dense_reward > 0.1 and exp.actual_reward > 0) or \
                       (exp.dense_reward < -0.1 and exp.actual_reward < 0) or \
                       (abs(exp.dense_reward) <= 0.1 and abs(exp.actual_reward) <= 0.5):
                        correct += 1
            
            if total > 0:
                summary["prediction_accuracy"] = correct / total
        
        # TD 学习整体状态
        td_status = self.get_td_learning_status()
        summary["td_learning"] = {
            "ensemble_accuracy": td_status.get("ensemble_accuracy", 0),
            "total_td_rules": td_status.get("total_td_rules", 0)
        }
        
        return summary
    
    def build_dense_reward_prompt(
        self,
        dense_reward: float,
        critic_scores: Dict[str, CriticScore],
        suggestions: List[str]
    ) -> str:
        """构建稠密奖励提示词，用于注入主Agent"""
        parts = [f"### 评论家预评估（稠密奖励 = {dense_reward:+.3f}）"]
        
        for name, score in critic_scores.items():
            emoji = "✓" if score.score > 0.1 else "✗" if score.score < -0.1 else "→"
            parts.append(
                f"- {name}: {emoji} {score.score:+.2f} (置信度 {score.confidence:.0%}) | {score.reasoning[:50]}..."
            )
        
        if suggestions:
            parts.append("\n**改进建议**:")
            for s in suggestions[:3]:
                parts.append(f"  - {s}")
        
        return "\n".join(parts)

