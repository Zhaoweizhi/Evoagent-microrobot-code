"""
显式记忆架构模块

基于 Reflexion 论文的设计，将记忆显式分为：
1. ShortTermMemory (Trajectory) - 当前 episode 的轨迹
2. LongTermMemory (Experience) - 跨 episode 的经验

方案 B 增强：
3. Block 系统 - 有固定字符限制的内存块，硬性约束不爆炸
4. 统一 Context 组合 - 替代分散的 build_xxx_context

这种显式分离让架构更清晰，便于理解和维护。
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Callable
from loguru import logger


# ============== Block 系统（方案 B）==============

# 默认 Block 字符限制
DEFAULT_BLOCK_LIMITS = {
    "persona": 1500,        # 任务描述/约束/目标
    "best_result": 800,     # 当前最佳结果
    "active_rules": 1200,   # 当前激活的规则
    "strategy": 1000,       # 策略提示
    "feedback": 600,        # 人类反馈
    "trajectory": 800,      # 短期轨迹
    "experience": 800,      # RAG/经验检索
}

# Token 预算分配（总计约 6000 tokens）
TOKEN_BUDGET = {
    "core_blocks": 3500,    # Block 系统
    "summary": 1000,        # 历史摘要
    "conversation": 1500,   # 最近对话
}

# 字符到 token 的估算比例（中文约 1.5-2 字符/token）
CHARS_PER_TOKEN_EST = 1.8


@dataclass
class Block:
    """内存块 - 有固定字符限制的结构化内存单元"""
    
    name: str
    label: str  # 显示标签
    value: str = ""
    limit: int = 2000
    description: str = ""
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if len(self.value) > self.limit:
            self.value = self.value[:self.limit]
    
    @property
    def current_length(self) -> int:
        return len(self.value)
    
    @property
    def remaining_space(self) -> int:
        return max(0, self.limit - len(self.value))
    
    @property
    def usage_ratio(self) -> float:
        return len(self.value) / self.limit if self.limit > 0 else 0
    
    def update(self, new_value: str, truncate: bool = True) -> bool:
        """更新 Block 内容"""
        if len(new_value) > self.limit:
            if truncate:
                new_value = new_value[:self.limit]
            else:
                return False
        self.value = new_value
        self.last_updated = time.time()
        return True
    
    def clear(self):
        self.value = ""
        self.last_updated = time.time()
    
    def to_prompt(self) -> str:
        """转换为 prompt 格式"""
        if not self.value:
            return ""
        return f"### {self.label}\n{self.value}"


class CoreBlocks:
    """核心记忆块管理器"""
    
    def __init__(self):
        self.blocks: Dict[str, Block] = {}
        self._init_default_blocks()
    
    def _init_default_blocks(self):
        """初始化默认 Block"""
        defaults = [
            ("persona", "任务与约束", DEFAULT_BLOCK_LIMITS["persona"]),
            ("best_result", "当前最佳", DEFAULT_BLOCK_LIMITS["best_result"]),
            ("active_rules", "激活规则", DEFAULT_BLOCK_LIMITS["active_rules"]),
            ("strategy", "策略指导", DEFAULT_BLOCK_LIMITS["strategy"]),
            ("feedback", "反馈记录", DEFAULT_BLOCK_LIMITS["feedback"]),
            ("trajectory", "轨迹摘要", DEFAULT_BLOCK_LIMITS["trajectory"]),
            ("experience", "相关经验", DEFAULT_BLOCK_LIMITS["experience"]),
        ]
        for name, label, limit in defaults:
            self.blocks[name] = Block(name=name, label=label, limit=limit)
    
    def get(self, name: str) -> Optional[Block]:
        return self.blocks.get(name)
    
    def update(self, name: str, value: str, truncate: bool = True) -> bool:
        if name not in self.blocks:
            return False
        return self.blocks[name].update(value, truncate)
    
    def get_total_chars(self) -> int:
        return sum(b.current_length for b in self.blocks.values())
    
    def get_total_tokens(self) -> int:
        return int(self.get_total_chars() / CHARS_PER_TOKEN_EST)
    
    def compose(self, include_empty: bool = False) -> str:
        """组合所有 Block 为单一 prompt"""
        parts = []
        for block in self.blocks.values():
            if block.value or include_empty:
                prompt = block.to_prompt()
                if prompt:
                    parts.append(prompt)
        if not parts:
            return ""
        return "## 核心记忆\n\n" + "\n\n".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chars": self.get_total_chars(),
            "total_tokens": self.get_total_tokens(),
            "blocks": {
                name: {"chars": b.current_length, "limit": b.limit, "usage": f"{b.usage_ratio:.1%}"}
                for name, b in self.blocks.items()
            }
        }


class SummaryMemory:
    """摘要记忆 - 管理历史对话的压缩摘要"""
    
    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars
        self.summary: str = ""
        self.summarized_rounds: int = 0
        self.last_summary_time: float = 0
    
    def update(self, new_summary: str, rounds_covered: int):
        if len(new_summary) > self.max_chars:
            new_summary = new_summary[:self.max_chars]
        self.summary = new_summary
        self.summarized_rounds = rounds_covered
        self.last_summary_time = time.time()
    
    def to_prompt(self) -> str:
        if not self.summary:
            return ""
        return f"## 历史摘要（前 {self.summarized_rounds} 轮）\n\n{self.summary}"
    
    @property
    def current_length(self) -> int:
        return len(self.summary)
    
    def get_tokens(self) -> int:
        return int(len(self.summary) / CHARS_PER_TOKEN_EST)


# ==================== 短期记忆 ====================

@dataclass
class TrajectoryStep:
    """轨迹中的单步记录"""
    round: int
    params: Dict[str, float]
    fitness: Optional[float]
    reward: float
    success: bool
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    # 可选的额外信息
    critic_feedback: Optional[str] = None
    llm_reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ShortTermMemory:
    """
    短期记忆（Trajectory）
    
    存储当前 episode 的轨迹，包括：
    - 每轮的参数、fitness、reward
    - LLM 对话上下文
    - 当前/上一轮状态
    - 探索策略状态（每轮更新）
    
    特点：
    - 大部分不持久化，episode 结束后清空
    - 探索策略状态可选择性持久化
    - 用于 Actor 的即时决策
    """
    
    # 参数可行域（用于探索策略）
    PARAM_BOUNDS = {
        "lm": (1.0, 6.0),
        "tm": (0.3, 0.5),
        "ta": (0.35, 0.75),
        "dg": (0.30, 0.65),
        "hs": (1.2, 2.2),
        "wslot": (2.0, 2.8),
        "hslot": (0.8, 1.3),
        "s": (0.8, 1.2),
        "wa": (2.0, 2.0),
    }
    
    def __init__(
        self, 
        max_trajectory_length: int = 100,
        initial_epsilon: float = 0.3,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.98
    ):
        self.max_trajectory_length = max_trajectory_length
        
        # 轨迹：当前 episode 的完整历史
        self.trajectory: List[TrajectoryStep] = []
        
        # LLM 对话上下文
        self.messages: List[Dict[str, Any]] = []
        
        # 状态缓存
        self.current_state: Optional[Dict[str, float]] = None
        self.previous_state: Optional[Dict[str, float]] = None
        
        # 最近结果（用于快速判断趋势）
        self.recent_successes: List[bool] = []
        self.recent_fitness: List[float] = []
        
        # ========== 探索策略状态（从 strategy_state 移入）==========
        # 探索率（ε-greedy）
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        # 探索策略权重
        self.exploration_weights: Dict[str, float] = {
            "random": 0.15,
            "directed": 0.30,
            "perturbation": 0.25,
            "counterfactual": 0.10,
            "sensitivity": 0.10,
            "avoid_failure": 0.10
        }
        
        # 上一轮的调整方向（用于反事实探索）
        self.last_direction: Dict[str, float] = {}
        
        # 参数历史（用于即时敏感性分析）
        self.param_history: List[Dict[str, Any]] = []
        
        # 动态提示词片段（每轮更新）
        self.prompt_additions: List[str] = []
        
        # Episode 元信息
        self.episode_start_time: float = time.time()
        self.episode_id: str = f"ep_{int(time.time())}"
        self.iteration: int = 0
    
    def add_step(
        self,
        round: int,
        params: Dict[str, float],
        fitness: Optional[float],
        reward: float,
        success: bool,
        errors: List[str] = None,
        critic_feedback: str = None,
        llm_reasoning: str = None
    ) -> TrajectoryStep:
        """添加一步到轨迹，并更新探索策略状态"""
        step = TrajectoryStep(
            round=round,
            params=params.copy() if params else {},
            fitness=fitness,
            reward=reward,
            success=success,
            errors=errors or [],
            critic_feedback=critic_feedback,
            llm_reasoning=llm_reasoning
        )
        
        self.trajectory.append(step)
        self.iteration += 1
        
        # 更新最近结果
        self.recent_successes.append(success)
        if len(self.recent_successes) > 20:
            self.recent_successes.pop(0)
        
        if fitness is not None:
            self.recent_fitness.append(fitness)
            if len(self.recent_fitness) > 20:
                self.recent_fitness.pop(0)
        
        # 更新调整方向（用于反事实探索）
        if self.current_state and params:
            self.last_direction = {}
            for param in self.PARAM_BOUNDS:
                if param in self.current_state and param in params:
                    old_val = self.current_state.get(param)
                    new_val = params.get(param)
                    if old_val is not None and new_val is not None:
                        low, high = self.PARAM_BOUNDS[param]
                        if high > low:
                            self.last_direction[param] = (new_val - old_val) / (high - low)
        
        # 更新参数历史（用于敏感性分析）
        self.param_history.append({
            "old": self.current_state.copy() if self.current_state else {},
            "new": params.copy() if params else {},
            "fitness": fitness,
            "success": success
        })
        if len(self.param_history) > 30:
            self.param_history.pop(0)
        
        # 更新状态
        self.previous_state = self.current_state
        self.current_state = params.copy() if params else None
        
        # 更新探索率
        self._update_epsilon(success)
        
        # 更新动态提示词
        self._update_prompt_additions()
        
        # 限制轨迹长度
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
        
        return step
    
    # ========== 探索策略方法 ==========
    
    def should_explore(self) -> bool:
        """决定是否进行探索（ε-greedy）"""
        import random
        return random.random() < self.epsilon
    
    def _update_epsilon(self, last_success: bool):
        """自适应更新探索率"""
        self.epsilon *= self.epsilon_decay
        
        if self.recent_successes:
            recent_success_rate = sum(self.recent_successes) / len(self.recent_successes)
            if recent_success_rate > 0.6:
                self.epsilon -= 0.02  # 成功率高，减少探索
            elif recent_success_rate < 0.3:
                self.epsilon += 0.01  # 成功率低，增加探索
        
        self.epsilon = max(self.min_epsilon, min(0.5, self.epsilon))
    
    def _update_prompt_additions(self):
        """生成动态提示词片段"""
        self.prompt_additions = []
        
        # 基于趋势的提示
        trend = self.get_fitness_trend()
        if trend == "degrading":
            self.prompt_additions.append("⚠️ 最近 fitness 持续下降，建议尝试新方向或回退到之前的好配置")
        elif trend == "improving":
            self.prompt_additions.append("💡 当前方向有效，可以继续沿此方向小步调整")
        
        # 基于成功率的提示
        success_rate = self.get_recent_success_rate()
        if success_rate < 0.3:
            self.prompt_additions.append("⚠️ 近期成功率较低，请检查是否接近约束边界")
    
    def get_exploration_info(self) -> Dict[str, Any]:
        """获取探索策略信息"""
        return {
            "epsilon": round(self.epsilon, 4),
            "iteration": self.iteration,
            "recent_success_rate": self.get_recent_success_rate(),
            "trend": self.get_fitness_trend(),
            "exploration_weights": self.exploration_weights
        }
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """添加 LLM 消息"""
        self.messages.append(message)
    
    def get_recent_trajectory(self, k: int = 10) -> List[TrajectoryStep]:
        """获取最近 k 步轨迹"""
        return self.trajectory[-k:] if len(self.trajectory) > k else self.trajectory
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """获取轨迹摘要"""
        if not self.trajectory:
            return {"length": 0}
        
        successes = [s for s in self.trajectory if s.success]
        fitness_values = [s.fitness for s in self.trajectory if s.fitness is not None]
        
        return {
            "length": len(self.trajectory),
            "success_count": len(successes),
            "failure_count": len(self.trajectory) - len(successes),
            "success_rate": len(successes) / len(self.trajectory) if self.trajectory else 0,
            "best_fitness": min(fitness_values) if fitness_values else None,
            "worst_fitness": max(fitness_values) if fitness_values else None,
            "latest_fitness": fitness_values[-1] if fitness_values else None,
            "episode_duration": time.time() - self.episode_start_time
        }
    
    def get_recent_success_rate(self) -> float:
        """获取最近成功率"""
        if not self.recent_successes:
            return 0.0
        return sum(self.recent_successes) / len(self.recent_successes)
    
    def get_fitness_trend(self) -> str:
        """判断 fitness 趋势"""
        if len(self.recent_fitness) < 3:
            return "unknown"
        
        recent = self.recent_fitness[-5:]
        if len(recent) < 2:
            return "unknown"
        
        # 计算趋势
        improvements = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        if improvements >= len(recent) - 1:
            return "improving"
        elif improvements == 0:
            return "degrading"
        else:
            return "fluctuating"
    
    def build_trajectory_context(self, max_steps: int = 5) -> str:
        """构建轨迹上下文（用于注入 prompt）"""
        if not self.trajectory:
            return ""
        
        lines = ["【当前轨迹摘要】"]
        summary = self.get_trajectory_summary()
        lines.append(f"- 已执行 {summary['length']} 轮，成功率 {summary['success_rate']:.1%}")
        
        if summary['best_fitness'] is not None:
            lines.append(f"- 最佳 fitness: {summary['best_fitness']:.4f}")
        
        trend = self.get_fitness_trend()
        trend_emoji = {"improving": "📈", "degrading": "📉", "fluctuating": "〰️"}.get(trend, "❓")
        lines.append(f"- 趋势: {trend_emoji} {trend}")
        
        # 最近几步
        recent = self.get_recent_trajectory(max_steps)
        if recent:
            lines.append(f"\n【最近 {len(recent)} 轮】")
            for step in recent:
                status = "✅" if step.success else "❌"
                fitness_str = f"fitness={step.fitness:.4f}" if step.fitness else "N/A"
                lines.append(f"- 第{step.round}轮 {status} {fitness_str}")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """重置短期记忆（新 episode 开始时调用）"""
        self.trajectory = []
        self.messages = []
        self.current_state = None
        self.previous_state = None
        self.recent_successes = []
        self.recent_fitness = []
        self.episode_start_time = time.time()
        self.episode_id = f"ep_{int(time.time())}"
        logger.info(f"[ShortTermMemory] 已重置，新 episode: {self.episode_id}")


# ==================== 长期记忆 ====================

class LongTermMemory:
    """
    长期记忆（Experience）
    
    基于 Reflexion 架构，长期记忆只在触发时才进行深度思考：
    - ReflectionManager: 反思记录、规则、模式、敏感性（核心）
    - FeedbackHandler: 人工反馈
    - StrategyManager: 保留用于兼容（元学习等）
    
    特点：
    - 持久化存储，跨 episode 保留
    - 触发时才进行深度分析（不是每轮）
    - 存储抽象知识而非原始数据
    """
    
    # 参数可行域（用于敏感性分析）
    PARAM_BOUNDS = {
        "lm": (1.0, 6.0),
        "tm": (0.3, 0.5),
        "ta": (0.35, 0.75),
        "dg": (0.30, 0.65),
        "hs": (1.2, 2.2),
        "wslot": (2.0, 2.8),
        "hslot": (0.8, 1.3),
        "s": (0.8, 1.2),
        "wa": (2.0, 2.0),
        "n1": (50, 80),
        "n2": (2, 10),
    }
    
    def __init__(
        self,
        reflection_manager=None,
        feedback_handler=None,
        strategy_manager=None,  # 保留用于兼容
        storage_dir: str = "."
    ):
        self.storage_dir = storage_dir
        
        # 延迟导入，避免循环依赖
        from .reflection import ReflectionManager
        from .feedback import FeedbackHandler
        
        # 核心：反思管理器（包含 memory_stream）
        self.reflection_manager = reflection_manager or ReflectionManager(
            storage_path=os.path.join(storage_dir, "memory_stream.jsonl")
        )
        
        # 人工反馈
        self.feedback_handler = feedback_handler or FeedbackHandler(
            storage_path=os.path.join(storage_dir, "feedback_storage.json")
        )
        
        # 保留 strategy_manager 用于兼容（元学习等）
        self.strategy_manager = strategy_manager
        
        # 缓存：成功/失败模式（从 memory_stream 读取）
        self._cached_success_patterns: List[Dict] = []
        self._cached_failure_patterns: List[Dict] = []
        self._cached_sensitivity: Dict[str, float] = {}
        
        logger.info("[LongTermMemory] 长期记忆已初始化（Reflexion 架构）")
    
    def record_round(
        self,
        round: int,
        params: Dict[str, float],
        fitness: Optional[float],
        success: bool,
        errors: List[str] = None
    ) -> Optional[str]:
        """
        记录一轮结果到长期记忆
        
        这是统一的入口，会：
        1. 记录观察到 memory_stream
        2. 检查是否触发反思
        3. 触发时进行深度分析（模式、敏感性）
        
        Returns:
            如果触发反思，返回反思内容；否则返回 None
        """
        return self.reflection_manager.record_round(
            round=round,
            params=params,
            fitness=fitness,
            success=success,
            errors=errors
        )
    
    def analyze_and_store_patterns(
        self,
        round: int,
        param_history: List[Dict[str, Any]],
        best_fitness: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        触发时才执行：分析并存储成功/失败模式
        
        这是"深度思考"的一部分，只在反思触发时调用
        """
        result = {"success_patterns": 0, "failure_patterns": 0}
        
        if len(param_history) < 5:
            return result
        
        # 分析成功模式
        successes = [h for h in param_history if h.get("success") and h.get("fitness") is not None]
        if successes:
            # 找出最佳的几个
            best_successes = sorted(successes, key=lambda x: x["fitness"])[:3]
            for h in best_successes:
                key_factors = self._identify_key_factors(h["new"])
                self.reflection_manager.memory_stream.write_pattern(
                    round=round,
                    pattern_type="success",
                    params=h["new"],
                    fitness=h["fitness"],
                    key_factors=key_factors
                )
                result["success_patterns"] += 1
        
        # 分析失败模式
        failures = [h for h in param_history if not h.get("success")]
        if failures:
            # 只记录最近的几个失败
            recent_failures = failures[-3:]
            for h in recent_failures:
                self.reflection_manager.memory_stream.write_pattern(
                    round=round,
                    pattern_type="failure",
                    params=h["new"],
                    errors=h.get("errors", [])
                )
                result["failure_patterns"] += 1
        
        return result
    
    def analyze_and_store_sensitivity(
        self,
        round: int,
        param_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        触发时才执行：分析并存储参数敏感性
        
        这是"深度思考"的一部分，只在反思触发时调用
        """
        if len(param_history) < 10:
            return {}
        
        import math
        sensitivity = {}
        
        # 收集有效的 fitness 记录
        valid_records = [h for h in param_history if h.get("fitness") is not None]
        if len(valid_records) < 5:
            return {}
        
        fitness_values = [h["fitness"] for h in valid_records]
        
        for param in self.PARAM_BOUNDS:
            param_values = []
            for h in valid_records:
                val = h.get("new", {}).get(param)
                if val is not None:
                    param_values.append(val)
            
            if len(param_values) < 5 or len(param_values) != len(fitness_values):
                continue
            
            # 计算相关系数
            n = len(param_values)
            mean_p = sum(param_values) / n
            mean_f = sum(fitness_values) / n
            
            num = sum((param_values[i] - mean_p) * (fitness_values[i] - mean_f) for i in range(n))
            den_p = sum((p - mean_p) ** 2 for p in param_values)
            den_f = sum((f - mean_f) ** 2 for f in fitness_values)
            
            if den_p > 0 and den_f > 0:
                correlation = abs(num / math.sqrt(den_p * den_f))
                sensitivity[param] = round(correlation, 3)
        
        # 归一化
        if sensitivity:
            max_s = max(sensitivity.values())
            if max_s > 0:
                sensitivity = {k: round(v / max_s, 3) for k, v in sensitivity.items()}
            
            # 存储到 memory_stream
            self.reflection_manager.memory_stream.write_sensitivity(
                round=round,
                sensitivity_data=sensitivity
            )
        
        self._cached_sensitivity = sensitivity
        return sensitivity
    
    def _identify_key_factors(self, params: Dict[str, float]) -> List[str]:
        """识别关键因素"""
        factors = []
        for param, value in params.items():
            if param not in self.PARAM_BOUNDS:
                continue
            low, high = self.PARAM_BOUNDS[param]
            if high == low:
                continue
            rel_pos = (value - low) / (high - low)
            if rel_pos > 0.8:
                factors.append(f"{param} 较高 ({value:.2f})")
            elif rel_pos < 0.2:
                factors.append(f"{param} 较低 ({value:.2f})")
        return factors[:5]
    
    def build_reflection_context(
        self,
        max_reflections: int = 3,
        max_rules: int = 5
    ) -> str:
        """构建反思上下文（长期记忆的核心）"""
        return self.reflection_manager.build_reflection_prompt(
            max_reflections=max_reflections,
            max_rules=max_rules
        )
    
    def build_patterns_context(self) -> str:
        """构建模式上下文（从 memory_stream 读取）"""
        lines = []
        
        # 读取成功模式
        all_events = self.reflection_manager.memory_stream.read_all()
        patterns = [e for e in all_events if e.event_type == "pattern"]
        
        success_patterns = [p for p in patterns if p.metadata.get("pattern_type") == "success"]
        failure_patterns = [p for p in patterns if p.metadata.get("pattern_type") == "failure"]
        
        if success_patterns:
            lines.append("【成功模式】")
            for p in success_patterns[-3:]:
                lines.append(f"  - {p.content}")
        
        if failure_patterns:
            lines.append("【失败模式】")
            for p in failure_patterns[-3:]:
                lines.append(f"  - {p.content}")
        
        return "\n".join(lines) if lines else ""
    
    def build_sensitivity_context(self) -> str:
        """构建敏感性上下文（从 memory_stream 读取）"""
        all_events = self.reflection_manager.memory_stream.read_all()
        sens_events = [e for e in all_events if e.event_type == "sensitivity"]
        
        if not sens_events:
            return ""
        
        latest = sens_events[-1]
        return f"【参数敏感性】\n  {latest.content}"
    
    def build_full_context(self) -> str:
        """构建完整的长期记忆上下文"""
        parts = []
        
        # 1. 反思和规则（核心）
        reflection_ctx = self.build_reflection_context()
        if reflection_ctx:
            parts.append(reflection_ctx)
        
        # 2. 模式
        patterns_ctx = self.build_patterns_context()
        if patterns_ctx:
            parts.append(patterns_ctx)
        
        # 3. 敏感性
        sens_ctx = self.build_sensitivity_context()
        if sens_ctx:
            parts.append(sens_ctx)
        
        # 4. 兼容：如果有 strategy_manager，也获取其上下文
        if self.strategy_manager:
            try:
                strategy_ctx = self.strategy_manager.build_strategy_prompt()
                if strategy_ctx:
                    parts.append(strategy_ctx)
            except:
                pass
        
        return "\n\n".join(parts) if parts else ""
    
    def get_summary(self) -> Dict[str, Any]:
        """获取长期记忆摘要"""
        reflection_stats = self.reflection_manager.get_stats()
        
        # 统计 memory_stream 中的模式数量
        all_events = self.reflection_manager.memory_stream.read_all()
        patterns = [e for e in all_events if e.event_type == "pattern"]
        rules = [e for e in all_events if e.event_type == "rule"]
        reflections = [e for e in all_events if e.event_type == "reflection"]
        
        return {
            "memory_stream": {
                "total_events": len(all_events),
                "reflections": len(reflections),
                "rules": len(rules),
                "patterns": len(patterns),
            },
            "cached_sensitivity": self._cached_sensitivity,
            "feedback": {
                "total": len(self.feedback_handler.feedbacks) if hasattr(self.feedback_handler, 'feedbacks') else 0
            }
        }
    
    def save_all(self) -> None:
        """保存所有长期记忆"""
        # memory_stream 是自动保存的（append 模式）
        # 如果有 strategy_manager（兼容模式），也保存
        if self.strategy_manager:
            try:
                self.strategy_manager._save()
            except:
                pass
        logger.info("[LongTermMemory] 所有长期记忆已保存")
    
    def clear_all(self) -> None:
        """清空所有长期记忆（谨慎使用）"""
        # 清空文件
        files_to_clear = [
            os.path.join(self.storage_dir, "experience_buffer.json"),
            os.path.join(self.storage_dir, "strategy_state.json"),
            os.path.join(self.storage_dir, "memory_stream.jsonl"),
            os.path.join(self.storage_dir, "meta_knowledge.json"),
        ]
        
        for f in files_to_clear:
            if os.path.exists(f):
                os.remove(f)
        
        # 重新初始化
        self.__init__(storage_dir=self.storage_dir)
        logger.warning("[LongTermMemory] 所有长期记忆已清空！")


# ==================== 统一记忆管理器 ====================

class MemoryManager:
    """
    统一记忆管理器（方案 B 增强版）
    
    整合短期记忆、长期记忆和 Block 系统，提供统一的 context 组合接口。
    
    核心改进（解决 Memory 爆炸）：
    1. Block 系统：固定字符限制，硬性约束不爆炸
    2. 统一 compose_context()：替代分散的 build_xxx_context
    3. Token 预算：明确分配每类记忆的配额
    4. 摘要记忆：LLM 压缩历史而非简单截断
    """
    
    def __init__(
        self,
        storage_dir: str = ".",
        max_trajectory_length: int = 100,
        summarize_fn: Optional[Callable[[str], str]] = None
    ):
        self.storage_dir = storage_dir
        self.summarize_fn = summarize_fn
        
        # 短期记忆（当前 episode）
        self.short_term = ShortTermMemory(
            max_trajectory_length=max_trajectory_length
        )
        
        # 长期记忆（跨 episode）
        self.long_term = LongTermMemory(
            storage_dir=storage_dir
        )
        
        # ★ 方案 B：Block 系统
        self.core_blocks = CoreBlocks()
        
        # ★ 方案 B：摘要记忆
        self.summary_memory = SummaryMemory()
        
        logger.info("[MemoryManager] 记忆管理器已初始化（方案 B 增强版）")
        logger.info(f"  - 短期记忆: Trajectory (max {max_trajectory_length} steps)")
        logger.info(f"  - 长期记忆: Experience, Strategy, Reflection, Feedback")
        logger.info(f"  - Block 系统: {len(self.core_blocks.blocks)} 个内存块")
    
    def record_step(
        self,
        round: int,
        params: Dict[str, float],
        fitness: Optional[float],
        reward: float,
        success: bool,
        errors: List[str] = None,
        action: Dict[str, float] = None,
        critic_feedback: str = None,
        llm_reasoning: str = None,
        best_fitness: Optional[float] = None
    ) -> Tuple[TrajectoryStep, Optional[str]]:
        """
        记录一步（同时更新短期和长期记忆）
        
        短期记忆：每轮都更新（即时反馈）
        长期记忆：只在触发时进行深度分析
        
        Returns:
            (trajectory_step, reflection_if_triggered)
        """
        # 1. 更新短期记忆（每轮都更新）
        step = self.short_term.add_step(
            round=round,
            params=params,
            fitness=fitness,
            reward=reward,
            success=success,
            errors=errors,
            critic_feedback=critic_feedback,
            llm_reasoning=llm_reasoning
        )
        
        # 2. 更新长期记忆 - 记录观察并检查是否触发反思
        reflection = self.long_term.record_round(
            round=round,
            params=params or {},
            fitness=fitness,
            success=success,
            errors=errors
        )
        
        return step, reflection
    
    def build_actor_context(self) -> str:
        """
        构建 Actor 的完整上下文
        
        整合短期记忆（轨迹）和长期记忆（反思、模式、敏感性）
        """
        parts = []
        
        # 1. 短期记忆：当前轨迹 + 即时反馈
        trajectory_ctx = self.short_term.build_trajectory_context()
        if trajectory_ctx:
            parts.append(trajectory_ctx)
        
        # 短期记忆的动态提示词
        if self.short_term.prompt_additions:
            parts.append("【即时反馈】")
            parts.extend(self.short_term.prompt_additions)
        
        # 2. 长期记忆：反思、模式、敏感性
        long_term_ctx = self.long_term.build_full_context()
        if long_term_ctx:
            parts.append(long_term_ctx)
        
        return "\n\n".join(parts) if parts else ""
    
    # ========== 方案 B：Block 管理方法 ==========
    
    def update_persona(self, content: str) -> bool:
        """更新任务描述/约束 Block"""
        return self.core_blocks.update("persona", content)
    
    def update_best_result(self, fitness: float, params: Dict[str, float], 
                           extra_info: Optional[str] = None) -> bool:
        """更新当前最佳结果 Block"""
        param_text = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
            for k, v in params.items()
        )
        content = f"最佳 fitness: {fitness:.6f}\n参数: {param_text}"
        if extra_info:
            content += f"\n{extra_info}"
        return self.core_blocks.update("best_result", content)
    
    def update_active_rules(self, rules: List[str], max_rules: int = 5) -> bool:
        """更新激活的规则 Block"""
        selected = rules[:max_rules]
        content = "\n".join(f"- {rule}" for rule in selected)
        return self.core_blocks.update("active_rules", content)
    
    def update_strategy(self, strategy_text: str) -> bool:
        """更新策略提示 Block"""
        return self.core_blocks.update("strategy", strategy_text)
    
    def update_feedback(self, feedbacks: List[str], max_feedbacks: int = 3) -> bool:
        """更新人类反馈 Block"""
        selected = feedbacks[:max_feedbacks]
        content = "\n".join(f"- {fb}" for fb in selected)
        return self.core_blocks.update("feedback", content)
    
    def update_trajectory_block(self) -> bool:
        """从短期记忆更新轨迹 Block"""
        trajectory_ctx = self.short_term.build_trajectory_context(max_steps=3)
        return self.core_blocks.update("trajectory", trajectory_ctx)
    
    def update_experience(self, experience_text: str) -> bool:
        """更新经验/RAG Block"""
        return self.core_blocks.update("experience", experience_text)
    
    def update_summary(self, summary_text: str, rounds: int):
        """更新历史摘要"""
        self.summary_memory.update(summary_text, rounds)
    
    # ========== 方案 B：统一 Context 组合 ==========
    
    def compose_context(self, include_summary: bool = True) -> str:
        """
        统一的 Context 组合方法（方案 B 核心）
        
        替代原来分散的 build_xxx_context 方法，
        通过 Block 系统硬性约束总 token 数。
        
        Returns:
            组合后的完整上下文字符串
        """
        parts = []
        
        # 1. Core Blocks（有字符限制的内存块）
        core_prompt = self.core_blocks.compose()
        if core_prompt:
            parts.append(core_prompt)
        
        # 2. Summary Memory（历史摘要）
        if include_summary:
            summary_prompt = self.summary_memory.to_prompt()
            if summary_prompt:
                parts.append(summary_prompt)
        
        return "\n\n---\n\n".join(parts) if parts else ""
    
    def compose_as_system_message(self, **kwargs) -> Optional[Dict[str, str]]:
        """组合为单一 system message"""
        content = self.compose_context(**kwargs)
        if not content:
            return None
        return {"role": "system", "content": content}
    
    def get_total_tokens(self) -> int:
        """获取当前总 token 数"""
        core_tokens = self.core_blocks.get_total_tokens()
        summary_tokens = self.summary_memory.get_tokens()
        return core_tokens + summary_tokens
    
    def get_block_stats(self) -> Dict[str, Any]:
        """获取 Block 统计信息"""
        return {
            "total_tokens": self.get_total_tokens(),
            "budget": TOKEN_BUDGET,
            "core_blocks": self.core_blocks.get_stats(),
            "summary": {
                "chars": self.summary_memory.current_length,
                "tokens": self.summary_memory.get_tokens(),
                "rounds": self.summary_memory.summarized_rounds
            }
        }
    
    def log_block_stats(self, log_fn: Optional[Callable] = None):
        """输出 Block 统计信息"""
        stats = self.get_block_stats()
        log = log_fn or logger.info
        
        log(f"[MemoryManager] 总 Token: ~{stats['total_tokens']}")
        for name, info in stats['core_blocks']['blocks'].items():
            if info['chars'] > 0:
                log(f"  - {name}: {info['chars']} chars ({info['usage']})")
        if stats['summary']['chars'] > 0:
            log(f"  - summary: {stats['summary']['chars']} chars ({stats['summary']['rounds']} rounds)")
    
    def new_episode(self) -> None:
        """开始新 episode（重置短期记忆，保留长期记忆）"""
        # 保存长期记忆
        self.long_term.save_all()
        
        # 重置短期记忆
        self.short_term.reset()
        
        logger.info("[MemoryManager] 新 episode 开始，短期记忆已重置")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        return {
            "short_term": self.short_term.get_trajectory_summary(),
            "long_term": self.long_term.get_summary()
        }
    
    # ========== 便捷属性 ==========
    
    @property
    def current_state(self) -> Optional[Dict[str, float]]:
        return self.short_term.current_state
    
    @current_state.setter
    def current_state(self, value: Dict[str, float]):
        self.short_term.current_state = value
    
    @property
    def previous_state(self) -> Optional[Dict[str, float]]:
        return self.short_term.previous_state
    
    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self.short_term.messages
    
    @property
    def trajectory(self) -> List[TrajectoryStep]:
        return self.short_term.trajectory
    
    @property
    def epsilon(self) -> float:
        return self.short_term.epsilon
    
    @property
    def iteration(self) -> int:
        return self.short_term.iteration
    
    @property
    def strategy_manager(self):
        """兼容属性：返回 long_term 中的 strategy_manager（如果有）"""
        return self.long_term.strategy_manager
    
    @property
    def reflection_manager(self):
        return self.long_term.reflection_manager
    
    @property
    def memory_stream(self):
        return self.long_term.reflection_manager.memory_stream


# ==================== 便捷函数 ====================

def create_memory_manager(
    storage_dir: str = ".",
    max_trajectory_length: int = 100
) -> MemoryManager:
    """创建记忆管理器的便捷函数"""
    return MemoryManager(
        storage_dir=storage_dir,
        max_trajectory_length=max_trajectory_length
    )
