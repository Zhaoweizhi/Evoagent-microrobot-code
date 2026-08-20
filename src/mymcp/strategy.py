"""
策略管理模块 - 完整策略库实现

功能：
1. 探索-利用平衡（ε-greedy）
2. 探索策略选择
3. 策略参数动态调整
4. 策略状态持久化
5. 成功模式库（识别成功规律）
6. 失败模式库（避免重蹈覆辙）
7. 参数敏感性分析
8. 动态提示词生成
9. 检索策略优化
10. 元学习支持（跨任务知识迁移）
"""

import os
import json
import random
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from loguru import logger

# 元学习支持
try:
    from .meta_learning import (
        MetaLearningAgent,
        MetaKnowledgeBase,
        ParameterNormalizer,
        MetaKnowledgeExtractor,
        create_default_task_config,
        ParamCategory,
        ParamRole,
    )
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False
    logger.warning("元学习模块未加载，跨任务知识迁移功能不可用")

# ExpeL 对比批评支持
try:
    from .expel_critique import RuleManager, RuleWithConfidence
    EXPEL_AVAILABLE = True
except ImportError:
    EXPEL_AVAILABLE = False
    logger.warning("ExpeL 模块未加载，对比批评功能不可用")

DEFAULT_STRATEGY_FILE = "strategy_state.json"


class ExplorationStrategy(Enum):
    """探索策略类型"""
    RANDOM = "random"           # 完全随机探索
    DIRECTED = "directed"       # 定向探索（基于历史趋势）
    PERTURBATION = "perturbation"  # 小扰动探索
    COUNTERFACTUAL = "counterfactual"  # 反事实探索（尝试相反方向）
    SENSITIVITY = "sensitivity"  # 敏感性探索（调整最敏感参数）
    AVOID_FAILURE = "avoid_failure"  # 避免失败区域的探索
    DISCRETE_JUMP = "discrete_jump"  # 离散派生绕组容量边界探索


@dataclass
class SuccessPattern:
    """成功模式"""
    params: Dict[str, float]
    fitness: float
    key_factors: List[str]  # 识别出的关键因素
    frequency: int = 1
    avg_fitness: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "params": self.params,
            "fitness": self.fitness,
            "key_factors": self.key_factors,
            "frequency": self.frequency,
            "avg_fitness": self.avg_fitness
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "SuccessPattern":
        return cls(
            params=d.get("params", {}),
            fitness=d.get("fitness", 0),
            key_factors=d.get("key_factors", []),
            frequency=d.get("frequency", 1),
            avg_fitness=d.get("avg_fitness", 0)
        )


@dataclass
class FailurePattern:
    """失败模式"""
    params: Dict[str, float]
    errors: List[str]
    avoid_rules: List[str]  # 应该避免的规则
    frequency: int = 1
    
    def to_dict(self) -> Dict:
        return {
            "params": self.params,
            "errors": self.errors,
            "avoid_rules": self.avoid_rules,
            "frequency": self.frequency
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "FailurePattern":
        return cls(
            params=d.get("params", {}),
            errors=d.get("errors", []),
            avoid_rules=d.get("avoid_rules", []),
            frequency=d.get("frequency", 1)
        )


@dataclass
class StrategyConfig:
    """策略配置"""
    # 探索率参数
    initial_epsilon: float = 0.3
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.98
    
    # 探索策略权重
    exploration_weights: Dict[str, float] = field(default_factory=lambda: {
        "random": 0.15,
        "directed": 0.30,
        "perturbation": 0.25,
        "counterfactual": 0.10,
        "sensitivity": 0.10,
        "avoid_failure": 0.10
    })
    
    # 扰动参数，用于覆盖相邻离散派生量边界
    perturbation_scale: float = 0.15
    
    # 离散派生量边界探索配置
    discrete_jump_threshold: int = 8      # 连续多少轮不变后提高边界邻域采样概率
    discrete_jump_scale: float = 0.25     # 边界邻域采样的步长放大倍数
    
    # 自适应参数
    success_boost: float = 0.02
    failure_penalty: float = 0.01
    
    # 模式识别参数
    pattern_similarity_threshold: float = 0.15  # 参数相似度阈值
    max_patterns: int = 50  # 最大模式数量
    sensitivity_window: int = 30  # 敏感性分析窗口（增大以提供更好的统计分析）
    
    # 在线元学习参数
    meta_extract_interval: int = 50  # 每 N 轮触发一次元知识提取


class StrategyManager:
    """完整策略管理器"""
    
    # 参数可行域（用于策略生成）
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
    
    # 敏感性分析参数范围（包含所有可能影响 fitness 的参数）
    SENSITIVITY_PARAM_BOUNDS = {
        "lm": (1.0, 6.0),
        "tm": (0.3, 0.5),
        "ta": (0.35, 0.75),
        "dg": (0.30, 0.65),
        "hs": (1.2, 2.2),
        "wslot": (2.0, 2.8),
        "hslot": (0.8, 1.3),
        "s": (0.8, 1.2),
        "wa": (2.0, 2.0),
        # 线圈参数（对 fitness 影响巨大）
        "n1": (50, 80),
        "n2": (2, 10),
        "twall": (0.1, 0.5),
        "tb": (0.8, 1.2),
        # 尺寸参数
        "la": (5.0, 7.0),
        "ha": (3.5, 5.0),
        "ws": (2.0, 3.5),
        "ls": (3.0, 4.5),
    }
    
    def __init__(
        self, 
        config: Optional[StrategyConfig] = None, 
        storage_path: str = DEFAULT_STRATEGY_FILE,
        enable_meta_learning: bool = True,
        meta_knowledge_path: str = "meta_knowledge.json",
        domain: str = "electromagnetic_actuator"
    ):
        self.config = config or StrategyConfig()
        self.storage_path = storage_path
        self.domain = domain
        
        # 基础状态
        self.epsilon = self.config.initial_epsilon
        self.iteration = 0
        self.recent_results: List[bool] = []
        self.last_direction: Dict[str, float] = {}
        self.success_regions: List[Dict[str, float]] = []
        
        # === 新增：完整策略库 ===
        # 1. 成功模式库
        self.success_patterns: List[SuccessPattern] = []
        
        # 2. 失败模式库
        self.failure_patterns: List[FailurePattern] = []
        
        # 3. 参数敏感性
        self.param_sensitivity: Dict[str, float] = {p: 0.5 for p in self.SENSITIVITY_PARAM_BOUNDS}
        self.param_history: List[Dict[str, Any]] = []  # 用于计算敏感性
        self.fitness_history: List[float] = []  # 新增：fitness 历史用于相关性分析
        
        # 4. 动态提示词片段
        self.prompt_additions: List[str] = []
        
        # 5. ★ 离散变量 (n1, n2) 历史追踪
        self.n1_history: List[int] = []
        self.n2_history: List[int] = []
        
        # 5. 学习到的规则（保留向后兼容）
        self.learned_rules: List[str] = []
        
        # 6. ★ ExpeL 规则管理器（带置信度的规则库）
        self.expel_enabled = EXPEL_AVAILABLE
        self.rule_manager: Optional[Any] = None
        if self.expel_enabled:
            self.rule_manager = RuleManager(
                max_rules=20,
                storage_path="expel_rules.json"
            )
            logger.info(f"[ExpeL] 规则管理器已初始化，当前规则数: {len(self.rule_manager.rules)}")
        
        # 7. 参数重要性权重（用于检索）
        self.param_importance: Dict[str, float] = {p: 1.0 for p in self.PARAM_BOUNDS}
        
        # === 蒸馏原则（跨任务迁移） ===
        self.distilled_principles = None  # DistilledPrinciples 对象
        
        # === 元学习支持 ===
        self.enable_meta_learning = enable_meta_learning and META_LEARNING_AVAILABLE
        self.meta_agent: Optional[Any] = None
        self.meta_prior_prompt: str = ""
        self.meta_exploration_suggestions: List[str] = []
        
        if self.enable_meta_learning:
            self._init_meta_learning(meta_knowledge_path)
        
        # 加载已有状态
        self._load()
    
    # ==================== 探索-利用决策 ====================
    
    def should_explore(self) -> bool:
        """决定是否进行探索（ε-greedy）"""
        return random.random() < self.epsilon
    
    def select_exploration_strategy(self) -> ExplorationStrategy:
        """选择探索策略（考虑当前状态动态调整权重）"""
        
        # 检查 n2 是否长期保持不变或在小范围内波动
        n2_unchanged = self.get_rounds_since_change(self.n2_history)
        n2_stagnation = self.get_stagnation_info(self.n2_history, window=15)
        threshold = getattr(self.config, 'discrete_jump_threshold', 8)
        
        # 当离散派生量长期不变时，增加边界邻域采样
        n2_should_jump = (
            n2_unchanged >= threshold or 
            (n2_stagnation["is_stagnant"] and n2_stagnation["rounds_in_window"] >= 8)
        )
        
        if n2_should_jump:
            # n2 停滞时，以一定概率采用离散边界邻域探索
            if random.random() < 0.5:
                if n2_stagnation["is_stagnant"] and n2_unchanged < threshold:
                    logger.info(f"[离散边界探索] n2 在 [{n2_stagnation['min_val']}, {n2_stagnation['max_val']}] 波动 {n2_stagnation['rounds_in_window']} 轮，增加边界邻域采样")
                else:
                    logger.info(f"[离散边界探索] n2 已保持 {n2_unchanged} 轮，增加边界邻域采样")
                return ExplorationStrategy.DISCRETE_JUMP
        
        weights = self.config.exploration_weights.copy()
        
        # 动态调整权重
        # 如果失败模式多，增加避免失败策略的权重
        if len(self.failure_patterns) > 5:
            weights["avoid_failure"] = min(0.25, weights.get("avoid_failure", 0.1) * 1.5)
        
        # 如果敏感性分析有结果，增加敏感性探索权重
        high_sensitivity_params = [p for p, s in self.param_sensitivity.items() if s > 0.7]
        if high_sensitivity_params:
            weights["sensitivity"] = min(0.25, weights.get("sensitivity", 0.1) * 1.5)
        
        # 如果成功模式多，增加定向探索权重
        if len(self.success_patterns) > 3:
            weights["directed"] = min(0.4, weights.get("directed", 0.3) * 1.2)
        
        # 如果 n2 接近停滞，增加离散边界邻域探索权重
        if n2_unchanged >= threshold // 2 or (n2_stagnation["is_stagnant"] and n2_stagnation["rounds_in_window"] >= 5):
            weights["discrete_jump"] = 0.25
        
        # 归一化
        strategies = list(weights.keys())
        probs = [weights[s] for s in strategies]
        total = sum(probs)
        probs = [p / total for p in probs]
        
        r = random.random()
        cumsum = 0
        for strategy, prob in zip(strategies, probs):
            cumsum += prob
            if r < cumsum:
                try:
                    return ExplorationStrategy(strategy)
                except ValueError:
                    continue
        return ExplorationStrategy.RANDOM
    
    def generate_exploration_action(
        self,
        current_state: Dict[str, float],
        strategy: Optional[ExplorationStrategy] = None
    ) -> Dict[str, float]:
        """生成探索动作"""
        if strategy is None:
            strategy = self.select_exploration_strategy()
        
        logger.debug(f"使用探索策略: {strategy.value}")
        
        # 确保 current_state 中的 None 值被替换为默认值
        safe_state = {}
        for param, (low, high) in self.PARAM_BOUNDS.items():
            val = current_state.get(param)
            if val is None:
                safe_state[param] = round((low + high) / 2, 2)
            else:
                safe_state[param] = val
        
        if strategy == ExplorationStrategy.RANDOM:
            result = self._random_exploration(safe_state)
        elif strategy == ExplorationStrategy.DIRECTED:
            result = self._directed_exploration(safe_state)
        elif strategy == ExplorationStrategy.PERTURBATION:
            result = self._perturbation_exploration(safe_state)
        elif strategy == ExplorationStrategy.COUNTERFACTUAL:
            result = self._counterfactual_exploration(safe_state)
        elif strategy == ExplorationStrategy.SENSITIVITY:
            result = self._sensitivity_exploration(safe_state)
        elif strategy == ExplorationStrategy.AVOID_FAILURE:
            result = self._avoid_failure_exploration(safe_state)
        elif strategy == ExplorationStrategy.DISCRETE_JUMP:
            result = self._discrete_jump_exploration(safe_state)
        else:
            result = self._random_exploration(safe_state)
        
        # 确保返回结果不包含 None
        return {k: v for k, v in result.items() if v is not None}
    
    def _random_exploration(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """完全随机探索"""
        new_state = {}
        for param, (low, high) in self.PARAM_BOUNDS.items():
            if low == high:
                new_state[param] = low
            else:
                new_state[param] = round(random.uniform(low, high), 2)
        return new_state
    
    def _directed_exploration(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """定向探索：向成功模式靠拢"""
        new_state = current_state.copy()
        
        # 优先使用成功模式
        if self.success_patterns:
            # 选择 fitness 最好的成功模式
            best_pattern = min(self.success_patterns, key=lambda p: p.fitness)
            target = best_pattern.params
            alpha = random.uniform(0.4, 0.8)
            
            for param in self.PARAM_BOUNDS:
                if param in target and param in new_state:
                    target_val = target[param]
                    current_val = new_state[param]
                    # 跳过 None 值
                    if target_val is None or current_val is None:
                        continue
                    low, high = self.PARAM_BOUNDS[param]
                    if low != high:
                        new_val = alpha * target_val + (1 - alpha) * current_val
                        new_state[param] = round(max(low, min(high, new_val)), 2)
        elif self.success_regions:
            # 退化到成功区域
            target = random.choice(self.success_regions)
            alpha = random.uniform(0.3, 0.7)
            for param in self.PARAM_BOUNDS:
                if param in target and param in new_state:
                    target_val = target[param]
                    current_val = new_state[param]
                    # 跳过 None 值
                    if target_val is None or current_val is None:
                        continue
                    low, high = self.PARAM_BOUNDS[param]
                    if low != high:
                        new_val = alpha * target_val + (1 - alpha) * current_val
                        new_state[param] = round(max(low, min(high, new_val)), 2)
        else:
            return self._perturbation_exploration(current_state)
        
        return new_state
    
    def _perturbation_exploration(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """小扰动探索"""
        new_state = current_state.copy()
        scale = self.config.perturbation_scale
        
        adjustable = [p for p in self.PARAM_BOUNDS if self.PARAM_BOUNDS[p][0] != self.PARAM_BOUNDS[p][1]]
        num_params = random.randint(1, min(3, len(adjustable)))
        params_to_perturb = random.sample(adjustable, num_params)
        
        for param in params_to_perturb:
            low, high = self.PARAM_BOUNDS[param]
            current = new_state.get(param)
            # 如果是 None，使用中点值
            if current is None:
                current = (low + high) / 2
            delta = (high - low) * scale * random.gauss(0, 1)
            new_val = current + delta
            new_state[param] = round(max(low, min(high, new_val)), 2)
        
        return new_state
    
    def _counterfactual_exploration(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """反事实探索：尝试相反方向"""
        new_state = current_state.copy()
        
        if not self.last_direction:
            return self._random_exploration(current_state)
        
        for param, direction in self.last_direction.items():
            if param in new_state and param in self.PARAM_BOUNDS:
                current = new_state[param]
                # 跳过 None 值
                if current is None:
                    continue
                low, high = self.PARAM_BOUNDS[param]
                if low == high:
                    continue
                delta = -direction * (high - low) * 0.15
                new_val = current + delta
                new_state[param] = round(max(low, min(high, new_val)), 2)
        
        return new_state
    
    def _sensitivity_exploration(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """敏感性探索：重点调整最敏感的参数"""
        new_state = current_state.copy()
        
        # 找出最敏感的可调参数（必须在 PARAM_BOUNDS 中）
        sorted_params = sorted(
            [(p, s) for p, s in self.param_sensitivity.items() 
             if p in self.PARAM_BOUNDS and self.PARAM_BOUNDS.get(p, (0, 0))[0] != self.PARAM_BOUNDS.get(p, (0, 0))[1]],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 调整前 2-3 个最敏感的参数
        top_params = [p for p, _ in sorted_params[:random.randint(2, 3)]]
        
        for param in top_params:
            if param in self.PARAM_BOUNDS:
                low, high = self.PARAM_BOUNDS[param]
                current = new_state.get(param)
                # 如果是 None，使用中点值
                if current is None:
                    current = (low + high) / 2
                # 较大幅度调整
                delta = (high - low) * random.uniform(-0.2, 0.2)
                new_val = current + delta
                new_state[param] = round(max(low, min(high, new_val)), 2)
        
        return new_state
    
    def _avoid_failure_exploration(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """避免失败区域的探索"""
        max_attempts = 20
        
        for _ in range(max_attempts):
            # 生成候选
            candidate = self._perturbation_exploration(current_state)
            
            # 检查是否在失败区域附近
            is_near_failure = False
            for fp in self.failure_patterns:
                if self._params_similar(candidate, fp.params, threshold=0.1):
                    is_near_failure = True
                    break
            
            if not is_near_failure:
                return candidate
        
        # 所有尝试都失败，返回随机
        return self._random_exploration(current_state)
    
    def _discrete_jump_exploration(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """
        离散派生绕组容量边界探索。
        
        n2 = floor(9 × (hs - hslot))
        
        三种邻域采样方式：
        1. 增大 hs（直接有效）
        2. 减小 hslot（需要先降低 ta/tb_ratio 以满足约束 hslot ≥ tb + 0.1）
        3. 同时调整 hs↑ + ta↓ → hslot↓
        """
        new_state = current_state.copy()
        
        # 获取当前参数
        hs = current_state.get("hs", 1.65)
        hslot = current_state.get("hslot", 1.26)
        ta = current_state.get("ta", 0.6)
        tb_ratio = current_state.get("tb_ratio", 1.75)
        
        # 计算当前 n2
        current_n2 = max(1, int(9 * (hs - hslot)))
        target_n2 = current_n2 + 1
        
        # 计算到下一个 n2 离散边界所需的 (hs - hslot) 值
        # 向下取整时，需要 9 * (hs - hslot) >= target_n2。
        required_gap = (target_n2 + 0.05) / 9
        current_gap = hs - hslot
        gap_deficit = required_gap - current_gap  # 还差多少
        
        # 参数边界
        hs_low, hs_high = self.PARAM_BOUNDS.get("hs", (1.2, 2.2))
        hslot_low, hslot_high = self.PARAM_BOUNDS.get("hslot", (0.8, 1.3))
        ta_low, ta_high = self.PARAM_BOUNDS.get("ta", (0.35, 0.75))
        
        # 随机选择边界邻域采样策略
        strategy = random.choice(["hs_up", "hslot_down", "combined"])
        
        logger.info(f"[离散边界探索] 当前 n2={current_n2}, 邻近边界 n2={target_n2}, "
                   f"当前 gap={current_gap:.3f}, 需要 gap≥{required_gap:.3f}, "
                   f"策略: {strategy}")
        
        if strategy == "hs_up":
            # 方案1：增大 hs
            new_hs = hs + gap_deficit + 0.02  # 加一点余量
            new_hs = min(new_hs, hs_high)  # 不超过上限
            new_state["hs"] = round(new_hs, 3)
            logger.info(f"   → hs: {hs:.3f} → {new_state['hs']:.3f}")
            
        elif strategy == "hslot_down":
            # 方案2：减小 hslot（需要先处理约束）
            # 约束: hslot ≥ tb + 0.1 = ta × tb_ratio + 0.1
            current_tb = ta * tb_ratio
            hslot_min = current_tb + 0.1
            
            # 目标 hslot
            target_hslot = hs - required_gap - 0.02
            
            if target_hslot >= hslot_min and target_hslot >= hslot_low:
                # 可以直接降低 hslot
                new_state["hslot"] = round(max(target_hslot, hslot_low), 3)
                logger.info(f"   → hslot: {hslot:.3f} → {new_state['hslot']:.3f} (直接)")
            else:
                # 需要先降低 ta 或 tb_ratio
                # 计算需要的 tb 值
                required_tb = target_hslot - 0.1
                if required_tb > 0:
                    # 降低 ta 来实现
                    new_ta = required_tb / tb_ratio
                    new_ta = max(new_ta, ta_low)  # 不低于下限
                    new_state["ta"] = round(new_ta, 3)
                    # 更新 hslot
                    new_hslot_min = new_ta * tb_ratio + 0.1
                    new_state["hslot"] = round(max(new_hslot_min, hslot_low), 3)
                    logger.info(f"   → ta: {ta:.3f} → {new_state['ta']:.3f}")
                    logger.info(f"   → hslot: {hslot:.3f} → {new_state['hslot']:.3f} (联动)")
                else:
                    # 无法通过降低 hslot 实现，转为增大 hs
                    new_hs = hs + gap_deficit + 0.02
                    new_state["hs"] = round(min(new_hs, hs_high), 3)
                    logger.info(f"   → 无法降低 hslot，转为增大 hs: {hs:.3f} → {new_state['hs']:.3f}")
        
        else:  # combined
            # 方案3：同时调整（各承担一半）
            half_deficit = gap_deficit / 2 + 0.01
            
            # 增大 hs
            new_hs = min(hs + half_deficit, hs_high)
            new_state["hs"] = round(new_hs, 3)
            
            # 尝试减小 hslot
            current_tb = ta * tb_ratio
            target_hslot = hslot - half_deficit
            hslot_min = current_tb + 0.1
            
            if target_hslot >= hslot_min and target_hslot >= hslot_low:
                new_state["hslot"] = round(target_hslot, 3)
            else:
                # 需要先降低 ta
                target_tb = target_hslot - 0.1
                if target_tb > 0:
                    new_ta = max(target_tb / tb_ratio, ta_low)
                    new_state["ta"] = round(new_ta, 3)
                    new_hslot_min = new_ta * tb_ratio + 0.1
                    new_state["hslot"] = round(max(new_hslot_min, target_hslot, hslot_low), 3)
            
            logger.info(f"   → hs: {hs:.3f} → {new_state['hs']:.3f}")
            logger.info(f"   → hslot: {hslot:.3f} → {new_state.get('hslot', hslot):.3f}")
            if "ta" in new_state and new_state["ta"] != ta:
                logger.info(f"   → ta: {ta:.3f} → {new_state['ta']:.3f}")
        
        # 最终检查：确保所有参数都在可行域内
        for param in new_state:
            if param in self.PARAM_BOUNDS:
                low, high = self.PARAM_BOUNDS[param]
                new_state[param] = round(max(low, min(high, new_state[param])), 3)
        
        return new_state
    
    # ==================== 策略更新 ====================
    
    def record_constraint_violation(
        self,
        params: Dict[str, float],
        errors: List[str]
    ):
        """记录 validate 阶段的约束违规（独立于仿真结果）"""
        # 直接更新失败模式，不影响其他状态
        self._update_failure_patterns(params, errors)
        self._save()
    
    def update_after_result(
        self,
        old_state: Dict[str, float],
        new_state: Dict[str, float],
        success: bool,
        fitness: Optional[float] = None,
        errors: Optional[List[str]] = None,
        best_fitness: Optional[float] = None  # ★新增：历史最佳fitness
    ):
        """根据结果更新策略 - 增强版（含软失败判断）"""
        self.iteration += 1
        
        # ★新增：判断是否"软失败"（仿真成功但fitness明显变差）
        is_soft_failure = False
        soft_failure_reason = None
        if success and fitness is not None and best_fitness is not None:
            # 如果fitness比历史最佳差超过20%，标记为软失败
            # 注意：fitness越小越好，所以fitness > best_fitness * 0.8表示变差
            # 但为了避免陷入局部最优，我们用较宽松的阈值
            if best_fitness < 0:  # fitness为负数（越小越好）
                threshold = best_fitness * 0.8  # 允许20%的波动
                if fitness > threshold:
                    is_soft_failure = True
                    degradation = (fitness - best_fitness) / abs(best_fitness) * 100
                    soft_failure_reason = f"fitness退化{degradation:.1f}%（当前{fitness:.2f} vs 最佳{best_fitness:.2f}）"
        
        # 记录结果：真成功 / 软失败 / 硬失败
        effective_success = success and not is_soft_failure
        self.recent_results.append(effective_success)
        if len(self.recent_results) > 20:
            self.recent_results.pop(0)
        
        # 1. 记录调整方向
        self.last_direction = {}
        for param in self.PARAM_BOUNDS:
            if param in old_state and param in new_state:
                old_val = old_state[param]
                new_val = new_state[param]
                # 跳过 None 值
                if old_val is None or new_val is None:
                    continue
                diff = new_val - old_val
                low, high = self.PARAM_BOUNDS[param]
                if high > low:
                    self.last_direction[param] = diff / (high - low)
        
        # 2. 更新成功/失败模式（增强版）
        if success and fitness is not None:
            # 仿真成功，记录成功模式
            self._update_success_patterns(new_state, fitness)
            self.success_regions.append(new_state.copy())
            if len(self.success_regions) > 20:
                self.success_regions.pop(0)
            
            # ★新增：如果是软失败，也记录为失败模式（用于学习）
            if is_soft_failure and soft_failure_reason:
                self._update_failure_patterns(new_state, [f"[软失败] {soft_failure_reason}"])
        
        elif not success and errors:
            # 硬失败（约束违规或仿真失败）
            self._update_failure_patterns(new_state, errors)
        
        # 3. 更新参数敏感性
        self._update_param_sensitivity(old_state, new_state, fitness, effective_success)
        
        # 4. 更新探索率（软失败只轻微惩罚，避免过度保守）
        if is_soft_failure:
            # 软失败：探索率小幅上升（鼓励继续探索）
            self.epsilon = min(self.config.initial_epsilon, 
                             self.epsilon + self.config.failure_penalty * 0.5)
        else:
            self._update_epsilon(effective_success)
        
        # 5. 生成动态提示词
        self._update_prompt_additions()
        
        # 6. 提取学习规则
        self._extract_learned_rules()
        
        # 7. ★在线元知识提取（每 meta_extract_interval 轮触发一次）
        meta_extract_result = None
        if self.enable_meta_learning and self.meta_agent:
            meta_extract_result = self._online_meta_extract()
        
        # 8. 持久化
        self._save()
        
        # 返回是否软失败，供日志使用
        return {
            "is_soft_failure": is_soft_failure, 
            "reason": soft_failure_reason,
            "meta_extract": meta_extract_result
        }
    
    def _update_success_patterns(self, state: Dict[str, float], fitness: float):
        """更新成功模式库"""
        # 检查是否与现有模式相似
        for pattern in self.success_patterns:
            if self._params_similar(state, pattern.params):
                # 更新现有模式
                pattern.frequency += 1
                pattern.avg_fitness = (pattern.avg_fitness * (pattern.frequency - 1) + fitness) / pattern.frequency
                if fitness < pattern.fitness:
                    pattern.fitness = fitness
                    pattern.params = state.copy()
                return
        
        # 添加新模式
        key_factors = self._identify_key_factors(state, is_success=True)
        new_pattern = SuccessPattern(
            params=state.copy(),
            fitness=fitness,
            key_factors=key_factors,
            frequency=1,
            avg_fitness=fitness
        )
        self.success_patterns.append(new_pattern)
        
        # 限制数量
        if len(self.success_patterns) > self.config.max_patterns:
            # 移除频率最低的
            self.success_patterns.sort(key=lambda p: p.frequency, reverse=True)
            self.success_patterns = self.success_patterns[:self.config.max_patterns]
    
    def _update_failure_patterns(self, state: Dict[str, float], errors: List[str]):
        """更新失败模式库"""
        # 检查是否与现有模式相似
        for pattern in self.failure_patterns:
            if self._params_similar(state, pattern.params):
                pattern.frequency += 1
                # 合并错误
                for e in errors:
                    if e not in pattern.errors:
                        pattern.errors.append(e)
                return
        
        # 添加新模式
        avoid_rules = self._generate_avoid_rules(state, errors)
        new_pattern = FailurePattern(
            params=state.copy(),
            errors=errors.copy(),
            avoid_rules=avoid_rules,
            frequency=1
        )
        self.failure_patterns.append(new_pattern)
        
        # 限制数量
        if len(self.failure_patterns) > self.config.max_patterns:
            self.failure_patterns.sort(key=lambda p: p.frequency, reverse=True)
            self.failure_patterns = self.failure_patterns[:self.config.max_patterns]
    
    def _update_param_sensitivity(
        self,
        old_state: Dict[str, float],
        new_state: Dict[str, float],
        fitness: Optional[float],
        success: bool
    ):
        """更新参数敏感性分析"""
        # 记录历史（包含 fitness）
        self.param_history.append({
            "old": old_state.copy(),
            "new": new_state.copy(),
            "fitness": fitness,
            "success": success
        })
        
        # 记录 fitness 历史
        if fitness is not None:
            self.fitness_history.append(fitness)
        
        # 扩大敏感性分析窗口（至少 20 条记录才能有意义的相关性分析）
        window_size = max(self.config.sensitivity_window, 30)
        if len(self.param_history) > window_size:
            self.param_history.pop(0)
        if len(self.fitness_history) > window_size:
            self.fitness_history.pop(0)
        
        # 计算敏感性（至少需要 5 条有效记录）
        if len(self.param_history) >= 5:
            self._calculate_sensitivity()
    
    def _calculate_sensitivity(self):
        """
        使用多种方法计算参数敏感性：
        1. 相关性分析：参数值与 fitness 的皮尔逊相关系数
        2. 变化影响：参数变化导致的 fitness 变化
        3. 离散参数特殊处理（如 n2）
        """
        from collections import defaultdict
        import math
        
        # 收集所有参数值和对应的 fitness
        param_values: Dict[str, List[float]] = defaultdict(list)
        fitness_values: List[float] = []
        
        # 从历史记录中提取参数值
        for record in self.param_history:
            if record["fitness"] is None:
                continue
            fitness_values.append(record["fitness"])
            new_state = record["new"]
            for param in self.SENSITIVITY_PARAM_BOUNDS:
                if param in new_state and new_state[param] is not None:
                    param_values[param].append(float(new_state[param]))
        
        if len(fitness_values) < 5:
            return
        
        # 方法1：计算皮尔逊相关系数（参数值 vs fitness）
        def pearson_correlation(x: List[float], y: List[float]) -> float:
            """计算皮尔逊相关系数"""
            if len(x) != len(y) or len(x) < 3:
                return 0.0
            n = len(x)
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
            denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
            
            if denom_x * denom_y == 0:
                return 0.0
            return numerator / (denom_x * denom_y)
        
        # 方法2：计算参数变化导致的 fitness 变化（变化敏感性）
        def calculate_change_impact(param: str) -> float:
            """计算参数变化对 fitness 的影响程度"""
            impacts = []
            for i, record in enumerate(self.param_history[1:], 1):
                if record["fitness"] is None:
                    continue
                prev_record = self.param_history[i - 1]
                if prev_record["fitness"] is None:
                    continue
                
                old_state = prev_record["new"]
                new_state = record["new"]
                
                if param not in old_state or param not in new_state:
                    continue
                
                old_val = old_state.get(param)
                new_val = new_state.get(param)
                if old_val is None or new_val is None:
                    continue
                
                # 归一化参数变化
                low, high = self.SENSITIVITY_PARAM_BOUNDS.get(param, (0, 1))
                param_range = high - low if high > low else 1
                param_change = abs(new_val - old_val) / param_range
                
                # fitness 变化
                fitness_change = abs(record["fitness"] - prev_record["fitness"])
                
                if param_change > 0.01:  # 只考虑有意义的变化
                    # 计算变化比例：fitness 变化 / 参数变化
                    impact = fitness_change / param_change if param_change > 0 else 0
                    impacts.append(impact)
            
            return sum(impacts) / len(impacts) if impacts else 0.0
        
        # 综合计算敏感性
        sensitivities: Dict[str, float] = {}
        
        for param in self.SENSITIVITY_PARAM_BOUNDS:
            if param not in param_values or len(param_values[param]) < 3:
                sensitivities[param] = 0.0
                continue
            
            # 确保长度匹配
            p_vals = param_values[param]
            f_vals = fitness_values[:len(p_vals)]
            
            if len(p_vals) != len(f_vals):
                min_len = min(len(p_vals), len(f_vals))
                p_vals = p_vals[:min_len]
                f_vals = f_vals[:min_len]
            
            # 方法1：相关性（注意：fitness 是负的，越小越好，所以取绝对值）
            correlation = abs(pearson_correlation(p_vals, f_vals))
            
            # 方法2：变化影响
            change_impact = calculate_change_impact(param)
            
            # 方法3：参数值变化范围（标准差/范围）反映参数被调整的程度
            if len(p_vals) > 1:
                mean_val = sum(p_vals) / len(p_vals)
                variance = sum((v - mean_val) ** 2 for v in p_vals) / len(p_vals)
                std_val = math.sqrt(variance)
                low, high = self.SENSITIVITY_PARAM_BOUNDS.get(param, (0, 1))
                param_range = high - low if high > low else 1
                exploration_ratio = std_val / param_range if param_range > 0 else 0
            else:
                exploration_ratio = 0
            
            # 综合得分：相关性权重最高，变化影响次之，探索程度作为调节
            # 如果一个参数被大量调整且与 fitness 高度相关，说明非常敏感
            base_sensitivity = correlation * 0.6 + min(change_impact / 10, 0.4) * 0.4
            
            # 如果探索程度低但相关性高，增加敏感性权重
            if exploration_ratio < 0.1 and correlation > 0.3:
                base_sensitivity *= 1.2
            
            sensitivities[param] = base_sensitivity
        
        # 归一化到 [0, 1]
        max_sens = max(sensitivities.values()) if sensitivities and max(sensitivities.values()) > 0 else 1
        for p in sensitivities:
            self.param_sensitivity[p] = round(sensitivities[p] / max_sens, 3)
    
    def _update_epsilon(self, last_success: bool):
        """自适应更新探索率"""
        self.epsilon *= self.config.epsilon_decay
        
        if self.recent_results:
            recent_success_rate = sum(self.recent_results) / len(self.recent_results)
            if recent_success_rate > 0.6:
                self.epsilon -= self.config.success_boost
            elif recent_success_rate < 0.3:
                self.epsilon += self.config.failure_penalty
        
        self.epsilon = max(self.config.min_epsilon, min(0.5, self.epsilon))
    
    # ========== ★ 离散变量探索追踪机制 ==========
    
    def record_discrete_variables(self, n1: int, n2: int):
        """记录当前轮的 n1, n2 值"""
        self.n1_history.append(n1)
        self.n2_history.append(n2)
        
        # 保持历史长度合理
        if len(self.n1_history) > 100:
            self.n1_history.pop(0)
        if len(self.n2_history) > 100:
            self.n2_history.pop(0)
    
    def get_rounds_since_change(self, history: List[int]) -> int:
        """计算离散变量连续多少轮没有变化"""
        if len(history) < 2:
            return 0
        
        count = 1
        current = history[-1]
        for val in reversed(history[:-1]):
            if val == current:
                count += 1
            else:
                break
        return count
    
    def get_stagnation_info(self, history: List[int], window: int = 15) -> Dict[str, Any]:
        """
        检测离散变量是否长期在某个范围内波动但未突破
        
        Args:
            history: 历史记录
            window: 检测窗口大小
        
        Returns:
            {
                "is_stagnant": bool,  # 是否停滞
                "min_val": int,       # 窗口内最小值
                "max_val": int,       # 窗口内最大值
                "current": int,       # 当前值
                "rounds_in_window": int,  # 窗口内的轮数
                "never_exceeded": bool,   # 是否从未超过 max_val+1
            }
        """
        if len(history) < 5:
            return {"is_stagnant": False, "min_val": 0, "max_val": 0, "current": 0, "rounds_in_window": 0, "never_exceeded": True}
        
        # 取最近 window 轮的数据
        recent = history[-window:] if len(history) >= window else history
        
        min_val = min(recent)
        max_val = max(recent)
        current = history[-1]
        
        # 检查是否"停滞"：长期在一个小范围内波动（差值 <= 1）
        is_stagnant = (max_val - min_val) <= 1 and len(recent) >= window // 2
        
        return {
            "is_stagnant": is_stagnant,
            "min_val": min_val,
            "max_val": max_val,
            "current": current,
            "rounds_in_window": len(recent),
            "never_exceeded": True  # 暂时不使用全局历史，只看窗口
        }
    
    def build_discrete_variable_exploration_prompt(
        self,
        current_params: Dict[str, float],
        n1: int,
        n2: int,
        delta_hs_to_next_n2: float,
        delta_lm_to_next_n1: float,
        threshold: int = 10  # 连续多少轮不变就提示
    ) -> Optional[str]:
        """
        构建离散派生量边界探索提示（主动引导，非强制）
        
        触发条件（满足任一即可）：
        1. n2 连续 threshold 轮不变
        2. n2 长期在某个范围波动
        """
        n1_unchanged_rounds = self.get_rounds_since_change(self.n1_history)
        n2_unchanged_rounds = self.get_rounds_since_change(self.n2_history)
        
        # 检测长期波动的情况
        n2_stagnation = self.get_stagnation_info(self.n2_history, window=15)
        n1_stagnation = self.get_stagnation_info(self.n1_history, window=15)
        
        # 判断是否需要增加 n2 边界邻域探索
        # 条件1：连续不变超过阈值
        # 条件2：长期在小范围波动（如 3-4）且窗口足够大
        n2_should_explore = (
            n2_unchanged_rounds >= threshold or 
            (n2_stagnation["is_stagnant"] and n2_stagnation["rounds_in_window"] >= 8)
        )
        
        n1_should_explore = (
            n1_unchanged_rounds >= threshold or
            (n1_stagnation["is_stagnant"] and n1_stagnation["rounds_in_window"] >= 8)
        )
        
        lines = []
        
        # n2 探索提示（优先级更高，因为 n2 影响绕组容量）
        # n2 = floor(9 * (hs - hslot))，可通过 hs 或 hslot 调整到相邻离散边界
        if n2_should_explore:
            current_hs = current_params.get("hs", 0)
            current_hslot = current_params.get("hslot", 1.0)
            current_ta = current_params.get("ta", 0.68)
            current_tb_ratio = current_params.get("tb_ratio", 1.5)
            current_tb = current_ta * current_tb_ratio  # tb = ta × tb_ratio
            
            hs_upper = self.PARAM_BOUNDS.get("hs", (0, 2.2))[1]
            hslot_lower = self.PARAM_BOUNDS.get("hslot", (0.8, 1.3))[0]
            ta_lower = self.PARAM_BOUNDS.get("ta", (0.3, 1.0))[0]
            tb_ratio_lower = 1.5  # tb_ratio 下限
            
            new_hs = current_hs + delta_hs_to_next_n2
            
            # 计算通过降低 hslot 到达相邻 n2 边界所需的值
            # 需要 9 * (hs - hslot_new) = n2 + 0.5
            # hslot_new = hs - (n2 + 0.5) / 9
            hslot_for_jump = current_hs - (n2 + 0.5) / 9
            delta_hslot_to_jump = current_hslot - hslot_for_jump  # 需要减少的量
            
            # ★ 约束: hslot ≥ tb + 0.1，所以 hslot_min = tb + 0.1
            hslot_min_by_tb = current_tb + 0.1
            # 如果要降低 hslot 到 hslot_for_jump，需要 tb 满足: tb ≤ hslot_for_jump - 0.1
            tb_required = hslot_for_jump - 0.1
            
            if delta_hs_to_next_n2 > 0 or delta_hslot_to_jump > 0:
                lines.append("═" * 50)
                lines.append("【离散派生量边界探索建议：n2】")
                
                # 根据触发原因给出不同的提示（弱化语气，仅作提示）
                if n2_unchanged_rounds >= threshold:
                    lines.append(f"   ⏱️ n2（线圈层数）已连续 {n2_unchanged_rounds} 轮保持为 {n2}")
                elif n2_stagnation["is_stagnant"]:
                    lines.append(f"   📊 n2 在最近 {n2_stagnation['rounds_in_window']} 轮内在 [{n2_stagnation['min_val']}, {n2_stagnation['max_val']}] 波动")
                    lines.append(f"   当前 n2={n2}，可考虑在相邻边界附近采样")
                lines.append(f"   📐 公式: n2 = floor(9 × (hs - hslot))")
                lines.append("")
                
                # 简化方案提示
                if new_hs <= hs_upper:
                    lines.append(f"   可选方案：增大 hs 约 +{delta_hs_to_next_n2:.3f}mm，使 n2 接近 {n2+1}")
                else:
                    lines.append(f"   可选方案：减小 hslot 约 -{delta_hslot_to_jump:.3f}mm，使 n2 接近 {n2+1}")
                
                lines.append(f"   🔬 物理意义：层数增加 → 总匝数增加 → 推力系数 kb 可能提升")
                lines.append("═" * 50)
        
        # n1 探索提示（弱化语气，仅作提示）
        if n1_should_explore:
            current_lm = current_params.get("lm", 0)
            lm_upper = self.PARAM_BOUNDS.get("lm", (0, 6.0))[1]
            new_lm = current_lm + delta_lm_to_next_n1
            
            if delta_lm_to_next_n1 > 0:
                if lines:
                    lines.append("")
                else:
                    lines.append("═" * 50)
                lines.append("【离散派生量边界探索建议：n1】")
                
                # 根据触发原因给出不同的提示
                if n1_unchanged_rounds >= threshold:
                    lines.append(f"   ⏱️ n1（匝数/层）已连续 {n1_unchanged_rounds} 轮保持为 {n1}")
                elif n1_stagnation["is_stagnant"]:
                    lines.append(f"   📊 n1 在最近 {n1_stagnation['rounds_in_window']} 轮内在 [{n1_stagnation['min_val']}, {n1_stagnation['max_val']}] 波动")
                
                if new_lm <= lm_upper:
                    lines.append(f"   💡 可选尝试：将 lm 从 {current_lm:.3f} 调整到约 {new_lm:.3f}")
                    lines.append(f"   效果预期：n1 接近 {n1+1}")
                else:
                    lines.append(f"   ℹ️ 注意：理想 lm={new_lm:.3f} 已接近上限 {lm_upper}")
                
                if not lines[-1].endswith("═" * 50):
                    lines.append("═" * 50)
        
        if not lines:
            return None
        
        return "\n".join(lines)
    
    def _update_prompt_additions(self):
        """生成动态提示词片段"""
        self.prompt_additions = []
        
        # 1. 基于失败模式的警告
        if self.failure_patterns:
            # 找出最频繁的失败模式
            frequent_failures = sorted(self.failure_patterns, key=lambda p: p.frequency, reverse=True)[:3]
            for fp in frequent_failures:
                if fp.avoid_rules:
                    self.prompt_additions.append(f"⚠️ 警告：{fp.avoid_rules[0]}（历史失败 {fp.frequency} 次）")
        
        # 2. 基于成功模式的建议
        if self.success_patterns:
            best = min(self.success_patterns, key=lambda p: p.fitness)
            if best.key_factors:
                self.prompt_additions.append(f"💡 成功经验：{', '.join(best.key_factors[:3])}")
        
        # 3. 基于敏感性的提示
        high_sens = [(p, s) for p, s in self.param_sensitivity.items() if s > 0.6]
        if high_sens:
            high_sens.sort(key=lambda x: x[1], reverse=True)
            params_str = ", ".join([f"{p}(敏感度{s:.2f})" for p, s in high_sens[:3]])
            self.prompt_additions.append(f"🔑 关键参数：{params_str}")
        
        # 4. 基于学习规则的提示
        for rule in self.learned_rules[:2]:
            self.prompt_additions.append(f"📌 规则：{rule}")
    
    def _extract_learned_rules(self):
        """从经验中提取学习规则 - 增强版"""
        new_rules = []
        
        # 1. 从失败模式提取规则
        error_counts: Dict[str, int] = defaultdict(int)
        for fp in self.failure_patterns:
            for error in fp.errors:
                error_counts[error] += fp.frequency
        
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for error, count in common_errors:
            if count >= 3:
                related_params = self._find_params_related_to_error(error)
                if related_params:
                    new_rules.append(f"避免 {related_params}，否则容易出现：{error[:30]}...")
        
        # 2. 从成功模式提取规则
        if len(self.success_patterns) >= 3:
            common_features = self._find_common_features()
            for feature in common_features[:2]:
                new_rules.append(f"成功配置通常满足：{feature}")
        
        # 3. ★新增：从参数历史分析变化趋势与 fitness 的关系（输出物理化结论）
        if len(self.param_history) >= 3:
            # 分析每个参数的变化方向与 fitness 改善的关系
            param_trends = self._analyze_param_fitness_correlation()
            for param, trend_info in param_trends.items():
                if trend_info["confidence"] >= 0.6:  # 置信度阈值
                    direction = "增大" if trend_info["direction"] > 0 else "减小"
                    corr = trend_info["correlation"]
                    samples = trend_info.get("sample_count", 0)
                    phys_expl = self._build_physical_explanation(param, direction)
                    new_rules.append(
                        f"{param} {direction}有助于改善 fitness（相关系数 {corr:.2f}，样本 {samples}）; 物理解释：{phys_expl}"
                    )
        
        # 4. ★调整：不再给出“区间范围”类结论，改为组合趋势/物理因果
        # 5. ★新增：识别参数组合规律
        if len(self.param_history) >= 5:
            combo_rules = self._extract_combo_rules()
            new_rules.extend(combo_rules)
        
        # 6. ★新增：引入评论家 + 主策略库的物理/约束类规则，辅助元学习
        external_snippets = self._load_strategy_snippets(limit=6)
        if external_snippets:
            new_rules.extend(external_snippets)
        
        # 合并规则（去重）
        self.learned_rules = list(set(self.learned_rules + new_rules))[:15]
    
    def _analyze_param_fitness_correlation(self) -> Dict[str, Dict]:
        """分析参数变化与 fitness 变化的相关性"""
        correlations = {}
        
        # 用相邻两条记录计算 fitness 差值
        for param in self.PARAM_BOUNDS:
            changes = []
            fitness_diffs = []
            
            for i in range(1, len(self.param_history)):
                prev_entry = self.param_history[i - 1]
                curr_entry = self.param_history[i]
                
                old = curr_entry.get("old", {})
                new = curr_entry.get("new", {})
                if param not in old or param not in new:
                    continue
                if old.get(param) is None or new.get(param) is None:
                    continue
                
                prev_fitness = prev_entry.get("fitness")
                curr_fitness = curr_entry.get("fitness")
                
                if prev_fitness is None or curr_fitness is None:
                    continue
                
                param_change = new[param] - old[param]
                fitness_diff = curr_fitness - prev_fitness  # 负值更好（fitness 越小越好）
                
                if abs(param_change) > 0.001:  # 只看有变化的
                    changes.append(param_change)
                    fitness_diffs.append(fitness_diff)
            
            if len(changes) >= 3:
                # 计算皮尔逊相关系数
                n = len(changes)
                mean_c = sum(changes) / n
                mean_f = sum(fitness_diffs) / n
                
                num = sum((c - mean_c) * (f - mean_f) for c, f in zip(changes, fitness_diffs))
                den_c = sum((c - mean_c) ** 2 for c in changes)
                den_f = sum((f - mean_f) ** 2 for f in fitness_diffs)
                
                if den_c > 0 and den_f > 0:
                    correlation = num / (den_c * den_f) ** 0.5
                    
                    # 判断趋势方向：correlation < 0 表示参数增大时 fitness 减小（改善）
                    correlations[param] = {
                        "correlation": correlation,
                        "direction": -1 if correlation < 0 else 1,  # -1表示增大改善
                        "confidence": abs(correlation),
                        "sample_count": n
                    }
        
        return correlations
    
    def _extract_range_rules(self) -> List[str]:
        """从成功区域提取参数范围建议"""
        rules = []
        
        for param in self.PARAM_BOUNDS:
            values = [region.get(param) for region in self.success_regions 
                     if param in region and region[param] is not None]
            
            if len(values) >= 5:
                min_val = min(values)
                max_val = max(values)
                mean_val = sum(values) / len(values)
                
                # 如果范围比较集中，给出建议
                param_range = self.PARAM_BOUNDS[param][1] - self.PARAM_BOUNDS[param][0]
                if param_range > 0:
                    spread = (max_val - min_val) / param_range
                    if spread < 0.3:  # 成功值集中在较小范围内
                        rules.append(f"{param} 建议保持在 {min_val:.2f}~{max_val:.2f} 范围内（成功率高）")
        
        return rules[:3]

    def _build_physical_explanation(self, param: str, direction_word: str) -> str:
        """根据参数和方向给出简短物理机理说明，避免空泛"""
        # 默认泛化描述
        generic = "在保持线圈匝数不变时，调整该尺寸会改变磁路长度/漏磁，从而影响主磁通与推力"
        # 针对常见参数补充物理直觉
        physics_map = {
            "dg": "减小齿间距可降低磁路长度与漏磁，主磁通更集中，推力系数 kb 有望提升",
            "hs": "减小槽高可缩短磁路、降低漏磁；但需注意 2*dg+tb-hs 约束，防止几何失效",
            "ta": "增大轭厚可缓解饱和，减小则易降低磁通承载能力",
            "tm": "减小磁铁厚度会降低磁势，增大会提高磁通但增加质量",
            "lm": "增大磁铁长度可增加有效磁路面积，提升磁密，但会增加质量与体积",
        }
        # 特别强调“保持 n2 不变”的场景
        coil_note = "在保证 n2 不变的条件下，"
        core = physics_map.get(param, generic)
        if param in {"dg", "hs", "ta", "tm", "lm"}:
            return coil_note + core
        return core

    def _load_strategy_snippets(self, limit: int = 6) -> List[str]:
        """
        从评论家策略库 + 主 Agent 策略库抽取可复用的物理/约束类规则，辅助元学习。
        仅提取含有参数/物理含义的片段，避免泛化语句。
        """
        snippets: List[str] = []
        import glob
        import json
        from pathlib import Path
        
        base = Path("critic_experience")
        for path in glob.glob(str(base / "*_strategy.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for s in data.get("strategies", []):
                    # 简单过滤：包含参数名或物理关键词才收
                    if any(key in s for key in ["lm", "tm", "ta", "dg", "hs", "wslot", "hslot", "s", "kb", "磁", "饱和", "漏磁"]):
                        snippets.append(f"评论家经验：{s}")
            except Exception:
                continue

        # 主 Agent 策略库（self.storage_path / strategy_state.json）
        try:
            strategy_path = Path(self.storage_path)
            if strategy_path.exists():
                with open(strategy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for s in data.get("learned_rules", []):
                    if any(key in s for key in ["lm", "tm", "ta", "dg", "hs", "wslot", "hslot", "s", "kb", "磁", "饱和", "漏磁"]):
                        snippets.append(f"主策略库：{s}")
        except Exception:
            pass

        # 去重并截断
        dedup = []
        seen = set()
        for s in snippets:
            if s not in seen:
                dedup.append(s)
                seen.add(s)
            if len(dedup) >= limit:
                break
        return dedup
    
    def _extract_combo_rules(self) -> List[str]:
        """识别参数组合规律"""
        rules = []
        
        # 分析最佳 fitness 对应的参数特征
        best_entries = sorted(
            [e for e in self.param_history if e.get("fitness") is not None and e.get("success")],
            key=lambda x: x["fitness"]
        )[:5]
        
        if len(best_entries) >= 3:
            # 找出最佳配置的共同特征
            common_high = []  # 偏高的参数
            common_low = []   # 偏低的参数
            
            for param in self.PARAM_BOUNDS:
                low, high = self.PARAM_BOUNDS[param]
                if high == low:
                    continue
                    
                values = [e["new"].get(param) for e in best_entries 
                         if e["new"].get(param) is not None]
                if len(values) < 3:
                    continue
                    
                avg = sum(values) / len(values)
                rel_pos = (avg - low) / (high - low)
                
                if rel_pos > 0.7:
                    common_high.append(f"{param}({avg:.2f})")
                elif rel_pos < 0.3:
                    common_low.append(f"{param}({avg:.2f})")
            
            if common_high:
                rules.append(f"最佳配置中通常偏高的参数：{', '.join(common_high)}")
            if common_low:
                rules.append(f"最佳配置中通常偏低的参数：{', '.join(common_low)}")
        
        return rules[:2]
    
    # ==================== 辅助方法 ====================
    
    def _params_similar(self, p1: Dict[str, float], p2: Dict[str, float], threshold: float = None) -> bool:
        """判断两个参数组合是否相似"""
        if threshold is None:
            threshold = self.config.pattern_similarity_threshold
        
        total_diff = 0
        count = 0
        for param in self.PARAM_BOUNDS:
            if param in p1 and param in p2:
                low, high = self.PARAM_BOUNDS[param]
                if high > low:
                    diff = abs(p1[param] - p2[param]) / (high - low)
                    total_diff += diff
                    count += 1
        
        if count == 0:
            return False
        
        avg_diff = total_diff / count
        return avg_diff < threshold
    
    def _identify_key_factors(self, state: Dict[str, float], is_success: bool) -> List[str]:
        """识别关键因素"""
        factors = []
        
        for param, value in state.items():
            if param not in self.PARAM_BOUNDS:
                continue
            low, high = self.PARAM_BOUNDS[param]
            if high == low:
                continue
            
            # 相对位置
            rel_pos = (value - low) / (high - low)
            
            if rel_pos > 0.8:
                factors.append(f"{param} 较高 ({value:.2f})")
            elif rel_pos < 0.2:
                factors.append(f"{param} 较低 ({value:.2f})")
        
        return factors[:5]
    
    def _generate_avoid_rules(self, state: Dict[str, float], errors: List[str]) -> List[str]:
        """根据失败状态和错误生成避免规则"""
        rules = []
        
        for error in errors:
            # 解析错误，识别相关参数
            error_lower = error.lower()
            
            if "dg" in error_lower or "twall" in error_lower:
                if state.get("dg", 0) < 0.35:
                    rules.append(f"dg 不要低于 0.35（当前 {state.get('dg', 0):.2f} 导致失败）")
            
            if "n2" in error_lower:
                if state.get("hs", 0) < 1.5:
                    rules.append(f"hs 低于 1.5 时 n2 难以达标（当前 hs={state.get('hs', 0):.2f}）")
            
            if "hslot" in error_lower and "tb" in error_lower:
                rules.append(f"hslot 与 tb 的差值需要足够大")
            
            if "2dg + tb" in error_lower:
                rules.append(f"增大 dg 或减小 hs 以满足 2dg + tb - hs ≥ 0.1")
        
        # 如果没有具体规则，生成通用规则
        if not rules:
            for param, value in state.items():
                if param in self.PARAM_BOUNDS:
                    low, high = self.PARAM_BOUNDS[param]
                    if high > low:
                        rel_pos = (value - low) / (high - low)
                        if rel_pos < 0.1 or rel_pos > 0.9:
                            rules.append(f"避免 {param}={value:.2f}（边界值）")
        
        return rules[:3]
    
    def _find_params_related_to_error(self, error: str) -> str:
        """找出与错误相关的参数"""
        error_lower = error.lower()
        related = []
        
        param_keywords = {
            "dg": ["dg", "气隙", "twall"],
            "hs": ["hs", "hslot", "n2", "2dg"],
            "ta": ["ta", "tb"],
            "hslot": ["hslot", "tb"],
            "wslot": ["wslot", "wa", "ws"],
            "s": ["行程", "ls", "lm"],
        }
        
        for param, keywords in param_keywords.items():
            for kw in keywords:
                if kw in error_lower:
                    related.append(param)
                    break
        
        if related:
            return ", ".join(set(related))
        return ""
    
    def _find_common_features(self) -> List[str]:
        """找出成功模式的共同特征"""
        if len(self.success_patterns) < 3:
            return []
        
        features = []
        param_ranges: Dict[str, List[float]] = defaultdict(list)
        
        for sp in self.success_patterns:
            for param, value in sp.params.items():
                param_ranges[param].append(value)
        
        for param, values in param_ranges.items():
            if len(values) >= 3:
                avg = sum(values) / len(values)
                std = (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5
                
                low, high = self.PARAM_BOUNDS.get(param, (0, 1))
                range_size = high - low if high > low else 1
                
                # 如果标准差较小，说明成功案例在这个参数上比较集中
                if std / range_size < 0.15:
                    features.append(f"{param} ≈ {avg:.2f} (集中)")
        
        return features
    
    # ==================== 提示词生成 ====================
    
    def build_strategy_prompt(self, strategy_only: bool = False, transfer_mode: str = None) -> str:
        """构建策略相关的提示词
        
        Args:
            strategy_only: 若为 True（经验迁移场景），只输出结论性语句，不输出具体参数值
            transfer_mode: 迁移模式 - 'distilled' 时仅注入蒸馏原则，跳过任务特定内容
        """
        use_distilled = (transfer_mode in ("distilled",) and self.distilled_principles is not None)
        
        parts = []
        
        # 0. 设计变量可行域（始终需要，确保 LLM 使用正确的范围）
        parts.append("【设计变量可行域】")
        parts.append("请在以下范围内选择参数值：")
        for param, (low, high) in self.PARAM_BOUNDS.items():
            if param == "wa":
                parts.append(f"  - {param}: 固定值 {low}")
            elif param == "tm":
                parts.append(f"  - {param}: [{low}, {high}] (连续变量，可取任意值)")
            else:
                parts.append(f"  - {param}: [{low}, {high}]")
        parts.append("  - tb_ratio: [1.5, 2.0]")
        parts.append("")
        
        # ★ 蒸馏原则注入（跨任务迁移的核心）
        if self.distilled_principles is not None:
            parts.append(self.distilled_principles.to_prompt())
            parts.append("")
            # hybrid 模式：同时有参数数据和物理原则，指导 LLM 以原则为主框架
            if transfer_mode == "hybrid":
                parts.append("⚠️ **混合迁移模式指令**：")
                parts.append("你同时获得了「蒸馏物理原则（L2/L3）」和「历史参数数据」两类知识。请遵循以下规则：")
                parts.append("1. **推理框架**：每个设计决策必须首先基于 L2/L3 物理原则进行推理，解释物理机制")
                parts.append("2. **参数参考**：历史参数数据可作为数值锚点和目标参考（如目标 n2 范围、fitness 可达水平）")
                parts.append("3. **输出格式**：在你的推理中必须标注引用来源，如「根据迁移经验[L3物理定律#2]，减小气隙 → 磁阻下降 → 磁通提升，因此将 dg 从 0.40 → 0.35」")
                parts.append("4. **原则优先**：当参数数据与物理原则冲突时，以物理原则为准")
                parts.append("")
        
        # 在纯蒸馏模式下，跳过任务特定的成功模式/参数敏感性等（它们绑定了源任务参数名）
        if use_distilled:
            # 蒸馏模式：不注入源任务的参数绑定规则/模式/敏感性
            # 仅保留本轮新产生的规则（created_at > 运行启动时间的规则）
            if self.expel_enabled and self.rule_manager and self.rule_manager.rules:
                import time as _time
                run_start = getattr(self, '_run_start_time', _time.time())
                new_rules = [r for r in self.rule_manager.rules if r.created_at > run_start]
                if new_rules:
                    lines = ["【本轮学习规则】"]
                    for i, rule in enumerate(sorted(new_rules, key=lambda r: r.confidence, reverse=True)[:5], 1):
                        lines.append(f"  {i}. [{rule.confidence}] {rule.text}")
                    parts.append("\n".join(lines))
            return "\n".join(parts) if parts else ""
        
        # --- 以下为 raw / hybrid / 无迁移 的完整策略输出 ---
        
        # 0.5 元学习先验知识（跨任务迁移的知识）
        if self.enable_meta_learning and self.meta_prior_prompt:
            parts.append("【元学习先验知识】")
            parts.append(self.meta_prior_prompt)
        
        # 1. 动态提示词（结论性，保留）
        if self.prompt_additions:
            parts.append("\n【策略学习洞察】")
            for addition in self.prompt_additions:
                if addition != self.meta_prior_prompt:
                    parts.append(addition)
        
        # 2. 成功模式建议
        if self.success_patterns:
            parts.append("\n【成功模式参考】")
            best_patterns = sorted(self.success_patterns, key=lambda p: p.fitness)[:2]
            for i, sp in enumerate(best_patterns, 1):
                if strategy_only:
                    parts.append(f"  模式{i} (fitness={sp.fitness:.4f}, 出现{sp.frequency}次): 结论：该区域可行且较优，可在此附近探索")
                else:
                    params_str = ", ".join([f"{k}={v:.2f}" for k, v in list(sp.params.items())[:5]])
                    parts.append(f"  模式{i} (fitness={sp.fitness:.4f}, 出现{sp.frequency}次): {params_str}...")
        
        # 3. 失败模式警告（结论性，保留）
        if self.failure_patterns:
            parts.append("\n【失败模式警告】")
            worst_patterns = sorted(self.failure_patterns, key=lambda p: p.frequency, reverse=True)[:2]
            for fp in worst_patterns:
                if fp.avoid_rules:
                    parts.append(f"  ⚠️ {fp.avoid_rules[0]}")
        
        # 4. 参数敏感性
        high_sens = [(p, s) for p, s in self.param_sensitivity.items() if s > 0.5]
        if high_sens:
            high_sens.sort(key=lambda x: x[1], reverse=True)
            parts.append("\n【参数敏感性】")
            if strategy_only:
                parts.append(f"  高敏感参数: {', '.join([p for p, _ in high_sens[:4]])}（建议重点调整）")
            else:
                parts.append(f"  高敏感参数: {', '.join([f'{p}({s:.2f})' for p, s in high_sens[:4]])}")
                parts.append(f"  建议重点调整这些参数")
        
        # 5. 元学习探索建议
        if self.enable_meta_learning and self.meta_exploration_suggestions:
            parts.append("\n【跨任务学习建议】")
            for suggestion in self.meta_exploration_suggestions[:3]:
                parts.append(f"  • {suggestion}")
        
        # 6. ★ ExpeL 对比学习规则（带置信度）
        if self.expel_enabled and self.rule_manager and self.rule_manager.rules:
            expel_rules_text = self.rule_manager.get_rules_text(strategy_only=strategy_only)
            if expel_rules_text:
                parts.append(f"\n{expel_rules_text}")
        
        return "\n".join(parts) if parts else ""
    
    def get_avoidance_suggestions(self, proposed_state: Dict[str, float]) -> List[str]:
        """检查提议的参数是否接近失败区域，返回建议"""
        suggestions = []
        
        for fp in self.failure_patterns:
            if self._params_similar(proposed_state, fp.params, threshold=0.12):
                for rule in fp.avoid_rules:
                    suggestions.append(f"⚠️ 接近失败区域：{rule}")
        
        return suggestions[:3]
    
    # ==================== 状态管理 ====================
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略状态信息"""
        recent_success_rate = (
            sum(self.recent_results) / len(self.recent_results)
            if self.recent_results else 0
        )
        
        # 基础信息
        info = {
            "epsilon": self.epsilon,
            "iteration": self.iteration,
            "recent_success_rate": recent_success_rate,
            "success_regions_count": len(self.success_regions),
            "success_patterns_count": len(self.success_patterns),
            "failure_patterns_count": len(self.failure_patterns),
            "learned_rules_count": len(self.learned_rules),
            "exploration_weights": self.config.exploration_weights,
            # 元学习信息
            "meta_learning_enabled": self.enable_meta_learning,
            "meta_knowledge_summary": self.get_meta_knowledge_summary() if self.enable_meta_learning else {},
            # ExpeL 对比学习信息
            "expel_enabled": self.expel_enabled,
            "expel_rules_count": len(self.rule_manager.rules) if self.expel_enabled and self.rule_manager else 0,
            "expel_summary": self.rule_manager.get_summary() if self.expel_enabled and self.rule_manager else {},
        }
        
        return info
    
    def get_full_state(self) -> Dict[str, Any]:
        """获取完整状态（用于 Web 展示）"""
        recent_success_rate = (
            sum(self.recent_results) / len(self.recent_results)
            if self.recent_results else 0
        )
        return {
            "epsilon": round(self.epsilon, 4),
            "iteration": self.iteration,
            "recent_success_rate": round(recent_success_rate, 3),
            "recent_results": self.recent_results[-20:],
            "last_direction": self.last_direction,
            "success_regions": self.success_regions[-10:],
            "success_regions_count": len(self.success_regions),
            "success_patterns": [sp.to_dict() for sp in self.success_patterns[-10:]],
            "success_patterns_count": len(self.success_patterns),
            "failure_patterns": [fp.to_dict() for fp in self.failure_patterns[-10:]],
            "failure_patterns_count": len(self.failure_patterns),
            "param_sensitivity": self.param_sensitivity,
            "prompt_additions": self.prompt_additions,
            "learned_rules": self.learned_rules,
            "config": {
                "initial_epsilon": self.config.initial_epsilon,
                "min_epsilon": self.config.min_epsilon,
                "epsilon_decay": self.config.epsilon_decay,
                "exploration_weights": self.config.exploration_weights,
            }
        }
    
    def clamp_to_bounds(self, state: Dict[str, float]) -> Dict[str, float]:
        """将参数裁剪到可行域"""
        clamped = {}
        for param, value in state.items():
            if param in self.PARAM_BOUNDS:
                low, high = self.PARAM_BOUNDS[param]
                clamped[param] = round(max(low, min(high, value)), 2)
            else:
                clamped[param] = value
        return clamped
    
    def suggest_param_focus(self, sensitivities: Optional[Dict[str, float]] = None) -> List[str]:
        """根据敏感性建议重点关注的参数"""
        sens = sensitivities or self.param_sensitivity
        if not sens:
            return []
        
        sorted_params = sorted(sens.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_params[:3]]
    
    def reset(self):
        """重置策略状态"""
        self.epsilon = self.config.initial_epsilon
        self.iteration = 0
        self.recent_results = []
        self.last_direction = {}
        self.success_regions = []
        self.success_patterns = []
        self.failure_patterns = []
        self.param_sensitivity = {p: 0.5 for p in self.PARAM_BOUNDS}
        self.param_history = []
        self.prompt_additions = []
        self.learned_rules = []
        self._save()
        logger.info("[RL] 策略状态已完全重置")
    
    # ==================== 元学习支持 ====================
    
    def _init_meta_learning(self, meta_knowledge_path: str):
        """初始化元学习组件"""
        if not META_LEARNING_AVAILABLE:
            return
        
        try:
            # 从环境变量获取 LLM 配置
            llm_api_key = os.environ.get("OPENAI_API_KEY")
            llm_base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            llm_model = os.environ.get("META_LLM_MODEL", "openai/gpt-4o-mini")
            
            self.meta_agent = MetaLearningAgent(
                knowledge_base_path=meta_knowledge_path,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                enable_llm_analysis=bool(llm_api_key)  # 有 API key 才启用
            )
            # 使用当前任务的参数边界设置
            self.meta_agent.setup_from_bounds(self.PARAM_BOUNDS, self.domain)
            
            # 尝试加载先验知识
            transfer_result = self.meta_agent.transfer_to_new_task(
                create_default_task_config(self.PARAM_BOUNDS, self.domain)
            )
            
            self.meta_prior_prompt = transfer_result.get("prior_prompt", "")
            self.meta_exploration_suggestions = transfer_result.get("exploration_suggestions", [])
            
            summary = self.meta_agent.get_knowledge_summary()
            llm_status = "✅" if llm_api_key else "❌"
            logger.info(
                f"🧠 元学习已初始化 | 规则: {summary['total_rules']} | "
                f"模式: {summary['total_patterns']} | 高置信度规则: {summary['high_confidence_rules']} | "
                f"LLM分析: {llm_status}"
            )
            
            # 如果有先验知识，添加到 prompt_additions
            if self.meta_prior_prompt:
                self.prompt_additions.append(self.meta_prior_prompt)
                
        except Exception as e:
            logger.warning(f"元学习初始化失败: {e}")
            self.meta_agent = None
    
    def _online_meta_extract(self) -> Optional[Dict[str, Any]]:
        """
        在线元知识提取 - 在优化过程中实时触发
        
        触发条件：
        1. 迭代次数是 meta_extract_interval 的倍数
        2. 积累了足够的经验（>= 5 条）
        
        Returns:
            提取结果，如果未触发则返回 None
        """
        # 获取提取间隔（默认每 5 轮提取一次）
        extract_interval = getattr(self.config, 'meta_extract_interval', 5)
        
        # 检查是否达到触发条件
        if self.iteration % extract_interval != 0:
            return None
        
        if len(self.param_history) < 5:
            return None
        
        logger.info(f"🧠 [在线元学习] 触发元知识提取 (第 {self.iteration} 轮)")
        
        # 调用提取
        result = self.extract_meta_knowledge()
        
        if "error" not in result:
            # 提取成功，更新 prompt_additions 中的元学习部分
            self._update_meta_prompt_in_additions()
            logger.info(
                f"🧠 [在线元学习] 提取完成 | 新规则: {result.get('extracted_rules', 0)} | "
                f"新模式: {result.get('generated_patterns', 0)}"
            )
        
        return result
    
    def _update_meta_prompt_in_additions(self):
        """更新 prompt_additions 中的元学习先验知识"""
        if not self.meta_prior_prompt:
            return
        
        # 移除旧的元学习 prompt（如果有）
        self.prompt_additions = [
            p for p in self.prompt_additions 
            if not p.startswith("## 从历史优化任务中学习到的先验知识")
        ]
        
        # 添加新的元学习 prompt
        self.prompt_additions.append(self.meta_prior_prompt)

    def extract_meta_knowledge(self, experiences: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        从当前经验中提取元知识
        
        Args:
            experiences: 经验列表，如果为 None 则使用 param_history
        
        Returns:
            提取结果摘要
        """
        if not self.enable_meta_learning or not self.meta_agent:
            return {"error": "元学习未启用"}
        
        # 如果没有传入经验，使用 param_history
        if experiences is None:
            experiences = [
                {
                    "params": h.get("new", {}),
                    "fitness": h.get("fitness", 0),
                    "success": h.get("success", True),
                    "result": {"status": "ok" if h.get("success", True) else "failed"},
                }
                for h in self.param_history
            ]
        
        if len(experiences) < 5:
            return {"error": f"经验不足 ({len(experiences)} < 5)"}
        
        try:
            result = self.meta_agent.learn_from_experiences(experiences)
            
            # 更新先验 prompt
            if self.meta_agent.knowledge_base:
                rules = self.meta_agent.knowledge_base.get_applicable_rules(min_confidence=0.4)
                patterns = self.meta_agent.knowledge_base.get_applicable_patterns(self.domain, min_confidence=0.4)
                self.meta_prior_prompt = self.meta_agent._build_prior_prompt(rules, patterns)
            
            logger.info(f"🧠 元知识提取完成 | 新规则: {result['extracted_rules']} | 新模式: {result['generated_patterns']}")
            return result
            
        except Exception as e:
            logger.error(f"元知识提取失败: {e}")
            return {"error": str(e)}
    
    def get_meta_prior_prompt(self) -> str:
        """获取元学习先验知识 prompt"""
        return self.meta_prior_prompt
    
    def get_meta_exploration_suggestions(self) -> List[str]:
        """获取基于元知识的探索建议"""
        return self.meta_exploration_suggestions
    
    def get_normalized_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """获取归一化后的参数（用于跨任务比较）"""
        if not self.enable_meta_learning or not self.meta_agent or not self.meta_agent.normalizer:
            return {}
        
        normalized = self.meta_agent.normalizer.normalize(params)
        return {name: np.normalized for name, np in normalized.items()}
    
    def get_meta_knowledge_summary(self) -> Dict[str, Any]:
        """获取元知识库摘要"""
        if not self.enable_meta_learning or not self.meta_agent:
            return {"enabled": False}
        
        summary = self.meta_agent.get_knowledge_summary()
        summary["enabled"] = True
        summary["prior_prompt_length"] = len(self.meta_prior_prompt)
        summary["exploration_suggestions"] = len(self.meta_exploration_suggestions)
        return summary
    
    def export_for_transfer(self) -> Dict[str, Any]:
        """
        导出当前任务的知识，用于迁移到新任务
        
        Returns:
            可迁移的知识包
        """
        # 先提取元知识
        self.extract_meta_knowledge()
        
        return {
            "domain": self.domain,
            "param_bounds": self.PARAM_BOUNDS,
            "param_sensitivity": self.param_sensitivity,
            "learned_rules": self.learned_rules,
            "success_patterns_count": len(self.success_patterns),
            "failure_patterns_count": len(self.failure_patterns),
            "meta_knowledge_summary": self.get_meta_knowledge_summary(),
            "meta_prior_prompt": self.meta_prior_prompt,
        }
    
    def _save(self):
        """保存策略状态"""
        try:
            state = {
                "epsilon": self.epsilon,
                "iteration": self.iteration,
                "recent_results": self.recent_results,
                "last_direction": self.last_direction,
                "success_regions": self.success_regions,
                "success_patterns": [sp.to_dict() for sp in self.success_patterns],
                "failure_patterns": [fp.to_dict() for fp in self.failure_patterns],
                "param_sensitivity": self.param_sensitivity,
                "param_history": self.param_history[-self.config.sensitivity_window:],
                "prompt_additions": self.prompt_additions,
                "learned_rules": self.learned_rules,
                "n1_history": self.n1_history[-100:],  # ★ 保存离散变量历史
                "n2_history": self.n2_history[-100:],
                "config": {
                    "initial_epsilon": self.config.initial_epsilon,
                    "min_epsilon": self.config.min_epsilon,
                    "epsilon_decay": self.config.epsilon_decay,
                    "exploration_weights": self.config.exploration_weights,
                    "perturbation_scale": self.config.perturbation_scale,
                    "success_boost": self.config.success_boost,
                    "failure_penalty": self.config.failure_penalty,
                }
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存策略状态失败: {e}")
    
    def _load(self):
        """加载策略状态"""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            self.epsilon = state.get("epsilon", self.config.initial_epsilon)
            self.iteration = state.get("iteration", 0)
            self.recent_results = state.get("recent_results", [])
            self.last_direction = state.get("last_direction", {})
            self.success_regions = state.get("success_regions", [])
            
            # 加载成功模式
            self.success_patterns = [
                SuccessPattern.from_dict(d) for d in state.get("success_patterns", [])
            ]
            
            # 加载失败模式
            self.failure_patterns = [
                FailurePattern.from_dict(d) for d in state.get("failure_patterns", [])
            ]
            
            self.param_sensitivity = state.get("param_sensitivity", self.param_sensitivity)
            self.param_history = state.get("param_history", [])
            self.prompt_additions = state.get("prompt_additions", [])
            self.learned_rules = state.get("learned_rules", [])
            
            # ★ 加载离散变量历史
            self.n1_history = state.get("n1_history", [])
            self.n2_history = state.get("n2_history", [])
            
            if "config" in state:
                cfg = state["config"]
                self.config.exploration_weights = cfg.get(
                    "exploration_weights", self.config.exploration_weights
                )
            
            logger.info(f"[RL] 已加载策略状态: epsilon={self.epsilon:.3f}, "
                       f"iteration={self.iteration}, "
                       f"成功模式={len(self.success_patterns)}, "
                       f"失败模式={len(self.failure_patterns)}, "
                       f"学习规则={len(self.learned_rules)}")
        except Exception as e:
            logger.warning(f"加载策略状态失败: {e}")
