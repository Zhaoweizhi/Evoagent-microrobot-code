"""
元学习模块：实现跨任务知识迁移

核心功能：
1. 参数归一化：将具体参数值映射到 [0,1] 区间
2. 抽象规则：从具体经验中提取通用模式（统计 + LLM 混合）
3. 元知识库：存储可迁移的优化知识
4. 任务适配：将元知识应用到新任务
5. LLM 深度分析：理解物理机理，生成高层抽象规则
"""

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)

# LLM 客户端
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI 未安装，LLM 元知识分析功能不可用")


# ========== 参数类型和角色定义 ==========

class ParamCategory(Enum):
    """参数类型分类"""
    LENGTH = "length"           # 长度类：la, ha, ws, ls, lm, tm 等
    RATIO = "ratio"             # 比例类：tb_ratio, 各种比值
    GAP = "gap"                 # 间隙类：dg, twall, clearance
    COUNT = "count"             # 计数类：n1, n2, 匝数
    ANGLE = "angle"             # 角度类
    MATERIAL = "material"       # 材料属性类
    OTHER = "other"


class ParamRole(Enum):
    """参数在优化中的角色"""
    PRIMARY_DIMENSION = "primary_dimension"     # 主尺寸：决定整体大小
    SECONDARY_DIMENSION = "secondary_dimension" # 次尺寸：细节尺寸
    ASPECT_RATIO = "aspect_ratio"               # 长宽比类
    CLEARANCE = "clearance"                     # 间隙/余量
    WINDING = "winding"                         # 绕组相关
    MAGNETIC = "magnetic"                       # 磁路相关
    STRUCTURAL = "structural"                   # 结构相关
    OTHER = "other"


# ========== 数据结构定义 ==========

@dataclass
class NormalizedParam:
    """归一化后的参数表示"""
    name: str                           # 原始参数名
    raw_value: float                    # 原始值
    normalized: float                   # 归一化值 [0,1]
    category: str                       # 参数类型
    role: str                           # 参数角色
    min_bound: float                    # 下界
    max_bound: float                    # 上界
    sensitivity: float = 0.5           # 敏感度（初始中等）


@dataclass
class AbstractRule:
    """抽象规则（可跨任务迁移）"""
    rule_id: str
    rule_type: str                      # monotonic_effect, optimal_range, correlation, constraint
    param_category: str                 # 适用的参数类型
    param_role: str                     # 适用的参数角色
    direction: Optional[str] = None     # increase/decrease
    effect: Optional[str] = None        # improve_fitness/degrade_fitness/violate_constraint
    optimal_range: Optional[Tuple[float, float]] = None  # 最优归一化范围
    confidence: float = 0.5
    sample_count: int = 0               # 支撑样本数
    context: Optional[str] = None       # 适用条件描述
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class MetaPattern:
    """元模式（从多条规则中提取的更高层模式）"""
    pattern_id: str
    description: str                    # 自然语言描述
    applicable_domains: List[str]       # 适用领域
    conditions: List[str]               # 适用条件
    recommendations: List[str]          # 推荐操作
    anti_patterns: List[str]            # 反模式（应避免的操作）
    confidence: float = 0.5
    source_rules: List[str] = field(default_factory=list)  # 来源规则ID


@dataclass 
class TaskConfig:
    """任务配置（描述一个优化任务）"""
    task_name: str
    domain: str                         # 领域：electromagnetic_actuator, motor, etc.
    parameters: List[Dict[str, Any]]    # 参数定义列表
    constraints: List[Dict[str, Any]]   # 约束定义列表
    objectives: List[Dict[str, Any]]    # 目标定义列表
    
    @classmethod
    def from_yaml(cls, path: str) -> "TaskConfig":
        """从 YAML 文件加载"""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TaskConfig":
        """从字典加载"""
        return cls(**data)


# ========== 参数类型自动识别 ==========

# 参数名到类型的映射规则
PARAM_CATEGORY_RULES = {
    # 长度类
    ParamCategory.LENGTH: [
        "la", "ha", "ws", "ls", "lm", "tm", "ta", "tb", "hs", "wslot", "hslot",
        "wa", "twall", "length", "width", "height", "thickness"
    ],
    # 比例类
    ParamCategory.RATIO: [
        "ratio", "tb_ratio", "aspect", "proportion"
    ],
    # 间隙类
    ParamCategory.GAP: [
        "dg", "gap", "clearance", "spacing", "s"
    ],
    # 计数类
    ParamCategory.COUNT: [
        "n1", "n2", "turns", "count", "number", "layers"
    ],
}

PARAM_ROLE_RULES = {
    # 主尺寸
    ParamRole.PRIMARY_DIMENSION: ["la", "ha", "length", "width", "height"],
    # 次尺寸
    ParamRole.SECONDARY_DIMENSION: ["ws", "ls", "hs", "wslot", "hslot"],
    # 间隙
    ParamRole.CLEARANCE: ["dg", "twall", "gap", "clearance", "s"],
    # 绕组
    ParamRole.WINDING: ["n1", "n2", "turns", "lm", "tm"],
    # 磁路
    ParamRole.MAGNETIC: ["ta", "tb", "wa"],
    # 比例
    ParamRole.ASPECT_RATIO: ["ratio", "tb_ratio", "aspect"],
}


def infer_param_category(param_name: str) -> ParamCategory:
    """根据参数名推断类型"""
    name_lower = param_name.lower()
    for category, keywords in PARAM_CATEGORY_RULES.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return ParamCategory.OTHER


def infer_param_role(param_name: str) -> ParamRole:
    """根据参数名推断角色"""
    name_lower = param_name.lower()
    for role, keywords in PARAM_ROLE_RULES.items():
        for kw in keywords:
            if kw in name_lower:
                return role
    return ParamRole.OTHER


# ========== 元学习核心类 ==========

class MetaKnowledgeBase:
    """元知识库：存储和管理可迁移的优化知识"""
    
    def __init__(self, save_path: str = "meta_knowledge.json"):
        self.save_path = save_path
        self.abstract_rules: Dict[str, AbstractRule] = {}
        self.meta_patterns: Dict[str, MetaPattern] = {}
        self.domain_statistics: Dict[str, Dict] = {}  # 领域统计信息
        self._load()
    
    def _load(self):
        """加载已有的元知识"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 加载抽象规则
                for rule_id, rule_data in data.get("abstract_rules", {}).items():
                    self.abstract_rules[rule_id] = AbstractRule(**rule_data)
                
                # 加载元模式
                for pattern_id, pattern_data in data.get("meta_patterns", {}).items():
                    self.meta_patterns[pattern_id] = MetaPattern(**pattern_data)
                
                self.domain_statistics = data.get("domain_statistics", {})
                
                logger.info(f"📚 加载元知识库: {len(self.abstract_rules)} 条规则, {len(self.meta_patterns)} 条模式")
            except Exception as e:
                logger.warning(f"加载元知识库失败: {e}")
    
    def save(self):
        """保存元知识库"""
        try:
            data = {
                "abstract_rules": {k: asdict(v) for k, v in self.abstract_rules.items()},
                "meta_patterns": {k: asdict(v) for k, v in self.meta_patterns.items()},
                "domain_statistics": self.domain_statistics,
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"💾 元知识库已保存: {self.save_path}")
        except Exception as e:
            logger.warning(f"保存元知识库失败: {e}")
    
    def add_rule(self, rule: AbstractRule):
        """添加或更新抽象规则"""
        existing = self.abstract_rules.get(rule.rule_id)
        if existing:
            # 更新置信度（加权平均）
            total_samples = existing.sample_count + rule.sample_count
            if total_samples > 0:
                rule.confidence = (
                    existing.confidence * existing.sample_count + 
                    rule.confidence * rule.sample_count
                ) / total_samples
                rule.sample_count = total_samples
        
        self.abstract_rules[rule.rule_id] = rule
        self.save()
    
    def add_pattern(self, pattern: MetaPattern):
        """添加元模式"""
        self.meta_patterns[pattern.pattern_id] = pattern
        self.save()
    
    def get_applicable_rules(
        self, 
        param_category: Optional[str] = None,
        param_role: Optional[str] = None,
        min_confidence: float = 0.3
    ) -> List[AbstractRule]:
        """获取适用的规则"""
        rules = []
        for rule in self.abstract_rules.values():
            if rule.confidence < min_confidence:
                continue
            if param_category and rule.param_category != param_category:
                continue
            if param_role and rule.param_role != param_role:
                continue
            rules.append(rule)
        
        # 按置信度排序
        rules.sort(key=lambda r: r.confidence, reverse=True)
        return rules
    
    def get_applicable_patterns(
        self,
        domain: str,
        min_confidence: float = 0.3
    ) -> List[MetaPattern]:
        """获取适用的元模式"""
        patterns = []
        for pattern in self.meta_patterns.values():
            if pattern.confidence < min_confidence:
                continue
            if domain in pattern.applicable_domains or "all" in pattern.applicable_domains:
                patterns.append(pattern)
        
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns


class ParameterNormalizer:
    """参数归一化器"""
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        """
        Args:
            param_bounds: 参数边界 {param_name: (min, max)}
        """
        self.param_bounds = param_bounds
        self.param_info: Dict[str, Dict] = {}
        
        # 自动推断参数类型和角色
        for name, (min_val, max_val) in param_bounds.items():
            self.param_info[name] = {
                "category": infer_param_category(name).value,
                "role": infer_param_role(name).value,
                "min": min_val,
                "max": max_val,
            }
    
    def normalize(self, params: Dict[str, float]) -> Dict[str, NormalizedParam]:
        """将原始参数归一化"""
        result = {}
        for name, value in params.items():
            if name not in self.param_bounds:
                continue
            
            min_val, max_val = self.param_bounds[name]
            range_val = max_val - min_val
            
            if range_val > 0:
                normalized = (value - min_val) / range_val
                normalized = max(0.0, min(1.0, normalized))  # 裁剪到 [0,1]
            else:
                normalized = 0.5
            
            info = self.param_info[name]
            result[name] = NormalizedParam(
                name=name,
                raw_value=value,
                normalized=normalized,
                category=info["category"],
                role=info["role"],
                min_bound=min_val,
                max_bound=max_val,
            )
        
        return result
    
    def denormalize(self, normalized_params: Dict[str, float]) -> Dict[str, float]:
        """将归一化参数还原"""
        result = {}
        for name, norm_value in normalized_params.items():
            if name not in self.param_bounds:
                continue
            
            min_val, max_val = self.param_bounds[name]
            result[name] = min_val + norm_value * (max_val - min_val)
        
        return result
    
    def get_param_by_category(self, category: str) -> List[str]:
        """获取指定类型的所有参数名"""
        return [
            name for name, info in self.param_info.items()
            if info["category"] == category
        ]
    
    def get_param_by_role(self, role: str) -> List[str]:
        """获取指定角色的所有参数名"""
        return [
            name for name, info in self.param_info.items()
            if info["role"] == role
        ]


class MetaKnowledgeExtractor:
    """
    元知识提取器：从具体经验中提取通用规则
    
    采用混合方式：
    1. 统计分析：提取确定性的数值规律
    2. LLM 深度分析：理解物理机理，生成高层抽象规则
    """
    
    # LLM 分析 Prompt 模板
    LLM_ANALYSIS_PROMPT = """你是一位电磁设计优化专家。请基于以下从优化实验中提取的统计事实，进行 **因果关系分析**。

## 优化任务
领域：{domain}
参数数量：{param_count}
样本数量：{sample_count}

## 统计事实

### 1. 参数-性能相关性
{correlation_facts}

### 2. 最优参数范围
{range_facts}

### 3. 参数协同关系
{synergy_facts}

### 4. 约束违反模式
{constraint_facts}

## 请完成以下 **因果关系分析**

重点分析：
1. **参数-约束因果关系**：哪些参数是影响特定约束（如磁饱和 B_max、体积、质量）的 **真正原因**，哪些参数是次要因素或无关因素
2. **影响方向**：参数的增大/减小会如何影响约束（缓解/加剧）
3. **因果机理**：从物理原理解释为什么某个参数会影响某个约束
4. **通用性判断**：这个因果关系在其他类似电磁设计中是否也成立

请以 JSON 格式返回分析结果（不要添加 markdown 代码块）：

{{
  "causal_rules": [
    {{
      "constraint_name": "约束名称（如：磁饱和、体积、质量、装配间隙）",
      "primary_params": ["直接影响该约束的关键参数"],
      "secondary_params": ["有一定影响但非关键的参数"],
      "irrelevant_params": ["对该约束几乎无影响的参数"],
      "effects": [
        {{
          "param": "参数名",
          "direction": "increase/decrease",
          "effect": "alleviates/aggravates",
          "description": "参数增大/减小会缓解/加剧该约束"
        }}
      ],
      "physics_mechanism": "从物理原理解释这个因果关系",
      "generalizability": "high/medium/low，是否可推广到其他电磁设计",
      "confidence": 0.0到1.0的置信度
    }}
  ],
  "meta_rules": [
    {{
      "rule_name": "规则简称",
      "description": "规则的详细描述",
      "physics_reason": "从物理/工程角度解释为什么这个规律存在",
      "applicable_domains": ["适用的领域列表"],
      "param_categories": ["适用的参数类型：length/ratio/gap/count"],
      "recommendations": ["具体的操作建议"],
      "anti_patterns": ["应该避免的操作"],
      "confidence": 0.0到1.0的置信度
    }}
  ],
  "optimization_insights": [
    "关于整体优化策略的洞察"
  ],
  "transfer_suggestions": [
    "关于如何将这些知识迁移到新任务的建议"
  ]
}}

要求：
1. **因果优先**：区分相关性和因果性，指出哪些参数是真正的原因
2. **通用性**：规则要足够抽象，能迁移到类似的优化任务
3. **可操作**：给出明确的参数调整方向（增大/减小）
4. **物理解释**：用电磁学/机械原理解释因果机理
5. 至少分析 2-3 个主要约束的因果关系
"""

    def __init__(
        self, 
        normalizer: ParameterNormalizer,
        knowledge_base: MetaKnowledgeBase,
        domain: str = "electromagnetic_actuator",
        llm_client: Optional[Any] = None,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        enable_llm_analysis: bool = True
    ):
        self.normalizer = normalizer
        self.knowledge_base = knowledge_base
        self.domain = domain
        
        # LLM 配置
        self.enable_llm_analysis = enable_llm_analysis and OPENAI_AVAILABLE
        self.llm_model = llm_model
        self.llm_client = llm_client
        
        # 如果没有传入客户端，尝试从环境变量创建
        if self.enable_llm_analysis and self.llm_client is None:
            try:
                api_key = llm_api_key or os.environ.get("OPENAI_API_KEY")
                base_url = llm_base_url or os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
                if api_key:
                    self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
                    logger.info(f"🤖 元知识 LLM 分析已启用 | model={llm_model}")
                else:
                    self.enable_llm_analysis = False
                    logger.warning("未找到 OPENAI_API_KEY，LLM 元知识分析已禁用")
            except Exception as e:
                self.enable_llm_analysis = False
                logger.warning(f"LLM 客户端初始化失败: {e}")
    
    def extract_from_experiences(
        self, 
        experiences: List[Dict],
        min_samples: int = 5
    ) -> List[AbstractRule]:
        """
        从经验列表中提取抽象规则
        
        Args:
            experiences: 经验列表，每个包含 {params, fitness, success, ...}
            min_samples: 最小样本数要求
        
        Returns:
            提取的抽象规则列表
        """
        if len(experiences) < min_samples:
            logger.info(f"样本数不足 ({len(experiences)} < {min_samples})，跳过元知识提取")
            return []
        
        extracted_rules = []
        
        # 1. 提取单调性规则（参数变化方向与 fitness 的关系）
        monotonic_rules = self._extract_monotonic_rules(experiences)
        extracted_rules.extend(monotonic_rules)
        
        # 2. 提取最优范围规则
        range_rules = self._extract_optimal_range_rules(experiences)
        extracted_rules.extend(range_rules)
        
        # 3. 提取参数相关性规则
        correlation_rules = self._extract_correlation_rules(experiences)
        extracted_rules.extend(correlation_rules)
        
        # 4. 提取约束违反模式
        constraint_rules = self._extract_constraint_rules(experiences)
        extracted_rules.extend(constraint_rules)
        
        logger.info(f"📊 统计分析提取了 {len(extracted_rules)} 条规则")
        
        # ========== 5. LLM 深度分析（混合方式核心）==========
        llm_rules = []
        llm_insights = {}
        
        if self.enable_llm_analysis and len(extracted_rules) >= 2:
            try:
                # 收集统计事实
                stats_facts = self._collect_statistics_facts(
                    experiences, 
                    monotonic_rules, 
                    range_rules, 
                    correlation_rules, 
                    constraint_rules
                )
                
                # 调用 LLM 进行深度分析
                llm_result = self._llm_deep_analysis(stats_facts, experiences)
                
                if llm_result:
                    # 解析 LLM 生成的规则
                    llm_rules = self._parse_llm_rules(llm_result)
                    extracted_rules.extend(llm_rules)
                    
                    # 保存洞察
                    llm_insights = {
                        "optimization_insights": llm_result.get("optimization_insights", []),
                        "transfer_suggestions": llm_result.get("transfer_suggestions", []),
                    }
                    
                    logger.info(f"🤖 LLM 分析生成了 {len(llm_rules)} 条高层规则")
                    
            except Exception as e:
                logger.warning(f"LLM 元知识分析失败: {e}")
        
        # 保存到知识库
        for rule in extracted_rules:
            self.knowledge_base.add_rule(rule)
        
        # 保存 LLM 洞察到知识库的统计信息中
        if llm_insights:
            self.knowledge_base.domain_statistics[self.domain] = {
                "optimization_insights": llm_insights.get("optimization_insights", []),
                "transfer_suggestions": llm_insights.get("transfer_suggestions", []),
                "last_analysis": datetime.now().isoformat(),
            }
            self.knowledge_base.save()
        
        logger.info(f"🧠 总共提取了 {len(extracted_rules)} 条抽象规则（统计: {len(extracted_rules) - len(llm_rules)}, LLM: {len(llm_rules)}）")
        return extracted_rules
    
    # ==================== LLM 分析相关方法 ====================
    
    def _collect_statistics_facts(
        self,
        experiences: List[Dict],
        monotonic_rules: List[AbstractRule],
        range_rules: List[AbstractRule],
        correlation_rules: List[AbstractRule],
        constraint_rules: List[AbstractRule]
    ) -> Dict[str, Any]:
        """收集统计事实，供 LLM 分析使用"""
        
        # 1. 相关性事实
        correlation_facts = []
        for rule in monotonic_rules:
            dir_cn = "增大" if rule.direction == "increase" else "减小"
            effect_cn = "改善" if rule.effect == "improve_fitness" else "恶化"
            correlation_facts.append(
                f"- {rule.param_category} 类参数{dir_cn}与 fitness {effect_cn}相关 "
                f"(置信度: {rule.confidence:.0%}, 样本数: {rule.sample_count})"
            )
        
        # 2. 最优范围事实
        range_facts = []
        for rule in range_rules:
            if rule.optimal_range:
                lo, hi = rule.optimal_range
                range_facts.append(
                    f"- {rule.param_category} 类参数在归一化范围 [{lo:.0%}, {hi:.0%}] 内表现最佳 "
                    f"(样本数: {rule.sample_count})"
                )
        
        # 3. 协同关系事实
        synergy_facts = []
        for rule in correlation_rules:
            if rule.rule_type == "correlation":
                dir_cn = "正" if rule.direction == "positive" else "负"
                synergy_facts.append(
                    f"- {rule.param_category} 存在{dir_cn}相关性 "
                    f"(置信度: {rule.confidence:.0%})"
                )
        
        # 4. 约束违反事实
        constraint_facts = []
        for rule in constraint_rules:
            if rule.optimal_range:
                lo, hi = rule.optimal_range
                constraint_facts.append(
                    f"- {rule.param_category} 类参数在 [{lo:.0%}, {hi:.0%}] 范围容易违反约束 "
                    f"({rule.context})"
                )
        
        # 5. 计算基本统计
        fitness_values = [
            exp.get("fitness") or exp.get("reward", 0) 
            for exp in experiences 
            if exp.get("fitness") is not None or exp.get("reward") is not None
        ]
        
        success_count = sum(
            1 for exp in experiences 
            if exp.get("success", True) and exp.get("result", {}).get("status") != "constraint_violation"
        )
        
        return {
            "domain": self.domain,
            "param_count": len(self.normalizer.param_bounds),
            "sample_count": len(experiences),
            "success_rate": success_count / len(experiences) if experiences else 0,
            "fitness_range": (min(fitness_values), max(fitness_values)) if fitness_values else (0, 0),
            "correlation_facts": "\n".join(correlation_facts) if correlation_facts else "暂无数据",
            "range_facts": "\n".join(range_facts) if range_facts else "暂无数据",
            "synergy_facts": "\n".join(synergy_facts) if synergy_facts else "暂无数据",
            "constraint_facts": "\n".join(constraint_facts) if constraint_facts else "暂无数据",
        }
    
    def _llm_deep_analysis(self, stats_facts: Dict[str, Any], experiences: List[Dict]) -> Optional[Dict]:
        """调用 LLM 进行深度分析"""
        if not self.llm_client:
            return None
        
        # 构建 prompt
        prompt = self.LLM_ANALYSIS_PROMPT.format(**stats_facts)
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一位电磁设计优化专家，擅长从实验数据中提取通用的设计规律。请始终返回有效的 JSON 格式。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.3,  # 较低温度保证一致性
                )
                
                raw_response = response.choices[0].message.content or ""
                
                # 解析 JSON
                result = self._parse_llm_json(raw_response)
                if result:
                    return result
                    
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"LLM 分析失败 (尝试 {attempt + 1})，重试中... | {e}")
                    time.sleep(1)
                else:
                    logger.error(f"LLM 分析最终失败: {e}")
        
        return None
    
    def _parse_llm_json(self, raw_response: str) -> Optional[Dict]:
        """解析 LLM 返回的 JSON"""
        # 清理 markdown 代码块
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 部分
        try:
            match = re.search(r'\{[\s\S]*\}', raw_response)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        
        logger.warning(f"无法解析 LLM 返回的 JSON: {raw_response[:200]}...")
        return None
    
    def _parse_llm_rules(self, llm_result: Dict) -> List[AbstractRule]:
        """将 LLM 生成的规则转换为 AbstractRule 对象"""
        rules = []
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # 1. 解析因果规则（新增）
        causal_rules = llm_result.get("causal_rules", [])
        for i, causal_data in enumerate(causal_rules):
            try:
                constraint_name = causal_data.get("constraint_name", "unknown")
                primary_params = causal_data.get("primary_params", [])
                effects = causal_data.get("effects", [])
                
                # 为每个因果规则创建一条抽象规则
                rule = AbstractRule(
                    rule_id=f"causal_{self.domain}_{constraint_name}_{i}_{timestamp}",
                    rule_type="causal_relation",  # 新的规则类型
                    param_category="mixed",
                    param_role="causal",
                    confidence=causal_data.get("confidence", 0.7),
                    sample_count=0,
                    context=json.dumps({
                        "constraint_name": constraint_name,
                        "primary_params": primary_params,
                        "secondary_params": causal_data.get("secondary_params", []),
                        "irrelevant_params": causal_data.get("irrelevant_params", []),
                        "effects": effects,
                        "physics_mechanism": causal_data.get("physics_mechanism", ""),
                        "generalizability": causal_data.get("generalizability", "medium"),
                    }, ensure_ascii=False),
                )
                rules.append(rule)
                
            except Exception as e:
                logger.warning(f"解析因果规则失败: {e}")
                continue
        
        # 2. 解析元规则（原有逻辑）
        meta_rules = llm_result.get("meta_rules", [])
        for i, rule_data in enumerate(meta_rules):
            try:
                # 确定参数类型
                param_categories = rule_data.get("param_categories", [])
                param_category = param_categories[0] if param_categories else "mixed"
                
                # 创建抽象规则
                rule = AbstractRule(
                    rule_id=f"llm_{self.domain}_{i}_{timestamp}",
                    rule_type="llm_insight",
                    param_category=param_category,
                    param_role="mixed",
                    confidence=rule_data.get("confidence", 0.6),
                    sample_count=0,  # LLM 规则不基于样本
                    context=json.dumps({
                        "rule_name": rule_data.get("rule_name", ""),
                        "description": rule_data.get("description", ""),
                        "physics_reason": rule_data.get("physics_reason", ""),
                        "applicable_domains": rule_data.get("applicable_domains", []),
                        "recommendations": rule_data.get("recommendations", []),
                        "anti_patterns": rule_data.get("anti_patterns", []),
                    }, ensure_ascii=False),
                )
                rules.append(rule)
                
            except Exception as e:
                logger.warning(f"解析 LLM 规则失败: {e}")
                continue
        
        return rules
    
    def _extract_monotonic_rules(self, experiences: List[Dict]) -> List[AbstractRule]:
        """提取单调性规则：参数增大/减小与 fitness 改善的关系"""
        rules = []
        
        # 按参数类型和角色分组分析
        for category in ParamCategory:
            param_names = self.normalizer.get_param_by_category(category.value)
            if not param_names:
                continue
            
            # 收集该类型参数的变化与 fitness 变化的数据
            delta_fitness_by_direction = {"increase": [], "decrease": []}
            
            for i in range(1, len(experiences)):
                prev_exp = experiences[i - 1]
                curr_exp = experiences[i]
                
                prev_fitness = prev_exp.get("fitness") or prev_exp.get("reward", 0)
                curr_fitness = curr_exp.get("fitness") or curr_exp.get("reward", 0)
                
                if prev_fitness is None or curr_fitness is None:
                    continue
                
                fitness_delta = curr_fitness - prev_fitness
                
                prev_params = prev_exp.get("params") or prev_exp.get("state", {})
                curr_params = curr_exp.get("params") or curr_exp.get("state", {})
                
                # 计算该类型参数的平均变化方向
                param_deltas = []
                for pname in param_names:
                    if pname in prev_params and pname in curr_params:
                        # 归一化后的变化
                        prev_norm = self.normalizer.normalize({pname: prev_params[pname]})
                        curr_norm = self.normalizer.normalize({pname: curr_params[pname]})
                        if pname in prev_norm and pname in curr_norm:
                            delta = curr_norm[pname].normalized - prev_norm[pname].normalized
                            param_deltas.append(delta)
                
                if param_deltas:
                    avg_delta = statistics.mean(param_deltas)
                    if avg_delta > 0.05:  # 增大
                        delta_fitness_by_direction["increase"].append(fitness_delta)
                    elif avg_delta < -0.05:  # 减小
                        delta_fitness_by_direction["decrease"].append(fitness_delta)
            
            # 分析结果
            for direction, fitness_deltas in delta_fitness_by_direction.items():
                if len(fitness_deltas) < 3:
                    continue
                
                avg_fitness_change = statistics.mean(fitness_deltas)
                positive_ratio = sum(1 for d in fitness_deltas if d > 0) / len(fitness_deltas)
                
                # 如果有明显的单调关系
                if positive_ratio > 0.6:
                    effect = "improve_fitness"
                    confidence = positive_ratio
                elif positive_ratio < 0.4:
                    effect = "degrade_fitness"
                    confidence = 1 - positive_ratio
                else:
                    continue  # 关系不明显
                
                # 获取该类型参数的主要角色
                roles = set()
                for pname in param_names:
                    info = self.normalizer.param_info.get(pname, {})
                    roles.add(info.get("role", "other"))
                main_role = list(roles)[0] if len(roles) == 1 else "mixed"
                
                rule = AbstractRule(
                    rule_id=f"mono_{category.value}_{direction}_{self.domain}",
                    rule_type="monotonic_effect",
                    param_category=category.value,
                    param_role=main_role,
                    direction=direction,
                    effect=effect,
                    confidence=confidence,
                    sample_count=len(fitness_deltas),
                    context=f"基于 {self.domain} 领域 {len(fitness_deltas)} 个样本"
                )
                rules.append(rule)
        
        return rules
    
    def _extract_optimal_range_rules(self, experiences: List[Dict]) -> List[AbstractRule]:
        """提取最优范围规则：哪个归一化区间的 fitness 最好"""
        rules = []
        
        # 按参数类型分析
        for category in ParamCategory:
            param_names = self.normalizer.get_param_by_category(category.value)
            if not param_names:
                continue
            
            # 将归一化空间分成 5 个区间
            bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            bin_fitness: Dict[int, List[float]] = {i: [] for i in range(5)}
            
            for exp in experiences:
                params = exp.get("params") or exp.get("state", {})
                fitness = exp.get("fitness") or exp.get("reward", 0)
                
                if fitness is None:
                    continue
                
                # 计算该类型参数的平均归一化值
                norm_values = []
                for pname in param_names:
                    if pname in params:
                        norm = self.normalizer.normalize({pname: params[pname]})
                        if pname in norm:
                            norm_values.append(norm[pname].normalized)
                
                if norm_values:
                    avg_norm = statistics.mean(norm_values)
                    bin_idx = min(4, int(avg_norm * 5))
                    bin_fitness[bin_idx].append(fitness)
            
            # 找到最优区间
            best_bin = -1
            best_avg = float("-inf")
            
            for bin_idx, fitnesses in bin_fitness.items():
                if len(fitnesses) >= 2:
                    avg = statistics.mean(fitnesses)
                    if avg > best_avg:
                        best_avg = avg
                        best_bin = bin_idx
            
            if best_bin >= 0 and len(bin_fitness[best_bin]) >= 3:
                optimal_range = bins[best_bin]
                
                # 计算置信度
                total_samples = sum(len(f) for f in bin_fitness.values())
                confidence = len(bin_fitness[best_bin]) / total_samples if total_samples > 0 else 0.5
                
                rule = AbstractRule(
                    rule_id=f"range_{category.value}_{self.domain}",
                    rule_type="optimal_range",
                    param_category=category.value,
                    param_role="mixed",
                    optimal_range=optimal_range,
                    confidence=min(0.9, confidence + 0.3),  # 基础置信度
                    sample_count=len(bin_fitness[best_bin]),
                    context=f"{category.value} 类参数在归一化 {optimal_range} 范围内表现最佳"
                )
                rules.append(rule)
        
        return rules
    
    def _extract_correlation_rules(self, experiences: List[Dict]) -> List[AbstractRule]:
        """提取参数相关性规则：哪些参数应该协同变化"""
        rules = []
        
        # 收集参数变化数据
        param_deltas: Dict[str, List[float]] = {}
        fitness_deltas: List[float] = []
        
        for i in range(1, len(experiences)):
            prev_exp = experiences[i - 1]
            curr_exp = experiences[i]
            
            prev_fitness = prev_exp.get("fitness") or prev_exp.get("reward", 0)
            curr_fitness = curr_exp.get("fitness") or curr_exp.get("reward", 0)
            
            if prev_fitness is None or curr_fitness is None:
                continue
            
            fitness_deltas.append(curr_fitness - prev_fitness)
            
            prev_params = prev_exp.get("params") or prev_exp.get("state", {})
            curr_params = curr_exp.get("params") or curr_exp.get("state", {})
            
            for pname in self.normalizer.param_bounds:
                if pname in prev_params and pname in curr_params:
                    prev_norm = self.normalizer.normalize({pname: prev_params[pname]})
                    curr_norm = self.normalizer.normalize({pname: curr_params[pname]})
                    if pname in prev_norm and pname in curr_norm:
                        delta = curr_norm[pname].normalized - prev_norm[pname].normalized
                        if pname not in param_deltas:
                            param_deltas[pname] = []
                        param_deltas[pname].append(delta)
        
        if len(fitness_deltas) < 5:
            return rules
        
        # 计算参数间的相关性
        param_names = list(param_deltas.keys())
        for i, p1 in enumerate(param_names):
            for p2 in param_names[i + 1:]:
                if len(param_deltas[p1]) != len(param_deltas[p2]):
                    continue
                
                # 简单相关性计算
                n = len(param_deltas[p1])
                if n < 5:
                    continue
                
                mean1 = statistics.mean(param_deltas[p1])
                mean2 = statistics.mean(param_deltas[p2])
                
                cov = sum((param_deltas[p1][j] - mean1) * (param_deltas[p2][j] - mean2) for j in range(n)) / n
                std1 = statistics.stdev(param_deltas[p1]) if n > 1 else 1
                std2 = statistics.stdev(param_deltas[p2]) if n > 1 else 1
                
                if std1 > 0 and std2 > 0:
                    correlation = cov / (std1 * std2)
                else:
                    continue
                
                # 高相关性（协同变化）
                if abs(correlation) > 0.6:
                    cat1 = self.normalizer.param_info[p1]["category"]
                    cat2 = self.normalizer.param_info[p2]["category"]
                    
                    direction = "positive" if correlation > 0 else "negative"
                    
                    rule = AbstractRule(
                        rule_id=f"corr_{cat1}_{cat2}_{direction}_{self.domain}",
                        rule_type="correlation",
                        param_category=f"{cat1}+{cat2}",
                        param_role="mixed",
                        direction=direction,
                        confidence=min(0.9, abs(correlation)),
                        sample_count=n,
                        context=f"{cat1} 类和 {cat2} 类参数存在{direction}相关性"
                    )
                    rules.append(rule)
        
        return rules
    
    def _extract_constraint_rules(self, experiences: List[Dict]) -> List[AbstractRule]:
        """提取约束违反规则：哪些参数区间容易违反约束"""
        rules = []
        
        # 收集违反约束的样本
        violation_samples: Dict[str, List[float]] = {}  # category -> normalized values
        success_samples: Dict[str, List[float]] = {}
        
        for exp in experiences:
            params = exp.get("params") or exp.get("state", {})
            success = exp.get("success", True)
            result = exp.get("result", {})
            
            # 判断是否违反约束
            is_violation = (
                not success or 
                result.get("status") == "constraint_violation" or
                "constraint" in str(result.get("errors", [])).lower()
            )
            
            target_dict = violation_samples if is_violation else success_samples
            
            for pname, pvalue in params.items():
                if pname not in self.normalizer.param_bounds:
                    continue
                
                norm = self.normalizer.normalize({pname: pvalue})
                if pname not in norm:
                    continue
                
                category = self.normalizer.param_info[pname]["category"]
                if category not in target_dict:
                    target_dict[category] = []
                target_dict[category].append(norm[pname].normalized)
        
        # 分析每个类型的约束违反区间
        for category in set(list(violation_samples.keys()) + list(success_samples.keys())):
            violations = violation_samples.get(category, [])
            successes = success_samples.get(category, [])
            
            if len(violations) < 3:
                continue
            
            # 计算违反约束样本的平均归一化值
            avg_violation = statistics.mean(violations)
            
            # 确定危险区间
            if avg_violation < 0.3:
                danger_range = (0.0, 0.3)
                context = "过小"
            elif avg_violation > 0.7:
                danger_range = (0.7, 1.0)
                context = "过大"
            else:
                continue  # 违反分布在中间，规律不明显
            
            # 计算置信度
            total = len(violations) + len(successes)
            confidence = len(violations) / total if total > 0 else 0.5
            
            rule = AbstractRule(
                rule_id=f"constraint_{category}_{context}_{self.domain}",
                rule_type="constraint",
                param_category=category,
                param_role="mixed",
                effect="violate_constraint",
                optimal_range=danger_range,  # 这里存的是危险范围
                confidence=confidence,
                sample_count=len(violations),
                context=f"{category} 类参数{context}时容易违反约束"
            )
            rules.append(rule)
        
        return rules
    
    def generate_meta_patterns(self) -> List[MetaPattern]:
        """从抽象规则生成更高层的元模式"""
        patterns = []
        
        # 模式1：尺寸协同
        length_rules = self.knowledge_base.get_applicable_rules(
            param_category="length", min_confidence=0.5
        )
        if len(length_rules) >= 2:
            pattern = MetaPattern(
                pattern_id=f"pattern_length_synergy_{self.domain}",
                description="长度类参数应协同调整，避免单独改变某一尺寸",
                applicable_domains=[self.domain, "all"],
                conditions=["多个长度参数存在几何约束"],
                recommendations=[
                    "同时调整相关的长度参数",
                    "保持关键比例在合理范围",
                ],
                anti_patterns=[
                    "单独大幅改变某个长度参数",
                    "忽视几何约束关系",
                ],
                confidence=0.7,
                source_rules=[r.rule_id for r in length_rules[:3]],
            )
            patterns.append(pattern)
        
        # 模式2：间隙敏感性
        gap_rules = self.knowledge_base.get_applicable_rules(
            param_category="gap", min_confidence=0.4
        )
        if gap_rules:
            pattern = MetaPattern(
                pattern_id=f"pattern_gap_sensitivity_{self.domain}",
                description="间隙类参数对性能影响大，需谨慎调整",
                applicable_domains=[self.domain, "electromagnetic_actuator"],
                conditions=["存在气隙或间隙参数"],
                recommendations=[
                    "间隙参数小幅调整",
                    "先固定其他参数再优化间隙",
                ],
                anti_patterns=[
                    "大幅改变间隙参数",
                    "间隙过小导致磁饱和",
                ],
                confidence=0.65,
                source_rules=[r.rule_id for r in gap_rules[:2]],
            )
            patterns.append(pattern)
        
        # 模式3：探索策略
        all_rules = list(self.knowledge_base.abstract_rules.values())
        high_conf_rules = [r for r in all_rules if r.confidence > 0.6]
        
        if len(high_conf_rules) >= 3:
            pattern = MetaPattern(
                pattern_id=f"pattern_exploration_{self.domain}",
                description="基于已学习规则的探索策略",
                applicable_domains=[self.domain, "all"],
                conditions=["有足够的历史经验"],
                recommendations=[
                    "优先探索高敏感度参数",
                    "在最优范围内精细搜索",
                    "避开约束违反的危险区间",
                ],
                anti_patterns=[
                    "在低置信度区域大量探索",
                    "忽视已知的约束规律",
                ],
                confidence=0.75,
                source_rules=[r.rule_id for r in high_conf_rules[:5]],
            )
            patterns.append(pattern)
        
        # 保存模式
        for pattern in patterns:
            self.knowledge_base.add_pattern(pattern)
        
        logger.info(f"🎯 生成了 {len(patterns)} 条元模式")
        return patterns


class MetaLearningAgent:
    """元学习智能体：管理知识迁移和应用"""
    
    def __init__(
        self,
        knowledge_base_path: str = "meta_knowledge.json",
        current_task_config: Optional[TaskConfig] = None,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        enable_llm_analysis: bool = True
    ):
        self.knowledge_base = MetaKnowledgeBase(knowledge_base_path)
        self.current_task = current_task_config
        self.normalizer: Optional[ParameterNormalizer] = None
        self.extractor: Optional[MetaKnowledgeExtractor] = None
        
        # LLM 配置
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.enable_llm_analysis = enable_llm_analysis
        
        if current_task_config:
            self._setup_for_task(current_task_config)
    
    def _setup_for_task(self, task_config: TaskConfig):
        """为当前任务设置归一化器和提取器"""
        # 从任务配置中提取参数边界
        param_bounds = {}
        for param in task_config.parameters:
            name = param["name"]
            range_val = param.get("range", [0, 1])
            param_bounds[name] = (range_val[0], range_val[1])
        
        self.normalizer = ParameterNormalizer(param_bounds)
        self.extractor = MetaKnowledgeExtractor(
            self.normalizer,
            self.knowledge_base,
            domain=task_config.domain,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            enable_llm_analysis=self.enable_llm_analysis
        )
    
    def setup_from_bounds(
        self, 
        param_bounds: Dict[str, Tuple[float, float]], 
        domain: str = "electromagnetic_actuator"
    ):
        """直接从参数边界设置（简化接口）"""
        self.normalizer = ParameterNormalizer(param_bounds)
        self.extractor = MetaKnowledgeExtractor(
            self.normalizer,
            self.knowledge_base,
            domain=domain,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            enable_llm_analysis=self.enable_llm_analysis
        )
    
    def learn_from_experiences(self, experiences: List[Dict]) -> Dict[str, Any]:
        """从经验中学习并提取元知识"""
        if not self.extractor:
            return {"error": "未设置任务配置"}
        
        # 提取抽象规则
        rules = self.extractor.extract_from_experiences(experiences)
        
        # 生成元模式
        patterns = self.extractor.generate_meta_patterns()
        
        return {
            "extracted_rules": len(rules),
            "generated_patterns": len(patterns),
            "rules": [asdict(r) for r in rules],
            "patterns": [asdict(p) for p in patterns],
        }
    
    def transfer_to_new_task(self, new_task_config: TaskConfig) -> Dict[str, Any]:
        """将元知识迁移到新任务"""
        # 设置新任务
        self._setup_for_task(new_task_config)
        
        # 获取适用的规则和模式
        applicable_rules = self.knowledge_base.get_applicable_rules(min_confidence=0.4)
        applicable_patterns = self.knowledge_base.get_applicable_patterns(
            new_task_config.domain, min_confidence=0.4
        )
        
        # 构建先验知识 prompt
        prior_prompt = self._build_prior_prompt(applicable_rules, applicable_patterns)
        
        # 生成初始探索建议
        exploration_suggestions = self._generate_exploration_suggestions(applicable_rules)
        
        return {
            "applicable_rules": len(applicable_rules),
            "applicable_patterns": len(applicable_patterns),
            "prior_prompt": prior_prompt,
            "exploration_suggestions": exploration_suggestions,
            "param_mapping": {
                name: info for name, info in self.normalizer.param_info.items()
            } if self.normalizer else {},
        }
    
    def _build_prior_prompt(
        self, 
        rules: List[AbstractRule], 
        patterns: List[MetaPattern]
    ) -> str:
        """构建先验知识 prompt"""
        lines = ["## 从历史优化任务中学习到的先验知识\n"]
        
        if patterns:
            lines.append("### 优化模式")
            for p in patterns[:3]:
                lines.append(f"- **{p.description}**")
                if p.recommendations:
                    lines.append(f"  - 建议: {'; '.join(p.recommendations[:2])}")
                if p.anti_patterns:
                    lines.append(f"  - 避免: {'; '.join(p.anti_patterns[:2])}")
        
        if rules:
            lines.append("\n### 参数调整规则")
            
            # 按规则类型分组
            mono_rules = [r for r in rules if r.rule_type == "monotonic_effect"]
            range_rules = [r for r in rules if r.rule_type == "optimal_range"]
            constraint_rules = [r for r in rules if r.rule_type == "constraint"]
            causal_rules = [r for r in rules if r.rule_type == "causal_relation"]
            llm_rules = [r for r in rules if r.rule_type == "llm_insight"]
            
            # 因果关系规则（优先显示，最重要）
            if causal_rules:
                lines.append("\n### ⚡ 参数-约束因果关系（关键知识）")
                for r in causal_rules[:5]:
                    try:
                        context_data = json.loads(r.context) if r.context else {}
                        constraint_name = context_data.get("constraint_name", "")
                        primary_params = context_data.get("primary_params", [])
                        secondary_params = context_data.get("secondary_params", [])
                        irrelevant_params = context_data.get("irrelevant_params", [])
                        effects = context_data.get("effects", [])
                        physics_mechanism = context_data.get("physics_mechanism", "")
                        
                        if constraint_name:
                            lines.append(f"\n**约束: {constraint_name}** (置信度 {r.confidence:.0%})")
                        if primary_params:
                            lines.append(f"  - 🔴 关键参数: {', '.join(primary_params)}")
                        if secondary_params:
                            lines.append(f"  - 🟡 次要参数: {', '.join(secondary_params)}")
                        if irrelevant_params:
                            lines.append(f"  - ⚪ 无关参数: {', '.join(irrelevant_params)}")
                        if effects:
                            effect_lines = []
                            for eff in effects[:4]:
                                param = eff.get("param", "")
                                direction = "增大" if eff.get("direction") == "increase" else "减小"
                                effect = "缓解" if eff.get("effect") == "alleviates" else "加剧"
                                desc = eff.get("description", "")
                                effect_lines.append(f"{param}{direction}会{effect}此约束")
                            lines.append(f"  - 📋 影响: {'; '.join(effect_lines)}")
                        if physics_mechanism:
                            lines.append(f"  - 💡 机理: {physics_mechanism[:100]}")
                    except Exception:
                        pass
            
            if mono_rules:
                lines.append("\n**单调性规律**:")
                for r in mono_rules[:3]:
                    effect_cn = "改善" if r.effect == "improve_fitness" else "恶化"
                    dir_cn = "增大" if r.direction == "increase" else "减小"
                    lines.append(f"- {r.param_category} 类参数{dir_cn}倾向于{effect_cn} fitness (置信度 {r.confidence:.0%})")
            
            if range_rules:
                lines.append("**最优范围**:")
                for r in range_rules[:3]:
                    if r.optimal_range:
                        lines.append(f"- {r.param_category} 类参数在 {r.optimal_range[0]:.0%}-{r.optimal_range[1]:.0%} 范围内表现最佳")
            
            if constraint_rules:
                lines.append("**约束注意**:")
                for r in constraint_rules[:3]:
                    lines.append(f"- {r.param_category} 类参数{r.context or '需注意约束'}")
            
            # LLM 深度洞察规则
            if llm_rules:
                lines.append("\n### 深度洞察（LLM 分析）")
                for r in llm_rules[:5]:
                    try:
                        context_data = json.loads(r.context) if r.context else {}
                        rule_name = context_data.get("rule_name", "")
                        description = context_data.get("description", "")
                        physics_reason = context_data.get("physics_reason", "")
                        recommendations = context_data.get("recommendations", [])
                        anti_patterns = context_data.get("anti_patterns", [])
                        
                        if rule_name:
                            lines.append(f"\n**{rule_name}** (置信度 {r.confidence:.0%})")
                        if description:
                            lines.append(f"  - 描述: {description}")
                        if physics_reason:
                            lines.append(f"  - 物理机理: {physics_reason}")
                        if recommendations:
                            lines.append(f"  - 建议: {'; '.join(recommendations[:2])}")
                        if anti_patterns:
                            lines.append(f"  - 避免: {'; '.join(anti_patterns[:2])}")
                    except Exception:
                        pass
        
        # 添加领域洞察
        if hasattr(self, 'knowledge_base') and self.knowledge_base:
            domain_stats = self.knowledge_base.domain_statistics.get(
                self.current_task.domain if hasattr(self, 'current_task') and self.current_task else "electromagnetic_actuator",
                {}
            )
            insights = domain_stats.get("optimization_insights", [])
            if insights:
                lines.append("\n### 优化策略洞察")
                for insight in insights[:3]:
                    lines.append(f"- {insight}")
        
        return "\n".join(lines)
    
    def _generate_exploration_suggestions(self, rules: List[AbstractRule]) -> List[str]:
        """生成探索建议"""
        suggestions = []
        
        # 基于规则生成建议
        high_conf_rules = [r for r in rules if r.confidence > 0.6]
        
        for rule in high_conf_rules[:5]:
            if rule.rule_type == "monotonic_effect":
                dir_cn = "增大" if rule.direction == "increase" else "减小"
                if rule.effect == "improve_fitness":
                    suggestions.append(f"优先尝试{dir_cn} {rule.param_category} 类参数")
            
            elif rule.rule_type == "optimal_range" and rule.optimal_range:
                lo, hi = rule.optimal_range
                suggestions.append(f"{rule.param_category} 类参数建议在 {lo:.0%}-{hi:.0%} 范围内探索")
            
            elif rule.rule_type == "constraint":
                suggestions.append(f"注意: {rule.context}")
        
        # 添加通用建议
        suggestions.append("初始阶段优先探索高敏感度参数")
        suggestions.append("在找到较好区域后进行精细搜索")
        
        return suggestions
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """获取当前元知识库摘要"""
        return {
            "total_rules": len(self.knowledge_base.abstract_rules),
            "total_patterns": len(self.knowledge_base.meta_patterns),
            "rules_by_type": self._count_by_type(
                self.knowledge_base.abstract_rules.values(), "rule_type"
            ),
            "rules_by_category": self._count_by_type(
                self.knowledge_base.abstract_rules.values(), "param_category"
            ),
            "high_confidence_rules": len([
                r for r in self.knowledge_base.abstract_rules.values()
                if r.confidence > 0.6
            ]),
        }
    
    def _count_by_type(self, items, attr: str) -> Dict[str, int]:
        """按属性统计数量"""
        counts = {}
        for item in items:
            val = getattr(item, attr, "unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts


# ========== 便捷函数 ==========

def create_default_task_config(param_bounds: Dict[str, Tuple[float, float]], domain: str = "electromagnetic_actuator") -> TaskConfig:
    """从参数边界创建默认任务配置"""
    parameters = []
    for name, (min_val, max_val) in param_bounds.items():
        category = infer_param_category(name).value
        role = infer_param_role(name).value
        parameters.append({
            "name": name,
            "category": category,
            "role": role,
            "range": [min_val, max_val],
        })
    
    return TaskConfig(
        task_name="auto_generated_task",
        domain=domain,
        parameters=parameters,
        constraints=[],
        objectives=[{"name": "fitness", "direction": "maximize", "weight": 1.0}],
    )


# ========== 经验蒸馏：跨任务迁移核心 ==========

DISTILL_PROMPT = """你是一位跨领域的电磁设计优化专家。以下是从一个**特定电磁执行器（E-core 结构）**优化任务中积累的经验数据。

你的任务是将这些经验蒸馏为**两个层级**的可迁移知识，使其可应用于**不同拓扑结构**（E-core、U-core、圆筒型、盘式等）的电磁执行器优化。

## ⚠️ 关键要求

1. **禁止使用源任务的具体参数名**（如 lm, tm, ta, dg, hs, hslot, wslot, s, tb_ratio）
   → 改用物理角色名：「永磁体厚度」「磁路截面(轭部)厚度」「工作气隙」「绕组窗口高度」「槽深」「槽宽」「齿部厚度比」「磁铁长度」等
2. **禁止使用具体参数数值**（如 4.72, 0.50, 2.2）
   → 改用定性描述或归一化位置：「接近上界」「中等偏大」「范围中部」
3. 每条规则必须包含**因果方向**（↑/↓ 对目标的影响）和**物理机理**

## 参数名到物理角色的参考映射（源任务）
- lm → 磁铁长度 (沿运动方向)
- tm → 永磁体厚度 (垂直于磁化方向)
- ta → 磁路截面/轭部厚度
- dg → 工作气隙长度
- hs → 绕组窗口高度（槽区总高度）
- wslot → 槽宽
- hslot → 槽深
- s → 极距/齿距比例因子
- tb_ratio → 齿部厚度与轭部厚度之比
- n1 → 每层匝数（离散，由几何派生）
- n2 → 层数（离散跳变变量，由几何派生）

## 源经验数据

### 学习到的规则（含物理解释）
{learned_rules}

### 策略洞察
{prompt_additions}

### 参数敏感性排序（对 fitness 的影响程度）
{sensitivity_info}

### 成功模式概要
{success_summary}

### 失败模式与规避规则
{failure_summary}

### ExpeL 对比学习规则
{expel_rules}

### 经验统计
{experience_stats}

## 输出要求

请输出 JSON（不加 markdown 代码块），包含两个层级：

{{
  "L3_universal_principles": {{
    "physical_laws": [
      "纯物理机理层面的通用定律，适用于任何电磁执行器。格式：'[现象] → [因果关系] → [设计建议]'"
    ],
    "optimization_methodology": [
      "优化方法论层面的通用策略（离散变量处理、多目标权衡、约束边界行为等）"
    ]
  }},
  "L2_transferable_rules": [
    {{
      "rule": "用物理角色名表述的具体调参规则（保留因果方向和影响强度）",
      "physics_role": "涉及的物理角色（如'磁路截面厚度'、'工作气隙'）",
      "direction": "increase/decrease（该角色参数应增大还是减小）",
      "effect": "对目标函数的影响方向和强度（如'显著改善'、'轻微恶化'）",
      "mechanism": "物理机理简述",
      "confidence": "high/medium/low",
      "caveat": "适用条件或注意事项（可选）"
    }}
  ],
  "L2_sensitivity_ranking": [
    "按影响力排序的物理角色列表（从最敏感到最不敏感），使用物理角色名"
  ],
  "L2_constraint_patterns": [
    {{
      "pattern": "用物理概念描述的约束违反模式",
      "frequency": "高频/中频/低频",
      "root_cause": "物理层面的根本原因",
      "prevention": "预防策略"
    }}
  ]
}}

要求：
- L3 层：3-5 条，高度抽象，任何电磁结构都成立
- L2 调参规则：5-8 条，保留因果方向和机理，但用物理角色名替代参数名
- L2 敏感性：按物理角色排序
- L2 约束模式：2-4 条，描述高频失败场景
- 每条规则必须有**可操作的方向性建议**，不要写空泛的"注意xxx"类语句"""


@dataclass
class DistilledPrinciples:
    """蒸馏后的跨任务可迁移知识（L3 + L2 两级）"""
    
    # L3: 高度抽象的通用原理
    physical_laws: List[str] = field(default_factory=list)
    optimization_methodology: List[str] = field(default_factory=list)
    
    # L2: 用物理角色名表述的可迁移规则
    transferable_rules: List[Dict[str, str]] = field(default_factory=list)
    sensitivity_ranking: List[str] = field(default_factory=list)
    constraint_patterns: List[Dict[str, str]] = field(default_factory=list)
    
    # 兼容旧格式
    physical_principles: List[str] = field(default_factory=list)
    constraint_strategies: List[str] = field(default_factory=list)
    optimization_heuristics: List[str] = field(default_factory=list)
    failure_avoidance: List[str] = field(default_factory=list)
    
    source_dir: str = ""
    distilled_at: str = ""
    
    def to_prompt(self) -> str:
        """生成注入 LLM 的提示词（两级知识结构）"""
        lines = ["【跨任务迁移知识（蒸馏自历史优化经验）】"]
        lines.append("以下知识从历史优化任务中蒸馏提炼，已用物理概念替代具体参数名。")
        lines.append("请将这些物理角色（如「磁路截面厚度」「工作气隙」）映射到当前任务的对应参数，并在你的推理中应用。")
        lines.append("")
        lines.append("⚠️ **引用要求**：当你基于以下迁移知识做出设计决策时，请明确标注来源，格式如：")
        lines.append("  「根据迁移经验[L2规则#2]，增大磁路截面厚度（对应本任务的 ta）可缓解饱和，因此本轮将 ta 从 0.55 → 0.60」")
        lines.append("  「根据迁移经验[L2敏感性排序]，绕组窗口高度是最敏感参数（对应本任务的 hs），优先调整」")
        lines.append("")
        
        # === L3: 通用原理 ===
        has_l3 = self.physical_laws or self.optimization_methodology
        # 兼容旧格式
        phys = self.physical_laws or self.physical_principles
        opt = self.optimization_methodology or self.optimization_heuristics
        
        if phys:
            lines.append("## L3 通用物理定律")
            for i, p in enumerate(phys, 1):
                lines.append(f"  {i}. {p}")
        
        if opt:
            lines.append("\n## L3 优化方法论")
            for i, p in enumerate(opt, 1):
                lines.append(f"  {i}. {p}")
        
        # === L2: 可迁移调参规则 ===
        if self.transferable_rules:
            lines.append("\n## L2 可迁移调参规则（请映射到当前任务的对应参数）")
            for i, rule in enumerate(self.transferable_rules, 1):
                r = rule.get("rule", "")
                direction = rule.get("direction", "")
                effect = rule.get("effect", "")
                mechanism = rule.get("mechanism", "")
                confidence = rule.get("confidence", "")
                caveat = rule.get("caveat", "")
                
                dir_symbol = "↑" if direction == "increase" else "↓" if direction == "decrease" else ""
                conf_tag = {"high": "★★★", "medium": "★★", "low": "★"}.get(confidence, "")
                
                lines.append(f"  {i}. {conf_tag} {r}")
                if mechanism:
                    lines.append(f"     机理: {mechanism}")
                if caveat:
                    lines.append(f"     注意: {caveat}")
        
        # === L2: 敏感性排序 ===
        if self.sensitivity_ranking:
            lines.append("\n## L2 参数敏感性排序（影响力从高到低）")
            lines.append(f"  {' > '.join(self.sensitivity_ranking)}")
        
        # === L2: 约束违反模式 ===
        patterns = self.constraint_patterns
        # 兼容旧格式
        if not patterns and (self.constraint_strategies or self.failure_avoidance):
            if self.constraint_strategies:
                lines.append("\n## 约束处理策略")
                for i, p in enumerate(self.constraint_strategies, 1):
                    lines.append(f"  {i}. {p}")
            if self.failure_avoidance:
                lines.append("\n## 失败规避")
                for i, p in enumerate(self.failure_avoidance, 1):
                    lines.append(f"  {i}. {p}")
        elif patterns:
            lines.append("\n## L2 高频约束违反模式")
            for i, cp in enumerate(patterns, 1):
                pat = cp.get("pattern", "")
                freq = cp.get("frequency", "")
                root = cp.get("root_cause", "")
                prev = cp.get("prevention", "")
                lines.append(f"  {i}. [{freq}] {pat}")
                if root:
                    lines.append(f"     根因: {root}")
                if prev:
                    lines.append(f"     预防: {prev}")
        
        return "\n".join(lines)
    
    def save(self, path: str = "distilled_principles.json"):
        data = {
            "L3_universal_principles": {
                "physical_laws": self.physical_laws,
                "optimization_methodology": self.optimization_methodology,
            },
            "L2_transferable_rules": self.transferable_rules,
            "L2_sensitivity_ranking": self.sensitivity_ranking,
            "L2_constraint_patterns": self.constraint_patterns,
            # 兼容旧格式字段
            "physical_principles": self.physical_principles,
            "constraint_strategies": self.constraint_strategies,
            "optimization_heuristics": self.optimization_heuristics,
            "failure_avoidance": self.failure_avoidance,
            "source_dir": self.source_dir,
            "distilled_at": self.distilled_at,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[Distill] 💾 蒸馏原则已保存: {path}")
    
    @classmethod
    def load(cls, path: str = "distilled_principles.json") -> Optional["DistilledPrinciples"]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            l3 = data.get("L3_universal_principles", {})
            
            dp = cls(
                physical_laws=l3.get("physical_laws", []),
                optimization_methodology=l3.get("optimization_methodology", []),
                transferable_rules=data.get("L2_transferable_rules", []),
                sensitivity_ranking=data.get("L2_sensitivity_ranking", []),
                constraint_patterns=data.get("L2_constraint_patterns", []),
                # 兼容旧格式
                physical_principles=data.get("physical_principles", []),
                constraint_strategies=data.get("constraint_strategies", []),
                optimization_heuristics=data.get("optimization_heuristics", []),
                failure_avoidance=data.get("failure_avoidance", []),
                source_dir=data.get("source_dir", ""),
                distilled_at=data.get("distilled_at", ""),
            )
            
            n_l3 = len(dp.physical_laws) + len(dp.optimization_methodology)
            n_l2 = len(dp.transferable_rules) + len(dp.constraint_patterns)
            logger.info(f"[Distill] 📚 已加载蒸馏知识: L3={n_l3}条, L2规则={len(dp.transferable_rules)}条, "
                        f"约束模式={len(dp.constraint_patterns)}条 | 来源: {dp.source_dir}")
            return dp
        except Exception as e:
            logger.warning(f"[Distill] 加载蒸馏原则失败: {e}")
            return None


def _read_json_safe(path: str) -> Any:
    """安全读取 JSON 文件"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _collect_raw_experience(source_dir: str) -> Dict[str, str]:
    """从源经验目录收集原始经验数据，格式化为蒸馏提示词的输入"""
    
    strategy_data = _read_json_safe(os.path.join(source_dir, "strategy_state.json"))
    expel_data = _read_json_safe(os.path.join(source_dir, "expel_rules.json"))
    experience_data = _read_json_safe(os.path.join(source_dir, "experience_buffer.json"))
    
    # 学习到的规则
    learned_rules = "无"
    if strategy_data and strategy_data.get("learned_rules"):
        rules = strategy_data["learned_rules"]
        learned_rules = "\n".join(f"- {r}" for r in rules[:15])
    
    # 策略洞察
    prompt_additions = "无"
    if strategy_data and strategy_data.get("prompt_additions"):
        adds = strategy_data["prompt_additions"]
        prompt_additions = "\n".join(f"- {a}" for a in adds[:10])
    
    # 参数敏感性
    sensitivity_info = "无"
    if strategy_data and strategy_data.get("param_sensitivity"):
        sens = strategy_data["param_sensitivity"]
        sorted_sens = sorted(sens.items(), key=lambda x: x[1], reverse=True)
        sensitivity_info = "\n".join(
            f"- {name}: 敏感度 {val:.2f}" for name, val in sorted_sens if val > 0.3
        )
    
    # 成功模式
    success_summary = "无"
    if strategy_data and strategy_data.get("success_patterns"):
        patterns = strategy_data["success_patterns"]
        lines = []
        for i, sp in enumerate(patterns[:3], 1):
            fitness = sp.get("fitness", "N/A")
            freq = sp.get("frequency", 0)
            params = sp.get("params", {})
            param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:6])
            lines.append(f"- 模式{i}: fitness={fitness}, 频次={freq}, 参数={param_str}")
        success_summary = "\n".join(lines)
    
    # 失败模式
    failure_summary = "无"
    if strategy_data and strategy_data.get("failure_patterns"):
        fps = strategy_data["failure_patterns"]
        lines = []
        for fp in fps[:5]:
            avoid = fp.get("avoid_rules", [])
            freq = fp.get("frequency", 0)
            for rule in avoid[:2]:
                lines.append(f"- [频次{freq}] {rule}")
        failure_summary = "\n".join(lines) if lines else "无"
    
    # ExpeL 规则
    expel_rules = "无"
    if expel_data and isinstance(expel_data, list):
        lines = []
        for rule in expel_data[:10]:
            text = rule.get("text", "")
            conf = rule.get("confidence", 0)
            lines.append(f"- [置信度{conf}] {text}")
        expel_rules = "\n".join(lines) if lines else "无"
    
    # 经验统计
    experience_stats = "无"
    if experience_data and isinstance(experience_data, list):
        total = len(experience_data)
        success = sum(1 for e in experience_data if e.get("result_status") == "ok")
        failed = total - success
        
        fitnesses = [e.get("fitness") for e in experience_data
                     if e.get("fitness") is not None and e.get("result_status") == "ok"]
        best_fitness = min(fitnesses) if fitnesses else "N/A"
        
        errors = {}
        for e in experience_data:
            for err in e.get("errors", []):
                errors[err] = errors.get(err, 0) + 1
        top_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:5]
        
        lines = [
            f"- 总经验数: {total}（成功: {success}, 失败: {failed}）",
            f"- 成功率: {success/total*100:.1f}%" if total > 0 else "",
            f"- 历史最佳 fitness: {best_fitness}",
        ]
        if top_errors:
            lines.append("- 高频错误:")
            for err, cnt in top_errors:
                lines.append(f"  - [{cnt}次] {err}")
        experience_stats = "\n".join(lines)
    
    return {
        "learned_rules": learned_rules,
        "prompt_additions": prompt_additions,
        "sensitivity_info": sensitivity_info,
        "success_summary": success_summary,
        "failure_summary": failure_summary,
        "expel_rules": expel_rules,
        "experience_stats": experience_stats,
    }


def distill_experience_for_transfer(
    source_dir: str,
    llm_api_key: str,
    llm_base_url: str = "https://openrouter.ai/api/v1",
    llm_model: str = "openai/gpt-4o",
    save_path: str = "distilled_principles.json",
) -> Optional[DistilledPrinciples]:
    """
    经验蒸馏：将任务特定的原始经验通过 LLM 蒸馏为结构无关的通用原则。
    
    这是跨任务迁移的核心函数。它读取源经验目录中的 strategy_state.json、
    expel_rules.json、experience_buffer.json，提取关键信息后调用 LLM
    进行物理层面的抽象，生成不依赖具体参数名/值的通用优化原则。
    
    Args:
        source_dir: 源经验目录（包含 strategy_state.json 等）
        llm_api_key: LLM API Key
        llm_base_url: LLM API Base URL
        llm_model: LLM 模型名
        save_path: 蒸馏结果保存路径
    
    Returns:
        DistilledPrinciples 或 None（失败时）
    """
    if not OPENAI_AVAILABLE:
        logger.error("[Distill] OpenAI SDK 未安装，无法进行经验蒸馏")
        return None
    
    import re as _re
    
    # 1. 收集原始经验
    logger.info(f"[Distill] 📖 正在从 {source_dir} 收集原始经验...")
    raw_data = _collect_raw_experience(source_dir)
    
    # 2. 构建蒸馏提示词
    prompt = DISTILL_PROMPT.format(**raw_data)
    
    # 3. 调用 LLM
    logger.info(f"[Distill] 🤖 正在调用 LLM 进行经验蒸馏... (model={llm_model})")
    try:
        client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "你是一位精通电磁学和优化算法的跨领域工程专家。请严格按照要求输出 JSON 格式。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4500,
        )
        result_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[Distill] LLM 调用失败: {e}")
        return None
    
    # 4. 解析 JSON
    try:
        json_match = _re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result_text = json_match.group()
        result = json.loads(result_text)
    except json.JSONDecodeError as e:
        logger.error(f"[Distill] JSON 解析失败: {e}\n原始输出: {result_text[:500]}")
        return None
    
    # 5. 构建 DistilledPrinciples（兼容新旧两种 JSON 格式）
    l3 = result.get("L3_universal_principles", {})
    
    principles = DistilledPrinciples(
        # 新格式 L3
        physical_laws=l3.get("physical_laws", []),
        optimization_methodology=l3.get("optimization_methodology", []),
        # 新格式 L2
        transferable_rules=result.get("L2_transferable_rules", []),
        sensitivity_ranking=result.get("L2_sensitivity_ranking", []),
        constraint_patterns=result.get("L2_constraint_patterns", []),
        # 兼容旧格式
        physical_principles=result.get("physical_principles", []),
        constraint_strategies=result.get("constraint_strategies", []),
        optimization_heuristics=result.get("optimization_heuristics", []),
        failure_avoidance=result.get("failure_avoidance", []),
        source_dir=source_dir,
        distilled_at=datetime.now().isoformat(),
    )
    
    n_l3 = len(principles.physical_laws) + len(principles.optimization_methodology)
    n_l2_rules = len(principles.transferable_rules)
    n_l2_constraints = len(principles.constraint_patterns)
    # 旧格式兼容计数
    n_old = (len(principles.physical_principles) + len(principles.constraint_strategies)
             + len(principles.optimization_heuristics) + len(principles.failure_avoidance))
    
    logger.info(f"[Distill] ✅ 蒸馏完成:")
    if n_l3 > 0:
        logger.info(f"[Distill]   L3 通用原理: {n_l3} 条 (物理定律={len(principles.physical_laws)}, 方法论={len(principles.optimization_methodology)})")
    if n_l2_rules > 0:
        logger.info(f"[Distill]   L2 可迁移规则: {n_l2_rules} 条")
    if principles.sensitivity_ranking:
        logger.info(f"[Distill]   L2 敏感性排序: {' > '.join(principles.sensitivity_ranking[:4])}...")
    if n_l2_constraints > 0:
        logger.info(f"[Distill]   L2 约束模式: {n_l2_constraints} 条")
    if n_old > 0 and n_l3 == 0:
        logger.info(f"[Distill]   (旧格式) 共 {n_old} 条原则")
    
    # 6. 保存
    principles.save(save_path)
    
    return principles

