"""
多LLM评估模块

功能：
1. 调用多个先进LLM对优化结果进行交叉评估
2. 支持OpenRouter API统一调用不同模型
3. 客观指标（约束、性能）+ 主观评审（物理合理性、鲁棒性、可制造性、创新性）
4. 不允许LLM评估自己优化出来的结果
"""

import asyncio
import json
import os
import csv
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from openai import AsyncOpenAI
from loguru import logger


# ============== 配置常量 ==============

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PENALTY_THRESHOLD = 1e5  # 惩罚值阈值，超过此值视为仿真失败

# 可用的评审LLM列表（通过OpenRouter调用，选用最新顶级模型）
AVAILABLE_REVIEW_LLMS = [
    "openai/gpt-5.2",                     # GPT-5.2
    "anthropic/claude-opus-4.5",          # Claude Opus 4.5
    "anthropic/claude-sonnet-4.5",        # Claude Sonnet 4.5
    "google/gemini-3-pro",                # Gemini 3 Pro
    "qwen/qwen3-max",                     # Qwen 3 Max
    "deepseek/deepseek-v3.2-speciale",    # DeepSeek V3.2 Speciale
    "amazon/nova-premier-v1",             # Nova Premier V1
    "x-ai/grok-4",                        # Grok 4
]

# 算法与其使用的优化LLM映射
# 如果设计是由某个LLM优化产生的，该LLM不参与评审自己的结果
# 传入的是纯数字数据，评审时其他LLM都可以打分
ALGORITHM_LLM_MAPPING = {
    # LLM优化算法 - 需要排除对应的LLM评审自己的结果
    "LLM-RL-GPT": "openai/gpt-5.2",
    "LLM-RL-Claude-Opus": "anthropic/claude-opus-4.5",
    "LLM-RL-Claude-Sonnet": "anthropic/claude-sonnet-4.5",
    "LLM-RL-Gemini": "google/gemini-3-pro",
    "LLM-RL-Qwen": "qwen/qwen3-max",
    "LLM-RL-DeepSeek": "deepseek/deepseek-v3.2-speciale",
    "LLM-RL-Nova": "amazon/nova-premier-v1",
    "LLM-RL-Grok": "x-ai/grok-4",
    
    # 非LLM算法 - 所有LLM都可以评审
    "GA": None,           # 遗传算法
    "PSO": None,          # 粒子群优化
    "Random": None,       # 随机搜索
    "Manual": None,       # 人工设计
    "LLM-RL": None,       # 默认LLM-RL（如未指定具体LLM）
}

# 评估维度权重
WEIGHTS = {
    "constraint": 0.20,           # 约束满足度
    "performance": 0.40,          # 性能(fitness)
    "physical_rationality": 0.15, # 物理合理性
    "robustness": 0.10,           # 鲁棒性
    "manufacturability": 0.10,    # 可制造性
    "innovation": 0.05,           # 创新性
}

# 参数边界定义
PARAM_BOUNDS = {
    "lm": (1.0, 6.0),
    "tm": (0.3, 0.5),
    "ta": (0.35, 0.75),
    "dg": (0.30, 0.65),
    "hs": (1.2, 2.2),
    "wslot": (2.0, 2.8),
    "hslot": (0.8, 1.3),
    "s": (0.8, 1.2),
}


# ============== 数据结构 ==============

@dataclass
class DesignData:
    """设计数据（评估输入）"""
    # 必需的设计参数
    lm: float           # 永磁体长度 (mm)
    tm: float           # 永磁体厚度 (mm)
    ta: float           # 电枢厚度 (mm)
    dg: float           # 气隙 (mm)
    hs: float           # 定子高度 (mm)
    wslot: float        # 槽宽 (mm)
    hslot: float        # 槽高 (mm)
    s: float            # 行程 (mm)
    
    # 性能指标
    fitness: float      # 综合适应度（越小越好，负数）
    avg_B: float        # 平均磁通密度 (T)
    B_max: float        # 最大磁通密度 (T)
    
    # 可选的附加指标
    kb: Optional[float] = None          # 推力线性度系数
    pb: Optional[float] = None          # 功率系数
    volume: Optional[float] = None      # 体积 (m³)
    mass_total: Optional[float] = None  # 总质量 (kg)
    
    # 约束状态
    status: str = "ok"                  # "ok" 或 "constraint_violation"
    errors: str = ""                    # 约束违规详情
    
    # 元数据
    algorithm: str = "unknown"          # 产生该设计的算法
    iteration: int = 0                  # 迭代轮次
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, str], algorithm: str = "unknown") -> "DesignData":
        """从CSV行解析设计数据"""
        def safe_float(val, default=0.0):
            try:
                return float(val) if val else default
            except:
                return default
        
        return cls(
            lm=safe_float(row.get("lm")),
            tm=safe_float(row.get("tm")),
            ta=safe_float(row.get("ta")),
            dg=safe_float(row.get("dg")),
            hs=safe_float(row.get("hs")),
            wslot=safe_float(row.get("wslot")),
            hslot=safe_float(row.get("hslot")),
            s=safe_float(row.get("s")),
            fitness=safe_float(row.get("fitness"), 1e6),
            avg_B=safe_float(row.get("avg_B")),
            B_max=safe_float(row.get("B_sat")),  # CSV中是B_sat
            kb=safe_float(row.get("kb")) if row.get("kb") else None,
            pb=safe_float(row.get("pb")) if row.get("pb") else None,
            volume=safe_float(row.get("volume")) if row.get("volume") else None,
            mass_total=safe_float(row.get("mass_total")) if row.get("mass_total") else None,
            status=row.get("status", "ok"),
            errors=row.get("errors", ""),
            algorithm=algorithm,
            iteration=int(row.get("iteration", 0)),
        )


@dataclass
class LLMReview:
    """单个LLM的评审结果"""
    llm_name: str
    physical_rationality: float
    physical_rationality_reason: str
    robustness: float
    robustness_reason: str
    manufacturability: float
    manufacturability_reason: str
    innovation: float
    innovation_reason: str
    overall_comment: str
    raw_response: str = ""


@dataclass
class EvaluationResult:
    """完整评估结果"""
    design_id: str
    algorithm: str
    design_data: DesignData
    
    # 客观评分
    constraint_score: float = 0.0
    constraint_details: str = ""
    performance_score: float = 0.0
    performance_rank: int = 0
    performance_total: int = 0
    
    # 主观评分（多LLM平均）
    physical_rationality_score: float = 0.0
    robustness_score: float = 0.0
    manufacturability_score: float = 0.0
    innovation_score: float = 0.0
    
    # LLM评审详情
    llm_reviews: List[LLMReview] = field(default_factory=list)
    consistency_coefficient: float = 0.0
    
    # 综合评分
    total_score: float = 0.0
    weighted_details: Dict[str, float] = field(default_factory=dict)
    
    def calculate_total(self):
        """计算加权总分"""
        self.weighted_details = {
            "constraint": self.constraint_score * WEIGHTS["constraint"],
            "performance": self.performance_score * WEIGHTS["performance"],
            "physical_rationality": self.physical_rationality_score * WEIGHTS["physical_rationality"],
            "robustness": self.robustness_score * WEIGHTS["robustness"],
            "manufacturability": self.manufacturability_score * WEIGHTS["manufacturability"],
            "innovation": self.innovation_score * WEIGHTS["innovation"],
        }
        self.total_score = sum(self.weighted_details.values())


# ============== 评估器 ==============

class MultiLLMEvaluator:
    """多LLM评价系统"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        review_llms: Optional[List[str]] = None,
        num_reviewers: Optional[int] = None,
        timeout: int = 120,
    ):
        """
        初始化评价器
        
        Args:
            api_key: OpenRouter API密钥
            base_url: API基础URL
            review_llms: 可用的评审LLM列表
            num_reviewers: 每个设计使用几个LLM评审，默认None表示使用全部可用LLM
            timeout: API调用超时时间
        """
        self.api_key = api_key
        self.base_url = base_url
        self.available_llms = review_llms or AVAILABLE_REVIEW_LLMS
        # 默认使用全部LLM评审（除了自己）
        self.num_reviewers = num_reviewers if num_reviewers else len(self.available_llms)
        self.timeout = timeout
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        
        # 用于性能排名的fitness列表
        self.all_fitness_values: List[float] = []
        
        logger.info(f"多LLM评估器初始化完成，可用模型: {len(self.available_llms)}个")
    
    # ---------- 客观评分 ----------
    
    def score_constraint(self, design: DesignData) -> Tuple[float, str]:
        """计算约束满足度得分"""
        # 硬约束：约束违规直接0分
        if design.status != "ok":
            return 0.0, f"约束违规: {design.errors}"
        
        score = 10.0
        details = []
        
        # 磁饱和检查
        if design.B_max >= 2.0:
            return 0.0, f"磁饱和: B_max={design.B_max:.3f}T >= 2.0T"
        elif design.B_max >= 2.0:
            score -= 2.0
            details.append(f"接近饱和: B_max={design.B_max:.3f}T")
        elif design.B_max >= 1.6:
            score -= 1.0
            details.append(f"B_max={design.B_max:.3f}T")
        
        # 边界余量检查
        margin_penalty = 0.0
        for param_name, (low, high) in PARAM_BOUNDS.items():
            value = getattr(design, param_name, None)
            if value is None:
                continue
            
            range_size = high - low
            margin_low = (value - low) / range_size
            margin_high = (high - value) / range_size
            min_margin = min(margin_low, margin_high)
            
            if min_margin < 0.05:
                margin_penalty += 0.3
                details.append(f"{param_name}贴近边界")
            elif min_margin < 0.10:
                margin_penalty += 0.1
        
        score -= min(margin_penalty, 2.0)
        
        return max(0.0, score), "; ".join(details) if details else "约束良好"
    
    def score_performance(self, design: DesignData) -> Tuple[float, int, int]:
        """
        计算性能得分（基于fitness排名）
        
        Returns:
            (得分, 排名, 总数)
        """
        fitness = design.fitness
        
        # 惩罚值检测
        if fitness >= PENALTY_THRESHOLD:
            return 0.0, 0, 0
        
        # 添加到fitness列表用于排名
        if fitness not in self.all_fitness_values:
            self.all_fitness_values.append(fitness)
        
        # 基于排名评分
        valid_fitness = sorted([f for f in self.all_fitness_values if f < PENALTY_THRESHOLD])
        
        if not valid_fitness:
            return 5.0, 1, 1
        
        # fitness越小越好
        rank = valid_fitness.index(fitness) + 1 if fitness in valid_fitness else len(valid_fitness)
        total = len(valid_fitness)
        percentile = rank / total
        
        # 排名转分数
        if percentile <= 0.05:
            base_score = 10.0
        elif percentile <= 0.10:
            base_score = 9.5
        elif percentile <= 0.20:
            base_score = 9.0
        elif percentile <= 0.30:
            base_score = 8.0
        elif percentile <= 0.50:
            base_score = 7.0
        elif percentile <= 0.70:
            base_score = 5.0
        elif percentile <= 0.90:
            base_score = 3.0
        else:
            base_score = 1.0
        
        # 磁饱和惩罚
        saturation_penalty = 0.0
        if design.B_max >= 2.0:
            saturation_penalty = 2.0
        elif design.B_max >= 1.6:
            saturation_penalty = 1.0
        
        return max(0.0, base_score - saturation_penalty), rank, total
    
    # ---------- LLM评审 ----------
    
    def _build_review_prompt(self, design: DesignData) -> str:
        """构建LLM评审提示词"""
        params_str = f"""
- lm (永磁体长度): {design.lm:.3f} mm (范围: 1.0-6.0)
- tm (永磁体厚度): {design.tm:.3f} mm (范围: 0.3-0.5)
- ta (电枢厚度): {design.ta:.3f} mm (范围: 0.35-0.75)
- dg (气隙): {design.dg:.3f} mm (范围: 0.30-0.65)
- hs (定子高度): {design.hs:.3f} mm (范围: 1.2-2.2)
- wslot (槽宽): {design.wslot:.3f} mm (范围: 2.0-2.8)
- hslot (槽高): {design.hslot:.3f} mm (范围: 0.8-1.3)
- s (行程): {design.s:.3f} mm (范围: 0.8-1.2)
"""
        
        performance_str = f"""
- 综合fitness: {design.fitness:.6f} (越小越好)
- 平均磁通密度 avg_B: {design.avg_B:.4f} T
- 最大磁通密度 B_max: {design.B_max:.4f} T
- 推力线性度 kb: {design.kb if design.kb else 'N/A'}
- 功率系数 pb: {design.pb if design.pb else 'N/A'}
"""
        
        return f"""# 电磁执行器设计方案专家评审

## 你的角色
你是一位资深的微型电磁执行器设计专家，擅长音圈式直线电机的设计与优化。
请对以下设计方案进行专业、客观的评审。

**重要说明**：
- 约束满足度和性能(fitness)已通过工具客观计算，你只需评审以下4个主观维度
- 请基于电磁学原理和工程经验给出评分，评分范围0-10分

## 设计参数
{params_str}

## 性能指标
{performance_str}

## 评审维度（每项0-10分）

### 1. 物理合理性 (physical_rationality)
评估参数配置是否符合电磁学原理：
- 气隙dg与永磁体厚度tm的比例是否合理（通常气隙应小于永磁体厚度）
- 槽宽wslot与槽高hslot的比例是否适合线圈绑扎
- 永磁体长度lm与整体尺寸的协调性
- 磁路设计是否有利于磁通闭合

### 2. 鲁棒性 (robustness)
评估设计对制造误差的容忍度：
- 参数是否远离边界，有足够安全余量
- 关键尺寸（如气隙dg）对性能的敏感性
- 对材料特性波动的鲁棒性
- 工作点是否稳定

### 3. 可制造性 (manufacturability)
评估加工和装配难度：
- 最小特征尺寸（尤其是气隙dg={design.dg:.2f}mm）是否易于加工
- 尺寸精度要求是否合理
- 装配工艺可行性（如何保证气隙均匀）
- 材料选择的可行性

### 4. 创新性 (innovation)
评估设计的新颖程度：
- 参数组合是否突破传统设计范式
- 是否探索了非常规的设计空间
- 性能指标是否超越常规水平

## 输出格式
请严格按以下JSON格式输出，不要添加其他内容：
```json
{{
  "physical_rationality": <0-10的数字>,
  "physical_rationality_reason": "<评分理由，一句话>",
  "robustness": <0-10的数字>,
  "robustness_reason": "<评分理由，一句话>",
  "manufacturability": <0-10的数字>,
  "manufacturability_reason": "<评分理由，一句话>",
  "innovation": <0-10的数字>,
  "innovation_reason": "<评分理由，一句话>",
  "overall_comment": "<总体评价，两到三句话>"
}}
```
"""
    
    async def _call_llm(self, llm_name: str, prompt: str) -> str:
        """调用单个LLM"""
        try:
            response = await self.client.chat.completions.create(
                model=llm_name,
                messages=[
                    {"role": "system", "content": "你是一位电磁执行器设计专家，请客观评审设计方案。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 降低随机性，增加一致性
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"调用 {llm_name} 失败: {e}")
            return ""
    
    def _parse_llm_response(self, llm_name: str, response: str) -> Optional[LLMReview]:
        """解析LLM响应"""
        try:
            # 提取JSON部分
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                logger.warning(f"{llm_name} 响应中未找到JSON")
                return None
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            return LLMReview(
                llm_name=llm_name,
                physical_rationality=float(data.get("physical_rationality", 5)),
                physical_rationality_reason=data.get("physical_rationality_reason", ""),
                robustness=float(data.get("robustness", 5)),
                robustness_reason=data.get("robustness_reason", ""),
                manufacturability=float(data.get("manufacturability", 5)),
                manufacturability_reason=data.get("manufacturability_reason", ""),
                innovation=float(data.get("innovation", 5)),
                innovation_reason=data.get("innovation_reason", ""),
                overall_comment=data.get("overall_comment", ""),
                raw_response=response
            )
        except Exception as e:
            logger.error(f"解析 {llm_name} 响应失败: {e}")
            return None
    
    async def _llm_review(self, design: DesignData, llm_name: str) -> Optional[LLMReview]:
        """使用单个LLM进行评审"""
        prompt = self._build_review_prompt(design)
        response = await self._call_llm(llm_name, prompt)
        
        if not response:
            return None
        
        return self._parse_llm_response(llm_name, response)
    
    def _select_reviewers(self, optimization_llm: Optional[str]) -> List[str]:
        """
        选择评审LLM（排除优化使用的LLM）
        
        默认使用全部可用LLM（除了自己），求平均分
        """
        candidates = [llm for llm in self.available_llms if llm != optimization_llm]
        # 使用全部可用的评审LLM
        if self.num_reviewers >= len(self.available_llms):
            return candidates
        return candidates[:self.num_reviewers]
    
    def _aggregate_scores(self, reviews: List[LLMReview]) -> Dict[str, float]:
        """聚合多个LLM评分"""
        if not reviews:
            return {
                "physical_rationality": 5.0,
                "robustness": 5.0,
                "manufacturability": 5.0,
                "innovation": 5.0,
            }
        
        return {
            "physical_rationality": sum(r.physical_rationality for r in reviews) / len(reviews),
            "robustness": sum(r.robustness for r in reviews) / len(reviews),
            "manufacturability": sum(r.manufacturability for r in reviews) / len(reviews),
            "innovation": sum(r.innovation for r in reviews) / len(reviews),
        }
    
    def _calculate_consistency(self, reviews: List[LLMReview]) -> float:
        """计算评审一致性系数"""
        if len(reviews) < 2:
            return 1.0
        
        dimensions = ["physical_rationality", "robustness", "manufacturability", "innovation"]
        total_variance = 0.0
        
        for dim in dimensions:
            scores = [getattr(r, dim) for r in reviews]
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            total_variance += variance
        
        avg_variance = total_variance / len(dimensions)
        # 方差越小，一致性越高（归一化到0-1）
        consistency = max(0, 1 - avg_variance / 25)  # 假设最大方差为25
        return round(consistency, 3)
    
    # ---------- 主评估流程 ----------
    
    async def evaluate_design(
        self,
        design: DesignData,
        design_id: Optional[str] = None,
        optimization_llm: Optional[str] = None,
    ) -> EvaluationResult:
        """
        评估单个设计方案
        
        Args:
            design: 设计数据（纯数字，所有LLM都可以评审）
            design_id: 设计唯一标识
            optimization_llm: 优化时使用的LLM名称（如有），该LLM将被排除在评审外
                             例如: "anthropic/claude-3.5-sonnet", "openai/gpt-4o"
                             如果是非LLM算法（GA/PSO等），传None即可
        
        说明:
            - 传入的是纯数字数据，评审时所有LLM都可以打分
            - 但如果设计是由某个LLM优化产生的，需要排除该LLM的自我评价
        """
        if design_id is None:
            design_id = f"{design.algorithm}_{design.iteration}"
        
        result = EvaluationResult(
            design_id=design_id,
            algorithm=design.algorithm,
            design_data=design,
        )
        
        # ===== 第一阶段：客观评分 =====
        
        # 约束评分
        result.constraint_score, result.constraint_details = self.score_constraint(design)
        
        # 约束不通过，直接返回0分
        if result.constraint_score == 0:
            result.total_score = 0.0
            logger.info(f"设计 {design_id} 约束不满足，评分为0")
            return result
        
        # 性能评分
        result.performance_score, result.performance_rank, result.performance_total = \
            self.score_performance(design)
        
        # 性能为0（仿真失败），跳过LLM评审
        if result.performance_score == 0:
            result.calculate_total()
            logger.info(f"设计 {design_id} 性能为0，跳过LLM评审")
            return result
        
        # ===== 第二阶段：LLM主观评审 =====
        
        # 确定需要排除的LLM
        # 优先使用传入的optimization_llm，其次查找算法映射
        exclude_llm = optimization_llm or ALGORITHM_LLM_MAPPING.get(design.algorithm)
        reviewers = self._select_reviewers(exclude_llm)
        
        if exclude_llm:
            logger.info(f"设计 {design_id} 由 {exclude_llm} 优化，排除其自评")
        logger.info(f"设计 {design_id} 使用 {len(reviewers)} 个LLM评审: {[r.split('/')[-1] for r in reviewers]}")
        
        # 并发调用多个LLM
        tasks = [self._llm_review(design, llm) for llm in reviewers]
        reviews = await asyncio.gather(*tasks)
        
        # 过滤无效响应
        valid_reviews = [r for r in reviews if r is not None]
        result.llm_reviews = valid_reviews
        
        if valid_reviews:
            # 聚合评分
            aggregated = self._aggregate_scores(valid_reviews)
            result.physical_rationality_score = aggregated["physical_rationality"]
            result.robustness_score = aggregated["robustness"]
            result.manufacturability_score = aggregated["manufacturability"]
            result.innovation_score = aggregated["innovation"]
            
            # 计算一致性
            result.consistency_coefficient = self._calculate_consistency(valid_reviews)
        
        # 计算综合评分
        result.calculate_total()
        
        logger.info(f"设计 {design_id} 评估完成，总分: {result.total_score:.2f}")
        
        return result
    
    async def evaluate_batch(
        self,
        designs: List[DesignData],
        algorithm: str = "unknown",
        optimization_llm: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """
        批量评估多个设计
        
        Args:
            designs: 设计数据列表
            algorithm: 算法名称
            optimization_llm: 优化时使用的LLM（如有），该LLM不参与评审
        """
        results = []
        for i, design in enumerate(designs):
            design.algorithm = algorithm
            result = await self.evaluate_design(
                design, 
                f"{algorithm}_{i}",
                optimization_llm=optimization_llm
            )
            results.append(result)
        return results
    
    async def compare_algorithms(
        self,
        algorithm_results: Dict[str, List[DesignData]],
        algorithm_llm_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        对比多个算法的优化效果
        
        Args:
            algorithm_results: {算法名: [该算法产生的设计列表]}
            algorithm_llm_mapping: {算法名: 优化时使用的LLM}
                                   例如: {"LLM-RL": "anthropic/claude-3.5-sonnet"}
                                   如果某算法是由LLM优化的，需要指定，该LLM将不参与评审
                                   非LLM算法（GA/PSO）不需要指定
        
        Returns:
            包含评估结果和排名的字典
        """
        all_evaluations: Dict[str, Dict] = {}
        llm_mapping = algorithm_llm_mapping or {}
        
        # 先收集所有fitness用于排名
        for algo_name, designs in algorithm_results.items():
            for design in designs:
                if design.fitness < PENALTY_THRESHOLD:
                    if design.fitness not in self.all_fitness_values:
                        self.all_fitness_values.append(design.fitness)
        
        # 评估每个算法的设计
        for algo_name, designs in algorithm_results.items():
            # 获取该算法对应的优化LLM（如有）
            optimization_llm = llm_mapping.get(algo_name)
            
            logger.info(f"评估算法: {algo_name}, 设计数量: {len(designs)}, 优化LLM: {optimization_llm or '无'}")
            
            # 更新算法名
            for design in designs:
                design.algorithm = algo_name
            
            # 批量评估（排除优化LLM的自评）
            evaluations = await self.evaluate_batch(designs, algo_name, optimization_llm)
            
            # 统计
            valid_evals = [e for e in evaluations if e.total_score > 0]
            if valid_evals:
                best = max(valid_evals, key=lambda x: x.total_score)
                avg_score = sum(e.total_score for e in valid_evals) / len(valid_evals)
            else:
                best = evaluations[0] if evaluations else None
                avg_score = 0.0
            
            all_evaluations[algo_name] = {
                "best_result": best,
                "average_score": avg_score,
                "valid_count": len(valid_evals),
                "total_count": len(evaluations),
                "all_results": evaluations,
            }
        
        # 生成排名
        ranking = sorted(
            all_evaluations.items(),
            key=lambda x: x[1]["best_result"].total_score if x[1]["best_result"] else 0,
            reverse=True
        )
        
        return {
            "evaluations": all_evaluations,
            "ranking": [
                {
                    "rank": i + 1,
                    "algorithm": name,
                    "best_score": data["best_result"].total_score if data["best_result"] else 0,
                    "avg_score": data["average_score"],
                    "valid_count": data["valid_count"],
                }
                for i, (name, data) in enumerate(ranking)
            ],
            "timestamp": datetime.now().isoformat(),
        }


# ============== 工具函数 ==============

def load_designs_from_csv(csv_path: str, algorithm: str = "unknown") -> List[DesignData]:
    """从CSV文件加载设计数据"""
    designs = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            design = DesignData.from_csv_row(row, algorithm)
            designs.append(design)
    
    logger.info(f"从 {csv_path} 加载了 {len(designs)} 个设计")
    return designs


def save_evaluation_report(
    results: Dict[str, Any],
    output_path: str = "evaluation_report.json"
):
    """保存评估报告"""
    
    def serialize(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return obj
    
    serialized = serialize(results)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估报告已保存到: {output_path}")


def print_ranking_table(results: Dict[str, Any]):
    """打印排名表格"""
    ranking = results.get("ranking", [])
    
    print("\n" + "=" * 80)
    print("                          算法优化效果综合排名")
    print("=" * 80)
    print(f"{'排名':<6}{'算法':<20}{'最佳分':<12}{'平均分':<12}{'有效设计':<12}")
    print("-" * 80)
    
    for item in ranking:
        print(f"{item['rank']:<6}{item['algorithm']:<20}{item['best_score']:<12.2f}{item['avg_score']:<12.2f}{item['valid_count']:<12}")
    
    print("=" * 80 + "\n")


# ============== 命令行入口 ==============

async def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多LLM评估系统")
    parser.add_argument("--csv", type=str, required=True, help="优化结果CSV文件路径")
    parser.add_argument("--algorithm", type=str, default="LLM-RL", help="算法名称")
    parser.add_argument("--api-key", type=str, help="OpenRouter API密钥")
    parser.add_argument("--output", type=str, default="evaluation_report.json", help="输出报告路径")
    parser.add_argument("--num-reviewers", type=int, default=3, help="评审LLM数量")
    
    args = parser.parse_args()
    
    # 获取API密钥
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("请设置 OPENROUTER_API_KEY 环境变量或通过 --api-key 参数提供")
        return
    
    # 加载设计数据
    designs = load_designs_from_csv(args.csv, args.algorithm)
    
    # 创建评估器
    evaluator = MultiLLMEvaluator(
        api_key=api_key,
        num_reviewers=args.num_reviewers,
    )
    
    # 评估
    results = await evaluator.compare_algorithms({args.algorithm: designs})
    
    # 输出结果
    print_ranking_table(results)
    save_evaluation_report(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())

