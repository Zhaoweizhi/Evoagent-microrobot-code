"""
ExpeL 风格的对比批评模块

基于 AAAI 2024 论文 "ExpeL: LLM Agents Are Experiential Learners"
核心思想：通过对比成功/失败案例，用 LLM 提取可迁移的元知识

功能：
1. RuleWithConfidence - 带置信度的规则
2. RuleManager - 规则库管理（ADD/EDIT/REMOVE/AGREE）
3. ContrastCritique - 对比批评器（LLM驱动）
4. update_rules() - ExpeL 原版规则更新逻辑
"""

import re
import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from loguru import logger

# 尝试导入 OpenAI 客户端
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI 未安装，对比批评将使用简化模式")


@dataclass
class RuleWithConfidence:
    """ExpeL 风格的带置信度规则"""
    text: str                    # 规则文本
    confidence: int = 2          # 置信度计数器（初始=2）
    source: str = "unknown"      # 来源：compare/success/failure
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def agree(self) -> None:
        """同意规则，置信度 +1"""
        self.confidence += 1
        self.last_updated = time.time()
    
    def challenge(self, is_full: bool = False) -> None:
        """挑战规则，置信度 -1（列表满时 -3）"""
        self.confidence -= 3 if is_full else 1
        self.last_updated = time.time()
    
    def edit(self, new_text: str) -> None:
        """修改规则文本，置信度 +1"""
        self.text = new_text
        self.confidence += 1
        self.last_updated = time.time()
    
    @property
    def is_valid(self) -> bool:
        """规则是否有效（置信度 > 0）"""
        return self.confidence > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RuleWithConfidence":
        return cls(**d)


class RuleManager:
    """规则库管理器 - ExpeL 风格"""
    
    def __init__(
        self,
        max_rules: int = 20,
        storage_path: str = "expel_rules.json"
    ):
        self.max_rules = max_rules
        self.storage_path = storage_path
        self.rules: List[RuleWithConfidence] = []
        self._load()
    
    def _load(self) -> None:
        """从文件加载规则"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.rules = [RuleWithConfidence.from_dict(d) for d in data]
                logger.info(f"[ExpeL] 已加载 {len(self.rules)} 条规则")
            except Exception as e:
                logger.warning(f"[ExpeL] 加载规则失败: {e}")
                self.rules = []
    
    def _save(self) -> None:
        """保存规则到文件"""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self.rules], f, 
                         ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[ExpeL] 保存规则失败: {e}")
    
    def update(self, operations: List[Tuple[str, str]]) -> Dict[str, int]:
        """
        根据操作更新规则库
        
        Args:
            operations: [(操作类型, 规则文本), ...]
                操作类型: "ADD", "AGREE 1", "EDIT 2", "REMOVE 3"
        
        Returns:
            统计信息
        """
        stats = {"added": 0, "agreed": 0, "edited": 0, "removed": 0, "pruned": 0}
        is_full = len(self.rules) >= self.max_rules
        
        for op, text in operations:
            op_type = op.split()[0].upper()
            op_num = int(op.split()[1]) - 1 if len(op.split()) > 1 else None
            
            if op_type == "ADD":
                # 检查是否已存在相似规则
                if not self._find_similar_rule(text):
                    self.rules.append(RuleWithConfidence(
                        text=text, 
                        confidence=2, 
                        source="compare"
                    ))
                    stats["added"] += 1
                    logger.debug(f"[ExpeL] ADD: {text[:50]}...")
                    
            elif op_type == "AGREE" and op_num is not None:
                if 0 <= op_num < len(self.rules):
                    self.rules[op_num].agree()
                    stats["agreed"] += 1
                    logger.debug(f"[ExpeL] AGREE #{op_num+1}: confidence → {self.rules[op_num].confidence}")
                    
            elif op_type == "EDIT" and op_num is not None:
                if 0 <= op_num < len(self.rules):
                    self.rules[op_num].edit(text)
                    stats["edited"] += 1
                    logger.debug(f"[ExpeL] EDIT #{op_num+1}: {text[:50]}...")
                    
            elif op_type == "REMOVE" and op_num is not None:
                if 0 <= op_num < len(self.rules):
                    self.rules[op_num].challenge(is_full)
                    stats["removed"] += 1
                    logger.debug(f"[ExpeL] REMOVE #{op_num+1}: confidence → {self.rules[op_num].confidence}")
        
        # 移除无效规则（置信度 <= 0）
        before_count = len(self.rules)
        self.rules = [r for r in self.rules if r.is_valid]
        stats["pruned"] = before_count - len(self.rules)
        
        # 按置信度降序排列
        self.rules.sort(key=lambda r: r.confidence, reverse=True)
        
        # 限制最大数量
        if len(self.rules) > self.max_rules:
            self.rules = self.rules[:self.max_rules]
        
        self._save()
        return stats
    
    def _find_similar_rule(self, text: str, threshold: float = 0.8) -> Optional[RuleWithConfidence]:
        """找相似规则（简单字符串匹配）"""
        text_lower = text.lower()
        for rule in self.rules:
            rule_lower = rule.text.lower()
            # 简单的包含检测
            if text_lower in rule_lower or rule_lower in text_lower:
                return rule
            # 计算词汇重叠
            words1 = set(text_lower.split())
            words2 = set(rule_lower.split())
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap > threshold:
                    return rule
        return None
    
    def get_rules_text(self, strategy_only: bool = False) -> str:
        """获取规则文本（用于提示词）
        
        Args:
            strategy_only: 若为 True，只取高置信度前5条，优先结论性规则
        """
        if not self.rules:
            return ""
        
        lines = ["【元知识规则库】（按置信度排序）"]
        max_rules = 5 if strategy_only else 10
        sorted_rules = sorted(self.rules, key=lambda r: r.confidence, reverse=True)[:max_rules]
        for i, rule in enumerate(sorted_rules, 1):
            lines.append(f"  {i}. [{rule.confidence}] {rule.text}")
        return "\n".join(lines)
    
    def get_rules_for_prompt(self) -> str:
        """获取用于 LLM 提示的规则列表"""
        if not self.rules:
            return "（暂无规则）"
        return "\n".join([f"{i}. {r.text}" for i, r in enumerate(self.rules, 1)])
    
    def get_summary(self) -> Dict[str, Any]:
        """获取规则库摘要"""
        return {
            "total_rules": len(self.rules),
            "avg_confidence": sum(r.confidence for r in self.rules) / len(self.rules) if self.rules else 0,
            "max_confidence": max((r.confidence for r in self.rules), default=0),
            "sources": {
                "compare": len([r for r in self.rules if r.source == "compare"]),
                "success": len([r for r in self.rules if r.source == "success"]),
                "failure": len([r for r in self.rules if r.source == "failure"]),
            }
        }


class ContrastCritique:
    """
    对比批评器 - ExpeL 核心
    
    通过对比成功和失败案例，用 LLM 提取元知识
    """
    
    # 对比批评的 Prompt 模板
    COMPARE_PROMPT_TEMPLATE = """你是一个优化专家，正在分析电磁执行器设计的成功和失败案例。

## 任务
对比以下两个相似参数配置的结果，提取设计规则（包括具体规则和宏观规则）。

## 成功案例
参数配置: {success_params}
结果: fitness = {success_fitness:.4f}
{success_extra}

## 失败案例  
参数配置: {fail_params}
错误/问题: {fail_errors}
{fail_extra}

## 现有规则
{existing_rules}

## 要求
分析成功和失败的关键差异，执行以下操作（每条规则最多1个操作，总共最多6个操作）：

操作格式：
- ADD: <新发现的规则>
- AGREE <规则编号>: <同意该规则的理由>
- EDIT <规则编号>: <修改后的规则文本>
- REMOVE <规则编号>: <删除理由>

**请同时提取两类规则**：

### A. 具体规则（项目内可用）
1. 必须是具体的、可操作的设计指导
2. 包含具体的参数名称和数值范围
3. 解释物理/工程原因
4. 避免空泛的描述

示例：
- "当 hs 接近上限 2.2 时，必须将 hslot 降到 1.1 以下以维持 n2≥7"
- "dg < 0.35 会导致约束违规，建议保持 dg ≥ 0.4"

### B. 宏观规则（可迁移到其他项目）
1. 抽象为**原理层面的规律**，避免绑定到具体参数名
2. 用"参数类别/物理量"表述（如：气隙、磁路截面、线圈匝数、结构厚度、饱和裕度）
3. 解释物理/工程原因，并说明**适用条件**
4. 给出调整方向/权衡关系

示例：
- "[宏观] 磁路截面接近饱和时，应优先增加截面或降低磁通密度，否则性能会因饱和快速恶化。"
- "[宏观] 当气隙减小以提升磁力时，需同步增加线圈驱动能力，否则会引发约束或热负荷问题。"

请输出你的操作（具体规则和宏观规则各1-2条）："""

    SUCCESS_PROMPT_TEMPLATE = """你是一个优化专家，正在分析多个成功的电磁执行器设计案例。

## 成功案例
{success_cases}

## 现有规则
{existing_rules}

## 要求
从这些成功案例中提取设计规则（包括具体规则和宏观规则）。执行以下操作（最多6个）：

操作格式：
- ADD: <新规则>
- AGREE <规则编号>: <理由>
- EDIT <规则编号>: <修改后的规则>
- REMOVE <规则编号>: <理由>

**请同时提取两类规则**：
- **具体规则**：包含参数名和数值范围，项目内可用
- **宏观规则**：以"[宏观]"开头，抽象为原理层面，可迁移到其他项目

请输出你的操作："""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        llm_model: str = "gpt-4o-mini",
        rule_manager: Optional[RuleManager] = None
    ):
        self.llm = llm_client
        self.llm_model = llm_model
        self.rule_manager = rule_manager or RuleManager()
        self._critique_count = 0
        
        # ★方案C：已分析配对去重（避免重复分析相同的成功/失败配对）
        # 存储格式：set of (success_exp_id, fail_exp_id)
        self._analyzed_pairs: set = set()
        self._skipped_count = 0  # 跳过计数（用于统计）
    
    def is_pair_analyzed(self, success_exp_id: str, fail_exp_id: str) -> bool:
        """★方案C：检查配对是否已分析过"""
        pair_key = (success_exp_id, fail_exp_id)
        return pair_key in self._analyzed_pairs
    
    def mark_pair_analyzed(self, success_exp_id: str, fail_exp_id: str) -> None:
        """★方案C：标记配对已分析"""
        pair_key = (success_exp_id, fail_exp_id)
        self._analyzed_pairs.add(pair_key)
    
    async def compare_critique(
        self,
        success_state: Dict[str, float],
        success_fitness: float,
        fail_state: Dict[str, float],
        fail_errors: List[str],
        success_extra: str = "",
        fail_extra: str = "",
        success_exp_id: str = "",
        fail_exp_id: str = ""
    ) -> List[Tuple[str, str]]:
        """
        对比批评：比较成功和失败案例
        
        Args:
            success_exp_id, fail_exp_id: ★方案C 新增，用于去重
        
        Returns:
            操作列表 [(操作类型, 规则文本), ...]
        """
        # ★方案C：去重检查
        if success_exp_id and fail_exp_id:
            if self.is_pair_analyzed(success_exp_id, fail_exp_id):
                self._skipped_count += 1
                logger.debug(f"[ExpeL] 跳过已分析配对: {success_exp_id} vs {fail_exp_id}")
                return []
            # 标记为已分析
            self.mark_pair_analyzed(success_exp_id, fail_exp_id)
        
        self._critique_count += 1
        
        # 构建 prompt
        prompt = self.COMPARE_PROMPT_TEMPLATE.format(
            success_params=self._format_params(success_state),
            success_fitness=success_fitness,
            success_extra=success_extra,
            fail_params=self._format_params(fail_state),
            fail_errors=", ".join(fail_errors) if fail_errors else "fitness 较差",
            fail_extra=fail_extra,
            existing_rules=self.rule_manager.get_rules_for_prompt()
        )
        
        # 调用 LLM
        response = await self._call_llm(prompt)
        
        # 解析操作
        operations = self._parse_operations(response)
        
        logger.info(f"[ExpeL] 对比批评 #{self._critique_count}: 提取 {len(operations)} 个操作")
        
        return operations
    
    async def success_critique(
        self,
        success_cases: List[Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """
        成功批评：从多个成功案例提取通用规则
        """
        if len(success_cases) < 2:
            return []
        
        # 格式化案例
        cases_text = []
        for i, case in enumerate(success_cases[:5], 1):  # 最多5个
            params = self._format_params(case.get("state", {}))
            fitness = case.get("fitness", 0)
            cases_text.append(f"案例{i}: {params}\n  fitness = {fitness:.4f}")
        
        prompt = self.SUCCESS_PROMPT_TEMPLATE.format(
            success_cases="\n\n".join(cases_text),
            existing_rules=self.rule_manager.get_rules_for_prompt()
        )
        
        response = await self._call_llm(prompt)
        operations = self._parse_operations(response)
        
        # 标记来源为 success
        return [(op, text) for op, text in operations]
    
    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        if self.llm is None:
            logger.warning("[ExpeL] LLM 未配置，返回空结果")
            return ""
        
        try:
            # ★方案B：辅助调用使用 1024（足够输出规则）
            response = await self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个电磁执行器优化专家，擅长从案例中提取设计规则（包括具体规则和可迁移的宏观规律）。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024  # ★合理限制：足够输出完整规则
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[ExpeL] LLM 调用失败: {e}")
            return ""
    
    def _format_params(self, params: Dict[str, float]) -> str:
        """格式化参数"""
        if not params:
            return "（无参数）"
        return ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                         for k, v in params.items() if v is not None])
    
    def _parse_operations(self, llm_output: str) -> List[Tuple[str, str]]:
        """Parse ADD, AGREE, EDIT and REMOVE operations from model output."""
        if not llm_output:
            return []
        
        operations = []
        
        # 匹配 ADD: xxx, AGREE 1: xxx, EDIT 2: xxx, REMOVE 3: xxx
        pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+)?)\s*[:：]\s*(.+?)(?=\n(?:REMOVE|EDIT|ADD|AGREE)|$)'
        matches = re.findall(pattern, llm_output, re.IGNORECASE | re.DOTALL)
        
        banned_words = ['ADD', 'AGREE', 'EDIT', 'REMOVE', '操作', '规则']
        
        for op, text in matches:
            text = text.strip()
            # 过滤无效规则
            if not text:
                continue
            if len(text) < 10:  # 太短
                continue
            if any(w in text for w in banned_words):  # 包含禁止词
                continue
            if not text.endswith(('.', '。', '！', '!')) and len(text) < 50:
                # 不完整的句子（除非很长）
                continue
            
            operations.append((op.upper(), text))
        
        return operations[:6]  # 最多6个操作（具体+宏观各3条）
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "critique_count": self._critique_count,
            "skipped_count": self._skipped_count,  # ★方案C：跳过的配对数
            "analyzed_pairs": len(self._analyzed_pairs),  # ★方案C：已分析配对数
            "rule_summary": self.rule_manager.get_summary()
        }


# ==================== 便捷函数 ====================

def create_expel_system(
    llm_client: Optional[Any] = None,
    llm_model: str = "gpt-4o-mini",
    max_rules: int = 20,
    storage_path: str = "expel_rules.json"
) -> Tuple[RuleManager, ContrastCritique]:
    """
    创建 ExpeL 系统
    
    Returns:
        (rule_manager, contrast_critique)
    """
    rule_manager = RuleManager(max_rules=max_rules, storage_path=storage_path)
    critique = ContrastCritique(
        llm_client=llm_client,
        llm_model=llm_model,
        rule_manager=rule_manager
    )
    return rule_manager, critique
