"""
人类反馈处理模块

功能：
1. 接收、存储、检索人类反馈
2. 根据当前状态匹配相关反馈
3. 支持多种反馈类型（纠正、建议、警告、确认）
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
from loguru import logger


class FeedbackType(Enum):
    """反馈类型"""
    CORRECTION = "correction"   # 纠正型：指出错误
    SUGGESTION = "suggestion"   # 建议型：提出改进方向
    WARNING = "warning"         # 警告型：提醒约束或风险
    CONFIRMATION = "confirmation"  # 确认型：肯定当前方向


class FeedbackPriority(Enum):
    """反馈优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4  # 立即应用


@dataclass
class Feedback:
    """单条反馈"""
    id: str
    text: str
    feedback_type: str = FeedbackType.SUGGESTION.value
    priority: int = FeedbackPriority.MEDIUM.value
    related_params: List[str] = field(default_factory=list)  # 相关参数名
    timestamp: float = field(default_factory=time.time)
    applied: bool = False  # 是否已应用
    strength: float = 1.0  # ★新增：引导强度 (0.0~1.0)
                           # 1.0 = 强烈引导，0.5 = 中等参考，0.1 = 弱参考，0.0 = 暂时关闭
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Feedback":
        # 兼容旧数据（没有 strength 字段）
        if "strength" not in d:
            d["strength"] = 1.0
        return cls(**d)
    
    @property
    def strength_label(self) -> str:
        """返回强度的文字描述"""
        if self.strength >= 0.9:
            return "🔴 强烈"
        elif self.strength >= 0.6:
            return "🟠 中等"
        elif self.strength >= 0.3:
            return "🟡 一般"
        elif self.strength > 0:
            return "🟢 弱"
        else:
            return "⚪ 关闭"


class FeedbackHandler:
    """反馈处理器"""
    
    def __init__(self, storage_path: Optional[str] = None, max_feedbacks: int = 100):
        self.storage_path = storage_path or "feedback_storage.json"
        self.max_feedbacks = max_feedbacks
        self.feedbacks: List[Feedback] = []
        self._load()
    
    def _load(self):
        """从文件加载反馈"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.feedbacks = [Feedback.from_dict(d) for d in data]
                logger.info(f"已加载 {len(self.feedbacks)} 条历史反馈")
            except Exception as e:
                logger.warning(f"加载反馈存储失败: {e}")
                self.feedbacks = []
    
    def _save(self):
        """保存反馈到文件"""
        try:
            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([fb.to_dict() for fb in self.feedbacks], f, 
                         ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存反馈存储失败: {e}")
    
    def add_feedback(
        self,
        text: str,
        feedback_type: str = "suggestion",
        priority: int = 2,
        related_params: Optional[List[str]] = None
    ) -> Feedback:
        """添加新反馈"""
        fb_id = f"fb_{int(time.time() * 1000)}_{len(self.feedbacks)}"
        feedback = Feedback(
            id=fb_id,
            text=text,
            feedback_type=feedback_type,
            priority=priority,
            related_params=related_params or []
        )
        self.feedbacks.append(feedback)
        
        # 限制数量，移除旧的低优先级反馈
        if len(self.feedbacks) > self.max_feedbacks:
            # 按优先级和时间排序，保留重要的
            self.feedbacks.sort(key=lambda x: (x.priority, x.timestamp), reverse=True)
            self.feedbacks = self.feedbacks[:self.max_feedbacks]
        
        self._save()
        logger.info(f"新增反馈: [{feedback_type}] {text[:50]}...")
        return feedback
    
    def get_relevant_feedbacks(
        self,
        current_params: Optional[Dict[str, float]] = None,
        param_names: Optional[List[str]] = None,
        limit: int = 5,
        include_applied: bool = False,
        min_strength: float = 0.1  # ★新增：最低强度阈值
    ) -> List[Feedback]:
        """获取与当前状态相关的反馈"""
        candidates = self.feedbacks if include_applied else [
            fb for fb in self.feedbacks if not fb.applied
        ]
        
        # ★过滤掉强度低于阈值的反馈（强度为0表示暂时关闭）
        candidates = [fb for fb in candidates if fb.strength >= min_strength]
        
        # 按相关性排序（强度影响排序权重）
        def relevance_score(fb: Feedback) -> float:
            score = fb.priority * 10  # 优先级权重
            score *= fb.strength  # ★强度作为乘数
            # 如果指定了参数，匹配相关参数
            if param_names and fb.related_params:
                overlap = len(set(fb.related_params) & set(param_names))
                score += overlap * 5 * fb.strength
            # 时间衰减：越新越重要
            age_hours = (time.time() - fb.timestamp) / 3600
            score -= min(age_hours * 0.1, 5)  # 最多减 5 分
            return score
        
        candidates.sort(key=relevance_score, reverse=True)
        return candidates[:limit]
    
    def get_urgent_feedbacks(self) -> List[Feedback]:
        """获取紧急反馈（需立即应用）"""
        return [
            fb for fb in self.feedbacks 
            if fb.priority >= FeedbackPriority.URGENT.value and not fb.applied
        ]
    
    def mark_applied(self, feedback_id: str):
        """标记反馈已应用"""
        for fb in self.feedbacks:
            if fb.id == feedback_id:
                fb.applied = True
                self._save()
                return
    
    def build_feedback_context(
        self,
        current_params: Optional[Dict[str, float]] = None,
        limit: int = 3,
        min_priority: int = 0
    ) -> str:
        """构建反馈上下文，用于注入提示词"""
        relevant = self.get_relevant_feedbacks(
            current_params=current_params,
            limit=limit
        )
        
        # ★过滤低优先级反馈
        if min_priority > 0:
            relevant = [fb for fb in relevant if fb.priority >= min_priority]
        
        if not relevant:
            return ""
        
        lines = ["## 专家反馈提醒\n"]
        for fb in relevant:
            type_label = {
                "correction": "⚠️ 纠正",
                "suggestion": "💡 建议",
                "warning": "🚨 警告",
                "confirmation": "✅ 确认"
            }.get(fb.feedback_type, "📝 反馈")
            
            # ★根据强度添加不同的引导语
            if fb.strength >= 0.9:
                prefix = "【必须遵守】"
            elif fb.strength >= 0.6:
                prefix = "【重要参考】"
            elif fb.strength >= 0.3:
                prefix = "【可参考】"
            else:
                prefix = "【仅供参考】"
            
            lines.append(f"- {type_label} {prefix}: {fb.text}")
        
        return "\n".join(lines)
    
    def update_strength(self, feedback_id: str, strength: float) -> bool:
        """更新反馈的引导强度"""
        strength = max(0.0, min(1.0, strength))  # 限制在 0~1 范围
        for fb in self.feedbacks:
            if fb.id == feedback_id:
                fb.strength = strength
                self._save()
                logger.info(f"更新反馈强度: {feedback_id} -> {strength:.0%}")
                return True
        return False
    
    def extract_constraints_from_feedbacks(self) -> Dict[str, Any]:
        """从反馈中提取约束规则"""
        constraints = {
            "param_bounds": {},  # 参数边界建议
            "forbidden_regions": [],  # 禁止区域
            "preferred_directions": {}  # 推荐调整方向
        }
        
        for fb in self.feedbacks:
            if fb.feedback_type == FeedbackType.WARNING.value:
                # 解析警告中的约束信息（简化实现）
                text_lower = fb.text.lower()
                for param in fb.related_params:
                    if "不要超过" in fb.text or "不能超过" in fb.text:
                        # 尝试提取上界
                        constraints["preferred_directions"][param] = "decrease"
                    elif "不要低于" in fb.text or "不能低于" in fb.text:
                        constraints["preferred_directions"][param] = "increase"
        
        return constraints
    
    def clear_old_feedbacks(self, hours: float = 24):
        """清理过期反馈"""
        cutoff = time.time() - hours * 3600
        old_count = len(self.feedbacks)
        self.feedbacks = [
            fb for fb in self.feedbacks 
            if fb.timestamp > cutoff or fb.priority >= FeedbackPriority.HIGH.value
        ]
        removed = old_count - len(self.feedbacks)
        if removed > 0:
            self._save()
            logger.info(f"清理了 {removed} 条过期反馈")

