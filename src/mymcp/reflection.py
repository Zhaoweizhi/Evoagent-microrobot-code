"""
Reflexion 式自我反思模块

基于论文 "Reflexion: Language Agents with Verbal Reinforcement Learning"
实现结构化的自我反思机制，用于改进 LLM Agent 的决策能力。

核心组件：
1. MemoryStream - 统一的记忆流（jsonl 格式）
2. ReflectionTrigger - 反思触发条件检测
3. SelfReflector - 结构化反思生成
4. ReflectionInjector - 反思内容注入到 prompt
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
from loguru import logger


class EventType(Enum):
    """记忆事件类型"""
    OBSERVATION = "observation"      # 观察：仿真结果
    ACTION = "action"                # 动作：参数调整
    REFLECTION = "reflection"        # 反思：自我总结
    RULE = "rule"                    # 规则：提取的经验
    CRITIC_FEEDBACK = "critic"       # 评论家反馈
    PATTERN = "pattern"              # 模式：成功/失败模式
    SENSITIVITY = "sensitivity"      # 敏感性：参数敏感性分析


class TriggerReason(Enum):
    """反思触发原因"""
    STAGNATION = "stagnation"        # 停滞：连续多轮无改善
    VIOLATION = "violation"          # 违规：约束违反
    DEGRADATION = "degradation"      # 退化：fitness 大幅下降
    PERIODIC = "periodic"            # 周期性：每 N 轮
    MILESTONE = "milestone"          # 里程碑：达成特定目标


@dataclass
class MemoryEvent:
    """记忆事件"""
    id: str
    timestamp: str
    round: int
    event_type: str  # EventType value
    content: str
    
    # 可选字段
    fitness_before: Optional[float] = None
    fitness_after: Optional[float] = None
    is_improvement: Optional[bool] = None
    importance: float = 5.0  # 1-10 重要性评分
    trigger_reason: Optional[str] = None  # TriggerReason value
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEvent":
        return cls(**d)
    
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class MemoryStream:
    """
    记忆流管理器
    
    使用 JSONL 格式存储，支持：
    - 追加写入
    - 按类型/重要性检索
    - 滑窗读取
    """
    
    def __init__(self, storage_path: str = "memory_stream.jsonl"):
        self.storage_path = storage_path
        self._event_count = 0
        self._load_count()
    
    def _load_count(self):
        """统计已有事件数量"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self._event_count = sum(1 for _ in f)
    
    def _generate_id(self, event_type: str) -> str:
        """生成事件 ID"""
        self._event_count += 1
        return f"{event_type}_{int(time.time() * 1000)}_{self._event_count}"
    
    def append(self, event: MemoryEvent) -> None:
        """追加事件到记忆流"""
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(event.to_jsonl() + "\n")
    
    def write_observation(
        self,
        round: int,
        params: Dict[str, float],
        fitness: Optional[float],
        success: bool,
        errors: List[str] = None
    ) -> MemoryEvent:
        """写入观察事件"""
        content_parts = [f"第 {round} 轮仿真结果:"]
        
        if success and fitness is not None:
            content_parts.append(f"- 状态: 成功")
            content_parts.append(f"- fitness: {fitness:.4f}")
        else:
            content_parts.append(f"- 状态: 失败")
            if errors:
                content_parts.append(f"- 错误: {'; '.join(errors[:3])}")
        
        # 记录关键参数
        key_params = ["lm", "tm", "ta", "dg", "hs", "wslot", "hslot"]
        param_str = ", ".join([f"{k}={params.get(k, 0):.2f}" for k in key_params if k in params])
        content_parts.append(f"- 参数: {param_str}")
        
        event = MemoryEvent(
            id=self._generate_id("obs"),
            timestamp=datetime.now().isoformat(),
            round=round,
            event_type=EventType.OBSERVATION.value,
            content="\n".join(content_parts),
            fitness_after=fitness,
            is_improvement=None,  # 需要外部设置
            importance=6.0 if success else 4.0,
            metadata={"params": params, "success": success, "errors": errors or []}
        )
        self.append(event)
        return event
    
    def write_reflection(
        self,
        round: int,
        reflection_content: str,
        trigger_reason: TriggerReason,
        fitness_before: Optional[float] = None,
        fitness_after: Optional[float] = None,
        importance: float = 8.0
    ) -> MemoryEvent:
        """写入反思事件"""
        event = MemoryEvent(
            id=self._generate_id("ref"),
            timestamp=datetime.now().isoformat(),
            round=round,
            event_type=EventType.REFLECTION.value,
            content=reflection_content,
            fitness_before=fitness_before,
            fitness_after=fitness_after,
            is_improvement=fitness_after < fitness_before if fitness_before and fitness_after else None,
            importance=importance,
            trigger_reason=trigger_reason.value
        )
        self.append(event)
        return event
    
    def write_rule(
        self,
        round: int,
        rule_content: str,
        source: str = "reflection",  # reflection / critic / meta
        importance: float = 7.0
    ) -> MemoryEvent:
        """写入规则事件"""
        event = MemoryEvent(
            id=self._generate_id("rule"),
            timestamp=datetime.now().isoformat(),
            round=round,
            event_type=EventType.RULE.value,
            content=rule_content,
            importance=importance,
            metadata={"source": source}
        )
        self.append(event)
        return event
    
    def write_pattern(
        self,
        round: int,
        pattern_type: str,  # "success" / "failure"
        params: Dict[str, float],
        fitness: Optional[float] = None,
        errors: List[str] = None,
        key_factors: List[str] = None,
        importance: float = 7.0
    ) -> MemoryEvent:
        """写入模式事件（成功/失败模式）"""
        if pattern_type == "success":
            content = f"成功模式 (fitness={fitness:.4f}): " + ", ".join(key_factors or [])
        else:
            content = f"失败模式: " + "; ".join((errors or [])[:3])
        
        event = MemoryEvent(
            id=self._generate_id("pat"),
            timestamp=datetime.now().isoformat(),
            round=round,
            event_type=EventType.PATTERN.value,
            content=content,
            fitness_after=fitness,
            importance=importance,
            metadata={
                "pattern_type": pattern_type,
                "params": params,
                "errors": errors or [],
                "key_factors": key_factors or []
            }
        )
        self.append(event)
        return event
    
    def write_sensitivity(
        self,
        round: int,
        sensitivity_data: Dict[str, float],
        top_params: List[str] = None,
        importance: float = 6.0
    ) -> MemoryEvent:
        """写入参数敏感性分析事件"""
        top_params = top_params or sorted(
            sensitivity_data.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        content = "参数敏感性分析: " + ", ".join(
            [f"{p}({s:.2f})" for p, s in (top_params if isinstance(top_params[0], tuple) else [(p, sensitivity_data.get(p, 0)) for p in top_params])]
        )
        
        event = MemoryEvent(
            id=self._generate_id("sens"),
            timestamp=datetime.now().isoformat(),
            round=round,
            event_type=EventType.SENSITIVITY.value,
            content=content,
            importance=importance,
            metadata={"sensitivity": sensitivity_data}
        )
        self.append(event)
        return event
    
    def read_all(self) -> List[MemoryEvent]:
        """读取所有事件"""
        if not os.path.exists(self.storage_path):
            return []
        
        events = []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(MemoryEvent.from_dict(json.loads(line)))
                    except Exception as e:
                        logger.warning(f"解析记忆事件失败: {e}")
        return events
    
    def read_recent(self, k: int = 10) -> List[MemoryEvent]:
        """读取最近 k 个事件"""
        all_events = self.read_all()
        return all_events[-k:] if len(all_events) > k else all_events
    
    def read_by_type(self, event_type: EventType, k: int = 10) -> List[MemoryEvent]:
        """按类型读取事件"""
        all_events = self.read_all()
        filtered = [e for e in all_events if e.event_type == event_type.value]
        return filtered[-k:] if len(filtered) > k else filtered
    
    def read_reflections(self, k: int = 5) -> List[MemoryEvent]:
        """读取最近的反思"""
        return self.read_by_type(EventType.REFLECTION, k)
    
    def read_rules(self, k: int = 10) -> List[MemoryEvent]:
        """读取规则"""
        return self.read_by_type(EventType.RULE, k)
    
    def read_high_importance(self, min_importance: float = 7.0, k: int = 10) -> List[MemoryEvent]:
        """读取高重要性事件"""
        all_events = self.read_all()
        filtered = [e for e in all_events if e.importance >= min_importance]
        return filtered[-k:] if len(filtered) > k else filtered
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆流统计"""
        all_events = self.read_all()
        if not all_events:
            return {"total": 0}
        
        type_counts = {}
        for e in all_events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
        
        return {
            "total": len(all_events),
            "by_type": type_counts,
            "latest_round": all_events[-1].round if all_events else 0,
            "avg_importance": sum(e.importance for e in all_events) / len(all_events)
        }
    
    def clear(self) -> None:
        """清空记忆流"""
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        self._event_count = 0
        logger.info(f"[Reflection] 记忆流已清空: {self.storage_path}")


class ReflectionTrigger:
    """
    反思触发器
    
    检测是否需要触发自我反思，支持多种触发条件：
    - 停滞检测：连续 N 轮无改善
    - 违规检测：约束违反
    - 退化检测：fitness 大幅下降
    - 周期触发：每 N 轮
    """
    
    def __init__(
        self,
        stagnation_window: int = 5,      # 停滞窗口
        degradation_threshold: float = 0.15,  # 退化阈值（相对下降比例）
        periodic_interval: int = 10,      # 周期间隔
        enable_periodic: bool = True
    ):
        self.stagnation_window = stagnation_window
        self.degradation_threshold = degradation_threshold
        self.periodic_interval = periodic_interval
        self.enable_periodic = enable_periodic
        
        # 内部状态
        self._fitness_history: List[float] = []
        self._violation_count: int = 0
        self._last_reflection_round: int = 0  # ★ 初始化为 0，避免第一轮就触发周期性反思
        self._best_fitness: Optional[float] = None
    
    def update(
        self,
        round: int,
        fitness: Optional[float],
        success: bool,
        errors: List[str] = None
    ) -> None:
        """更新触发器状态"""
        if fitness is not None:
            self._fitness_history.append(fitness)
            if len(self._fitness_history) > 20:
                self._fitness_history.pop(0)
            
            if self._best_fitness is None or fitness < self._best_fitness:
                self._best_fitness = fitness
        
        if not success:
            self._violation_count += 1
        else:
            self._violation_count = max(0, self._violation_count - 1)
    
    def check(self, round: int) -> Tuple[bool, Optional[TriggerReason]]:
        """
        检查是否需要触发反思
        
        Returns:
            (should_trigger, reason)
        """
        # ★ 冷却期检查：反思后需要等待至少 stagnation_window 轮才能再次触发
        # 这避免了连续每轮都触发反思的问题
        cooldown_rounds = self.stagnation_window
        if round - self._last_reflection_round < cooldown_rounds:
            return False, None
        
        # 1. 停滞检测
        if self._check_stagnation():
            return True, TriggerReason.STAGNATION
        
        # 2. 违规检测（连续违规）
        if self._violation_count >= 3:
            self._violation_count = 0  # 重置
            return True, TriggerReason.VIOLATION
        
        # 3. 退化检测
        if self._check_degradation():
            return True, TriggerReason.DEGRADATION
        
        # 4. 周期触发
        if self.enable_periodic:
            if round - self._last_reflection_round >= self.periodic_interval:
                return True, TriggerReason.PERIODIC
        
        return False, None
    
    def _check_stagnation(self) -> bool:
        """
        检查是否停滞
        
        停滞定义：最近 N 轮的最佳 fitness 相比之前 N 轮的最佳 fitness 没有改善
        需要至少 2*N 轮数据才能判断
        """
        window = self.stagnation_window
        if len(self._fitness_history) < window * 2:
            return False
        
        # 最近 N 轮的最佳值
        recent = self._fitness_history[-window:]
        recent_best = min(recent)  # fitness 越小越好
        
        # 之前 N 轮的最佳值
        previous = self._fitness_history[-window*2:-window]
        previous_best = min(previous)
        
        # 如果最近的最佳没有比之前的最佳改善至少 1%，认为停滞
        # 对于负数：recent_best < previous_best 表示改善
        improvement_ratio = (previous_best - recent_best) / abs(previous_best) if previous_best != 0 else 0
        
        # 改善不足 1% 认为停滞
        return improvement_ratio < 0.01
    
    def _check_degradation(self) -> bool:
        """
        检查是否退化
        
        只在有足够历史数据后才检查，避免早期波动频繁触发
        """
        # ★ 需要至少 stagnation_window 轮数据才检查退化
        # 这避免了早期波动频繁触发反思
        if len(self._fitness_history) < self.stagnation_window:
            return False
        
        if self._best_fitness is None:
            return False
        
        current = self._fitness_history[-1]
        
        # 相对退化（fitness 为负数，越小越好）
        if self._best_fitness < 0:
            relative_change = (current - self._best_fitness) / abs(self._best_fitness)
            return relative_change > self.degradation_threshold
        
        return False
    
    def mark_reflection_done(self, round: int) -> None:
        """标记反思已完成"""
        self._last_reflection_round = round
    
    def get_context(self) -> Dict[str, Any]:
        """获取触发器上下文（用于反思生成）"""
        return {
            "fitness_history": self._fitness_history[-10:],
            "best_fitness": self._best_fitness,
            "violation_count": self._violation_count,
            "last_reflection_round": self._last_reflection_round
        }


class SelfReflector:
    """
    自我反思生成器
    
    使用结构化模板生成反思内容，遵循 Reflexion 论文的设计：
    - 四段式结构
    - 聚焦可行动的改进
    - 避免泛泛而谈
    """
    
    # 反思模板（四段式）
    REFLECTION_TEMPLATE = """## 第 {round} 轮自我反思 ({trigger_reason})

### 1. 发生了什么
{what_happened}

### 2. 为什么（根因分析）
{why_analysis}

### 3. 学到什么（可迁移的教训）
{lessons_learned}

### 4. 下一步建议（具体行动）
{next_actions}
"""
    
    def __init__(
        self,
        memory_stream: MemoryStream,
        llm_client: Optional[Any] = None,  # 可选：用 LLM 生成更智能的反思
        llm_model: Optional[str] = None,   # 使用的模型名称
        max_reflection_length: int = 500
    ):
        self.memory_stream = memory_stream
        self.llm_client = llm_client
        self.llm_model = llm_model or "gpt-4o-mini"  # 默认用便宜的模型
        self.max_reflection_length = max_reflection_length
    
    def generate(
        self,
        round: int,
        trigger_reason: TriggerReason,
        context: Dict[str, Any]
    ) -> str:
        """
        生成结构化反思
        
        Args:
            round: 当前轮次
            trigger_reason: 触发原因
            context: 上下文信息（包含 fitness 历史、最近参数等）
        
        Returns:
            反思内容字符串
        """
        # ★ 如果有 LLM client，用 LLM 生成更智能的反思
        if self.llm_client is not None:
            try:
                return self._generate_llm_reflection(round, trigger_reason, context)
            except Exception as e:
                import traceback
                logger.warning(f"[Reflection] LLM 反思失败，回退到模板: {type(e).__name__}: {e}")
                logger.debug(f"[Reflection] 详细堆栈:\n{traceback.format_exc()}")
        
        # 回退到模板方法
        if trigger_reason == TriggerReason.STAGNATION:
            return self._generate_stagnation_reflection(round, context)
        elif trigger_reason == TriggerReason.VIOLATION:
            return self._generate_violation_reflection(round, context)
        elif trigger_reason == TriggerReason.DEGRADATION:
            return self._generate_degradation_reflection(round, context)
        else:
            return self._generate_periodic_reflection(round, context)
    
    def _generate_llm_reflection(
        self,
        round: int,
        trigger_reason: TriggerReason,
        context: Dict[str, Any]
    ) -> str:
        """使用 LLM 生成反思"""
        import asyncio
        
        fitness_history = context.get("fitness_history", [])
        best_fitness = context.get("best_fitness")
        recent_params = context.get("recent_params", [])
        recent_errors = context.get("recent_errors", [])
        
        # 构建简洁的数据摘要
        best_fitness_str = f"{best_fitness:.4f}" if isinstance(best_fitness, (int, float)) else "N/A"
        recent_fitness_str = (
            "[" + ", ".join(f"{f:.3f}" for f in fitness_history[-5:]) + "]"
            if fitness_history else "N/A"
        )
        data_summary = (
            f"当前轮次: {round}\n"
            f"触发原因: {trigger_reason.value}\n"
            f"历史最佳 fitness: {best_fitness_str}\n"
            f"最近 fitness: {recent_fitness_str}\n"
        )
        
        if recent_params and len(recent_params) >= 2:
            last = recent_params[-1]
            prev = recent_params[-2]
            changes = []
            for k in last:
                if k in prev and abs(last[k] - prev[k]) > 0.001:
                    changes.append(f"{k}: {prev[k]:.3f}→{last[k]:.3f}")
            if changes:
                data_summary += f"本轮参数变化: {', '.join(changes[:6])}\n"
        
        if recent_errors:
            data_summary += f"最近错误: {recent_errors[-2:]}\n"
        
        # 让 LLM 直接生成完整的结构化反思
        prompt = f"""你是一个电磁执行器优化的反思模块。基于以下数据生成结构化反思。

{data_summary}

请按以下格式输出（每部分 1-3 句话，总共不超过 200 字）：

## 第 {round} 轮自我反思 ({trigger_reason.value})

### 1. 发生了什么
[描述当前状态和触发原因]

### 2. 为什么（根因分析）
[分析具体原因，结合参数变化]

### 3. 学到什么（可迁移的教训）
[提炼可复用的规律，如"增大 hs 会改变 n2 对应的绕组容量区间"]

### 4. 下一步建议（具体行动）
[给出具体参数调整建议，如"将 hs 从 1.8 增大到 2.0"，不要泛泛而谈]"""

        # 异步调用 LLM（兼容 AsyncOpenAI）
        # ★方案B：辅助调用使用 1024（足够输出规则）
        async def _call_llm():
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,  # ★ 使用配置的模型（与主 Agent 相同）
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,  # ★合理限制：足够输出完整反思
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        
        # ★ 修复：在新线程中创建独立事件循环执行异步调用
        # 这是从同步代码调用异步函数的标准方式
        import concurrent.futures
        
        def run_async_in_thread():
            """在新线程中运行异步代码"""
            return asyncio.run(_call_llm())
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(run_async_in_thread)
                reflection = future.result(timeout=90)  # 90 秒超时（给 OpenRouter 足够时间）
        except concurrent.futures.TimeoutError:
            raise TimeoutError("LLM 反思调用超时（90秒），可能是 API 响应慢")
        except Exception as e:
            # 捕获所有异常并重新抛出，确保错误信息完整
            import traceback
            tb = traceback.format_exc()
            raise RuntimeError(f"LLM 反思调用失败: {type(e).__name__}: {e}\n{tb}")
        
        # LLM 直接生成完整结构，不再套模板
        return reflection
    
    def _generate_stagnation_reflection(self, round: int, context: Dict[str, Any]) -> str:
        """生成停滞反思 - 基于实际数据分析"""
        fitness_history = context.get("fitness_history", [])
        best_fitness = context.get("best_fitness")
        recent_params = context.get("recent_params", [])
        
        # 分析停滞原因
        what_happened = f"连续多轮 fitness 未能改善，当前最佳 fitness: {best_fitness:.4f}" if best_fitness else "连续多轮无有效仿真结果"
        
        # ★ 基于实际数据分析
        why_analysis = "数据分析：\n"
        if len(fitness_history) >= 3:
            recent_5 = fitness_history[-5:] if len(fitness_history) >= 5 else fitness_history
            variance = self._calculate_variance(recent_5)
            avg_fitness = sum(recent_5) / len(recent_5)
            why_analysis += f"- 最近 {len(recent_5)} 轮 fitness 范围: [{min(recent_5):.4f}, {max(recent_5):.4f}]\n"
            why_analysis += f"- 平均 fitness: {avg_fitness:.4f}，方差: {variance:.6f}\n"
            if variance < 0.001:
                why_analysis += "- 波动极小，可能陷入局部最优\n"
        
        # ★ 分析参数变化趋势
        lessons = ""
        if len(recent_params) >= 2:
            first_params = recent_params[0]
            last_params = recent_params[-1]
            changed_params = []
            unchanged_params = []
            for key in first_params:
                if key in last_params:
                    delta = abs(last_params[key] - first_params[key])
                    if delta > 0.001:
                        changed_params.append(f"{key}({first_params[key]:.3f}→{last_params[key]:.3f})")
                    else:
                        unchanged_params.append(key)
            if changed_params:
                lessons += f"- 变化的参数: {', '.join(changed_params[:5])}\n"
            if unchanged_params:
                lessons += f"- 未变化的参数: {', '.join(unchanged_params[:5])}\n"
        if not lessons:
            lessons = "- 参数数据不足，无法分析变化趋势"
        
        # ★ 具体建议
        actions = ""
        if len(recent_params) >= 1 and recent_params[-1]:
            last = recent_params[-1]
            if 'hs' in last and last['hs'] < 2.0:
                actions += f"- 当前 hs={last['hs']:.3f}，可尝试增大并检查相邻 n2 边界\n"
            if 'lm' in last and last['lm'] < 5.5:
                actions += f"- 当前 lm={last['lm']:.3f}，可尝试增大并检查相邻 n1 边界\n"
        if not actions:
            actions = "- 尝试调整 hs 或 lm，并检查相邻离散派生量边界"
        
        return self.REFLECTION_TEMPLATE.format(
            round=round,
            trigger_reason="停滞",
            what_happened=what_happened,
            why_analysis=why_analysis,
            lessons_learned=lessons,
            next_actions=actions
        )
    
    def _generate_violation_reflection(self, round: int, context: Dict[str, Any]) -> str:
        """生成违规反思"""
        recent_errors = context.get("recent_errors", [])
        recent_params = context.get("recent_params", {})
        
        what_happened = "连续多轮出现约束违规"
        if recent_errors:
            what_happened += f"：\n- " + "\n- ".join(recent_errors[:3])
        
        why_analysis = "违规通常因为：\n"
        why_analysis += "- 参数超出物理可行域\n"
        why_analysis += "- 几何约束（如 2dg+tb-hs≥0.1）未满足\n"
        why_analysis += "- 线圈匝数 n2 难以在当前几何下实现"
        
        lessons = "- 约束违规区域应被记忆并避免\n- 调整前应先检查约束裕度\n- 某些参数组合天然不可行"
        
        actions = "- 回退到上一个可行设计\n- 减小步长，谨慎探索边界\n- 优先调整不涉及约束的参数"
        
        return self.REFLECTION_TEMPLATE.format(
            round=round,
            trigger_reason="约束违规",
            what_happened=what_happened,
            why_analysis=why_analysis,
            lessons_learned=lessons,
            next_actions=actions
        )
    
    def _generate_degradation_reflection(self, round: int, context: Dict[str, Any]) -> str:
        """生成退化反思 - 基于实际数据分析"""
        fitness_history = context.get("fitness_history", [])
        best_fitness = context.get("best_fitness")
        recent_params = context.get("recent_params", [])
        
        current_fitness = fitness_history[-1] if fitness_history else None
        degradation = ((current_fitness - best_fitness) / abs(best_fitness) * 100) if current_fitness and best_fitness else 0
        
        what_happened = f"fitness 相比最佳退化了 {degradation:.1f}%（当前: {current_fitness:.4f} vs 最佳: {best_fitness:.4f}）" if current_fitness else "fitness 出现显著退化"
        
        # ★ 分析导致退化的参数变化
        why_analysis = "参数变化分析：\n"
        if len(recent_params) >= 2:
            prev = recent_params[-2] if len(recent_params) >= 2 else {}
            curr = recent_params[-1] if recent_params else {}
            significant_changes = []
            for key in curr:
                if key in prev:
                    delta = curr[key] - prev[key]
                    if abs(delta) > 0.01:
                        direction = "↑" if delta > 0 else "↓"
                        significant_changes.append(f"{key}{direction}{abs(delta):.3f}")
            if significant_changes:
                why_analysis += f"- 本轮主要变化: {', '.join(significant_changes)}\n"
                why_analysis += "- 这些变化可能导致了 fitness 退化\n"
            else:
                why_analysis += "- 本轮参数变化很小，退化可能是数值波动\n"
        else:
            why_analysis += "- 参数数据不足"
        
        # ★ 基于数据的具体建议
        lessons = ""
        if len(recent_params) >= 2:
            prev = recent_params[-2] if len(recent_params) >= 2 else {}
            curr = recent_params[-1] if recent_params else {}
            for key in curr:
                if key in prev:
                    delta = curr[key] - prev[key]
                    if abs(delta) > 0.01:
                        reverse = "减小" if delta > 0 else "增大"
                        lessons += f"- 本轮 {key} {reverse} 可能更好（逆转本轮调整）\n"
        if not lessons:
            lessons = "- 数据不足，建议谨慎调整"
        
        actions = f"- 考虑回退到 fitness={best_fitness:.4f} 附近的配置\n" if best_fitness else ""
        actions += "- 减小调整步长，避免跳过最优区域"
        
        return self.REFLECTION_TEMPLATE.format(
            round=round,
            trigger_reason="退化",
            what_happened=what_happened,
            why_analysis=why_analysis,
            lessons_learned=lessons,
            next_actions=actions
        )
    
    def _generate_periodic_reflection(self, round: int, context: Dict[str, Any]) -> str:
        """生成周期性反思 - 阶段性数据总结"""
        fitness_history = context.get("fitness_history", [])
        best_fitness = context.get("best_fitness")
        recent_params = context.get("recent_params", [])
        
        # 计算进展
        if len(fitness_history) >= 2:
            initial = fitness_history[0]
            current = fitness_history[-1]
            improvement = ((initial - current) / abs(initial) * 100) if initial else 0
            what_happened = f"已完成 {round} 轮优化，fitness 从 {initial:.4f} 改善到 {current:.4f}（改善 {improvement:.1f}%）"
        else:
            what_happened = f"已完成 {round} 轮优化"
        
        # ★ 详细数据分析
        why_analysis = "阶段性数据：\n"
        if best_fitness:
            why_analysis += f"- 历史最佳 fitness: {best_fitness:.4f}\n"
        if len(fitness_history) >= 5:
            recent_5 = fitness_history[-5:]
            why_analysis += f"- 最近 5 轮 fitness: {[f'{f:.3f}' for f in recent_5]}\n"
            why_analysis += f"- 最近 5 轮最佳: {min(recent_5):.4f}\n"
        
        # ★ 参数变化范围分析
        lessons = ""
        if len(recent_params) >= 3:
            param_ranges = {}
            for p in recent_params[-5:]:
                for k, v in p.items():
                    if k not in param_ranges:
                        param_ranges[k] = [v, v]
                    else:
                        param_ranges[k][0] = min(param_ranges[k][0], v)
                        param_ranges[k][1] = max(param_ranges[k][1], v)
            active_params = [(k, r[1]-r[0]) for k, r in param_ranges.items() if r[1]-r[0] > 0.01]
            if active_params:
                active_params.sort(key=lambda x: -x[1])
                lessons += f"- 调整最活跃的参数: {', '.join([f'{k}(Δ{d:.3f})' for k,d in active_params[:4]])}\n"
            static_params = [k for k, r in param_ranges.items() if r[1]-r[0] <= 0.01]
            if static_params:
                lessons += f"- 较少调整的参数: {', '.join(static_params[:4])}\n"
        if not lessons:
            lessons = "- 数据不足，无法分析参数调整模式"
        
        actions = f"- 当前距离最佳 fitness 差距: {abs(fitness_history[-1] - best_fitness):.4f}\n" if fitness_history and best_fitness else ""
        actions += "- 评估是否需要更大幅度探索未调整的参数"
        
        return self.REFLECTION_TEMPLATE.format(
            round=round,
            trigger_reason="周期性回顾",
            what_happened=what_happened,
            why_analysis=why_analysis,
            lessons_learned=lessons,
            next_actions=actions
        )
    
    def _calculate_variance(self, values: List[float]) -> float:
        """计算方差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
    
    def extract_actionable_rules(self, reflection: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        从反思中提取可行动的规则（LLM 增强版）
        
        Args:
            reflection: 反思内容
            context: 可选的上下文信息（包含 fitness_history, recent_params 等）
        
        Returns:
            提取的规则列表（最多 5 条）
        """
        # 如果有 LLM client，使用 LLM 提取更智能的规则
        if self.llm_client and self.llm_model:
            try:
                return self._extract_rules_with_llm(reflection, context)
            except Exception as e:
                logger.warning(f"[Reflection] LLM 规则提取失败: {e}，回退到简单提取")
        
        # 回退：简单规则提取
        return self._extract_rules_simple(reflection)
    
    def _extract_rules_simple(self, reflection: str) -> List[str]:
        """简单规则提取（基于文本解析）"""
        rules = []
        lines = reflection.split("\n")
        in_actions_section = False
        
        for line in lines:
            if "下一步建议" in line or "行动" in line or "学到什么" in line:
                in_actions_section = True
                continue
            if in_actions_section and line.strip().startswith("- "):
                rule = line.strip()[2:].strip()
                if rule and len(rule) > 10:  # 过滤太短的
                    rules.append(rule)
        
        return rules[:5]
    
    def _extract_rules_with_llm(self, reflection: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """使用 LLM 提取高质量规则"""
        import asyncio
        
        # 构建上下文摘要
        context_summary = ""
        if context:
            fitness_history = context.get("fitness_history", [])
            recent_params = context.get("recent_params", [])
            
            if fitness_history:
                recent_5 = fitness_history[-5:]
                context_summary += f"最近 fitness: {[f'{f:.3f}' for f in recent_5]}\n"
            
            if recent_params and len(recent_params) >= 2:
                last = recent_params[-1]
                prev = recent_params[-2]
                changes = []
                for k in last:
                    if k in prev and abs(last[k] - prev[k]) > 0.001:
                        changes.append(f"{k}: {prev[k]:.3f}→{last[k]:.3f}")
                if changes:
                    context_summary += f"参数变化: {', '.join(changes[:6])}\n"
        
        prompt = f"""你是一个电磁执行器优化的规则提取器。从以下反思中提取可复用的优化规则。

## 反思内容
{reflection}

{f"## 上下文数据" + chr(10) + context_summary if context_summary else ""}

## 要求
请提取 3-5 条**具体、可行动、可复用**的规则。每条规则应该：
1. 包含具体的参数名（如 hs, lm, ta, hslot 等）
2. 给出明确的数值或方向（如"增大到 2.0"或"保持在 1.5-1.8 范围"）
3. 解释因果关系（如"因为这会改变 n2 对应的绕组容量区间"）

## 输出格式
每条规则一行，以 "- " 开头，例如：
- 当 hs < 1.8 且 n2 长期不变时，可将 hs 增大 0.1-0.15mm 并检查相邻 n2 边界
- ta 降到 0.5 以下可以放松 hslot 的约束下限（因为 hslot ≥ ta×tb_ratio + 0.1）

只输出规则，不要其他内容。"""

        # ★方案B：辅助调用使用 1024（足够输出规则）
        async def _call_llm():
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,  # ★合理限制：足够输出完整规则
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        
        # ★ 修复：在新线程中运行异步调用，增加超时
        import concurrent.futures
        
        def run_async_in_thread():
            return asyncio.run(_call_llm())
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_async_in_thread)
            result = future.result(timeout=60)  # 增加超时到 60 秒
        
        # 解析 LLM 输出
        rules = []
        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                rule = line[2:].strip()
                if rule and len(rule) > 15:  # 过滤太短的
                    rules.append(rule)
        
        logger.info(f"[Reflection] 🧠 LLM 提取了 {len(rules)} 条规则")
        return rules[:5]


class ReflectionManager:
    """
    反思管理器（整合所有组件）
    
    提供简洁的 API 供 MCPClient 调用
    """
    
    def __init__(
        self,
        storage_path: str = "memory_stream.jsonl",
        stagnation_window: int = 5,
        degradation_threshold: float = 0.15,
        periodic_interval: int = 10,
        enable_periodic: bool = True,
        llm_client: Optional[Any] = None,  # ★ 可选：传入 LLM client 用于智能反思
        llm_model: Optional[str] = None    # ★ 使用的模型名称
    ):
        self.memory_stream = MemoryStream(storage_path)
        self.trigger = ReflectionTrigger(
            stagnation_window=stagnation_window,
            degradation_threshold=degradation_threshold,
            periodic_interval=periodic_interval,
            enable_periodic=enable_periodic
        )
        self.reflector = SelfReflector(self.memory_stream, llm_client=llm_client, llm_model=llm_model)
        self.llm_client = llm_client  # 保存引用
        self.llm_model = llm_model
        
        # 缓存
        self._recent_errors: List[str] = []
        self._recent_params: List[Dict[str, float]] = []
    
    def record_round(
        self,
        round: int,
        params: Dict[str, float],
        fitness: Optional[float],
        success: bool,
        errors: List[str] = None
    ) -> Optional[str]:
        """
        记录一轮结果，检查是否需要反思
        
        Returns:
            如果触发反思，返回反思内容；否则返回 None
        """
        # 1. 写入观察
        self.memory_stream.write_observation(
            round=round,
            params=params,
            fitness=fitness,
            success=success,
            errors=errors
        )
        
        # 2. 更新触发器
        self.trigger.update(round, fitness, success, errors)
        
        # 3. 更新缓存
        if errors:
            self._recent_errors.extend(errors)
            self._recent_errors = self._recent_errors[-10:]
        self._recent_params.append(params)
        if len(self._recent_params) > 10:
            self._recent_params.pop(0)
        
        # 4. 检查是否需要反思
        should_reflect, reason = self.trigger.check(round)
        
        if should_reflect and reason:
            return self._do_reflection(round, reason)
        
        return None
    
    def _do_reflection(self, round: int, reason: TriggerReason) -> str:
        """执行反思"""
        # 构建上下文
        context = self.trigger.get_context()
        context["recent_errors"] = self._recent_errors
        context["recent_params"] = self._recent_params
        
        # 生成反思
        reflection = self.reflector.generate(round, reason, context)
        
        # 写入记忆流
        self.memory_stream.write_reflection(
            round=round,
            reflection_content=reflection,
            trigger_reason=reason,
            fitness_before=context.get("fitness_history", [None])[-2] if len(context.get("fitness_history", [])) >= 2 else None,
            fitness_after=context.get("fitness_history", [None])[-1] if context.get("fitness_history") else None
        )
        
        # ★ 启用 LLM 规则提取（已升级为智能提取）
        rules = self.reflector.extract_actionable_rules(reflection, context)
        for rule in rules:
            self.memory_stream.write_rule(
                round=round,
                rule_content=rule,
                source="reflection"
            )
        
        # 标记反思完成
        self.trigger.mark_reflection_done(round)
        
        logger.info(f"[Reflection] 🪞 第 {round} 轮触发反思 ({reason.value})，提取了 {len(rules)} 条规则")
        
        return reflection
    
    def build_reflection_prompt(self, max_reflections: int = 3, max_rules: int = 5) -> str:
        """
        构建反思相关的 prompt 片段，用于注入到 Actor
        
        Returns:
            可直接添加到 system prompt 的文本
        """
        parts = []
        
        # 1. 最近的反思
        reflections = self.memory_stream.read_reflections(max_reflections)
        if reflections:
            parts.append("【历史反思】")
            for ref in reflections:
                # 只取关键部分，避免过长
                content = ref.content
                if len(content) > 300:
                    # 提取"下一步建议"部分
                    if "下一步建议" in content:
                        idx = content.index("下一步建议")
                        content = "..." + content[idx:idx+300]
                    else:
                        content = content[:300] + "..."
                parts.append(f"[第{ref.round}轮 - {ref.trigger_reason}]\n{content}\n")
        
        # 2. 提取的规则
        rules = self.memory_stream.read_rules(max_rules)
        if rules:
            parts.append("【从反思中学到的规则】")
            for rule in rules:
                parts.append(f"- {rule.content}")
        
        return "\n".join(parts) if parts else ""
    
    def get_recent_rules(self, max_rules: int = 5) -> List[str]:
        """
        获取最近的规则列表（用于 Block 系统）
        
        Returns:
            规则文本列表
        """
        rules = self.memory_stream.read_rules(max_rules)
        return [rule.content for rule in rules] if rules else []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取反思统计"""
        stream_stats = self.memory_stream.get_stats()
        trigger_context = self.trigger.get_context()
        
        return {
            "memory_stream": stream_stats,
            "trigger": {
                "best_fitness": trigger_context["best_fitness"],
                "last_reflection_round": trigger_context["last_reflection_round"],
                "violation_count": trigger_context["violation_count"]
            }
        }
    
    def clear(self) -> None:
        """清空反思记忆"""
        self.memory_stream.clear()
        self._recent_errors = []
        self._recent_params = []
        logger.info("[Reflection] 反思记忆已清空")


# 便捷函数
def create_reflection_manager(
    storage_path: str = "memory_stream.jsonl",
    stagnation_window: int = 5,
    periodic_interval: int = 10
) -> ReflectionManager:
    """创建反思管理器的便捷函数"""
    return ReflectionManager(
        storage_path=storage_path,
        stagnation_window=stagnation_window,
        periodic_interval=periodic_interval
    )
