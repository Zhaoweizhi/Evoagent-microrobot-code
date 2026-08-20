# 实现MCP客户端，参考https://modelcontextprotocol.io/quickstart/client
import asyncio
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
import json
import os
import sys
import math
from contextlib import AsyncExitStack
import time
from loguru import logger
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
from openai import OpenAI, AsyncOpenAI
from .mcp_adapter import BaseMCPAdapter

# RL 增强模块
from .feedback import FeedbackHandler, FeedbackType, FeedbackPriority
from .experience import ExperienceBuffer, Experience
from .strategy import StrategyManager, StrategyConfig, ExplorationStrategy
from .reward import RewardCalculator, RewardConfig

# 评论家模块
from .critic import CriticEnsemble, CriticScore

# Actor-Critic 系统
from .reflection import ReflectionManager, TriggerReason

# 统一记忆架构
from .memory import MemoryManager, ShortTermMemory, LongTermMemory

# Actor-Critic 系统
from .value_function import StateValueFunction, ActorCriticSystem, ValueEstimate

# ExpeL 对比批评
try:
    from .expel_critique import ContrastCritique, RuleManager, create_expel_system
    EXPEL_AVAILABLE = True
except ImportError:
    EXPEL_AVAILABLE = False

if TYPE_CHECKING:
    from .rag import RAGEngine


# 写一个adapter函数，将MCP tool中的字段转换为openai接口需要的tool_schema
LLM_TIMEOUT = 90  # ★ 从 600 秒改为 90 秒，超时后自动重试

# 上下文管理常量
MAX_CONTEXT_TOKENS = 200000  # 提前触发压缩，防止超限
CHARS_PER_TOKEN = 1.8  # 更接近中文实际比例
KEEP_RECENT_ROUNDS = 10  # ★新方案A：只保留最近 10 轮完整对话
PERIODIC_COMPRESS_INTERVAL = 20  # ★新方案C：每 20 轮强制压缩一次


class MCPClient:

    def __init__(self,
                 api_key: Optional[str],
                 base_url: str,
                 adapter: BaseMCPAdapter,
                 llm_timeout: int = LLM_TIMEOUT,
                 model: str = "qwen-plus",
                 stream: bool = True,
                 rag_engine: Optional["RAGEngine"] = None,
                 # RL 增强组件（可选，在 optimize 时自动初始化）
                 feedback_handler: Optional[FeedbackHandler] = None,
                 experience_buffer: Optional[ExperienceBuffer] = None,
                 strategy_manager: Optional[StrategyManager] = None,
                 reward_calculator: Optional[RewardCalculator] = None,
                 # 评论家集群（Actor-Critic架构）
                 critic_ensemble: Optional[CriticEnsemble] = None,
                 # Actor-Critic 系统
                 actor_critic_system: Optional[ActorCriticSystem] = None):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = AsyncOpenAI(api_key=api_key,
                               base_url=base_url,
                               timeout=llm_timeout)
        self.base_url = base_url
        self.model = model
        self.stream = stream
        self.adapter = adapter
        self.rag_engine = rag_engine

        # ★ 终端输出策略：
        # - 默认保持原行为：LLM streaming 会把分段内容实时打印到终端
        # - 如希望“终端与 log 一致（仅输出一次完整回复）”，可设置环境变量：
        #   MCP_STREAM_TO_CONSOLE=0
        #   这样会关闭 streaming 的分段打印，仅在每轮结束时输出一次“完整回复”
        self._stream_to_console: bool = os.getenv("MCP_STREAM_TO_CONSOLE", "1").lower() not in ("0", "false", "no")
        
        # ★ 详细输出模式（LLM 思考过程完整可见）：
        # - 默认：简洁模式，输出【批评】【总结】【方案】共100字内
        # - 设置 MCP_VERBOSE_LLM=1 或 LLM_VERBOSE=1：详细模式，输出完整的目标复述、候选参数、约束检查、
        #   物理机理分析、与历史最佳对比、下一轮调整思路等，便于论文/调试
        self._verbose_llm_output: bool = (
            os.getenv("MCP_VERBOSE_LLM", "0").lower() in ("1", "true", "yes") or
            os.getenv("LLM_VERBOSE", "0").lower() in ("1", "true", "yes")
        )
        
        # RL 增强组件
        self.feedback_handler = feedback_handler
        self.experience_buffer = experience_buffer
        self.strategy_manager = strategy_manager
        self.reward_calculator = reward_calculator
        
        # 评论家集群（用于稠密奖励）
        self.critic_ensemble = critic_ensemble
        
        # Actor-Critic 系统（整合 V(s) 和动作评论家）
        self.actor_critic_system = actor_critic_system
        
        # Reflexion 反思管理器
        self.reflection_manager: Optional[ReflectionManager] = None
        
        # 统一记忆管理器（整合短期/长期记忆）
        self.memory_manager: Optional[MemoryManager] = None
        
        # 当前状态（用于 RL）
        self._current_state: Optional[Dict[str, float]] = None
        self._previous_state: Optional[Dict[str, float]] = None

        # 评论家评分缓存（用于记录实际结果后更新）
        self._last_critic_scores: Optional[Dict[str, CriticScore]] = None
        self._last_dense_reward: float = 0.0
        self._last_value_estimate: Optional[ValueEstimate] = None
        
        # 评论家反馈缓存（用于注入 LLM 上下文）
        self._pending_critic_feedback: Optional[str] = None
        # 磁饱和警告缓存（用于注入 LLM 上下文）
        self._pending_saturation_warning: Optional[str] = None
        # Google 严格工具序场景下，延后到下一轮 user 提示中注入
        self._deferred_system_notes: List[str] = []
        
        # ★滑窗奖励融合配置（开关）
        self._enable_reward_smoothing: bool = False  # 是否启用滑窗平滑
        self._reward_window_size: int = 3            # 滑窗大小（默认 3 轮）
        self._reward_history_window: List[float] = []  # 滑窗奖励历史
        
        # ★策略库全部暴露开关（前期探索用）
        self._full_strategy_exposure: bool = False  # 是否把策略库全部暴露给 LLM（不筛选）
        
        # ★统一开关：离散探索提示 + 派生变量状态提示（默认开启）
        self._enable_discrete_guidance: bool = True
        
        # ★磁饱和软约束（拉格朗日乘子）配置
        self._saturation_as_constraint: bool = False
        self._sat_threshold_t: float = 2.0          # B_max 阈值（T）
        self._sat_scale_t: float = 0.2              # 归一化尺度（T），g = max(0, (B-阈值)/scale)
        self._sat_lambda: float = 0.0               # 拉格朗日乘子 λ ≥ 0
        self._sat_lambda_lr: float = 0.05           # λ 更新学习率
        self._sat_target: float = 0.0               # 允许的平均违背（0=尽量不违背）
        
        # ★ ExpeL 对比批评系统
        self.contrast_critique: Optional[Any] = None
        self._expel_enabled: bool = False
        
        # ★约束违规重试机制
        self._constraint_violation_in_round: bool = False  # 本轮是否发生约束违规
        self._last_constraint_errors: List[str] = []       # 最近的约束错误信息
        
        # ★每轮只执行一次仿真的控制标志
        self._sim_executed_this_round: bool = False
        
        # ★AEDT 周期性清理配置（防止长跑卡顿/端口不释放）
        # 可通过环境变量覆盖：AEDT_CLEANUP_INTERVAL / AEDT_CLEANUP_COOLDOWN
        self._aedt_cleanup_interval = int(os.getenv("AEDT_CLEANUP_INTERVAL", "50"))
        self._aedt_cleanup_cooldown = float(os.getenv("AEDT_CLEANUP_COOLDOWN", "5"))
        self._simulation_count = 0  # 仿真成功完成计数器
        
        # ★邮件通知配置
        self._notify_email: Optional[str] = None
        self._smtp_server: str = "smtp.qq.com"
        self._smtp_port: int = 587
        self._smtp_password: Optional[str] = None
        # 工具调用超时看门狗（秒）
        self._tool_timeout_seconds: int = int(os.getenv("MCP_TOOL_TIMEOUT_SECONDS", "180"))
        self._maxwell_timeout_seconds: int = int(os.getenv("MAXWELL_SIM_TIMEOUT_SECONDS", "900"))
        self._last_timeout_alert_round: int = -1

    def _log(self, message: str, level: str = "info", end: str = "\n", console: bool = False) -> None:
        """写入日志文件（默认不输出到终端，避免重复）。
        
        设计原则：
        - 日志文件是持久化记录，始终写入
        - 终端输出用 _console() 方法控制，避免重复
        - streaming 开启时 LLM 回复已实时显示，不需要重复打印
        """
        if level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)
        if console:
            print(message, end=end, flush=True)
    
    def _console(self, message: str, end: str = "\n", important: bool = False) -> None:
        """仅输出到终端（不写日志，用于避免重复）。"""
        print(message, end=end, flush=True)
        if important:
            logger.info(message)
    
    def _log_round_header(self, iteration: int, total: int = None) -> None:
        """打印清晰的轮次开始分隔符"""
        sep = "=" * 60
        header = f"\n{sep}\n【第 {iteration} 轮】"
        if total:
            header += f" / {total}"
        header += f"\n{sep}"
        self._console(header, important=True)
    
    def _log_round_summary(self, iteration: int, params: dict = None, 
                           fitness: float = None, n1: int = None, n2: int = None,
                           prev_params: dict = None, prev_fitness: float = None,
                           prev_n1: int = None, prev_n2: int = None) -> None:
        """打印轮次结束摘要（关键变化一目了然）"""
        sep = "-" * 60
        lines_out = [sep, f"【第 {iteration} 轮摘要】"]
        
        # fitness 变化
        if fitness is not None:
            if prev_fitness is not None:
                delta = fitness - prev_fitness
                arrow = "+" if delta > 0 else ("-" if delta < 0 else "=")
                lines_out.append(f"  fitness: {prev_fitness:.4f} {arrow}> {fitness:.4f} (d={delta:+.4f})")
            else:
                lines_out.append(f"  fitness: {fitness:.4f}")
        
        # n1/n2 变化（离散变量，最重要）
        if n1 is not None and n2 is not None:
            n1_str = f"{prev_n1}->{n1}" if prev_n1 is not None and prev_n1 != n1 else str(n1)
            n2_str = f"{prev_n2}->{n2}" if prev_n2 is not None and prev_n2 != n2 else str(n2)
            changed = (prev_n1 != n1 or prev_n2 != n2) if prev_n1 is not None else False
            status = "[CHANGED]" if changed else "[no change]"
            lines_out.append(f"  n1={n1_str}, n2={n2_str} {status}")
        
        # 参数变化（只显示变化的）
        if params and prev_params:
            changes = []
            for k, v in params.items():
                prev_v = prev_params.get(k)
                if prev_v is not None and abs(v - prev_v) > 0.001:
                    changes.append(f"{k}: {prev_v:.3f}->{v:.3f}")
            if changes:
                lines_out.append("  params changed: " + ", ".join(changes[:5]))
            else:
                lines_out.append("  params: no change")
        
        lines_out.append(sep)
        self._console("\n".join(lines_out), important=True)

    def _cleanup_aedt_processes(self):
        """清理残留的 AEDT 进程，防止长跑卡顿/端口不释放"""
        import subprocess
        import time
        
        process_names = ["ansysedt.exe", "ansysedtsv.exe"]
        cleaned = False
        
        for proc_name in process_names:
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", proc_name],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode == 0:
                    cleaned = True
            except Exception:
                pass
        
        if cleaned:
            self._log(f"[AEDT] 已清理残留 AEDT 进程")
        
        # 清理后等待，让端口释放
        if self._aedt_cleanup_cooldown > 0:
            time.sleep(self._aedt_cleanup_cooldown)

    def _check_and_cleanup_aedt(self):
        """检查是否需要周期性清理 AEDT"""
        if self._aedt_cleanup_interval <= 0:
            return  # 禁用
        
        if self._simulation_count > 0 and self._simulation_count % self._aedt_cleanup_interval == 0:
            self._log(f"[AEDT] 触发周期性清理 (每 {self._aedt_cleanup_interval} 次仿真)", level="warning")
            self._cleanup_aedt_processes()

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """估算消息列表的 token 数量"""
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        # 工具调用也要计算
        for msg in messages:
            if "tool_calls" in msg:
                total_chars += len(json.dumps(msg["tool_calls"], ensure_ascii=False))
        return int(total_chars / CHARS_PER_TOKEN)
    
    def _slim_tool_result(self, tool_name: str, tool_content: str) -> str:
        """★新方案B：精简工具返回结果，只保留关键字段
        
        对于 run_maxwell_simulation，只保留：
        - status, fitness, n1, n2, total_turns
        - avg_B, B_sat, kb, pb
        - errors（如果有）
        
        删除冗余字段：logs, fld_file, derived_dimensions 等
        """
        if tool_name != "run_maxwell_simulation":
            # 其他工具不处理
            return tool_content
        
        try:
            payload = json.loads(tool_content)
            result = payload.get("result", {})
            
            # 只保留关键字段
            slim_result = {
                "status": result.get("status"),
                "fitness": result.get("fitness"),
                "avg_B": result.get("avg_B"),
                "B_sat": result.get("B_sat"),
                "kb": result.get("kb"),
                "pb": result.get("pb"),
            }
            
            # 匝数信息
            turns = result.get("turns", {})
            if turns:
                slim_result["turns"] = {
                    "n1": turns.get("n1"),
                    "n2": turns.get("n2"),
                    "total": turns.get("total"),
                }
            
            # 饱和信息（如果有）
            if result.get("is_saturated"):
                slim_result["is_saturated"] = True
                slim_result["saturation_region"] = result.get("saturation_region", "")
                slim_result["saturation_suggestion"] = result.get("saturation_suggestion", "")
            
            # 错误信息（如果有）
            errors = result.get("errors")
            if errors:
                slim_result["errors"] = errors
            
            # 重新构建精简的 payload（不包含 logs）
            slim_payload = {"result": slim_result}
            
            return json.dumps(slim_payload, ensure_ascii=False, indent=2)
            
        except Exception:
            # 解析失败，返回原内容
            return tool_content

    def _sanitize_tool_calls(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理孤立的 tool 消息，确保每个 tool 响应都有对应的 assistant tool_call
        
        OpenAI API 要求：每个 role=tool 的消息必须有配对的 assistant 消息中的 tool_calls。
        压缩后可能破坏这种对应关系，导致 "No tool call found for function call output" 错误。
        """
        if not messages:
            return messages
        
        # 收集所有 assistant 消息中的 tool_call_id
        valid_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg.get("tool_calls", []):
                    call_id = tc.get("id")
                    if call_id:
                        valid_call_ids.add(call_id)
        
        # 过滤掉孤立的 tool 消息
        sanitized = []
        removed_count = 0
        for msg in messages:
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id and tool_call_id not in valid_call_ids:
                    # 孤立的 tool 消息，跳过
                    removed_count += 1
                    continue
            sanitized.append(msg)
        
        if removed_count > 0:
            self._log(f"[Context] 清理了 {removed_count} 条孤立的 tool 消息")
        
        return sanitized

    def _requires_strict_tool_turn_protocol(self) -> bool:
        """
        某些 Google/Gemini 兼容端要求严格的 tool 调用顺序：
        function call 后必须紧跟 function response，期间不允许插入其他消息。
        """
        model_name = str(getattr(self, "model", "") or "").lower()
        base_url = str(getattr(self, "base_url", "") or "").lower()
        return (
            ("gemini" in model_name)
            or ("google" in model_name)
            or ("generativelanguage" in base_url)
            or ("google" in base_url)
            or ("aistudio" in base_url)
        )

    def _compress_context(self, messages: List[Dict[str, Any]], best_fitness: float, 
                          best_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        压缩上下文，保留关键信息：
        - 系统提示（前面的 system messages）
        - 最近 N 轮对话
        - 最佳结果摘要
        """
        if not messages:
            return messages
        
        # 分离系统消息和对话历史
        #
        # 重要：不能“永久保留所有 role=system 的消息”，因为在 optimize 过程中会不断注入 system
        # （评论家反馈/饱和提示/实时反馈等），这些会无限累积导致上下文仍然爆炸。
        # 这里仅保留“开头的系统消息前缀”（RAG/主 system_prompt/初始策略等），其余一律作为可裁剪历史。
        system_messages: List[Dict[str, Any]] = []
        conversation_messages: List[Dict[str, Any]] = []
        saw_non_system = False
        for msg in messages:
            if not saw_non_system and msg.get("role") == "system":
                system_messages.append(msg)
            else:
                saw_non_system = True
                conversation_messages.append(msg)

        # 额外保险：系统前缀也要限长（通常 2~3 条足够）
        keep_system_n = 3
        if len(system_messages) > keep_system_n:
            system_messages = system_messages[:keep_system_n]
        
        # 估算每轮对话大约占用的消息数（user + assistant + tool_call + tool_result）
        msgs_per_round = 4
        keep_msgs = KEEP_RECENT_ROUNDS * msgs_per_round
        
        if len(conversation_messages) <= keep_msgs:
            # 不需要压缩
            return messages
        
        # 保留最近 N 轮对话
        recent_messages = conversation_messages[-keep_msgs:]
        
        # 创建压缩摘要
        removed_rounds = (len(conversation_messages) - keep_msgs) // msgs_per_round
        summary = f"""[上下文压缩摘要]
已完成 {removed_rounds} 轮历史迭代（已压缩）。
当前最佳 fitness: {best_fitness:.6f}
"""
        if best_params:
            param_text = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                                   for k, v in best_params.items() if v is not None)
            summary += f"最佳参数: {param_text}\n"
        
        summary += """
请继续优化，目标是进一步降低 fitness。
保持调用工具顺序：validate_maxwell_design → run_maxwell_simulation"""
        
        summary_message = {"role": "system", "content": summary}
        
        # 重新组合消息：系统消息(前缀) + 摘要 + 最近对话
        compressed = system_messages + [summary_message] + recent_messages
        
        # ★关键：清理压缩后可能产生的孤立 tool 消息，避免 API 报错
        compressed = self._sanitize_tool_calls(compressed)
        
        self._log(f"[Context] 上下文已压缩：{len(messages)} → {len(compressed)} 条消息")
        self._log(f"[Context] 压缩了 {removed_rounds} 轮历史对话")
        
        return compressed

    def _check_and_compress_context(self, messages: List[Dict[str, Any]], 
                                     best_fitness: float,
                                     best_params: Optional[Dict] = None,
                                     current_iteration: int = 0) -> List[Dict[str, Any]]:
        """检查上下文长度，必要时进行压缩
        
        ★新方案A+C：主动压缩策略
        1. 只保留最近 N 轮完整对话（不等超限就压缩）
        2. 每 20 轮强制压缩一次
        """
        estimated_tokens = self._estimate_tokens(messages)
        
        # ★新方案C：定期强制压缩（每 PERIODIC_COMPRESS_INTERVAL 轮）
        if current_iteration > 0 and current_iteration % PERIODIC_COMPRESS_INTERVAL == 0:
            self._log(f"[Context] 🔄 定期压缩（第 {current_iteration} 轮）...")
            messages = self._compress_context(messages, best_fitness, best_params)
            new_tokens = self._estimate_tokens(messages)
            self._log(f"[Context] ✅ 定期压缩完成：{estimated_tokens:,} → {new_tokens:,} tokens")
            return messages
        
        # ★新方案A：主动压缩 - 消息数超过阈值就压缩（不等 token 超限）
        # 估算：每轮约 4 条消息，保留 10 轮 = 40 条 + 系统消息约 5 条 = 45 条
        max_messages = KEEP_RECENT_ROUNDS * 4 + 10  # 留一些余量
        if len(messages) > max_messages:
            self._log(f"[Context] 📦 主动压缩（消息数 {len(messages)} > {max_messages}）...")
            messages = self._compress_context(messages, best_fitness, best_params)
            new_tokens = self._estimate_tokens(messages)
            self._log(f"[Context] ✅ 主动压缩完成：{len(messages)} 条消息，约 {new_tokens:,} tokens")
            return messages
        
        # 兜底：token 超限时压缩
        if estimated_tokens > MAX_CONTEXT_TOKENS:
            self._log(f"[Context] ⚠️ 上下文接近限制：约 {estimated_tokens:,} tokens，开始压缩...", level="warning")
            messages = self._compress_context(messages, best_fitness, best_params)
            new_tokens = self._estimate_tokens(messages)
            self._log(f"[Context] ✅ 压缩完成：{estimated_tokens:,} → {new_tokens:,} tokens")

            # 若压缩后仍偏大（估算误差/系统前缀过长），继续走强制压缩兜底
            if new_tokens > MAX_CONTEXT_TOKENS:
                self._log(f"[Context] ⚠️ 压缩后仍偏大（约 {new_tokens:,} tokens），触发强制压缩...", level="warning")
                messages = self._force_compress_context(messages, best_fitness, best_params)
                forced_tokens = self._estimate_tokens(messages)
                self._log(f"[Context] ✅ 强制压缩完成：约 {forced_tokens:,} tokens")
        
        return messages

    def _force_compress_context(self, messages: List[Dict[str, Any]], 
                                 best_fitness: float,
                                 best_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """强制压缩上下文（API 返回超限错误时使用）
        
        比普通压缩更激进：只保留最近 3 轮对话
        """
        if not messages:
            return messages
        
        # 分离系统消息和对话历史
        system_messages = []
        conversation_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                conversation_messages.append(msg)
        
        # 强制只保留最近 3 轮对话（非常激进）
        force_keep_rounds = 3
        msgs_per_round = 4
        keep_msgs = force_keep_rounds * msgs_per_round
        
        # 保留最近的消息
        recent_messages = conversation_messages[-keep_msgs:] if len(conversation_messages) > keep_msgs else conversation_messages
        
        # 计算被压缩的轮数
        removed_count = len(conversation_messages) - len(recent_messages)
        removed_rounds = removed_count // msgs_per_round
        
        # 创建强制压缩摘要
        summary = f"""[强制上下文压缩 - API 超限触发]
已完成约 {removed_rounds} 轮历史迭代（已全部压缩以满足 API 限制）。
当前最佳 fitness: {best_fitness:.6f}
"""
        if best_params:
            param_text = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                                   for k, v in list(best_params.items())[:8])  # 只保留前8个参数
            summary += f"最佳参数: {param_text}\n"
        
        summary += """
请继续优化，目标是进一步降低 fitness。
保持调用工具顺序：validate_maxwell_design → run_maxwell_simulation"""
        
        summary_message = {"role": "system", "content": summary}
        
        # 保留前几条 system（通常包含 RAG + 主 system_prompt + 关键约束/策略），避免把核心指令压没
        keep_system_n = 3
        kept_system = system_messages[:keep_system_n] if system_messages else []
        compressed = kept_system + [summary_message] + recent_messages
        
        # ★关键：清理压缩后可能产生的孤立 tool 消息，避免 API 报错
        compressed = self._sanitize_tool_calls(compressed)
        
        self._log(f"[Context] 🔥 强制压缩：{len(messages)} → {len(compressed)} 条消息")
        self._log(f"[Context] 强制删除了 {removed_rounds} 轮历史对话")
        
        return compressed

    async def connect_to_mcp_server_stdio(self, server_script_path: str):
        """连接到一个MCP服务端 stdio模式

        Args:
            server_script_path (str): Path to the server script (.py or .js)
        """

        # 判断文件格式
        is_python = server_script_path.endswith('.py')

        # 目前仅支持python脚本
        if not is_python:
            raise ValueError("仅支持python脚本")

        # 使用当前解释器，避免在多环境场景下出现依赖不一致
        command = sys.executable if is_python else None

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "1")

        # StdioServerParameters类就是封装一句命令行的命令（类），比如python server.py
        server_params = StdioServerParameters(command=command,
                                              args=[server_script_path],
                                              env=env)

        # 官方提供，建议背板：使用AsyncExitStack管理异步资源，建立与该进程的标准输入输出通信管道
        # 将stdio客户端添加到异步上下文栈中，这个过程会：
        # 1.启动MCP服务器进程
        # 2.建立与该进程的标准输入输出通信管道
        # 3.返回一个包含读写接口的传输对象
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))  # 执行命令起服务，并和它连接上
        self.stdio, self.write = stdio_transport  # 获取对服务端进程的读取和写入接口
        # 将客户端会话添加到异步上下文栈中，这个过程会：
        # 1.初始化客户端会话
        # 2.建立与服务器的会话连接
        # 3.设置必要的会话参数和状态
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write))

        # 整个过程的数据流如下：
        # 客户端代码（比如发送消息） -> self.write -> 服务端进程的标准输入
        # 服务端进程的标准输出 -> self.stdio -> 客户端代码（比如接收消息）

        # 初始化会话，保障客户端<->服务端通信正常
        await self.session.initialize()

        # 客户端向服务端发送一个请求获取工具列表，服务端返回工具列表
        response = await self.session.list_tools()
        tools = response.tools

        self._log('\n已和MCP服务端连接完成，工具包含： ' + str([tool.name for tool in tools]))

    async def connect_to_mcp_server_sse(self, server_url: str):
        """使用SSE模式连接到MCP服务端(测试中……)

        Args:
            server_url (str): MCP服务端的SSE URL
        """
        from mcp.client.sse import sse_client

        # 使用AsyncExitStack管理异步资源，建立与SSE服务器的连接
        streams_context = sse_client(url=server_url)
        streams = await self.exit_stack.enter_async_context(streams_context)

        # 创建会话
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(*streams))

        # 初始化会话，保障客户端<->服务端通信正常
        await self.session.initialize()

        # 获取工具列表
        response = await self.session.list_tools()
        tools = response.tools

        print('\n已和MCP服务端SSE连接完成，工具包含：', [tool.name for tool in tools])

    async def _call_tool_directly(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """直接调用 MCP 工具（不经过 LLM）
        
        用于第二阶段：远程 Actor 自己做决策时，直接调用工具执行
        
        Args:
            tool_name: 工具名称（validate_maxwell_design 或 run_maxwell_simulation）
            params: 参数字典
        
        Returns:
            工具返回的结果字典
        """
        self._log(f"[MCP Direct] 直接调用工具 {tool_name}，参数: {params}")
        
        try:
            timeout_seconds = self._maxwell_timeout_seconds if tool_name == "run_maxwell_simulation" else self._tool_timeout_seconds
            tool_response = await asyncio.wait_for(
                self.session.call_tool(
                    name=tool_name,
                    arguments=params,
                ),
                timeout=timeout_seconds,
            )
            
            # 解析响应
            result = {}
            if getattr(tool_response, "content", None):
                for block in tool_response.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "text" and hasattr(block, "text"):
                        try:
                            result = json.loads(block.text)
                        except json.JSONDecodeError:
                            result = {"raw": block.text}
            
            self._log(f"[MCP Direct] 工具 {tool_name} 返回: status={result.get('status', 'unknown')}")
            
            # 如果是仿真结果，更新内部状态
            if tool_name == "run_maxwell_simulation" and result.get("status") == "ok":
                self._last_sim_result = {
                    "iteration": getattr(self, '_current_iteration', 0),
                    "result": result,
                    "params": params,
                }
                # 周期性清理 AEDT 进程
                self._simulation_count += 1
                self._check_and_cleanup_aedt()
            
            return result
            
        except Exception as e:
            self._log(f"[MCP Direct] 工具 {tool_name} 调用失败: {e}", level="error")
            return {"status": "error", "errors": [str(e)]}

    async def _chat_with_tools(
        self,
        messages,
        llm_timeout: Optional[int] = None,
        iteration: Optional[int] = None,
        stream_override: Optional[bool] = None,
        tool_choice_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """核心对话逻辑，支持工具调用。返回 (模型回复, 是否执行过工具)
        
        Args:
            messages: 对话消息
            llm_timeout: LLM 超时
            iteration: 当前迭代轮次（用于将工具结果写入 CSV）
            stream_override: 覆盖是否流式（用于兼容部分端点流式工具调用异常）
            tool_choice_override: 覆盖 tool_choice（例如强制 validate_maxwell_design）
        """

        response = await self.session.list_tools()
        available_tools = self.adapter.convert_to_tool_schema(response.tools)

        use_stream = self.stream if stream_override is None else stream_override
        tool_choice = tool_choice_override if tool_choice_override is not None else "auto"

        # ★方案B：合理限制 max_tokens（避免 API 按 max_tokens 计费时浪费）
        # 主循环：4096 足够输出参数+分析+工具调用
        llm_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": available_tools,
            "tool_choice": tool_choice,
            "stream": use_stream,
            "max_tokens": 4096,  # ★合理限制：足够输出完整内容
        }
        if llm_timeout:
            llm_kwargs["timeout"] = llm_timeout
        
        # ★ 使用 asyncio.wait_for 强制超时（防止 API 无限期阻塞）
        effective_timeout = llm_timeout or LLM_TIMEOUT
        try:
            response = await asyncio.wait_for(
                self.llm.chat.completions.create(**llm_kwargs),
                timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"LLM API 调用超时（{effective_timeout}秒），请检查网络或更换模型")

        collected_content: List[str] = []
        collected_tool_calls = []
        current_tool_call = None

        # 处理流式和非流式两种响应格式
        if use_stream:
            # 流式响应：逐 chunk 处理
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    collected_content.append(chunk_content)
                    if self._stream_to_console:
                        print(chunk_content, end="", flush=True)

                if chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        if tool_call.index is not None:
                            if (current_tool_call is None
                                    or tool_call.index != current_tool_call[
                                        "index"]):
                                if current_tool_call:
                                    collected_tool_calls.append(current_tool_call)
                                current_tool_call = {
                                    "id": tool_call.id or "",
                                    "type": "function",
                                    "index": tool_call.index,
                                    "function": {
                                        "name": "",
                                        "arguments": ""
                                    },
                                }

                        if tool_call.function and tool_call.function.name:
                            current_tool_call["function"]["name"
                                                          ] = tool_call.function.name
                        if tool_call.function and tool_call.function.arguments:
                            current_tool_call["function"][
                                "arguments"] += tool_call.function.arguments
        else:
            # 非流式响应：直接从 response 提取（不在这里打印，由调用方打印）
            if response.choices and response.choices[0].message:
                msg = response.choices[0].message
                if msg.content:
                    collected_content.append(msg.content)
                
                # 处理工具调用
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        collected_tool_calls.append({
                            "id": tc.id or "",
                            "type": "function",
                            "index": 0,
                            "function": {
                                "name": tc.function.name if tc.function else "",
                                "arguments": tc.function.arguments if tc.function else ""
                            },
                        })

        if current_tool_call:
            collected_tool_calls.append(current_tool_call)
        
        # 记录模型计划调用的工具，便于日志留痕
        if collected_tool_calls:
            for tc in collected_tool_calls:
                self._log(
                    f"[MCP Client] 收到工具调用: {tc['function']['name']} args={tc['function']['arguments']}"
                )

        if not collected_tool_calls:
            return "".join(collected_content).strip(), False

        final_response_parts: List[str] = []
        remaining_tool_calls = collected_tool_calls[:]
        tool_executed = False

        while remaining_tool_calls:
            tool_call_entry = remaining_tool_calls.pop(0)
            tool_name_to_call = tool_call_entry["function"]["name"]
            
            # ★每轮只允许执行一次 run_maxwell_simulation，跳过后续重复调用
            if tool_name_to_call == "run_maxwell_simulation" and getattr(self, '_sim_executed_this_round', False):
                self._log(f"[MCP Client] ⚠️ 本轮已执行过仿真，跳过重复的 run_maxwell_simulation 调用")
                continue
            
            assistant_message = {
                "role": "assistant",
                "content": "".join(collected_content).strip(),
                "tool_calls": [tool_call_entry],
            }

            raw_arguments = tool_call_entry["function"]["arguments"]
            parsed_arguments = None
            if raw_arguments:
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    parsed_arguments = {"input": raw_arguments}

            self._log(
                f"[MCP Client] 调用工具 {tool_name_to_call}，参数: {parsed_arguments}")
            timeout_seconds = self._maxwell_timeout_seconds if tool_name_to_call == "run_maxwell_simulation" else self._tool_timeout_seconds
            try:
                tool_response = await asyncio.wait_for(
                    self.session.call_tool(
                        name=tool_name_to_call,
                        arguments=parsed_arguments,
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                timeout_msg = f"工具 {tool_name_to_call} 调用超时（{timeout_seconds}秒）"
                self._log(f"[MCP Client] ⏱️ {timeout_msg}", level="error")
                # Maxwell 卡住时发送告警邮件（同一轮只发一次，避免刷屏）
                current_round = int(iteration) if iteration is not None else -1
                if (
                    tool_name_to_call == "run_maxwell_simulation"
                    and self._notify_email
                    and current_round != self._last_timeout_alert_round
                ):
                    self._last_timeout_alert_round = current_round
                    self._send_notification_email(
                        "MCP ALERT - Maxwell No Response",
                        f"Maxwell simulation timed out.\n\nRound: {current_round}\nTimeout: {timeout_seconds}s\nPlease check Maxwell status.",
                    )
                raise TimeoutError(timeout_msg)
            tool_executed = True

            tool_content_parts = []
            if getattr(tool_response, "content", None):
                for block in tool_response.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "text" and hasattr(block, "text"):
                        tool_content_parts.append(block.text)
                    else:
                        try:
                            tool_content_parts.append(
                                json.dumps(block.model_dump(),
                                           ensure_ascii=False))
                        except Exception:
                            tool_content_parts.append(str(block))

            if not tool_content_parts and getattr(tool_response,
                                                  "structuredContent", None):
                tool_content_parts.append(
                    json.dumps(tool_response.structuredContent,
                               ensure_ascii=False))

            if not tool_content_parts:
                tool_content_parts.append(
                    tool_response.model_dump_json(ensure_ascii=False))

            tool_content = "\n".join(tool_content_parts)
            # 简化输出：只打印关键信息，不打印完整 JSON
            tool_name = tool_call_entry['function']['name']
            summary = self._summarize_tool_response(tool_name, tool_content)
            self._log(f"[MCP Client] 工具 {tool_name} 返回：{summary}")

            # 如果是仿真结果，记录本轮结果（无论 status 是否 ok，便于 CSV 落盘）
            if tool_name == "run_maxwell_simulation":
                # ★标记本轮已执行仿真（用于限制每轮只执行一次）
                self._sim_executed_this_round = True
                
                try:
                    payload = json.loads(tool_content)
                    result = payload.get("result", {})
                    # 只保留设计变量，过滤掉 project_name, design_name 等非设计参数（包含 tb_ratio）
                    design_params = {}
                    design_var_names = {"lm", "tm", "ta", "dg", "hs", "wslot", "hslot", "s", "wa", "tb_ratio"}
                    for k, v in (parsed_arguments or {}).items():
                        if k in design_var_names and v is not None:
                            design_params[k] = v

                    # ★补充：将仿真返回的派生/线圈/尺寸参数也纳入，用于敏感性分析 & 前端展示
                    try:
                        sensitivity_keys = (
                            self.strategy_manager.SENSITIVITY_PARAM_BOUNDS
                            if self.strategy_manager else StrategyManager.SENSITIVITY_PARAM_BOUNDS
                        )
                    except Exception:
                        sensitivity_keys = design_var_names

                    for k in sensitivity_keys:
                        if k in design_params:
                            continue
                        if k in result and result[k] is not None:
                            try:
                                design_params[k] = float(result[k])
                            except Exception:
                                design_params[k] = result[k]
                    self._last_sim_result = {
                        # 这里的 iteration 已经是外层传入的当前轮次编号（从 1 开始），无需再 +1
                        "iteration": iteration,
                        "params": design_params,
                        "result": result,
                    }
                    
                    # 周期性清理 AEDT 进程
                    self._simulation_count += 1
                    self._check_and_cleanup_aedt()
                    
                    # ★磁饱和警告检测：分别检测上轭(ta)和中间块(tb)区域
                    is_saturated = result.get("is_saturated", False)
                    is_saturated_ta = result.get("is_saturated_ta", False)
                    is_saturated_tb = result.get("is_saturated_tb", False)
                    b_mean_ta = result.get("B_mean_ta", 0)
                    b_mean_tb = result.get("B_mean_tb", 0)
                    saturation_region = result.get("saturation_region", "")
                    
                    if is_saturated:
                        # 构建详细的饱和提示
                        parts = []
                        if is_saturated_ta:
                            parts.append(f"上轭(ta)区域={b_mean_ta:.3f}T")
                        if is_saturated_tb:
                            parts.append(f"中间块(tb)区域={b_mean_tb:.3f}T")
                        region_info = "、".join(parts) if parts else f"B={max(b_mean_ta, b_mean_tb):.3f}T"
                        
                        self._log(f"[Auto] 📌 磁饱和提示：{region_info} ≥ 2.0T（fitness 正常计算）")
                        # 添加警告消息到即将发送给 LLM 的上下文
                        self._pending_saturation_warning = (
                            f"📌 提示：本轮 {region_info}，可能出现局部磁饱和，推力增长可能不均匀，fitness 照常计算，可继续探索。"
                        )
                    else:
                        self._pending_saturation_warning = None
                    
                    # ★Actor-Critic 更新：记录实际结果，更新价值函数和评论家
                    actual_fitness = result.get("fitness")
                    actual_reward = 0.0
                    if actual_fitness is not None:
                        # 计算实际奖励：fitness 降低为正奖励
                        if hasattr(self, '_best_fitness_for_critic') and self._best_fitness_for_critic:
                            delta = self._best_fitness_for_critic - actual_fitness
                            actual_reward = 1.0 if delta > 0 else -0.5 if delta < 0 else 0.0
                    
                    # 使用 Actor-Critic 系统更新
                    if self.actor_critic_system:
                        try:
                            params_before = self._current_state or {}
                            update_result = self.actor_critic_system.update_after_result(
                                state=params_before,
                                next_state=design_params,
                                fitness=actual_fitness if actual_fitness else float('inf'),
                                reward=actual_reward
                            )
                            
                            # 显示 TD 误差
                            td_error = update_result.get("td_error", 0)
                            pred_acc = update_result.get("prediction_accuracy")
                            self._log(f"[AC] 📊 TD误差: {td_error:+.3f} | 预测准确性: {'✓' if pred_acc else '✗' if pred_acc is not None else 'N/A'}")
                            
                            # ★基于 TD 误差更新评论家（方案A + B）
                            if self.critic_ensemble and td_error != 0:
                                td_update_results = self.critic_ensemble.update_with_td_error(
                                    td_error=td_error,
                                    params_before=params_before,
                                    params_after=design_params
                                )
                                
                                # 输出评论家 TD 学习日志
                                new_rules_count = sum(len(r.get("new_rules", [])) for r in td_update_results.values())
                                if new_rules_count > 0:
                                    self._log(f"[Critic] 📚 TD 学习新增 {new_rules_count} 条规则")
                                
                                # 显示置信度变化
                                for name, result in td_update_results.items():
                                    old_conf = result.get("old_confidence", 0.5)
                                    new_conf = result.get("new_confidence", 0.5)
                                    correct = result.get("prediction_correct", False)
                                    
                                    if abs(new_conf - old_conf) > 0.01:
                                        change = "↑" if new_conf > old_conf else "↓"
                                        emoji = "✓" if correct else "✗"
                                        self._log(f"[Critic] {emoji} {name}: 置信度 {old_conf:.2f} {change} {new_conf:.2f}")
                            
                        except Exception as e:
                            self._log(f"[AC] ⚠️ Actor-Critic 更新异常: {e}", level="warning")
                    
                    # 兼容旧的评论家集群
                    elif self.critic_ensemble and self._last_critic_scores:
                        try:
                            params_before = self._current_state or {}
                            self.critic_ensemble.record_actual_result(
                                params_before=params_before,
                                params_after=design_params,
                                actual_result=result,
                                actual_reward=actual_reward
                            )
                        except Exception as e:
                            self._log(f"[Critic] ⚠️ 评论家校准异常: {e}", level="warning")
                    
                    # 更新用于评论家的最佳 fitness
                    if actual_fitness and (not hasattr(self, '_best_fitness_for_critic') or 
                                           self._best_fitness_for_critic is None or
                                           actual_fitness < self._best_fitness_for_critic):
                        self._best_fitness_for_critic = actual_fitness
                    
                except Exception:
                    # 解析失败则忽略，保持稳健
                    pass
            
            # ★新增：记录 validate 阶段的约束违规（用于失败模式学习 + 经验存储）
            elif tool_name == "validate_maxwell_design":
                try:
                    payload = json.loads(tool_content)
                    status = payload.get("status", "")
                    # 提取参数（包含 tb_ratio）
                    design_params = {}
                    design_var_names = {"lm", "tm", "ta", "dg", "hs", "wslot", "hslot", "s", "wa", "tb_ratio"}
                    for k, v in (parsed_arguments or {}).items():
                        if k in design_var_names and v is not None:
                            design_params[k] = v
                    errors = payload.get("errors", [])
                    
                    if status == "constraint_violation" and design_params:
                        # ★标记本轮发生了约束违规（用于重试机制）
                        self._constraint_violation_in_round = True
                        self._last_constraint_errors = errors
                        
                        # 记录到策略管理器
                        if self.strategy_manager and errors:
                            self.strategy_manager.record_constraint_violation(
                                params=design_params,
                                errors=errors
                            )
                            self._log(f"[RL] 📝 记录约束违规: {errors[:2]}...")
                        
                        # ★新增：也存储到经验缓冲区（失败经验同样重要）
                        if self.experience_buffer:
                            self.experience_buffer.store(
                                iteration=iteration,
                                state=design_params,
                                action={},  # validate 阶段没有 action
                                result={
                                    "status": "constraint_violation",
                                    "errors": errors,
                                    "source": "validate"
                                },
                                reward=-10.0,  # 约束违规给负奖励
                                llm_reasoning="validate阶段约束违规"
                            )
                            # ★立即强制保存，防止程序中断丢失数据
                            self.experience_buffer.save()
                            self._log(f"[RL] 📝 存储失败经验（validate阶段）并已保存")
                    
                    # ★Actor-Critic 预评估：当 validate 通过时，在 run_maxwell 之前进行评估
                    elif status == "ok" and design_params:
                        # 获取上一轮参数作为对比基准
                        params_before = self._current_state or {}
                        params_after = design_params
                        
                        # ★使用完整的 Actor-Critic 系统
                        if self.actor_critic_system and params_before and params_after:
                            try:
                                best_info = f"当前最佳 fitness: {getattr(self, '_best_fitness_for_critic', 'N/A')}"
                                dense_reward, evaluation, suggestions = await self.actor_critic_system.evaluate_before_action(
                                    current_state=params_before,
                                    proposed_state=params_after,
                                    additional_context=best_info
                                )
                                
                                # 缓存结果
                                self._last_dense_reward = dense_reward
                                self._last_critic_scores = evaluation.get("critic_scores", {})
                                
                                # 提取价值函数估计
                                v_current = evaluation.get("v_current", {})
                                v_proposed = evaluation.get("v_proposed", {})
                                advantage = evaluation.get("advantage", 0)
                                
                                # 日志输出完整的 Actor-Critic 评估结果
                                self._log("=" * 60)
                                self._log(f"[AC] 🎭 【Actor-Critic 预评估】")
                                self._log(f"[AC] 📊 状态价值 V(s): {v_current.get('value', 0):+.3f} → V(s'): {v_proposed.get('value', 0):+.3f}")
                                self._log(f"[AC] 📈 优势估计 A(s,a): {advantage:+.3f}")
                                self._log(f"[AC] 💰 稠密奖励: {dense_reward:+.3f}")
                                
                                # 显示各评论家评分
                                if self._last_critic_scores:
                                    self._log(f"[AC] --- 动作评论家评分 ---")
                                    for name, score_data in self._last_critic_scores.items():
                                        score_val = score_data.get("score", 0) if isinstance(score_data, dict) else getattr(score_data, "score", 0)
                                        emoji = "✓" if score_val > 0.1 else "✗" if score_val < -0.1 else "→"
                                        self._log(f"[AC]   {emoji} {name}: {score_val:+.2f}")
                                
                                if suggestions:
                                    self._log(f"[AC] 💡 建议: {'; '.join(suggestions[:2])}")
                                
                                # ★关键：将评论家反馈注入到 messages 中，让 LLM 在决定是否 run_maxwell 前看到
                                if abs(dense_reward) > 0.2 or suggestions:
                                    feedback_msg = self._build_critic_feedback_message(
                                        dense_reward, advantage, v_current, v_proposed, suggestions,
                                        params_before=params_before, params_after=params_after
                                    )
                                    # 这个反馈会被 LLM 在下一次调用时看到
                                    self._pending_critic_feedback = feedback_msg
                                    self._log(f"[AC] 📝 评论家反馈已缓存，将在后续注入")
                                    self._log(f"\n{'─'*50}\n{feedback_msg}\n{'─'*50}")
                                
                                self._log("=" * 60)
                                
                            except Exception as e:
                                self._log(f"[AC] ⚠️ Actor-Critic 评估异常: {e}", level="warning")
                                self._last_critic_scores = None
                                self._last_dense_reward = 0.0
                        
                        # 兼容旧的评论家集群（如果没有 actor_critic_system）
                        elif self.critic_ensemble and params_before and params_after:
                            try:
                                best_info = f"当前最佳 fitness: {getattr(self, '_best_fitness_for_critic', 'N/A')}"
                                dense_reward, critic_scores, suggestions = await self.critic_ensemble.evaluate(
                                    params_before=params_before,
                                    params_after=params_after,
                                    additional_context=best_info
                                )
                                
                                self._last_critic_scores = critic_scores
                                self._last_dense_reward = dense_reward
                                
                                self._log("-" * 50)
                                self._log(f"[Critic] 🎭 【评论家预评估】稠密奖励 = {dense_reward:+.3f}")
                                for name, score in critic_scores.items():
                                    emoji = "✓" if score.score > 0.1 else "✗" if score.score < -0.1 else "→"
                                    self._log(f"[Critic] {emoji} {name}: {score.score:+.2f} (置信度 {score.confidence:.0%})")
                                if suggestions:
                                    self._log(f"[Critic] 💡 建议: {'; '.join(suggestions[:2])}")
                                self._log("-" * 50)
                                
                            except Exception as e:
                                self._log(f"[Critic] ⚠️ 评论家评估异常: {e}", level="warning")
                                self._last_critic_scores = None
                                self._last_dense_reward = 0.0
                        
                except Exception:
                    pass

            # ★新方案B：精简工具返回结果，只保留关键字段（减少 token 消耗）
            tool_name = tool_call_entry["function"]["name"]
            slim_content = self._slim_tool_result(tool_name, tool_content)
            
            tool_message = {
                "role": "tool",
                "content": tool_content,
                "tool_call_id": tool_call_entry["id"],
            }
            messages.append(assistant_message)
            messages.append(tool_message)
            
            strict_tool_turn = self._requires_strict_tool_turn_protocol()

            # ★关键：注入评论家反馈到 LLM messages（在 validate 通过后）
            if hasattr(self, '_pending_critic_feedback') and self._pending_critic_feedback:
                if strict_tool_turn:
                    # Google 严格序：不能在 tool 响应后立刻插入 system 消息，延后到下一轮 user 提示
                    self._deferred_system_notes.append(self._pending_critic_feedback)
                else:
                    critic_feedback_msg = {
                        "role": "system",
                        "content": self._pending_critic_feedback
                    }
                    messages.append(critic_feedback_msg)
                # 评论家反馈已在 pre_evaluate_action 时输出，此处不再重复
                self._pending_critic_feedback = None  # 清空，避免重复注入
            
            # ★注入磁饱和警告到 LLM messages（仿真完成后）
            if hasattr(self, '_pending_saturation_warning') and self._pending_saturation_warning:
                if strict_tool_turn:
                    # Google 严格序：延后注入，避免破坏 tool call/response 邻接关系
                    self._deferred_system_notes.append(self._pending_saturation_warning)
                else:
                    saturation_msg = {
                        "role": "system",
                        "content": self._pending_saturation_warning
                    }
                    messages.append(saturation_msg)
                # 磁饱和警告已在仿真返回时输出，此处不再重复
                self._pending_saturation_warning = None  # 清空，避免重复注入

            # ★方案B：followup 调用也限制 max_tokens
            followup_kwargs = {
                "model": self.model,
                "messages": messages,
                "tools": available_tools,  # 继续提供工具列表，允许模型链式调用
                "tool_choice": tool_choice,
                "stream": use_stream,
                "max_tokens": 4096,  # ★合理限制：足够输出完整内容
            }
            if llm_timeout:
                followup_kwargs["timeout"] = llm_timeout
            
            # ★ 使用 asyncio.wait_for 强制超时
            effective_timeout = llm_timeout or LLM_TIMEOUT
            try:
                response = await asyncio.wait_for(
                    self.llm.chat.completions.create(**followup_kwargs),
                    timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"LLM API followup 调用超时（{effective_timeout}秒）")

            collected_content = []
            collected_tool_calls = []
            current_tool_call = None

            # 处理流式和非流式两种响应格式
            if use_stream:
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        collected_content.append(chunk_content)
                        if self._stream_to_console:
                            print(chunk_content, end="", flush=True)

                    if chunk.choices[0].delta.tool_calls:
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            if tool_call.index is not None:
                                if (current_tool_call is None
                                        or tool_call.index != current_tool_call[
                                            "index"]):
                                    if current_tool_call:
                                        collected_tool_calls.append(
                                            current_tool_call)
                                    current_tool_call = {
                                        "id": tool_call.id or "",
                                        "type": "function",
                                        "index": tool_call.index,
                                        "function": {
                                            "name": "",
                                            "arguments": ""
                                        },
                                    }

                            if tool_call.function and tool_call.function.name:
                                current_tool_call["function"]["name"
                                                              ] = tool_call.function.name
                            if tool_call.function and tool_call.function.arguments:
                                current_tool_call["function"][
                                    "arguments"] += tool_call.function.arguments
            else:
                # 非流式响应：直接从 response 提取
                if response.choices and response.choices[0].message:
                    msg = response.choices[0].message
                    if msg.content:
                        collected_content.append(msg.content)
                    
                    # 处理工具调用
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            collected_tool_calls.append({
                                "id": tc.id or "",
                                "type": "function",
                                "index": 0,
                                "function": {
                                    "name": tc.function.name if tc.function else "",
                                    "arguments": tc.function.arguments if tc.function else ""
                                },
                            })

            if current_tool_call:
                collected_tool_calls.append(current_tool_call)
            
            if collected_tool_calls:
                for tc in collected_tool_calls:
                    self._log(
                        f"[MCP Client] 收到工具调用: {tc['function']['name']} args={tc['function']['arguments']}"
                    )

            final_response_parts.append("".join(collected_content).strip())
            remaining_tool_calls.extend(collected_tool_calls)

        return "\n".join(final_response_parts).strip(), tool_executed

    async def process_query(self, query: str):
        """单轮交互：用户输入 -> 工具调用 -> 返回答案"""
        messages = []
        if self.rag_engine:
            rag_context = await self.rag_engine.build_context(query)
            if rag_context:
                messages.append({"role": "system", "content": rag_context})

        messages.append({"role": "user", "content": query})
        response_text, _ = await self._chat_with_tools(
            messages, llm_timeout=None)
        return response_text

    # 磁饱和惩罚阈值：fitness 超过此值认为是饱和惩罚，不计入有效收敛判断
    SATURATION_PENALTY_THRESHOLD = 1e5

    async def optimize(self,
                       system_prompt: str,
                       max_iterations: int = 50,
                       min_iterations: int = 10,
                       convergence_window: int = 20,
                       convergence_threshold: float = 0.01,
                       results_file: Optional[str] = None,
                       warm_start: Optional[dict] = None,
                       enable_rl: bool = True,
                       # ★TD(n) 批处理模式配置
                       enable_batch_mode: bool = False,
                       batch_interval: int = 3,
                       # ★滑窗奖励平滑配置
                       enable_reward_smoothing: bool = False,
                       reward_window_size: int = 3) -> None:
        """自动化多轮迭代，支持强化学习增强
        
        Args:
            system_prompt: 系统提示词
            max_iterations: 最大迭代次数
            min_iterations: 最小迭代次数（收敛判断前至少运行这么多轮）
            convergence_window: 收敛判断窗口大小（连续多少轮无改进视为收敛，默认 20）
            convergence_threshold: 收敛阈值（fitness 改进比例小于此值视为无改进）
            results_file: 保存每轮结果的 CSV 文件路径
            warm_start: 上一轮最优设计参数，指导模型从该点附近搜索
            enable_rl: 是否启用 RL 增强（经验回放、策略管理、奖励计算）
            enable_batch_mode: 是否启用 TD(n) 批处理模式（评论家每 n 轮生成一次规则）
            batch_interval: 批处理间隔（默认 3 轮）
            enable_reward_smoothing: 是否启用滑窗奖励平滑
            reward_window_size: 滑窗大小（默认 3 轮）
        """
        # ★设置滑窗平滑配置
        self._enable_reward_smoothing = enable_reward_smoothing
        self._reward_window_size = reward_window_size
        self._reward_history_window = []  # 重置滑窗历史
        
        if enable_reward_smoothing:
            self._log(f"[RL] 📊 滑窗奖励平滑已启用 | 窗口大小: {reward_window_size}")
        if enable_batch_mode:
            self._log(f"[RL] 📦 TD({batch_interval}) 批处理模式已启用 | 每 {batch_interval} 轮生成规则")
        
        # ========== RL 增强：初始化组件 ==========
        if enable_rl:
            if self.feedback_handler is None:
                self.feedback_handler = FeedbackHandler(storage_path="feedback_storage.json")
            if self.experience_buffer is None:
                self.experience_buffer = ExperienceBuffer(storage_path="experience_buffer.json")
            if self.strategy_manager is None:
                self.strategy_manager = StrategyManager(
                    enable_meta_learning=False,  # ★ 禁用旧版元学习，改用 ExpeL 对比批评
                    meta_knowledge_path="meta_knowledge.json",
                    domain="electromagnetic_actuator"
                )
            # ★ 注入蒸馏原则（跨任务迁移）
            if getattr(self, '_distilled_principles', None) is not None:
                import time as _time
                self.strategy_manager.distilled_principles = self._distilled_principles
                self.strategy_manager._run_start_time = _time.time()
                self._log("[Transfer] 📚 蒸馏原则已注入策略管理器")
            if self.reward_calculator is None:
                self.reward_calculator = RewardCalculator()
            
            # 初始化反思管理器（Reflexion）
            # ★ 可通过环境变量 REFLECTION_USE_LLM=0 禁用 LLM 反思（使用模板方法，更快更稳定）
            use_llm_reflection = os.environ.get("REFLECTION_USE_LLM", "1") == "1"
            if self.reflection_manager is None:
                self.reflection_manager = ReflectionManager(
                    storage_path="memory_stream.jsonl",
                    stagnation_window=5,
                    degradation_threshold=0.15,
                    periodic_interval=10,
                    enable_periodic=True,
                    llm_client=self.llm if use_llm_reflection else None,
                    llm_model=self.model if use_llm_reflection else None
                )
                mode = f"使用 {self.model} 进行智能反思" if use_llm_reflection else "使用模板方法（更快）"
                self._log(f"[Reflection] 🪞 反思管理器已初始化（{mode}）")
            
            # 初始化统一记忆管理器（整合短期/长期记忆）
            if self.memory_manager is None:
                self.memory_manager = MemoryManager(
                    storage_dir=".",
                    max_trajectory_length=100
                )
                # 复用已初始化的组件
                self.memory_manager.long_term.strategy_manager = self.strategy_manager  # 兼容
                self.memory_manager.long_term.reflection_manager = self.reflection_manager
                self.memory_manager.long_term.feedback_handler = self.feedback_handler
                self._log("=" * 60)
                self._log("[Memory] 🧠 统一记忆架构已初始化（Reflexion 架构）")
                self._log("[Memory]   - 短期记忆 (Trajectory): 轨迹 + 探索策略 + 即时反馈")
                self._log("[Memory]   - 长期记忆 (Experience): 反思 + 规则 + 模式（触发时思考）")
                self._log("=" * 60)
            
            # 显示 RL 状态（旧版元学习已禁用，改用 ExpeL）
            self._log(f"[RL] 强化学习增强已启用 | epsilon={self.strategy_manager.epsilon:.2f}")
            
            # ★ ExpeL 对比批评初始化
            if EXPEL_AVAILABLE:
                try:
                    rule_manager = self.strategy_manager.rule_manager if self.strategy_manager.expel_enabled else None
                    if rule_manager is None:
                        rule_manager, _ = create_expel_system(
                            llm_client=self.llm,
                            llm_model=self.model,
                            max_rules=20,
                            storage_path="expel_rules.json"
                        )
                    self.contrast_critique = ContrastCritique(
                        llm_client=self.llm,
                        llm_model=self.model,
                        rule_manager=rule_manager
                    )
                    self._expel_enabled = True
                    self._log(f"[ExpeL] ✅ 对比批评已启用 | 当前规则数: {len(rule_manager.rules)}")
                except Exception as e:
                    self._log(f"[ExpeL] ⚠️ 初始化失败: {e}", level="warning")
                    self._expel_enabled = False
        
        # ========== 方案 B：使用 Block 系统构建初始消息 ==========
        # 输出规范（作为 persona 的一部分）
        if self._verbose_llm_output:
            output_constraint = """【输出规范 - 详细可解释性模式】
每轮输出请完整包含以下结构，便于追溯思考过程与论文引用：
1) 目标与可行域复述
2) 候选参数（说明选取理由）及 validate 结果（含约束违规时的修正过程）
3) 仿真结果（fitness、kb、pb、avg_B、体积、质量、饱和状态等）
4) 物理机理/原因：为何这样调整参数、约束边界条件、对 fitness 各分量的预期影响
5) 与历史最佳的对比 & 是否改进
6) 下一轮调整思路（含具体可操作方向）
7) 本轮摘要（参数→validate→仿真→下一步原因）
最后用【批评】【总结】【方案】三行收尾。禁止空洞词汇，请给出具体数值与物理解释。

📌 **可溯源引用要求**（便于论文论证可解释性/可迁移性/可控性）：
   当决策参考以下任一来源时，必须显式写出对应格式：
   • **经验/策略**：「根据经验库/策略库中的 [具体知识]，我们做出了 [具体改变]」
     例：根据策略库中 hslot 与 tb 差值需足够大的历史失败警告，我们保持 hslot-tb≥0.2 余量
   • **文献 RAG**：「根据文献 [具体内容/思路]，我们做出了 [具体改变]」
     例：根据文献中气隙磁阻与磁通密度的关系，我们减小 dg 以提升气隙磁密
   • **人工反馈**：「根据人工反馈 [具体内容]，我们做出了 [具体改变]」
     例：根据人工反馈“加大 n2”的要求，我们通过调整 hs/hslot 以增大可布线空间
"""
        else:
            output_constraint = """【输出规范 - 批评性工程思维】
每轮输出必须包含以下三部分（总共 100 字以内）：
1️⃣ 【批评】当前方案有什么问题？为什么没有更大改进？
2️⃣ 【总结】关键数值变化（具体数字）
3️⃣ 【方案】下一步具体怎么做（明确数值）
⛔ 禁止空洞词汇和庆祝性文字
"""
        
        # 初始化 Block 系统
        if self.memory_manager:
            # 1. Persona Block：任务描述 + 输出规范
            persona_content = f"{system_prompt}\n\n{output_constraint}"
            self.memory_manager.update_persona(persona_content)
            
            # 2. Experience Block：RAG + 历史经验
            experience_parts = []
            if self.rag_engine:
                rag_context = await self.rag_engine.build_context(system_prompt)
                if rag_context:
                    # RAG 内容截取关键部分
                    experience_parts.append(rag_context[:800] if len(rag_context) > 800 else rag_context)
            
            if enable_rl and self.experience_buffer:
                strategy_only = getattr(self, '_experience_transferred', False)
                transfer_mode = getattr(self, '_transfer_mode', None)
                # hybrid 模式：暴露完整参数，让 LLM 有数值锚点（L2/L3 引导由 strategy prompt 负责）
                if transfer_mode == "hybrid":
                    strategy_only = False
                exp_context = self.experience_buffer.build_experience_context(
                    current_state=warm_start or {},
                    include_similar=2,
                    include_best=2,
                    strategy_only=strategy_only,
                    transfer_mode=transfer_mode
                )
                if exp_context:
                    experience_parts.append(exp_context)
            
            if experience_parts:
                self.memory_manager.update_experience("\n\n".join(experience_parts))
            
            # 3. Feedback Block：人类反馈
            if enable_rl and self.feedback_handler:
                if self.feedback_handler.feedbacks:
                    feedback_texts = [f.text for f in self.feedback_handler.feedbacks[:3]]
                    self.memory_manager.update_feedback(feedback_texts)
                    self._log("[RL] 🧑‍🏫 人类反馈已更新到 Block")
            
            # 4. Strategy Block：策略提示
            if enable_rl and self.strategy_manager:
                strategy_only = getattr(self, '_experience_transferred', False)
                transfer_mode = getattr(self, '_transfer_mode', None)
                if transfer_mode == "hybrid":
                    strategy_only = False
                strategy_prompt = self.strategy_manager.build_strategy_prompt(
                    strategy_only=strategy_only,
                    transfer_mode=transfer_mode
                )
                if strategy_prompt:
                    # 蒸馏/混合模式需要更多空间容纳 L2/L3 原则和引用指令
                    max_len = 3000 if transfer_mode in ("distilled", "hybrid") else 1000
                    self.memory_manager.update_strategy(strategy_prompt[:max_len])
                    self._log("[RL] 📚 策略知识已更新到 Block")
            
            # 5. Active Rules Block：反思规则
            if enable_rl and self.reflection_manager:
                rules = self.reflection_manager.get_recent_rules(max_rules=5)
                if rules:
                    self.memory_manager.update_active_rules(rules)
                    self._log(f"[Reflection] 🪞 {len(rules)} 条规则已更新到 Block")
            
            # 6. Trajectory Block：短期轨迹
            if enable_rl:
                self.memory_manager.update_trajectory_block()
                summary = self.memory_manager.short_term.get_trajectory_summary()
                if summary.get("length", 0) > 0:
                    self._log(f"[Memory] 📍 轨迹({summary['length']}步)已更新到 Block")
            
            # ★ 使用统一的 compose_context 生成单一 system message
            unified_context = self.memory_manager.compose_context()
            messages = [{"role": "system", "content": unified_context}]
            
            # 输出 Block 统计信息
            if self._verbose_llm_output:
                self._log("[LLM] 📝 详细输出模式已启用（完整思考过程将写入 log）")
            self._log("=" * 60)
            self._log("[方案B] 📦 Memory Block 统计:")
            self.memory_manager.log_block_stats(self._log)
            self._log("=" * 60)
        else:
            # 兼容模式：没有 memory_manager 时使用原来的方式
            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "system", "content": output_constraint})

        # warm-start
        if warm_start:
            warm_text = ", ".join(f"{k}={v:g}" for k, v in warm_start.items())
            messages.append({
                "role": "system",
                "content": f"上一轮最优设计（请优先在此附近微调）：{warm_text}"
            })
            self._current_state = warm_start.copy()
        
        iteration = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        # 收敛判断相关
        best_fitness = float('inf')
        best_params = None             # 最佳参数（用于上下文压缩摘要）
        no_improvement_count = 0
        fitness_history = []           # 全量 fitness 历史（含饱和惩罚）
        valid_fitness_history = []     # 有效 fitness 历史（排除饱和惩罚值）
        successful_sim_count = 0       # ★成功仿真次数（排除 skipped/llm_failed/约束违规）
        saturation_count = 0           # 连续饱和次数
        total_saturation_count = 0     # 总饱和次数
        skipped_count = 0              # ★跳过/失败轮次计数
        reward_history = []            # RL 奖励历史
        
        # 初始化 CSV 文件
        import csv
        if not results_file:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # 构建标签列表
            tags = []
            # 0. 模型标识：加入命名，便于区分不同模型实验
            model_name = str(getattr(self, "model", "unknown") or "unknown").strip()
            safe_model_name = "".join(
                ch if (ch.isalnum() or ch in ("-", "_", ".")) else "-"
                for ch in model_name
            ).strip("-_.")
            if not safe_model_name:
                safe_model_name = "unknown"
            # 控制长度，避免文件名过长
            safe_model_name = safe_model_name[:40]
            tags.append(f"m-{safe_model_name}")
            # 1. 结构类型：E / EI
            tags.append("E" if os.getenv("USE_E_ONLY_SIMULATION", "0") == "1" else "EI")
            # 1.5 线径：非默认值时标注（如 w0.06 表示 0.06mm）
            wire_diameter = os.getenv("WIRE_DIAMETER_MM", "0.05")
            if wire_diameter != "0.05":
                tags.append(f"w{wire_diameter}")
            # 2. RAG 状态：RAG / noRAG
            tags.append("RAG" if self.rag_engine else "noRAG")
            # 3. RL 状态：RL / noRL
            tags.append("RL" if enable_rl else "noRL")
            # 4. 离散引导开关：默认开启不标注，关闭时显式标注
            if not self._enable_discrete_guidance:
                tags.append("noDiscreteGuidance")
            # 5. 经验迁移：有则加 Transfer，无则不加
            if getattr(self, '_experience_transferred', False):
                tags.append("Transfer")
            # 6. 人工反馈：有则加 FB，无则不加
            has_feedback = (self.feedback_handler and 
                           hasattr(self.feedback_handler, 'feedbacks') and 
                           len(self.feedback_handler.feedbacks) > 0)
            if has_feedback:
                tags.append("FB")
            results_file = f"AgenticOPT_{timestamp}_{'_'.join(tags)}.csv"
        elif os.path.exists(results_file):
            base, ext = os.path.splitext(results_file)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            results_file = f"{base}_{timestamp}{ext or '.csv'}"

        results_dir = os.path.dirname(results_file)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        csv_file = open(results_file, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'iteration', 'status', 'fitness', 'avg_B', 'B_sat', 'kb', 'pb',
            'volume_r', 'mass_r', 'kb_r', 'pb_r',
            'volume', 'mass_total', 'mass_mover', 'mass_stator',
            'la', 'ha', 'ws', 'ls', 'tb', 'twall',
            'lm', 'tm', 'ta', 'dg', 'hs', 'wslot', 'hslot', 's', 'wa', 'tb_ratio',
            'n1', 'n2', 'total_turns',
            'result_source', 'result_description', 'fld_file', 'fld_bsat_file', 'errors',
            'B_mean_ta', 'B_mean_tb', 'is_saturated_ta', 'is_saturated_tb',  # 新增：两面平均|B|和饱和标志
            'saturation_region', 'saturation_suggestion',  # 饱和分析字段
            'reward', 'is_exploration',  # RL 增强字段
            'dense_reward', 'critic_magnetic', 'critic_performance', 'critic_constraint', 'critic_magnitude'  # 评论家字段
        ])
        self._log(f"[Auto] 结果将保存到：{results_file}")
        self._last_sim_result = None
        # 用于避免重复写入同一 iteration（也用于写占位行保持连续）
        self._written_csv_iterations: set[int] = set()
        
        # 反馈输入文件路径
        feedback_input_file = "feedback_input.txt"
        
        # ★约束违规重试机制：最大重试次数
        MAX_CONSTRAINT_RETRIES = 3
        constraint_retry_count = 0  # 当前轮次的重试计数
        
        while iteration < max_iterations:
            is_exploration = False
            exploration_suggestion = ""
            
            # ★重置本轮约束违规标志（但不重置 retry_count，它在成功仿真后才重置）
            self._constraint_violation_in_round = False
            self._last_constraint_errors = []
            
            # ★重置本轮仿真执行标志：确保每轮只执行一次 run_maxwell_simulation
            self._sim_executed_this_round = False
            
            # ========== 上下文长度检查与压缩 ==========
            # ★新方案A+C：传入当前轮次，支持定期压缩
            messages = self._check_and_compress_context(messages, best_fitness, best_params, current_iteration=iteration)
            
            # ========== RL 增强：检查反馈输入文件 ==========
            if enable_rl and self.feedback_handler:
                new_feedbacks = self._check_feedback_file(feedback_input_file)
                if new_feedbacks:
                    self._log("=" * 60)
                    self._log(f"[RL] 🧑‍🏫 【检测到 {len(new_feedbacks)} 条新人类反馈】")
                    for fb_text, fb_type, fb_priority in new_feedbacks:
                        self.feedback_handler.add_feedback(
                            text=fb_text,
                            feedback_type=fb_type,
                            priority=fb_priority
                        )
                        priority_emoji = "🚨" if fb_priority >= 4 else "⚠️" if fb_priority >= 3 else "💡"
                        self._log(f"[RL] {priority_emoji} [{fb_type}] {fb_text}")
                    self._log("=" * 60)
                
                # ★每轮都重新注入人工反馈（防止被上下文压缩挤掉）
                # 无论是否有新反馈，只要 feedback_handler 里有内容就注入
                feedback_context = self.feedback_handler.build_feedback_context(limit=5)
                if feedback_context:
                    messages.append({"role": "system", "content": f"[人工反馈约束]\n{feedback_context}"})
            
            # ========== RL 增强：探索-利用决策 ==========
            if enable_rl and self.strategy_manager:
                self._log("-" * 50)
                self._log(f"[RL] 🎲 【探索-利用决策】")
                self._log(f"[RL] 当前探索率 ε = {self.strategy_manager.epsilon:.4f}")
                
                # 检查 n2 是否长期不变，并在需要时提高边界邻域探索概率
                n2_force_explore = False
                n2_high_plateau = False
                if self.strategy_manager.n2_history:
                    n2_unchanged = self.strategy_manager.get_rounds_since_change(self.strategy_manager.n2_history)
                    n2_stagnation = self.strategy_manager.get_stagnation_info(self.strategy_manager.n2_history, window=15)
                    threshold = getattr(self.strategy_manager.config, 'discrete_jump_threshold', 8)
                    current_n2 = self.strategy_manager.n2_history[-1]
                    recent_max_n2 = max(self.strategy_manager.n2_history)
                    high_n2_threshold = getattr(self.strategy_manager.config, "high_n2_threshold", 9)
                    # 多轮无显著改进时，转向连续参数精调
                    stall_rounds_threshold = 5
                    fitness_stalled = no_improvement_count >= stall_rounds_threshold
                    is_stagnant = n2_unchanged >= threshold or (
                        n2_stagnation["is_stagnant"] and n2_stagnation["rounds_in_window"] >= threshold
                    )
                    # 高位平台期（如 n2 在 [8,9] 长时间波动）改为连续参数精调
                    n2_high_plateau = (
                        is_stagnant
                        and current_n2 >= high_n2_threshold
                        and current_n2 >= recent_max_n2 - 1
                    )
                    # 若多轮无改进且 n2 未出现新变化，也转向连续参数精调
                    if fitness_stalled and current_n2 >= recent_max_n2 - 1:
                        n2_high_plateau = True
                    if is_stagnant:
                        if n2_high_plateau:
                            if fitness_stalled:
                                self._log(
                                    f"[RL] ℹ️ 多轮改不动检测: 连续{no_improvement_count}轮无显著改进，"
                                    "转入利用/精调"
                                )
                            else:
                                self._log(
                                    f"[RL] ℹ️ n2 高位平台检测: n2={current_n2}, 连续{n2_unchanged}轮不变，"
                                    "转入利用/精调"
                                )
                        else:
                            n2_force_explore = True
                            self._log(f"[RL] ⚠️ n2 停滞检测: 连续{n2_unchanged}轮不变 或 在[{n2_stagnation.get('min_val')},{n2_stagnation.get('max_val')}]波动")
                
                if n2_force_explore or self.strategy_manager.should_explore():
                    is_exploration = True
                    strategy = self.strategy_manager.select_exploration_strategy()
                    if n2_high_plateau and strategy == ExplorationStrategy.DISCRETE_JUMP:
                        strategy = ExplorationStrategy.PERTURBATION
                        self._log("[RL] 高位平台策略切换: discrete_jump -> perturbation")
                    strategy_desc = {
                        "random": "🎲 完全随机探索 - 在可行域内随机采样",
                        "directed": "🧭 定向探索 - 向成功区域靠拢",
                        "perturbation": "🔧 小扰动探索 - 在当前状态附近微调",
                        "counterfactual": "🔄 反事实探索 - 尝试相反方向",
                        "discrete_jump": "离散派生量边界探索"
                    }.get(strategy.value, strategy.value)
                    self._log(f"[RL] 决策: 🔍 【探索】 {strategy_desc}")
                    
                    # 生成探索建议
                    if self._current_state:
                        explore_state = self.strategy_manager.generate_exploration_action(
                            self._current_state, strategy
                        )
                        # 过滤 None 值并格式化
                        explore_items = [(k, v) for k, v in explore_state.items() if v is not None]
                        if explore_items:
                            explore_text = ", ".join(f"{k}={v:.2f}" for k, v in explore_items)
                            exploration_suggestion = f"\n\n💡 **探索建议**：本轮尝试以下参数附近搜索：{explore_text}"
                            self._log(f"[RL] 探索目标: {explore_text}")
                else:
                    self._log(f"[RL] 决策: 🎯 【利用】 基于历史最佳进行优化")
                    if self.strategy_manager.success_regions:
                        self._log(f"[RL] 可用成功区域: {len(self.strategy_manager.success_regions)} 个")
                self._log("-" * 50)
            
            # ★注入 Actor-Critic 上下文（如果有）
            ac_context = ""
            if self.actor_critic_system and self._current_state:
                try:
                    ac_context = self.actor_critic_system.build_context_for_llm(self._current_state)
                    if ac_context:
                        ac_context = f"\n\n{ac_context}\n"
                except Exception as e:
                    self._log(f"[AC] 构建上下文失败: {e}", level="warning")
            
            # 构建迭代提示
            iteration_prompt_content = (
                f"请执行第{iteration + 1}轮迭代。"
                f"{ac_context}"
                "先复述目标与可行域，提出候选参数，然后按顺序调用：\n"
                "1. validate_maxwell_design 检查约束\n"
                "2. 若 validate 返回 status=ok，立即调用 run_maxwell_simulation 执行仿真\n"
                "3. 若 validate 返回 constraint_violation，修正参数后重新 validate\n"
                "禁止编造结果，必须调用工具获取真实数据。\n\n"
                "**重要：在分析结果和给出下一轮调整建议时，请基于你对电磁学的理解，说明：**\n"
                "1. 为什么这样调整参数（物理机理/原因）\n"
                "2. 相关的约束边界条件\n"
                "3. 对 fitness 各分量（kb/pb/体积/质量）的预期影响\n"
                "（可参考文献知识，但请用自己的理解总结）\n\n"
                + (
                    "📝 **输出格式**：按 persona 中的详细结构完整输出（目标复述、候选参数、validate/仿真过程、物理机理、历史对比、下一轮思路、摘要），最后用【批评】【总结】【方案】收尾。"
                    + (" 引用经验/策略时写「根据经验库/策略库中的 XX，我们做出了 XX 改变」；引用文献时写「根据文献 XX，我们做出了 XX 改变」；引用人工反馈时写「根据人工反馈 XX，我们做出了 XX 改变」。" if self._verbose_llm_output else "")
                    if self._verbose_llm_output
                    else "📝 **输出格式**：【批评】当前问题 →【总结】数值变化 →【方案】具体下一步（共100字内）。禁止空洞词汇。"
                )
                + f"{exploration_suggestion}"
            )

            # Google 严格工具序兼容：把上一轮延后的 system 信息并入本轮 user 提示
            if self._deferred_system_notes:
                deferred_text = "\n\n".join(self._deferred_system_notes[-4:])
                iteration_prompt_content = (
                    "[来自上一轮的系统反馈，请在本轮决策中吸收]\n"
                    f"{deferred_text}\n\n"
                    f"{iteration_prompt_content}"
                )
                self._deferred_system_notes.clear()
            
            # RL 增强：检查紧急反馈
            if enable_rl and self.feedback_handler:
                urgent = self.feedback_handler.get_urgent_feedbacks()
                if urgent:
                    urgent_text = "\n".join(f"⚠️ {fb.text}" for fb in urgent)
                    iteration_prompt_content += f"\n\n**紧急提醒**：\n{urgent_text}"
            
            # ★ 离散变量探索追踪：当 n1/n2 长期不变时，主动建议探索
            # 优先使用 best_params，如果没有则用 _current_state
            params_for_check = best_params or self._current_state
            
            # ★统一开关：控制离散探索提示与调试输出
            if self._enable_discrete_guidance:
                # ★ 调试：始终输出当前状态，方便追踪
                n2_hist_len = len(self.strategy_manager.n2_history) if self.strategy_manager else 0
                self._console(f"[DEBUG] params_for_check={params_for_check is not None}, n2_history长度={n2_hist_len}")
            
            if self._enable_discrete_guidance and enable_rl and self.strategy_manager and params_for_check:
                derived_info = self._compute_derived_variable_info(params_for_check)
                if derived_info:
                    n1 = derived_info.get("n1", 0)
                    n2 = derived_info.get("n2", 0)
                    delta_hs = derived_info.get("delta_hs_to_next_n2", 0)
                    delta_hslot = derived_info.get("delta_hslot_to_next_n2", 0)
                    delta_lm = derived_info.get("delta_lm_to_next_n1", 0)
                    
                    n1_unchanged = self.strategy_manager.get_rounds_since_change(self.strategy_manager.n1_history)
                    n2_unchanged = self.strategy_manager.get_rounds_since_change(self.strategy_manager.n2_history)
                    n2_stagnation = self.strategy_manager.get_stagnation_info(self.strategy_manager.n2_history, window=15)
                    
                    # ★ 调试：显示停滞检测结果
                    self._console(f"[DEBUG] n2={n2}, 连续不变={n2_unchanged}轮, 波动停滞={n2_stagnation['is_stagnant']}, 窗口={n2_stagnation['rounds_in_window']}轮")
                    
                    discrete_prompt = self.strategy_manager.build_discrete_variable_exploration_prompt(
                        current_params=params_for_check,
                        n1=n1,
                        n2=n2,
                        delta_hs_to_next_n2=delta_hs,
                        delta_lm_to_next_n1=delta_lm,
                        threshold=5  # 降低阈值，更早触发
                    )
                    if discrete_prompt:
                        # ★ 方案 B 改进：把离散探索建议合并到 user message 而非追加 system message
                        iteration_prompt_content += f"\n\n{discrete_prompt}"
                        # ★ 重要信息：输出到终端
                        self._console("=" * 60)
                        self._console("[RL] 🔔 【离散变量探索建议已注入】", important=True)
                        self._console(f"[RL] n1={n1} 已保持 {n1_unchanged} 轮，n2={n2} 已保持 {n2_unchanged} 轮")
                        self._console(f"[RL] n2 相邻边界: hs +{delta_hs:.3f}mm 或 hslot -{delta_hslot:.3f}mm，n2 接近 {n2+1}")
                        self._console("=" * 60)
                    else:
                        # ★ 调试：为什么没有生成提示词
                        self._console(f"[DEBUG] 提示词未生成: 阈值=5, n2_unchanged={n2_unchanged}, stagnant={n2_stagnation['is_stagnant']}, window={n2_stagnation['rounds_in_window']}")
            
            iteration_prompt = {"role": "user", "content": iteration_prompt_content}
            messages.append(iteration_prompt)
            # * 清晰的轮次分隔符
            self._log_round_header(iteration + 1, max_iterations)
            self._log(f"[Auto] 开始第{iteration + 1}轮迭代...")
            
            # ========== LLM 做决策 ==========
            attempt = 0
            tool_used = False
            response_text = ""
            last_llm_error_msg = ""
            
            while not tool_used and attempt < 3:
                attempt += 1
                try:
                    # 某些兼容端点在「流式 + tool_calls」场景会偶发返回空流，
                    # 这里做分级兜底：第2次关闭流式，第3次强制要求调用 validate_maxwell_design。
                    stream_override = None
                    tool_choice_override = None
                    if attempt == 2:
                        stream_override = False
                    elif attempt == 3:
                        stream_override = False
                        tool_choice_override = {
                            "type": "function",
                            "function": {"name": "validate_maxwell_design"}
                        }

                    response_text, tool_used = await self._chat_with_tools(
                        messages,
                        llm_timeout=LLM_TIMEOUT,
                        iteration=iteration + 1,
                        stream_override=stream_override,
                        tool_choice_override=tool_choice_override,
                    )
                    # 终端/日志输出策略：
                    # - 日志始终记录完整回复（便于回溯）
                    # - 终端：streaming 开启时已有实时输出，不再重复 print
                    self._log(
                        f"[Auto] 第{iteration + 1}轮完整回复：\n{response_text}\n",
                        console=(not self._stream_to_console),
                    )
                    
                    # ★ 检测违规词汇
                    banned_words = ["史诗", "传奇", "钻石", "永恒", "丰碑", "巅峰", "奇迹", 
                                   "革命", "突破", "最高纪录", "人类历史", "不朽", "荣耀",
                                   "璀璨", "辉煌", "伟大", "震撼", "完美", "极致", "终极"]
                    found_banned = [w for w in banned_words if w in response_text]
                    if found_banned:
                        self._log(f"⚠️ [输出违规] 检测到禁止词汇: {found_banned}。LLM 未遵守输出规范！", level="warning")
                    
                    consecutive_failures = 0
                    break
                except Exception as exc:
                    error_msg = str(exc)
                    last_llm_error_msg = error_msg
                    
                    # ★ 检测 API 超时，立即重试
                    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "超时" in error_msg or "timeout" in error_msg.lower():
                        self._log(
                            f"[Auto] ⏱️ 第{iteration + 1}轮第 {attempt} 次 API 超时，立即重试...",
                            level="warning")
                        # ★ 连续超时 2 次时发送邮件提醒
                        if attempt >= 2 and self._notify_email:
                            if not hasattr(self, '_last_api_timeout_alert_round') or self._last_api_timeout_alert_round != iteration:
                                self._last_api_timeout_alert_round = iteration
                                self._send_notification_email(
                                    "MCP ALERT - API Timeout",
                                    f"API timeout detected.\n\nRound: {iteration + 1}\nAttempt: {attempt}\nError: {error_msg[:200]}"
                                )
                        await asyncio.sleep(2)  # 短暂等待后重试
                        continue
                    
                    self._log(
                        f"[Auto] 第{iteration + 1}轮第 {attempt} 次失败：{exc}，5s 后重试…",
                        level="warning")
                    
                    if "Port may be in use" in error_msg or "GRPC" in error_msg:
                        self._log(f"[Auto] 检测到严重错误（端口占用），终止迭代！",
                                  level="error")
                        # ★ 发送异常邮件通知
                        if self._notify_email:
                            self._send_notification_email(
                                "MCP ERROR - Maxwell Port/GRPC Error",
                                f"Optimization stopped due to critical error.\n\nError: {error_msg}\nRound: {iteration + 1}"
                            )
                        raise RuntimeError(f"Maxwell 仿真严重错误: {error_msg}")
                    
                    # ★检测上下文超限错误，立即触发强制压缩
                    em = error_msg.lower()
                    if ("maximum context length" in em) or ("context length" in em) or ("maximum context" in em):
                        self._log(f"[Context] ⚠️ 检测到上下文超限，立即触发强制压缩...", level="warning")
                        messages = self._force_compress_context(messages, best_fitness, best_params)
                        self._log(f"[Context] ✅ 强制压缩完成，消息数: {len(messages)}")
                        # 压缩后立即重试，不等待
                        continue
                    
                await asyncio.sleep(5)
            else:
                consecutive_failures += 1
                self._log(
                    f"[Auto] 第{iteration + 1}轮多次失败（连续失败 {consecutive_failures} 次）",
                    level="warning")
                
                if consecutive_failures >= max_consecutive_failures:
                    # ★ 发送异常邮件通知
                    if self._notify_email:
                        self._send_notification_email(
                            "MCP ERROR - Consecutive Failures",
                            f"Optimization stopped due to {max_consecutive_failures} consecutive failures.\n\nRound: {iteration + 1}\nLast error: {last_llm_error_msg or 'unknown'}"
                        )
                    raise RuntimeError(
                        f"连续 {max_consecutive_failures} 轮迭代失败，终止优化！请检查网络或模型配置。")
                
                # 写入占位行，保证 iteration 连续（即使本轮 LLM 没有成功触发工具调用）
                try:
                    self._write_placeholder_csv_row(
                        csv_writer=csv_writer,
                        iteration=iteration + 1,
                        status="llm_failed",
                        errors=[
                            f"LLM 未能在 3 次尝试内成功触发工具调用。last_error={last_llm_error_msg or 'unknown'}"
                        ],
                        params=self._current_state or {},
                        result_source="Internal",
                        result_description="本轮 LLM 调用/工具触发失败，写入占位行以保持 iteration 连续。"
                    )
                    csv_file.flush()
                except Exception as _e:
                    self._log(f"[DEBUG-CSV] ⚠️ 写入占位行失败: {_e}", level="warning")

                iteration += 1
                continue

            if not tool_used:
                self._log(
                    "[Auto] 本轮未触发工具调用，强制要求模型按 validate → run 顺序执行。",
                    level="warning")
                messages.append({
                    "role": "user",
                    "content": (
                        "你必须调用工具！请立即给出参数并按顺序调用：\n"
                        "1. 先调用 validate_maxwell_design\n"
                        "2. 若通过则调用 run_maxwell_simulation\n"
                        "不要只输出文字，必须实际调用工具！"
                    )
                })
                await asyncio.sleep(2)
                continue

            # ========== RL 增强：处理结果、计算奖励、存储经验 ==========
            current_reward = 0.0
            result_for_rl = {}
            new_state = {}
            
            # 调试日志：检查 _last_sim_result 状态
            _stored_iter = self._last_sim_result.get("iteration") if self._last_sim_result else None
            _expected_iter = iteration + 1
            if self._last_sim_result:
                if _stored_iter != _expected_iter:
                    self._log(f"[DEBUG] ⚠️ iteration 不匹配：stored={_stored_iter}, expected={_expected_iter}，仍尝试处理")
            else:
                self._log(f"[DEBUG] ⚠️ _last_sim_result 为空！iteration={_expected_iter}")
            
            # 优先使用工具返回的结构化结果
            # ★修复：放宽条件 - 只要 _last_sim_result 有有效结果且未被处理过就处理
            _already_processed = self._last_sim_result.get("_processed", False) if self._last_sim_result else True
            _has_result = bool(self._last_sim_result.get("result")) if self._last_sim_result else False
            self._log(f"[DEBUG-CSV] iteration={_expected_iter} | has_sim_result={bool(self._last_sim_result)} | has_result={_has_result} | already_processed={_already_processed}")
            if self._last_sim_result and self._last_sim_result.get("result") and not _already_processed:
                # 标记已处理，防止重复写入
                self._last_sim_result["_processed"] = True
                result_for_rl = self._last_sim_result.get("result", {})
                new_state = self._last_sim_result.get("params", {})
                
                # ★提前提取 current_fitness，供后续使用（无论 enable_rl 是否为 True）
                current_fitness = result_for_rl.get("fitness")
                
                # ★成功仿真后重置约束违规重试计数
                constraint_retry_count = 0
                
                # RL 增强：计算奖励
                # 使用 _current_state（上一轮状态）作为比较基准
                if enable_rl and self.reward_calculator:
                    current_reward = self.reward_calculator.calculate(
                        result=result_for_rl,
                        old_state=self._current_state,  # 修复：使用上一轮状态
                        new_state=new_state,
                        is_exploration=is_exploration
                    )
                    
                    # ★整合评论家稠密奖励（Actor-Critic融合）
                    # 使用 tanh 归一化，确保两个奖励量级统一
                    if (self.critic_ensemble or self.actor_critic_system) and self._last_dense_reward != 0:
                        # 归一化 r_real 到 [-1, 1] 范围（使用 tanh 软压缩）
                        reward_scale = 5.0  # 控制压缩程度：r_real=±5 时约为 ±0.76
                        r_real_norm = math.tanh(current_reward / reward_scale)
                        
                        # r_dense 已经是 [-1, 1]，不需要归一化
                        r_dense = self._last_dense_reward
                        
                        # ★滑窗平滑模式：对 r_real_norm 做滑窗平均
                        if self._enable_reward_smoothing:
                            self._reward_history_window.append(r_real_norm)
                            # 保持窗口大小
                            if len(self._reward_history_window) > self._reward_window_size:
                                self._reward_history_window.pop(0)
                            # 计算滑窗平均
                            r_real_smoothed = sum(self._reward_history_window) / len(self._reward_history_window)
                            self._log(f"[RL] 滑窗平滑({len(self._reward_history_window)}/{self._reward_window_size}): r_real_norm={r_real_norm:.3f} → r_smoothed={r_real_smoothed:.3f}")
                            r_real_norm = r_real_smoothed
                        
                        # 融合（两者权重接近，评论家占 40%）
                        alpha = 0.4  # 评论家权重
                        beta = 0.6   # 真实奖励权重
                        r_total_norm = beta * r_real_norm + alpha * r_dense
                        
                        # ★磁饱和软约束（拉格朗日乘子法）：从奖励中扣除 λ·g
                        # g = max(0, (B_max - threshold) / scale) 约束违背程度（归一化）
                        if self._saturation_as_constraint:
                            b_max_val = 0.0
                            try:
                                b_max_val = float(result_for_rl.get("B_sat", result_for_rl.get("B_max", 0.0)) or 0.0)
                            except Exception:
                                b_max_val = 0.0
                            
                            g = 0.0
                            if self._sat_scale_t > 0:
                                g = max(0.0, (b_max_val - self._sat_threshold_t) / self._sat_scale_t)
                            g = min(1.0, g)  # 限幅到 [0,1]，避免过大惩罚破坏奖励尺度
                            
                            penalty = self._sat_lambda * g
                            if penalty != 0.0 or g > 0:
                                r_total_norm = r_total_norm - penalty
                                self._log(f"[Constraint] 磁饱和软约束: B_max={b_max_val:.3f}T | g={g:.3f} | λ={self._sat_lambda:.3f} | penalty={penalty:.3f} | r_total→{r_total_norm:.3f}")
                            
                            # 原始-对偶更新：λ ← max(0, λ + lr·(g - target))
                            self._sat_lambda = max(0.0, self._sat_lambda + self._sat_lambda_lr * (g - self._sat_target))
                            self._log(f"[Constraint] λ 更新: lr={self._sat_lambda_lr:.3f} target={self._sat_target:.3f} → λ={self._sat_lambda:.3f}")
                        
                        # 限幅，保证融合奖励稳定
                        r_total_norm = max(-1.0, min(1.0, r_total_norm))
                        
                        # 反归一化回原始量级（可选，便于理解）
                        # combined_reward = r_total_norm * reward_scale
                        combined_reward = r_total_norm  # 保持归一化，范围 [-1, 1]
                        
                        self._log(f"[RL] 奖励融合（归一化）: r_real={current_reward:.2f}→{r_real_norm:.3f} | r_dense={r_dense:.3f} | r_total={combined_reward:.3f}")
                        current_reward = combined_reward
                    
                    reward_history.append(current_reward)
                    self._log(f"[RL] 最终奖励 = {current_reward:.2f} (探索={is_exploration})")
                
                # RL 增强：存储经验
                if enable_rl and self.experience_buffer and new_state:
                    action = {}
                    if self._current_state:  # 修复：使用上一轮状态
                        for k in new_state:
                            if k in self._current_state:
                                action[k] = new_state[k] - self._current_state[k]
                    
                    # ★获取上一轮 fitness，用于判断是否改善
                    _prev_fitness = fitness_history[-2] if len(fitness_history) >= 2 else None
                    _curr_fitness = result_for_rl.get("fitness")
                    _improved = (_curr_fitness < _prev_fitness) if (_curr_fitness is not None and _prev_fitness is not None) else None
                    
                    exp = self.experience_buffer.store(
                        iteration=iteration + 1,
                        state=new_state,
                        action=action,
                        result=result_for_rl,
                        reward=current_reward,
                        llm_reasoning=response_text[:500],
                        prev_fitness=_prev_fitness  # ★新增：传入上一轮 fitness
                    )
                    # ★立即强制保存，防止程序中断丢失数据
                    self.experience_buffer.save()
                    _improved_str = "✓改善" if _improved else ("✗未改善" if _improved is False else "N/A")
                    self._log(f"[RL] 📝 存储经验并已保存 (iteration={iteration + 1}, fitness_improved={_improved_str})")
                    
                    # ★ ExpeL 对比批评：事件驱动触发（方案C+D）
                    # 只在有意义的事件后触发，避免每轮都调用 LLM
                    if self._expel_enabled and self.contrast_critique and exp:
                        _fitness_for_expel = result_for_rl.get("fitness")
                        
                        # ★方案D：判断事件类型
                        _expel_event = "normal"  # 默认普通轮次（会被跳过）
                        
                        # 事件1：发现新最佳 fitness
                        if _fitness_for_expel is not None and _fitness_for_expel < best_fitness:
                            _expel_event = "new_best"
                        # 事件2：约束违规
                        elif result_for_rl.get("status") == "constraint_violation":
                            _expel_event = "constraint_violation"
                        # 事件3：显著退化（fitness 恶化 >10%）
                        elif _prev_fitness is not None and _fitness_for_expel is not None:
                            if _prev_fitness != 0 and (_fitness_for_expel - _prev_fitness) / abs(_prev_fitness) > 0.10:
                                _expel_event = "significant_degradation"
                        
                        await self._trigger_expel_critique(exp, _fitness_for_expel, event_type=_expel_event)
                
                # RL 增强：更新策略
                if enable_rl and self.strategy_manager and new_state:
                    success = result_for_rl.get("status") == "ok"
                    # current_fitness 已在上面提取，无需重复
                    old_epsilon = self.strategy_manager.epsilon
                    old_success_count = len(self.strategy_manager.success_regions)
                    old_success_patterns = len(self.strategy_manager.success_patterns)
                    old_failure_patterns = len(self.strategy_manager.failure_patterns)
                    
                    # 提取错误信息（用于失败模式学习）
                    errors = result_for_rl.get("errors", []) if not success else []
                    
                    # ★修复：使用 _current_state（上一轮状态）而非 _previous_state（上上轮状态）
                    # old_state 应该是本轮开始前的状态，即上一轮的结果
                    old_state_for_update = self._current_state or {}
                    
                    update_result = self.strategy_manager.update_after_result(
                        old_state=old_state_for_update,
                        new_state=new_state,
                        success=success,
                        fitness=current_fitness,
                        errors=errors,
                        best_fitness=best_fitness  # 传入历史最佳
                    )
                    
                    # 策略更新日志
                    new_epsilon = self.strategy_manager.epsilon
                    new_success_count = len(self.strategy_manager.success_regions)
                    new_success_patterns = len(self.strategy_manager.success_patterns)
                    new_failure_patterns = len(self.strategy_manager.failure_patterns)
                    epsilon_change = new_epsilon - old_epsilon
                    
                    self._log("-" * 50)
                    self._log(f"[RL] 🎯 【策略更新】第 {self.strategy_manager.iteration} 轮")
                    
                    # ★新增：显示软失败信息
                    if update_result and update_result.get("is_soft_failure"):
                        result_emoji = "⚠️ 软失败"
                        self._log(f"[RL] 本轮结果: {result_emoji}")
                        self._log(f"[RL] 原因: {update_result.get('reason')}")
                    else:
                        result_emoji = "✅ 成功" if success else "❌ 失败"
                        self._log(f"[RL] 本轮结果: {result_emoji}")
                    
                    # 探索率变化
                    epsilon_emoji = "📉" if epsilon_change < 0 else "📈" if epsilon_change > 0 else "➡️"
                    self._log(f"[RL] 探索率 ε: {old_epsilon:.4f} → {new_epsilon:.4f} ({epsilon_emoji} {epsilon_change:+.4f})")
                    
                    # 成功模式变化
                    if new_success_patterns > old_success_patterns:
                        self._log(f"[RL] 🌟 新增成功模式！总计: {new_success_patterns}")
                        if self.strategy_manager.success_patterns:
                            best = min(self.strategy_manager.success_patterns, key=lambda p: p.fitness)
                            self._log(f"[RL] 📊 最佳模式 fitness={best.fitness:.4f}")
                    
                    # 失败模式变化
                    if new_failure_patterns > old_failure_patterns:
                        self._log(f"[RL] ⚠️ 新增失败模式！总计: {new_failure_patterns}")
                        if errors:
                            self._log(f"[RL] 错误类型: {errors[0][:50]}...")
                    
                    # 成功区域变化
                    if new_success_count > old_success_count:
                        self._log(f"[RL] 📍 新增成功区域！总计: {new_success_count}")
                    
                    # 显示学习到的规则（如果有新规则）
                    if self.strategy_manager.learned_rules:
                        self._log(f"[RL] 📚 已学习规则: {len(self.strategy_manager.learned_rules)} 条")
                    
                    # 最近成功率
                    if self.strategy_manager.recent_results:
                        rate = sum(self.strategy_manager.recent_results) / len(self.strategy_manager.recent_results)
                        self._log(f"[RL] 最近成功率: {rate:.1%} ({sum(self.strategy_manager.recent_results)}/{len(self.strategy_manager.recent_results)})")
                    self._log("-" * 50)
                
                # ★ Reflexion：记录本轮并检查是否触发反思
                if self.reflection_manager:
                    reflection_result = self.reflection_manager.record_round(
                        round=iteration,
                        params=new_state or {},
                        fitness=current_fitness,
                        success=success,
                        errors=errors
                    )
                    if reflection_result:
                        # 触发了反思，输出反思内容摘要
                        self._log("=" * 60)
                        self._log("[Reflection] 🪞 【自我反思已触发】")
                        # 只显示前几行
                        reflection_lines = reflection_result.split("\n")[:8]
                        for line in reflection_lines:
                            if line.strip():
                                self._log(f"[Reflection] {line}")
                        self._log("[Reflection] （完整反思已写入 memory_stream.jsonl）")
                        self._log("=" * 60)
                
                # ★ 更新短期记忆（Trajectory）
                if self.memory_manager:
                    self.memory_manager.short_term.add_step(
                        round=iteration,
                        params=new_state or {},
                        fitness=current_fitness,
                        reward=current_reward,
                        success=success,
                        errors=errors,
                        critic_feedback=self._pending_critic_feedback,
                        llm_reasoning=None  # 可选：从 LLM 响应中提取
                    )
                
                # 更新状态
                self._previous_state = self._current_state
                if new_state:
                    self._current_state = new_state.copy()
                
                # 写入 CSV（包含 RL 字段和评论家字段）
                self._save_iteration_result_to_csv(
                    csv_writer, self._last_sim_result,
                    reward=current_reward, is_exploration=is_exploration,
                    dense_reward=self._last_dense_reward,
                    critic_scores=self._last_critic_scores
                )
                csv_file.flush()
                self._log(f"[Auto] 本轮结果已写入 CSV，status={result_for_rl.get('status')}")
                
                # ★ 每轮结束后：自动备份本轮可迁移经验文件（按轮次落盘，便于重复实验/回放）
                self._backup_experience_files(results_file, iteration=iteration + 1)

                # * 打印轮次摘要（让每轮差异一目了然）
                n1_val = result_for_rl.get("n1")
                n2_val = result_for_rl.get("n2")
                if n1_val is None and self._last_sim_result:
                    sim_result = self._last_sim_result.get("result", {})
                    n1_val = sim_result.get("n1")
                    n2_val = sim_result.get("n2")
                self._log_round_summary(
                    iteration=iteration + 1,
                    params=new_state,
                    fitness=current_fitness,
                    n1=n1_val,
                    n2=n2_val,
                    prev_params=self._previous_state,
                    prev_fitness=fitness_history[-2] if len(fitness_history) >= 2 else None,
                    prev_n1=self.strategy_manager.n1_history[-2] if self.strategy_manager and len(self.strategy_manager.n1_history) >= 2 else None,
                    prev_n2=self.strategy_manager.n2_history[-2] if self.strategy_manager and len(self.strategy_manager.n2_history) >= 2 else None
                )
            else:
                # ★调试：记录为什么条件不满足导致 CSV 未写入
                self._log(f"[DEBUG-CSV] ⚠️ 跳过 CSV 写入: has_sim_result={bool(self._last_sim_result)} | has_result={_has_result} | already_processed={_already_processed}", level="warning")
                # 写入占位行，保证 iteration 连续（典型原因：没有新的结构化结果/结果已处理/迭代号不匹配）
                try:
                    skip_reasons = []
                    if not self._last_sim_result:
                        skip_reasons.append("no_last_sim_result")
                    if self._last_sim_result and not _has_result:
                        skip_reasons.append("no_result")
                    if _already_processed:
                        skip_reasons.append("already_processed")
                    if _stored_iter != _expected_iter:
                        skip_reasons.append(f"stored_iter={_stored_iter}")
                    reason = " | ".join(skip_reasons) if skip_reasons else "unknown"

                    self._write_placeholder_csv_row(
                        csv_writer=csv_writer,
                        iteration=_expected_iter,
                        status="skipped",
                        errors=[f"CSV 写入被跳过：{reason}"],
                        params=self._current_state or {},
                        result_source="Internal",
                        result_description="本轮未获得新的仿真结构化结果，写入占位行以保持 iteration 连续。"
                    )
                    csv_file.flush()
                except Exception as _e:
                    self._log(f"[DEBUG-CSV] ⚠️ 写入 skipped 占位行失败: {_e}", level="warning")

            # ========== 收敛判断：统一使用结构化仿真结果 ==========
            try:
                # ★ 优先从 _last_sim_result 获取 fitness（结构化数据，更可靠）
                current_fitness = None
                if self._last_sim_result:
                    sim_result = self._last_sim_result.get("result", {})
                    current_fitness = sim_result.get("fitness")
                
                # 兜底：如果没有仿真结果，尝试从 LLM 回复中解析（不推荐，仅兼容旧逻辑）
                if current_fitness is None:
                    current_fitness = self._extract_fitness_from_response(response_text)
                    if current_fitness is not None:
                        self._log(f"[Auto] ⚠️ 使用 LLM 回复解析 fitness（兜底）: {current_fitness:.6e}")
                
                if current_fitness is not None:
                    fitness_history.append(current_fitness)
                    successful_sim_count += 1  # ★成功获得 fitness 的轮次
                    
                    # ========== 磁饱和检测与处理 ==========
                    result_is_saturated = False
                    if self._last_sim_result:
                        result_is_saturated = self._last_sim_result.get("result", {}).get("is_saturated", False)
                    is_penalty_fitness = current_fitness >= self.SATURATION_PENALTY_THRESHOLD
                    
                    if result_is_saturated:
                        # 仿真返回的磁饱和标志：fitness 正常计算，但记录饱和状态
                        saturation_count += 1
                        total_saturation_count += 1
                        b_max_val = self._last_sim_result.get("result", {}).get("B_sat", "N/A") if self._last_sim_result else "N/A"
                        self._log(f"[Auto] ⚠️ 磁饱和检测：B_max={b_max_val}T ≥ 2.0T (fitness={current_fitness:.4f}，正常计算，连续饱和 {saturation_count} 轮)")
                        
                        # 磁饱和时 fitness 仍然有效，参与收敛判断
                        valid_fitness_history.append(current_fitness)
                        
                        # 连续饱和过多时发出警告
                        if saturation_count >= convergence_window:
                            self._log(f"[Auto] ⚠️ 警告：连续 {saturation_count} 轮出现磁饱和，建议增大 ta 或 dg 以降低磁通密度")
                    elif is_penalty_fitness:
                        # 旧逻辑兼容：fitness 是惩罚值（约束违规等）
                        saturation_count += 1
                        total_saturation_count += 1
                        self._log(f"[Auto] ⚠️ 约束违规检测：fitness={current_fitness:.2e} (连续违规 {saturation_count} 轮)")
                        
                        # 惩罚值不参与收敛判断
                        if saturation_count >= convergence_window:
                            self._log(f"[Auto] ⚠️ 警告：连续 {saturation_count} 轮约束违规，建议检查参数边界")
                    else:
                        # 有效值：加入有效历史并参与收敛判断
                        saturation_count = 0  # 重置连续饱和计数
                        valid_fitness_history.append(current_fitness)
                    
                    # 检查是否有改进（最小化问题：Δ = best - current > 0 表示变好）
                    if best_fitness == float('inf'):
                        # 初始状态：任何有限的 fitness 都是改进
                        delta = float('inf')
                        threshold_abs = 0.0
                    else:
                        # 正常状态：用 abs(best_fitness) 做比例阈值，避免负数导致符号翻转
                        delta = best_fitness - current_fitness  # Δ > 0 means improvement
                        threshold_abs = abs(best_fitness) * convergence_threshold if best_fitness != 0 else convergence_threshold
                    
                    if delta > threshold_abs:
                        best_fitness = current_fitness
                        best_params = self._current_state.copy() if self._current_state else None
                        no_improvement_count = 0
                        self._log(f"[Auto] 🎯 发现更优解！fitness = {best_fitness:.6e} (改进 Δ={delta:.6e})")
                        
                        # ★ 方案 B：更新 best_result Block
                        if self.memory_manager and best_params:
                            self.memory_manager.update_best_result(best_fitness, best_params)
                    else:
                        no_improvement_count += 1
                        self._log(f"[Auto] 本轮无显著改进（连续 {no_improvement_count}/{convergence_window} 轮无改进, Δ={delta:.6e}）")
                    
                    # ========== 收敛判断（连续 convergence_window 轮无改进）==========
                    # ★使用 successful_sim_count 而非 valid_iterations，确保 skipped/llm_failed 不计入最小轮次
                    if successful_sim_count >= min_iterations and no_improvement_count >= convergence_window:
                        self._log(f"\n[Auto] ===== 优化已收敛 =====")
                        self._log(f"[Auto] 共完成 {iteration + 1} 轮迭代（成功仿真 {successful_sim_count} 轮，跳过/失败 {skipped_count} 轮）")
                        self._log(f"[Auto] 连续 {convergence_window} 轮无显著改进")
                        self._log(f"[Auto] 最佳 fitness = {best_fitness:.6e}")
                        self._log(f"[Auto] ===========================\n")
                        break
                else:
                    # 本轮没有有效 fitness（可能是约束违规导致未运行仿真，或 skipped/llm_failed）
                    skipped_count += 1  # ★统计跳过/失败轮次
                    self._log(f"[Auto] ⚠️ 本轮无有效 fitness，不计入最小迭代次数（已跳过 {skipped_count} 轮）")

            except Exception as e:
                self._log(f"[Auto] ⚠️ 收敛判断异常: {e}", level="warning")

            # ========== 约束违规重试机制 ==========
            # 如果本轮只有约束违规而没有成功仿真，让 LLM 重新尝试（不增加 iteration）
            # ★修复：使用 _sim_executed_this_round 判断，避免因 _processed=True 导致误判
            # 只要本轮已执行过仿真（无论结果如何），就不再触发约束违规重试
            if self._constraint_violation_in_round and not self._sim_executed_this_round:
                constraint_retry_count += 1
                if constraint_retry_count < MAX_CONSTRAINT_RETRIES:
                    error_summary = "; ".join(self._last_constraint_errors[:3]) if self._last_constraint_errors else "参数不满足约束"
                    self._log(f"[Auto] 🔄 约束违规重试 ({constraint_retry_count}/{MAX_CONSTRAINT_RETRIES}): {error_summary}")
                    
                    # 追加消息让 LLM 重新提出参数
                    retry_prompt = (
                        f"⚠️ 约束违规！错误: {error_summary}\n\n"
                        f"请根据错误信息修正参数，重新调用 validate_maxwell_design。\n"
                        f"这是第 {constraint_retry_count} 次重试，请仔细检查约束条件。"
                    )
                    messages.append({"role": "user", "content": retry_prompt})
                    
                    # 重置约束违规标志，准备下一次尝试
                    self._constraint_violation_in_round = False
                    self._last_constraint_errors = []
                    continue  # 不增加 iteration，重新执行循环
                else:
                    self._log(f"[Auto] ⚠️ 约束违规重试次数已达上限 ({MAX_CONSTRAINT_RETRIES})，跳过本轮", level="warning")

            iteration += 1
        
        # ========== RL 增强：保存经验和输出统计 ==========
        if enable_rl and self.experience_buffer:
            self.experience_buffer.save()
        
        # ========== 元学习：已禁用旧版，改用 ExpeL 对比批评 ==========
        # （旧版元知识提取代码已移除，ExpeL 规则在每轮迭代后自动更新）
            patterns = self.experience_buffer.analyze_patterns()
            self._log(f"\n[RL] ===== 学习统计 =====")
            self._log(f"[RL] 总经验数: {patterns.get('total_experiences', 0)}")
            self._log(f"[RL] 成功率: {patterns.get('success_rate', 0)*100:.1f}%")
            if patterns.get('best_fitness') is not None:
                self._log(f"[RL] 历史最佳 fitness: {patterns['best_fitness']:.6e}")
            if reward_history:
                avg_reward = sum(reward_history) / len(reward_history)
                self._log(f"[RL] 平均奖励: {avg_reward:.2f}")
            self._log(f"[RL] =======================\n")
        
        if enable_rl and self.strategy_manager:
            strategy_info = self.strategy_manager.get_strategy_info()
            self._log(f"[RL] 最终探索率 ε = {strategy_info['epsilon']:.3f}")
            self._log(f"[RL] 成功区域数: {strategy_info['success_regions_count']}")
        
        # ========== Actor-Critic 统计 ==========
        if self.actor_critic_system:
            ac_summary = self.actor_critic_system.get_summary()
            self._log(f"\n[AC] ===== Actor-Critic 统计 =====")
            
            # 价值函数统计
            vf_summary = ac_summary.get("value_function", {})
            self._log(f"[AC] 状态价值函数 V(s):")
            self._log(f"[AC]   总状态数: {vf_summary.get('total_states', 0)} | 有效: {vf_summary.get('valid_states', 0)}")
            if vf_summary.get("best_fitness"):
                self._log(f"[AC]   历史最佳 fitness: {vf_summary.get('best_fitness'):.4f}")
            if "avg_value" in vf_summary:
                self._log(f"[AC]   平均价值: {vf_summary.get('avg_value', 0):.3f}")
            
            # 评论家集群统计
            if "critic_ensemble" in ac_summary:
                ce_summary = ac_summary["critic_ensemble"]
                self._log(f"[AC] 动作评论家集群:")
                self._log(f"[AC]   总评估次数: {ce_summary.get('total_evaluations', 0)}")
                if "prediction_accuracy" in ce_summary:
                    self._log(f"[AC]   预测准确率: {ce_summary['prediction_accuracy']*100:.1f}%")
                
                # TD 学习统计
                td_learning = ce_summary.get("td_learning", {})
                if td_learning:
                    self._log(f"[AC]   TD学习准确率: {td_learning.get('ensemble_accuracy', 0)*100:.1f}%")
                    self._log(f"[AC]   TD学习规则数: {td_learning.get('total_td_rules', 0)}")
                
                for name, info in ce_summary.get("critics", {}).items():
                    conf = info.get("dynamic_confidence", 0.5)
                    acc = info.get("prediction_accuracy", 0.5)
                    td_rules = info.get("td_rules_count", 0)
                    self._log(f"[AC]   {name}: 置信度={conf:.2f} | 准确率={acc:.1%} | TD规则={td_rules}")
            
            self._log(f"[AC] ================================\n")
        
        # ========== 评论家 TD 学习详细日志 ==========
        if self.critic_ensemble:
            td_log = self.critic_ensemble.build_td_learning_log()
            self._log(td_log)
        
        # 兼容旧的评论家统计
        elif self.critic_ensemble:
            critic_summary = self.critic_ensemble.get_summary()
            self._log(f"\n[Critic] ===== 评论家统计 =====")
            self._log(f"[Critic] 总评估次数: {critic_summary.get('total_evaluations', 0)}")
            if "prediction_accuracy" in critic_summary:
                self._log(f"[Critic] 预测准确率: {critic_summary['prediction_accuracy']*100:.1f}%")
            for name, info in critic_summary.get("critics", {}).items():
                self._log(f"[Critic] {name}: 经验={info.get('experience_count', 0)} | 策略={info.get('strategy_count', 0)}")
            self._log(f"[Critic] ===========================\n")
        
        # ========== ExpeL 对比批评统计 ==========
        if self._expel_enabled and self.contrast_critique:
            expel_stats = self.contrast_critique.get_stats()
            rule_summary = expel_stats.get("rule_summary", {})
            self._log(f"\n[ExpeL] ===== 对比批评统计 =====")
            self._log(f"[ExpeL] 对比批评次数: {expel_stats.get('critique_count', 0)}")
            self._log(f"[ExpeL] 跳过次数（去重）: {expel_stats.get('skipped_count', 0)}")  # ★方案C
            self._log(f"[ExpeL] 已分析配对数: {expel_stats.get('analyzed_pairs', 0)}")  # ★方案C
            self._log(f"[ExpeL] 规则总数: {rule_summary.get('total_rules', 0)}")
            self._log(f"[ExpeL] 平均置信度: {rule_summary.get('avg_confidence', 0):.2f}")
            self._log(f"[ExpeL] 规则来源: 对比={rule_summary.get('sources', {}).get('compare', 0)} | 成功={rule_summary.get('sources', {}).get('success', 0)}")
            if self.contrast_critique.rule_manager.rules:
                self._log(f"[ExpeL] 置信度最高规则:")
                for i, rule in enumerate(self.contrast_critique.rule_manager.rules[:3], 1):
                    self._log(f"[ExpeL]   {i}. [{rule.confidence}] {rule.text[:60]}...")
            self._log(f"[ExpeL] =============================\n")
        
        # 关闭 CSV 文件
        csv_file.close()
        
        # 输出最终统计
        self._log(f"\n[Auto] ===== 优化结束 =====")
        self._log(f"[Auto] 共完成 {iteration} 轮迭代")
        self._log(f"[Auto] 成功仿真: {successful_sim_count} 轮 | 跳过/失败: {skipped_count} 轮 | 磁饱和: {total_saturation_count} 轮")
        if iteration > 0:
            self._log(f"[Auto] 有效率: {successful_sim_count/iteration*100:.1f}% | 跳过率: {skipped_count/iteration*100:.1f}%")
        if valid_fitness_history:
            self._log(f"[Auto] 最佳 fitness = {best_fitness:.6e}")
        elif fitness_history:
            self._log(f"[Auto] ⚠️ 全部迭代均触发磁饱和，无有效 fitness")
        self._log(f"[Auto] 结果已保存到：{results_file}")
        self._log(f"[Auto] =======================\n")
        
        # ★ 优化结束后：再做一次最终备份（根目录覆盖式，便于快速取用最新版本）
        self._backup_experience_files(results_file, iteration=None)
        
        # ★ 发送邮件通知
        if self._notify_email:
            fitness_str = f"{best_fitness:.4f}" if valid_fitness_history else "N/A"
            email_subject = f"MCP Optimization Finished - Fitness: {fitness_str}"
            email_body = f"Optimization completed.\n\nTotal rounds: {iteration}\nSuccessful: {successful_sim_count}\nBest fitness: {fitness_str}\nResult file: {results_file}"
            self._send_notification_email(email_subject, email_body)

    def _backup_experience_files(self, results_file: str, iteration: Optional[int] = None) -> None:
        """
        自动备份本轮所有可迁移经验文件，命名与 CSV 结果文件对应。
        
        备份文件包括：
        - experience_buffer.json    经验回放缓冲区
        - expel_rules.json          ExpeL 对比学习规则
        - strategy_state.json       策略管理器状态
        - memory_stream.jsonl       Reflexion 反思记录
        - critic_experience/        评论家集群经验（整个目录）
        """
        import shutil
        
        # 根据 CSV 文件名生成备份目录名
        # 例如 AgenticOPT_20260205-204912_EI_RAG_RL.csv → AgenticOPT_20260205-204912_EI_RAG_RL_experience/
        base_name = os.path.splitext(os.path.basename(results_file))[0]
        results_dir = os.path.dirname(results_file) or "."
        base_backup_dir = os.path.join(results_dir, f"{base_name}_experience")
        # 每轮落盘：base_backup_dir/round_0001/...
        backup_dir = (
            os.path.join(base_backup_dir, f"round_{int(iteration):04d}")
            if iteration is not None
            else base_backup_dir
        )
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            # 需要备份的单文件列表
            single_files = [
                "experience_buffer.json",
                "expel_rules.json",
                "strategy_state.json",
                "memory_stream.jsonl",
            ]
            
            copied_count = 0
            for fname in single_files:
                src = os.path.join(results_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(backup_dir, fname))
                    copied_count += 1
            
            # 备份 critic_experience 目录
            critic_src = os.path.join(results_dir, "critic_experience")
            critic_dst = os.path.join(backup_dir, "critic_experience")
            if os.path.isdir(critic_src):
                if os.path.exists(critic_dst):
                    shutil.rmtree(critic_dst)
                shutil.copytree(critic_src, critic_dst)
                critic_file_count = len([f for f in os.listdir(critic_dst) if os.path.isfile(os.path.join(critic_dst, f))])
                copied_count += critic_file_count
            
            tag = f"round_{int(iteration):04d}" if iteration is not None else "final"
            self._log(f"[Backup] 📦 经验文件已备份到：{backup_dir}（{copied_count} 个文件，{tag}）")
            
        except Exception as e:
            self._log(f"[Backup] ⚠️ 经验备份失败: {e}", level="warning")

    def _send_notification_email(
        self,
        subject: str,
        body: str
    ) -> bool:
        """
        发送邮件通知。
        
        Args:
            subject: 邮件主题
            body: 邮件正文
            
        Returns:
            bool: 是否发送成功
        """
        if not self._notify_email:
            return False
        
        if not self._smtp_password:
            self._log("[Notify] ⚠️ 未配置 SMTP 授权码，无法发送邮件", level="warning")
            return False
        
        import smtplib
        from email.mime.text import MIMEText
        from email.header import Header
        
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = Header(subject, "utf-8")
            msg["From"] = self._notify_email
            msg["To"] = self._notify_email
            
            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                server.starttls()
                server.login(self._notify_email, self._smtp_password)
                server.sendmail(self._notify_email, [self._notify_email], msg.as_string())
            
            self._log(f"[Notify] ✅ 邮件通知已发送到 {self._notify_email}")
            return True
            
        except Exception as e:
            self._log(f"[Notify] ❌ 邮件发送失败: {e}", level="error")
            return False

    async def _trigger_expel_critique(
        self, 
        exp: "Experience", 
        current_fitness: Optional[float],
        event_type: str = "normal"
    ) -> None:
        """
        ExpeL 对比批评触发器
        
        ★方案C+D：事件驱动触发 + 去重
        - 只在有意义的事件后触发（新最佳、约束违规、显著退化）
        - 跳过已分析过的配对
        
        Args:
            exp: 当前经验
            current_fitness: 当前 fitness
            event_type: 触发事件类型
                - "new_best": 发现新最佳 fitness
                - "constraint_violation": 约束违规
                - "significant_degradation": 显著退化（fitness 恶化 >10%）
                - "normal": 普通轮次（将被跳过以节省 token）
        """
        if not self._expel_enabled or not self.contrast_critique:
            return
        
        if not self.experience_buffer:
            return
        
        # ★方案D：事件驱动 - 只在有意义的事件后触发
        # 普通轮次跳过，节省 LLM 调用
        if event_type == "normal":
            self._log(f"[ExpeL] ⏭️ 普通轮次，跳过对比批评（节省 token）")
            return
        
        try:
            # 简化日志（减少 DEBUG 输出）
            total_exp = len(self.experience_buffer.experiences)
            successful_exp = len([e for e in self.experience_buffer.experiences if e.is_success])
            self._log(f"[ExpeL] 📊 经验库: 总={total_exp}, 成功={successful_exp}, 失败={total_exp - successful_exp}")
            self._log(f"[ExpeL] 🎯 触发事件: {event_type} | 当前经验: is_success={exp.is_success}")
            
            # 找与当前经验相似但结果相反的经验
            contrast_exp = self.experience_buffer.retrieve_contrast_pair(
                current_state=exp.state,
                current_is_success=exp.is_success,
                distance_threshold=1.5
            )
            
            if contrast_exp is None:
                self._log(f"[ExpeL] ⚠️ 未找到对比配对")
                return
            
            # 确定成功和失败的经验
            if exp.is_success:
                success_exp_id = exp.id
                fail_exp_id = contrast_exp.id
                success_state = exp.state
                success_fitness = exp.fitness or 0.0
                fail_state = contrast_exp.state
                fail_errors = contrast_exp.errors or ["fitness 较差"]
            else:
                success_exp_id = contrast_exp.id
                fail_exp_id = exp.id
                success_state = contrast_exp.state
                success_fitness = contrast_exp.fitness or 0.0
                fail_state = exp.state
                fail_errors = exp.errors or ["仿真失败"]
            
            self._log(f"[ExpeL] 🔍 发现对比配对，触发对比批评...")
            
            # ★方案C：调用对比批评（带去重）
            operations = await self.contrast_critique.compare_critique(
                success_state=success_state,
                success_fitness=success_fitness,
                fail_state=fail_state,
                fail_errors=fail_errors,
                success_exp_id=success_exp_id,  # ★方案C：传入 ID 用于去重
                fail_exp_id=fail_exp_id
            )
            
            if operations:
                # 更新规则库
                stats = self.contrast_critique.rule_manager.update(operations)
                self._log(f"[ExpeL] ✅ 规则更新: +{stats['added']} 新增, {stats['agreed']} 同意, {stats['edited']} 修改, {stats['removed']} 删除")
                
                # 同步到 strategy_manager（如果有）
                if self.strategy_manager and self.strategy_manager.expel_enabled:
                    self.strategy_manager.rule_manager = self.contrast_critique.rule_manager
            elif operations == []:
                # 可能是去重跳过
                expel_stats = self.contrast_critique.get_stats()
                self._log(f"[ExpeL] ⏭️ 无新规则（可能已分析过，已跳过 {expel_stats.get('skipped_count', 0)} 次）")
            
        except Exception as e:
            self._log(f"[ExpeL] ⚠️ 对比批评失败: {e}", level="warning")

    def _save_iteration_result_to_csv(
        self, csv_writer, sim_result: dict,
        reward: float = 0.0, is_exploration: bool = False,
        dense_reward: float = 0.0, critic_scores: Optional[Dict[str, Any]] = None
    ):
        """将 run_maxwell_simulation 的结构化结果写入 CSV（含 RL 字段和评论家字段）"""
        result = sim_result.get("result", {})
        params = sim_result.get("params", {})
        iteration = sim_result.get("iteration", "")

        status = result.get("status", "")
        fitness = result.get("fitness", "")
        avg_B = result.get("avg_B", "")
        b_sat = result.get("B_sat", "")
        kb = result.get("kb", "")
        pb = result.get("pb", "")
        volume = result.get("volume", "")
        mass_total = result.get("mass_total", "")
        mass_mover = result.get("mass_mover", "")
        mass_stator = result.get("mass_stator", "")
        fld_bsat = result.get("fld_bsat_file", "")

        derived = result.get("derived_dimensions", {}) if isinstance(result, dict) else {}
        la = derived.get("la", "")
        ha = derived.get("ha", "")
        ws = derived.get("ws", "")
        ls = derived.get("ls", "")
        tb = derived.get("tb", "")
        twall = derived.get("twall", "")

        # 参数字段（未传入的为空）
        lm = params.get("lm", "")
        tm = params.get("tm", "")
        ta = params.get("ta", "")
        dg = params.get("dg", "")
        hs = params.get("hs", "")
        wslot = params.get("wslot", "")
        hslot = params.get("hslot", "")
        s = params.get("s", "")
        wa = params.get("wa", "")
        tb_ratio = params.get("tb_ratio", "")
        
        # 饱和分析字段（新增：两面平均|B|）
        b_mean_ta = result.get("B_mean_ta", "")
        b_mean_tb = result.get("B_mean_tb", "")
        is_saturated_ta = result.get("is_saturated_ta", "")
        is_saturated_tb = result.get("is_saturated_tb", "")
        saturation_region = result.get("saturation_region", "")
        saturation_suggestion = result.get("saturation_suggestion", "")

        # 匝数信息来自 result.turns
        turns = result.get("turns", {}) if isinstance(result, dict) else {}
        n1 = turns.get("n1", "")
        n2 = turns.get("n2", "")
        total_turns = turns.get("total", "")

        result_source = result.get("result_source", "")
        result_description = result.get("result_description", "")
        fld_file = result.get("fld_file", "")
        errors = result.get("errors", "")
        if isinstance(errors, list):
            errors = "; ".join(str(e) for e in errors)
        

        # 归一化指标（若缺失则留空）
        volume_refer = 6.75e-8
        mass_refer =3.794e-4
        kb_refer = 0.2391
        pb_refer = 0.8192

        volume_r = volume / volume_refer if volume not in ("", None) else ""
        mass_r = mass_total / mass_refer if mass_total not in ("", None) else ""
        kb_r = kb / kb_refer if kb not in ("", None) else ""
        pb_r = pb / pb_refer if pb not in ("", None) else ""

        # 提取评论家评分
        critic_magnetic = ""
        critic_performance = ""
        critic_constraint = ""
        critic_magnitude = ""
        if critic_scores:
            critic_magnetic = critic_scores.get("magnetic", {})
            if hasattr(critic_magnetic, "score"):
                critic_magnetic = critic_magnetic.score
            elif isinstance(critic_magnetic, dict):
                critic_magnetic = critic_magnetic.get("score", "")
            
            critic_performance = critic_scores.get("performance", {})
            if hasattr(critic_performance, "score"):
                critic_performance = critic_performance.score
            elif isinstance(critic_performance, dict):
                critic_performance = critic_performance.get("score", "")
            
            critic_constraint = critic_scores.get("constraint", {})
            if hasattr(critic_constraint, "score"):
                critic_constraint = critic_constraint.score
            elif isinstance(critic_constraint, dict):
                critic_constraint = critic_constraint.get("score", "")
            
            critic_magnitude = critic_scores.get("magnitude", {})
            if hasattr(critic_magnitude, "score"):
                critic_magnitude = critic_magnitude.score
            elif isinstance(critic_magnitude, dict):
                critic_magnitude = critic_magnitude.get("score", "")

        csv_writer.writerow([
            iteration, status, fitness, avg_B, b_sat, kb, pb,
            volume_r, mass_r, kb_r, pb_r,
            volume, mass_total, mass_mover, mass_stator,
            la, ha, ws, ls, tb, twall,
            lm, tm, ta, dg, hs, wslot, hslot, s, wa, tb_ratio,
            n1, n2, total_turns,
            result_source, result_description, fld_file, fld_bsat, errors,
            b_mean_ta, b_mean_tb, is_saturated_ta, is_saturated_tb,  # 两面平均|B|和饱和标志
            saturation_region, saturation_suggestion,  # 饱和分析字段
            reward, is_exploration,  # RL 增强字段
            dense_reward, critic_magnetic, critic_performance, critic_constraint, critic_magnitude  # 评论家字段
        ])

        # 记录已写入的 iteration，避免重复写同一轮（也为占位行逻辑服务）
        try:
            if not hasattr(self, "_written_csv_iterations"):
                self._written_csv_iterations = set()
            if iteration not in ("", None):
                self._written_csv_iterations.add(int(iteration))
        except Exception:
            pass
        
        # ★ 记录 n1, n2 历史（用于离散变量探索追踪）
        if self.strategy_manager and n1 != "" and n2 != "":
            try:
                self.strategy_manager.record_discrete_variables(int(n1), int(n2))
            except (ValueError, TypeError):
                pass  # 如果转换失败，忽略

    def _write_placeholder_csv_row(
        self,
        csv_writer,
        iteration: int,
        status: str,
        errors: Any,
        params: Optional[Dict[str, Any]] = None,
        result_source: str = "Internal",
        result_description: str = ""
    ) -> bool:
        """写入占位行以保持 CSV iteration 连续（不会重复写同一 iteration）。"""
        try:
            if not hasattr(self, "_written_csv_iterations"):
                self._written_csv_iterations = set()
            if int(iteration) in self._written_csv_iterations:
                return False
        except Exception:
            # 如果 iteration 无法转换，仍继续尝试写入（但不做去重）
            pass

        if errors is None:
            errors = []
        if isinstance(errors, str):
            errors = [errors]

        sim_result = {
            "iteration": iteration,
            "params": params or {},
            "result": {
                "status": status,
                "errors": errors,
                "result_source": result_source,
                "result_description": result_description,
            },
        }
        self._save_iteration_result_to_csv(csv_writer, sim_result)
        return True

    def _build_critic_feedback_message(
        self,
        dense_reward: float,
        advantage: float,
        v_current: Dict,
        v_proposed: Dict,
        suggestions: List[str],
        params_before: Optional[Dict[str, float]] = None,
        params_after: Optional[Dict[str, float]] = None
    ) -> str:
        """构建评论家反馈消息，用于注入到 LLM 提示中（简化版：只包含即时评估和建议）"""
        parts = ["[Actor-Critic 预评估反馈]"]
        
        # ========== 0. 派生变量状态（n1, n2 离散边界信息）==========
        if self._enable_discrete_guidance and params_after:
            derived_info = self._compute_derived_variable_info(params_after)
            if derived_info:
                parts.append("")
                parts.append("【派生变量状态】（整数绕组容量带来的离散边界）")
                parts.append(f"  当前: n1={derived_info['n1']} (匝/层), n2={derived_info['n2']} (层数), 总匝数={derived_info['n1']*derived_info['n2']}")
                parts.append(f"  n1 相邻边界: 增大 lm 约 {derived_info['delta_lm_to_next_n1']:+.3f}mm 时 n1 接近 {derived_info['n1']+1}")
                parts.append(f"  n2 相邻边界: 增大 hs 约 {derived_info['delta_hs_to_next_n2']:+.3f}mm 或减小 hslot 约 {derived_info['delta_hslot_to_next_n2']:.3f}mm 时 n2 接近 {derived_info['n2']+1}")
                parts.append("  提示：可在离散边界附近进行邻域采样，并同时满足几何约束。")
        
        # ========== 1. 即时评估 ==========
        parts.append("")
        parts.append("📊 【即时评估】")
        
        # 价值变化
        v_cur = v_current.get("value", 0)
        v_prop = v_proposed.get("value", 0)
        if advantage > 0.2:
            parts.append(f"  ✓ 价值预测: 提议状态价值较高 V(s')={v_prop:+.2f} > V(s)={v_cur:+.2f}")
        elif advantage < -0.2:
            parts.append(f"  ⚠️ 价值预测: 提议状态价值较低 V(s')={v_prop:+.2f} < V(s)={v_cur:+.2f}")
        else:
            parts.append(f"  → 价值预测: V(s')={v_prop:+.2f} ≈ V(s)={v_cur:+.2f}")
        
        # 稠密奖励
        if dense_reward > 0.3:
            parts.append(f"  ✓ 评论家预测: 方向正确 (稠密奖励={dense_reward:+.2f})")
        elif dense_reward < -0.3:
            parts.append(f"  ⚠️ 评论家预测: 方向可能有误 (稠密奖励={dense_reward:+.2f})")
        else:
            parts.append(f"  → 稠密奖励: {dense_reward:+.2f}")
        
        # ========== 2. 即时建议（来自评论家的具体建议）==========
        if suggestions:
            parts.append("")
            parts.append("💡 【即时建议】")
            for s in suggestions[:5]:  # 最多显示5条建议
                parts.append(f"  - {s}")
        
        # ========== 3. 策略库知识（可选：全部暴露）==========
        if self._full_strategy_exposure:
            # 计算变化的参数
            changed_params = set()
            if params_before and params_after:
                for k in params_after:
                    if k in params_before and params_before.get(k) != params_after.get(k):
                        changed_params.add(k)
            
            strategy_content = self._load_and_filter_critic_strategies(
                changed_params, 
                full_exposure=True
            )
            if strategy_content:
                parts.append("")
                parts.append("📚 【策略库知识（全部）】")
                parts.append(strategy_content)
                parts.append("")
                parts.append("💡 提示：前期探索可以大胆尝试，步子可以跨大一点！")
        
        return "\n".join(parts)
    
    def _compute_derived_variable_info(self, params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """计算派生变量 n1, n2 的当前值和相邻离散边界距离
        
        公式（来自 ActuatorDesignVariables）：
        - wcoil = 0.05 mm (固定线径)
        - ls = lm - s
        - twall = 0.5 * max(hs - hslot, 0)
        - n1 = max(1, round(0.8 * ls / wcoil)) = max(1, round(16 * (lm - s)))
        - n2 = max(1, floor(0.9 * twall / wcoil)) = max(1, floor(9 * max(hs - hslot, 0)))
        """
        try:
            lm = params.get("lm", 4.5)
            s = params.get("s", 1.0)
            hs = params.get("hs", 1.8)
            hslot = params.get("hslot", 1.0)
            
            wcoil = float(os.getenv("WIRE_DIAMETER_MM", "0.05"))  # 从环境变量读取线径
            
            # 计算 ls 和 twall（与 ActuatorDesignVariables 保持一致）
            ls = lm - s
            twall = 0.5 * max(hs - hslot, 0)
            
            # 计算当前 n1, n2
            # n1 = round(0.8 * ls / wcoil) = round(16 * (lm - s))
            # n2 = floor(0.9 * twall / wcoil) = floor(18 * twall) = floor(9 * (hs - hslot))
            n1_raw = 0.8 * ls / wcoil  # = 16 * (lm - s)
            n2_raw = 0.9 * twall / wcoil  # = 18 * twall = 9 * (hs - hslot)
            
            # n1 可由环境变量选择取整方式；n2 固定向下取整。
            floor_mode = os.getenv("TURNS_FLOOR_MODE", "0").lower() in ("1", "true", "yes", "on")
            if floor_mode:
                n1 = max(1, int(n1_raw))
            else:
                n1 = max(1, int(round(n1_raw)))
            n2 = max(1, int(n2_raw))
            
            # 计算到下一个离散边界的距离
            # n1 跟随其可选取整方式；n2 始终需要 raw >= n2 + 1。
            n1_threshold = 1.0 if floor_mode else 0.5
            n2_threshold = 1.0
            
            # n1 相邻边界：0.8 * ls / wcoil >= n1 + threshold
            # lm >= (n1 + threshold) * wcoil / 0.8 + s
            lm_next_n1 = (n1 + n1_threshold) * wcoil / 0.8 + s
            delta_lm_to_next_n1 = lm_next_n1 - lm
            
            # n2 相邻边界：0.9 * twall / wcoil >= n2 + threshold
            # hs >= 2 * (n2 + threshold) * wcoil / 0.9 + hslot
            twall_next_n2 = (n2 + n2_threshold) * wcoil / 0.9
            hs_next_n2 = 2 * twall_next_n2 + hslot
            delta_hs_to_next_n2 = hs_next_n2 - hs
            
            # 计算通过减小 hslot 接近相邻 n2 边界所需的变化量
            # hslot_next <= hs - 2 * (n2 + threshold) * wcoil / 0.9
            hslot_next_n2 = hs - 2 * (n2 + n2_threshold) * wcoil / 0.9
            delta_hslot_to_next_n2 = hslot - hslot_next_n2  # 需要减小的量（正数）
            
            return {
                "n1": n1,
                "n2": n2,
                "n1_raw": n1_raw,
                "n2_raw": n2_raw,
                "delta_lm_to_next_n1": delta_lm_to_next_n1,
                "delta_hs_to_next_n2": delta_hs_to_next_n2,
                "delta_hslot_to_next_n2": delta_hslot_to_next_n2,
                "hslot_next_n2": hslot_next_n2,
            }
        except Exception:
            return None
    
    def _load_and_filter_critic_strategies(self, changed_params: set, full_exposure: bool = False) -> str:
        """读取并筛选评论家策略库，返回与当前参数变化相关的策略
        
        Args:
            changed_params: 变化的参数集合
            full_exposure: 是否暴露全部策略（不筛选）
        """
        from pathlib import Path
        
        # 策略库路径
        critic_dir = Path(__file__).resolve().parents[2] / "critic_experience"
        
        # 策略库配置
        strategy_files = {
            "magnetic": ("🧲 磁路策略", ["ta", "tb", "tb_ratio", "dg", "tm", "lm", "B_max", "B_sat", "磁饱和"]),
            "performance": ("📈 性能策略", ["fitness", "kb", "pb", "体积", "质量", "性能"]),
            "constraint": ("⚠️ 约束策略", ["wslot", "hslot", "s", "n1", "n2", "约束", "边界"]),
            "magnitude": ("📏 幅度策略", ["改动", "步长", "幅度", "调整"]),
        }
        
        result_parts = []
        
        for critic_type, (label, keywords) in strategy_files.items():
            strategy_file = critic_dir / f"{critic_type}_strategy.json"
            if not strategy_file.exists():
                continue
            
            try:
                with open(strategy_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    strategies = data.get("strategies", [])
                    confidence = data.get("dynamic_confidence", 0.5)
                    
                    if not strategies:
                        continue
                    
                    # ★ 全部暴露模式：不筛选，直接返回所有策略
                    if full_exposure:
                        relevant_strategies = strategies
                    else:
                        # 筛选相关策略
                        relevant_strategies = []
                        for s in strategies:
                            # 检查策略是否与变化的参数相关
                            is_relevant = False
                            
                            # 方法1：策略中提到了变化的参数
                            for param in changed_params:
                                if param.lower() in s.lower():
                                    is_relevant = True
                                    break
                            
                            # 方法2：策略中包含该评论家的关键词
                            if not is_relevant:
                                for kw in keywords:
                                    if kw.lower() in s.lower():
                                        is_relevant = True
                                        break
                            
                            # 方法3：TD 学习规则总是相关
                            if "TD误差" in s or "预测需要更谨慎" in s or "历史预测偏差" in s:
                                is_relevant = True
                            
                            if is_relevant:
                                relevant_strategies.append(s)
                        
                        # 如果没有筛选到相关策略，取前2条通用策略
                        if not relevant_strategies and strategies:
                            relevant_strategies = strategies[:2]
                    
                    if relevant_strategies:
                        conf_emoji = "🟢" if confidence > 0.6 else "🟡" if confidence > 0.4 else "🔴"
                        result_parts.append(f"  {label} {conf_emoji}(置信度:{confidence:.0%})")
                        # ★ 全部暴露模式：不限制条数
                        max_strategies = len(relevant_strategies) if full_exposure else 4
                        for s in relevant_strategies[:max_strategies]:
                            result_parts.append(f"    • {s}")
                        
            except Exception as e:
                continue
        
        return "\n".join(result_parts)

    def _summarize_tool_response(self, tool_name: str, tool_content: str) -> str:
        """提取工具返回的关键信息，简化输出"""
        try:
            data = json.loads(tool_content)
        except json.JSONDecodeError:
            return tool_content[:200] + "..." if len(tool_content) > 200 else tool_content
        
        if tool_name == "validate_maxwell_design":
            status = data.get("status", "unknown")
            errors = data.get("errors", [])
            derived = data.get("derived", {})
            if status == "ok":
                n1 = derived.get("n1", "?")
                n2 = derived.get("n2", "?")
                return f"status=ok | n1={n1}, n2={n2}, turns={n1*n2 if isinstance(n1,int) and isinstance(n2,int) else '?'}"
            else:
                return f"status={status} | errors: {errors[:2]}..." if len(errors) > 2 else f"status={status} | errors: {errors}"
        
        elif tool_name == "run_maxwell_simulation":
            result = data.get("result", {})
            status = result.get("status", "unknown")
            if status == "ok":
                fitness = result.get("fitness", 0)
                avg_B = result.get("avg_B", 0)
                kb = result.get("kb", 0)
                mass = result.get("mass_total", 0)
                return f"status=ok | fitness={fitness:.4e} | avg_B={avg_B:.4f} | kb={kb:.4f} | mass={mass:.6f}"
            else:
                errors = result.get("errors", [])
                return f"status={status} | errors: {errors[:2]}..."
        
        else:
            # 其他工具，截断显示
            return str(data)[:300] + "..." if len(str(data)) > 300 else str(data)

    def _extract_result_from_response(self, response_text: str) -> list:
        """从模型回复中提取完整结果数据，用于保存到 CSV"""
        import re
        # 查找 run_maxwell_simulation 的 JSON 返回
        pattern = r'\{[^{}]*"result":\s*\{[^{}]*"status":\s*"ok"[^{}]*\}[^{}]*\}'
        
        # 尝试找到仿真结果的 JSON
        try:
            # 更简单的方式：直接搜索关键字段
            fitness_match = re.search(r'"fitness":\s*([\d.eE+-]+)', response_text)
            avg_B_match = re.search(r'"avg_B":\s*([\d.eE+-]+)', response_text)
            kb_match = re.search(r'"kb":\s*([\d.eE+-]+)', response_text)
            pb_match = re.search(r'"pb":\s*([\d.eE+-]+)', response_text)
            mass_match = re.search(r'"mass_total":\s*([\d.eE+-]+)', response_text)
            
            # 提取设计参数
            lm_match = re.search(r'lm[=:]\s*([\d.]+)', response_text)
            tm_match = re.search(r'tm[=:]\s*([\d.]+)', response_text)
            ta_match = re.search(r'ta[=:]\s*([\d.]+)', response_text)
            dg_match = re.search(r'dg[=:]\s*([\d.]+)', response_text)
            hs_match = re.search(r'hs[=:]\s*([\d.]+)', response_text)
            # wwall 不再作为输入存储，twall 为派生值
            wslot_match = re.search(r'wslot[=:]\s*([\d.]+)', response_text)
            hslot_match = re.search(r'hslot[=:]\s*([\d.]+)', response_text)
            s_match = re.search(r'"s":\s*([\d.]+)', response_text)
            wa_match = re.search(r'wa[=:]\s*([\d.]+)', response_text)
            n1_match = re.search(r'"n1":\s*(\d+)', response_text)
            n2_match = re.search(r'"n2":\s*(\d+)', response_text)
            turns_match = re.search(r'"total":\s*(\d+)', response_text)
            
            if fitness_match:
                return [
                    'ok',
                    float(fitness_match.group(1)) if fitness_match else '',
                    float(avg_B_match.group(1)) if avg_B_match else '',
                    float(kb_match.group(1)) if kb_match else '',
                    float(pb_match.group(1)) if pb_match else '',
                    float(mass_match.group(1)) if mass_match else '',
                    float(lm_match.group(1)) if lm_match else '',
                    float(tm_match.group(1)) if tm_match else '',
                    float(ta_match.group(1)) if ta_match else '',
                    float(dg_match.group(1)) if dg_match else '',
                    float(hs_match.group(1)) if hs_match else '',
                    '',
                    float(wslot_match.group(1)) if wslot_match else '',
                    float(hslot_match.group(1)) if hslot_match else '',
                    float(s_match.group(1)) if s_match else '',
                    float(wa_match.group(1)) if wa_match else '',
                    int(n1_match.group(1)) if n1_match else '',
                    int(n2_match.group(1)) if n2_match else '',
                    int(turns_match.group(1)) if turns_match else '',
                ]
        except Exception:
            pass
        return None

    def _extract_fitness_from_response(self, response_text: str) -> float:
        """从模型回复中提取 fitness 值"""
        import re
        # 匹配常见的 fitness 格式
        patterns = [
            r'"fitness":\s*([\d.eE+-]+)',
            r'fitness[=:]\s*([\d.eE+-]+)',
            r'fitness\s*[:：]\s*([\d.eE+-]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    # ========== RL 增强：人类反馈接口 ==========
    
    def add_feedback(
        self,
        text: str,
        feedback_type: str = "suggestion",
        priority: int = 2,
        related_params: Optional[List[str]] = None
    ) -> bool:
        """添加人类反馈
        
        Args:
            text: 反馈内容
            feedback_type: 类型（correction/suggestion/warning/confirmation）
            priority: 优先级（1=低, 2=中, 3=高, 4=紧急）
            related_params: 相关参数名列表
        
        Returns:
            是否添加成功
        """
        if self.feedback_handler is None:
            self.feedback_handler = FeedbackHandler()
        
        try:
            self.feedback_handler.add_feedback(
                text=text,
                feedback_type=feedback_type,
                priority=priority,
                related_params=related_params or []
            )
            return True
        except Exception as e:
            self._log(f"添加反馈失败: {e}", level="error")
            return False
    
    def get_rl_status(self) -> Dict[str, Any]:
        """获取 RL 增强组件状态"""
        status = {
            "feedback_count": 0,
            "experience_count": 0,
            "epsilon": 0.0,
            "success_rate": 0.0,
            "best_fitness": None
        }
        
        if self.feedback_handler:
            status["feedback_count"] = len(self.feedback_handler.feedbacks)
        
        if self.experience_buffer:
            patterns = self.experience_buffer.analyze_patterns()
            status["experience_count"] = patterns.get("total_experiences", 0)
            status["success_rate"] = patterns.get("success_rate", 0)
            status["best_fitness"] = patterns.get("best_fitness")
        
        if self.strategy_manager:
            status["epsilon"] = self.strategy_manager.epsilon
        
        return status
    
    def clear_experience(self):
        """清空经验缓冲"""
        if self.experience_buffer:
            self.experience_buffer.experiences = []
            self.experience_buffer.save()
            self._log("[RL] 经验缓冲已清空")
    
    def clear_feedback(self):
        """清空反馈"""
        if self.feedback_handler:
            self.feedback_handler.feedbacks = []
            self.feedback_handler._save()
            self._log("[RL] 反馈已清空")
    
    def _check_feedback_file(self, filepath: str) -> List[Tuple[str, str, int]]:
        """检查反馈输入文件，返回新反馈列表
        
        文件格式（每行一条反馈）：
        - 普通反馈：直接写内容
        - 紧急反馈：以 ! 开头
        - 警告类型：以 [warning] 开头
        - 纠正类型：以 [correction] 开头
        
        示例：
        注意 dg 要大于 0.35
        ![warning] 立即停止探索 n2<3 的区域
        [correction] hslot 必须大于 tb + 0.1
        
        Returns:
            List of (text, feedback_type, priority)
        """
        if not os.path.exists(filepath):
            return []
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if not lines:
                return []
            
            # 清空文件（已读取）
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("")
            
            feedbacks = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                fb_type = "suggestion"
                priority = 2
                
                # 解析紧急标记
                if line.startswith("!"):
                    priority = 4
                    line = line[1:].strip()
                
                # 解析类型标记
                if line.startswith("[warning]"):
                    fb_type = "warning"
                    line = line[9:].strip()
                    if priority < 3:
                        priority = 3
                elif line.startswith("[correction]"):
                    fb_type = "correction"
                    line = line[12:].strip()
                    if priority < 3:
                        priority = 3
                elif line.startswith("[suggestion]"):
                    fb_type = "suggestion"
                    line = line[12:].strip()
                elif line.startswith("[confirmation]"):
                    fb_type = "confirmation"
                    line = line[14:].strip()
                
                if line:
                    feedbacks.append((line, fb_type, priority))
            
            return feedbacks
        except Exception as e:
            self._log(f"[RL] 读取反馈文件失败: {e}", level="warning")
            return []

