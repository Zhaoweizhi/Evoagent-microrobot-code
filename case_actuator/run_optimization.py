import argparse
import asyncio
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

# --- Windows 控制台中文乱码修复（强制 UTF-8 输出） ---
def _force_utf8_stdio() -> None:
    if os.name != "nt":
        return
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_force_utf8_stdio()

# Add the project's `src/` (one level up) to sys.path so `import mymcp` works
# without requiring `pip install -e .`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from loguru import logger

from mymcp.client import MCPClient
from mymcp.mcp_adapter import MCPOpenAIAdapter
from mymcp.rag import RAGConfig, RAGEngine
from mymcp.critic import CriticEnsemble, DEFAULT_CRITIC_MODELS
from mymcp.value_function import StateValueFunction, ActorCriticSystem

# 追加文件日志，便于长期留档
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/run_{time}.log",
    encoding="utf-8",
    enqueue=True,
    rotation="20 MB",
    retention="7 days",
)

DEFAULT_CLOUD_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_CLOUD_MODEL = "gpt-5.2"
DEFAULT_LOCAL_MODEL = "qwen2.5:7b-instruct"
BASE_DIR = _PROJECT_ROOT
DEFAULT_LITER_DIR = os.path.join(BASE_DIR, "liter")
DEFAULT_LOG_DIR = os.path.join(_HERE, "logs")
# RAG corpus directory (PDF / log literature). NOT shipped in this repo;
# users must place their own PDFs under <project_root>/liter/ if they want to
# use RAG.  See README for details.
DEFAULT_RAG_DIR = DEFAULT_LITER_DIR
DEFAULT_RAG_CACHE = os.path.join(DEFAULT_LITER_DIR, ".rag_cache.json")
DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
SERVER_SCRIPT = os.path.join(_PROJECT_ROOT, "src", "mymcp", "server.py")
AUTO_KEYWORDS = ["必须遵循", "多轮", "run_maxwell_simulation", "validate_maxwell_design"]
DEFAULT_AUTO_ITERATIONS = 200


@dataclass
class LLMConfig:
    mode: str
    base_url: str
    model: str
    api_key: Optional[str]


def should_use_auto(prompt: str) -> bool:

    lower_prompt = prompt.lower().strip()
    if lower_prompt.startswith("auto:"):
        return True
    # 检测是否包含工具名或关键字
    has_tool = ("run_maxwell_simulation" in lower_prompt or 
                "validate_maxwell_design" in lower_prompt)
    has_keyword = any(keyword in prompt for keyword in AUTO_KEYWORDS)
    return has_tool and has_keyword


def extract_system_prompt(prompt: str) -> str:
    if prompt.lower().startswith("auto:"):
        return prompt.split(":", 1)[1].strip()
    return prompt


def parse_manual_warm_start(params: str) -> Optional[Dict[str, float]]:
    """解析命令行传入的手动 warm-start 设计，示例：
    lm=4.7,tm=0.4,ta=0.5,dg=0.3,hs=1.5,wslot=2.3,hslot=1.1,s=1.0,wa=2.0,tb_ratio=1.8
    tb_ratio 为可选项，不传则由 LLM 自主探索。
    """
    if not params:
        return None
    base_keys = {"lm", "tm", "ta", "dg", "hs", "wslot", "hslot", "s", "wa"}
    optional_keys = {"tb_ratio"}
    allowed_keys = base_keys | optional_keys
    result: Dict[str, float] = {}
    try:
        parts = [p.strip() for p in params.replace(";", ",").split(",") if p.strip()]
        for part in parts:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k or k not in allowed_keys:
                continue
            result[k] = float(v)
        # 必须包含所有基础字段，tb_ratio 可选
        if base_keys.issubset(result.keys()):
            return result
    except Exception as exc:
        logger.warning(f"解析 warm-start 参数失败: {exc}")
    return None


def load_warm_start_design(csv_path: str) -> Optional[dict]:
    """从上一轮优化 CSV 中读取最优设计参数（fitness 最小者）。"""
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            best_row = None
            best_fitness = float("inf")
            for row in reader:
                try:
                    fitness = float(row.get("fitness", "inf"))
                except Exception:
                    continue
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_row = row
            if not best_row:
                return None
            base_fields = ["lm", "tm", "ta", "dg", "hs", "wslot", "hslot", "s", "wa"]
            optional_fields = ["tb_ratio"]
            fields = base_fields + optional_fields
            design = {}
            for key in fields:
                val = best_row.get(key)
                # 对必填字段进行严格检查
                if key in base_fields:
                    if val is None or val == "":
                        return None
                    try:
                        design[key] = float(val)
                    except Exception:
                        return None
                else:
                    # 可选字段（如 tb_ratio），只有在存在且可解析时才加入
                    if val is None or val == "":
                        continue
                    try:
                        design[key] = float(val)
                    except Exception:
                        continue
            return design
    except Exception as exc:
        logger.warning(f"读取 warm-start CSV 失败: {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maxwell MCP 超级助手")
    parser.add_argument("--mode",
                        choices=["cloud", "local"],
                        default=os.getenv("LLM_MODE", "cloud"),
                        help="选择云端 or 本地大模型")
    parser.add_argument("--base-url",
                        default=os.getenv("LLM_BASE_URL"),
                        help="自定义 LLM API Base URL")
    parser.add_argument("--model",
                        default=os.getenv("LLM_MODEL"),
                        help="LLM 模型名称")
    parser.add_argument("--api-key",
                        default=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
                        help="LLM API Key（本地模式可为空）")
    parser.add_argument("--auto-iterations",
                        type=int,
                        default=None,
                        help="最大自动优化轮数（默认 200）")
    parser.add_argument("--min-iterations",
                        type=int,
                        default=50,
                        help="最小迭代轮数（收敛判断前至少运行这么多轮，默认 30）")
    parser.add_argument("--convergence-window",
                        type=int,
                        default=20,
                        help="收敛窗口（连续多少轮无改进视为收敛，默认 20）")
    parser.add_argument("--server-script",
                        default=SERVER_SCRIPT,
                        help="自定义 MCP 服务端脚本路径")
    parser.add_argument("--rag",
                        dest="enable_rag",
                        action="store_true",
                        default=True,
                        help="启用基于本地 liter 文档的 RAG（默认开启）")
    parser.add_argument("--no-rag",
                        dest="enable_rag",
                        action="store_false",
                        help="禁用 RAG")
    parser.add_argument("--rag-dir",
                        default=DEFAULT_RAG_DIR,
                        help="RAG 文档目录，支持用分号或竖线分隔多个目录，例如 liter;logs")
    parser.add_argument("--rag-cache",
                        default=DEFAULT_RAG_CACHE,
                        help="RAG embedding 缓存文件")
    parser.add_argument("--rag-top-k",
                        type=int,
                        default=4,
                        help="每次检索返回的文档片段数量")
    parser.add_argument("--rag-embedding-model",
                        default=DEFAULT_EMBED_MODEL,
                        help="RAG 使用的 embedding 模型")
    parser.add_argument("--rag-chunk-size",
                        type=int,
                        default=900,
                        help="RAG 文档分块大小")
    parser.add_argument("--rag-chunk-overlap",
                        type=int,
                        default=150,
                        help="RAG 文档分块重叠长度")
    parser.add_argument("--embedding-base-url",
                        default=os.getenv("EMBEDDING_BASE_URL"),
                        help="Embedding 专用 API Base URL（默认用 OpenAI 官方）")
    parser.add_argument("--embedding-api-key",
                        default=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
                        help="Embedding 专用 API Key（默认读 OPENAI_API_KEY）")
    # 视觉 RAG 增强
    parser.add_argument("--enable-vision",
                        dest="enable_vision",
                        action="store_true",
                        default=True,
                        help="启用 RAG 视觉增强（自动识别图片并生成描述）")
    parser.add_argument("--no-vision",
                        dest="enable_vision",
                        action="store_false",
                        help="禁用 RAG 视觉增强")
    parser.add_argument("--use-cached-images",
                        dest="use_cached_images_only",
                        action="store_true",
                        default=False,
                        help="只使用已缓存的图片描述，跳过新图片处理（加速启动）")
    parser.add_argument("--warm-start-params",
                        default=None,
                        help="手动指定 warm-start 设计，格式如 lm=4.7,tm=0.4,ta=0.5,dg=0.3,hs=1.5,wslot=2.3,hslot=1.1,s=1.0,wa=2.0")
    parser.add_argument("--warm-start-csv",
                        default=None,
                        help="从上一轮优化结果 CSV 中读取最优设计作为起点")
    # 强化学习开关
    parser.add_argument("--enable-rl",
                        dest="enable_rl",
                        action="store_true",
                        default=True,
                        help="启用强化学习增强（经验回放、策略管理、奖励计算，默认开启）")
    parser.add_argument("--no-rl",
                        dest="enable_rl",
                        action="store_false",
                        help="禁用强化学习增强")
    parser.add_argument("--verbose-llm",
                        dest="verbose_llm",
                        action="store_true",
                        default=os.getenv("MCP_VERBOSE_LLM", "0").lower() in ("1", "true", "yes") or os.getenv("LLM_VERBOSE", "0").lower() in ("1", "true", "yes"),
                        help="LLM 详细输出模式：输出完整思考过程（目标复述、候选参数、物理机理、历史对比等），便于论文/调试。也可设 MCP_VERBOSE_LLM=1")
    
    # ========== 评论家Agent配置 ==========
    parser.add_argument("--enable-critic",
                        dest="enable_critic",
                        action="store_true",
                        default=True,
                        help="启用评论家Agent集群（提供稠密奖励，默认开启）")
    parser.add_argument("--no-critic",
                        dest="enable_critic",
                        action="store_false",
                        help="禁用评论家Agent集群")
    parser.add_argument("--critic-base-url",
                        default=None,
                        help="评论家API Base URL（默认与主LLM相同）")
    parser.add_argument("--critic-api-key",
                        default=None,
                        help="评论家API Key（默认与主LLM相同）")
    # 各评论家模型配置
    parser.add_argument("--critic-magnetic-model",
                        default=DEFAULT_CRITIC_MODELS.get("magnetic"),
                        help=f"磁路评论家模型（默认 {DEFAULT_CRITIC_MODELS.get('magnetic')}）")
    parser.add_argument("--critic-performance-model",
                        default=DEFAULT_CRITIC_MODELS.get("performance"),
                        help=f"性能评论家模型（默认 {DEFAULT_CRITIC_MODELS.get('performance')}）")
    parser.add_argument("--critic-constraint-model",
                        default=DEFAULT_CRITIC_MODELS.get("constraint"),
                        help=f"约束评论家模型（默认 {DEFAULT_CRITIC_MODELS.get('constraint')}）")
    parser.add_argument("--critic-magnitude-model",
                        default=DEFAULT_CRITIC_MODELS.get("magnitude"),
                        help=f"幅度评论家模型（默认 {DEFAULT_CRITIC_MODELS.get('magnitude')}）")
    # 评论家权重配置
    parser.add_argument("--critic-weights",
                        default=None,
                        help="评论家权重配置，格式: magnetic=0.35,performance=0.30,constraint=0.20,magnitude=0.15")
    # 启用的评论家
    parser.add_argument("--enabled-critics",
                        default="magnetic,performance,constraint,magnitude",
                        help="启用的评论家列表，逗号分隔（默认全部启用）")
    # 评论家经验库目录
    parser.add_argument("--critic-experience-dir",
                        default="critic_experience",
                        help="评论家经验库目录（默认 critic_experience）")
    
    # ★TD(n) 批处理模式配置
    parser.add_argument("--enable-batch-mode",
                        dest="enable_batch_mode",
                        action="store_true",
                        default=True,
                        help="启用 TD(n) 批处理模式（评论家每 n 轮生成一次规则，默认开启）")
    parser.add_argument("--no-batch-mode",
                        dest="enable_batch_mode",
                        action="store_false",
                        help="禁用 TD(n) 批处理模式")
    parser.add_argument("--batch-interval",
                        type=int,
                        default=3,
                        help="TD(n) 批处理间隔（默认 3 轮）")
    
    # ★策略库全部暴露（前期大步探索）
    parser.add_argument("--full-strategy-exposure",
                        dest="full_strategy_exposure",
                        action="store_true",
                        default=True,
                        help="把策略库全部暴露给 LLM（不筛选），适合前期大步探索（默认开启）")
    parser.add_argument("--no-full-strategy-exposure",
                        dest="full_strategy_exposure",
                        action="store_false",
                        help="关闭策略库全部暴露，只显示与当前参数相关的策略")
    
    # ★统一开关：离散探索提示 + 派生变量状态提示
    parser.add_argument("--enable-discrete-guidance",
                        dest="enable_discrete_guidance",
                        action="store_true",
                        default=True,
                        help="启用离散探索提示与派生变量状态提示（默认开启）")
    parser.add_argument("--no-discrete-guidance",
                        dest="enable_discrete_guidance",
                        action="store_false",
                        help="关闭离散探索提示与派生变量状态提示")
    
    # ★磁饱和软约束（拉格朗日乘子）：作为约束而非稠密奖励惩罚
    parser.add_argument("--saturation-as-constraint",
                        dest="saturation_as_constraint",
                        action="store_true",
                        default=False,
                        help="将磁饱和作为软约束（拉格朗日乘子）处理，而不是作为 dense reward 惩罚")
    parser.add_argument("--no-saturation-as-constraint",
                        dest="saturation_as_constraint",
                        action="store_false",
                        help="不启用磁饱和软约束（默认）")
    parser.add_argument("--sat-threshold",
                        type=float,
                        default=2.0,
                        help="磁饱和阈值 B_max（T，默认 2.0）")
    parser.add_argument("--sat-scale",
                        type=float,
                        default=0.2,
                        help="约束违背归一化尺度（T，默认 0.2；g=max(0,(B-th)/scale)，并截断到[0,1]）")
    parser.add_argument("--sat-lambda-lr",
                        type=float,
                        default=0.05,
                        help="拉格朗日乘子 λ 更新学习率（默认 0.05）")
    parser.add_argument("--sat-target",
                        type=float,
                        default=0.0,
                        help="允许的平均约束违背目标 g（默认 0.0）")
    
    # ★滑窗奖励平滑配置
    parser.add_argument("--enable-reward-smoothing",
                        dest="enable_reward_smoothing",
                        action="store_true",
                        default=True,
                        help="启用滑窗奖励平滑（对真实奖励做滑窗平均，默认开启）")
    parser.add_argument("--no-reward-smoothing",
                        dest="enable_reward_smoothing",
                        action="store_false",
                        help="禁用滑窗奖励平滑")
    parser.add_argument("--reward-window-size",
                        type=int,
                        default=3,
                        help="滑窗奖励平滑窗口大小（默认 3 轮）")
    
    # ★流式输出配置
    parser.add_argument("--stream",
                        dest="enable_stream",
                        action="store_true",
                        default=True,
                        help="启用流式输出（默认开启）")
    parser.add_argument("--no-stream",
                        dest="enable_stream",
                        action="store_false",
                        help="禁用流式输出（更稳定，但无法实时看到生成过程）")
    
    # ★经验迁移：从指定目录加载已有经验
    parser.add_argument("--load-experience-dir",
                        default=None,
                        help="从指定目录加载经验文件进行迁移学习（如 AgenticOPT_xxx_experience）")
    parser.add_argument("--transfer-mode",
                        choices=["raw", "distilled", "hybrid"],
                        default="raw",
                        help="经验迁移模式: raw=原始拷贝(同结构), distilled=蒸馏原则(跨结构), hybrid=两者皆用")
    
    # ★邮件通知：优化结束时发送邮件
    parser.add_argument("--notify-email",
                        default=None,
                        help="优化结束时发送通知邮件到指定地址（如 xxx@qq.com）")
    parser.add_argument("--smtp-server",
                        default="smtp.qq.com",
                        help="SMTP服务器地址（默认 smtp.qq.com）")
    parser.add_argument("--smtp-port",
                        type=int,
                        default=587,
                        help="SMTP端口（默认 587）")
    parser.add_argument("--smtp-password",
                        default=None,
                        help="SMTP授权码（也可通过环境变量 SMTP_PASSWORD 设置）")
    
    return parser.parse_args()


def _resolve_experience_source_dir(source_dir: str) -> str:
    """解析经验目录的实际路径（处理 round_XXXX 子目录）"""
    import re as _re
    
    has_root_files = any(
        os.path.exists(os.path.join(source_dir, f))
        for f in ["experience_buffer.json", "strategy_state.json", "expel_rules.json"]
    )
    
    if has_root_files:
        return source_dir
    
    round_dirs = []
    if os.path.isdir(source_dir):
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path) and _re.match(r"round_\d+", item):
                round_num = int(_re.search(r"\d+", item).group())
                round_dirs.append((round_num, item_path))
    
    if round_dirs:
        round_dirs.sort(key=lambda x: x[0], reverse=True)
        return round_dirs[0][1]
    
    return source_dir


def load_experience_from_dir(source_dir: str) -> bool:
    """
    从指定目录加载经验文件到当前工作目录，用于经验迁移。
    
    支持两种目录结构：
    1. 直接包含经验文件的目录
    2. 包含 round_XXXX 子目录的备份目录（自动选择最新的 round）
    
    Args:
        source_dir: 经验目录路径（如 AgenticOPT_xxx_experience）
    
    Returns:
        bool: 是否成功加载
    """
    import shutil
    import re
    
    if not os.path.isdir(source_dir):
        logger.error(f"经验目录不存在: {source_dir}")
        return False
    
    # 需要复制的经验文件
    experience_files = [
        "experience_buffer.json",    # 经验回放缓冲区
        "strategy_state.json",       # 策略管理器状态
        "expel_rules.json",          # ExpeL 对比学习规则
        "memory_stream.jsonl",       # Reflexion 反思记录
    ]
    
    # 需要复制的目录
    experience_dirs = [
        "critic_experience",         # 评论家集群经验
    ]
    
    # 检查是否需要从 round_XXXX 子目录加载
    actual_source_dir = source_dir
    
    # 检查根目录是否有经验文件
    has_root_files = any(
        os.path.exists(os.path.join(source_dir, f)) for f in experience_files
    )
    
    if not has_root_files:
        # 查找最新的 round_XXXX 子目录
        round_dirs = []
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path) and re.match(r"round_\d+", item):
                round_num = int(re.search(r"\d+", item).group())
                round_dirs.append((round_num, item_path))
        
        if round_dirs:
            # 选择最新的 round（编号最大）
            round_dirs.sort(key=lambda x: x[0], reverse=True)
            actual_source_dir = round_dirs[0][1]
            logger.info(f"[Transfer] 从最新轮次加载: {os.path.basename(actual_source_dir)}")
        else:
            logger.warning(f"[Transfer] 目录中没有经验文件或 round 子目录")
            return False
    
    copied_files = []
    copied_dirs = []
    
    # 复制文件
    for filename in experience_files:
        src_path = os.path.join(actual_source_dir, filename)
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, filename)
                copied_files.append(filename)
                logger.info(f"[Transfer] 已加载: {filename}")
            except Exception as e:
                logger.warning(f"[Transfer] 复制 {filename} 失败: {e}")
    
    # 复制目录
    for dirname in experience_dirs:
        src_path = os.path.join(actual_source_dir, dirname)
        if os.path.isdir(src_path):
            try:
                # 如果目标目录存在，先删除
                if os.path.exists(dirname):
                    shutil.rmtree(dirname)
                shutil.copytree(src_path, dirname)
                copied_dirs.append(dirname)
                logger.info(f"[Transfer] 已加载目录: {dirname}/")
            except Exception as e:
                logger.warning(f"[Transfer] 复制目录 {dirname} 失败: {e}")
    
    if copied_files or copied_dirs:
        logger.info(f"[Transfer] ✅ 经验迁移完成: {len(copied_files)} 个文件, {len(copied_dirs)} 个目录")
        logger.info(f"[Transfer] 来源: {actual_source_dir}")
        return True
    else:
        logger.warning(f"[Transfer] ⚠️ 未找到可迁移的经验文件")
        return False


def resolve_llm_config(args: argparse.Namespace) -> LLMConfig:
    mode = (args.mode or "cloud").lower()
    if mode not in {"cloud", "local"}:
        mode = "cloud"

    base_url = args.base_url
    if not base_url:
        base_url = (DEFAULT_CLOUD_BASE_URL if mode == "cloud" else
                    DEFAULT_LOCAL_BASE_URL)

    model = args.model
    if not model:
        model = DEFAULT_CLOUD_MODEL if mode == "cloud" else DEFAULT_LOCAL_MODEL

    if mode == "cloud":
        api_key = (args.api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
                   or os.getenv("OPENROUTER_API_KEY") or os.getenv("DASHSCOPE_API_KEY"))
        if not api_key:
            raise RuntimeError(
                "云端模式需要 API Key：请设置 OPENAI_API_KEY/OPENROUTER_API_KEY/LLM_API_KEY，或通过 --api-key 显式传入。")
    else:
        api_key = args.api_key or os.getenv("LOCAL_LLM_API_KEY")

    return LLMConfig(mode=mode, base_url=base_url, model=model, api_key=api_key)


async def main():
    args = parse_args()
    if args.verbose_llm:
        os.environ["MCP_VERBOSE_LLM"] = "1"
    config = resolve_llm_config(args)
    server_script = args.server_script
    auto_iterations = args.auto_iterations or DEFAULT_AUTO_ITERATIONS
    min_iterations = args.min_iterations
    convergence_window = args.convergence_window
    warm_start_design = (parse_manual_warm_start(args.warm_start_params) or
                         load_warm_start_design(args.warm_start_csv))
    rag_engine = None
    
    # ★经验迁移：在 RL 组件初始化前加载经验
    experience_transferred = False
    transfer_mode = getattr(args, 'transfer_mode', 'raw')
    distilled_principles = None
    if args.load_experience_dir:
        logger.info("=" * 60)
        logger.info(f"[Transfer] 🔄 开始经验迁移... (模式: {transfer_mode})")
        
        if transfer_mode in ("raw", "hybrid"):
            experience_transferred = load_experience_from_dir(args.load_experience_dir)
        
        if transfer_mode in ("distilled", "hybrid"):
            from mymcp.meta_learning import distill_experience_for_transfer, DistilledPrinciples
            
            # 确定源经验的实际目录（可能需要解析 round_XXXX），统一用绝对路径
            actual_source = os.path.abspath(_resolve_experience_source_dir(args.load_experience_dir))
            
            # 先检查是否已有蒸馏缓存（路径统一 normalize 后比较）
            cached = DistilledPrinciples.load("distilled_principles.json")
            cache_hit = (cached is not None and
                         os.path.normpath(cached.source_dir) == os.path.normpath(actual_source))
            if cache_hit:
                logger.info("[Transfer] 📂 使用已缓存的蒸馏原则")
                distilled_principles = cached
            else:
                logger.info(f"[Transfer] 🔄 缓存未命中，正在调用 LLM 蒸馏...")
                if cached:
                    logger.info(f"[Transfer]   缓存路径: {cached.source_dir}")
                    logger.info(f"[Transfer]   实际路径: {actual_source}")
                distilled_principles = distill_experience_for_transfer(
                    source_dir=actual_source,
                    llm_api_key=config.api_key,
                    llm_base_url=config.base_url,
                    llm_model=config.model,
                    save_path="distilled_principles.json",
                )
            
            if distilled_principles:
                experience_transferred = True
                logger.info("[Transfer] ✅ 蒸馏原则就绪")
            else:
                logger.warning("[Transfer] ⚠️ 蒸馏失败，回退到 raw 模式")
                if transfer_mode == "distilled":
                    experience_transferred = load_experience_from_dir(args.load_experience_dir)
        
        logger.info("=" * 60)

    print(sys.executable)

    if not os.path.exists(server_script):
        raise FileNotFoundError(f"无法找到 MCP 服务端脚本：{server_script}")

    logger.info("LLM 配置：mode=%s base_url=%s model=%s", config.mode,
                config.base_url, config.model)
    logger.info("优化配置：max={} min={} convergence_window={}",
                auto_iterations, min_iterations, convergence_window)
    logger.info("功能开关：RAG={} | RL={} | Vision={} | Critic={}",
                "✅" if args.enable_rag else "❌",
                "✅" if args.enable_rl else "❌",
                "✅" if args.enable_vision else "❌",
                "✅" if args.enable_critic else "❌")
    logger.info("LLM 输出模式：%s（--verbose-llm 或 MCP_VERBOSE_LLM=1 可启用详细模式，含完整思考过程）", "详细" if args.verbose_llm else "简洁")
    
    # 显示仿真配置
    wire_diameter = os.getenv("WIRE_DIAMETER_MM", "0.05")
    n1_floor = os.getenv("TURNS_FLOOR_MODE", "0").lower() in ("1", "true", "yes", "on")
    logger.info("仿真配置：线径={}mm | n1取整={} | n2取整=向下取整 | 结构={}",
                wire_diameter,
                "向下取整" if n1_floor else "四舍五入",
                "E-core" if os.getenv("USE_E_ONLY_SIMULATION") else "EI-core")
    if not config.api_key:
        logger.warning("当前未提供 LLM API Key，将以无鉴权方式访问本地服务。")

    if warm_start_design:
        logger.info(f"已加载 warm-start 设计: {warm_start_design}")
    elif args.warm_start_params:
        logger.warning("手动 warm-start 参数未解析出有效设计，默认从头优化。")
    elif args.warm_start_csv:
        logger.warning(f"未能从 {args.warm_start_csv} 读取有效的 warm-start 设计，默认从头优化。")

    adapter = MCPOpenAIAdapter()
    
    # ========== 初始化评论家集群 ==========
    critic_ensemble = None
    if args.enable_critic:
        # 解析评论家模型配置
        critic_models = {
            "magnetic": args.critic_magnetic_model,
            "performance": args.critic_performance_model,
            "constraint": args.critic_constraint_model,
            "magnitude": args.critic_magnitude_model,
        }
        
        # 解析评论家权重
        critic_weights = None
        if args.critic_weights:
            try:
                critic_weights = {}
                for part in args.critic_weights.split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        critic_weights[k.strip()] = float(v.strip())
            except Exception as e:
                logger.warning(f"解析评论家权重失败: {e}，使用默认权重")
        
        # 解析启用的评论家
        enabled_critics = [c.strip() for c in args.enabled_critics.split(",") if c.strip()]
        
        # 评论家API配置（默认与主LLM相同）
        critic_base_url = args.critic_base_url or config.base_url
        critic_api_key = args.critic_api_key or config.api_key
        
        try:
            critic_ensemble = CriticEnsemble(
                base_url=critic_base_url,
                api_key=critic_api_key,
                models=critic_models,
                weights=critic_weights,
                experience_dir=args.critic_experience_dir,
                enabled_critics=enabled_critics,
                # ★批处理模式配置
                enable_batch_mode=args.enable_batch_mode,
                batch_interval=args.batch_interval
            )
            logger.info("评论家集群已初始化，启用: %s", enabled_critics)
            logger.info("评论家模型配置: %s", critic_models)
            if args.enable_batch_mode:
                logger.info("TD(%d) 批处理模式已启用", args.batch_interval)
        except Exception as e:
            logger.error(f"评论家集群初始化失败: {e}")
            critic_ensemble = None
    
    # ========== 初始化 Actor-Critic 系统 ==========
    actor_critic_system = None
    if args.enable_critic:
        try:
            # 状态价值函数
            value_function = StateValueFunction(
                storage_path="state_value_history.json",
                k_neighbors=5,
                distance_threshold=0.3
            )
            
            # Actor-Critic 系统（整合 V(s) 和评论家集群）
            actor_critic_system = ActorCriticSystem(
                critic_ensemble=critic_ensemble,
                value_function=value_function,
                alpha_dense=0.3,  # 稠密奖励权重
                beta_real=0.7    # 真实奖励权重
            )
            logger.info("Actor-Critic 系统已初始化 | α=0.3, β=0.7")
            logger.info("状态价值函数 V(s): k=5 近邻估计")
        except Exception as e:
            logger.error(f"Actor-Critic 系统初始化失败: {e}")
            actor_critic_system = None
    
    if args.enable_rag:
        # 若未单独指定 embedding 的 base_url/api_key，则默认继承主 LLM 配置（更符合 OpenRouter 一体化用法）
        embedding_base_url = args.embedding_base_url or config.base_url
        embedding_api_key = args.embedding_api_key or config.api_key
        rag_config = RAGConfig(doc_dir=args.rag_dir,
                               cache_path=args.rag_cache,
                               base_url=config.base_url,
                               api_key=config.api_key,
                               embedding_model=args.rag_embedding_model,
                               embedding_base_url=embedding_base_url,
                               embedding_api_key=embedding_api_key,
                               chunk_size=args.rag_chunk_size,
                               chunk_overlap=args.rag_chunk_overlap,
                               top_k=args.rag_top_k,
                               enable_vision=args.enable_vision,
                               use_cached_images_only=args.use_cached_images_only)
        # 视觉模型默认使用主 LLM（如 gpt-5.1）
        rag_engine = RAGEngine(rag_config, vision_model=config.model)
        try:
            await rag_engine.prepare()
            logger.info("RAG 已加载，来源目录：%s", args.rag_dir)
        except Exception as exc:
            logger.error(f"RAG 初始化失败：{exc}")
            rag_engine = None

    # 流式输出配置：命令行参数优先，本地部署自动关闭
    use_stream = args.enable_stream
    if "localhost" in config.base_url or "127.0.0.1" in config.base_url:
        use_stream = False
        logger.info("检测到本地部署模型，已关闭流式输出")
    elif not args.enable_stream:
        logger.info("流式输出已手动关闭（--no-stream）")
    
    client = MCPClient(api_key=config.api_key,
                       base_url=config.base_url,
                       adapter=adapter,
                       model=config.model,
                       stream=use_stream,
                       rag_engine=rag_engine,
                       critic_ensemble=critic_ensemble,
                       actor_critic_system=actor_critic_system)
    
    # ★策略库全部暴露开关（前期大步探索）
    client._full_strategy_exposure = args.full_strategy_exposure
    if args.full_strategy_exposure:
        logger.info("策略库全部暴露模式已启用（适合前期大步探索）")
    else:
        logger.info("策略库筛选模式（只显示相关策略）")
    
    # ★统一开关：离散探索提示 + 派生变量状态提示
    client._enable_discrete_guidance = args.enable_discrete_guidance
    if args.enable_discrete_guidance:
        logger.info("离散探索提示/派生变量提示已启用（默认）")
    else:
        logger.info("离散探索提示/派生变量提示已关闭")
    
    # ★经验迁移标记（用于 CSV 文件名和提示词构建）
    client._experience_transferred = experience_transferred
    client._transfer_mode = transfer_mode if experience_transferred else None
    client._distilled_principles = distilled_principles
    if experience_transferred:
        mode_label = {"raw": "原始拷贝", "distilled": "蒸馏原则", "hybrid": "混合"}
        logger.info(f"经验迁移模式已启用 ({mode_label.get(transfer_mode, transfer_mode)})，CSV 文件名将包含 Transfer 标记")
    
    # ★邮件通知配置
    if args.notify_email:
        client._notify_email = args.notify_email
        client._smtp_server = args.smtp_server
        client._smtp_port = args.smtp_port
        client._smtp_password = args.smtp_password or os.getenv("SMTP_PASSWORD")
        if client._smtp_password:
            logger.info(f"邮件通知已启用，结束时将发送到: {args.notify_email}")
        else:
            logger.warning(f"邮件通知已配置，但未设置 SMTP 授权码（--smtp-password 或 SMTP_PASSWORD 环境变量）")
    
    # ★磁饱和软约束（拉格朗日乘子）开关
    client._saturation_as_constraint = args.saturation_as_constraint
    client._sat_threshold_t = float(args.sat_threshold)
    client._sat_scale_t = float(args.sat_scale)
    client._sat_lambda_lr = float(args.sat_lambda_lr)
    client._sat_target = float(args.sat_target)
    if args.saturation_as_constraint:
        logger.info(f"磁饱和软约束已启用：B_th={client._sat_threshold_t:.3f}T scale={client._sat_scale_t:.3f}T lr={client._sat_lambda_lr:.3f} target={client._sat_target:.3f}")
        # 关键：将磁路评论家从 dense_reward 聚合里排除（仍运行并给建议），避免“双重惩罚”
        if critic_ensemble is not None and hasattr(critic_ensemble, "exclude_from_dense_reward"):
            critic_ensemble.exclude_from_dense_reward.add("magnetic")
            logger.info("已将 magnetic 评论家从 dense_reward 聚合中排除（仍保留其建议/评分）")

    try:
        # 连接本地 MCP Server（会自动以 stdio 模式启动 server.py）
        await client.connect_to_mcp_server_stdio(
            server_script_path=server_script)
    except Exception as exc:
        logger.error(f"连接 MCP 服务端失败：{exc}")
        await client.exit_stack.aclose()
        return

    logger.warning("MCP 客户端已连接，输入 quit/exit 可退出。")
    logger.warning("特殊命令：feedback:<内容> 添加反馈 | status 查看RL状态 | clear 清空经验")
    logger.warning("💡 优化运行时，可往 feedback_input.txt 写入反馈，下轮自动读取")

    try:
        while True:
            prompt = input(
                "我是赵唯至开发的 Maxwell MCP 超级助手，请输入你的需求：")
            prompt_lower = prompt.lower().strip()
            
            if prompt_lower in {"quit", "exit"}:
                logger.warning("再见!")
                break
            
            # ========== 人类反馈命令 ==========
            if prompt_lower.startswith("feedback:"):
                feedback_text = prompt[9:].strip()
                if feedback_text:
                    # 解析反馈类型和优先级
                    fb_type = "suggestion"
                    priority = 2
                    if feedback_text.startswith("!"):
                        priority = 4  # 紧急
                        feedback_text = feedback_text[1:].strip()
                    if feedback_text.startswith("[warning]"):
                        fb_type = "warning"
                        feedback_text = feedback_text[9:].strip()
                    elif feedback_text.startswith("[correction]"):
                        fb_type = "correction"
                        feedback_text = feedback_text[12:].strip()
                    
                    if client.add_feedback(feedback_text, feedback_type=fb_type, priority=priority):
                        logger.warning(f"✅ 反馈已添加: [{fb_type}] {feedback_text}")
                    else:
                        logger.error("❌ 反馈添加失败")
                else:
                    logger.warning("请提供反馈内容，格式: feedback:<内容>")
                continue
            
            # ========== 查看 RL 状态 ==========
            if prompt_lower == "status":
                status = client.get_rl_status()
                logger.warning(f"\n===== RL 增强状态 =====")
                logger.warning(f"反馈数量: {status['feedback_count']}")
                logger.warning(f"经验数量: {status['experience_count']}")
                logger.warning(f"成功率: {status['success_rate']*100:.1f}%")
                logger.warning(f"探索率 ε: {status['epsilon']:.3f}")
                if status['best_fitness'] is not None:
                    logger.warning(f"历史最佳 fitness: {status['best_fitness']:.6e}")
                logger.warning(f"=========================\n")
                continue
            
            # ========== 清空经验/反馈 ==========
            if prompt_lower == "clear":
                client.clear_experience()
                logger.warning("经验缓冲已清空")
                continue
            if prompt_lower == "clear feedback":
                client.clear_feedback()
                logger.warning("反馈已清空")
                continue

            logger.warning("MCP 智能体正在运行中……")
            try:
                if should_use_auto(prompt):
                    system_prompt = extract_system_prompt(prompt)
                    await client.optimize(
                        system_prompt=system_prompt,
                        max_iterations=auto_iterations,
                        min_iterations=min_iterations,
                        convergence_window=convergence_window,
                        warm_start=warm_start_design,
                        enable_rl=args.enable_rl,
                        # ★TD(n) 批处理模式和滑窗奖励平滑
                        enable_batch_mode=args.enable_batch_mode,
                        batch_interval=args.batch_interval,
                        enable_reward_smoothing=args.enable_reward_smoothing,
                        reward_window_size=args.reward_window_size)
                else:
                    response = await client.process_query(prompt)
                    if response:
                        print(response)
                logger.warning("MCP 智能体执行完成")
            except Exception as exc:
                logger.error(f"MCP 智能体运行错误: {exc}")
                # ★ 发送异常邮件通知
                if client._notify_email and client._smtp_password:
                    try:
                        client._send_notification_email(
                            "MCP ERROR - Optimization Exception",
                            f"Optimization stopped due to exception.\n\nError: {exc}"
                        )
                    except Exception:
                        pass  # 邮件发送失败不影响主流程
    except KeyboardInterrupt:
        logger.warning("再见!")
        # ★ 用户中断也发送通知
        if client._notify_email and client._smtp_password:
            try:
                client._send_notification_email(
                    "MCP INFO - User Interrupted",
                    "Optimization was interrupted by user (Ctrl+C)."
                )
            except Exception:
                pass
    finally:
        # 确保断开与 MCP Server 的连接并回收资源
        await client.exit_stack.aclose()


if __name__ == "__main__":
    asyncio.run(main())

