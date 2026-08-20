"""
main_robot.py - 机器人腿部结构优化入口
=======================================
与 main_mcp.py（直线电机优化）完全对称，只切换：
  1. MCP Server → robot_server.py
  2. 系统提示词 → 机器人腿优化场景
  3. 参数名 / CSV 列名 → 机器人腿设计变量
  4. 触发关键词 → robot 相关

所有 RL / RAG / Critic / Memory / ExpeL / Reflexion 组件**原封不动**复用。
"""
import argparse
import asyncio
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional


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

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from loguru import logger

from mymcp.client import MCPClient
from mymcp.mcp_adapter import MCPOpenAIAdapter
from mymcp.rag import RAGConfig, RAGEngine
from mymcp.critic import CriticEnsemble, DEFAULT_CRITIC_MODELS
from mymcp.value_function import StateValueFunction, ActorCriticSystem

os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/robot_{time}.log",
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
DEFAULT_LITER_DIR = os.path.join(BASE_DIR, "liter_robot")
DEFAULT_LOG_DIR = os.path.join(_HERE, "logs")
DEFAULT_RAG_DIR = DEFAULT_LITER_DIR
DEFAULT_RAG_CACHE = None
DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
# Local server lives in case_robot_leg/server.py
SERVER_SCRIPT = os.path.join(_HERE, "server.py")

AUTO_KEYWORDS = ["必须遵循", "多轮", "run_robot_simulation", "validate_robot_design"]
DEFAULT_AUTO_ITERATIONS = 200

# 机器人腿优化的系统提示词
ROBOT_SYSTEM_PROMPT = """你是一个机器人腿部结构优化专家。你的任务是通过迭代调参，最大化爬虫微型机器人在 1 秒仿真时间内的 X 方向净位移，同时保持 Y 方向振幅尽可能小（行走稳定性）。

【设计变量与范围】
| 变量 | 含义 | 单位 | 下界 | 上界 |
|------|------|------|------|------|
| m | 前腿 BF 长度 | 米 | 0 | 0.006 |
| n | 后腿 GD 长度 | 米 | 0 | 0.005 |
| alpha | 后脚与地面夹角 | 度 | 80 | 130 |
| beta | 前脚与地面夹角 | 度 | 50 | 120 |
| DIST_BETTERY | 电池距离 | 米 | 0 | 0.009 |

【物理约束（7 类）】
1. 步幅极限：前后腿连接不超过机身跨度 L_AB
2. 机身触地：机身底角 C 点 Y 坐标 > 0
3. 后腿构型：防止后腿反向折叠
4. 穿模干涉：关键点(G/D/F)不得侵入机身矩形
5. 高度限制：机身最高点 ≤ 15mm
6. 前腿关节角 D12 ≤ 90°
7. DF 跨度 ≤ 13mm

【优化目标 fitness（越大越好）】
fitness = Net_Displacement_X（X方向净位移，单位 mm）
若 Displacement_Range_Y > 10mm，fitness 强制为 0（视为异常）。
目标：最大化机器人在 1 秒内的 X 方向行走距离。

【仿真流程】
参数 → SolidWorks 3D建模 → Adams 动力学仿真(1秒) → 读取位移/速度结果

【工具使用规范】
每轮必须遵循以下顺序：
1. validate_robot_design 检查参数是否满足 7 类约束
2. 若 validate 返回 status=ok，调用 run_robot_simulation 执行仿真
3. 分析 fitness 和各指标，规划下一轮参数调整

【参数调整指导】
- m, n 控制前后腿长度，直接影响步幅和稳定性
- alpha 控制后腿张角，影响后脚着地点和机身倾角
- beta 控制前腿张角，影响前脚着地点
- alpha 和 beta 的组合决定了机身俯仰角和重心高度
- DIST_BETTERY 影响重心位置和行走对称性
- 注意：参数间存在耦合，调整一个可能导致约束违规

请在每轮输出中包含：批评当前方案问题、关键数值变化、下一步调整方案。"""


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
    has_tool = ("run_robot_simulation" in lower_prompt or
                "validate_robot_design" in lower_prompt)
    has_keyword = any(kw in prompt for kw in AUTO_KEYWORDS)
    return has_tool and has_keyword


def extract_system_prompt(prompt: str) -> str:
    if prompt.lower().startswith("auto:"):
        return prompt.split(":", 1)[1].strip()
    return prompt


def parse_manual_warm_start(params: str) -> Optional[Dict[str, float]]:
    if not params:
        return None
    base_keys = {"m", "n", "alpha", "beta", "DIST_BETTERY"}
    result: Dict[str, float] = {}
    try:
        parts = [p.strip() for p in params.replace(";", ",").split(",") if p.strip()]
        for part in parts:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            if k in base_keys:
                result[k] = float(v)
        if base_keys.issubset(result.keys()):
            return result
    except Exception as exc:
        logger.warning(f"解析 warm-start 参数失败: {exc}")
    return None


def load_warm_start_design(csv_path: str) -> Optional[dict]:
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
            fields = ["m", "n", "alpha", "beta", "DIST_BETTERY"]
            design = {}
            for key in fields:
                val = best_row.get(key)
                if val is None or val == "":
                    return None
                design[key] = float(val)
            return design
    except Exception as exc:
        logger.warning(f"读取 warm-start CSV 失败: {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot Leg MCP 优化助手")
    parser.add_argument("--mode", choices=["cloud", "local"],
                        default=os.getenv("LLM_MODE", "cloud"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--api-key",
                        default=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
                        or os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--auto-iterations", type=int, default=None)
    parser.add_argument("--min-iterations", type=int, default=50)
    parser.add_argument("--convergence-window", type=int, default=20)
    parser.add_argument("--server-script", default=SERVER_SCRIPT)
    # RAG
    parser.add_argument("--rag", dest="enable_rag", action="store_true", default=True)
    parser.add_argument("--no-rag", dest="enable_rag", action="store_false")
    parser.add_argument("--rag-dir", default=DEFAULT_RAG_DIR)
    parser.add_argument("--rag-cache", default=DEFAULT_RAG_CACHE)
    parser.add_argument("--rag-top-k", type=int, default=4)
    parser.add_argument("--rag-embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--rag-chunk-size", type=int, default=900)
    parser.add_argument("--rag-chunk-overlap", type=int, default=150)
    parser.add_argument("--embedding-base-url", default=os.getenv("EMBEDDING_BASE_URL"))
    parser.add_argument("--embedding-api-key",
                        default=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--enable-vision", dest="enable_vision", action="store_true", default=False)
    parser.add_argument("--no-vision", dest="enable_vision", action="store_false")
    parser.add_argument("--use-cached-images", dest="use_cached_images_only",
                        action="store_true", default=False)
    # Warm start
    parser.add_argument("--warm-start-params", default=None)
    parser.add_argument("--warm-start-csv", default=None)
    # RL
    parser.add_argument("--enable-rl", dest="enable_rl", action="store_true", default=True)
    parser.add_argument("--no-rl", dest="enable_rl", action="store_false")
    parser.add_argument("--single-var-exploit", dest="single_var_exploit", action="store_true", default=False,
                        help="利用模式下限制每轮只改1个参数")
    parser.add_argument("--no-single-var-exploit", dest="single_var_exploit", action="store_false",
                        help="利用模式下允许同时改多个参数 (默认)")
    parser.add_argument("--force-explore", dest="force_explore", action="store_true", default=False,
                        help="探索轮强制执行: 绕过 LLM, 直接用 RL 生成的参数调用 validate→simulate")
    parser.add_argument("--no-force-explore", dest="force_explore", action="store_false")
    parser.add_argument("--verbose-llm", dest="verbose_llm", action="store_true",
                        default=os.getenv("MCP_VERBOSE_LLM", "0").lower() in ("1", "true", "yes"))
    # Critic
    parser.add_argument("--enable-critic", dest="enable_critic", action="store_true", default=True)
    parser.add_argument("--no-critic", dest="enable_critic", action="store_false")
    parser.add_argument("--critic-base-url", default=None)
    parser.add_argument("--critic-api-key", default=None)
    parser.add_argument("--critic-kinematics-model",
                        default=DEFAULT_CRITIC_MODELS.get("kinematics"))
    parser.add_argument("--critic-robot-performance-model",
                        default=DEFAULT_CRITIC_MODELS.get("robot_performance"))
    parser.add_argument("--critic-robot-constraint-model",
                        default=DEFAULT_CRITIC_MODELS.get("robot_constraint"))
    parser.add_argument("--critic-magnitude-model",
                        default=DEFAULT_CRITIC_MODELS.get("magnitude"))
    parser.add_argument("--critic-weights", default=None)
    parser.add_argument("--enabled-critics",
                        default="kinematics,robot_performance,robot_constraint,magnitude")
    parser.add_argument("--critic-experience-dir", default="critic_experience")
    # Batch mode
    parser.add_argument("--enable-batch-mode", dest="enable_batch_mode",
                        action="store_true", default=True)
    parser.add_argument("--no-batch-mode", dest="enable_batch_mode", action="store_false")
    parser.add_argument("--batch-interval", type=int, default=3)
    # Strategy
    parser.add_argument("--full-strategy-exposure", dest="full_strategy_exposure",
                        action="store_true", default=True)
    parser.add_argument("--no-full-strategy-exposure", dest="full_strategy_exposure",
                        action="store_false")
    parser.add_argument("--enable-discrete-guidance", dest="enable_discrete_guidance",
                        action="store_true", default=False)
    parser.add_argument("--no-discrete-guidance", dest="enable_discrete_guidance",
                        action="store_false")
    # Reward smoothing
    parser.add_argument("--enable-reward-smoothing", dest="enable_reward_smoothing",
                        action="store_true", default=True)
    parser.add_argument("--no-reward-smoothing", dest="enable_reward_smoothing",
                        action="store_false")
    parser.add_argument("--reward-window-size", type=int, default=3)
    # Stream
    parser.add_argument("--stream", dest="enable_stream", action="store_true", default=True)
    parser.add_argument("--no-stream", dest="enable_stream", action="store_false")
    # Transfer
    parser.add_argument("--load-experience-dir", default=None)
    parser.add_argument("--transfer-mode", choices=["raw", "distilled", "hybrid"], default="raw")
    # Force amplitude
    parser.add_argument("--force-amplitude", type=float, default=None,
                        help="Adams 推力幅值 (N)，默认 0.02025 N (=1500*0.001*13.5E-03)")
    return parser.parse_args()


def resolve_llm_config(args: argparse.Namespace) -> LLMConfig:
    mode = (args.mode or "cloud").lower()
    if mode not in {"cloud", "local"}:
        mode = "cloud"
    base_url = args.base_url or (DEFAULT_CLOUD_BASE_URL if mode == "cloud" else DEFAULT_LOCAL_BASE_URL)
    model = args.model or (DEFAULT_CLOUD_MODEL if mode == "cloud" else DEFAULT_LOCAL_MODEL)
    if mode == "cloud":
        api_key = (args.api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
                   or os.getenv("OPENROUTER_API_KEY") or os.getenv("DASHSCOPE_API_KEY"))
        if not api_key:
            raise RuntimeError("云端模式需要 API Key")
    else:
        api_key = args.api_key or os.getenv("LOCAL_LLM_API_KEY")
    return LLMConfig(mode=mode, base_url=base_url, model=model, api_key=api_key)


async def main():
    args = parse_args()
    if args.verbose_llm:
        os.environ["MCP_VERBOSE_LLM"] = "1"
    if args.force_amplitude is not None:
        os.environ["ROBOT_FORCE_AMPLITUDE"] = str(args.force_amplitude)
        logger.info(f"Adams 推力幅值设为: {args.force_amplitude} N")
    config = resolve_llm_config(args)
    server_script = args.server_script
    auto_iterations = args.auto_iterations or DEFAULT_AUTO_ITERATIONS
    min_iterations = args.min_iterations
    convergence_window = args.convergence_window
    warm_start_design = (parse_manual_warm_start(args.warm_start_params) or
                         load_warm_start_design(args.warm_start_csv))
    if not warm_start_design:
        logger.info("未指定 warm-start，Agent 将自行选择初始参数")
    rag_engine = None

    if not os.path.exists(server_script):
        raise FileNotFoundError(f"无法找到 MCP 服务端脚本：{server_script}")

    logger.info("=" * 60)
    logger.info("  Robot Leg Optimization - Agentic Framework")
    logger.info("=" * 60)
    logger.info("LLM 配置：mode=%s base_url=%s model=%s", config.mode, config.base_url, config.model)
    logger.info("优化配置：max={} min={} convergence_window={}",
                auto_iterations, min_iterations, convergence_window)
    logger.info("功能开关：RAG={} | RL={} | Critic={}",
                "ON" if args.enable_rag else "OFF",
                "ON" if args.enable_rl else "OFF",
                "ON" if args.enable_critic else "OFF")

    adapter = MCPOpenAIAdapter()

    # Critic 集群
    critic_ensemble = None
    if args.enable_critic:
        critic_models = {
            "kinematics": args.critic_kinematics_model,
            "robot_performance": args.critic_robot_performance_model,
            "robot_constraint": args.critic_robot_constraint_model,
            "magnitude": args.critic_magnitude_model,
        }
        critic_weights = None
        if args.critic_weights:
            try:
                critic_weights = {}
                for part in args.critic_weights.split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        critic_weights[k.strip()] = float(v.strip())
            except Exception as e:
                logger.warning(f"解析评论家权重失败: {e}")
        enabled_critics = [c.strip() for c in args.enabled_critics.split(",") if c.strip()]
        critic_base_url = args.critic_base_url or config.base_url
        critic_api_key = args.critic_api_key or config.api_key
        try:
            critic_ensemble = CriticEnsemble(
                base_url=critic_base_url, api_key=critic_api_key,
                models=critic_models, weights=critic_weights,
                experience_dir=args.critic_experience_dir,
                enabled_critics=enabled_critics,
                enable_batch_mode=args.enable_batch_mode,
                batch_interval=args.batch_interval,
            )
            logger.info("评论家集群已初始化: %s", enabled_critics)
        except Exception as e:
            logger.error(f"评论家初始化失败: {e}")

    # Actor-Critic
    actor_critic_system = None
    if args.enable_critic and critic_ensemble:
        try:
            value_function = StateValueFunction(
                storage_path="state_value_history.json", k_neighbors=5, distance_threshold=0.3,
                domain="robot_leg")
            actor_critic_system = ActorCriticSystem(
                critic_ensemble=critic_ensemble, value_function=value_function,
                alpha_dense=0.3, beta_real=0.7)
            logger.info("Actor-Critic 系统已初始化")
        except Exception as e:
            logger.error(f"Actor-Critic 初始化失败: {e}")

    # RAG
    if args.enable_rag:
        args.rag_dir = args.rag_dir.strip()
        os.makedirs(args.rag_dir, exist_ok=True)
        rag_cache = args.rag_cache or os.path.join(args.rag_dir, ".rag_cache.json")
        embedding_base_url = args.embedding_base_url or config.base_url
        embedding_api_key = args.embedding_api_key or config.api_key
        rag_config = RAGConfig(
            doc_dir=args.rag_dir, cache_path=rag_cache,
            base_url=config.base_url, api_key=config.api_key,
            embedding_model=args.rag_embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            chunk_size=args.rag_chunk_size, chunk_overlap=args.rag_chunk_overlap,
            top_k=args.rag_top_k,
            enable_vision=args.enable_vision,
            use_cached_images_only=args.use_cached_images_only)
        rag_engine = RAGEngine(rag_config, vision_model=config.model)
        try:
            await rag_engine.prepare()
            logger.info("RAG 已加载: %s", args.rag_dir)
        except Exception as exc:
            logger.error(f"RAG 初始化失败：{exc}")
            rag_engine = None

    # Stream
    use_stream = args.enable_stream
    if "localhost" in config.base_url or "127.0.0.1" in config.base_url:
        use_stream = False

    client = MCPClient(
        api_key=config.api_key, base_url=config.base_url,
        adapter=adapter, model=config.model, stream=use_stream,
        rag_engine=rag_engine, critic_ensemble=critic_ensemble,
        actor_critic_system=actor_critic_system,
    )

    # 设置 domain 为 robot_leg（让 client 使用通用 CSV 和工具名映射）
    client._domain = "robot_leg"

    client._full_strategy_exposure = args.full_strategy_exposure
    client._enable_discrete_guidance = args.enable_discrete_guidance
    client._experience_transferred = False
    client._transfer_mode = None

    if warm_start_design:
        logger.info(f"已加载 warm-start: {warm_start_design}")

    try:
        await client.connect_to_mcp_server_stdio(server_script_path=server_script)
    except Exception as exc:
        logger.error(f"连接 MCP 服务端失败：{exc}")
        await client.exit_stack.aclose()
        return

    logger.warning("Robot Leg MCP 客户端已连接，输入 quit/exit 退出。")
    logger.warning("输入 auto:<提示词> 启动自动优化，或直接输入问题进行单轮交互。")

    try:
        while True:
            prompt = input("Robot Leg 优化助手> ")
            prompt_lower = prompt.lower().strip()

            if prompt_lower in {"quit", "exit"}:
                logger.warning("再见!")
                break

            if prompt_lower.startswith("feedback:"):
                fb = prompt[9:].strip()
                if fb and client.add_feedback(fb):
                    logger.warning(f"反馈已添加: {fb}")
                continue

            if prompt_lower == "status":
                status = client.get_rl_status()
                logger.warning(f"经验={status['experience_count']} 成功率={status['success_rate']*100:.1f}%")
                continue

            if prompt_lower == "clear":
                client.clear_experience()
                logger.warning("经验已清空")
                continue

            logger.warning("Agent 运行中...")
            try:
                if should_use_auto(prompt):
                    system_prompt = extract_system_prompt(prompt)
                    final_prompt = f"{ROBOT_SYSTEM_PROMPT}\n\n{system_prompt}"
                    await client.optimize(
                        system_prompt=final_prompt,
                        max_iterations=auto_iterations,
                        min_iterations=min_iterations,
                        convergence_window=convergence_window,
                        warm_start=warm_start_design,
                        enable_rl=args.enable_rl,
                        single_var_exploit=args.single_var_exploit,
                        enable_batch_mode=args.enable_batch_mode,
                        batch_interval=args.batch_interval,
                        enable_reward_smoothing=args.enable_reward_smoothing,
                        reward_window_size=args.reward_window_size,
                        force_explore=args.force_explore)
                else:
                    response = await client.process_query(
                        f"你是一个机器人腿部结构优化助手，可以回答关于机器人腿部设计的问题。\n\n用户: {prompt}")
                    if response:
                        print(response)
                logger.warning("Agent 执行完成")
            except Exception as exc:
                logger.error(f"Agent 运行错误: {exc}")
    except KeyboardInterrupt:
        logger.warning("用户中断")
    finally:
        await client.exit_stack.aclose()


if __name__ == "__main__":
    asyncio.run(main())
