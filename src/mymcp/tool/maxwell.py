import asyncio
import io
import json
import os
import random
import re
import sys
import time
import logging
import warnings
import subprocess
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ===================== 彻底禁用 PyAEDT 控制台输出 =====================
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _wants_aedt_gui() -> bool:
    # 默认关闭 GUI；仅在明确设置 AEDT_NON_GRAPHICAL=0 时打开 GUI
    return (not _env_bool("AEDT_NON_GRAPHICAL", True)) or (not _env_bool("AEDT_CLOSE_ON_EXIT", True))


# 注意：这里不要再强制覆盖用户的 GUI 配置。
# - 默认：关闭 GUI
# - 若用户设 AEDT_NON_GRAPHICAL=0，则同步设置 PYAEDT_NON_GRAPHICAL=0
if _wants_aedt_gui():
    os.environ["PYAEDT_NON_GRAPHICAL"] = "0"
else:
    os.environ.setdefault("PYAEDT_NON_GRAPHICAL", "1")
os.environ["PYAEDT_CONSOLE_LOG"] = "0"
os.environ["PYAEDT_DESKTOP_LOGS"] = "0"
os.environ["PYAEDT_DESKTOP_LOG"] = "0"
os.environ["PYAEDT_LOGS"] = "off"
os.environ["PYAEDT_LOG_LEVEL"] = "ERROR"
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

warnings.filterwarnings("ignore", module="pyaedt")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

for logger_name in ["pyaedt", "pyaedt.desktop", "pyaedt.generic", "Global"]:
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.CRITICAL)
    _logger.propagate = False
    _logger.handlers = []
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(logging.CRITICAL)
    _logger.addHandler(_handler)

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)


def _kill_aedt_processes():
    """强制杀掉所有 AEDT 相关进程，防止端口占用"""
    process_names = [
        "ansysedt.exe",
        "ansysedtsv.exe",
        "desktopMessageServer.exe",
        "ANSYSMessagesService.exe",
        "AnsoftMonitorSvc.exe",
    ]
    
    for proc_name in process_names:
        for attempt in range(3):
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", proc_name],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0 or b"not found" in result.stderr.lower():
                    break
            except Exception:
                pass
            time.sleep(0.5)
    
    time.sleep(3)

_import_stdout = io.StringIO()
_import_stderr = io.StringIO()
with redirect_stdout(_import_stdout), redirect_stderr(_import_stderr):
    from maxwell_pyaedt_run import (ActuatorDesignVariables,
                                    evaluate_design_fitness,
                                    WA_FIXED)
from .constraints import _build_design_kwargs

import_logs = []

DEFAULT_PROJECT = "PyAEDT_Project"
DEFAULT_DESIGN = "Maxwell3DDesign_PyAEDT"
DEFAULT_SETUP = "Setup1"
# ★ 强制所有 Maxwell 输出放到统一目录，避免散落在项目根目录
MAXWELL_OUTPUT_BASE = "maxwell_outputs"
DEFAULT_OUTPUT_ROOT = os.path.join(MAXWELL_OUTPUT_BASE, "default")
WCOIL_FIXED = 0.05
WEIGHTS_FIXED: Tuple[float, float, float, float] = (0.5, 0.5, 4.0, 1.0)


async def run_maxwell_simulation(
        lm: float,
        tm: float,
        ta: float,
        dg: float,
        hs: float,
        wslot: float,
        hslot: float,
        s: float,
        tb_ratio: float,  # 自由变量，LLM 必须传入，范围 [1.6, 2.0]
        project_name: Optional[str] = None,
        design_name: Optional[str] = None,
        setup_name: Optional[str] = None,
        output_root: Optional[str] = None) -> str:
    """调用 Maxwell 仿真一次并返回 fitness 以及各项指标。"""
    loop = asyncio.get_running_loop()

    # 为可选参数填充默认值
    project_name = project_name or DEFAULT_PROJECT
    design_name = design_name or DEFAULT_DESIGN
    setup_name = setup_name or DEFAULT_SETUP
    
    # ★ 强制所有输出放到 maxwell_outputs 目录下，避免散落在项目根目录
    if output_root:
        # 提取 LLM 指定的子目录名（去掉 ./ 前缀）
        subdir = output_root.lstrip("./").lstrip(".\\")
        output_root = os.path.join(MAXWELL_OUTPUT_BASE, subdir)
    else:
        output_root = DEFAULT_OUTPUT_ROOT

    # 构建设计参数，tb_ratio 是自由变量
    design, design_kwargs = _build_design_kwargs(lm, tm, ta, dg, hs,
                                                 wslot, hslot, s, tb_ratio)
    design_kwargs["wcoil"] = WCOIL_FIXED

    weights: Tuple[float, float, float, float] = WEIGHTS_FIXED

    def _run_simulation() -> str:
        stderr_target = sys.stderr
        log_messages = import_logs[:]

        def log(msg: str):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            full_msg = f"[Maxwell Tool][{timestamp}] {msg}"
            log_messages.append(full_msg)
            print(full_msg, file=stderr_target, flush=True)

        py_stdout = io.StringIO()
        py_stderr = io.StringIO()
        result = None

        try:
            with redirect_stdout(py_stdout), redirect_stderr(py_stderr):
                os.makedirs(output_root, exist_ok=True)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                run_dir = os.path.join(
                    output_root,
                    f"mcp_eval_{timestamp}_{random.randint(0, 9999):04d}",
                )
                os.makedirs(run_dir, exist_ok=True)
                log(f"准备仿真，run_dir={run_dir}")

                design = ActuatorDesignVariables(**design_kwargs)
                log(f"设计参数: {design}")

                result = evaluate_design_fitness(
                    design,
                    weight_factors=weights,
                    project_name=project_name,
                    design_name=design_name,
                    setup_name=setup_name,
                    output_dir=run_dir,
                )
        finally:
            # 注意：不再在 finally 中强制杀进程，避免杀掉下一个仿真正在启动的 AEDT
            # PyAEDT 的 close_on_exit=True 会自然关闭 AEDT
            if _wants_aedt_gui():
                log("检测到 GUI/保留会话模式，保留 AEDT 进程。")
            else:
                log("仿真结束，AEDT 将自动关闭（close_on_exit=True）")

        _ = py_stdout.getvalue()
        _ = py_stderr.getvalue()

        if result is None:
            result = {"status": "error", "message": "仿真未能完成"}

        log(
            f"仿真完成，status={result.get('status')} fitness={result.get('fitness')}"
        )
        return json.dumps(
            {
                "result": result,
                "logs": log_messages,
            },
            ensure_ascii=False,
            indent=2,
        )

    # ===== 超时/重试配置 =====
    # MAXWELL_SIMULATION_TIMEOUT: 单次仿真最大等待秒数
    # MAXWELL_MAX_RETRIES: 超时后最大重试次数（不含首次）
    # MAXWELL_RETRY_SLEEP: 超时后重试前等待秒数
    SIMULATION_TIMEOUT = int(os.environ.get("MAXWELL_SIMULATION_TIMEOUT", "300"))
    MAX_RETRIES = int(os.environ.get("MAXWELL_MAX_RETRIES", "2"))
    RETRY_SLEEP = int(os.environ.get("MAXWELL_RETRY_SLEEP", "5"))

    async def force_cleanup_aedt() -> None:
        """异步强制清理 AEDT 相关进程（仅在非 GUI 模式下使用）。"""
        try:
            await loop.run_in_executor(None, _kill_aedt_processes)
        except Exception:
            # 清理失败不应阻塞错误返回
            pass

    for attempt in range(MAX_RETRIES + 1):
        try:
            print(
                f"[Maxwell Tool] 仿真开始（尝试 {attempt + 1}/{MAX_RETRIES + 1}，超时={SIMULATION_TIMEOUT}s）",
                file=sys.stderr,
                flush=True,
            )
            return await asyncio.wait_for(
                loop.run_in_executor(None, _run_simulation),
                timeout=SIMULATION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            print(
                f"[Maxwell Tool] ⚠️ 仿真超时（{SIMULATION_TIMEOUT}s）",
                file=sys.stderr,
                flush=True,
            )
            # 非 GUI 模式下清理 AEDT，避免卡死占用端口/句柄
            if not _wants_aedt_gui():
                print(
                    "[Maxwell Tool] 🔄 正在清理 AEDT 进程...",
                    file=sys.stderr,
                    flush=True,
                )
                await force_cleanup_aedt()
                print(
                    "[Maxwell Tool] ✅ AEDT 进程已清理",
                    file=sys.stderr,
                    flush=True,
                )

            if attempt < MAX_RETRIES:
                print(
                    f"[Maxwell Tool] 等待 {RETRY_SLEEP}s 后重试...",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(RETRY_SLEEP)
                continue

            error_payload_with_logs = {
                "result": {
                    "status": "error",
                    "message": f"仿真超时（{SIMULATION_TIMEOUT}s），已重试 {MAX_RETRIES} 次仍失败",
                },
                "logs": [f"仿真超时（{SIMULATION_TIMEOUT}s），已清理 AEDT 进程并放弃重试"],
            }
            return json.dumps(error_payload_with_logs, ensure_ascii=False, indent=2)
        except Exception as exc:
            error_payload_with_logs = {
                "result": {"status": "error", "message": str(exc)},
                "logs": [f"仿真过程异常：{exc}"],
            }
            print(
                f"[Maxwell Tool] 仿真过程异常：{exc}",
                file=sys.stderr,
                flush=True,
            )
            return json.dumps(error_payload_with_logs, ensure_ascii=False, indent=2)
