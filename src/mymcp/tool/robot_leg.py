"""
robot_leg.py - 机器人腿部结构优化 MCP 工具
============================================
提供两个 MCP 工具，与 Maxwell 工具完全对称：
  validate_robot_design  ←→  validate_maxwell_design
  run_robot_simulation   ←→  run_maxwell_simulation

设计变量（5 个）：
  m            前腿长度 (单位：米, 范围 [0, 0.006])
  n            后腿长度 (单位：米, 范围 [0, 0.005])
  alpha        后脚与地面夹角 (单位：度, 范围 [80, 130])
  beta         前脚与地面夹角 (单位：度, 范围 [50, 120])
  DIST_BETTERY 电池距离 (单位：米, 范围 [0, 0.009])

仿真流程：写 input.csv → run.py(SolidWorks + Adams) → 读 final_res.csv
"""

import asyncio
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parents[3] / "package"

def _kill_sw_adams_global():
    """杀掉残留的 SolidWorks / Adams 进程"""
    try:
        import psutil
    except ImportError:
        return
    targets = {"SLDWORKS.exe", "sldworks.exe", "aview.exe",
               "adams_main.exe", "adamssolve.exe"}
    for p in psutil.process_iter(["name", "pid"]):
        try:
            if p.info["name"] in targets:
                print(f"[kill] {p.info['name']} PID={p.info['pid']}", flush=True)
                p.kill()
        except Exception:
            pass
DATA_DIR = PACKAGE_DIR / "data"
CODE_DIR = PACKAGE_DIR / "code"
RESULT_DIR = PACKAGE_DIR / "result"
INPUT_CSV = DATA_DIR / "input.csv"
PARAMS_CSV = DATA_DIR / "params.csv"
LIMITED_CSV = DATA_DIR / "limited.csv"
FINAL_RES_CSV = RESULT_DIR / "final_res.csv"
RUN_SCRIPT = CODE_DIR / "run.py"

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "m": (0.0, 0.006),
    "n": (0.0, 0.005),
    "alpha": (80.0, 130.0),
    "beta": (50.0, 120.0),
    "DIST_BETTERY": (0.0, 0.009),
}

PARAM_DECIMALS: Dict[str, int] = {
    "m": 5,              # 0.01mm = 1e-5 m → 5位小数
    "n": 5,
    "alpha": 1,          # 0.1°
    "beta": 1,
    "DIST_BETTERY": 5,
}


def snap_params(m: float, n: float, alpha: float, beta: float,
                DIST_BETTERY: float):
    """将参数截断到全局精度: 长度→0.01mm, 角度→0.1°"""
    return (
        round(m, PARAM_DECIMALS["m"]),
        round(n, PARAM_DECIMALS["n"]),
        round(alpha, PARAM_DECIMALS["alpha"]),
        round(beta, PARAM_DECIMALS["beta"]),
        round(DIST_BETTERY, PARAM_DECIMALS["DIST_BETTERY"]),
    )


BODY_BC = 7.26  # 与 run.py 一致
BODY_AC = 1.95
BODY_AG = 2.20
BODY_L = 7.26
BODY_H = 5.34
L_AB = math.sqrt(BODY_BC**2 + BODY_AC**2)
THETA_AB_BODY = math.atan2(-BODY_AC, BODY_BC)
MAX_HEIGHT = 15.0
MAX_B_DEG = 90.0
DF_LIMIT = 13.0
DELTA = 5.8  # 与 run.py 保持一致


def _check_point_in_body(P_x, P_y, C_x, C_y, theta_body, point_name) -> Tuple[bool, str]:
    v_x = P_x - C_x
    v_y = P_y - C_y
    p_x = v_x * math.cos(theta_body) + v_y * math.sin(theta_body)
    p_y = -v_x * math.sin(theta_body) + v_y * math.cos(theta_body)
    eps = 1e-4
    if (eps < p_x < BODY_L - eps) and (eps < p_y < BODY_H - eps):
        return False, f"碰撞干涉：点 {point_name} 侵入机身内部 (局部坐标: x={p_x:.2f}, y={p_y:.2f})"
    return True, f"点 {point_name} 安全 (局部坐标: x={p_x:.2f}, y={p_y:.2f})"


def _validate_design(m: float, n: float, alpha: float, beta: float,
                     DIST_BETTERY: float) -> Dict:
    """纯 Python 约束检查，不调用仿真软件。返回与 validate_maxwell_design 相同的 JSON 结构。"""
    errors: List[str] = []
    warnings: List[str] = []
    design = {"m": m, "n": n, "alpha": alpha, "beta": beta, "DIST_BETTERY": DIST_BETTERY}

    # 范围检查
    for name, (lo, hi) in PARAM_BOUNDS.items():
        val = design[name]
        if val < lo or val > hi:
            errors.append(f"{name}={val} 超出范围 [{lo}, {hi}]")

    if errors:
        return {"status": "constraint_violation", "errors": errors, "design": design, "derived": {}}

    m_mm = m * 1000.0
    n_mm = n * 1000.0

    beta_rad = math.radians(beta)
    alpha_rad = math.radians(alpha)

    D_x, D_y = 0.0, 0.0
    G_x = -n_mm * math.cos(alpha_rad)
    G_y = n_mm * math.sin(alpha_rad)
    A_x = G_x + BODY_AG * math.sin(alpha_rad)
    A_y = G_y + BODY_AG * math.cos(alpha_rad)
    B_y = m_mm * math.sin(beta_rad)
    dy = B_y - A_y

    # 限制1: 步幅极限
    if abs(dy) > L_AB:
        errors.append(f"步幅极限：|Δy|={abs(dy):.4f}mm > L_AB={L_AB:.4f}mm，腿长和角度组合超过机身跨度")

    if errors:
        return {"status": "constraint_violation", "errors": errors, "design": design, "derived": {}}

    dx = math.sqrt(L_AB**2 - dy**2)
    theta_AB_gnd = math.atan2(dy, dx)
    theta_body = theta_AB_gnd - THETA_AB_BODY
    cef_deg = math.degrees(theta_body)

    C_x = A_x + BODY_AC * math.sin(theta_body)
    C_y = A_y - BODY_AC * math.cos(theta_body)
    B_x = C_x + BODY_BC * math.cos(theta_body)
    F_x = B_x + m_mm * math.cos(beta_rad)

    Y_BR = C_y + BODY_L * math.sin(theta_body)
    Y_H_top = C_y + BODY_H * math.cos(theta_body)
    Y_TR = C_y + BODY_L * math.sin(theta_body) + BODY_H * math.cos(theta_body)
    H_max = max(C_y, Y_BR, Y_H_top, Y_TR) + DELTA

    # 限制2: 机身触地
    if C_y <= 0:
        errors.append(f"机身触地：C点高度 Y_C={C_y:.4f}mm ≤ 0")

    # 限制3: 后腿构型
    config_val = A_y * math.cos(theta_body) - A_x * math.sin(theta_body)
    if config_val <= 0:
        errors.append(f"构型奇异：后腿反向折叠 (判定值={config_val:.4f})")

    # 限制4: 穿模检测
    for px, py, pname in [(G_x, G_y, "G(后膝关节)"), (D_x, D_y, "D(后脚垫)"), (F_x, 0.0, "F(前脚垫)")]:
        ok, msg = _check_point_in_body(px, py, C_x, C_y, theta_body, pname)
        if not ok:
            errors.append(msg)

    # 限制5: 高度限制
    if H_max > MAX_HEIGHT:
        errors.append(f"高度超限：H_max={H_max:.4f}mm > {MAX_HEIGHT}mm")

    # 关节角
    a_rad = math.acos(max(-1.0, min(1.0, -math.cos(theta_body + alpha_rad))))
    b_rad = math.acos(max(-1.0, min(1.0, -math.cos(theta_body + beta_rad))))
    a_deg = math.degrees(a_rad)
    b_deg = math.degrees(b_rad)

    # 限制6: 前腿关节角
    if b_deg > MAX_B_DEG:
        errors.append(f"前腿关节角超限：D12={b_deg:.4f}° > {MAX_B_DEG}°")

    # 限制7: DF 跨度
    DF_span = abs(F_x)
    if DF_span > DF_LIMIT:
        errors.append(f"DF跨度超限：DF={DF_span:.4f}mm > {DF_LIMIT}mm")

    derived = {
        "D4_n": n, "D10_m": m,
        "D6_rear_angle": a_deg, "D12_front_angle": b_deg,
        "D2_body_pitch": cef_deg + 90.0,
        "C_y": C_y, "H_max": H_max, "DF_span": DF_span,
        "body_pitch_deg": cef_deg,
    }

    status = "ok" if not errors else "constraint_violation"
    return {"status": status, "errors": errors, "design": design, "derived": derived}


async def validate_robot_design(
    m: float,
    n: float,
    alpha: float,
    beta: float,
    DIST_BETTERY: float,
) -> str:
    """检查一组机器人腿部设计参数是否满足全部物理约束。

    参数范围：
    - m: 前腿长度 [0.0020, 0.0035] 米
    - n: 后腿长度 [0.0020, 0.0035] 米
    - alpha: 后脚与地面夹角 [91, 155] 度
    - beta: 前脚与地面夹角 [50, 89] 度
    - DIST_BETTERY: 电池距离 [0.0055, 0.0065] 米

    7 类约束：步幅极限、机身触地、后腿构型、穿模干涉(G/D/F)、高度限制、前腿关节角、DF跨度。
    """
    m, n, alpha, beta, DIST_BETTERY = snap_params(m, n, alpha, beta, DIST_BETTERY)
    result = _validate_design(m, n, alpha, beta, DIST_BETTERY)
    return json.dumps(result, ensure_ascii=False, indent=2)


def _write_input_csv(m: float, n: float, alpha: float, beta: float,
                     DIST_BETTERY: float) -> None:
    m, n, alpha, beta, DIST_BETTERY = snap_params(m, n, alpha, beta, DIST_BETTERY)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(INPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        for name, val in [("m", m), ("n", n), ("alpha", alpha),
                          ("beta", beta), ("DIST_BETTERY", DIST_BETTERY)]:
            writer.writerow([name, val])


def _read_final_results() -> Optional[Dict[str, float]]:
    if not FINAL_RES_CSV.exists():
        return None
    results = {}
    with open(FINAL_RES_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return None
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                try:
                    results[key] = float(row[1].strip())
                except ValueError:
                    results[key] = row[1].strip()
    return results if results else None


RANGE_Y_LIMIT = 2.0
RANGE_Y_FLY_LIMIT = 10.0    # Range_Y > 10mm → 物理性起飞，不重跑
FLY_NET_Y_THRESHOLD = 2.0   # Net_Displacement_Y > +2mm → 机器人飞上天，不重跑
MAX_RETRY_ON_RANGE_Y = 3
MAX_RETRY_ON_SIM_FAIL = 2   # simulation_failed / run.py exit!=0 重跑次数
FALL_NET_Y_THRESHOLD = -1.0  # Net_Displacement_Y < -1mm → 判定为摔倒
ADAMS_RAW_CSV = RESULT_DIR / "results.csv"


def _build_trajectory_diagnosis(params: dict) -> str:
    """读取 Adams 原始 results.csv 时序数据，生成 Y 轨迹诊断报告供 LLM 分析。"""
    if not ADAMS_RAW_CSV.exists():
        return ""
    try:
        times, ys, xs = [], [], []
        with open(ADAMS_RAW_CSV, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('A') and 'B' in line and 'C' in line:
                data_start = i + 1
                break
        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    xs.append(float(parts[0]))
                    times.append(float(parts[1]))
                    ys.append(float(parts[2]))
                except ValueError:
                    continue
        if len(times) < 10:
            return ""

        y_mm = [v * 1000 for v in ys]
        x_mm = [v * 1000 for v in xs]
        y0, y_final = y_mm[0], y_mm[-1]
        y_min, y_max = min(y_mm), max(y_mm)
        y_min_idx = y_mm.index(y_min)
        t_at_y_min = times[y_min_idx]
        net_y = y_final - y0
        net_x = x_mm[-1] - x_mm[0]

        n_pts = len(times)
        segments = [0, n_pts // 5, 2 * n_pts // 5, 3 * n_pts // 5, 4 * n_pts // 5, n_pts - 1]
        timeline = []
        for idx in segments:
            if idx < n_pts:
                timeline.append(f"  t={times[idx]:.3f}s: X={x_mm[idx]:.2f}mm, Y={y_mm[idx]:.2f}mm")

        report = [
            "=== 仿真轨迹诊断（供分析用）===",
            f"参数: m={params.get('m',0)*1000:.2f}mm, n={params.get('n',0)*1000:.2f}mm, "
            f"alpha={params.get('alpha',0):.1f}°, beta={params.get('beta',0):.1f}°, "
            f"DIST_BETTERY={params.get('DIST_BETTERY',0)*1000:.2f}mm",
            f"Y初始={y0:.2f}mm, Y最低={y_min:.2f}mm(t={t_at_y_min:.3f}s), "
            f"Y终值={y_final:.2f}mm, Y净位移={net_y:.2f}mm",
            f"X净位移={net_x:.2f}mm, 仿真时长={times[-1]:.3f}s, 数据点={n_pts}",
            "--- 时间线采样 ---",
        ]
        report.extend(timeline)

        if net_y < -1.0:
            if t_at_y_min < times[-1] * 0.3:
                report.append(f"[诊断] Y 在仿真前 30% 时间内就降至最低点 → 快速摔倒，"
                              f"可能是重心偏前（DIST_BETTERY）或腿部角度导致初始姿态不稳")
            else:
                report.append(f"[诊断] Y 持续下降 → 行走过程中逐渐失衡摔倒")
            report.append(f"[物理背景] DIST_BETTERY 是电池到前腿的距离，"
                          f"越小→电池越靠前→重心偏前，越大→电池越靠后→重心偏后。"
                          f"请结合 Y 轨迹走势和各参数综合分析摔倒原因。")
        elif (y_max - y_min) > 2.0 and net_y >= -1.0:
            report.append(f"[诊断] Y 振幅 {y_max - y_min:.2f}mm 较大但未摔倒 → 可能是软件波动或行走不稳")

        return "\n".join(report)
    except Exception:
        return ""


def _compute_fitness(results: Dict) -> float:
    """计算 fitness（越大越好）。

    fitness = Net_Displacement_X（X方向净位移，mm）。
    """
    return results.get("Net_Displacement_X", 0.0)


import random as _random


def _perturb_params(m, n, alpha, beta, DIST_BETTERY, attempt: int):
    """对参数施加小幅随机扰动（每次 attempt 加大幅度），并 clamp 到合法范围。"""
    scale = 0.02 * attempt
    def _jitter(val, lo, hi):
        rng = hi - lo
        delta = _random.uniform(-scale * rng, scale * rng)
        return max(lo, min(hi, val + delta))
    return (
        _jitter(m, *PARAM_BOUNDS["m"]),
        _jitter(n, *PARAM_BOUNDS["n"]),
        _jitter(alpha, *PARAM_BOUNDS["alpha"]),
        _jitter(beta, *PARAM_BOUNDS["beta"]),
        _jitter(DIST_BETTERY, *PARAM_BOUNDS["DIST_BETTERY"]),
    )


_sim_counter = 0


def _archive_result_dir(status: str, params: dict):
    """将 result/ 中的非 .bin 文件复制到 result_archive/iter_NNNN/ 便于 debug。"""
    import shutil
    global _sim_counter
    _sim_counter += 1
    archive = RESULT_DIR.parent / "result_archive" / f"iter_{_sim_counter:04d}"
    archive.mkdir(parents=True, exist_ok=True)
    for f in RESULT_DIR.iterdir():
        if f.is_file() and f.suffix not in ('.bin',):
            try:
                shutil.copy2(f, archive / f.name)
            except Exception:
                pass
    meta = {"iter": _sim_counter, "status": status, "params": params}
    (archive / "_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_single_sim(cur_m, cur_n, cur_alpha, cur_beta, cur_dist,
                     log_fn, log_messages):
    """执行一次完整仿真，返回 (result_dict_or_None, json_str_or_None)。
    result_dict_or_None: 仿真结果 dict（含 range_y 等），仅成功时非 None。
    json_str_or_None: 非 None 表示应直接返回该 JSON（错误/失败）。
    """
    check = _validate_design(cur_m, cur_n, cur_alpha, cur_beta, cur_dist)
    if check["status"] != "ok":
        return None, json.dumps({
            "result": {
                "status": "constraint_violation",
                "errors": check["errors"],
                "design": check["design"],
                "derived": check["derived"],
            },
            "logs": log_messages,
        }, ensure_ascii=False, indent=2), check

    _write_input_csv(cur_m, cur_n, cur_alpha, cur_beta, cur_dist)
    log_fn(f"已写入 input.csv: m={cur_m}, n={cur_n}, alpha={cur_alpha}, "
           f"beta={cur_beta}, DIST={cur_dist}")

    for f in [FINAL_RES_CSV, RESULT_DIR / "results.csv"]:
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass

    log_fn("启动 SolidWorks + Adams 仿真流水线...")
    log_dir = PACKAGE_DIR.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = log_dir / "run_subprocess.log"

    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    sub_env["PYTHONUNBUFFERED"] = "1"
    sub_env["PYTHONUTF8"] = "1"
    force_amp = os.environ.get("ROBOT_FORCE_AMPLITUDE")
    if force_amp:
        sub_env["ROBOT_FORCE_AMPLITUDE"] = force_amp

    sim_timeout = int(os.environ.get("ROBOT_SIMULATION_TIMEOUT", "600"))
    stall_timeout = int(os.environ.get("ROBOT_STALL_TIMEOUT", "600"))

    def _kill_sw_adams():
        """杀掉残留的 SolidWorks / Adams 进程"""
        try:
            import psutil
        except ImportError:
            return
        targets = {"SLDWORKS.exe", "sldworks.exe", "aview.exe",
                   "adams_main.exe", "adamssolve.exe"}
        for p in psutil.process_iter(["name", "pid"]):
            try:
                if p.info["name"] in targets:
                    log_fn(f"[看门狗] 杀掉残留进程 {p.info['name']} (PID {p.info['pid']})")
                    p.kill()
            except Exception:
                pass

    try:
        lf = open(run_log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", str(RUN_SCRIPT)],
            cwd=str(CODE_DIR),
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=sub_env,
        )

        start_ts = time.time()
        last_size = 0
        last_activity_ts = start_ts

        while proc.poll() is None:
            time.sleep(5)
            elapsed = time.time() - start_ts
            if elapsed > sim_timeout:
                log_fn(f"[看门狗] 总超时 {sim_timeout}s，强制终止")
                proc.kill()
                _kill_sw_adams()
                break
            try:
                cur_size = run_log_path.stat().st_size
            except OSError:
                cur_size = last_size
            if cur_size != last_size:
                last_size = cur_size
                last_activity_ts = time.time()
            elif time.time() - last_activity_ts > stall_timeout:
                log_fn(f"[看门狗] 日志 {stall_timeout}s 无新输出，判定 SW 卡住，强制终止")
                proc.kill()
                _kill_sw_adams()
                break

        proc.wait(timeout=10)
        lf.close()

        log_fn(f"run.py 退出码: {proc.returncode} (日志: {run_log_path})")
        if proc.returncode != 0:
            tail = ""
            try:
                with open(run_log_path, "r", encoding="utf-8", errors="replace") as rf:
                    lines = rf.readlines()
                    tail = "".join(lines[-20:])
            except Exception:
                pass
            log_fn(f"run.py 失败，退出码: {proc.returncode}\n--- 日志尾部 ---\n{tail}")
            _archive_result_dir(f"exit_{proc.returncode}",
                                {"m": cur_m, "n": cur_n, "alpha": cur_alpha,
                                 "beta": cur_beta, "DIST_BETTERY": cur_dist})
            return None, json.dumps({
                "result": {
                    "status": "simulation_failed",
                    "errors": [f"run.py 退出码 {proc.returncode}"],
                    "log_tail": tail,
                },
                "logs": log_messages,
            }, ensure_ascii=False, indent=2), check
    except OSError as exc:
        log_fn(f"启动子进程失败: {exc}")
        return None, json.dumps({
            "result": {"status": "error", "errors": [f"启动子进程失败: {exc}"]},
            "logs": log_messages,
        }, ensure_ascii=False, indent=2), check

    results = _read_final_results()
    cur_params = {"m": cur_m, "n": cur_n, "alpha": cur_alpha,
                  "beta": cur_beta, "DIST_BETTERY": cur_dist}
    if not results:
        log_fn("未能读取 final_res.csv")
        _archive_result_dir("no_final_res", cur_params)
        return None, json.dumps({
            "result": {"status": "error", "errors": ["final_res.csv 不存在或为空"]},
            "logs": log_messages,
        }, ensure_ascii=False, indent=2), check

    _archive_result_dir("ok", cur_params)
    return results, None, check


async def run_robot_simulation(
    m: float,
    n: float,
    alpha: float,
    beta: float,
    DIST_BETTERY: float,
) -> str:
    """运行 SolidWorks + Adams 完整仿真流水线，返回位移/速度等性能指标及 fitness。

    Displacement_Range_Y > 2mm 时的处理：
    - Net_Displacement_Y < -1mm → 判定为摔倒（重心失稳），直接 fitness=0，不重跑
    - Net_Displacement_Y >= -1mm → 疑似软件波动，用相同参数直接重跑（最多 3 次）
    """
    m, n, alpha, beta, DIST_BETTERY = snap_params(m, n, alpha, beta, DIST_BETTERY)
    loop = asyncio.get_running_loop()
    log_messages: List[str] = []

    def log(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        full = f"[RobotLeg Tool][{ts}] {msg}"
        log_messages.append(full)
        print(full, file=sys.stderr, flush=True)

    def _run() -> str:
        cur_m, cur_n, cur_alpha, cur_beta, cur_dist = m, n, alpha, beta, DIST_BETTERY
        sim_fail_retries = 0

        for attempt in range(1 + MAX_RETRY_ON_RANGE_Y + MAX_RETRY_ON_SIM_FAIL):
            results, err_json, check = _run_single_sim(
                cur_m, cur_n, cur_alpha, cur_beta, cur_dist,
                log, log_messages)

            if err_json is not None:
                sim_fail_retries += 1
                if sim_fail_retries <= MAX_RETRY_ON_SIM_FAIL:
                    log(f"⚠️ 仿真失败，清理进程后 10s 重试 ({sim_fail_retries}/{MAX_RETRY_ON_SIM_FAIL})")
                    _kill_sw_adams_global()
                    import time as _time
                    _time.sleep(10)
                    continue
                return err_json

            # run.py 已检测到起飞，直接跳出不重跑
            if results.get("Robot_Flew", 0) == 1:
                log(f"🚀 run.py 检测到机器人起飞（Y 大幅上升），fitness=0，不重跑")
                range_y = results.get("Displacement_Range_Y", 999.0)
                break

            # run.py 已检测到摔倒，直接跳出不重跑
            if results.get("Robot_Fell", 0) == 1:
                log(f"⚠️ run.py 检测到摔倒（Y 早期大幅下降），fitness=0")
                range_y = results.get("Displacement_Range_Y", 999.0)
                break

            range_y = results.get("Displacement_Range_Y", 0.0)
            if range_y <= RANGE_Y_LIMIT:
                break

            net_y = results.get("Net_Displacement_Y", 0.0)

            # 1) 机器人摔倒（Y 大幅下降）→ 不重跑
            if net_y < FALL_NET_Y_THRESHOLD:
                log(f"⚠️ Range_Y={range_y:.2f}mm，Net_Y={net_y:.2f}mm "
                    f"→ 判定为摔倒（重心失稳），不重跑，fitness=0")
                break

            # 2) 机器人飞上天（Range_Y 极大 或 Net_Y 大幅正向）→ 物理性起飞，不重跑
            if range_y > RANGE_Y_FLY_LIMIT or net_y > FLY_NET_Y_THRESHOLD:
                log(f"🚀 Range_Y={range_y:.2f}mm，Net_Y={net_y:.2f}mm "
                    f"→ 判定为物理性起飞（参数导致机器人飞离地面），不重跑，fitness=0")
                break

            # 3) Range_Y 在 2~10mm 之间且 Net_Y 接近零 → 疑似软件小波动，可重跑
            if attempt < MAX_RETRY_ON_RANGE_Y:
                log(f"⚠️ Range_Y={range_y:.2f}mm（2~10mm 之间），"
                    f"Net_Y={net_y:.2f}mm（接近零），疑似软件波动，"
                    f"原参数不变，直接重跑 ({attempt+1}/{MAX_RETRY_ON_RANGE_Y})")
            else:
                log(f"⚠️ 重试 {MAX_RETRY_ON_RANGE_Y} 次后 Range_Y 仍>{RANGE_Y_LIMIT}mm，"
                    f"以最后一次结果返回 (fitness=0)")

        fell_flag = results.get("Robot_Fell", 0) == 1
        flew_flag = results.get("Robot_Flew", 0) == 1
        fitness = 0.0 if (fell_flag or flew_flag or range_y > RANGE_Y_LIMIT) else _compute_fitness(results)
        log(f"仿真完成: displacement_x={results.get('Net_Displacement_X')}, "
            f"range_y={results.get('Displacement_Range_Y')}, fitness={fitness:.6f}"
            + (f" (retried, params: m={cur_m}, n={cur_n}, alpha={cur_alpha}, "
               f"beta={cur_beta}, DIST={cur_dist})" if attempt > 0 else ""))

        if flew_flag:
            status_tag = "robot_flew"
        elif fell_flag:
            status_tag = "robot_fell"
        elif range_y > RANGE_Y_LIMIT:
            net_y = results.get("Net_Displacement_Y", 0.0)
            if net_y < FALL_NET_Y_THRESHOLD:
                status_tag = "robot_fell"
            elif range_y > RANGE_Y_FLY_LIMIT or net_y > FLY_NET_Y_THRESHOLD:
                status_tag = "robot_flew"
            else:
                status_tag = "range_y_exceeded_software"
        else:
            status_tag = "ok"

        cur_params = {"m": cur_m, "n": cur_n, "alpha": cur_alpha,
                      "beta": cur_beta, "DIST_BETTERY": cur_dist}
        trajectory_diag = _build_trajectory_diagnosis(cur_params)

        result_payload = {
            "status": status_tag,
            "fitness": fitness,
            "Net_Displacement_X": results.get("Net_Displacement_X", 0),
            "Displacement_Range_Y": results.get("Displacement_Range_Y", 0),
            "Net_Displacement_Y": results.get("Net_Displacement_Y", 0),
            "Net_Displacement_Z": results.get("Net_Displacement_Z", 0),
            "Total_Net_Displacement_3D": results.get("Total_Net_Displacement_3D", 0),
            "Average_Velocity_X": results.get("Average_Velocity_X", 0),
            "Total_Average_Velocity_3D": results.get("Total_Average_Velocity_3D", 0),
            "Simulation_Time": results.get("Simulation_Time", 0),
            "Data_Points": results.get("Data_Points", 0),
            "design": {"m": cur_m, "n": cur_n, "alpha": cur_alpha,
                       "beta": cur_beta, "DIST_BETTERY": cur_dist},
            "derived": check.get("derived", {}),
            "retries": attempt,
        }
        if trajectory_diag:
            result_payload["trajectory_diagnosis"] = trajectory_diag

        return json.dumps({
            "result": result_payload,
            "logs": log_messages,
        }, ensure_ascii=False, indent=2)

    TIMEOUT = int(os.environ.get("ROBOT_SIMULATION_TIMEOUT", "600"))
    total_timeout = TIMEOUT * (1 + MAX_RETRY_ON_RANGE_Y) + 30
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=total_timeout,
        )
    except asyncio.TimeoutError:
        return json.dumps({
            "result": {"status": "error", "message": f"仿真超时（{total_timeout}s）"},
            "logs": log_messages,
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({
            "result": {"status": "error", "message": str(exc)},
            "logs": [f"仿真异常：{exc}"],
        }, ensure_ascii=False, indent=2)
