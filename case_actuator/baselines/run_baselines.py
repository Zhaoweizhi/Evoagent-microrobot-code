#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线算法运行入口
用于与 LLM 优化进行对比实验

使用方法:
    # 遗传算法 (200次评估)
    python baselines/run_baselines.py --algo ga --max-evals 200 --pop-size 20
    
    # 贝叶斯优化 (200次评估)
    python baselines/run_baselines.py --algo bo --max-evals 200 --n-initial 10
    
    # 粒子群算法 (200次评估)
    python baselines/run_baselines.py --algo pso --max-evals 200 --swarm-size 20
    
    # 运行全部算法
    python baselines/run_baselines.py --algo all --max-evals 200
"""
# ============================================================
# 关键：在导入任何其他库之前，先设置 OpenMP 环境变量并预加载 PyTorch
# 这是为了解决 Anaconda numpy + PyTorch 的 OpenMP 冲突问题
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 尝试预加载 PyTorch（必须在 numpy 之前）
try:
    import torch
    _TORCH_PRELOADED = True
except Exception:
    _TORCH_PRELOADED = False

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# --- Windows 控制台中文乱码修复（强制 UTF-8 输出） ---
# 说明：Cursor/PowerShell 后台会话经常导致 stdout/stderr 仍按 GBK 编码输出，
# 进而在日志文件/终端回放里出现 “����”。这里在入口处统一强制 UTF-8。
def _force_utf8_stdio() -> None:
    if os.name != "nt":
        return
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Python < 3.7 或某些环境下 reconfigure 不可用，忽略即可
        pass


_force_utf8_stdio()

# Resolve the project root (paper_code_release/) regardless of nesting depth.
# This file lives at:  <project_root>/case_actuator/baselines/run_baselines.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loguru import logger

# 配置日志
os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)
logger.add(
    PROJECT_ROOT / "logs" / "baselines_{time}.log",
    encoding="utf-8",
    rotation="20 MB",
    retention="7 days",
)


def _send_notification_email(
    notify_email: str,
    smtp_server: str,
    smtp_port: int,
    smtp_password: str,
    subject: str,
    body: str,
) -> bool:
    """发送邮件通知（失败不抛异常，返回 False）。"""
    if not notify_email or not smtp_password:
        return False

    import smtplib
    from email.mime.text import MIMEText
    from email.header import Header

    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = notify_email
        msg["To"] = notify_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(notify_email, smtp_password)
            server.sendmail(notify_email, [notify_email], msg.as_string())
        return True
    except Exception as e:
        logger.warning(f"[Notify] 邮件发送失败: {e}")
        return False


def _find_latest_results_csv(output_dir: str, prefix: str) -> Optional[str]:
    """在 output_dir 中找到最新的 <prefix>_results_*.csv 文件路径。"""
    try:
        p = Path(output_dir)
        if not p.exists():
            return None
        candidates = list(p.glob(f"{prefix}_results_*.csv"))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(candidates[0])
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="基线优化算法运行器")
    
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ga", "bo", "pso", "smac", "bore", "pfn", "pom", "all"],
        default="ga",
        help="选择算法: ga, bo, pso, smac, bore, pfn, pom, all"
    )
    
    parser.add_argument(
        "--max-evals",
        type=int,
        default=200,
        help="最大仿真评估次数（默认 200）"
    )
    
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=200,
        help="最小迭代次数（收敛判断前至少运行这么多轮，默认 200）"
    )
    
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=40,
        help="收敛窗口（连续多少轮无改进视为收敛，默认 40）"
    )
    
    parser.add_argument(
        "--avg-window",
        type=int,
        default=10,
        help="早停平均窗口（默认 10）"
    )
    
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.01,
        help="收敛阈值（改进比例小于此值视为无改进，默认 0.01）"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT),
        help="输出目录（默认项目根目录）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑（支持 GA、BO、PSO）"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="断点文件路径（GA: GA_checkpoint.pkl, BO: BO_checkpoint.pkl, PSO: PSO_checkpoint.pkl）"
    )
    
    # GA 参数
    parser.add_argument("--pop-size", type=int, default=20, help="[GA] 种群大小")
    parser.add_argument("--crossover-prob", type=float, default=0.8, help="[GA] 交叉概率")
    parser.add_argument("--mutation-prob", type=float, default=0.1, help="[GA] 变异概率")
    parser.add_argument("--elite-size", type=int, default=2, help="[GA] 每代精英保留数量（默认 2）")
    
    # BO 参数
    parser.add_argument("--n-initial", type=int, default=10, help="[BO] 初始采样数量")
    parser.add_argument("--acq-func", type=str, default="EI", help="[BO] 采集函数 (EI/PI/LCB)")

    # PSO 参数
    parser.add_argument("--swarm-size", type=int, default=20, help="[PSO] 粒子群大小")
    parser.add_argument("--w", type=float, default=0.7, help="[PSO] 惯性权重")
    parser.add_argument("--c1", type=float, default=1.5, help="[PSO] 认知因子")
    parser.add_argument("--c2", type=float, default=1.5, help="[PSO] 社会因子")

    # SMAC 参数
    parser.add_argument("--smac-initial", type=int, default=20, help="[SMAC] 初始采样数量")
    parser.add_argument("--smac-trees", type=int, default=10, help="[SMAC] Random Forest 树数量")

    # BORE 参数
    parser.add_argument("--bore-initial", type=int, default=20, help="[BORE] 初始采样数量")
    parser.add_argument("--bore-gamma", type=float, default=0.25, help="[BORE] 分位数阈值（top γ 作为好点，默认 0.25）")
    parser.add_argument("--bore-hidden", type=str, default="64,32", help="[BORE] 神经网络隐藏层（逗号分隔，默认 64,32）")
    parser.add_argument("--bore-epochs", type=int, default=100, help="[BORE] 训练轮数（默认 100）")
    parser.add_argument("--bore-candidates", type=int, default=1000, help="[BORE] 每轮候选点数量（默认 1000）")

    # PFN-CEI 参数
    parser.add_argument("--pfn-initial", type=int, default=10, help="[PFN] 初始采样数量")
    parser.add_argument("--pfn-candidates", type=int, default=5000, help="[PFN] 每轮候选点数量（默认 5000）")
    parser.add_argument("--pfn-device", type=str, default="cpu", help="[PFN] 计算设备 (cpu/cuda:0)")

    # POM 参数
    parser.add_argument("--pom-pop-size", type=int, default=50, help="[POM] 种群大小")
    parser.add_argument("--pom-device", type=str, default="cpu", help="[POM] 计算设备 (cpu/cuda:0)")
    parser.add_argument("--pom-model", type=str, default=None, help="[POM] 预训练模型路径")

    # ★邮件通知：基线实验结束/异常时发送邮件
    parser.add_argument("--notify-email",
                        default=None,
                        help="基线实验结束时发送通知邮件到指定地址（如 xxx@qq.com）")
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


def _inject_email_config(optimizer, args):
    """注入邮件通知配置到优化器"""
    smtp_password = args.smtp_password or os.getenv("SMTP_PASSWORD")
    if args.notify_email and smtp_password:
        optimizer._notify_email = args.notify_email
        optimizer._smtp_server = args.smtp_server
        optimizer._smtp_port = int(args.smtp_port)
        optimizer._smtp_password = smtp_password
        logger.info(f"[{optimizer.algorithm_name}] 已启用邮件通知: {args.notify_email}")


def run_ga(args):
    """运行遗传算法"""
    from baselines.ga_optimizer import GAOptimizer
    elite_size = max(2, int(args.elite_size))
    if elite_size != int(args.elite_size):
        logger.warning(f"[GA] elite-size={args.elite_size} 过小，已自动提升为 2")
    
    optimizer = GAOptimizer(
        max_evals=args.max_evals,
        min_iterations=args.min_iterations,
        convergence_window=args.convergence_window,
        avg_window=args.avg_window,
        convergence_threshold=args.convergence_threshold,
        pop_size=args.pop_size,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        elite_size=elite_size,
        output_dir=args.output_dir,
        seed=args.seed,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
    )
    _inject_email_config(optimizer, args)
    return optimizer.run()


def run_bo(args):
    """运行贝叶斯优化"""
    from baselines.bo_optimizer import BOOptimizer
    
    # 确定 checkpoint 路径（默认或用户指定）
    checkpoint_path = args.checkpoint if args.checkpoint else None
    if checkpoint_path is None and args.resume:
        checkpoint_path = str(Path(args.output_dir) / "BO_checkpoint.pkl")
    
    optimizer = BOOptimizer(
        max_evals=args.max_evals,
        min_iterations=args.min_iterations,
        convergence_window=args.convergence_window,
        avg_window=args.avg_window,
        convergence_threshold=args.convergence_threshold,
        n_initial=args.n_initial,
        acq_func=args.acq_func,
        output_dir=args.output_dir,
        seed=args.seed,
        resume=args.resume,
        checkpoint_path=checkpoint_path,
    )
    _inject_email_config(optimizer, args)
    return optimizer.run()


def run_pso(args):
    """运行粒子群算法"""
    from baselines.pso_optimizer import PSOOptimizer
    
    # 确定 checkpoint 路径（默认或用户指定）
    checkpoint_path = args.checkpoint if args.checkpoint else None
    if checkpoint_path is None and args.resume:
        checkpoint_path = str(Path(args.output_dir) / "PSO_checkpoint.pkl")
    
    optimizer = PSOOptimizer(
        max_evals=args.max_evals,
        min_iterations=args.min_iterations,
        convergence_window=args.convergence_window,
        avg_window=args.avg_window,
        convergence_threshold=args.convergence_threshold,
        swarm_size=args.swarm_size,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        output_dir=args.output_dir,
        seed=args.seed,
        resume=args.resume,
        checkpoint_path=checkpoint_path,
    )
    _inject_email_config(optimizer, args)
    return optimizer.run()


def run_smac(args):
    """运行 SMAC（Random Forest surrogate）"""
    from baselines.smac_optimizer import SMACOptimizer
    
    checkpoint_path = args.checkpoint if args.checkpoint else None
    if checkpoint_path is None and args.resume:
        checkpoint_path = str(Path(args.output_dir) / "SMAC_checkpoint.pkl")
    
    optimizer = SMACOptimizer(
        max_evals=args.max_evals,
        min_iterations=args.min_iterations,
        convergence_window=args.convergence_window,
        avg_window=args.avg_window,
        convergence_threshold=args.convergence_threshold,
        n_initial=args.smac_initial,
        n_trees=args.smac_trees,
        output_dir=args.output_dir,
        seed=args.seed,
        resume=args.resume,
        checkpoint_path=checkpoint_path,
    )
    _inject_email_config(optimizer, args)
    return optimizer.run()


def run_bore(args):
    """运行 BORE（神经网络密度比估计）"""
    from baselines.bore_optimizer import BOREOptimizer
    
    checkpoint_path = args.checkpoint if args.checkpoint else None
    if checkpoint_path is None and args.resume:
        checkpoint_path = str(Path(args.output_dir) / "BORE_checkpoint.pkl")
    
    # 解析隐藏层配置
    hidden_dims = [int(x.strip()) for x in args.bore_hidden.split(",")]
    
    optimizer = BOREOptimizer(
        max_evals=args.max_evals,
        min_iterations=args.min_iterations,
        convergence_window=args.convergence_window,
        avg_window=args.avg_window,
        convergence_threshold=args.convergence_threshold,
        n_initial=args.bore_initial,
        gamma=args.bore_gamma,
        hidden_dims=hidden_dims,
        n_epochs=args.bore_epochs,
        n_candidates=args.bore_candidates,
        output_dir=args.output_dir,
        seed=args.seed,
        resume=args.resume,
        checkpoint_path=checkpoint_path,
    )
    _inject_email_config(optimizer, args)
    return optimizer.run()


def run_pfn(args):
    """运行 PFN-CEI（预训练 Transformer 约束贝叶斯优化）"""
    from baselines.pfn_cei_optimizer import PFNCEIOptimizer
    
    checkpoint_path = args.checkpoint if args.checkpoint else None
    if checkpoint_path is None and args.resume:
        checkpoint_path = str(Path(args.output_dir) / "PFN_CEI_checkpoint.pkl")
    
    optimizer = PFNCEIOptimizer(
        max_evals=args.max_evals,
        min_iterations=args.min_iterations,
        convergence_window=args.convergence_window,
        avg_window=args.avg_window,
        convergence_threshold=args.convergence_threshold,
        initial_samples=args.pfn_initial,
        n_candidates=args.pfn_candidates,
        device=args.pfn_device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    # 设置检查点路径
    if checkpoint_path:
        optimizer.checkpoint_path = Path(checkpoint_path)
    
    _inject_email_config(optimizer, args)
    return optimizer.run()


def run_pom(args):
    """运行 POM（pretrained optimisation model）"""
    from baselines.pom_optimizer import POMOptimizer

    optimizer = POMOptimizer(
        max_evals=args.max_evals,
        min_iterations=args.min_iterations,
        convergence_window=args.convergence_window,
        avg_window=args.avg_window,
        convergence_threshold=args.convergence_threshold,
        pop_size=args.pom_pop_size,
        device=args.pom_device,
        model_path=args.pom_model,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    _inject_email_config(optimizer, args)
    return optimizer.run()


def main():
    args = parse_args()

    notify_email = args.notify_email
    smtp_server = args.smtp_server
    smtp_port = int(args.smtp_port)
    smtp_password = args.smtp_password or os.getenv("SMTP_PASSWORD")
    notify_enabled = bool(notify_email and smtp_password)
    if args.notify_email and not smtp_password:
        logger.warning("[Notify] 已配置 --notify-email，但未设置 SMTP 授权码（--smtp-password 或 SMTP_PASSWORD）")
    
    logger.info("=" * 70)
    logger.info(f"基线优化实验 | 算法={args.algo} | max_evals={args.max_evals} | seed={args.seed}")
    logger.info("=" * 70)
    
    results = {}
    csv_paths: Dict[str, str] = {}
    start_ts = datetime.now()

    try:
        if args.algo in ["ga", "all"]:
            logger.info("\n" + "=" * 30 + " GA " + "=" * 30)
            result = run_ga(args)
            if result:
                results["GA"] = result.fitness
                logger.info(f"[GA] 最优 fitness: {result.fitness:.6f}")
                p = _find_latest_results_csv(args.output_dir, "GA")
                if p:
                    csv_paths["GA"] = p
        
        if args.algo in ["bo", "all"]:
            logger.info("\n" + "=" * 30 + " BO " + "=" * 30)
            result = run_bo(args)
            if result:
                results["BO"] = result.fitness
                logger.info(f"[BO] 最优 fitness: {result.fitness:.6f}")
                p = _find_latest_results_csv(args.output_dir, "BO")
                if p:
                    csv_paths["BO"] = p

        if args.algo in ["pso", "all"]:
            logger.info("\n" + "=" * 30 + " PSO " + "=" * 30)
            result = run_pso(args)
            if result:
                results["PSO"] = result.fitness
                logger.info(f"[PSO] 最优 fitness: {result.fitness:.6f}")
                p = _find_latest_results_csv(args.output_dir, "PSO")
                if p:
                    csv_paths["PSO"] = p
        
        if args.algo in ["smac", "all"]:
            logger.info("\n" + "=" * 30 + " SMAC " + "=" * 30)
            result = run_smac(args)
            if result:
                results["SMAC"] = result.fitness
                logger.info(f"[SMAC] 最优 fitness: {result.fitness:.6f}")
                p = _find_latest_results_csv(args.output_dir, "SMAC")
                if p:
                    csv_paths["SMAC"] = p
        
        if args.algo in ["bore", "all"]:
            logger.info("\n" + "=" * 30 + " BORE " + "=" * 30)
            result = run_bore(args)
            if result:
                results["BORE"] = result.fitness
                logger.info(f"[BORE] 最优 fitness: {result.fitness:.6f}")
                p = _find_latest_results_csv(args.output_dir, "BORE")
                if p:
                    csv_paths["BORE"] = p
        
        if args.algo in ["pfn", "all"]:
            logger.info("\n" + "=" * 30 + " PFN-CEI " + "=" * 30)
            result = run_pfn(args)
            if result:
                results["PFN_CEI"] = result.fitness
                logger.info(f"[PFN-CEI] 最优 fitness: {result.fitness:.6f}")
                p = _find_latest_results_csv(args.output_dir, "PFN_CEI")
                if p:
                    csv_paths["PFN_CEI"] = p

        if args.algo in ["pom", "all"]:
            logger.info("\n" + "=" * 30 + " POM " + "=" * 30)
            result = run_pom(args)
            if result:
                results["POM"] = result.fitness
                logger.info(f"[POM] 最优 fitness: {result.fitness:.6f}")
                p = _find_latest_results_csv(args.output_dir, "POM")
                if p:
                    csv_paths["POM"] = p
    except KeyboardInterrupt:
        logger.warning("用户中断 (Ctrl+C)")
        if notify_enabled:
            _send_notification_email(
                notify_email=notify_email,
                smtp_server=smtp_server,
                smtp_port=smtp_port,
                smtp_password=smtp_password,
                subject=f"Baseline INFO - User Interrupted ({args.algo})",
                body=f"Baseline run interrupted by user.\n\nalgo={args.algo}\nseed={args.seed}\nmax_evals={args.max_evals}\noutput_dir={args.output_dir}",
            )
        raise
    except Exception as exc:
        logger.error(f"基线运行异常: {exc}")
        if notify_enabled:
            partial = "\n".join([f"- {k}: {v:.6f}" for k, v in sorted(results.items(), key=lambda x: x[1])]) or "(no results yet)"
            _send_notification_email(
                notify_email=notify_email,
                smtp_server=smtp_server,
                smtp_port=smtp_port,
                smtp_password=smtp_password,
                subject=f"Baseline ERROR - {args.algo} (seed={args.seed})",
                body=f"Baseline run stopped due to exception.\n\nalgo={args.algo}\nseed={args.seed}\nmax_evals={args.max_evals}\noutput_dir={args.output_dir}\n\nPartial results:\n{partial}\n\nError:\n{exc}",
            )
        raise
    
    # 汇总结果
    logger.info("\n" + "=" * 70)
    logger.info("实验结果汇总:")
    logger.info("=" * 70)
    for algo, fitness in sorted(results.items(), key=lambda x: x[1]):
        logger.info(f"  {algo}: {fitness:.6f}")
    logger.info("=" * 70)

    # ★ 运行结束邮件通知
    if notify_enabled:
        elapsed = datetime.now() - start_ts
        best_algo = None
        best_fit = None
        if results:
            best_algo, best_fit = sorted(results.items(), key=lambda x: x[1])[0]

        summary_lines = []
        for algo, fitness in sorted(results.items(), key=lambda x: x[1]):
            line = f"- {algo}: {fitness:.6f}"
            if algo in csv_paths:
                line += f" | {csv_paths[algo]}"
            summary_lines.append(line)
        summary = "\n".join(summary_lines) if summary_lines else "(no results)"

        subject = f"Baseline Finished - {args.algo} (seed={args.seed})"
        if best_algo is not None and best_fit is not None:
            subject = f"Baseline Finished - best {best_algo} {best_fit:.6f} (seed={args.seed})"

        _send_notification_email(
            notify_email=notify_email,
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            smtp_password=smtp_password,
            subject=subject,
            body=(
                "Baseline run completed.\n\n"
                f"algo={args.algo}\n"
                f"seed={args.seed}\n"
                f"max_evals={args.max_evals}\n"
                f"elapsed={elapsed}\n"
                f"output_dir={args.output_dir}\n\n"
                f"Results:\n{summary}"
            ),
        )


if __name__ == "__main__":
    main()
