#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线算法统一运行入口（机器人腿任务）
用于与 LLM Agent 优化进行对比实验

使用方法:
    # 遗传算法
    python baselines/run_baselines.py --algo ga --max-evals 200 --pop-size 20

    # 贝叶斯优化 (GP+EI)
    python baselines/run_baselines.py --algo bo --max-evals 200 --n-initial 10

    # 粒子群算法
    python baselines/run_baselines.py --algo pso --max-evals 200 --swarm-size 20

    # SMAC (Random Forest + EI)
    python baselines/run_baselines.py --algo smac --max-evals 200

    # BORE (密度比估计)
    python baselines/run_baselines.py --algo bore --max-evals 200

    # PFN-CEI (预训练 Transformer)
    python baselines/run_baselines.py --algo pfncei --max-evals 200

    # POM (Portfolio of Methods)
    python baselines/run_baselines.py --algo pom --max-evals 200

    # 运行全部算法
    python baselines/run_baselines.py --algo all --max-evals 200

    # dry-run（不调用仿真，快速验证 CSV 格式）
    python baselines/run_baselines.py --algo ga --max-evals 20 --dry-run
"""
import os
import sys
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="机器人腿优化基线运行器")

    parser.add_argument(
        "--algo",
        type=str,
        choices=["ga", "bo", "pso", "smac", "bore", "pfncei", "pom", "all"],
        default="ga",
        help="选择算法: ga, bo, pso, smac, bore, pfncei, pom, all"
    )
    parser.add_argument("--max-evals", type=int, default=200,
                        help="最大仿真评估次数（默认 200）")
    parser.add_argument("--min-evals-before-stop", type=int, default=100,
                        help="早停前至少完成的真实评估次数（默认 100）")
    parser.add_argument("--patience-evals", type=int, default=40,
                        help="达到最少评估次数后，连续多少次无提升则收敛停止（默认 40）")
    parser.add_argument("--significant-improvement-ratio", type=float, default=0.01,
                        help="早停显著提升阈值；例如 0.01 表示 best 至少提升 1%% 才重置 patience（默认 0.01）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT),
                        help="输出目录（默认项目根目录）")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="跳过真仿真，快速验证 CSV 格式")

    # 断点续跑
    parser.add_argument("--resume", type=str, default=None,
                        help="断点文件路径，从该断点恢复继续优化")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="断点保存路径（默认自动在 output-dir 下生成）")

    # GA 参数
    parser.add_argument("--pop-size", type=int, default=20, help="[GA] 种群大小")
    parser.add_argument("--crossover-eta", type=float, default=15, help="[GA] SBX 分布指数")
    parser.add_argument("--mutation-prob", type=float, default=0.2, help="[GA] 变异概率")
    parser.add_argument("--mutation-eta", type=float, default=20, help="[GA] 变异分布指数")

    # BO 参数
    parser.add_argument("--n-initial", type=int, default=20, help="[BO/SMAC/BORE/PFN/POM] 初始采样数量")
    parser.add_argument("--acq-func", type=str, default="EI", help="[BO] 采集函数 (EI/PI/LCB)")

    # PSO 参数
    parser.add_argument("--swarm-size", type=int, default=20, help="[PSO] 粒子群大小")
    parser.add_argument("--w", type=float, default=0.7, help="[PSO] 惯性权重")
    parser.add_argument("--c1", type=float, default=1.5, help="[PSO] 认知因子")
    parser.add_argument("--c2", type=float, default=1.5, help="[PSO] 社会因子")

    # SMAC 参数
    parser.add_argument("--smac-trees", type=int, default=10, help="[SMAC] Random Forest 树数量")
    parser.add_argument(
        "--smac-backend",
        type=str,
        choices=["official", "rf"],
        default="official",
        help="[SMAC] 后端选择：official 使用官方 smac 包；rf 使用旧本地 RandomForest 补充模式",
    )

    # BORE 参数
    parser.add_argument("--bore-gamma", type=float, default=0.25, help="[BORE] 分位数阈值")
    parser.add_argument("--bore-hidden", type=str, default="64,32", help="[BORE] 隐藏层（逗号分隔）")
    parser.add_argument("--bore-epochs", type=int, default=100, help="[BORE] 训练轮数")
    parser.add_argument("--bore-candidates", type=int, default=1000, help="[BORE] 候选点数量")
    parser.add_argument(
        "--bore-backend",
        type=str,
        choices=["auto", "official", "torch", "sklearn"],
        default="official",
        help="[BORE] 后端选择：official 强制使用官方 bore 的 minimize_multi_start 候选优化路径；auto 为旧兼容模式；torch/sklearn 为本地候选筛选补充模式",
    )

    # PFN-CEI 参数
    parser.add_argument("--pfn-candidates", type=int, default=5000, help="[PFN] 候选点数量")
    parser.add_argument("--pfn-device", type=str, default="cpu", help="[PFN] 设备 (cpu/cuda:0)")
    parser.add_argument("--pfn-model", type=str, default=None, help="[PFN] 预训练模型路径")

    # POM 参数
    parser.add_argument("--pom-candidates", type=int, default=1000, help="[POM] 候选点数量")

    return parser.parse_args()


def configure_early_stop(optimizer, args):
    optimizer.min_evals_before_stop = args.min_evals_before_stop
    optimizer.patience_evals = args.patience_evals
    optimizer.significant_improvement_ratio = args.significant_improvement_ratio
    return optimizer


async def run_ga(args):
    from case_robot_leg.baselines.ga_optimizer import GAOptimizer
    optimizer = GAOptimizer(
        max_evals=args.max_evals,
        pop_size=args.pop_size,
        crossover_eta=args.crossover_eta,
        mutation_prob=args.mutation_prob,
        mutation_eta=args.mutation_eta,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    configure_early_stop(optimizer, args)
    await optimizer.run(dry_run=args.dry_run, resume_checkpoint=args.resume,
                        checkpoint_path=args.checkpoint)
    return optimizer


async def run_bo(args):
    from case_robot_leg.baselines.bo_optimizer import BOOptimizer
    optimizer = BOOptimizer(
        max_evals=args.max_evals,
        n_initial=args.n_initial,
        acq_func=args.acq_func,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    configure_early_stop(optimizer, args)
    await optimizer.run(dry_run=args.dry_run, resume_checkpoint=args.resume,
                        checkpoint_path=args.checkpoint)
    return optimizer


async def run_pso(args):
    from case_robot_leg.baselines.pso_optimizer import PSOOptimizer
    optimizer = PSOOptimizer(
        max_evals=args.max_evals,
        swarm_size=args.swarm_size,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    configure_early_stop(optimizer, args)
    await optimizer.run(dry_run=args.dry_run, resume_checkpoint=args.resume,
                        checkpoint_path=args.checkpoint)
    return optimizer


async def run_smac(args):
    from case_robot_leg.baselines.smac_optimizer import SMACOptimizer
    optimizer = SMACOptimizer(
        max_evals=args.max_evals,
        n_initial=args.n_initial,
        n_trees=args.smac_trees,
        backend=args.smac_backend,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    configure_early_stop(optimizer, args)
    await optimizer.run(dry_run=args.dry_run, resume_checkpoint=args.resume,
                        checkpoint_path=args.checkpoint)
    return optimizer


async def run_bore(args):
    from case_robot_leg.baselines.bore_optimizer import BOREOptimizer
    hidden_dims = [int(x.strip()) for x in args.bore_hidden.split(",")]
    optimizer = BOREOptimizer(
        max_evals=args.max_evals,
        n_initial=args.n_initial,
        gamma=args.bore_gamma,
        hidden_dims=hidden_dims,
        n_epochs=args.bore_epochs,
        n_candidates=args.bore_candidates,
        backend=args.bore_backend,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    configure_early_stop(optimizer, args)
    await optimizer.run(dry_run=args.dry_run, resume_checkpoint=args.resume,
                        checkpoint_path=args.checkpoint)
    return optimizer


async def run_pfncei(args):
    from case_robot_leg.baselines.pfn_cei_optimizer import PFNCEIOptimizer
    optimizer = PFNCEIOptimizer(
        max_evals=args.max_evals,
        n_initial=args.n_initial,
        n_candidates=args.pfn_candidates,
        device=args.pfn_device,
        model_path=args.pfn_model,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    configure_early_stop(optimizer, args)
    await optimizer.run(dry_run=args.dry_run, resume_checkpoint=args.resume,
                        checkpoint_path=args.checkpoint)
    return optimizer


async def run_pom(args):
    from case_robot_leg.baselines.pom_optimizer import POMOptimizer
    optimizer = POMOptimizer(
        max_evals=args.max_evals,
        n_initial=args.n_initial,
        n_candidates=args.pom_candidates,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    configure_early_stop(optimizer, args)
    await optimizer.run(dry_run=args.dry_run, resume_checkpoint=args.resume,
                        checkpoint_path=args.checkpoint)
    return optimizer


async def main():
    args = parse_args()

    print("=" * 70)
    print(f"机器人腿优化基线实验 | 算法={args.algo} | max_evals={args.max_evals} | seed={args.seed}")
    print(f"输出目录: {args.output_dir}")
    if args.dry_run:
        print("⚠️  DRY-RUN 模式（不调用真仿真）")
    print("=" * 70)

    results = {}
    algo_list = []
    if args.algo == "all":
        algo_list = ["ga", "pso", "bo", "smac", "bore", "pfncei", "pom"]
    else:
        algo_list = [args.algo]

    for algo in algo_list:
        print(f"\n{'='*30} {algo.upper()} {'='*30}")
        try:
            if algo == "ga":
                opt = await run_ga(args)
            elif algo == "bo":
                opt = await run_bo(args)
            elif algo == "pso":
                opt = await run_pso(args)
            elif algo == "smac":
                opt = await run_smac(args)
            elif algo == "bore":
                opt = await run_bore(args)
            elif algo == "pfncei":
                opt = await run_pfncei(args)
            elif algo == "pom":
                opt = await run_pom(args)
            else:
                print(f"未知算法: {algo}")
                continue

            result_key = opt.algorithm_name if algo == "bore" else algo.upper()
            results[result_key] = opt.best_fitness
        except KeyboardInterrupt:
            print(f"\n[{algo.upper()}] 用户中断")
            break
        except Exception as e:
            print(f"\n[{algo.upper()}] 运行异常: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    if results:
        print(f"\n{'='*70}")
        print("实验结果汇总:")
        print("=" * 70)
        for algo, fitness in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {algo:<10s}: {fitness:.4f}")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
