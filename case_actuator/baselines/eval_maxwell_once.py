#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在 agent_maxwell 环境中执行一次 Maxwell 评估。

输入:  JSON 文件，格式 {"params": {...}}
输出: JSON 文件，格式 {"status": "...", "fitness": float, "errors": [...], "raw_result": {...}}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# 确保在子进程环境中也能导入项目包（baselines/src）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Maxwell design point")
    parser.add_argument("--input", required=True, help="输入 JSON 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    return parser.parse_args()


def _safe_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    try:
        if not in_path.exists():
            _safe_write(
                out_path,
                {
                    "status": "input_error",
                    "fitness": 1_000_000.0,
                    "errors": [f"输入文件不存在: {in_path}"],
                    "raw_result": {},
                },
            )
            return 2

        data = json.loads(in_path.read_text(encoding="utf-8"))
        params = data.get("params", {})
        if not isinstance(params, dict):
            _safe_write(
                out_path,
                {
                    "status": "input_error",
                    "fitness": 1_000_000.0,
                    "errors": ["输入 JSON 中 params 非 dict"],
                    "raw_result": {},
                },
            )
            return 2

        # 延迟导入，确保在 agent_maxwell 环境下执行
        from baselines.base_optimizer import BaseOptimizer

        class _OneShotEvaluator(BaseOptimizer):
            def optimize(self):
                raise NotImplementedError

        evaluator = _OneShotEvaluator(max_evals=1, output_dir=".", algorithm_name="MAXWELL_SINGLE_EVAL")
        result = evaluator.evaluate(params)

        fitness = float(result.fitness)
        if not math.isfinite(fitness):
            fitness = 1_000_000.0

        payload = {
            "status": result.status,
            "fitness": fitness,
            "errors": result.errors or [],
            "raw_result": result.raw_result or {},
        }
        _safe_write(out_path, payload)
        return 0

    except Exception as e:
        _safe_write(
            out_path,
            {
                "status": "execution_error",
                "fitness": 1_000_000.0,
                "errors": [f"{type(e).__name__}: {e}"],
                "raw_result": {},
            },
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
