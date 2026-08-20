"""Genetic Algorithm baseline for Maxwell actuator optimization.

Runs a GA on the same objective as `evaluate_design_fitness` and optionally
compares against an LLM-run CSV (min fitness row).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

from maxwell_pyaedt_run import (
    ActuatorDesignVariables,
    PENALTY_FITNESS,
    WA_FIXED,
    evaluate_design_fitness,
)

# Variable order and bounds (mm)
VAR_ORDER: Tuple[str, ...] = (
    "lm",
    "tm",
    "ta",
    "dg",
    "hs",
    "wslot",
    "hslot",
    "s",
    "wa",
)
BOUNDS: Dict[str, Tuple[float, float]] = {
    "lm": (0.0, 6.0),
    "tm": (0.3, 0.5),
    "ta": (0.4, 0.5),
    "dg": (0.25, 0.65),
    "hs": (1.2, 2.2),
    "wslot": (2.0, 2.8),
    "hslot": (0.8, 1.3),
    "s": (0.8, 1.2),
    "wa": (WA_FIXED, WA_FIXED),
}


def _clamp(name: str, value: float) -> float:
    low, high = BOUNDS[name]
    return max(low, min(value, high))


def _random_genome() -> List[float]:
    genome: List[float] = []
    for name in VAR_ORDER:
        low, high = BOUNDS[name]
        genome.append(random.uniform(low, high))
    return genome


def _mutate(genome: List[float], rate: float, sigma_frac: float) -> List[float]:
    mutated = genome[:]
    for i, name in enumerate(VAR_ORDER):
        if random.random() < rate:
            low, high = BOUNDS[name]
            span = high - low
            sigma = max(span * sigma_frac, 1e-3)
            mutated[i] = _clamp(name, mutated[i] + random.gauss(0, sigma))
    return mutated


def _blend_crossover(p1: List[float], p2: List[float]) -> Tuple[List[float], List[float]]:
    alpha = random.random()
    c1: List[float] = []
    c2: List[float] = []
    for i, name in enumerate(VAR_ORDER):
        low, high = BOUNDS[name]
        v1, v2 = p1[i], p2[i]
        c1.append(_clamp(name, alpha * v1 + (1 - alpha) * v2))
        c2.append(_clamp(name, alpha * v2 + (1 - alpha) * v1))
    return c1, c2


def _tournament(pop: List[List[float]], fitnesses: List[float], k: int) -> List[float]:
    best_idx = None
    for _ in range(k):
        idx = random.randrange(len(pop))
        if best_idx is None or fitnesses[idx] < fitnesses[best_idx]:
            best_idx = idx
    return pop[best_idx][:]


def _design_from_genome(genome: List[float]) -> ActuatorDesignVariables:
    kwargs = {name: val for name, val in zip(VAR_ORDER, genome)}
    return ActuatorDesignVariables(**kwargs)


def evaluate_genome(
    genome: List[float],
    weight_factors: Sequence[float],
    project_name: str,
    design_name: str,
    setup_name: str,
    output_root: str,
) -> Dict[str, object]:
    design = _design_from_genome(genome)
    pre_errors = design.validate_without_B()
    if pre_errors:
        return {
            "status": "constraint_violation",
            "fitness": PENALTY_FITNESS,
            "errors": pre_errors,
            "design": design,
        }

    eval_dir = os.path.join(output_root, f"eval_{int(time.time() * 1000)}_{random.randint(0, 9999):04d}")
    os.makedirs(eval_dir, exist_ok=True)
    result = evaluate_design_fitness(
        design,
        weight_factors=weight_factors,
        penalty_value=PENALTY_FITNESS,
        project_name=project_name,
        design_name=design_name,
        setup_name=setup_name,
        output_dir=eval_dir,
    )
    result["design"] = design
    return result


def run_ga(
    population_size: int,
    generations: int,
    mutation_rate: float,
    mutation_sigma: float,
    crossover_rate: float,
    tournament_k: int,
    weight_factors: Sequence[float],
    project_name: str,
    design_name: str,
    setup_name: str,
    output_root: str,
    seed: int | None = None,
) -> Dict[str, object]:
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_root, exist_ok=True)
    cache: Dict[Tuple[float, ...], Dict[str, object]] = {}
    population = [_random_genome() for _ in range(population_size)]

    def eval_once(genome: List[float]) -> Dict[str, object]:
        key = tuple(round(v, 3) for v in genome)
        if key in cache:
            return cache[key]
        record = evaluate_genome(
            genome,
            weight_factors=weight_factors,
            project_name=project_name,
            design_name=design_name,
            setup_name=setup_name,
            output_root=output_root,
        )
        cache[key] = record
        return record

    records = [eval_once(g) for g in population]
    best = min(records, key=lambda r: r.get("fitness", PENALTY_FITNESS))
    history = [(0, best["fitness"])]

    for gen in range(1, generations + 1):
        fitnesses = [r.get("fitness", PENALTY_FITNESS) for r in records]
        new_pop: List[List[float]] = []
        while len(new_pop) < population_size:
            p1 = _tournament(population, fitnesses, tournament_k)
            p2 = _tournament(population, fitnesses, tournament_k)
            if random.random() < crossover_rate:
                c1, c2 = _blend_crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            c1 = _mutate(c1, mutation_rate, mutation_sigma)
            c2 = _mutate(c2, mutation_rate, mutation_sigma)
            new_pop.extend([c1, c2])
        population = new_pop[:population_size]
        records = [eval_once(g) for g in population]
        gen_best = min(records, key=lambda r: r.get("fitness", PENALTY_FITNESS))
        if gen_best.get("fitness", PENALTY_FITNESS) < best.get("fitness", PENALTY_FITNESS):
            best = gen_best
        history.append((gen, best.get("fitness", PENALTY_FITNESS)))

    return {
        "best": best,
        "history": history,
        "evaluations": len(cache),
        "cache": cache,
    }


def _design_to_dict(design: ActuatorDesignVariables) -> Dict[str, float]:
    return {name: getattr(design, name) for name in VAR_ORDER}


def save_ga_results(cache: Dict[Tuple[float, ...], Dict[str, object]], path: str) -> None:
    fieldnames = [
        "fitness",
        "status",
        "avg_B",
        "B_sat",
        "volume",
        "mass_total",
        "mass_mover",
        "mass_stator",
        "kb",
        "pb",
    ] + list(VAR_ORDER)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in cache.values():
            row = {key: rec.get(key) for key in fieldnames}
            design = rec.get("design")
            if isinstance(design, ActuatorDesignVariables):
                row.update(_design_to_dict(design))
            writer.writerow(row)


def load_llm_best(csv_path: str) -> Tuple[float | None, Dict[str, float]]:
    if not csv_path or not os.path.exists(csv_path):
        return None, {}
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
                return None, {}
            design = {}
            for name in VAR_ORDER:
                val = best_row.get(name)
                if val is None or val == "":
                    continue
                try:
                    design[name] = float(val)
                except Exception:
                    pass
            return best_fitness, design
    except Exception:
        return None, {}


def main() -> None:
    parser = argparse.ArgumentParser(description="GA baseline for Maxwell actuator")
    parser.add_argument("--project", default="PyAEDT_Project")
    parser.add_argument("--design", default="Maxwell3DDesign_PyAEDT")
    parser.add_argument("--setup", default="Setup1")
    parser.add_argument("--weights", nargs=4, type=float, default=[0.5, 0.5, 4.0, 1.0])
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--mutation-sigma", type=float, default=0.1)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default="ga_baseline_runs")
    parser.add_argument("--llm-results-csv", default=None, help="LLM 运行生成的 CSV，用于对比最优 fitness")
    args = parser.parse_args()

    start = time.perf_counter()
    result = run_ga(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        crossover_rate=args.crossover_rate,
        tournament_k=args.tournament_k,
        weight_factors=tuple(args.weights),
        project_name=args.project,
        design_name=args.design,
        setup_name=args.setup,
        output_root=args.output_root,
        seed=args.seed,
    )

    best = result["best"]
    best_fitness = best.get("fitness", PENALTY_FITNESS)
    best_design = best.get("design")
    if isinstance(best_design, ActuatorDesignVariables):
        print("GA 最优设计:", _design_to_dict(best_design))
    print(f"GA 最优 fitness: {best_fitness}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.output_root, f"ga_results_{timestamp}.csv")
    save_ga_results(result["cache"], csv_path)
    print(f"GA 结果已写入: {csv_path}")

    llm_best_fitness, llm_design = load_llm_best(args.llm_results_csv)
    if llm_best_fitness is not None:
        print(f"LLM 最优 fitness: {llm_best_fitness}")
        if llm_design:
            print(f"LLM 最优设计: {llm_design}")
        diff = llm_best_fitness - best_fitness
        print(f"对比：LLM - GA = {diff}")
    else:
        if args.llm_results_csv:
            print(f"未能从 {args.llm_results_csv} 读取 LLM 最优值，跳过对比。")

    elapsed = time.perf_counter() - start
    print(f"耗时 {elapsed:.2f}s，评估次数 {result['evaluations']}。")


if __name__ == "__main__":
    main()
