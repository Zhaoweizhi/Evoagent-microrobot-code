# -*- coding: utf-8 -*-
"""Genetic-algorithm optimizer for the robot-leg task.

The implementation uses simulated-binary crossover, polynomial mutation,
tournament selection, elitism and constraint-aware resampling.
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_optimizer import (
    BaseOptimizer, PARAM_NAMES, PARAM_BOUNDS, PARAM_DECIMALS,
    PENALTY_FITNESS, snap, clamp, random_params, generate_valid_params,
)


class GAOptimizer(BaseOptimizer):
    """遗传算法优化器（机器人腿）"""

    def __init__(
        self,
        max_evals: int = 200,
        pop_size: int = 20,
        crossover_eta: float = 15,
        mutation_prob: float = 0.2,
        mutation_eta: float = 20,
        tournament_k: int = 3,
        elite_ratio: float = 0.1,
        output_dir: str = ".",
        seed: int = 42,
    ):
        super().__init__(
            max_evals=max_evals,
            output_dir=output_dir,
            algorithm_name="GA",
            seed=seed,
        )
        self.pop_size = pop_size
        self.crossover_eta = crossover_eta
        self.mutation_prob = mutation_prob
        self.mutation_eta = mutation_eta
        self.tournament_k = tournament_k
        self.elite_size = max(1, int(pop_size * elite_ratio))
        self.n_gen = max(1, max_evals // pop_size)

    def _tournament_select(self, pop: List[Dict], fitnesses: List[float],
                           rng: random.Random) -> Dict[str, float]:
        indices = rng.sample(range(len(pop)), min(self.tournament_k, len(pop)))
        best_i = max(indices, key=lambda i: fitnesses[i])
        return {name: pop[best_i][name] for name in PARAM_NAMES}

    def _sbx_crossover(self, p1: Dict, p2: Dict, rng: random.Random) -> Tuple[Dict, Dict]:
        c1, c2 = {}, {}
        eta = self.crossover_eta
        for name in PARAM_NAMES:
            lo, hi = PARAM_BOUNDS[name]
            if rng.random() < 0.5 and abs(p1[name] - p2[name]) > 1e-14:
                u = rng.random()
                if u <= 0.5:
                    beta_q = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                v1 = 0.5 * ((1 + beta_q) * p1[name] + (1 - beta_q) * p2[name])
                v2 = 0.5 * ((1 - beta_q) * p1[name] + (1 + beta_q) * p2[name])
                c1[name] = clamp(name, v1)
                c2[name] = clamp(name, v2)
            else:
                c1[name] = p1[name]
                c2[name] = p2[name]
        return snap(c1), snap(c2)

    def _polynomial_mutate(self, ind: Dict, rng: random.Random) -> Dict:
        result = dict(ind)
        eta = self.mutation_eta
        for name in PARAM_NAMES:
            if rng.random() < self.mutation_prob:
                lo, hi = PARAM_BOUNDS[name]
                y = result[name]
                delta1 = (y - lo) / (hi - lo + 1e-14)
                delta2 = (hi - y) / (hi - lo + 1e-14)
                u = rng.random()
                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1))
                    deltaq = val ** (1.0 / (eta + 1)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1))
                    deltaq = 1.0 - val ** (1.0 / (eta + 1))
                result[name] = clamp(name, y + deltaq * (hi - lo))
        return snap(result)

    def _resample_or_mutate(self, ind: Dict, rng: random.Random,
                            max_attempts: int = 200) -> Dict:
        """变异后若不满足约束，尝试重采样"""
        for _ in range(max_attempts):
            mutant = self._polynomial_mutate(ind, rng)
            if self.check_constraints(mutant, source="mutation_resample"):
                return mutant
        # 兜底
        p, _ = generate_valid_params(rng)
        self.check_constraints(p, source="generate_valid_fallback")
        return p

    async def optimize(self, dry_run: bool = False):
        rng = random.Random(self.seed)
        population: List[Dict[str, float]] = []
        fitnesses: List[float] = []

        print(f"[GA] pop={self.pop_size}, gen={self.n_gen}, max_evals={self.max_evals}, "
              f"seed={self.seed}")
        print(f"[GA] elite={self.elite_size}, cx_eta={self.crossover_eta}, "
              f"mut_prob={self.mutation_prob}")
        print("=" * 60)

        # 从断点恢复种群状态
        start_gen = 0
        if hasattr(self, '_checkpoint_save_path') and self._checkpoint_save_path.exists():
            state = self.load_checkpoint(self._checkpoint_save_path)
            if state and "population" in state:
                population = state["population"]
                fitnesses = state["fitnesses"]
                start_gen = state.get("generation", 0) + 1
                print(f"[GA] 从断点恢复: gen={start_gen-1}, pop={len(population)}, "
                      f"eval_count={self.eval_count}, best={self.best_fitness:.4f}")

        if not population:
            # 初始种群
            self.current_generation = 0
            print(f"[GA] === 第 0/{self.n_gen} 代 (初始种群) ===")
            for i in range(self.pop_size):
                if self.eval_count >= self.max_evals:
                    break
                ind = self._gen_valid_init(random.Random(self.seed + i))
                population.append(ind)

                rec = await self.evaluate_one(ind, dry_run=dry_run)
                fit = rec["fitness"]
                fitnesses.append(fit)
                self.finalize_ok_proposal(rec)
                self.write_eval_row(rec, generation=0)

                self.record_evaluation(fit, ind)

                status_icon = "\u2705" if rec["status"] == "ok" else "\u274c"
                print(f"  [{self.eval_count:4d}] {status_icon} {rec['status']:<22s} "
                      f"fitness={fit:8.4f}  best={self.best_fitness:.4f}  "
                      f"validate={self.proposal_counter}")

            # 初始种群断点
            self.save_checkpoint(self._checkpoint_save_path, extra_state={
                "population": population,
                "fitnesses": fitnesses,
                "generation": 0,
            })
            start_gen = 1

        # 进化
        for gen in range(start_gen, self.n_gen + 1):
            if self.eval_count >= self.max_evals:
                print(f"[GA] 达到 max_evals={self.max_evals}，停止")
                break
            if self.should_stop_early():
                break
            self.current_generation = gen
            print(f"\n{'='*60}")
            print(f"[GA] === 第 {gen}/{self.n_gen} 代 === "
                  f"eval={self.eval_count}/{self.max_evals} best={self.best_fitness:.4f}")
            print(f"{'='*60}")

            # 精英保留
            elite_idx = sorted(range(len(fitnesses)),
                               key=lambda i: fitnesses[i], reverse=True)[:self.elite_size]
            new_pop = [population[i].copy() for i in elite_idx]
            new_fit = [fitnesses[i] for i in elite_idx]

            while (len(new_pop) < self.pop_size and
                   self.eval_count < self.max_evals and
                   not self.should_stop_early()):
                p1 = self._tournament_select(population, fitnesses, rng)
                p2 = self._tournament_select(population, fitnesses, rng)

                if rng.random() < 0.9:
                    c1, c2 = self._sbx_crossover(p1, p2, rng)
                else:
                    c1, c2 = dict(p1), dict(p2)

                c1 = self._polynomial_mutate(c1, rng)
                c2 = self._polynomial_mutate(c2, rng)

                for child in [c1, c2]:
                    if (len(new_pop) >= self.pop_size or
                            self.eval_count >= self.max_evals or
                            self.should_stop_early()):
                        break
                    if not self.check_constraints(child, source="crossover_mutation"):
                        child = self._resample_or_mutate(child, rng)

                    rec = await self.evaluate_one(child, dry_run=dry_run)
                    fit = rec["fitness"]
                    new_pop.append(child)
                    new_fit.append(fit)
                    self.finalize_ok_proposal(rec)
                    self.write_eval_row(rec, generation=gen)

                    if self.record_evaluation(fit, child):
                        print(f"  [{self.eval_count:4d}] 🎯 NEW BEST fitness={fit:8.4f}")
                    else:
                        status_icon = "✅" if rec["status"] == "ok" else "❌"
                        print(f"  [{self.eval_count:4d}] {status_icon} {rec['status']:<22s} "
                              f"fitness={fit:8.4f}  best={self.best_fitness:.4f}")

            population = new_pop
            fitnesses = new_fit

            # 每代结束自动保存断点
            self.save_checkpoint(self._checkpoint_save_path, extra_state={
                "population": population,
                "fitnesses": fitnesses,
                "generation": gen,
            })

    def _gen_valid_init(self, rng: random.Random) -> Dict[str, float]:
        """生成满足约束的初始个体（失败过程全部记录）"""
        for _ in range(5000):
            p = random_params(rng)
            if self.check_constraints(p, source="init_random"):
                return p
        return random_params(rng)
