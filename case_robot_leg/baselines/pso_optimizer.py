# -*- coding: utf-8 -*-
"""Particle-swarm optimization for the robot-leg task.

The implementation uses inertia-weight decay, cognitive and social terms,
constraint-aware position repair and the common evaluation logger.
"""
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_optimizer import (
    BaseOptimizer, PARAM_NAMES, PARAM_BOUNDS,
    PENALTY_FITNESS, snap, clamp, random_params,
    generate_valid_params, validate_quick,
)


class Particle:
    """PSO 粒子"""

    def __init__(self, position: Dict[str, float], velocity: Dict[str, float]):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_fitness = PENALTY_FITNESS
        self.fitness = PENALTY_FITNESS


class PSOOptimizer(BaseOptimizer):
    """粒子群优化器（机器人腿）"""

    def __init__(
        self,
        max_evals: int = 200,
        swarm_size: int = 20,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        w_decay: float = 0.99,
        v_max_ratio: float = 0.2,
        output_dir: str = ".",
        seed: int = 42,
    ):
        super().__init__(
            max_evals=max_evals,
            output_dir=output_dir,
            algorithm_name="PSO",
            seed=seed,
        )
        self.swarm_size = swarm_size
        self.w = w
        self.w_init = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        self.v_max_ratio = v_max_ratio
        self.n_iterations = max(1, max_evals // swarm_size)

        self.global_best_position: Optional[Dict[str, float]] = None
        self.global_best_fitness = PENALTY_FITNESS

    def _init_swarm(self, rng: random.Random) -> List[Particle]:
        swarm = []
        for i in range(self.swarm_size):
            # 生成满足约束的初始位置
            pos = self._gen_valid_position(random.Random(self.seed + i))
            # 随机速度
            vel = {}
            for name in PARAM_NAMES:
                lo, hi = PARAM_BOUNDS[name]
                vel[name] = rng.uniform(-1, 1) * (hi - lo) * 0.1
            swarm.append(Particle(pos, vel))
        return swarm

    def _gen_valid_position(self, rng: random.Random) -> Dict[str, float]:
        """生成满足约束的位置（失败全部记录）"""
        for _ in range(5000):
            p = random_params(rng)
            if self.check_constraints(p, source="init_random"):
                return p
        return random_params(rng)

    def _repair_position(self, position: Dict[str, float],
                         old_position: Dict[str, float],
                         rng: random.Random) -> Dict[str, float]:
        """修复不满足约束的位置"""
        # 先做边界限制
        repaired = {}
        for name in PARAM_NAMES:
            repaired[name] = clamp(name, position[name])
        repaired = snap(repaired)

        ok, _ = validate_quick(repaired)
        if ok:
            self.log_proposal(repaired, "pso_repair", "ok", [])
            return repaired

        # 在原位置和新位置之间插值
        for _ in range(100):
            alpha = rng.uniform(0.1, 0.9)
            mixed = {}
            for name in PARAM_NAMES:
                mixed[name] = clamp(name, alpha * old_position[name] + (1 - alpha) * repaired[name])
            mixed = snap(mixed)
            ok, _ = validate_quick(mixed)
            if ok:
                self.log_proposal(mixed, "pso_repair", "ok", [])
                return mixed

        # 兜底重采样
        p, _ = generate_valid_params(rng)
        self.log_proposal(p, "pso_resample", "ok", [])
        return p

    def _update_velocity(self, particle: Particle, rng: random.Random):
        for name in PARAM_NAMES:
            lo, hi = PARAM_BOUNDS[name]
            r1 = rng.random()
            r2 = rng.random()

            cognitive = self.c1 * r1 * (particle.best_position[name] - particle.position[name])
            social = self.c2 * r2 * (self.global_best_position[name] - particle.position[name])
            particle.velocity[name] = self.w * particle.velocity[name] + cognitive + social

            v_max = (hi - lo) * self.v_max_ratio
            particle.velocity[name] = max(-v_max, min(v_max, particle.velocity[name]))

    def _update_position(self, particle: Particle, rng: random.Random):
        old_position = particle.position.copy()
        new_position = {}
        for name in PARAM_NAMES:
            lo, hi = PARAM_BOUNDS[name]
            new_val = particle.position[name] + particle.velocity[name]
            if new_val < lo:
                new_val = lo
                particle.velocity[name] *= -0.5
            elif new_val > hi:
                new_val = hi
                particle.velocity[name] *= -0.5
            new_position[name] = new_val
        new_position = snap(new_position)

        # 约束检查
        if not self.check_constraints(new_position, source="pso_move"):
            new_position = self._repair_position(new_position, old_position, rng)

        particle.position = new_position

    async def optimize(self, dry_run: bool = False):
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        print(f"[PSO] swarm={self.swarm_size}, iter={self.n_iterations}, "
              f"w={self.w}, c1={self.c1}, c2={self.c2}")
        print("=" * 60)

        # 断点恢复
        swarm = None
        start_iter = 1
        if hasattr(self, '_checkpoint_save_path') and self._checkpoint_save_path.exists():
            state = self.load_checkpoint(self._checkpoint_save_path)
            if state and "swarm" in state:
                swarm_state = state["swarm"]
                swarm = []
                for s in swarm_state:
                    p = Particle(s["position"], s["velocity"])
                    p.best_position = s["best_position"]
                    p.best_fitness = s["best_fitness"]
                    swarm.append(p)
                self.global_best_position = state.get("global_best_position")
                self.global_best_fitness = state.get("global_best_fitness", PENALTY_FITNESS)
                self.w = state.get("w", self.w)
                start_iter = state.get("iteration", 0) + 1
                print(f"[PSO] 从断点恢复: iter={start_iter-1}, "
                      f"eval_count={self.eval_count}, best={self.best_fitness:.4f}")

        if swarm is None:
            # 初始化粒子群
            swarm = self._init_swarm(rng)

            # 评估初始粒子群
            self.current_generation = 0
            print(f"[PSO] === 初始评估 ({self.swarm_size} 粒子) ===")
            for i, particle in enumerate(swarm):
                if self.eval_count >= self.max_evals:
                    break

                rec = await self.evaluate_one(particle.position, dry_run=dry_run)
                fit = rec["fitness"]
                particle.fitness = fit
                self.finalize_ok_proposal(rec)
                self.write_eval_row(rec, generation=0)

                if fit > particle.best_fitness:
                    particle.best_fitness = fit
                    particle.best_position = particle.position.copy()

                if fit > self.global_best_fitness:
                    self.global_best_fitness = fit
                    self.global_best_position = particle.position.copy()

                self.record_evaluation(fit, particle.position)

                status_icon = "\u2705" if rec["status"] == "ok" else "\u274c"
                print(f"  [{self.eval_count:4d}] {status_icon} fitness={fit:8.4f}  "
                      f"best={self.best_fitness:.4f}")

            if self.global_best_position is None:
                self.global_best_position = swarm[0].position.copy()

            # 保存初始断点
            swarm_state = [{
                "position": p.position, "velocity": p.velocity,
                "best_position": p.best_position, "best_fitness": p.best_fitness,
            } for p in swarm]
            self.save_checkpoint(self._checkpoint_save_path, extra_state={
                "swarm": swarm_state,
                "iteration": 0,
                "global_best_position": self.global_best_position,
                "global_best_fitness": self.global_best_fitness,
                "w": self.w,
            })

        # 迭代优化
        for iteration in range(start_iter, self.n_iterations + 1):
            if self.eval_count >= self.max_evals:
                print(f"[PSO] 达到 max_evals={self.max_evals}，停止")
                break
            if self.should_stop_early():
                break

            self.current_generation = iteration
            self.w = self.w_init * (self.w_decay ** iteration)

            print(f"\n[PSO] === 第 {iteration}/{self.n_iterations} 轮 === "
                  f"w={self.w:.3f} eval={self.eval_count}/{self.max_evals}")

            for particle in swarm:
                if self.eval_count >= self.max_evals or self.should_stop_early():
                    break

                self._update_velocity(particle, rng)
                self._update_position(particle, rng)

                rec = await self.evaluate_one(particle.position, dry_run=dry_run)
                fit = rec["fitness"]
                particle.fitness = fit
                self.finalize_ok_proposal(rec)
                self.write_eval_row(rec, generation=iteration)

                if fit > particle.best_fitness:
                    particle.best_fitness = fit
                    particle.best_position = particle.position.copy()

                if fit > self.global_best_fitness:
                    self.global_best_fitness = fit
                    self.global_best_position = particle.position.copy()

                if self.record_evaluation(fit, particle.position):
                    print(f"  [{self.eval_count:4d}] 🎯 NEW BEST fitness={fit:8.4f}")
                else:
                    status_icon = "✅" if rec["status"] == "ok" else "❌"
                    print(f"  [{self.eval_count:4d}] {status_icon} fitness={fit:8.4f}  "
                          f"best={self.best_fitness:.4f}")

            print(f"[PSO] 第{iteration}轮完成 | global_best={self.global_best_fitness:.4f}")

            # 每轮保存断点
            swarm_state = [{
                "position": p.position, "velocity": p.velocity,
                "best_position": p.best_position, "best_fitness": p.best_fitness,
            } for p in swarm]
            self.save_checkpoint(self._checkpoint_save_path, extra_state={
                "swarm": swarm_state,
                "iteration": iteration,
                "global_best_position": self.global_best_position,
                "global_best_fitness": self.global_best_fitness,
                "w": self.w,
            })
