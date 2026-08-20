"""
经验缓冲模块

功能：
1. 记录每轮交互的状态、动作、结果
2. 检索相似经验
3. 分析成功/失败模式
"""

import json
import os
import time
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from loguru import logger


@dataclass
class Experience:
    """单条经验记录"""
    id: str
    iteration: int
    
    # 状态：参数快照
    state: Dict[str, float]  # lm, tm, ta, dg, hs, wslot, hslot, s, wa
    
    # 动作：参数调整
    action: Dict[str, float]  # 相对于前一状态的变化量
    
    # 结果
    result_status: str  # ok, constraint_violation, simulation_failed
    fitness: Optional[float] = None
    avg_B: Optional[float] = None
    kb: Optional[float] = None
    pb: Optional[float] = None
    n1: Optional[int] = None
    n2: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    
    # 奖励
    reward: float = 0.0
    
    # ★新增：fitness 是否改善（用于 ExpeL 对比学习）
    # None = 未知（第一轮或未计算），True = 改善，False = 不变或变差
    fitness_improved: Optional[bool] = None
    prev_fitness: Optional[float] = None  # 上一轮 fitness，用于计算改善
    
    # 元信息
    timestamp: float = field(default_factory=time.time)
    llm_reasoning: str = ""  # 大模型的思考过程摘要
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experience":
        # 兼容旧数据：如果没有 fitness_improved 字段，设为 None
        if "fitness_improved" not in d:
            d["fitness_improved"] = None
        if "prev_fitness" not in d:
            d["prev_fitness"] = None
        return cls(**d)
    
    @property
    def is_success(self) -> bool:
        """
        ★优化后的成功定义：
        1. 仿真必须成功执行（status=ok）
        2. 且 fitness 必须有改善（不降反升视为失败）
        
        对于 ExpeL 对比学习，这能产生更多有意义的"失败"经验
        """
        if self.result_status != "ok":
            return False
        
        # 如果有 fitness_improved 标记，使用它
        if self.fitness_improved is not None:
            return self.fitness_improved
        
        # 兼容旧数据：只检查 status
        return True


class ExperienceBuffer:
    """经验回放缓冲区"""
    
    PARAM_NAMES = ["lm", "tm", "ta", "dg", "hs", "wslot", "hslot", "s", "wa"]
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_size: int = 500,
        auto_save_interval: int = 10
    ):
        self.storage_path = storage_path or "experience_buffer.json"
        self.max_size = max_size
        self.auto_save_interval = auto_save_interval
        self.experiences: List[Experience] = []
        self._unsaved_count = 0
        self._load()
    
    def _load(self):
        """从文件加载经验"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.experiences = [Experience.from_dict(d) for d in data]
                logger.info(f"已加载 {len(self.experiences)} 条历史经验")
            except Exception as e:
                logger.warning(f"加载经验缓冲失败: {e}")
                self.experiences = []
    
    def _save(self, force: bool = False):
        """保存经验到文件"""
        if not force and self._unsaved_count < self.auto_save_interval:
            return
        try:
            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([exp.to_dict() for exp in self.experiences], f,
                         ensure_ascii=False, indent=2)
            self._unsaved_count = 0
        except Exception as e:
            logger.warning(f"保存经验缓冲失败: {e}")
    
    def store(
        self,
        iteration: int,
        state: Dict[str, float],
        action: Dict[str, float],
        result: Dict[str, Any],
        reward: float,
        llm_reasoning: str = "",
        prev_fitness: Optional[float] = None  # ★新增：上一轮 fitness，用于计算改善
    ) -> Experience:
        """存储一条经验
        
        Args:
            prev_fitness: 上一轮的 fitness 值。如果提供，将计算 fitness_improved：
                         - True: fitness 降低（改善，因为 fitness 越小越好）
                         - False: fitness 不变或升高（未改善/变差）
                         - None: 第一轮或未提供
        """
        exp_id = f"exp_{int(time.time() * 1000)}_{iteration}"
        
        # ★计算 fitness 是否改善（fitness 越小越好）
        current_fitness = result.get("fitness")
        fitness_improved = None
        if current_fitness is not None and prev_fitness is not None:
            # fitness 降低了才算改善
            fitness_improved = current_fitness < prev_fitness
        
        experience = Experience(
            id=exp_id,
            iteration=iteration,
            state=state,
            action=action,
            result_status=result.get("status", "unknown"),
            fitness=current_fitness,
            avg_B=result.get("avg_B"),
            kb=result.get("kb"),
            pb=result.get("pb"),
            n1=result.get("turns", {}).get("n1") if isinstance(result.get("turns"), dict) else None,
            n2=result.get("turns", {}).get("n2") if isinstance(result.get("turns"), dict) else None,
            errors=result.get("errors", []),
            reward=reward,
            fitness_improved=fitness_improved,  # ★新增
            prev_fitness=prev_fitness,  # ★新增
            llm_reasoning=llm_reasoning
        )
        
        self.experiences.append(experience)
        self._unsaved_count += 1
        
        # 限制大小，移除最旧的低价值经验
        if len(self.experiences) > self.max_size:
            self._prune()
        
        self._save()
        return experience
    
    def _prune(self):
        """修剪缓冲区，保留高价值经验"""
        # 始终保留成功的经验
        successful = [exp for exp in self.experiences if exp.is_success]
        failed = [exp for exp in self.experiences if not exp.is_success]
        
        # 成功经验按 fitness 排序（假设越小越好）
        successful.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'))
        
        # 失败经验按时间排序，保留最新的
        failed.sort(key=lambda x: x.timestamp, reverse=True)
        
        # 保留策略：成功的占 60%，失败的占 40%
        max_success = int(self.max_size * 0.6)
        max_failed = self.max_size - min(len(successful), max_success)
        
        self.experiences = successful[:max_success] + failed[:max_failed]
    
    def _param_distance(self, state1: Dict[str, float], state2: Dict[str, float]) -> float:
        """计算两个参数状态的归一化距离"""
        # 参数的典型范围（用于归一化）
        ranges = {
            "lm": 6.0, "tm": 0.1, "ta": 0.4, "dg": 0.4,
            "hs": 1.0, "wslot": 0.8, "hslot": 0.5, "s": 0.4, "wa": 1.0
        }
        
        dist_sq = 0.0
        for param in self.PARAM_NAMES:
            v1 = state1.get(param, 0)
            v2 = state2.get(param, 0)
            r = ranges.get(param, 1.0)
            dist_sq += ((v1 - v2) / r) ** 2
        
        return math.sqrt(dist_sq)
    
    def retrieve_similar(
        self,
        current_state: Dict[str, float],
        k: int = 5,
        only_success: bool = False
    ) -> List[Experience]:
        """检索与当前状态相似的经验"""
        candidates = self.experiences
        if only_success:
            candidates = [exp for exp in candidates if exp.is_success]
        
        if not candidates:
            return []
        
        # 按距离排序
        scored = [
            (exp, self._param_distance(current_state, exp.state))
            for exp in candidates
        ]
        scored.sort(key=lambda x: x[1])
        
        return [exp for exp, _ in scored[:k]]
    
    def retrieve_best(self, k: int = 5) -> List[Experience]:
        """检索最佳（fitness 最小）的成功经验"""
        successful = [exp for exp in self.experiences if exp.is_success and exp.fitness is not None]
        successful.sort(key=lambda x: x.fitness)
        return successful[:k]
    
    def retrieve_contrast_pair(
        self,
        current_state: Dict[str, float],
        current_is_success: bool,
        distance_threshold: float = 1.5
    ) -> Optional[Experience]:
        """
        ExpeL 对比学习：找与当前状态相似但结果相反的经验
        
        Args:
            current_state: 当前参数状态
            current_is_success: 当前是否成功
            distance_threshold: 距离阈值，小于此值认为相似
        
        Returns:
            相似但结果相反的经验，如果没有则返回 None
        """
        # 找结果相反的经验
        if current_is_success:
            # 当前成功，找相似的失败
            candidates = [exp for exp in self.experiences if not exp.is_success]
        else:
            # 当前失败，找相似的成功
            candidates = [exp for exp in self.experiences if exp.is_success]
        
        if not candidates:
            return None
        
        # 按距离排序，找最相似的
        scored = [
            (exp, self._param_distance(current_state, exp.state))
            for exp in candidates
        ]
        scored.sort(key=lambda x: x[1])
        
        # 返回最相似且距离在阈值内的
        if scored and scored[0][1] < distance_threshold:
            return scored[0][0]
        
        return None
    
    def retrieve_contrast_pairs_batch(
        self,
        k: int = 5
    ) -> List[Tuple["Experience", "Experience"]]:
        """
        批量获取成功/失败配对（用于批量对比批评）
        
        Returns:
            [(成功经验, 失败经验), ...] 配对列表
        """
        successful = [exp for exp in self.experiences if exp.is_success]
        failed = [exp for exp in self.experiences if not exp.is_success]
        
        if not successful or not failed:
            return []
        
        pairs = []
        used_fail_ids = set()
        
        for succ_exp in successful[:k*2]:  # 多取一些，避免配对失败
            best_fail = None
            best_dist = float('inf')
            
            for fail_exp in failed:
                if fail_exp.id in used_fail_ids:
                    continue
                dist = self._param_distance(succ_exp.state, fail_exp.state)
                if dist < best_dist:
                    best_dist = dist
                    best_fail = fail_exp
            
            if best_fail and best_dist < 2.0:  # 距离阈值
                pairs.append((succ_exp, best_fail))
                used_fail_ids.add(best_fail.id)
            
            if len(pairs) >= k:
                break
        
        return pairs
    
    def retrieve_recent(self, k: int = 10) -> List[Experience]:
        """检索最近的经验"""
        sorted_exp = sorted(self.experiences, key=lambda x: x.timestamp, reverse=True)
        return sorted_exp[:k]
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """分析成功/失败模式"""
        if not self.experiences:
            return {"success_rate": 0, "patterns": []}
        
        successful = [exp for exp in self.experiences if exp.is_success]
        failed = [exp for exp in self.experiences if not exp.is_success]
        
        success_rate = len(successful) / len(self.experiences)
        
        # 分析成功经验的参数分布
        success_param_stats = {}
        if successful:
            for param in self.PARAM_NAMES:
                values = [exp.state.get(param, 0) for exp in successful]
                success_param_stats[param] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        # 分析常见错误
        error_counts: Dict[str, int] = {}
        for exp in failed:
            for err in exp.errors:
                # 简化错误消息
                key = err[:50] if len(err) > 50 else err
                error_counts[key] = error_counts.get(key, 0) + 1
        
        top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_experiences": len(self.experiences),
            "success_count": len(successful),
            "failure_count": len(failed),
            "success_rate": success_rate,
            "success_param_stats": success_param_stats,
            "top_errors": top_errors,
            "best_fitness": min((exp.fitness for exp in successful if exp.fitness), default=None)
        }
    
    def build_experience_context(
        self,
        current_state: Dict[str, float],
        include_similar: int = 2,
        include_best: int = 2,
        strategy_only: bool = False,
        transfer_mode: str = None
    ) -> str:
        """构建经验上下文，用于注入提示词
        
        Args:
            strategy_only: 若为 True（经验迁移场景），只输出结论性语句，不输出具体参数
            transfer_mode: 迁移模式 - 'distilled' 时跳过旧任务参数经验，仅保留统计信息
        """
        # 纯蒸馏模式：不注入源任务的具体参数经验（它们属于另一个结构）
        if transfer_mode == "distilled":
            return self._build_distilled_experience_context()
        
        lines = []
        
        # 最佳经验
        best = self.retrieve_best(k=include_best)
        if best:
            lines.append("## 历史最佳设计参考")
            for i, exp in enumerate(best, 1):
                if strategy_only:
                    lines.append(f"{i}. fitness={exp.fitness:.4f}, n1={exp.n1}, n2={exp.n2}（结论：该区域可行且较优）")
                else:
                    params_str = ", ".join(f"{k}={v:.2f}" for k, v in exp.state.items())
                    lines.append(f"{i}. fitness={exp.fitness:.4f}, n1={exp.n1}, n2={exp.n2}")
                    lines.append(f"   参数: {params_str}")
        
        # 相似经验
        if not strategy_only:
            similar = self.retrieve_similar(current_state, k=include_similar, only_success=True)
            if similar:
                lines.append("\n## 相似成功案例")
                for i, exp in enumerate(similar, 1):
                    lines.append(f"{i}. fitness={exp.fitness:.4f} (状态相似)")
        
        # 统计摘要（结论性，保留）
        patterns = self.analyze_patterns()
        if patterns["success_rate"] > 0:
            lines.append(f"\n## 优化统计")
            lines.append(f"- 成功率: {patterns['success_rate']*100:.1f}%")
            if patterns["best_fitness"] is not None:
                lines.append(f"- 历史最佳 fitness: {patterns['best_fitness']:.4f}")
        
        # 常见错误提醒（结论性，保留）
        if patterns.get("top_errors"):
            lines.append("\n## 常见约束违规（请避免）")
            for err, cnt in patterns["top_errors"][:3]:
                lines.append(f"- {err} (出现 {cnt} 次)")
        
        return "\n".join(lines) if lines else ""
    
    def _build_distilled_experience_context(self) -> str:
        """蒸馏模式下的经验上下文：仅输出结构无关的统计信息"""
        lines = []
        
        # 仅保留本轮（非迁移）的经验
        current_run_exps = [e for e in self.experiences if getattr(e, '_from_current_run', True)]
        
        if current_run_exps:
            success = [e for e in current_run_exps if e.result_status == "ok" and e.fitness is not None]
            if success:
                best = min(success, key=lambda e: e.fitness)
                lines.append("## 本轮优化进展")
                lines.append(f"- 已完成 {len(current_run_exps)} 轮，成功 {len(success)} 轮")
                lines.append(f"- 当前最佳 fitness: {best.fitness:.4f}")
        
        # 常见约束违规仍然有用（结构无关）
        patterns = self.analyze_patterns()
        if patterns.get("top_errors"):
            lines.append("\n## 常见约束违规（请避免）")
            for err, cnt in patterns["top_errors"][:3]:
                lines.append(f"- {err} (出现 {cnt} 次)")
        
        return "\n".join(lines) if lines else ""
    
    def get_param_sensitivity(self) -> Dict[str, float]:
        """估计参数敏感性（fitness 对参数变化的响应）"""
        if len(self.experiences) < 10:
            return {}
        
        successful = [exp for exp in self.experiences if exp.is_success and exp.fitness is not None]
        if len(successful) < 5:
            return {}
        
        sensitivities = {}
        for param in self.PARAM_NAMES:
            values = [(exp.state.get(param, 0), exp.fitness) for exp in successful]
            if len(values) < 5:
                continue
            
            # 简单相关性估计
            x_mean = sum(v[0] for v in values) / len(values)
            y_mean = sum(v[1] for v in values) / len(values)
            
            num = sum((v[0] - x_mean) * (v[1] - y_mean) for v in values)
            den_x = sum((v[0] - x_mean) ** 2 for v in values)
            den_y = sum((v[1] - y_mean) ** 2 for v in values)
            
            if den_x > 0 and den_y > 0:
                correlation = num / math.sqrt(den_x * den_y)
                sensitivities[param] = abs(correlation)
        
        return sensitivities
    
    def save(self):
        """强制保存"""
        self._save(force=True)

