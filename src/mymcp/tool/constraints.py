import json
from typing import Tuple, Optional

from maxwell_pyaedt_run import ActuatorDesignVariables, WA_FIXED


def _build_design_kwargs(lm,
                         tm,
                         ta,
                         dg,
                         hs,
                         wslot,
                         hslot,
                         s,
                         tb_ratio: float) -> Tuple[ActuatorDesignVariables, dict]:
    """构建设计参数字典，wa 固定为 2.0"""
    kwargs = {
        "lm": lm,
        "tm": tm,
        "ta": ta,
        "dg": dg,
        "hs": hs,
        "wslot": wslot,
        "hslot": hslot,
        "s": s,
        "tb_ratio": tb_ratio,
        "wa": WA_FIXED,  # 固定为 2.0
    }
    design = ActuatorDesignVariables(**kwargs)
    return design, kwargs


async def validate_maxwell_design(
        lm: float,
        tm: float,
        ta: float,
        dg: float,
        hs: float,
        wslot: float,
        hslot: float,
        s: float,
        tb_ratio: float,  # 自由变量，LLM 必须传入，范围 [1.6, 2.0]
) -> str:
    """检查一组 Maxwell 几何参数是否满足全部前置约束。"""
    design, kwargs = _build_design_kwargs(lm, tm, ta, dg, hs, wslot,
                                          hslot, s, tb_ratio)
    errors = design.validate_without_B()
    status = "ok" if not errors else "constraint_violation"
    payload = {
        "status": status,
        "errors": errors,
        "design": kwargs,
        "derived": {
            "s": design.s,
            "hslot": design.hslot,
            "n1": design.n1,
            "n2": design.n2,
            "total_turns": design.total_turns,
            "la": design.la,
            "ha": design.ha,
            "ws": design.ws,
            "ls": design.ls,
            "tb": design.tb,
            "twall": design.twall,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
