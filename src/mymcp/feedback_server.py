"""
反馈 Web 服务

启动方式：
    python -m mymcp.feedback_server --port 8888

访问 http://localhost:8888 即可使用 Web 界面提交反馈
"""

import os
import time
import json
import argparse
from pathlib import Path

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("请先安装依赖: pip install fastapi uvicorn")
    exit(1)

# 反馈文件路径（相对于项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_INPUT_FILE = PROJECT_ROOT / "feedback_input.txt"
FEEDBACK_STORAGE_FILE = PROJECT_ROOT / "feedback_storage.json"
STRATEGY_STATE_FILE = PROJECT_ROOT / "strategy_state.json"
EXPERIENCE_BUFFER_FILE = PROJECT_ROOT / "experience_buffer.json"
META_KNOWLEDGE_FILE = PROJECT_ROOT / "meta_knowledge.json"

# 评论家相关文件路径
CRITIC_EXPERIENCE_DIR = PROJECT_ROOT / "critic_experience"
VALUE_FUNCTION_FILE = PROJECT_ROOT / "value_function_history.json"

app = FastAPI(title="Maxwell 优化反馈系统")


# ========== API 端点 ==========

@app.post("/api/feedback")
async def add_feedback(request: Request):
    """添加反馈"""
    try:
        data = await request.json()
        text = data.get("text", "").strip()
        fb_type = data.get("type", "suggestion")
        priority = data.get("priority", 2)
        urgent = data.get("urgent", False)
        
        if not text:
            return JSONResponse({"success": False, "error": "反馈内容不能为空"}, status_code=400)
        
        # 构建反馈行
        line = ""
        if urgent:
            line += "!"
        if fb_type != "suggestion":
            line += f"[{fb_type}] "
        line += text
        
        # 追加到文件
        with open(FEEDBACK_INPUT_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        
        return JSONResponse({
            "success": True,
            "message": f"反馈已添加，将在下轮迭代生效",
            "feedback": line
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/status")
async def get_status():
    """获取当前状态"""
    status = {
        "pending_feedbacks": [],
        "stored_feedbacks": [],
        "experience_count": 0,
        "best_fitness": None
    }
    
    # 读取待处理反馈
    if FEEDBACK_INPUT_FILE.exists():
        with open(FEEDBACK_INPUT_FILE, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
            status["pending_feedbacks"] = lines
    
    # 读取已存储反馈
    if FEEDBACK_STORAGE_FILE.exists():
        try:
            with open(FEEDBACK_STORAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                status["stored_feedbacks"] = data[-10:]  # 最近 10 条
        except:
            pass
    
    # 读取经验统计
    exp_file = PROJECT_ROOT / "experience_buffer.json"
    if exp_file.exists():
        try:
            with open(exp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                status["experience_count"] = len(data)
                # 找最佳 fitness
                ok_results = [e for e in data if e.get("result_status") == "ok" and e.get("fitness")]
                if ok_results:
                    status["best_fitness"] = min(e["fitness"] for e in ok_results)
        except:
            pass
    
    return JSONResponse(status)


@app.delete("/api/feedback")
async def clear_pending():
    """清空待处理反馈"""
    with open(FEEDBACK_INPUT_FILE, "w", encoding="utf-8") as f:
        f.write("")
    return JSONResponse({"success": True, "message": "待处理反馈已清空"})


@app.get("/api/feedback/history")
async def get_feedback_history():
    """获取所有历史反馈"""
    feedbacks = []
    if FEEDBACK_STORAGE_FILE.exists():
        try:
            with open(FEEDBACK_STORAGE_FILE, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
        except:
            pass
    return JSONResponse({"feedbacks": feedbacks})


@app.put("/api/feedback/{feedback_id}")
async def update_feedback(feedback_id: str, request: Request):
    """更新指定反馈"""
    try:
        data = await request.json()
        new_text = data.get("text", "").strip()
        if not new_text:
            return JSONResponse({"success": False, "error": "反馈内容不能为空"}, status_code=400)
        
        feedbacks = []
        if FEEDBACK_STORAGE_FILE.exists():
            with open(FEEDBACK_STORAGE_FILE, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
        
        # 查找并更新
        found = False
        for fb in feedbacks:
            if fb.get("id") == feedback_id:
                fb["text"] = new_text
                fb["feedback_type"] = data.get("type", fb.get("feedback_type", "suggestion"))
                fb["priority"] = data.get("priority", fb.get("priority", 2))
                # ★新增：更新强度
                if "strength" in data:
                    fb["strength"] = max(0.0, min(1.0, float(data["strength"])))
                found = True
                break
        
        if not found:
            return JSONResponse({"success": False, "error": "未找到该反馈"}, status_code=404)
        
        with open(FEEDBACK_STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)
        
        return JSONResponse({"success": True, "message": "反馈已更新"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.patch("/api/feedback/{feedback_id}/strength")
async def update_feedback_strength(feedback_id: str, request: Request):
    """单独更新反馈的引导强度"""
    try:
        data = await request.json()
        strength = data.get("strength", 1.0)
        strength = max(0.0, min(1.0, float(strength)))
        
        feedbacks = []
        if FEEDBACK_STORAGE_FILE.exists():
            with open(FEEDBACK_STORAGE_FILE, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
        
        found = False
        for fb in feedbacks:
            if fb.get("id") == feedback_id:
                fb["strength"] = strength
                found = True
                break
        
        if not found:
            return JSONResponse({"success": False, "error": "未找到该反馈"}, status_code=404)
        
        with open(FEEDBACK_STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)
        
        return JSONResponse({
            "success": True, 
            "message": f"引导强度已更新为 {strength:.0%}",
            "strength": strength
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.delete("/api/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    """删除指定反馈"""
    try:
        feedbacks = []
        if FEEDBACK_STORAGE_FILE.exists():
            with open(FEEDBACK_STORAGE_FILE, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
        
        original_len = len(feedbacks)
        feedbacks = [fb for fb in feedbacks if fb.get("id") != feedback_id]
        
        if len(feedbacks) == original_len:
            return JSONResponse({"success": False, "error": "未找到该反馈"}, status_code=404)
        
        with open(FEEDBACK_STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)
        
        return JSONResponse({"success": True, "message": "反馈已删除"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/strategy")
async def get_strategy():
    """获取完整策略状态"""
    strategy = {
        "epsilon": 0.3,
        "iteration": 0,
        "recent_success_rate": 0,
        "recent_results": [],
        "last_direction": {},
        "success_regions": [],
        "success_regions_count": 0,
        "success_patterns": [],
        "success_patterns_count": 0,
        "failure_patterns": [],
        "failure_patterns_count": 0,
        "param_sensitivity": {},
        "prompt_additions": [],
        "learned_rules": [],
        "config": {}
    }
    
    if STRATEGY_STATE_FILE.exists():
        try:
            with open(STRATEGY_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                strategy["epsilon"] = data.get("epsilon", 0.3)
                strategy["iteration"] = data.get("iteration", 0)
                strategy["recent_results"] = data.get("recent_results", [])[-20:]
                strategy["last_direction"] = data.get("last_direction", {})
                strategy["success_regions"] = data.get("success_regions", [])[-10:]
                strategy["success_regions_count"] = len(data.get("success_regions", []))
                
                # 成功模式
                strategy["success_patterns"] = data.get("success_patterns", [])[-10:]
                strategy["success_patterns_count"] = len(data.get("success_patterns", []))
                
                # 失败模式
                strategy["failure_patterns"] = data.get("failure_patterns", [])[-10:]
                strategy["failure_patterns_count"] = len(data.get("failure_patterns", []))
                
                # 参数敏感性
                strategy["param_sensitivity"] = data.get("param_sensitivity", {})
                
                # 动态提示词和学习规则
                strategy["prompt_additions"] = data.get("prompt_additions", [])
                strategy["learned_rules"] = data.get("learned_rules", [])
                
                strategy["config"] = data.get("config", {})
                
                # 计算成功率
                results = strategy["recent_results"]
                if results:
                    strategy["recent_success_rate"] = sum(results) / len(results)
        except:
            pass
    
    return JSONResponse(strategy)


@app.delete("/api/strategy")
async def reset_strategy():
    """重置策略状态"""
    try:
        if STRATEGY_STATE_FILE.exists():
            os.remove(STRATEGY_STATE_FILE)
        return JSONResponse({"success": True, "message": "策略状态已重置"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/experience")
async def get_experience():
    """获取经验数据"""
    experiences = []
    stats = {
        "total": 0,
        "success": 0,
        "failure": 0,
        "best_fitness": None,
        "worst_fitness": None,
        "avg_fitness": None
    }
    
    if EXPERIENCE_BUFFER_FILE.exists():
        try:
            with open(EXPERIENCE_BUFFER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                stats["total"] = len(data)
                
                fitnesses = []
                for exp in data:
                    if exp.get("result_status") == "ok":
                        stats["success"] += 1
                        if exp.get("fitness"):
                            fitnesses.append(exp["fitness"])
                    else:
                        stats["failure"] += 1
                
                if fitnesses:
                    stats["best_fitness"] = min(fitnesses)
                    stats["worst_fitness"] = max(fitnesses)
                    stats["avg_fitness"] = sum(fitnesses) / len(fitnesses)
                
                # 返回最近 20 条
                experiences = data[-20:]
        except:
            pass
    
    return JSONResponse({"experiences": experiences, "stats": stats})


@app.delete("/api/experience")
async def clear_experience():
    """清空经验数据"""
    try:
        if EXPERIENCE_BUFFER_FILE.exists():
            os.remove(EXPERIENCE_BUFFER_FILE)
        return JSONResponse({"success": True, "message": "经验数据已清空"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/meta-knowledge")
async def get_meta_knowledge():
    """获取元学习知识库数据"""
    result = {
        "summary": {
            "total_rules": 0,
            "total_patterns": 0,
            "llm_rules": 0,
            "high_confidence_rules": 0,
            "rules_by_type": {},
            "rules_by_category": {},
        },
        "rules": {
            "monotonic": [],
            "optimal_range": [],
            "correlation": [],
            "constraint": [],
            "llm_insight": [],
        },
        "patterns": [],
        "domain_insights": {
            "optimization_insights": [],
            "transfer_suggestions": [],
        },
        "updated_at": None,
    }
    
    if not META_KNOWLEDGE_FILE.exists():
        return JSONResponse(result)
    
    try:
        with open(META_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        abstract_rules = data.get("abstract_rules", {})
        meta_patterns = data.get("meta_patterns", {})
        domain_statistics = data.get("domain_statistics", {})
        
        # 统计摘要
        rules_by_type = {}
        rules_by_category = {}
        high_conf = 0
        llm_count = 0
        
        for rule_id, rule in abstract_rules.items():
            rule_type = rule.get("rule_type", "unknown")
            category = rule.get("param_category", "unknown")
            confidence = rule.get("confidence", 0)
            
            rules_by_type[rule_type] = rules_by_type.get(rule_type, 0) + 1
            rules_by_category[category] = rules_by_category.get(category, 0) + 1
            
            if confidence > 0.6:
                high_conf += 1
            if rule_type == "llm_insight":
                llm_count += 1
            
            # 分类存储规则
            rule_data = {
                "id": rule_id,
                "type": rule_type,
                "category": category,
                "role": rule.get("param_role", ""),
                "direction": rule.get("direction", ""),
                "effect": rule.get("effect", ""),
                "optimal_range": rule.get("optimal_range"),
                "confidence": confidence,
                "sample_count": rule.get("sample_count", 0),
                "context": rule.get("context", ""),
                "created_at": rule.get("created_at", ""),
            }
            
            if rule_type == "monotonic_effect":
                result["rules"]["monotonic"].append(rule_data)
            elif rule_type == "optimal_range":
                result["rules"]["optimal_range"].append(rule_data)
            elif rule_type == "correlation":
                result["rules"]["correlation"].append(rule_data)
            elif rule_type == "constraint":
                result["rules"]["constraint"].append(rule_data)
            elif rule_type == "llm_insight":
                # 解析 LLM 规则的 context JSON
                try:
                    context_data = json.loads(rule.get("context", "{}"))
                    rule_data["rule_name"] = context_data.get("rule_name", "")
                    rule_data["description"] = context_data.get("description", "")
                    rule_data["physics_reason"] = context_data.get("physics_reason", "")
                    rule_data["recommendations"] = context_data.get("recommendations", [])
                    rule_data["anti_patterns"] = context_data.get("anti_patterns", [])
                except:
                    pass
                result["rules"]["llm_insight"].append(rule_data)
        
        result["summary"]["total_rules"] = len(abstract_rules)
        result["summary"]["total_patterns"] = len(meta_patterns)
        result["summary"]["llm_rules"] = llm_count
        result["summary"]["high_confidence_rules"] = high_conf
        result["summary"]["rules_by_type"] = rules_by_type
        result["summary"]["rules_by_category"] = rules_by_category
        
        # 元模式
        for pattern_id, pattern in meta_patterns.items():
            result["patterns"].append({
                "id": pattern_id,
                "description": pattern.get("description", ""),
                "applicable_domains": pattern.get("applicable_domains", []),
                "conditions": pattern.get("conditions", []),
                "recommendations": pattern.get("recommendations", []),
                "anti_patterns": pattern.get("anti_patterns", []),
                "confidence": pattern.get("confidence", 0),
                "source_rules": pattern.get("source_rules", []),
            })
        
        # 领域洞察（取第一个有数据的领域）
        for domain, stats in domain_statistics.items():
            if stats.get("optimization_insights") or stats.get("transfer_suggestions"):
                result["domain_insights"]["optimization_insights"] = stats.get("optimization_insights", [])
                result["domain_insights"]["transfer_suggestions"] = stats.get("transfer_suggestions", [])
                break
        
        result["updated_at"] = data.get("updated_at")
        
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), **result}, status_code=500)


@app.get("/api/expel-rules")
async def get_expel_rules():
    """获取元学习规则（ExpeL + Reflexion）"""
    result = {
        "summary": {
            "total_rules": 0,
            "avg_confidence": 0,
            "max_confidence": 0,
            "sources": {"compare": 0, "success": 0, "failure": 0, "reflection": 0},
        },
        "rules": [],
        "updated_at": None,
    }
    
    all_rules = []
    
    # 1. 读取 ExpeL 规则（如果有）
    expel_file = Path("expel_rules.json")
    if expel_file.exists():
        try:
            with open(expel_file, "r", encoding="utf-8") as f:
                expel_data = json.load(f)
            for rule in expel_data:
                all_rules.append({
                    "text": rule.get("text", ""),
                    "confidence": rule.get("confidence", 2),
                    "source": rule.get("source", "compare"),
                    "created_at": rule.get("created_at"),
                    "last_updated": rule.get("last_updated"),
                })
            result["updated_at"] = expel_file.stat().st_mtime
        except Exception as e:
            pass
    
    # 2. 读取 Reflexion 规则（从 memory_stream.jsonl）
    if MEMORY_STREAM_FILE.exists():
        try:
            with open(MEMORY_STREAM_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("event_type") == "rule":
                            # 转换为统一格式
                            all_rules.append({
                                "text": event.get("content", ""),
                                "confidence": int(event.get("importance", 7)),  # 用 importance 作为置信度
                                "source": "reflection",
                                "created_at": event.get("timestamp"),
                                "last_updated": event.get("timestamp"),
                                "round": event.get("round"),
                            })
                    except:
                        pass
            # 更新时间取最新的
            ms_mtime = MEMORY_STREAM_FILE.stat().st_mtime
            if result["updated_at"] is None or ms_mtime > result["updated_at"]:
                result["updated_at"] = ms_mtime
        except Exception as e:
            pass
    
    if not all_rules:
        return JSONResponse(result)
    
    # 去重：相似规则只保留置信度最高的
    seen_texts = {}
    for rule in all_rules:
        text_key = rule["text"][:50].lower()  # 用前50字符作为key
        if text_key not in seen_texts or rule["confidence"] > seen_texts[text_key]["confidence"]:
            seen_texts[text_key] = rule
    
    unique_rules = list(seen_texts.values())
    
    # 按置信度排序
    unique_rules.sort(key=lambda r: r.get("confidence", 0), reverse=True)
    
    # 统计
    total = len(unique_rules)
    confidences = [r.get("confidence", 0) for r in unique_rules]
    sources = {"compare": 0, "success": 0, "failure": 0, "reflection": 0}
    
    for rule in unique_rules:
        src = rule.get("source", "unknown")
        if src in sources:
            sources[src] += 1
    
    result["summary"] = {
        "total_rules": total,
        "avg_confidence": sum(confidences) / total if total else 0,
        "max_confidence": max(confidences) if confidences else 0,
        "sources": sources,
    }
    
    # 规则列表
    for i, rule in enumerate(unique_rules):
        result["rules"].append({
            "index": i + 1,
            "text": rule.get("text", ""),
            "confidence": rule.get("confidence", 0),
            "source": rule.get("source", "unknown"),
            "created_at": rule.get("created_at"),
            "last_updated": rule.get("last_updated"),
            "round": rule.get("round"),
        })
    
    return JSONResponse(result)


@app.delete("/api/expel-rules")
async def clear_expel_rules():
    """清空 ExpeL 规则"""
    expel_file = Path("expel_rules.json")
    if expel_file.exists():
        expel_file.unlink()
    return JSONResponse({"success": True, "message": "ExpeL规则已清空"})


@app.get("/api/critic")
async def get_critic_status():
    """获取评论家集群状态"""
    result = {
        "critics": {},
        "td_learning": {
            "ensemble_accuracy": 0.0,
            "total_td_rules": 0
        },
        "value_function": {
            "total_states": 0,
            "best_fitness": None,
            "avg_value": 0.0
        },
        "ensemble_experiences": [],
        "updated_at": None
    }
    
    # 读取各评论家的策略和 TD 统计
    critic_types = ["magnetic", "performance", "constraint", "magnitude"]
    
    for critic_type in critic_types:
        critic_info = {
            "model": "-",
            "enabled": False,
            "experience_count": 0,
            "strategy_count": 0,
            "strategies": [],
            "dynamic_confidence": 0.5,
            "prediction_accuracy": 0.5,
            "td_statistics": {},
            "total_predictions": 0,
            "correct_predictions": 0
        }
        
        # 读取经验文件
        exp_file = CRITIC_EXPERIENCE_DIR / f"{critic_type}_experience.json"
        if exp_file.exists():
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    experiences = json.load(f)
                    critic_info["experience_count"] = len(experiences)
                    critic_info["enabled"] = True
            except:
                pass
        
        # 读取策略文件
        strategy_file = CRITIC_EXPERIENCE_DIR / f"{critic_type}_strategy.json"
        if strategy_file.exists():
            try:
                with open(strategy_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    strategies = data.get("strategies", [])
                    critic_info["strategy_count"] = len(strategies)
                    critic_info["strategies"] = strategies[-10:]  # 最近10条
                    critic_info["dynamic_confidence"] = data.get("dynamic_confidence", 0.5)
                    critic_info["total_predictions"] = data.get("total_predictions", 0)
                    critic_info["correct_predictions"] = data.get("correct_predictions", 0)
                    if critic_info["total_predictions"] > 0:
                        critic_info["prediction_accuracy"] = critic_info["correct_predictions"] / critic_info["total_predictions"]
            except:
                pass
        
        # 读取 TD 统计文件
        td_file = CRITIC_EXPERIENCE_DIR / f"{critic_type}_td_stats.json"
        if td_file.exists():
            try:
                with open(td_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    critic_info["td_statistics"] = data.get("statistics", {})
                    critic_info["dynamic_confidence"] = data.get("dynamic_confidence", critic_info["dynamic_confidence"])
                    critic_info["total_predictions"] = data.get("total_predictions", critic_info["total_predictions"])
                    critic_info["correct_predictions"] = data.get("correct_predictions", critic_info["correct_predictions"])
                    if critic_info["total_predictions"] > 0:
                        critic_info["prediction_accuracy"] = critic_info["correct_predictions"] / critic_info["total_predictions"]
            except:
                pass
        
        result["critics"][critic_type] = critic_info
    
    # 计算集群整体统计
    total_correct = sum(c["correct_predictions"] for c in result["critics"].values())
    total_predictions = sum(c["total_predictions"] for c in result["critics"].values())
    total_td_rules = sum(
        len([s for s in c["strategies"] if "TD误差" in s])
        for c in result["critics"].values()
    )
    
    if total_predictions > 0:
        result["td_learning"]["ensemble_accuracy"] = total_correct / total_predictions
    result["td_learning"]["total_td_rules"] = total_td_rules
    
    # 读取价值函数历史
    if VALUE_FUNCTION_FILE.exists():
        try:
            with open(VALUE_FUNCTION_FILE, "r", encoding="utf-8") as f:
                vf_data = json.load(f)
                states = vf_data.get("states", [])
                result["value_function"]["total_states"] = len(states)
                
                if states:
                    fitnesses = [s.get("fitness") for s in states if s.get("fitness") and s.get("fitness") < 1e5]
                    if fitnesses:
                        result["value_function"]["best_fitness"] = min(fitnesses)
                    
                    values = [s.get("value", 0) for s in states]
                    if values:
                        result["value_function"]["avg_value"] = sum(values) / len(values)
                
                result["updated_at"] = vf_data.get("updated_at")
        except:
            pass
    
    return JSONResponse(result)


@app.delete("/api/meta-knowledge")
async def clear_meta_knowledge():
    """清空元知识库"""
    try:
        if META_KNOWLEDGE_FILE.exists():
            os.remove(META_KNOWLEDGE_FILE)
        return JSONResponse({"success": True, "message": "元知识已清空"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.delete("/api/critic/{critic_type}/experience")
async def clear_critic_experience(critic_type: str):
    """清空指定评论家的经验"""
    try:
        exp_file = CRITIC_EXPERIENCE_DIR / f"{critic_type}_experience.json"
        if exp_file.exists():
            os.remove(exp_file)
        return JSONResponse({"success": True, "message": f"{critic_type} 经验已清空"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.delete("/api/critic/{critic_type}/td-stats")
async def clear_critic_td_stats(critic_type: str):
    """清空指定评论家的 TD 统计"""
    try:
        td_file = CRITIC_EXPERIENCE_DIR / f"{critic_type}_td_stats.json"
        if td_file.exists():
            os.remove(td_file)
        return JSONResponse({"success": True, "message": f"{critic_type} TD统计已清空"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.delete("/api/critic/all")
async def clear_all_critic_data():
    """清空所有评论家数据"""
    try:
        critic_types = ["magnetic", "performance", "constraint", "magnitude"]
        for critic_type in critic_types:
            exp_file = CRITIC_EXPERIENCE_DIR / f"{critic_type}_experience.json"
            td_file = CRITIC_EXPERIENCE_DIR / f"{critic_type}_td_stats.json"
            strategy_file = CRITIC_EXPERIENCE_DIR / f"{critic_type}_strategy.json"
            if exp_file.exists():
                os.remove(exp_file)
            if td_file.exists():
                os.remove(td_file)
            if strategy_file.exists():
                os.remove(strategy_file)
        
        # 清空价值函数历史（兼容旧命名）
        if VALUE_FUNCTION_FILE.exists():
            os.remove(VALUE_FUNCTION_FILE)
        legacy_vf_file = PROJECT_ROOT / "state_value_history.json"
        if legacy_vf_file.exists():
            os.remove(legacy_vf_file)
        
        return JSONResponse({"success": True, "message": "所有评论家数据已清空"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ========== 统一记忆架构 API ==========

@app.get("/api/memory")
async def get_memory_status():
    """获取统一记忆架构状态（短期 + 长期）"""
    result = {
        "short_term": {
            "description": "当前 episode 的轨迹 (Trajectory)",
            "note": "短期记忆在运行时维护，这里显示相关文件状态"
        },
        "long_term": {
            "description": "跨 episode 的经验 (Experience)",
            "components": {}
        }
    }
    
    # 长期记忆组件
    components = {
        "experience_buffer": {
            "file": "experience_buffer.json",
            "description": "历史设计经验"
        },
        "strategy_state": {
            "file": "strategy_state.json",
            "description": "策略学习状态"
        },
        "meta_knowledge": {
            "file": "meta_knowledge.json",
            "description": "元学习知识"
        },
        "memory_stream": {
            "file": "memory_stream.jsonl",
            "description": "反思记录"
        },
        "feedback_storage": {
            "file": "feedback_storage.json",
            "description": "人工反馈"
        }
    }
    
    for name, info in components.items():
        file_path = PROJECT_ROOT / info["file"]
        comp_info = {
            "description": info["description"],
            "exists": file_path.exists(),
            "size_kb": round(file_path.stat().st_size / 1024, 2) if file_path.exists() else 0
        }
        
        # 尝试读取摘要信息
        if file_path.exists() and info["file"].endswith(".json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        comp_info["item_count"] = len(data)
                    elif isinstance(data, dict):
                        comp_info["key_count"] = len(data)
            except:
                pass
        elif file_path.exists() and info["file"].endswith(".jsonl"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    comp_info["line_count"] = sum(1 for _ in f)
            except:
                pass
        
        result["long_term"]["components"][name] = comp_info
    
    return JSONResponse(result)


# ========== Reflexion 反思模块 API ==========

MEMORY_STREAM_FILE = PROJECT_ROOT / "memory_stream.jsonl"

@app.get("/api/reflection")
async def get_reflection_status(limit: int = 100, offset: int = 0, event_type: str = None):
    """获取反思模块状态
    
    参数:
        limit: 返回事件数量，默认 100
        offset: 偏移量（从末尾计算），默认 0
        event_type: 筛选事件类型（observation/reflection/rule/pattern/sensitivity）
    """
    result = {
        "memory_stream": {
            "exists": MEMORY_STREAM_FILE.exists(),
            "total_events": 0,
            "by_type": {},
            "recent_events": [],
            "pagination": {"limit": limit, "offset": offset}
        }
    }
    
    if MEMORY_STREAM_FILE.exists():
        try:
            events = []
            type_counts = {}
            with open(MEMORY_STREAM_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            events.append(event)
                            et = event.get("event_type", "unknown")
                            type_counts[et] = type_counts.get(et, 0) + 1
                        except:
                            pass
            
            result["memory_stream"]["total_events"] = len(events)
            result["memory_stream"]["by_type"] = type_counts
            
            # 按类型筛选
            if event_type:
                events = [e for e in events if e.get("event_type") == event_type]
                result["memory_stream"]["filtered_count"] = len(events)
            
            # 分页：从末尾往前取
            if offset > 0:
                events = events[-(offset + limit):-offset] if offset < len(events) else []
            else:
                events = events[-limit:]
            
            result["memory_stream"]["recent_events"] = events
            result["memory_stream"]["has_more"] = offset + limit < result["memory_stream"]["total_events"]
        except Exception as e:
            result["error"] = str(e)
    
    return JSONResponse(result)


@app.get("/api/reflection/reflections")
async def get_reflections():
    """获取所有反思记录"""
    reflections = []
    
    if MEMORY_STREAM_FILE.exists():
        try:
            with open(MEMORY_STREAM_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            if event.get("event_type") == "reflection":
                                reflections.append(event)
                        except:
                            pass
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    
    return JSONResponse({"reflections": reflections})


@app.get("/api/reflection/rules")
async def get_reflection_rules():
    """获取从反思中提取的规则"""
    rules = []
    
    if MEMORY_STREAM_FILE.exists():
        try:
            with open(MEMORY_STREAM_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            if event.get("event_type") == "rule":
                                rules.append(event)
                        except:
                            pass
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    
    return JSONResponse({"rules": rules})


@app.delete("/api/reflection")
async def clear_reflection():
    """清空反思记忆流"""
    try:
        if MEMORY_STREAM_FILE.exists():
            os.remove(MEMORY_STREAM_FILE)
        return JSONResponse({"success": True, "message": "反思记忆已清空"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ========== 远程 Actor (Agentic RL) API ==========

# 远程 Actor URL 配置（可通过环境变量设置）
REMOTE_ACTOR_URL = os.getenv("REMOTE_ACTOR_URL", "http://localhost:8000")

@app.get("/api/actor/status")
async def get_actor_status():
    """获取远程 Actor 的训练状态"""
    import requests
    
    result = {
        "connected": False,
        "model_loaded": False,
        "gpu_available": False,
        "gpu_count": 0,
        "update_count": 0,
        "buffer_size": 0,
        "total_steps": 0,
        "training_stats": {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "rewards": []
        },
        "error": None
    }
    
    try:
        # 获取健康状态
        health_response = requests.get(f"{REMOTE_ACTOR_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health = health_response.json()
            result["connected"] = True
            result["model_loaded"] = health.get("model_loaded", False)
            result["gpu_available"] = health.get("gpu_available", False)
            result["gpu_count"] = health.get("gpu_count", 0)
            result["update_count"] = health.get("update_count", 0)
            result["buffer_size"] = health.get("buffer_size", 0)
        
        # 获取训练统计
        stats_response = requests.get(f"{REMOTE_ACTOR_URL}/stats", timeout=5)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            result["total_steps"] = stats.get("total_steps", 0)
            result["update_count"] = stats.get("update_count", 0)
            result["buffer_size"] = stats.get("buffer_size", 0)
            
            recent_stats = stats.get("recent_stats", {})
            result["training_stats"]["policy_loss"] = recent_stats.get("policy_loss", [])
            result["training_stats"]["value_loss"] = recent_stats.get("value_loss", [])
            result["training_stats"]["entropy"] = recent_stats.get("entropy", [])
            result["training_stats"]["kl_divergence"] = recent_stats.get("kl_divergence", [])
            result["training_stats"]["rewards"] = recent_stats.get("rewards", [])
            result["training_stats"]["bc_loss"] = recent_stats.get("bc_loss", [])
            result["training_stats"]["dpo_loss"] = recent_stats.get("dpo_loss", [])  # ★ DPO Loss
            
            # ★ 训练模式和 DPO 专属信息
            result["mode"] = stats.get("mode", "ppo")
            result["pairs_count"] = stats.get("pairs_count", 0)
            result["total_experiences"] = stats.get("total_experiences", 0)
    
    except requests.exceptions.ConnectionError:
        result["error"] = "无法连接到远程 Actor 服务器"
    except requests.exceptions.Timeout:
        result["error"] = "连接远程 Actor 超时"
    except Exception as e:
        result["error"] = str(e)
    
    return JSONResponse(result)


@app.post("/api/actor/force-update")
async def force_actor_update():
    """强制触发 PPO 更新"""
    import requests
    
    try:
        response = requests.post(f"{REMOTE_ACTOR_URL}/update", timeout=60)
        if response.status_code == 200:
            return JSONResponse({"success": True, "result": response.json()})
        else:
            return JSONResponse({"success": False, "error": response.text}, status_code=response.status_code)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/actor/save-checkpoint")
async def save_actor_checkpoint():
    """保存 Actor 检查点"""
    import requests
    
    try:
        response = requests.post(f"{REMOTE_ACTOR_URL}/save_checkpoint", timeout=30)
        if response.status_code == 200:
            return JSONResponse({"success": True, "result": response.json()})
        else:
            return JSONResponse({"success": False, "error": response.text}, status_code=response.status_code)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/actor/set-url")
async def set_actor_url(request: Request):
    """设置远程 Actor URL"""
    global REMOTE_ACTOR_URL
    try:
        data = await request.json()
        url = data.get("url", "").strip()
        if url:
            REMOTE_ACTOR_URL = url
            return JSONResponse({"success": True, "url": REMOTE_ACTOR_URL})
        else:
            return JSONResponse({"success": False, "error": "URL 不能为空"}, status_code=400)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ========== Web 页面 ==========

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maxwell 优化反馈系统</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e8e8e8;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00d9ff;
            margin-bottom: 30px;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            transition: all 0.3s;
        }
        .tab:hover { background: rgba(255, 255, 255, 0.1); }
        .tab.active {
            background: rgba(0, 217, 255, 0.2);
            border-color: #00d9ff;
            color: #00d9ff;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .card h2 {
            color: #00d9ff;
            font-size: 1.2em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        textarea {
            width: 100%;
            height: 100px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #fff;
            padding: 12px;
            font-size: 14px;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #00d9ff;
            box-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
        }
        .options {
            display: flex;
            gap: 15px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        .option-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        select {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            color: #fff;
            padding: 8px 12px;
            cursor: pointer;
        }
        label {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
        }
        input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        button {
            background: linear-gradient(135deg, #00d9ff, #0099cc);
            border: none;
            border-radius: 8px;
            color: #fff;
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        .btn-danger {
            background: linear-gradient(135deg, #ff4757, #cc0000);
        }
        .btn-small {
            padding: 8px 16px;
            font-size: 14px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }
        .stat-item {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00d9ff;
        }
        .stat-value.green { color: #2ed573; }
        .stat-value.yellow { color: #ffa502; }
        .stat-value.red { color: #ff4757; }
        .stat-label {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        .feedback-list, .region-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .feedback-item, .region-item {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 8px;
            font-size: 13px;
            border-left: 3px solid #00d9ff;
        }
        .feedback-item.warning { border-left-color: #ffa502; }
        .feedback-item.correction { border-left-color: #ff4757; }
        .region-item { font-family: monospace; }
        .progress-bar {
            height: 8px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d9ff, #2ed573);
            transition: width 0.3s;
        }
        .results-bar {
            display: flex;
            height: 24px;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        .results-success {
            background: #2ed573;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            color: #000;
        }
        .results-failure {
            background: #ff4757;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            color: #fff;
        }
        .direction-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .direction-item {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            padding: 8px;
            text-align: center;
            font-size: 12px;
        }
        .direction-value {
            font-weight: bold;
            margin-top: 4px;
        }
        .direction-value.positive { color: #2ed573; }
        .direction-value.negative { color: #ff4757; }
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #00d9ff;
            color: #000;
            padding: 15px 25px;
            border-radius: 8px;
            font-weight: bold;
            transform: translateX(150%);
            transition: transform 0.3s;
            z-index: 1000;
        }
        .toast.show { transform: translateX(0); }
        .toast.error { background: #ff4757; color: #fff; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        th { color: #00d9ff; }
        .exp-ok { color: #2ed573; }
        .exp-fail { color: #ff4757; }
        .history-item {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 3px solid #00d9ff;
            position: relative;
        }
        .history-item.warning { border-left-color: #ffa502; }
        .history-item.correction { border-left-color: #ff4757; }
        .history-item.confirmation { border-left-color: #2ed573; }
        .history-item .fb-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .history-item .fb-type {
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(0, 217, 255, 0.2);
            color: #00d9ff;
        }
        .history-item .fb-time {
            font-size: 11px;
            color: #666;
        }
        .history-item .fb-text {
            font-size: 14px;
            line-height: 1.5;
            margin-bottom: 10px;
        }
        .history-item .fb-actions {
            display: flex;
            gap: 8px;
        }
        .history-item .fb-actions button {
            padding: 4px 12px;
            font-size: 12px;
            background: rgba(255, 255, 255, 0.1);
        }
        .history-item .fb-actions button:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        .history-item .fb-actions .btn-edit { background: rgba(0, 217, 255, 0.3); }
        .history-item .fb-actions .btn-delete { background: rgba(255, 71, 87, 0.3); }
        .strength-control {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
            padding: 8px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
        }
        .strength-control label {
            font-size: 12px;
            color: #888;
            min-width: 60px;
        }
        .strength-slider {
            flex: 1;
            -webkit-appearance: none;
            height: 6px;
            background: linear-gradient(to right, #2ed573, #ffa502, #ff4757);
            border-radius: 3px;
            cursor: pointer;
        }
        .strength-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #fff;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }
        .strength-value {
            min-width: 50px;
            text-align: right;
            font-weight: bold;
            font-size: 13px;
        }
        .strength-value.high { color: #ff4757; }
        .strength-value.medium { color: #ffa502; }
        .strength-value.low { color: #2ed573; }
        .strength-value.off { color: #666; }
        .edit-modal {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .edit-modal.show { display: flex; }
        .edit-modal-content {
            background: #1a1a2e;
            border: 1px solid rgba(0, 217, 255, 0.3);
            border-radius: 12px;
            padding: 24px;
            width: 90%;
            max-width: 500px;
        }
        .edit-modal h3 {
            color: #00d9ff;
            margin-bottom: 16px;
        }
        .edit-modal textarea {
            margin-bottom: 16px;
        }
        .edit-modal .modal-actions {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Maxwell 优化控制台</h1>
        
        <!-- 标签页 -->
        <div class="tabs">
            <div class="tab active" onclick="switchTab('feedback')">📝 反馈</div>
            <div class="tab" onclick="switchTab('strategy')">🎯 策略</div>
            <div class="tab" onclick="switchTab('experience')">📚 经验</div>
            <div class="tab" onclick="switchTab('meta')">🧠 元学习</div>
            <div class="tab" onclick="switchTab('reflection')">🪞 反思</div>
            <div class="tab" onclick="switchTab('critic')">🎭 评论家</div>
            <div class="tab" onclick="switchTab('actor')">🤖 远程Actor</div>
        </div>
        
        <!-- 反馈页 -->
        <div id="tab-feedback" class="tab-content active">
            <!-- 提交反馈 -->
            <div class="card">
                <h2>📝 提交反馈</h2>
                <textarea id="feedbackText" placeholder="输入你的反馈...&#10;例如：注意 dg 要大于 0.35&#10;或：twall 过小会导致 n2 无法达到 4"></textarea>
                <div class="options">
                    <div class="option-group">
                        <span>类型:</span>
                        <select id="feedbackType">
                            <option value="suggestion">💡 建议</option>
                            <option value="warning">⚠️ 警告</option>
                            <option value="correction">🔴 纠正</option>
                            <option value="confirmation">✅ 确认</option>
                        </select>
                    </div>
                    <label>
                        <input type="checkbox" id="urgentCheck">
                        <span>🚨 紧急</span>
                    </label>
                </div>
                <button onclick="submitFeedback()">提交反馈</button>
            </div>
            
            <!-- 状态面板 -->
            <div class="card">
                <h2>📊 状态面板</h2>
                <div class="status-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="pendingCount">-</div>
                        <div class="stat-label">待处理反馈</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="storedCount">-</div>
                        <div class="stat-label">已存储反馈</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="expCount">-</div>
                        <div class="stat-label">经验数量</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="bestFitness">-</div>
                        <div class="stat-label">最佳 Fitness</div>
                    </div>
                </div>
                <button onclick="refreshAll()" style="margin-top: 15px;" class="btn-small">刷新状态</button>
            </div>
            
            <!-- 待处理反馈 -->
            <div class="card">
                <h2>⏳ 待处理反馈（下轮生效）</h2>
                <div class="feedback-list" id="pendingList">
                    <div style="color: #666;">暂无</div>
                </div>
                <button class="btn-danger btn-small" onclick="clearPending()" style="margin-top: 15px;">清空待处理</button>
            </div>
            
            <!-- 历史反馈 -->
            <div class="card">
                <h2>📚 已存储的历史反馈</h2>
                <div class="feedback-list" id="historyList" style="max-height: 400px;">
                    <div style="color: #666;">暂无</div>
                </div>
            </div>
        </div>
        
        <!-- 策略页 -->
        <div id="tab-strategy" class="tab-content">
            <div class="card">
                <h2>🎯 策略状态</h2>
                <div class="status-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="epsilon">-</div>
                        <div class="stat-label">探索率 ε</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="iteration">-</div>
                        <div class="stat-label">迭代次数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="successRate">-</div>
                        <div class="stat-label">成功率</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value green" id="successPatternsCount">-</div>
                        <div class="stat-label">成功模式</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value red" id="failurePatternsCount">-</div>
                        <div class="stat-label">失败模式</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="rulesCount">-</div>
                        <div class="stat-label">学习规则</div>
                    </div>
                </div>
                
                <!-- 探索率进度条 -->
                <div style="margin-top: 20px;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;">
                        <span>探索 (ε)</span>
                        <span>利用 (1-ε)</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="epsilonBar" style="width: 30%;"></div>
                    </div>
                </div>
                
                <!-- 最近结果条 -->
                <div style="margin-top: 20px;">
                    <div style="font-size: 12px; color: #888; margin-bottom: 5px;">最近 20 轮结果</div>
                    <div class="results-bar" id="resultsBar">
                        <div class="results-success" style="width: 0%;">-</div>
                        <div class="results-failure" style="width: 100%;">-</div>
                    </div>
                </div>
                
                <button onclick="refreshStrategy()" style="margin-top: 15px;" class="btn-small">刷新策略</button>
                <button onclick="resetStrategy()" style="margin-top: 15px;" class="btn-danger btn-small">重置策略</button>
            </div>
            
            <!-- 参数敏感性 -->
            <div class="card">
                <h2>📊 参数敏感性分析</h2>
                <div class="direction-grid" id="sensitivityGrid">
                    <div style="color: #666;">暂无数据</div>
                </div>
            </div>
            
            <!-- 学习规则 -->
            <div class="card">
                <h2>📚 学习到的规则</h2>
                <div class="feedback-list" id="rulesList">
                    <div style="color: #666;">暂无规则</div>
                </div>
            </div>
            
            <!-- 成功模式 -->
            <div class="card">
                <h2>✅ 成功模式（最佳 5 个）</h2>
                <div class="region-list" id="successPatternsList">
                    <div style="color: #666;">暂无</div>
                </div>
            </div>
            
            <!-- 失败模式 -->
            <div class="card">
                <h2>⚠️ 失败模式（最频繁 5 个）</h2>
                <div class="feedback-list" id="failurePatternsList">
                    <div style="color: #666;">暂无</div>
                </div>
            </div>
            
            <!-- 上次调整方向 -->
            <div class="card">
                <h2>📐 上次调整方向</h2>
                <div class="direction-grid" id="directionGrid">
                    <div style="color: #666;">暂无数据</div>
                </div>
            </div>
        </div>
        
        <!-- 经验页 -->
        <div id="tab-experience" class="tab-content">
            <div class="card">
                <h2>📚 经验统计</h2>
                <div class="status-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="expTotal">-</div>
                        <div class="stat-label">总经验</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value green" id="expSuccess">-</div>
                        <div class="stat-label">成功</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value red" id="expFailure">-</div>
                        <div class="stat-label">失败</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="expBest">-</div>
                        <div class="stat-label">最佳 Fitness</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="expAvg">-</div>
                        <div class="stat-label">平均 Fitness</div>
                    </div>
                </div>
                <button onclick="refreshExperience()" style="margin-top: 15px;" class="btn-small">刷新经验</button>
                <button onclick="clearExperience()" style="margin-top: 15px;" class="btn-danger btn-small">清空经验</button>
            </div>
            
            <!-- 最近经验表 -->
            <div class="card">
                <h2>📋 最近 20 条经验</h2>
                <div style="overflow-x: auto;">
                    <table id="expTable">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>状态</th>
                                <th>Fitness</th>
                                <th>奖励</th>
                                <th>关键参数</th>
                            </tr>
                        </thead>
                        <tbody id="expBody">
                            <tr><td colspan="5" style="color: #666;">暂无数据</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- 元学习页（ExpeL + Reflexion 规则） -->
        <div id="tab-meta" class="tab-content">
            <!-- 元学习规则汇总 -->
            <div class="card" style="border-left: 4px solid #9b59b6;">
                <h2>🧠 元学习规则库 <span style="font-size: 12px; color: #888;">（ExpeL 对比 + Reflexion 反思提取）</span></h2>
                <div class="status-grid" style="margin-bottom: 15px;">
                    <div class="stat-item">
                        <div class="stat-value" id="expelTotalRules">-</div>
                        <div class="stat-label">规则总数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value green" id="expelAvgConf">-</div>
                        <div class="stat-label">平均置信度</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value yellow" id="expelCompare">-</div>
                        <div class="stat-label">ExpeL对比</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value blue" id="expelReflection">-</div>
                        <div class="stat-label">Reflexion反思</div>
                    </div>
                </div>
                <div class="feedback-list" id="expelRulesList" style="max-height: 500px;">
                    <div style="color: #666;">暂无规则（运行优化后将自动生成）</div>
                </div>
                <div style="margin-top: 15px; display: flex; gap: 10px;">
                    <button onclick="refreshExpel()" class="btn-small">刷新规则</button>
                    <button onclick="clearExpel()" class="btn-danger btn-small">清空所有规则</button>
                </div>
                <div id="expelUpdatedAt" style="margin-top: 10px; font-size: 12px; color: #666;"></div>
            </div>
        </div>
        
        <!-- 反思页 (Reflexion) -->
        <div id="tab-reflection" class="tab-content">
            <!-- 反思统计 -->
            <div class="card">
                <h2>🪞 反思统计 (Reflexion)</h2>
                <div class="status-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="reflectionTotal">-</div>
                        <div class="stat-label">记忆事件</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value yellow" id="reflectionReflections">-</div>
                        <div class="stat-label">反思次数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value green" id="reflectionRules">-</div>
                        <div class="stat-label">提取规则</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value blue" id="reflectionObservations">-</div>
                        <div class="stat-label">观察记录</div>
                    </div>
                </div>
                <div style="margin-top: 15px; display: flex; gap: 10px;">
                    <button onclick="refreshReflection()" class="btn-small">刷新</button>
                    <button onclick="clearReflection()" class="btn-danger btn-small">清空反思记忆</button>
                </div>
            </div>
            
            <!-- 最近反思 -->
            <div class="card">
                <h2>📝 最近反思</h2>
                <div class="feedback-list" id="reflectionList" style="max-height: 500px;">
                    <div style="color: #666;">暂无反思记录</div>
                </div>
                <div style="margin-top: 10px; display: flex; gap: 10px;">
                    <button onclick="loadMoreReflections()" class="btn-small" id="loadMoreReflBtn">加载更多反思</button>
                    <span id="reflLoadInfo" style="color: #888; font-size: 12px; line-height: 32px;"></span>
                </div>
            </div>
            
            <!-- 提取的规则 -->
            <div class="card">
                <h2>📌 从反思中提取的规则</h2>
                <div class="feedback-list" id="reflectionRuleList" style="max-height: 400px;">
                    <div style="color: #666;">暂无规则</div>
                </div>
                <div style="margin-top: 10px; display: flex; gap: 10px;">
                    <button onclick="loadMoreRules()" class="btn-small" id="loadMoreRuleBtn">加载更多规则</button>
                    <span id="ruleLoadInfo" style="color: #888; font-size: 12px; line-height: 32px;"></span>
                </div>
            </div>
            
            <!-- 最近观察 -->
            <div class="card">
                <h2>👁️ 最近观察</h2>
                <div class="feedback-list" id="reflectionObsList" style="max-height: 500px;">
                    <div style="color: #666;">暂无观察</div>
                </div>
                <div style="margin-top: 10px; display: flex; gap: 10px;">
                    <button onclick="loadMoreObservations()" class="btn-small" id="loadMoreObsBtn">加载更多观察</button>
                    <span id="obsLoadInfo" style="color: #888; font-size: 12px; line-height: 32px;"></span>
                </div>
            </div>
        </div>
        
        <!-- 评论家页 -->
        <div id="tab-critic" class="tab-content">
            <!-- 评论家集群统计 -->
            <div class="card">
                <h2>🎭 评论家集群状态 (Actor-Critic)</h2>
                <div class="status-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="criticEnsembleAcc">-</div>
                        <div class="stat-label">集群准确率</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value yellow" id="criticTDRules">-</div>
                        <div class="stat-label">TD学习规则</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="criticVFStates">-</div>
                        <div class="stat-label">V(s) 状态数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value green" id="criticBestFitness">-</div>
                        <div class="stat-label">最佳 Fitness</div>
                    </div>
                </div>
                <div id="criticUpdatedAt" style="margin-top: 10px; font-size: 12px; color: #666;"></div>
                <div style="margin-top: 15px; display: flex; gap: 10px;">
                    <button onclick="refreshCritic()" class="btn-small">刷新状态</button>
                    <button onclick="clearAllCritic()" class="btn-danger btn-small">清空所有数据</button>
                </div>
            </div>
            
            <!-- 各评论家详情 -->
            <div class="card">
                <h2>📊 评论家详情</h2>
                <div id="criticDetailGrid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px;">
                    <div style="color: #666;">加载中...</div>
                </div>
            </div>
            
            <!-- TD 学习统计 -->
            <div class="card">
                <h2>📈 TD 误差学习统计 <span style="font-size: 12px; color: #888;">（参数改动 vs 实际效果）</span></h2>
                <div id="criticTDStatsList" class="feedback-list" style="max-height: 400px;">
                    <div style="color: #666;">暂无 TD 统计</div>
                </div>
            </div>
            
            <!-- 评论家策略库 -->
            <div class="card">
                <h2>📚 评论家策略库 <span style="font-size: 12px; color: #888;">（已学习的规则）</span></h2>
                <div id="criticStrategiesList" class="feedback-list" style="max-height: 400px;">
                    <div style="color: #666;">暂无策略</div>
                </div>
            </div>
            
            <!-- 价值函数 V(s) 状态 -->
            <div class="card">
                <h2>💎 状态价值函数 V(s) <span style="font-size: 12px; color: #888;">（历史状态评估）</span></h2>
                <div class="status-grid" style="margin-bottom: 15px;">
                    <div class="stat-item">
                        <div class="stat-value" id="vfTotalStates">-</div>
                        <div class="stat-label">总状态数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="vfAvgValue">-</div>
                        <div class="stat-label">平均价值</div>
                    </div>
                </div>
                <div style="font-size: 12px; color: #888; margin-bottom: 10px;">
                    V(s) 使用 k-NN 估计：根据历史相似状态加权平均计算当前状态价值
                </div>
                <div class="progress-bar" style="height: 12px;">
                    <div class="progress-fill" id="vfValueBar" style="width: 50%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #666; margin-top: 5px;">
                    <span>差 (-1)</span>
                    <span>中等 (0)</span>
                    <span>优 (+1)</span>
                </div>
            </div>
        </div>
        
        <!-- 远程Actor (Agentic RL) 页 -->
        <div id="tab-actor" class="tab-content">
            <!-- 连接状态 -->
            <div class="card">
                <h2>🤖 远程 Actor 状态 (Agentic RL)</h2>
                <div class="status-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="actorConnected">❌</div>
                        <div class="stat-label">连接状态</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="actorGpuCount">-</div>
                        <div class="stat-label">GPU 数量</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="actorUpdateCount">0</div>
                        <div class="stat-label" id="updateCountLabel">更新次数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="actorBufferSize">0</div>
                        <div class="stat-label">经验缓冲大小</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="actorTotalSteps">0</div>
                        <div class="stat-label">总步数</div>
                    </div>
                </div>
                
                <!-- Actor URL 配置 -->
                <div style="margin-top: 15px; display: flex; gap: 10px; align-items: center;">
                    <input type="text" id="actorUrlInput" placeholder="http://localhost:8000" 
                           style="flex: 1; padding: 8px; border: 1px solid #333; border-radius: 4px; background: #1a1a2e; color: #e8e8e8;">
                    <button onclick="setActorUrl()" class="btn-small">设置 URL</button>
                    <button onclick="refreshActor()" class="btn-small">刷新状态</button>
                </div>
                
                <div style="margin-top: 10px; display: flex; gap: 10px;">
                    <button onclick="forceActorUpdate()" class="btn-small" style="background: #4ecdc4;">🔄 强制 PPO 更新</button>
                    <button onclick="saveActorCheckpoint()" class="btn-small" style="background: #a55eea;">💾 保存检查点</button>
                </div>
            </div>
            
            <!-- 训练统计（根据模式显示 PPO 或 DPO） -->
            <div class="card">
                <h2>📊 <span id="trainingModeTitle">PPO</span> 训练统计</h2>
                <div style="margin-bottom: 10px;">
                    <span style="padding: 4px 12px; border-radius: 12px; font-size: 12px;" id="trainingModeBadge">PPO</span>
                </div>
                <div id="actorError" style="color: #ff6b6b; margin-bottom: 10px; display: none;"></div>
                
                <!-- PPO 损失曲线 -->
                <div id="ppoStatsSection" style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <div>
                        <h3 style="font-size: 14px; margin-bottom: 10px;">📉 Policy Loss</h3>
                        <div id="policyLossChart" style="height: 150px; background: #1a1a2e; border-radius: 8px; padding: 10px; position: relative;">
                            <canvas id="policyLossCanvas" style="width: 100%; height: 100%;"></canvas>
                        </div>
                        <div style="font-size: 12px; color: #888; margin-top: 5px;">
                            最新值: <span id="policyLossLatest">-</span>
                        </div>
                    </div>
                    <div>
                        <h3 style="font-size: 14px; margin-bottom: 10px;">📉 Value Loss</h3>
                        <div id="valueLossChart" style="height: 150px; background: #1a1a2e; border-radius: 8px; padding: 10px; position: relative;">
                            <canvas id="valueLossCanvas" style="width: 100%; height: 100%;"></canvas>
                        </div>
                        <div style="font-size: 12px; color: #888; margin-top: 5px;">
                            最新值: <span id="valueLossLatest">-</span>
                        </div>
                    </div>
                    <div>
                        <h3 style="font-size: 14px; margin-bottom: 10px;">🎲 Entropy</h3>
                        <div id="entropyChart" style="height: 150px; background: #1a1a2e; border-radius: 8px; padding: 10px; position: relative;">
                            <canvas id="entropyCanvas" style="width: 100%; height: 100%;"></canvas>
                        </div>
                        <div style="font-size: 12px; color: #888; margin-top: 5px;">
                            最新值: <span id="entropyLatest">-</span>
                        </div>
                    </div>
                    <div>
                        <h3 style="font-size: 14px; margin-bottom: 10px;">📏 KL Divergence</h3>
                        <div id="klChart" style="height: 150px; background: #1a1a2e; border-radius: 8px; padding: 10px; position: relative;">
                            <canvas id="klCanvas" style="width: 100%; height: 100%;"></canvas>
                        </div>
                        <div style="font-size: 12px; color: #888; margin-top: 5px;">
                            最新值: <span id="klLatest">-</span>
                        </div>
                    </div>
                </div>
                
                <!-- BC Loss（两种模式都显示） -->
                <div style="margin-top: 15px;">
                    <h3 style="font-size: 14px; margin-bottom: 10px;">🎯 BC Loss (行为克隆) - PPO/DPO 共用</h3>
                    <div id="bcLossChart" style="height: 150px; background: #1a1a2e; border-radius: 8px; padding: 10px; position: relative;">
                        <canvas id="bcLossCanvas" style="width: 100%; height: 100%;"></canvas>
                    </div>
                    <div style="font-size: 12px; color: #888; margin-top: 5px;">
                        最新值: <span id="bcLossLatest">-</span>
                        <span style="margin-left: 15px; color: #4ecdc4;">（下降表示 7B 正在学习 GPT 的决策方式）</span>
                    </div>
                </div>
                
                <!-- DPO 专属统计（当模式为 DPO 时显示） -->
                <div id="dpoStatsSection" style="display: none; margin-top: 20px; padding-top: 20px; border-top: 1px solid #333;">
                    <h3 style="font-size: 14px; margin-bottom: 15px; color: #a55eea;">📊 DPO 专属指标</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div>
                            <h3 style="font-size: 14px; margin-bottom: 10px;">📉 DPO Loss</h3>
                            <div id="dpoLossChart" style="height: 150px; background: #1a1a2e; border-radius: 8px; padding: 10px; position: relative;">
                                <canvas id="dpoLossCanvas" style="width: 100%; height: 100%;"></canvas>
                            </div>
                            <div style="font-size: 12px; color: #888; margin-top: 5px;">
                                最新值: <span id="dpoLossLatest">-</span>
                            </div>
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 10px;">
                            <div style="background: #1a1a2e; padding: 15px; border-radius: 8px;">
                                <div style="font-size: 12px; color: #888;">偏好对数量</div>
                                <div style="font-size: 24px; color: #a55eea;" id="dpoPairsCount">0</div>
                            </div>
                            <div style="background: #1a1a2e; padding: 15px; border-radius: 8px;">
                                <div style="font-size: 12px; color: #888;">总经验数</div>
                                <div style="font-size: 24px; color: #4ecdc4;" id="dpoTotalExperiences">0</div>
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding: 10px; background: #1a1a2e; border-radius: 8px; font-size: 12px; color: #888;">
                        <strong style="color: #a55eea;">DPO 原理：</strong>
                        直接从偏好数据学习，不需要奖励模型。preferred = fitness 更优的决策，comparison = fitness 较低的决策。
                    </div>
                </div>
            </div>
            
            <!-- 训练阶段说明 -->
            <div class="card">
                <h2>📖 训练阶段说明</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div style="padding: 15px; background: #1a1a2e; border-radius: 8px; border-left: 3px solid #4ecdc4;">
                        <h3 style="color: #4ecdc4; font-size: 14px;">第一阶段：跟着老师学</h3>
                        <p style="font-size: 12px; color: #888; margin-top: 8px;">
                            GPT-5.2 做决策，7B 观察并收集经验。<br>
                            使用命令: <code>--use-remote-actor</code>
                        </p>
                    </div>
                    <div style="padding: 15px; background: #1a1a2e; border-radius: 8px; border-left: 3px solid #a55eea;">
                        <h3 style="color: #a55eea; font-size: 14px;">第二阶段：7B 自己干</h3>
                        <p style="font-size: 12px; color: #888; margin-top: 8px;">
                            7B 自己做决策，自己探索。<br>
                            使用命令: <code>--use-remote-actor --actor-decision</code>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="toast" id="toast"></div>
    
    <!-- 编辑弹窗 -->
    <div class="edit-modal" id="editModal">
        <div class="edit-modal-content">
            <h3>✏️ 编辑反馈</h3>
            <input type="hidden" id="editFeedbackId">
            <textarea id="editText" placeholder="修改反馈内容..."></textarea>
            <div class="options">
                <div class="option-group">
                    <span>类型:</span>
                    <select id="editType">
                        <option value="suggestion">💡 建议</option>
                        <option value="warning">⚠️ 警告</option>
                        <option value="correction">🔴 纠正</option>
                        <option value="confirmation">✅ 确认</option>
                    </select>
                </div>
            </div>
            <!-- 引导强度 -->
            <div class="strength-control">
                <label>引导强度:</label>
                <input type="range" class="strength-slider" id="editStrength" min="0" max="100" value="100" oninput="updateStrengthDisplay('edit')">
                <span class="strength-value high" id="editStrengthValue">100%</span>
            </div>
            <div style="font-size: 11px; color: #666; margin-bottom: 15px;">
                🔴 100%=必须遵守 | 🟠 60%=重要参考 | 🟡 30%=可参考 | 🟢 10%=弱参考 | ⚪ 0%=暂时关闭
            </div>
            <div class="modal-actions">
                <button onclick="closeEditModal()" style="background: rgba(255,255,255,0.1);">取消</button>
                <button onclick="saveEdit()">保存修改</button>
            </div>
        </div>
    </div>
    
    <script>
        function showToast(msg, isError = false) {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.className = 'toast show' + (isError ? ' error' : '');
            setTimeout(() => toast.className = 'toast', 3000);
        }
        
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            const tabIndex = {'feedback': 1, 'strategy': 2, 'experience': 3, 'meta': 4, 'reflection': 5, 'critic': 6, 'actor': 7}[tab] || 1;
            document.querySelector(`.tab:nth-child(${tabIndex})`).classList.add('active');
            document.getElementById('tab-' + tab).classList.add('active');
            
            if (tab === 'strategy') refreshStrategy();
            if (tab === 'experience') refreshExperience();
            if (tab === 'meta') refreshExpel();  // 只用 ExpeL，旧版元学习已删除
            if (tab === 'reflection') refreshReflection();
            if (tab === 'critic') refreshCritic();
            if (tab === 'actor') refreshActor();
        }
        
        async function submitFeedback() {
            const text = document.getElementById('feedbackText').value.trim();
            if (!text) { showToast('请输入反馈内容', true); return; }
            
            const type = document.getElementById('feedbackType').value;
            const urgent = document.getElementById('urgentCheck').checked;
            
            try {
                const res = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, type, urgent })
                });
                const data = await res.json();
                if (data.success) {
                    showToast('✅ ' + data.message);
                    document.getElementById('feedbackText').value = '';
                    refreshStatus();
                } else {
                    showToast('❌ ' + data.error, true);
                }
            } catch (e) {
                showToast('❌ 请求失败', true);
            }
        }
        
        async function refreshStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                document.getElementById('pendingCount').textContent = data.pending_feedbacks.length;
                document.getElementById('storedCount').textContent = data.stored_feedbacks.length;
                document.getElementById('expCount').textContent = data.experience_count;
                document.getElementById('bestFitness').textContent = 
                    data.best_fitness ? data.best_fitness.toExponential(2) : '-';
                
                const list = document.getElementById('pendingList');
                if (data.pending_feedbacks.length > 0) {
                    list.innerHTML = data.pending_feedbacks.map(fb => {
                        let cls = 'feedback-item';
                        if (fb.includes('[warning]')) cls += ' warning';
                        if (fb.includes('[correction]')) cls += ' correction';
                        return `<div class="${cls}">${fb}</div>`;
                    }).join('');
                } else {
                    list.innerHTML = '<div style="color: #666;">暂无</div>';
                }
            } catch (e) {
                console.error(e);
            }
        }
        
        async function refreshStrategy() {
            try {
                const res = await fetch('/api/strategy');
                const data = await res.json();
                
                // 基础状态
                document.getElementById('epsilon').textContent = data.epsilon.toFixed(3);
                document.getElementById('iteration').textContent = data.iteration;
                document.getElementById('successRate').textContent = (data.recent_success_rate * 100).toFixed(0) + '%';
                document.getElementById('successPatternsCount').textContent = data.success_patterns_count || 0;
                document.getElementById('failurePatternsCount').textContent = data.failure_patterns_count || 0;
                document.getElementById('rulesCount').textContent = (data.learned_rules || []).length;
                
                // 更新探索率条
                const epsPct = Math.round(data.epsilon * 100);
                document.getElementById('epsilonBar').style.width = epsPct + '%';
                
                // 更新结果条
                const results = data.recent_results || [];
                const successCount = results.filter(r => r).length;
                const total = results.length || 1;
                const successPct = Math.round(successCount / total * 100);
                const failPct = 100 - successPct;
                document.getElementById('resultsBar').innerHTML = 
                    `<div class="results-success" style="width: ${successPct}%;">${successCount} 成功</div>` +
                    `<div class="results-failure" style="width: ${failPct}%;">${total - successCount} 失败</div>`;
                
                // 参数敏感性
                const sensGrid = document.getElementById('sensitivityGrid');
                const sens = data.param_sensitivity || {};
                const sensEntries = Object.entries(sens)
                    .filter(([_, v]) => typeof v === 'number' && v > 0);
                if (sensEntries.length > 0) {
                    const sorted = sensEntries.sort((a, b) => b[1] - a[1]);
                    sensGrid.innerHTML = sorted.map(([k, v]) => {
                        const pct = Math.round(v * 100);
                        const cls = v > 0.6 ? 'positive' : v < 0.3 ? '' : '';
                        const bar = '█'.repeat(Math.round(v * 10)) + '░'.repeat(10 - Math.round(v * 10));
                        return `<div class="direction-item">${k}<div class="direction-value ${cls}">${bar} ${pct}%</div></div>`;
                    }).join('');
                } else {
                    sensGrid.innerHTML = '<div style="color: #666;">暂无数据</div>';
                }
                
                // 学习规则
                const rulesList = document.getElementById('rulesList');
                const rules = data.learned_rules || [];
                if (rules.length > 0) {
                    rulesList.innerHTML = rules.map((r, i) => 
                        `<div class="feedback-item">📌 ${r}</div>`
                    ).join('');
                } else {
                    rulesList.innerHTML = '<div style="color: #666;">暂无规则</div>';
                }
                
                // 成功模式
                const successList = document.getElementById('successPatternsList');
                const successPatterns = data.success_patterns || [];
                if (successPatterns.length > 0) {
                    const sorted = [...successPatterns].sort((a, b) => a.fitness - b.fitness).slice(0, 5);
                    successList.innerHTML = sorted.map((sp, i) => {
                        const params = Object.entries(sp.params || {}).slice(0, 5).map(([k,v]) => `${k}=${v}`).join(', ');
                        const factors = (sp.key_factors || []).slice(0, 2).join(', ');
                        return `<div class="region-item">
                            <strong>#${i+1}</strong> fitness=${sp.fitness?.toExponential(2) || '-'} (出现${sp.frequency || 1}次)<br>
                            <span style="font-size:11px;">${params}...</span><br>
                            ${factors ? `<span style="color:#2ed573;font-size:11px;">关键: ${factors}</span>` : ''}
                        </div>`;
                    }).join('');
                } else {
                    successList.innerHTML = '<div style="color: #666;">暂无</div>';
                }
                
                // 失败模式
                const failureList = document.getElementById('failurePatternsList');
                const failurePatterns = data.failure_patterns || [];
                if (failurePatterns.length > 0) {
                    const sorted = [...failurePatterns].sort((a, b) => b.frequency - a.frequency).slice(0, 5);
                    failureList.innerHTML = sorted.map((fp, i) => {
                        const rules = (fp.avoid_rules || []).slice(0, 2).join('; ');
                        const errors = (fp.errors || []).slice(0, 1).join('');
                        return `<div class="feedback-item warning">
                            <strong>⚠️ 失败${fp.frequency || 1}次</strong><br>
                            ${rules ? `<span style="color:#ffa502;">${rules}</span><br>` : ''}
                            <span style="font-size:11px;color:#888;">${errors.substring(0, 60)}...</span>
                        </div>`;
                    }).join('');
                } else {
                    failureList.innerHTML = '<div style="color: #666;">暂无</div>';
                }
                
                // 更新调整方向
                const dirGrid = document.getElementById('directionGrid');
                const dirs = data.last_direction || {};
                if (Object.keys(dirs).length > 0) {
                    dirGrid.innerHTML = Object.entries(dirs).map(([k, v]) => {
                        const cls = v > 0 ? 'positive' : v < 0 ? 'negative' : '';
                        const sign = v > 0 ? '↑' : v < 0 ? '↓' : '→';
                        return `<div class="direction-item">${k}<div class="direction-value ${cls}">${sign} ${(v*100).toFixed(1)}%</div></div>`;
                    }).join('');
                } else {
                    dirGrid.innerHTML = '<div style="color: #666;">暂无数据</div>';
                }
            } catch (e) {
                console.error(e);
            }
        }
        
        async function refreshExperience() {
            try {
                const res = await fetch('/api/experience');
                const data = await res.json();
                const stats = data.stats;
                
                document.getElementById('expTotal').textContent = stats.total;
                document.getElementById('expSuccess').textContent = stats.success;
                document.getElementById('expFailure').textContent = stats.failure;
                document.getElementById('expBest').textContent = stats.best_fitness ? stats.best_fitness.toExponential(2) : '-';
                document.getElementById('expAvg').textContent = stats.avg_fitness ? stats.avg_fitness.toExponential(2) : '-';
                
                // 更新经验表
                const tbody = document.getElementById('expBody');
                const exps = data.experiences || [];
                if (exps.length > 0) {
                    tbody.innerHTML = exps.map((e, i) => {
                        const ok = e.result_status === 'ok';
                        const params = e.state ? Object.entries(e.state).slice(0, 4).map(([k,v]) => `${k}=${v}`).join(', ') + '...' : '-';
                        return `<tr>
                            <td>${i+1}</td>
                            <td class="${ok ? 'exp-ok' : 'exp-fail'}">${ok ? '✅' : '❌'}</td>
                            <td>${e.fitness ? e.fitness.toExponential(2) : '-'}</td>
                            <td>${e.reward ? e.reward.toFixed(2) : '-'}</td>
                            <td style="font-size:11px;">${params}</td>
                        </tr>`;
                    }).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="5" style="color: #666;">暂无数据</td></tr>';
                }
            } catch (e) {
                console.error(e);
            }
        }
        
        async function clearPending() {
            if (!confirm('确定清空待处理反馈？')) return;
            try {
                await fetch('/api/feedback', { method: 'DELETE' });
                showToast('✅ 已清空');
                refreshStatus();
            } catch (e) { showToast('❌ 失败', true); }
        }
        
        async function resetStrategy() {
            if (!confirm('确定重置策略状态？这将清除所有学习到的策略！')) return;
            try {
                await fetch('/api/strategy', { method: 'DELETE' });
                showToast('✅ 策略已重置');
                refreshStrategy();
            } catch (e) { showToast('❌ 失败', true); }
        }
        
        async function clearExperience() {
            if (!confirm('确定清空所有经验数据？')) return;
            try {
                await fetch('/api/experience', { method: 'DELETE' });
                showToast('✅ 经验已清空');
                refreshExperience();
            } catch (e) { showToast('❌ 失败', true); }
        }
        
        // ========== 元学习规则（ExpeL + Reflexion） ==========
        
        async function refreshExpel() {
            try {
                const res = await fetch('/api/expel-rules');
                const data = await res.json();
                
                // 统计摘要
                const summary = data.summary || {};
                document.getElementById('expelTotalRules').textContent = summary.total_rules || 0;
                document.getElementById('expelAvgConf').textContent = (summary.avg_confidence || 0).toFixed(1);
                document.getElementById('expelCompare').textContent = summary.sources?.compare || 0;
                document.getElementById('expelReflection').textContent = summary.sources?.reflection || 0;
                
                // 更新时间
                if (data.updated_at) {
                    document.getElementById('expelUpdatedAt').textContent = 
                        `最后更新: ${new Date(data.updated_at * 1000).toLocaleString('zh-CN')}`;
                }
                
                // 渲染规则列表
                renderExpelRules(data.rules || []);
                
            } catch (e) {
                console.error('刷新元学习规则失败:', e);
            }
        }
        
        function renderExpelRules(rules) {
            const list = document.getElementById('expelRulesList');
            if (rules.length === 0) {
                list.innerHTML = '<div style="color: #666;">暂无规则（运行优化后将自动生成）</div>';
                return;
            }
            
            list.innerHTML = rules.map(r => {
                // 置信度颜色
                let confColor = '#888';
                if (r.confidence >= 7) confColor = '#2ed573';
                else if (r.confidence >= 5) confColor = '#ffa502';
                else if (r.confidence <= 2) confColor = '#ff4757';
                
                // 来源图标和颜色
                const sourceInfo = {
                    'compare': { icon: '🔄', label: 'ExpeL对比', color: '#9b59b6' },
                    'success': { icon: '✅', label: 'ExpeL成功', color: '#2ed573' },
                    'failure': { icon: '❌', label: 'ExpeL失败', color: '#ff4757' },
                    'reflection': { icon: '🪞', label: 'Reflexion反思', color: '#3498db' }
                }[r.source] || { icon: '📝', label: r.source, color: '#888' };
                
                // 轮次信息
                const roundInfo = r.round ? `第${r.round}轮` : '';
                
                return `<div class="feedback-item" style="border-left-color: ${sourceInfo.color}; padding: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 12px; color: #888;">
                            #${r.index} ${sourceInfo.icon} ${sourceInfo.label} ${roundInfo}
                        </span>
                        <span style="font-size: 14px; font-weight: bold; color: ${confColor};">
                            置信度: ${r.confidence}
                        </span>
                    </div>
                    <div style="font-size: 14px; line-height: 1.5;">${r.text}</div>
                </div>`;
            }).join('');
        }
        
        async function clearExpel() {
            if (!confirm('确定清空所有规则？这将删除 ExpeL 对比规则和清空 memory_stream 中的规则。')) return;
            try {
                await fetch('/api/expel-rules', { method: 'DELETE' });
                showToast('✅ ExpeL 规则已清空');
                refreshExpel();
            } catch (e) { showToast('❌ 失败', true); }
        }
        
        // ========== 反思 (Reflexion) 相关 ==========
        
        async function refreshReflection() {
            // 重置分页状态
            resetObsPagination();
            
            try {
                const res = await fetch('/api/reflection?limit=500');  // ★ 从 100 增加到 500
                const data = await res.json();
                
                const ms = data.memory_stream || {};
                const byType = ms.by_type || {};
                
                // 更新统计
                document.getElementById('reflectionTotal').textContent = ms.total_events || 0;
                document.getElementById('reflectionReflections').textContent = byType.reflection || 0;
                document.getElementById('reflectionRules').textContent = byType.rule || 0;
                document.getElementById('reflectionObservations').textContent = byType.observation || 0;
                
                // 显示最近事件
                const events = ms.recent_events || [];
                const reflections = events.filter(e => e.event_type === 'reflection').slice(-50);  // ★ 显示最近 50 条反思
                const rules = events.filter(e => e.event_type === 'rule').slice(-30);  // ★ 显示最近 30 条规则
                const observations = events.filter(e => e.event_type === 'observation').slice(-30);  // ★ 从 10 增加到 30
                
                // 反思列表
                const reflectionList = document.getElementById('reflectionList');
                if (reflections.length > 0) {
                    reflectionList.innerHTML = reflections.reverse().map(r => `
                        <div class="feedback-card">
                            <div class="feedback-header">
                                <span class="type-badge" style="background: #9333ea;">第 ${r.round} 轮 - ${r.trigger_reason || '周期性'}</span>
                                <span class="feedback-time">${new Date(r.timestamp).toLocaleString()}</span>
                            </div>
                            <pre style="white-space: pre-wrap; font-size: 12px; background: #1e1e2e; padding: 10px; border-radius: 4px; margin-top: 8px;">${r.content || ''}</pre>
                        </div>
                    `).join('');
                } else {
                    reflectionList.innerHTML = '<div style="color: #666;">暂无反思记录</div>';
                }
                
                // 规则列表
                const ruleList = document.getElementById('reflectionRuleList');
                if (rules.length > 0) {
                    ruleList.innerHTML = rules.reverse().map(r => `
                        <div class="feedback-card">
                            <div class="feedback-header">
                                <span class="type-badge" style="background: #059669;">第 ${r.round} 轮</span>
                                <span class="feedback-priority" style="background: #374151;">重要性: ${r.importance || 5}</span>
                            </div>
                            <div class="feedback-content">${r.content || ''}</div>
                        </div>
                    `).join('');
                } else {
                    ruleList.innerHTML = '<div style="color: #666;">暂无规则</div>';
                }
                
                // 观察列表
                const obsList = document.getElementById('reflectionObsList');
                if (observations.length > 0) {
                    obsList.innerHTML = observations.reverse().map(o => {
                        const meta = o.metadata || {};
                        const statusClass = meta.success ? 'green' : 'red';
                        return `
                            <div class="feedback-card">
                                <div class="feedback-header">
                                    <span class="type-badge" style="background: ${meta.success ? '#059669' : '#dc2626'};">第 ${o.round} 轮</span>
                                    <span class="feedback-time">${new Date(o.timestamp).toLocaleString()}</span>
                                </div>
                                <div class="feedback-content">${o.content || ''}</div>
                            </div>
                        `;
                    }).join('');
                } else {
                    obsList.innerHTML = '<div style="color: #666;">暂无观察</div>';
                }
                
            } catch (e) {
                console.error('刷新反思失败:', e);
                showToast('❌ 刷新反思失败', true);
            }
        }
        
        async function clearReflection() {
            if (!confirm('确定清空反思记忆？这将删除所有历史反思和规则。')) return;
            try {
                await fetch('/api/reflection', { method: 'DELETE' });
                showToast('✅ 反思记忆已清空');
                refreshReflection();
            } catch (e) {
                showToast('❌ 清空失败', true);
            }
        }
        
        // ========== 分页状态 ==========
        // 观察记录
        let obsOffset = 0;
        const obsLimit = 100;
        let allObservations = [];
        
        // 反思记录
        let reflOffset = 0;
        const reflLimit = 50;
        let allReflections = [];
        
        // 规则记录
        let ruleOffset = 0;
        const ruleLimit = 50;
        let allRules = [];
        
        // ========== 加载更多函数 ==========
        async function loadMoreReflections() {
            try {
                const res = await fetch(`/api/reflection?limit=${reflLimit}&offset=${reflOffset}&event_type=reflection`);
                const data = await res.json();
                const ms = data.memory_stream || {};
                const events = ms.recent_events || [];
                
                if (events.length === 0) {
                    showToast('没有更多反思记录了');
                    document.getElementById('loadMoreReflBtn').disabled = true;
                    return;
                }
                
                allReflections = [...events, ...allReflections];
                reflOffset += events.length;
                
                const reflList = document.getElementById('reflectionList');
                reflList.innerHTML = allReflections.map(r => `
                    <div class="feedback-card">
                        <div class="feedback-header">
                            <span class="type-badge" style="background: #9333ea;">第 ${r.round} 轮 - ${r.trigger_reason || '周期性'}</span>
                            <span class="feedback-time">${new Date(r.timestamp).toLocaleString()}</span>
                        </div>
                        <pre style="white-space: pre-wrap; font-size: 12px; background: #1e1e2e; padding: 10px; border-radius: 4px; margin-top: 8px;">${r.content || ''}</pre>
                    </div>
                `).join('');
                
                const total = ms.filtered_count || ms.total_events || 0;
                document.getElementById('reflLoadInfo').textContent = `已加载 ${allReflections.length} / ${total} 条`;
                
                if (!ms.has_more) {
                    document.getElementById('loadMoreReflBtn').disabled = true;
                    document.getElementById('loadMoreReflBtn').textContent = '已加载全部';
                }
            } catch (e) {
                console.error('加载反思失败:', e);
                showToast('❌ 加载失败', true);
            }
        }
        
        async function loadMoreRules() {
            try {
                const res = await fetch(`/api/reflection?limit=${ruleLimit}&offset=${ruleOffset}&event_type=rule`);
                const data = await res.json();
                const ms = data.memory_stream || {};
                const events = ms.recent_events || [];
                
                if (events.length === 0) {
                    showToast('没有更多规则了');
                    document.getElementById('loadMoreRuleBtn').disabled = true;
                    return;
                }
                
                allRules = [...events, ...allRules];
                ruleOffset += events.length;
                
                const ruleList = document.getElementById('reflectionRuleList');
                ruleList.innerHTML = allRules.map(r => `
                    <div class="feedback-card">
                        <div class="feedback-header">
                            <span class="type-badge" style="background: #059669;">第 ${r.round} 轮</span>
                            <span class="feedback-priority" style="background: #374151;">重要性: ${r.importance || 5}</span>
                        </div>
                        <div class="feedback-content">${r.content || ''}</div>
                    </div>
                `).join('');
                
                const total = ms.filtered_count || ms.total_events || 0;
                document.getElementById('ruleLoadInfo').textContent = `已加载 ${allRules.length} / ${total} 条`;
                
                if (!ms.has_more) {
                    document.getElementById('loadMoreRuleBtn').disabled = true;
                    document.getElementById('loadMoreRuleBtn').textContent = '已加载全部';
                }
            } catch (e) {
                console.error('加载规则失败:', e);
                showToast('❌ 加载失败', true);
            }
        }
        
        async function loadMoreObservations() {
            try {
                const res = await fetch(`/api/reflection?limit=${obsLimit}&offset=${obsOffset}&event_type=observation`);
                const data = await res.json();
                const ms = data.memory_stream || {};
                const events = ms.recent_events || [];
                
                if (events.length === 0) {
                    showToast('没有更多观察记录了');
                    document.getElementById('loadMoreObsBtn').disabled = true;
                    return;
                }
                
                // 累加到已有数据
                allObservations = [...events, ...allObservations];
                obsOffset += events.length;
                
                // 更新显示
                const obsList = document.getElementById('reflectionObsList');
                obsList.innerHTML = allObservations.map(o => {
                    const meta = o.metadata || {};
                    return `
                        <div class="feedback-card">
                            <div class="feedback-header">
                                <span class="type-badge" style="background: ${meta.success ? '#059669' : '#dc2626'};">第 ${o.round} 轮</span>
                                <span class="feedback-time">${new Date(o.timestamp).toLocaleString()}</span>
                            </div>
                            <div class="feedback-content">${o.content || ''}</div>
                        </div>
                    `;
                }).join('');
                
                // 更新信息
                const total = ms.filtered_count || ms.total_events || 0;
                document.getElementById('obsLoadInfo').textContent = `已加载 ${allObservations.length} / ${total} 条`;
                
                if (!ms.has_more) {
                    document.getElementById('loadMoreObsBtn').disabled = true;
                    document.getElementById('loadMoreObsBtn').textContent = '已加载全部';
                }
            } catch (e) {
                console.error('加载观察失败:', e);
                showToast('❌ 加载失败', true);
            }
        }
        
        // 重置所有分页状态
        function resetAllPagination() {
            // 观察
            obsOffset = 0;
            allObservations = [];
            const obsBtn = document.getElementById('loadMoreObsBtn');
            if (obsBtn) { obsBtn.disabled = false; obsBtn.textContent = '加载更多观察'; }
            const obsInfo = document.getElementById('obsLoadInfo');
            if (obsInfo) obsInfo.textContent = '';
            
            // 反思
            reflOffset = 0;
            allReflections = [];
            const reflBtn = document.getElementById('loadMoreReflBtn');
            if (reflBtn) { reflBtn.disabled = false; reflBtn.textContent = '加载更多反思'; }
            const reflInfo = document.getElementById('reflLoadInfo');
            if (reflInfo) reflInfo.textContent = '';
            
            // 规则
            ruleOffset = 0;
            allRules = [];
            const ruleBtn = document.getElementById('loadMoreRuleBtn');
            if (ruleBtn) { ruleBtn.disabled = false; ruleBtn.textContent = '加载更多规则'; }
            const ruleInfo = document.getElementById('ruleLoadInfo');
            if (ruleInfo) ruleInfo.textContent = '';
        }
        
        // 兼容旧名称
        function resetObsPagination() {
            resetAllPagination();
        }
        
        // ========== 评论家相关 ==========
        
        async function refreshCritic() {
            try {
                const res = await fetch('/api/critic');
                const data = await res.json();
                
                // 集群统计
                const tdLearning = data.td_learning || {};
                const vf = data.value_function || {};
                
                const ensembleAcc = tdLearning.ensemble_accuracy || 0;
                document.getElementById('criticEnsembleAcc').textContent = (ensembleAcc * 100).toFixed(1) + '%';
                document.getElementById('criticEnsembleAcc').className = 'stat-value ' + (ensembleAcc > 0.6 ? 'green' : ensembleAcc > 0.4 ? 'yellow' : 'red');
                
                document.getElementById('criticTDRules').textContent = tdLearning.total_td_rules || 0;
                document.getElementById('criticVFStates').textContent = vf.total_states || 0;
                
                const bestFit = vf.best_fitness;
                document.getElementById('criticBestFitness').textContent = bestFit ? bestFit.toExponential(2) : '-';
                
                // 更新时间
                if (data.updated_at) {
                    document.getElementById('criticUpdatedAt').textContent = 
                        `最后更新: ${new Date(data.updated_at).toLocaleString('zh-CN')}`;
                }
                
                // 价值函数
                document.getElementById('vfTotalStates').textContent = vf.total_states || 0;
                document.getElementById('vfAvgValue').textContent = (vf.avg_value || 0).toFixed(3);
                const vfPct = ((vf.avg_value || 0) + 1) / 2 * 100;  // -1~1 -> 0~100
                document.getElementById('vfValueBar').style.width = Math.max(0, Math.min(100, vfPct)) + '%';
                
                // 渲染各评论家详情
                renderCriticDetails(data.critics || {});
                
                // 渲染 TD 统计
                renderTDStatistics(data.critics || {});
                
                // 渲染策略库
                renderCriticStrategies(data.critics || {});
                
            } catch (e) {
                console.error('加载评论家数据失败:', e);
            }
        }
        
        function renderCriticDetails(critics) {
            const grid = document.getElementById('criticDetailGrid');
            
            const criticLabels = {
                'magnetic': { name: '🧲 磁路评论家', desc: '评估 ta, tb_ratio, dg 对磁饱和的影响', color: '#ff6b6b' },
                'performance': { name: '📈 性能评论家', desc: '预估整体参数改动对 fitness 的影响', color: '#4ecdc4' },
                'constraint': { name: '⚠️ 约束评论家', desc: '快速约束预检，评估违规风险', color: '#ffe66d' },
                'magnitude': { name: '📏 幅度评论家', desc: '评估参数改动幅度的合理性', color: '#a55eea' }
            };
            
            let html = '';
            for (const [type, info] of Object.entries(critics)) {
                const label = criticLabels[type] || { name: type, desc: '', color: '#888' };
                const conf = info.dynamic_confidence || 0.5;
                const acc = info.prediction_accuracy || 0.5;
                const total = info.total_predictions || 0;
                const correct = info.correct_predictions || 0;
                const expCount = info.experience_count || 0;
                const stratCount = info.strategy_count || 0;
                const tdCount = Object.keys(info.td_statistics || {}).length;
                
                // 置信度颜色和图标
                const confEmoji = conf > 0.6 ? '📈' : conf < 0.4 ? '📉' : '➡️';
                const confClass = conf > 0.6 ? 'green' : conf < 0.4 ? 'red' : 'yellow';
                const accClass = acc > 0.6 ? 'green' : acc < 0.4 ? 'red' : 'yellow';
                
                html += `
                <div style="background: rgba(0,0,0,0.2); border-radius: 8px; padding: 15px; border-left: 4px solid ${label.color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <strong style="color: ${label.color};">${label.name}</strong>
                        <span style="font-size: 11px; color: #666;">${info.enabled ? '✓ 已启用' : '✗ 未启用'}</span>
                    </div>
                    <div style="font-size: 11px; color: #888; margin-bottom: 12px;">${label.desc}</div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px;">
                        <div>
                            <span style="color: #666;">置信度:</span>
                            <span class="${confClass}">${confEmoji} ${(conf * 100).toFixed(0)}%</span>
                        </div>
                        <div>
                            <span style="color: #666;">准确率:</span>
                            <span class="${accClass}">${(acc * 100).toFixed(0)}%</span>
                        </div>
                        <div>
                            <span style="color: #666;">预测:</span>
                            <span>${correct}/${total}</span>
                        </div>
                        <div>
                            <span style="color: #666;">TD统计:</span>
                            <span>${tdCount}条</span>
                        </div>
                        <div>
                            <span style="color: #666;">经验:</span>
                            <span>${expCount}条</span>
                        </div>
                        <div>
                            <span style="color: #666;">策略:</span>
                            <span>${stratCount}条</span>
                        </div>
                    </div>
                    
                    <div class="progress-bar" style="margin-top: 10px; height: 6px;">
                        <div class="progress-fill" style="width: ${conf * 100}%; background: linear-gradient(90deg, ${label.color}, ${label.color}88);"></div>
                    </div>
                </div>`;
            }
            
            grid.innerHTML = html || '<div style="color: #666;">暂无评论家数据</div>';
        }
        
        function renderTDStatistics(critics) {
            const list = document.getElementById('criticTDStatsList');
            let allStats = [];
            
            const criticLabels = {
                'magnetic': '🧲',
                'performance': '📈',
                'constraint': '⚠️',
                'magnitude': '📏'
            };
            
            for (const [type, info] of Object.entries(critics)) {
                const stats = info.td_statistics || {};
                for (const [key, stat] of Object.entries(stats)) {
                    allStats.push({
                        critic: type,
                        emoji: criticLabels[type] || '📋',
                        ...stat
                    });
                }
            }
            
            if (allStats.length === 0) {
                list.innerHTML = '<div style="color: #666;">暂无 TD 统计数据（需要运行优化积累经验）</div>';
                return;
            }
            
            // 按样本数排序
            allStats.sort((a, b) => (b.count || 0) - (a.count || 0));
            
            list.innerHTML = allStats.slice(0, 20).map(stat => {
                const direction = stat.direction === 'increase' ? '↑ 增大' : '↓ 减小';
                const tdErr = stat.avg_td_error || 0;
                const predErr = stat.avg_prediction_error || 0;
                const acc = stat.accuracy_rate || 0.5;
                
                const tdClass = tdErr > 0.1 ? 'green' : tdErr < -0.1 ? 'red' : '';
                const predBias = predErr > 0.1 ? '偏保守' : predErr < -0.1 ? '偏乐观' : '准确';
                const predClass = predErr > 0.1 ? 'green' : predErr < -0.1 ? 'red' : 'yellow';
                const accClass = acc > 0.6 ? 'green' : acc < 0.4 ? 'red' : 'yellow';
                
                return `<div class="feedback-item" style="border-left-color: #00d9ff;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <strong>${stat.emoji} ${stat.param_name} ${direction}</strong>
                        <span style="font-size: 11px; color: #666;">样本: ${stat.count || 0}</span>
                    </div>
                    <div style="font-size: 12px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 5px;">
                        <span>TD误差: <span class="${tdClass}">${tdErr >= 0 ? '+' : ''}${tdErr.toFixed(3)}</span></span>
                        <span>预测: <span class="${predClass}">${predBias}</span></span>
                        <span>准确率: <span class="${accClass}">${(acc * 100).toFixed(0)}%</span></span>
                    </div>
                </div>`;
            }).join('');
        }
        
        function renderCriticStrategies(critics) {
            const list = document.getElementById('criticStrategiesList');
            let allStrategies = [];
            
            const criticLabels = {
                'magnetic': { emoji: '🧲', name: '磁路', color: '#ff6b6b' },
                'performance': { emoji: '📈', name: '性能', color: '#4ecdc4' },
                'constraint': { emoji: '⚠️', name: '约束', color: '#ffe66d' },
                'magnitude': { emoji: '📏', name: '幅度', color: '#a55eea' }
            };
            
            for (const [type, info] of Object.entries(critics)) {
                const strategies = info.strategies || [];
                const label = criticLabels[type] || { emoji: '📋', name: type, color: '#888' };
                strategies.forEach(s => {
                    allStrategies.push({
                        critic: type,
                        ...label,
                        text: s,
                        isTD: s.includes('TD误差')
                    });
                });
            }
            
            if (allStrategies.length === 0) {
                list.innerHTML = '<div style="color: #666;">暂无策略规则（评论家会在优化过程中自动学习）</div>';
                return;
            }
            
            // TD 规则优先显示
            allStrategies.sort((a, b) => (b.isTD ? 1 : 0) - (a.isTD ? 1 : 0));
            
            list.innerHTML = allStrategies.map(s => {
                const tdBadge = s.isTD ? '<span style="background: #00d9ff33; color: #00d9ff; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin-left: 5px;">TD学习</span>' : '';
                return `<div class="feedback-item" style="border-left-color: ${s.color};">
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 12px; color: ${s.color};">${s.emoji} ${s.name}</span>
                        ${tdBadge}
                    </div>
                    <div style="font-size: 13px;">${s.text}</div>
                </div>`;
            }).join('');
        }
        
        async function clearAllCritic() {
            if (!confirm('确定清空所有评论家数据？\\n这将清除：经验、TD统计、价值函数历史')) return;
            
            try {
                const res = await fetch('/api/critic/all', { method: 'DELETE' });
                const data = await res.json();
                if (data.success) {
                    showToast('✅ 所有评论家数据已清空');
                    refreshCritic();
                } else {
                    showToast('❌ ' + data.error, true);
                }
            } catch (e) {
                showToast('❌ 请求失败', true);
            }
        }
        
        // ========== 远程 Actor 相关 ==========
        
        async function refreshActor() {
            try {
                const res = await fetch('/api/actor/status');
                const data = await res.json();
                
                // 更新状态显示
                document.getElementById('actorConnected').textContent = data.connected ? '✅ 已连接' : '❌ 未连接';
                document.getElementById('actorConnected').style.color = data.connected ? '#4ecdc4' : '#ff6b6b';
                document.getElementById('actorGpuCount').textContent = data.gpu_count || '-';
                document.getElementById('actorUpdateCount').textContent = data.update_count || 0;
                document.getElementById('actorBufferSize').textContent = data.buffer_size || 0;
                document.getElementById('actorTotalSteps').textContent = data.total_steps || 0;
                
                // ★ 根据训练模式更新显示
                const mode = (data.mode || 'ppo').toUpperCase();
                document.getElementById('trainingModeTitle').textContent = mode;
                document.getElementById('updateCountLabel').textContent = mode + ' 更新次数';  // ★ 动态标签
                const badge = document.getElementById('trainingModeBadge');
                badge.textContent = mode;
                if (mode === 'DPO') {
                    badge.style.background = '#a55eea';
                    badge.style.color = '#fff';
                    document.getElementById('dpoStatsSection').style.display = 'block';
                    document.getElementById('ppoStatsSection').style.display = 'none';  // ★ 隐藏 PPO 专属指标
                } else {
                    badge.style.background = '#4ecdc4';
                    badge.style.color = '#1a1a2e';
                    document.getElementById('dpoStatsSection').style.display = 'none';
                    document.getElementById('ppoStatsSection').style.display = 'grid';  // ★ 显示 PPO 专属指标
                }
                
                // 显示错误信息
                const errorDiv = document.getElementById('actorError');
                if (data.error) {
                    errorDiv.textContent = '⚠️ ' + data.error;
                    errorDiv.style.display = 'block';
                } else {
                    errorDiv.style.display = 'none';
                }
                
                // 绘制训练曲线（PPO）
                const stats = data.training_stats || {};
                drawSimpleChart('policyLossCanvas', stats.policy_loss || [], '#ff6b6b', 'policyLossLatest');
                drawSimpleChart('valueLossCanvas', stats.value_loss || [], '#4ecdc4', 'valueLossLatest');
                drawSimpleChart('entropyCanvas', stats.entropy || [], '#ffe66d', 'entropyLatest');
                drawSimpleChart('klCanvas', stats.kl_divergence || [], '#a55eea', 'klLatest');
                drawSimpleChart('bcLossCanvas', stats.bc_loss || [], '#ff6b9d', 'bcLossLatest');
                
                // ★ DPO 专属统计
                if (mode === 'DPO') {
                    drawSimpleChart('dpoLossCanvas', stats.dpo_loss || [], '#a55eea', 'dpoLossLatest');
                    document.getElementById('dpoPairsCount').textContent = data.pairs_count || 0;
                    document.getElementById('dpoTotalExperiences').textContent = data.total_experiences || 0;
                }
                
            } catch (e) {
                console.error('加载 Actor 状态失败:', e);
                document.getElementById('actorConnected').textContent = '❌ 请求失败';
            }
        }
        
        function drawSimpleChart(canvasId, data, color, latestId) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            const width = canvas.parentElement.clientWidth - 20;
            const height = 130;
            
            canvas.width = width;
            canvas.height = height;
            
            ctx.clearRect(0, 0, width, height);
            
            // 更新最新值
            const latestEl = document.getElementById(latestId);
            if (latestEl) {
                latestEl.textContent = data.length > 0 ? data[data.length - 1].toFixed(4) : '-';
            }
            
            if (data.length < 2) {
                ctx.fillStyle = '#666';
                ctx.font = '12px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('暂无数据', width / 2, height / 2);
                return;
            }
            
            // 计算范围
            const minVal = Math.min(...data);
            const maxVal = Math.max(...data);
            const range = maxVal - minVal || 1;
            
            // 绘制网格
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 4; i++) {
                const y = height * i / 4;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }
            
            // 绘制曲线
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let i = 0; i < data.length; i++) {
                const x = (i / (data.length - 1)) * width;
                const y = height - ((data[i] - minVal) / range) * (height - 20) - 10;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            
            // 绘制点
            ctx.fillStyle = color;
            for (let i = 0; i < data.length; i++) {
                const x = (i / (data.length - 1)) * width;
                const y = height - ((data[i] - minVal) / range) * (height - 20) - 10;
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        
        async function setActorUrl() {
            const url = document.getElementById('actorUrlInput').value.trim();
            if (!url) {
                showToast('❌ 请输入 URL', true);
                return;
            }
            
            try {
                const res = await fetch('/api/actor/set-url', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url })
                });
                const data = await res.json();
                if (data.success) {
                    showToast('✅ URL 已设置: ' + data.url);
                    refreshActor();
                } else {
                    showToast('❌ ' + data.error, true);
                }
            } catch (e) {
                showToast('❌ 请求失败', true);
            }
        }
        
        async function forceActorUpdate() {
            if (!confirm('确定强制执行 PPO 更新？')) return;
            
            try {
                showToast('⏳ 正在执行 PPO 更新...');
                const res = await fetch('/api/actor/force-update', { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    showToast('✅ PPO 更新完成');
                    refreshActor();
                } else {
                    showToast('❌ ' + data.error, true);
                }
            } catch (e) {
                showToast('❌ 请求失败', true);
            }
        }
        
        async function saveActorCheckpoint() {
            try {
                showToast('⏳ 正在保存检查点...');
                const res = await fetch('/api/actor/save-checkpoint', { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    showToast('✅ 检查点已保存: ' + (data.result?.path || ''));
                } else {
                    showToast('❌ ' + data.error, true);
                }
            } catch (e) {
                showToast('❌ 请求失败', true);
            }
        }
        
        function refreshAll() {
            refreshStatus();
            refreshStrategy();
            refreshExperience();
        }
        
        // ========== 历史反馈相关 ==========
        
        async function refreshHistory() {
            try {
                const res = await fetch('/api/feedback/history');
                const data = await res.json();
                const list = document.getElementById('historyList');
                const feedbacks = data.feedbacks || [];
                
                if (feedbacks.length > 0) {
                    list.innerHTML = feedbacks.map(fb => {
                        const fbType = fb.feedback_type || fb.type || 'suggestion';
                        const typeClass = fbType === 'warning' ? 'warning' : 
                                         fbType === 'correction' ? 'correction' : 
                                         fbType === 'confirmation' ? 'confirmation' : '';
                        const typeEmoji = fbType === 'warning' ? '⚠️' : 
                                         fbType === 'correction' ? '🔴' : 
                                         fbType === 'confirmation' ? '✅' : '💡';
                        const time = fb.timestamp ? new Date(fb.timestamp * 1000).toLocaleString('zh-CN') : '';
                        const strength = fb.strength !== undefined ? fb.strength : 1.0;
                        const strengthPct = Math.round(strength * 100);
                        const strengthClass = strength >= 0.6 ? 'high' : strength >= 0.3 ? 'medium' : strength > 0 ? 'low' : 'off';
                        const strengthLabel = strength >= 0.9 ? '🔴 强烈' : 
                                             strength >= 0.6 ? '🟠 中等' : 
                                             strength >= 0.3 ? '🟡 一般' : 
                                             strength > 0 ? '🟢 弱' : '⚪ 关闭';
                        return `<div class="history-item ${typeClass}" style="${strength === 0 ? 'opacity: 0.5;' : ''}">
                            <div class="fb-header">
                                <span class="fb-type">${typeEmoji} ${fbType}</span>
                                <span class="fb-time">${time}</span>
                            </div>
                            <div class="fb-text">${fb.text}</div>
                            <div class="strength-control">
                                <label>引导强度:</label>
                                <input type="range" class="strength-slider" min="0" max="100" value="${strengthPct}" 
                                       onchange="updateStrength('${fb.id}', this.value)" 
                                       oninput="this.nextElementSibling.textContent = this.value + '%'; this.nextElementSibling.className = 'strength-value ' + (this.value >= 60 ? 'high' : this.value >= 30 ? 'medium' : this.value > 0 ? 'low' : 'off')">
                                <span class="strength-value ${strengthClass}">${strengthPct}%</span>
                            </div>
                            <div class="fb-actions">
                                <button class="btn-edit" onclick="openEditModal('${fb.id}', '${escapeHtml(fb.text)}', '${fbType}', ${strength})">✏️ 编辑</button>
                                <button class="btn-delete" onclick="deleteFeedback('${fb.id}')">🗑️ 删除</button>
                            </div>
                        </div>`;
                    }).join('');
                } else {
                    list.innerHTML = '<div style="color: #666;">暂无历史反馈</div>';
                }
            } catch (e) {
                console.error(e);
            }
        }
        
        function escapeHtml(str) {
            return str.replace(/'/g, "\\'").replace(/"/g, '\\"').replace(/\\n/g, ' ');
        }
        
        function openEditModal(id, text, type, strength = 1.0) {
            document.getElementById('editFeedbackId').value = id;
            document.getElementById('editText').value = text.replace(/\\'/g, "'").replace(/\\"/g, '"');
            document.getElementById('editType').value = type;
            document.getElementById('editStrength').value = Math.round(strength * 100);
            updateStrengthDisplay('edit');
            document.getElementById('editModal').classList.add('show');
        }
        
        function closeEditModal() {
            document.getElementById('editModal').classList.remove('show');
        }
        
        function updateStrengthDisplay(prefix) {
            const slider = document.getElementById(prefix + 'Strength');
            const display = document.getElementById(prefix + 'StrengthValue');
            const val = parseInt(slider.value);
            display.textContent = val + '%';
            display.className = 'strength-value ' + (val >= 60 ? 'high' : val >= 30 ? 'medium' : val > 0 ? 'low' : 'off');
        }
        
        async function updateStrength(id, value) {
            const strength = parseInt(value) / 100;
            try {
                const res = await fetch(`/api/feedback/${id}/strength`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ strength })
                });
                const data = await res.json();
                if (data.success) {
                    showToast(`✅ 引导强度已更新为 ${value}%`);
                } else {
                    showToast('❌ ' + data.error, true);
                    refreshHistory();
                }
            } catch (e) {
                showToast('❌ 请求失败', true);
            }
        }
        
        async function saveEdit() {
            const id = document.getElementById('editFeedbackId').value;
            const text = document.getElementById('editText').value.trim();
            const type = document.getElementById('editType').value;
            const strength = parseInt(document.getElementById('editStrength').value) / 100;
            
            if (!text) {
                showToast('反馈内容不能为空', true);
                return;
            }
            
            try {
                const res = await fetch(`/api/feedback/${id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, type, strength })
                });
                const data = await res.json();
                if (data.success) {
                    showToast('✅ 反馈已更新');
                    closeEditModal();
                    refreshHistory();
                } else {
                    showToast('❌ ' + data.error, true);
                }
            } catch (e) {
                showToast('❌ 请求失败', true);
            }
        }
        
        async function deleteFeedback(id) {
            if (!confirm('确定删除这条反馈？')) return;
            
            try {
                const res = await fetch(`/api/feedback/${id}`, { method: 'DELETE' });
                const data = await res.json();
                if (data.success) {
                    showToast('✅ 反馈已删除');
                    refreshHistory();
                    refreshStatus();
                } else {
                    showToast('❌ ' + data.error, true);
                }
            } catch (e) {
                showToast('❌ 请求失败', true);
            }
        }
        
        // 点击弹窗外部关闭
        document.getElementById('editModal').addEventListener('click', function(e) {
            if (e.target === this) closeEditModal();
        });
        
        // 更新 refreshAll
        function refreshAll() {
            refreshStatus();
            refreshHistory();
            refreshStrategy();
            refreshExperience();
        }
        
        // 初始化
        refreshStatus();
        refreshHistory();
        setInterval(refreshStatus, 5000);
        setInterval(refreshHistory, 10000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE


def main():
    parser = argparse.ArgumentParser(description="反馈 Web 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8888, help="监听端口")
    args = parser.parse_args()
    
    print(f"\n🚀 反馈服务启动中...")
    print(f"📍 访问地址: http://{args.host}:{args.port}")
    print(f"📁 反馈文件: {FEEDBACK_INPUT_FILE}\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

