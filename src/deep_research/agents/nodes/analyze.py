"""分析与反思节点：生成结论并判断是否需要补搜。"""

import json
import logging

from .base import (
    _invoke_json_agent,
    _fallback_analysis,
    log_inputs,
)
from ..state import ResearchState
from deep_research.utils import colorize

logger = logging.getLogger("deep_research")


def analyze_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """分析节点：基于证据池生成结论并评估证据完备性。"""
    logger.info("%s 开始 | agent=%s", colorize("[analyze]", "cyan"), colorize(agent_name, "magenta"))
    fallback = _fallback_analysis(state)
    payload, content, messages = _invoke_json_agent(
        state,
        "请基于证据池输出结论映射 JSON，并评估证据完备性：\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"证据池：{json.dumps(state.get('evidence_pool', []), ensure_ascii=False)}\n"
        f"审计标记：{json.dumps(state.get('audit_flags', []), ensure_ascii=False)}",
        agent, agent_name, "analyze", fallback,
    )
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else fallback["findings"]
    claim_map = payload.get("claim_map") if isinstance(payload.get("claim_map"), list) else fallback["claim_map"]
    needs_more_research = payload.get("needs_more_research", False)
    missing_gaps = payload.get("missing_gaps", [])
    analysis_summary = payload.get("analysis_summary", content)
    return {
        "analysis": analysis_summary,
        "findings": findings,
        "claim_map": claim_map,
        "needs_more_research": needs_more_research,
        "missing_gaps": missing_gaps,
        "messages": messages,
    }


def reflect_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """反思节点：根据分析师反馈生成补搜计划。"""
    logger.info("%s 开始 | agent=%s", colorize("[reflect]", "cyan"), colorize(agent_name, "magenta"))
    missing_gaps = state.get("missing_gaps", [])
    log_inputs("reflect", agent_name, {"missing_gaps": str(missing_gaps)})

    fallback = {
        "reflection_summary": "默认补搜",
        "supplementary_queries": [{"section_id": "gap_1", "query": state["query"], "source_preference": "hybrid", "reason": "fallback"}],
    }

    prompt = (
        f"分析师指出当前证据不足以完全回答问题，存在以下信息缺口：\n{json.dumps(missing_gaps, ensure_ascii=False)}\n\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"已执行过的搜索计划：\n{json.dumps(state.get('search_plan', []), ensure_ascii=False)}\n"
        f"已执行过的补搜计划：\n{json.dumps(state.get('supplementary_queries', []), ensure_ascii=False)}\n\n"
        "请生成新的补搜计划以填补缺口。"
    )

    payload, content, messages = _invoke_json_agent(
        state, prompt, agent, agent_name, "reflect", fallback,
    )

    return {
        "iteration": state.get("iteration", 0) + 1,
        "supplementary_queries": payload.get("supplementary_queries", fallback["supplementary_queries"]),
        "messages": messages,
    }
