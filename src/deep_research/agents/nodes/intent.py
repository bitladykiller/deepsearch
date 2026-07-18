"""意图识别与直接回答节点。"""

import re
import logging

from langchain_core.messages import HumanMessage

from .base import (
    _invoke_json_agent,
    _last_content,
    with_memory_context,
)
from ..state import ResearchState
from deep_research.utils import colorize, emit

logger = logging.getLogger("deep_research")


def detect_intent(query: str) -> str:
    """基于关键词规则引擎判断用户意图：direct 或 multiagent。"""
    normalized_query = query.strip()
    force_multiagent_keywords = [
        "调查", "调研", "来源", "证据", "检索统计", "来源清单",
        "重大新闻", "热门项目", "趋势", "新闻", "最新", "盘点",
    ]
    if re.search(r"20\d{2}年", normalized_query) and any(
        word in normalized_query for word in ["趋势", "新闻", "调研", "调查", "盘点"]
    ):
        return "multiagent"
    if any(word in query for word in force_multiagent_keywords):
        return "multiagent"
    keywords = [
        "调研", "研究", "调查", "盘点", "热门", "趋势", "榜单", "分析",
        "方案", "架构", "设计", "对比", "报告", "代码", "实现", "落地",
        "检索", "知识库", "证据", "来源", "溯源", "资料", "手册", "验证",
        "数据", "模型",
    ]
    return "multiagent" if any(word in query for word in keywords) else "direct"


def intent_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """意图识别节点：调用 Agent 判断路由方向。"""
    logger.info("%s 开始 | agent=%s", colorize("[intent]", "cyan"), colorize(agent_name, "magenta"))
    rule_route = detect_intent(state["query"])
    prompt = (
        f"用户问题：{state['query']}\n"
        f"规则引擎初判：{rule_route}\n"
        "请输出 JSON：{\"route\":\"direct|multiagent\",\"reason\":\"...\"}"
    )
    payload, content, messages = _invoke_json_agent(
        state, prompt, agent, agent_name, "intent",
        {"route": rule_route, "reason": "rule"},
    )
    route = str(payload.get("route", rule_route)).strip().lower()
    if route not in {"direct", "multiagent"}:
        route = rule_route
    logger.info("%s 路由: %s", colorize("[intent]", "green"), route)
    return {"intent": route, "draft": content, "messages": messages}


def direct_answer_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """直接回答节点：简单问答无需走研究流程。"""
    logger.info("%s 开始 | agent=%s", colorize("[direct_answer]", "cyan"), colorize(agent_name, "magenta"))
    prompt = f"用户问题：{state['query']}"
    human = HumanMessage(content=with_memory_context(state, prompt))
    result = agent.invoke({"messages": [human]})
    content = _last_content(result).strip()
    emit("direct_answer", content)
    return {
        "intent": "direct",
        "final": content,
        "draft": content,
        "analysis_summary": content,
        "needs_more_research": False,
        "messages": [human, result["messages"][-1]],
    }
