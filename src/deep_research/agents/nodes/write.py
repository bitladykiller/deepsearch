"""写作节点：生成最终 Markdown 研究报告。"""

import json
import logging
import re

from langchain_core.messages import HumanMessage

from .base import (
    _last_content,
    _validate_and_fix_citations,
    _ensure_reference_section,
    _render_fallback_report,
    with_memory_context,
    emit,
)
from ..state import ResearchState
from deep_research.utils import colorize

logger = logging.getLogger("deep_research")


def write_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """写作节点：撰写最终 Markdown 研报。"""
    logger.info("%s 开始 | agent=%s", colorize("[write]", "cyan"), colorize(agent_name, "magenta"))
    valid_source_ids = [str(item.get("source_id", "")).strip() for item in state.get("source_index", []) if item.get("source_id")]
    valid_source_ids = [item for item in valid_source_ids if item][:80]
    valid_source_ids_set = set(valid_source_ids)

    prompt = (
        "请严格根据以下信息撰写最终的 Markdown 研报。请直接输出正文，绝对不要输出任何 JSON 结构，也不要复述你的指令。\n\n"
        f"核心问题：{state['query']}\n"
        f"子问题拆解：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n\n"
        "【分析结论 (Findings)】：\n"
        f"{json.dumps(state.get('findings', []), ensure_ascii=False)}\n\n"
        "【可用来源索引 (source_index)】：\n"
        f"{json.dumps(state.get('source_index', []), ensure_ascii=False)}\n\n"
        "【合法引用ID列表】：\n"
        f"{json.dumps(valid_source_ids, ensure_ascii=False)}\n\n"
        "【可能存在的风险/冲突 (Audit Flags)】：\n"
        f"{json.dumps(state.get('audit_flags', []), ensure_ascii=False)}\n\n"
        "要求：正文必须使用合法引用ID（例如 [WEB1_1-1]、[LOC1_1-3]）；禁止使用不存在的编号。"
        "结尾不需要你来列举引用列表，系统会自动拼接。"
    )
    human = HumanMessage(content=with_memory_context(state, prompt))

    result = agent.invoke({"messages": [human]})
    content = _last_content(result)

    # 强制清理可能的错误 JSON 代码块
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```markdown\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"```$", "", content.strip())

    # 校验并修正引用ID，移除非法引用
    content, used_citation_ids = _validate_and_fix_citations(content, valid_source_ids_set)

    final_content = _ensure_reference_section(content, state)
    emit("write", final_content)
    return {"draft": final_content, "final": final_content, "messages": [human, result["messages"][-1]]}
