"""证据裁判节点：评分、去重、冲突审计。"""

import json
import logging

from .base import (
    _invoke_json_agent,
    _fallback_audit,
    _score_evidence,
    _dedupe_sources,
)
from ..state import ResearchState
from deep_research.utils import colorize

logger = logging.getLogger("deep_research")


def deep_dive_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """证据裁判节点：对 web 和 local 证据进行评分、去重、冲突审计。"""
    logger.info("%s 开始 | agent=%s", colorize("[deep_dive]", "cyan"), colorize(agent_name, "magenta"))
    if not state.get("web_evidence") and not state.get("local_evidence"):
        logger.info("%s 等待检索结果", colorize("[deep_dive]", "yellow"))
        return {}
    fallback = _fallback_audit(state)
    payload, content, messages = _invoke_json_agent(
        state,
        "请对 web 与 local 证据进行评分、去重、冲突审计，并只输出 JSON。\n"
        f"问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"web_evidence：{json.dumps(state.get('web_evidence', []), ensure_ascii=False)}\n"
        f"local_evidence：{json.dumps(state.get('local_evidence', []), ensure_ascii=False)}",
        agent, agent_name, "deep_dive", fallback,
    )
    payload_pool = payload.get("evidence_pool") if isinstance(payload.get("evidence_pool"), list) else []
    raw_evidence = state.get("web_evidence", []) + state.get("local_evidence", [])
    allowed_source_ids = {str(item.get("source_id", "")).strip() for item in raw_evidence if item.get("source_id")}
    evidence_pool = []
    for item in payload_pool:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("source_id", "")).strip()
        if sid and sid in allowed_source_ids:
            evidence_pool.append(item)
    if not evidence_pool:
        evidence_pool = fallback["evidence_pool"]
    existing_ids = {str(item.get("source_id", "")).strip() for item in evidence_pool if isinstance(item, dict)}
    for record in raw_evidence:
        sid = str(record.get("source_id", "")).strip()
        if not sid or sid in existing_ids:
            continue
        score, reason = _score_evidence(record)
        evidence_pool.append({
            "source_id": sid,
            "source_type": record.get("source_type", "source"),
            "title": record.get("title") or sid,
            "url": record.get("url", ""),
            "doc_id": record.get("doc_id", ""),
            "snippet": record.get("snippet", ""),
            "supports_questions": record.get("supports_questions", []),
            "reliability_score": score,
            "reliability_reason": reason,
            "source_label": record.get("title") or record.get("doc_id") or record.get("url") or sid,
        })
        existing_ids.add(sid)
    audit_flags = payload.get("audit_flags") if isinstance(payload.get("audit_flags"), list) else fallback["audit_flags"]
    source_index = []
    for item in evidence_pool:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("source_id", "")).strip()
        if not sid:
            continue
        source_index.append({
            "source_id": sid,
            "label": item.get("title") or item.get("source_label") or sid,
            "locator": item.get("url") or item.get("doc_id") or "",
            "source_type": item.get("source_type", "source"),
        })
    source_index = _dedupe_sources(source_index, ["source_id"])
    return {
        "deep_dive": payload.get("summary", content),
        "audit": payload.get("summary", content),
        "evidence_pool": evidence_pool,
        "audit_flags": audit_flags,
        "source_index": source_index,
        "messages": messages,
    }
