"""基础工具函数模块：所有节点共用的辅助函数。

本模块包含 JSON 解析、搜索计划构建、证据过滤/评分、引用校验等
共享逻辑，供 intent / plan / search / evidence / analyze / write 等
节点模块按需导入。
"""

import json
import logging
import re
from functools import partial

from langchain_core.messages import HumanMessage

from ..state import ResearchState
from ..tools import bocha_web_search_records, search_knowledge_base_records
from deep_research.utils import colorize, emit, collect_tool_calls, with_memory_context, log_inputs

logger = logging.getLogger("mult_agents")


# ---------------------------------------------------------------------------
# Agent 绑定
# ---------------------------------------------------------------------------

def bind_agent(node_func, agent, agent_name: str):
    """将 agent 和 agent_name 绑定到节点函数，返回单参数 callable。"""
    return partial(node_func, agent=agent, agent_name=agent_name)


# ---------------------------------------------------------------------------
# JSON / LLM 交互辅助
# ---------------------------------------------------------------------------

def _last_content(result) -> str:
    """从 Agent 结果中提取最后一条消息的文本内容。"""
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _extract_json_block(text: str) -> str:
    """从 LLM 输出中提取 JSON 块（兼容 markdown 代码块格式）。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _load_json(text: str, fallback: dict) -> dict:
    """安全解析 JSON 字符串，失败时返回 fallback。"""
    try:
        value = json.loads(_extract_json_block(text))
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return fallback


def _invoke_json_agent(state: ResearchState, prompt: str, agent, agent_name: str, node: str, fallback: dict) -> tuple[dict, str, list]:
    """调用 Agent 并解析 JSON 输出，返回 (payload_dict, raw_content, messages)。"""
    human = HumanMessage(content=with_memory_context(state, prompt))
    # Optimization: Do NOT pass state["messages"] to avoid token accumulation
    # Each node only needs its specific instruction and the current state data
    result = agent.invoke({"messages": [human]})
    tools, tool_outputs = collect_tool_calls(result["messages"])
    logger.info("%s 工具: %s", colorize(f"[{node}]", "green"), ", ".join(tools) if tools else "无")
    for item in tool_outputs[:5]:
        logger.info("%s 工具输出: %s", colorize(f"[{node}]", "green"), item[:400])
    logger.info("%s LLM调用: 是 | 思考: 不可见", colorize(f"[{node}]", "yellow"))
    content = _last_content(result)
    emit(node, content)
    return _load_json(content, fallback), content, [human, result["messages"][-1]]


# ---------------------------------------------------------------------------
# 规划辅助
# ---------------------------------------------------------------------------

def _default_plan(state: ResearchState) -> dict:
    """生成默认规划 JSON，用于 LLM 未返回有效规划时的兜底。"""
    return {
        "objective": state["query"],
        "sub_questions": [state["query"]],
        "outline": [
            {
                "id": "sec_1",
                "title": "默认大纲",
                "description": "默认生成的大纲",
                "section_type": "mixed",
                "requires_data": False,
                "requires_chart": False,
                "priority": 1,
                "search_queries": [state["query"]],
                "status": "pending",
            }
        ],
        "research_questions": [state["query"]],
        "budget": {"max_rounds": 2, "max_sources": 12, "max_tokens": 12000, "max_seconds": 45},
    }


def _guess_primary_entity(query: str) -> str:
    """从查询中猜测主要实体名称（ASCII 术语优先，其次中文短语）。"""
    lowered = query.lower()
    ascii_terms = re.findall(r"[a-z][a-z0-9_-]{2,}", lowered)
    for term in ascii_terms:
        if term not in {"latest", "trend", "news", "agent", "open", "using"}:
            return term
    chinese_terms = re.findall(r"[一-鿿]{2,}", query)
    for term in chinese_terms:
        if term not in {"帮我", "调查", "最新", "使用趋势", "是什么", "多少", "情况"}:
            return term
    return ""


def _derive_direct_search_queries(query: str) -> list[str]:
    """根据用户原始问题生成直接检索词列表。"""
    base_query = query.strip()
    if not base_query:
        return []
    entity = _guess_primary_entity(base_query)
    candidates = [base_query]
    if entity:
        candidates.extend(
            [
                f"{entity}是什么",
                f"{entity} GitHub",
                f"{entity} 官方文档",
                f"{entity} 使用趋势",
                f"{entity} AI Agent",
            ]
        )
    else:
        candidates.extend(
            [
                f"{base_query} 是什么",
                f"{base_query} GitHub",
                f"{base_query} 官方文档",
            ]
        )
    deduped: list[str] = []
    for item in candidates:
        text = item.strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped[:6]


# ---------------------------------------------------------------------------
# 查询相关辅助
# ---------------------------------------------------------------------------

def _extract_query_terms(query: str) -> list[str]:
    """从查询文本中提取关键词（中文 2+ 字符、英文 3+ 字符），排除停用词。"""
    parts = re.findall(r"[一-鿿]{2,}|[A-Za-z0-9_-]{3,}", query.lower())
    terms = []
    stopwords = {"什么", "如何", "以及", "一个", "关于", "这个", "那个", "进行", "基于", "附带", "来源", "清单"}
    for part in parts:
        if part in stopwords:
            continue
        terms.append(part)
    return terms[:12]


def _estimate_relevance(query: str, text: str) -> float:
    """估算文本与查询的相关性得分（0~1），基于关键词命中比例。"""
    terms = _extract_query_terms(query)
    if not terms:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / max(len(terms), 1)


def _is_query_grounded(candidate: str, user_query: str) -> bool:
    """检查候选查询是否基于用户原始问题（至少一个关键词重叠）。"""
    candidate_terms = set(_extract_query_terms(candidate))
    user_terms = set(_extract_query_terms(user_query))
    if not candidate_terms or not user_terms:
        return False
    if _guess_primary_entity(user_query) and _guess_primary_entity(user_query) in candidate.lower():
        return True
    overlap = candidate_terms & user_terms
    return len(overlap) >= 1


def _derive_search_plan(outline: list[dict], sub_questions: list[str], _research_questions: list[str], query: str) -> list[dict]:
    """根据大纲和子问题推导搜索计划。"""
    plan: list[dict] = []
    for direct_query in _derive_direct_search_queries(query):
        plan.append(
            {
                "section_id": "user_query",
                "query": direct_query,
                "source_preference": "hybrid",
                "reason": "围绕用户原始问题生成的直接检索词",
            }
        )
    for section in outline:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "sec")
        for item in section.get("search_queries", []) or []:
            text = str(item).strip()
            if text and _is_query_grounded(text, query):
                plan.append(
                    {
                        "section_id": section_id,
                        "query": text,
                        "source_preference": "hybrid",
                        "reason": f"来自大纲章节 {section_id}",
                    }
                )
    if not plan:
        plan.append({"section_id": "sec_1", "query": query, "source_preference": "hybrid", "reason": "fallback"})
    deduped = _dedupe_sources(plan, ["query"])
    return deduped[:6]


def _build_queries(state: ResearchState, source_preference: str) -> list[dict]:
    """从 state 中构建指定来源类型的查询列表。"""
    queries: list[dict] = []

    # Check if we are in re-search iteration
    iteration = state.get("iteration", 0)
    if iteration > 0 and state.get("supplementary_queries"):
        base_plan = state.get("supplementary_queries", [])
    else:
        base_plan = state.get("search_plan", [])

    for item in base_plan:
        if not isinstance(item, dict):
            continue
        pref = item.get("source_preference", "hybrid")
        if pref in (source_preference, "hybrid"):
            query = str(item.get("query", "")).strip()
            if query:
                queries.append(item)
    if not queries:
        queries.append({"section_id": "sec_1", "query": state["query"], "source_preference": source_preference, "reason": "fallback"})
    return queries[:6]


# ---------------------------------------------------------------------------
# 记录过滤
# ---------------------------------------------------------------------------

def _is_bad_web_domain(domain: str) -> bool:
    """判断是否为低质量网页域名。"""
    value = domain.lower()
    blocked = ["datasheet", "bdtic", "doc88", "elecfans", "down"]
    return any(item in value for item in blocked)


def _is_official_domain(domain: str) -> bool:
    """判断是否为官方或权威机构域名。"""
    value = domain.lower()
    return value.endswith(".gov.cn") or value.endswith(".gov") or value.endswith(".edu") or value.endswith(".edu.cn") or "gov" in value or "official" in value


def _filter_web_records(query: str, records: list[dict]) -> tuple[list[dict], dict]:
    """过滤网页记录，移除低相关性、黑名单域名和空记录。"""
    kept = []
    stats = {"raw_count": len(records), "kept_count": 0, "dropped_irrelevant": 0, "dropped_domain": 0, "dropped_empty": 0}
    for record in records:
        title = str(record.get("title", ""))
        snippet = str(record.get("snippet", ""))
        domain = str(record.get("domain", ""))
        if not title and not snippet:
            stats["dropped_empty"] += 1
            continue
        if _is_bad_web_domain(domain):
            stats["dropped_domain"] += 1
            continue
        relevance = _estimate_relevance(query, f"{title}\n{snippet}")
        record["relevance_score"] = relevance
        if relevance < 0.2 and not _is_official_domain(domain):
            stats["dropped_irrelevant"] += 1
            continue
        kept.append(record)
    stats["kept_count"] = len(kept)
    return kept, stats


def _filter_local_records(query: str, records: list[dict]) -> tuple[list[dict], dict]:
    """过滤本地知识库记录，移除低相关性和空记录。"""
    kept = []
    stats = {"raw_count": len(records), "kept_count": 0, "dropped_irrelevant": 0, "dropped_missing_doc": 0, "dropped_empty": 0}
    for record in records:
        title = str(record.get("title", ""))
        snippet = str(record.get("snippet", ""))
        doc_id = str(record.get("doc_id", "")).strip()
        if not snippet:
            stats["dropped_empty"] += 1
            continue
        relevance = _estimate_relevance(query, f"{title}\n{snippet}")
        record["relevance_score"] = relevance
        if not doc_id and relevance < 0.35:
            stats["dropped_missing_doc"] += 1
            continue
        if relevance < 0.2:
            stats["dropped_irrelevant"] += 1
            continue
        kept.append(record)
    stats["kept_count"] = len(kept)
    return kept, stats


# ---------------------------------------------------------------------------
# 记录格式化与处理
# ---------------------------------------------------------------------------

def _format_raw_records(records: list[dict], source_type: str) -> str:
    """将原始记录格式化为 JSON 行，用于 LLM 上下文注入。"""
    if not records:
        return "[]"
    lines = []
    for record in records[:40]:
        locator = record.get("url") or record.get("doc_id") or ""
        lines.append(
            json.dumps(
                {
                    "source_id": record.get("source_id"),
                    "title": record.get("title"),
                    "url": record.get("url", ""),
                    "doc_id": record.get("doc_id", ""),
                    "snippet": str(record.get("snippet", ""))[:500],
                    "source_type": source_type,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _minimal_record_filter(records: list[dict], required_any: list[str]) -> list[dict]:
    """最低限度过滤：至少有一个 required_any 字段非空。"""
    kept: list[dict] = []
    for record in records:
        if any(str(record.get(field, "")).strip() for field in required_any):
            kept.append(record)
    return kept


def _assign_source_ids(records: list[dict], prefix: str) -> list[dict]:
    """为记录列表分配 source_id（格式：prefix-序号）。"""
    assigned: list[dict] = []
    for index, record in enumerate(records, 1):
        item = dict(record)
        item["source_id"] = f"{prefix}-{index}"
        assigned.append(item)
    return assigned


def _enrich_evidence_from_raw(evidence: list[dict], raw_records: list[dict]) -> list[dict]:
    """从原始记录中补充 evidence 中可能丢失的 url、domain、title 等字段。"""
    raw_lookup = {str(r.get("source_id", "")).strip(): r for r in raw_records if r.get("source_id")}
    enriched = []
    for ev in evidence:
        item = dict(ev)
        sid = str(item.get("source_id", "")).strip()
        raw = raw_lookup.get(sid, {})
        # 补充 url（如 LLM 没有保留）
        if not item.get("url") and raw.get("url"):
            item["url"] = raw["url"]
        # 补充 domain
        if not item.get("domain") and raw.get("domain"):
            item["domain"] = raw["domain"]
        # 补充 title（如 LLM 没有保留）
        if not item.get("title") and raw.get("title"):
            item["title"] = raw["title"]
        enriched.append(item)
    return enriched


def _prune_evidence_to_allowed_sources(evidence: list[dict], allowed_source_ids: set[str]) -> list[dict]:
    """只保留允许来源 ID 列表中的证据。"""
    kept: list[dict] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        if source_id and source_id in allowed_source_ids:
            kept.append(item)
    return kept


def _summarize_records(records: list[dict]) -> list[dict]:
    """生成记录摘要（最多 5 条），用于检索 trace 记录。"""
    summary: list[dict] = []
    for record in records[:5]:
        summary.append(
            {
                "source_id": record.get("source_id"),
                "title": record.get("title", ""),
                "locator": record.get("url") or record.get("doc_id") or "",
                "snippet": str(record.get("snippet", ""))[:160],
            }
        )
    return summary


def _normalize_source_ids(values) -> list[str]:
    """归一化来源 ID 列表：去重、去空、保持顺序。"""
    normalized: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _finalize_query_traces(query_traces: list[dict], kept_ids: set[str], rejected_ids: list[str], reject_reason: str) -> list[dict]:
    """为每个查询 trace 记录标记 kept/rejected 的 source_id 及样本。"""
    normalized_rejected = set(_normalize_source_ids(rejected_ids))
    finalized: list[dict] = []
    for trace in query_traces:
        raw_items = [item for item in trace.get("raw_records", []) if isinstance(item, dict)]
        kept_records = [item for item in raw_items if str(item.get("source_id", "")).strip() in kept_ids]
        rejected_records = [
            item
            for item in raw_items
            if str(item.get("source_id", "")).strip() in normalized_rejected or str(item.get("source_id", "")).strip() not in kept_ids
        ]
        trace_item = dict(trace)
        trace_item["raw_source_ids"] = _normalize_source_ids(item.get("source_id") for item in raw_items)
        trace_item["kept_source_ids"] = _normalize_source_ids(item.get("source_id") for item in kept_records)
        trace_item["rejected_source_ids"] = _normalize_source_ids(item.get("source_id") for item in rejected_records)
        trace_item["kept_count"] = len(trace_item["kept_source_ids"])
        trace_item["rejected_count"] = len(trace_item["rejected_source_ids"])
        trace_item["kept_records"] = kept_records[:3]
        trace_item["rejected_records"] = rejected_records[:3]
        if reject_reason:
            trace_item["reject_reason"] = reject_reason
        finalized.append(trace_item)
    return finalized


# ---------------------------------------------------------------------------
# 证据 fallback 构建
# ---------------------------------------------------------------------------

def _fallback_web_evidence(records: list[dict]) -> dict:
    """网页证据兜底：直接从原始记录构建结构化证据列表。"""
    evidence = []
    for record in records:
        evidence.append(
            {
                "source_id": record.get("source_id"),
                "title": record.get("title"),
                "url": record.get("url", ""),
                "snippet": record.get("snippet", ""),
                "domain": record.get("domain", ""),
                "source_type": "web",
                "reliability_hint": "official" if _is_official_domain(record.get("domain", "")) else "unknown",
                "supports": [],
                "notes": "",
            }
        )
    return {"summary": "完成网页证据采集。", "evidence": evidence, "gaps": []}


def _fallback_local_evidence(records: list[dict]) -> dict:
    """本地知识库证据兜底：直接从原始记录构建结构化证据列表。"""
    evidence = []
    for record in records:
        evidence.append(
            {
                "source_id": record.get("source_id"),
                "doc_id": record.get("doc_id", ""),
                "title": record.get("title", "") or record.get("source_id", ""),
                "snippet": record.get("snippet", ""),
                "source_type": "local",
                "reliability_hint": "internal",
                "supports": [],
                "notes": "",
            }
        )
    return {"summary": "完成本地知识库证据采集。", "evidence": evidence, "gaps": []}


# ---------------------------------------------------------------------------
# 证据评分与去重
# ---------------------------------------------------------------------------

def _score_evidence(record: dict) -> tuple[float, str]:
    """根据来源类型和域名对证据评分，返回 (score, reason)。"""
    source_type = record.get("source_type")
    if source_type == "local":
        return 0.92, "企业内部知识库证据，默认高可信"
    domain = str(record.get("domain", "")).lower()
    if _is_official_domain(domain):
        return 0.88, "官方或权威机构域名"
    if any(word in domain for word in ["news", "finance", "reuters", "bloomberg", "people", "xinhuanet"]):
        return 0.72, "主流媒体域名"
    if domain:
        return 0.58, "普通互联网来源，需要交叉验证"
    return 0.45, "来源信息不完整"


def _dedupe_sources(items: list[dict], key_fields: list[str]) -> list[dict]:
    """根据指定字段去重，保持顺序。"""
    seen = set()
    results = []
    for item in items:
        key = tuple(str(item.get(field, "")).strip() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# 审计 / 分析 fallback
# ---------------------------------------------------------------------------

def _fallback_audit(state: ResearchState) -> dict:
    """证据审计兜底：对所有证据评分、归一化，检查假设覆盖。"""
    evidence_pool = []
    source_index = []
    audit_flags = []
    for record in state.get("web_evidence", []) + state.get("local_evidence", []):
        score, reason = _score_evidence(record)
        normalized = dict(record)
        normalized["reliability_score"] = score
        normalized["reliability_reason"] = reason
        normalized["source_label"] = record.get("title") or record.get("doc_id") or record.get("url") or record.get("source_id")
        normalized.setdefault("supports", [])
        normalized.setdefault("refutes", [])
        evidence_pool.append(normalized)
        locator = record.get("url") or record.get("doc_id") or ""
        if score < 0.6:
            audit_flags.append({"type": "low_confidence", "target": record.get("source_id"), "reason": reason})
        else:
            source_index.append(
                {
                    "source_id": record.get("source_id"),
                    "label": normalized["source_label"],
                    "locator": locator or "未提供定位信息",
                    "source_type": record.get("source_type", "source"),
                }
            )
    for hypo in state.get("hypotheses", []):
        hypo_id = hypo.get("id")
        related = [item for item in evidence_pool if hypo_id in item.get("supports", []) or hypo_id in item.get("refutes", [])]
        if not related:
            audit_flags.append({"type": "missing_evidence", "target": hypo_id, "reason": "缺少直接关联证据"})
    return {
        "summary": "完成证据评分与审计。",
        "evidence_pool": evidence_pool,
        "audit_flags": audit_flags,
        "source_index": _dedupe_sources(source_index, ["source_id"]),
    }


def _fallback_analysis(state: ResearchState) -> dict:
    """分析节点兜底：生成默认分析结论。"""
    source_ids = [item.get("source_id") for item in state.get("evidence_pool", [])[:3] if item.get("source_id")]
    findings = [
        {
            "claim_id": "c_1",
            "claim": f"围绕“{state['query']}”已完成多源检索，初步证据表明问题可以从网络与本地知识库双侧支撑。",
            "confidence": "medium" if source_ids else "low",
            "source_ids": source_ids,
        }
    ]
    hypothesis_status = []
    for hypo in state.get("hypotheses", []):
        hypothesis_status.append(
            {
                "id": hypo.get("id"),
                "status": "verified" if source_ids else "uncertain",
                "reason": "已有可用证据池" if source_ids else "证据不足",
                "source_ids": source_ids,
            }
        )
    return {
        "analysis_summary": "完成结论归纳与假设状态整理。",
        "hypothesis_status": hypothesis_status,
        "findings": findings,
        "claim_map": [{"claim_id": item["claim_id"], "source_ids": item["source_ids"]} for item in findings],
        "next_actions": [] if source_ids else ["补充更多高质量来源"],
    }


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------

def _render_fallback_report(state: ResearchState) -> str:
    """当 LLM 写作失败时生成兜底 Markdown 报告。"""
    lines = ["# 调研结果", "", "## 执行摘要", state.get("analysis", "暂无分析结果"), ""]
    lines.append("## 任务规划与假设状态")
    for hypo in state.get("hypotheses", []):
        status = hypo.get("status", "unverified")
        lines.append(f"- {hypo.get('id', 'h')}: {hypo.get('content', '')} | 状态: {status}")
    lines.append("")
    lines.append("## 核心结论")
    for finding in state.get("findings", []):
        refs = "".join(f"[{source_id}]" for source_id in finding.get("source_ids", []))
        lines.append(f"- {finding.get('claim', '')} {refs}".rstrip())
    lines.append("")
    lines.append("## 风险与不确定性")
    if state.get("audit_flags"):
        for flag in state["audit_flags"]:
            lines.append(f"- {flag.get('type')}: {flag.get('reason')} ({flag.get('target')})")
    else:
        lines.append("- 当前未发现明显冲突。")
    lines.append("")
    lines.append("## 检索统计")
    web_stats = state.get("web_retrieval_stats", {})
    local_stats = state.get("local_retrieval_stats", {})
    if web_stats or local_stats:
        lines.append(f"- 网络检索：queries={web_stats.get('query_count', 0)} raw={web_stats.get('raw_count', 0)} kept={web_stats.get('kept_count', 0)} dropped={web_stats.get('dropped_count', 0)}")
        lines.append(f"- 本地检索：queries={local_stats.get('query_count', 0)} raw={local_stats.get('raw_count', 0)} kept={local_stats.get('kept_count', 0)} dropped={local_stats.get('dropped_count', 0)}")
    else:
        lines.append("- 未记录检索统计。")
    lines.append("")
    lines.append("## 引用列表")
    for source in state.get("source_index", []):
        source_type = source.get("source_type", "source")
        lines.append(f"- {source.get('source_id')} [{source_type}]: {source.get('label')} | {source.get('locator')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 来源索引构建
# ---------------------------------------------------------------------------

def _build_source_lookup(state: ResearchState) -> dict[str, dict]:
    """从 state 的多个来源字段构建统一的 source_id -> 来源信息 lookup。"""
    lookup: dict[str, dict] = {}

    def _put(source_id: str, source_type: str, label: str, locator: str):
        if not source_id:
            return
        item = lookup.get(source_id)
        if not item:
            lookup[source_id] = {
                "source_id": source_id,
                "source_type": source_type or "source",
                "label": label or source_id,
                "locator": locator or "",
            }
            return
        if (not item.get("locator")) and locator:
            item["locator"] = locator
        if (not item.get("label")) and label:
            item["label"] = label
        if item.get("source_type") in {"source", ""} and source_type:
            item["source_type"] = source_type

    for source in state.get("source_index", []):
        _put(
            str(source.get("source_id", "")).strip(),
            str(source.get("source_type", "source")).strip(),
            str(source.get("label", "")).strip(),
            str(source.get("locator", "")).strip(),
        )
    for ev in state.get("evidence_pool", []):
        _put(
            str(ev.get("source_id", "")).strip(),
            str(ev.get("source_type", "source")).strip(),
            str(ev.get("title") or ev.get("source_label") or "").strip(),
            str(ev.get("url") or ev.get("doc_id") or "").strip(),
        )
    for ev in state.get("web_evidence", []):
        _put(
            str(ev.get("source_id", "")).strip(),
            "web",
            str(ev.get("title", "")).strip(),
            str(ev.get("url") or "").strip(),
        )
    for ev in state.get("local_evidence", []):
        _put(
            str(ev.get("source_id", "")).strip(),
            "local",
            str(ev.get("title") or ev.get("doc_id") or "").strip(),
            str(ev.get("doc_id") or "").strip(),
        )
    for key, item in lookup.items():
        if key.startswith("LOC"):
            item["source_type"] = "local"
        elif key.startswith("WEB"):
            item["source_type"] = "web"
    return lookup


# ---------------------------------------------------------------------------
# 引用校验与渲染
# ---------------------------------------------------------------------------

def _extract_citation_ids(content: str) -> list[str]:
    """从正文中提取所有引用ID [XXX]。"""
    pattern = r'\[([A-Z]+\d+_\d+-\d+)\]'
    matches = re.findall(pattern, content)
    return list(dict.fromkeys(matches))  # 去重保序


def _validate_and_fix_citations(content: str, valid_source_ids: set[str]) -> tuple[str, list[str]]:
    """校验正文中的引用ID，移除非法引用，返回修正后的内容和实际使用的合法引用列表。"""
    pattern = r'\[([A-Z]+\d+_\d+-\d+)\]'

    def replace_citation(match):
        citation_id = match.group(1)
        if citation_id in valid_source_ids:
            return f"[{citation_id}]"
        else:
            # 非法引用，直接移除
            return ""

    fixed_content = re.sub(pattern, replace_citation, content)
    # 提取修正后实际使用的合法引用
    used_ids = [cid for cid in _extract_citation_ids(fixed_content) if cid in valid_source_ids]
    return fixed_content, used_ids


def _render_reference_list(state: ResearchState) -> str:
    """渲染参考文献列表 Markdown，优先按正文引用顺序，去重本地来源。"""
    lines = ["## 参考资料"]
    lookup = _build_source_lookup(state)

    # 1. 优先从正文 draft 中按出现顺序提取实际引用的 source_id
    draft_content = state.get("draft", "") or state.get("final", "")
    cited_ids: list[str] = []
    if draft_content:
        for sid in _extract_citation_ids(draft_content):
            if sid in lookup and sid not in cited_ids:
                cited_ids.append(sid)

    # 2. 如果正文无引用，降级到 findings
    if not cited_ids:
        for finding in state.get("findings", []):
            for sid in finding.get("source_ids", []):
                text = str(sid).strip()
                if text and text not in cited_ids and text in lookup:
                    cited_ids.append(text)

    # 3. 再降级：全量 lookup
    if not cited_ids:
        cited_ids = list(lookup.keys())

    # 4. 对 local 来源按 locator 去重展示（同一文件多个 chunk 只展示一次）
    seen_locators: set[str] = set()
    display_ids: list[str] = []
    web_ids: list[str] = []
    local_ids: list[str] = []

    for sid in cited_ids:
        source = lookup.get(sid)
        if not source:
            continue
        source_type = source.get("source_type", "")
        locator = source.get("locator", "").strip()

        if source_type == "local":
            # 同一文件路径只保留第一次出现的 source_id 做代表
            dedup_key = locator or sid
            if dedup_key in seen_locators:
                continue
            seen_locators.add(dedup_key)
            local_ids.append(sid)
        else:
            web_ids.append(sid)

    # 5. 排列顺序：WEB 在前（保持原始引用顺序），LOCAL 跟后
    display_ids = web_ids + local_ids

    if not display_ids:
        display_ids = cited_ids[:15]

    for sid in display_ids:
        source = lookup.get(sid)
        if not source:
            continue
        locator = source.get("locator", "").strip()
        label = source.get("label", "").strip()
        source_type = source.get("source_type", "source")
        source_id = source.get("source_id", sid)

        if not locator:
            locator = "链接暂不可用" if source_type == "web" else "本地知识库"

        lines.append(f"- [{source_id}] [{source_type}]: {label} | {locator}")

    if len(lines) == 1:
        lines.append("- 暂无参考资料")
    return "\n".join(lines)


def _render_execution_appendix(state: ResearchState) -> str:
    """渲染规划与检索明细附录 Markdown。"""
    lines = ["## 规划与检索明细", "", "### 执行概览"]
    search_plan = state.get("search_plan", [])
    web_stats = state.get("web_retrieval_stats", {})
    local_stats = state.get("local_retrieval_stats", {})
    lines.append(f"- 规划生成研究问题数: {len(state.get('research_questions', []))}")
    lines.append(f"- 规划生成搜索步骤数: {len(search_plan)}")

    iteration = state.get("iteration", 0)
    lines.append(f"- 经过 {iteration + 1} 轮检索迭代")
    if state.get("needs_more_research"):
        lines.append(f"- 信息缺口: {state.get('missing_gaps', [])}")

    lines.append(
        f"- 实际执行网页检索问题数: {web_stats.get('query_count', 0)} | 原始命中: {web_stats.get('raw_count', 0)} | 保留证据: {web_stats.get('kept_count', 0)} | 丢弃: {web_stats.get('dropped_count', 0)}"
    )
    lines.append(
        f"- 实际执行本地检索问题数: {local_stats.get('query_count', 0)} | 原始命中: {local_stats.get('raw_count', 0)} | 保留证据: {local_stats.get('kept_count', 0)} | 丢弃: {local_stats.get('dropped_count', 0)}"
    )
    lines.append("")
    lines.append("### 问题拆解明细")
    for sq in state.get("sub_questions", []):
        lines.append(f"- {sq}")
    if not state.get("sub_questions"):
        lines.append("- 无")
    lines.append("")
    lines.append("### 规划输出")
    outline = state.get("outline", [])
    if outline:
        for section in outline:
            lines.append(
                f"- {section.get('id')}: {section.get('title')} | {section.get('description')} | search_queries={section.get('search_queries', [])}"
            )
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("### 研究问题")
    for index, question in enumerate(state.get("research_questions", []), 1):
        lines.append(f"- Q{index}: {question}")
    if not state.get("research_questions"):
        lines.append("- 无")
    lines.append("")
    lines.append("### 搜索计划")
    for index, item in enumerate(state.get("search_plan", []), 1):
        lines.append(
            f"- S{index}: section={item.get('section_id')} | query={item.get('query')} | source={item.get('source_preference')} | reason={item.get('reason')}"
        )
    if not state.get("search_plan"):
        lines.append("- 无")
    lines.append("")
    if state.get("supplementary_queries"):
        lines.append("### 补搜计划")
        for index, item in enumerate(state.get("supplementary_queries", []), 1):
            lines.append(f"- S{index} (补搜): query={item.get('query')} | reason={item.get('reason')}")
        lines.append("")
    lines.append("### 网页检索明细")
    for index, trace in enumerate(state.get("web_search_trace", []), 1):
        lines.append(
            f"- WQ{index}: section={trace.get('section_id')} | query={trace.get('query')} | reason={trace.get('reason')} | raw={trace.get('raw_count', 0)} | kept={trace.get('kept_count', 0)} | rejected={trace.get('rejected_count', 0)}"
        )
        lines.append(f"  - raw_ids={trace.get('raw_source_ids', [])}")
        lines.append(f"  - kept_ids={trace.get('kept_source_ids', [])}")
        lines.append(f"  - rejected_ids={trace.get('rejected_source_ids', [])}")
        if trace.get("reject_reason"):
            lines.append(f"  - reject_reason={trace.get('reject_reason')}")
        lines.append("  - raw_samples:")
        for item in trace.get("raw_records", [])[:3]:
            lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("kept_records"):
            lines.append("  - kept_samples:")
            for item in trace.get("kept_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("rejected_records"):
            lines.append("  - rejected_samples:")
            for item in trace.get("rejected_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
    if not state.get("web_search_trace"):
        lines.append("- 无")
    lines.append("")
    lines.append("### 本地检索明细")
    for index, trace in enumerate(state.get("local_rag_trace", []), 1):
        lines.append(
            f"- LQ{index}: section={trace.get('section_id')} | query={trace.get('query')} | reason={trace.get('reason')} | raw={trace.get('raw_count', 0)} | kept={trace.get('kept_count', 0)} | rejected={trace.get('rejected_count', 0)}"
        )
        lines.append(f"  - raw_ids={trace.get('raw_source_ids', [])}")
        lines.append(f"  - kept_ids={trace.get('kept_source_ids', [])}")
        lines.append(f"  - rejected_ids={trace.get('rejected_source_ids', [])}")
        if trace.get("reject_reason"):
            lines.append(f"  - reject_reason={trace.get('reject_reason')}")
        lines.append("  - raw_samples:")
        for item in trace.get("raw_records", [])[:3]:
            lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("kept_records"):
            lines.append("  - kept_samples:")
            for item in trace.get("kept_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("rejected_records"):
            lines.append("  - rejected_samples:")
            for item in trace.get("rejected_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
    if not state.get("local_rag_trace"):
        lines.append("- 无")
    return "\n".join(lines)


def _ensure_reference_section(content: str, state: ResearchState) -> str:
    """确保报告正文末尾包含参考资料章节。"""
    base = content.rstrip()
    references = _render_reference_list(state)
    if "## 引用列表" in base or "## 来源清单" in base or "## 参考资料" in base:
        return base
    return f"{base}\n\n{references}"
