"""Web 检索工具：封装 Bocha 网络搜索及相关检索接口。"""

import json
import logging
import os
import urllib.error
import urllib.request

from langchain_core.tools import tool

logger = logging.getLogger("deep_research")


def bocha_web_search_records(query: str, count: int = 8) -> list[dict]:
    """使用 Bocha Web Search API 执行网络搜索，返回标准化记录列表。

    Args:
        query: 搜索关键词。
        count: 返回结果数量上限，默认 8。

    Returns:
        包含 source_id、title、url、snippet、domain、source_type、published_at 字段的字典列表。
    """
    api_key = os.getenv("BOCHA_API_KEY", "").strip()
    logger.info("[bocha_web_search] 开始搜索 | query=%s | count=%s", query, count)
    logger.info("[bocha_web_search] API Key 状态 | 是否配置=%s | Key前缀=%s", bool(api_key), api_key[:8] + "..." if api_key else "None")
    if not api_key:
        logger.warning("[bocha_web_search] 未配置 BOCHA_API_KEY，跳过搜索")
        return []
    payload = {
        "query": query,
        "summary": True,
        "freshness": "noLimit",
        "count": count,
    }
    request = urllib.request.Request(
        url="https://api.bocha.cn/v1/web-search",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        logger.info("[bocha_web_search] 发送请求 | url=%s", request.full_url)
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            logger.info("[bocha_web_search] 收到响应 | status=%s | content_length=%s", response.status, len(raw))
        result = json.loads(raw)
        logger.info("[bocha_web_search] 解析响应成功 | data字段存在=%s", "data" in result)
    except urllib.error.HTTPError as e:
        logger.error("[bocha_web_search] HTTP 错误 | code=%s | reason=%s", e.code, e.reason)
        return []
    except urllib.error.URLError as e:
        logger.error("[bocha_web_search] URL 错误 | reason=%s", e.reason)
        return []
    except json.JSONDecodeError as e:
        logger.error("[bocha_web_search] JSON 解析错误 | error=%s", e)
        return []
    except Exception as e:
        logger.error("[bocha_web_search] 未知错误 | error=%s | type=%s", e, type(e).__name__)
        return []
    data = result.get("data", {})
    pages = data.get("webPages", [])
    logger.info("[bocha_web_search] 解析数据 | webPages类型=%s", type(pages).__name__)
    if isinstance(pages, dict):
        if isinstance(pages.get("value"), list):
            pages = pages.get("value", [])
        elif isinstance(pages.get("items"), list):
            pages = pages.get("items", [])
        else:
            pages = []
    if not isinstance(pages, list):
        logger.warning("[bocha_web_search] webPages 格式异常 | type=%s", type(pages).__name__)
        return []
    logger.info("[bocha_web_search] 获取网页数量 | total=%s", len(pages))
    records: list[dict] = []
    for idx, page in enumerate(pages[:count], 1):
        if not isinstance(page, dict):
            logger.warning("[bocha_web_search] 第 %s 条记录格式异常 | type=%s", idx, type(page).__name__)
            continue
        url = str(page.get("url") or "").strip()
        domain = ""
        if "://" in url:
            domain = url.split("://", 1)[1].split("/", 1)[0]
        title = page.get("name") or f"web_result_{idx}"
        snippet = page.get("summary") or ""
        logger.info("[bocha_web_search] 解析记录 %s | title=%s | url=%s | snippet长度=%s", idx, title[:50], domain, len(snippet))
        records.append(
            {
                "source_id": f"WEB-{idx}",
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": domain,
                "source_type": "web",
                "published_at": page.get("datePublished") or page.get("dateLastCrawled") or "",
            }
        )
    logger.info("[bocha_web_search] 搜索完成 | 返回记录数=%s", len(records))
    return records


@tool
def web_search_stub(query: str) -> str:
    """网络检索接口（Bocha Web Search）。"""
    records = bocha_web_search_records(query, count=5)
    if not records:
        return "未配置 BOCHA_API_KEY，无法执行网络检索。"
    lines = ["Bocha 检索结果："]
    for idx, record in enumerate(records, 1):
        lines.append(f"{idx}. {record['title']}")
        url = record.get("url", "")
        if url:
            lines.append(f"   链接: {url}")
        snippet = record.get("snippet", "")
        if snippet:
            lines.append(f"   摘要: {snippet[:200]}")
    return "\n".join(lines)


@tool
def news_search_stub(query: str) -> str:
    """模拟新闻检索接口。"""
    return f"未配置新闻检索服务，收到查询: {query}"


@tool
def finance_search_stub(query: str) -> str:
    """模拟金融检索接口。"""
    return f"未配置金融检索服务，收到查询: {query}"


@tool
def extract_url_content_stub(url: str) -> str:
    """模拟 URL 内容抽取接口。"""
    return f"未配置URL解析服务，收到URL: {url}"
