"""知识库工具：封装本地 RAG 知识库查询接口。"""

import logging
from typing import Optional

from langchain_core.tools import tool

from deep_research.rag.core import RAGSystem, RAGConfig

logger = logging.getLogger("mult_agents")

# 全局 RAG 系统实例
_RAG_SYSTEM: Optional[RAGSystem] = None


def init_rag_system(api_key: str, config: Optional[RAGConfig] = None):
    """初始化全局 RAG 系统。

    Args:
        api_key: RAG 系统所需的 API Key。
        config: 可选的 RAGConfig 配置对象。
    """
    global _RAG_SYSTEM
    if _RAG_SYSTEM is None:
        try:
            _RAG_SYSTEM = RAGSystem(api_key, config)
        except Exception as e:
            print(f"RAG 系统初始化失败: {e}")


def search_knowledge_base_records(query: str, limit: int = 5) -> list[dict]:
    """查询本地知识库并返回标准化记录列表。

    Args:
        query: 查询关键词。
        limit: 返回结果数量上限，默认 5。

    Returns:
        知识库命中记录的字典列表；RAG 系统未初始化时返回空列表。
    """
    if _RAG_SYSTEM is None:
        return []
    try:
        return _RAG_SYSTEM.search_records(query, k=limit)
    except Exception:
        return []


@tool
def search_knowledge_base(query: str) -> str:
    """
    查询本地知识库/向量数据库。
    当用户询问关于专业知识、历史文档或私有数据时使用此工具。
    输入应该是具体的查询问题。
    """
    if _RAG_SYSTEM is None:
        return "错误：RAG 系统未初始化或连接失败。请检查 Milvus 服务状态。"
    return _RAG_SYSTEM.search(query)
