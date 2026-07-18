"""Agent 构建模块：创建 AgentBundle、初始化记忆与 checkpointer。"""

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from ..config import AppConfig
from ..prompts import PROMPTS
from ..rag.core import RAGConfig
from ..tools import (
    extract_requirements,
    outline_from_topics,
    dedupe_lines,
    web_search_stub,
    local_docs_lookup_stub,
    search_knowledge_base,
    init_rag_system,
    merge_notes,
    summarize_points,
    simple_calculator,
)

logger = logging.getLogger("deep_research")


@dataclass(frozen=True)
class AgentBundle:
    """所有 Agent 实例的容器。"""

    intent_router: any
    planner: any
    scout_web: any
    scout_local: any
    evidence_judge: any
    analyst: any
    direct_responder: any
    writer: any


def build_agent(model: str, api_key: str, prompt_key: str, temperature: float, tools: list):
    """构建单个 Agent 实例。"""
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    llm = ChatTongyi(model=model, temperature=temperature)
    prompt = PROMPTS[prompt_key]
    return create_agent(model=llm, tools=tools, system_prompt=prompt)


def build_agents(model: str, api_key: str, config: AppConfig) -> AgentBundle:
    """构建所有 Agent 实例。"""
    rag_config = RAGConfig(
        milvus_host=config.milvus_host,
        milvus_port=config.milvus_port,
        collection_name=config.milvus_collection,
    )
    init_rag_system(api_key=api_key, config=rag_config)
    # 每个 Agent 不绑定 tools，只做信息抽取，降低 System Prompt 长度
    return AgentBundle(
        intent_router=build_agent(model, api_key, "intent_router", 0.0, []),
        planner=build_agent(model, api_key, "plan", 0.3, []),
        scout_web=build_agent(model, api_key, "web_search", 0.4, []),
        scout_local=build_agent(model, api_key, "local_rag", 0.4, []),
        evidence_judge=build_agent(model, api_key, "deep_dive", 0.2, []),
        analyst=build_agent(model, api_key, "analyze", 0.3, []),
        direct_responder=build_agent(model, api_key, "direct_answer", 0.2, []),
        writer=build_agent(model, api_key, "write", 0.4, []),
    )


def build_memory_manager(config: AppConfig):
    """初始化记忆管理器，失败时返回 None。"""
    from ..memory import MemoryManager

    if not config.enable_memory:
        return None
    try:
        return MemoryManager(
            short_term_ttl=config.short_term_ttl_seconds,
            short_term_max_messages=config.short_term_max_messages,
            short_term_summary_threshold=config.short_term_summary_threshold,
            tenant_id=config.tenant_id,
            short_term_backend=config.short_term_backend,
            long_term_backend=config.long_term_backend,
            long_term_scope=config.long_term_scope,
            save_conversation_task=config.save_conversation_task,
            enable_milvus=config.enable_milvus,
            redis_url=config.redis_url,
            postgres_dsn=config.postgres_dsn,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            milvus_collection=config.milvus_collection,
            embedding_api_key=config.api_key,
        )
    except Exception as exc:
        logger.exception("初始化 MemoryManager 失败，已禁用外部记忆: %s", exc)
        return None


_CHECKPOINTER_CONTEXT = None


def build_checkpointer(config: AppConfig):
    """构建 LangGraph checkpointer（PostgreSQL > Redis > 内存降级）。"""
    global _CHECKPOINTER_CONTEXT
    backend = config.checkpointer_backend

    if backend in {"postgres", "auto"} and config.enable_memory and config.postgres_dsn:
        postgres_saver = None
        postgres_import_error = ""
        try:
            module = importlib.import_module("langgraph.checkpoint.postgres")
            postgres_saver = getattr(module, "PostgresSaver", None)
        except Exception as exc:
            postgres_import_error = str(exc)
        if postgres_saver is None:
            try:
                module = importlib.import_module("langgraph_checkpoint_postgres")
                postgres_saver = getattr(module, "PostgresSaver", None)
            except Exception as exc:
                postgres_import_error = postgres_import_error or str(exc)
        if postgres_saver is None:
            message = (
                "PostgreSQL checkpointer 模块不可用。请安装: pip install langgraph-checkpoint-postgres "
                f"| import_error={postgres_import_error or 'unknown'}"
            )
            if backend == "postgres":
                logger.warning(message)
            else:
                logger.info(message)
        else:
            try:
                _CHECKPOINTER_CONTEXT = postgres_saver.from_conn_string(config.postgres_dsn)
                checkpointer = _CHECKPOINTER_CONTEXT.__enter__()
                checkpointer.setup()
                logger.info("使用 PostgreSQL checkpointer")
                return checkpointer
            except Exception as exc:
                logger.warning("PostgreSQL checkpointer 初始化失败: %s", exc)

    if backend in {"redis", "auto"} and config.enable_memory and config.redis_url:
        from langgraph.checkpoint.redis import RedisSaver

        candidate_urls = [config.redis_url]
        if "redis://root:" in config.redis_url:
            candidate_urls.append(config.redis_url.replace("redis://root:", "redis://:"))
        last_exc = None
        for url in candidate_urls:
            try:
                _CHECKPOINTER_CONTEXT = RedisSaver.from_conn_string(url)
                checkpointer = _CHECKPOINTER_CONTEXT.__enter__()
                checkpointer.setup()
                logger.info("使用 Redis checkpointer")
                return checkpointer
            except Exception as exc:
                last_exc = exc
        if last_exc and "FT._LIST" in str(last_exc):
            logger.warning(
                "Redis checkpointer 依赖 RediSearch(FT._LIST)。当前 Redis 非 Redis Stack，已降级。"
            )
        else:
            logger.warning("Redis checkpointer 初始化失败，降级内存: %s", last_exc)

    logger.info("使用内存 checkpointer")
    return InMemorySaver()


def cleanup_checkpointer():
    """清理全局 checkpointer 上下文。"""
    global _CHECKPOINTER_CONTEXT
    if _CHECKPOINTER_CONTEXT:
        _CHECKPOINTER_CONTEXT.__exit__(None, None, None)
        _CHECKPOINTER_CONTEXT = None
