"""CLI 入口：交互式 REPL 与单次查询模式。"""

import argparse
import json
import logging
import sys

from langchain_core.messages import HumanMessage

from .config import AppConfig
from .agents.builder import (
    AgentBundle,
    build_agents,
    build_checkpointer,
    build_memory_manager,
    cleanup_checkpointer,
)
from .agents.graph import build_app as build_workflow_app
from .agents.state import create_initial_state
from .memory import MemoryManager
from .utils import colorize, emit, collect_tool_calls, with_memory_context, log_inputs


logger = logging.getLogger("deep_research")

MEMORY_MANAGER: MemoryManager | None = None


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Research multi-agent runner")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tenant-id", type=str, default=None)
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--thread-id", type=str, default=None)
    parser.add_argument("--short-term-backend", choices=["postgres", "redis", "memory"], default=None)
    parser.add_argument("--long-term-backend", choices=["postgres", "sqlite", "disabled"], default=None)
    parser.add_argument("--long-term-scope", choices=["user", "thread"], default=None)
    parser.add_argument("--save-conversation-task", choices=["true", "false"], default=None)
    parser.add_argument("--checkpointer-backend", choices=["postgres", "redis", "memory", "auto"], default=None)
    parser.add_argument("--enable-memory", choices=["true", "false"], default=None)
    parser.add_argument("--enable-milvus", choices=["true", "false"], default=None)
    parser.add_argument("--memory-top-k", type=int, default=None)
    parser.add_argument("--once-query", type=str, default=None)
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.from_file(args.config) if args.config else AppConfig.from_file()
    overrides = {
        "tenant_id": args.tenant_id,
        "user_id": args.user_id,
        "thread_id": args.thread_id,
        "short_term_backend": args.short_term_backend,
        "long_term_backend": args.long_term_backend,
        "long_term_scope": args.long_term_scope,
        "checkpointer_backend": args.checkpointer_backend,
        "memory_top_k": args.memory_top_k,
    }
    if args.enable_memory is not None:
        overrides["enable_memory"] = args.enable_memory == "true"
    if args.enable_milvus is not None:
        overrides["enable_milvus"] = args.enable_milvus == "true"
    if args.save_conversation_task is not None:
        overrides["save_conversation_task"] = args.save_conversation_task == "true"
    config = config.with_overrides(**overrides)
    logger.info(
        "%s tenant=%s user=%s thread=%s short=%s long=%s scope=%s save_task=%s checkpointer=%s milvus=%s",
        colorize("[config]", "cyan"),
        config.tenant_id,
        config.user_id,
        config.thread_id,
        config.short_term_backend,
        config.long_term_backend,
        config.long_term_scope,
        config.save_conversation_task,
        config.checkpointer_backend,
        config.enable_milvus,
    )
    return config


def run_query(app, config: AppConfig, query: str):
    global MEMORY_MANAGER
    memory_context = ""
    if MEMORY_MANAGER:
        try:
            memory_context = MEMORY_MANAGER.build_personalized_prompt_context(
                user_id=config.user_id,
                thread_id=config.thread_id,
                query=query,
                tenant_id=config.tenant_id,
                max_memories=config.memory_top_k,
            )
        except Exception as exc:
            logger.warning("%s 读取记忆失败，忽略本轮注入: %s", colorize("[memory]", "yellow"), exc)
    state = create_initial_state(
        query=query,
        max_iterations=config.max_iterations,
        user_id=config.user_id,
        tenant_id=config.tenant_id,
        memory_context=memory_context,
    )
    result = app.invoke(
        state,
        {"configurable": {"thread_id": config.thread_id}},
    )
    final = result["final"]
    if MEMORY_MANAGER:
        try:
            MEMORY_MANAGER.persist_turn(
                tenant_id=config.tenant_id,
                user_id=config.user_id,
                thread_id=config.thread_id,
                query=query,
                answer=final,
            )
        except Exception as exc:
            logger.warning("%s 持久化记忆失败，已跳过: %s", colorize("[memory]", "yellow"), exc)
    return final


def read_user_input(prompt: str = "你: ") -> str:
    try:
        return input(prompt)
    except UnicodeDecodeError:
        print(prompt, end="", flush=True)
        raw = sys.stdin.buffer.readline()
        if raw == b"":
            raise EOFError
        encoding = sys.stdin.encoding or "utf-8"
        recovered = raw.decode(encoding, errors="replace").rstrip("\r\n")
        logger.warning("%s 检测到输入编码异常，已使用容错解码。", colorize("[input]", "yellow"))
        return recovered


def main():
    global MEMORY_MANAGER
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_cli_args()
    config = build_runtime_config(args)
    MEMORY_MANAGER = build_memory_manager(config)
    agents = build_agents(config.model, config.api_key, config)
    checkpointer = build_checkpointer(config)
    app = build_workflow_app(agents, checkpointer)
    if args.once_query:
        response = run_query(app, config, args.once_query)
        print(f"\nAI: {response}\n")
    else:
        while True:
            try:
                query = read_user_input("你: ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in {"quit", "exit", "退出"}:
                break
            if query.lower() in {"/memory", "memory-status"} and MEMORY_MANAGER:
                print(json.dumps(MEMORY_MANAGER.get_memory_stats(config.user_id), ensure_ascii=False, indent=2))
                continue
            if query.lower() in {"/memory-trace", "memory-trace"} and MEMORY_MANAGER:
                print(json.dumps(MEMORY_MANAGER.get_last_trace(), ensure_ascii=False, indent=2))
                continue
            response = run_query(app, config, query)
            print(f"\nAI: {response}\n")
    cleanup_checkpointer()


if __name__ == "__main__":
    main()
