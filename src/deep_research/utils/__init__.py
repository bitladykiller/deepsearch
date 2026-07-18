"""共享日志与终端输出工具。"""

import logging
import os

logger = logging.getLogger("deep_research")

# ANSI 终端颜色
ANSI = {
    "reset": "\033[0m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
}


def colorize(text: str, color: str) -> str:
    """给文本添加 ANSI 颜色，NO_COLOR 环境变量可禁用。"""
    if os.getenv("NO_COLOR"):
        return text
    code = ANSI.get(color, "")
    if not code:
        return text
    return f"{code}{text}{ANSI['reset']}"


def emit(node: str, content: str):
    """输出节点结果摘要到日志。"""
    preview = content.replace("\n", " ")
    if len(preview) > 400:
        preview = preview[:400] + "..."
    logger.info("%s 输出: %s", colorize(f"[{node}]", "yellow"), preview)


def collect_tool_calls(messages) -> tuple[list, list]:
    """从消息列表中提取工具调用名称和输出。"""
    tools = []
    tool_outputs = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                if name:
                    tools.append(name)
        name = getattr(msg, "name", None)
        msg_type = getattr(msg, "type", None)
        if msg_type == "tool" and name:
            tools.append(name)
            output = getattr(msg, "content", "")
            if output:
                tool_outputs.append(f"{name}: {output}")
    return tools, tool_outputs


def with_memory_context(state: dict, user_prompt: str) -> str:
    """在用户 prompt 前注入跨会话记忆上下文。"""
    memory_context = state.get("memory_context", "").strip()
    if not memory_context:
        return user_prompt
    return f"{user_prompt}\n\n[跨会话记忆]\n{memory_context}"


def log_inputs(node: str, agent_name: str, payload: dict):
    """记录节点输入到日志（截断长字段）。"""
    preview = {
        key: (value[:200] + "..." if isinstance(value, str) and len(value) > 200 else value)
        for key, value in payload.items()
    }
    logger.info(
        "%s 输入 | agent=%s | data=%s",
        colorize(f"[{node}]", "cyan"),
        colorize(agent_name, "magenta"),
        preview,
    )
