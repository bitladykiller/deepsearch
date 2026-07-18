"""通用工具：封装时间、计算器、文本处理、模拟接口等辅助工具函数。"""

import ast
import operator
from datetime import datetime

from langchain_core.tools import tool

# 计算器支持的安全运算符映射
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _eval_node(node):
    """递归求值 AST 节点，仅支持安全的算术运算。

    Args:
        node: ast 表达式节点。

    Returns:
        计算结果（int 或 float）。

    Raises:
        ValueError: 表达式包含不支持的运算符或结构时抛出。
    """
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("Unsupported expression")


@tool
def get_current_time() -> str:
    """返回当前时间的 ISO 字符串。"""
    return datetime.now().isoformat()


@tool
def simple_calculator(expression: str) -> str:
    """计算简单算术表达式并返回结果。"""
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree.body)
    return str(result)


@tool
def extract_requirements(text: str) -> str:
    """从文本中提取需求要点列表。"""
    items = [part.strip() for part in text.replace("\n", " ").split("。") if part.strip()]
    return "\n".join(f"- {item}" for item in items[:8])


@tool
def outline_from_topics(topics: str) -> str:
    """根据主题列表生成编号大纲。"""
    raw = topics.replace("\n", ",")
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return "\n".join(f"{idx+1}. {item}" for idx, item in enumerate(items[:10]))


@tool
def merge_notes(note_a: str, note_b: str) -> str:
    """合并两段文本为一段笔记。"""
    return f"{note_a}\n{note_b}".strip()


@tool
def summarize_points(text: str) -> str:
    """从文本中抽取要点列表。"""
    sentences = [s.strip() for s in text.replace("\n", " ").split("。") if s.strip()]
    points = sentences[:6]
    return "\n".join(f"- {p}" for p in points)


@tool
def dedupe_lines(text: str) -> str:
    """对文本按行去重并输出。"""
    seen = set()
    lines = []
    for line in text.splitlines():
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines)


@tool
def local_docs_lookup_stub(query: str) -> str:
    """模拟本地检索接口。"""
    return f"未配置本地检索服务，收到查询: {query}"


@tool
def local_vector_search_stub(query: str) -> str:
    """模拟向量数据库检索接口。"""
    return f"未配置向量数据库，收到查询: {query}"


@tool
def optimize_query(query: str) -> str:
    """对检索问题进行改写与优化。"""
    return f"优化后的查询建议: {query}"


@tool
def explain_term(term: str) -> str:
    """解释领域术语。"""
    return f"{term} 需要结合上下文进一步解释"


@tool
def python_inter(code: str) -> str:
    """模拟 Python 执行环境。"""
    return f"未配置Python执行环境，收到代码: {code}"


@tool
def fig_inter(spec: str) -> str:
    """模拟绘图执行环境。"""
    return f"未配置绘图环境，收到图表需求: {spec}"


@tool
def amap_weather(city: str) -> str:
    """模拟高德天气查询。"""
    return f"未配置高德API，收到天气查询: {city}"


@tool
def amap_geocode(address: str) -> str:
    """模拟高德地理编码。"""
    return f"未配置高德API，收到地理编码请求: {address}"


@tool
def amap_poi_search(query: str) -> str:
    """模拟高德 POI 检索。"""
    return f"未配置高德API，收到POI检索: {query}"


@tool
def amap_route_plan(origin: str, destination: str) -> str:
    """模拟高德路径规划。"""
    return f"未配置高德API，收到路径规划: {origin} -> {destination}"


@tool
def sql_inter(query: str) -> str:
    """模拟 SQL 执行接口。"""
    return f"未配置数据库，收到SQL: {query}"


@tool
def extract_data_stub(query: str) -> str:
    """模拟数据抽取接口。"""
    return f"未配置数据抽取环境，收到请求: {query}"


@tool
def execute_terminal_command(command: str) -> str:
    """模拟终端命令执行接口。"""
    return f"未配置终端执行环境，收到命令: {command}"


@tool
def file_operation_stub(request: str) -> str:
    """模拟文件操作接口。"""
    return f"未配置文件操作环境，收到请求: {request}"
