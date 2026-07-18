"""工具包：聚合所有工具函数，对外提供统一入口。

使用方式:
    from deep_research.agents.tools import (
        init_rag_system,
        search_knowledge_base_records,
        bocha_web_search_records,
        web_search_stub,
        search_knowledge_base,
        safe_list_dir,
        safe_read_file,
        get_current_time,
        simple_calculator,
        # ... 其他 @tool 函数
    )
"""

# ── Web 检索工具 ──────────────────────────────────────────────
from .web import (
    bocha_web_search_records,
    extract_url_content_stub,
    finance_search_stub,
    news_search_stub,
    web_search_stub,
)

# ── 知识库工具 ────────────────────────────────────────────────
from .knowledge import (
    _RAG_SYSTEM,
    init_rag_system,
    search_knowledge_base,
    search_knowledge_base_records,
)

# ── 文件操作工具 ──────────────────────────────────────────────
from .file_ops import (
    _safe_path,
    _workspace_root,
    safe_list_dir,
    safe_move_file,
    safe_read_file,
    safe_write_file,
)

# ── 通用工具 ─────────────────────────────────────────────────
from .utility import (
    ALLOWED_OPERATORS,
    _eval_node,
    amap_geocode,
    amap_poi_search,
    amap_route_plan,
    amap_weather,
    dedupe_lines,
    execute_terminal_command,
    explain_term,
    extract_data_stub,
    extract_requirements,
    fig_inter,
    file_operation_stub,
    get_current_time,
    local_docs_lookup_stub,
    local_vector_search_stub,
    merge_notes,
    optimize_query,
    outline_from_topics,
    python_inter,
    simple_calculator,
    sql_inter,
    summarize_points,
)

__all__ = [
    # web
    "bocha_web_search_records",
    "web_search_stub",
    "news_search_stub",
    "finance_search_stub",
    "extract_url_content_stub",
    # knowledge
    "_RAG_SYSTEM",
    "init_rag_system",
    "search_knowledge_base_records",
    "search_knowledge_base",
    # file_ops
    "_workspace_root",
    "_safe_path",
    "safe_list_dir",
    "safe_read_file",
    "safe_write_file",
    "safe_move_file",
    # utility
    "ALLOWED_OPERATORS",
    "_eval_node",
    "get_current_time",
    "simple_calculator",
    "extract_requirements",
    "outline_from_topics",
    "merge_notes",
    "summarize_points",
    "dedupe_lines",
    "local_docs_lookup_stub",
    "local_vector_search_stub",
    "optimize_query",
    "explain_term",
    "python_inter",
    "fig_inter",
    "amap_weather",
    "amap_geocode",
    "amap_poi_search",
    "amap_route_plan",
    "sql_inter",
    "extract_data_stub",
    "execute_terminal_command",
    "file_operation_stub",
]
