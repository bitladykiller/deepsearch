# 08 - API 层与 CLI

## 1. 概述

系统提供两种使用方式：
- **FastAPI REST API**：通过 HTTP 请求触发研究任务，支持同步和 SSE 流式输出
- **CLI 命令行**：交互式 REPL 或单次查询模式

---

## 2. FastAPI 应用（api/app.py）

### 2.1 应用工厂

```python
def create_app() -> FastAPI:
    settings = APISettings.from_env()
    app = FastAPI(title=settings.app_name)
    
    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(health_router)
    app.include_router(research_router)
    
    return app

app = create_app()
```

### 2.2 CORS 配置

默认允许的源：
- `http://localhost:5173`（Vite 开发服务器）
- `http://127.0.0.1:5173`

Docker 环境下，前端通过 Nginx 代理到同一域名，CORS 不是问题。但在本地开发时，前端和后端在不同端口，需要 CORS。

### 2.3 日志配置

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.getLogger("deep_research").setLevel(logging.INFO)
```

所有模块使用统一的 logger 名称 `deep_research`，子模块使用 `deep_research.memory`、`deep_research.api` 等。

---

## 3. API 路由

### 3.1 健康检查（health.py）

```python
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="deepresearch-backend")
```

**响应**：
```json
{
  "status": "ok",
  "service": "deepresearch-backend"
}
```

### 3.2 同步研究接口（research.py）

```python
@router.post("/api/v1/research/run", response_model=ResearchResponse)
async def run_research(
    payload: ResearchRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> ResearchResponse:
    final = await workflow_service.run(
        query=payload.query,
        user_id=payload.user_id,
        thread_id=payload.thread_id,
        tenant_id=payload.tenant_id,
        max_iterations=payload.max_iterations,
        enable_memory=payload.enable_memory,
    )
    return ResearchResponse(
        query=payload.query,
        user_id=payload.user_id,
        thread_id=payload.thread_id,
        tenant_id=payload.tenant_id,
        final=final,
    )
```

**请求体** (`ResearchRequest`)：
```json
{
  "query": "2025年 AI Agent 发展趋势",
  "user_id": "user_001",
  "thread_id": "thread_001",
  "tenant_id": "tenant_001",
  "max_iterations": 3,
  "enable_memory": true
}
```

**响应体** (`ResearchResponse`)：
```json
{
  "query": "2025年 AI Agent 发展趋势",
  "user_id": "user_001",
  "thread_id": "thread_001",
  "tenant_id": "tenant_001",
  "final": "# 2025年 AI Agent 发展趋势\n\n## 核心摘要\n..."
}
```

### 3.3 SSE 流式研究接口

```python
@router.post("/api/v1/research/stream")
async def stream_research(
    payload: ResearchRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> StreamingResponse:
    async def event_stream():
        # 1. 发送初始确认事件
        start_event = {"type": "status", "message": "任务已接收，正在初始化多智能体链路"}
        yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"
        
        # 2. 流式推送节点进度
        async for event in workflow_service.stream_events(...):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

**SSE 事件类型**：

| 事件类型 | 说明 | 示例 |
|----------|------|------|
| `status` | 初始确认 | `{"type":"status","message":"任务已接收"}` |
| `phase` | 节点进度 | `{"type":"phase","node":"plan","message":"Planner 正在拆解问题"}` |
| `route` | 路由结果 | `{"type":"route","message":"已走多智能体研究路径"}` |
| `final` | 最终报告 | `{"type":"final","final":"# 研究报告\n..."}` |
| `error` | 错误信息 | `{"type":"error","message":"..."}` |

**SSE 数据格式**：
```
data: {"type":"status","message":"任务已接收，正在初始化多智能体链路"}

data: {"type":"phase","node":"intent","message":"Intent Router 正在识别问题意图"}

data: {"type":"phase","node":"plan","message":"Planner 正在拆解问题"}

data: {"type":"phase","node":"web_search","message":"Web Scout 正在检索网络证据"}

data: {"type":"route","message":"已走多智能体研究路径"}

data: {"type":"final","query":"...","final":"# 研究报告\n..."}
```

---

## 4. WorkflowService（deps.py）

### 4.1 单例模式

```python
_SERVICE: "WorkflowService | None" = None

def get_workflow_service() -> "WorkflowService":
    global _SERVICE
    if _SERVICE is None:
        from ..config import APISettings
        settings = APISettings.from_env()
        _SERVICE = WorkflowService(settings.config_path)
    return _SERVICE
```

整个应用生命周期内只有一个 WorkflowService 实例。

### 4.2 懒加载初始化

```python
class WorkflowService:
    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            # 1. 加载配置
            base_config = AppConfig.from_file(self._config_path)
            # 2. 初始化记忆管理器
            self._memory_manager = build_memory_manager(base_config)
            # 3. 构建所有 Agent
            agents = build_agents(base_config.model, base_config.api_key, base_config)
            # 4. 构建 checkpointer
            checkpointer = build_checkpointer(base_config)
            # 5. 编译工作流
            self._app = build_workflow_app(agents, checkpointer)
            self._base_config = base_config
            self._initialized = True
```

使用双重检查锁（double-checked locking）确保线程安全。

### 4.3 运行时配置构建

```python
def _build_runtime_config(self, user_id, thread_id, tenant_id, max_iterations, enable_memory):
    overrides = {
        "user_id": user_id,
        "thread_id": thread_id,
        "tenant_id": tenant_id,
        "max_iterations": max_iterations or self._base_config.max_iterations,
    }
    if enable_memory is not None:
        overrides["enable_memory"] = enable_memory
    return self._base_config.with_overrides(**overrides)
```

每次 API 请求时，使用请求参数覆盖基础配置。

### 4.4 同步执行流程

```python
def _run_sync(self, query, user_id, thread_id, tenant_id, max_iterations, enable_memory):
    # 1. 确保初始化
    self._ensure_initialized()
    
    # 2. 构建运行时配置
    runtime_config = self._build_runtime_config(...)
    
    # 3. 获取记忆上下文
    memory_context = ""
    if self._memory_manager and runtime_config.enable_memory:
        memory_context = self._memory_manager.build_personalized_prompt_context(...)
    
    # 4. 创建初始状态
    state = create_initial_state(
        query=query,
        max_iterations=runtime_config.max_iterations,
        user_id=runtime_config.user_id,
        tenant_id=runtime_config.tenant_id,
        memory_context=memory_context,
    )
    
    # 5. 执行工作流
    result = self._app.invoke(
        state,
        {"configurable": {"thread_id": runtime_config.thread_id}},
    )
    
    # 6. 持久化记忆
    if self._memory_manager and runtime_config.enable_memory:
        self._memory_manager.persist_turn(
            tenant_id=..., user_id=..., thread_id=...,
            query=query, answer=final,
        )
    
    return final, route
```

### 4.5 流式执行流程

```python
async def stream_events(self, query, user_id, thread_id, tenant_id, ...):
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def emit(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(event), loop)
    
    def worker() -> None:
        try:
            # 在工作线程中执行同步工作流
            final, route = self._run_sync_with_events(
                query=query, ..., emit=emit,
            )
            # 发送路由结果事件
            emit({"type": "route", "message": "已走多智能体研究路径"})
            # 发送最终报告事件
            emit({"type": "final", "final": final})
        except Exception as exc:
            emit({"type": "error", "message": str(exc)})
        finally:
            emit({"type": "__done__"})
    
    # 在工作线程中执行
    Thread(target=worker, daemon=True).start()
    
    # 异步等待事件并 yield
    while True:
        event = await queue.get()
        if event.get("type") == "__done__":
            break
        yield event
```

**线程模型**：
```
主线程（FastAPI 事件循环）
    │
    ├── stream_events() 异步生成器
    │   └── yield event（等待 queue）
    │
    └── worker 线程（同步执行工作流）
        ├── _run_sync_with_events()
        │   └── app.stream(state, config, stream_mode="updates")
        │       └── emit(event) → queue.put(event)
        └── emit({"type": "__done__"})
```

### 4.6 节点消息映射

```python
@staticmethod
def _node_message(node_name: str) -> str:
    mapping = {
        "intent": "Intent Router 正在识别问题意图",
        "direct_answer": "Direct Responder 正在快速作答",
        "plan": "Planner 正在拆解问题",
        "web_search": "Web Scout 正在检索网络证据",
        "local_rag": "Local Scout 正在检索本地知识库",
        "deep_dive": "Evidence Judge 正在进行证据裁判",
        "analyze": "Analyst 正在生成结论",
        "reflect": "Reflect 正在生成补搜计划",
        "write": "Writer 正在撰写最终报告",
    }
    return mapping.get(node_name, f"{node_name} 正在执行")
```

---

## 5. CLI 入口（cli.py）

### 5.1 命令行参数

```python
def parse_cli_args():
    parser = argparse.ArgumentParser(description="Deep Research multi-agent runner")
    parser.add_argument("--config", type=str, default=None)          # 配置文件路径
    parser.add_argument("--tenant-id", type=str, default=None)       # 租户 ID
    parser.add_argument("--user-id", type=str, default=None)         # 用户 ID
    parser.add_argument("--thread-id", type=str, default=None)       # 线程 ID
    parser.add_argument("--short-term-backend", choices=["postgres", "redis", "memory"])
    parser.add_argument("--long-term-backend", choices=["postgres", "sqlite", "disabled"])
    parser.add_argument("--long-term-scope", choices=["user", "thread"])
    parser.add_argument("--save-conversation-task", choices=["true", "false"])
    parser.add_argument("--checkpointer-backend", choices=["postgres", "redis", "memory", "auto"])
    parser.add_argument("--enable-memory", choices=["true", "false"])
    parser.add_argument("--enable-milvus", choices=["true", "false"])
    parser.add_argument("--memory-top-k", type=int)
    parser.add_argument("--once-query", type=str, default=None)      # 单次查询模式
```

### 5.2 启动流程

```python
def main():
    # 1. 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    # 2. 解析命令行参数
    args = parse_cli_args()
    
    # 3. 构建运行时配置
    config = build_runtime_config(args)
    
    # 4. 初始化记忆管理器
    MEMORY_MANAGER = build_memory_manager(config)
    
    # 5. 构建 Agent
    agents = build_agents(config.model, config.api_key, config)
    
    # 6. 构建 checkpointer
    checkpointer = build_checkpointer(config)
    
    # 7. 编译工作流
    app = build_workflow_app(agents, checkpointer)
    
    # 8. 单次查询或交互模式
    if args.once_query:
        response = run_query(app, config, args.once_query)
        print(f"\nAI: {response}\n")
    else:
        # 交互式 REPL
        while True:
            query = read_user_input("你: ").strip()
            if query.lower() in {"quit", "exit", "退出"}:
                break
            if query.lower() in {"/memory", "memory-status"}:
                print(json.dumps(MEMORY_MANAGER.get_memory_stats(config.user_id), ...))
                continue
            response = run_query(app, config, query)
            print(f"\nAI: {response}\n")
    
    # 9. 清理
    cleanup_checkpointer()
```

### 5.3 run_query 函数

```python
def run_query(app, config, query):
    # 1. 获取记忆上下文
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
            logger.warning("读取记忆失败，忽略本轮注入: %s", exc)
    
    # 2. 创建初始状态
    state = create_initial_state(
        query=query,
        max_iterations=config.max_iterations,
        user_id=config.user_id,
        tenant_id=config.tenant_id,
        memory_context=memory_context,
    )
    
    # 3. 执行工作流
    result = app.invoke(state, {"configurable": {"thread_id": config.thread_id}})
    
    # 4. 持久化记忆
    if MEMORY_MANAGER:
        MEMORY_MANAGER.persist_turn(
            tenant_id=config.tenant_id,
            user_id=config.user_id,
            thread_id=config.thread_id,
            query=query,
            answer=result["final"],
        )
    
    return result["final"]
```

### 5.4 特殊命令

| 命令 | 作用 |
|------|------|
| `quit` / `exit` / `退出` | 退出 CLI |
| `/memory` / `memory-status` | 显示记忆系统状态（JSON 格式） |
| `/memory-trace` / `memory-trace` | 显示最近一次记忆注入的详细追踪 |

### 5.5 入口点配置

```toml
# pyproject.toml
[project.scripts]
deep-research = "deep_research.cli:main"
```

安装后可以直接运行：
```bash
deep-research --once-query "什么是 RAG？"
deep-research  # 交互模式
```

---

## 6. 共享工具函数（utils/__init__.py）

### 6.1 ANSI 颜色

```python
ANSI = {
    "reset": "\033[0m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
}

def colorize(text: str, color: str) -> str:
    if os.getenv("NO_COLOR"):
        return text
    code = ANSI.get(color, "")
    return f"{code}{text}{ANSI['reset']}" if code else text
```

`NO_COLOR` 环境变量可以禁用颜色输出（Docker 环境中默认禁用）。

### 6.2 emit 函数

```python
def emit(node: str, content: str):
    preview = content.replace("\n", " ")
    if len(preview) > 400:
        preview = preview[:400] + "..."
    logger.info("%s 输出: %s", colorize(f"[{node}]", "yellow"), preview)
```

每个节点执行完成后调用 `emit` 输出摘要到日志。

### 6.3 collect_tool_calls

```python
def collect_tool_calls(messages) -> tuple[list, list]:
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
```

从 LangChain 消息列表中提取工具调用信息，用于日志记录。

### 6.4 with_memory_context

```python
def with_memory_context(state: dict, user_prompt: str) -> str:
    memory_context = state.get("memory_context", "").strip()
    if not memory_context:
        return user_prompt
    return f"{user_prompt}\n\n[跨会话记忆]\n{memory_context}"
```

在用户 prompt 前注入跨会话记忆上下文。所有节点的 `_invoke_json_agent` 调用都会经过这个函数。

---

## 7. 前端对接要点（补充）

主 UI 在 `front/agent_front/src/App.vue`，只走 **SSE** `/api/v1/research/stream`。

```
关键实现细节（完整图见 docs/11）:
1. fetch + ReadableStream + buffer.split('\\n\\n') 防粘包
2. 事件 type: status | phase | route | final | error
3. phase 写入 progressLogs（去重，窗口约 6 条）
4. final 删除 status 气泡，markdownToHtml 轻量渲染
5. 请求体仅带 query/user_id/thread_id/tenant_id
6. 新建会话不自动更换 threadId → 后端记忆可能连续
```

Schema 约束（`api/schemas/research.py`）：

| 字段 | 约束 |
|------|------|
| query | 必填，min_length=1 |
| max_iterations | 可选，1..6；null 用服务端配置 |
| enable_memory | 可选 bool；null 用服务端配置 |
