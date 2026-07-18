# Deep Research

> 基于 LangGraph 的多智能体深度研究系统，融合网络检索与本地知识库，自动生成带引用的深度研报。

---

## 📖 项目简介

Deep Research 是一个多智能体（Multi-Agent）深度研究系统，能够针对用户的复杂问题，自动执行以下流程：

1. **意图识别** — 判断问题是简单问答还是需要深度研究
2. **任务规划** — 将问题拆解为子问题，生成搜索计划
3. **双源检索** — 并行执行网络搜索（Bocha API）和本地知识库检索（Milvus 向量数据库）
4. **证据裁判** — 对所有证据评分、去重、冲突审计
5. **深度分析** — 生成结论，评估证据完备性
6. **迭代补搜** — 若证据不足，自动生成补搜计划并重复检索（最多 N 轮）
7. **报告撰写** — 生成带引用标注的 Markdown 格式深度研报

系统支持会话记忆（短期 + 长期）、多租户隔离、SSE 实时流式输出。

---

## 🏗️ 系统架构

### 整体架构图

```
                           ┌─────────────────────────────────────┐
                           │           用户浏览器                  │
                           │     http://localhost:3000            │
                           └──────────────┬──────────────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────────────┐
                           │    Frontend (Nginx + Vue SPA)        │
                           │    · 静态文件服务                     │
                           │    · /api → Backend 反向代理          │
                           │    · /health → Backend 反向代理       │
                           └──────────────┬───────────────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────────────┐
                           │    Backend (FastAPI + Uvicorn)       │
                           │    · POST /api/v1/research/run       │
                           │    · POST /api/v1/research/stream    │
                           │    · GET  /health                    │
                           └──────────────┬───────────────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────────────┐
                           │    LangGraph Workflow Engine         │
                           │    · 9 个节点的状态图                 │
                           │    · 条件路由 + 迭代循环              │
                           └───┬──────────┬──────────┬────────────┘
                               │          │          │
                    ┌──────────┘    ┌─────┘    ┌─────┘
                    ▼               ▼          ▼
           ┌──────────────┐ ┌──────────┐ ┌──────────────┐
           │   MySQL      │ │  Redis   │ │   Milvus     │
           │  记忆存储     │ │  缓存    │ │  向量数据库   │
           │  (短期+长期)  │ │  +Ckpt   │ │  (RAG 检索)  │
           └──────────────┘ └──────────┘ └──────────────┘
```

### 工作流节点图

```
                         ┌─────────┐
                         │  START  │
                         └────┬────┘
                              │
                              ▼
                      ┌───────────────┐
                      │    intent     │  意图识别
                      └───────┬───────┘
                       ┌──────┴──────┐
                       │             │
                  (direct)      (multiagent)
                       │             │
                       ▼             ▼
              ┌──────────────┐  ┌──────────┐
              │direct_answer │  │   plan   │  任务规划
              │  直接回答     │  └────┬─────┘
              └──────┬───────┘       │
                     │          ┌────┴─────┐
                     │          │          │
                     │          ▼          ▼
                     │   ┌────────────┐ ┌────────────┐
                     │   │ web_search │ │ local_rag  │  并行检索
                     │   │  网络搜索   │ │ 本地知识库  │
                     │   └─────┬──────┘ └─────┬──────┘
                     │         │              │
                     │         └──────┬───────┘
                     │                │
                     │                ▼
                     │        ┌──────────────┐
                     │        │  deep_dive   │  证据裁判
                     │        └──────┬───────┘
                     │               │
                     │               ▼
                     │        ┌──────────────┐
                     │        │   analyze    │  分析结论
                     │        └──────┬───────┘
                     │          ┌────┴────┐
                     │          │         │
                     │    (sufficient)  (gaps found)
                     │          │         │
                     │          │         ▼
                     │          │   ┌──────────┐
                     │          │   │ reflect  │  补搜规划
                     │          │   └────┬─────┘
                     │          │        │
                     │          │   ┌────┴─────┐
                     │          │   │          │
                     │          │   ▼          ▼
                     │          │  web_search  local_rag  ← 循环
                     │          │
                     │          ▼
                     │   ┌──────────────┐
                     │   │    write     │  撰写报告
                     │   └──────┬───────┘
                     │          │
                     ▼          ▼
                      ┌─────────────┐
                      │     END     │
                      └─────────────┘
```

---

## 📁 项目结构

```
deep_research/
├── pyproject.toml                  # Python 项目配置（包名、依赖、入口点）
├── requirements.txt                # pip 依赖锁定
├── environment.yml                 # Conda 环境配置
├── config.json                     # 运行时配置文件
├── .env.example                    # 环境变量模板
├── .env                            # 环境变量（不提交到 Git）
├── main.py                         # 根入口（调用 cli.py）
│
├── Dockerfile                      # 后端 Docker 镜像
├── docker-compose.yml              # Docker Compose 编排（5 个服务）
├── .dockerignore                   # Docker 构建排除规则
│
├── src/
│   └── deep_research/              # 主 Python 包
│       ├── __init__.py
│       ├── config.py               # 统一配置（AppConfig + APISettings）
│       ├── cli.py                  # CLI 入口（交互式 REPL / 单次查询）
│       │
│       ├── agents/                 # Agent 核心
│       │   ├── builder.py          # Agent 构建工厂
│       │   ├── graph.py            # LangGraph 工作流定义
│       │   ├── state.py            # ResearchState 状态定义
│       │   ├── prompts.py          # 所有 Agent 的 System Prompt
│       │   ├── nodes/              # 工作流节点实现
│       │   │   ├── base.py         # 共享辅助函数
│       │   │   ├── intent.py       # 意图识别 + 直接回答
│       │   │   ├── plan.py         # 任务规划
│       │   │   ├── search.py       # 网络搜索 + 本地检索
│       │   │   ├── evidence.py     # 证据裁判
│       │   │   ├── analyze.py      # 分析 + 反思
│       │   │   └── write.py        # 报告撰写
│       │   └── tools/              # Agent 可调用的工具
│       │       ├── web.py          # 网络搜索工具
│       │       ├── knowledge.py    # 知识库工具
│       │       ├── file_ops.py     # 文件操作工具
│       │       └── utility.py      # 通用工具（计算器、时间等）
│       │
│       ├── rag/                    # RAG 检索系统
│       │   ├── core.py             # RAGSystem（Milvus 向量存储）
│       │   ├── bm25_retriever.py   # BM25+ 稀疏检索
│       │   ├── hybrid_retriever.py # 混合检索器（向量 + BM25）
│       │   ├── rrf_fusion.py       # RRF 融合算法
│       │   └── ingest.py           # 文档导入脚本
│       │
│       ├── memory/                 # 记忆系统
│       │   ├── base.py             # 基础类型（MemoryEntry, MemoryType）
│       │   ├── short_term.py       # 短期记忆（会话缓冲）
│       │   ├── long_term.py        # 长期记忆（语义 + 情景）
│       │   ├── manager.py          # 记忆管理器（统一接口）
│       │   └── utils.py            # 记忆工具函数
│       │
│       ├── api/                    # FastAPI API 层
│       │   ├── app.py              # 应用工厂
│       │   ├── deps.py             # WorkflowService 依赖注入
│       │   ├── routers/
│       │   │   ├── health.py       # GET /health
│       │   │   └── research.py     # POST /api/v1/research/run|stream
│       │   └── schemas/
│       │       ├── health.py       # HealthResponse
│       │       └── research.py     # ResearchRequest/Response
│       │
│       └── utils/
│           └── __init__.py         # 共享工具（ANSI 颜色、日志辅助）
│
├── tests/                          # 测试
│   ├── test_bocha_api.py           # Bocha API 集成测试
│   └── rag/
│       ├── test_bm25_retriever.py  # BM25 检索器单元测试
│       ├── test_hybrid_retriever.py# 混合检索器单元测试
│       └── test_rrf_fusion.py      # RRF 融合算法单元测试
│
├── front/
│   └── agent_front/                # 前端（Vue 3 + TypeScript + Vite）
│       ├── Dockerfile              # 前端 Docker 镜像
│       ├── nginx.conf              # Nginx 配置（SPA + API 代理）
│       ├── src/
│       │   ├── main.ts             # Vue 入口
│       │   └── App.vue             # 主组件（聊天界面）
│       └── ...
│
└── data/                           # 运行时数据（Git 忽略）
    └── .gitkeep
```

---

## 🧩 核心模块详解

### 1. Agent 系统 (`agents/`)

系统包含 8 个专业 Agent，由 `AgentBundle` 统一管理：

| Agent | 角色 | Prompt Key | Temperature | 职责 |
|-------|------|------------|-------------|------|
| `intent_router` | IntentRouter | `intent_router` | 0.0 | 意图识别：direct / multiagent |
| `planner` | ChiefArchitect | `plan` | 0.3 | 任务拆解、大纲生成、搜索计划 |
| `scout_web` | WebScout | `web_search` | 0.4 | 网络证据整理与过滤 |
| `scout_local` | LocalRAGScout | `local_rag` | 0.4 | 本地知识库证据整理与过滤 |
| `evidence_judge` | EvidenceJudge | `deep_dive` | 0.2 | 证据评分、去重、冲突审计 |
| `analyst` | Analyst | `analyze` | 0.3 | 结论生成、证据完备性评估 |
| `direct_responder` | DirectResponder | `direct_answer` | 0.2 | 简单问答直接回答 |
| `writer` | Writer | `write` | 0.4 | 最终 Markdown 研报撰写 |

所有 Agent 使用 **DashScope ChatTongyi**（通义千问）作为 LLM 后端，System Prompt 强制 JSON 输出。

### 2. RAG 检索系统 (`rag/`)

```
                    用户查询
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
    ┌───────────────┐    ┌───────────────┐
    │  向量检索      │    │  BM25 检索     │
    │  (Milvus)      │    │  (关键词匹配)   │
    │  text-embedding│    │  jieba 分词    │
    │  -v1           │    │  BM25+ 评分    │
    └───────┬───────┘    └───────┬───────┘
            │                     │
            └──────────┬──────────┘
                       ▼
               ┌───────────────┐
               │  RRF 融合     │
               │  (倒数排名融合) │
               └───────┬───────┘
                       ▼
                  混合检索结果
```

**向量检索**：使用 DashScope `text-embedding-v1` 模型，通过 `langchain-milvus` 连接 Milvus 向量数据库。

**BM25 检索**：完整的 BM25+ 实现，支持三种分词器：
- `SimpleTokenizer` — 正则分词（英文单词 + 中文二元/三元组）
- `ChineseTokenizer` — jieba 精确分词
- `MixedTokenizer` — 中英文混合分词

**RRF 融合**：`RRF_score(d) = Σ(w_i / (k + rank_i(d)))`，支持自适应权重调整。

### 3. 记忆系统 (`memory/`)

```
┌─────────────────────────────────────────────────────────────┐
│                    MemoryManager                            │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐  │
│  │   短期记忆       │  │   长期记忆       │  │  向量索引   │  │
│  │                 │  │                 │  │  (Milvus)   │  │
│  │  · Redis 列表   │  │  · MySQL 表     │  │            │  │
│  │  · MySQL 表     │  │  · SQLite       │  │  · 语义搜索 │  │
│  │  · 内存缓冲     │  │  · 语义记忆     │  │  · 情景搜索 │  │
│  │                 │  │  · 情景记忆     │  │            │  │
│  │  自动压缩摘要   │  │  · 用户画像     │  │  text-     │  │
│  │  TTL 过期       │  │  · 事实知识     │  │  embedding  │  │
│  └─────────────────┘  └─────────────────┘  │  -v1        │  │
│                                             └────────────┘  │
│                                                             │
│  核心能力：                                                  │
│  · 会话摘要压缩（LLM 驱动）                                  │
│  · 自动偏好提取（"记住"/"我喜欢" 等关键词触发）               │
│  · 个性化 Prompt 注入（用户画像 + 历史记忆 + 对话摘要）       │
│  · 多租户隔离（tenant_id + user_id + thread_id）             │
└─────────────────────────────────────────────────────────────┘
```

**四种记忆类型**：

| 类型 | 枚举值 | 说明 | 存储 |
|------|--------|------|------|
| 短期记忆 | `SHORT_TERM` | 当前会话上下文 | Redis / MySQL / 内存 |
| 语义记忆 | `SEMANTIC` | 用户画像、事实知识 | MySQL + Milvus |
| 情景记忆 | `EPISODIC` | 历史任务、执行轨迹 | MySQL + Milvus |
| 程序记忆 | `PROCEDURAL` | 系统提示、行为模式 | 已定义，未启用 |

### 4. API 层 (`api/`)

**端点**：

| 方法 | 路径 | 说明 |
|------|------|------ |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/research/run` | 同步研究（返回最终报告） |
| `POST` | `/api/v1/research/stream` | SSE 流式研究（实时进度） |

**SSE 流式事件类型**：

| 事件类型 | 说明 | 示例 |
|----------|------|------|
| `status` | 初始确认 | `{"type":"status","message":"任务已接收"}` |
| `phase` | 节点进度 | `{"type":"phase","node":"plan","message":"Planner 正在拆解问题"}` |
| `route` | 路由结果 | `{"type":"route","message":"已走多智能体研究路径"}` |
| `final` | 最终报告 | `{"type":"final","final":"# 研究报告\n..."}` |
| `error` | 错误信息 | `{"type":"error","message":"..."}` |

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

---

## ⚙️ 配置说明

### 配置优先级

```
环境变量 > config.json > 代码默认值
```

### 核心配置项 (`AppConfig`)

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| `api_key` | `DASHSCOPE_API_KEY` | （必填） | DashScope API Key |
| `model` | `MODEL` | `qwen-plus` | LLM 模型名 |
| `max_iterations` | `MAX_ITERATIONS` | `3` | 最大研究迭代轮数 |
| `enable_memory` | `ENABLE_MEMORY` | `true` | 是否启用记忆系统 |
| `short_term_backend` | `SHORT_TERM_BACKEND` | `mysql` | 短期记忆后端 |
| `long_term_backend` | `LONG_TERM_BACKEND` | `mysql` | 长期记忆后端 |
| `checkpointer_backend` | `CHECKPOINTER_BACKEND` | `redis` | LangGraph checkpointer |
| `enable_milvus` | `ENABLE_MILVUS` | `true` | 是否启用 Milvus |
| `mysql_dsn` | `MYSQL_DSN` | `mysql+pymysql://...` | MySQL 连接串 |
| `redis_url` | `REDIS_URL` | `redis://redis:6379` | Redis 连接串 |
| `milvus_host` | `MILVUS_HOST` | `milvus` | Milvus 主机 |

完整配置项见 `config.py` 和 `.env.example`。

---

## 🚀 快速开始

### 方式一：Docker Compose（推荐）

```bash
# 1. 克隆项目
git clone <repo-url>
cd deep_research

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填写 DASHSCOPE_API_KEY 和 BOCHA_API_KEY

# 3. 启动所有服务
docker compose up -d --build

# 4. 查看状态
docker compose ps

# 5. 访问
# 前端：http://localhost:3000
# API：http://localhost:8080/health
```

**Docker 服务架构**：

| 服务 | 镜像 | 对外端口 | 内部端口 |
|------|------|----------|----------|
| frontend | nginx:alpine + Vue SPA | **3000** | 80 |
| backend | python:3.11-slim + FastAPI | **8080** | 8000 |
| mysql | mysql:8.0.30 | ❌ | 3306 |
| redis | redis:7-alpine | ❌ | 6379 |
| milvus | milvusdb/milvus:v2.6.14 | ❌ | 19530 |

**命名卷**：`mysql_data`、`redis_data`、`milvus_data`、`app_data`

**网络**：`deep_research_net`（bridge，所有服务内部通信）

### 方式二：本地开发

```bash
# 1. 安装依赖
pip install -e .

# 2. 启动基础设施（MySQL、Redis、Milvus）
docker compose up -d mysql redis milvus

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，将连接地址改为 localhost

# 4. 启动后端
python -m deep_research.cli --once-query "测试查询"
# 或交互模式
python -m deep_research.cli

# 5. 启动前端
cd front/agent_front
npm install
npm run dev
```

---

## 🔌 API 使用示例

### 同步请求

```bash
curl -X POST http://localhost:8080/api/v1/research/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "2025年 AI Agent 发展趋势",
    "user_id": "user_001",
    "thread_id": "thread_001",
    "tenant_id": "tenant_001",
    "max_iterations": 2,
    "enable_memory": true
  }'
```

### SSE 流式请求

```bash
curl -X POST http://localhost:8080/api/v1/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "LangChain vs LlamaIndex 对比分析"}' \
  --no-buffer
```

### Python 调用

```python
import httpx

response = httpx.post(
    "http://localhost:8080/api/v1/research/run",
    json={"query": "什么是 RAG？", "user_id": "demo"},
    timeout=300,
)
print(response.json()["final"])
```

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行 RAG 模块测试
pytest tests/rag/ -v

# 运行单个测试文件
pytest tests/rag/test_bm25_retriever.py -v
```

**测试覆盖**：

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_bm25_retriever.py` | 分词器、BM25 索引、IDF 计算、评分、批量查询、持久化 |
| `test_hybrid_retriever.py` | 混合检索、向量/BM25 单独模式、缓存、异步、权重调整 |
| `test_rrf_fusion.py` | RRF 融合、权重、空结果、归一化、自适应融合、多阶段融合 |
| `test_bocha_api.py` | Bocha Web Search API 集成测试 |

---

## 🛠️ 技术栈

| 层 | 技术 |
|----|------|
| **LLM** | DashScope ChatTongyi（通义千问） |
| **Embedding** | DashScope `text-embedding-v1` |
| **工作流引擎** | LangGraph（StateGraph） |
| **Web 框架** | FastAPI + Uvicorn |
| **向量数据库** | Milvus（langchain-milvus） |
| **关系数据库** | MySQL 8.0（pymysql） |
| **缓存** | Redis 7（langgraph-checkpoint-redis） |
| **稀疏检索** | BM25+（rank-bm25 + jieba） |
| **前端** | Vue 3 + TypeScript + Vite |
| **反向代理** | Nginx |
| **容器化** | Docker + Docker Compose |

---

## 📝 开发说明

### 添加新的 Agent 节点

1. 在 `agents/nodes/` 下创建新模块
2. 实现节点函数 `def my_node(state: ResearchState, agent, agent_name: str) -> ResearchState`
3. 在 `agents/nodes/__init__.py` 中导出
4. 在 `agents/graph.py` 中添加节点和边
5. 在 `agents/prompts.py` 中添加对应的 System Prompt

### 添加新的工具

1. 在 `agents/tools/` 下创建新模块
2. 使用 `@tool` 装饰器定义工具函数
3. 在 `agents/tools/__init__.py` 中导出

### 修改记忆后端

编辑 `config.json` 或 `.env` 中的 `short_term_backend` 和 `long_term_backend`：
- `mysql` — 使用 MySQL（生产推荐）
- `redis` — 使用 Redis（仅短期记忆）
- `memory` / `sqlite` — 本地内存 / SQLite（开发测试）

---

## 📄 许可证

MIT License
