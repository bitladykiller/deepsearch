# 12 - Bocha 响应样例 · MySQL 表字段 · SSE 抓包时间线

> 本篇把三块“对照源码就能落地排障”的材料写透：  
> 1) Bocha Web Search 请求/响应结构与解析兼容  
> 2) 记忆系统 MySQL 四张表逐字段说明 + 读写 SQL  
> 3) 一次真实形态的 SSE 事件时间线（direct / multiagent / 迭代 / 失败）

相关源码：
- `src/deep_research/agents/tools/web.py`
- `src/deep_research/memory/manager.py`
- `src/deep_research/api/routers/research.py`
- `src/deep_research/api/deps.py`
- `front/agent_front/src/App.vue`

---

## 1. Bocha Web Search：请求、响应、解析全对照

### 1.1 调用位置

```
web_search_node
  → bocha_web_search_records(query_text, count=4)
      → POST https://api.bocha.cn/v1/web-search
```

主工作流每个 search plan 查询默认 `count=4`。  
`tests/test_bocha_api.py` 里还有 `ai-search` 端点探测，**主链路不使用 ai-search**。

### 1.2 请求规格

#### HTTP

| 项 | 值 |
|----|-----|
| Method | `POST` |
| URL | `https://api.bocha.cn/v1/web-search` |
| Header | `Authorization: Bearer ${BOCHA_API_KEY}` |
| Header | `Content-Type: application/json` |
| Timeout | 30s（`urlopen`） |
| Body 编码 | UTF-8 JSON（`ensure_ascii=False`） |

#### 请求体（源码构造）

```json
{
  "query": "AI Agent 2025 技术趋势",
  "summary": true,
  "freshness": "noLimit",
  "count": 4
}
```

| 字段 | 类型 | 含义 | 源码取值 |
|------|------|------|----------|
| query | string | 检索词 | `search_plan` / `supplementary_queries` 中的 query |
| summary | bool | 是否返回摘要 | 固定 `true` |
| freshness | string | 时间新鲜度 | 固定 `"noLimit"` |
| count | int | 条数上限 | 节点传 `4`；tool stub 用 `5` |

#### 环境变量

```
BOCHA_API_KEY=sk-xxxxxxxx   # 未配置 → 直接 return []，不发请求
```

### 1.3 响应结构：三种兼容形态

源码解析逻辑（`web.py`）：

```
result = json.loads(raw)
data = result.get("data", {})
pages = data.get("webPages", [])

if isinstance(pages, dict):
    if isinstance(pages.get("value"), list):
        pages = pages["value"]
    elif isinstance(pages.get("items"), list):
        pages = pages["items"]
    else:
        pages = []
if not isinstance(pages, list):
    return []
```

因此合法形态有三种：

#### 形态 A：`webPages` 直接是数组（最常见期望）

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "webPages": [
      {
        "name": "2025 AI Agent 趋势报告",
        "url": "https://example.com/ai-agent-2025",
        "summary": "报告指出多 Agent 协作与工具调用将成为主流……",
        "datePublished": "2025-11-02T08:00:00Z",
        "dateLastCrawled": "2025-11-03T01:20:00Z",
        "siteName": "Example Research",
        "displayUrl": "https://example.com/ai-agent-2025"
      },
      {
        "name": "LangChain vs CrewAI",
        "url": "https://blog.example.org/compare",
        "summary": "框架对比……",
        "datePublished": "2025-09-18",
        "dateLastCrawled": null
      }
    ],
    "queryContext": {
      "originalQuery": "AI Agent 2025 技术趋势"
    }
  }
}
```

#### 形态 B：`webPages.value` 为数组

```json
{
  "code": 200,
  "data": {
    "webPages": {
      "value": [
        {
          "name": "标题 A",
          "url": "https://a.example/x",
          "summary": "摘要 A",
          "datePublished": "2025-01-01"
        }
      ],
      "totalEstimatedMatches": 1280
    }
  }
}
```

#### 形态 C：`webPages.items` 为数组

```json
{
  "data": {
    "webPages": {
      "items": [
        {
          "name": "标题 B",
          "url": "https://b.example/y",
          "summary": "摘要 B",
          "dateLastCrawled": "2025-02-02T12:00:00Z"
        }
      ]
    }
  }
}
```

#### 形态 D：无法识别 → 空列表

```json
{ "data": { "webPages": { "foo": 1 } } }
{ "data": { "webPages": "not-a-list" } }
{ "data": {} }
```

### 1.4 单条 page → 内部 record 字段映射

```
Bocha page 字段          →  内部 record 字段
─────────────────────────────────────────────
name                     →  title（空则 web_result_{idx}）
url                      →  url
url 的 host 部分         →  domain
summary                  →  snippet
datePublished
  或 dateLastCrawled     →  published_at
(固定)                   →  source_type = "web"
(临时)                   →  source_id = "WEB-{idx}"
```

标准化后示例：

```json
{
  "source_id": "WEB-1",
  "title": "2025 AI Agent 趋势报告",
  "url": "https://example.com/ai-agent-2025",
  "snippet": "报告指出多 Agent 协作与工具调用将成为主流……",
  "domain": "example.com",
  "source_type": "web",
  "published_at": "2025-11-02T08:00:00Z"
}
```

### 1.5 节点侧二次改写（非常关键）

`bocha_web_search_records` 返回的 `WEB-1` **不是最终 ID**。

```
web_search_node:
  prefix = f"WEB{iteration+1}"           # 第1轮 WEB1，第2轮 WEB2
  records = _assign_source_ids(records, f"{prefix}_{query_index}")
  → WEB1_1-1, WEB1_1-2, WEB1_2-1, ...

同时附加:
  section_id   = plan item.section_id
  search_query = plan item.query
```

**生命周期**：

```
Bocha 原始:     WEB-1
节点重写:       WEB1_2-3
deep_dive:      evidence_pool / source_index 保持 WEB1_2-3
write 正文:     [WEB1_2-3]
参考资料列表:   - [WEB1_2-3] [web]: title | url
```

### 1.6 错误响应样例与系统行为

#### 未配置 Key

```
BOCHA_API_KEY 空
→ logger.warning 未配置
→ return []
→ web_search_node:
   "未检索到可用网页证据，已跳过网页上下文注入。"
```

#### HTTP 401/403

```json
{
  "code": 401,
  "msg": "Invalid API key"
}
```

```
HTTPError → logger.error → return []
```

#### HTTP 429

```json
{
  "code": 429,
  "msg": "rate limit exceeded"
}
```

```
同样降级为空列表（当前无重试/backoff）
```

#### 超时 / DNS

```
URLError: timed out
→ logger.error → []
```

#### 非 JSON 正文

```
JSONDecodeError → []
```

### 1.7 解析决策全图

```
bocha_web_search_records(query, count)
  │
  ├─ KEY 空? ──yes──► []
  │
  ├─ POST /v1/web-search
  │     ├─ HTTPError ──► []
  │     ├─ URLError ───► []
  │     ├─ JSON err ───► []
  │     └─ other ──────► []
  │
  ├─ data.webPages
  │     ├─ list ──────────────► 用 list
  │     ├─ dict.value list ───► 用 value
  │     ├─ dict.items list ───► 用 items
  │     └─ else ──────────────► []
  │
  ├─ for page in pages[:count]
  │     skip 非 dict
  │     map → record
  │
  └─ return records
```

### 1.8 与测试脚本的差异

| 项 | 主链路 `web.py` | `tests/test_bocha_api.py` |
|----|-----------------|---------------------------|
| 端点 | 仅 web-search | web-search + ai-search |
| count | 4（节点） | 10 |
| 用途 | 生产取证 | 手工探测 API |
| 密钥 | 环境变量 | **不应硬编码**；请改用 `os.getenv("BOCHA_API_KEY")` |

> 安全提醒：若测试文件中曾写入真实 Key，应轮换密钥并从仓库历史中清理，文档与示例一律使用占位符。

### 1.9 手工 curl 对照

```bash
curl -sS -X POST 'https://api.bocha.cn/v1/web-search' \
  -H "Authorization: Bearer ${BOCHA_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "AI Agent 2025",
    "summary": true,
    "freshness": "noLimit",
    "count": 4
  }' | jq .
```

检查点：
1. 顶层是否有 `data`
2. `data.webPages` 是 list 还是 dict
3. 单条是否含 `name` / `url` / `summary`
4. 与日志 `[bocha_web_search] 返回记录数=` 是否一致

---

## 2. MySQL 记忆表：逐字段 + 读写路径

初始化位置：`MemoryManager._init_postgres()`（名称为历史遗留，实际连 MySQL）。

建表条件：
- 长期表：`enable_long_term and long_term_backend == "mysql"`
- 短期表：`short_term_backend == "mysql"`
- 失败：warning + 清空 `_mysql_conn_params`，降级 SQLite/内存

### 2.1 ER 关系图

```
┌────────────────────┐       ┌─────────────────────────┐
│   user_profiles    │       │     memory_entries      │
│ PK(tenant,user)    │       │ PK(id)                  │
│ profile JSON       │       │ tenant,user,thread      │
└─────────┬──────────┘       │ type, namespace          │
          │                  │ content/summary/meta    │
          │ 逻辑关联          └───────────┬─────────────┘
          │ (无 FK)                       │
          │                               │ 逻辑关联
          ▼                               ▼
┌────────────────────┐       ┌─────────────────────────┐
│ short_term_messages│       │ short_term_summaries    │
│ PK(id)             │       │ PK(tenant,user,thread)  │
│ tenant,user,thread │       │ summary TEXT            │
│ role, content      │       └─────────────────────────┘
└────────────────────┘
```

> 全部逻辑关联，**无外键约束**。隔离靠应用层 `tenant_id + user_id (+ thread_id)`。

---

### 2.2 表 `memory_entries`（长期记忆条目）

#### DDL（源码）

```sql
CREATE TABLE IF NOT EXISTS memory_entries (
    id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    thread_id VARCHAR(255),
    memory_type VARCHAR(50) NOT NULL,
    namespace VARCHAR(255),
    content JSON NOT NULL,
    summary TEXT,
    metadata JSON NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE INDEX idx_memory_entries_lookup
ON memory_entries (tenant_id, user_id, memory_type, created_at DESC);
-- 重复创建索引时 ignore
```

#### 字段字典

| 字段 | 类型 | 空 | 默认 | 含义 | 写入来源 |
|------|------|----|------|------|----------|
| id | VARCHAR(255) | NO | UUID | 记忆主键 | `uuid4()` / MemoryEntry.id |
| tenant_id | VARCHAR(255) | NO | - | 租户 | metadata.tenant_id / default_tenant |
| user_id | VARCHAR(255) | NO | - | 用户 | entry.user_id |
| thread_id | VARCHAR(255) | YES | NULL | 会话线程 | 可选；scope=thread 时过滤 |
| memory_type | VARCHAR(50) | NO | - | `semantic` / `episodic` 等 | MemoryType.value |
| namespace | VARCHAR(255) | YES | NULL | 逻辑分区 | `user_profile` / `facts/x` / `tasks/y` |
| content | JSON | NO | - | 结构化内容 | dict 直接 dump；str 包成 `{"text":...}` |
| summary | TEXT | YES | NULL | 检索摘要 | fact 前 500 字 / task outcome |
| metadata | JSON | NO | `{}` | 扩展元数据 | tenant/category/task_type 等 |
| created_at | TIMESTAMP | NO | CURRENT | 创建时间 | entry.created_at |
| updated_at | TIMESTAMP | NO | auto | 更新时间 | ON UPDATE / UPSERT |

#### 典型 content 形态

```json
// save_fact
{"text": "用户叫小明", "category": "user_fact"}

// save_task
{
  "task_type": "conversation",
  "task_data": {"query": "调研 AI Agent"},
  "outcome": "最终报告前 1200 字..."
}
```

#### 写入 SQL（UPSERT）

```sql
INSERT INTO memory_entries
(id, tenant_id, user_id, thread_id, memory_type, namespace,
 content, summary, metadata, created_at, updated_at)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
ON DUPLICATE KEY UPDATE
  content = VALUES(content),
  summary = VALUES(summary),
  metadata = VALUES(metadata),
  updated_at = NOW();
```

#### 检索 SQL（关键词降级路径）

```sql
SELECT id, memory_type, namespace, content, metadata, created_at
FROM memory_entries
WHERE tenant_id = %s
  AND user_id = %s
  AND memory_type = %s
  AND (summary LIKE %s OR CAST(content AS CHAR) LIKE %s)  -- query 非空时
  AND namespace = %s   -- 可选
  AND thread_id = %s   -- scope=thread 时
ORDER BY created_at DESC
LIMIT %s;
```

索引利用：
- 等值 `tenant_id, user_id, memory_type` + 时间排序 → `idx_memory_entries_lookup`
- `LIKE %xx%` 无法很好用 BTree，大数据量应更依赖 Milvus

---

### 2.3 表 `user_profiles`（用户画像）

#### DDL

```sql
CREATE TABLE IF NOT EXISTS user_profiles (
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    profile JSON NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, user_id)
);
```

#### 字段字典

| 字段 | 类型 | 含义 |
|------|------|------|
| tenant_id | VARCHAR(255) | 租户，联合主键左部 |
| user_id | VARCHAR(255) | 用户，联合主键右部 |
| profile | JSON | 画像对象（可含 preferences 列表等） |
| updated_at | TIMESTAMP | 最后更新 |

#### 典型 profile

```json
{
  "preferences": [
    "我喜欢简洁的回答",
    "我对 AI Agent 很感兴趣"
  ],
  "_last_updated": "2026-07-18T12:00:00"
}
```

#### UPSERT

```sql
INSERT INTO user_profiles (tenant_id, user_id, profile, updated_at)
VALUES (%s, %s, %s, NOW())
ON DUPLICATE KEY UPDATE
  profile = VALUES(profile),
  updated_at = NOW();
```

#### 读取

```sql
SELECT profile FROM user_profiles
WHERE tenant_id = %s AND user_id = %s;
```

写画像时若启用 Milvus，会同步把 `json.dumps(profile)` 索引进记忆向量库，namespace=`user_profile`。

---

### 2.4 表 `short_term_messages`（短期消息）

#### DDL

```sql
CREATE TABLE IF NOT EXISTS short_term_messages (
    id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    thread_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_short_term_lookup
ON short_term_messages (tenant_id, user_id, thread_id, created_at DESC);
```

#### 字段字典

| 字段 | 含义 | 取值 |
|------|------|------|
| id | 消息主键 | uuid4 |
| tenant_id / user_id / thread_id | 会话三维键 | 请求透传 |
| role | 角色 | `human` / `ai` / `system` |
| content | 文本 | 用户 query 或助手 final |
| created_at | 写入时间 | NOW() |

#### 写入（每条消息一行）

```sql
INSERT INTO short_term_messages
(id, tenant_id, user_id, thread_id, role, content, created_at)
VALUES (%s,%s,%s,%s,%s,%s,NOW());
```

`persist_turn` 每轮至少写 2 行：human + ai。

#### 顺序读取

```sql
SELECT role, content
FROM short_term_messages
WHERE tenant_id=%s AND user_id=%s AND thread_id=%s
ORDER BY created_at ASC;
```

#### 压缩流程（消息数 > max_messages）

```
history = 全部消息 ASC
split_at = len - summary_threshold
to_summarize = history[:split_at]
keep = history[split_at:]
new_summary = LLM/规则摘要(existing_summary, to_summarize)

DELETE FROM short_term_messages WHERE tenant/user/thread
逐条 INSERT keep
UPSERT short_term_summaries
```

---

### 2.5 表 `short_term_summaries`（会话摘要）

#### DDL

```sql
CREATE TABLE IF NOT EXISTS short_term_summaries (
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    thread_id VARCHAR(255) NOT NULL,
    summary TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, user_id, thread_id)
);
```

#### 字段字典

| 字段 | 含义 |
|------|------|
| tenant_id, user_id, thread_id | 联合主键 = 一个会话一份摘要 |
| summary | 压缩后的历史摘要（中文 100–300 字目标） |
| updated_at | 摘要更新时间 |

#### UPSERT

```sql
INSERT INTO short_term_summaries (tenant_id, user_id, thread_id, summary, updated_at)
VALUES (%s,%s,%s,%s,NOW())
ON DUPLICATE KEY UPDATE summary=VALUES(summary), updated_at=NOW();
```

读取后在 `get_short_term_messages(include_summary=True)` 时变为：

```
SystemMessage("历史对话摘要：{summary}") + 近期 Human/AI 消息
```

---

### 2.6 配置默认与连接串

```
MYSQL_DSN=mysql+pymysql://deepresearch:deepresearch123@mysql:3306/deepresearch?charset=utf8mb4
```

`_parse_mysql_dsn` 解析为：

```python
{
  "host": "mysql",
  "port": 3306,
  "user": "deepresearch",
  "password": "...",
  "database": "deepresearch",
  "charset": "utf8mb4",
}
```

### 2.7 运维常用 SQL

```sql
-- 看某用户长期记忆
SELECT id, memory_type, namespace, LEFT(summary,80), created_at
FROM memory_entries
WHERE tenant_id='default_tenant' AND user_id='user01'
ORDER BY created_at DESC LIMIT 20;

-- 看会话消息量
SELECT thread_id, COUNT(*) cnt
FROM short_term_messages
WHERE tenant_id='default_tenant' AND user_id='user01'
GROUP BY thread_id;

-- 看摘要
SELECT thread_id, LEFT(summary,120), updated_at
FROM short_term_summaries
WHERE tenant_id='default_tenant' AND user_id='user01';

-- 清某一会话短期
DELETE FROM short_term_messages
WHERE tenant_id=? AND user_id=? AND thread_id=?;
DELETE FROM short_term_summaries
WHERE tenant_id=? AND user_id=? AND thread_id=?;
```

### 2.8 与 Redis 短期键对照

| MySQL | Redis |
|-------|-------|
| short_term_messages 多行 | `ma:short:{t}:{u}:{th}` List |
| short_term_summaries 一行 | `ma:short:summary:{t}:{u}:{th}` String |
| 无 TTL 字段（靠业务清理） | `EXPIRE` = short_term_ttl_seconds |
| 压缩 = DELETE+INSERT | 压缩 = DEL list + RPUSH keep + SET summary |

---

## 3. SSE 抓包时间线（协议级）

### 3.1 传输层格式

服务端：

```
media_type = "text/event-stream"
每条: f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
```

首包由路由层直接发：

```json
{"type":"status","message":"任务已接收，正在初始化多智能体链路"}
```

后续由 `WorkflowService.stream_events` worker 线程 `emit`。

### 3.2 事件类型契约

| type | 何时 | 主要字段 | 前端处理 |
|------|------|----------|----------|
| status | 请求刚进入 | message | progress |
| phase | 每个图节点完成 | node, message | `[node] message` 进 progress |
| route | 全图结束后 | message | progress |
| final | 全图结束后 | query, user_id, thread_id, tenant_id, final | 渲染助手消息 |
| error | 异常 | message | 失败气泡 |
| __done__ | 内部结束符 | - | **不发给前端**（生成器 break） |

`phase.message` 映射见 `WorkflowService._node_message`。

### 3.3 场景 A：直接回答（direct）时间线

假设 query=`你是谁？`，耗时量级示意（非精确 benchmark）：

```
t+0.00s  Client POST /api/v1/research/stream
t+0.01s  <-- data: {"type":"status","message":"任务已接收，正在初始化多智能体链路"}

         [若首次请求，此处可能阻塞数秒做 _ensure_initialized]
t+0.05s  worker 开始 _run_sync_with_events
t+0.10s  build_personalized_prompt_context（记忆读）
t+0.40s  intent 节点完成
t+0.40s  <-- data: {"type":"phase","node":"intent","message":"Intent Router 正在识别问题意图"}
t+1.80s  direct_answer 节点完成（写入 final）
t+1.80s  <-- data: {"type":"phase","node":"direct_answer","message":"Direct Responder 正在快速作答"}
t+1.85s  persist_turn
t+1.90s  <-- data: {"type":"route","message":"已走直接回答路径"}
t+1.90s  <-- data: {"type":"final","query":"你是谁？","user_id":"user01","thread_id":"thread01","tenant_id":"default_tenant","final":"我是 DeepResearch 助手..."}
t+1.90s  worker emit __done__（仅内部）
t+1.91s  流关闭
```

**原始 SSE 字节流形态**：

```
data: {"type": "status", "message": "任务已接收，正在初始化多智能体链路"}

data: {"type": "phase", "node": "intent", "message": "Intent Router 正在识别问题意图"}

data: {"type": "phase", "node": "direct_answer", "message": "Direct Responder 正在快速作答"}

data: {"type": "route", "message": "已走直接回答路径"}

data: {"type": "final", "query": "你是谁？", "user_id": "user01", "thread_id": "thread01", "tenant_id": "default_tenant", "final": "我是 DeepResearch 助手，一个多智能体深度研究系统。"}

```

### 3.4 场景 B：单轮深度研究（multiagent，无补搜）

query=`帮我调查 2025 年 AI Agent 发展趋势`

```
t+0.0   status
t+0.5   phase intent
t+4.0   phase plan
t+4.0   ┌ fan-out 并行
        │ web_search ........ (Bocha × N + LLM)
        │ local_rag  ........ (Milvus × N + LLM)
t+12.0  phase web_search     ⎤ 顺序取决于谁先完成；
t+13.0  phase local_rag      ⎦ stream 按完成顺序推 phase
t+18.0  phase deep_dive
t+23.0  phase analyze         needs_more_research=false
t+35.0  phase write           final 长 Markdown
t+35.5  route "已走多智能体研究路径"
t+35.5  final { final: "# 2025年...\n## 参考资料\n..." }
```

**phase 顺序注意**：
- `web_search` 与 `local_rag` 的 phase **谁先谁后不固定**（并行）。
- 不会先看到 `deep_dive` 再看到检索 phase（fan-in 之后）。

示例片段：

```
data: {"type":"status","message":"任务已接收，正在初始化多智能体链路"}

data: {"type":"phase","node":"intent","message":"Intent Router 正在识别问题意图"}

data: {"type":"phase","node":"plan","message":"Planner 正在拆解问题"}

data: {"type":"phase","node":"local_rag","message":"Local Scout 正在检索本地知识库"}

data: {"type":"phase","node":"web_search","message":"Web Scout 正在检索网络证据"}

data: {"type":"phase","node":"deep_dive","message":"Evidence Judge 正在进行证据裁判"}

data: {"type":"phase","node":"analyze","message":"Analyst 正在生成结论"}

data: {"type":"phase","node":"write","message":"Writer 正在撰写最终报告"}

data: {"type":"route","message":"已走多智能体研究路径"}

data: {"type":"final","query":"帮我调查 2025 年 AI Agent 发展趋势","user_id":"user01","thread_id":"thread01","tenant_id":"default_tenant","final":"# 2025年 AI Agent 发展趋势\n\n## 核心摘要\n..."}
```

### 3.5 场景 C：多轮迭代（reflect 循环一次）

```
intent → plan → web∥local → deep_dive → analyze(needs_more=true)
  → reflect → web∥local → deep_dive → analyze(needs_more=false)
  → write → route → final
```

SSE phase 序列示例：

```
status
phase intent
phase plan
phase web_search
phase local_rag
phase deep_dive
phase analyze
phase reflect          ← 出现 reflect 即发生补搜
phase web_search       ← 第二轮
phase local_rag
phase deep_dive
phase analyze
phase write
route multiagent
final
```

对应 state 侧：
- 第一次 analyze 后 `iteration` 仍为 0
- reflect 后 `iteration=1`，`supplementary_queries` 生效
- 第二轮 source_id 前缀 `WEB2_*` / `LOC2_*`

### 3.6 场景 D：失败

```
status
phase intent
...
data: {"type":"error","message":"缺少 DASHSCOPE_API_KEY 配置..."}
流结束（finally __done__）
```

前端：
- catch → 删 status 气泡
- push assistant `请求失败：...`

### 3.7 场景 E：stream 未拿到 final 的兜底

```
_run_sync_with_events:
  for update in app.stream(...):
      从 node_output.final 收集
  if not final:
      result = app.invoke(state, config)   # 再跑一遍整图！
      final = result["final"]
```

含义：
- 正常 write/direct_answer 会带 final，不触发。
- 若 stream 丢字段或异常路径导致 final 空，可能 **双倍耗时** 再 invoke。
- 排障时若日志出现两次完整节点序列，检查是否命中该兜底。

### 3.8 前端拆包与粘包对照

#### 正常一包一事

```
chunk = 'data: {"type":"phase","node":"plan","message":"..."}\n\n'
buffer.split('\n\n') → [完整事件, '']
pop 剩余 '' 
parse 成功
```

#### 粘包（一包多事）

```
chunk = 'data: {...phase...}\n\ndata: {...phase...}\n\n'
split → 两条完整 + 残余
循环 parse 两条
```

#### 半包（一事多包）

```
chunk1 = 'data: {"type":"fi'
chunk2 = 'nal","final":"# 报告"}\n\n'
buffer 累积后再 split，避免 JSON.parse 失败
```

### 3.9 curl 抓包命令

```bash
curl -N -X POST 'http://localhost:8080/api/v1/research/stream' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "你是谁？",
    "user_id": "user01",
    "thread_id": "thread_sse_demo",
    "tenant_id": "default_tenant"
  }'
```

`-N` / `--no-buffer` 关闭 curl 输出缓冲，便于看流式到达。

经前端 Nginx：

```bash
curl -N -X POST 'http://localhost:3000/api/v1/research/stream' \
  -H 'Content-Type: application/json' \
  -d '{"query":"你是谁？","user_id":"user01","thread_id":"t1","tenant_id":"default_tenant"}'
```

### 3.10 时间线与日志对照表

| SSE 事件 | 后端大致日志关键字 |
|----------|-------------------|
| status | uvicorn access `/stream` |
| phase intent | `[intent] 开始` / `路由: direct\|multiagent` |
| phase plan | `[plan] 开始` |
| phase web_search | `[bocha_web_search]` / `[web_search_node]` |
| phase local_rag | `[local_rag]` / milvus |
| phase deep_dive | `[deep_dive]` |
| phase analyze | `[analyze]` |
| phase reflect | `[reflect]` |
| phase write | `[write]` / emit write 输出预览 |
| route | worker 结束后 emit |
| final | 同刻；随后 `[memory] turn persisted` |
| error | exception 栈 / ValueError 文案 |

### 3.11 线程时序图（stream）

```
Async event loop                     Worker Thread
      │                                    │
      │ start Thread(worker)               │
      │ await queue.get()                  │
      │                                    │ _ensure_initialized
      │                                    │ memory_context
      │                                    │ app.stream loop
      │◄──── phase intent ────────────────│ emit
      │ yield to client                    │
      │ await queue.get()                  │
      │◄──── phase ... ───────────────────│
      │                                    │ persist_turn
      │◄──── route ───────────────────────│
      │◄──── final ───────────────────────│
      │◄──── __done__ ────────────────────│
      │ break（不 yield __done__）          │
      ▼                                    ▼
```

---

## 4. 三块材料如何一起排障

```
1) 网页证据为空
   → curl Bocha，看 webPages 形态是否 A/B/C
   → 看 BOCHA_API_KEY
   → 看日志返回记录数

2) 记忆不生效
   → SHOW TABLES; 是否有四张表
   → short_term_messages 是否写入
   → tenant/user/thread 是否与前端一致
   → threadId 是否每次新建会话都变

3) 前端一直转圈 / 半截
   → curl -N 看 SSE 是否推到 final/error
   → 是否只有 status 无 phase（卡在初始化）
   → Nginx 300s 超时
   → 是否命中 stream 后再 invoke 兜底（双倍耗时）
```

---

## 5. 交叉索引

| 主题 | 文档 |
|------|------|
| web_search 节点流程 | 05 / 06 / 10 |
| 记忆 manager 总览 | 07 / 11 |
| SSE API 与前端 | 08 / 11 |
| 本篇样例与时间线 | **12（本文）** |
