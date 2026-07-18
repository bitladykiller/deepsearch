# 04 - Agent 系统

## 1. 概述

Deep Research 包含 8 个专业 Agent，每个 Agent 由以下组件构成：

```
Agent = LLM (ChatTongyi) + System Prompt + Temperature + Tools
```

所有 Agent 通过 `AgentBundle` 数据类统一管理，由 `build_agents` 工厂函数创建。

### 1.1 Agent 与 Node 的关系

Agent 和 Node 是两个不同层次的概念：

| 概念 | 职责 | 定义位置 |
|------|------|----------|
| Agent | LLM 实例 + System Prompt + 工具 | `builder.py` |
| Node | 工作流节点逻辑（调用 Agent + 处理输入输出） | `nodes/*.py` |

一个 Node 可以使用任意 Agent。例如：
- `plan_node` 使用 `planner` Agent
- `reflect_node` 也使用 `planner` Agent
- `intent_node` 使用 `intent_router` Agent
- `direct_answer_node` 使用 `direct_responder` Agent

---

## 2. AgentBundle 数据类

### 2.1 定义

```python
@dataclass(frozen=True)
class AgentBundle:
    """所有 Agent 实例的容器。"""
    intent_router: any       # 意图路由器
    planner: any             # 任务规划师
    scout_web: any           # 网络搜索员
    scout_local: any         # 本地知识库搜索员
    evidence_judge: any      # 证据裁判
    analyst: any             # 分析师
    direct_responder: any    # 直接回答者
    writer: any              # 报告撰写者
```

### 2.2 字段类型

所有字段类型为 `any`，实际运行时是 LangChain 的 AgentExecutor 实例。这是 `create_agent` 函数的返回值。

---

## 3. 8 个 Agent 详解

### 3.1 intent_router（意图路由器）

| 属性 | 值 |
|------|-----|
| Prompt Key | `intent_router` |
| Temperature | 0.0（完全确定性） |
| 工具 | 无 |
| 输入 | 用户原始问题 |
| 输出 | JSON：`{"route":"direct|multiagent","reason":"..."}` |

**System Prompt**：
```
你是 IntentRouter，负责把用户问题路由到 direct 或 multiagent。
你必须只输出 JSON，格式固定为：
{"route":"direct|multiagent","reason":"..."}

判断标准：
1) 问候、自我介绍、简单问答（如"你是谁""今天天气如何"）=> direct
2) 需要检索、多来源证据、分析、对比、报告 => multiagent
```

**设计要点**：
- Temperature=0.0 确保路由结果确定性，避免同一问题被不同路由
- 实际执行时还有规则引擎 `detect_intent()` 做初判，LLM 做二次确认
- 如果 LLM 输出格式异常，回退到规则引擎结果

### 3.2 planner（任务规划师）

| 属性 | 值 |
|------|-----|
| Prompt Key | `plan` |
| Temperature | 0.3（略带创造性） |
| 工具 | 无 |
| 输入 | 用户问题 |
| 输出 | JSON：包含 objective、sub_questions、outline、budget |

**System Prompt 核心要求**：
```json
{
  "objective": "...",
  "sub_questions": ["核心问题", "扩展子问题1", "扩展子问题2"],
  "outline": [
    {
      "id": "sec_1",
      "title": "...",
      "description": "...",
      "section_type": "mixed",
      "requires_data": true,
      "requires_chart": false,
      "priority": 1,
      "search_queries": ["检索词1", "检索词2"],
      "status": "pending"
    }
  ],
  "budget": {
    "max_rounds": 2,
    "max_sources": 12,
    "max_tokens": 12000,
    "max_seconds": 45
  }
}
```

**设计要点**：
- sub_questions 必须包含 1 个核心原问题 + 2-3 个扩展子问题
- search_queries 是针对子问题的自然语言检索词
- outline 定义报告大纲结构，每个章节有独立的搜索词
- Temperature=0.3 允许一定程度的创造性拆解

### 3.3 scout_web（网络搜索员）

| 属性 | 值 |
|------|-----|
| Prompt Key | `web_search` |
| Temperature | 0.4（中等创造性） |
| 工具 | 无 |
| 输入 | 用户问题 + 子问题列表 + 原始网页证据 |
| 输出 | JSON：包含 summary、evidence、gaps、rejected_source_ids |

**System Prompt 核心职责**：
- 判断每条证据是否与原问题或子问题相关
- 保留包含核心实体有效信息的证据
- 丢弃明显无关或广告内容
- evidence 中只能出现输入里存在的 source_id
- 不能编造来源

**证据结构**：
```json
{
  "source_id": "WEB-1",
  "title": "...",
  "url": "...",
  "snippet": "...",
  "domain": "...",
  "source_type": "web",
  "reliability_hint": "official|media|community|unknown",
  "supports_questions": ["问题1"],
  "notes": "..."
}
```

### 3.4 scout_local（本地知识库搜索员）

| 属性 | 值 |
|------|-----|
| Prompt Key | `local_rag` |
| Temperature | 0.4 |
| 工具 | 无 |
| 输入 | 用户问题 + 子问题列表 + 知识库检索结果 |
| 输出 | JSON：与 scout_web 结构类似，但 source_type 为 "local" |

**与 scout_web 的区别**：
- 来源是本地知识库（Milvus 向量检索），不是网页
- 每条证据有 `doc_id` 字段而非 `url`
- `reliability_hint` 固定为 `"internal"`
- 不存在域名相关性判断

### 3.5 evidence_judge（证据裁判）

| 属性 | 值 |
|------|-----|
| Prompt Key | `deep_dive` |
| Temperature | 0.2（高度确定性） |
| 工具 | 无 |
| 输入 | web_evidence + local_evidence + sub_questions |
| 输出 | JSON：包含 evidence_pool、audit_flags、source_index |

**System Prompt 核心职责**：
- 对所有证据评分（reliability_score）
- 去重（合并重复来源）
- 冲突审计（标记互相矛盾的证据）
- 本地知识库和官方站点优先高分
- 自媒体和论坛低分
- 冲突必须显式标记

**审计标记类型**：
```json
{
  "type": "low_confidence|conflict|missing_evidence",
  "target": "问题1",
  "reason": "..."
}
```

### 3.6 analyst（分析师）

| 属性 | 值 |
|------|-----|
| Prompt Key | `analyze` |
| Temperature | 0.3 |
| 工具 | 无 |
| 输入 | 证据池 + 审计标记 + 子问题 |
| 输出 | JSON：包含 findings、claim_map、needs_more_research、missing_gaps |

**System Prompt 核心职责**：
- 从证据池中形成结论
- 评估证据是否足够回答所有子问题
- 如果不足，指出 missing_gaps 并设置 needs_more_research=true
- 每个结论必须绑定来源 source_id

**结论结构**：
```json
{
  "claim_id": "c_1",
  "claim": "结论文本",
  "confidence": "high|medium|low",
  "source_ids": ["WEB1_1-1", "LOC1_1-2"]
}
```

### 3.7 direct_responder（直接回答者）

| 属性 | 值 |
|------|-----|
| Prompt Key | `direct_answer` |
| Temperature | 0.2 |
| 工具 | 无 |
| 输入 | 用户问题 |
| 输出 | 自然语言回答（非 JSON） |

**System Prompt**：
```
你是 DeepResearch 助手。当问题是简单问答或闲聊时，直接回答用户，
不要走研究报告结构。要求：简洁、自然、准确。
如果用户问天气但未提供城市，请先提示补充城市。
```

**设计要点**：
- 这是唯一直接输出自然语言的 Agent
- 不需要 JSON 解析
- 不经过证据裁判和分析流程
- 直接将输出写入 `state["final"]`

### 3.8 writer（报告撰写者）

| 属性 | 值 |
|------|-----|
| Prompt Key | `write` |
| Temperature | 0.4（中等创造性） |
| 工具 | 无 |
| 输入 | 问题 + 子问题 + findings + source_index + audit_flags |
| 输出 | Markdown 格式研究报告（非 JSON） |

**System Prompt 核心要求**：
- 报告至少 2000-3000 字以上
- 包含标题、核心摘要、详细分析、总结与展望
- 正文中使用上标引用（如 `[WEB1_1-1]`）
- 禁止输出 JSON 格式
- 禁止编造引用序号
- 结尾不需要列举引用列表（系统自动拼接）

**报告结构**：
```markdown
# 标题

## 核心摘要
（200 字左右）

## 详细分析
（主体部分，极其详实，每个 finding 展开为长篇段落）

## 总结与展望
（深度洞见 + 风险提示）

## 参考资料
（系统自动拼接，不需要 writer 生成）
```

---

## 4. Agent 构建流程

### 4.1 build_agent 函数

```python
def build_agent(model: str, api_key: str, prompt_key: str, temperature: float, tools: list):
    """构建单个 Agent 实例。"""
    # 1. 设置环境变量（DashScope 需要）
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    
    # 2. 创建 LLM 实例
    llm = ChatTongyi(model=model, temperature=temperature)
    
    # 3. 获取 System Prompt
    prompt = PROMPTS[prompt_key]
    
    # 4. 创建 Agent
    return create_agent(model=llm, tools=tools, system_prompt=prompt)
```

### 4.2 build_agents 函数

```python
def build_agents(model: str, api_key: str, config: AppConfig) -> AgentBundle:
    """构建所有 Agent 实例。"""
    # 1. 初始化 RAG 系统（全局单例）
    rag_config = RAGConfig(
        milvus_host=config.milvus_host,
        milvus_port=config.milvus_port,
        collection_name=config.milvus_collection,
    )
    init_rag_system(api_key=api_key, config=rag_config)
    
    # 2. 构建 8 个 Agent（都不绑定 tools）
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
```

### 4.3 为什么 Agent 不绑定 Tools

```python
# 每个 Agent 的 tools 参数都是空列表 []
build_agent(model, api_key, "intent_router", 0.0, [])
```

这是一个重要的设计决策。原因：

1. **降低 System Prompt 长度**：绑定 tools 会在 System Prompt 中注入工具描述，增加 token 消耗
2. **节点逻辑控制工具调用**：工具调用（如 `bocha_web_search_records`）由节点函数直接调用，而非通过 Agent 的 Function Calling
3. **避免 LLM 幻觉调用**：不让 LLM 自己决定调用什么工具，而是由代码精确控制
4. **结果可预测性**：每个节点的工具调用是确定性的，不受 LLM 输出影响

---

## 5. System Prompt 设计原则

### 5.1 所有 Prompt 都要求 JSON 输出

除 `direct_answer` 和 `write` 外，所有 Agent 的 System Prompt 都包含：

```
你必须只输出 JSON，不要输出 markdown，不要补充解释。
```

这是因为：
- JSON 结构化输出便于程序解析
- 避免 LLM 输出冗长的解释文本
- 字段名固定，便于下游节点读取

### 5.2 强制 JSON Schema

每个 JSON 输出的 Prompt 都定义了完整的字段结构，例如：

```
JSON 结构固定为：
{
  "objective": "...",
  "sub_questions": ["..."],
  "outline": [...],
  "budget": {...}
}
```

这样 LLM 输出的 JSON 格式相对稳定，减少解析失败的概率。

### 5.3 容错设计

即使 LLM 输出格式异常，系统也有兜底机制：

```python
payload, content, messages = _invoke_json_agent(
    state, prompt, agent, agent_name, "plan", fallback,
)
```

`fallback` 参数是一个预定义的默认 JSON，当 LLM 输出无法解析时使用。

---

## 6. LLM 后端：DashScope ChatTongyi

### 6.1 技术栈

```python
from langchain_community.chat_models import ChatTongyi

llm = ChatTongyi(model="qwen-plus", temperature=0.3)
```

ChatTongyi 是 LangChain 对阿里云 DashScope 通义千问模型的封装。

### 6.2 模型选择

| 模型 | 用途 | 特点 |
|------|------|------|
| `qwen-plus` | 默认模型 | 平衡性能和成本 |
| `qwen-turbo` | 低成本场景 | 更快但能力稍弱 |
| `qwen-max` | 高质量需求 | 最强但成本最高 |

### 6.3 API Key 管理

```python
# 方式 1：环境变量（推荐）
os.environ["DASHSCOPE_API_KEY"] = api_key

# 方式 2：构造函数参数
llm = ChatTongyi(model="qwen-plus", dashscope_api_key=api_key)
```

系统使用方式 1，在 `build_agent` 中通过环境变量注入 API Key。

---

## 7. Prompts 模块完整清单

### 7.1 核心研究 Prompt

| Prompt Key | 角色 | 输出格式 | 使用节点 |
|------------|------|----------|----------|
| `intent_router` | IntentRouter | JSON | intent |
| `plan` | ChiefArchitect | JSON | plan, reflect |
| `web_search` | WebScout | JSON | web_search |
| `local_rag` | LocalRAGScout | JSON | local_rag |
| `deep_dive` | EvidenceJudge | JSON | deep_dive |
| `analyze` | Analyst | JSON | analyze |
| `reflect` | ResearchPlanner | JSON | reflect |
| `write` | Writer | Markdown | write |
| `direct_answer` | DirectResponder | 自然语言 | direct_answer |

### 7.2 辅助 Agent Prompt（未在主流程中使用）

| Prompt Key | 角色 | 用途 |
|------------|------|------|
| `codegen` | CodeWizard | 代码骨架生成 |
| `rag_agent` | 知识库检索专家 | 独立知识库查询 |
| `python_agent` | 数据科学专家 | Python 代码执行 |
| `amap_agent` | 地理位置专家 | 高德地图 API |
| `file_agent` | 文件管理专家 | 安全文件操作 |
| `sql_agent` | 数据库专家 | SQL 操作 |
| `terminal_agent` | 终端专家 | 安全命令执行 |
| `web_search_agent` | 网络检索专家 | 独立网络搜索 |

这些辅助 Agent Prompt 定义了更多角色，但在当前主工作流中未被使用，可能用于扩展场景。
