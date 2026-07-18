# 06 - RAG 检索系统（完整流程与子图）

> 本文档对 `src/deep_research/rag/` 下每个文件、每条数据路径、每个算法子图进行事无巨细的叙述。  
> 所有图均为严谨 ASCII 流程图，可直接对照源码阅读。

---

## 0. 模块总览图

### 0.1 文件与职责

```
src/deep_research/rag/
├── __init__.py              # 统一导出 BM25 / RRF / Hybrid 公开 API
├── core.py                  # RAGSystem：向量存储 + 文本切分 + 入库 + 检索
├── bm25_retriever.py        # BM25+ 稀疏检索：分词 → 索引 → 评分 → 持久化
├── rrf_fusion.py            # RRF 融合：标准 / 加权 / 自适应 / 多阶段
├── hybrid_retriever.py      # 混合检索器：向量 ∥ BM25 → RRF → 结果
└── ingest.py                # 文档入库 CLI 脚本（文件收集 → 切分 → 向量化 → Milvus）
```

### 0.2 模块依赖关系图

```
                    ┌─────────────────────┐
                    │     ingest.py       │
                    │  (文档入库入口)      │
                    └──────────┬──────────┘
                               │ 依赖
                               ▼
                    ┌─────────────────────┐
                    │      core.py        │
                    │    RAGSystem        │
                    │  · DashScope Embed  │
                    │  · TextSplitter     │
                    │  · Milvus Store     │
                    └──────────┬──────────┘
                               │ 被 local_rag_node 调用
                               │ (search_knowledge_base_records)
                               ▼
              ┌────────────────────────────────┐
              │   agents/tools/knowledge.py    │
              │   agents/nodes/search.py       │
              └────────────────────────────────┘

独立可组合子图（当前主工作流以 core 向量检索为主，Hybrid 作为完整能力保留）：

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ bm25_retriever   │     │   rrf_fusion     │     │ hybrid_retriever │
│                  │◄────│                  │◄────│                  │
│ Document         │     │ RRFFusion        │     │ HybridRetriever  │
│ Tokenizer*       │     │ AdaptiveRRF      │     │ RetrievalCache   │
│ BM25Index        │     │ MultiStageFusion │     │ create_hybrid_*  │
│ BM25Retriever    │     │ ScoreNormalizer  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         ▲                         ▲                        │
         └─────────────────────────┴────────────────────────┘
                    hybrid 内部组合调用
```

### 0.3 主工作流中的 RAG 位置

```
用户问题
   │
   ▼
plan 节点 ──► search_plan / supplementary_queries
   │
   ├──────────────────────┬──────────────────────┐
   ▼                      ▼                      │
web_search            local_rag ◄── 本文档重点     │
(Bocha API)               │                      │
   │                      │                      │
   │              ┌───────┴────────┐             │
   │              │ RAG 检索子系统  │             │
   │              │                │             │
   │              │ query 列表     │             │
   │              │   │            │             │
   │              │   ▼            │             │
   │              │ Milvus 向量检索│             │
   │              │   │            │             │
   │              │   ▼            │             │
   │              │ list[record]   │             │
   │              │ LOC 前缀 ID    │             │
   │              └───────┬────────┘             │
   │                      │                      │
   └──────────┬───────────┘                      │
              ▼                                  │
         deep_dive ◄─────────────────────────────┘
              │
              ▼
           analyze → write
```

### 0.4 两种检索能力的关系（必须区分）

```
┌─────────────────────────────────────────────────────────────────┐
│ A. 主工作流实际路径（local_rag_node）                              │
│                                                                 │
│   search_knowledge_base_records(query)                          │
│       → RAGSystem.search_records(query, k)                      │
│           → Milvus.similarity_search                            │
│           → 标准化为 {source_id, doc_id, title, snippet, ...}  │
│                                                                 │
│   说明：当前线上主路径是「纯向量检索」，不走 HybridRetriever。     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ B. 模块完整能力路径（hybrid_retriever.py 提供）                   │
│                                                                 │
│   HybridRetriever.search(query)                                 │
│       → 并行：vector_search ∥ bm25_search                       │
│       → RRFFusion.fuse                                          │
│       → 融合排序结果                                            │
│                                                                 │
│   说明：这是完整混合检索实现，可供后续替换 local_rag 或独立调用。  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 整体 RAG 数据生命周期总图

```
═══════════════════════ 入库阶段（Offline / Ingest） ═══════════════════════

  磁盘文件(.md/.txt)                内存文本字符串
         │                                │
         ▼                                ▼
  ┌──────────────┐                 ┌──────────────┐
  │ _collect_paths│                │ ingest_text  │
  │ 文件收集/过滤  │                │ 直接入库入口  │
  └──────┬───────┘                 └──────┬───────┘
         │                                │
         └────────────┬───────────────────┘
                      ▼
             ┌─────────────────┐
             │  读取 UTF-8 文本  │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ RecursiveChar   │
             │ TextSplitter    │  chunk_size=500, overlap=50
             │ 递归分隔符切分   │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ List[Document]  │  page_content + metadata{source}
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ DashScope Embed │  text-embedding-v1 → 向量
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ Milvus 写入      │  collection, auto_id=True
             └────────┬────────┘
                      ▼
                 向量知识库就绪


═══════════════════════ 检索阶段（Online / Query） ════════════════════════

  用户查询 / search_plan 中的 query
                      │
                      ▼
             ┌─────────────────┐
             │ query 文本       │
             └────────┬────────┘
                      ▼
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│ 路径 A：核心向量  │       │ 路径 B：混合检索  │
│ RAGSystem       │       │ HybridRetriever │
│ similarity_     │       │ vector ∥ BM25   │
│ search          │       │ → RRF           │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
             ┌─────────────────┐
             │ 标准化 records   │
             │ source_id=LOC-* │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ local_rag_node  │
             │ LLM 证据整理     │
             └────────┬────────┘
                      ▼
                 local_evidence
```

---

## 2. 文件解析与入库全流程（ingest + core）

### 2.1 入口：ingest.py 总流程图

```
main()
  │
  ├─(1) logging.basicConfig
  │
  ├─(2) AppConfig.from_file()
  │      │
  │      ├─ 读 config.json
  │      ├─ 环境变量覆盖
  │      └─ 得到 api_key / milvus_* / collection
  │
  ├─(3) 组装 RAGConfig
  │      │
  │      collection_name = COLLECTION_NAME or config.milvus_collection
  │      milvus_host     = MILVUS_HOST or config.milvus_host
  │      milvus_port     = MILVUS_PORT or config.milvus_port
  │      embedding_model = "text-embedding-v1"
  │      chunk_size      = 500
  │      chunk_overlap   = 50
  │
  ├─(4) RAGSystem(api_key, rag_cfg)
  │      └─ 见 §3 初始化子图
  │
  ├─(5) INPUT_PATH.expanduser().resolve()
  │      │
  │      ├─ 不存在 → FileNotFoundError
  │      └─ 存在 → 继续
  │
  ├─(6) paths = _collect_paths(input_path)
  │      └─ 见 §2.2 文件收集子图
  │
  ├─(7) if not paths → ValueError("未找到可入库文件")
  │
  ├─(8) total_chunks = rag.ingest_paths(paths)
  │      └─ 见 §2.3 / §2.4
  │
  └─(9) print("入库完成 | 文件数=N | chunk数=M | collection=...")
```

### 2.2 文件收集子图：`_collect_paths`

```
_collect_paths(input_path: Path) → list[Path]
  │
  ├─ Case A: input_path.is_file() == True
  │     │
  │     └─ return [input_path]
  │        （单文件直接入库，不检查扩展名）
  │
  └─ Case B: input_path 是目录
        │
        ├─ patterns = ("*.txt", "*.md", "*.markdown")
        │
        ├─ for pat in patterns:
        │     paths.extend(sorted(input_path.rglob(pat)))
        │
        │   rglob 语义：
        │   ┌─────────────────────────────────────────┐
        │   │ knowledge_base/                         │
        │   │ ├── a.md          ← 命中 *.md           │
        │   │ ├── b.txt         ← 命中 *.txt          │
        │   │ ├── notes.markdown← 命中 *.markdown     │
        │   │ ├── c.pdf         ← 不命中，忽略         │
        │   │ └── sub/                                │
        │   │     └── d.md      ← 递归命中 *.md       │
        │   └─────────────────────────────────────────┘
        │
        └─ return paths  （已排序，顺序稳定）
```

**收集规则汇总**：

| 输入类型 | 行为 | 支持扩展名 |
|----------|------|------------|
| 单个文件 | 直接加入列表 | 任意（不校验后缀） |
| 目录 | `rglob` 递归扫描 | `.txt` / `.md` / `.markdown` |
| 空目录/无匹配 | 返回 `[]` → 上层抛 `ValueError` | - |

### 2.3 批量入库子图：`RAGSystem.ingest_paths`

```
ingest_paths(paths: Iterable[Path]) → int
  │
  total = 0
  │
  for path in paths:
  │   │
  │   ├─(1) text = path.read_text(encoding="utf-8")
  │   │       │
  │   │       ├─ 成功 → 得到完整字符串
  │   │       └─ 编码错误 → 抛出异常（当前实现不吞错）
  │   │
  │   ├─(2) n = ingest_text(text, source=str(path))
  │   │       └─ 见 §2.4
  │   │
  │   └─(3) total += n
  │
  return total   # 所有文件产生的 chunk 总数
```

**单文件数据形态变化**：

```
磁盘:
  /data/kb/agent.md   (UTF-8 bytes)
        │
        ▼ read_text
内存字符串:
  "# AI Agent\n\nLangChain 是...\n\n## 趋势\n..."
        │
        ▼ ingest_text(text, source="/data/kb/agent.md")
List[Document]:
  [
    Document(page_content="chunk0...", metadata={"source": "/data/kb/agent.md"}),
    Document(page_content="chunk1...", metadata={"source": "/data/kb/agent.md"}),
    ...
  ]
        │
        ▼ add_documents → Embedding → Milvus
持久化向量:
  collection 中 N 条向量记录，每条带 metadata.source
```

### 2.4 单文本入库子图：`ingest_text` + `add_documents`

```
ingest_text(text: str, source: str) → int
  │
  ├─(1) docs = text_splitter.create_documents(
  │         [text],
  │         metadatas=[{"source": source}]
  │     )
  │     │
  │     └─ 见 §2.5 递归切分子图
  │
  ├─(2) return add_documents(docs)
  │
add_documents(documents: list[Document]) → int
  │
  ├─(1) vectorstore.add_documents(documents)
  │     │
  │     ├─ 对每个 Document.page_content 调 Embedding API
  │     ├─ 得到向量 list[float]
  │     ├─ 写入 Milvus collection
  │     └─ auto_id=True → Milvus 自动分配主键
  │
  └─(2) return len(documents)
```

### 2.5 文本递归切分子图（核心算法）

配置：

```
chunk_size    = 500
chunk_overlap = 50
separators    = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
length_fn     = len   # 按字符数，不是 token 数
```

#### 2.5.1 分隔符优先级树

```
文本过长 (>500 字符)？
  │
  ├─ 是 → 尝试用当前分隔符切开
  │        │
  │        separators[0] = "\n\n"  （段落）
  │             │
  │             ├─ 切开后每段仍可能 >500
  │             │     └─ 对超长段递归，改用 separators[1]
  │             │
  │             separators[1] = "\n"   （行）
  │             separators[2] = "。"   （中文句号）
  │             separators[3] = "！"
  │             separators[4] = "？"
  │             separators[5] = "；"
  │             separators[6] = "，"
  │             separators[7] = " "    （空格）
  │             separators[8] = ""     （逐字符硬切，最后兜底）
  │
  └─ 否 → 该段作为一个 chunk 输出
```

#### 2.5.2 切分 + 重叠示意图

```
原始文本（简化，竖线为字符位置）:

0        50       100      ...     450      500      550      600
|---------|---------|-------...------|---------|---------|---------|
[============== chunk 0: [0, 500) ==============]
                                      [======== chunk 1: [450, 950) ========]
                                      ^-- overlap=50
                                                            [==== chunk 2 ...

规则：
1. 尽量在分隔符处断开，避免半句切断
2. 相邻 chunk 重叠 50 字符，保留边界语义
3. metadata 中每个 chunk 都复制同一 source
```

#### 2.5.3 完整切分决策伪流程

```
split(text, seps):
  if len(text) <= chunk_size:
      return [text]

  sep = seps[0]
  if sep == "":
      # 硬切：按 chunk_size 滑动窗口
      return hard_cut_with_overlap(text, chunk_size, chunk_overlap)

  parts = text.split(sep)
  chunks = []
  buffer = ""
  for part in parts:
      candidate = buffer + sep + part if buffer else part
      if len(candidate) <= chunk_size:
          buffer = candidate
      else:
          if buffer:
              chunks.append(buffer)
          if len(part) > chunk_size:
              # 当前 part 本身超长 → 递归用更细分隔符
              chunks.extend(split(part, seps[1:]))
              buffer = ""
          else:
              buffer = part
  if buffer:
      chunks.append(buffer)

  # 再根据 overlap 对相邻 chunk 做边界重叠拼接（由 splitter 内部完成）
  return apply_overlap(chunks, chunk_overlap)
```

### 2.6 Embedding 与写入 Milvus 子图

```
Document.page_content (str)
        │
        ▼
DashScopeEmbeddings.embed_documents([...])
        │
        │  model = text-embedding-v1
        │  API: DashScope Embedding
        │
        ▼
vector: List[float]   # 维度由模型决定（text-embedding-v1 常用 1536）
        │
        ▼
Milvus insert:
  ┌──────────────────────────────────────────────┐
  │ collection = mult_agent_knowledge / 配置名     │
  │ fields:                                      │
  │   · auto primary key                         │
  │   · vector                                   │
  │   · text / page_content                      │
  │   · metadata.source = 文件路径                 │
  └──────────────────────────────────────────────┘
        │
        ▼
写入成功 → 可被 similarity_search 召回
```

### 2.7 入库异常路径图

```
                    入库开始
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   路径不存在      目录无匹配文件    读文件编码失败
        │              │              │
        ▼              ▼              ▼
 FileNotFoundError  ValueError     UnicodeDecodeError
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
                 中断，不部分提交保证
                 （当前实现无事务回滚层；
                  已成功写入的 chunk 会留在
                  向量库中，需人工清理）

        Milvus 连接失败
               │
               ▼
        logger.error / 后续 search 返回 []
```

---

## 3. RAGSystem 初始化与检索（core.py）

### 3.1 初始化总图

```
RAGSystem.__init__(api_key, config)
  │
  ├─(1) self.config = config or RAGConfig()
  │
  ├─(2) self.embeddings = DashScopeEmbeddings(
  │         model=config.embedding_model,          # text-embedding-v1
  │         dashscope_api_key=api_key
  │     )
  │
  ├─(3) self.text_splitter = RecursiveCharacterTextSplitter(
  │         chunk_size=500,
  │         chunk_overlap=50,
  │         length_function=len,
  │         separators=["\n\n","\n","。","！","？","；","，"," ",""]
  │     )
  │
  ├─(4) _connect_to_milvus()
  │       │
  │       ├─ connections.connect(alias="default", host, port)
  │       ├─ 成功 → 连接就绪
  │       └─ 失败 → logger.error，不抛致命异常
  │
  └─(5) self.vectorstore = Milvus / langchain_milvus.Milvus(
            embedding_function=self.embeddings,
            collection_name=config.collection_name,
            connection_args={"uri": f"http://{host}:{port}"},
            auto_id=True
        )
```

### 3.2 langchain-milvus 后端选择子图

```
try:
  from langchain_milvus import Milvus as _MilvusVectorStore
  _MILVUS_BACKEND = "langchain_milvus"          # 优先新包
except ImportError:
  from langchain_community.vectorstores import Milvus as _MilvusVectorStore
  _MILVUS_BACKEND = "langchain_community"       # 降级旧包
```

### 3.3 在线检索子图：`search_records`

```
search_records(query: str, k: int = 5) → list[dict]
  │
  ├─(1) if not utility.has_collection(collection_name):
  │       return []     # 集合不存在，静默空结果
  │
  ├─(2) docs = vectorstore.similarity_search(query, k=k)
  │       │
  │       ├─ query → Embedding → query_vector
  │       ├─ Milvus ANN 检索 top-k
  │       └─ 返回 List[Document]
  │
  └─(3) 标准化循环:
        for idx, doc in enumerate(docs, 1):
          metadata = doc.metadata or {}
          source   = metadata.get("source") or ""
          title    = Path(source).name if source else f"本地知识片段-{idx}"
          records.append({
            "source_id": f"LOC-{idx}",     # 注意：此处是临时 ID
            "doc_id": source,              # 原文件路径
            "title": title,                # 文件名
            "snippet": doc.page_content,   # chunk 正文
            "source_type": "local",
            "metadata": metadata,
          })

说明：
  local_rag_node 会再次调用 _assign_source_ids，
  把 LOC-{idx} 重写为 LOC{iteration}_{query_index}-{n}
  例如：LOC1_2-3
```

### 3.4 `search`（面向人的字符串结果）子图

```
search(query, k=3) → str
  │
  ├─ records = search_records(query, k)
  │
  ├─ empty → "未找到相关信息。"
  │
  ├─ 有结果 → 拼接:
  │     "检索到的相关信息：\n"
  │     "1. {snippet}\n   (来源: {doc_id})\n"
  │     ...
  │
  └─ exception → "检索过程中发生错误: {exc}"
```

### 3.5 主工作流调用链（knowledge.py → core.py）

```
local_rag_node
  │
  │ queries = _build_queries(state, "local")
  │
  for item in queries:
  │   │
  │   search_knowledge_base_records(query, limit=4)
  │         │
  │         ├─ if _RAG_SYSTEM is None: return []
  │         └─ return _RAG_SYSTEM.search_records(query, k=4)
  │               │
  │               └─ core.RAGSystem.search_records
  │
  _assign_source_ids(records, f"LOC{iteration+1}_{query_index}")
  │
  _dedupe_sources / _minimal_record_filter
  │
  LLM scout_local 整理 evidence
  │
  写入 state["local_evidence"]
```

---

## 4. BM25 稀疏检索完整子图（bm25_retriever.py）

### 4.1 类关系图

```
                         ┌────────────────┐
                         │  BM25Config    │
                         └───────┬────────┘
                                 │ 配置
                                 ▼
┌──────────────┐         ┌────────────────┐         ┌─────────────┐
│ BaseTokenizer│◄────────│ BM25Retriever  │────────►│  BM25Index  │
└──────┬───────┘         └───────┬────────┘         └─────────────┘
       │                         │
       │ 实现                    │ 管理
       ▼                         ▼
┌──────────────┐         ┌────────────────┐
│SimpleTokenizer│         │ list[Document] │
│ChineseTokenizer│        │ doc_ids        │
│MixedTokenizer │         └────────────────┘
└──────────────┘
```

### 4.2 BM25 总数据流

```
文档入库（add_documents / add_texts）
  │
  ├─ Document(doc_id, content, metadata)
  ├─ tokenizer.tokenize(content) → tokens
  ├─ （可选）按 min_df / max_df 过滤词
  └─ BM25Index.build(tokenized_corpus)
        │
        ├─ 统计 doc_lengths / avgdl
        ├─ 统计 term_freqs（每文档 TF）
        ├─ 统计 doc_freqs（DF）
        └─ 计算 idf_cache

查询（search）
  │
  ├─ query_tokens = tokenizer.tokenize(query)
  ├─ scores = index.get_scores(query_tokens)
  ├─ （可选）query_weights 加权
  ├─ argsort 取 top_k
  └─ return [(Document, score), ...]
```

### 4.3 分词器选择子图

```
BM25Retriever._create_tokenizer()
  │
  tokenizer_type = config.tokenizer.lower()
  │
  ├─ "chinese"
  │     try: ChineseTokenizer(jieba)
  │     except ImportError: 回退 SimpleTokenizer
  │
  ├─ "mixed"
  │     try: MixedTokenizer(jieba + 英文正则)
  │     except ImportError: 回退 SimpleTokenizer
  │
  └─ 其他 / "simple"
        SimpleTokenizer
```

### 4.4 SimpleTokenizer 子图（含样例）

```
输入: "RAG 检索系统使用 LangChain 与 BM25，版本 2"
  │
  ├─ lowercase → "rag 检索系统使用 langchain 与 bm25，版本 2"
  │
  ├─ 英文正则 [a-zA-Z]+
  │     → ["rag", "langchain", "bm25"]
  │
  ├─ 中文连续段 [一-鿿]+
  │     → ["检索系统使用", "与", "版本"]
  │     对每个中文段:
  │       单字: 检,索,系,统,使,用
  │       双字滑动: 检索,索系,系统统,统使,使用
  │       三字滑动: 检索系,索系统,系统使,统使用
  │
  ├─ 数字 \d+
  │     → ["2"]
  │
  └─ _filter_tokens
        去掉停用词/空串
        → 最终 tokens 列表
```

### 4.5 ChineseTokenizer 子图

```
输入文本
  │
  ├─ lowercase（可选）
  │
  ├─ jieba.cut(text)   # 精确模式
  │     "检索系统使用LangChain"
  │       → ["检索", "系统", "使用", "LangChain"]  （示例）
  │
  ├─ strip 空白
  │
  └─ _filter_tokens(stopwords)
```

### 4.6 MixedTokenizer 子图

```
输入文本
  │
  ├─ lowercase
  │
  ├─ 拆分三路并行抽取:
  │     ├─ 英文 [a-zA-Z]+     → 完整英文词
  │     ├─ 中文段 → jieba 分词 → 中文词
  │     └─ 数字 \d+           → 数字 token
  │
  └─ 统一 _filter_tokens
```

### 4.7 停用词过滤子图

```
tokens = ["的", "RAG", "是", "检索", "系统", ""]
  │
  _filter_tokens:
  │  keep t if (t 非空) and (t not in stopwords)
  │
  → ["RAG", "检索", "系统"]

stopwords 来源:
  DEFAULT_CHINESE_STOPWORDS ∪ DEFAULT_ENGLISH_STOPWORDS
  或用户自定义 config.stopwords
  enable_stopwords=False 时 stopwords 视为空集
```

### 4.8 BM25Index.build 子图

```
tokenized_corpus = [
  ["rag", "检索", "系统"],          # doc0
  ["bm25", "检索", "算法"],         # doc1
  ["向量", "检索", "milvus"],       # doc2
]
  │
  ├─ doc_count = 3
  ├─ doc_lengths = [3, 3, 3]
  ├─ avgdl = 3.0
  │
  ├─ term_freqs:
  │     doc0: {rag:1, 检索:1, 系统:1}
  │     doc1: {bm25:1, 检索:1, 算法:1}
  │     doc2: {向量:1, 检索:1, milvus:1}
  │
  ├─ doc_freqs:
  │     检索: 3
  │     rag:1, 系统:1, bm25:1, 算法:1, 向量:1, milvus:1
  │
  └─ idf_cache[term] = log((N - df + 0.5)/(df + 0.5) + 1)
        且 max(idf, epsilon=0.25)
```

### 4.9 BM25 评分公式子图

```
对查询 Q = {q1, q2, ...}，文档 D：

                    f(qi, D) * (k1 + 1)
score += IDF(qi) * ─────────────────────────────────────────
                   f(qi, D) + k1 * (1 - b + b * |D|/avgdl)

再乘查询词权重因子: (1 + log(query_tf(qi)))

参数默认:
  k1 = 1.5   # 词频饱和
  b  = 0.75  # 长度归一
  epsilon = 0.25
```

### 4.10 get_scores 执行图

```
query_tokens = ["检索", "rag"]
  │
  scores = zeros(doc_count)
  │
  for term, qf in Counter(query_tokens):
  │   idf = get_idf(term)
  │   for each doc_idx:
  │     tf = term_freqs[doc_idx].get(term, 0)
  │     if tf == 0: continue
  │     numerator   = tf * (k1 + 1)
  │     denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
  │     score = idf * numerator / denominator
  │     score *= (1 + log(qf))
  │     scores[doc_idx] += score
  │
  （可选）scores *= doc_weights
  │
  return scores  # np.ndarray shape=(N,)
```

### 4.11 search / batch_search 子图

```
search(query, top_k=10, query_weights=None)
  │
  ├─ if not _built: return []
  ├─ tokens = tokenize(query)
  ├─ if not tokens: return []
  ├─ scores = index.get_scores(tokens)
  ├─ if query_weights: scores = _apply_query_weights(...)
  ├─ top_indices = argsort(scores)[::-1][:top_k]
  └─ return [(documents[i], scores[i]) for i in top_indices if scores[i] > 0]


batch_search(queries, top_k)
  │
  ├─ tokenized_queries = [tokenize(q) for q in queries]
  ├─ all_scores = index.get_batch_scores(...)  # shape=(Q, N)
  └─ 每个 query 独立 top_k
```

### 4.12 min_df / max_df 词过滤子图

```
_filter_by_df(tokenized_corpus)
  │
  ├─ 统计全局 DF
  ├─ max_df_count = int(max_df * n_docs)   # 默认 0.95N
  ├─ valid_terms = {t | min_df <= df(t) <= max_df_count}
  └─ 每个文档只保留 valid_terms 中的 token

效果:
  · 去掉极低频噪声词（min_df>1 时）
  · 去掉几乎每篇都出现的无区分度词（max_df）
```

### 4.13 持久化子图（save / load）

```
save(path)
  │
  pickle.dump({
    config,
    documents,
    doc_ids,
    index: {doc_count, avgdl, doc_lengths, doc_freqs, term_freqs, idf_cache}
  })

load(path)
  │
  pickle.load → 恢复上述字段
  _built = True
  重建 tokenizer（按 config）
```

---

## 5. RRF 融合完整子图（rrf_fusion.py）

### 5.1 类层次图

```
FusionStrategy (ABC)
  ├── RRFFusionStrategy          # 基于排名的 RRF
  └── WeightedScoreFusionStrategy# 归一化分数加权求和

RRFFusion
  ├── 持有 RRFFusionStrategy
  ├── fuse / fuse_with_strategy
  └── update_weights / set_k

AdaptiveRRFFusion(RRFFusion)
  └── fuse_adaptive + 历史反馈权重

MultiStageFusion
  ├── stage1_fusion: 组内 RRF
  └── stage2_fusion: 组间 RRF

工具:
  DocIdExtractor   # 统一抽取文档 ID
  ScoreNormalizer  # min-max / softmax / rank
```

### 5.2 标准 RRF 融合总图

```
输入:
  results_list = [
    vector: [(docA, 0.95), (docB, 0.82), (docC, 0.71)],
    bm25:   [(docB, 8.5),  (docD, 6.2),  (docA, 4.1)],
  ]
  weights = {vector: 1.0, bm25: 1.0}
  k = 60
  top_k = 5

步骤:
  1) 对每个检索器结果，按排名 rank=1..n
  2) 对每个文档累计:
        RRF += w / (k + rank)
  3) 按累计分降序
  4) 截断 top_k
  5) （可选）normalize / min_score_threshold 过滤
```

### 5.3 数值演算子图（必须看懂）

```
k=60, w=1

docA:
  vector rank=1 → 1/(60+1)=0.016393
  bm25   rank=3 → 1/(60+3)=0.015873
  sum = 0.032266

docB:
  vector rank=2 → 1/(60+2)=0.016129
  bm25   rank=1 → 1/(60+1)=0.016393
  sum = 0.032522

docC:
  vector rank=3 → 1/(60+3)=0.015873
  bm25   无     → 0
  sum = 0.015873

docD:
  vector 无     → 0
  bm25   rank=2 → 1/(60+2)=0.016129
  sum = 0.016129

排序: docB > docA > docD > docC
```

### 5.4 DocIdExtractor 决策树

```
get_doc_id(doc)
  │
  ├─ hasattr(doc, "doc_id") → str(doc.doc_id)          # BM25 Document
  │
  ├─ hasattr(doc, "metadata")
  │     ├─ metadata["source"] → 用 source
  │     ├─ metadata["doc_id"] → 用 doc_id
  │     └─ metadata["id"]     → 用 id
  │
  └─ fallback: "doc-{hash(content/page_content/str(doc))}"
```

### 5.5 WeightedScoreFusion 子图

```
对每个检索器结果:
  1) 对该路分数做 min-max 归一化到 [0,1]
  2) 乘以该路 weight
合并:
  3) 同 doc_id 分数相加
  4) 排序 top_k

与 RRF 区别:
  · RRF 只看排名，不看原始分尺度
  · Weighted 依赖分数可比性，必须先归一化
```

### 5.6 AdaptiveRRF 权重计算子图

```
对每个检索器结果列表 R:
  if R 为空:
      weight = 0.1
  else:
      quality ≈ min(1, len(R)/10) * (与最高分相关的启发式)
      weight = max(0.5, quality * 2)

全部 weight 再归一化，使 sum=1
→ 作为本轮 fuse 的动态 weights
```

### 5.7 MultiStageFusion 子图

```
retriever_groups = {
  "vector_group": [vec_results_1, vec_results_2],
  "bm25_group":   [bm25_results],
}

Stage 1（组内）:
  vector_group → RRF 融合 → fused_vector
  bm25_group  → 仅 1 路，直接使用

Stage 2（组间）:
  RRF([fused_vector, bm25_group]) → final top_k
```

---

## 6. 混合检索器完整子图（hybrid_retriever.py）

### 6.1 组件装配图

```
HybridRetriever
  │
  ├─ vectorstore ──────────────► Milvus / 任意 similarity_search_with_score
  ├─ bm25_retriever ───────────► BM25Retriever
  ├─ rrf_fusion ───────────────► RRFFusion / Adaptive / MultiStage
  └─ _cache? ──────────────────► RetrievalCache (可选)
```

### 6.2 search_with_details 总流程图

```
search_with_details(query, top_k, filters)
  │
  ├─ t0 = now()
  │
  ├─ cache hit?
  │     yes → return cached HybridSearchResult
  │     no  ↓
  │
  ├─ enable_parallel?
  │     yes → _parallel_search(query, filters)
  │     no  → vector then bm25 串行
  │
  │   得到:
  │     vector_results: List[(doc, score)]
  │     bm25_results:   List[(doc, score)]
  │
  ├─ fused = rrf_fusion.fuse(
  │       [vector_results, bm25_results],
  │       names=["vector","bm25"],
  │       top_k=top_k
  │   )
  │
  ├─ min_score_threshold 过滤
  │
  ├─ 组装 HybridSearchResult:
  │     fused_results / vector_results / bm25_results
  │     query / latency_ms / stats
  │
  ├─ cache set
  │
  └─ return result
```

### 6.3 并行检索子图

```
_parallel_search
  │
  ├─ 事件循环正在运行？
  │     yes → ThreadPoolExecutor(max_workers=2)
  │            ├─ future_vector = submit(_vector_search)
  │            └─ future_bm25   = submit(_bm25_search)
  │            wait both
  │
  ├─ 事件循环未运行？
  │     → run_in_executor + asyncio.gather
  │
  └─ RuntimeError 回退
        → 串行 _vector_search; _bm25_search
```

时间线：

```
串行:  |---- vector ----|---- bm25 ----|
并行:  |---- vector ----|
       |---- bm25 ------|
总耗时 ≈ max(vector, bm25)
```

### 6.4 向量检索子图

```
_vector_search(query, filters)
  │
  ├─ cache get("vector")
  │
  ├─ vectorstore.similarity_search_with_score(query, k=vector_top_k, filter?)
  │     返回 (doc, distance)
  │
  ├─ distance → similarity
  │     similarity = 1 - score   (score>=0 时)
  │     # 假设距离越小越相似（余弦距离场景）
  │
  ├─ cache set
  └─ return [(doc, similarity), ...]

失败 → logger.error → []
```

### 6.5 BM25 检索子图

```
_bm25_search(query)
  │
  ├─ cache get("bm25")
  ├─ bm25_retriever.search(query, top_k=bm25_top_k)
  ├─ cache set
  └─ return [(Document, bm25_score), ...]

失败 → []
```

### 6.6 统计信息子图

```
_compute_stats(vector_results, bm25_results, fused)
  │
  vector_ids = {doc_id}
  bm25_ids   = {doc_id}
  overlap    = vector_ids ∩ bm25_ids

返回:
  {
    vector_count,
    bm25_count,
    fused_count,
    overlap_count,
    overlap_ratio = |overlap| / min(|V|,|B|)
  }
```

### 6.7 缓存键设计子图

```
RetrievalCache
  key = f"{retriever_type}:{normalize(query)}"
  normalize = lower + 合并空白

示例:
  hybrid:什么是 rag
  vector:什么是 rag
  bm25:什么是 rag

TTL 过期:
  now - timestamp >= ttl → 删除并视为未命中
```

### 6.8 工厂函数装配图

```
create_hybrid_retriever(vectorstore, top_k, vector_top_k, bm25_top_k,
                        k1, b, rrf_k, vector_weight, bm25_weight, ...)
  │
  ├─ BM25Config(k1, b)
  ├─ RRFConfig(k=rrf_k, weights={vector, bm25})
  ├─ HybridRetrieverConfig(...)
  └─ return HybridRetriever(vectorstore, config)
```

---

## 7. 主工作流中的 local_rag 证据链路（跨模块）

### 7.1 从 query 到 local_evidence 全图

```
state.search_plan / supplementary_queries
  │
  ▼
_build_queries(state, "local")
  │  过滤 source_preference in {local, hybrid}
  │  最多 6 条
  ▼
for query_index, item in queries:
  │
  │  records = search_knowledge_base_records(item.query, limit=4)
  │     └─ RAGSystem.search_records → Milvus top-4 chunks
  │
  │  records = _assign_source_ids(records, f"LOC{iter+1}_{query_index}")
  │     LOC1_1-1, LOC1_1-2, ...
  │
  │  附加 section_id / search_query
  │  写入 local_rag_trace 原始摘要
  │
  ▼
合并 raw_records
  │
  ▼
_dedupe_sources(raw, ["doc_id","snippet"])
_minimal_record_filter(raw, ["snippet","title","doc_id"])
  │
  ▼
无记录？
  yes → 返回提示字符串，保留已有 local_evidence
  no  ↓
LLM scout_local 输出 JSON evidence
  │
  ▼
_prune_evidence_to_allowed_sources  # 禁止幻觉 source_id
  │
  ▼
state.local_evidence = existing + new
state.local_retrieval_stats 更新
state.local_rag_trace finalize(kept/rejected)
```

### 7.2 source_id 生命周期图

```
Milvus 返回阶段:
  source_id = "LOC-1" / "LOC-2"   （临时，按本次 top-k 序号）

local_rag_node 重写:
  source_id = "LOC{iteration}_{plan_step}-{n}"
  例: 第1轮第2个查询第3条 → LOC1_2-3

deep_dive:
  进入 evidence_pool / source_index，保持同一 ID

write:
  正文引用 [LOC1_2-3]
  末尾参考资料按 ID 回查 label/locator
```

### 7.3 与 web_search 的对称结构图

```
            plan / reflect
                 │
        ┌────────┴────────┐
        ▼                 ▼
   web_search          local_rag
        │                 │
   Bocha API           Milvus RAG
        │                 │
   WEB{i}_{q}-{n}      LOC{i}_{q}-{n}
        │                 │
        └────────┬────────┘
                 ▼
             deep_dive
           evidence_pool
```

---

## 8. 配置参数全表（RAG 相关）

### 8.1 RAGConfig

| 字段 | 默认值 | 含义 |
|------|--------|------|
| milvus_host | 127.0.0.1 | 向量库主机 |
| milvus_port | 19530 | 向量库端口 |
| collection_name | mult_agent_knowledge | 知识库集合名（core 默认） |
| embedding_model | text-embedding-v1 | Embedding 模型 |
| chunk_size | 500 | 切分块大小（字符） |
| chunk_overlap | 50 | 块重叠（字符） |

> 注意：记忆系统的 Milvus collection 默认名是 `mult_agent_memory`，与知识库 collection 不同。

### 8.2 BM25Config

| 字段 | 默认值 | 含义 |
|------|--------|------|
| tokenizer | chinese | simple/chinese/mixed |
| k1 | 1.5 | TF 饱和 |
| b | 0.75 | 长度归一 |
| epsilon | 0.25 | IDF 下限 |
| lowercase | true | 小写化 |
| enable_stopwords | true | 停用词 |
| min_df | 1 | 最小文档频 |
| max_df | 0.95 | 最大文档频比例 |

### 8.3 HybridRetrieverConfig / RRFConfig

| 字段 | 默认值 | 含义 |
|------|--------|------|
| top_k | 10 | 最终返回数 |
| vector_top_k | 20 | 向量候选 |
| bm25_top_k | 20 | BM25 候选 |
| enable_parallel | true | 并行检索 |
| enable_cache | false | 结果缓存 |
| cache_ttl | 3600 | 缓存秒 |
| fusion_strategy | rrf | rrf/adaptive/multistage |
| rrf.k | 60 | RRF 平滑 |
| weights.vector/bm25 | 1.0/1.0 | 路权 |

---

## 9. 端到端示例（文件 → 检索 → 证据）

### 9.1 入库示例

```
输入文件: /kb/ai_agent.md
内容摘要: "2025年 AI Agent 将强化工具调用与多Agent协作..."

切分:
  chunk0 (0-500)
  chunk1 (450-950)   # overlap 50
  ...

写入 Milvus:
  每条 metadata.source = "/kb/ai_agent.md"
```

### 9.2 查询示例

```
query = "2025 AI Agent 发展趋势"

RAGSystem.search_records(query, k=4)
  →
  [
    {source_id:"LOC-1", doc_id:"/kb/ai_agent.md", title:"ai_agent.md",
     snippet:"2025年 AI Agent 将强化工具调用...", source_type:"local"},
    ...
  ]

local_rag_node 重编号后:
  LOC1_1-1, LOC1_1-2, ...

LLM 过滤后进入 local_evidence
  → deep_dive 评分（local 默认约 0.92）
  → write 引用 [LOC1_1-1]
```

### 9.3 若走 Hybrid 路径（模块能力）

```
query
  ├─ vector top20
  ├─ bm25 top20
  └─ RRF(k=60) → top10
        同时出现在两路的文档通常排名上升（见 §5.3）
```

---

## 10. 错误、降级与边界条件图

```
                    查询进入
                       │
        ┌──────────────┼────────────────┐
        ▼              ▼                ▼
   RAG 未初始化    collection 不存在   Embedding/Milvus 异常
        │              │                │
        ▼              ▼                ▼
   return []        return []      logger + 空结果/错误字符串
        │              │                │
        └──────────────┴────────────────┘
                       │
                       ▼
              local_rag_node 看到空 raw_records
                       │
                       ▼
         "未检索到可用本地知识库证据，已跳过本地上下文注入。"
         不阻断 web_search / deep_dive（可仅用网页证据继续）


BM25 侧边界:
  · 索引未 build → search 返回 []
  · 查询分词后为空 → []
  · jieba 缺失 → 回退 SimpleTokenizer


RRF 侧边界:
  · results_list 全空 → []
  · 某路为空 → 该路不贡献分数
  · k <= 0 → 配置阶段 ValueError
```

---

## 11. 源码阅读地图（按调用顺序）

```
入库:
  ingest.py::main
    → _collect_paths
    → RAGSystem.ingest_paths
      → ingest_text
        → RecursiveCharacterTextSplitter.create_documents
        → add_documents
          → Milvus.add_documents
          → DashScopeEmbeddings

在线（主工作流）:
  agents/nodes/search.py::local_rag_node
    → agents/tools/knowledge.py::search_knowledge_base_records
      → rag/core.py::RAGSystem.search_records
        → Milvus.similarity_search
    → base._assign_source_ids / _dedupe / LLM 整理

完整混合能力（可选）:
  hybrid_retriever.py::HybridRetriever.search_with_details
    → _vector_search
    → _bm25_search (bm25_retriever.py)
    → rrf_fusion.py::RRFFusion.fuse
```

---

## 12. 本章小结

1. **文件解析入库**：路径收集 → UTF-8 读取 → 递归分隔符切分（500/50）→ Embedding → Milvus。  
2. **主工作流检索**：当前 `local_rag` 走 `RAGSystem` 纯向量检索，再由节点重写 `LOC{i}_{q}-{n}` 并做 LLM 证据筛选。  
3. **完整混合能力**：`HybridRetriever` 提供向量∥BM25 + RRF 的完整图，含并行、缓存、自适应融合。  
4. **所有关键子图**（切分、分词、BM25 建索引、RRF 数值、并行检索、source_id 生命周期、异常降级）均已在上文单独画出，可直接对照源码逐框验证。
