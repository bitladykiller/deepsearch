# 06 - RAG 检索系统

## 1. 概述

RAG（Retrieval-Augmented Generation，检索增强生成）系统是 Deep Research 的本地知识库检索能力核心。它整合了三种检索技术：

```
                   用户查询
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
   ┌───────────────┐    ┌───────────────┐
   │  向量检索      │    │  BM25 检索     │
   │  (Milvus)      │    │  (关键词匹配)   │
   │                │    │                │
   │  语义相似度     │    │  词频统计       │
   │  模糊匹配      │    │  精确匹配       │
   │  同义词理解     │    │  专有名词       │
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

### 1.1 三种检索技术的互补关系

| 特性 | 向量检索 | BM25 检索 | RRF 融合 |
|------|---------|----------|----------|
| 匹配方式 | 语义相似 | 关键词匹配 | 排名融合 |
| 优势 | 模糊匹配、同义词、跨语言 | 精确匹配、专有名词、低频词 | 取长补短 |
| 劣势 | 可能忽略精确关键词 | 无法理解同义词 | - |
| 索引类型 | 稠密向量 | 稀疏倒排索引 | - |
| 存储 | Milvus | 内存 | - |

---

## 2. RAGSystem 核心类（core.py）

### 2.1 RAGConfig 配置

```python
@dataclass(frozen=True)
class RAGConfig:
    milvus_host: str = "127.0.0.1"      # Milvus 主机
    milvus_port: int = 19530             # Milvus 端口
    collection_name: str = "mult_agent_knowledge"  # 集合名
    embedding_model: str = "text-embedding-v1"     # Embedding 模型
    chunk_size: int = 500                # 文本分块大小
    chunk_overlap: int = 50              # 分块重叠大小
```

### 2.2 RAGSystem 初始化流程

```python
class RAGSystem:
    def __init__(self, api_key: str, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.api_key = api_key
        
        # 1. 创建 Embedding 模型
        self.embeddings = DashScopeEmbeddings(
            model=self.config.embedding_model,
            dashscope_api_key=self.api_key,
        )
        
        # 2. 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,     # 每块 500 字符
            chunk_overlap=self.config.chunk_overlap, # 重叠 50 字符
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        
        # 3. 连接 Milvus
        self._connect_to_milvus()
        
        # 4. 创建向量存储
        self.vectorstore = MilvusVectorStore(
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
            connection_args={"uri": f"http://{self.config.milvus_host}:{self.config.milvus_port}"},
            auto_id=True,  # Milvus 自动生成 ID
        )
```

### 2.3 Embedding 模型

使用 DashScope 的 `text-embedding-v1` 模型：

| 属性 | 值 |
|------|-----|
| 维度 | 1536 |
| 支持语言 | 中文、英文 |
| 最大输入 | 2048 tokens |
| 提供商 | 阿里云 DashScope |

### 2.4 文本分割器

`RecursiveCharacterTextSplitter` 的分割策略：

```
优先使用分隔符的顺序：
1. \n\n（段落分隔）
2. \n（换行分隔）
3. 。（中文句号）
4. ！（中文感叹号）
5. ？（中文问号）
6. ；（中文分号）
7. ，（中文逗号）
8. （空格）
9. ""（字符级分割）
```

分块参数：
- `chunk_size=500`：每个分块最多 500 个字符
- `chunk_overlap=50`：相邻分块重叠 50 个字符（避免语义断裂）

### 2.5 核心方法

#### search_records（结构化检索）

```python
def search_records(self, query: str, k: int = 5) -> list[dict]:
    # 1. 检查集合是否存在
    if not utility.has_collection(self.config.collection_name):
        return []
    
    # 2. 执行向量相似度搜索
    docs = self.vectorstore.similarity_search(query, k=k)
    
    # 3. 标准化为记录列表
    records = []
    for idx, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        source = str(metadata.get("source") or "").strip()
        title = Path(source).name if source else f"本地知识片段-{idx}"
        records.append({
            "source_id": f"LOC-{idx}",
            "doc_id": source,
            "title": title,
            "snippet": doc.page_content,
            "source_type": "local",
            "metadata": metadata,
        })
    return records
```

#### ingest_text（文本导入）

```python
def ingest_text(self, text: str, source: str) -> int:
    # 1. 分割文本为 chunks
    docs = self.text_splitter.create_documents(
        [text], 
        metadatas=[{"source": source}]
    )
    # 2. 添加到向量存储
    return self.add_documents(docs)
```

#### ingest_paths（文件批量导入）

```python
def ingest_paths(self, paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        total += self.ingest_text(text, source=str(path))
    return total
```

---

## 3. BM25 检索器（bm25_retriever.py）

### 3.1 BM25 算法原理

BM25（Best Matching 25）是一种经典的稀疏检索算法，基于词频（TF）和逆文档频率（IDF）计算文档相关性。

**BM25 公式**：
```
score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
```

其中：
- `qi`：查询中的第 i 个词
- `f(qi, D)`：词 qi 在文档 D 中的词频
- `|D|`：文档 D 的长度
- `avgdl`：所有文档的平均长度
- `k1`：词频饱和参数（推荐 1.2-2.0）
- `b`：文档长度归一化参数（推荐 0.75）
- `IDF(qi)`：词 qi 的逆文档频率

**IDF 公式（BM25+ 变体）**：
```
IDF(qi) = log((N - df + 0.5) / (df + 0.5) + 1)
```

其中：
- `N`：文档总数
- `df`：包含词 qi 的文档数量
- 这种变体避免了负 IDF 值

### 3.2 BM25Config 配置

```python
@dataclass
class BM25Config:
    tokenizer: str = "chinese"       # 分词器类型
    k1: float = 1.5                  # 词频饱和参数
    b: float = 0.75                  # 文档长度归一化参数
    epsilon: float = 0.25            # IDF 平滑参数
    index_path: Optional[str] = None # 索引存储路径
    lowercase: bool = True           # 是否转小写
    enable_stopwords: bool = True    # 是否启用停用词
    stopwords: Optional[list[str]] = None  # 自定义停用词
    min_df: int = 1                  # 最小文档频率
    max_df: float = 0.95             # 最大文档频率（百分比）
```

### 3.3 分词器

系统实现三种分词器：

#### SimpleTokenizer（简单分词器）

```python
class SimpleTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> list[str]:
        tokens = []
        
        # 英文单词提取
        english_words = re.findall(r"[a-zA-Z]+", text)
        tokens.extend(english_words)
        
        # 中文分词：滑动窗口
        chinese_chars = re.findall(r"[一-鿿]+", text)
        for chars in chinese_chars:
            # 单字
            for char in chars:
                tokens.append(char)
            # 双字词（滑动窗口）
            for i in range(len(chars) - 1):
                tokens.append(chars[i:i + 2])
            # 三字词
            for i in range(len(chars) - 2):
                tokens.append(chars[i:i + 3])
        
        # 数字提取
        numbers = re.findall(r"\d+", text)
        tokens.extend(numbers)
        
        return self._filter_tokens(tokens)
```

#### ChineseTokenizer（jieba 分词器）

```python
class ChineseTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> list[str]:
        tokens = list(self.jieba.cut(text))
        tokens = [t.strip() for t in tokens if t.strip()]
        return self._filter_tokens(tokens)
```

#### MixedTokenizer（混合分词器）

结合 jieba 和简单分词器：
- 英文部分使用正则提取（保持完整性）
- 中文部分使用 jieba 分词
- 数字部分使用正则提取

### 3.4 BM25Index 索引类

```python
class BM25Index:
    def __init__(self, k1=1.5, b=0.75, epsilon=0.25):
        self.doc_count: int = 0           # 文档数量
        self.avgdl: float = 0.0           # 平均文档长度
        self.doc_lengths: list[int] = []  # 每个文档的长度
        self.doc_freqs: dict[str, int] = {}   # 词 → 文档频率
        self.term_freqs: list[dict[str, int]] = []  # 每个文档的词频
        self.idf_cache: dict[str, float] = {}  # IDF 缓存
```

#### build 方法（构建索引）

```python
def build(self, tokenized_corpus: list[list[str]]):
    self.doc_count = len(tokenized_corpus)
    self.doc_lengths = [len(doc) for doc in tokenized_corpus]
    self.avgdl = sum(self.doc_lengths) / self.doc_count
    
    self.doc_freqs = {}
    self.term_freqs = []
    
    for doc_tokens in tokenized_corpus:
        # 计算词频
        tf = Counter(doc_tokens)
        self.term_freqs.append(dict(tf))
        
        # 更新文档频率
        for term in set(doc_tokens):
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
    
    # 计算 IDF
    self._compute_idf()
```

#### get_scores 方法（计算 BM25 分数）

```python
def get_scores(self, query_tokens: list[str]) -> np.ndarray:
    scores = np.zeros(self.doc_count)
    query_tf = Counter(query_tokens)
    
    for term, query_freq in query_tf.items():
        idf = self.get_idf(term)
        
        for doc_idx in range(self.doc_count):
            tf = self.term_freqs[doc_idx].get(term, 0)
            if tf == 0:
                continue
            
            doc_len = self.doc_lengths[doc_idx]
            # BM25 核心公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score = idf * numerator / denominator
            
            # 查询词权重
            score *= (1 + math.log(query_freq))
            scores[doc_idx] += score
    
    return scores
```

### 3.5 BM25Retriever 检索器

```python
class BM25Retriever:
    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()
        self.documents: list[Document] = []  # 文档列表
        self.doc_ids: list[str] = []         # 文档 ID 列表
        self.index = BM25Index(...)          # BM25 索引
        self.tokenizer = self._create_tokenizer()  # 分词器
```

#### search 方法

```python
def search(self, query: str, top_k: int = 10) -> list[tuple[Document, float]]:
    # 1. 分词
    query_tokens = self.tokenizer.tokenize(query)
    
    # 2. 计算 BM25 分数
    scores = self.index.get_scores(query_tokens)
    
    # 3. 排序并返回 top-k
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((self.documents[idx], float(scores[idx])))
    
    return results
```

#### 持久化

```python
def save(self, path: str):
    data = {
        "config": self.config,
        "documents": self.documents,
        "doc_ids": self.doc_ids,
        "index": {
            "doc_count": self.index.doc_count,
            "avgdl": self.index.avgdl,
            "doc_lengths": self.index.doc_lengths,
            "doc_freqs": self.index.doc_freqs,
            "term_freqs": self.index.term_freqs,
            "idf_cache": self.index.idf_cache,
        },
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
```

---

## 4. RRF 融合算法（rrf_fusion.py）

### 4.1 RRF 公式

```
RRF_score(d) = Σ (w_i / (k + rank_i(d)))
```

其中：
- `k`：平滑常数（经典值 60），值越小排名靠前的文档优势越大
- `rank_i(d)`：文档 d 在第 i 个检索器中的排名位置
- `w_i`：第 i 个检索器的权重

**RRF 的优势**：
- 无需分数归一化（基于排名，不是原始分数）
- 对异常值鲁棒
- 支持任意数量的检索器
- 可配置权重

### 4.2 RRFConfig 配置

```python
@dataclass
class RRFConfig:
    k: int = 60                           # 平滑常数
    weights: dict[str, float] = None      # 检索器权重 {"vector": 1.0, "bm25": 1.0}
    normalize_scores: bool = False        # 是否归一化最终分数
    min_score_threshold: float = 0.0      # 最低分数阈值
    dedup_strategy: str = "best"          # 去重策略：first 或 best
```

### 4.3 RRFFusionStrategy 融合策略

```python
class RRFFusionStrategy(FusionStrategy):
    def fuse(self, results_list, retriever_names, weights, top_k):
        doc_scores = defaultdict(float)
        doc_appearances = defaultdict(list)
        
        for results, name in zip(results_list, retriever_names):
            weight = weights.get(name, 1.0)
            
            for rank, (doc, _) in enumerate(results, start=1):
                doc_id = DocIdExtractor.get_doc_id(doc)
                rrf_score = weight / (self.k + rank)
                doc_scores[doc_id] += rrf_score
                doc_appearances[doc_id].append(name)
        
        # 排序并返回 top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if doc_id in doc_map:
                results.append((doc_map[doc_id], score))
        
        return results
```

### 4.4 融合示例

假设有两个检索器的结果：

```
向量检索结果：
  排名 1: doc_A (score: 0.95)
  排名 2: doc_B (score: 0.82)
  排名 3: doc_C (score: 0.71)

BM25 检索结果：
  排名 1: doc_B (score: 8.5)
  排名 2: doc_D (score: 6.2)
  排名 3: doc_A (score: 4.1)
```

RRF 融合计算（k=60, w=1.0）：

```
doc_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
doc_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
doc_C: 1/(60+3) + 0       = 0.0159
doc_D: 0       + 1/(60+2) = 0.0161
```

融合排名：doc_B > doc_A > doc_D > doc_C

注意：虽然 doc_A 在向量检索中排名第一，但 doc_B 在两个检索器中都有较高排名，因此综合得分最高。

### 4.5 AdaptiveRRFFusion 自适应融合

```python
class AdaptiveRRFFusion(RRFFusion):
    def _compute_adaptive_weights(self, results_list, retriever_names):
        weights = {}
        
        for name, results in zip(retriever_names, results_list):
            if not results:
                weights[name] = 0.1  # 无结果的检索器给予最低权重
                continue
            
            scores = [score for _, score in results]
            max_score = max(scores)
            
            # 基于最高分和结果数量的启发式
            quality = min(1.0, len(results) / 10.0) * max_score
            weights[name] = max(0.5, quality * 2.0)
        
        # 归一化权重
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return weights
```

自适应融合根据检索结果的质量自动调整权重：
- 结果数量多且分数高的检索器获得更高权重
- 无结果的检索器获得最低权重（0.1）

---

## 5. 混合检索器（hybrid_retriever.py）

### 5.1 HybridRetrieverConfig 配置

```python
@dataclass
class HybridRetrieverConfig:
    bm25_config: Optional[BM25Config] = None
    rrf_config: Optional[RRFConfig] = None
    top_k: int = 10                 # 最终返回数量
    vector_top_k: int = 20          # 向量检索候选数
    bm25_top_k: int = 20            # BM25 检索候选数
    enable_parallel: bool = True    # 是否并行检索
    enable_cache: bool = False      # 是否启用缓存
    cache_ttl: int = 3600           # 缓存 TTL（秒）
    min_score_threshold: float = 0.0
    fusion_strategy: str = "rrf"    # 融合策略：rrf, weighted, adaptive
```

### 5.2 检索流程

```python
def search_with_details(self, query, top_k=None, filters=None):
    # 1. 检查缓存
    if self._cache:
        cached = self._cache.get(query, "hybrid")
        if cached:
            return cached
    
    # 2. 并行或串行检索
    if self.config.enable_parallel:
        vector_results, bm25_results = self._parallel_search(query, filters)
    else:
        vector_results = self._vector_search(query, filters)
        bm25_results = self._bm25_search(query)
    
    # 3. RRF 融合
    fused = self.rrf_fusion.fuse(
        results_list=[vector_results, bm25_results],
        retriever_names=["vector", "bm25"],
        top_k=top_k,
    )
    
    # 4. 分数阈值过滤
    if self.config.min_score_threshold > 0:
        fused = [(doc, score) for doc, score in fused if score >= threshold]
    
    # 5. 构建结果
    result = HybridSearchResult(
        fused_results=[...],
        vector_results=[...],
        bm25_results=[...],
        query=query,
        latency_ms=latency_ms,
        stats=self._compute_stats(vector_results, bm25_results, fused),
    )
    
    # 6. 缓存结果
    if self._cache:
        self._cache.set(query, "hybrid", result)
    
    return result
```

### 5.3 并行检索实现

```python
def _parallel_search(self, query, filters):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已在异步上下文中，使用线程池
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                vector_future = executor.submit(self._vector_search, query, filters)
                bm25_future = executor.submit(self._bm25_search, query)
                vector_results = vector_future.result()
                bm25_results = bm25_future.result()
        else:
            # 使用异步
            async def run_parallel():
                vector_task = loop.run_in_executor(None, self._vector_search, query, filters)
                bm25_task = loop.run_in_executor(None, self._bm25_search, query)
                return await asyncio.gather(vector_task, bm25_task)
            vector_results, bm25_results = loop.run_until_complete(run_parallel())
    except RuntimeError:
        # 回退到串行
        vector_results = self._vector_search(query, filters)
        bm25_results = self._bm25_search(query)
    
    return vector_results, bm25_results
```

### 5.4 向量检索细节

```python
def _vector_search(self, query, filters):
    kwargs = {"k": self.config.vector_top_k}
    if filters:
        kwargs["filter"] = filters
    
    # similarity_search_with_score 返回 (Document, distance) 列表
    docs_with_scores = self.vectorstore.similarity_search_with_score(query, **kwargs)
    
    # 转换距离为相似度
    results = []
    for doc, score in docs_with_scores:
        similarity = 1 - score if score >= 0 else score  # 余弦距离 → 相似度
        results.append((doc, similarity))
    
    return results
```

### 5.5 缓存机制

```python
class RetrievalCache:
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._cache: dict[str, tuple[list, datetime]] = {}
    
    def get(self, query, retriever_type):
        key = f"{retriever_type}:{self._normalize_query(query)}"
        if key in self._cache:
            results, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.ttl:
                return results
            else:
                del self._cache[key]
        return None
```

缓存 key 格式：`{retriever_type}:{normalized_query}`

- `hybrid:什么是 rag`
- `vector:langchain 对比 llamaindex`
- `bm25:deep research agent`

---

## 6. 在主工作流中的使用

### 6.1 RAG 系统初始化

在 `builder.py` 的 `build_agents` 中初始化：

```python
rag_config = RAGConfig(
    milvus_host=config.milvus_host,
    milvus_port=config.milvus_port,
    collection_name=config.milvus_collection,
)
init_rag_system(api_key=api_key, config=rag_config)
```

`init_rag_system` 创建全局单例 `_RAG_SYSTEM`。

### 6.2 在 local_rag_node 中调用

```python
# search.py 中
records = search_knowledge_base_records(str(item.get("query", "")), limit=4)
```

`search_knowledge_base_records` 调用 `_RAG_SYSTEM.search_records(query, k=limit)`。

### 6.3 数据流

```
local_rag_node
    → _build_queries(state, "local")     # 构建查询列表
    → search_knowledge_base_records(query)  # Milvus 向量搜索
    → _assign_source_ids(records, prefix)   # 分配 LOC 前缀 ID
    → _invoke_json_agent(...)               # LLM 整理证据
    → _prune_evidence_to_allowed_sources()  # 校验 source_id
    → 返回 local_evidence
```

---

## 7. 文档导入（ingest.py）

### 7.1 导入脚本

```python
# ingest.py
from deep_research.rag.core import RAGSystem, RAGConfig

config = RAGConfig(
    milvus_host="localhost",
    milvus_port=19530,
    collection_name="mult_agent_knowledge",
)
rag = RAGSystem(api_key="your-api-key", config=config)

# 导入单个文本
rag.ingest_text("这是一段测试文本...", source="test.txt")

# 批量导入文件
from pathlib import Path
rag.ingest_paths(Path("./knowledge_base").glob("*.md"))
```

### 7.2 导入流程

```
文件路径列表
    │
    ├── 遍历每个文件
    │   ├── 读取文件内容 (read_text)
    │   ├── 文本分割 (text_splitter.create_documents)
    │   │   ├── 按 500 字符分块
    │   │   ├── 50 字符重叠
    │   │   └── 每块附带 metadata {"source": file_path}
    │   └── 添加到 Milvus (vectorstore.add_documents)
    │       ├── 调用 DashScope Embedding API 生成向量
    │       └── 写入 Milvus 集合
    │
    └── 返回导入文档总数
```
