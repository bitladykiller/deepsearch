"""
RAG (Retrieval-Augmented Generation) 模块

提供完整的检索增强生成功能，包括：
- BM25 稀疏检索
- 向量稠密检索
- RRF 混合检索融合
- 文档摄入和管理

模块结构：
├── core.py           - 核心向量存储和检索
├── bm25_retriever.py - BM25 稀疏检索器
├── rrf_fusion.py     - RRF 融合算法
├── hybrid_retriever.py - 混合检索器
└── ingest.py         - 文档摄入工具

使用示例：
    >>> from rag import HybridRetriever, create_hybrid_retriever
    >>> retriever = create_hybrid_retriever(vectorstore, top_k=10)
    >>> results = retriever.search("查询文本")
"""

# ------------------------------------------------------------------ //
# BM25 相关
# ------------------------------------------------------------------ //

from .bm25_retriever import (
    # 配置类
    BM25Config,
    # 检索器
    BM25Retriever,
    BM25Index,
    # 文档类
    Document,
    # 分词器
    BaseTokenizer,
    SimpleTokenizer,
    ChineseTokenizer,
    MixedTokenizer,
    # 停用词
    DEFAULT_CHINESE_STOPWORDS,
    DEFAULT_ENGLISH_STOPWORDS,
)

# ------------------------------------------------------------------ //
# RRF 相关
# ------------------------------------------------------------------ //

from .rrf_fusion import (
    # 配置类
    RRFConfig,
    # 融合器
    RRFFusion,
    AdaptiveRRFFusion,
    MultiStageFusion,
    # 策略
    FusionStrategy,
    RRFFusionStrategy,
    WeightedScoreFusionStrategy,
    # 工具类
    DocIdExtractor,
    ScoreNormalizer,
)

# ------------------------------------------------------------------ //
# 混合检索相关
# ------------------------------------------------------------------ //

from .hybrid_retriever import (
    # 配置类
    HybridRetrieverConfig,
    # 检索器
    HybridRetriever,
    # 结果类
    RetrievalResult,
    HybridSearchResult,
    # 缓存
    RetrievalCache,
    # 工厂函数
    create_hybrid_retriever,
)

# ------------------------------------------------------------------ //
# 导出列表
# ------------------------------------------------------------------ //

__all__ = [
    # BM25 配置
    "BM25Config",
    # BM25 检索器
    "BM25Retriever",
    "BM25Index",
    # BM25 文档
    "Document",
    # BM25 分词器
    "BaseTokenizer",
    "SimpleTokenizer",
    "ChineseTokenizer",
    "MixedTokenizer",
    # BM25 停用词
    "DEFAULT_CHINESE_STOPWORDS",
    "DEFAULT_ENGLISH_STOPWORDS",
    # RRF 配置
    "RRFConfig",
    # RRF 融合器
    "RRFFusion",
    "AdaptiveRRFFusion",
    "MultiStageFusion",
    # RRF 策略
    "FusionStrategy",
    "RRFFusionStrategy",
    "WeightedScoreFusionStrategy",
    # RRF 工具
    "DocIdExtractor",
    "ScoreNormalizer",
    # 混合检索配置
    "HybridRetrieverConfig",
    # 混合检索器
    "HybridRetriever",
    # 混合检索结果
    "RetrievalResult",
    "HybridSearchResult",
    # 混合检索缓存
    "RetrievalCache",
    # 工厂函数
    "create_hybrid_retriever",
]