"""
混合检索器

整合向量检索、BM25 检索和 RRF 融合的多路检索系统。

特性：
- 向量稠密检索（语义相似）
- BM25 稀疏检索（关键词匹配）
- RRF 混合融合（取长补短）
- 异步检索支持
- 检索效果评估
- 可配置的检索策略
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

from langchain_core.documents import Document as LCDocument

from .bm25_retriever import BM25Config, BM25Retriever, Document
from .rrf_fusion import AdaptiveRRFFusion, MultiStageFusion, RRFConfig, RRFFusion

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ //
# 配置类
# ------------------------------------------------------------------ //


@dataclass
class HybridRetrieverConfig:
    """
    混合检索器配置
    
    Attributes:
        bm25_config: BM25 检索器配置
        rrf_config: RRF 融合器配置
        top_k: 最终返回的文档数量
        vector_top_k: 向量检索的候选文档数量
        bm25_top_k: BM25 检索的候选文档数量
        enable_parallel: 是否启用并行检索
        enable_cache: 是否启用检索缓存
        cache_ttl: 缓存过期时间（秒）
        min_score_threshold: 最低分数阈值
        fusion_strategy: 融合策略，可选 "rrf", "weighted", "adaptive"
    """
    bm25_config: Optional[BM25Config] = None
    rrf_config: Optional[RRFConfig] = None
    top_k: int = 10
    vector_top_k: int = 20
    bm25_top_k: int = 20
    enable_parallel: bool = True
    enable_cache: bool = False
    cache_ttl: int = 3600
    min_score_threshold: float = 0.0
    fusion_strategy: str = "rrf"
    
    def __post_init__(self):
        if self.bm25_config is None:
            self.bm25_config = BM25Config()
        if self.rrf_config is None:
            self.rrf_config = RRFConfig()


# ------------------------------------------------------------------ //
# 检索结果
# ------------------------------------------------------------------ //


@dataclass
class RetrievalResult:
    """
    检索结果封装
    
    Attributes:
        document: 文档对象
        score: 相关性分数
        source: 来源检索器（"vector", "bm25", "hybrid"）
        metadata: 额外元数据
    """
    document: Any
    score: float
    source: str = "hybrid"
    metadata: dict = field(default_factory=dict)
    
    def to_tuple(self) -> tuple[Any, float]:
        """转换为 (文档, 分数) 元组"""
        return (self.document, self.score)


@dataclass
class HybridSearchResult:
    """
    混合检索完整结果
    
    包含融合结果和各检索器的中间结果，便于分析和调试
    """
    fused_results: list[RetrievalResult]
    vector_results: list[RetrievalResult]
    bm25_results: list[RetrievalResult]
    query: str
    latency_ms: float
    stats: dict = field(default_factory=dict)
    
    @property
    def total_results(self) -> int:
        return len(self.fused_results)


# ------------------------------------------------------------------ //
# 检索缓存
# ------------------------------------------------------------------ //


class RetrievalCache:
    """
    检索结果缓存
    
    使用简单的内存缓存，支持 TTL 过期
    """
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._cache: dict[str, tuple[list, datetime]] = {}
    
    def _normalize_query(self, query: str) -> str:
        """规范化查询字符串"""
        return " ".join(query.lower().split())
    
    def get(self, query: str, retriever_type: str) -> Optional[list]:
        """获取缓存"""
        key = f"{retriever_type}:{self._normalize_query(query)}"
        if key in self._cache:
            results, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.ttl:
                return results
            else:
                del self._cache[key]
        return None
    
    def set(self, query: str, retriever_type: str, results: list) -> None:
        """设置缓存"""
        key = f"{retriever_type}:{self._normalize_query(query)}"
        self._cache[key] = (results, datetime.now())
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()


# ------------------------------------------------------------------ //
# 混合检索器
# ------------------------------------------------------------------ //


class HybridRetriever:
    """
    混合检索器
    
    整合向量检索和 BM25 检索，使用 RRF 进行融合
    
    特性：
    - 向量检索：语义相似度，适合模糊匹配、同义词
    - BM25 检索：关键词匹配，适合精确匹配、专有名词
    - RRF 融合：取长补短，提高召回率
    
    Example:
        >>> config = HybridRetrieverConfig(top_k=10)
        >>> retriever = HybridRetriever(vectorstore, config)
        >>> results = retriever.search("查询文本")
    """
    
    def __init__(
        self,
        vectorstore: Any,
        config: Optional[HybridRetrieverConfig] = None,
    ):
        """
        初始化混合检索器
        
        Args:
            vectorstore: 向量存储对象（需实现 similarity_search_with_score 方法）
            config: 混合检索配置
        """
        self.vectorstore = vectorstore
        self.config = config or HybridRetrieverConfig()
        self.bm25_retriever = BM25Retriever(config=self.config.bm25_config)
        self._cache = RetrievalCache(ttl=self.config.cache_ttl) if self.config.enable_cache else None
        
        # 根据策略选择融合器
        self._init_fusion()
    
    def _init_fusion(self) -> None:
        """初始化融合器"""
        strategy = self.config.fusion_strategy.lower()
        
        if strategy == "adaptive":
            self.rrf_fusion = AdaptiveRRFFusion(config=self.config.rrf_config)
        elif strategy == "multistage":
            self.rrf_fusion = MultiStageFusion(
                stage1_config=self.config.rrf_config,
                stage2_config=self.config.rrf_config,
            )
        else:
            self.rrf_fusion = RRFFusion(config=self.config.rrf_config)
    
    # ------------------------------------------------------------------ //
    # 文档管理
    # ------------------------------------------------------------------ //
    
    def add_documents(self, documents: list[LCDocument]) -> int:
        """
        添加文档到检索器
        
        文档会同时添加到 BM25 索引中
        
        Args:
            documents: LangChain Document 列表
            
        Returns:
            添加的文档数量
        """
        bm25_docs = [
            Document(
                doc_id=doc.metadata.get("source", f"doc-{i}"),
                content=doc.page_content,
                metadata=doc.metadata,
            )
            for i, doc in enumerate(documents)
        ]
        self.bm25_retriever.add_documents(bm25_docs)
        logger.info(f"混合检索器索引了 {len(documents)} 个文档")
        return len(documents)
    
    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        doc_ids: Optional[list[str]] = None,
    ) -> int:
        """
        添加文本列表
        
        Args:
            texts: 文本内容列表
            metadatas: 元数据列表
            doc_ids: 文档 ID 列表
            
        Returns:
            添加的文档数量
        """
        metadatas = metadatas or [{} for _ in texts]
        doc_ids = doc_ids or [f"doc-{i}" for i in range(len(texts))]
        
        documents = [
            LCDocument(page_content=text, metadata={**meta, "source": doc_id})
            for text, meta, doc_id in zip(texts, metadatas, doc_ids)
        ]
        
        return self.add_documents(documents)
    
    def clear(self) -> None:
        """清空所有索引"""
        self.bm25_retriever.clear()
        if self._cache:
            self._cache.clear()
    
    # ------------------------------------------------------------------ //
    # 检索
    # ------------------------------------------------------------------ //
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[tuple[Any, float]]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            filters: 元数据过滤条件（用于向量检索）
            
        Returns:
            (文档, 分数) 元组列表
        """
        result = self.search_with_details(query, top_k, filters)
        return [r.to_tuple() for r in result.fused_results]
    
    def search_with_details(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> HybridSearchResult:
        """
        执行混合检索并返回详细结果
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            filters: 元数据过滤条件
            
        Returns:
            完整的检索结果对象
        """
        import time
        start_time = time.time()
        
        top_k = top_k or self.config.top_k
        
        # 检查缓存
        if self._cache:
            cached = self._cache.get(query, "hybrid")
            if cached:
                logger.debug(f"命中缓存: {query}")
                return cached
        
        # 并行或串行检索
        if self.config.enable_parallel:
            vector_results, bm25_results = self._parallel_search(query, filters)
        else:
            vector_results = self._vector_search(query, filters)
            bm25_results = self._bm25_search(query)
        
        # 融合
        fused = self.rrf_fusion.fuse(
            results_list=[vector_results, bm25_results],
            retriever_names=["vector", "bm25"],
            top_k=top_k,
        )
        
        # 应用分数阈值
        if self.config.min_score_threshold > 0:
            fused = [
                (doc, score) for doc, score in fused
                if score >= self.config.min_score_threshold
            ]
        
        # 构建结果
        latency_ms = (time.time() - start_time) * 1000
        
        result = HybridSearchResult(
            fused_results=[
                RetrievalResult(document=doc, score=score, source="hybrid")
                for doc, score in fused
            ],
            vector_results=[
                RetrievalResult(document=doc, score=score, source="vector")
                for doc, score in vector_results
            ],
            bm25_results=[
                RetrievalResult(document=doc, score=score, source="bm25")
                for doc, score in bm25_results
            ],
            query=query,
            latency_ms=latency_ms,
            stats=self._compute_stats(vector_results, bm25_results, fused),
        )
        
        # 缓存结果
        if self._cache:
            self._cache.set(query, "hybrid", result)
        
        return result
    
    def _parallel_search(
        self,
        query: str,
        filters: Optional[dict] = None,
    ) -> tuple[list[tuple[Any, float]], list[tuple[Any, float]]]:
        """并行执行向量检索和 BM25 检索"""
        try:
            # 尝试异步并行
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已经在异步上下文中，使用线程池
                import concurrent.futures
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    vector_future = executor.submit(self._vector_search, query, filters)
                    bm25_future = executor.submit(self._bm25_search, query)
                    
                    vector_results = vector_future.result()
                    bm25_results = bm25_future.result()
            else:
                # 可以使用异步
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
    
    def _vector_search(
        self,
        query: str,
        filters: Optional[dict] = None,
    ) -> list[tuple[Any, float]]:
        """执行向量检索"""
        try:
            # 检查缓存
            if self._cache:
                cached = self._cache.get(query, "vector")
                if cached:
                    return cached
            
            # 执行检索
            kwargs = {"k": self.config.vector_top_k}
            if filters:
                kwargs["filter"] = filters
            
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, **kwargs)
            
            # 转换分数（假设距离越小越相似，转换为相似度）
            results = []
            for doc, score in docs_with_scores:
                # 对于余弦距离，相似度 = 1 - 距离
                # 对于欧氏距离，可能需要不同的转换
                similarity = 1 - score if score >= 0 else score
                results.append((doc, similarity))
            
            # 缓存结果
            if self._cache:
                self._cache.set(query, "vector", results)
            
            return results
        
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def _bm25_search(self, query: str) -> list[tuple[Any, float]]:
        """执行 BM25 检索"""
        try:
            # 检查缓存
            if self._cache:
                cached = self._cache.get(query, "bm25")
                if cached:
                    return cached
            
            results = self.bm25_retriever.search(query, top_k=self.config.bm25_top_k)
            
            # 缓存结果
            if self._cache:
                self._cache.set(query, "bm25", results)
            
            return results
        
        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []
    
    def _compute_stats(
        self,
        vector_results: list,
        bm25_results: list,
        fused_results: list,
    ) -> dict:
        """计算检索统计信息"""
        # 找出同时出现在两个检索器中的文档
        vector_ids = {self._get_doc_id(doc) for doc, _ in vector_results}
        bm25_ids = {self._get_doc_id(doc) for doc, _ in bm25_results}
        overlap = vector_ids & bm25_ids
        
        return {
            "vector_count": len(vector_results),
            "bm25_count": len(bm25_results),
            "fused_count": len(fused_results),
            "overlap_count": len(overlap),
            "overlap_ratio": len(overlap) / min(len(vector_ids), len(bm25_ids)) if vector_ids and bm25_ids else 0,
        }
    
    def _get_doc_id(self, doc: Any) -> str:
        """获取文档 ID"""
        if hasattr(doc, "doc_id"):
            return doc.doc_id
        if hasattr(doc, "metadata"):
            return doc.metadata.get("source", doc.metadata.get("id", str(hash(doc.page_content if hasattr(doc, "page_content") else str(doc)))))
        return str(hash(doc))
    
    # ------------------------------------------------------------------ //
    # 单独检索
    # ------------------------------------------------------------------ //
    
    def search_vector_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[tuple[Any, float]]:
        """
        仅执行向量检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件
            
        Returns:
            检索结果
        """
        top_k = top_k or self.config.top_k
        results = self._vector_search(query, filters)
        return results[:top_k]
    
    def search_bm25_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        query_weights: Optional[dict[str, float]] = None,
    ) -> list[tuple[Any, float]]:
        """
        仅执行 BM25 检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            query_weights: 查询词权重
            
        Returns:
            检索结果
        """
        top_k = top_k or self.config.top_k
        results = self.bm25_retriever.search(query, top_k=top_k, query_weights=query_weights)
        return results
    
    # ------------------------------------------------------------------ //
    # 异步检索
    # ------------------------------------------------------------------ //
    
    async def asearch(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[tuple[Any, float]]:
        """
        异步执行混合检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件
            
        Returns:
            检索结果
        """
        top_k = top_k or self.config.top_k
        
        # 并行执行两个检索
        vector_task = asyncio.create_task(self._a_vector_search(query, filters))
        bm25_task = asyncio.create_task(self._a_bm25_search(query))
        
        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
        
        # 融合
        fused = self.rrf_fusion.fuse(
            results_list=[vector_results, bm25_results],
            retriever_names=["vector", "bm25"],
            top_k=top_k,
        )
        
        return fused
    
    async def _a_vector_search(
        self,
        query: str,
        filters: Optional[dict] = None,
    ) -> list[tuple[Any, float]]:
        """异步向量检索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._vector_search, query, filters)
    
    async def _a_bm25_search(self, query: str) -> list[tuple[Any, float]]:
        """异步 BM25 检索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._bm25_search, query)
    
    # ------------------------------------------------------------------ //
    # 权重调整
    # ------------------------------------------------------------------ //
    
    def set_weights(self, vector_weight: float, bm25_weight: float) -> None:
        """
        设置检索器权重
        
        Args:
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重
        """
        if hasattr(self.rrf_fusion, "update_weights"):
            self.rrf_fusion.update_weights({
                "vector": vector_weight,
                "bm25": bm25_weight,
            })
        else:
            self.config.rrf_config.weights = {
                "vector": vector_weight,
                "bm25": bm25_weight,
            }
        logger.info(f"权重已更新: vector={vector_weight}, bm25={bm25_weight}")
    
    def set_rrf_k(self, k: int) -> None:
        """
        设置 RRF 的 k 参数
        
        Args:
            k: 新的 k 值
        """
        if hasattr(self.rrf_fusion, "set_k"):
            self.rrf_fusion.set_k(k)
        else:
            self.config.rrf_config.k = k
            self._init_fusion()
        logger.info(f"RRF k 值已更新为: {k}")
    
    # ------------------------------------------------------------------ //
    # 持久化
    # ------------------------------------------------------------------ //
    
    def save_bm25_index(self, path: str) -> None:
        """
        保存 BM25 索引到文件
        
        Args:
            path: 保存路径
        """
        self.bm25_retriever.save(path)
    
    def load_bm25_index(self, path: str) -> None:
        """
        从文件加载 BM25 索引
        
        Args:
            path: 文件路径
        """
        self.bm25_retriever.load(path)
    
    # ------------------------------------------------------------------ //
    # 统计与诊断
    # ------------------------------------------------------------------ //
    
    def get_stats(self) -> dict:
        """
        获取检索器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "bm25": self.bm25_retriever.get_stats(),
            "config": {
                "top_k": self.config.top_k,
                "vector_top_k": self.config.vector_top_k,
                "bm25_top_k": self.config.bm25_top_k,
                "fusion_strategy": self.config.fusion_strategy,
                "enable_parallel": self.config.enable_parallel,
                "enable_cache": self.config.enable_cache,
            },
            "cache_size": len(self._cache._cache) if self._cache else 0,
        }
    
    def get_fusion_stats(self) -> dict:
        """
        获取最近一次融合的统计信息
        
        Returns:
            融合统计字典
        """
        if hasattr(self.rrf_fusion, "get_stats"):
            return self.rrf_fusion.get_stats()
        return {}
    
    def __len__(self) -> int:
        return len(self.bm25_retriever)


# ------------------------------------------------------------------ //
# 工厂函数
# ------------------------------------------------------------------ //


def create_hybrid_retriever(
    vectorstore: Any,
    top_k: int = 10,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
    **kwargs,
) -> HybridRetriever:
    """
    创建混合检索器的便捷工厂函数
    
    Args:
        vectorstore: 向量存储
        top_k: 最终返回数量
        vector_top_k: 向量检索候选数
        bm25_top_k: BM25 检索候选数
        k1: BM25 k1 参数
        b: BM25 b 参数
        rrf_k: RRF k 参数
        vector_weight: 向量检索权重
        bm25_weight: BM25 检索权重
        **kwargs: 其他配置参数
        
    Returns:
        配置好的混合检索器
        
    Example:
        >>> retriever = create_hybrid_retriever(
        ...     vectorstore=vs,
        ...     top_k=10,
        ...     vector_weight=1.2,  # 向量检索更重要的场景
        ... )
    """
    bm25_config = BM25Config(k1=k1, b=b)
    rrf_config = RRFConfig(
        k=rrf_k,
        weights={"vector": vector_weight, "bm25": bm25_weight},
    )
    
    config = HybridRetrieverConfig(
        bm25_config=bm25_config,
        rrf_config=rrf_config,
        top_k=top_k,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        **kwargs,
    )
    
    return HybridRetriever(vectorstore=vectorstore, config=config)
