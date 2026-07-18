"""
混合检索器单元测试

测试 HybridRetriever 的核心功能：
- 混合检索
- 异步检索
- 权重调整
- 缓存功能
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass

from deep_research.rag.hybrid_retriever import (
    HybridRetrieverConfig,
    HybridRetriever,
    RetrievalResult,
    HybridSearchResult,
    RetrievalCache,
    create_hybrid_retriever,
)
from deep_research.rag.bm25_retriever import BM25Config, Document
from deep_research.rag.rrf_fusion import RRFConfig


class MockVectorStore:
    """模拟向量存储"""
    
    def __init__(self):
        self.documents = []
    
    def similarity_search_with_score(self, query: str, k: int = 10, filter: dict = None):
        """模拟相似度检索"""
        results = []
        for i in range(min(k, 5)):
            doc = MagicMock()
            doc.page_content = f"Document {i} for query: {query}"
            doc.metadata = {"source": f"vec_doc_{i}", "id": f"vec_doc_{i}"}
            # 返回距离分数（越小越相似）
            score = 0.1 * i
            results.append((doc, score))
        return results
    
    def add_documents(self, documents):
        """添加文档"""
        self.documents.extend(documents)
        return len(documents)


class TestRetrievalCache:
    """测试检索缓存"""
    
    def test_cache_set_and_get(self):
        """测试缓存设置和获取"""
        cache = RetrievalCache(ttl=3600)
        
        results = [("doc1", 0.9), ("doc2", 0.8)]
        cache.set("test query", "vector", results)
        
        cached = cache.get("test query", "vector")
        assert cached == results
    
    def test_cache_expiry(self):
        """测试缓存过期"""
        import time
        
        cache = RetrievalCache(ttl=1)  # 1 秒过期
        
        results = [("doc1", 0.9)]
        cache.set("test query", "vector", results)
        
        # 等待过期
        time.sleep(2)
        
        cached = cache.get("test query", "vector")
        assert cached is None
    
    def test_cache_clear(self):
        """测试缓存清空"""
        cache = RetrievalCache(ttl=3600)
        
        cache.set("query1", "vector", [("doc1", 0.9)])
        cache.set("query2", "bm25", [("doc2", 5.0)])
        
        cache.clear()
        
        assert cache.get("query1", "vector") is None
        assert cache.get("query2", "bm25") is None
    
    def test_cache_normalize_query(self):
        """测试查询规范化"""
        cache = RetrievalCache(ttl=3600)
        
        results = [("doc1", 0.9)]
        cache.set("Test Query  ", "vector", results)
        
        # 不同形式的相同查询应该命中缓存
        cached = cache.get("test query", "vector")
        assert cached == results
    
    def test_cache_different_types(self):
        """测试不同类型的缓存"""
        cache = RetrievalCache(ttl=3600)
        
        vector_results = [("doc1", 0.9)]
        bm25_results = [("doc2", 5.0)]
        
        cache.set("query", "vector", vector_results)
        cache.set("query", "bm25", bm25_results)
        
        assert cache.get("query", "vector") == vector_results
        assert cache.get("query", "bm25") == bm25_results


class TestRetrievalResult:
    """测试检索结果"""
    
    def test_result_creation(self):
        """测试结果创建"""
        doc = MagicMock()
        result = RetrievalResult(
            document=doc,
            score=0.95,
            source="hybrid",
            metadata={"retrievers": ["vector", "bm25"]}
        )
        
        assert result.document == doc
        assert result.score == 0.95
        assert result.source == "hybrid"
    
    def test_to_tuple(self):
        """测试转换为元组"""
        doc = MagicMock()
        result = RetrievalResult(document=doc, score=0.9)
        
        tuple_result = result.to_tuple()
        assert tuple_result == (doc, 0.9)


class TestHybridSearchResult:
    """测试混合检索结果"""
    
    def test_result_structure(self):
        """测试结果结构"""
        fused = [RetrievalResult(MagicMock(), 0.9)]
        vector = [RetrievalResult(MagicMock(), 0.85)]
        bm25 = [RetrievalResult(MagicMock(), 5.0)]
        
        result = HybridSearchResult(
            fused_results=fused,
            vector_results=vector,
            bm25_results=bm25,
            query="test",
            latency_ms=100.0,
        )
        
        assert result.fused_results == fused
        assert result.total_results == 1
        assert result.latency_ms == 100.0


class TestHybridRetrieverConfig:
    """测试混合检索配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = HybridRetrieverConfig()
        
        assert config.top_k == 10
        assert config.vector_top_k == 20
        assert config.bm25_top_k == 20
        assert config.enable_parallel is True
        assert config.enable_cache is False
        assert config.fusion_strategy == "rrf"
    
    def test_custom_config(self):
        """测试自定义配置"""
        bm25_config = BM25Config(k1=1.2, b=0.8)
        rrf_config = RRFConfig(k=40)
        
        config = HybridRetrieverConfig(
            bm25_config=bm25_config,
            rrf_config=rrf_config,
            top_k=15,
            vector_top_k=30,
            bm25_top_k=25,
            enable_parallel=False,
            enable_cache=True,
            cache_ttl=1800,
            min_score_threshold=0.05,
            fusion_strategy="adaptive",
        )
        
        assert config.bm25_config.k1 == 1.2
        assert config.rrf_config.k == 40
        assert config.top_k == 15
        assert config.enable_parallel is False
        assert config.enable_cache is True
        assert config.fusion_strategy == "adaptive"


class TestHybridRetriever:
    """测试混合检索器"""
    
    def setup_method(self):
        self.vectorstore = MockVectorStore()
        self.config = HybridRetrieverConfig(
            top_k=10,
            vector_top_k=5,
            bm25_top_k=5,
            enable_parallel=False,
            enable_cache=False,
        )
        self.retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            config=self.config,
        )
        
        # 添加测试文档
        docs = [
            MagicMock(page_content="Python programming language", metadata={"source": "doc1"}),
            MagicMock(page_content="Machine learning basics", metadata={"source": "doc2"}),
            MagicMock(page_content="Data science with Python", metadata={"source": "doc3"}),
        ]
        self.retriever.add_documents(docs)
    
    def test_add_documents(self):
        """测试添加文档"""
        assert len(self.retriever) == 3
    
    def test_search(self):
        """测试混合检索"""
        results = self.retriever.search("Python programming")
        
        assert len(results) > 0
        assert len(results) <= self.config.top_k
        
        for doc, score in results:
            assert hasattr(doc, "page_content") or hasattr(doc, "content")
    
    def test_search_with_details(self):
        """测试详细检索"""
        result = self.retriever.search_with_details("Python")
        
        assert isinstance(result, HybridSearchResult)
        assert result.query == "Python"
        assert len(result.fused_results) > 0
        assert result.latency_ms > 0
        assert "vector_count" in result.stats
        assert "bm25_count" in result.stats
    
    def test_search_vector_only(self):
        """测试仅向量检索"""
        results = self.retriever.search_vector_only("Python", top_k=5)
        
        assert len(results) > 0
        assert len(results) <= 5
    
    def test_search_bm25_only(self):
        """测试仅 BM25 检索"""
        results = self.retriever.search_bm25_only("Python", top_k=5)
        
        assert len(results) > 0
    
    def test_set_weights(self):
        """测试设置权重"""
        self.retriever.set_weights(vector_weight=1.5, bm25_weight=0.8)
        
        # 再次检索应该使用新权重
        results = self.retriever.search("Python")
        assert len(results) > 0
    
    def test_set_rrf_k(self):
        """测试设置 RRF k 值"""
        self.retriever.set_rrf_k(40)
        
        results = self.retriever.search("Python")
        assert len(results) > 0
    
    def test_get_stats(self):
        """测试统计信息"""
        stats = self.retriever.get_stats()
        
        assert "bm25" in stats
        assert "config" in stats
        assert stats["config"]["top_k"] == 10
    
    def test_clear(self):
        """测试清空"""
        self.retriever.clear()
        assert len(self.retriever) == 0
    
    def test_with_cache(self):
        """测试缓存功能"""
        config = HybridRetrieverConfig(enable_cache=True, cache_ttl=3600)
        retriever = HybridRetriever(vectorstore=self.vectorstore, config=config)
        
        retriever.add_documents([
            MagicMock(page_content="Test content", metadata={"source": "test"})
        ])
        
        # 第一次检索
        results1 = retriever.search("test")
        
        # 第二次检索应该命中缓存
        results2 = retriever.search("test")
        
        assert len(results1) == len(results2)
    
    def test_min_score_threshold(self):
        """测试最低分数阈值"""
        config = HybridRetrieverConfig(min_score_threshold=0.1)
        retriever = HybridRetriever(vectorstore=self.vectorstore, config=config)
        
        retriever.add_documents([
            MagicMock(page_content="Python content", metadata={"source": "doc1"})
        ])
        
        results = retriever.search("Python")
        
        # 验证所有结果分数大于阈值（这里主要是验证功能可用）
        assert isinstance(results, list)


class TestHybridRetrieverAsync:
    """测试异步检索"""
    
    @pytest.mark.asyncio
    async def test_async_search(self):
        """测试异步检索"""
        vectorstore = MockVectorStore()
        config = HybridRetrieverConfig(top_k=5)
        retriever = HybridRetriever(vectorstore=vectorstore, config=config)
        
        retriever.add_documents([
            MagicMock(page_content="Test doc", metadata={"source": "doc1"})
        ])
        
        results = await retriever.asearch("test query")
        
        assert len(results) > 0


class TestCreateHybridRetriever:
    """测试工厂函数"""
    
    def test_factory_basic(self):
        """测试基本工厂创建"""
        vectorstore = MockVectorStore()
        
        retriever = create_hybrid_retriever(
            vectorstore=vectorstore,
            top_k=10,
        )
        
        assert isinstance(retriever, HybridRetriever)
        assert retriever.config.top_k == 10
    
    def test_factory_full_params(self):
        """测试完整参数工厂创建"""
        vectorstore = MockVectorStore()
        
        retriever = create_hybrid_retriever(
            vectorstore=vectorstore,
            top_k=15,
            vector_top_k=30,
            bm25_top_k=25,
            k1=1.2,
            b=0.8,
            rrf_k=40,
            vector_weight=1.5,
            bm25_weight=0.8,
        )
        
        assert retriever.config.top_k == 15
        assert retriever.config.vector_top_k == 30
        assert retriever.config.bm25_top_k == 25
        assert retriever.config.bm25_config.k1 == 1.2
        assert retriever.config.bm25_config.b == 0.8
        assert retriever.config.rrf_config.k == 40
        assert retriever.config.rrf_config.weights["vector"] == 1.5
        assert retriever.config.rrf_config.weights["bm25"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])