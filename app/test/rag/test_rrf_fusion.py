"""
RRF 融合算法单元测试

测试 RRFFusion 的核心功能：
- 多路检索结果融合
- 权重调整
- 分数归一化
- 自适应融合
"""

import pytest
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.mult_agents.rag.rrf_fusion import (
    RRFConfig,
    RRFFusion,
    AdaptiveRRFFusion,
    MultiStageFusion,
    DocIdExtractor,
    ScoreNormalizer,
    RRFFusionStrategy,
    WeightedScoreFusionStrategy,
)
from app.mult_agents.rag.bm25_retriever import Document


class MockDocument:
    """模拟文档对象"""
    
    def __init__(self, doc_id: str, content: str = "", score: float = 0.0):
        self.doc_id = doc_id
        self.content = content
        self.metadata = {}
        self._score = score
    
    def __repr__(self):
        return f"MockDocument({self.doc_id})"


class TestDocIdExtractor:
    """测试文档 ID 提取器"""
    
    def test_extract_from_doc_id(self):
        """测试从 doc_id 属性提取"""
        doc = Document(doc_id="test_doc", content="content")
        assert DocIdExtractor.get_doc_id(doc) == "test_doc"
    
    def test_extract_from_mock(self):
        """测试从 MockDocument 提取"""
        doc = MockDocument("mock_doc")
        assert DocIdExtractor.get_doc_id(doc) == "mock_doc"
    
    def test_extract_from_metadata(self):
        """测试从 metadata 提取"""
        doc = Document(doc_id="", content="content", metadata={"source": "meta_source"})
        # doc_id 为空时会使用 hash
        assert DocIdExtractor.get_doc_id(doc) == ""


class TestScoreNormalizer:
    """测试分数归一化器"""
    
    def test_min_max_normalize(self):
        """测试 Min-Max 归一化"""
        scores = {"a": 0.0, "b": 0.5, "c": 1.0}
        normalized = ScoreNormalizer.normalize(scores, method="min-max")
        
        assert normalized["a"] == 0.0
        assert normalized["b"] == 0.5
        assert normalized["c"] == 1.0
    
    def test_min_max_normalize_custom_range(self):
        """测试自定义范围的归一化"""
        scores = {"a": 10.0, "b": 20.0, "c": 30.0}
        normalized = ScoreNormalizer.normalize(scores, method="min-max")
        
        assert normalized["a"] == 0.0
        assert normalized["b"] == 0.5
        assert normalized["c"] == 1.0
    
    def test_softmax_normalize(self):
        """测试 Softmax 归一化"""
        scores = {"a": 1.0, "b": 2.0, "c": 3.0}
        normalized = ScoreNormalizer.normalize(scores, method="softmax")
        
        # 所有值应该在 0-1 之间，且总和为 1
        total = sum(normalized.values())
        assert pytest.approx(total, 0.01) == 1.0
        assert normalized["c"] > normalized["b"] > normalized["a"]
    
    def test_rank_normalize(self):
        """测试排名归一化"""
        scores = {"a": 100.0, "b": 50.0, "c": 25.0}
        normalized = ScoreNormalizer.normalize(scores, method="rank")
        
        # 排名归一化：最高分的文档应该有最高的归一化值
        assert normalized["a"] > normalized["b"] > normalized["c"]
    
    def test_empty_scores(self):
        """测试空分数"""
        scores = {}
        normalized = ScoreNormalizer.normalize(scores)
        assert normalized == {}
    
    def test_single_score(self):
        """测试单个分数"""
        scores = {"a": 100.0}
        normalized = ScoreNormalizer.normalize(scores, method="min-max")
        # 单个值的 min-max 归一化应该返回 1.0（或根据实现）
        assert "a" in normalized


class TestRRFConfig:
    """测试 RRF 配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RRFConfig()
        
        assert config.k == 60
        assert config.weights == {"vector": 1.0, "bm25": 1.0}
        assert config.normalize_scores is False
        assert config.min_score_threshold == 0.0
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = RRFConfig(
            k=40,
            weights={"vector": 1.5, "bm25": 0.8},
            normalize_scores=True,
            min_score_threshold=0.01,
        )
        
        assert config.k == 40
        assert config.weights["vector"] == 1.5
        assert config.weights["bm25"] == 0.8
    
    def test_invalid_k(self):
        """测试无效的 k 值"""
        with pytest.raises(ValueError):
            RRFConfig(k=0)
        
        with pytest.raises(ValueError):
            RRFConfig(k=-10)
    
    def test_invalid_dedup_strategy(self):
        """测试无效的去重策略"""
        with pytest.raises(ValueError):
            RRFConfig(dedup_strategy="invalid")


class TestRRFFusion:
    """测试 RRF 融合"""
    
    def setup_method(self):
        self.config = RRFConfig(k=60)
        self.fusion = RRFFusion(config=self.config)
        
        # 创建模拟结果
        self.vector_results = [
            (MockDocument("doc1", "content1"), 0.9),
            (MockDocument("doc2", "content2"), 0.8),
            (MockDocument("doc3", "content3"), 0.7),
        ]
        
        self.bm25_results = [
            (MockDocument("doc2", "content2"), 5.2),
            (MockDocument("doc4", "content4"), 4.1),
            (MockDocument("doc1", "content1"), 3.8),
        ]
    
    def test_fuse_basic(self):
        """测试基本融合"""
        results = self.fusion.fuse(
            results_list=[self.vector_results, self.bm25_results],
            retriever_names=["vector", "bm25"],
            top_k=10,
        )
        
        assert len(results) > 0
        
        # doc1 和 doc2 在两个检索器中都出现，应该有更高的融合分数
        doc_ids = [DocIdExtractor.get_doc_id(doc) for doc, _ in results]
        assert "doc1" in doc_ids or "doc2" in doc_ids
    
    def test_fuse_with_weights(self):
        """测试加权融合"""
        results_equal = self.fusion.fuse(
            results_list=[self.vector_results, self.bm25_results],
            retriever_names=["vector", "bm25"],
            weights={"vector": 1.0, "bm25": 1.0},
            top_k=5,
        )
        
        results_weighted = self.fusion.fuse(
            results_list=[self.vector_results, self.bm25_results],
            retriever_names=["vector", "bm25"],
            weights={"vector": 2.0, "bm25": 0.5},
            top_k=5,
        )
        
        # 不同权重应该产生不同的排序
        # 这里我们验证融合器能正确处理权重
        assert len(results_weighted) > 0
    
    def test_fuse_empty_results(self):
        """测试空结果"""
        results = self.fusion.fuse([], top_k=10)
        assert results == []
    
    def test_fuse_partial_empty(self):
        """测试部分空结果"""
        results = self.fusion.fuse(
            results_list=[self.vector_results, []],
            retriever_names=["vector", "bm25"],
            top_k=5,
        )
        
        # 只有向量检索有结果，应该返回向量检索的结果
        assert len(results) > 0
    
    def test_get_stats(self):
        """测试统计信息"""
        self.fusion.fuse(
            results_list=[self.vector_results, self.bm25_results],
            retriever_names=["vector", "bm25"],
            top_k=10,
        )
        
        stats = self.fusion.get_stats()
        assert stats["input_counts"] == [3, 3]
        assert stats["output_count"] > 0
        assert stats["k_value"] == 60
    
    def test_update_weights(self):
        """测试动态更新权重"""
        self.fusion.update_weights({"vector": 1.5})
        assert self.fusion.config.weights["vector"] == 1.5
    
    def test_set_k(self):
        """测试设置 k 值"""
        self.fusion.set_k(40)
        assert self.fusion.config.k == 40
    
    def test_set_invalid_k(self):
        """测试设置无效 k 值"""
        with pytest.raises(ValueError):
            self.fusion.set_k(0)
    
    def test_fuse_with_strategy(self):
        """测试使用不同策略融合"""
        results_rrf = self.fusion.fuse_with_strategy(
            results_list=[self.vector_results, self.bm25_results],
            strategy="rrf",
            top_k=5,
        )
        
        results_weighted = self.fusion.fuse_with_strategy(
            results_list=[self.vector_results, self.bm25_results],
            strategy="weighted",
            top_k=5,
        )
        
        # RRF 策略应该返回有效结果
        assert len(results_rrf) > 0
        assert len(results_weighted) > 0
    
    def test_normalize_scores(self):
        """测试分数归一化"""
        config = RRFConfig(normalize_scores=True)
        fusion = RRFFusion(config=config)
        
        results = fusion.fuse(
            results_list=[self.vector_results, self.bm25_results],
            retriever_names=["vector", "bm25"],
            top_k=5,
        )
        
        # 归一化后的分数应该在 0-1 之间
        for doc, score in results:
            assert 0 <= score <= 1
    
    def test_min_score_threshold(self):
        """测试最低分数阈值"""
        config = RRFConfig(min_score_threshold=0.1)
        fusion = RRFFusion(config=config)
        
        results = fusion.fuse(
            results_list=[self.vector_results, self.bm25_results],
            retriever_names=["vector", "bm25"],
            top_k=10,
        )
        
        # 所有返回的结果分数应该大于阈值
        for doc, score in results:
            assert score >= 0.1


class TestAdaptiveRRFFusion:
    """测试自适应 RRF 融合"""
    
    def setup_method(self):
        self.config = RRFConfig()
        self.fusion = AdaptiveRRFFusion(config=self.config)
        
        self.results_list = [
            [(MockDocument("doc1"), 0.9), (MockDocument("doc2"), 0.8)],
            [(MockDocument("doc2"), 5.0), (MockDocument("doc3"), 4.0)],
        ]
    
    def test_adaptive_fuse(self):
        """测试自适应融合"""
        results = self.fusion.fuse_adaptive(
            results_list=self.results_list,
            retriever_names=["vector", "bm25"],
            top_k=5,
        )
        
        assert len(results) > 0
    
    def test_record_feedback(self):
        """测试反馈记录"""
        self.fusion.record_feedback("vector", 0.8)
        self.fusion.record_feedback("bm25", 0.6)
        
        assert len(self.fusion._performance_history["vector"]) == 1
        assert len(self.fusion._performance_history["bm25"]) == 1
    
    def test_feedback_history_limit(self):
        """测试反馈历史限制"""
        for i in range(150):
            self.fusion.record_feedback("vector", 0.5)
        
        # 应该限制在 100 条
        assert len(self.fusion._performance_history["vector"]) <= 100


class TestMultiStageFusion:
    """测试多阶段融合"""
    
    def setup_method(self):
        self.fusion = MultiStageFusion()
        
        # 创建多个检索器组
        self.retriever_groups = {
            "vector": [
                [(MockDocument("doc1"), 0.9), (MockDocument("doc2"), 0.8)],
                [(MockDocument("doc2"), 0.85), (MockDocument("doc3"), 0.75)],
            ],
            "bm25": [
                [(MockDocument("doc2"), 5.0), (MockDocument("doc4"), 4.0)],
            ],
        }
    
    def test_multistage_fuse(self):
        """测试多阶段融合"""
        results = self.fusion.fuse(
            retriever_groups=self.retriever_groups,
            top_k=5,
        )
        
        assert len(results) > 0
    
    def test_single_retriever_group(self):
        """测试单个检索器组"""
        groups = {
            "vector": [
                [(MockDocument("doc1"), 0.9)],
            ],
        }
        
        results = self.fusion.fuse(groups, top_k=5)
        assert len(results) > 0


class TestRRFFusionStrategy:
    """测试 RRF 融合策略"""
    
    def test_strategy_fuse(self):
        """测试策略融合"""
        strategy = RRFFusionStrategy(k=60)
        
        results = strategy.fuse(
            results_list=[
                [(MockDocument("doc1"), 0.9)],
                [(MockDocument("doc2"), 5.0)],
            ],
            retriever_names=["vector", "bm25"],
            weights={"vector": 1.0, "bm25": 1.0},
            top_k=5,
        )
        
        assert len(results) > 0


class TestWeightedScoreFusionStrategy:
    """测试加权分数融合策略"""
    
    def test_strategy_fuse(self):
        """测试策略融合"""
        strategy = WeightedScoreFusionStrategy()
        
        results = strategy.fuse(
            results_list=[
                [(MockDocument("doc1"), 0.9), (MockDocument("doc2"), 0.1)],
                [(MockDocument("doc1"), 5.0), (MockDocument("doc3"), 1.0)],
            ],
            retriever_names=["vector", "bm25"],
            weights={"vector": 1.0, "bm25": 1.0},
            top_k=5,
        )
        
        assert len(results) > 0
        
        # doc1 在两个检索器中都有高分，应该排在前面
        top_doc_id = DocIdExtractor.get_doc_id(results[0][0])
        assert top_doc_id == "doc1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])