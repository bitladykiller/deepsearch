"""
BM25 检索器单元测试

测试 BM25Retriever 的核心功能：
- 文档索引和检索
- 分词器功能
- 批量查询
- 权重调整
- 持久化
"""

import pytest
import tempfile
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.mult_agents.rag.bm25_retriever import (
    BM25Config,
    BM25Retriever,
    BM25Index,
    Document,
    SimpleTokenizer,
    ChineseTokenizer,
    DEFAULT_CHINESE_STOPWORDS,
    DEFAULT_ENGLISH_STOPWORDS,
)


class TestSimpleTokenizer:
    """测试简单分词器"""
    
    def setup_method(self):
        self.tokenizer = SimpleTokenizer(lowercase=True, enable_stopwords=True)
    
    def test_tokenize_english(self):
        """测试英文分词"""
        text = "Hello World Python Programming"
        tokens = self.tokenizer.tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens
        assert "programming" in tokens
    
    def test_tokenize_chinese(self):
        """测试中文分词"""
        text = "深度学习是人工智能的分支"
        tokens = self.tokenizer.tokenize(text)
        # 应该包含单字和双字词
        assert len(tokens) > 0
        # 不应该包含停用词
        assert "是" not in tokens
        assert "的" not in tokens
    
    def test_tokenize_mixed(self):
        """测试中英文混合"""
        text = "Python是一种编程语言"
        tokens = self.tokenizer.tokenize(text)
        assert "python" in tokens
        assert len(tokens) > 1
    
    def test_stopwords_filter(self):
        """测试停用词过滤"""
        text = "这是一个测试"
        tokens_with_stopwords = SimpleTokenizer(lowercase=True, enable_stopwords=False).tokenize(text)
        tokens_without_stopwords = self.tokenizer.tokenize(text)
        
        # 启用停用词过滤后，结果应该更少
        assert len(tokens_without_stopwords) <= len(tokens_with_stopwords)
    
    def test_add_stopwords(self):
        """测试添加停用词"""
        tokenizer = SimpleTokenizer(lowercase=True, enable_stopwords=True)
        tokenizer.add_stopwords("custom_stopword")
        
        tokens = tokenizer.tokenize("custom_stopword test")
        assert "custom_stopword" not in tokens
        assert "test" in tokens


class TestBM25Index:
    """测试 BM25 索引"""
    
    def test_build_index(self):
        """测试索引构建"""
        index = BM25Index(k1=1.5, b=0.75)
        corpus = [
            ["hello", "world"],
            ["python", "programming"],
            ["hello", "python"],
        ]
        index.build(corpus)
        
        assert index.doc_count == 3
        assert index.avgdl > 0
        assert "hello" in index.doc_freqs
        assert index.doc_freqs["hello"] == 2  # 出现在 2 个文档中
    
    def test_idf_computation(self):
        """测试 IDF 计算"""
        index = BM25Index()
        corpus = [["term1", "term2"], ["term1", "term3"]]
        index.build(corpus)
        
        # term1 出现在所有文档，IDF 应该较低
        # term2 只出现在一个文档，IDF 应该较高
        idf_term1 = index.get_idf("term1")
        idf_term2 = index.get_idf("term2")
        
        # 出现频率低的词 IDF 应该更高
        assert idf_term2 > idf_term1
    
    def test_get_scores(self):
        """测试分数计算"""
        index = BM25Index()
        corpus = [
            ["python", "programming", "language"],
            ["java", "programming", "language"],
            ["python", "data", "science"],
        ]
        index.build(corpus)
        
        # 查询包含 python
        scores = index.get_scores(["python"])
        
        # 应该返回 3 个文档的分数
        assert len(scores) == 3
        # 第一个和第三个文档包含 python，分数应该更高
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]
    
    def test_batch_scores(self):
        """测试批量分数计算"""
        index = BM25Index()
        corpus = [["python", "programming"], ["java", "development"]]
        index.build(corpus)
        
        queries = [["python"], ["java"]]
        scores = index.get_batch_scores(queries)
        
        assert scores.shape == (2, 2)
        assert scores[0, 0] > scores[0, 1]  # python query 匹配第一个文档
        assert scores[1, 1] > scores[1, 0]  # java query 匹配第二个文档


class TestBM25Retriever:
    """测试 BM25 检索器"""
    
    def setup_method(self):
        self.config = BM25Config(tokenizer="simple", k1=1.5, b=0.75)
        self.retriever = BM25Retriever(config=self.config)
        
        # 添加测试文档
        self.documents = [
            Document(doc_id="doc1", content="Python is a programming language"),
            Document(doc_id="doc2", content="Java is also a programming language"),
            Document(doc_id="doc3", content="Machine learning uses Python"),
        ]
        self.retriever.add_documents(self.documents)
    
    def test_add_documents(self):
        """测试添加文档"""
        assert len(self.retriever) == 3
        assert "doc1" in self.retriever
    
    def test_search(self):
        """测试检索功能"""
        results = self.retriever.search("Python programming", top_k=2)
        
        assert len(results) <= 2
        assert len(results) > 0
        
        # 检查结果格式
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert score > 0
    
    def test_search_chinese(self):
        """测试中文检索"""
        config = BM25Config(tokenizer="simple", enable_stopwords=True)
        retriever = BM25Retriever(config=config)
        
        docs = [
            Document(doc_id="c1", content="深度学习是人工智能的分支"),
            Document(doc_id="c2", content="机器学习使用神经网络"),
        ]
        retriever.add_documents(docs)
        
        results = retriever.search("深度学习", top_k=2)
        assert len(results) > 0
    
    def test_batch_search(self):
        """测试批量检索"""
        queries = ["Python", "Java", "Machine learning"]
        results = self.retriever.batch_search(queries, top_k=2)
        
        assert len(results) == 3
        for query_results in results:
            assert isinstance(query_results, list)
    
    def test_query_weights(self):
        """测试查询词权重"""
        # 给 "python" 更高的权重
        results = self.retriever.search(
            "Python programming",
            top_k=3,
            query_weights={"python": 2.0}
        )
        
        # 包含 python 的文档应该排在前面
        if results:
            top_doc = results[0][0]
            assert "python" in top_doc.content.lower()
    
    def test_get_stats(self):
        """测试统计信息"""
        stats = self.retriever.get_stats()
        
        assert stats["doc_count"] == 3
        assert stats["vocab_size"] > 0
        assert stats["avg_doc_length"] > 0
        assert stats["is_built"] is True
    
    def test_persistence(self):
        """测试持久化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bm25_index.pkl"
            
            # 保存
            self.retriever.save(str(path))
            assert path.exists()
            
            # 加载
            new_retriever = BM25Retriever(config=self.config)
            new_retriever.load(str(path))
            
            assert len(new_retriever) == 3
            
            # 验证检索结果一致
            results1 = self.retriever.search("Python", top_k=2)
            results2 = new_retriever.search("Python", top_k=2)
            
            assert len(results1) == len(results2)
    
    def test_remove_document(self):
        """测试删除文档"""
        assert self.retriever.remove_document("doc1")
        assert len(self.retriever) == 2
        assert "doc1" not in self.retriever
    
    def test_get_document(self):
        """测试获取文档"""
        doc = self.retriever.get_document("doc1")
        assert doc is not None
        assert doc.doc_id == "doc1"
        
        assert self.retriever.get_document("nonexistent") is None
    
    def test_clear(self):
        """测试清空"""
        self.retriever.clear()
        assert len(self.retriever) == 0
        assert not self.retriever._built


class TestDocument:
    """测试文档类"""
    
    def test_document_creation(self):
        """测试文档创建"""
        doc = Document(
            doc_id="test",
            content="Test content",
            metadata={"key": "value"}
        )
        
        assert doc.doc_id == "test"
        assert doc.content == "Test content"
        assert doc.metadata["key"] == "value"
    
    def test_document_hash_and_eq(self):
        """测试文档哈希和相等性"""
        doc1 = Document(doc_id="test", content="content1")
        doc2 = Document(doc_id="test", content="content2")
        doc3 = Document(doc_id="other", content="content1")
        
        assert hash(doc1) == hash(doc2)
        assert doc1 == doc2
        assert doc1 != doc3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
