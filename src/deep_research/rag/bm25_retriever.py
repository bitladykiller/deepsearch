"""
BM25 检索器实现

基于 BM25 算法的稀疏检索实现，用于关键词匹配和词汇检索。
实现了完整的 BM25+ 变体，支持：
- IDF 缓存优化
- 批量查询
- 查询词权重调整
- 多种分词器
- 停用词过滤
"""

import logging
import math
import pickle
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ //
# 配置类
# ------------------------------------------------------------------ //


@dataclass
class BM25Config:
    """
    BM25 检索器配置
    
    Attributes:
        tokenizer: 分词器类型，可选 "chinese", "simple", "mixed"
        k1: BM25 参数，控制词频饱和度，推荐值 1.2-2.0
        b: BM25 参数，控制文档长度归一化，推荐值 0.75
        epsilon: IDF 平滑参数，避免除零
        index_path: 索引存储路径
        lowercase: 是否转换为小写
        enable_stopwords: 是否启用停用词过滤
        stopwords: 自定义停用词列表
        min_df: 最小文档频率，低于此值的词将被过滤
        max_df: 最大文档频率，高于此值的词将被过滤（百分比）
    """
    tokenizer: str = "chinese"
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    index_path: Optional[str] = None
    lowercase: bool = True
    enable_stopwords: bool = True
    stopwords: Optional[list[str]] = None
    min_df: int = 1
    max_df: float = 0.95


# ------------------------------------------------------------------ //
# 停用词表
# ------------------------------------------------------------------ //

DEFAULT_CHINESE_STOPWORDS = {
    # 常用虚词
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "她", "他", "它", "们", "这个", "那个", "什么", "怎么",
    "为", "与", "及", "等", "或", "且", "但", "如", "而", "因", "所以", "因为",
    "如果", "虽然", "但是", "然而", "不过", "只是", "可以", "可能", "应该", "需要",
    # 标点符号
    "，", "。", "！", "？", "；", "：", """, """, "'", "'", "（", "）", "【", "】",
    # 其他常见停用词
    "之", "者", "所", "以", "于", "其", "乃", "乎", "矣", "焉", "哉", "兮",
}

DEFAULT_ENGLISH_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "dare", "ought", "used",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between", "under",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither", "not",
    "only", "own", "same", "than", "too", "very", "just", "also",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "been", "being", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "no", "any", "about", "then", "once", "now",
}


# ------------------------------------------------------------------ //
# 分词器基类和实现
# ------------------------------------------------------------------ //


class BaseTokenizer(ABC):
    """分词器基类"""
    
    def __init__(
        self,
        lowercase: bool = True,
        stopwords: Optional[set[str]] = None,
        enable_stopwords: bool = True,
    ):
        self.lowercase = lowercase
        self.enable_stopwords = enable_stopwords
        self._stopwords = stopwords or self._get_default_stopwords()
    
    def _get_default_stopwords(self) -> set[str]:
        """获取默认停用词表"""
        return DEFAULT_CHINESE_STOPWORDS | DEFAULT_ENGLISH_STOPWORDS
    
    @property
    def stopwords(self) -> set[str]:
        """获取当前停用词表"""
        if not self.enable_stopwords:
            return set()
        return self._stopwords
    
    def add_stopwords(self, words: Union[str, list[str]]) -> None:
        """添加停用词"""
        if isinstance(words, str):
            self._stopwords.add(words.lower() if self.lowercase else words)
        else:
            for w in words:
                self._stopwords.add(w.lower() if self.lowercase else w)
    
    def remove_stopwords(self, words: Union[str, list[str]]) -> None:
        """移除停用词"""
        if isinstance(words, str):
            self._stopwords.discard(words.lower() if self.lowercase else words)
        else:
            for w in words:
                self._stopwords.discard(w.lower() if self.lowercase else w)
    
    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        """过滤停用词和空白词"""
        return [t for t in tokens if t and t not in self.stopwords]
    
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """分词方法，子类必须实现"""
        pass


class SimpleTokenizer(BaseTokenizer):
    """
    简单分词器（中英文通用）
    
    使用正则表达式进行分词，支持：
    - 英文单词提取
    - 中文双字切分（滑动窗口）
    - 数字提取
    - 停用词过滤
    """
    
    def tokenize(self, text: str) -> list[str]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        if self.lowercase:
            text = text.lower()
        
        import re
        
        tokens = []
        
        # 提取英文单词
        english_words = re.findall(r"[a-zA-Z]+", text)
        tokens.extend(english_words)
        
        # 中文分词：使用滑动窗口提取双字词
        chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)
        for chars in chinese_chars:
            # 单字
            for char in chars:
                tokens.append(char)
            # 双字词（滑动窗口）
            for i in range(len(chars) - 1):
                token = chars[i:i + 2]
                tokens.append(token)
            # 三字词（可选）
            for i in range(len(chars) - 2):
                token = chars[i:i + 3]
                tokens.append(token)
        
        # 提取数字
        numbers = re.findall(r"\d+", text)
        tokens.extend(numbers)
        
        # 过滤停用词
        return self._filter_tokens(tokens)


class ChineseTokenizer(BaseTokenizer):
    """
    中文分词器（使用 jieba）
    
    特点：
    - 使用 jieba 进行精确分词
    - 支持自定义词典
    - 支持停用词过滤
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        stopwords: Optional[set[str]] = None,
        enable_stopwords: bool = True,
        user_dict: Optional[str] = None,
    ):
        super().__init__(lowercase, stopwords, enable_stopwords)
        self.user_dict = user_dict
        self._jieba = None
        self._initialized = False
    
    @property
    def jieba(self):
        """延迟加载 jieba"""
        if self._jieba is None:
            try:
                import jieba
                self._jieba = jieba
                if self.user_dict:
                    self._jieba.load_userdict(self.user_dict)
                self._initialized = True
            except ImportError:
                raise ImportError("请安装 jieba: pip install jieba")
        return self._jieba
    
    def tokenize(self, text: str) -> list[str]:
        """
        使用 jieba 进行中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        if self.lowercase:
            text = text.lower()
        
        tokens = list(self.jieba.cut(text))
        tokens = [t.strip() for t in tokens if t.strip()]
        
        return self._filter_tokens(tokens)


class MixedTokenizer(BaseTokenizer):
    """
    混合分词器
    
    结合 jieba 和简单分词器，支持更精细的中文分词：
    - 使用 jieba 进行主要分词
    - 保留英文单词的完整性
    - 支持停用词过滤
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        stopwords: Optional[set[str]] = None,
        enable_stopwords: bool = True,
        user_dict: Optional[str] = None,
    ):
        super().__init__(lowercase, stopwords, enable_stopwords)
        self._jieba_tokenizer = None
        self.user_dict = user_dict
    
    @property
    def jieba_tokenizer(self) -> ChineseTokenizer:
        """延迟初始化 jieba 分词器"""
        if self._jieba_tokenizer is None:
            self._jieba_tokenizer = ChineseTokenizer(
                lowercase=self.lowercase,
                stopwords=self._stopwords,
                enable_stopwords=False,  # 最后统一过滤
                user_dict=self.user_dict,
            )
        return self._jieba_tokenizer
    
    def tokenize(self, text: str) -> list[str]:
        """
        混合分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        if self.lowercase:
            text = text.lower()
        
        import re
        
        tokens = []
        
        # 分离中英文
        # 英文部分
        english_parts = re.findall(r"[a-zA-Z]+", text)
        tokens.extend(english_parts)
        
        # 中文部分使用 jieba
        chinese_parts = re.findall(r"[\u4e00-\u9fff]+", text)
        for part in chinese_parts:
            tokens.extend(self.jieba_tokenizer.tokenize(part))
        
        # 数字
        numbers = re.findall(r"\d+", text)
        tokens.extend(numbers)
        
        return self._filter_tokens(tokens)


# ------------------------------------------------------------------ //
# 文档类
# ------------------------------------------------------------------ //


@dataclass
class Document:
    """
    文档对象
    
    Attributes:
        doc_id: 文档唯一标识符
        content: 文档内容
        metadata: 文档元数据字典
    """
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.doc_id)
    
    def __eq__(self, other):
        if isinstance(other, Document):
            return self.doc_id == other.doc_id
        return False


# ------------------------------------------------------------------ //
# BM25 索引类
# ------------------------------------------------------------------ //


class BM25Index:
    """
    BM25 索引类
    
    实现完整的 BM25+ 算法，包括：
    - 文档频率 (DF) 统计
    - 逆文档频率 (IDF) 计算
    - 文档长度归一化
    - IDF 缓存
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """
        初始化 BM25 索引
        
        Args:
            k1: 词频饱和参数
            b: 文档长度归一化参数
            epsilon: IDF 平滑参数
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # 索引数据
        self.doc_count: int = 0
        self.avgdl: float = 0.0  # 平均文档长度
        self.doc_lengths: list[int] = []
        self.doc_freqs: dict[str, int] = {}  # 词 -> 文档频率
        self.term_freqs: list[dict[str, int]] = []  # 每个文档的词频
        self.idf_cache: dict[str, float] = {}  # IDF 缓存
        self._idf_updated: bool = False
    
    def build(self, tokenized_corpus: list[list[str]]) -> None:
        """
        构建索引
        
        Args:
            tokenized_corpus: 分词后的文档列表
        """
        self.doc_count = len(tokenized_corpus)
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0
        
        self.doc_freqs = {}
        self.term_freqs = []
        
        for doc_tokens in tokenized_corpus:
            # 计算当前文档的词频
            tf = Counter(doc_tokens)
            self.term_freqs.append(dict(tf))
            
            # 更新文档频率
            for term in set(doc_tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        # 计算 IDF
        self._compute_idf()
        self._idf_updated = True
        
        logger.info(f"BM25 索引构建完成: {self.doc_count} 文档, {len(self.doc_freqs)} 词项")
    
    def _compute_idf(self) -> None:
        """计算所有词的 IDF 值"""
        self.idf_cache = {}
        for term, df in self.doc_freqs.items():
            self.idf_cache[term] = self._compute_idf_for_term(df)
    
    def _compute_idf_for_term(self, df: int) -> float:
        """
        计算单个词的 IDF 值（使用 BM25+ 变体）
        
        IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        
        这种变体避免了负 IDF 值
        
        Args:
            df: 文档频率
            
        Returns:
            IDF 值
        """
        n = self.doc_count
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        return max(idf, self.epsilon)  # 保证非负
    
    def get_idf(self, term: str) -> float:
        """
        获取词的 IDF 值
        
        Args:
            term: 词项
            
        Returns:
            IDF 值，如果词不在词表中返回默认值
        """
        if term in self.idf_cache:
            return self.idf_cache[term]
        # 对于未见过的词，返回最大 IDF（假设 df = 1）
        return self._compute_idf_for_term(1)
    
    def get_scores(self, query_tokens: list[str], doc_weights: Optional[list[float]] = None) -> np.ndarray:
        """
        计算查询与所有文档的 BM25 分数
        
        BM25 公式:
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
        
        Args:
            query_tokens: 查询分词结果
            doc_weights: 可选的文档权重（用于加权）
            
        Returns:
            各文档的 BM25 分数数组
        """
        scores = np.zeros(self.doc_count)
        
        # 查询词频
        query_tf = Counter(query_tokens)
        
        for term, query_freq in query_tf.items():
            idf = self.get_idf(term)
            
            for doc_idx in range(self.doc_count):
                tf = self.term_freqs[doc_idx].get(term, 0)
                if tf == 0:
                    continue
                
                doc_len = self.doc_lengths[doc_idx]
                # BM25 分数计算
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score = idf * numerator / denominator
                
                # 查询词权重：query_freq 次出现的词有额外权重
                score *= (1 + math.log(query_freq))
                
                scores[doc_idx] += score
        
        # 应用文档权重
        if doc_weights is not None:
            scores *= np.array(doc_weights)
        
        return scores
    
    def get_batch_scores(self, query_tokens_list: list[list[str]]) -> np.ndarray:
        """
        批量计算多个查询的分数（向量化优化）
        
        Args:
            query_tokens_list: 多个查询的分词结果列表
            
        Returns:
            二维数组，shape=(num_queries, num_docs)
        """
        num_queries = len(query_tokens_list)
        scores = np.zeros((num_queries, self.doc_count))
        
        for i, query_tokens in enumerate(query_tokens_list):
            scores[i] = self.get_scores(query_tokens)
        
        return scores
    
    def clear(self) -> None:
        """清空索引"""
        self.doc_count = 0
        self.avgdl = 0.0
        self.doc_lengths = []
        self.doc_freqs = {}
        self.term_freqs = []
        self.idf_cache = {}
        self._idf_updated = False


# ------------------------------------------------------------------ //
# BM25 检索器
# ------------------------------------------------------------------ //


class BM25Retriever:
    """
    BM25 检索器
    
    提供完整的 BM25 检索功能，包括：
    - 文档索引和检索
    - 批量查询
    - 查询词权重调整
    - 索引持久化
    """
    
    def __init__(self, config: Optional[BM25Config] = None):
        """
        初始化 BM25 检索器
        
        Args:
            config: BM25 配置对象
        """
        self.config = config or BM25Config()
        self.documents: list[Document] = []
        self.doc_ids: list[str] = []
        self.index = BM25Index(
            k1=self.config.k1,
            b=self.config.b,
            epsilon=self.config.epsilon,
        )
        self.tokenizer = self._create_tokenizer()
        self._built: bool = False
    
    def _create_tokenizer(self) -> BaseTokenizer:
        """创建分词器"""
        tokenizer_type = self.config.tokenizer.lower()
        
        common_kwargs = {
            "lowercase": self.config.lowercase,
            "stopwords": set(self.config.stopwords) if self.config.stopwords else None,
            "enable_stopwords": self.config.enable_stopwords,
        }
        
        if tokenizer_type == "chinese":
            try:
                return ChineseTokenizer(**common_kwargs)
            except ImportError:
                logger.warning("jieba 未安装，回退到简单分词器")
                return SimpleTokenizer(**common_kwargs)
        elif tokenizer_type == "mixed":
            try:
                return MixedTokenizer(**common_kwargs)
            except ImportError:
                return SimpleTokenizer(**common_kwargs)
        else:
            return SimpleTokenizer(**common_kwargs)
    
    # ------------------------------------------------------------------ //
    # 文档管理
    # ------------------------------------------------------------------ //
    
    def add_documents(self, documents: list[Document]) -> int:
        """
        添加文档到检索器
        
        Args:
            documents: 文档列表
            
        Returns:
            添加的文档数量
        """
        for doc in documents:
            self.documents.append(doc)
            self.doc_ids.append(doc.doc_id)
        
        self._build_index()
        logger.info(f"添加 {len(documents)} 个文档，总文档数: {len(self.documents)}")
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
            metadatas: 元数据列表（可选）
            doc_ids: 文档 ID 列表（可选）
            
        Returns:
            添加的文档数量
        """
        metadatas = metadatas or [{} for _ in texts]
        doc_ids = doc_ids or [f"doc-{i}" for i in range(len(self.documents), len(self.documents) + len(texts))]
        
        documents = [
            Document(doc_id=doc_id, content=text, metadata=meta)
            for doc_id, text, meta in zip(doc_ids, texts, metadatas)
        ]
        return self.add_documents(documents)
    
    def remove_document(self, doc_id: str) -> bool:
        """
        移除文档
        
        Args:
            doc_id: 文档 ID
            
        Returns:
            是否成功移除
        """
        if doc_id not in self.doc_ids:
            return False
        
        idx = self.doc_ids.index(doc_id)
        self.documents.pop(idx)
        self.doc_ids.pop(idx)
        self._build_index()
        return True
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        获取文档
        
        Args:
            doc_id: 文档 ID
            
        Returns:
            文档对象，不存在返回 None
        """
        if doc_id in self.doc_ids:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        return None
    
    # ------------------------------------------------------------------ //
    # 索引构建
    # ------------------------------------------------------------------ //
    
    def _build_index(self) -> None:
        """构建 BM25 索引"""
        if not self.documents:
            return
        
        # 分词
        tokenized_corpus = [
            self.tokenizer.tokenize(doc.content) for doc in self.documents
        ]
        
        # 过滤低频和高频词（可选）
        if self.config.min_df > 1 or self.config.max_df < 1.0:
            tokenized_corpus = self._filter_by_df(tokenized_corpus)
        
        # 构建索引
        self.index.build(tokenized_corpus)
        self._built = True
    
    def _filter_by_df(self, tokenized_corpus: list[list[str]]) -> list[list[str]]:
        """
        根据文档频率过滤词项
        
        Args:
            tokenized_corpus: 分词后的文档列表
            
        Returns:
            过滤后的文档列表
        """
        # 计算文档频率
        df = Counter()
        for doc_tokens in tokenized_corpus:
            for term in set(doc_tokens):
                df[term] += 1
        
        n_docs = len(tokenized_corpus)
        max_df_count = int(self.config.max_df * n_docs)
        
        # 确定要保留的词
        valid_terms = {
            term for term, count in df.items()
            if self.config.min_df <= count <= max_df_count
        }
        
        # 过滤
        return [
            [term for term in doc_tokens if term in valid_terms]
            for doc_tokens in tokenized_corpus
        ]
    
    # ------------------------------------------------------------------ //
    # 检索
    # ------------------------------------------------------------------ //
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        query_weights: Optional[dict[str, float]] = None,
    ) -> list[tuple[Document, float]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            query_weights: 查询词权重字典，用于调整某些词的重要性
            
        Returns:
            (文档, 分数) 元组列表，按分数降序排列
        """
        if not self._built:
            logger.warning("索引未构建，无法检索")
            return []
        
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []
        
        # 计算分数
        scores = self.index.get_scores(query_tokens)
        
        # 应用查询词权重
        if query_weights:
            scores = self._apply_query_weights(query_tokens, scores, query_weights)
        
        # 排序并返回 top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def batch_search(
        self,
        queries: list[str],
        top_k: int = 10,
    ) -> list[list[tuple[Document, float]]]:
        """
        批量检索（优化版）
        
        Args:
            queries: 查询文本列表
            top_k: 每个查询返回的文档数量
            
        Returns:
            每个查询的检索结果列表
        """
        if not self._built:
            return [[] for _ in queries]
        
        # 分词
        tokenized_queries = [self.tokenizer.tokenize(q) for q in queries]
        
        # 批量计算分数
        all_scores = self.index.get_batch_scores(tokenized_queries)
        
        # 对每个查询排序
        results = []
        for i, scores in enumerate(all_scores):
            top_indices = np.argsort(scores)[::-1][:top_k]
            query_results = [
                (self.documents[idx], float(scores[idx]))
                for idx in top_indices
                if scores[idx] > 0
            ]
            results.append(query_results)
        
        return results
    
    def _apply_query_weights(
        self,
        query_tokens: list[str],
        scores: np.ndarray,
        query_weights: dict[str, float],
    ) -> np.ndarray:
        """
        应用查询词权重
        
        Args:
            query_tokens: 查询分词结果
            scores: 原始分数
            query_weights: 词权重字典
            
        Returns:
            加权后的分数
        """
        weighted_scores = scores.copy()
        
        for term, weight in query_weights.items():
            if term in query_tokens:
                # 对包含该词的文档增加权重
                for doc_idx, tf_dict in enumerate(self.index.term_freqs):
                    if term in tf_dict:
                        weighted_scores[doc_idx] *= weight
        
        return weighted_scores
    
    # ------------------------------------------------------------------ //
    # 持久化
    # ------------------------------------------------------------------ //
    
    def save(self, path: Optional[str] = None) -> None:
        """
        保存索引到文件
        
        Args:
            path: 保存路径，未指定则使用配置中的路径
        """
        path = path or self.config.index_path
        if not path:
            raise ValueError("未指定保存路径")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
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
        
        logger.info(f"BM25 索引已保存到: {path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        从文件加载索引
        
        Args:
            path: 文件路径，未指定则使用配置中的路径
        """
        path = path or self.config.index_path
        if not path:
            raise ValueError("未指定加载路径")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"索引文件不存在: {path}")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.config = data["config"]
        self.documents = data["documents"]
        self.doc_ids = data["doc_ids"]
        
        # 恢复索引
        self.index.doc_count = data["index"]["doc_count"]
        self.index.avgdl = data["index"]["avgdl"]
        self.index.doc_lengths = data["index"]["doc_lengths"]
        self.index.doc_freqs = data["index"]["doc_freqs"]
        self.index.term_freqs = data["index"]["term_freqs"]
        self.index.idf_cache = data["index"]["idf_cache"]
        
        self._built = True
        self.tokenizer = self._create_tokenizer()
        
        logger.info(f"BM25 索引已加载，文档数: {len(self.documents)}")
    
    def clear(self) -> None:
        """清空所有数据"""
        self.documents = []
        self.doc_ids = []
        self.index.clear()
        self._built = False
    
    # ------------------------------------------------------------------ //
    # 统计信息
    # ------------------------------------------------------------------ //
    
    def get_stats(self) -> dict:
        """
        获取索引统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "doc_count": len(self.documents),
            "vocab_size": len(self.index.doc_freqs),
            "avg_doc_length": self.index.avgdl,
            "total_terms": sum(self.index.doc_lengths),
            "is_built": self._built,
        }
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.doc_ids
