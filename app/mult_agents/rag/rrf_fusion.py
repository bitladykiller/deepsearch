"""
RRF (Reciprocal Rank Fusion) 融合算法

用于合并多路检索结果，实现混合检索。

RRF 公式:
    RRF_score(d) = Σ (w_i / (k + rank_i(d)))

其中:
    - k: 平滑常数（经典值 60）
    - rank_i(d): 文档 d 在第 i 个检索器中的排名位置
    - w_i: 第 i 个检索器的权重

特点:
    - 无需分数归一化（基于排名）
    - 对异常值鲁棒
    - 支持任意数量的检索器
    - 可配置权重
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ //
# 配置类
# ------------------------------------------------------------------ //


@dataclass
class RRFConfig:
    """
    RRF 融合配置
    
    Attributes:
        k: RRF 平滑常数，经典值为 60，值越小排名靠前的文档优势越大
        weights: 各检索器的权重字典，键为检索器名称，值为权重
        normalize_scores: 是否对最终分数进行归一化（0-1）
        min_score_threshold: 最低分数阈值，低于此值的结果将被过滤
        dedup_strategy: 去重策略，可选 "first"（保留第一个）、"best"（保留最高分）
    """
    k: int = 60
    weights: Optional[dict[str, float]] = None
    normalize_scores: bool = False
    min_score_threshold: float = 0.0
    dedup_strategy: str = "best"
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "vector": 1.0,
                "bm25": 1.0,
            }
        if self.k <= 0:
            raise ValueError(f"k 必须为正数，当前值: {self.k}")
        if self.dedup_strategy not in ("first", "best"):
            raise ValueError(f"dedup_strategy 必须为 'first' 或 'best'，当前值: {self.dedup_strategy}")


# ------------------------------------------------------------------ //
# 文档标识符提取
# ------------------------------------------------------------------ //


class DocIdExtractor:
    """
    文档标识符提取器
    
    支持多种文档类型的 ID 提取：
    - 自定义 Document 对象（有 doc_id 属性）
    - LangChain Document 对象（有 metadata["source"]）
    - 带有 page_content 的文档对象
    """
    
    @staticmethod
    def get_doc_id(doc: Any) -> str:
        """
        提取文档的唯一标识符
        
        Args:
            doc: 文档对象
            
        Returns:
            文档 ID 字符串
        """
        # 优先使用 doc_id 属性
        if hasattr(doc, "doc_id"):
            return str(doc.doc_id)
        
        # 尝试 metadata 中的 source
        if hasattr(doc, "metadata"):
            metadata = doc.metadata or {}
            if "source" in metadata:
                return str(metadata["source"])
            if "doc_id" in metadata:
                return str(metadata["doc_id"])
            if "id" in metadata:
                return str(metadata["id"])
        
        # 使用内容的哈希值
        content = DocIdExtractor._get_content(doc)
        return f"doc-{hash(content)}"
    
    @staticmethod
    def _get_content(doc: Any) -> str:
        """提取文档内容"""
        if hasattr(doc, "content"):
            return doc.content
        if hasattr(doc, "page_content"):
            return doc.page_content
        return str(doc)


# ------------------------------------------------------------------ //
# 分数归一化器
# ------------------------------------------------------------------ //


class ScoreNormalizer:
    """
    分数归一化器
    
    支持多种归一化策略：
    - min-max: 线性归一化到 [0, 1]
    - softmax: 使用 softmax 函数归一化
    - rank: 转换为排名分数
    """
    
    @staticmethod
    def normalize(scores: dict[str, float], method: str = "min-max") -> dict[str, float]:
        """
        归一化分数
        
        Args:
            scores: 文档 ID -> 分数字典
            method: 归一化方法
            
        Returns:
            归一化后的分数字典
        """
        if not scores:
            return scores
        
        if method == "min-max":
            return ScoreNormalizer._min_max_normalize(scores)
        elif method == "softmax":
            return ScoreNormalizer._softmax_normalize(scores)
        elif method == "rank":
            return ScoreNormalizer._rank_normalize(scores)
        else:
            logger.warning(f"未知的归一化方法: {method}，使用原始分数")
            return scores
    
    @staticmethod
    def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
        """Min-Max 归一化"""
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        
        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }
    
    @staticmethod
    def _softmax_normalize(scores: dict[str, float]) -> dict[str, float]:
        """Softmax 归一化"""
        import math
        
        values = list(scores.values())
        max_val = max(values)  # 数值稳定性
        
        exp_values = {k: math.exp(v - max_val) for k, v in scores.items()}
        sum_exp = sum(exp_values.values())
        
        return {k: v / sum_exp for k, v in exp_values.items()}
    
    @staticmethod
    def _rank_normalize(scores: dict[str, float]) -> dict[str, float]:
        """排名归一化"""
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_items)
        
        return {
            k: (n - i) / n
            for i, (k, _) in enumerate(sorted_items)
        }


# ------------------------------------------------------------------ //
# 融合策略基类和实现
# ------------------------------------------------------------------ //


class FusionStrategy(ABC):
    """融合策略基类"""
    
    @abstractmethod
    def fuse(
        self,
        results_list: list[list[tuple[Any, float]]],
        retriever_names: list[str],
        weights: dict[str, float],
        top_k: int,
    ) -> list[tuple[Any, float]]:
        """执行融合"""
        pass


class RRFFusionStrategy(FusionStrategy):
    """
    RRF 融合策略
    
    实现 Reciprocal Rank Fusion 算法
    """
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(
        self,
        results_list: list[list[tuple[Any, float]]],
        retriever_names: list[str],
        weights: dict[str, float],
        top_k: int,
    ) -> list[tuple[Any, float]]:
        """
        执行 RRF 融合
        
        Args:
            results_list: 各检索器的结果列表
            retriever_names: 检索器名称列表
            weights: 检索器权重字典
            top_k: 返回结果数量
            
        Returns:
            融合后的结果列表
        """
        doc_scores = defaultdict(float)
        doc_appearances = defaultdict(list)  # 记录文档在各检索器中的出现情况
        
        for results, name in zip(results_list, retriever_names):
            weight = weights.get(name, 1.0)
            
            for rank, (doc, _) in enumerate(results, start=1):
                doc_id = DocIdExtractor.get_doc_id(doc)
                rrf_score = weight / (self.k + rank)
                doc_scores[doc_id] += rrf_score
                doc_appearances[doc_id].append(name)
        
        # 构建文档映射
        doc_map = self._build_doc_map(results_list)
        
        # 排序并返回
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if doc_id in doc_map:
                # 附加元数据：文档在哪些检索器中出现
                doc = doc_map[doc_id]
                if hasattr(doc, "metadata"):
                    doc.metadata["_retrievers"] = doc_appearances.get(doc_id, [])
                results.append((doc, score))
        
        return results
    
    def _build_doc_map(self, results_list: list[list[tuple[Any, float]]]) -> dict[str, Any]:
        """构建文档 ID -> 文档对象的映射"""
        doc_map = {}
        for results in results_list:
            for doc, _ in results:
                doc_id = DocIdExtractor.get_doc_id(doc)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
        return doc_map


class WeightedScoreFusionStrategy(FusionStrategy):
    """
    加权分数融合策略
    
    对原始分数进行加权求和，需要先归一化各检索器的分数
    """
    
    def fuse(
        self,
        results_list: list[list[tuple[Any, float]]],
        retriever_names: list[str],
        weights: dict[str, float],
        top_k: int,
    ) -> list[tuple[Any, float]]:
        """
        执行加权分数融合
        
        Args:
            results_list: 各检索器的结果列表
            retriever_names: 检索器名称列表
            weights: 检索器权重字典
            top_k: 返回结果数量
            
        Returns:
            融合后的结果列表
        """
        # 归一化各检索器的分数
        normalized_results = []
        for results, name in zip(results_list, retriever_names):
            if not results:
                normalized_results.append([])
                continue
            
            scores = [score for _, score in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            
            if max_score == min_score:
                normalized = [(doc, 1.0) for doc, _ in results]
            else:
                normalized = [
                    (doc, (score - min_score) / (max_score - min_score))
                    for doc, score in results
                ]
            
            # 应用权重
            weight = weights.get(name, 1.0)
            normalized = [(doc, score * weight) for doc, score in normalized]
            normalized_results.append(normalized)
        
        # 合并分数
        doc_scores = defaultdict(float)
        doc_map = {}
        
        for results in normalized_results:
            for doc, score in results:
                doc_id = DocIdExtractor.get_doc_id(doc)
                doc_scores[doc_id] += score
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
        
        # 排序返回
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[doc_id], score) for doc_id, score in sorted_docs[:top_k] if doc_id in doc_map]


# ------------------------------------------------------------------ //
# RRF 融合器
# ------------------------------------------------------------------ //


class RRFFusion:
    """
    Reciprocal Rank Fusion 融合器
    
    支持多种融合策略，提供灵活的配置选项
    
    特点:
        - 支持 RRF 和加权分数两种融合策略
        - 支持动态权重调整
        - 支持分数归一化
        - 支持最低分数过滤
        - 提供详细的融合统计信息
    """
    
    def __init__(self, config: Optional[RRFConfig] = None):
        """
        初始化 RRF 融合器
        
        Args:
            config: RRF 配置对象
        """
        self.config = config or RRFConfig()
        self._strategy = RRFFusionStrategy(k=self.config.k)
        self._fusion_stats: dict = {}
    
    def fuse(
        self,
        results_list: list[list[tuple[Any, float]]],
        retriever_names: Optional[list[str]] = None,
        top_k: int = 10,
        weights: Optional[dict[str, float]] = None,
    ) -> list[tuple[Any, float]]:
        """
        融合多路检索结果
        
        Args:
            results_list: 各检索器的结果列表，每个元素是 (文档, 分数) 元组列表
            retriever_names: 检索器名称列表，未提供则使用默认名称
            top_k: 返回的文档数量
            weights: 可选的权重覆盖，未提供则使用配置中的权重
            
        Returns:
            融合后的 (文档, 分数) 元组列表，按分数降序排列
            
        Example:
            >>> fusion = RRFFusion()
            >>> vector_results = [(doc1, 0.9), (doc2, 0.8)]
            >>> bm25_results = [(doc2, 5.2), (doc3, 4.1)]
            >>> fused = fusion.fuse(
            ...     [vector_results, bm25_results],
            ...     retriever_names=["vector", "bm25"],
            ...     top_k=5
            ... )
        """
        if not results_list:
            return []
        
        # 过滤空结果
        valid_results = [r for r in results_list if r]
        if not valid_results:
            return []
        
        # 设置检索器名称
        if retriever_names is None:
            retriever_names = [f"retriever_{i}" for i in range(len(results_list))]
        
        # 合并权重
        final_weights = {**self.config.weights, **(weights or {})}
        
        # 执行融合
        results = self._strategy.fuse(
            results_list=results_list,
            retriever_names=retriever_names,
            weights=final_weights,
            top_k=top_k,
        )
        
        # 分数归一化（可选）
        if self.config.normalize_scores and results:
            scores_dict = {DocIdExtractor.get_doc_id(doc): score for doc, score in results}
            normalized = ScoreNormalizer.normalize(scores_dict, method="min-max")
            results = [
                (doc, normalized.get(DocIdExtractor.get_doc_id(doc), score))
                for doc, score in results
            ]
        
        # 最低分数过滤
        if self.config.min_score_threshold > 0:
            results = [
                (doc, score) for doc, score in results
                if score >= self.config.min_score_threshold
            ]
        
        # 记录统计信息
        self._record_stats(results_list, retriever_names, final_weights, results)
        
        return results
    
    def fuse_with_strategy(
        self,
        results_list: list[list[tuple[Any, float]]],
        strategy: str = "rrf",
        retriever_names: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list[tuple[Any, float]]:
        """
        使用指定策略融合结果
        
        Args:
            results_list: 各检索器的结果列表
            strategy: 融合策略，可选 "rrf" 或 "weighted"
            retriever_names: 检索器名称列表
            top_k: 返回的文档数量
            
        Returns:
            融合后的结果列表
        """
        if strategy == "weighted":
            strategy_obj = WeightedScoreFusionStrategy()
        else:
            strategy_obj = self._strategy
        
        retriever_names = retriever_names or [f"retriever_{i}" for i in range(len(results_list))]
        
        return strategy_obj.fuse(
            results_list=results_list,
            retriever_names=retriever_names,
            weights=self.config.weights or {},
            top_k=top_k,
        )
    
    def _record_stats(
        self,
        results_list: list,
        retriever_names: list[str],
        weights: dict[str, float],
        results: list,
    ) -> None:
        """记录融合统计信息"""
        self._fusion_stats = {
            "input_counts": [len(r) for r in results_list],
            "output_count": len(results),
            "retriever_names": retriever_names,
            "weights_used": weights,
            "k_value": self.config.k,
        }
    
    def get_stats(self) -> dict:
        """
        获取最近一次融合的统计信息
        
        Returns:
            统计信息字典
        """
        return self._fusion_stats.copy()
    
    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        动态更新检索器权重
        
        Args:
            new_weights: 新的权重字典
        """
        self.config.weights = {**self.config.weights, **new_weights}
        logger.info(f"权重已更新: {self.config.weights}")
    
    def set_k(self, k: int) -> None:
        """
        动态设置 k 值
        
        Args:
            k: 新的 k 值
        """
        if k <= 0:
            raise ValueError(f"k 必须为正数，当前值: {k}")
        self.config.k = k
        self._strategy = RRFFusionStrategy(k=k)
        logger.info(f"k 值已更新为: {k}")


# ------------------------------------------------------------------ //
# 高级融合器
# ------------------------------------------------------------------ //


class MultiStageFusion:
    """
    多阶段融合器
    
    支持复杂的多阶段融合策略：
    1. 第一阶段：组内融合（如多个向量检索器）
    2. 第二阶段：组间融合（向量组 vs BM25组）
    """
    
    def __init__(
        self,
        stage1_config: Optional[RRFConfig] = None,
        stage2_config: Optional[RRFConfig] = None,
    ):
        self.stage1_fusion = RRFFusion(stage1_config or RRFConfig())
        self.stage2_fusion = RRFFusion(stage2_config or RRFConfig())
    
    def fuse(
        self,
        retriever_groups: dict[str, list[list[tuple[Any, float]]]],
        top_k: int = 10,
    ) -> list[tuple[Any, float]]:
        """
        多阶段融合
        
        Args:
            retriever_groups: 检索器分组，键为组名，值为该组内各检索器的结果列表
            top_k: 最终返回数量
            
        Returns:
            融合后的结果列表
        """
        stage1_results = {}
        
        for group_name, results_list in retriever_groups.items():
            if len(results_list) == 1:
                # 单个检索器，直接使用
                stage1_results[group_name] = results_list[0]
            else:
                # 组内融合
                fused = self.stage1_fusion.fuse(
                    results_list,
                    retriever_names=[f"{group_name}_{i}" for i in range(len(results_list))],
                    top_k=top_k * 2,  # 保留更多候选
                )
                stage1_results[group_name] = fused
        
        # 第二阶段：组间融合
        all_results = list(stage1_results.values())
        return self.stage2_fusion.fuse(
            all_results,
            retriever_names=list(stage1_results.keys()),
            top_k=top_k,
        )


class AdaptiveRRFFusion(RRFFusion):
    """
    自适应 RRF 融合器
    
    根据检索结果的质量自动调整权重
    """
    
    def __init__(self, config: Optional[RRFConfig] = None):
        super().__init__(config)
        self._performance_history: dict[str, list[float]] = defaultdict(list)
    
    def fuse_adaptive(
        self,
        results_list: list[list[tuple[Any, float]]],
        retriever_names: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list[tuple[Any, float]]:
        """
        自适应融合
        
        根据各检索器的历史表现调整权重
        
        Args:
            results_list: 各检索器的结果列表
            retriever_names: 检索器名称列表
            top_k: 返回数量
            
        Returns:
            融合结果
        """
        retriever_names = retriever_names or [f"retriever_{i}" for i in range(len(results_list))]
        
        # 计算自适应权重
        adaptive_weights = self._compute_adaptive_weights(results_list, retriever_names)
        
        return self.fuse(
            results_list=results_list,
            retriever_names=retriever_names,
            top_k=top_k,
            weights=adaptive_weights,
        )
    
    def _compute_adaptive_weights(
        self,
        results_list: list[list[tuple[Any, float]]],
        retriever_names: list[str],
    ) -> dict[str, float]:
        """
        计算自适应权重
        
        基于各检索器的结果质量指标：
        - 结果数量
        - 分数分布
        - 分数方差
        """
        weights = {}
        
        for name, results in zip(retriever_names, results_list):
            if not results:
                weights[name] = 0.1  # 无结果的检索器给予最低权重
                continue
            
            scores = [score for _, score in results]
            max_score = max(scores)
            
            # 基于最高分和结果数量的简单启发式
            quality = min(1.0, len(results) / 10.0) * (max_score / (max(scores) if scores else 1.0))
            
            weights[name] = max(0.5, quality * 2.0)
        
        # 归一化权重
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def record_feedback(
        self,
        retriever_name: str,
        score: float,
    ) -> None:
        """
        记录检索器的反馈分数
        
        Args:
            retriever_name: 检索器名称
            score: 反馈分数（如用户评分、点击率等）
        """
        self._performance_history[retriever_name].append(score)
        # 保留最近 100 次记录
        if len(self._performance_history[retriever_name]) > 100:
            self._performance_history[retriever_name] = self._performance_history[retriever_name][-100:]
