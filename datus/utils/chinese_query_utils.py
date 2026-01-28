# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 0.2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Chinese Query Processing Utilities

This module provides enhanced processing for Chinese text2sql queries:
- Bilingual LLM-based query rewriting (Chinese -> English database terms)
- Chinese text segmentation and keyword extraction
- Dynamic similarity threshold based on query complexity
- Query complexity scoring

The goal is to bridge the gap between Chinese business queries and English database schemas.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig


class QueryComplexity(Enum):
    """Query complexity levels for dynamic threshold adjustment."""
    SIMPLE = "simple"      # Single concept, few keywords
    MEDIUM = "medium"      # Multiple concepts, some aggregation
    COMPLEX = "complex"    # Multiple joins, complex aggregations, time windows


@dataclass
class ChineseQueryAnalysis:
    """Analysis result for a Chinese query."""
    original_query: str
    is_chinese: bool
    chinese_ratio: float
    complexity: QueryComplexity
    complexity_score: float
    keywords: List[str]
    entities: List[str]  # Named entities like 铂智3X
    business_terms: List[str]
    suggested_threshold: float
    suggested_top_n: int
    translation: Optional[str] = None  # English translation for schema matching


def contains_chinese(text: str) -> bool:
    """Detect if text contains Chinese characters."""
    if not text:
        return False
    for char in text:
        code_point = ord(char)
        if (0x4E00 <= code_point <= 0x9FFF) or (0x3400 <= code_point <= 0x4DBF):
            return True
    return False


def calculate_chinese_ratio(text: str) -> float:
    """Calculate the ratio of Chinese characters in text."""
    if not text:
        return 0.0
    chinese_chars = sum(1 for char in text if contains_chinese(char))
    return chinese_chars / len(text)


def extract_chinese_entities(text: str) -> List[str]:
    """
    Extract named entities from Chinese text.
    
    Captures:
    - Mixed alphanumeric patterns (e.g., 铂智3X, 车型A)
    - Product names, model numbers
    - Date patterns (2024年, Q1)
    """
    entities = []
    
    # Mixed Chinese-English patterns (e.g., 铂智3X, 宝马X5)
    # Exclude common leading/trailing words that are not part of entity names
    EXCLUDE_PREFIXES = ['查询', '分析', '统计', '获取', '查看', '计算']
    EXCLUDE_SUFFIXES = ['数据', '信息', '情况', '结果']
    
    mixed_pattern = r'[\u4e00-\u9fff]+[A-Za-z0-9]+|[A-Za-z0-9]+[\u4e00-\u9fff]+'
    raw_entities = re.findall(mixed_pattern, text)
    
    for entity in raw_entities:
        # Clean up entity by removing common prefixes/suffixes
        clean_entity = entity
        for prefix in EXCLUDE_PREFIXES:
            if clean_entity.startswith(prefix):
                clean_entity = clean_entity[len(prefix):]
        for suffix in EXCLUDE_SUFFIXES:
            if clean_entity.endswith(suffix):
                clean_entity = clean_entity[:-len(suffix)]
        if clean_entity and len(clean_entity) >= 2:
            entities.append(clean_entity)
    
    # Standalone date/time patterns
    date_pattern = r'(\d{4}年|\d{4}-\d{2}|Q[1-4]|\d{2}月|\d{2}日)'
    entities.extend(re.findall(date_pattern, text))
    
    return list(set(entities))


def extract_chinese_keywords(text: str, min_length: int = 2) -> List[str]:
    """
    Extract keywords from Chinese text using business term dictionary + bigram analysis.
    
    Args:
        text: Input Chinese text
        min_length: Minimum keyword length
        
    Returns:
        List of extracted keywords
    """
    # Common business terms for automotive/dealership domain
    BUSINESS_TERMS = {
        "线索", "试驾", "订单", "漏斗", "到店", "转化", "统计", "渠道",
        "有效", "意向", "车型", "车种", "客户", "门店", "销售",
        "潜客", "留档", "成交", "交车", "库存", "入库", "出库",
        "月度", "季度", "年度", "同比", "环比", "趋势", "分布",
        "数量", "金额", "占比", "率", "额", "量", "数",
    }
    
    # Stop words to filter out
    STOP_WORDS = {
        "的", "和", "与", "及", "或", "按", "对", "在", "中", "以", "于", "为",
        "了", "是", "有", "我", "他", "她", "它", "们", "这", "那", "个",
        "什么", "怎么", "如何", "请", "问", "查询", "分析", "查看", "获取",
    }
    
    keywords = []
    
    # 1. Extract known business terms
    for term in BUSINESS_TERMS:
        if term in text:
            keywords.append(term)
    
    # 2. Generate bigrams for remaining Chinese text
    chinese_segments = re.findall(r'[\u4e00-\u9fff]+', text)
    
    for segment in chinese_segments:
        # Skip short segments
        if len(segment) < min_length:
            continue
        
        # Single word if length is 2
        if len(segment) == 2:
            if segment not in STOP_WORDS and segment not in keywords:
                keywords.append(segment)
            continue
        
        # Generate overlapping bigrams for longer segments
        for i in range(len(segment) - 1):
            bigram = segment[i:i+2]
            if bigram not in STOP_WORDS and bigram not in keywords:
                # Check if both characters are meaningful
                if not any(char in STOP_WORDS for char in bigram):
                    keywords.append(bigram)
    
    return keywords


def analyze_query_complexity(query: str) -> Tuple[QueryComplexity, float]:
    """
    Analyze query complexity for dynamic threshold adjustment.
    
    Scoring factors:
    - Query length
    - Number of distinct concepts
    - Presence of aggregation indicators
    - Presence of time/filters
    - Number of relationships/joins implied
    
    Returns:
        Tuple of (complexity_level, score_0_to_1)
    """
    if not query:
        return QueryComplexity.SIMPLE, 0.0
    
    score = 0.0
    
    # Length factor (normalized)
    length_score = min(len(query) / 50, 1.0) * 0.2
    score += length_score
    
    # Count distinct business concepts
    CONCEPT_INDICATORS = {
        "线索": "lead", "试驾": "testdrive", "订单": "order",
        "客户": "customer", "车型": "vehicle", "渠道": "channel",
        "门店": "dealer", "转化": "conversion", "漏斗": "funnel",
        "统计": "statistics", "分析": "analysis", "对比": "comparison",
    }
    
    concept_count = sum(1 for indicator in CONCEPT_INDICATORS if indicator in query)
    concept_score = min(concept_count / 4, 1.0) * 0.3
    score += concept_score
    
    # Aggregation indicators
    AGGREGATION_WORDS = ["汇总", "统计", "总计", "平均", "最大", "最小", "求和", "计算", "占比", "率"]
    agg_count = sum(1 for word in AGGREGATION_WORDS if word in query)
    agg_score = min(agg_count / 2, 1.0) * 0.2
    score += agg_score
    
    # Time/filter indicators
    TIME_WORDS = ["按月", "按周", "按日", "月度", "季度", "年度", "同比", "环比", "趋势", "最近"]
    time_count = sum(1 for word in TIME_WORDS if word in query)
    time_score = min(time_count / 2, 1.0) * 0.15
    score += time_score
    
    # Relationship/join indicators
    RELATIONSHIP_WORDS = ["与", "和", "以及", "的", "按", "根据", "通过"]
    rel_count = sum(1 for word in RELATIONSHIP_WORDS if word in query)
    rel_score = min(rel_count / 3, 1.0) * 0.15
    score += rel_score
    
    # Determine complexity level
    if score < 0.35:
        complexity = QueryComplexity.SIMPLE
    elif score < 0.65:
        complexity = QueryComplexity.MEDIUM
    else:
        complexity = QueryComplexity.COMPLEX
    
    return complexity, min(score, 1.0)


def calculate_dynamic_threshold(
    query: str,
    base_threshold: float,
    complexity: QueryComplexity,
    complexity_score: float,
    chinese_ratio: float,
) -> float:
    """
    Calculate dynamic similarity threshold based on query characteristics.
    
    Args:
        query: Original query text
        base_threshold: Base similarity threshold from config
        complexity: Query complexity level
        complexity_score: Complexity score (0-1)
        chinese_ratio: Ratio of Chinese characters (0-1)
        
    Returns:
        Adjusted similarity threshold
    """
    threshold = base_threshold
    
    # Chinese queries need lower threshold due to cross-lingual embedding mismatch
    if chinese_ratio > 0.3:
        # Reduce threshold based on Chinese ratio
        # More Chinese = lower threshold needed
        chinese_reduction = 0.1 + (chinese_ratio * 0.2)  # 0.1 to 0.3 reduction
        threshold -= chinese_reduction
    
    # Adjust based on complexity
    # Complex queries should use slightly higher threshold to be more precise
    # Simple queries should use lower threshold to cast wider net
    if complexity == QueryComplexity.SIMPLE:
        threshold -= 0.05  # Wider net for simple queries
    elif complexity == QueryComplexity.COMPLEX:
        threshold += 0.05  # Tighter threshold for complex queries
    
    # Fine-tune based on complexity score
    # Higher complexity score = slightly lower threshold (need more context)
    complexity_adjustment = (0.5 - complexity_score) * 0.1
    threshold += complexity_adjustment
    
    # Ensure threshold stays in reasonable bounds
    return max(0.15, min(0.7, threshold))


def suggest_top_n(complexity: QueryComplexity, base_top_n: int = 20) -> int:
    """Suggest top_n based on query complexity."""
    multipliers = {
        QueryComplexity.SIMPLE: 0.8,    # Fewer tables needed
        QueryComplexity.MEDIUM: 1.0,    # Standard amount
        QueryComplexity.COMPLEX: 1.5,   # More tables needed
    }
    return int(base_top_n * multipliers.get(complexity, 1.0))


async def rewrite_chinese_query_bilingual(
    query: str,
    agent_config: "AgentConfig",
    model_name: str = "",
) -> Optional[str]:
    """
    Rewrite Chinese query into bilingual form optimized for English schema search.
    
    This uses an LLM to:
    1. Extract key Chinese business terms
    2. Translate to English database terms
    3. Combine for maximum schema matching effectiveness
    
    Example:
        Input: "分析铂智3X车型从有效线索到店转化的月度漏斗数据"
        Output: "铂智3X Bozhi_3X 线索 lead 到店到店 转化 conversion 漏斗 funnel 月度 monthly 车型 vehicle"
    
    Args:
        query: Original Chinese query
        agent_config: Agent configuration
        model_name: Optional model override
        
    Returns:
        Bilingual enhanced query string, or None if processing fails
    """
    if not query or not contains_chinese(query):
        return None
    
    try:
        from datus.models.base import LLMBaseModel
        
        llm_model = LLMBaseModel.create_model(
            agent_config=agent_config,
            model_name=model_name or None
        )
    except Exception as e:
        return None
    
    # Extract entities first to preserve them
    entities = extract_chinese_entities(query)
    entity_hint = f"Preserve these entities exactly: {', '.join(entities)}. " if entities else ""
    
    prompt = f"""You are a bilingual data analyst specializing in Chinese-to-English database schema matching.

Task: Rewrite the Chinese business query into a bilingual keyword string for semantic schema search.

Rules:
1. Keep all Chinese product names, model numbers, and proper nouns AS-IS
2. Add English database equivalents for business terms:
   - 线索/潜客 → lead/clue
   - 试驾 → testdrive/test_drive
   - 订单 → order
   - 客户 → customer
   - 车型/车种 → vehicle/model
   - 门店/经销商 → dealer
   - 渠道 → channel
   - 转化 → conversion
   - 漏斗 → funnel
   - 到店 → showroom/arrival
   - 月度/月 → monthly/month
   - 统计/汇总 → statistics/summary
3. Output format: ChineseTerm EnglishTerm ChineseTerm EnglishTerm ...
4. Keep it concise (10-15 tokens max)
5. {entity_hint}

Return ONLY a JSON object: {{"enhanced_query": "<bilingual keywords>"}}

Chinese Query: {query}
"""
    
    try:
        response = llm_model.generate_with_json_output(prompt)
        
        if isinstance(response, dict):
            enhanced = response.get("enhanced_query", "")
            if isinstance(enhanced, str) and enhanced.strip():
                return enhanced.strip()
    except Exception as e:
        pass
    
    return None


async def analyze_chinese_query(
    query: str,
    agent_config: Optional["AgentConfig"] = None,
    base_threshold: float = 0.5,
    base_top_n: int = 20,
    use_llm_rewrite: bool = True,
) -> ChineseQueryAnalysis:
    """
    Comprehensive analysis of a Chinese query with all enhancements.
    
    This is the main entry point for Chinese query processing.
    
    Args:
        query: Original query string
        agent_config: Optional agent configuration for LLM rewrite
        base_threshold: Base similarity threshold
        base_top_n: Base number of results to return
        use_llm_rewrite: Whether to use LLM bilingual rewrite
        
    Returns:
        ChineseQueryAnalysis with all processing results
    """
    # Basic analysis
    is_chinese = contains_chinese(query)
    chinese_ratio = calculate_chinese_ratio(query)
    
    # Extract components
    entities = extract_chinese_entities(query) if is_chinese else []
    keywords = extract_chinese_keywords(query) if is_chinese else []
    
    # Complexity analysis
    complexity, complexity_score = analyze_query_complexity(query)
    
    # LLM bilingual rewrite (optional)
    translation = None
    if use_llm_rewrite and is_chinese and agent_config:
        translation = await rewrite_chinese_query_bilingual(query, agent_config)
    
    # Calculate dynamic threshold
    suggested_threshold = calculate_dynamic_threshold(
        query=query,
        base_threshold=base_threshold,
        complexity=complexity,
        complexity_score=complexity_score,
        chinese_ratio=chinese_ratio,
    )
    
    # Suggest top_n
    suggested_top_n = suggest_top_n(complexity, base_top_n)
    
    # Extract business terms from keywords
    BUSINESS_TERMS = {
        "线索", "试驾", "订单", "漏斗", "到店", "转化",
        "客户", "车型", "渠道", "门店", "统计", "分析",
    }
    business_terms = [k for k in keywords if k in BUSINESS_TERMS]
    
    return ChineseQueryAnalysis(
        original_query=query,
        is_chinese=is_chinese,
        chinese_ratio=chinese_ratio,
        complexity=complexity,
        complexity_score=complexity_score,
        keywords=keywords,
        entities=entities,
        business_terms=business_terms,
        suggested_threshold=suggested_threshold,
        suggested_top_n=suggested_top_n,
        translation=translation,
    )


# Backward compatibility
def _contains_chinese(text: str) -> bool:
    """Backward-compatible wrapper for contains_chinese."""
    return contains_chinese(text)
