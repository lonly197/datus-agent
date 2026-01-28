#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Term extraction utilities for business configuration generation.
"""

import re
from typing import Dict, List, Set

from ..shared import (
    STOP_WORDS,
    SYNONYM_MAP,
    TECHNICAL_KEYWORDS,
    METRIC_SUFFIXES,
)


class TermExtractor:
    """业务术语提取器"""

    def __init__(self, min_term_length: int = 2):
        self.min_term_length = min_term_length

    def extract_business_keywords(self, text: str) -> List[str]:
        """
        从业务定义中提取高价值业务关键词

        策略：提取业务场景关键词，如：
        - "首触"、"有效线索"、"试驾预约"、"订单转化"
        - 过滤通用词汇（的、和、是等）
        """
        if not text:
            return []

        keywords = []

        for match in re.finditer(r"[\u4e00-\u9fa5]{2,8}", text):
            kw = match.group()
            if kw in STOP_WORDS or len(kw) < self.min_term_length:
                continue
            if kw in TECHNICAL_KEYWORDS:
                continue
            keywords.append(kw)

        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:10]

    def extract_metric_terms(self, metric_name: str) -> List[str]:
        """
        从指标名称中提取业务术语

        处理逻辑：
        1. 去除常见指标后缀（数、量、率、占比、金额、次数等）
        2. 提取核心业务概念（如"有效线索数"→"有效线索"）
        3. 保留原始指标名称作为备选
        """
        if not metric_name:
            return []

        terms = []
        clean_name = re.sub(r'[（(].*?[）)]', '', metric_name).strip()

        # 逐层去除后缀
        current = clean_name
        for suffix in METRIC_SUFFIXES:
            if current.endswith(suffix) and len(current) > len(suffix) + self.min_term_length:
                core_term = current[:-len(suffix)]
                if len(core_term) >= self.min_term_length and self._is_meaningful_term(core_term):
                    terms.append(core_term)
                current = core_term

        # 提取2-8字的中文业务词
        for match in re.finditer(r"[\u4e00-\u9fa5]{%d,8}" % self.min_term_length, clean_name):
            kw = match.group()
            if self._is_meaningful_term(kw):
                terms.append(kw)

        # 特定业务术语识别
        special_patterns = [
            r'(首触)\w*',
            r'(有效\w+)',
            r'(原始\w+)',
            r'(人工\w+)',
            r'(自然\w+)',
            r'(战败\w*)',
        ]
        for pattern in special_patterns:
            for match in re.finditer(pattern, clean_name):
                term = match.group(1)
                if len(term) >= self.min_term_length and term not in terms:
                    terms.append(term)

        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen and len(term) >= self.min_term_length:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms[:5]

    def extract_core_keywords(self, text: str) -> List[str]:
        """提取核心业务关键词（用于表名匹配）"""
        keywords = []
        for match in re.finditer(r"[\u4e00-\u9fa5]{2,8}", text):
            kw = match.group()
            if kw not in ['明细', '汇总', '统计', '计算', '结果', '数据', '信息']:
                keywords.append(kw)
        return keywords

    def extract_meaningful_keywords(self, text: str) -> List[str]:
        """提取有意义的关键词（过滤技术词汇和停用词）"""
        if not text:
            return []

        keywords = []

        # 提取中文短语
        for match in re.finditer(r"[\u4e00-\u9fa5]{%d,10}" % self.min_term_length, text):
            kw = match.group()
            if kw not in STOP_WORDS and self._is_meaningful_term(kw):
                keywords.append(kw)

        # 提取英文/数字混合的业务词汇
        for match in re.finditer(r"[a-z_][a-z0-9_]{%d,}" % (self.min_term_length - 1), text.lower()):
            kw = match.group()
            if self.min_term_length <= len(kw) <= 40 and self._is_meaningful_term(kw):
                keywords.append(kw)

        # 同义词替换
        normalized = []
        for kw in keywords:
            normalized.append(SYNONYM_MAP.get(kw, kw))

        return list(set(normalized))

    def _is_meaningful_term(self, term: str) -> bool:
        """判断术语是否有业务意义"""
        if not term or len(term) < 2:
            return False

        technical_terms = {
            'id', 'code', 'name', 'status', 'type', 'flag', 'time', 'date',
            'create', 'update', 'delete', 'insert', 'select', 'from', 'where',
            'table', 'column', 'field', 'index', 'key', 'value',
            'dealer_clue_code', 'original_clue_code', 'customer_id',
        }

        if term.lower() in technical_terms:
            return False

        if re.match(r'^\d+$', term):
            return False

        if term.startswith('_'):
            return False

        return True

    def extract_keywords(self, text: str) -> List[str]:
        """从文本提取关键词（简单实现）"""
        if not text:
            return []

        keywords = []

        for match in re.finditer(r"[\u4e00-\u9fa5]{%d,10}" % self.min_term_length, text):
            kw = match.group()
            if kw not in STOP_WORDS and len(kw) >= self.min_term_length:
                keywords.append(kw)

        for match in re.finditer(r"[a-z_][a-z0-9_]{%d,}" % (self.min_term_length - 1), text.lower()):
            kw = match.group()
            if kw not in STOP_WORDS:
                keywords.append(kw)

        normalized = [SYNONYM_MAP.get(kw, kw) for kw in keywords]
        return list(set(normalized))

    def extract_table_refs(self, calc_logic: str) -> List[str]:
        """从计算公式中提取表引用"""
        refs = []
        for match in re.finditer(r"\b(t_[a-z_]+|dws_[a-z_]+|dwd_[a-z_]+|dim_[a-z_]+)\b", calc_logic.lower()):
            refs.append(match.group())
        return refs
