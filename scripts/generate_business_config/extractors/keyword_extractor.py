#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Keyword extraction utilities for business configuration generation.
"""

import re
from typing import Dict, List, Optional

from ..shared import STOP_WORDS, SYNONYM_MAP, TECHNICAL_TERMS

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class KeywordExtractor:
    """关键词提取器"""

    def __init__(self, min_length: int = 2):
        self.min_length = min_length

    def extract(self, text: str) -> List[str]:
        """提取有意义的关键词"""
        if not text:
            return []

        keywords = []

        # 提取中文短语
        for match in re.finditer(r"[\u4e00-\u9fa5]{%d,10}" % self.min_length, text):
            kw = match.group()
            if kw not in STOP_WORDS and self._is_business_term(kw):
                keywords.append(kw)

        # 提取英文/数字混合词
        for match in re.finditer(r"[a-z_][a-z0-9_]{%d,}" % (self.min_length - 1), text.lower()):
            kw = match.group()
            if self._is_business_term(kw):
                keywords.append(kw)

        # 同义词替换
        normalized = [SYNONYM_MAP.get(kw, kw) for kw in keywords]
        return list(set(normalized))

    def extract_from_comments(self, text: str) -> List[str]:
        """从注释中提取关键词"""
        if not text:
            return []

        keywords = []
        for match in re.finditer(r"[\u4e00-\u9fa5]{%d,8}" % self.min_length, text):
            kw = match.group()
            if kw not in STOP_WORDS and len(kw) >= self.min_length:
                keywords.append(kw)

        return list(set(keywords))

    def _is_business_term(self, term: str) -> bool:
        """判断是否为业务术语"""
        if not term or len(term) < self.min_length:
            return False

        if term.lower() in TECHNICAL_TERMS:
            return False

        if re.match(r'^\d+$', term):
            return False

        if term.startswith('_'):
            return False

        return True


class BusinessTermMapping:
    """带置信度的业务术语映射项"""

    def __init__(self, term: str, targets: List[str], source: str, confidence: float = 0.5):
        self.term = term
        self.targets = set(targets) if isinstance(targets, list) else {targets}
        self.sources = {source: confidence}
        self.confidence = confidence
        self.metadata: Dict = {}

    def merge(self, other: 'BusinessTermMapping') -> 'BusinessTermMapping':
        """合并两个映射项"""
        self.targets.update(other.targets)
        for src, conf in other.sources.items():
            if src not in self.sources or conf > self.sources[src]:
                self.sources[src] = conf
        self.confidence = max(self.confidence, other.confidence)
        return self

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "targets": sorted(list(self.targets)),
            "confidence": round(self.confidence, 2),
            "sources": self.sources,
        }
