#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared constants and utilities for business config generation.
"""

from typing import Dict, List, Set

# 停用词列表（用于关键词提取）
STOP_WORDS: Set[str] = {
    "的", "和", "或", "与", "及", "是", "在", "用于", "表示", "指",
    "对", "从", "到", "为", "有", "由", "等", "可", "请", "需",
    "以", "根据", "按照", "依据", "包括", "包含", "涉及",
}

# 常见同义词映射（业务术语标准化）
SYNONYM_MAP: Dict[str, str] = {
    "车种": "车型",
    "车系": "车型系列",
    "dealership": "经销店",
    "4s店": "经销店",
    "门店": "经销店",
}

# 技术词汇集合（用于过滤）
TECHNICAL_TERMS: Set[str] = {
    'id', 'code', 'name', 'status', 'type', 'flag', 'time', 'date',
    'create', 'update', 'delete', 'insert', 'select', 'from', 'where',
    'table', 'column', 'field', 'index', 'key', 'value',
    'dealer_clue_code', 'original_clue_code', 'customer_id',
    'engine', 'key', 'duplicate', 'distributed', 'random', 'min', 'max', 'properties',
}

# 常见指标后缀
METRIC_SUFFIXES: List[str] = [
    '数量', '数', '量', '率', '占比', '比例', '金额', '次数', '天数', '时长',
    '目标', '实绩', '合计', '汇总', '统计', '平均', '最大', '最小',
    '及时', '完成', '达成', '转化', '变更', '新增', '活跃'
]

# 常见技术词汇（用于关键词过滤）
TECHNICAL_KEYWORDS: Set[str] = {
    '明细', '汇总', '统计', '计算', '结果', '数据', '信息', '字段', '表名',
}


def is_meaningful_term(term: str, min_length: int = 2) -> bool:
    """判断术语是否有业务意义
    
    Args:
        term: 待判断的术语
        min_length: 最小长度要求，默认2
        
    Returns:
        bool: 如果有业务意义返回 True，否则返回 False
    """
    import re

    if not term or len(term) < min_length:
        return False

    if term.lower() in TECHNICAL_TERMS:
        return False

    if re.match(r'^\d+$', term):
        return False

    if term.startswith('_'):
        return False

    return True
