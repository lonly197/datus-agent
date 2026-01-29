#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared constants and utilities for business config generation.
"""

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Set, Optional, Pattern


class TablePriority(IntEnum):
    """表优先级枚举，数值越小优先级越高"""
    DIM = 1      # 维度表 - 最高优先级
    DWD = 2      # 明细事实表
    DWS = 3      # 汇总事实表
    ADS = 4      # 应用数据表
    ODS = 5      # 操作数据表 - 最低优先级
    UNKNOWN = 99  # 未知类型


# 表前缀到优先级的映射
TABLE_PREFIX_PRIORITY: Dict[str, TablePriority] = {
    'dim_': TablePriority.DIM,
    'dwd_': TablePriority.DWD,
    'dws_': TablePriority.DWS,
    'ads_': TablePriority.ADS,
    'ods_': TablePriority.ODS,
}

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

# 严格过滤（Text2SQL 优化）默认配置
STRICT_STOP_TERMS: Set[str] = {
    "时间", "日期", "编号", "代码", "名称", "数量", "类型", "状态", "标识", "字段", "备注", "其他",
    "信息", "金额", "数", "值", "率", "比", "原因", "结果", "方式", "单位", "版本", "类别", "对象",
    "维度", "指标", "数据", "明细", "汇总", "合计", "总数", "占比", "比例", "详情", "场景",
    "新增", "累计", "月度", "日度", "周度", "年度",
    # 业务过于泛化的核心词（仅当等于该词时过滤）
    "客户", "订单", "线索", "门店", "销售", "试驾", "渠道", "任务",
}


@dataclass
class StrictTermFilterConfig:
    enabled: bool = False
    max_mapping_count: int = 15
    min_length: int = 3
    drop_boolean_prefix: bool = True
    drop_english: bool = True
    drop_digits: bool = True
    drop_symbols: bool = True
    drop_bracketed: bool = True
    allow_terms: Set[str] = field(default_factory=set)
    allow_patterns: List[Pattern] = field(default_factory=list)


def is_meaningful_term(term: str, min_length: int = 2) -> bool:
    """判断术语是否有业务意义
    
    Args:
        term: 待判断的术语
        min_length: 最小长度要求，默认2
        
    Returns:
        bool: 如果有业务意义返回 True，否则返回 False
    """
    if not term or len(term) < min_length:
        return False

    if term.lower() in TECHNICAL_TERMS:
        return False

    if re.match(r'^\d+$', term):
        return False

    if term.startswith('_'):
        return False

    return True


def get_table_priority(table_name: str) -> TablePriority:
    """根据表名获取优先级
    
    Args:
        table_name: 表名
        
    Returns:
        TablePriority: 表的优先级
    """
    if not table_name:
        return TablePriority.UNKNOWN
    
    table_lower = table_name.lower()
    for prefix, priority in TABLE_PREFIX_PRIORITY.items():
        if table_lower.startswith(prefix):
            return priority
    return TablePriority.UNKNOWN


def should_include_table(table_name: str, max_priority: TablePriority = TablePriority.ADS) -> bool:
    """判断是否应该包含该表
    
    Args:
        table_name: 表名
        max_priority: 最大允许的优先级（默认为ADS，即包含DIM/DWD/DWS/ADS，排除ODS）
        
    Returns:
        bool: 如果应该包含返回 True
    """
    priority = get_table_priority(table_name)
    return priority != TablePriority.UNKNOWN and priority <= max_priority


# 用于清洗文本的正则表达式模式
# 注意：emoji范围不能与CJK字符范围（\u4e00-\u9fff）重叠
TEXT_CLEANING_PATTERNS = {
    # 移除emoji - 使用明确的emoji范围，避免与CJK字符重叠
    'emoji': re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons: 😀-🙏
        "\U0001F300-\U0001F5FF"  # symbols & pictographs: 🌀-🗿
        "\U0001F680-\U0001F6FF"  # transport & map: 🚀-🛿
        "\U0001F1E0-\U0001F1FF"  # flags: 🇦-🇿
        "\U00002702-\U000027B0"  # dingbats: ✂-➰
        "\U0001F900-\U0001F9FF"  # supplemental symbols: 🦀-🧿
        "\U00002600-\U000026FF"  # misc symbols: ☀-⛿
        "\U0001F018-\U0001F270"  # 更多emoji
        "\U00002300-\U000023FF"  # misc technical: ⌀-⏿
        "]+",
        flags=re.UNICODE
    ),
    # 移除行首序号（如 1.、①、(1)、（1）等）- 仅匹配行首
    'numbered_list': re.compile(r'^[\s]*(?:\d+[\.、]|\([\d一二三四五六七八九十]+\)|（[\d一二三四五六七八九十]+）|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])[\s]*'),
    # 行内序号标记（用于替换为空格而非删除）
    'inline_number': re.compile(r'\([\d一二三四五六七八九十]+\)|（[\d一二三四五六七八九十]+）|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]'),
    # 多余空格
    'extra_spaces': re.compile(r'\s+'),
    # 多余换行
    'extra_newlines': re.compile(r'\n+'),
    # 特殊符号（保留基本标点）
    'special_chars': re.compile(r'[*#^~|\\]'),
}


def clean_excel_text(text: Optional[str], remove_newlines: bool = False) -> str:
    """清洗Excel单元格文本
    
    处理内容：
    - 移除emoji
    - 移除序号（如 1.、①、(1)等）
    - 规范化空格和换行
    - 移除特殊符号
    - 去除首尾空白
    
    Args:
        text: 输入文本
        remove_newlines: 是否移除所有换行（默认保留，替换为空格）
        
    Returns:
        str: 清洗后的文本
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 移除emoji
    text = TEXT_CLEANING_PATTERNS['emoji'].sub('', text)
    
    # 移除行首序号标记
    text = TEXT_CLEANING_PATTERNS['numbered_list'].sub('', text)
    # 将行内序号替换为空格（避免与前后文字粘连）
    text = TEXT_CLEANING_PATTERNS['inline_number'].sub(' ', text)
    
    # 移除特殊符号
    text = TEXT_CLEANING_PATTERNS['special_chars'].sub('', text)
    
    # 处理换行
    if remove_newlines:
        text = text.replace('\n', ' ').replace('\r', ' ')
    else:
        # 将多个换行替换为单个
        text = TEXT_CLEANING_PATTERNS['extra_newlines'].sub('\n', text)
    
    # 规范化空格
    text = TEXT_CLEANING_PATTERNS['extra_spaces'].sub(' ', text)
    
    # 去除首尾空白
    text = text.strip()
    
    return text


def extract_clean_keywords(text: str, min_length: int = 2, max_length: int = 20) -> List[str]:
    """从文本中提取清洗后的关键词

    Args:
        text: 输入文本
        min_length: 最小长度
        max_length: 最大长度

    Returns:
        List[str]: 关键词列表
    """
    if not text:
        return []

    # 先清洗文本
    cleaned = clean_excel_text(text, remove_newlines=True)
    if not cleaned:
        return []

    keywords = []

    # 提取中文词汇
    for match in re.finditer(r'[\u4e00-\u9fa5]{' + str(min_length) + r',' + str(max_length) + r'}', cleaned):
        kw = match.group()
        if kw not in STOP_WORDS and is_meaningful_term(kw, min_length):
            keywords.append(kw)

    # 提取英文/数字业务词汇
    for match in re.finditer(r'[a-z_][a-z0-9_]{' + str(min_length - 1) + r',}', cleaned.lower()):
        kw = match.group()
        if kw not in TECHNICAL_TERMS and len(kw) <= 40:
            keywords.append(kw)

    # 去重并保持顺序
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return unique_keywords


# ============================================================
# 业务术语质量过滤器
# ============================================================

# 否定词列表（不应作为业务术语开头）
NEGATION_WORDS: Set[str] = {'不', '无', '非', '未', '别', '莫', '勿'}

# 冗余描述模式
REDUNDANT_PATTERNS: List[re.Pattern] = [
    re.compile(r'.+的数量$'),
    re.compile(r'.+的订单数$'),
    re.compile(r'.+的线索数$'),
    re.compile(r'.+的试驾数$'),
    re.compile(r'.+的任务数$'),
    re.compile(r'.+的数量$'),
    re.compile(r'.+的订单数量$'),
    re.compile(r'.+的线索数量$'),
    re.compile(r'.+的试驾数量$'),
    re.compile(r'^不包含.+'),
    re.compile(r'^不区分.+'),
    re.compile(r'^未.+'),
    re.compile(r'^非.+'),
]

# 过滤的词尾模式（不完整的短语）
INVALID_ENDINGS: Set[str] = {'的', '了', '在', '是', '为', '于', '和', '或', '与', '及', '中', '上', '下'}

# 推荐的关键词长度范围
RECOMMENDED_MIN_LENGTH = 3
RECOMMENDED_MAX_LENGTH = 12


def is_valid_business_term(term: str) -> bool:
    """判断术语是否为有效的业务术语（用于 table_keywords 和 term_to_table）

    过滤规则：
    1. 长度范围检查（3-12字符）
    2. 否定词开头检查
    3. 冗余描述模式检查
    4. 词尾完整性检查
    5. 必须是完整词汇（非片段）

    Args:
        term: 待检查的术语

    Returns:
        bool: 如果是有效业务术语返回 True
    """
    if not term or not isinstance(term, str):
        return False

    # 1. 长度检查
    term_len = len(term)
    if term_len < RECOMMENDED_MIN_LENGTH or term_len > RECOMMENDED_MAX_LENGTH:
        return False

    # 2. 否定词开头检查
    if any(term.startswith(neg) for neg in NEGATION_WORDS):
        return False

    # 3. 冗余描述模式检查
    for pattern in REDUNDANT_PATTERNS:
        if pattern.match(term):
            return False

    # 4. 词尾检查（不能以助词结尾）
    if term[-1] in INVALID_ENDINGS:
        return False

    # 5. 检查是否包含完整业务概念（至少有1个中文词根）
    has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', term))
    if not has_chinese:
        # 英文术语需要更严格的检查
        if not re.match(r'^[a-z][a-z0-9_]{2,}$', term):
            return False

    return True


def _is_allowed_term(term: str, config: StrictTermFilterConfig) -> bool:
    if term in config.allow_terms:
        return True
    for pattern in config.allow_patterns:
        if pattern.search(term):
            return True
    return False


def is_strict_business_term(term: str, config: StrictTermFilterConfig) -> bool:
    """严格业务术语过滤（用于提升 text2sql 精度）"""
    if not config.enabled:
        return is_valid_business_term(term)

    if not is_valid_business_term(term):
        return False

    if _is_allowed_term(term, config):
        return True

    if len(term) < config.min_length:
        return False

    if config.drop_boolean_prefix and term.startswith("是否"):
        return False

    if term in STRICT_STOP_TERMS:
        return False

    if config.drop_bracketed and re.search(r"[\\(\\)（）\\[\\]【】]", term):
        return False

    if config.drop_english and re.search(r"[A-Za-z]", term):
        return False

    if config.drop_digits and re.search(r"\\d", term):
        return False

    if config.drop_symbols and re.search(r"[+_/#\\\\:\\-]", term):
        return False

    return True


def filter_business_terms(terms: List[str]) -> List[str]:
    """批量过滤业务术语

    Args:
        terms: 术语列表

    Returns:
        List[str]: 过滤后的术语列表
    """
    return [t for t in terms if is_valid_business_term(t)]


def clean_table_keywords(
    keywords: Dict[str, str],
    strict_config: Optional[StrictTermFilterConfig] = None,
) -> Dict[str, str]:
    """清理 table_keywords，移除无效的关键词

    Args:
        keywords: 原始关键词映射 {关键词: 表名}

    Returns:
        Dict[str, str]: 清理后的关键词映射
    """
    if strict_config is None:
        strict_config = StrictTermFilterConfig(enabled=False)
    return {k: v for k, v in keywords.items() if is_strict_business_term(k, strict_config)}


def clean_term_to_table(
    term_to_table: Dict[str, Set[str]],
    strict_config: Optional[StrictTermFilterConfig] = None,
) -> Dict[str, Set[str]]:
    """清理 term_to_table，移除无效的术语

    Args:
        term_to_table: 原始术语映射 {术语: {表1, 表2, ...}}

    Returns:
        Dict[str, Set[str]]: 清理后的术语映射
    """
    if strict_config is None:
        strict_config = StrictTermFilterConfig(enabled=False)
    cleaned: Dict[str, Set[str]] = {}
    for term, tables in term_to_table.items():
        if not is_strict_business_term(term, strict_config):
            continue
        if strict_config.enabled and strict_config.max_mapping_count > 0:
            if len(tables) > strict_config.max_mapping_count and not _is_allowed_term(term, strict_config):
                continue
        cleaned[term] = tables
    return cleaned


def clean_term_to_schema(
    term_to_schema: Dict[str, Set[str]],
    strict_config: Optional[StrictTermFilterConfig] = None,
) -> Dict[str, Set[str]]:
    """清理 term_to_schema，移除无效的术语并去重

    Args:
        term_to_schema: 原始字段映射 {术语: {字段1, 字段2, ...}}

    Returns:
        Dict[str, Set[str]]: 清理后的字段映射
    """
    if strict_config is None:
        strict_config = StrictTermFilterConfig(enabled=False)
    cleaned: Dict[str, Set[str]] = {}
    for term, fields in term_to_schema.items():
        if not is_strict_business_term(term, strict_config):
            continue
        unique_fields = set(fields)
        if strict_config.enabled and strict_config.max_mapping_count > 0:
            if len(unique_fields) > strict_config.max_mapping_count and not _is_allowed_term(term, strict_config):
                continue
        cleaned[term] = unique_fields
    return cleaned


def filter_term_to_schema_by_table_priority(
    term_to_schema: Dict[str, Set[str]],
    max_table_priority: TablePriority = TablePriority.ADS,
    keep_bare_if_has_valid: bool = True,
) -> Dict[str, Set[str]]:
    """按表优先级过滤 term_to_schema，移除仅来自低优先级表或仅剩裸字段的术语。

    规则：
    - 仅保留 table.column 中表名满足 max_table_priority 的映射
    - 若术语没有任何合法 table.column，则整个术语丢弃（避免仅裸字段污染）
    - 若存在合法 table.column，可选择保留裸字段（默认保留）
    """
    cleaned: Dict[str, Set[str]] = {}

    for term, fields in term_to_schema.items():
        qualified_fields: Set[str] = set()
        bare_fields: Set[str] = set()

        for field in fields:
            if "." in field:
                table_name, col_name = field.split(".", 1)
                if should_include_table(table_name, max_table_priority):
                    qualified_fields.add(f"{table_name}.{col_name}")
            else:
                bare_fields.add(field)

        if not qualified_fields:
            # 没有合法表映射，直接丢弃该术语
            continue

        if keep_bare_if_has_valid:
            cleaned[term] = qualified_fields | bare_fields
        else:
            cleaned[term] = qualified_fields

    return cleaned
