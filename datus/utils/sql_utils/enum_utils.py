# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Enum value extraction utilities.

This module provides functions for extracting enum-like values from SQL comments,
particularly useful for parsing Chinese database documentation.
"""

import re
from typing import List, Optional, Set, Tuple

# =============================================================================
# PRE-COMPILED REGEX PATTERNS FOR ENUM EXTRACTION
# =============================================================================

_ENUM_HINT_RE = re.compile(
    r"(枚举|取值|值域|范围|状态|类型|可选|选项|option|enum)",
    re.IGNORECASE,
)

_ENUM_SEGMENT_RE = re.compile(
    r"[（(【\[]([^）)】\]]+)[）)】\]]"
)

_ENUM_CODE_RE = re.compile(
    r"(?:-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,5}|true|false|yes|no|on|off|y|n)",
    re.IGNORECASE,
)

_ENUM_PAIR_RE = re.compile(
    r"(?P<code>-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,5}|true|false|yes|no|on|off|y|n)"
    r"\s*(?P<sep>[:：=\-—~、\.．\)）])\s*"
    r"(?P<label>[^;；,，、/\n]+?)\s*"
    r"(?=(?:-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,5}|true|false|yes|no|on|off|y|n)"
    r"\s*(?:[:：=\-—~、\.．\)）])|$)",
    re.IGNORECASE,
)

# 新增：支持无分隔符结尾的枚举值（如 "（1:长期订单,2:无效订单,3:自动取消）"）
# 匹配模式：code sep label (sep label)*
_ENUM_PAIR_RE_FLEXIBLE = re.compile(
    r"(?P<code>-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,5}|true|false|yes|no|on|off|y|n)"
    r"\s*(?P<sep>[:：=\-—~、\.．])\s*"
    r"(?P<label>(?:[^,，、;；:：=()（）\-—~/\n]|_(?!_))+?)"
    r"\s*(?=(?:\s*(?P<next_code>-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,5}|true|false|yes|no|on|off|y|n)"
    r"\s*[:：=\-—~、\.．\)]?)|[\）\n]|$)",
    re.IGNORECASE,
)


# =============================================================================
# ENUM EXTRACTION FUNCTIONS
# =============================================================================

def extract_enum_values_from_comment(comment: str) -> List[Tuple[str, str]]:
    """
    Extract enum-like code/label pairs from a comment string.

    Example: "客户类型（0：售前客户、1：售后客户）" -> [("0", "售前客户"), ("1", "售后客户")]
    Example: "订单状态（0:录入、1:生产、2:配车）" -> [("0", "录入"), ("1", "生产"), ("2", "配车")]

    Supports multiple formats:
    - 中文括号: "（1:选项1,2:选项2）"
    - 英文括号: "(1:option1,2:option2)"
    - Various separators: ":", "：", "=", "-", "—", "~", "、"
    - Last item without separator
    """
    if not comment or not isinstance(comment, str):
        return []

    allow_single = bool(_ENUM_HINT_RE.search(comment))

    # Prefer enum content inside parentheses/brackets first.
    candidates: List[str] = []
    for segment in _ENUM_SEGMENT_RE.findall(comment):
        segment = segment.strip()
        if segment:
            candidates.append(segment)
    if not candidates:
        candidates = [comment]

    numeric_pairs: List[Tuple[str, str]] = []
    alpha_pairs: List[Tuple[str, str]] = []

    for text in candidates:
        for match in _ENUM_PAIR_RE.finditer(text):
            code = match.group("code").strip()
            label = _clean_label(match.group("label"))
            if not code or not label:
                continue
            if re.fullmatch(r"-?\d+(?:\.\d+)?", code):
                numeric_pairs.append((code, label))
            else:
                alpha_pairs.append((code, label))

        if not numeric_pairs and not alpha_pairs:
            for match in _ENUM_PAIR_RE_FLEXIBLE.finditer(text):
                code = match.group("code").strip()
                label = _clean_label(match.group("label"))
                if not code or not label:
                    continue
                if re.fullmatch(r"-?\d+(?:\.\d+)?", code):
                    numeric_pairs.append((code, label))
                else:
                    alpha_pairs.append((code, label))

        if not numeric_pairs and not alpha_pairs:
            pairs = _parse_enum_pairs_simple(text)
            for code, label in pairs:
                if re.fullmatch(r"-?\d+(?:\.\d+)?", code):
                    numeric_pairs.append((code, label))
                else:
                    alpha_pairs.append((code, label))

    allow_alpha = allow_single or bool(numeric_pairs)
    pairs: List[Tuple[str, str]] = []
    pairs.extend(numeric_pairs)

    if allow_alpha and alpha_pairs:
        for code, label in alpha_pairs:
            code_lower = code.lower()
            if code_lower in {"id", "name", "type", "status"}:
                continue
            if len(code_lower) > 6 and code_lower not in {"true", "false"}:
                continue
            pairs.append((code, label))

    if len(pairs) < 2 and not allow_single:
        return []

    deduped: List[Tuple[str, str]] = []
    seen_codes: Set[str] = set()
    for code, label in pairs:
        code_key = code.lower() if re.search(r"[A-Za-z]", code) else code
        if code_key in seen_codes:
            continue
        seen_codes.add(code_key)
        deduped.append((code, label))

    return deduped


def _parse_enum_pairs_simple(text: str) -> List[Tuple[str, str]]:
    """
    Simple enum pair parser that handles Chinese enumeration formats.

    Example: "0:录入、1:生产、2:配车" -> [("0", "录入"), ("1", "生产"), ("2", "配车")]

    Supports:
    - 数字 codes: 0, 1, 2, -1, 3.14
    - Letter codes: active, yes, true, n
    - Separators: :, ：, =, -, —, ~, 、, ．, .
    - Delimiters: 、, ,, ，, or end of segment
    """
    if not text:
        return []

    result = []
    # 使用更简单的方法：先按分隔符拆分，再解析每对
    # 匹配模式：code (sep) label

    # 1. 首先找到所有 code-label 对
    # 代码模式：数字或字母代码
    code_pattern = re.compile(
        r"^(?P<code>-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,5}|true|false|yes|no|on|off|y|n)"
    )

    # 2. 按分隔符拆分整个文本
    # 分隔符包括：、,, ， 以及它们的组合
    delimiter_pattern = re.compile(r"[、，,]+")

    segments = delimiter_pattern.split(text)
    if len(segments) <= 1:
        # 如果没有分隔符，尝试其他方式
        segments = [text]

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # 3. 在每段中查找 code 和 label
        # 尝试匹配 code (sep) label 模式
        pair_pattern = re.compile(
            r"(?P<code>-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,5}|true|false|yes|no|on|off|y|n)"
            r"\s*[:：=\-—~、\.．]*\s*"
            r"(?P<label>.+)",
            re.IGNORECASE,
        )

        match = pair_pattern.match(segment)
        if match:
            code = match.group("code").strip()
            label = _clean_label(match.group("label"))
            if code and label and len(label) >= 1:
                result.append((code, label))
        else:
            # 如果无法按模式匹配，尝试直接使用整段作为 label
            # 只有当整段不是纯代码时才添加
            label = _clean_label(segment)
            if label and not re.match(r"^\s*(?:-?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]{0,5}|true|false|yes|no|on|off|y|n)\s*$", label):
                # 如果当前已有 pairs，且这是最后一段，可以作为最后一个 label
                pass

    return result


def _clean_label(value: str) -> str:
    """Clean enum label value."""
    cleaned = value.strip()
    return cleaned.strip(" ;；,，、")


def _extract_table_comment_after_columns(sql: str) -> Optional[str]:
    """
    Extract table-level COMMENT after the column list, avoiding column comments.
    """
    if not sql or not isinstance(sql, str):
        return None

    create_match = re.search(r"CREATE\s+TABLE", sql, re.IGNORECASE)
    start_scan = create_match.end() if create_match else 0

    in_single_quote = False
    in_double_quote = False
    in_backtick = False
    escaped = False
    paren_count = 0
    start_idx = -1
    end_idx = -1

    for i in range(start_scan, len(sql)):
        char = sql[i]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_single_quote:
            if char == "'":
                in_single_quote = False
            continue
        if in_double_quote:
            if char == '"':
                in_double_quote = False
            continue
        if in_backtick:
            if char == "`":
                in_backtick = False
            continue
        if char == "'":
            in_single_quote = True
            continue
        if char == '"':
            in_double_quote = True
            continue
        if char == "`":
            in_backtick = True
            continue
        if char == '(':
            if paren_count == 0:
                start_idx = i
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count == 0 and start_idx >= 0:
                end_idx = i
                break

    if end_idx < 0:
        return None

    tail = sql[end_idx + 1 :]
    tail_upper = tail.upper()
    stop_tokens = ("DISTRIBUTED BY", "PROPERTIES", "PARTITION BY", "ORDER BY")
    stop_idx = len(tail)
    for token in stop_tokens:
        idx = tail_upper.find(token)
        if idx != -1:
            stop_idx = min(stop_idx, idx)
    comment_scope = tail[:stop_idx]

    comment_match = re.search(
        r'COMMENT\s*(?:=\s*)?([\'"])(.*?)\1',
        comment_scope,
        re.IGNORECASE | re.DOTALL,
    )
    if not comment_match:
        return None

    return comment_match.group(2)
