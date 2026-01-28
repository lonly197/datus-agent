#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Header parsing utilities for business configuration generation.
"""

import re
from typing import Dict, List, Optional

from datus.utils.loggings import get_logger

from ..shared import STOP_WORDS, SYNONYM_MAP, is_meaningful_term

logger = get_logger(__name__)


class HeaderParser:
    """表头解析器，支持多列名映射和模糊匹配"""

    @staticmethod
    def extract_field(
        row: Dict,
        possible_names: List[str],
        debug: bool = False
    ) -> str:
        """
        从行中提取字段值，尝试多个可能的列名

        支持多种匹配策略：
        1. 精确匹配
        2. 大小写不敏感匹配
        3. 前缀匹配
        4. 子串匹配

        Args:
            row: 数据行字典
            possible_names: 可能的列名列表
            debug: 是否打印调试信息

        Returns:
            str: 提取的字段值
        """
        row_keys = list(row.keys())

        for name in possible_names:
            # 1. 精确匹配
            if name in row and row[name]:
                return HeaderParser.clean_string(row[name])

            # 2. 大小写不敏感匹配
            for key in row_keys:
                if key and key.lower() == name.lower():
                    return HeaderParser.clean_string(row[key])

            # 3. 前缀匹配
            for key in row_keys:
                if key and (key.startswith(name) or name in key):
                    if row[key]:
                        return HeaderParser.clean_string(row[key])

        if debug:
            logger.debug(f"[字段提取] 未找到匹配: {possible_names}, 可用列: {row_keys[:15]}...")
        return ""

    @staticmethod
    def clean_string(s: str) -> str:
        """清理字符串，去除空白和特殊字符"""
        if not s:
            return ""
        return s.strip().replace("\n", " ").replace("\r", "").replace("\t", " ")

    @staticmethod
    def is_valid_table_name(name: str, verbose: bool = False) -> bool:
        """验证是否为有效的表名"""
        if not name or len(name) < 2:
            if verbose and name:
                logger.debug(f"[过滤] 表名太短或为空: '{name}'")
            return False

        if not re.match(r'^[a-zA-Z]', name):
            if verbose:
                logger.debug(f"[过滤] 表名不以字母开头: '{name}'")
            return False

        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            if verbose:
                logger.debug(f"[过滤] 表名含非法字符: '{name}'")
            return False

        return True

    @staticmethod
    def is_valid_column_name(name: str) -> bool:
        """验证是否为有效的字段名"""
        if not name or len(name) < 1:
            return False
        if not re.match(r'^[a-zA-Z_]', name):
            return False
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return False
        return True

    @staticmethod
    def is_meaningful_term(term: str) -> bool:
        """判断术语是否有业务意义（委托给 shared 模块的统一实现）"""
        return is_meaningful_term(term)
