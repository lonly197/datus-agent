#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
CSV reader for business configuration generation.
Handles multi-line headers and complex CSV structures.
"""

import csv
import io
from pathlib import Path
from typing import Dict, List

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CsvReader:
    """CSV文件读取器，支持多行表头智能检测"""

    # 数据架构CSV的关键列名
    ARCH_EXPECTED_COLUMNS = ["物理表名", "字段名", "属性（中文）", "分析对象（中文）"]

    # 指标清单CSV的关键列名
    METRICS_EXPECTED_COLUMNS = ["业务活动", "指标编码", "指标名称"]

    def read_architecture_csv(
        self,
        csv_path: Path,
        min_term_length: int = 2
    ) -> List[Dict]:
        """
        从数据架构CSV生成业务术语映射

        处理多行表头CSV（前5行为说明/标题行）

        Args:
            csv_path: 数据架构详细设计CSV文件路径
            min_term_length: 最小术语长度

        Returns:
            List[Dict]: 解析后的数据行
        """
        logger.info(f"Processing architecture CSV: {csv_path}")

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        header_row_idx = self._find_header_row(lines, self.ARCH_EXPECTED_COLUMNS)

        csv_content = ''.join(lines[header_row_idx:])
        reader = csv.DictReader(io.StringIO(csv_content))

        if reader.fieldnames:
            logger.info(f"[CSV解析] 数据架构文件列名: {list(reader.fieldnames)}")

        return list(reader)

    def read_metrics_csv(
        self,
        csv_path: Path,
        min_term_length: int = 2
    ) -> List[Dict]:
        """
        从指标清单CSV读取数据

        处理多行表头CSV（前7行为说明/标题行）

        Args:
            csv_path: 指标清单CSV文件路径
            min_term_length: 最小术语长度

        Returns:
            List[Dict]: 解析后的数据行
        """
        logger.info(f"Processing metrics CSV: {csv_path}")

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        header_row_idx = self._find_header_row(lines, self.METRICS_EXPECTED_COLUMNS, default=7)

        csv_content = ''.join(lines[header_row_idx:])
        reader = csv.DictReader(io.StringIO(csv_content))

        if reader.fieldnames:
            logger.info(f"[CSV解析] 指标清单文件列名: {list(reader.fieldnames)}")

        return list(reader)

    def _find_header_row(
        self,
        lines: List[str],
        expected_columns: List[str],
        default: int = 0
    ) -> int:
        """
        找到包含关键列名的行作为表头

        Args:
            lines: CSV文件的所有行
            expected_columns: 期望的列名列表
            default: 默认表头行索引

        Returns:
            int: 表头行索引
        """
        for i, line in enumerate(lines[:20]):
            match_count = sum(1 for col in expected_columns if col in line)
            if match_count >= 2:
                logger.info(f"[CSV解析] 找到表头在第 {i+1} 行: {line[:100]}...")
                return i

        logger.info(f"[CSV解析] 未找到期望列名，使用默认第 {default + 1} 行")
        return default
