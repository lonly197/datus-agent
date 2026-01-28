#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Excel reader for business configuration generation.

This module provides functionality to read Excel files with support for
multi-row headers and complex Excel structures commonly found in 
data architecture design documents.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ExcelReader:
    """Excel文件读取器，支持多行表头智能处理
    
    处理复杂表头结构的数据架构设计文档：
    - 第1行：分组标题
    - 第2行：实际列名（物理表名、字段名等）
    - 第3行：列说明/注释
    """

    def __init__(self):
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available. Install with: pip install pandas openpyxl")

    def read_with_header(
        self,
        xlsx_path: Path,
        header_rows: int,
        sheet_name: Optional[str] = None
    ) -> List[Dict]:
        """
        读取Excel文件，智能处理多行表头

        针对复杂表头结构：
        - 数据架构设计文档：
          - 第1行：分组标题
          - 第2行：实际列名（物理表名、字段名等）★ 使用这行
          - 第3行：列说明/注释

        Args:
            xlsx_path: Excel文件路径
            header_rows: 表头行数
            sheet_name: 工作表名称或索引

        Returns:
            List[Dict]: 数据行列表
        """
        if not PANDAS_AVAILABLE:
            logger.error("pandas not available. Install with: pip install pandas openpyxl")
            return []

        try:
            sheet_name = self._resolve_sheet_name(xlsx_path, sheet_name)
            actual_header_row = header_rows - 2 if header_rows >= 2 else 0

            logger.info(f"[Excel读取] 读取 {xlsx_path.name}, 使用第 {actual_header_row + 1} 行作为列名")

            df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=actual_header_row)

            # 对于2行表头的情况，检查是否需要读取第2行补充列名
            if header_rows == 2:
                df = self._merge_header_rows(xlsx_path, sheet_name, actual_header_row, df)

            df = self._clean_columns(df)
            records = self._df_to_records(df)

            logger.info(f"[Excel读取] 有效数据行: {len(records)}")
            return records

        except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError, 
                PermissionError, OSError, ValueError) as e:
            logger.error(f"[Excel读取] 失败: {type(e).__name__}: {e}")
            return []

    def _resolve_sheet_name(
        self,
        xlsx_path: Path,
        sheet_name: Optional[str]
    ) -> str:
        """解析工作表名称或索引"""
        if sheet_name is not None:
            try:
                return int(sheet_name)
            except ValueError:
                return sheet_name

        xl = pd.ExcelFile(xlsx_path)
        sheet_names = xl.sheet_names

        if "Sheet1" in sheet_names:
            logger.info(f"[Excel读取] 自动选择工作表: 'Sheet1'")
            return "Sheet1"
        else:
            logger.info(f"[Excel读取] 自动选择第一个工作表: '{sheet_names[0]}'")
            return 0

    def _merge_header_rows(
        self,
        xlsx_path: Path,
        sheet_name: str,
        actual_header_row: int,
        df
    ):
        """合并多行表头的补充列名"""
        df_second_row = pd.read_excel(
            xlsx_path,
            sheet_name=sheet_name,
            header=None,
            skiprows=actual_header_row + 1,
            nrows=1
        )

        if df_second_row.empty:
            return df

        second_row_values = df_second_row.iloc[0].tolist()
        original_columns = df.columns.tolist()

        merged_columns = []
        for i, col in enumerate(original_columns):
            col_str = str(col) if col is not None else ""
            if 'Unnamed' in col_str and i < len(second_row_values):
                second_val = str(second_row_values[i]) if second_row_values[i] is not None else ""
                if second_val and second_val != 'nan':
                    merged_columns.append(second_val)
                else:
                    merged_columns.append(col_str)
            else:
                merged_columns.append(col_str)

        if merged_columns != original_columns:
            logger.info(f"[Excel读取] 检测到第2行补充列名，已合并")
            df.columns = merged_columns

        return df

    def _clean_columns(self, df) -> 'pd.DataFrame':
        """清理列名，处理空值和重复"""
        def clean_col(col):
            if pd.isna(col):
                return f"Unnamed_{id(col)}"
            s = str(col).strip()
            s = ' '.join(s.split())
            return s

        clean_columns = [clean_col(col) for col in df.columns]
        seen = {}
        unique_columns = []

        for col in clean_columns:
            if col in seen:
                seen[col] += 1
                unique_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_columns.append(col)

        df.columns = unique_columns

        logger.info(f"[Excel读取] 清理后的列名(前30): {df.columns.tolist()[:30]}")
        if len(df.columns) > 30:
            logger.info(f"[Excel读取] 清理后的列名(31-): {df.columns.tolist()[30:]}")

        return df

    def _df_to_records(self, df) -> List[Dict]:
        """将DataFrame转换为字典列表"""
        records = []

        for idx, row in df.iterrows():
            if row.isna().all():
                continue

            record = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, pd.Series):
                    val = val.dropna().iloc[0] if not val.dropna().empty else ""
                if pd.isna(val):
                    record[col] = ""
                else:
                    record[col] = str(val).strip()
            records.append(record)

        return records
