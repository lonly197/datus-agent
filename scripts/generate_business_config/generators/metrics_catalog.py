#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Metrics catalog generator for business configuration.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..readers import ExcelReader, HeaderParser
from ..extractors import TermExtractor
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class MetricsCatalogGenerator:
    """指标目录生成器"""

    def __init__(self, min_term_length: int = 2):
        self.min_term_length = min_term_length
        self.excel_reader = ExcelReader()
        self.term_extractor = TermExtractor(min_term_length)

    def generate(
        self,
        xlsx_path: Path,
        header_rows: int = 2,
        sheet_name: Optional[str] = None
    ) -> List[Dict]:
        """
        从指标清单Excel生成指标目录

        Returns:
            List[Dict]: 指标目录条目列表
        """
        logger.info(f"Generating metrics catalog from: {xlsx_path}")

        records = self.excel_reader.read_with_header(xlsx_path, header_rows, sheet_name)

        if not records:
            logger.warning("[指标目录] 未读取到有效数据")
            return []

        return self._process_records(records)

    def _process_records(self, records: List[Dict]) -> List[Dict]:
        """处理记录列表生成指标目录"""
        metrics = []

        for row in records:
            entry = self._process_row(row)
            if entry:
                metrics.append(entry)

        logger.info(f"[指标目录] 生成了 {len(metrics)} 个指标条目")
        return metrics

    def _process_row(self, row: Dict) -> Optional[Dict]:
        """处理单行数据"""
        metric_code = HeaderParser.extract_field(row, ["指标编码", "指标编码 -固定值（勿改）"])
        metric_name = HeaderParser.extract_field(row, ["指标名称"])

        if not metric_code or not metric_name:
            return None

        biz_def = HeaderParser.extract_field(row, ["业务定义及说明", "业务定义"])
        calc_logic = HeaderParser.extract_field(row, ["计算公式/业务逻辑", "计算公式", "业务逻辑"])
        dimensions = HeaderParser.extract_field(row, ["公共维度", "维度"])
        source_model = HeaderParser.extract_field(row, ["来源dws模型", "dws模型", "来源模型", "来源表"])
        biz_activity = HeaderParser.extract_field(row, ["业务活动"])
        category1 = HeaderParser.extract_field(row, ["分类1", "分类一", "一级分类"])
        category2 = HeaderParser.extract_field(row, ["分类2", "分类二", "二级分类"])

        explanation_parts = []
        if biz_def:
            explanation_parts.append(f"【业务定义】{biz_def}")
        if calc_logic:
            explanation_parts.append(f"【计算逻辑】{calc_logic}")
        if dimensions:
            explanation_parts.append(f"【分析维度】{dimensions}")
        if source_model:
            explanation_parts.append(f"【来源表】{source_model}")
        if category1 or category2:
            cats = [c for c in [category1, category2] if c]
            explanation_parts.append(f"【业务分类】{' / '.join(cats)}")

        explanation = "\n".join(explanation_parts) if explanation_parts else biz_def or metric_name

        subject_path = ["Metrics"]
        if biz_activity:
            subject_path.append(biz_activity)
        if category1:
            subject_path.append(category1)

        return {
            "subject_path": subject_path,
            "name": metric_code,
            "terminology": metric_name,
            "explanation": explanation,
        }
