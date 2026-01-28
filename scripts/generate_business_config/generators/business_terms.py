#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Business terms generator for business configuration.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..readers import ExcelReader, CsvReader, HeaderParser
from ..extractors import TermExtractor, KeywordExtractor
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class BusinessTermsGenerator:
    """业务术语生成器"""

    def __init__(self, min_term_length: int = 2):
        self.min_term_length = min_term_length
        self.excel_reader = ExcelReader()
        self.csv_reader = CsvReader()
        self.term_extractor = TermExtractor(min_term_length)
        self.keyword_extractor = KeywordExtractor(min_term_length)

    def generate_from_architecture_xlsx(
        self,
        xlsx_path: Path,
        header_rows: int = 3,
        sheet_name: Optional[str] = None
    ) -> Dict:
        """从数据架构Excel生成业务术语映射"""
        logger.info(f"Processing architecture Excel: {xlsx_path}")

        term_to_table: Dict[str, Set[str]] = defaultdict(set)
        term_to_schema: Dict[str, Set[str]] = defaultdict(set)
        table_keywords: Dict[str, str] = {}

        stats = {
            "total_rows": 0,
            "valid_rows": 0,
            "tables_found": set(),
            "terms_extracted": 0,
        }

        records = self.excel_reader.read_with_header(xlsx_path, header_rows, sheet_name)

        if not records:
            logger.warning("[Excel读取] 未读取到有效数据")
            return self._empty_result()

        for row in records:
            stats["total_rows"] += 1
            self._process_architecture_row(row, term_to_table, term_to_schema, table_keywords, stats)

        logger.info(
            f"[Excel解析] 数据架构处理完成: {stats['valid_rows']}/{stats['total_rows']} 有效行, "
            f"{len(stats['tables_found'])} 个表, "
            f"{stats['terms_extracted']} 个术语提取"
        )

        return self._build_result(term_to_table, term_to_schema, table_keywords, stats)

    def _process_architecture_row(
        self,
        row: Dict,
        term_to_table: Dict[str, Set[str]],
        term_to_schema: Dict[str, Set[str]],
        table_keywords: Dict[str, str],
        stats: Dict
    ):
        """处理单行数据架构记录"""
        table_name = HeaderParser.extract_field(row, ["物理表名", "table_name", "表名"], debug=(stats['total_rows']==1))
        column_name = HeaderParser.extract_field(row, ["字段名", "column_name", "列名", "column"], debug=(stats['total_rows']==1))
        attr_def = HeaderParser.extract_field(row, ["属性业务定义", "属性定义"], debug=(stats['total_rows']==1))
        attr_cn = HeaderParser.extract_field(row, ["属性（中文）", "属性中文名", "属性名称"], debug=(stats['total_rows']==1))
        obj_name = HeaderParser.extract_field(row, ["分析对象（中文）", "分析对象中文名", "分析对象"], debug=(stats['total_rows']==1))
        obj_en = HeaderParser.extract_field(row, ["分析对象（英文）", "分析对象英文名"], debug=(stats['total_rows']==1))
        logic_entity = HeaderParser.extract_field(row, ["逻辑实体（中文）", "逻辑实体中文名", "逻辑实体"], debug=(stats['total_rows']==1))
        logic_entity_def = HeaderParser.extract_field(row, ["逻辑实体业务含义", "逻辑实体定义", "逻辑实体说明"], debug=(stats['total_rows']==1))

        if not HeaderParser.is_valid_table_name(table_name) or not HeaderParser.is_valid_column_name(column_name):
            return

        stats["valid_rows"] += 1
        stats["tables_found"].add(table_name)

        # 分析对象 -> 表映射
        if obj_name and len(obj_name) >= self.min_term_length and HeaderParser.is_meaningful_term(obj_name):
            term_to_table[obj_name].add(table_name)
            if obj_en and HeaderParser.is_meaningful_term(obj_en):
                term_to_table[obj_en.lower()].add(table_name)

        # 逻辑实体 -> 表映射
        if logic_entity and len(logic_entity) >= self.min_term_length and HeaderParser.is_meaningful_term(logic_entity):
            term_to_table[logic_entity].add(table_name)
            table_keywords[logic_entity] = table_name

            if logic_entity_def:
                keywords = self.term_extractor.extract_meaningful_keywords(logic_entity_def)
                for kw in keywords:
                    table_keywords[kw] = table_name
                    term_to_table[kw].add(table_name)

        # 属性（中文）-> 字段映射
        if attr_cn and len(attr_cn) >= self.min_term_length and HeaderParser.is_meaningful_term(attr_cn):
            term_to_schema[attr_cn].add(f"{table_name}.{column_name}")
            term_to_schema[attr_cn].add(column_name)

        # 属性业务定义 -> 关键词映射
        if attr_def and len(attr_def) >= self.min_term_length:
            keywords = self.term_extractor.extract_meaningful_keywords(attr_def)
            for kw in keywords:
                if len(kw) <= 50:
                    term_to_schema[kw].add(column_name)
                    term_to_schema[kw].add(f"{table_name}.{column_name}")
                    stats["terms_extracted"] += 1

    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            "term_to_table": {},
            "term_to_schema": {},
            "table_keywords": {},
            "_stats": {"total_rows": 0, "valid_rows": 0, "tables_count": 0, "terms_count": 0},
        }

    def _build_result(
        self,
        term_to_table: Dict[str, Set[str]],
        term_to_schema: Dict[str, Set[str]],
        table_keywords: Dict[str, str],
        stats: Dict
    ) -> Dict:
        """构建结果字典"""
        return {
            "term_to_table": dict(term_to_table),
            "term_to_schema": dict(term_to_schema),
            "table_keywords": table_keywords,
            "_stats": {
                "total_rows": stats["total_rows"],
                "valid_rows": stats["valid_rows"],
                "tables_count": len(stats["tables_found"]),
                "terms_count": stats["terms_extracted"],
            },
        }

    def generate_from_architecture_csv(self, csv_path: Path) -> Dict:
        """从数据架构CSV生成业务术语映射"""
        logger.info(f"Processing architecture CSV: {csv_path}")

        term_to_table: Dict[str, Set[str]] = defaultdict(set)
        term_to_schema: Dict[str, Set[str]] = defaultdict(set)
        table_keywords: Dict[str, str] = {}

        stats = {"total_rows": 0, "valid_rows": 0, "tables_found": set(), "terms_extracted": 0}

        records = self.csv_reader.read_architecture_csv(csv_path, self.min_term_length)

        for row in records:
            stats["total_rows"] += 1

            table_name = HeaderParser.extract_field(row, ["物理表名", "table_name", "表名"])
            column_name = HeaderParser.extract_field(row, ["字段名", "column_name", "列名"])
            attr_def = HeaderParser.extract_field(row, ["属性业务定义", "业务定义", "注释", "comment"])
            attr_cn = HeaderParser.extract_field(row, ["属性（中文）", "属性", "column_cn", "字段中文名"])
            obj_name = HeaderParser.extract_field(row, ["分析对象（中文）", "分析对象", "object_cn"])
            obj_en = HeaderParser.extract_field(row, ["分析对象（英文）", "object_en"])
            logic_entity = HeaderParser.extract_field(row, ["逻辑实体（中文）", "逻辑实体", "entity_cn", "逻辑表名"])
            logic_entity_def = HeaderParser.extract_field(row, ["逻辑实体业务含义", "实体业务含义", "entity_def"])

            if not HeaderParser.is_valid_table_name(table_name) or not HeaderParser.is_valid_column_name(column_name):
                continue

            stats["valid_rows"] += 1
            stats["tables_found"].add(table_name)

            if obj_name and len(obj_name) >= self.min_term_length and HeaderParser.is_meaningful_term(obj_name):
                term_to_table[obj_name].add(table_name)
                if obj_en and HeaderParser.is_meaningful_term(obj_en):
                    term_to_table[obj_en.lower()].add(table_name)

            if logic_entity and len(logic_entity) >= self.min_term_length and HeaderParser.is_meaningful_term(logic_entity):
                term_to_table[logic_entity].add(table_name)
                table_keywords[logic_entity] = table_name

                if logic_entity_def:
                    keywords = self.term_extractor.extract_meaningful_keywords(logic_entity_def)
                    for kw in keywords:
                        table_keywords[kw] = table_name
                        term_to_table[kw].add(table_name)

            if attr_cn and len(attr_cn) >= self.min_term_length and HeaderParser.is_meaningful_term(attr_cn):
                term_to_schema[attr_cn].add(f"{table_name}.{column_name}")
                term_to_schema[attr_cn].add(column_name)

            if attr_def and len(attr_def) >= self.min_term_length:
                keywords = self.term_extractor.extract_meaningful_keywords(attr_def)
                for kw in keywords:
                    if len(kw) <= 50:
                        term_to_schema[kw].add(column_name)
                        term_to_schema[kw].add(f"{table_name}.{column_name}")
                        stats["terms_extracted"] += 1

        logger.info(
            f"[CSV解析] 数据架构处理完成: {stats['valid_rows']}/{stats['total_rows']} 有效行"
        )

        return self._build_result(term_to_table, term_to_schema, table_keywords, stats)

    def generate_from_metrics_xlsx(
        self,
        xlsx_path: Path,
        existing_terms: Dict,
        header_rows: int = 2,
        sheet_name: Optional[str] = None
    ) -> Dict:
        """从指标清单Excel提取业务术语"""
        logger.info(f"Processing metrics Excel for text2sql: {xlsx_path}")

        term_to_table = defaultdict(set, existing_terms.get("term_to_table", {}))
        table_keywords = existing_terms.get("table_keywords", {})
        term_to_schema = defaultdict(set, existing_terms.get("term_to_schema", {}))

        records = self.excel_reader.read_with_header(xlsx_path, header_rows, sheet_name)

        if not records:
            return self._empty_result()

        for row in records:
            self._process_metrics_row(row, term_to_table, term_to_schema, table_keywords)

        return self._build_result(term_to_table, term_to_schema, table_keywords, {"valid_metrics": len(records)})

    def _process_metrics_row(
        self,
        row: Dict,
        term_to_table: Dict[str, Set[str]],
        term_to_schema: Dict[str, Set[str]],
        table_keywords: Dict[str, str]
    ):
        """处理单行指标数据"""
        metric_name = HeaderParser.extract_field(row, ["指标名称"])
        biz_def = HeaderParser.extract_field(row, ["业务定义及说明", "业务定义"])
        biz_activity = HeaderParser.extract_field(row, ["业务活动"])
        category1 = HeaderParser.extract_field(row, ["分类1", "分类一", "一级分类"])
        category2 = HeaderParser.extract_field(row, ["分类2", "分类二", "二级分类"])
        source_model = HeaderParser.extract_field(row, ["来源dws模型", "dws模型", "来源模型", "来源表"])

        if not metric_name:
            return

        if source_model and HeaderParser.is_valid_table_name(source_model):
            term_to_table[metric_name].add(source_model)

            if biz_activity:
                composite_term = f"{biz_activity}_{metric_name}"
                term_to_table[composite_term].add(source_model)

        if biz_def:
            keywords = self.term_extractor.extract_business_keywords(biz_def)
            for kw in keywords:
                if source_model and HeaderParser.is_valid_table_name(source_model):
                    term_to_table[kw].add(source_model)
                    if len(kw) >= 4:
                        table_keywords[kw] = source_model

        if metric_name:
            metric_terms = self.term_extractor.extract_metric_terms(metric_name)
            for term in metric_terms:
                if source_model and HeaderParser.is_valid_table_name(source_model):
                    term_to_table[term].add(source_model)
                    if len(term) >= 2:
                        table_keywords[term] = source_model

        for category in [category1, category2]:
            if category and len(category) >= self.min_term_length and HeaderParser.is_meaningful_term(category):
                clean_cat = category.strip()
                if clean_cat and clean_cat.lower() not in ['nan', 'none', 'null', '']:
                    if source_model and HeaderParser.is_valid_table_name(source_model):
                        term_to_table[clean_cat].add(source_model)
                        if len(clean_cat) >= 2:
                            table_keywords[clean_cat] = source_model
