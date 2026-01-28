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
from ..shared import (
    TablePriority, 
    get_table_priority, 
    should_include_table,
    clean_excel_text,
    extract_clean_keywords
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class BusinessTermsGenerator:
    """业务术语生成器
    
    支持表优先级过滤：
    - 优先使用 DWD/DWS/DIM 表
    - 其次考虑 ADS 表  
    - 尽量少使用 ODS 表
    """

    def __init__(
        self, 
        min_term_length: int = 2,
        max_table_priority: TablePriority = TablePriority.ADS,
        enable_text_cleaning: bool = True
    ):
        self.min_term_length = min_term_length
        self.max_table_priority = max_table_priority
        self.enable_text_cleaning = enable_text_cleaning
        self.excel_reader = ExcelReader()
        self.csv_reader = CsvReader()
        self.term_extractor = TermExtractor(min_term_length)
        self.keyword_extractor = KeywordExtractor(min_term_length)
        
        # 统计信息
        self.stats = {
            "tables_filtered_by_priority": 0,
            "tables_by_priority": {p.name: 0 for p in TablePriority},
        }

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
        """处理单行数据架构记录（支持表优先级过滤和文本清洗）"""
        # 提取原始字段
        table_name = HeaderParser.extract_field(row, ["物理表名", "table_name", "表名"], debug=(stats['total_rows']==1))
        column_name = HeaderParser.extract_field(row, ["字段名", "column_name", "列名", "column"], debug=(stats['total_rows']==1))
        attr_def = HeaderParser.extract_field(row, ["属性业务定义", "属性定义"], debug=(stats['total_rows']==1))
        attr_cn = HeaderParser.extract_field(row, ["属性（中文）", "属性中文名", "属性名称"], debug=(stats['total_rows']==1))
        obj_name = HeaderParser.extract_field(row, ["分析对象（中文）", "分析对象中文名", "分析对象"], debug=(stats['total_rows']==1))
        obj_en = HeaderParser.extract_field(row, ["分析对象（英文）", "分析对象英文名"], debug=(stats['total_rows']==1))
        logic_entity = HeaderParser.extract_field(row, ["逻辑实体（中文）", "逻辑实体中文名", "逻辑实体"], debug=(stats['total_rows']==1))
        logic_entity_def = HeaderParser.extract_field(row, ["逻辑实体业务含义", "逻辑实体定义", "逻辑实体说明"], debug=(stats['total_rows']==1))

        # 文本清洗
        if self.enable_text_cleaning:
            attr_def = clean_excel_text(attr_def, remove_newlines=True)
            attr_cn = clean_excel_text(attr_cn)
            obj_name = clean_excel_text(obj_name)
            obj_en = clean_excel_text(obj_en)
            logic_entity = clean_excel_text(logic_entity)
            logic_entity_def = clean_excel_text(logic_entity_def, remove_newlines=True)

        # 验证表名和列名
        if not HeaderParser.is_valid_table_name(table_name) or not HeaderParser.is_valid_column_name(column_name):
            return

        # 表优先级过滤
        if not should_include_table(table_name, self.max_table_priority):
            self.stats["tables_filtered_by_priority"] += 1
            return

        # 统计表优先级分布
        priority = get_table_priority(table_name)
        self.stats["tables_by_priority"][priority.name] += 1

        stats["valid_rows"] += 1
        stats["tables_found"].add(table_name)

        # 分析对象 -> 表映射
        if obj_name and len(obj_name) >= self.min_term_length and HeaderParser.is_meaningful_term(obj_name):
            term_to_table[obj_name].add(table_name)
            if obj_en and HeaderParser.is_meaningful_term(obj_en):
                term_to_table[obj_en.lower()].add(table_name)

        # 逻辑实体 -> 表映射（限制关键词长度，避免过长描述）
        if logic_entity and len(logic_entity) >= self.min_term_length and HeaderParser.is_meaningful_term(logic_entity):
            term_to_table[logic_entity].add(table_name)
            # 只有较短的逻辑实体名才作为关键词
            if len(logic_entity) <= 30:
                table_keywords[logic_entity] = table_name

            if logic_entity_def:
                # 使用清洗后的关键词提取
                keywords = extract_clean_keywords(logic_entity_def, min_length=self.min_term_length, max_length=10)
                for kw in keywords:
                    # 只添加有意义且不太长的关键词
                    if len(kw) >= self.min_term_length and len(kw) <= 20:
                        table_keywords[kw] = table_name
                        term_to_table[kw].add(table_name)

        # 属性（中文）-> 字段映射
        if attr_cn and len(attr_cn) >= self.min_term_length and HeaderParser.is_meaningful_term(attr_cn):
            term_to_schema[attr_cn].add(f"{table_name}.{column_name}")
            term_to_schema[attr_cn].add(column_name)

        # 属性业务定义 -> 关键词映射（限制关键词质量和数量）
        if attr_def and len(attr_def) >= self.min_term_length:
            keywords = extract_clean_keywords(attr_def, min_length=self.min_term_length, max_length=10)
            for kw in keywords:
                # 限制关键词长度和质量
                if len(kw) <= 30:
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
                "tables_filtered_by_priority": self.stats["tables_filtered_by_priority"],
                "tables_by_priority": self.stats["tables_by_priority"],
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
        """处理单行指标数据（支持文本清洗和表优先级过滤）"""
        # 提取原始字段
        metric_name = HeaderParser.extract_field(row, ["指标名称"])
        biz_def = HeaderParser.extract_field(row, ["业务定义及说明", "业务定义"])
        biz_activity = HeaderParser.extract_field(row, ["业务活动"])
        category1 = HeaderParser.extract_field(row, ["分类1", "分类一", "一级分类"])
        category2 = HeaderParser.extract_field(row, ["分类2", "分类二", "二级分类"])
        source_model = HeaderParser.extract_field(row, ["来源dws模型", "dws模型", "来源模型", "来源表"])

        # 文本清洗
        if self.enable_text_cleaning:
            metric_name = clean_excel_text(metric_name)
            biz_def = clean_excel_text(biz_def, remove_newlines=True)
            biz_activity = clean_excel_text(biz_activity)
            category1 = clean_excel_text(category1)
            category2 = clean_excel_text(category2)

        if not metric_name:
            return

        # 表优先级过滤
        if source_model and HeaderParser.is_valid_table_name(source_model):
            if not should_include_table(source_model, self.max_table_priority):
                self.stats["tables_filtered_by_priority"] += 1
                return
            
            # 统计表优先级
            priority = get_table_priority(source_model)
            self.stats["tables_by_priority"][priority.name] += 1

            term_to_table[metric_name].add(source_model)

            # 添加业务活动作为命名空间前缀
            if biz_activity:
                composite_term = f"{biz_activity}_{metric_name}"
                term_to_table[composite_term].add(source_model)
                # 单独添加业务活动映射
                if len(biz_activity) >= self.min_term_length:
                    term_to_table[biz_activity].add(source_model)

        # 从业务定义提取关键词（使用清洗后的提取）
        if biz_def:
            keywords = extract_clean_keywords(biz_def, min_length=self.min_term_length, max_length=10)
            for kw in keywords:
                if source_model and should_include_table(source_model, self.max_table_priority):
                    term_to_table[kw].add(source_model)
                    # 只添加有意义且不太短的关键词
                    if len(kw) >= 4 and len(kw) <= 20:
                        table_keywords[kw] = source_model

        # 从指标名称提取术语
        if metric_name:
            metric_terms = self.term_extractor.extract_metric_terms(metric_name)
            for term in metric_terms:
                if source_model and should_include_table(source_model, self.max_table_priority):
                    term_to_table[term].add(source_model)
                    if len(term) >= 2 and len(term) <= 20:
                        table_keywords[term] = source_model

        # 处理分类
        for category in [category1, category2]:
            if category and len(category) >= self.min_term_length and HeaderParser.is_meaningful_term(category):
                clean_cat = category.strip()
                if clean_cat and clean_cat.lower() not in ['nan', 'none', 'null', '']:
                    if source_model and should_include_table(source_model, self.max_table_priority):
                        term_to_table[clean_cat].add(source_model)
                        if len(clean_cat) >= 2 and len(clean_cat) <= 20:
                            table_keywords[clean_cat] = source_model
