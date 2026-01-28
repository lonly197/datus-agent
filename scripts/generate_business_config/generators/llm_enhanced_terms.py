#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
LLM-enhanced business terms generator.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from datus.models.base import LLMBaseModel
from datus.utils.loggings import get_logger

from ..readers import ExcelReader, CsvReader, HeaderParser
from ..extractors import TermExtractor
from ..processors import LLMTextRewriter
from ..shared import (
    TablePriority,
    get_table_priority,
    should_include_table,
    clean_excel_text,
    extract_clean_keywords
)

logger = get_logger(__name__)


class LLMEnhancedBusinessTermsGenerator:
    """LLM增强的业务术语生成器
    
    特性：
    - 表优先级过滤（DWD/DWS/DIM > ADS > ODS）
    - 文本清洗（去除emoji、序号、特殊符号等）
    - LLM智能改写指标定义
    """

    def __init__(
        self,
        agent_config=None,
        namespace: str = "",
        min_term_length: int = 2,
        use_llm: bool = False,
        max_table_priority: TablePriority = TablePriority.ADS,
        enable_text_cleaning: bool = True
    ):
        self.min_term_length = min_term_length
        self.use_llm = use_llm
        self.max_table_priority = max_table_priority
        self.enable_text_cleaning = enable_text_cleaning
        self.llm_model = None
        self.text_rewriter = None

        if use_llm and agent_config:
            try:
                self.llm_model = LLMBaseModel.create_model(agent_config=agent_config)
                self.text_rewriter = LLMTextRewriter(agent_config, use_llm=True)
                logger.info("LLM model initialized for enhanced extraction and rewriting")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM model: {e}, falling back to regex mode")
                self.use_llm = False

        self.excel_reader = ExcelReader()
        self.csv_reader = CsvReader()
        self.term_extractor = TermExtractor(min_term_length)
        
        # 统计信息
        self.stats = {
            "tables_filtered_by_priority": 0,
            "tables_by_priority": {p.name: 0 for p in TablePriority},
            "llm_rewrites": 0,
        }

    def generate_from_architecture_xlsx(
        self,
        xlsx_path: Path,
        header_rows: int = 3,
        sheet_name: Optional[str] = None
    ) -> Dict:
        """从数据架构Excel生成业务术语映射（LLM增强版）"""
        logger.info(f"Processing architecture Excel with LLM: {xlsx_path}")

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

        # LLM增强
        if self.use_llm and self.llm_model:
            logger.info("Starting LLM enhancement for term disambiguation...")
            term_to_table, term_to_schema = self._llm_enhance_terms(term_to_table, term_to_schema)

        return self._build_result(term_to_table, term_to_schema, table_keywords, stats)

    def _process_architecture_row(
        self,
        row: Dict,
        term_to_table: Dict[str, Set[str]],
        term_to_schema: Dict[str, Set[str]],
        table_keywords: Dict[str, str],
        stats: Dict
    ):
        """处理单行数据架构记录（支持表优先级过滤、文本清洗和LLM改写）"""
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

        # 逻辑实体 -> 表映射
        if logic_entity and len(logic_entity) >= self.min_term_length and HeaderParser.is_meaningful_term(logic_entity):
            term_to_table[logic_entity].add(table_name)
            if len(logic_entity) <= 30:
                table_keywords[logic_entity] = table_name

            if logic_entity_def:
                keywords = extract_clean_keywords(logic_entity_def, min_length=self.min_term_length, max_length=10)
                for kw in keywords:
                    if len(kw) >= self.min_term_length and len(kw) <= 20:
                        table_keywords[kw] = table_name
                        term_to_table[kw].add(table_name)

        # 属性（中文）-> 字段映射
        if attr_cn and len(attr_cn) >= self.min_term_length and HeaderParser.is_meaningful_term(attr_cn):
            term_to_schema[attr_cn].add(f"{table_name}.{column_name}")
            term_to_schema[attr_cn].add(column_name)

            # 使用LLM改写字段定义（如果启用）
            if self.use_llm and self.text_rewriter and attr_def:
                try:
                    rewritten = self.text_rewriter.rewrite_field_definition(
                        column_name, attr_cn, attr_def
                    )
                    for term in rewritten.get("search_terms", []):
                        if len(term) <= 30:
                            term_to_schema[term].add(column_name)
                            term_to_schema[term].add(f"{table_name}.{column_name}")
                    self.stats["llm_rewrites"] += 1
                except Exception as e:
                    logger.debug(f"LLM改写失败 '{column_name}': {e}")

        # 属性业务定义 -> 关键词映射
        if attr_def and len(attr_def) >= self.min_term_length:
            keywords = extract_clean_keywords(attr_def, min_length=self.min_term_length, max_length=10)
            for kw in keywords:
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
                "total_rows": stats.get("total_rows", 0),
                "valid_rows": stats.get("valid_rows", 0),
                "tables_count": len(stats.get("tables_found", set())),
                "terms_count": stats.get("terms_extracted", 0),
                "tables_filtered_by_priority": self.stats["tables_filtered_by_priority"],
                "tables_by_priority": self.stats["tables_by_priority"],
                "llm_rewrites": self.stats["llm_rewrites"],
            },
        }

    def _llm_enhance_terms(
        self,
        term_to_table: Dict[str, Set[str]],
        term_to_schema: Dict[str, Set[str]]
    ) -> tuple:
        """使用LLM增强术语理解"""
        # 对于有歧义的术语（映射到多个表），使用LLM进行消歧
        ambiguous_terms = {
            term: tables for term, tables in term_to_table.items()
            if len(tables) > 1
        }

        if not ambiguous_terms:
            logger.info("No ambiguous terms found for LLM enhancement")
            return term_to_table, term_to_schema

        logger.info(f"Found {len(ambiguous_terms)} ambiguous terms for LLM enhancement")

        enhanced_count = 0
        for term, tables in ambiguous_terms.items():
            enhancement = self._llm_enhance_term_extraction(term, list(tables))
            if enhancement and enhancement.get("enhanced") and enhancement.get("confidence", 0) > 0.6:
                primary_table = enhancement.get("primary_table", "")
                if primary_table and primary_table in tables:
                    # 重新排序，将primary表放在首位
                    ordered_tables = [primary_table] + [t for t in tables if t != primary_table]
                    term_to_table[term] = set(ordered_tables)
                    enhanced_count += 1

        logger.info(f"LLM enhanced {enhanced_count}/{len(ambiguous_terms)} ambiguous terms")
        return term_to_table, term_to_schema

    def _llm_enhance_term_extraction(
        self,
        term: str,
        table_candidates: List[str]
    ) -> Optional[Dict]:
        """使用LLM增强术语理解"""
        if not self.use_llm or not self.llm_model:
            return {"enhanced": False, "confidence": 0.0}

        prompt = f"""You are a data warehouse business analyst. Analyze the business term and determine the most relevant tables.

## Input
Business Term: {term}
Candidate Tables: {table_candidates}

## Task
1. Analyze the semantic meaning of the business term
2. Evaluate relevance of each candidate table (score 0-1)
3. Identify if this term has ambiguity

## Output Format
Return ONLY a JSON object:
{{
  "primary_table": "most relevant table name",
  "primary_confidence": 0.0-1.0,
  "ambiguity": true/false,
  "confidence": 0.0-1.0
}}
"""
        try:
            response = self.llm_model.generate_with_json_output(prompt)
            if isinstance(response, dict):
                return {
                    "enhanced": True,
                    "primary_table": response.get("primary_table", ""),
                    "primary_confidence": response.get("primary_confidence", 0.0),
                    "ambiguity": response.get("ambiguity", False),
                    "confidence": response.get("confidence", 0.0),
                }
        except Exception as e:
            logger.debug(f"LLM enhancement failed for term '{term}': {e}")

        return {"enhanced": False, "confidence": 0.0}

    def generate_from_metrics_xlsx(
        self,
        xlsx_path: Path,
        existing_terms: Dict,
        header_rows: int = 2,
        sheet_name: Optional[str] = None
    ) -> Dict:
        """从指标清单Excel提取业务术语（LLM增强版）"""
        logger.info(f"Processing metrics Excel with LLM: {xlsx_path}")

        term_to_table = defaultdict(set, existing_terms.get("term_to_table", {}))
        table_keywords = existing_terms.get("table_keywords", {})
        term_to_schema = defaultdict(set, existing_terms.get("term_to_schema", {}))

        records = self.excel_reader.read_with_header(xlsx_path, header_rows, sheet_name)

        if not records:
            logger.warning("[Excel读取] 指标清单未读取到有效数据")
            return self._build_result(term_to_table, term_to_schema, table_keywords, {"valid_metrics": 0})

        for row in records:
            self._process_metrics_row(row, term_to_table, term_to_schema, table_keywords)

        # LLM增强：对指标术语进行消歧
        if self.use_llm and self.llm_model:
            logger.info("Starting LLM enhancement for metrics term disambiguation...")
            term_to_table, term_to_schema = self._llm_enhance_terms(term_to_table, term_to_schema)

        logger.info(f"[Excel解析] 指标清单处理完成: {len(records)} 个指标")

        return self._build_result(term_to_table, term_to_schema, table_keywords, {"valid_metrics": len(records)})

    def _process_metrics_row(
        self,
        row: Dict,
        term_to_table: Dict[str, Set[str]],
        term_to_schema: Dict[str, Set[str]],
        table_keywords: Dict[str, str]
    ):
        """处理单行指标数据（支持文本清洗、表优先级过滤和LLM改写）"""
        # 提取原始字段
        metric_name = HeaderParser.extract_field(row, ["指标名称"])
        biz_def = HeaderParser.extract_field(row, ["业务定义及说明", "业务定义"])
        biz_activity = HeaderParser.extract_field(row, ["业务活动"])
        category1 = HeaderParser.extract_field(row, ["分类1", "分类一", "一级分类"])
        category2 = HeaderParser.extract_field(row, ["分类2", "分类二", "二级分类"])
        source_model = HeaderParser.extract_field(row, ["来源dws模型", "dws模型", "来源模型", "来源表"])
        calc_logic = HeaderParser.extract_field(row, ["计算公式/业务逻辑", "计算公式", "计算逻辑"])

        # 文本清洗
        if self.enable_text_cleaning:
            metric_name = clean_excel_text(metric_name)
            biz_def = clean_excel_text(biz_def, remove_newlines=True)
            biz_activity = clean_excel_text(biz_activity)
            category1 = clean_excel_text(category1)
            category2 = clean_excel_text(category2)
            calc_logic = clean_excel_text(calc_logic, remove_newlines=True)

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

            # 添加业务活动映射
            if biz_activity:
                composite_term = f"{biz_activity}_{metric_name}"
                term_to_table[composite_term].add(source_model)
                if len(biz_activity) >= self.min_term_length:
                    term_to_table[biz_activity].add(source_model)

            # 使用LLM改写指标定义（如果启用）
            if self.use_llm and self.text_rewriter and biz_def:
                try:
                    rewritten = self.text_rewriter.rewrite_metric_definition(
                        metric_name, biz_def, calc_logic
                    )
                    
                    # 添加核心概念
                    for concept in rewritten.get("core_concepts", []):
                        if len(concept) <= 20:
                            term_to_table[concept].add(source_model)
                    
                    # 添加检索关键词
                    for term in rewritten.get("search_terms", []):
                        if len(term) <= 20:
                            term_to_table[term].add(source_model)
                            table_keywords[term] = source_model
                    
                    # 添加同义词映射
                    for term, synonyms in rewritten.get("synonyms", {}).items():
                        for syn in synonyms:
                            if len(syn) <= 20:
                                term_to_table[syn].add(source_model)
                    
                    self.stats["llm_rewrites"] += 1
                except Exception as e:
                    logger.debug(f"LLM改写失败 '{metric_name}': {e}")

        # 从业务定义提取关键词（使用清洗后的提取）
        if biz_def:
            keywords = extract_clean_keywords(biz_def, min_length=self.min_term_length, max_length=10)
            for kw in keywords:
                if source_model and should_include_table(source_model, self.max_table_priority):
                    term_to_table[kw].add(source_model)
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
