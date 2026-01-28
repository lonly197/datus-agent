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

logger = get_logger(__name__)


class LLMEnhancedBusinessTermsGenerator:
    """LLM增强的业务术语生成器"""

    def __init__(
        self,
        agent_config=None,
        namespace: str = "",
        min_term_length: int = 2,
        use_llm: bool = False
    ):
        self.min_term_length = min_term_length
        self.use_llm = use_llm
        self.llm_model = None

        if use_llm and agent_config:
            try:
                self.llm_model = LLMBaseModel.create_model(agent_config=agent_config)
                logger.info("LLM model initialized for enhanced extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM model: {e}, falling back to regex mode")
                self.use_llm = False

        self.excel_reader = ExcelReader()
        self.csv_reader = CsvReader()
        self.term_extractor = TermExtractor(min_term_length)

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
            if enhancement.get("enhanced") and enhancement.get("confidence", 0) > 0.6:
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
    ) -> Dict:
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
