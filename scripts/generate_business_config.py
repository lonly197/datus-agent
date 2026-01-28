#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Business Configuration Generator

结合数据架构设计文档和DDL元数据，生成/更新业务术语配置文件 business_terms.yml。
支持两种输入格式：
1. Excel文件 (.xlsx) - 推荐，支持多行表头（原始格式）
2. CSV文件 (.csv) - 兼容，需确保表头正确

使用示例：
    # 仅从数据架构生成
    python scripts/generate_business_config.py \
        --config=conf/agent.yml \
        --namespace=test \
        --arch-csv=/path/to/数据架构详细设计v2.3.csv

    # 从数据架构和指标清单生成
    python scripts/generate_business_config.py \
        --config=conf/agent.yml \
        --namespace=test \
        --arch-csv=/path/to/数据架构详细设计v2.3.csv \
        --metrics-csv=/path/to/指标清单v2.4.csv \
        --output=conf/business_terms.yml
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

# Optional: pandas for Excel support
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Install with: pip install pandas openpyxl")

from datus.configuration.agent_config_loader import load_agent_config
from datus.models.base import LLMBaseModel
from datus.storage.cache import get_storage_cache_instance
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage
from datus.storage.schema_metadata.llm_enhanced_extract import (
    extract_business_metadata_with_llm,
    parse_design_requirement_with_llm,
)
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)

# 停用词列表（用于关键词提取）
STOP_WORDS = {
    "的", "和", "或", "与", "及", "是", "在", "用于", "表示", "指",
    "对", "从", "到", "为", "有", "由", "等", "可", "请", "需",
    "以", "根据", "按照", "依据", "包括", "包含", "涉及",
}

# 常见同义词映射（业务术语标准化）
SYNONYM_MAP = {
    "车种": "车型",
    "车系": "车型系列",
    " dealership": "经销店",
    "4s店": "经销店",
    "门店": "经销店",
}


class BusinessTermMapping:
    """
    带置信度的业务术语映射项
    
    支持多源合并和冲突解决
    """
    def __init__(self, term: str, targets: List[str], source: str, confidence: float = 0.5):
        self.term = term
        self.targets = set(targets) if isinstance(targets, list) else {targets}
        self.sources = {source: confidence}
        self.confidence = confidence
        self.metadata = {}
    
    def merge(self, other: 'BusinessTermMapping') -> 'BusinessTermMapping':
        """合并两个映射项，保留高置信度的来源"""
        self.targets.update(other.targets)
        for src, conf in other.sources.items():
            if src not in self.sources or conf > self.sources[src]:
                self.sources[src] = conf
        self.confidence = max(self.confidence, other.confidence)
        return self
    
    def to_dict(self) -> Dict:
        """转换为字典格式（用于YAML输出）"""
        return {
            "targets": sorted(list(self.targets)),
            "confidence": round(self.confidence, 2),
            "sources": self.sources,
        }


class BusinessConfigGenerator:
    """
    业务术语配置生成器（支持LLM增强）
    
    三级处理策略：
    1. Regex Extraction (快速规则提取，置信度0.6-0.9)
    2. Pattern Matching (模式匹配，置信度0.4-0.7)
    3. LLM Enhancement (LLM增强理解，置信度0.5-0.9)
    """

    # 置信度阈值配置
    CONFIDENCE_THRESHOLDS = {
        "high": 0.8,      # 高置信度：直接采用，无需审核
        "medium": 0.6,    # 中置信度：建议采用，可抽样审核
        "low": 0.4,       # 低置信度：仅供参考，需人工审核
    }

    def __init__(self, agent_config, namespace: str, use_llm: bool = False, llm_model=None):
        self.agent_config = agent_config
        self.namespace = namespace
        self.db_path = agent_config.rag_storage_path()
        self.schema_storage = SchemaStorage(
            db_path=self.db_path,
            embedding_model=get_db_embedding_model()
        )
        
        # LLM 支持
        self.use_llm = use_llm
        self.llm_model = llm_model
        if use_llm and not llm_model:
            try:
                self.llm_model = LLMBaseModel.create_model(agent_config=agent_config)
                logger.info("LLM model initialized for enhanced extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM model: {e}, falling back to regex mode")
                self.use_llm = False
        
        # 带置信度的术语映射存储
        self._term_mappings: Dict[str, BusinessTermMapping] = {}
        
        logger.info(f"Initialized generator for namespace: {namespace}, db_path: {self.db_path}, use_llm: {use_llm}")

    def _read_excel_with_header(self, xlsx_path: Path, header_rows: int, sheet_name: int = 0) -> List[Dict]:
        """
        读取Excel文件，智能处理多行表头
        
        针对复杂表头结构（如数据架构设计文档）：
        - 第1行：分组标题（资产目录/业务属性/技术属性/管理属性）
        - 第2行：实际列名（物理表名、字段名等）★ 使用这行
        - 第3行：列说明/注释
        
        Args:
            xlsx_path: Excel文件路径
            header_rows: 表头行数（用于确定实际列名所在行）
            sheet_name: 工作表索引，默认第一个
            
        Returns:
            List[Dict]: 数据行列表
        """
        if not PANDAS_AVAILABLE:
            logger.error("pandas not available. Install with: pip install pandas openpyxl")
            return []
        
        try:
            # 智能表头检测：
            # - 如果 header_rows=3，表示第2行（索引1）是实际列名
            # - 如果 header_rows=2，表示第1行（索引0）是实际列名
            actual_header_row = header_rows - 2 if header_rows >= 2 else 0
            
            logger.info(f"[Excel读取] 读取 {xlsx_path.name}, 使用第 {actual_header_row + 1} 行作为列名")
            
            # 读取Excel，使用实际列名行作为header
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=actual_header_row)
            logger.info(f"[Excel读取] 数据行数: {len(df)}")
            
            # 清理列名（去除空格、换行符）
            def clean_col(col):
                if pd.isna(col):
                    return f"Unnamed_{id(col)}"
                s = str(col).strip()
                # 移除换行符和多余空格
                s = ' '.join(s.split())
                return s
            
            # 清理并处理重复列名
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
            
            # 转换为字典列表，跳过空行
            records = []
            for idx, row in df.iterrows():
                # 检查关键字段是否都为空
                if row.isna().all():
                    continue
                    
                record = {}
                for col in df.columns:
                    val = row[col]
                    # 处理NaN和空值（val可能是单个值或Series）
                    if isinstance(val, pd.Series):
                        # 如果是Series，取第一个非空值
                        val = val.dropna().iloc[0] if not val.dropna().empty else ""
                    if pd.isna(val):
                        record[col] = ""
                    else:
                        record[col] = str(val).strip()
                records.append(record)
            
            logger.info(f"[Excel读取] 有效数据行: {len(records)}")
            return records
            
        except Exception as e:
            logger.error(f"[Excel读取] 失败: {e}")
            return []

    def generate_from_architecture_xlsx(
        self,
        xlsx_path: Path,
        header_rows: int = 3,
        min_term_length: int = 2
    ) -> Dict:
        """
        从数据架构Excel生成业务术语映射（支持多行表头）
        
        Args:
            xlsx_path: Excel文件路径
            header_rows: 表头行数（数据架构设计文档为3行）
            min_term_length: 最小术语长度
        """
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

        records = self._read_excel_with_header(xlsx_path, header_rows)
        
        if not records:
            logger.warning("[Excel读取] 未读取到有效数据")
            return {
                "term_to_table": {},
                "term_to_schema": {},
                "table_keywords": {},
                "_stats": stats,
            }

        # 调试：打印第一条记录的所有字段
        if records:
            logger.info(f"[Excel解析] 第一条记录的列: {list(records[0].keys())[:20]}...")
            logger.info(f"[Excel解析] 第一条记录样本: {{'物理表名': {records[0].get('物理表名', 'N/A')}, '字段名': {records[0].get('字段名', 'N/A')}, '分析对象': {records[0].get('分析对象（中文）', 'N/A')}}})")
        
        for row in records:
            stats["total_rows"] += 1

            # 提取关键字段（启用调试）
            table_name = self._extract_field(row, ["物理表名", "table_name", "表名"], debug=(stats['total_rows']==1))
            column_name = self._extract_field(row, ["字段名", "column_name", "列名", "column"], debug=(stats['total_rows']==1))
            attr_def = self._extract_field(row, ["属性业务定义", "属性定义"], debug=(stats['total_rows']==1))
            attr_cn = self._extract_field(row, ["属性（中文）", "属性中文名", "属性名称"], debug=(stats['total_rows']==1))
            obj_name = self._extract_field(row, ["分析对象（中文）", "分析对象中文名", "分析对象"], debug=(stats['total_rows']==1))
            obj_en = self._extract_field(row, ["分析对象（英文）", "分析对象英文名"], debug=(stats['total_rows']==1))
            logic_entity = self._extract_field(row, ["逻辑实体（中文）", "逻辑实体中文名", "逻辑实体"], debug=(stats['total_rows']==1))
            logic_entity_def = self._extract_field(row, ["逻辑实体业务含义", "逻辑实体定义", "逻辑实体说明"], debug=(stats['total_rows']==1))

            # 验证表名和字段名
            table_valid = self._is_valid_table_name(table_name, verbose=(stats['total_rows'] <= 10))
            column_valid = self._is_valid_column_name(column_name)
            if not table_valid or not column_valid:
                if stats['total_rows'] <= 20:
                    logger.debug(f"[过滤] 表 '{table_name}' 或字段 '{column_name}' 验证失败 (表:{table_valid}, 字段:{column_valid})")
                continue

            stats["valid_rows"] += 1
            stats["tables_found"].add(table_name)

            # 1. 分析对象 -> 表映射（业务对象粒度）
            if obj_name and len(obj_name) >= min_term_length and self._is_meaningful_term(obj_name):
                term_to_table[obj_name].add(table_name)
                if obj_en and self._is_meaningful_term(obj_en):
                    term_to_table[obj_en.lower()].add(table_name)

            # 2. 逻辑实体 -> 表映射（作为表关键词）
            if logic_entity and len(logic_entity) >= min_term_length and self._is_meaningful_term(logic_entity):
                term_to_table[logic_entity].add(table_name)
                table_keywords[logic_entity] = table_name

                # 从逻辑实体业务含义提取关键词
                if logic_entity_def:
                    keywords = self._extract_meaningful_keywords(logic_entity_def, min_term_length)
                    for kw in keywords:
                        table_keywords[kw] = table_name
                        term_to_table[kw].add(table_name)

            # 3. 属性（中文）-> 字段映射
            if attr_cn and len(attr_cn) >= min_term_length and self._is_meaningful_term(attr_cn):
                term_to_schema[attr_cn].add(f"{table_name}.{column_name}")
                term_to_schema[attr_cn].add(column_name)

            # 4. 属性业务定义 -> 提取有意义的关键词映射到字段
            if attr_def and len(attr_def) >= min_term_length:
                keywords = self._extract_meaningful_keywords(attr_def, min_term_length)
                for kw in keywords:
                    if len(kw) <= 50:
                        term_to_schema[kw].add(column_name)
                        term_to_schema[kw].add(f"{table_name}.{column_name}")
                        stats["terms_extracted"] += 1

        logger.info(
            f"[Excel解析] 数据架构处理完成: {stats['valid_rows']}/{stats['total_rows']} 有效行, "
            f"{len(stats['tables_found'])} 个表, "
            f"{stats['terms_extracted']} 个术语提取"
        )
        
        # 详细调试信息
        if stats['tables_found']:
            sample_tables = list(stats['tables_found'])[:5]
            logger.info(f"[Excel解析] 样本表名: {sample_tables}")
        
        if term_to_table:
            sample_terms = list(term_to_table.keys())[:10]
            logger.info(f"[术语提取] 样本术语(前10): {sample_terms}")
            logger.debug(f"[术语提取] 全部术语({len(term_to_table)}): {list(term_to_table.keys())}")
        
        if table_keywords:
            sample_keywords = list(table_keywords.keys())[:5]
            logger.info(f"[关键词] 样本表关键词(前5): {sample_keywords}")

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

    def generate_from_metrics_xlsx(
        self,
        xlsx_path: Path,
        existing_terms: Dict,
        header_rows: int = 2,
        min_term_length: int = 2
    ) -> Dict:
        """
        从指标清单Excel补充业务术语映射（支持多行表头）
        
        Args:
            xlsx_path: Excel文件路径
            existing_terms: 已有的术语映射
            header_rows: 表头行数（指标清单为2行）
            min_term_length: 最小术语长度
        """
        logger.info(f"Processing metrics Excel: {xlsx_path}")

        term_to_table = defaultdict(set, existing_terms.get("term_to_table", {}))
        term_to_schema = defaultdict(set, existing_terms.get("term_to_schema", {}))
        table_keywords = existing_terms.get("table_keywords", {})

        stats = {"total_metrics": 0, "valid_metrics": 0, "tables_added": set()}

        records = self._read_excel_with_header(xlsx_path, header_rows)
        
        if not records:
            logger.warning("[Excel读取] 指标清单未读取到有效数据")
            return {
                "term_to_table": dict(term_to_table),
                "term_to_schema": dict(term_to_schema),
                "table_keywords": table_keywords,
                "_stats": {**existing_terms.get("_stats", {}), **{
                    "metrics_count": 0,
                    "metrics_tables": 0,
                }},
            }

        for row in records:
            stats["total_metrics"] += 1

            # 使用 _extract_field 支持多种列名变体
            metric_code = self._extract_field(row, ["指标编码", "指标编码 -固定值（勿改）"], debug=(stats['total_metrics']==1))
            metric_name = self._extract_field(row, ["指标名称"], debug=(stats['total_metrics']==1))
            biz_def = self._extract_field(row, ["业务定义及说明", "业务定义"], debug=(stats['total_metrics']==1))
            calc_logic = self._extract_field(row, ["计算公式/业务逻辑", "计算公式", "业务逻辑"], debug=(stats['total_metrics']==1))
            source_model = self._extract_field(row, ["来源dws模型", "dws模型", "来源模型"], debug=(stats['total_metrics']==1))
            biz_activity = self._extract_field(row, ["业务活动"], debug=(stats['total_metrics']==1))

            if not metric_name:
                continue

            stats["valid_metrics"] += 1

            # 1. 指标名称 -> 来源表映射（验证表名有效性）
            if source_model and self._is_valid_table_name(source_model):
                if len(metric_name) >= min_term_length and self._is_meaningful_term(metric_name):
                    term_to_table[metric_name].add(source_model)
                    stats["tables_added"].add(source_model)

                    # 添加业务活动分类映射
                    if biz_activity and self._is_meaningful_term(biz_activity):
                        term_to_table[f"{biz_activity}_{metric_name}"].add(source_model)

            # 2. 从业务定义提取有意义的关键词
            if biz_def:
                keywords = self._extract_meaningful_keywords(biz_def, min_term_length)
                for kw in keywords:
                    if source_model and self._is_valid_table_name(source_model):
                        term_to_table[kw].add(source_model)

            # 3. 从计算公式提取表名引用
            if calc_logic:
                table_refs = self._extract_table_refs(calc_logic)
                for ref in table_refs:
                    if self._is_valid_table_name(ref):
                        term_to_schema[metric_name].add(ref)

        logger.info(
            f"[Excel解析] 指标清单处理完成: {stats['valid_metrics']}/{stats['total_metrics']} 有效指标, "
            f"{len(stats['tables_added'])} 个来源表"
        )
        
        # 详细调试信息
        if stats['tables_added']:
            logger.info(f"[指标提取] 来源表示例: {list(stats['tables_added'])[:5]}")
        
        # 显示新增的关键术语映射
        new_terms = set(term_to_table.keys()) - set(existing_terms.get("term_to_table", {}).keys())
        if new_terms:
            logger.info(f"[术语合并] 指标清单新增术语({len(new_terms)}个): {list(new_terms)[:10]}")

        return {
            "term_to_table": dict(term_to_table),
            "term_to_schema": dict(term_to_schema),
            "table_keywords": table_keywords,
            "_stats": {**existing_terms.get("_stats", {}), **{
                "metrics_count": stats["valid_metrics"],
                "metrics_tables": len(stats["tables_added"]),
            }},
        }

    def generate_from_architecture_csv(
        self,
        csv_path: Path,
        min_term_length: int = 2
    ) -> Dict:
        """
        从数据架构CSV生成业务术语映射

        处理多行表头CSV（前5行为说明/标题行）

        Args:
            csv_path: 数据架构详细设计CSV文件路径
            min_term_length: 最小术语长度，过滤单字术语

        Returns:
            Dict with term_to_table, term_to_schema, table_keywords
        """
        logger.info(f"Processing architecture CSV: {csv_path}")

        term_to_table: Dict[str, Set[str]] = defaultdict(set)
        term_to_schema: Dict[str, Set[str]] = defaultdict(set)
        table_keywords: Dict[str, str] = {}

        # 统计信息
        stats = {
            "total_rows": 0,
            "valid_rows": 0,
            "tables_found": set(),
            "terms_extracted": 0,
        }

        # 智能检测表头：读取前10行找到真正的列名行
        # 真正的列名行包含"物理表名"、"字段名"等关键词
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        
        # 找到包含关键列名的行作为表头
        header_row_idx = 0
        expected_columns = ["物理表名", "字段名", "属性（中文）", "分析对象（中文）"]
        for i, line in enumerate(lines[:10]):  # 只检查前10行
            if any(col in line for col in expected_columns):
                header_row_idx = i
                logger.info(f"[CSV解析] 找到表头在第 {i+1} 行: {line[:100]}...")
                break
        
        # 从表头行开始读取
        import io
        csv_content = ''.join(lines[header_row_idx:])
        reader = csv.DictReader(io.StringIO(csv_content))
        
        # 打印实际的列名用于调试
        if reader.fieldnames:
            logger.info(f"[CSV解析] 数据架构文件列名: {list(reader.fieldnames)}")
            logger.debug(f"[CSV解析] 完整列名详情: {reader.fieldnames}")
        
        for row in reader:
                stats["total_rows"] += 1

                # 提取关键字段（使用实际的列名，兼容不同版本）
                # 尝试多种可能的列名
                table_name = self._extract_field(row, ["物理表名", "table_name", "表名"])
                column_name = self._extract_field(row, ["字段名", "column_name", "列名"])
                attr_def = self._extract_field(row, ["属性业务定义", "业务定义", "注释", "comment"])
                attr_cn = self._extract_field(row, ["属性（中文）", "属性", "column_cn", "字段中文名"])
                obj_name = self._extract_field(row, ["分析对象（中文）", "分析对象", "object_cn"])
                obj_en = self._extract_field(row, ["分析对象（英文）", "object_en"])
                logic_entity = self._extract_field(row, ["逻辑实体（中文）", "逻辑实体", "entity_cn", "逻辑表名"])
                logic_entity_def = self._extract_field(row, ["逻辑实体业务含义", "实体业务含义", "entity_def"])

                # 验证表名和字段名 - 必须是以字母开头的有效标识符（启用详细调试）
                table_valid = self._is_valid_table_name(table_name, verbose=(stats['total_rows'] <= 10))
                column_valid = self._is_valid_column_name(column_name)
                if not table_valid or not column_valid:
                    if stats['total_rows'] <= 20:  # 只在前20行显示调试信息
                        logger.debug(f"[过滤] 表 '{table_name}' 或字段 '{column_name}' 验证失败 (表:{table_valid}, 字段:{column_valid})")
                    continue

                stats["valid_rows"] += 1
                stats["tables_found"].add(table_name)

                # 1. 分析对象 -> 表映射（业务对象粒度）
                if obj_name and len(obj_name) >= min_term_length and self._is_meaningful_term(obj_name):
                    term_to_table[obj_name].add(table_name)
                    # 同时添加英文映射（如果存在且有效）
                    if obj_en and self._is_meaningful_term(obj_en):
                        term_to_table[obj_en.lower()].add(table_name)

                # 2. 逻辑实体 -> 表映射（作为表关键词）
                if logic_entity and len(logic_entity) >= min_term_length and self._is_meaningful_term(logic_entity):
                    term_to_table[logic_entity].add(table_name)
                    table_keywords[logic_entity] = table_name

                    # 从逻辑实体业务含义提取关键词（仅提取有意义的关键词）
                    if logic_entity_def:
                        keywords = self._extract_meaningful_keywords(logic_entity_def, min_term_length)
                        for kw in keywords:
                            table_keywords[kw] = table_name
                            term_to_table[kw].add(table_name)

                # 3. 属性（中文）-> 字段映射
                if attr_cn and len(attr_cn) >= min_term_length and self._is_meaningful_term(attr_cn):
                    term_to_schema[attr_cn].add(f"{table_name}.{column_name}")
                    # 也添加纯字段名映射
                    term_to_schema[attr_cn].add(column_name)

                # 4. 属性业务定义 -> 提取有意义的关键词映射到字段
                if attr_def and len(attr_def) >= min_term_length:
                    # 提取有意义的关键词（避免提取技术词汇）
                    keywords = self._extract_meaningful_keywords(attr_def, min_term_length)
                    for kw in keywords:
                        # 限制关键词长度，避免过长的短语
                        if len(kw) <= 50:
                            term_to_schema[kw].add(column_name)
                            term_to_schema[kw].add(f"{table_name}.{column_name}")
                            stats["terms_extracted"] += 1

        logger.info(
            f"[CSV解析] 数据架构处理完成: {stats['valid_rows']}/{stats['total_rows']} 有效行, "
            f"{len(stats['tables_found'])} 个表, "
            f"{stats['terms_extracted']} 个术语提取"
        )
        
        # 详细调试信息
        if stats['tables_found']:
            sample_tables = list(stats['tables_found'])[:5]
            logger.info(f"[CSV解析] 样本表名: {sample_tables}")
        
        if term_to_table:
            sample_terms = list(term_to_table.keys())[:10]
            logger.info(f"[术语提取] 样本术语(前10): {sample_terms}")
            logger.debug(f"[术语提取] 全部术语({len(term_to_table)}): {list(term_to_table.keys())}")
        
        if table_keywords:
            sample_keywords = list(table_keywords.keys())[:5]
            logger.info(f"[关键词] 样本表关键词(前5): {sample_keywords}")

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

    def generate_from_metrics_csv(
        self,
        csv_path: Path,
        existing_terms: Dict,
        min_term_length: int = 2
    ) -> Dict:
        """
        从指标清单CSV补充业务术语映射

        处理多行表头CSV（前7行为说明/标题行）

        Args:
            csv_path: 指标清单CSV文件路径
            existing_terms: 已有的术语映射（从数据架构生成）
            min_term_length: 最小术语长度

        Returns:
            合并后的术语映射
        """
        logger.info(f"Processing metrics CSV: {csv_path}")

        term_to_table = defaultdict(set, existing_terms.get("term_to_table", {}))
        term_to_schema = defaultdict(set, existing_terms.get("term_to_schema", {}))
        table_keywords = existing_terms.get("table_keywords", {})

        stats = {"total_metrics": 0, "valid_metrics": 0, "tables_added": set()}

        # 智能检测表头：指标清单CSV结构复杂，需要检测真正的列名行
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        
        # 找到包含关键列名（业务活动、指标编码、指标名称）的行作为表头
        header_row_idx = 7  # 默认第8行
        expected_cols = ["业务活动", "指标编码", "指标名称"]
        for i, line in enumerate(lines[:20]):  # 检查前20行
            # 检查是否包含至少2个关键列名
            match_count = sum(1 for col in expected_cols if col in line)
            if match_count >= 2:
                header_row_idx = i
                logger.info(f"[CSV解析] 指标清单表头在第 {i+1} 行: {line[:80]}...")
                break
        
        # 从表头行开始读取
        import io
        csv_content = ''.join(lines[header_row_idx:])
        reader = csv.DictReader(io.StringIO(csv_content))
        
        # 打印列名用于调试
        if reader.fieldnames:
            logger.info(f"[CSV解析] 指标清单文件列名: {list(reader.fieldnames)}")
            logger.debug(f"[CSV解析] 完整列名详情: {reader.fieldnames}")
        
        for row in reader:
            stats["total_metrics"] += 1

            # 尝试多种可能的列名
            metric_code = self._extract_field(row, ["指标编码", "metric_code", "编码"])
            metric_name = self._extract_field(row, ["指标名称", "metric_name", "指标"])
            biz_def = self._extract_field(row, ["业务定义及说明", "业务定义", "定义"])
            calc_logic = self._extract_field(row, ["计算公式/业务逻辑", "计算公式", "业务逻辑"])
            source_model = self._extract_field(row, ["来源dws模型", "来源表", "source_table"])
            biz_activity = self._extract_field(row, ["业务活动", "activity"])

            if not metric_name:
                continue

            stats["valid_metrics"] += 1

            # 1. 指标名称 -> 来源表映射（验证表名有效性）
            if source_model and self._is_valid_table_name(source_model):
                if len(metric_name) >= min_term_length and self._is_meaningful_term(metric_name):
                    term_to_table[metric_name].add(source_model)
                    stats["tables_added"].add(source_model)

                    # 添加业务活动分类映射
                    if biz_activity and self._is_meaningful_term(biz_activity):
                        term_to_table[f"{biz_activity}_{metric_name}"].add(source_model)

            # 2. 从业务定义提取有意义的关键词
            if biz_def:
                keywords = self._extract_meaningful_keywords(biz_def, min_term_length)
                for kw in keywords:
                    if source_model and self._is_valid_table_name(source_model):
                        term_to_table[kw].add(source_model)

            # 3. 从计算公式提取表名引用
            if calc_logic:
                table_refs = self._extract_table_refs(calc_logic)
                for ref in table_refs:
                    if self._is_valid_table_name(ref):
                        term_to_schema[metric_name].add(ref)

        logger.info(
            f"[CSV解析] 指标清单处理完成: {stats['valid_metrics']}/{stats['total_metrics']} 有效指标, "
            f"{len(stats['tables_added'])} 个来源表"
        )
        
        # 详细调试信息
        if stats['tables_added']:
            logger.info(f"[指标提取] 来源表示例: {list(stats['tables_added'])[:5]}")
        
        # 显示新增的关键术语映射
        new_terms = set(term_to_table.keys()) - set(existing_terms.get("term_to_table", {}).keys())
        if new_terms:
            logger.info(f"[术语合并] 指标清单新增术语({len(new_terms)}个): {list(new_terms)[:10]}")

        return {
            "term_to_table": dict(term_to_table),
            "term_to_schema": dict(term_to_schema),
            "table_keywords": table_keywords,
            "_stats": {**existing_terms.get("_stats", {}), **{
                "metrics_count": stats["valid_metrics"],
                "metrics_tables": len(stats["tables_added"]),
            }},
        }

    def merge_with_ddl_comments(self, business_terms: Dict) -> Dict:
        """
        将CSV提取的术语与LanceDB中的DDL comments进行合并

        合并策略：
        1. CSV中有但DDL没有的 -> 保留（设计文档优先）
        2. DDL中有但CSV没有的 -> 补充（实际落地的字段）
        3. 两者都有 -> 以CSV为准，但添加DDL作为别名
        """
        logger.info("Merging with DDL comments from LanceDB...")

        term_to_table = defaultdict(set, business_terms.get("term_to_table", {}))
        term_to_schema = defaultdict(set, business_terms.get("term_to_schema", {}))

        ddl_stats = {"tables_checked": 0, "comments_found": 0, "terms_added": 0}

        try:
            # 从LanceDB读取所有schema
            self.schema_storage._ensure_table_ready()
            all_records = self.schema_storage.table.to_pandas()

            for _, row in all_records.iterrows():
                table_name = row.get("table_name", "")
                col_comments_json = row.get("column_comments", "{}")
                table_comment = row.get("table_comment", "")

                if not table_name:
                    continue

                ddl_stats["tables_checked"] += 1

                # 1. 表注释作为关键词（使用有意义的关键词提取）
                if table_comment and len(table_comment) > 2:
                    keywords = self._extract_meaningful_keywords(table_comment, min_length=3)
                    for kw in keywords:
                        if self._is_meaningful_term(kw):
                            term_to_table[kw].add(table_name)

                # 2. 字段注释映射
                try:
                    col_comments = json.loads(col_comments_json) if col_comments_json else {}
                except json.JSONDecodeError:
                    continue

                for col_name, comment in col_comments.items():
                    if not comment or len(comment) < 2:
                        continue

                    ddl_stats["comments_found"] += 1

                    # 将注释作为术语映射到字段
                    # 只添加有意义的注释（非技术注释）
                    if self._is_meaningful_comment(comment):
                        term_to_schema[comment].add(f"{table_name}.{col_name}")
                        term_to_schema[comment].add(col_name)
                        ddl_stats["terms_added"] += 1

                        # 提取有意义的关键词
                        keywords = self._extract_meaningful_keywords(comment, min_length=2)
                        for kw in keywords:
                            if self._is_meaningful_term(kw):
                                term_to_schema[kw].add(col_name)

        except Exception as e:
            logger.warning(f"Failed to merge DDL comments: {e}")

        logger.info(
            f"[DDL合并] 完成: {ddl_stats['tables_checked']} 表检查, "
            f"{ddl_stats['comments_found']} 注释发现, "
            f"{ddl_stats['terms_added']} 术语添加"
        )
        
        # 详细调试：显示新增的术语样本
        if ddl_stats['terms_added'] > 0:
            new_ddl_terms = list(term_to_schema.keys())[-20:]  # 最后添加的20个
            logger.info(f"[DDL合并] DDL新增术语样本(后20): {new_ddl_terms}")
            
            # 统计按表分部的术语
            tables_with_terms = defaultdict(int)
            for term, fields in term_to_schema.items():
                for f in fields:
                    if '.' in f:
                        table = f.split('.')[0]
                        tables_with_terms[table] += 1
            
            if tables_with_terms:
                top_tables = sorted(tables_with_terms.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"[DDL合并] 术语最多的表(Top5): {top_tables}")

        return {
            "term_to_table": dict(term_to_table),
            "term_to_schema": dict(term_to_schema),
            "table_keywords": business_terms.get("table_keywords", {}),
            "_stats": {**business_terms.get("_stats", {}), **{
                "ddl_tables": ddl_stats["tables_checked"],
                "ddl_terms": ddl_stats["terms_added"],
            }},
        }

    def _extract_field(self, row: Dict, possible_names: List[str], debug: bool = False) -> str:
        """
        从行中提取字段值，尝试多个可能的列名
        
        支持多种匹配策略：
        1. 精确匹配
        2. 大小写不敏感匹配
        3. 前缀匹配（用于处理带注释的列名，如 "指标编码\n-固定值（勿改）"）
        4. 子串匹配（用于处理复杂列名）
        """
        row_keys = list(row.keys())
        
        for name in possible_names:
            # 1. 尝试精确匹配
            if name in row and row[name]:
                return self._clean_string(row[name])
            
            # 2. 尝试大小写不敏感匹配
            for key in row_keys:
                if key and key.lower() == name.lower():
                    return self._clean_string(row[key])
            
            # 3. 尝试前缀匹配（处理带换行符/注释的列名）
            for key in row_keys:
                if key and (key.startswith(name) or name in key):
                    if row[key]:
                        return self._clean_string(row[key])
        
        # 调试：打印失败的匹配
        if debug:
            logger.debug(f"[字段提取] 未找到匹配: {possible_names}, 可用列: {row_keys[:15]}...")
        return ""

    def _is_valid_table_name(self, name: str, verbose: bool = False) -> bool:
        """验证是否为有效的表名（以字母开头，只包含字母数字下划线）"""
        if not name or len(name) < 2:
            if verbose and name:
                logger.debug(f"[过滤] 表名太短或为空: '{name}'")
            return False
        # 必须以字母开头
        if not re.match(r'^[a-zA-Z]', name):
            if verbose:
                logger.debug(f"[过滤] 表名不以字母开头: '{name}'")
            return False
        # 只能包含字母、数字、下划线
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            if verbose:
                logger.debug(f"[过滤] 表名含非法字符: '{name}'")
            return False
        # 排除常见技术关键词
        technical_keywords = {'engine', 'key', 'duplicate', 'distributed', 'random', 'min', 'max', 'properties'}
        if name.lower() in technical_keywords:
            if verbose:
                logger.debug(f"[过滤] 技术关键词表名: '{name}'")
            return False
        return True

    def _is_valid_column_name(self, name: str) -> bool:
        """验证是否为有效的字段名"""
        if not name or len(name) < 1:
            return False
        # 必须以字母开头
        if not re.match(r'^[a-zA-Z_]', name):
            return False
        # 只能包含字母、数字、下划线
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return False
        return True

    def _is_meaningful_term(self, term: str) -> bool:
        """判断术语是否有业务意义（过滤技术词汇）"""
        if not term or len(term) < 2:
            return False
        
        # 排除纯技术词汇
        technical_terms = {
            'id', 'code', 'name', 'status', 'type', 'flag', 'time', 'date',
            'create', 'update', 'delete', 'insert', 'select', 'from', 'where',
            'table', 'column', 'field', 'index', 'key', 'value',
            'dealer_clue_code', 'original_clue_code', 'customer_id',  # 字段名本身
        }
        
        if term.lower() in technical_terms:
            return False
        
        # 排除纯数字
        if re.match(r'^\d+$', term):
            return False
        
        # 排除下划线开头的（通常是临时/内部字段）
        if term.startswith('_'):
            return False
        
        return True

    def _extract_meaningful_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """提取有意义的关键词（过滤技术词汇和停用词）"""
        if not text:
            return []

        keywords = []
        
        # 1. 提取中文短语（2-10字）
        for match in re.finditer(r"[\u4e00-\u9fa5]{%d,10}" % min_length, text):
            kw = match.group()
            # 过滤停用词和纯技术词汇
            if kw not in STOP_WORDS and self._is_meaningful_term(kw):
                keywords.append(kw)
        
        # 2. 提取英文/数字混合的业务词汇（如 is_valid_clue, test_drive）
        # 排除纯技术字段名（如 id, code, flag）
        for match in re.finditer(r"[a-z_][a-z0-9_]{%d,}" % (min_length - 1), text.lower()):
            kw = match.group()
            # 过滤太短或太长的
            if len(kw) < min_length or len(kw) > 40:
                continue
            # 过滤纯技术词汇
            if self._is_meaningful_term(kw):
                keywords.append(kw)
        
        # 3. 同义词替换
        normalized = []
        for kw in keywords:
            normalized.append(SYNONYM_MAP.get(kw, kw))
        
        return list(set(normalized))  # 去重

    def _clean_string(self, s: str) -> str:
        """清理字符串，去除空白和特殊字符"""
        if not s:
            return ""
        return s.strip().replace("\n", " ").replace("\r", "").replace("\t", " ")

    def _extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """从文本提取关键词（简单实现）"""
        if not text:
            return []

        keywords = []

        # 1. 提取中文词汇（2-10字）
        # 匹配连续中文字符
        for match in re.finditer(r"[\u4e00-\u9fa5]{%d,10}" % min_length, text):
            kw = match.group()
            if kw not in STOP_WORDS and len(kw) >= min_length:
                keywords.append(kw)

        # 2. 提取英文/数字混合词（如 is_valid_clue）
        for match in re.finditer(r"[a-z_][a-z0-9_]{%d,}" % (min_length - 1), text.lower()):
            kw = match.group()
            if kw not in STOP_WORDS:
                keywords.append(kw)

        # 3. 同义词替换
        normalized = []
        for kw in keywords:
            normalized.append(SYNONYM_MAP.get(kw, kw))

        return list(set(normalized))  # 去重

    def _extract_table_refs(self, calc_logic: str) -> List[str]:
        """从计算公式中提取表引用"""
        refs = []
        # 匹配 t_xxx 或 dws_xxx 等表名模式
        for match in re.finditer(r"\b(t_[a-z_]+|dws_[a-z_]+|dwd_[a-z_]+|dim_[a-z_]+)\b", calc_logic.lower()):
            refs.append(match.group())
        return refs

    def _is_meaningful_comment(self, comment: str) -> bool:
        """判断注释是否有业务意义（过滤技术注释）"""
        if not comment:
            return False

        # 过滤纯技术注释
        technical_patterns = [
            r"^\d+$",  # 纯数字
            r"^[a-z_]+$",  # 纯小写英文（可能是字段名本身）
            r"^\d{4}-\d{2}-\d{2}",  # 日期格式
            r"^(主键|外键|索引|unique|primary|key|idx|fk|pk)$",  # 纯技术词汇
        ]

        for pattern in technical_patterns:
            if re.match(pattern, comment.strip(), re.IGNORECASE):
                return False

        return len(comment) >= 4  # 有意义的注释通常至少有4个字符

    def add_mapping_with_confidence(
        self,
        term: str,
        targets: List[str],
        source: str,
        confidence: float,
        metadata: Dict = None
    ):
        """
        添加带置信度的术语映射
        
        合并策略：
        - 同术语同来源：更新置信度（取最高）
        - 同术语不同来源：合并targets，记录多来源
        """
        if term not in self._term_mappings:
            self._term_mappings[term] = BusinessTermMapping(term, targets, source, confidence)
        else:
            existing = self._term_mappings[term]
            existing.targets.update(targets)
            if source in existing.sources:
                # 同来源取最高置信度
                existing.sources[source] = max(existing.sources[source], confidence)
            else:
                existing.sources[source] = confidence
            existing.confidence = max(existing.confidence, confidence)
        
        if metadata:
            self._term_mappings[term].metadata.update(metadata)

    def llm_enhance_term_extraction(
        self,
        term: str,
        context: str,
        table_candidates: List[str]
    ) -> Dict:
        """
        使用LLM增强术语理解，解决歧义和冲突
        
        适用于：
        - 术语有歧义（如"客户"可能指厂端/店端/线索客户）
        - 多表冲突（多个表都匹配同一术语）
        - 需要语义理解（如"首触"→"首次接触的原始线索"）
        """
        if not self.use_llm or not self.llm_model:
            return {"enhanced": False, "confidence": 0.0}
        
        prompt = f"""You are a data warehouse business analyst. Analyze the business term and determine the most relevant tables.

## Input
Business Term: {term}
Context: {context}
Candidate Tables: {table_candidates}

## Task
1. Analyze the semantic meaning of the business term
2. Evaluate relevance of each candidate table (score 0-1)
3. Identify if this term has ambiguity (e.g., "客户" could be factory customer, dealer customer, or lead customer)

## Output Format
Return ONLY a JSON object:
{{
  "term_analysis": "brief semantic analysis of the term",
  "primary_table": "most relevant table name",
  "primary_confidence": 0.0-1.0,
  "secondary_tables": ["other relevant tables"],
  "ambiguity": true/false,
  "disambiguation": "if ambiguous, explain different contexts",
  "related_terms": ["synonyms or related business concepts"],
  "confidence": 0.0-1.0
}}
"""
        try:
            response = self.llm_model.generate_with_json_output(prompt)
            if isinstance(response, dict):
                return {
                    "enhanced": True,
                    "analysis": response.get("term_analysis", ""),
                    "primary_table": response.get("primary_table", ""),
                    "primary_confidence": response.get("primary_confidence", 0.0),
                    "secondary_tables": response.get("secondary_tables", []),
                    "ambiguity": response.get("ambiguity", False),
                    "related_terms": response.get("related_terms", []),
                    "confidence": response.get("confidence", 0.0),
                }
        except Exception as e:
            logger.debug(f"LLM enhancement failed for term '{term}': {e}")
        
        return {"enhanced": False, "confidence": 0.0}

    def resolve_conflicts_with_llm(self) -> Dict[str, BusinessTermMapping]:
        """
        使用LLM解决术语映射冲突
        
        冲突场景：
        1. 同一术语映射到多个表（歧义）
        2. 多个术语指向同一表（冗余）
        3. 设计文档与实际DDL不一致
        """
        if not self.use_llm:
            return self._term_mappings
        
        resolved = {}
        
        for term, mapping in self._term_mappings.items():
            # 只处理有冲突的映射（多targets或低置信度）
            if len(mapping.targets) > 1 or mapping.confidence < self.CONFIDENCE_THRESHOLDS["medium"]:
                context = f"Sources: {mapping.sources}, Metadata: {mapping.metadata}"
                table_candidates = list(mapping.targets)
                
                enhancement = self.llm_enhance_term_extraction(term, context, table_candidates)
                
                if enhancement.get("enhanced") and enhancement.get("confidence", 0) > 0.6:
                    # 使用LLM建议作为主映射
                    primary = enhancement.get("primary_table", "")
                    if primary in mapping.targets:
                        # 重新排序targets，将primary放在首位
                        ordered_targets = [primary] + [t for t in mapping.targets if t != primary]
                        mapping.targets = set(ordered_targets)
                        mapping.confidence = enhancement.get("confidence", mapping.confidence)
                        mapping.metadata["llm_analysis"] = enhancement.get("analysis", "")
                        mapping.metadata["related_terms"] = enhancement.get("related_terms", [])
            
            resolved[term] = mapping
        
        return resolved

    def save_to_yaml(self, business_terms: Dict, output_path: Path, include_confidence: bool = False):
        """
        保存业务术语配置到YAML文件
        
        Args:
            business_terms: 术语映射字典
            output_path: 输出路径
            include_confidence: 是否包含置信度信息（调试用）
        """
        # 如果启用了置信度模式，先解决冲突
        if self._term_mappings and include_confidence:
            resolved = self.resolve_conflicts_with_llm()
            
            # 按置信度分组输出
            output = {
                "_metadata": {
                    "high_confidence_threshold": self.CONFIDENCE_THRESHOLDS["high"],
                    "medium_confidence_threshold": self.CONFIDENCE_THRESHOLDS["medium"],
                },
                "high_confidence": {},      # >= 0.8
                "medium_confidence": {},    # 0.6 - 0.8
                "low_confidence": {},       # < 0.6
                "term_to_table": {},
                "term_to_schema": {},
                "table_keywords": {},
            }
            
            for term, mapping in resolved.items():
                entry = mapping.to_dict() if include_confidence else sorted(list(mapping.targets))
                
                if mapping.confidence >= self.CONFIDENCE_THRESHOLDS["high"]:
                    output["high_confidence"][term] = entry
                elif mapping.confidence >= self.CONFIDENCE_THRESHOLDS["medium"]:
                    output["medium_confidence"][term] = entry
                else:
                    output["low_confidence"][term] = entry
                
                # 同时输出扁平格式供生产使用
                output["term_to_table"][term] = sorted(list(mapping.targets))
        else:
            # 传统模式：转换set为sorted list
            output = {
                "term_to_table": {
                    k: sorted(list(v)) if isinstance(v, (set, list)) else v
                    for k, v in business_terms.get("term_to_table", {}).items()
                },
                "term_to_schema": {
                    k: sorted(list(v)) if isinstance(v, (set, list)) else v
                    for k, v in business_terms.get("term_to_schema", {}).items()
                },
                "table_keywords": business_terms.get("table_keywords", {}),
            }

        # 添加生成信息注释
        header = f"""# Business Terms Configuration
# Generated by generate_business_config.py
# Namespace: {self.namespace}
# Total mappings: {len(output['term_to_table'])} table terms, {len(output['term_to_schema'])} schema terms
#
# Load order:
# 1) Environment variable BUSINESS_TERM_CONFIG (file path)
# 2) conf/business_terms.yml (repo/local override)
# 3) ~/.datus/conf/business_terms.yml (per-user override)

"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.dump(
                output,
                f,
                allow_unicode=True,
                sort_keys=True,
                default_flow_style=False,
                indent=2,
            )

        logger.info(f"Business terms saved to: {output_path}")

        # 打印统计报告
        stats = business_terms.get("_stats", {})
        print("\n" + "=" * 70)
        print("BUSINESS TERMS GENERATION REPORT")
        print("=" * 70)
        print(f"Output file: {output_path}")
        print(f"Table terms: {len(output['term_to_table'])}")
        print(f"Schema terms: {len(output['term_to_schema'])}")
        print(f"Table keywords: {len(output.get('table_keywords', {}))}")
        
        # 显示样本术语
        if output['term_to_table']:
            sample = list(output['term_to_table'].keys())[:5]
            print(f"\nSample table terms: {sample}")
        if output['term_to_schema']:
            sample = list(output['term_to_schema'].keys())[:5]
            print(f"Sample schema terms: {sample}")
        
        if stats:
            print("\nGeneration stats:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate business terms configuration from design documents (with optional LLM enhancement)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # [推荐] 从Excel生成（支持多行表头）
  python scripts/generate_business_config.py \\
      --config=conf/agent.yml \\
      --namespace=test \\
      --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \\
      --metrics-xlsx=/path/to/指标清单v2.4.xlsx \\
      --merge-ddl \\
      --verbose

  # [兼容] 从CSV生成（需确保表头正确）
  python scripts/generate_business_config.py \\
      --config=conf/agent.yml \\
      --namespace=test \\
      --arch-csv=/path/to/数据架构详细设计v2.3.csv \\
      --metrics-csv=/path/to/指标清单v2.4.csv

  # Debug mode with confidence scores
  python scripts/generate_business_config.py \\
      --config=conf/agent.yml \\
      --namespace=test \\
      --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \\
      --use-llm \\
      --include-confidence \\
      --output=conf/business_terms_debug.yml
        """,
    )

    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", required=True, help="Target namespace")
    parser.add_argument("--arch-csv", help="Path to 数据架构详细设计 CSV file (legacy)")
    parser.add_argument("--metrics-csv", help="Path to 指标清单 CSV file (legacy, optional)")
    parser.add_argument("--arch-xlsx", help="Path to 数据架构详细设计 Excel file (推荐, 支持多行表头)")
    parser.add_argument("--metrics-xlsx", help="Path to 指标清单 Excel file (推荐, 支持多行表头)")
    parser.add_argument(
        "--output",
        default="conf/business_terms.yml",
        help="Output YAML file path (default: conf/business_terms.yml)",
    )
    parser.add_argument(
        "--merge-ddl",
        action="store_true",
        help="Merge with existing DDL comments from LanceDB",
    )
    parser.add_argument(
        "--min-term-length",
        type=int,
        default=2,
        help="Minimum term length (default: 2)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM enhancement for term disambiguation and conflict resolution",
    )
    parser.add_argument(
        "--include-confidence",
        action="store_true",
        help="Include confidence scores in output (for debugging)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Verbose output (-v for INFO, -vv for DEBUG)",
    )

    args = parser.parse_args()

    # 初始化日志（根据verbose级别设置）
    # configure_logging 使用 debug 参数: False=INFO级别, True=DEBUG级别
    if args.verbose >= 2:
        configure_logging(debug=True)  # DEBUG级别
    else:
        configure_logging(debug=False)  # INFO级别（默认）

    # 加载配置
    logger.info(f"Loading agent config from: {args.config}")
    agent_config = load_agent_config(config=args.config)
    agent_config.current_namespace = args.namespace

    # 初始化生成器（支持LLM增强）
    generator = BusinessConfigGenerator(
        agent_config, 
        args.namespace,
        use_llm=args.use_llm
    )
    
    if args.use_llm:
        logger.info("LLM enhancement enabled for term extraction and conflict resolution")

    # 生成业务术语
    business_terms = None

    # 优先处理Excel文件（推荐方式）
    if args.arch_xlsx:
        if not PANDAS_AVAILABLE:
            logger.error("pandas is required for Excel support. Install with: pip install pandas openpyxl")
            sys.exit(1)
        
        arch_path = Path(args.arch_xlsx)
        if not arch_path.exists():
            logger.error(f"Architecture Excel not found: {arch_path}")
            sys.exit(1)

        business_terms = generator.generate_from_architecture_xlsx(
            arch_path, header_rows=3, min_term_length=args.min_term_length
        )

    elif args.arch_csv:
        arch_path = Path(args.arch_csv)
        if not arch_path.exists():
            logger.error(f"Architecture CSV not found: {arch_path}")
            sys.exit(1)

        business_terms = generator.generate_from_architecture_csv(
            arch_path, min_term_length=args.min_term_length
        )

    # 优先处理Excel文件（推荐方式）
    if args.metrics_xlsx:
        if not PANDAS_AVAILABLE:
            logger.error("pandas is required for Excel support. Install with: pip install pandas openpyxl")
            sys.exit(1)
        
        metrics_path = Path(args.metrics_xlsx)
        if not metrics_path.exists():
            logger.error(f"Metrics Excel not found: {metrics_path}")
            sys.exit(1)

        if business_terms is None:
            business_terms = {
                "term_to_table": {},
                "term_to_schema": {},
                "table_keywords": {},
            }

        business_terms = generator.generate_from_metrics_xlsx(
            metrics_path, business_terms, header_rows=2, min_term_length=args.min_term_length
        )

    elif args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        if not metrics_path.exists():
            logger.error(f"Metrics CSV not found: {metrics_path}")
            sys.exit(1)

        if business_terms is None:
            business_terms = {
                "term_to_table": {},
                "term_to_schema": {},
                "table_keywords": {},
            }

        business_terms = generator.generate_from_metrics_csv(
            metrics_path, business_terms, min_term_length=args.min_term_length
        )

    if business_terms is None:
        logger.error("No input files provided. Use --arch-xlsx/--metrics-xlsx (recommended) or --arch-csv/--metrics-csv")
        sys.exit(1)

    # 合并DDL注释
    if args.merge_ddl:
        business_terms = generator.merge_with_ddl_comments(business_terms)

    # 保存输出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_to_yaml(business_terms, output_path, include_confidence=args.include_confidence)

    logger.info("Business configuration generation complete!")


if __name__ == "__main__":
    main()
