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
from typing import Dict, List, Set, Tuple

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
from datus.storage.ext_knowledge.store import ExtKnowledgeStore
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

    def _read_excel_with_header(self, xlsx_path: Path, header_rows: int, sheet_name = None) -> List[Dict]:
        """
        读取Excel文件，智能处理多行表头
        
        针对复杂表头结构：
        - 数据架构设计文档：
          - 第1行：分组标题（资产目录/业务属性/技术属性/管理属性）
          - 第2行：实际列名（物理表名、字段名等）★ 使用这行
          - 第3行：列说明/注释
        
        - 指标清单（特殊）：
          - 第1行：主要列名（部分列为空，如"来源dws模型"）
          - 第2行：子列名/补充列名（包含"来源dws模型"等）★ 合并使用
        
        Args:
            xlsx_path: Excel文件路径
            header_rows: 表头行数（用于确定实际列名所在行）
            sheet_name: 工作表名称或索引（None时自动检测：优先'Sheet1'，否则第一个）
            
        Returns:
            List[Dict]: 数据行列表
        """
        if not PANDAS_AVAILABLE:
            logger.error("pandas not available. Install with: pip install pandas openpyxl")
            return []
        
        try:
            # 如果没有指定sheet_name，自动检测
            if sheet_name is None:
                # 获取所有sheet名称
                xl = pd.ExcelFile(xlsx_path)
                sheet_names = xl.sheet_names
                
                # 优先查找 "Sheet1"，否则使用第一个sheet
                if "Sheet1" in sheet_names:
                    sheet_name = "Sheet1"
                    logger.info(f"[Excel读取] 自动选择工作表: 'Sheet1'")
                else:
                    sheet_name = 0  # 第一个sheet
                    logger.info(f"[Excel读取] 自动选择第一个工作表: '{sheet_names[0]}'")
            else:
                # 尝试将数字字符串转为整数
                try:
                    sheet_name = int(sheet_name)
                    logger.info(f"[Excel读取] 使用工作表索引: {sheet_name}")
                except ValueError:
                    logger.info(f"[Excel读取] 使用工作表名称: '{sheet_name}'")
            
            # 智能表头检测：
            # - 如果 header_rows=3，表示第2行（索引1）是实际列名
            # - 如果 header_rows=2，表示第1行（索引0）是实际列名
            actual_header_row = header_rows - 2 if header_rows >= 2 else 0
            
            logger.info(f"[Excel读取] 读取 {xlsx_path.name}, 使用第 {actual_header_row + 1} 行作为列名")
            
            # 读取Excel，使用实际列名行作为header
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=actual_header_row)
            logger.info(f"[Excel读取] 数据行数: {len(df)}")
            
            # 对于2行表头的情况（如指标清单），检查是否需要读取第2行补充列名
            if header_rows == 2:
                # 读取第2行（作为数据）来获取补充列名
                df_second_row = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, skiprows=actual_header_row+1, nrows=1)
                if not df_second_row.empty:
                    # 获取第2行的值作为补充列名
                    second_row_values = df_second_row.iloc[0].tolist()
                    original_columns = df.columns.tolist()
                    
                    # 合并列名：如果原列名是Unnamed，使用第2行的值
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
                    
                    # 检查是否有更新
                    if merged_columns != original_columns:
                        logger.info(f"[Excel读取] 检测到第2行补充列名，已合并")
                        df.columns = merged_columns
            
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
            if len(df.columns) > 30:
                logger.info(f"[Excel读取] 清理后的列名(31-): {df.columns.tolist()[30:]}")
            
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
        min_term_length: int = 2,
        sheet_name = None
    ) -> Dict:
        """
        从数据架构Excel生成业务术语映射（支持多行表头）
        
        Args:
            xlsx_path: Excel文件路径
            header_rows: 表头行数（数据架构设计文档为3行）
            min_term_length: 最小术语长度
            sheet_name: 工作表名称或索引（None时自动检测）
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

        records = self._read_excel_with_header(xlsx_path, header_rows, sheet_name)
        
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

    def _load_architecture_entity_mapping(self, arch_xlsx_path: Path) -> Dict[str, List[str]]:
        """
        从数据架构Excel加载逻辑实体（中文）到物理表名的映射
        
        用于将Sheet1的"来源dws模型"中文描述（如"原始线索聚合明细"）
        映射到物理表名（如"dws_obtain_original_clue_2h_di"）
        
        Returns:
            Dict[逻辑实体中文名, 物理表名列表]
        """
        entity_to_tables = {}
        
        try:
            df = pd.read_excel(arch_xlsx_path, header=1)  # 第2行是实际列名
            logger.info(f"[数据架构] 加载了 {len(df)} 行架构数据")
            
            for _, row in df.iterrows():
                table_name = str(row.get('物理表名', '')).strip() if pd.notna(row.get('物理表名')) else ''
                entity = str(row.get('逻辑实体（中文）', '')).strip() if pd.notna(row.get('逻辑实体（中文）')) else ''
                
                if not table_name or not entity:
                    continue
                if table_name == '定义物理表名' or entity == '定义逻辑实体中文名':
                    continue  # 跳过表头行
                
                if entity not in entity_to_tables:
                    entity_to_tables[entity] = []
                if table_name not in entity_to_tables[entity]:
                    entity_to_tables[entity].append(table_name)
            
            logger.info(f"[数据架构] 建立了 {len(entity_to_tables)} 个逻辑实体映射")
            
            # 显示包含"明细"的实体样例
            sample_entities = [e for e in entity_to_tables.keys() if '明细' in e][:5]
            for e in sample_entities:
                logger.info(f"[数据架构] 样例: {e} -> {entity_to_tables[e][:2]}")
            
        except Exception as e:
            logger.warning(f"[数据架构] 加载逻辑实体映射失败: {e}")
        
        return entity_to_tables

    def _match_chinese_model_to_entity(
        self, 
        chinese_model: str, 
        entity_to_tables: Dict[str, List[str]]
    ) -> List[str]:
        """
        将Sheet1的中文模型描述（如"原始线索聚合明细"）匹配到物理表名
        
        匹配策略：
        1. 精确匹配
        2. 包含匹配（chinese_model 是 entity 的子串）
        3. 关键词匹配（提取核心业务词）
        """
        if not chinese_model or len(chinese_model) < 2:
            return []
        
        chinese_clean = chinese_model.strip()
        
        # 1. 精确匹配
        if chinese_clean in entity_to_tables:
            return entity_to_tables[chinese_clean]
        
        # 2. 包含匹配（Sheet1的"原始线索聚合明细"匹配"原始线索聚合明细（油+电）"）
        for entity, tables in entity_to_tables.items():
            if chinese_clean in entity:
                return tables
        
        # 3. 关键词匹配（提取核心业务词如"线索"、"试驾"等）
        keywords = self._extract_core_keywords(chinese_clean)
        if keywords:
            best_match = None
            best_score = 0
            for entity, tables in entity_to_tables.items():
                score = sum(1 for kw in keywords if kw in entity)
                if score > best_score:
                    best_score = score
                    best_match = tables
            if best_match and best_score >= 2:  # 至少匹配2个关键词
                return best_match
        
        return []

    def _load_metrics_code_to_tables_mapping(self, xlsx_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        从Sheet2读取指标映射关系
        
        Sheet2包含人工维护的指标-表关联信息，返回两个映射：
        1. 指标编码 -> 物理表名列表
        2. 中文描述 -> 物理表名列表（用于匹配Sheet1的"来源dws模型"列）
        
        Returns:
            Tuple[Dict[编码, 表名列表], Dict[中文描述, 表名列表]]
        """
        code_to_tables = {}
        chinese_to_tables = {}  # 新增：中文描述 -> 表名
        
        try:
            # 尝试读取第二个工作表（Sheet2）
            xl = pd.ExcelFile(xlsx_path)
            sheet_names = xl.sheet_names
            
            if len(sheet_names) < 2:
                logger.info("[Sheet2] Excel只有一个工作表，跳过Sheet2读取")
                return code_to_tables, chinese_to_tables
            
            # 读取Sheet2，使用第一行作为表头
            df_sheet2 = pd.read_excel(xlsx_path, sheet_name=1, header=0)
            logger.info(f"[Sheet2] 读取到 {len(df_sheet2)} 行维护数据")
            
            for _, row in df_sheet2.iterrows():
                # 提取指标编码
                code = str(row.get('编码', '')).strip() if pd.notna(row.get('编码')) else ''
                
                # 提取指标定义（中文描述）
                metric_def = str(row.get('指标定义', '')).strip() if pd.notna(row.get('指标定义')) else ''
                
                # 提取指标涉及表名（可能包含多行，如 "a:table1\nb:table2"）
                tables_raw = str(row.get('指标涉及表名', '')).strip() if pd.notna(row.get('指标涉及表名')) else ''
                if not tables_raw or tables_raw == 'nan':
                    continue
                
                # 解析表名（去除前缀如 "a:", "b:"）
                tables = []
                for line in tables_raw.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # 去除前缀（如 "a:", "b:", "table_a:" 等）
                    if ':' in line:
                        table_name = line.split(':', 1)[1].strip()
                    else:
                        table_name = line
                    if table_name and self._is_valid_table_name(table_name):
                        tables.append(table_name)
                
                if not tables:
                    continue
                
                # 映射1: 编码 -> 表名
                if code and code != 'nan':
                    code_to_tables[code] = tables
                
                # 映射2: 中文描述 -> 表名（用于匹配Sheet1）
                # 注意：Sheet2的"指标定义"是详细业务定义，不是"来源dws模型"值
                if metric_def and metric_def != 'nan':
                    clean_def = ' '.join(metric_def.split())
                    chinese_to_tables[clean_def] = tables
                    # 同时存储前10个字符的简短版本（应对截断情况）
                    if len(clean_def) > 10:
                        chinese_to_tables[clean_def[:10]] = tables
            
            logger.info(f"[Sheet2] 成功解析 {len(code_to_tables)} 个编码映射, {len(chinese_to_tables)} 个中文描述映射")
            
        except Exception as e:
            logger.warning(f"[Sheet2] 读取失败: {e}")
        
        return code_to_tables, chinese_to_tables

    def _match_chinese_model_name_to_table(
        self, 
        chinese_name: str, 
        available_tables: List[str],
        table_comments: Dict[str, str]
    ) -> List[str]:
        """
        将中文模型名称（如'原始线索聚合明细'）匹配到物理表名
        
        策略：
        1. 直接匹配 table_comment
        2. 模糊匹配（包含关系）
        3. 关键词匹配
        """
        if not chinese_name or len(chinese_name) < 2:
            return []
        
        matched_tables = []
        chinese_lower = chinese_name.lower()
        
        for table_name in available_tables:
            comment = table_comments.get(table_name, '')
            
            # 1. 精确匹配 comment
            if comment and chinese_name == comment:
                matched_tables.append(table_name)
                continue
            
            # 2. 模糊匹配（中文名包含在comment中）
            if comment and chinese_name in comment:
                matched_tables.append(table_name)
                continue
            
            # 3. 关键词匹配（提取核心业务词）
            # 如 "原始线索聚合明细" -> 匹配包含 "线索" 的表
            keywords = self._extract_core_keywords(chinese_name)
            for kw in keywords:
                if kw in table_name.lower() or (comment and kw in comment):
                    matched_tables.append(table_name)
                    break
        
        return list(set(matched_tables))  # 去重

    def _extract_core_keywords(self, text: str) -> List[str]:
        """提取核心业务关键词（用于表名匹配）"""
        keywords = []
        # 匹配2-8字的中文业务词
        for match in re.finditer(r"[\u4e00-\u9fa5]{2,8}", text):
            kw = match.group()
            # 过滤常见停用词
            if kw not in ['明细', '汇总', '统计', '计算', '结果', '数据', '信息']:
                keywords.append(kw)
        return keywords

    def _extract_business_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """
        从业务定义中提取高价值业务关键词（用于text2sql）
        
        策略：提取业务场景关键词，如：
        - "首触"、"有效线索"、"试驾预约"、"订单转化"
        - 过滤通用词汇（的、和、是等）
        """
        if not text:
            return []
        
        # 业务关键词模式：2-8字中文业务词
        keywords = []
        for match in re.finditer(r"[\u4e00-\u9fa5]{2,8}", text):
            kw = match.group()
            # 停用词过滤
            if kw in STOP_WORDS or len(kw) < min_length:
                continue
            # 技术词汇过滤
            if kw in ['明细', '汇总', '统计', '计算', '结果', '数据', '信息', '字段', '表名']:
                continue
            keywords.append(kw)
        
        # 去重并保持顺序
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]  # 限制数量，避免噪音

    def _extract_metric_terms(self, metric_name: str, min_length: int = 2) -> List[str]:
        """
        从指标名称中提取业务术语
        
        处理逻辑：
        1. 去除常见指标后缀（数、量、率、占比、金额、次数等）
        2. 提取核心业务概念（如"有效线索数"→"有效线索"）
        3. 保留原始指标名称作为备选
        
        示例：
        - "原始线索数" → ["原始线索", "线索"]
        - "首触及时率" → ["首触及时", "首触"]
        - "有效线索占比" → ["有效线索"]
        - "线索目标（下发量目标）" → ["线索目标", "下发量目标", "线索", "下发量"]
        """
        if not metric_name:
            return []
        
        terms = []
        
        # 清理指标名称（去除括号内容，保留主要部分）
        clean_name = re.sub(r'[（(].*?[）)]', '', metric_name).strip()
        
        # 常见指标后缀（需要去除的）
        metric_suffixes = [
            '数量', '数', '量', '率', '占比', '比例', '金额', '次数', '天数', '时长',
            '目标', '实绩', '合计', '汇总', '统计', '平均', '最大', '最小',
            '及时', '完成', '达成', '转化', '变更', '新增', '活跃'
        ]
        
        # 策略1: 逐层去除后缀，提取核心概念
        current = clean_name
        for suffix in metric_suffixes:
            if current.endswith(suffix) and len(current) > len(suffix) + min_length:
                core_term = current[:-len(suffix)]
                if len(core_term) >= min_length and self._is_meaningful_term(core_term):
                    terms.append(core_term)
                # 继续处理剩余部分（如"有效线索数"→提取"线索"）
                current = core_term
        
        # 策略2: 提取2-8字的中文业务词
        for match in re.finditer(r"[\u4e00-\u9fa5]{%d,8}" % min_length, clean_name):
            kw = match.group()
            if self._is_meaningful_term(kw):
                terms.append(kw)
        
        # 策略3: 特定业务术语识别（基于常见模式）
        special_patterns = [
            r'(首触)\w*',  # 首触及时率 → 首触
            r'(有效\w+)',  # 有效线索 → 有效线索
            r'(原始\w+)',  # 原始线索 → 原始线索
            r'(人工\w+)',  # 人工战败 → 人工战败
            r'(自然\w+)',  # 自然线索 → 自然线索
            r'(战败\w*)',  # 战败线索 → 战败
        ]
        for pattern in special_patterns:
            for match in re.finditer(pattern, clean_name):
                term = match.group(1)
                if len(term) >= min_length and term not in terms:
                    terms.append(term)
        
        # 去重并保持顺序
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen and len(term) >= min_length:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:5]  # 限制数量

    def _load_table_comments_from_lancedb(self) -> Dict[str, str]:
        """
        从LanceDB加载所有表的table_comment信息
        
        用于中文描述到物理表名的模糊匹配
        
        Returns:
            Dict[table_name, table_comment]
        """
        table_comments = {}
        
        try:
            import lancedb
            
            # 直接连接LanceDB，不依赖schema_storage
            db = lancedb.connect(self.db_path)
            
            # 检查schema_metadata表是否存在
            if "schema_metadata" not in db.table_names():
                logger.warning("[LanceDB] schema_metadata表不存在")
                return table_comments
            
            # 读取所有schema记录
            table = db.open_table("schema_metadata")
            all_records = table.to_pandas()
            
            if all_records is None or all_records.empty:
                logger.warning("[LanceDB] schema表为空")
                return table_comments
            
            logger.info(f"[LanceDB] 从 {self.db_path} 读取到 {len(all_records)} 条schema记录")
            
            # 提取table_name和table_comment
            for _, row in all_records.iterrows():
                table_name = row.get('table_name', '')
                comment = row.get('table_comment', '')
                
                if table_name and isinstance(table_name, str):
                    table_comments[table_name] = str(comment) if comment else ''
            
            # 显示样本
            sample_tables = list(table_comments.keys())[:3]
            logger.info(f"[LanceDB] 加载了 {len(table_comments)} 个表的comment信息")
            logger.info(f"[LanceDB] 样例: {[(t, table_comments[t][:30]) for t in sample_tables]}")
            
        except Exception as e:
            logger.warning(f"[LanceDB] 加载表comment失败: {e}")
            import traceback
            logger.debug(f"[LanceDB] 错误详情: {traceback.format_exc()}")
        
        return table_comments

    def _match_chinese_to_table_by_comment(
        self, 
        chinese_desc: str, 
        table_comments: Dict[str, str]
    ) -> List[str]:
        """
        通过table_comment将中文描述匹配到物理表名
        
        策略：
        1. 精确匹配：chinese_desc == table_comment
        2. 包含匹配：chinese_desc 在 table_comment 中
        3. 关键词匹配：提取关键词匹配
        """
        if not chinese_desc or len(chinese_desc) < 2:
            return []
        
        matched_tables = []
        chinese_clean = chinese_desc.strip()
        
        for table_name, comment in table_comments.items():
            if not comment:
                continue
            
            comment_clean = comment.strip()
            
            # 1. 精确匹配
            if chinese_clean == comment_clean:
                matched_tables.append(table_name)
                continue
            
            # 2. 包含匹配（中文描述是comment的子串）
            if chinese_clean in comment_clean:
                matched_tables.append(table_name)
                continue
            
            # 3. 关键词匹配（提取核心业务词）
            keywords = self._extract_core_keywords(chinese_clean)
            for kw in keywords:
                if len(kw) >= 4 and kw in comment_clean:  # 至少4个字的关键词
                    matched_tables.append(table_name)
                    break
        
        return list(set(matched_tables))  # 去重

    def generate_from_metrics_xlsx(
        self,
        xlsx_path: Path,
        existing_terms: Dict,
        header_rows: int = 2,
        min_term_length: int = 2,
        sheet_name = None,
        arch_xlsx_path: Path = None
    ) -> Dict:
        """
        从指标清单Excel提取text2sql关键映射
        
        【设计聚焦】只提取对text2sql有直接价值的映射：
        1. 指标名称 -> 物理表名（核心，用于schema discovery）
        2. 业务关键词 -> 表名（用于表排序boost）
        
        【数据来源】
        - Sheet2: 优先，包含人工维护的指标编码->物理表名映射
        - Sheet1: 补充，提供指标名称和业务定义
        
        【关键原则】
        - 不输出指标编码（M-0018），这只是内部关联媒介
        - 不强求100%表名匹配，业务术语本身就有检索价值
        - 简化term_to_schema，字段映射准确率不高，由LLM从表结构推断
        
        Args:
            xlsx_path: Excel文件路径
            existing_terms: 已有的术语映射
            header_rows: 表头行数（指标清单Sheet1为2行）
            min_term_length: 最小术语长度
            sheet_name: 工作表名称或索引（None时自动检测，Sheet1为主表）
        """
        logger.info(f"Processing metrics Excel for text2sql: {xlsx_path}")

        # 核心输出：只保留text2sql真正需要的映射
        term_to_table = defaultdict(set, existing_terms.get("term_to_table", {}))
        table_keywords = existing_terms.get("table_keywords", {})
        # term_to_schema简化：只保留高置信度的字段映射
        term_to_schema = defaultdict(set, existing_terms.get("term_to_schema", {}))

        stats = {
            "total_metrics": 0,         # 处理的总指标数
            "valid_metrics": 0,         # 有效指标数（有名称）
            "with_tables": 0,           # 成功关联到表的指标数
            "tables_added": set(),      # 涉及的物理表集合
            "sheet2_matched": 0,        # 通过Sheet2编码精确匹配的数量
            "entity_matched": 0,        # 通过数据架构逻辑实体匹配的数量
            "sheet2_chinese_matched": 0,  # 通过Sheet2中文描述匹配的数量
            "lancedb_matched": 0,       # 通过LanceDB table_comment匹配的数量
            "keyword_matched": 0,       # 通过关键词弱匹配的数量
        }

        # 步骤1: 从Sheet2加载编码->表名映射，同时构建中文描述->表名映射
        code_to_tables, chinese_to_tables = self._load_metrics_code_to_tables_mapping(xlsx_path)
        
        # 步骤1.5: 从数据架构Excel加载逻辑实体->表名映射（关键！）
        # 用于将Sheet1的"来源dws模型"中文描述（如"原始线索聚合明细"）映射到物理表名
        entity_to_tables = {}
        if arch_xlsx_path and arch_xlsx_path.exists():
            entity_to_tables = self._load_architecture_entity_mapping(arch_xlsx_path)
        
        # 步骤1.6: 从LanceDB加载table_comment信息（用于中文描述匹配）
        lancedb_table_comments = self._load_table_comments_from_lancedb()

        # 步骤2: 读取Sheet1（主指标清单）
        records = self._read_excel_with_header(xlsx_path, header_rows, sheet_name)
        
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

        # 调试：检查是否读取到"来源dws模型"列
        if records:
            all_cols = list(records[0].keys())
            logger.info(f"[指标清单] 总列数: {len(all_cols)}")
            dws_cols = [c for c in all_cols if 'dws' in c.lower() or '来源' in c or '模型' in c]
            logger.info(f"[指标清单] DWS相关列: {dws_cols}")

        for row in records:
            stats["total_metrics"] += 1

            # 提取核心字段（text2sql相关）
            metric_code = self._extract_field(row, ["指标编码", "指标编码 -固定值（勿改）"])
            metric_name = self._extract_field(row, ["指标名称"])
            biz_def = self._extract_field(row, ["业务定义及说明", "业务定义"])
            biz_activity = self._extract_field(row, ["业务活动"])
            # 关键：尝试多种可能的"来源dws模型"列名变体
            source_model = self._extract_field(row, [
                "来源dws模型", "dws模型", "来源模型", 
                "模型", "source_model", "来源表"
            ])
            
            # 提取分类字段（用于业务术语提取）
            category1 = self._extract_field(row, ["分类1", "分类一", "一级分类"])
            category2 = self._extract_field(row, ["分类2", "分类二", "二级分类"])

            if not metric_name:
                continue

            stats["valid_metrics"] += 1

            # === 核心：获取物理表名（多策略，优先级递减）===
            source_tables = []
            match_source = None
            
            # 策略1: Sheet2编码精确映射（最高置信度）
            if metric_code and metric_code in code_to_tables:
                source_tables = code_to_tables[metric_code]
                stats["sheet2_matched"] += 1
                match_source = "sheet2_code"
            
            # 策略2: Sheet1的"来源dws模型"是物理表名
            elif source_model and self._is_valid_table_name(source_model):
                source_tables = [source_model]
                match_source = "sheet1_direct"
            
            # 策略3: Sheet1的"来源dws模型"是中文描述，用数据架构逻辑实体映射匹配（关键！）
            elif source_model and entity_to_tables:
                matched = self._match_chinese_model_to_entity(source_model, entity_to_tables)
                if matched:
                    source_tables = matched
                    match_source = "arch_entity"
                    stats["entity_matched"] = stats.get("entity_matched", 0) + 1
                    if stats.get("entity_matched", 0) <= 3:
                        logger.info(f"[实体匹配] {metric_name}: '{source_model}' -> {matched}")
            
            # 策略4: Sheet1的"来源dws模型"是中文描述，用Sheet2映射匹配
            elif source_model:
                # 清理输入的中文描述
                clean_source = ' '.join(source_model.split())
                # 尝试完整匹配
                if clean_source in chinese_to_tables:
                    source_tables = chinese_to_tables[clean_source]
                    match_source = "sheet2_chinese"
                    stats["sheet2_chinese_matched"] += 1
                # 尝试前10字符匹配（应对截断）
                elif len(clean_source) > 10 and clean_source[:10] in chinese_to_tables:
                    source_tables = chinese_to_tables[clean_source[:10]]
                    match_source = "sheet2_chinese_prefix"
                    stats["sheet2_chinese_matched"] += 1
                # 策略5: 使用LanceDB的table_comment进行模糊匹配
                elif lancedb_table_comments:
                    matched = self._match_chinese_to_table_by_comment(
                        clean_source, lancedb_table_comments
                    )
                    if matched:
                        source_tables = matched
                        match_source = "lancedb_comment"
                        stats["lancedb_matched"] = stats.get("lancedb_matched", 0) + 1
            
            # === 修改：无论是否匹配到表，都提取业务术语 ===
            
            if source_tables:
                # 成功匹配到表，建立指标名称->表名映射
                stats["with_tables"] += 1
                
                # === 核心输出1: 指标名称 -> 表名 ===
                for table_name in source_tables:
                    term_to_table[metric_name].add(table_name)
                    stats["tables_added"].add(table_name)

                # === 核心输出2: 业务活动+指标名 -> 表名（增加上下文）===
                if biz_activity:
                    composite_term = f"{biz_activity}_{metric_name}"
                    for table_name in source_tables:
                        term_to_table[composite_term].add(table_name)

                # === 核心输出3: 从业务定义提取关键词 -> 表名 ===
                if biz_def:
                    # 提取业务关键词（如"首触"、"有效线索"）
                    keywords = self._extract_business_keywords(biz_def, min_term_length)
                    for kw in keywords:
                        for table_name in source_tables:
                            term_to_table[kw].add(table_name)
                        # 同时加入table_keywords（用于表排序）
                        if len(kw) >= 4:  # 较长关键词更可靠
                            table_keywords[kw] = list(source_tables)[0]
                
                # === 核心输出3.5: 从指标名称提取业务术语 -> 表名 ===
                # 指标名称包含关键业务概念（如"有效线索数"→"有效线索"、"首触及时率"→"首触"）
                if metric_name:
                    metric_terms = self._extract_metric_terms(metric_name, min_term_length)
                    for term in metric_terms:
                        for table_name in source_tables:
                            term_to_table[term].add(table_name)
                        if len(term) >= 2:
                            table_keywords[term] = list(source_tables)[0]
                        # 调试输出
                        if stats.get('metric_terms_added', 0) < 5:
                            logger.info(f"[指标术语] '{term}' -> {source_tables[0]} (来自指标: {metric_name})")
                        stats["metric_terms_added"] = stats.get("metric_terms_added", 0) + 1
                
                # === 核心输出4: 从分类字段提取业务术语 -> 表名 ===
                # 分类1、分类2包含关键业务术语（如"订单"、"试驾"、"线索"等）
                for category in [category1, category2]:
                    if category and len(category) >= min_term_length and self._is_meaningful_term(category):
                        clean_cat = category.strip()
                        if clean_cat and clean_cat.lower() not in ['nan', 'none', 'null', '']:
                            for table_name in source_tables:
                                term_to_table[clean_cat].add(table_name)
                            # 同时加入table_keywords（用于表排序）
                            if len(clean_cat) >= 2:
                                table_keywords[clean_cat] = list(source_tables)[0]
                            # 调试输出前几个分类术语
                            if stats.get('category_terms_added', 0) < 5:
                                logger.info(f"[分类术语] '{clean_cat}' -> {source_tables[0]} (来自: {metric_name})")
                            stats["category_terms_added"] = stats.get("category_terms_added", 0) + 1
            
            else:
                # 未匹配到表，尝试从业务定义提取关键词，并通过其他方式关联表
                if biz_def:
                    keywords = self._extract_business_keywords(biz_def, min_term_length)
                    
                    # 尝试通过关键词匹配数据架构中的表
                    for kw in keywords:
                        if len(kw) < 4:  # 太短的关键词跳过
                            continue
                            
                        # 在数据架构的逻辑实体中查找匹配的表
                        matched_tables = []
                        for entity, tables in entity_to_tables.items():
                            if kw in entity:
                                matched_tables.extend(tables)
                        
                        # 去重并限制数量
                        matched_tables = list(set(matched_tables))[:3]
                        
                        if matched_tables:
                            for table_name in matched_tables:
                                term_to_table[kw].add(table_name)
                            # 记录这种弱关联
                            if len(kw) >= 4:
                                table_keywords[kw] = matched_tables[0]
                            
                            if stats.get('keyword_matched', 0) < 5:
                                logger.info(f"[关键词弱匹配] '{kw}' -> {matched_tables[:2]} (来自: {metric_name})")
                            stats["keyword_matched"] = stats.get("keyword_matched", 0) + 1

        # 统计各种匹配来源
        sheet2_chinese_matched = stats.get('sheet2_chinese_matched', 0)
        lancedb_matched = stats.get('lancedb_matched', 0)
        entity_matched = stats.get('entity_matched', 0)
        keyword_matched = stats.get('keyword_matched', 0)
        
        logger.info(
            f"[指标清单] text2sql映射生成: {stats['with_tables']}/{stats['valid_metrics']} 指标关联表, "
            f"{len(stats['tables_added'])} 个物理表, "
            f"{keyword_matched}个关键词弱匹配"
        )
        logger.info(
            f"[指标清单] 匹配来源: {stats['sheet2_matched']}个编码精确匹配, "
            f"{entity_matched}个数据架构实体匹配, "
            f"{sheet2_chinese_matched}个Sheet2中文匹配, "
            f"{lancedb_matched}个LanceDB注释匹配"
        )
        
        # 输出关键业务术语样本（用于验证）
        key_terms = ['有效线索', '首触', '试驾', '订单', '客户']
        found_terms = [t for t in key_terms if t in term_to_table]
        logger.info(f"[术语验证] 关键业务术语覆盖: {found_terms}")
        
        # 调试：显示未覆盖的关键术语
        missing_terms = [t for t in key_terms if t not in term_to_table]
        if missing_terms:
            logger.info(f"[术语验证] 未覆盖的关键术语: {missing_terms}")
        
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

    def generate_metrics_catalog_from_xlsx(
        self,
        xlsx_path: Path,
        header_rows: int = 2,
        sheet_name = None
    ) -> List[Dict]:
        """
        从指标清单Excel生成指标目录（用于导入ext_knowledge）
        
        每条记录包含：
        - subject_path: [业务活动, 指标层级]
        - name: 指标编码
        - terminology: 指标名称
        - explanation: 业务定义+计算公式+维度+来源表
        - metadata: 来源表、维度、指标层级等
        
        Args:
            xlsx_path: 指标清单Excel文件路径
            header_rows: 表头行数
            sheet_name: 工作表名称或索引
            
        Returns:
            List[Dict]: 指标目录条目列表
        """
        logger.info(f"Generating metrics catalog from: {xlsx_path}")
        
        metrics = []
        records = self._read_excel_with_header(xlsx_path, header_rows, sheet_name)
        
        if not records:
            logger.warning("[指标目录] 未读取到有效数据")
            return metrics
        
        for row in records:
            # 提取关键字段
            metric_code = self._extract_field(row, ["指标编码", "指标编码 -固定值（勿改）"])
            metric_name = self._extract_field(row, ["指标名称"])
            biz_def = self._extract_field(row, ["业务定义及说明", "业务定义"])
            calc_logic = self._extract_field(row, ["计算公式/业务逻辑", "计算公式", "业务逻辑"])
            dimensions = self._extract_field(row, ["公共维度", "维度"])
            source_model = self._extract_field(row, ["来源dws模型", "dws模型", "来源模型", "来源表"])
            biz_activity = self._extract_field(row, ["业务活动"])
            metric_level = self._extract_field(row, ["指标层级", "层级"])
            category1 = self._extract_field(row, ["分类1", "分类一", "一级分类"])
            category2 = self._extract_field(row, ["分类2", "分类二", "二级分类"])
            
            if not metric_code or not metric_name:
                continue
            
            # 构建详细解释文本
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
            
            # 构建subject_path
            subject_path = ["Metrics"]
            if biz_activity:
                subject_path.append(biz_activity)
            if category1:
                subject_path.append(category1)
            
            # 注意：ext_knowledge表的schema只支持特定字段，不要添加额外字段如metadata
            metric_entry = {
                "subject_path": subject_path,
                "name": metric_code,
                "terminology": metric_name,
                "explanation": explanation,
            }
            metrics.append(metric_entry)
        
        logger.info(f"[指标目录] 生成了 {len(metrics)} 个指标条目")
        return metrics

    def import_metrics_to_ext_knowledge(
        self,
        metrics_catalog: List[Dict],
        clear_existing: bool = False
    ) -> int:
        """
        将指标目录导入到 LanceDB 的 ext_knowledge 表
        
        Args:
            metrics_catalog: 指标目录列表
            clear_existing: 是否清空已有指标数据
            
        Returns:
            int: 成功导入的条目数
        """
        if not metrics_catalog:
            logger.warning("[ExtKnowledge] 没有指标数据需要导入")
            return 0
        
        try:
            # 初始化 ExtKnowledgeStore
            ext_store = ExtKnowledgeStore(db_path=self.db_path)
            
            # 检查表是否存在且schema是否正确
            import lancedb
            db = lancedb.connect(self.db_path)
            table_exists = "ext_knowledge" in db.table_names()
            
            if table_exists:
                temp_table = db.open_table("ext_knowledge")
                schema_fields = set(temp_table.schema.names)
                expected_fields = {"name", "subject_node_id", "created_at", "terminology", "explanation", "vector"}
                
                if not expected_fields.issubset(schema_fields):
                    logger.warning(f"[ExtKnowledge] 表schema不匹配，字段: {schema_fields}")
                    logger.warning("[ExtKnowledge] 删除旧表并重新创建...")
                    db.drop_table("ext_knowledge")
                    # 重新初始化 store 以创建新表
                    ext_store = ExtKnowledgeStore(db_path=self.db_path)
            
            # 确保表已准备好
            ext_store._ensure_table_ready()
            logger.info(f"[ExtKnowledge] 表schema: {ext_store.table.schema}")
            
            # 如果需要清空现有Metrics数据
            if clear_existing:
                logger.info("[ExtKnowledge] 清空现有Metrics分类数据...")
                # 注意：这里假设ExtKnowledgeStore有删除功能，如果没有可以跳过
                # ext_store.delete_by_subject_path(["Metrics"])
            
            # 批量导入
            logger.info(f"[ExtKnowledge] 开始导入 {len(metrics_catalog)} 个指标...")
            
            # 调试：显示第一个条目的字段
            if metrics_catalog:
                first_entry = metrics_catalog[0]
                logger.info(f"[ExtKnowledge] 第一个条目字段: {list(first_entry.keys())}")
                logger.info(f"[ExtKnowledge] 第一个条目: {first_entry}")
            
            # 分批导入，便于定位问题
            batch_size = 50
            imported_count = 0
            for i in range(0, len(metrics_catalog), batch_size):
                batch = metrics_catalog[i:i+batch_size]
                try:
                    ext_store.batch_store_knowledge(batch)
                    imported_count += len(batch)
                    logger.info(f"[ExtKnowledge] 已导入 {imported_count}/{len(metrics_catalog)} 个指标")
                except Exception as batch_e:
                    logger.error(f"[ExtKnowledge] 批次 {i//batch_size + 1} 导入失败: {batch_e}")
                    # 尝试单条导入定位问题
                    for j, entry in enumerate(batch):
                        try:
                            ext_store.batch_store_knowledge([entry])
                            imported_count += 1
                        except Exception as entry_e:
                            logger.error(f"[ExtKnowledge] 条目 {i+j} 导入失败: {entry}")
                            logger.error(f"[ExtKnowledge] 错误: {entry_e}")
                            break
                    break
            
            logger.info(f"[ExtKnowledge] 成功导入 {imported_count} 个指标到 ext_knowledge 表")
            return imported_count
            
        except Exception as e:
            logger.error(f"[ExtKnowledge] 导入失败: {e}")
            import traceback
            logger.error(f"[ExtKnowledge] 错误详情: {traceback.format_exc()}")
            return 0

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

  # 指定工作表名称或索引
  python scripts/generate_business_config.py \\
      --config=conf/agent.yml \\
      --namespace=test \\
      --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \\
      --arch-sheet-name="Sheet1" \\
      --metrics-xlsx=/path/to/指标清单v2.4.xlsx \\
      --metrics-sheet-name=0

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

  # 导入指标到 ext_knowledge 表（用于Text2SQL语义检索）
  python scripts/generate_business_config.py \\
      --config=conf/agent.yml \\
      --namespace=test \\
      --metrics-xlsx=/path/to/指标清单v2.4.xlsx \\
      --import-to-lancedb \\
      --verbose
        """,
    )

    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", required=True, help="Target namespace")
    parser.add_argument("--arch-csv", help="Path to 数据架构详细设计 CSV file (legacy)")
    parser.add_argument("--metrics-csv", help="Path to 指标清单 CSV file (legacy, optional)")
    parser.add_argument("--arch-xlsx", help="Path to 数据架构详细设计 Excel file (推荐, 支持多行表头)")
    parser.add_argument("--metrics-xlsx", help="Path to 指标清单 Excel file (推荐, 支持多行表头)")
    parser.add_argument(
        "--arch-sheet-name",
        dest="arch_sheet_name",
        default=None,
        help="Sheet name or index for architecture Excel (default: first sheet or 'Sheet1')",
    )
    parser.add_argument(
        "--metrics-sheet-name",
        dest="metrics_sheet_name",
        default=None,
        help="Sheet name or index for metrics Excel (default: first sheet or 'Sheet1')",
    )
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
    parser.add_argument(
        "--import-to-lancedb",
        action="store_true",
        help="Import metrics catalog to ext_knowledge table in LanceDB",
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
            arch_path, header_rows=3, min_term_length=args.min_term_length,
            sheet_name=args.arch_sheet_name
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

        # 传入数据架构Excel路径，用于中文描述到物理表名的映射
        arch_path = Path(args.arch_xlsx) if args.arch_xlsx else None
        business_terms = generator.generate_from_metrics_xlsx(
            metrics_path, business_terms, header_rows=2, min_term_length=args.min_term_length,
            sheet_name=args.metrics_sheet_name, arch_xlsx_path=arch_path
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

    # 导入指标到 ext_knowledge 表
    if args.import_to_lancedb and args.metrics_xlsx:
        logger.info("[主流程] 开始导入指标到 ext_knowledge 表...")
        metrics_catalog = generator.generate_metrics_catalog_from_xlsx(
            Path(args.metrics_xlsx), header_rows=2, sheet_name=args.metrics_sheet_name
        )
        if metrics_catalog:
            imported_count = generator.import_metrics_to_ext_knowledge(metrics_catalog)
            logger.info(f"[主流程] 指标导入完成: {imported_count} 个指标")
        else:
            logger.warning("[主流程] 没有生成指标目录，跳过导入")
    elif args.import_to_lancedb and not args.metrics_xlsx:
        logger.warning("[主流程] --import-to-lancedb 需要 --metrics-xlsx 参数")

    logger.info("Business configuration generation complete!")


if __name__ == "__main__":
    main()
