#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema Enrichment from Design Document

结合数据架构设计文档（Excel）和 LanceDB 中的 Schema 元数据，
通过设计文档提供的丰富业务信息，补全和增强 Schema 元数据。

核心功能：
1. 表注释增强：从"逻辑实体业务含义"补充 table_comment
2. 列注释增强：从"属性业务定义"补充 column_comments
3. 业务标签提取：从"主题域分组"、"分析对象"提取 business_tags
4. 枚举值推断：基于"数据安全分类"和字段名模式推断枚举值
5. 关系元数据增强：基于逻辑实体关联推断表关系

使用示例：
    python scripts/enrich_schema_from_design.py \
        --config=/root/.datus/conf/agent.yml \
        --namespace=test \
        --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \
        --dry-run

    python scripts/enrich_schema_from_design.py \
        --config=/root/.datus/conf/agent.yml \
        --namespace=test \
        --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \
        --apply
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Install with: pip install pandas openpyxl")

try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)


@dataclass
class DesignDocRecord:
    """数据架构设计文档单条记录"""
    # 资产目录
    l1_theme: str = ""  # L1-主题域分组
    l2_theme: str = ""  # L2-主题域
    l3_theme: str = ""  # L3/L4-分析主题
    
    # 业务属性
    analysis_obj_code: str = ""  # 分析对象编码
    analysis_obj_cn: str = ""    # 分析对象（中文）
    analysis_obj_en: str = ""    # 分析对象（英文）
    logic_entity_cn: str = ""    # 逻辑实体（中文）
    logic_entity_en: str = ""    # 逻辑实体（英文）
    logic_entity_def: str = ""   # 逻辑实体业务含义
    attr_cn: str = ""            # 属性（中文）
    attr_en: str = ""            # 属性（英文）
    attr_def: str = ""           # 属性业务定义
    
    # 技术属性
    table_name: str = ""         # 物理表名
    column_name: str = ""        # 字段名
    
    # 管理属性
    data_owner: str = ""         # 数据Owner
    security_class: str = ""     # 安全分类
    security_level: str = ""     # 安全等级


@dataclass
class EnrichmentResult:
    """单个表的 enrichment 结果"""
    table_name: str
    original_comment: str = ""
    enriched_comment: str = ""
    enriched_columns: Dict[str, str] = field(default_factory=dict)
    enriched_tags: List[str] = field(default_factory=list)
    enriched_enums: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    
    @property
    def has_changes(self) -> bool:
        return bool(
            self.enriched_comment or 
            self.enriched_columns or 
            self.enriched_tags or 
            self.enriched_enums
        )


class SchemaEnricher:
    """
    Schema 元数据增强器
    
    通过数据架构设计文档的业务信息，补全 LanceDB 中的 Schema 元数据。
    """
    
    def __init__(self, agent_config, namespace: str):
        self.agent_config = agent_config
        self.namespace = namespace
        self.db_path = agent_config.rag_storage_path()
        self.schema_storage = SchemaStorage(
            db_path=self.db_path,
            embedding_model=get_db_embedding_model()
        )
        
        # 设计文档数据索引
        self._table_to_records: Dict[str, List[DesignDocRecord]] = defaultdict(list)
        self._column_to_records: Dict[str, List[DesignDocRecord]] = defaultdict(list)
        self._entity_to_tables: Dict[str, List[str]] = defaultdict(list)
        
        # LanceDB 中的实际表名集合（用于匹配）
        self._lancedb_tables: Set[str] = set()
        
        logger.info(f"Initialized SchemaEnricher for namespace: {namespace}")
        logger.info(f"Database path: {self.db_path}")

    def _load_lancedb_tables(self):
        """加载 LanceDB 中的所有表名"""
        try:
            self.schema_storage._ensure_table_ready()
            all_data = self.schema_storage.table.to_pandas()
            self._lancedb_tables = set(all_data['table_name'].str.lower().tolist())
            logger.info(f"Loaded {len(self._lancedb_tables)} tables from LanceDB")
            
            # 显示样本
            sample = list(self._lancedb_tables)[:10]
            logger.info(f"Sample LanceDB tables: {sample}")
            
            # 统计表名前缀分布
            prefix_counts = {}
            for t in self._lancedb_tables:
                prefix = t.split('_')[0] if '_' in t else 'other'
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            logger.info(f"Table prefix distribution: {dict(sorted(prefix_counts.items(), key=lambda x: -x[1])[:5])}")
            
        except Exception as e:
            logger.error(f"Failed to load LanceDB tables: {e}")

    def load_design_document(self, xlsx_path: Path, sheet_name=None) -> int:
        """
        加载数据架构设计文档 Excel
        
        Args:
            xlsx_path: Excel 文件路径
            sheet_name: 工作表名称或索引
            
        Returns:
            加载的有效记录数
        """
        if not PANDAS_AVAILABLE:
            logger.error("pandas is required for Excel support")
            return 0
        
        try:
            # 自动检测工作表
            if sheet_name is None:
                xl = pd.ExcelFile(xlsx_path)
                sheet_names = xl.sheet_names
                if "Sheet1" in sheet_names:
                    sheet_name = "Sheet1"
                    logger.info(f"Auto-selected sheet: 'Sheet1'")
                else:
                    sheet_name = 0
                    logger.info(f"Auto-selected first sheet: '{sheet_names[0]}'")
            
            # 数据架构设计文档结构：
            # 第1行：分组标题（资产目录/业务属性/技术属性/管理属性）
            # 第2行：实际列名（物理表名、字段名等）★ 使用这行
            # 第3行：列说明/注释
            logger.info(f"Reading Excel with header=1 (row 2 as column names)")
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=1)
            
            # 清理列名
            df.columns = [str(c).strip().replace('\n', ' ').replace('\r', '') for c in df.columns]
            logger.info(f"Excel columns: {df.columns.tolist()[:10]}...")
            
            # 过滤掉表头说明行（物理表名为空或是说明文字的行）
            df = self._filter_header_rows(df)
            
            logger.info(f"Loaded {len(df)} data rows from design document")
            
            valid_count = 0
            invalid_table_names = []
            
            for _, row in df.iterrows():
                record = self._parse_row(row)
                if record and record.table_name and record.column_name:
                    # 标准化表名（小写）
                    table_name_lower = record.table_name.lower()
                    
                    self._table_to_records[table_name_lower].append(record)
                    col_key = f"{table_name_lower}.{record.column_name.lower()}"
                    self._column_to_records[col_key].append(record)
                    
                    # 建立逻辑实体到表的映射
                    if record.logic_entity_cn:
                        entity_lower = record.logic_entity_cn.lower()
                        if table_name_lower not in self._entity_to_tables[entity_lower]:
                            self._entity_to_tables[entity_lower].append(table_name_lower)
                    
                    valid_count += 1
                elif record and record.table_name:
                    # 记录无效表名用于调试
                    invalid_table_names.append(record.table_name)
            
            # 去重并限制调试输出
            invalid_table_names = list(dict.fromkeys(invalid_table_names))[:20]
            
            logger.info(f"Parsed {valid_count} valid records with both table and column names")
            logger.info(f"Indexed {len(self._table_to_records)} unique tables")
            logger.info(f"Indexed {len(self._entity_to_tables)} logic entities")
            
            if invalid_table_names:
                logger.warning(f"Sample invalid/skipped table names: {invalid_table_names[:10]}")
            
            # 显示样本表名
            sample_tables = list(self._table_to_records.keys())[:10]
            logger.info(f"Sample indexed tables: {sample_tables}")
            
            # 对比 LanceDB 表名
            common_tables = set(self._table_to_records.keys()) & self._lancedb_tables
            logger.info(f"Matched tables with LanceDB: {len(common_tables)}/{len(self._lancedb_tables)}")
            if common_tables:
                logger.info(f"Sample matched: {list(common_tables)[:5]}")
            
            # 显示未匹配的样本
            unmatched_lancedb = list(self._lancedb_tables - set(self._table_to_records.keys()))[:10]
            if unmatched_lancedb:
                logger.warning(f"Sample unmatched LanceDB tables: {unmatched_lancedb}")
            
            unmatched_excel = list(set(self._table_to_records.keys()) - self._lancedb_tables)[:10]
            if unmatched_excel:
                logger.warning(f"Sample unmatched Excel tables: {unmatched_excel}")
            
            return valid_count
            
        except Exception as e:
            logger.error(f"Failed to load design document: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return 0
    
    def _filter_header_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤掉表头说明行"""
        original_len = len(df)
        
        # 条件1：过滤掉 #N/A 值
        df = df.replace('#N/A', pd.NA).replace('N/A', pd.NA)
        
        # 条件2：过滤掉说明行（多种可能的说明文字）
        header_keywords = [
            '定义分析业务对象',
            '定义逻辑实体',
            '逻辑实体的业务含义',
            '定义物理表名',
            '定义属性',
            '分析业务对象（中文）名称',
            '分析业务对象（英文）名称',
            '定义逻辑实体中文名',
            '定义逻辑实体英文名',
        ]
        
        # 检查所有列是否包含说明关键词
        for col in df.columns:
            for keyword in header_keywords:
                df = df[~df[col].astype(str).str.contains(keyword, na=False, regex=False)]
        
        # 条件3：逻辑实体（英文）列必须是有效的表名格式
        # 优先使用逻辑实体（英文）列，因为这是实际的物理表名
        logic_entity_col = None
        for possible in ["逻辑实体（英文）", "逻辑实体英文"]:
            if possible in df.columns:
                logic_entity_col = possible
                break
        
        table_col = None
        for possible in ["物理表名", "表名"]:
            if possible in df.columns:
                table_col = possible
                break
        
        # 优先使用逻辑实体（英文）列
        if logic_entity_col:
            df = df[df[logic_entity_col].notna()]
            df = df[df[logic_entity_col].astype(str).str.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', na=False)]
            df = df[~df[logic_entity_col].astype(str).str.contains(r'^定义|^说明', regex=True, na=False)]
        elif table_col:
            df = df[df[table_col].notna()]
            df = df[df[table_col].astype(str).str.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', na=False)]
        
        filtered_len = len(df)
        logger.info(f"Filtered {original_len - filtered_len} header/invalid rows, {filtered_len} data rows remaining")
        
        return df
    
    def _parse_row(self, row: pd.Series) -> Optional[DesignDocRecord]:
        """解析单行数据为 DesignDocRecord"""
        try:
            record = DesignDocRecord()
            
            # 辅助函数：安全获取字段值（尝试多种可能的列名）
            def get_field(possible_names: List[str]) -> str:
                for name in possible_names:
                    # 精确匹配
                    if name in row.index:
                        val = row[name]
                        if pd.notna(val):
                            return str(val).strip()
                    # 模糊匹配（处理带换行符的列名）
                    for col in row.index:
                        if name in col and pd.notna(row[col]):
                            return str(row[col]).strip()
                return ""
            
            # 资产目录
            record.l1_theme = get_field(["L1-主题域分组", "L1主题域分组", "L1"])
            record.l2_theme = get_field(["L2-主题域", "L2主题域", "L2"])
            record.l3_theme = get_field(["L3/L4-分析主题", "L3分析主题", "L3", "L3/L4"])
            
            # 业务属性
            record.analysis_obj_code = get_field(["分析对象编码"])
            record.analysis_obj_cn = get_field(["分析对象（中文）", "分析对象"])
            record.analysis_obj_en = get_field(["分析对象（英文）", "分析对象英文"])
            record.logic_entity_cn = get_field(["逻辑实体（中文）", "逻辑实体"])
            record.logic_entity_en = get_field(["逻辑实体（英文）", "逻辑实体英文"])
            record.logic_entity_def = get_field(["逻辑实体业务含义", "逻辑实体含义", "逻辑实体说明"])
            record.attr_cn = get_field(["属性（中文）", "属性"])
            record.attr_en = get_field(["属性（英文）", "属性英文"])
            record.attr_def = get_field(["*属性业务定义", "属性业务定义", "业务定义", "属性定义"])
            
            # 技术属性
            # 关键：逻辑实体（英文）列实际上是物理表名！
            record.logic_entity_en = get_field(["逻辑实体（英文）", "逻辑实体英文"])
            record.table_name = get_field(["物理表名", "table_name", "表名"])
            record.column_name = get_field(["字段名", "column_name", "列名", "属性英文名"])
            
            # 优先使用逻辑实体（英文）作为表名（这才是实际的物理表名）
            if record.logic_entity_en and not record.table_name:
                record.table_name = record.logic_entity_en
            elif record.logic_entity_en and record.table_name:
                # 两者都有，优先使用逻辑实体（英文）
                if record.logic_entity_en != record.table_name:
                    logger.debug(f"Table name mismatch: logic_entity_en={record.logic_entity_en}, table_name={record.table_name}")
                record.table_name = record.logic_entity_en
            
            # 管理属性
            record.data_owner = get_field(["数据Owner", "数据 Owner"])
            record.security_class = get_field(["安全分类"])
            record.security_level = get_field(["安全等级"])
            
            # 清理表名和字段名
            if record.table_name:
                record.table_name = record.table_name.strip().lower()
                # 过滤掉说明行
                if self._is_header_text(record.table_name):
                    record.table_name = ""
            
            if record.column_name:
                record.column_name = record.column_name.strip().lower()
                # 过滤掉说明行
                if self._is_header_text(record.column_name):
                    record.column_name = ""
            
            return record if record.table_name else None
            
        except Exception as e:
            logger.debug(f"Failed to parse row: {e}")
            return None
    
    def _is_header_text(self, text: str) -> bool:
        """判断是否为表头说明文字"""
        if not text:
            return True
        header_keywords = [
            '定义物理表名', '定义逻辑实体', '定义属性', '定义字段',
            '定义分析业务对象', '定义逻辑实体中文名', '定义逻辑实体英文名',
            '逻辑实体的业务含义', '分析业务对象（中文）名称', '分析业务对象（英文）名称',
            '物理表名', '逻辑实体', '属性业务定义',
            'table_name', 'column_name', 'field_name',
            '#N/A', 'N/A',
        ]
        return any(keyword in text for keyword in header_keywords) or len(text) < 2
    
    def enrich_table_metadata(self, table_name: str, existing_record: Dict) -> EnrichmentResult:
        """
        增强单个表的元数据
        
        Args:
            table_name: 表名
            existing_record: LanceDB 中现有记录
            
        Returns:
            EnrichmentResult 包含所有增强信息
        """
        result = EnrichmentResult(table_name=table_name)
        
        # 获取原始注释
        result.original_comment = existing_record.get("table_comment", "") or ""
        
        # 标准化表名进行匹配
        table_name_lower = table_name.lower()
        
        # 查找设计文档中的记录
        design_records = self._table_to_records.get(table_name_lower, [])
        
        if design_records:
            result.match_source = "exact"
        else:
            # 尝试模糊匹配
            design_records = self._fuzzy_match_table(table_name_lower)
            if design_records:
                result.match_source = "fuzzy"
                logger.debug(f"Fuzzy matched table '{table_name}' with {len(design_records)} records")
        
        if not design_records:
            return result
        
        # 1. 增强表注释（从逻辑实体业务含义）
        result.enriched_comment = self._enrich_table_comment(design_records, result.original_comment)
        
        # 2. 增强列注释
        existing_col_comments = self._parse_json_column_comments(
            existing_record.get("column_comments", "{}"))
        result.enriched_columns = self._enrich_column_comments(
            table_name_lower, design_records, existing_col_comments)
        
        # 3. 增强业务标签
        existing_tags = existing_record.get("business_tags", []) or []
        if isinstance(existing_tags, str):
            try:
                existing_tags = json.loads(existing_tags)
            except:
                existing_tags = []
        result.enriched_tags = self._enrich_business_tags(design_records, existing_tags)
        
        # 4. 推断枚举值
        existing_enums = self._parse_json_column_enums(
            existing_record.get("column_enums", "{}"))
        result.enriched_enums = self._infer_enum_values(
            table_name_lower, design_records, result.enriched_columns, existing_enums)
        
        return result
    
    def _fuzzy_match_table(self, table_name: str) -> List[DesignDocRecord]:
        """基于逻辑实体名称模糊匹配表"""
        matched_records = []
        
        # 提取表名关键词（去除前缀如 ods_, dwd_ 等）
        clean_name = re.sub(r'^(ods|dwd|dws|dim|ads|tmp|stg)_', '', table_name)
        clean_name = re.sub(r'_(di|df|tmp|temp|bak|\d+)$', '', clean_name)
        
        # 在逻辑实体中查找匹配
        for entity, tables in self._entity_to_tables.items():
            entity_clean = entity.lower().replace('事实表', '').replace('明细', '').replace('汇总', '')
            # 计算相似度
            if clean_name in entity_clean or entity_clean in clean_name:
                # 获取该实体的所有记录
                for t in tables:
                    matched_records.extend(self._table_to_records.get(t, []))
                continue
            
            # 尝试计算关键词相似度
            if self._calculate_similarity(clean_name, entity_clean) > 0.5:
                for t in tables:
                    matched_records.extend(self._table_to_records.get(t, []))
        
        # 去重
        seen = set()
        unique_records = []
        for r in matched_records:
            key = (r.table_name, r.column_name)
            if key not in seen:
                seen.add(key)
                unique_records.append(r)
        
        return unique_records
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """计算两个字符串的相似度（简单版本）"""
        # 提取关键词
        keywords1 = set(re.findall(r'[a-z]+', s1.lower()))
        keywords2 = set(re.findall(r'[a-z]+', s2.lower()))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _enrich_table_comment(self, records: List[DesignDocRecord], existing: str) -> str:
        """增强表注释"""
        # 优先使用逻辑实体业务含义
        entity_defs = [r.logic_entity_def for r in records if r.logic_entity_def]
        if entity_defs:
            # 去重并合并
            unique_defs = list(dict.fromkeys(entity_defs))
            combined = "；".join(unique_defs[:3])  # 最多取3个
            if existing and combined not in existing:
                return f"{existing} ({combined})"
            return combined
        
        # 其次使用逻辑实体名称
        entity_names = [r.logic_entity_cn for r in records if r.logic_entity_cn]
        if entity_names:
            return entity_names[0]
        
        return existing
    
    def _enrich_column_comments(
        self, 
        table_name: str,
        records: List[DesignDocRecord], 
        existing: Dict[str, str]
    ) -> Dict[str, str]:
        """增强列注释"""
        enriched = {}
        
        for record in records:
            if not record.column_name:
                continue
            
            col = record.column_name.lower()
            
            # 优先使用属性业务定义
            if record.attr_def:
                # 如果现有注释较短或为空，使用业务定义
                existing_comment = existing.get(col, "")
                if not existing_comment or len(existing_comment) < len(record.attr_def):
                    enriched[col] = record.attr_def
            
            # 其次使用属性中文名
            elif record.attr_cn:
                enriched[col] = record.attr_cn
        
        return enriched
    
    def _enrich_business_tags(
        self, 
        records: List[DesignDocRecord], 
        existing: List[str]
    ) -> List[str]:
        """增强业务标签"""
        tags = set(existing) if existing else set()
        
        for record in records:
            # 从主题域提取标签
            if record.l1_theme:
                tags.add(self._normalize_tag(record.l1_theme))
            if record.l2_theme:
                tags.add(self._normalize_tag(record.l2_theme))
            if record.l3_theme:
                tags.add(self._normalize_tag(record.l3_theme))
            
            # 从分析对象提取标签
            if record.analysis_obj_cn:
                tags.add(self._normalize_tag(record.analysis_obj_cn))
            
            # 从逻辑实体提取标签
            if record.logic_entity_cn:
                # 提取关键词
                keywords = self._extract_keywords_from_entity(record.logic_entity_cn)
                tags.update(keywords)
        
        return sorted(list(tags))
    
    def _normalize_tag(self, tag: str) -> str:
        """标准化标签"""
        tag = tag.lower().strip()
        # 替换特殊字符
        tag = re.sub(r'[\s\-]+', '_', tag)
        # 去除多余下划线
        tag = re.sub(r'_+', '_', tag)
        return tag.strip('_')
    
    def _infer_enum_values(
        self,
        table_name: str,
        records: List[DesignDocRecord],
        col_comments: Dict[str, str],
        existing: Dict[str, List[Dict[str, str]]]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        基于安全分类和字段名模式推断枚举值
        """
        enums = dict(existing) if existing else {}
        
        # 常见枚举字段模式
        enum_patterns = {
            r'.*_type$': [
                {"value": "1", "label": "类型1"},
                {"value": "2", "label": "类型2"},
            ],
            r'.*_status$': [
                {"value": "0", "label": "无效/禁用"},
                {"value": "1", "label": "有效/启用"},
            ],
            r'.*_flag$': [
                {"value": "0", "label": "否"},
                {"value": "1", "label": "是"},
            ],
            r'^is_.*': [
                {"value": "0", "label": "否"},
                {"value": "1", "label": "是"},
            ],
            r'gender|sex': [
                {"value": "0", "label": "未知"},
                {"value": "1", "label": "男"},
                {"value": "2", "label": "女"},
            ],
        }
        
        for record in records:
            if not record.column_name:
                continue
            
            col = record.column_name.lower()
            
            # 跳过已有枚举值的字段
            if col in enums and enums[col]:
                continue
            
            # 基于字段名模式匹配
            for pattern, default_enums in enum_patterns.items():
                if re.match(pattern, col, re.IGNORECASE):
                    # 检查注释是否支持枚举推断
                    comment = col_comments.get(col, "")
                    if self._is_likely_enum_column(col, comment):
                        enums[col] = default_enums
                        logger.debug(f"Inferred enum for {col}: {default_enums}")
                    break
        
        return enums
    
    def _is_likely_enum_column(self, col_name: str, comment: str) -> bool:
        """判断字段是否可能是枚举类型"""
        # 检查字段名后缀
        enum_suffixes = ['_type', '_status', '_flag', '_code', '_level', '_category']
        if any(col_name.endswith(suffix) for suffix in enum_suffixes):
            return True
        
        # 检查注释中的关键词
        if comment:
            enum_keywords = ['类型', '状态', '标识', '级别', '分类', '是否']
            if any(kw in comment for kw in enum_keywords):
                return True
        
        return False
    
    def _extract_keywords_from_entity(self, entity_name: str) -> Set[str]:
        """从逻辑实体名称提取关键词"""
        keywords = set()
        
        # 常见业务关键词映射
        keyword_mapping = {
            '客户': 'customer',
            '线索': 'clue',
            '订单': 'order',
            '试驾': 'testdrive',
            '门店': 'dealer',
            '渠道': 'channel',
            '车型': 'vehicle',
            '事实表': 'fact',
            '明细': 'detail',
            '汇总': 'summary',
        }
        
        for cn, en in keyword_mapping.items():
            if cn in entity_name:
                keywords.add(en)
        
        return keywords
    
    def _parse_json_column_comments(self, json_str: str) -> Dict[str, str]:
        """解析 JSON 格式的列注释"""
        try:
            return json.loads(json_str) if json_str else {}
        except json.JSONDecodeError:
            return {}
    
    def _parse_json_column_enums(self, json_str: str) -> Dict[str, List[Dict[str, str]]]:
        """解析 JSON 格式的列枚举值"""
        try:
            return json.loads(json_str) if json_str else {}
        except json.JSONDecodeError:
            return {}
    
    def process_all_tables(self, dry_run: bool = True) -> Tuple[int, int, Dict]:
        """
        处理所有表
        
        Returns:
            (处理的表数, 有变更的表数, 统计信息)
        """
        self.schema_storage._ensure_table_ready()
        
        # 获取所有现有记录
        all_records = self.schema_storage.table.to_pandas()
        total = len(all_records)
        
        logger.info(f"Processing {total} tables from LanceDB")
        
        enriched_count = 0
        changes_summary = []
        
        # 统计匹配来源
        match_stats = {"exact": 0, "fuzzy": 0, "none": 0}
        
        for idx, row in all_records.iterrows():
            table_name = row.get("table_name", "")
            if not table_name:
                continue
            
            # 转换为字典
            existing_record = row.to_dict()
            
            # 执行增强
            result = self.enrich_table_metadata(table_name, existing_record)
            
            # 统计匹配来源
            if result.match_source == "exact":
                match_stats["exact"] += 1
            elif result.match_source == "fuzzy":
                match_stats["fuzzy"] += 1
            else:
                match_stats["none"] += 1
            
            if result.has_changes:
                enriched_count += 1
                changes_summary.append({
                    "table": table_name,
                    "match_source": result.match_source,
                    "comment_changed": bool(result.enriched_comment and result.enriched_comment != result.original_comment),
                    "columns_enriched": len(result.enriched_columns),
                    "tags_added": len(result.enriched_tags),
                    "enums_inferred": len(result.enriched_enums),
                })
                
                logger.info(f"[{idx+1}/{total}] {table_name} ({result.match_source}): "
                          f"comment={bool(result.enriched_comment)}, "
                          f"columns={len(result.enriched_columns)}, "
                          f"tags={len(result.enriched_tags)}")
            else:
                logger.debug(f"[{idx+1}/{total}] {table_name}: no changes")
        
        logger.info(f"Match statistics: exact={match_stats['exact']}, fuzzy={match_stats['fuzzy']}, none={match_stats['none']}")
        
        stats = {
            "total_tables": total,
            "enriched_tables": enriched_count,
            "match_rate": f"{enriched_count/total*100:.1f}%" if total > 0 else "0%",
            "match_stats": match_stats,
            "changes_detail": changes_summary[:20],  # 前20个详情
        }
        
        return total, enriched_count, stats
    
    def _apply_enrichment(self, result: EnrichmentResult):
        """将增强结果应用到 LanceDB"""
        try:
            # 获取现有记录
            escaped_identifier = result.table_name.replace('"', '\\"')
            existing = self.schema_storage.table.search()\
                .where(f'table_name = "{escaped_identifier}"')\
                .limit(1).to_arrow()
            
            if len(existing) == 0:
                logger.warning(f"Table {result.table_name} not found in LanceDB")
                return
            
            record = existing.to_pylist()[0]
            
            # 构建更新数据
            update_data = {
                "identifier": record["identifier"],
                "catalog_name": record.get("catalog_name", ""),
                "database_name": record.get("database_name", ""),
                "schema_name": record.get("schema_name", ""),
                "table_name": result.table_name,
                "table_type": record.get("table_type", ""),
                "definition": record.get("definition", ""),
            }
            
            # 应用增强
            if result.enriched_comment:
                update_data["table_comment"] = result.enriched_comment
            else:
                update_data["table_comment"] = record.get("table_comment", "")
            
            # 合并列注释
            existing_col_comments = self._parse_json_column_comments(
                record.get("column_comments", "{}"))
            merged_col_comments = {**existing_col_comments, **result.enriched_columns}
            update_data["column_comments"] = json.dumps(merged_col_comments, ensure_ascii=False)
            
            # 业务标签
            update_data["business_tags"] = result.enriched_tags
            
            # 合并枚举值
            existing_enums = self._parse_json_column_enums(
                record.get("column_enums", "{}"))
            merged_enums = {**existing_enums, **result.enriched_enums}
            update_data["column_enums"] = json.dumps(merged_enums, ensure_ascii=False)
            
            # 保留原有字段
            for field in ["row_count", "sample_statistics", "relationship_metadata", 
                         "metadata_version", "last_updated"]:
                if field in record:
                    update_data[field] = record[field]
            
            # 更新 metadata_version 标记为已增强
            update_data["metadata_version"] = 1
            update_data["last_updated"] = int(pd.Timestamp.now().timestamp())
            
            # 删除旧记录并插入新记录
            escaped_id = record["identifier"].replace('"', '\\"')
            self.schema_storage.table.delete(f'identifier = "{escaped_id}"')
            self.schema_storage.table.add([update_data])
            
            logger.debug(f"Applied enrichment to {result.table_name}")
            
        except Exception as e:
            logger.error(f"Failed to apply enrichment to {result.table_name}: {e}")
    
    def generate_report(self, stats: Dict, output_path: Optional[Path] = None):
        """生成增强报告"""
        report = []
        report.append("=" * 80)
        report.append("SCHEMA ENRICHMENT REPORT")
        report.append("=" * 80)
        report.append(f"Namespace: {self.namespace}")
        report.append(f"Database: {self.db_path}")
        report.append("")
        report.append(f"Total tables processed: {stats['total_tables']}")
        report.append(f"Tables enriched: {stats['enriched_tables']}")
        report.append(f"Match rate: {stats['match_rate']}")
        
        if 'match_stats' in stats:
            report.append("")
            report.append("Match source breakdown:")
            report.append(f"  Exact match: {stats['match_stats']['exact']} tables")
            report.append(f"  Fuzzy match: {stats['match_stats']['fuzzy']} tables")
            report.append(f"  No match: {stats['match_stats']['none']} tables")
        
        report.append("")
        
        if stats['changes_detail']:
            report.append("Sample changes:")
            for change in stats['changes_detail'][:10]:
                report.append(f"  - {change['table']}: "
                            f"comment={change['comment_changed']}, "
                            f"columns={change['columns_enriched']}, "
                            f"tags={change['tags_added']}, "
                            f"enums={change['enums_inferred']}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if output_path:
            output_path.write_text(report_text, encoding="utf-8")
            logger.info(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich LanceDB schema metadata from design documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes (dry run)
  python scripts/enrich_schema_from_design.py \\
      --config=/root/.datus/conf/agent.yml \\
      --namespace=test \\
      --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \\
      --dry-run \\
      --output=/tmp/enrichment_preview.txt

  # Apply enrichment
  python scripts/enrich_schema_from_design.py \\
      --config=/root/.datus/conf/agent.yml \\
      --namespace=test \\
      --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \\
      --apply

  # Save report to file
  python scripts/enrich_schema_from_design.py \\
      --config=/root/.datus/conf/agent.yml \\
      --namespace=test \\
      --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \\
      --apply \\
      --output=/tmp/enrichment_report.txt
        """,
    )
    
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", required=True, help="Target namespace")
    parser.add_argument("--arch-xlsx", required=True, help="Path to 数据架构详细设计 Excel file")
    parser.add_argument("--arch-sheet-name", default=None, help="Sheet name or index (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply enrichment to LanceDB")
    parser.add_argument("--output", help="Path to save report")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        parser.error("Please specify --dry-run or --apply")
    
    if args.dry_run and args.apply:
        parser.error("Cannot specify both --dry-run and --apply")
    
    # 初始化日志
    configure_logging(debug=args.verbose)
    
    # 加载配置
    logger.info(f"Loading agent config from: {args.config}")
    agent_config = load_agent_config(config=args.config)
    agent_config.current_namespace = args.namespace
    
    # 初始化 enricher
    enricher = SchemaEnricher(agent_config, args.namespace)
    
    # 先加载 LanceDB 表名（用于匹配验证）
    enricher._load_lancedb_tables()
    
    # 加载设计文档
    arch_path = Path(args.arch_xlsx)
    if not arch_path.exists():
        logger.error(f"Design document not found: {arch_path}")
        sys.exit(1)
    
    record_count = enricher.load_design_document(arch_path, args.arch_sheet_name)
    if record_count == 0:
        logger.error("No valid records found in design document")
        sys.exit(1)
    
    # 处理所有表
    total, enriched, stats = enricher.process_all_tables(dry_run=args.dry_run)
    
    # 生成报告
    output_path = Path(args.output) if args.output else None
    enricher.generate_report(stats, output_path)
    
    # 输出结果
    if args.dry_run:
        logger.info("Dry run completed. Use --apply to apply changes.")
    else:
        logger.info(f"Enrichment applied: {enriched}/{total} tables updated")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
