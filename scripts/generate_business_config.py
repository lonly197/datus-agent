#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Business Configuration Generator

结合数据架构设计文档和DDL元数据，生成/更新业务术语配置文件 business_terms.yml。
支持两种输入：
1. 数据架构详细设计CSV - 生成表/字段的业务术语映射
2. 指标清单CSV - 生成指标相关的业务术语（可选）

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

from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.cache import get_storage_cache_instance
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage
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


class BusinessConfigGenerator:
    """业务术语配置生成器"""

    def __init__(self, agent_config, namespace: str):
        self.agent_config = agent_config
        self.namespace = namespace
        self.db_path = agent_config.rag_storage_path()
        self.schema_storage = SchemaStorage(
            db_path=self.db_path,
            embedding_model=get_db_embedding_model()
        )
        logger.info(f"Initialized generator for namespace: {namespace}, db_path: {self.db_path}")

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

        # 读取前10行来确定实际的列名（处理多行表头）
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            # 跳过前5行的标题/说明行
            for _ in range(5):
                next(f, None)
            
            reader = csv.DictReader(f)
            
            # 打印实际的列名用于调试
            if reader.fieldnames:
                logger.debug(f"CSV columns: {reader.fieldnames}")
            
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

                # 验证表名和字段名 - 必须是以字母开头的有效标识符
                if not self._is_valid_table_name(table_name) or not self._is_valid_column_name(column_name):
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
            f"Architecture CSV processed: {stats['valid_rows']}/{stats['total_rows']} valid rows, "
            f"{len(stats['tables_found'])} tables, "
            f"{stats['terms_extracted']} terms extracted"
        )

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

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            # 跳过前7行的说明行
            for _ in range(7):
                next(f, None)
            
            reader = csv.DictReader(f)
            
            # 打印列名用于调试
            if reader.fieldnames:
                logger.debug(f"Metrics CSV columns: {reader.fieldnames}")
            
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
            f"Metrics CSV processed: {stats['valid_metrics']}/{stats['total_metrics']} valid metrics, "
            f"{len(stats['tables_added'])} source tables added"
        )

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

                # 1. 表注释作为关键词
                if table_comment and len(table_comment) > 2:
                    keywords = self._extract_keywords(table_comment, min_length=3)
                    for kw in keywords:
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

                        # 提取关键词
                        keywords = self._extract_keywords(comment, min_length=2)
                        for kw in keywords:
                            term_to_schema[kw].add(col_name)

        except Exception as e:
            logger.warning(f"Failed to merge DDL comments: {e}")

        logger.info(
            f"DDL merge complete: {ddl_stats['tables_checked']} tables checked, "
            f"{ddl_stats['comments_found']} comments found, "
            f"{ddl_stats['terms_added']} terms added"
        )

        return {
            "term_to_table": dict(term_to_table),
            "term_to_schema": dict(term_to_schema),
            "table_keywords": business_terms.get("table_keywords", {}),
            "_stats": {**business_terms.get("_stats", {}), **{
                "ddl_tables": ddl_stats["tables_checked"],
                "ddl_terms": ddl_stats["terms_added"],
            }},
        }

    def _extract_field(self, row: Dict, possible_names: List[str]) -> str:
        """从行中提取字段值，尝试多个可能的列名"""
        for name in possible_names:
            # 尝试精确匹配
            if name in row and row[name]:
                return self._clean_string(row[name])
            # 尝试大小写不敏感匹配
            for key in row.keys():
                if key and key.lower() == name.lower():
                    return self._clean_string(row[key])
        return ""

    def _is_valid_table_name(self, name: str) -> bool:
        """验证是否为有效的表名（以字母开头，只包含字母数字下划线）"""
        if not name or len(name) < 2:
            return False
        # 必须以字母开头
        if not re.match(r'^[a-zA-Z]', name):
            return False
        # 只能包含字母、数字、下划线
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return False
        # 排除常见技术关键词
        technical_keywords = {'engine', 'key', 'duplicate', 'distributed', 'random', 'min', 'max', 'properties'}
        if name.lower() in technical_keywords:
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

    def save_to_yaml(self, business_terms: Dict, output_path: Path):
        """保存业务术语配置到YAML文件"""
        # 转换set为sorted list，确保输出稳定
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

        # 打印统计
        stats = business_terms.get("_stats", {})
        print("\n" + "=" * 60)
        print("BUSINESS TERMS GENERATION REPORT")
        print("=" * 60)
        print(f"Output file: {output_path}")
        print(f"Table terms: {len(output['term_to_table'])}")
        print(f"Schema terms: {len(output['term_to_schema'])}")
        print(f"Table keywords: {len(output['table_keywords'])}")
        if stats:
            print("\nGeneration stats:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate business terms configuration from design documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from architecture CSV only
  python scripts/generate_business_config.py \\
      --config=conf/agent.yml \\
      --namespace=test \\
      --arch-csv=/path/to/数据架构详细设计v2.3.csv

  # Generate from both CSV files
  python scripts/generate_business_config.py \\
      --config=conf/agent.yml \\
      --namespace=test \\
      --arch-csv=/path/to/数据架构详细设计v2.3.csv \\
      --metrics-csv=/path/to/指标清单v2.4.csv \\
      --output=conf/business_terms.yml

  # Merge with existing DDL comments
  python scripts/generate_business_config.py \\
      --config=conf/agent.yml \\
      --namespace=test \\
      --arch-csv=/path/to/数据架构详细设计v2.3.csv \\
      --merge-ddl
        """,
    )

    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", required=True, help="Target namespace")
    parser.add_argument("--arch-csv", help="Path to 数据架构详细设计 CSV file")
    parser.add_argument("--metrics-csv", help="Path to 指标清单 CSV file (optional)")
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

    args = parser.parse_args()

    # 初始化日志
    configure_logging()

    # 加载配置
    logger.info(f"Loading agent config from: {args.config}")
    agent_config = load_agent_config(config=args.config)
    agent_config.current_namespace = args.namespace

    # 初始化生成器
    generator = BusinessConfigGenerator(agent_config, args.namespace)

    # 生成业务术语
    business_terms = None

    if args.arch_csv:
        arch_path = Path(args.arch_csv)
        if not arch_path.exists():
            logger.error(f"Architecture CSV not found: {arch_path}")
            sys.exit(1)

        business_terms = generator.generate_from_architecture_csv(
            arch_path, min_term_length=args.min_term_length
        )

    if args.metrics_csv:
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
        logger.error("No input CSV files provided. Use --arch-csv or --metrics-csv")
        sys.exit(1)

    # 合并DDL注释
    if args.merge_ddl:
        business_terms = generator.merge_with_ddl_comments(business_terms)

    # 保存输出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_to_yaml(business_terms, output_path)

    logger.info("Business configuration generation complete!")


if __name__ == "__main__":
    main()
