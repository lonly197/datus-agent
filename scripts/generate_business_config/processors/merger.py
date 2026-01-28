#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
DDL merger for business configuration.
"""

import json
from collections import defaultdict
from typing import Dict

from datus.utils.loggings import get_logger

from ..extractors import KeywordExtractor, TermExtractor

logger = get_logger(__name__)


class DdlMerger:
    """DDL注释合并器"""

    def __init__(self, schema_storage, min_term_length: int = 2):
        self.schema_storage = schema_storage
        self.keyword_extractor = KeywordExtractor(min_term_length)
        self.term_extractor = TermExtractor(min_term_length)

    def merge(self, business_terms: Dict) -> Dict:
        """
        将CSV提取的术语与LanceDB中的DDL comments进行合并

        合并策略：
        1. CSV中有但DDL没有的 -> 保留
        2. DDL中有但CSV没有的 -> 补充
        3. 两者都有 -> 以CSV为准，但添加DDL作为别名
        """
        logger.info("Merging with DDL comments from LanceDB...")

        term_to_table = defaultdict(set, business_terms.get("term_to_table", {}))
        term_to_schema = defaultdict(set, business_terms.get("term_to_schema", {}))

        ddl_stats = {"tables_checked": 0, "comments_found": 0, "terms_added": 0}

        try:
            self.schema_storage._ensure_table_ready()
            all_records = self.schema_storage.table.to_pandas()

            for _, row in all_records.iterrows():
                table_name = row.get("table_name", "")
                col_comments_json = row.get("column_comments", "{}")
                table_comment = row.get("table_comment", "")

                if not table_name:
                    continue

                ddl_stats["tables_checked"] += 1

                # 表注释作为关键词
                if table_comment and len(table_comment) > 2:
                    keywords = self.keyword_extractor.extract_from_comments(table_comment)
                    for kw in keywords:
                        if self._is_meaningful_comment(kw):
                            term_to_table[kw].add(table_name)

                # 字段注释映射
                try:
                    col_comments = json.loads(col_comments_json) if col_comments_json else {}
                except json.JSONDecodeError:
                    continue

                for col_name, comment in col_comments.items():
                    if not comment or len(comment) < 2:
                        continue

                    ddl_stats["comments_found"] += 1

                    if self._is_meaningful_comment(comment):
                        term_to_schema[comment].add(f"{table_name}.{col_name}")
                        term_to_schema[comment].add(col_name)
                        ddl_stats["terms_added"] += 1

                        keywords = self.keyword_extractor.extract_from_comments(comment)
                        for kw in keywords:
                            if self._is_meaningful_comment(kw):
                                term_to_schema[kw].add(col_name)

        except Exception as e:
            logger.warning(f"Failed to merge DDL comments: {e}")

        logger.info(
            f"[DDL合并] 完成: {ddl_stats['tables_checked']} 表检查, "
            f"{ddl_stats['comments_found']} 注释发现, "
            f"{ddl_stats['terms_added']} 术语添加"
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

    def _is_meaningful_comment(self, comment: str) -> bool:
        """判断注释是否有业务意义"""
        import re

        if not comment:
            return False

        technical_patterns = [
            r"^\d+$",
            r"^[a-z_]+$",
            r"^\d{4}-\d{2}-\d{2}",
            r"^(主键|外键|索引|unique|primary|key|idx|fk|pk)$",
        ]

        for pattern in technical_patterns:
            if re.match(pattern, comment.strip(), re.IGNORECASE):
                return False

        return len(comment) >= 4
