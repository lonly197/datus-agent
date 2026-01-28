#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ExtKnowledge importer for business configuration.
"""

from typing import Dict, List

import lancedb

from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ExtKnowledgeImporter:
    """ExtKnowledge表导入器"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def import_metrics(
        self,
        metrics_catalog: List[Dict],
        clear_existing: bool = False
    ) -> int:
        """
        将指标目录导入到LanceDB的ext_knowledge表

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
            ext_store = ExtKnowledgeStore(db_path=self.db_path)
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
                    ext_store = ExtKnowledgeStore(db_path=self.db_path)

            ext_store._ensure_table_ready()
            logger.info(f"[ExtKnowledge] 表schema: {ext_store.table.schema}")

            if clear_existing:
                logger.info("[ExtKnowledge] 清空现有Metrics分类数据...")

            logger.info(f"[ExtKnowledge] 开始导入 {len(metrics_catalog)} 个指标...")

            if metrics_catalog:
                first_entry = metrics_catalog[0]
                logger.info(f"[ExtKnowledge] 第一个条目字段: {list(first_entry.keys())}")

            # 分批导入
            batch_size = 50
            imported_count = 0

            for i in range(0, len(metrics_catalog), batch_size):
                batch = metrics_catalog[i:i + batch_size]
                try:
                    ext_store.batch_store_knowledge(batch)
                    imported_count += len(batch)
                    logger.info(f"[ExtKnowledge] 已导入 {imported_count}/{len(metrics_catalog)} 个指标")
                except Exception as batch_e:
                    logger.error(f"[ExtKnowledge] 批次 {i // batch_size + 1} 导入失败: {batch_e}")
                    for j, entry in enumerate(batch):
                        try:
                            ext_store.batch_store_knowledge([entry])
                            imported_count += 1
                        except Exception as entry_e:
                            logger.error(f"[ExtKnowledge] 条目 {i + j} 导入失败")
                            break
                    break

            logger.info(f"[ExtKnowledge] 成功导入 {imported_count} 个指标")
            return imported_count

        except Exception as e:
            logger.error(f"[ExtKnowledge] 导入失败: {e}")
            import traceback
            logger.error(f"[ExtKnowledge] 错误详情: {traceback.format_exc()}")
            return 0
