# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Performance optimization utilities."""

import hashlib
import json
import re
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from .types import ErrorType

from datus.utils.loggings import get_logger


class QueryCache:
    """Intelligent query result caching system."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _get_cache_key(self, tool_name: str, **kwargs) -> str:
        """Generate a deterministic cache key for the query."""
        # Normalize kwargs by sorting keys and excluding non-deterministic parameters
        cache_params = {k: v for k, v in kwargs.items() if k not in ["todo_id", "call_id"]}
        sorted_params = json.dumps(cache_params, sort_keys=True)
        key_content = f"{tool_name}:{sorted_params}"
        return hashlib.md5(key_content.encode()).hexdigest()

    def get(self, tool_name: str, **kwargs) -> Optional[Any]:
        """Retrieve cached result if available and not expired."""
        cache_key = self._get_cache_key(tool_name, **kwargs)

        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                logger.debug(f"Cache hit for {tool_name} with key {cache_key}")
                return entry["result"]

            # Remove expired entry
            del self.cache[cache_key]

        return None

    def set(self, tool_name: str, result: Any, **kwargs) -> None:
        """Cache the result."""
        cache_key = self._get_cache_key(tool_name, **kwargs)

        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

        self.cache[cache_key] = {"result": result, "timestamp": time.time()}

        logger.debug(f"Cached result for {tool_name} with key {cache_key}")

    def get_enhanced(self, tool_name: str, **kwargs) -> Optional[Any]:
        """Enhanced get method with tool-specific cache key generation."""
        # Generate tool-specific cache key
        cache_key = self._get_enhanced_cache_key(tool_name, **kwargs)

        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry["timestamp"] < self._get_tool_ttl(tool_name):
                logger.debug(f"Enhanced cache hit for {tool_name} with key {cache_key}")
                return entry["result"]

            # Remove expired entry
            del self.cache[cache_key]

        return None

    def set_enhanced(self, tool_name: str, result: Any, **kwargs) -> None:
        """Enhanced set method with tool-specific caching."""
        cache_key = self._get_enhanced_cache_key(tool_name, **kwargs)

        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

        self.cache[cache_key] = {"result": result, "timestamp": time.time()}

        logger.debug(f"Enhanced cached result for {tool_name} with key {cache_key}")

    def _get_enhanced_cache_key(self, tool_name: str, **kwargs) -> str:
        """Generate enhanced cache key with tool-specific logic."""
        if tool_name == "analyze_query_plan":
            # For query plan analysis, include normalized SQL hash
            sql_query = kwargs.get("sql_query", "")
            normalized_sql = self._normalize_sql_for_cache(sql_query)
            cache_params = {
                "tool": tool_name,
                "sql_hash": hashlib.md5(normalized_sql.encode()).hexdigest()[:16],
                "catalog": kwargs.get("catalog", ""),
                "database": kwargs.get("database", ""),
                "schema": kwargs.get("schema", ""),
            }
            sorted_params = json.dumps(cache_params, sort_keys=True)
            return hashlib.md5(sorted_params.encode()).hexdigest()

        elif tool_name in ["check_table_conflicts", "validate_partitioning"]:
            # For table-specific tools, include table name and metadata
            cache_params = {
                "tool": tool_name,
                "table_name": kwargs.get("table_name", ""),
                "catalog": kwargs.get("catalog", ""),
                "database": kwargs.get("database", ""),
                "schema": kwargs.get("schema", ""),
            }
            sorted_params = json.dumps(cache_params, sort_keys=True)
            return hashlib.md5(sorted_params.encode()).hexdigest()

        else:
            # Fallback to original method for existing tools
            return self._get_cache_key(tool_name, **kwargs)

    def _normalize_sql_for_cache(self, sql: str) -> str:
        """Normalize SQL query for consistent caching."""
        if not sql:
            return ""

        # Remove comments
        import re

        sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)

        # Normalize whitespace
        sql = " ".join(sql.split())

        # Convert to lowercase for basic normalization
        return sql.lower().strip()

    def _get_tool_ttl(self, tool_name: str) -> int:
        """Get TTL for specific tools."""
        tool_ttls = {
            "analyze_query_plan": 1800,  # 30 minutes (query plans can change)
            "check_table_conflicts": 3600,  # 1 hour (table conflicts don't change often)
            "validate_partitioning": 7200,  # 2 hours (partitioning changes infrequently)
        }
        return tool_ttls.get(tool_name, self.ttl_seconds)




class ToolBatchProcessor:
    """Batch processor for similar tool calls to improve efficiency."""

    def __init__(self):
        self.batches = {}  # tool_name -> list of (todo_item, params)

    def add_to_batch(self, tool_name: str, todo_item, params: Dict[str, Any]) -> None:
        """Add a tool call to the batch."""
        if tool_name not in self.batches:
            self.batches[tool_name] = []

        self.batches[tool_name].append((todo_item, params))

    def get_batch_size(self, tool_name: str) -> int:
        """Get the current batch size for a tool."""
        return len(self.batches.get(tool_name, []))

    def clear_batch(self, tool_name: str) -> List[Tuple]:
        """Clear and return the batch for a tool."""
        if tool_name in self.batches:
            batch = self.batches[tool_name]
            self.batches[tool_name] = []
            return batch
        return []

    def optimize_search_table_batch(self, batch: List[Tuple]) -> List[Tuple]:
        """Optimize search_table batch by consolidating similar queries."""
        if not batch:
            return batch

        # Group by similar query patterns
        query_groups = {}
        for todo_item, params in batch:
            query_text = params.get("query_text", "").lower().strip()

            # Find the most specific query (longest common prefix)
            found_group = False
            for group_key in query_groups:
                if query_text.startswith(group_key) or group_key.startswith(query_text):
                    # Use the more specific query as group key
                    new_key = max(group_key, query_text, key=len)
                    if new_key != group_key:
                        query_groups[new_key] = query_groups.pop(group_key)
                    query_groups[new_key].append((todo_item, params))
                    found_group = True
                    break

            if not found_group:
                query_groups[query_text] = [(todo_item, params)]

        # Consolidate groups: if we have multiple similar queries, keep only the most comprehensive one
        optimized_batch = []
        for group_key, items in query_groups.items():
            if len(items) == 1:
                optimized_batch.extend(items)
            else:
                # For multiple similar queries, use the one with highest top_n
                best_item = max(items, key=lambda x: x[1].get("top_n", 5))
                optimized_batch.append(best_item)
                logger.info(f"Optimized search_table batch: consolidated {len(items)} similar queries into 1")

        return optimized_batch

    def optimize_describe_table_batch(self, batch: List[Tuple]) -> List[Tuple]:
        """Optimize describe_table batch by removing duplicates."""
        if not batch:
            return batch

        # Remove duplicate table names
        seen_tables = set()
        unique_batch = []

        for todo_item, params in batch:
            table_name = params.get("table_name", "").lower().strip()
            if table_name and table_name not in seen_tables:
                seen_tables.add(table_name)
                unique_batch.append((todo_item, params))

        if len(unique_batch) < len(batch):
            logger.info(f"Optimized describe_table batch: removed {len(batch) - len(unique_batch)} duplicates")

        return unique_batch

    def optimize_analyze_query_plan_batch(self, batch: List[Tuple]) -> List[Tuple]:
        """Optimize analyze_query_plan batch by deduplicating similar SQL queries."""
        if not batch:
            return batch

        # Group by normalized SQL queries
        sql_groups = {}
        for todo_item, params in batch:
            sql_query = params.get("sql_query", "").strip()
            if sql_query:
                # Normalize SQL for grouping
                normalized_sql = self._normalize_sql_for_batch(sql_query)
                if normalized_sql not in sql_groups:
                    sql_groups[normalized_sql] = []
                sql_groups[normalized_sql].append((todo_item, params))

        # Keep only one query per unique normalized SQL
        optimized_batch = []
        for sql_hash, items in sql_groups.items():
            if len(items) > 1:
                # If multiple items for same SQL, keep the first one
                optimized_batch.append(items[0])
                logger.info(f"Optimized analyze_query_plan batch: deduplicated {len(items)} identical queries")
            else:
                optimized_batch.extend(items)

        return optimized_batch

    def optimize_check_table_conflicts_batch(self, batch: List[Tuple]) -> List[Tuple]:
        """Optimize check_table_conflicts batch by removing duplicate table checks."""
        if not batch:
            return batch

        # Remove duplicate table names
        seen_tables = set()
        unique_batch = []

        for todo_item, params in batch:
            table_name = params.get("table_name", "").lower().strip()
            if table_name and table_name not in seen_tables:
                seen_tables.add(table_name)
                unique_batch.append((todo_item, params))

        if len(unique_batch) < len(batch):
            logger.info(f"Optimized check_table_conflicts batch: removed {len(batch) - len(unique_batch)} duplicates")

        return unique_batch

    def optimize_validate_partitioning_batch(self, batch: List[Tuple]) -> List[Tuple]:
        """Optimize validate_partitioning batch by removing duplicate table validations."""
        if not batch:
            return batch

        # Remove duplicate table names (same logic as check_table_conflicts)
        seen_tables = set()
        unique_batch = []

        for todo_item, params in batch:
            table_name = params.get("table_name", "").lower().strip()
            if table_name and table_name not in seen_tables:
                seen_tables.add(table_name)
                unique_batch.append((todo_item, params))

        if len(unique_batch) < len(batch):
            logger.info(f"Optimized validate_partitioning batch: removed {len(batch) - len(unique_batch)} duplicates")

        return unique_batch

    def _normalize_sql_for_batch(self, sql: str) -> str:
        """Normalize SQL query for batch deduplication."""
        if not sql:
            return ""

        import re

        # Remove comments and normalize whitespace
        sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        sql = " ".join(sql.split())

        # Create a hash of the normalized SQL
        return hashlib.md5(sql.lower().encode()).hexdigest()

    def get_batch_optimizer(self, tool_name: str):
        """Get the appropriate batch optimizer for a tool."""
        optimizers = {
            "search_table": self.optimize_search_table_batch,
            "describe_table": self.optimize_describe_table_batch,
            "analyze_query_plan": self.optimize_analyze_query_plan_batch,
            "check_table_conflicts": self.optimize_check_table_conflicts_batch,
            "validate_partitioning": self.optimize_validate_partitioning_batch,
        }
        return optimizers.get(tool_name, lambda x: x)  # Return identity function if no optimizer


# Default keyword mapping for plan executor - tool name to list of matching phrases
DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP = {
    # Database tools
    "search_table": [
        "search for table",
        "搜索表结构",
        "查找数据库表",
        "find table in database",
        "search table schema",
        "查找表信息",
        "find database tables",
        "搜索数据库表",
        "探索数据库中的表结构",
        "找到试驾表和线索表",
        "确认表名",
        "表结构",
        "字段名",
    ],
    "describe_table": [
        "describe table",
        "检查表结构",
        "inspect table schema",
        "查看表结构",
        "examine table structure",
        "分析表结构",
        "describe table structure",
        "检查表定义",
        "查看表模式",
        "analyze table structure",
        "分析表元数据",
        "表定义",
        "表模式",
        "表元数据",
        "表字段",
    ],
    "execute_sql": [
        "execute sql query",
        "执行sql查询",
        "run sql statement",
        "执行sql语句",
        "run database query",
        "执行数据库查询",
        "execute the sql",
        "运行sql代码",
        "execute sql",
        "执行sql",
        "run the query",
        "执行查询",
        "运行sql",
        "执行sql语句",
        "查询执行",
    ],
    "read_query": [
        "run query",
        "执行查询",
        "execute database query",
        "运行数据库查询",
        "perform sql execution",
        "执行sql执行",
    ],
    # Metrics and reference SQL tools
    "search_metrics": [
        "search metrics",
        "查找指标",
        "find business metrics",
        "搜索业务指标",
        "look for kpis",
        "查找kpi",
        "search performance metrics",
        "查找绩效指标",
    ],
    "search_reference_sql": [
        "search sql examples",
        "查找参考sql",
        "find reference sql",
        "搜索sql模板",
        "look for sql patterns",
        "查找sql模式",
        "search similar sql",
        "查找相似sql",
    ],
    "list_subject_tree": [
        "list domains",
        "查看领域层级",
        "show domain structure",
        "显示业务分类",
        "explore domain layers",
        "浏览领域层级",
        "view business taxonomy",
        "查看业务分类法",
    ],
    # Semantic model tools
    "check_semantic_model_exists": [
        "check semantic model",
        "检查语义模型",
        "verify semantic model",
        "验证语义模型",
        "semantic model exists",
        "语义模型是否存在",
        "find semantic model",
        "查找语义模型",
    ],
    "check_metric_exists": [
        "check metric exists",
        "检查指标是否存在",
        "verify metric availability",
        "验证指标可用性",
        "metric exists",
        "指标是否存在",
        "find existing metric",
        "查找现有指标",
    ],
    "generate_sql_summary_id": [
        "generate summary id",
        "生成摘要标识",
        "create sql summary",
        "创建sql摘要",
        "generate sql id",
        "生成sql标识",
        "create summary identifier",
        "创建摘要标识符",
    ],
    # Time parsing tools
    "parse_temporal_expressions": [
        "parse date expressions",
        "解析日期表达式",
        "parse temporal expressions",
        "解析时间表达式",
        "analyze date ranges",
        "分析日期范围",
        "parse time periods",
        "解析时间段",
    ],
    "get_current_date": [
        "get current date",
        "获取当前日期",
        "current date",
        "今天日期",
        "today's date",
        "今日日期",
        "get today date",
        "获取今天日期",
    ],
    # File system tools
    "write_file": [
        "write file",
        "写入文件",
        "save to file",
        "保存到文件",
        "create file",
        "创建文件",
        "write content to file",
        "将内容写入文件",
    ],
    "read_file": [
        "read file",
        "读取文件",
        "load file",
        "加载文件",
        "open file",
        "打开文件",
        "read file content",
        "读取文件内容",
    ],
    "list_directory": [
        "list directory",
        "列出目录",
        "show directory contents",
        "显示目录内容",
        "list files",
        "列出文件",
        "directory listing",
        "目录列表",
    ],
    # Reporting tools
    "report": [
        "generate final report",
        "生成最终报告",
        "create comprehensive report",
        "创建综合报告",
        "write final report",
        "编写最终报告",
        "produce report",
        "生成报告",
        "生成最终html",
        "生成最终",
        "生成html",
        "生成报告文档",
    ],
    # Plan management tools
    "todo_write": [
        "create plan",
        "创建计划",
        "write execution plan",
        "编写执行计划",
        "generate todo list",
        "生成任务列表",
        "create task plan",
        "创建任务计划",
    ],
    "todo_update": [
        "update task status",
        "更新任务状态",
        "mark task complete",
        "标记任务完成",
        "update todo status",
        "更新待办状态",
        "change task state",
        "更改任务状态",
    ],
    "todo_read": [
        "read plan",
        "查看计划",
        "show execution plan",
        "显示执行计划",
        "check plan status",
        "检查计划状态",
        "view todo list",
        "查看任务列表",
    ],
}



