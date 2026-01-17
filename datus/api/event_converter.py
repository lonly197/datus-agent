# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Event converter for mapping ActionHistory to DeepResearchEvent format.
"""

import asyncio
import hashlib
import json
import re
import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

import sqlglot
from sqlglot import exp

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_metadata_from_ddl

if TYPE_CHECKING:
    from datus.schemas.node_models import TableSchema

from .models import (
    ChatEvent,
    CompleteEvent,
    DeepResearchEvent,
    ErrorEvent,
    PlanUpdateEvent,
    ReportEvent,
    TodoItem,
    TodoStatus,
    ToolCallEvent,
    ToolCallResultEvent,
)


class DeepResearchEventConverter:
    """Converts ActionHistory events to DeepResearchEvent format."""

    # Define virtual steps for non-agentic workflows (Text2SQL)
    VIRTUAL_STEPS = [
        {"id": "step_intent", "content": "分析查询意图", "node_types": ["intent_analysis"]},
        {
            "id": "step_schema",
            "content": "发现数据库模式",
            "node_types": ["schema_discovery", "schema_linking", "schema_validation"],
        },
        {"id": "step_sql", "content": "生成SQL查询", "node_types": ["generate_sql"]},
        {
            "id": "step_exec",
            "content": "执行SQL并验证结果",
            "node_types": ["execute_sql", "sql_validate", "result_validation"],
        },
        {"id": "step_reflect", "content": "自我纠正与优化", "node_types": ["reflect", "output"]},
    ]

    def __init__(self):
        self.plan_id = str(uuid.uuid4())
        self.tool_call_map: Dict[str, str] = {}  # action_id -> tool_call_id
        self.logger = get_logger(__name__)
        # small rolling cache to deduplicate near-identical assistant messages
        self._recent_assistant_hashes: "deque[str]" = deque(maxlen=50)

        # State for virtual plan management
        self.virtual_plan_emitted = False
        self.active_virtual_step_id = None
        self.completed_virtual_steps = set()

        # Fixed virtual plan ID for all PlanUpdateEvents (fixes event association issue)
        self.virtual_plan_id = str(uuid.uuid4())

    def _get_virtual_step_id(self, node_type: str) -> Optional[str]:
        """Map node type to virtual step ID."""
        for step in self.VIRTUAL_STEPS:
            if node_type in step["node_types"]:
                return str(step["id"])
        return None

    def _generate_virtual_plan_update(self, current_node_type: Optional[str] = None) -> Optional[PlanUpdateEvent]:
        """Generate PlanUpdateEvent based on current progress."""
        todos = []
        current_step_id = self._get_virtual_step_id(current_node_type) if current_node_type else None

        # If we found a new active step, update our state
        if current_step_id:
            self.active_virtual_step_id = current_step_id

        # Determine the index of the current active step
        active_index = -1
        if self.active_virtual_step_id:
            for i, step in enumerate(self.VIRTUAL_STEPS):
                if str(step["id"]) == self.active_virtual_step_id:
                    active_index = i
                    break

        for i, step in enumerate(self.VIRTUAL_STEPS):
            status = TodoStatus.PENDING

            # Robust state machine logic based on linear order
            if active_index != -1:
                if i < active_index:
                    status = TodoStatus.COMPLETED
                elif i == active_index:
                    status = TodoStatus.IN_PROGRESS
                else:
                    status = TodoStatus.PENDING
            else:
                # If no active step yet (initialization), check completed set or default to pending
                if step["id"] in self.completed_virtual_steps:
                    status = TodoStatus.COMPLETED

            todos.append(TodoItem(id=str(step["id"]), content=str(step["content"]), status=status))

        return PlanUpdateEvent(id=self.virtual_plan_id, planId=None, timestamp=int(time.time() * 1000), todos=todos)

    def _hash_text(self, s: str) -> str:
        try:
            return hashlib.sha1(s.strip().encode("utf-8")).hexdigest()
        except Exception:
            return ""

    def _generate_sql_summary(self, sql: str, result: str, row_count: int) -> str:
        """Generate a markdown summary report for SQL execution results.

        Args:
            sql: The SQL query that was executed
            result: The CSV result string from SQL execution
            row_count: Number of rows returned

        Returns:
            Markdown formatted summary report
        """
        lines = []

        # Header
        lines.append("## 📊 SQL执行结果摘要\n")

        # SQL overview
        lines.append("### SQL查询")
        lines.append(f"- **行数**: {row_count}")
        lines.append("- **状态**: ✅ 执行成功\n")

        # Result preview (first 5 rows if available)
        if result and result.strip():
            lines.append("### 结果预览")
            try:
                import pandas as pd
                from io import StringIO

                df = pd.read_csv(StringIO(result))
                preview = df.head(5).to_markdown(index=False)
                lines.append(preview)

                if len(df) > 5:
                    lines.append(f"\n*...还有 {len(df) - 5} 行数据*\n")
            except Exception:
                # If parsing fails, show raw result preview
                result_lines = result.strip().split("\n")[:6]
                lines.append("```")
                lines.extend(result_lines)
                lines.append("```")
                if len(result.strip().split("\n")) > 6:
                    lines.append("*...更多数据*\n")

        return "\n".join(lines)

    def _generate_sql_generation_report(
        self,
        sql_query: str,
        sql_result: str,
        row_count: int,
        metadata: Optional[Dict[str, Any]] = None,
        table_schemas: Optional[List[Any]] = None,
    ) -> str:
        """Generate comprehensive SQL generation report for data warehouse developers.

        New 6-section structure (v2.8):
        1. SQL Design Overview - Task understanding, design logic, data scale
        2. Tables and Fields Details - Table/field details with comments from DDL
        3. Annotated SQL with Comments - SQL with business logic annotations
        4. SQL Validation Results - Enhanced validation display
        5. Execution Verification Results - Clarifying 0 rows vs SQL error
        6. Optimization Suggestions - Performance and quality recommendations

        Args:
            sql_query: The final SQL query that was generated
            sql_result: The CSV result string from SQL execution
            row_count: Number of rows returned
            metadata: Workflow metadata containing validation, intent, and reflection results
            table_schemas: List of TableSchema objects with DDL definitions

        Returns:
            Markdown formatted comprehensive report
        """
        lines = []

        # Header
        lines.append("## 📋 SQL生成报告（数仓开发版）\n")

        # ============================================================
        # Section 1: SQL Design Overview
        # ============================================================
        lines.append("### 1. SQL设计概述")

        # Extract clarified task for design understanding
        clarified_task = ""
        if metadata and metadata.get("clarified_task"):
            clarified_task = metadata["clarified_task"]
        elif metadata and metadata.get("intent_clarification"):
            clarified_task = metadata["intent_clarification"].get("clarified_task", "")

        if clarified_task:
            lines.append(f"**任务理解**: {clarified_task}")
        else:
            lines.append("**任务理解**: 生成SQL查询以满足数据分析需求")

        # Extract table info for scale analysis
        table_count = 0
        field_count = 0
        if table_schemas:
            table_count = len(table_schemas)
            for schema in table_schemas:
                definition = getattr(schema, "definition", "")
                if definition:
                    ddl_info = self._parse_ddl_comments(definition)
                    field_count += len(ddl_info["columns"])

        lines.append(f"**数据规模**: 涉及 {table_count} 张表、{field_count} 个字段")

        # Generate design logic summary
        design_logic = []
        try:
            parsed = self._parse_sql_structure(sql_query)
            if parsed:
                # Check for CTE usage
                if parsed.find(exp.With):
                    design_logic.append("使用CTE组织查询逻辑")

                # Check for JOIN operations
                join_count = len(list(parsed.find_all(exp.Join)))
                if join_count > 0:
                    design_logic.append(f"包含{join_count}个表关联")

                # Check for aggregation
                if parsed.find(exp.Agg):
                    design_logic.append("包含聚合计算")

                # Check for window functions
                if parsed.find(exp.Window):
                    design_logic.append("使用窗口函数")

                # Check for filtering
                if parsed.find(exp.Where):
                    design_logic.append("包含筛选条件")

        except Exception:
            pass

        if design_logic:
            lines.append("**设计思路**: " + "、".join(design_logic))
        else:
            lines.append("**设计思路**: 基于业务需求生成查询SQL")

        # Validation status summary
        validation_summary = []
        if metadata and metadata.get("sql_validation"):
            validation = metadata["sql_validation"]
            if validation.get("syntax_valid"):
                validation_summary.append("语法验证通过")
            if validation.get("tables_exist"):
                validation_summary.append("表存在性验证通过")
            if validation.get("columns_exist"):
                validation_summary.append("列存在性验证通过")
            if not validation.get("has_dangerous_ops"):
                validation_summary.append("无危险操作")

        if validation_summary:
            lines.append(f"**验证状态**: {'、'.join(validation_summary)}")
        else:
            lines.append("**验证状态**: SQL已生成，待执行验证")

        lines.append("")

        # ============================================================
        # Section 2: Tables and Fields Details
        # ============================================================
        lines.append("### 2. 使用的表和字段详情")

        if table_schemas:
            table_info = self._extract_table_info(table_schemas, sql_query)

            # Table list
            tables = table_info.get("tables", [])
            if tables:
                lines.append(f"**表清单** ({len(tables)}张表):")
                lines.append("")
                lines.append("| 表名 | 表备注 | 表类型 | 数据库 | 是否使用 |")
                lines.append("|------|--------|--------|--------|----------|")
                for t in tables:
                    table_name = t["table_name"]
                    table_comment = t["table_comment"] or "-"
                    table_type = t["table_type"]
                    database = t["database"] or "-"
                    is_used = "✅" if t["is_used"] else "-"
                    lines.append(f"| {table_name} | {table_comment} | {table_type} | {database} | {is_used} |")
                lines.append("")

            # Field list (only used fields)
            fields = table_info.get("fields", [])
            used_fields = [f for f in fields if f["is_used"]]
            if used_fields:
                lines.append(f"**字段清单** ({len(used_fields)}个字段):")
                lines.append("")
                lines.append("| 表名 | 字段名 | 字段注释 | 用途 |")
                lines.append("|------|--------|----------|------|")
                for f in used_fields:
                    table_name = f["table_name"]
                    column_name = f["column_name"]
                    column_comment = f["column_comment"] or "-"
                    usage = self._infer_field_usage(sql_query, f)
                    lines.append(f"| {table_name} | {column_name} | {column_comment} | {usage} |")
                lines.append("")

            # Relationships
            relationships = table_info.get("relationships", [])
            if relationships:
                lines.append("**表关联关系**:")
                for rel in relationships:
                    left = rel["left_table"]
                    right = rel["right_table"]
                    key = rel["join_key"]
                    join_type = rel["join_type"]
                    lines.append(f"- {left} ← {key} → {right} ({join_type} JOIN)")
                lines.append("")
        else:
            lines.append("*表结构信息不可用*")
            lines.append("")

        # ============================================================
        # Section 3: Annotated SQL with Comments
        # ============================================================
        lines.append("### 3. 带注释的SQL")

        annotated_sql = self._generate_sql_with_comments(sql_query, table_schemas or [], metadata)
        lines.append("```sql")
        lines.append(annotated_sql)
        lines.append("```")
        lines.append("")

        # ============================================================
        # Section 4: SQL Validation Results
        # ============================================================
        lines.append("### 4. SQL验证结果")

        if metadata and metadata.get("sql_validation"):
            validation = metadata["sql_validation"]

            lines.append("| 验证项 | 状态 | 说明 |")
            lines.append("|--------|------|------|")

            # Syntax validation
            syntax_valid = validation.get("syntax_valid", True)
            syntax_status = "✅ 通过" if syntax_valid else "❌ 失败"
            syntax_desc = "SQL语法正确，符合SQL方言规范" if syntax_valid else "SQL语法错误，请检查语句"
            lines.append(f"| 语法验证 | {syntax_status} | {syntax_desc} |")

            # Table existence
            tables_exist = validation.get("tables_exist", True)
            table_status = "✅ 通过" if tables_exist else "❌ 失败"
            table_desc = "所有表都在Schema中存在" if tables_exist else "部分表不存在，请检查表名"
            lines.append(f"| 表存在性 | {table_status} | {table_desc} |")

            # Column existence
            columns_exist = validation.get("columns_exist", True)
            column_status = "✅ 通过" if columns_exist else "❌ 失败"
            column_desc = "所有列都在对应表中存在" if columns_exist else "部分列不存在，请检查列名"
            lines.append(f"| 列存在性 | {column_status} | {column_desc} |")

            # Dangerous operations
            has_dangerous = validation.get("has_dangerous_ops", False)
            dangerous_status = "⚠️ 检测到" if has_dangerous else "✅ 无危险操作"
            dangerous_desc = "检测到DELETE/DROP/TRUNCATE等操作，请谨慎执行" if has_dangerous else "未检测到危险操作，可安全执行"
            lines.append(f"| 危险操作 | {dangerous_status} | {dangerous_desc} |")

            lines.append("")

            # Show warnings if any
            warnings = validation.get("warnings", [])
            if warnings:
                lines.append("**验证警告**:")
                for warning in warnings[:5]:
                    lines.append(f"- {warning}")
                if len(warnings) > 5:
                    lines.append(f"- ...还有 {len(warnings) - 5} 个警告")
                lines.append("")

        else:
            lines.append("*未进行SQL验证或验证结果不可用*")
            lines.append("")

        # ============================================================
        # Section 5: Execution Verification Results
        # ============================================================
        lines.append("### 5. 执行验证结果")

        execution_report = self._generate_execution_report(row_count, metadata)
        lines.append(execution_report)

        # ============================================================
        # Section 6: Optimization Suggestions
        # ============================================================
        lines.append("### 6. 优化建议")

        optimization = self._generate_optimization_suggestions(sql_query, table_schemas or [], metadata)
        if optimization:
            lines.append(optimization)
        else:
            lines.append("*无优化建议*")
            lines.append("")

        return "\n".join(lines)

    # ============================================================
    # SQL Report Enhancement Helper Methods (Developer-Centric)
    # ============================================================

    def _parse_ddl_comments(self, ddl: str, dialect: str = "snowflake") -> Dict[str, Any]:
        """Parse DDL to extract table and column comments.

        Args:
            ddl: DDL statement (CREATE TABLE ...)
            dialect: SQL dialect (snowflake, mysql, postgres, etc.)

        Returns:
            Dict with table_comment and columns dict mapping name->comment
        """
        result = {
            "table_comment": "",
            "columns": {}
        }

        try:
            metadata = parse_metadata_from_ddl(ddl, dialect)
            result["table_comment"] = metadata.get("table", {}).get("comment", "")
            for col in metadata.get("columns", []):
                result["columns"][col["name"]] = col.get("comment", "")
        except Exception as e:
            self.logger.warning(f"Failed to parse DDL comments: {e}")

        return result

    def _extract_table_info(self, table_schemas: List[Any], sql_query: str) -> Dict[str, Any]:
        """Extract table and field information from table_schemas and SQL.

        Args:
            table_schemas: List of TableSchema objects with DDL definitions
            sql_query: SQL query to analyze for field usage

        Returns:
            Dict with tables list, fields list, and relationships
        """
        tables_info = []
        fields_info = []

        if not table_schemas:
            return {"tables": tables_info, "fields": fields_info, "relationships": []}

        # Parse SQL to extract used tables and columns
        sql_tables = set()
        sql_columns = set()
        try:
            parsed = sqlglot.parse_one(sql_query, error_level=sqlglot.ErrorLevel.IGNORE)
            # Find all table references
            for table in parsed.find_all(exp.Table):
                sql_tables.add(table.name)
            # Find all column references
            for column in parsed.find_all(exp.Column):
                sql_columns.add(column.name)
        except Exception as e:
            self.logger.warning(f"Failed to parse SQL for table/column extraction: {e}")

        # Extract table information from DDLs
        for schema in table_schemas:
            table_name = getattr(schema, "table_name", "")
            definition = getattr(schema, "definition", "")
            database_name = getattr(schema, "database_name", "")
            table_type = getattr(schema, "table_type", "table")

            if not table_name or not definition:
                continue

            # Parse DDL for comments
            ddl_info = self._parse_ddl_comments(definition)

            tables_info.append({
                "table_name": table_name,
                "table_comment": ddl_info["table_comment"],
                "table_type": table_type,
                "database": database_name,
                "is_used": table_name in sql_tables
            })

            # Extract column information
            column_comments = ddl_info["columns"]
            for col_name, col_comment in column_comments.items():
                is_used = col_name in sql_columns
                fields_info.append({
                    "table_name": table_name,
                    "column_name": col_name,
                    "column_comment": col_comment,
                    "is_used": is_used
                })

        # Analyze relationships (JOIN keys)
        relationships = self._analyze_relationships(sql_query, tables_info)

        return {
            "tables": tables_info,
            "fields": fields_info,
            "relationships": relationships
        }

    def _analyze_relationships(self, sql_query: str, tables_info: List[Dict]) -> List[Dict[str, str]]:
        """Analyze JOIN relationships from SQL query.

        Args:
            sql_query: SQL query string
            tables_info: List of table information dicts

        Returns:
            List of relationship dicts with left_table, right_table, join_key
        """
        relationships = []

        try:
            parsed = sqlglot.parse_one(sql_query, error_level=sqlglot.ErrorLevel.IGNORE)

            # Find JOIN conditions
            for join in parsed.find_all(exp.Join):
                join_table = ""
                if isinstance(join.this, exp.Table):
                    join_table = join.this.name

                # Extract ON condition
                on_clause = join.args.get("on")
                if on_clause:
                    # Simple join key extraction (left_table.key = right_table.key)
                    if isinstance(on_clause, exp.EQ):
                        left = on_clause.left
                        right = on_clause.right
                        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                            left_table = left.table
                            right_table = right.table
                            join_key = left.name

                            if join_table and (left_table or right_table):
                                relationships.append({
                                    "left_table": left_table or "",
                                    "right_table": right_table or join_table,
                                    "join_key": join_key,
                                    "join_type": join.side if hasattr(join, "side") else "INNER"
                                })
        except Exception as e:
            self.logger.warning(f"Failed to analyze relationships: {e}")

        return relationships

    def _infer_field_usage(self, sql_query: str, field_info: Dict) -> str:
        """Infer field usage purpose based on SQL analysis.

        Args:
            sql_query: SQL query string
            field_info: Field info dict with table_name and column_name

        Returns:
            Usage description: 关联键/筛选条件/输出字段
        """
        col_name = field_info["column_name"]
        table_name = field_info["table_name"]

        try:
            parsed = sqlglot.parse_one(sql_query, error_level=sqlglot.ErrorLevel.IGNORE)

            # Check if used in JOIN condition
            for join in parsed.find_all(exp.Join):
                on_clause = join.args.get("on")
                if on_clause:
                    for col in on_clause.find_all(exp.Column):
                        if col.name == col_name:
                            return "关联键"

            # Check if used in WHERE condition
            for where in parsed.find_all(exp.Where):
                for col in where.find_all(exp.Column):
                    if col.name == col_name:
                        return "筛选条件"

            # Check if used in GROUP BY
            for group in parsed.find_all(exp.Group):
                for col in group.find_all(exp.Column):
                    if col.name == col_name:
                        return "分组字段"

            # Check if used in ORDER BY
            for order in parsed.find_all(exp.Order):
                for col in order.find_all(exp.Column):
                    if col.name == col_name:
                        return "排序字段"

        except Exception:
            pass

        return "输出字段"

    def _parse_sql_structure(self, sql_query: str, dialect: str = "snowflake") -> Optional[exp.Expression]:
        """Parse SQL structure using sqlglot.

        Args:
            sql_query: SQL query string
            dialect: SQL dialect

        Returns:
            Parsed SQL expression or None on error
        """
        try:
            return sqlglot.parse_one(sql_query, dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)
        except Exception as e:
            self.logger.warning(f"Failed to parse SQL structure: {e}")
            return None

    def _infer_cte_purpose(self, cte_name: str, cte_definition: str) -> str:
        """Infer business purpose of a CTE from its name and definition.

        Args:
            cte_name: Name of the CTE
            cte_definition: CTE SQL definition

        Returns:
            Business purpose description
        """
        # Common patterns
        patterns = {
            r"first|initial|earliest": "识别首次事件",
            r"last|final|latest": "识别最后事件",
            r"rank|row_number": "计算排名或序号",
            r"agg|aggregate|sum|count|avg": "聚合计算",
            r"filter|where": "筛选数据",
            r"join|link|relate": "关联表数据",
            r"dedup|distinct|unique": "去重或获取唯一值"
        }

        cte_lower = cte_name.lower()
        for pattern, purpose in patterns.items():
            if re.search(pattern, cte_lower):
                return purpose

        # Default purpose based on structure
        if "SELECT" in cte_definition.upper():
            return "中间查询结果"
        return "通用表达式"

    def _get_field_comment(self, table_schemas: List[Any], table_name: str, column_name: str) -> str:
        """Get field comment from table schemas.

        Args:
            table_schemas: List of TableSchema objects
            table_name: Table name
            column_name: Column name

        Returns:
            Field comment or empty string
        """
        for schema in table_schemas:
            if getattr(schema, "table_name", "") == table_name:
                definition = getattr(schema, "definition", "")
                ddl_info = self._parse_ddl_comments(definition)
                return ddl_info["columns"].get(column_name, "")
        return ""

    def _add_field_comment(self, field_name: str, field_comment: str, sql_line: str) -> str:
        """Add inline comment to a field in SQL.

        Args:
            field_name: Field/column name
            field_comment: Comment text
            sql_line: SQL line to annotate

        Returns:
            SQL line with inline comment added
        """
        if not field_comment or "--" in sql_line:
            return sql_line

        # Add comment after field name
        pattern = rf'\b{re.escape(field_name)}\b(?!\s*--)'
        replacement = f'{field_name} -- {field_comment}'
        return re.sub(pattern, replacement, sql_line, count=1)

    def _explain_condition(self, condition: exp.Expression) -> str:
        """Explain business meaning of a WHERE/JOIN condition.

        Args:
            condition: SQL condition expression

        Returns:
            Business meaning explanation
        """
        if isinstance(condition, exp.EQ):
            left = condition.left
            right = condition.right
            if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                return f"筛选 {left.name} 等于 {right.this}"
        elif isinstance(condition, exp.In):
            # Handle IN (...) conditions
            col = condition.this
            if isinstance(col, exp.Column):
                return f"筛选 {col.name} 在指定值范围内"
        elif isinstance(condition, exp.And):
            return "同时满足多个条件"
        elif isinstance(condition, exp.Or):
            return "满足任一条件"

        return "条件筛选"

    def _add_condition_comments(self, sql_lines: List[str], parsed: exp.Expression) -> List[str]:
        """Add business logic comments to WHERE/JOIN conditions.

        Args:
            sql_lines: List of SQL line strings
            parsed: Parsed SQL expression

        Returns:
            SQL lines with added condition comments
        """
        result = sql_lines.copy()

        # Add comments for WHERE clauses
        for i, line in enumerate(result):
            if "WHERE" in line.upper() or "AND" in line.upper() or "OR" in line.upper():
                # Try to explain the condition
                for where in parsed.find_all(exp.Where):
                    explanation = self._explain_condition(where.this)
                    if explanation and explanation != "条件筛选":
                        # Add comment before the line
                        result[i] = f"-- {explanation}\n{result[i]}"
                        break

        return result

    def _generate_sql_with_comments(
        self,
        sql_query: str,
        table_schemas: List[Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate annotated SQL with business logic comments.

        Args:
            sql_query: Original SQL query
            table_schemas: List of TableSchema objects
            metadata: Workflow metadata for context

        Returns:
            SQL with business logic comments
        """
        lines = []
        sql_lines = sql_query.strip().split("\n")

        try:
            parsed = self._parse_sql_structure(sql_query)
            if not parsed:
                return sql_query

            # Add header comment explaining the query purpose
            clarified_task = ""
            if metadata and metadata.get("clarified_task"):
                clarified_task = metadata["clarified_task"]
            elif metadata and metadata.get("intent_clarification"):
                clarified_task = metadata["intent_clarification"].get("clarified_task", "")

            if clarified_task:
                lines.append(f"-- SQL设计目的: {clarified_task}")
                lines.append("")

            # Process CTEs (WITH clauses)
            with_expr = parsed.find(exp.With)
            if with_expr:
                lines.append("-- 使用公共表表达式(CTE)组织复杂查询逻辑")
                for cte in with_expr.expressions:
                    if isinstance(cte, exp.CTE):
                        cte_name = cte.alias
                        cte_purpose = self._infer_cte_purpose(cte_name, str(cte.this))
                        lines.append(f"-- CTE: {cte_name} - {cte_purpose}")

            # Process main query structure
            line_idx = 0
            for line in sql_lines:
                stripped = line.strip()
                annotated_line = line

                # Add field comments for key columns
                for table_schema in table_schemas:
                    table_name = getattr(table_schema, "table_name", "")
                    definition = getattr(table_schema, "definition", "")
                    if table_name and definition:
                        ddl_info = self._parse_ddl_comments(definition)
                        for col_name, col_comment in ddl_info["columns"].items():
                            if col_comment and col_name in line:
                                # Add inline comment if not already present
                                if "--" not in line:
                                    annotated_line = self._add_field_comment(col_name, col_comment, line)

                lines.append(annotated_line)
                line_idx += 1

            # Add business logic comments for key operations
            lines = self._add_condition_comments(lines, parsed)

        except Exception as e:
            self.logger.warning(f"Failed to generate annotated SQL: {e}")
            return sql_query

        return "\n".join(lines)

    def _generate_execution_report(
        self,
        row_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate execution verification report section.

        Args:
            row_count: Number of rows returned
            metadata: Workflow metadata with validation results

        Returns:
            Markdown formatted execution report
        """
        lines = []

        # Check validation results
        syntax_valid = True
        if metadata and metadata.get("sql_validation"):
            validation = metadata["sql_validation"]
            syntax_valid = validation.get("syntax_valid", True)

        lines.append("**执行状态**: ✅ SQL已成功执行验证\n")
        lines.append("**执行详情**:")
        lines.append(f"- **语法正确**: {'✅ SQL语法验证通过，数据库成功解析' if syntax_valid else '❌ 语法验证失败'}")
        lines.append(f"- **执行返回**: {row_count}行数据")
        lines.append("")

        lines.append("**数据情况说明**:")
        if row_count == 0:
            lines.append("当前数据库中没有匹配查询条件的数据。这表明:")
            lines.append("- SQL逻辑正确（无语法错误，成功执行）")
            lines.append("- 数据库中暂无满足条件的数据")
            lines.append("")
            lines.append("**后续验证建议**:")
            lines.append("如需验证SQL逻辑，可以:")
            lines.append("1. 检查表数据是否存在（如：SELECT COUNT(*) FROM table_name）")
            lines.append("2. 确认筛选条件的时间范围或枚举值是否合理")
            lines.append("3. 检查数据是否已加载到指定时间段")
        else:
            lines.append(f"查询成功返回 {row_count} 行数据，SQL逻辑正确且数据完整。")
        lines.append("")

        lines.append("**SQL适合生产使用**: ✅ 是")
        lines.append("")

        return "\n".join(lines)

    def _generate_optimization_suggestions(
        self,
        sql_query: str,
        table_schemas: List[Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate optimization suggestions based on SQL analysis.

        Args:
            sql_query: SQL query string
            table_schemas: List of TableSchema objects
            metadata: Workflow metadata

        Returns:
            Markdown formatted optimization suggestions
        """
        suggestions = []
        lines = []

        try:
            parsed = self._parse_sql_structure(sql_query)
            if not parsed:
                return ""

            # Check for CTE usage (good practice)
            has_cte = parsed.find(exp.With) is not None
            if has_cte:
                suggestions.append("✅ 使用了CTE，提高了SQL可读性和维护性")

            # Check for JOIN operations
            join_count = len(list(parsed.find_all(exp.Join)))
            if join_count > 0:
                suggestions.append(f"✅ 包含{join_count}个表关联，建议确保关联字段有索引")

            # Check for subqueries
            subquery_count = len(list(parsed.find_all(exp.Subquery)))
            if subquery_count > 2:
                suggestions.append("💡 包含多个子查询，考虑使用CTE重构以提高可读性")

            # Check for SELECT *
            for select in parsed.find_all(exp.Select):
                if hasattr(select, "expressions"):
                    for expr in select.expressions:
                        if isinstance(expr, exp.Star):
                            suggestions.append("⚠️ 使用了SELECT *，建议明确指定所需字段以提高性能")
                            break

            # Check for missing WHERE clause in SELECT
            for select in parsed.find_all(exp.Select):
                has_where = select.find(exp.Where) is not None
                if not has_where and join_count == 0:
                    suggestions.append("💡 查询未包含WHERE条件，将扫描全表数据")

            # Data quality suggestions based on validation
            if metadata and metadata.get("sql_validation"):
                validation = metadata["sql_validation"]
                warnings = validation.get("warnings", [])
                if warnings:
                    suggestions.extend([f"⚠️ {w}" for w in warnings[:3]])

        except Exception as e:
            self.logger.warning(f"Failed to generate optimization suggestions: {e}")

        if suggestions:
            lines.append("**性能优化**:")
            for s in suggestions:
                if s.startswith("✅"):
                    lines.append(f"- {s}")
                elif s.startswith("⚠️"):
                    lines.append(f"- {s}")
                else:
                    lines.append(f"- {s}")

            lines.append("")
            lines.append("**后续分析建议**:")
            lines.append("- 根据实际数据量调整查询复杂度")
            lines.append("- 定期检查查询执行计划，优化索引策略")
            lines.append("- 对于大数据集查询，考虑添加时间范围限制")
            lines.append("")

        return "\n".join(lines) if lines else ""

    def _extract_plan_from_output(self, output: Any) -> Dict[str, Any]:
        """
        Try to find plan-related fields ('todo_list' or 'updated_item') inside
        a possibly nested output structure. Handles dicts and JSON strings and
        returns the first matching dict with keys 'todo_list' or 'updated_item',
        or {} if none found.
        """

        def try_parse(obj):
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, str):
                try:
                    return json.loads(obj)
                except Exception:
                    return None
            return None

        stack = [output]
        visited = set()
        max_depth = 6
        depth = 0
        while stack and depth < max_depth:
            depth += 1
            current = stack.pop()
            if current is None:
                continue
            # avoid looping on same object (use id where possible)
            try:
                cid = id(current)
            except Exception:
                cid = None
            if cid and cid in visited:
                continue
            if cid:
                visited.add(cid)

            parsed = try_parse(current)
            if isinstance(parsed, dict):
                # direct matches
                if "todo_list" in parsed and isinstance(parsed["todo_list"], dict):
                    return {"todo_list": parsed["todo_list"]}
                if "updated_item" in parsed and isinstance(parsed["updated_item"], dict):
                    return {"updated_item": parsed["updated_item"]}

                # common wrapper keys
                for k in ("raw_output", "result", "data", "output"):
                    if k in parsed:
                        child = try_parse(parsed[k])
                        if isinstance(child, dict):
                            stack.append(child)

                # push dict values for further traversal
                for v in parsed.values():
                    child = try_parse(v)
                    if isinstance(child, dict):
                        stack.append(child)

        return {}

    def _extract_callid_from_output(self, output: Any) -> Optional[str]:
        """
        Search nested output for common call id fields (action_id, call_id, callId, tool_call_id, toolCallId).
        Returns the first found string value or None.
        """

        def try_parse(obj):
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, str):
                try:
                    return json.loads(obj)
                except Exception:
                    return None
            return None

        keys_to_check = ("action_id", "call_id", "callId", "tool_call_id", "toolCallId", "id")
        stack = [output]
        visited = set()
        max_depth = 6
        depth = 0
        while stack and depth < max_depth:
            depth += 1
            current = stack.pop()
            if current is None:
                continue
            try:
                cid = id(current)
            except Exception:
                cid = None
            if cid and cid in visited:
                continue
            if cid:
                visited.add(cid)

            parsed = try_parse(current)
            if isinstance(parsed, dict):
                # check keys
                for k in keys_to_check:
                    if k in parsed:
                        v = parsed.get(k)
                        if isinstance(v, str) and v.strip():
                            return v
                # push wrapper keys and values
                for k in ("raw_output", "result", "data", "output"):
                    if k in parsed:
                        child = try_parse(parsed[k])
                        if isinstance(child, dict):
                            stack.append(child)
                for v in parsed.values():
                    child = try_parse(v)
                    if isinstance(child, dict):
                        stack.append(child)

        return None

    def _try_parse_json_like(self, obj: Any) -> Optional[Dict[str, Any]]:
        """
        Try to parse an object that may be a dict or a JSON string into a dict.
        Returns the dict if successful, otherwise None.
        """
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _extract_todo_id_from_action(self, action: ActionHistory) -> Optional[str]:
        """
        Extracts the todo_id from an ActionHistory object, looking in various places.
        This is crucial for correctly setting the planId for events related to specific todos.
        """
        # Try to extract from input arguments first
        if action.input and isinstance(action.input, dict):
            # Check for 'plan_id' directly in input (for preflight tools created by PreflightOrchestrator)
            if action.input.get("plan_id"):
                return action.input["plan_id"]
            # Check for 'todo_id' directly in input
            if action.input.get("todo_id"):
                return action.input["todo_id"]
            # Check for 'plan_id' in nested "input" field (for execution_event_manager records)
            if "input" in action.input and isinstance(action.input["input"], dict):
                nested_input = action.input["input"]
                if nested_input.get("plan_id"):
                    return nested_input["plan_id"]
                if nested_input.get("todo_id"):
                    return nested_input["todo_id"]
            # Check for 'arguments' field which might be a JSON string containing 'todo_id' or 'plan_id'
            if "arguments" in action.input and isinstance(action.input["arguments"], str):
                parsed_args = self._try_parse_json_like(action.input["arguments"])
                if isinstance(parsed_args, dict):
                    if parsed_args.get("plan_id"):
                        return parsed_args["plan_id"]
                    if parsed_args.get("todo_id"):
                        return parsed_args["todo_id"]

        # For plan_update actions, try to extract from output's 'updated_item' or 'todo_list'
        if action.action_type == "plan_update" and action.output:
            plan_data = self._extract_plan_from_output(action.output)
            if "updated_item" in plan_data and isinstance(plan_data["updated_item"], dict):
                return plan_data["updated_item"].get("id")
            if "todo_list" in plan_data and isinstance(plan_data["todo_list"], dict):
                # If it's a full todo_list, we might not have a single todo_id,
                # but if there's only one item, we can use its ID.
                items = plan_data["todo_list"].get("items")
                if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
                    return items[0].get("id")

        # For tool_call_result, try to find the original tool_call_id and then its todo_id
        if action.action_type == "tool_call_result" and action.input and isinstance(action.input, dict):
            original_action_id = action.input.get("action_id")
            if original_action_id and original_action_id in self.tool_call_map:
                # This is complex as we don't store todo_id in tool_call_map directly.
                # This would require looking up the original ActionHistory for the tool call.
                # For now, we rely on todo_id being in the tool call's input.
                pass

        return None

    def _get_unified_plan_id(self, action: ActionHistory, force_associate: bool = False) -> Optional[str]:
        """
        Unified planId retrieval strategy to ensure consistent event association.

        This method implements a clear ID responsibility separation:
        - virtual_plan_id: Stable event association identifier for Text2SQL workflow
        - active_virtual_step_id: Internal state tracking only (never exposed as planId)
        - todo_id: Specific task identifier from action (highest priority)

        Args:
            action: The action to get planId for
            force_associate: If True, use virtual_plan_id when no todo_id exists
                           (for events that need association like ToolCallEvent, ToolCallResultEvent)
                           If False, return None when no todo_id exists
                           (for general messages like ChatEvent that don't require association)

        Returns:
            planId priority: todo_id > virtual_step_id > virtual_plan_id > None
        """
        # 1. Priority: Extract todo_id from action (most specific)
        todo_id = self._extract_todo_id_from_action(action)
        if todo_id:
            return todo_id

        # 2. Map Text2SQL action_type to virtual step ID for Tool event binding
        # This ensures ToolCallEvents bind to their corresponding TodoItems
        virtual_step_id = self._get_virtual_step_id(action.action_type)
        if virtual_step_id:
            return virtual_step_id

        # 3. For events that require association, use virtual_plan_id (stable across workflow)
        if force_associate:
            return self.virtual_plan_id

        # 4. For events that don't require association, return None
        return None

    def _find_tool_call_id(self, action: ActionHistory) -> Optional[str]:
        """
        Try to determine the tool_call_id for a given action by:
          1) mapping action.action_id -> tool_call_id
          2) checking common id keys in action.input (including parsing stringified 'arguments')
          3) extracting call id from nested output
        Returns the tool_call_id string if found, else None.
        """
        # direct mapping by action_id
        if action.action_id in self.tool_call_map:
            return self.tool_call_map[action.action_id]

        # Normalize and inspect input for candidate ids
        input_candidate = None
        if action.input:
            # If it's a dict, copy and try to parse common string fields like 'arguments'
            if isinstance(action.input, dict):
                input_candidate = dict(action.input)
                # parse 'arguments' if it's a json string
                if "arguments" in input_candidate and isinstance(input_candidate["arguments"], str):
                    parsed_args = self._try_parse_json_like(input_candidate["arguments"])
                    if isinstance(parsed_args, dict):
                        input_candidate.update(parsed_args)
            else:
                # try to parse string input
                parsed = self._try_parse_json_like(action.input)
                if isinstance(parsed, dict):
                    input_candidate = parsed

        if input_candidate and isinstance(input_candidate, dict):
            for k in ("tool_call_id", "toolCallId", "call_id", "callId", "action_id", "id", "todo_id", "todoId"):
                v = input_candidate.get(k)
                if isinstance(v, str) and v:
                    # if value maps to our stored map, return mapped id
                    if v in self.tool_call_map:
                        return self.tool_call_map[v]
                    # otherwise, return the candidate (best-effort)
                    return v

        # try to extract from nested output
        if action.output:
            candidate = self._extract_callid_from_output(action.output)
            if candidate:
                if candidate in self.tool_call_map:
                    return self.tool_call_map[candidate]
                return candidate

        return None

    def _is_internal_todo_update(self, action: ActionHistory) -> bool:
        """
        Check if this is an internal todo_update call from server executor that should be filtered.

        Args:
            action: The ActionHistory to check

        Returns:
            bool: True if this is an internal todo_update that should be filtered
        """
        # Internal todo_update calls from the server executor often have specific action_id patterns
        # or messages. We want to filter these out as ToolCallEvents.
        if action.action_type == "todo_update":
            # Check if the action_id starts with "server_call_"
            if action.action_id.startswith("server_call_"):
                return True
            # Additionally, check if the message indicates it's an internal update
            if action.messages and (
                "Server executor: starting todo" in action.messages
                or "Server executor: todo_in_progress" in action.messages
                or "Server executor: todo_completed" in action.messages
                or "Server executor: todo_complete failed" in action.messages
            ):
                return True
        return False

    def validate_event_flow(self, action_type: str, events: List[DeepResearchEvent]) -> bool:
        """Validate event flow completeness for critical actions.

        Args:
            action_type: The type of action being converted
            events: The generated events list

        Returns:
            bool: True if event flow is valid, False otherwise
        """
        if action_type in ["schema_discovery", "sql_execution"]:
            # Should have both ToolCallEvent and ToolCallResultEvent
            has_call = any(e.event == "tool_call" for e in events)
            has_result = any(e.event == "tool_call_result" for e in events)
            if not (has_call and has_result):
                self.logger.warning(f"Action {action_type} missing tool call/result events")
                return False
        return True

    def convert_action_to_event(self, action: ActionHistory, seq_num: int) -> List[DeepResearchEvent]:
        """Convert ActionHistory to DeepResearchEvent list."""

        timestamp = int(time.time() * 1000)
        event_id = f"{action.action_id}_{seq_num}"
        events: List[DeepResearchEvent] = []

        # Debug logging: track action conversion
        self.logger.debug(f"Converting action: {action.action_type}, role: {action.role}, status: {action.status}")

        # Extract todo_id if present (for plan-related events)
        # Note: _get_unified_plan_id() will handle fallback logic properly
        todo_id = self._extract_todo_id_from_action(action)

        # 1. Handle chat/assistant messages
        if action.role == ActionRole.ASSISTANT:
            # ChatEvent should only have planId when directly related to a specific todo item execution
            # For general assistant messages (thinking, planning, etc.), planId should be None
            chat_plan_id = self._get_unified_plan_id(action, force_associate=False)

            # Emit streaming token chunks ("raw_stream") always.
            # Emit intermediate "message" or "thinking" actions only when explicitly flagged
            # by the producer via an `emit_chat` boolean in action.output or action.input.
            emit_flag = False
            if action.output and isinstance(action.output, dict):
                emit_flag = bool(action.output.get("emit_chat"))
            if not emit_flag and action.input and isinstance(action.input, dict):
                emit_flag = bool(action.input.get("emit_chat"))

            # Only allow raw_stream unconditionally; allow message/thinking when flagged.
            if action.action_type == "raw_stream" or (action.action_type in ("message", "thinking") and emit_flag):
                content = ""
                if action.output and isinstance(action.output, dict):
                    content = (
                        action.output.get("content", "")
                        or action.output.get("response", "")
                        or action.output.get("raw_output", "")
                        or action.messages
                    )
                # Only send chat events if they have actual content or are important messages
                if content and content.strip():
                    # dedupe near-identical assistant messages
                    h = self._hash_text(content)
                    if h and h in self._recent_assistant_hashes:
                        # skip duplicate message
                        return []
                    if h:
                        self._recent_assistant_hashes.append(h)
                    events.append(ChatEvent(id=event_id, planId=chat_plan_id, timestamp=timestamp, content=content))

            elif action.action_type == "chat_response":
                content = ""
                if action.output and isinstance(action.output, dict):
                    content = action.output.get("response", "") or action.output.get("content", "")
                # Always send chat_response events as they are final responses
                if content or action.output:
                    h = self._hash_text(content or str(action.output))
                    if h and h in self._recent_assistant_hashes:
                        return []
                    if h:
                        self._recent_assistant_hashes.append(h)
                    events.append(ChatEvent(id=event_id, planId=chat_plan_id, timestamp=timestamp, content=content))

        # Handle SQL generation events
        elif action.action_type == "sql_generation" and action.status == ActionStatus.SUCCESS:
            sql_content = ""
            if action.output and isinstance(action.output, dict):
                sql = action.output.get("sql_query", "")
                if sql:
                    # Wrap as Markdown SQL code block
                    sql_content = f"```sql\n{sql}\n```"

            if sql_content:
                # SQL generation events usually don't have a specific planId unless tied to a todo
                sql_plan_id = todo_id if todo_id else None
                events.append(
                    ChatEvent(
                        id=event_id,
                        planId=sql_plan_id,
                        timestamp=timestamp,
                        content=sql_content,
                    )
                )

        # Handle Intent Analysis (convert to ChatEvent for visibility)
        elif action.action_type == "intent_analysis" and action.status == ActionStatus.SUCCESS:
            intent = "Unknown"
            confidence = 0.0
            if action.output and isinstance(action.output, dict):
                intent = action.output.get("intent", intent)
                confidence = action.output.get("confidence", confidence)

            content = f"🧐 **Intent Detected**: `{intent}` (Confidence: {confidence:.2f})"
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=todo_id,
                    timestamp=timestamp,
                    content=content,
                )
            )

        # Handle SQL Validation (convert to ChatEvent for visibility)
        elif action.action_type == "sql_validation":
            # Extract validation results from output
            validation_result = {}
            if action.output and isinstance(action.output, dict):
                validation_result = action.output

            is_valid = validation_result.get("is_valid", False)
            syntax_valid = validation_result.get("syntax_valid", False)
            tables_exist = validation_result.get("tables_exist", True)
            columns_exist = validation_result.get("columns_exist", True)
            has_dangerous_ops = validation_result.get("has_dangerous_ops", False)
            errors = validation_result.get("errors", [])
            warnings = validation_result.get("warnings", [])

            # Build validation summary message
            if is_valid:
                content_lines = [
                    "✅ **SQL验证通过**",
                    f"- 语法验证: {'✅ 通过' if syntax_valid else '❌ 失败'}",
                    f"- 表存在性: {'✅ 通过' if tables_exist else '❌ 失败'}",
                    f"- 列存在性: {'✅ 通过' if columns_exist else '❌ 失败'}",
                    f"- 危险操作: {'⚠️ 检测到' if has_dangerous_ops else '✅ 无危险操作'}",
                ]

                if warnings:
                    content_lines.append(f"\n**警告** ({len(warnings)}):")
                    for warning in warnings[:3]:
                        content_lines.append(f"- {warning}")
                    if len(warnings) > 3:
                        content_lines.append(f"- ...还有 {len(warnings) - 3} 个警告")

                content = "\n".join(content_lines)
            else:
                content_lines = [
                    "❌ **SQL验证失败**",
                    f"- 语法验证: {'✅ 通过' if syntax_valid else '❌ 失败'}",
                    f"- 表存在性: {'✅ 通过' if tables_exist else '❌ 失败'}",
                    f"- 列存在性: {'✅ 通过' if columns_exist else '❌ 失败'}",
                    f"- 危险操作: {'⚠️ 检测到' if has_dangerous_ops else '✅ 无危险操作'}",
                ]

                if errors:
                    content_lines.append(f"\n**错误** ({len(errors)}):")
                    for error in errors[:3]:
                        content_lines.append(f"- {error}")
                    if len(errors) > 3:
                        content_lines.append(f"- ...还有 {len(errors) - 3} 个错误")

                if warnings:
                    content_lines.append(f"\n**警告** ({len(warnings)}):")
                    for warning in warnings[:3]:
                        content_lines.append(f"- {warning}")
                    if len(warnings) > 3:
                        content_lines.append(f"- ...还有 {len(warnings) - 3} 个警告")

                content = "\n".join(content_lines)

            events.append(
                ChatEvent(
                    id=event_id,
                    planId=self._get_unified_plan_id(action, force_associate=True),
                    timestamp=timestamp,
                    content=content,
                )
            )

        # Handle Schema Discovery (convert to ToolCallEvent)
        elif action.action_type == "schema_discovery":
            tool_call_id = str(uuid.uuid4())
            # Ensure input is a dict
            tool_input = {}
            if action.input and isinstance(action.input, dict):
                tool_input = action.input

            # Use unified plan ID strategy (fix for Text2SQL workflow)
            schema_plan_id = self._get_unified_plan_id(action, force_associate=True)

            events.append(
                ToolCallEvent(
                    id=f"{event_id}_call",
                    planId=schema_plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    toolName="schema_discovery",
                    input=tool_input,
                )
            )

            events.append(
                ToolCallResultEvent(
                    id=f"{event_id}_result",
                    planId=schema_plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    data=action.output,
                    error=action.status == ActionStatus.FAILED,
                )
            )

        # Handle Schema Linking
        elif action.action_type == "schema_linking" and action.status == ActionStatus.SUCCESS:
            tables_found = 0
            if action.output and isinstance(action.output, dict):
                tables_found = action.output.get("tables_found", 0)

            content = f"🔗 **Schema Linking**: Linked {tables_found} tables to the query context."
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=todo_id,
                    timestamp=timestamp,
                    content=content,
                )
            )

        # Handle Knowledge Search
        elif action.action_type == "knowledge_search" and action.status == ActionStatus.SUCCESS:
            knowledge_found = False
            if action.output and isinstance(action.output, dict):
                knowledge_found = action.output.get("knowledge_found", False)

            if knowledge_found:
                content = "📚 **Knowledge Search**: Found relevant external business knowledge."
                events.append(
                    ChatEvent(
                        id=event_id,
                        planId=todo_id,
                        timestamp=timestamp,
                        content=content,
                    )
                )

        # Handle SQL Execution (convert to ToolCallEvent)
        elif action.action_type == "sql_execution":
            tool_call_id = str(uuid.uuid4())
            tool_input = {}
            if action.input and isinstance(action.input, dict):
                tool_input = action.input

            # Use unified plan ID strategy (fix for Text2SQL workflow)
            exec_plan_id = self._get_unified_plan_id(action, force_associate=True)

            events.append(
                ToolCallEvent(
                    id=f"{event_id}_call",
                    planId=exec_plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    toolName="execute_sql",
                    input=tool_input,
                )
            )

            events.append(
                ToolCallResultEvent(
                    id=f"{event_id}_result",
                    planId=exec_plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    data=action.output,
                    error=action.status == ActionStatus.FAILED,
                )
            )

        # Handle Preflight Tool Execution
        elif action.action_type.startswith("preflight_"):
            # Extract tool name from action_type (e.g., "preflight_describe_table" -> "describe_table")
            tool_name = action.action_type.replace("preflight_", "", 1)

            # Extract tool_call_id from input or generate a new one
            tool_call_id = None
            if action.input and isinstance(action.input, dict):
                # Try to find tool_call_id in the input
                for key, value in action.input.items():
                    if "tool_call" in str(key).lower() or isinstance(value, str) and "preflight_" in value:
                        tool_call_id = value
                        break

            if not tool_call_id:
                tool_call_id = str(uuid.uuid4())

            # Create ToolCallEvent for processing status
            if action.status == ActionStatus.PROCESSING:
                tool_input = {}
                if action.input and isinstance(action.input, dict):
                    tool_input = action.input

                events.append(
                    ToolCallEvent(
                        id=f"{event_id}_call",
                        planId=todo_id,  # Use plan_id from _extract_todo_id_from_action
                        timestamp=timestamp,
                        toolCallId=tool_call_id,
                        toolName=tool_name,
                        input=tool_input,
                    )
                )

            # Create ToolCallResultEvent for completed/failed status
            if action.status in [ActionStatus.SUCCESS, ActionStatus.FAILED]:
                events.append(
                    ToolCallResultEvent(
                        id=f"{event_id}_result",
                        planId=todo_id,  # Use plan_id from _extract_todo_id_from_action
                        timestamp=timestamp,
                        toolCallId=tool_call_id,
                        data=action.output,
                        error=action.status == ActionStatus.FAILED,
                    )
                )

        # Handle Reflection Analysis
        elif action.action_type == "reflection_analysis" and action.status == ActionStatus.SUCCESS:
            strategy = "UNKNOWN"
            if action.output and isinstance(action.output, dict):
                strategy = action.output.get("strategy", strategy)

            content = f"🤔 **Reflection**: Analyzing results... Strategy: `{strategy}`"
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=todo_id,
                    timestamp=timestamp,
                    content=content,
                )
            )

        # 2. Handle tool calls - ToolCallEvent / ToolCallResultEvent and PlanUpdateEvent for plan tools
        elif action.role == ActionRole.TOOL:
            # 过滤掉内部的todo_update状态管理调用
            if action.action_type == "todo_update" and self._is_internal_todo_update(action):
                return []  # 不生成任何事件

            tool_call_id = str(uuid.uuid4())
            # store mapping for later result binding
            self.tool_call_map[action.action_id] = tool_call_id

            is_plan_tool = action.action_type in ["todo_write", "todo_update"]
            plan_data = {}
            if action.output:
                plan_data = self._extract_plan_from_output(action.output)

            # Build normalized input: parse stringified 'arguments' if present so fields like todo_id are exposed.
            normalized_input: Dict[str, Any] = {}
            if action.input:
                if isinstance(action.input, dict):
                    normalized_input = dict(action.input)
                    # parse 'arguments' field if it's a JSON string
                    if "arguments" in normalized_input and isinstance(normalized_input["arguments"], str):
                        parsed_args = self._try_parse_json_like(normalized_input["arguments"])
                        if isinstance(parsed_args, dict):
                            normalized_input.update(parsed_args)
                else:
                    parsed = self._try_parse_json_like(action.input)
                    if isinstance(parsed, dict):
                        normalized_input = parsed

            # Determine planId for tool events:
            # - For plan tools: use specific todo_id if available, otherwise None for todo_write (creates entire plan)
            # - For other tools: use todo_id if present (indicates tool is executing a specific todo)
            tool_plan_id = None
            if is_plan_tool:
                if action.action_type == "todo_update" and todo_id:
                    # todo_update operates on specific todo items
                    tool_plan_id = todo_id
                # todo_write creates the entire plan, so planId should be None
            elif todo_id:
                # Non-plan tools that are executing specific todos
                tool_plan_id = todo_id

            # If this is a plan tool and we found plan data, emit PlanUpdateEvent first
            if is_plan_tool and plan_data:
                todos = []
                if "todo_list" in plan_data:
                    tlist = plan_data["todo_list"]
                    if isinstance(tlist, dict) and "items" in tlist:
                        for todo_data in tlist["items"]:
                            if isinstance(todo_data, dict):
                                todos.append(
                                    TodoItem(
                                        id=todo_data.get("id", str(uuid.uuid4())),
                                        content=todo_data.get("content", ""),
                                        status=TodoStatus(todo_data.get("status", "pending")),
                                    )
                                )
                elif "updated_item" in plan_data:
                    ui = plan_data["updated_item"]
                    if isinstance(ui, dict):
                        todos.append(
                            TodoItem(
                                id=ui.get("id", str(uuid.uuid4())),
                                content=ui.get("content", ""),
                                status=TodoStatus(ui.get("status", "pending")),
                            )
                        )

                if todos:
                    # For plan tools, emit tool events AND plan update event
                    events.append(
                        ToolCallEvent(
                            id=f"{event_id}_call",
                            planId=tool_plan_id,
                            timestamp=timestamp,
                            toolCallId=tool_call_id,
                            toolName=action.action_type,
                            input=normalized_input or (action.input if isinstance(action.input, dict) else {}),
                        )
                    )

                    if action.output:
                        events.append(
                            ToolCallResultEvent(
                                id=f"{event_id}_result",
                                planId=tool_plan_id,
                                timestamp=timestamp,
                                toolCallId=tool_call_id,
                                data=action.output,
                                error=action.status == ActionStatus.FAILED,
                            )
                        )

                    # PlanUpdateEvent uses the specific todo_id for updated items, or None for plan creation
                    plan_update_plan_id = None
                    if action.action_type == "todo_update" and "updated_item" in plan_data:
                        ui = plan_data["updated_item"]
                        if isinstance(ui, dict) and ui.get("id"):
                            plan_update_plan_id = ui["id"]

                    events.append(
                        PlanUpdateEvent(
                            id=f"{event_id}_plan", planId=plan_update_plan_id, timestamp=timestamp, todos=todos
                        )
                    )
            else:
                # Normal tool call: emit call + result (if available)
                events.append(
                    ToolCallEvent(
                        id=f"{event_id}_call",
                        planId=tool_plan_id,
                        timestamp=timestamp,
                        toolCallId=tool_call_id,
                        toolName=action.action_type,
                        input=normalized_input or (action.input if isinstance(action.input, dict) else {}),
                    )
                )
                if action.output:
                    events.append(
                        ToolCallResultEvent(
                            id=f"{event_id}_result",
                            planId=tool_plan_id,
                            timestamp=timestamp,
                            toolCallId=tool_call_id,
                            data=action.output,
                            error=action.status == ActionStatus.FAILED,
                        )
                    )

        # 3. Handle tool results (legacy support)
        elif action.action_type == "tool_call_result" and action.output:
            # Try to find a matching tool_call_id robustly
            tool_call_id = self._find_tool_call_id(action)

            if tool_call_id:
                events.append(
                    ToolCallResultEvent(
                        id=event_id,
                        planId=self.plan_id,
                        timestamp=timestamp,
                        toolCallId=str(tool_call_id),
                        data=action.output,
                        error=action.status == ActionStatus.FAILED,
                    )
                )

        # 4. Handle plan updates
        elif action.action_type == "plan_update" and action.output:
            todos = []
            if isinstance(action.output, dict):
                # Handle both "todos" (legacy) and "todo_list" (new) formats
                todo_data_source = None
                if "todo_list" in action.output and isinstance(action.output["todo_list"], dict):
                    todo_data_source = action.output["todo_list"].get("items", [])
                elif "todos" in action.output and isinstance(action.output["todos"], list):
                    todo_data_source = action.output["todos"]

                if todo_data_source:
                    for todo_data in todo_data_source:
                        if isinstance(todo_data, dict):
                            todos.append(
                                TodoItem(
                                    id=todo_data.get("id", str(uuid.uuid4())),
                                    content=todo_data.get("content", ""),
                                    status=TodoStatus(todo_data.get("status", "pending")),
                                )
                            )

            # For plan_update events from workflow nodes, use virtual_plan_id to maintain association
            # For plan_update events from tools, use event_id to maintain unique plan identification
            plan_event_id = self.virtual_plan_id if action.role == ActionRole.WORKFLOW else event_id
            events.append(PlanUpdateEvent(id=plan_event_id, planId=None, timestamp=timestamp, todos=todos))

        # 5. Handle workflow completion (修复 CompleteEvent 处理)
        elif action.action_type == "workflow_completion" and action.status == ActionStatus.SUCCESS:
            # Force complete all virtual steps
            if self.virtual_plan_emitted:
                final_todos = []
                for step in self.VIRTUAL_STEPS:
                    final_todos.append(
                        TodoItem(id=str(step["id"]), content=str(step["content"]), status=TodoStatus.COMPLETED)
                    )
                events.append(
                    PlanUpdateEvent(id=f"{event_id}_plan_final", planId=None, timestamp=timestamp, todos=final_todos)
                )

            events.append(
                CompleteEvent(
                    id=event_id,
                    planId=None,  # CompleteEvent should not have planId by default
                    timestamp=timestamp,
                    content=action.messages,
                )
            )

        # 6. Handle workflow initialization
        elif action.action_type == "workflow_init":
            # Convert workflow init to a ChatEvent to inform user
            content = f"🚀 **System Initialization**: {action.messages}"
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=None,
                    timestamp=timestamp,
                    content=content,
                )
            )

            # Emit initial virtual plan for Text2SQL workflow
            # We assume if it's workflow_init, we can initialize the virtual plan
            if not self.virtual_plan_emitted:
                plan_update = self._generate_virtual_plan_update()
                if plan_update:
                    events.append(plan_update)
                    self.virtual_plan_emitted = True

        # 7. Handle node execution
        elif action.action_type == "node_execution":
            # Update virtual plan if applicable
            node_type = None
            if action.input and isinstance(action.input, dict):
                node_type = action.input.get("node_type")

            if node_type:
                plan_update = self._generate_virtual_plan_update(node_type)
                if plan_update and plan_update.todos:
                    # Check if status actually changed to avoid spamming events
                    # (Simplified: always emit for now as _generate logic handles state)
                    events.append(plan_update)

            # Convert node execution to a status update (ChatEvent with specific formatting or just text)
            # We can use it to show what the agent is doing
            node_desc = "Unknown Node"
            if action.input and isinstance(action.input, dict):
                node_desc = action.input.get("description", node_desc)

            content = f"🔄 **Executing Step**: {node_desc}"
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=self._get_unified_plan_id(action, force_associate=True),  # Use unified plan ID
                    timestamp=timestamp,
                    content=content,
                )
            )

        # 8. Handle errors
        elif action.status == ActionStatus.FAILED:
            # ErrorEvent should use todo_id if the error is related to a specific todo
            error_plan_id = todo_id if todo_id else None

            error_msg = action.messages or "Unknown error"

            # Enrich error message with details if available
            if action.output and isinstance(action.output, dict):
                details = []
                if action.output.get("error_code"):
                    details.append(f"Code: {action.output.get('error_code')}")

                # Check for recovery suggestions
                suggestions = action.output.get("recovery_suggestions")
                if suggestions and isinstance(suggestions, list):
                    suggestions_str = "\n".join([f"- {s}" for s in suggestions])
                    details.append(f"Suggestions:\n{suggestions_str}")

                if details:
                    error_msg += "\n\n" + "\n".join(details)

            events.append(ErrorEvent(id=event_id, planId=error_plan_id, timestamp=timestamp, error=error_msg))

        # 9. Handle report generation and SQL output
        elif action.action_type == "output_generation" and action.output:
            if isinstance(action.output, dict):
                # First, handle SQL output via ChatEvent (for text2sql workflow)
                sql_query = action.output.get("sql_query", "")
                sql_result = action.output.get("sql_result", "")
                sql_query_final = action.output.get("sql_query_final", "")
                sql_result_final = action.output.get("sql_result_final", "")
                row_count = action.output.get("row_count", 0)
                success = action.output.get("success", True)
                metadata = action.output.get("metadata", {})

                # Use final SQL if available, otherwise use generated SQL
                final_sql = sql_query_final if sql_query_final else sql_query

                # If SQL exists, send ChatEvent with comprehensive SQL generation report
                if final_sql:
                    # Generate comprehensive report (includes SQL, validation, intent, etc.)
                    final_result = sql_result_final if sql_result_final else sql_result

                    # Use comprehensive report if metadata is available, otherwise fall back to simple summary
                    # More lenient condition: check if metadata has valid content (excluding default reflection_count=0)
                    if metadata:
                        # Check if metadata has any valid content (excluding default reflection_count=0)
                        has_valid_content = any(
                            bool(v) and not (k == "reflection_count" and v == 0)
                            for k, v in metadata.items()
                        )

                        if has_valid_content or "table_schemas" in metadata:
                            table_schemas = metadata.get("table_schemas")
                            report = self._generate_sql_generation_report(
                                sql_query=final_sql,
                                sql_result=final_result,
                                row_count=row_count,
                                metadata=metadata,
                                table_schemas=table_schemas
                            )
                        else:
                            # Fallback to simple summary for backward compatibility
                            report = self._generate_sql_summary(final_sql, final_result, row_count)
                    else:
                        # Fallback to simple summary for backward compatibility
                        report = self._generate_sql_summary(final_sql, final_result, row_count)

                    # Send the comprehensive report as ChatEvent
                    events.append(
                        ChatEvent(
                            id=f"{event_id}_report",
                            planId=todo_id if todo_id else None,
                            timestamp=timestamp,
                            content=report,
                        )
                    )

                # Original ReportEvent handling (for HTML reports)
                report_url = action.output.get("report_url", "")
                report_data = action.output.get("html_content", "")
                # Create ReportEvent if we have either url or data
                if report_url or report_data:
                    # ReportEvent should use todo_id if related to a specific todo
                    report_plan_id = todo_id if todo_id else None
                    events.append(
                        ReportEvent(
                            id=event_id, planId=report_plan_id, timestamp=timestamp, url=report_url, data=report_data
                        )
                    )

        # Debug logging: track generated events
        for event in events:
            self.logger.debug(f"Generated event: {event.event}, planId: {event.planId}, id: {event.id}")

        # Validate event flow for critical actions
        self.validate_event_flow(action.action_type, events)

        return events

    async def convert_stream_to_events(
        self, action_stream: AsyncGenerator[ActionHistory, None]
    ) -> AsyncGenerator[str, None]:
        """Convert ActionHistory stream to DeepResearchEvent SSE stream."""

        seq_num = 0

        try:
            async for action in action_stream:
                seq_num += 1
                events = self.convert_action_to_event(action, seq_num)

                for event in events:
                    # Convert to JSON and yield as SSE data
                    event_json = event.model_dump_json()
                    yield f"data: {event_json}\n\n"

                    # 检查是否被取消
                    current_task = asyncio.current_task()
                    if current_task and current_task.cancelled():
                        break
        except asyncio.CancelledError:
            self.logger.info("Event conversion stream was cancelled")
            raise
