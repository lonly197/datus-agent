# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SQL processing utilities for event conversion.

This module provides functions to generate SQL reports, parse DDL,
analyze relationships, and create annotated SQL with comments.
"""

import re
from typing import Any, Dict, List, Optional, Set

import sqlglot
from sqlglot import exp

from datus.utils.constants import DBType
from datus.utils.sql_utils import (
    extract_enhanced_metadata_from_ddl,
    extract_sql_symbols,
    parse_dialect,
    sanitize_ddl_for_storage,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def generate_sql_summary(sql: str, result: str, row_count: int) -> str:
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


def generate_sql_failure_report(metadata: Dict[str, Any]) -> str:
    """Generate a failure report for SQL generation failures."""
    meta = metadata or {}
    stage_key = meta.get("failure_stage", "") if isinstance(meta, dict) else ""
    stage_labels = {
        "schema_discovery": "Schema 发现",
        "schema_validation": "Schema 校验",
        "generate_sql": "SQL 生成",
        "sql_validation": "SQL 校验",
        "execute_sql": "SQL 执行",
        "result_validation": "结果校验",
        "reflect": "反思",
        "output": "输出",
    }
    stage_label = stage_labels.get(stage_key, stage_key or "未知")
    reason = meta.get("termination_reason", "") if isinstance(meta, dict) else ""

    lines = ["## ❌ SQL生成失败报告", ""]
    lines.append(f"**失败阶段**: {stage_label}")
    if reason:
        lines.append(f"**失败原因**: {reason}")
    else:
        lines.append("**失败原因**: SQL生成或校验未通过")

    retry_count = meta.get("sql_retry_count")
    retry_max = meta.get("sql_retry_max")
    retry_interval = meta.get("sql_retry_interval")
    if retry_count is not None or retry_max is not None:
        retry_display = f"{retry_count or 0}/{retry_max or 0}"
        interval_display = f"{retry_interval}s" if retry_interval is not None else "未设置"
        lines.append(f"**SQL重试**: {retry_display}，间隔 {interval_display}")

    sql_validation = meta.get("sql_validation")
    if isinstance(sql_validation, dict):
        lines.append("")
        lines.append("### SQL校验信息")
        syntax_valid = sql_validation.get("syntax_valid")
        tables_exist = sql_validation.get("tables_exist")
        columns_exist = sql_validation.get("columns_exist")
        dangerous = sql_validation.get("has_dangerous_ops")
        if syntax_valid is not None:
            lines.append(f"- 语法校验: {'✅通过' if syntax_valid else '❌失败'}")
        if tables_exist is not None:
            lines.append(f"- 表存在性: {'✅通过' if tables_exist else '❌失败'}")
        if columns_exist is not None:
            lines.append(f"- 列存在性: {'✅通过' if columns_exist else '❌失败'}")
        if dangerous is not None:
            lines.append(f"- 危险操作: {'❌存在' if dangerous else '✅无危险操作'}")

        errors = sql_validation.get("errors") or []
        if errors:
            lines.append("")
            lines.append("**错误明细**:")
            for item in errors[:5]:
                lines.append(f"- {item}")
            if len(errors) > 5:
                lines.append(f"- ...还有 {len(errors) - 5} 条错误")

        error_details = sql_validation.get("error_details") or []
        if error_details:
            lines.append("")
            lines.append("**错误分类**:")
            for detail in error_details:
                detail_type = detail.get("type")
                columns = detail.get("columns") or []
                if detail_type == "missing_columns_physical":
                    label = "物理列缺失"
                elif detail_type == "missing_columns_virtual":
                    label = "CTE/子查询输出列缺失"
                else:
                    label = f"未知错误类型({detail_type})"
                preview = ", ".join(columns[:5])
                if preview:
                    lines.append(f"- {label}: {preview}")
                else:
                    lines.append(f"- {label}: 未提供列信息")
                if len(columns) > 5:
                    lines.append(f"  ...还有 {len(columns) - 5} 列")

        warnings = sql_validation.get("warnings") or []
        if warnings:
            lines.append("")
            lines.append("**警告明细**:")
            for item in warnings[:5]:
                lines.append(f"- {item}")
            if len(warnings) > 5:
                lines.append(f"- ...还有 {len(warnings) - 5} 条警告")

    schema_validation = meta.get("schema_validation")
    if isinstance(schema_validation, dict):
        lines.append("")
        lines.append("### Schema 校验信息")
        coverage = schema_validation.get("coverage_score")
        threshold = schema_validation.get("coverage_threshold")
        if coverage is not None and threshold is not None:
            lines.append(f"- 覆盖率: {coverage:.2f}（阈值 {threshold:.2f}）")
        missing_defs = schema_validation.get("missing_definitions") or []
        if missing_defs:
            lines.append(f"- 缺失DDL: {', '.join(missing_defs[:5])}")
            if len(missing_defs) > 5:
                lines.append(f"- ...还有 {len(missing_defs) - 5} 个缺失DDL")
        missing_tables = schema_validation.get("missing_tables") or []
        if missing_tables:
            lines.append(f"- 可能缺失表: {', '.join(missing_tables[:5])}")
            if len(missing_tables) > 5:
                lines.append(f"- ...还有 {len(missing_tables) - 5} 个缺失表")
        suggestions = schema_validation.get("suggestions") or []
        if suggestions:
            lines.append("")
            lines.append("**建议**:")
            for item in suggestions[:5]:
                lines.append(f"- {item}")

    schema_report = meta.get("schema_discovery_failure_report")
    if isinstance(schema_report, dict):
        lines.append("")
        lines.append("### Schema 发现诊断")
        report_type = schema_report.get("report_type")
        if report_type:
            lines.append(f"- 报告类型: {report_type}")
        candidate_tables = schema_report.get("candidate_tables_count")
        if candidate_tables is not None:
            lines.append(f"- 候选表数量: {candidate_tables}")
        sections = schema_report.get("sections") or []
        possible_causes = []
        for section in sections:
            if isinstance(section, dict) and section.get("possible_causes"):
                possible_causes.extend(section.get("possible_causes") or [])
        if possible_causes:
            lines.append("**可能原因**:")
            for item in possible_causes[:5]:
                lines.append(f"- {item}")

    if len(lines) <= 3:
        lines.append("")
        lines.append("### 下一步建议")
        lines.append("- 检查任务描述是否包含关键业务术语和时间范围")
        lines.append("- 确认目标表与字段已导入并可被系统检索")
        lines.append("- 如需精确字段，请补充字段中文名或英文名")

    return "\n".join(lines)


def generate_sql_generation_report(
    sql_query: str,
    sql_result: str,
    row_count: int,
    metadata: Optional[Dict[str, Any]] = None,
    table_schemas: Optional[List[Any]] = None,
) -> str:
    """Generate comprehensive SQL generation report for data warehouse developers.

    6-section structure:
    1. SQL Design Overview
    2. Tables and Fields Details
    3. Annotated SQL with Comments
    4. SQL Validation Results
    5. Execution Verification Results
    6. Optimization Suggestions
    """
    lines: List[str] = []

    lines.append("## 📋 SQL生成报告（数仓开发版）\n")

    # Section 1: SQL Design Overview
    lines.append("### 1. SQL设计概述")

    clarified_task = ""
    if metadata and metadata.get("clarified_task"):
        clarified_task = metadata["clarified_task"]
    elif metadata and metadata.get("intent_clarification"):
        clarified_task = metadata["intent_clarification"].get("clarified_task", "")

    if clarified_task:
        lines.append(f"**任务理解**: {clarified_task}")
    else:
        lines.append("**任务理解**: 生成SQL查询以满足数据分析需求")

    table_count = 0
    field_count = 0
    table_info = None
    if table_schemas:
        table_info = extract_table_info(table_schemas, sql_query, logger=logger)
        table_count = len(table_info.get("tables", []))
        field_count = len(table_info.get("fields", []))

    lines.append(f"**数据规模**: 涉及 {table_count} 张表、{field_count} 个字段")

    design_logic: List[str] = []
    parsed = parse_sql_structure(sql_query)
    if parsed:
        if parsed.find(exp.With):
            design_logic.append("使用CTE组织查询逻辑")
        join_count = len(list(parsed.find_all(exp.Join)))
        if join_count > 0:
            design_logic.append(f"包含{join_count}个表关联")
        if parsed.find(exp.AggFunc):
            design_logic.append("包含聚合计算")
        if parsed.find(exp.Window):
            design_logic.append("使用窗口函数")
        if parsed.find(exp.Where):
            design_logic.append("包含筛选条件")

    if design_logic:
        lines.append("**设计思路**: " + "、".join(design_logic))
    else:
        lines.append("**设计思路**: 基于业务需求生成查询SQL")

    validation_summary: List[str] = []
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

    # Section 2: Tables and Fields Details
    lines.append("### 2. 使用的表和字段详情")
    if table_schemas:
        if table_info is None:
            table_info = extract_table_info(table_schemas, sql_query, logger=logger)
        tables = table_info.get("tables", [])
        if tables:
            lines.append(f"**表清单** ({len(tables)}张表):")
            lines.append("")
            lines.append("| 表名 | 表备注 | 表类型 | 数据库 | 是否使用 |")
            lines.append("|------|--------|--------|--------|----------|")
            for t in tables:
                lines.append(
                    "| {table} | {comment} | {t_type} | {db} | {used} |".format(
                        table=escape_markdown_table_cell(t.get("table_name", "")),
                        comment=escape_markdown_table_cell(t.get("table_comment", "-") or "-"),
                        t_type=escape_markdown_table_cell(t.get("table_type", "-") or "-"),
                        db=escape_markdown_table_cell(t.get("database", "-") or "-"),
                        used="✅" if t.get("is_used") else "-",
                    )
                )
            lines.append("")

        fields = table_info.get("fields", [])
        used_fields = [f for f in fields if f.get("is_used")]
        if used_fields:
            lines.append(f"**字段清单** ({len(used_fields)}个字段):")
            lines.append("")
            lines.append("| 表名 | 字段名 | 字段注释 | 用途 |")
            lines.append("|------|--------|----------|------|")
            for f in used_fields:
                usage = infer_field_usage(sql_query, f)
                lines.append(
                    "| {table} | {col} | {comment} | {usage} |".format(
                        table=escape_markdown_table_cell(f.get("table_name", "")),
                        col=escape_markdown_table_cell(f.get("column_name", "")),
                        comment=escape_markdown_table_cell(f.get("column_comment", "-") or "-"),
                        usage=escape_markdown_table_cell(usage),
                    )
                )
            lines.append("")

        relationships = table_info.get("relationships", [])
        if relationships:
            lines.append("**表关联关系**:")
            for rel in relationships:
                left = rel.get("left_table", "")
                right = rel.get("right_table", "")
                key = rel.get("join_key", "")
                join_type = rel.get("join_type", "INNER")
                lines.append(f"- {left} ← {key} → {right} ({join_type} JOIN)")
            lines.append("")
    else:
        lines.append("*表结构信息不可用*")
        lines.append("")

    # Section 3: Annotated SQL
    lines.append("### 3. 带注释的SQL")
    annotated_sql = generate_sql_with_comments(sql_query, table_schemas or [], metadata)
    lines.append("```sql")
    lines.append(annotated_sql)
    lines.append("```")
    lines.append("")

    # Section 4: SQL Validation Results
    lines.append("### 4. SQL验证结果")
    if metadata and metadata.get("sql_validation"):
        validation = metadata["sql_validation"]
        lines.append("| 验证项 | 状态 | 说明 |")
        lines.append("|--------|------|------|")
        syntax_valid = validation.get("syntax_valid", True)
        lines.append(
            f"| 语法验证 | {'✅ 通过' if syntax_valid else '❌ 失败'} | "
            f"{'SQL语法正确，符合SQL方言规范' if syntax_valid else 'SQL语法错误，请检查语句'} |"
        )
        tables_exist = validation.get("tables_exist", True)
        lines.append(
            f"| 表存在性 | {'✅ 通过' if tables_exist else '❌ 失败'} | "
            f"{'所有表都在Schema中存在' if tables_exist else '部分表不存在，请检查表名'} |"
        )
        columns_exist = validation.get("columns_exist", True)
        lines.append(
            f"| 列存在性 | {'✅ 通过' if columns_exist else '❌ 失败'} | "
            f"{'所有列都在对应表中存在' if columns_exist else '部分列不存在，请检查列名'} |"
        )
        has_dangerous = validation.get("has_dangerous_ops", False)
        lines.append(
            f"| 危险操作 | {'⚠️ 检测到' if has_dangerous else '✅ 无危险操作'} | "
            f"{'检测到DELETE/DROP/TRUNCATE等操作，请谨慎执行' if has_dangerous else '未检测到危险操作，可安全执行'} |"
        )
        lines.append("")
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

    # Section 5: Execution Verification Results
    lines.append("### 5. 执行验证结果")
    lines.append(generate_execution_report(row_count, metadata))

    # Section 6: Optimization Suggestions
    lines.append("### 6. 优化建议")
    optimization = generate_optimization_suggestions(sql_query, table_schemas or [], metadata)
    if optimization:
        lines.append(optimization)
    else:
        lines.append("*无优化建议*")
        lines.append("")

    return "\n".join(lines)


def format_diagnostic_report(report: Dict[str, Any]) -> str:
    """Format schema discovery failure report for user display.

    Args:
        report: Diagnostic report dictionary from schema_validation_node

    Returns:
        Markdown formatted diagnostic report
    """
    lines = []

    # Header
    lines.append("## ❌ Schema Discovery Failure Report\n")
    lines.append(f"**Report Type**: {report.get('report_type', 'Unknown')}\n")
    lines.append(f"**Timestamp**: {report.get('timestamp', 'Unknown')}\n")
    lines.append(f"**Database**: {report.get('database_name', 'Unknown')}\n")
    lines.append(f"**Namespace**: {report.get('namespace', 'Unknown')}\n")
    lines.append(f"**Task**: {report.get('task', 'Unknown')[:100]}...\n")

    # Format sections
    sections = report.get("sections", [])
    for section in sections:
        lines.append(f"### {section.get('title', 'Unknown Section')}\n")

        # Handle different section types
        if "findings" in section:
            findings = section["findings"]
            lines.append("**Findings**:")
            for key, value in findings.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        elif "possible_causes" in section:
            causes = section["possible_causes"]
            lines.append("**Possible Causes**:")
            for i, cause in enumerate(causes, 1):
                lines.append(f"{i}. {cause}")
            lines.append("")

        elif "steps" in section:
            steps = section["steps"]
            lines.append("**Steps**:")
            for step in steps:
                lines.append(f"- {step}")
            lines.append("")

        elif "commands" in section:
            commands = section["commands"]
            lines.append("**Commands**:")
            for cmd in commands:
                lines.append(f"```bash")
                lines.append(cmd)
                lines.append("```")
            lines.append("")

        elif "sql_query" in section:
            lines.append(f"**SQL Query**: `{section.get('sql_query', 'No SQL generated')[:100]}`")
            if "warning" in section:
                lines.append(f"\n⚠️ **Warning**: {section['warning']}")
            lines.append("")

        elif "recommendations" in section:
            recommendations = section["recommendations"]
            lines.append("**Recommendations**:")
            for rec in recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        else:
            # Generic section handling
            for key, value in section.items():
                if key != "title":
                    lines.append(f"**{key}**: {value}")
            lines.append("")

    return "\n".join(lines)


def escape_markdown_table_cell(text: Any) -> str:
    """Escape text for use in a Markdown table cell.

    Args:
        text: Input text (will be converted to string)

    Returns:
        Escaped string safe for Markdown table
    """
    if text is None:
        return "-"
    s = str(text)
    # Replace pipes with broken bar or escaped pipe
    s = s.replace("|", "&#124;")
    # Replace newlines with space
    s = s.replace("\n", " ")
    return s.strip()


def parse_ddl_comments(ddl: str, dialect: str = "snowflake", logger=None) -> Dict[str, Any]:
    """Parse DDL to extract table and column comments.

    Args:
        ddl: DDL statement (CREATE TABLE ...)
        dialect: SQL dialect (snowflake, mysql, postgres, etc.)
        logger: Logger instance

    Returns:
        Dict with table_comment and columns dict mapping name->comment
    """
    result = {
        "table_comment": "",
        "columns": {},
    }

    if not ddl:
        return result

    cleaned = sanitize_ddl_for_storage(ddl)
    tried_dialects = []

    if dialect:
        tried_dialects.append(parse_dialect(dialect))
    tried_dialects.extend([DBType.STARROCKS, DBType.MYSQL, DBType.SNOWFLAKE])
    tried_dialects = [d for d in tried_dialects if d]

    last_error = None
    for candidate in tried_dialects:
        try:
            metadata = extract_enhanced_metadata_from_ddl(
                cleaned,
                dialect=candidate,
                warn_on_invalid=False,
            )
            table_comment = metadata.get("table", {}).get("comment", "") if metadata else ""
            columns = metadata.get("columns", []) if metadata else []
            if table_comment or columns:
                result["table_comment"] = table_comment or ""
                for col in columns:
                    col_name = col.get("name")
                    if col_name:
                        result["columns"][col_name] = col.get("comment", "")
                return result
        except Exception as e:
            last_error = e

    if last_error and logger:
        logger.warning(f"Failed to parse DDL comments: {last_error}")

    return result


def extract_table_info(table_schemas: List[Any], sql_query: str, logger=None) -> Dict[str, Any]:
    """Extract table and field information from table_schemas and SQL.

    Args:
        table_schemas: List of TableSchema objects with DDL definitions
        sql_query: SQL query to analyze for field usage
        logger: Logger instance

    Returns:
        Dict with tables list, fields list, and relationships
    """
    tables_info = []
    fields_info = []
    seen_tables = set()
    seen_fields = set()

    if not table_schemas:
        return {"tables": tables_info, "fields": fields_info, "relationships": []}

    # Parse SQL to extract used tables and columns
    sql_tables = set()
    sql_columns = set()
    symbols = extract_sql_symbols(sql_query)
    virtual_columns = symbols.get("virtual_columns", set())
    try:
        parsed = sqlglot.parse_one(sql_query, error_level=sqlglot.ErrorLevel.IGNORE)
        # Find all table references
        for table in parsed.find_all(exp.Table):
            sql_tables.add(table.name)
        # Find all column references
        for column in parsed.find_all(exp.Column):
            sql_columns.add(column.name)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to parse SQL for table/column extraction: {e}")

    # Extract table information from DDLs
    for schema in table_schemas:
        table_name = getattr(schema, "table_name", "")
        definition = getattr(schema, "definition", "")
        database_name = getattr(schema, "database_name", "")
        table_type = getattr(schema, "table_type", "table")
        schema_name = getattr(schema, "schema_name", "")
        catalog_name = getattr(schema, "catalog_name", "")
        identifier = getattr(schema, "identifier", "")

        if not table_name or not definition:
            continue

        # Parse DDL for comments
        ddl_info = parse_ddl_comments(definition, logger=logger)

        dedupe_key = identifier or (catalog_name, database_name, schema_name, table_name, table_type)
        if dedupe_key in seen_tables:
            continue
        seen_tables.add(dedupe_key)

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
            field_key = (table_name, col_name)
            if field_key in seen_fields:
                continue
            seen_fields.add(field_key)
            is_used = (col_name in sql_columns) or (col_name.lower() in virtual_columns)
            fields_info.append({
                "table_name": table_name,
                "column_name": col_name,
                "column_comment": col_comment,
                "is_used": is_used
            })

    # Analyze relationships (JOIN keys)
    relationships = analyze_relationships(sql_query, tables_info)

    return {
        "tables": tables_info,
        "fields": fields_info,
        "relationships": relationships
    }


def analyze_relationships(sql_query: str, tables_info: List[Dict]) -> List[Dict[str, str]]:
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
                    left = on_clause.this
                    right = on_clause.expression
                    if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                        left_table = left.table
                        right_table = right.table
                        join_key = left.name if left.name == right.name else f"{left.name} = {right.name}"
                        join_type = str(join.side).upper() if join.side else "INNER"

                        relationships.append({
                            "left_table": left_table,
                            "right_table": right_table or join_table,
                            "join_key": join_key,
                            "join_type": join_type
                        })
    except Exception:
        pass

    return relationships


def parse_sql_structure(sql_query: str, dialect: str = "snowflake") -> Optional[exp.Expression]:
    """Parse SQL query into structured AST.

    Args:
        sql_query: SQL query string
        dialect: SQL dialect

    Returns:
        Parsed SQL expression or None if parsing fails
    """
    try:
        return sqlglot.parse_one(sql_query, dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception:
        return None


def infer_field_usage(sql_query: str, field_info: Dict) -> str:
    """Infer how a field is used in the SQL query.

    Args:
        sql_query: SQL query string
        field_info: Field information dict

    Returns:
        Usage description string
    """
    field_name = field_info.get("column_name", "")
    if not field_name:
        return "未知用途"

    try:
        parsed = sqlglot.parse_one(sql_query, error_level=sqlglot.ErrorLevel.IGNORE)
        if not parsed:
            return "未知用途"

        # Check if field is in SELECT
        for select in parsed.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Column) and projection.name == field_name:
                    return "输出字段"

        # Check if field is in WHERE clause
        for where in parsed.find_all(exp.Where):
            if field_name in str(where):
                return "筛选条件"

        # Check if field is in JOIN condition
        for join in parsed.find_all(exp.Join):
            if field_name in str(join):
                return "关联键"

        # Check if field is in GROUP BY
        for group in parsed.find_all(exp.Group):
            for expression in group.expressions:
                if isinstance(expression, exp.Column) and expression.name == field_name:
                    return "分组字段"

        return "未在查询中使用"
    except Exception:
        return "未知用途"


def get_field_comment(table_schemas: List[Any], table_name: str, column_name: str) -> str:
    """Get field comment from table schemas.

    Args:
        table_schemas: List of TableSchema objects
        table_name: Table name
        column_name: Column name

    Returns:
        Column comment or empty string
    """
    if not table_schemas:
        return ""

    for schema in table_schemas:
        if getattr(schema, "table_name", "") == table_name:
            definition = getattr(schema, "definition", "")
            if definition:
                ddl_info = parse_ddl_comments(definition)
                return ddl_info["columns"].get(column_name, "")

    return ""


def infer_cte_purpose(cte_name: str, cte_definition: str) -> str:
    """Infer business purpose of a CTE from its name and definition."""
    patterns = {
        r"first|initial|earliest": "识别首次事件",
        r"last|final|latest": "识别最后事件",
        r"rank|row_number": "计算排名或序号",
        r"agg|aggregate|sum|count|avg": "聚合计算",
        r"filter|where": "筛选数据",
        r"join|link|relate": "关联表数据",
        r"dedup|distinct|unique": "去重或获取唯一值",
    }
    cte_lower = cte_name.lower() if cte_name else ""
    for pattern, purpose in patterns.items():
        if re.search(pattern, cte_lower):
            return purpose
    if "SELECT" in (cte_definition or "").upper():
        return "中间查询结果"
    return "通用表达式"


def add_field_comment(field_name: str, field_comment: str, sql_line: str) -> str:
    """Add inline comment to a field in SQL."""
    if not field_comment or "--" in sql_line:
        return sql_line
    if len(field_name) > 256 or len(sql_line) > 4096:
        return sql_line
    pattern = rf"\\b{re.escape(field_name)}\\b(?!\\s*--)"
    replacement = f"{field_name} -- {field_comment}"
    return re.sub(pattern, replacement, sql_line, count=1)


def explain_condition(condition: exp.Expression) -> str:
    """Explain business meaning of a WHERE/JOIN condition."""
    if isinstance(condition, exp.EQ):
        left = condition.left
        right = condition.right
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            return f"筛选 {left.name} 等于 {right.this}"
    elif isinstance(condition, exp.In):
        col = condition.this
        if isinstance(col, exp.Column):
            return f"筛选 {col.name} 在指定值范围内"
    elif isinstance(condition, exp.And):
        return "同时满足多个条件"
    elif isinstance(condition, exp.Or):
        return "满足任一条件"
    return "条件筛选"


def add_condition_comments(sql_lines: List[str], parsed: exp.Expression) -> List[str]:
    """Add business logic comments to WHERE/JOIN conditions."""
    result = sql_lines.copy()
    for i, line in enumerate(result):
        if "WHERE" in line.upper() or "AND" in line.upper() or "OR" in line.upper():
            for where in parsed.find_all(exp.Where):
                explanation = explain_condition(where.this)
                if explanation and explanation != "条件筛选":
                    result[i] = f"-- {explanation}\n{result[i]}"
                    break
    return result


def generate_sql_with_comments(
    sql_query: str,
    table_schemas: List[Any],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Generate annotated SQL with business logic comments."""
    lines: List[str] = []
    sql_lines = sql_query.strip().split("\n")

    try:
        parsed = parse_sql_structure(sql_query)
        if not parsed:
            return sql_query

        clarified_task = ""
        if metadata and metadata.get("clarified_task"):
            clarified_task = metadata["clarified_task"]
        elif metadata and metadata.get("intent_clarification"):
            clarified_task = metadata["intent_clarification"].get("clarified_task", "")

        if clarified_task:
            lines.append(f"-- SQL设计目的: {clarified_task}")
            lines.append("")

        with_expr = parsed.find(exp.With)
        if with_expr:
            lines.append("-- 使用公共表表达式(CTE)组织复杂查询逻辑")
            for cte in with_expr.expressions:
                if isinstance(cte, exp.CTE):
                    cte_name = cte.alias
                    cte_purpose = infer_cte_purpose(cte_name, str(cte.this))
                    lines.append(f"-- CTE: {cte_name} - {cte_purpose}")

        for line in sql_lines:
            annotated_line = line
            for table_schema in table_schemas:
                table_name = getattr(table_schema, "table_name", "")
                definition = getattr(table_schema, "definition", "")
                if table_name and definition:
                    ddl_info = parse_ddl_comments(definition)
                    for col_name, col_comment in ddl_info["columns"].items():
                        if col_comment and col_name in line and "--" not in line:
                            annotated_line = add_field_comment(col_name, col_comment, line)
            lines.append(annotated_line)

        lines = add_condition_comments(lines, parsed)
    except Exception as e:
        logger.warning(f"Failed to generate annotated SQL: {e}")
        return sql_query

    return "\n".join(lines)


def generate_execution_report(row_count: int, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate execution verification report section."""
    lines: List[str] = []
    meta = metadata if isinstance(metadata, dict) else {}
    failure_stage = meta.get("failure_stage")
    termination_reason = meta.get("termination_reason")

    sql_validation = meta.get("sql_validation") if isinstance(meta.get("sql_validation"), dict) else {}
    result_validation = meta.get("result_validation") if isinstance(meta.get("result_validation"), dict) else {}

    syntax_valid = sql_validation.get("syntax_valid", True)
    tables_exist = sql_validation.get("tables_exist", True)
    columns_exist = sql_validation.get("columns_exist", True)
    has_dangerous = sql_validation.get("has_dangerous_ops", False)
    sql_validation_ok = bool(syntax_valid and tables_exist and columns_exist and not has_dangerous)

    result_is_valid = result_validation.get("is_valid", True)
    result_reason = result_validation.get("reason") or result_validation.get("suggestion") or ""

    execution_failed = failure_stage in ("execute_sql", "sql_execution")
    sql_validation_failed = failure_stage == "sql_validation" or not sql_validation_ok
    result_validation_failed = (
        failure_stage == "result_validation"
        or (result_validation != {} and not result_is_valid)
    )
    validation_failed = sql_validation_failed or result_validation_failed

    if execution_failed:
        lines.append("**执行状态**: ❌ SQL执行失败\n")
        lines.append("**执行详情**:")
        if termination_reason:
            lines.append(f"- **失败原因**: {termination_reason}")
        else:
            lines.append("- **失败原因**: 数据库执行失败（未返回结果）")
        lines.append("- **执行返回**: 未返回结果")
        lines.append("")
        lines.append("**SQL适合生产使用**: ❌ 否")
        lines.append("")
        return "\n".join(lines)

    if validation_failed:
        lines.append("**执行状态**: ⚠️ SQL执行完成，但验证未通过\n")
        lines.append("**执行详情**:")
        lines.append(
            f"- **语法正确**: {'✅ SQL语法验证通过，数据库成功解析' if syntax_valid else '❌ 语法验证失败'}"
        )
        lines.append(f"- **执行返回**: {row_count}行数据")
        lines.append("")
        lines.append("**验证失败类型**:")
        if sql_validation_failed:
            lines.append("- **SQL验证**: ❌ 未通过")
            if not syntax_valid:
                lines.append("  - 原因: SQL语法解析失败")
            if not tables_exist:
                lines.append("  - 原因: 存在未匹配的表")
            if not columns_exist:
                lines.append("  - 原因: 存在未匹配的字段")
            if has_dangerous:
                lines.append("  - 原因: 检测到危险操作")
        if result_validation_failed:
            lines.append("- **结果验证**: ❌ 未通过")
            if result_reason:
                lines.append(f"  - 原因: {result_reason}")
        lines.append("")
        lines.append("**SQL适合生产使用**: ⚠️ 需修正后再使用")
        lines.append("")
        return "\n".join(lines)

    lines.append("**执行状态**: ✅ SQL已成功执行验证\n")
    lines.append("**执行详情**:")
    lines.append(
        f"- **语法正确**: {'✅ SQL语法验证通过，数据库成功解析' if syntax_valid else '❌ 语法验证失败'}"
    )
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


def generate_optimization_suggestions(
    sql_query: str,
    table_schemas: List[Any],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Generate optimization suggestions based on SQL analysis."""
    suggestions: List[str] = []
    lines: List[str] = []

    try:
        parsed = parse_sql_structure(sql_query)
        if not parsed:
            return ""

        has_cte = parsed.find(exp.With) is not None
        if has_cte:
            suggestions.append("✅ 使用了CTE，提高了SQL可读性和维护性")

        join_count = len(list(parsed.find_all(exp.Join)))
        if join_count > 0:
            suggestions.append(f"✅ 包含{join_count}个表关联，建议确保关联字段有索引")

        subquery_count = len(list(parsed.find_all(exp.Subquery)))
        if subquery_count > 2:
            suggestions.append("💡 包含多个子查询，考虑使用CTE重构以提高可读性")

        for select in parsed.find_all(exp.Select):
            if hasattr(select, "expressions"):
                for expr in select.expressions:
                    if isinstance(expr, exp.Star):
                        suggestions.append("⚠️ 使用了SELECT *，建议明确指定所需字段以提高性能")
                        break

        for select in parsed.find_all(exp.Select):
            has_where = select.find(exp.Where) is not None
            if not has_where and join_count == 0:
                suggestions.append("💡 查询未包含WHERE条件，将扫描全表数据")

        if metadata and metadata.get("sql_validation"):
            validation = metadata["sql_validation"]
            warnings = validation.get("warnings", [])
            if warnings:
                suggestions.extend([f"⚠️ {w}" for w in warnings[:3]])
    except Exception as e:
        logger.warning(f"Failed to generate optimization suggestions: {e}")

    if suggestions:
        lines.append("**性能优化**:")
        for s in suggestions:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("**后续分析建议**:")
        lines.append("- 根据实际数据量调整查询复杂度")
        lines.append("- 定期检查查询执行计划，优化索引策略")
        lines.append("- 对于大数据集查询，考虑添加时间范围限制")
        lines.append("")

    return "\n".join(lines) if lines else ""
