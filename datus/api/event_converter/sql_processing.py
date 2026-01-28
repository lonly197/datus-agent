# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SQL processing utilities for event conversion.

This module provides functions to generate SQL reports, parse DDL,
analyze relationships, and create annotated SQL with comments.
"""

from typing import Any, Dict, List, Optional

import sqlglot
from sqlglot import exp

from datus.utils.constants import DBType
from datus.utils.sql_utils import (
    extract_enhanced_metadata_from_ddl,
    parse_dialect,
    sanitize_ddl_for_storage,
)
from datus.utils.loggings import get_logger


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
            is_used = col_name in sql_columns
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
