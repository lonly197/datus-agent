# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional, Union

from datus.schemas.node_models import Metric, TableSchema, TableValue
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

from ..utils.json_utils import to_pretty_str
from .prompt_manager import prompt_manager

logger = get_logger(__name__)


def _safe_parse_json(value: Optional[str]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value or not isinstance(value, str):
        return {}
    try:
        import json

        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _should_skip_text(text: str, haystack: str) -> bool:
    if not text:
        return True
    return text.strip() in haystack


def build_schema_metadata_summary(
    table_schemas: List[TableSchema],
    processed_schemas: str,
    max_per_table: int = 800,
    max_total: int = 4000,
) -> str:
    if not table_schemas:
        return ""

    haystack = processed_schemas or ""
    parts: List[str] = []
    total_len = 0

    for schema in table_schemas:
        block_lines: List[str] = []
        has_extra = False
        title = f"Table: {schema.table_name}"
        if schema.table_comment and not _should_skip_text(schema.table_comment, haystack):
            title = f"{title} - {schema.table_comment}"
            has_extra = True
        block_lines.append(title)

        if schema.business_tags:
            block_lines.append(f"Tags: {', '.join(schema.business_tags)}")
            has_extra = True

        column_comments = _safe_parse_json(schema.column_comments)
        if column_comments:
            items = [f"{k}({v})" for k, v in column_comments.items() if v and not _should_skip_text(v, haystack)]
            if items:
                block_lines.append(f"Columns: {', '.join(items[:10])}")
                has_extra = True

        column_enums = _safe_parse_json(schema.column_enums)
        if column_enums:
            enum_items = []
            for col, enums in column_enums.items():
                if not enums:
                    continue
                values = [str(item.get("value", "")) for item in enums if isinstance(item, dict)]
                values = [v for v in values if v]
                if values:
                    enum_items.append(f"{col}=[{', '.join(values[:10])}]")
            if enum_items:
                block_lines.append(f"Enums: {', '.join(enum_items[:5])}")
                has_extra = True

        relationship_metadata = _safe_parse_json(schema.relationship_metadata)
        if relationship_metadata:
            join_paths = relationship_metadata.get("join_paths") or relationship_metadata.get("foreign_keys")
            if join_paths:
                if isinstance(join_paths, list):
                    join_preview = ", ".join([str(item) for item in join_paths[:5]])
                else:
                    join_preview = str(join_paths)
                block_lines.append(f"Relations: {join_preview}")
                has_extra = True

        if schema.row_count:
            block_lines.append(f"Stats: row_count={schema.row_count}")
            has_extra = True

        block = "\n".join(block_lines)
        if not has_extra:
            continue
        if len(block) > max_per_table:
            block = block[:max_per_table] + "\n... (truncated)"

        if total_len + len(block) > max_total:
            break
        parts.append(block)
        total_len += len(block) + 1

    return "\n\n".join(parts)


def get_sql_prompt(
    database_type: str,
    table_schemas: Union[List[TableSchema], str],
    data_details: List[TableValue],
    metrics: List[Metric],
    question: str,
    external_knowledge: str = "",
    prompt_version: str = "1.1",
    context=None,
    max_table_schemas_length: int = 4000,
    max_data_details_length: int = 2000,
    max_context_length: int = 8000,
    max_value_length: int = 500,
    max_text_mark_length: int = 16,
    database_docs: str = "",
    current_date: str = None,
    date_ranges: str = "",
    include_schema_ddl: bool = False,
    validation_summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    if context is None:
        context = []

    if isinstance(table_schemas, str):
        processed_schemas = table_schemas
    else:
        processed_schemas = _render_schemas_with_budget(
            table_schemas,
            database_type,
            include_schema_ddl,
            max_table_schemas_length,
            validation_summary,
        )

    processed_schema_metadata = ""
    if not isinstance(table_schemas, str):
        processed_schema_metadata = build_schema_metadata_summary(table_schemas, processed_schemas)

    if data_details:
        processed_details = "\n---\n".join(
            detail.to_prompt(database_type, max_value_length, max_text_mark_length, processed_schemas)
            for detail in data_details
        )
    else:
        processed_details = ""

    processed_context = str(context)

    # Truncate if exceeds max length
    if len(processed_schemas) > max_table_schemas_length:
        logger.warning("Table schemas is too long, truncating to %s characters" % max_table_schemas_length)
        processed_schemas = processed_schemas[:max_table_schemas_length] + "\n... (truncated)"

    if len(processed_details) > max_data_details_length:
        logger.warning("Data details is too long, truncating to %s characters" % max_data_details_length)
        processed_details = processed_details[:max_data_details_length] + "\n... (truncated)"

    if len(processed_context) > max_context_length:
        logger.warning("Context is too long, truncating to %s characters" % max_context_length)
        processed_context = processed_context[:max_context_length] + "\n... (truncated)"

    # Add Snowflake specific notes
    database_notes = ""
    knowledge_content = "" if not external_knowledge else f"External Knowledge:\n{external_knowledge}"
    if database_type.lower() == DBType.SNOWFLAKE.value.lower():
        database_notes = (
            "\nEnclose all column names in double quotes to comply with Snowflake syntax requirements and avoid erros. "
            "When referencing table names in Snowflake SQL, you must include both the database_name and schema_name."
        )
    elif database_type.lower() == DBType.STARROCKS.value.lower():
        database_notes = ""

    processed_metrics = ""
    if metrics:
        processed_metrics = to_pretty_str([m.__dict__ for m in metrics])
    processed_validation = ""
    if validation_summary:
        processed_validation = to_pretty_str(validation_summary)

    system_content = prompt_manager.get_raw_template("gen_sql_system", version=prompt_version)
    user_content = prompt_manager.render_template(
        "gen_sql_user",
        database_type=database_type,
        database_notes=database_notes,
        processed_schemas=processed_schemas,
        processed_schema_metadata=processed_schema_metadata,
        processed_details=processed_details,
        metrics=processed_metrics,
        knowledge_content=knowledge_content,
        question=question,
        version=prompt_version,
        processed_context=processed_context,
        database_docs=database_docs,
        current_date=current_date,
        date_ranges=date_ranges,
        validation_summary=processed_validation,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _render_schemas_with_budget(
    table_schemas: List[TableSchema],
    database_type: str,
    include_schema_ddl: bool,
    max_length: int,
    validation_summary: Optional[Dict[str, Any]],
) -> str:
    priority_tables: List[str] = []
    if validation_summary and isinstance(validation_summary, dict):
        table_coverage = validation_summary.get("table_coverage", {})
        if isinstance(table_coverage, dict):
            priority_tables = sorted(
                table_coverage.keys(),
                key=lambda name: len(table_coverage.get(name, [])),
                reverse=True,
            )

    schema_by_name = {schema.table_name: schema for schema in table_schemas}
    ordered_schemas: List[TableSchema] = []
    for table_name in priority_tables:
        schema = schema_by_name.get(table_name)
        if schema:
            ordered_schemas.append(schema)

    for schema in table_schemas:
        if schema not in ordered_schemas:
            ordered_schemas.append(schema)

    parts: List[str] = []
    current_len = 0
    truncated = False
    for schema in ordered_schemas:
        full_prompt = schema.to_prompt(database_type, include_ddl=include_schema_ddl)
        prompt_to_use = full_prompt
        if include_schema_ddl and current_len + len(prompt_to_use) > max_length:
            prompt_to_use = schema.to_prompt(database_type, include_ddl=False)
        if current_len + len(prompt_to_use) > max_length:
            if not parts and max_length > 0:
                parts.append(prompt_to_use[:max_length] + "\n... (truncated)")
            else:
                truncated = True
            break
        parts.append(prompt_to_use)
        current_len += len(prompt_to_use) + 1

    if truncated:
        parts.append("... (truncated)")

    return "\n".join(parts)
