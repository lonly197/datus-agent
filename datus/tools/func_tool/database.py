# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage.metric.store import SemanticMetricsRAG
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.db_tools import BaseSqlConnector
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.compress_utils import DataCompressor
from datus.utils.constants import SUPPORT_DATABASE_DIALECTS, SUPPORT_SCHEMA_DIALECTS, DBType


@dataclass
class TableCoordinate:
    catalog: str = ""
    database: str = ""
    schema: str = ""
    table: str = ""


@dataclass(frozen=True)
class ScopedTablePattern:
    raw: str
    catalog: str = ""
    database: str = ""
    schema: str = ""
    table: str = ""

    def matches(self, coordinate: TableCoordinate) -> bool:
        return all(
            _pattern_matches(getattr(self, field), getattr(coordinate, field))
            for field in ("catalog", "database", "schema", "table")
        )


def _pattern_matches(pattern: str, value: str) -> bool:
    if not pattern or pattern in ("*", "%"):
        return True
    normalized_pattern = pattern.replace("%", "*")
    return fnmatchcase(value or "", normalized_pattern)


class DBFuncTool:
    def __init__(
        self,
        connector: BaseSqlConnector,
        agent_config: Optional[AgentConfig] = None,
        *,
        sub_agent_name: Optional[str] = None,
        scoped_tables: Optional[Iterable[str]] = None,
    ):
        self.connector = connector
        self.compressor = DataCompressor()
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.schema_rag = SchemaWithValueRAG(agent_config, sub_agent_name) if agent_config else None
        self._field_order = self._determine_field_order()
        self._scoped_patterns = self._load_scoped_patterns(scoped_tables)
        self._semantic_storage = SemanticMetricsRAG(agent_config, sub_agent_name) if agent_config else None
        self.has_schema = self.schema_rag and self.schema_rag.schema_store.table_size() > 0
        self.has_semantic_models = self._semantic_storage and self._semantic_storage.get_semantic_model_size() > 0

    def _reset_database_for_rag(self, database_name: str = "") -> str:
        if self.connector.dialect in (DBType.SQLITE, DBType.DUCKDB):
            return self.connector.database_name
        else:
            return database_name

    def _determine_field_order(self) -> Sequence[str]:
        dialect = getattr(self.connector, "dialect", "") or ""
        fields: List[str] = []
        if DBType.support_catalog(dialect):
            fields.append("catalog")
        if DBType.support_database(dialect) or dialect == DBType.SQLITE:
            fields.append("database")
        if DBType.support_schema(dialect):
            fields.append("schema")
        fields.append("table")
        return fields

    def _load_scoped_patterns(self, explicit_tokens: Optional[Iterable[str]]) -> List[ScopedTablePattern]:
        tokens: List[str] = []
        if explicit_tokens:
            tokens.extend(explicit_tokens)
        else:
            tokens.extend(self._resolve_scoped_context_tables())

        patterns: List[ScopedTablePattern] = []
        for token in tokens:
            scoped_pattern = self._parse_scope_token(token)
            if scoped_pattern:
                patterns.append(scoped_pattern)
        return patterns

    def _resolve_scoped_context_tables(self) -> Sequence[str]:
        if not self.agent_config:
            return []
        scoped_entries: List[str] = []

        if self.sub_agent_name:
            sub_agent_config = self._load_sub_agent_config(self.sub_agent_name)
            if sub_agent_config and sub_agent_config.scoped_context and sub_agent_config.scoped_context.tables:
                scoped_entries.extend(sub_agent_config.scoped_context.as_lists().tables)

        return scoped_entries

    def _load_sub_agent_config(self, sub_agent_name: str) -> Optional[SubAgentConfig]:
        if not self.agent_config:
            return None
        try:
            config = self.agent_config.sub_agent_config(sub_agent_name)
        except Exception:
            return None

        if not config:
            return None
        if isinstance(config, SubAgentConfig):
            return config

        try:
            return SubAgentConfig.model_validate(config)
        except Exception:
            return None

    def _parse_scope_token(self, token: str) -> Optional[ScopedTablePattern]:
        token = (token or "").strip()
        if not token:
            return None
        parts = [self._normalize_identifier_part(part) for part in token.split(".") if part.strip()]
        if not parts:
            return None
        values: Dict[str, str] = {field: "" for field in self._field_order}
        for idx, part in enumerate(parts[: len(self._field_order)]):
            field = self._field_order[idx]
            values[field] = part
        return ScopedTablePattern(raw=token, **values)

    def _get_semantic_model(
        self, catalog: str = "", database: str = "", schema: str = "", table_name: str = ""
    ) -> Dict[str, Any]:
        if not self.has_semantic_models:
            return {}
        result = self._semantic_storage.get_semantic_model(
            catalog_name=catalog,
            database_name=database,
            schema_name=schema,
            table_name=table_name,
            select_fields=["semantic_model_name", "dimensions", "measures", "semantic_model_desc"],
        )
        return {} if not result else result[0]

    @staticmethod
    def _normalize_identifier_part(value: Optional[str]) -> str:
        if value is None:
            return ""
        normalized = str(value).strip()
        if not normalized:
            return ""
        # Strip common quoting characters
        return normalized.strip("`\"'[]")

    def _default_field_value(self, field: str, explicit: Optional[str]) -> str:
        if field not in self._field_order:
            return ""
        if explicit:
            return self._normalize_identifier_part(explicit)

        fallback_attr_map = {
            "catalog": "catalog_name",
            "database": "database_name",
            "schema": "schema_name",
        }
        fallback_attr = fallback_attr_map.get(field)
        if fallback_attr and hasattr(self.connector, fallback_attr):
            return self._normalize_identifier_part(getattr(self.connector, fallback_attr))
        return ""

    def _build_table_coordinate(
        self,
        raw_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema: Optional[str] = "",
    ) -> TableCoordinate:
        coordinate = TableCoordinate(
            catalog=self._default_field_value("catalog", catalog),
            database=self._default_field_value("database", database),
            schema=self._default_field_value("schema", schema),
            table=self._normalize_identifier_part(raw_name),
        )
        parts = [self._normalize_identifier_part(part) for part in raw_name.split(".") if part.strip()]
        if parts:
            coordinate.table = parts[-1]
            idx = len(parts) - 2
            for field in reversed(self._field_order[:-1]):
                if idx < 0:
                    break
                setattr(coordinate, field, parts[idx])
                idx -= 1
        return coordinate

    def _table_matches_scope(self, coordinate: TableCoordinate) -> bool:
        if not self._scoped_patterns:
            return True
        return any(pattern.matches(coordinate) for pattern in self._scoped_patterns)

    def _filter_table_entries(
        self,
        entries: Sequence[Dict[str, Any]],
        catalog: Optional[str],
        database: Optional[str],
        schema: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not self._scoped_patterns:
            return list(entries)

        filtered: List[Dict[str, Any]] = []
        for entry in entries:
            coordinate = self._build_table_coordinate(
                raw_name=str(entry.get("name", "")),
                catalog=catalog,
                database=database,
                schema=schema,
            )
            if self._table_matches_scope(coordinate):
                filtered.append(entry)
        return filtered

    def _matches_catalog_database(self, pattern: ScopedTablePattern, catalog: str, database: str) -> bool:
        if pattern.catalog and not _pattern_matches(pattern.catalog, catalog):
            return False
        if pattern.database and not _pattern_matches(pattern.database, database):
            return False
        return True

    def _database_matches_scope(self, catalog: Optional[str], database: str) -> bool:
        if not self._scoped_patterns:
            return True
        catalog_value = self._default_field_value("catalog", catalog or "")
        database_value = self._default_field_value("database", database or "")

        wildcard_allowed = False
        for pattern in self._scoped_patterns:
            if not self._matches_catalog_database(pattern, catalog_value, database_value):
                continue
            if pattern.database:
                if _pattern_matches(pattern.database, database_value):
                    return True
                continue
            wildcard_allowed = True
        return wildcard_allowed

    def _schema_matches_scope(self, catalog: Optional[str], database: Optional[str], schema: str) -> bool:
        if not self._scoped_patterns:
            return True
        catalog_value = self._default_field_value("catalog", catalog or "")
        database_value = self._default_field_value("database", database or "")
        schema_value = self._default_field_value("schema", schema or "")

        wildcard_allowed = False
        for pattern in self._scoped_patterns:
            if not self._matches_catalog_database(pattern, catalog_value, database_value):
                continue
            if pattern.schema:
                if _pattern_matches(pattern.schema, schema_value):
                    return True
                continue
            wildcard_allowed = True
        return wildcard_allowed

    def available_tools(self) -> List[Tool]:
        bound_tools = []
        methods_to_convert: List[Callable] = [self.list_tables, self.describe_table]

        if self.schema_rag:
            methods_to_convert.append(self.search_table)

        methods_to_convert.extend(
            [
                self.read_query,
                self.get_table_ddl,
            ]
        )

        if self.connector.dialect in SUPPORT_DATABASE_DIALECTS:
            bound_tools.append(trans_to_function_tool(self.list_databases))

        if self.connector.dialect in SUPPORT_SCHEMA_DIALECTS:
            bound_tools.append(trans_to_function_tool(self.list_schemas))

        for bound_method in methods_to_convert:
            bound_tools.append(trans_to_function_tool(bound_method))
        return bound_tools

    def search_table(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n: int = 5,
        simple_sample_data: bool = True,
    ) -> FuncToolResult:
        """
        Retrieve table candidates by semantic similarity over stored schema metadata and optional sample rows.
        Use this tool  when the agent needs tables matching a natural-language description.
        This tool  helps find relevant tables by searching through table names, schemas (DDL),
        and sample data using semantic search.

        Use this tool when you need to:
        - Find tables related to a specific business concept or domain
        - Discover tables containing certain types of data
        - Locate tables for SQL query development
        - Understand what tables are available in a database

        **Application Guidance**:
        1. If table matches (via definition/description/dimensions/measures/sample_data), use it directly
        2. If partitioned (e.g., date-based in definition), explore correct partition via describe_table
        3. If no match, use list_tables for broader exploration

        Args:
            query_text: Description of the table you want (e.g. "daily active users per country").
            catalog_name: Optional catalog filter to narrow the search.
            database_name: Optional database filter to narrow the search.
            schema_name: Optional schema filter to narrow the search.
            top_n: Maximum number of rows to return after scoping filters.
            simple_sample_data: If True, sample rows omit catalog/database/schema fields for brevity.

        Returns:
            FuncToolResult where:
                - success=1 with result={"metadata": [...], "sample_data": [...]} when matches remain after filtering.
                - success=1 with result=[] and error message when no candidates survive the filters.
                - success=0 with error text if schema storage is unavailable or lookup fails.
        """
        if not self.has_schema:
            return FuncToolResult(success=0, error="Table search is unavailable because schema storage is not ready.")

        try:
            metadata, sample_values = self.schema_rag.search_similar(
                query_text,
                catalog_name=catalog_name,
                database_name=self._reset_database_for_rag(database_name),
                schema_name=schema_name,
                table_type="full",
                top_n=top_n,
            )
            result_dict: Dict[str, List[Dict[str, Any]]] = {"metadata": [], "sample_data": []}

            metadata_rows: List[Dict[str, Any]] = []
            if metadata:
                metadata_rows = metadata.select(
                    [
                        "catalog_name",
                        "database_name",
                        "schema_name",
                        "table_name",
                        "table_type",
                        "identifier",
                        "_distance",
                    ]
                ).to_pylist()
            if not metadata_rows:
                return FuncToolResult(success=0, error="No metadata rows found.")

            current_has_semantic = False
            if self.has_semantic_models:
                for metadata_row in metadata_rows:
                    semantic_model = self._get_semantic_model(
                        metadata_row["catalog_name"],
                        metadata_row["database_name"],
                        metadata_row["schema_name"],
                        metadata_row["table_name"],
                    )
                    if semantic_model:
                        current_has_semantic = True
                        metadata_row["semantic_model_name"] = semantic_model["semantic_model_name"]
                        metadata_row["description"] = semantic_model["semantic_model_desc"]
                        metadata_row["dimensions"] = semantic_model["dimensions"]
                        metadata_row["measures"] = semantic_model["measures"]
                        # Only enrich the top match to prioritize the most relevant table
                        break

            result_dict["metadata"] = metadata_rows
            if current_has_semantic:
                return FuncToolResult(success=1, result=result_dict)

            sample_rows: List[Dict[str, Any]] = []
            if sample_values:
                if simple_sample_data:
                    selected_fields = ["identifier", "table_type", "sample_rows", "_distance"]
                else:
                    selected_fields = [
                        "identifier",
                        "catalog_name",
                        "database_name",
                        "schema_name",
                        "table_type",
                        "table_name",
                        "sample_rows",
                        "_distance",
                    ]
                sample_rows = sample_values.select(selected_fields).to_pylist()
            result_dict["sample_data"] = sample_rows
            return FuncToolResult(result=result_dict)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def list_databases(self, catalog: Optional[str] = "", include_sys: Optional[bool] = False) -> FuncToolResult:
        """
        Enumerate databases accessible through the current connection.

        Args:
            catalog: Optional catalog to scope the lookup (dialect dependent).
            include_sys: Set True to include system databases; defaults to False.

        Returns:
            FuncToolResult with result as a list of database names ordered by the connector. On failure success=0 with
            an explanatory error message.
        """
        try:
            databases = self.connector.get_databases(catalog, include_sys=include_sys)
            filtered = [db for db in databases if self._database_matches_scope(catalog, db)]
            return FuncToolResult(result=filtered)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def list_schemas(
        self, catalog: Optional[str] = "", database: Optional[str] = "", include_sys: bool = False
    ) -> FuncToolResult:
        """
        List schema names under the supplied catalog/database coordinate.

        Args:
            catalog: Optional catalog filter. Leave blank to rely on connector defaults.
            database: Optional database filter. Leave blank to rely on connector defaults.
            include_sys: Set True to include system schemas; defaults to False.

        Returns:
            FuncToolResult with result holding the schema name list. On failure success=0 with an explanatory message.
        """
        try:
            if database and not self._database_matches_scope(catalog, database):
                return FuncToolResult(result=[])
            schemas = self.connector.get_schemas(catalog, database, include_sys=include_sys)
            filtered = [schema for schema in schemas if self._schema_matches_scope(catalog, database, schema)]
            return FuncToolResult(result=filtered)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def list_tables(
        self,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        include_views: Optional[bool] = True,
    ) -> FuncToolResult:
        """
        Return table-like objects (tables, views, materialized views) visible to the connector.
        Args:
            catalog: Optional catalog filter.
            database: Optional database filter.
            schema_name: Optional schema filter.
            include_views: When True (default) also include views and materialized views.

        Returns:
            FuncToolResult with result=[{"type": "table|view|materialized_view", "name": str}, ...]. On failure
            success=0 with an explanatory error message.
        """
        try:
            result = []
            for tb in self.connector.get_tables(catalog, database, schema_name):
                result.append({"type": "table", "name": tb})

            if include_views:
                # Add views
                try:
                    views = self.connector.get_views(catalog, database, schema_name)
                    for view in views:
                        result.append({"type": "view", "name": view})
                except (NotImplementedError, AttributeError):
                    # Some connectors may not support get_views
                    pass

                # Add materialized views
                try:
                    materialized_views = self.connector.get_materialized_views(catalog, database, schema_name)
                    for mv in materialized_views:
                        result.append({"type": "materialized_view", "name": mv})
                except (NotImplementedError, AttributeError):
                    # Some connectors may not support get_materialized_views
                    pass

            filtered_result = self._filter_table_entries(result, catalog, database, schema_name)
            return FuncToolResult(result=filtered_result)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def check_table_exists(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Quick check if table exists without fetching full schema.

        Args:
            table_name: Simple table name (unqualified)
            catalog: Optional catalog override
            database: Optional database override
            schema_name: Optional schema override

        Returns:
            FuncToolResult with result={"table_exists": bool, "suggestions": [...], "available_tables": [...]}
        """
        try:
            # Get all accessible tables
            tables = self.connector.get_tables(catalog_name=catalog, database_name=database, schema_name=schema_name)

            if not tables:
                return FuncToolResult(result={"table_exists": False, "suggestions": [], "available_tables": []})

            # Check if table exists (case-insensitive)
            # tables is a List[str], convert to dict format for consistent processing
            table_names_lower = {t.lower(): t for t in tables}
            exists = table_name.lower() in table_names_lower

            # Suggest similar table names if not found
            suggestions = []
            if not exists:
                import difflib

                available = list(table_names_lower.keys())
                matches = difflib.get_close_matches(table_name.lower(), available, n=3, cutoff=0.6)
                suggestions = [table_names_lower[m] for m in matches]

            return FuncToolResult(
                result={
                    "table_exists": exists,
                    "suggestions": suggestions,
                    "available_tables": list(tables)[:20],  # Limit for performance
                }
            )

        except Exception as e:
            from datus.utils.loggings import get_logger

            logger = get_logger(__name__)
            logger.error(f"Error checking table existence: {e}")
            return FuncToolResult(success=0, error=str(e))

    def describe_table(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Fetch detailed column metadata (and optional semantic model info) for the given table.
        When semantic models exist for the table, `table_info`
        includes additional description/dimension/measure fields.

        Args:
            table_name: Table identifier to describe; can be partially qualified.
            catalog: Optional catalog override. Leave blank to rely on connector defaults.
            database: Optional database override. Leave blank to rely on connector defaults.
            schema_name: Optional schema override. Leave blank to rely on connector defaults.

        Returns:
            FuncToolResult with result={"table_info": {...}, "columns": [...]}. Scope violations or connector errors
            surface as success=0 with an explanatory message.
        """
        try:
            coordinate = self._build_table_coordinate(
                raw_name=table_name,
                catalog=catalog,
                database=database,
                schema=schema_name,
            )
            if not self._table_matches_scope(coordinate):
                return FuncToolResult(
                    success=0,
                    error=f"Table '{table_name}' is outside the scoped context.",
                )
            column_result = self.connector.get_schema(
                catalog_name=catalog, database_name=database, schema_name=schema_name, table_name=table_name
            )
            table_info = {}
            if self.has_semantic_models:
                semantic_model = self._get_semantic_model(catalog, database, schema_name, table_name)
                if semantic_model:
                    table_info["semantic_model_name"] = semantic_model["semantic_model_name"]
                    table_info["description"] = semantic_model["semantic_model_desc"]
                    table_info["dimensions"] = semantic_model["dimensions"]
                    table_info["measures"] = semantic_model["measures"]
            return FuncToolResult(result={"table_info": table_info, "columns": column_result})
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def read_query(self, sql: str) -> FuncToolResult:
        """
        Execute arbitrary SQL and return the result rows (optionally compressed).

        Args:
            sql: SQL text to run against the connector.

        Returns:
            FuncToolResult with result=self.compressor.compress(rows) when successful. On failure success=0 with the
            underlying error message from the connector.
        """
        try:
            result = self.connector.execute_query(
                sql, result_format="arrow" if self.connector.dialect == DBType.SNOWFLAKE else "list"
            )
            if result.success:
                data = result.sql_return
                return FuncToolResult(result=self.compressor.compress(data))
            else:
                return FuncToolResult(success=0, error=result.error)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def get_table_ddl(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Return the connector's DDL definition for the requested table.

        Use this when the agent needs a full CREATE statement (e.g. for semantic modelling or schema verification).

        Args:
            table_name: Target table identifier (supports partial qualification).
            catalog: Optional catalog override.
            database: Optional database override.
            schema_name: Optional schema override.

        Returns:
            FuncToolResult with result containing identifier/catalog/database/schema/table_name/table_type/definition.
            Scoped-context mismatches or connector failures surface as success=0 with an explanatory message.
        """
        try:
            coordinate = self._build_table_coordinate(
                raw_name=table_name,
                catalog=catalog,
                database=database,
                schema=schema_name,
            )
            if not self._table_matches_scope(coordinate):
                return FuncToolResult(
                    success=0,
                    error=f"Table '{table_name}' is outside the scoped context.",
                )
            # Get tables with DDL
            tables_with_ddl = self.connector.get_tables_with_ddl(
                catalog_name=catalog, database_name=database, schema_name=schema_name, tables=[table_name]
            )

            if not tables_with_ddl:
                return FuncToolResult(success=0, error=f"Table '{table_name}' not found or no DDL available")

            # Return the first (and only) table's DDL
            table_info = tables_with_ddl[0]
            return FuncToolResult(result=table_info)

        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def validate_sql_syntax(self, sql: str) -> FuncToolResult:
        """
        Validate SQL syntax without executing the query.

        Args:
            sql: SQL statement to validate

        Returns:
            FuncToolResult with result containing validation status and extracted tables
        """
        try:
            import sqlglot
            from sqlglot import exp

            # Get dialect from agent config or use default
            dialect = None
            if self.agent_config and hasattr(self.agent_config, "db_type"):
                dialect = self.agent_config.db_type

            # Parse SQL syntax tree
            parsed = sqlglot.parse_one(sql, read=dialect)

            # Basic syntax checks
            issues = []

            # Check for basic SELECT/FROM/INSERT/UPDATE/DELETE structure
            has_main_operation = any(
                isinstance(node, (exp.Select, exp.Insert, exp.Update, exp.Delete)) for node in parsed.walk()
            )

            if not has_main_operation:
                issues.append("SQL语句缺少基本的查询/修改操作关键词(SELECT, INSERT, UPDATE, DELETE等)")

            # Extract table references
            tables = [table.name for table in parsed.find_all(exp.Table)]
            if not tables and has_main_operation:
                # Some valid operations might not have explicit table references (like SELECT 1)
                # Only flag as issue if it's clearly intended to have tables
                pass

            if issues:
                return FuncToolResult(success=0, error="; ".join(issues))

            return FuncToolResult(
                result={
                    "syntax_valid": True,
                    "tables_referenced": tables,
                    "sql_type": type(parsed).__name__,
                    "dialect": dialect or "default",
                }
            )

        except Exception as e:
            return FuncToolResult(success=0, error=f"SQL语法错误: {str(e)}")

    def check_table_exists(
        self, table_name: str, catalog: Optional[str] = "", database: Optional[str] = "", schema: Optional[str] = ""
    ) -> FuncToolResult:
        """
        Check if a table exists without retrieving its full schema.

        Args:
            table_name: Name of the table to check
            catalog: Optional catalog name
            database: Optional database name
            schema: Optional schema name

        Returns:
            FuncToolResult with result containing existence status and available tables
        """
        try:
            coordinate = self._build_table_coordinate(
                raw_name=table_name,
                catalog=catalog,
                database=database,
                schema=schema,
            )

            # Check if table matches scoped context
            if not self._table_matches_scope(coordinate):
                return FuncToolResult(success=0, error=f"Table '{table_name}' is outside the scoped context.")

            # Get tables list for the specified scope
            tables = self.connector.get_tables(catalog_name=catalog, database_name=database, schema_name=schema)

            # Extract table names
            table_names = [t.get("name", "") for t in tables if t.get("name")]
            table_exists = table_name in table_names

            # Get a few similar table names for suggestions if table doesn't exist
            suggestions = []
            if not table_exists:
                # Simple fuzzy matching based on substring or edit distance
                import difflib

                # Find tables with similar names (substring match or close edit distance)
                for existing_table in table_names[:20]:  # Limit to first 20 for performance
                    if table_name.lower() in existing_table.lower() or existing_table.lower() in table_name.lower():
                        suggestions.append(existing_table)
                    elif difflib.SequenceMatcher(None, table_name.lower(), existing_table.lower()).ratio() > 0.6:
                        suggestions.append(existing_table)

                # Limit suggestions to top 5
                suggestions = list(set(suggestions))[:5]

            return FuncToolResult(
                result={
                    "table_exists": table_exists,
                    "available_tables": table_names[:10],  # Return first 10 for context
                    "suggestions": suggestions if not table_exists else [],
                    "catalog": catalog,
                    "database": database,
                    "schema": schema,
                    "table_name": table_name,
                }
            )

        except Exception as e:
            return FuncToolResult(success=0, error=f"Table existence check failed: {str(e)}")

    def analyze_query_plan(
        self,
        sql: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Analyze SQL query execution plan to identify performance bottlenecks.

        Args:
            sql: SQL query to analyze
            catalog: Optional catalog override
            database: Optional database override
            schema_name: Optional schema override

        Returns:
            FuncToolResult with execution plan analysis
        """
        try:
            # Construct EXPLAIN query based on database dialect
            dialect = getattr(self.connector, "dialect", "").lower()

            if dialect in ["starrocks", "mysql", "mariadb"]:
                explain_query = f"EXPLAIN {sql}"
            elif dialect in ["postgresql", "postgres"]:
                explain_query = f"EXPLAIN ANALYZE {sql}"
            elif dialect == "duckdb":
                explain_query = f"EXPLAIN QUERY PLAN {sql}"
            elif dialect == "sqlite":
                explain_query = f"EXPLAIN QUERY PLAN {sql}"
            else:
                # Default fallback
                explain_query = f"EXPLAIN {sql}"

            # Execute the EXPLAIN query
            explain_result = self.connector.execute_explain(explain_query, "list")

            if not explain_result.success:
                return FuncToolResult(success=0, error=f"Failed to execute EXPLAIN query: {explain_result.error}")

            # Parse the execution plan
            plan_analysis = self._parse_execution_plan(explain_result.result, dialect)

            return FuncToolResult(result=plan_analysis)

        except Exception as e:
            return FuncToolResult(success=0, error=f"Query plan analysis failed: {str(e)}")

    def _parse_execution_plan(self, plan_data: Any, dialect: str) -> Dict[str, Any]:
        """
        Parse execution plan data and extract performance insights.

        Args:
            plan_data: Raw execution plan data from database
            dialect: Database dialect

        Returns:
            Structured analysis of the execution plan
        """
        try:
            # Initialize analysis structure
            analysis = {
                "success": True,
                "plan_text": "",
                "estimated_rows": 0,
                "estimated_cost": 0.0,
                "hotspots": [],
                "join_analysis": {"join_count": 0, "join_types": [], "join_order_issues": []},
                "index_usage": {"indexes_used": [], "missing_indexes": [], "index_effectiveness": "unknown"},
                "warnings": [],
            }

            # Convert plan data to string for analysis
            if isinstance(plan_data, list) and plan_data:
                plan_text = "\n".join([str(row) for row in plan_data if row])
            else:
                plan_text = str(plan_data)

            analysis["plan_text"] = plan_text

            # Parse based on dialect
            if dialect in ["starrocks", "mysql", "mariadb"]:
                self._parse_mysql_like_plan(plan_text, analysis)
            elif dialect in ["postgresql", "postgres"]:
                self._parse_postgres_plan(plan_text, analysis)
            elif dialect in ["duckdb", "sqlite"]:
                self._parse_sqlite_plan(plan_text, analysis)
            else:
                # Generic parsing for unknown dialects
                self._parse_generic_plan(plan_text, analysis)

            # Generate overall assessment
            self._generate_plan_assessment(analysis)

            return analysis

        except Exception as e:
            return {
                "success": False,
                "error": f"Plan parsing failed: {str(e)}",
                "plan_text": str(plan_data) if plan_data else "",
                "hotspots": [],
                "warnings": ["Failed to parse execution plan"],
            }

    def check_table_conflicts(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Check for potential table conflicts and duplicate data structures.

        Args:
            table_name: Table to check for conflicts
            catalog: Optional catalog override
            database: Optional database override
            schema_name: Optional schema override

        Returns:
            FuncToolResult with conflict analysis
        """
        try:
            # Build target table coordinate
            target_coordinate = self._build_table_coordinate(
                raw_name=table_name,
                catalog=catalog,
                database=database,
                schema=schema_name,
            )

            # Get target table metadata
            target_metadata = self._get_table_metadata(target_coordinate)
            if not target_metadata:
                return FuncToolResult(success=0, error=f"Target table '{table_name}' not found in metadata store")

            # Search for similar tables
            similar_tables = self._find_similar_tables(target_coordinate, target_metadata)

            # Analyze conflicts
            conflict_analysis = self._analyze_table_conflicts(target_metadata, similar_tables)

            return FuncToolResult(result=conflict_analysis)

        except Exception as e:
            return FuncToolResult(success=0, error=f"Table conflict check failed: {str(e)}")

    def _get_table_metadata(self, coordinate: TableCoordinate) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific table."""
        if not self.has_schema:
            return None

        try:
            # Search for the specific table
            schemas, _ = self.schema_rag.search_tables(
                tables=[coordinate.table],
                catalog_name=coordinate.catalog,
                database_name=coordinate.database,
                schema_name=coordinate.schema,
                dialect=self.connector.dialect,
            )

            if schemas:
                schema = schemas[0]
                return {
                    "table_name": schema.table_name,
                    "catalog": schema.catalog_name,
                    "database": schema.database_name,
                    "schema": schema.schema_name,
                    "columns": schema.columns or [],
                    "table_type": getattr(schema, "table_type", "table"),
                    "ddl_hash": self._calculate_ddl_hash(schema),
                }

            return None
        except Exception as e:
            logger.warning(f"Failed to get table metadata: {e}")
            return None

    def _calculate_ddl_hash(self, schema) -> str:
        """Calculate a hash of table DDL for comparison."""
        import hashlib

        # Create a normalized representation of table structure
        ddl_components = [schema.table_name or "", str(sorted(schema.columns or [], key=lambda x: x.get("name", "")))]

        ddl_string = "|".join(ddl_components)
        return hashlib.md5(ddl_string.encode()).hexdigest()[:16]

    def _find_similar_tables(
        self, target_coordinate: TableCoordinate, target_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find tables similar to the target table."""
        similar_tables = []

        if not self.has_schema:
            return similar_tables

        try:
            # Search by table name similarity (exact matches, prefix matches, etc.)
            candidates = self._search_similar_by_name(target_coordinate.table)

            for candidate in candidates:
                if self._is_same_table(candidate, target_coordinate):
                    continue  # Skip the target table itself

                similarity_score = self._calculate_table_similarity(target_metadata, candidate)
                if similarity_score > 0.3:  # Only include tables with meaningful similarity
                    candidate["similarity_score"] = similarity_score
                    similar_tables.append(candidate)

            # Sort by similarity score
            similar_tables.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

            return similar_tables[:10]  # Return top 10 most similar

        except Exception as e:
            logger.warning(f"Failed to find similar tables: {e}")
            return similar_tables

    def _search_similar_by_name(self, table_name: str) -> List[Dict[str, Any]]:
        """Search for tables with similar names."""
        candidates = []

        try:
            # Use schema search to find tables
            # This is a simplified approach - in practice, you might want to use
            # more sophisticated name matching
            search_query = f"table named {table_name}"

            metadata, _ = self.schema_rag.search_similar(
                query_text=search_query,
                catalog_name="",
                database_name="",
                schema_name="",
                table_type="full",
                top_n=50,  # Get more candidates for name matching
            )

            if metadata:
                for row in metadata.select(
                    ["catalog_name", "database_name", "schema_name", "table_name", "table_type", "identifier"]
                ).to_pylist():
                    candidates.append(
                        {
                            "table_name": row.get("table_name", ""),
                            "catalog": row.get("catalog_name", ""),
                            "database": row.get("database_name", ""),
                            "schema": row.get("schema_name", ""),
                            "table_type": row.get("table_type", ""),
                            "identifier": row.get("identifier", ""),
                        }
                    )

            # Also search for exact name matches in different schemas/databases
            # This would require additional queries against the schema store

        except Exception as e:
            logger.warning(f"Failed to search similar by name: {e}")

        return candidates

    def _is_same_table(self, candidate: Dict[str, Any], target: TableCoordinate) -> bool:
        """Check if candidate table is the same as target table."""
        return (
            candidate.get("table_name") == target.table
            and candidate.get("catalog") == target.catalog
            and candidate.get("database") == target.database
            and candidate.get("schema") == target.schema
        )

    def _calculate_table_similarity(self, target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Calculate similarity score between two tables."""
        score = 0.0

        # Name similarity (basic string similarity)
        target_name = target.get("table_name", "").lower()
        candidate_name = candidate.get("table_name", "").lower()

        if target_name == candidate_name:
            score += 0.5  # Exact name match
        elif target_name in candidate_name or candidate_name in target_name:
            score += 0.3  # Partial name match

        # Column similarity (if we can get column info)
        target_columns = target.get("columns", [])
        candidate_columns = candidate.get("columns", [])

        if target_columns and candidate_columns:
            # Compare column names and types
            target_col_names = {col.get("name", "").lower() for col in target_columns}
            candidate_col_names = {col.get("name", "").lower() for col in candidate_columns}

            intersection = target_col_names & candidate_col_names
            union = target_col_names | candidate_col_names

            if union:
                jaccard_similarity = len(intersection) / len(union)
                score += jaccard_similarity * 0.4  # Column overlap contributes to score

        # DDL hash similarity
        target_hash = target.get("ddl_hash", "")
        candidate_hash = candidate.get("ddl_hash", "")

        if target_hash and candidate_hash and target_hash == candidate_hash:
            score += 0.1  # Exact DDL match bonus

        return min(score, 1.0)  # Cap at 1.0

    def _analyze_table_conflicts(
        self, target_metadata: Dict[str, Any], similar_tables: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze conflicts between target table and similar tables."""
        analysis = {
            "success": True,
            "exists_similar": len(similar_tables) > 0,
            "target_table": {
                "name": target_metadata.get("table_name", ""),
                "columns": [col.get("name", "") for col in target_metadata.get("columns", [])],
                "ddl_hash": target_metadata.get("ddl_hash", ""),
                "estimated_rows": 0,  # Would need to get from actual table stats
            },
            "matches": [],
            "duplicate_build_risk": "low",
            "layering_violations": [],
            "error": "",
        }

        # Analyze each similar table
        for similar in similar_tables:
            similarity_score = similar.get("similarity_score", 0)
            conflict_type = self._classify_conflict_type(similarity_score, similar)

            match_info = {
                "table_name": similar.get("table_name", ""),
                "similarity_score": similarity_score,
                "conflict_type": conflict_type,
                "matching_columns": [],  # Would need more detailed column comparison
                "column_similarity": similarity_score,  # Simplified
                "business_conflict": self._assess_business_conflict(target_metadata, similar),
                "recommendation": self._generate_conflict_recommendation(conflict_type, similarity_score),
            }

            analysis["matches"].append(match_info)

        # Assess overall risk
        if analysis["matches"]:
            high_similarity = [m for m in analysis["matches"] if m["similarity_score"] > 0.8]
            if high_similarity:
                analysis["duplicate_build_risk"] = "high"
            elif len([m for m in analysis["matches"] if m["similarity_score"] > 0.6]) > 2:
                analysis["duplicate_build_risk"] = "medium"

        # Check for layering violations (simplified business logic check)
        layering_issues = self._check_layering_violations(target_metadata, similar_tables)
        analysis["layering_violations"] = layering_issues

        return analysis

    def _classify_conflict_type(self, similarity_score: float, table_info: Dict[str, Any]) -> str:
        """Classify the type of conflict."""
        if similarity_score > 0.9:
            return "duplicate"  # Nearly identical tables
        elif similarity_score > 0.7:
            return "similar_business"  # Similar business purpose
        elif similarity_score > 0.5:
            return "structural_overlap"  # Structural similarities
        else:
            return "minor_overlap"  # Minor similarities only

    def _assess_business_conflict(self, target: Dict[str, Any], candidate: Dict[str, Any]) -> str:
        """Assess business logic conflicts (simplified version)."""
        target_name = target.get("table_name", "").lower()
        candidate_name = candidate.get("table_name", "").lower()

        # Simple heuristic: similar names often indicate similar business purpose
        if "fact" in target_name and "fact" in candidate_name:
            return "可能存在事实表重复建设"
        elif "dim" in target_name and "dim" in candidate_name:
            return "可能存在维度表重复建设"
        elif any(keyword in target_name for keyword in ["user", "customer", "client"]) and any(
            keyword in candidate_name for keyword in ["user", "customer", "client"]
        ):
            return "可能存在用户相关数据重复"

        return "表结构相似，建议进一步评估业务需求"

    def _generate_conflict_recommendation(self, conflict_type: str, similarity_score: float) -> str:
        """Generate recommendations based on conflict type."""
        if conflict_type == "duplicate":
            return "建议删除重复表或合并数据，优先保留数据更完整的一份"
        elif conflict_type == "similar_business":
            return "建议评估是否可以复用现有表，或明确业务边界"
        elif conflict_type == "structural_overlap":
            return "建议检查是否可以标准化表结构设计"
        else:
            return "建议定期review表设计，避免逐渐偏离规范"

    def _check_layering_violations(self, target: Dict[str, Any], similar_tables: List[Dict[str, Any]]) -> List[str]:
        """Check for data warehouse layering violations."""
        violations = []
        target_name = target.get("table_name", "").lower()

        # Simplified layering checks
        if "ads_" in target_name:
            # ADS layer should not have direct duplicates in other layers
            for similar in similar_tables:
                similar_name = similar.get("table_name", "").lower()
                if any(layer in similar_name for layer in ["ods_", "dwd_", "dws_"]):
                    violations.append(f"ADS层表不应与{similar_name}直接对应，可能违反分层规范")

        elif "dws_" in target_name:
            # DWS layer aggregations should be unique
            duplicate_aggregations = [t for t in similar_tables if "dws_" in t.get("table_name", "").lower()]
            if duplicate_aggregations:
                violations.append("DWS层存在相似汇总逻辑，可能存在重复计算")

        return violations

    def validate_partitioning(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Validate table partitioning strategy and provide optimization recommendations.

        Args:
            table_name: Table to validate partitioning
            catalog: Optional catalog override
            database: Optional database override
            schema_name: Optional schema override

        Returns:
            FuncToolResult with partitioning analysis
        """
        try:
            # Build table coordinate
            coordinate = self._build_table_coordinate(
                raw_name=table_name,
                catalog=catalog,
                database=database,
                schema=schema_name,
            )

            # Get table DDL for partitioning analysis
            ddl_result = self.get_table_ddl(
                table_name=table_name, catalog=catalog, database=database, schema_name=schema_name
            )

            if not ddl_result.success:
                return FuncToolResult(success=0, error=f"Cannot retrieve table DDL: {ddl_result.error}")

            ddl_text = ddl_result.result.get("ddl", "")

            # Parse partitioning information
            partitioning_info = self._parse_partitioning_info(ddl_text, coordinate)

            # Validate partitioning strategy
            validation_results = self._validate_partitioning_strategy(partitioning_info, coordinate)

            # Generate recommendations
            recommendations = self._generate_partitioning_recommendations(partitioning_info, validation_results)

            result = {
                "success": True,
                "partitioned": partitioning_info["is_partitioned"],
                "partition_info": partitioning_info,
                "validation_results": validation_results,
                "issues": validation_results.get("issues", []),
                "recommended_partition": recommendations.get("recommended_partition", {}),
                "performance_impact": recommendations.get("performance_impact", {}),
                "error": "",
            }

            return FuncToolResult(result=result)

        except Exception as e:
            return FuncToolResult(success=0, error=f"Partitioning validation failed: {str(e)}")

    def _parse_partitioning_info(self, ddl_text: str, coordinate: TableCoordinate) -> Dict[str, Any]:
        """Parse partitioning information from DDL text."""
        info = {
            "is_partitioned": False,
            "partition_key": "",
            "partition_type": "",
            "partition_count": 0,
            "partition_expression": "",
            "partition_values": [],
            "subpartition_info": {},
        }

        if not ddl_text:
            return info

        ddl_lower = ddl_text.lower()

        # Check if table is partitioned
        if "partitioned by" in ddl_lower or "partition by" in ddl_lower:
            info["is_partitioned"] = True

            # Extract partition key and type
            self._extract_partition_details(ddl_text, info)

        return info

    def _extract_partition_details(self, ddl_text: str, info: Dict[str, Any]) -> None:
        """Extract detailed partitioning information from DDL."""
        import re

        ddl_lower = ddl_text.lower()

        # StarRocks partitioning patterns
        starrocks_patterns = [
            r"partitioned\s+by\s+\(([^)]+)\)",  # PARTITIONED BY (column)
            r"partition\s+by\s+\(([^)]+)\)",  # PARTITION BY (column)
            r"partitioned\s+by\s+date_trunc\([^)]+\)",  # PARTITIONED BY date_trunc
        ]

        for pattern in starrocks_patterns:
            match = re.search(pattern, ddl_lower, re.IGNORECASE)
            if match:
                partition_expr = match.group(1).strip()
                info["partition_expression"] = partition_expr

                # Try to extract partition key
                if "(" in partition_expr and ")" in partition_expr:
                    # Extract column name from function calls like date_trunc('day', column)
                    inner_match = re.search(r'[\'"]([^\'"]+)[\'"],?\s*([^)]+)', partition_expr)
                    if inner_match:
                        info["partition_key"] = inner_match.group(2).strip()
                        info["partition_type"] = "time_based"
                    else:
                        # Simple column partitioning
                        info["partition_key"] = partition_expr.strip("()")
                        info["partition_type"] = "range"
                else:
                    info["partition_key"] = partition_expr
                    info["partition_type"] = "range"

                break

        # Try to estimate partition count (simplified)
        # In practice, this would require querying system tables or parsing more DDL
        if info["partition_type"] == "time_based":
            info["partition_count"] = 30  # Assume monthly partitions for time-based
        else:
            info["partition_count"] = 10  # Default estimate

    def _validate_partitioning_strategy(
        self, partition_info: Dict[str, Any], coordinate: TableCoordinate
    ) -> Dict[str, Any]:
        """Validate partitioning strategy against best practices."""
        results = {
            "partition_key_valid": True,
            "granularity_appropriate": True,
            "data_distribution_even": True,
            "pruning_opportunities": True,
            "issues": [],
        }

        if not partition_info["is_partitioned"]:
            results["issues"].append(
                {
                    "severity": "medium",
                    "issue_type": "no_partitioning",
                    "description": "表未进行分区，可能影响查询性能和大表维护",
                    "recommendation": "建议根据数据特点和查询模式添加分区",
                }
            )
            return results

        partition_key = partition_info.get("partition_key", "").lower()
        partition_type = partition_info.get("partition_type", "")

        # Validate partition key choice
        if partition_type == "time_based":
            # Time-based partitioning validation
            time_keywords = ["create_time", "update_time", "event_time", "date", "time"]
            if not any(keyword in partition_key for keyword in time_keywords):
                results["partition_key_valid"] = False
                results["issues"].append(
                    {
                        "severity": "high",
                        "issue_type": "poor_key_choice",
                        "description": f"时间分区键'{partition_key}'不是标准时间字段",
                        "recommendation": "建议使用create_time、update_time等标准时间字段作为分区键",
                    }
                )

        elif partition_type == "range":
            # Range partitioning validation
            if not partition_key:
                results["partition_key_valid"] = False
                results["issues"].append(
                    {
                        "severity": "high",
                        "issue_type": "missing_partition_key",
                        "description": "分区表缺少明确的partition_key定义",
                        "recommendation": "明确指定分区键字段",
                    }
                )

        # Check partition granularity
        partition_count = partition_info.get("partition_count", 0)
        if partition_count > 1000:
            results["granularity_appropriate"] = False
            results["issues"].append(
                {
                    "severity": "medium",
                    "issue_type": "too_many_partitions",
                    "description": f"分区数量({partition_count})过多，可能影响查询性能",
                    "recommendation": "考虑增大分区粒度或使用动态分区",
                }
            )
        elif partition_count < 3:
            results["granularity_appropriate"] = False
            results["issues"].append(
                {
                    "severity": "low",
                    "issue_type": "too_few_partitions",
                    "description": f"分区数量({partition_count})过少，限制了分区裁剪效果",
                    "recommendation": "考虑减小分区粒度以提高查询性能",
                }
            )

        # Assess pruning opportunities (simplified)
        # In practice, this would analyze common query patterns
        if not partition_key:
            results["pruning_opportunities"] = False

        return results

    def _generate_partitioning_recommendations(
        self, partition_info: Dict[str, Any], validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate partitioning recommendations and performance impact assessment."""
        recommendations = {
            "recommended_partition": {},
            "performance_impact": {
                "query_speed_improvement": "unknown",
                "storage_efficiency": "neutral",
                "maintenance_overhead": "unknown",
            },
        }

        if not partition_info["is_partitioned"]:
            # Recommend partitioning for large tables
            recommendations["recommended_partition"] = {
                "suggested_key": "create_time",
                "suggested_type": "time_based",
                "estimated_partitions": 30,
                "rationale": "基于创建时间按月分区，适合大多数业务场景，支持时间范围查询优化",
            }

            recommendations["performance_impact"] = {
                "query_speed_improvement": "significant",
                "storage_efficiency": "improved",
                "maintenance_overhead": "acceptable",
            }

        else:
            issues = validation_results.get("issues", [])
            partition_key = partition_info.get("partition_key", "")

            if any(issue["issue_type"] == "poor_key_choice" for issue in issues):
                # Suggest better partition key
                table_name_lower = partition_info.get("table_name", "").lower()

                if any(keyword in table_name_lower for keyword in ["log", "event", "action"]):
                    suggested_key = "event_time"
                elif any(keyword in table_name_lower for keyword in ["order", "transaction"]):
                    suggested_key = "order_date"
                else:
                    suggested_key = "create_time"

                recommendations["recommended_partition"] = {
                    "suggested_key": suggested_key,
                    "suggested_type": "time_based",
                    "estimated_partitions": 30,
                    "rationale": f"建议使用{suggested_key}作为分区键，更符合业务查询模式",
                }

            elif any(issue["issue_type"] == "too_many_partitions" for issue in issues):
                # Suggest coarser granularity
                recommendations["recommended_partition"] = {
                    "suggested_key": partition_key,
                    "suggested_type": "time_based",
                    "estimated_partitions": 12,
                    "rationale": "建议改用季度或年度分区，减少分区数量",
                }

        return recommendations

    def _parse_mysql_like_plan(self, plan_text: str, analysis: Dict[str, Any]) -> None:
        """Parse MySQL-like execution plans (MySQL, StarRocks, MariaDB)."""
        lines = plan_text.strip().split("\n")

        for line in lines:
            line = line.strip().lower()

            # Check for table scans
            if "table scan" in line or "all" in line:
                analysis["hotspots"].append(
                    {
                        "reason": "full_table_scan",
                        "node": line,
                        "severity": "high",
                        "recommendation": "Consider adding appropriate indexes for WHERE conditions",
                    }
                )

            # Check for expensive joins
            if "join" in line:
                analysis["join_analysis"]["join_count"] += 1
                if "block nested loop" in line:
                    analysis["hotspots"].append(
                        {
                            "reason": "expensive_join",
                            "node": line,
                            "severity": "high",
                            "recommendation": "Consider adding indexes on join columns",
                        }
                    )
                    analysis["join_analysis"]["join_types"].append("block_nested_loop")

            # Check for filesort
            if "filesort" in line:
                analysis["hotspots"].append(
                    {
                        "reason": "filesort_operation",
                        "node": line,
                        "severity": "medium",
                        "recommendation": "Consider adding indexes to avoid sorting",
                    }
                )

    def _parse_postgres_plan(self, plan_text: str, analysis: Dict[str, Any]) -> None:
        """Parse PostgreSQL execution plans."""
        lines = plan_text.strip().split("\n")

        for line in lines:
            line = line.strip().lower()

            # Check for sequential scans
            if "seq scan" in line:
                analysis["hotspots"].append(
                    {
                        "reason": "sequential_scan",
                        "node": line,
                        "severity": "medium",
                        "recommendation": "Consider adding indexes for better performance",
                    }
                )

            # Check for hash joins
            if "hash join" in line:
                analysis["join_analysis"]["join_count"] += 1
                analysis["join_analysis"]["join_types"].append("hash_join")

            # Check for nested loop joins
            if "nested loop" in line:
                analysis["join_analysis"]["join_count"] += 1
                analysis["join_analysis"]["join_types"].append("nested_loop")

            # Extract cost estimates
            if "cost=" in line:
                cost_match = line.split("cost=")[1].split()[0]
                if ".." in cost_match:
                    total_cost = float(cost_match.split("..")[1])
                    analysis["estimated_cost"] = total_cost

            # Extract row estimates
            if "rows=" in line:
                rows_match = line.split("rows=")[1].split()[0]
                try:
                    row_count = int(float(rows_match))
                    analysis["estimated_rows"] = max(analysis["estimated_rows"], row_count)
                except ValueError:
                    pass

    def _parse_sqlite_plan(self, plan_text: str, analysis: Dict[str, Any]) -> None:
        """Parse SQLite/DuckDB execution plans."""
        lines = plan_text.strip().split("\n")

        for line in lines:
            line = line.strip().lower()

            # Check for table scans
            if "scan" in line and "table" in line:
                analysis["hotspots"].append(
                    {
                        "reason": "table_scan",
                        "node": line,
                        "severity": "low",
                        "recommendation": "Consider query optimization if performance is an issue",
                    }
                )

    def _parse_generic_plan(self, plan_text: str, analysis: Dict[str, Any]) -> None:
        """Generic parsing for unknown dialects."""
        lines = plan_text.strip().split("\n")

        # Basic heuristics for any execution plan
        for line in lines:
            line = line.strip().lower()

            # Look for common performance indicators
            if any(keyword in line for keyword in ["scan", "table scan", "full scan"]):
                analysis["hotspots"].append(
                    {
                        "reason": "potential_scan_operation",
                        "node": line,
                        "severity": "medium",
                        "recommendation": "Review query for potential optimization opportunities",
                    }
                )

            if "join" in line:
                analysis["join_analysis"]["join_count"] += 1

        analysis["warnings"].append("Using generic plan analysis - results may be limited")

    def _generate_plan_assessment(self, analysis: Dict[str, Any]) -> None:
        """Generate overall assessment and recommendations."""
        hotspots = analysis.get("hotspots", [])

        # Assess index effectiveness
        if hotspots:
            high_severity = len([h for h in hotspots if h.get("severity") == "high"])
            if high_severity > 0:
                analysis["index_usage"]["index_effectiveness"] = "poor"
                analysis["index_usage"]["missing_indexes"].append(
                    "Consider adding indexes for high-severity operations"
                )
            else:
                analysis["index_usage"]["index_effectiveness"] = "fair"
        else:
            analysis["index_usage"]["index_effectiveness"] = "good"

        # Generate warnings based on analysis
        if len(hotspots) > 5:
            analysis["warnings"].append("Multiple performance hotspots detected - comprehensive optimization needed")

        if analysis["join_analysis"]["join_count"] > 3:
            analysis["warnings"].append("Complex join operations detected - consider query restructuring")

        if analysis.get("estimated_cost", 0) > 10000:
            analysis["warnings"].append("High estimated query cost - optimization recommended")


def db_function_tool_instance(
    agent_config: AgentConfig, database_name: str = "", sub_agent_name: Optional[str] = None
) -> DBFuncTool:
    db_manager = db_manager_instance(agent_config.namespaces)
    return DBFuncTool(
        db_manager.get_conn(agent_config.current_namespace, database_name or agent_config.current_database),
        agent_config=agent_config,
        sub_agent_name=sub_agent_name,
    )


def db_function_tools(
    agent_config: AgentConfig, database_name: str = "", sub_agent_name: Optional[str] = None
) -> List[Tool]:
    return db_function_tool_instance(agent_config, database_name, sub_agent_name).available_tools()
