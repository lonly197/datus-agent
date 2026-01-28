# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Core database function tool implementation.

This module contains the main DBFuncTool class with initialization logic
and integration with other modules.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage.metric.store import SemanticMetricsRAG
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.db_tools import BaseSqlConnector
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.database.db_functions import DBFunctionsMixin
from datus.tools.func_tool.database.patterns import ScopedTablePattern, TableCoordinate
from datus.tools.func_tool.database.utils import (
    build_table_coordinate,
    matches_catalog_database,
    normalize_identifier_part,
    parse_execution_plan,
    parse_scope_token,
)
from datus.tools.func_tool.database.validation import (
    check_table_conflicts,
    validate_partitioning,
    validate_sql_syntax,
)
from datus.utils.compress_utils import DataCompressor
from datus.utils.constants import SUPPORT_DATABASE_DIALECTS, SUPPORT_SCHEMA_DIALECTS, DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DBFuncTool(DBFunctionsMixin):
    """
    Main database function tool for Datus.

    This class provides database operations including table listing,
    schema retrieval, DDL extraction, and query execution with support
    for scoped table access patterns.

    Attributes:
        connector: Database connector instance
        compressor: Data compression utility
        agent_config: Agent configuration
        sub_agent_name: Optional sub-agent name for scoping
        schema_rag: Schema RAG storage instance
        _field_order: Ordered list of coordinate fields for the dialect
        _scoped_patterns: List of scoped table patterns
        _semantic_storage: Semantic metrics storage instance
        has_schema: Whether schema storage is available
        has_semantic_models: Whether semantic models are available
    """

    def __init__(
        self,
        connector: BaseSqlConnector,
        agent_config: Optional[AgentConfig] = None,
        *,
        sub_agent_name: Optional[str] = None,
        scoped_tables: Optional[Iterable[str]] = None,
    ):
        """
        Initialize DBFuncTool with database connector and optional scoping.

        Args:
            connector: Database connector instance
            agent_config: Optional agent configuration
            sub_agent_name: Optional sub-agent name for scoped context
            scoped_tables: Optional iterable of scoped table patterns
        """
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
        """
        Reset database name for RAG operations based on dialect.

        For SQLite and DuckDB, returns the connector's database name.
        For other dialects, returns the provided database_name.

        Args:
            database_name: Database name from request

        Returns:
            Appropriate database name for RAG operations
        """
        if self.connector.dialect in (DBType.SQLITE, DBType.DUCKDB):
            return self.connector.database_name
        else:
            return database_name

    def _determine_field_order(self) -> Sequence[str]:
        """
        Determine the order of coordinate fields based on database dialect.

        Returns:
            List of field names in order for the current dialect
        """
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
        """
        Load scoped table patterns from explicit tokens or sub-agent config.

        Args:
            explicit_tokens: Optional iterable of explicit pattern tokens

        Returns:
            List of ScopedTablePattern objects
        """
        tokens: List[str] = []
        if explicit_tokens:
            tokens.extend(explicit_tokens)
        else:
            tokens.extend(self._resolve_scoped_context_tables())

        patterns: List[ScopedTablePattern] = []
        for token in tokens:
            scoped_pattern = parse_scope_token(token, self._field_order)
            if scoped_pattern:
                patterns.append(scoped_pattern)
        return patterns

    def _resolve_scoped_context_tables(self) -> Sequence[str]:
        """
        Resolve scoped context tables from sub-agent configuration.

        Returns:
            List of scoped table pattern strings
        """
        if not self.agent_config:
            return []
        scoped_entries: List[str] = []

        if self.sub_agent_name:
            sub_agent_config = self._load_sub_agent_config(self.sub_agent_name)
            if sub_agent_config and sub_agent_config.scoped_context and sub_agent_config.scoped_context.tables:
                scoped_entries.extend(sub_agent_config.scoped_context.as_lists().tables)

        return scoped_entries

    def _load_sub_agent_config(self, sub_agent_name: str) -> Optional[SubAgentConfig]:
        """
        Load sub-agent configuration by name.

        Args:
            sub_agent_name: Name of the sub-agent

        Returns:
            SubAgentConfig if found and valid, None otherwise
        """
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

    def _get_semantic_model(
        self, catalog: str = "", database: str = "", schema: str = "", table_name: str = ""
    ) -> Dict[str, Any]:
        """
        Get semantic model for a table.

        Args:
            catalog: Optional catalog name
            database: Optional database name
            schema: Optional schema name
            table_name: Table name

        Returns:
            Semantic model dictionary or empty dict if not found
        """
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

    def _default_field_value(self, field: str, explicit: Optional[str]) -> str:
        """
        Get default value for a coordinate field.

        Args:
            field: Field name (catalog, database, schema)
            explicit: Explicit value provided (if any)

        Returns:
            Field value from connector defaults or explicit value
        """
        if field not in self._field_order:
            return ""
        if explicit:
            return normalize_identifier_part(explicit)

        fallback_attr_map = {
            "catalog": "catalog_name",
            "database": "database_name",
            "schema": "schema_name",
        }
        fallback_attr = fallback_attr_map.get(field)
        if fallback_attr and hasattr(self.connector, fallback_attr):
            return normalize_identifier_part(getattr(self.connector, fallback_attr))
        return ""

    def _build_table_coordinate(
        self,
        raw_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema: Optional[str] = "",
    ) -> TableCoordinate:
        """
        Build a TableCoordinate from a raw table name and optional overrides.

        Args:
            raw_name: Raw table name (may be qualified)
            catalog: Optional catalog override
            database: Optional database override
            schema: Optional schema override

        Returns:
            TableCoordinate with all fields populated
        """
        return build_table_coordinate(
            raw_name=raw_name,
            field_order=self._field_order,
            default_field_value_func=self._default_field_value,
            catalog=catalog,
            database=database,
            schema=schema,
        )

    def _table_matches_scope(self, coordinate: TableCoordinate) -> bool:
        """
        Check if a table coordinate matches the scoped patterns.

        Args:
            coordinate: TableCoordinate to check

        Returns:
            True if coordinate matches scope or no scoping is configured
        """
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
        """
        Filter table entries based on scoped patterns.

        Args:
            entries: List of table entry dictionaries
            catalog: Optional catalog filter
            database: Optional database filter
            schema: Optional schema filter

        Returns:
            Filtered list of table entries
        """
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
        """
        Check if a pattern matches given catalog and database.

        Args:
            pattern: ScopedTablePattern to check
            catalog: Catalog value
            database: Database value

        Returns:
            True if pattern matches
        """
        return matches_catalog_database(pattern, catalog, database)

    def _database_matches_scope(self, catalog: Optional[str], database: str) -> bool:
        """
        Check if a database matches the scoped patterns.

        Args:
            catalog: Optional catalog value
            database: Database name

        Returns:
            True if database matches scope or no scoping is configured
        """
        if not self._scoped_patterns:
            return True
        catalog_value = self._default_field_value("catalog", catalog or "")
        database_value = self._default_field_value("database", database or "")

        wildcard_allowed = False
        for pattern in self._scoped_patterns:
            if not self._matches_catalog_database(pattern, catalog_value, database_value):
                continue
            if pattern.database:
                if pattern.database and pattern.database in ("*", "%"):
                    return True
                continue
            wildcard_allowed = True
        return wildcard_allowed

    def _schema_matches_scope(self, catalog: Optional[str], database: Optional[str], schema: str) -> bool:
        """
        Check if a schema matches the scoped patterns.

        Args:
            catalog: Optional catalog value
            database: Optional database value
            schema: Schema name

        Returns:
            True if schema matches scope or no scoping is configured
        """
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
                if pattern.matches(TableCoordinate(catalog=catalog_value, database=database_value, schema=schema_value, table="*")):
                    return True
                continue
            wildcard_allowed = True
        return wildcard_allowed

    def available_tools(self) -> List[Tool]:
        """
        Get list of available database tools.

        Returns:
            List of Tool objects for database operations
        """
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

    def validate_sql_syntax(self, sql: str) -> FuncToolResult:
        """
        Validate SQL syntax without executing the query.

        Args:
            sql: SQL statement to validate

        Returns:
            FuncToolResult with validation status and extracted tables
        """
        return validate_sql_syntax(sql, dialect=None, agent_config=self.agent_config)

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
        return check_table_conflicts(
            table_name=table_name,
            catalog=catalog,
            database=database,
            schema_name=schema_name,
            connector=self.connector,
            schema_rag=self.schema_rag,
            has_schema=self.has_schema,
            coordinate_builder=self._build_table_coordinate,
        )

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
            plan_analysis = parse_execution_plan(explain_result.result, dialect)

            return FuncToolResult(result=plan_analysis)

        except Exception as e:
            return FuncToolResult(success=0, error=f"Query plan analysis failed: {str(e)}")

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
        # Get table DDL for partitioning analysis
        ddl_result = self.get_table_ddl(table_name=table_name, catalog=catalog, database=database, schema_name=schema_name)

        if not ddl_result.success:
            return FuncToolResult(success=0, error=f"Cannot retrieve table DDL: {ddl_result.error}")

        ddl_text = ddl_result.result.get("ddl", "")

        return validate_partitioning(
            table_name=table_name,
            ddl_text=ddl_text,
            catalog=catalog,
            database=database,
            schema_name=schema_name,
            coordinate_builder=self._build_table_coordinate,
        )


def db_function_tool_instance(
    agent_config: AgentConfig, database_name: str = "", sub_agent_name: Optional[str] = None
) -> DBFuncTool:
    """
    Create a DBFuncTool instance from agent configuration.

    Args:
        agent_config: Agent configuration
        database_name: Optional database name
        sub_agent_name: Optional sub-agent name for scoping

    Returns:
        Configured DBFuncTool instance
    """
    from datus.tools.db_tools.db_manager import get_db_manager

    db_manager = get_db_manager(agent_config.namespaces)
    return DBFuncTool(
        db_manager.get_conn(agent_config.current_namespace, database_name or agent_config.current_database),
        agent_config=agent_config,
        sub_agent_name=sub_agent_name,
    )


def db_function_tools(
    agent_config: AgentConfig, database_name: str = "", sub_agent_name: Optional[str] = None
) -> List[Tool]:
    """
    Get available database tools from agent configuration.

    Args:
        agent_config: Agent configuration
        database_name: Optional database name
        sub_agent_name: Optional sub-agent name for scoping

    Returns:
        List of available Tool objects
    """
    return db_function_tool_instance(agent_config, database_name, sub_agent_name).available_tools()
