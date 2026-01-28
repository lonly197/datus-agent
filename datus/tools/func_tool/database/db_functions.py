# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Database function tools for Datus.

This module contains the main database operation methods for DBFuncTool,
including table listing, DDL retrieval, and data querying.
"""

from typing import Any, Dict, List, Optional

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.database.patterns import TableCoordinate, _pattern_matches
from datus.utils.constants import DBType


class DBFunctionsMixin:
    """
    Mixin class providing database operation methods.

    This mixin contains the core database functionality methods that are
    mixed into the DBFuncTool class.
    """

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

    def search_tables_by_comment(
        self,
        keyword: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Search for tables based on table/column comments in DDL.

        This is a fallback tool when SchemaStorage is not populated.
        It retrieves all DDLs from the database and searches for keyword in comments.

        Args:
            keyword: Search keyword to find in COMMENT statements
            catalog: Optional catalog filter
            database: Optional database filter
            schema_name: Optional schema filter

        Returns:
            FuncToolResult with result containing:
            - keyword: The search keyword
            - matching_tables: List of tables with matching comments
            - count: Number of matching tables
            Each matching table includes: table_name, comments, ddl_preview
        """
        try:
            import re

            # Build table coordinate for scope checking
            coordinate = self._build_table_coordinate(
                raw_name="*",  # Wildcard for all tables
                catalog=catalog,
                database=database,
                schema=schema_name,
            )

            # Get all tables with DDL from database
            tables_with_ddl = self.connector.get_tables_with_ddl(
                catalog_name=catalog, database_name=database, schema_name=schema_name
            )

            if not tables_with_ddl:
                return FuncToolResult(
                    result={
                        "keyword": keyword,
                        "matching_tables": [],
                        "count": 0,
                        "message": "No tables found in database or no DDL available",
                    }
                )

            matching_tables = []
            keyword_lower = keyword.lower()

            for table_info in tables_with_ddl:
                ddl = table_info.get("ddl", "")
                table_name = table_info.get("name", "")

                # Extract COMMENT statements from DDL
                # Matches: COMMENT 'text' or COMMENT "text"
                comments = re.findall(r"COMMENT\s+['\"](.+?)['\"]", ddl, re.IGNORECASE)

                # Check if keyword matches any comment
                matched_comments = []
                for comment in comments:
                    if keyword_lower in comment.lower():
                        matched_comments.append(comment)

                if matched_comments:
                    matching_tables.append(
                        {
                            "table_name": table_name,
                            "catalog": table_info.get("catalog", catalog),
                            "database": table_info.get("database", database),
                            "schema": table_info.get("schema", schema_name),
                            "comments": matched_comments,
                            "ddl_preview": ddl[:500] if len(ddl) > 500 else ddl,
                        }
                    )

            return FuncToolResult(
                result={
                    "keyword": keyword,
                    "matching_tables": matching_tables,
                    "count": len(matching_tables),
                }
            )

        except Exception as e:
            return FuncToolResult(
                success=0,
                error=f"Failed to search tables by comment: {str(e)}",
            )
