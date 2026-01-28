# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SQLValidateNode implementation for centralized SQL validation.

This node performs comprehensive SQL validation before execution:
- SQL syntax validation (via sqlglot)
- SQL pattern validation (dangerous operations, best practices)
- Table/column existence validation (via schema metadata)
- Execution permissions validation (optional)
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_config import DEFAULT_MAX_REFLECTION_ROUNDS
from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import SQLContext, SQLValidateInput, TableSchema
from datus.utils.constants import DBType
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import validate_and_suggest_sql_fixes
from datus.agent.workflow_status import WorkflowTerminationStatus
from datus.utils.env import get_env_int
from datus.utils.sql_utils import extract_enhanced_metadata_from_ddl

logger = get_logger(__name__)


class SQLValidateNode(Node):
    """
    Dedicated SQL validation node.

    Validates:
    1. SQL syntax (via sqlglot)
    2. SQL patterns (dangerous operations, best practices)
    3. Table/column existence (via schema metadata)
    4. Execution permissions (optional)

    On validation failure, sets termination status to skip to reflect node.
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[BaseInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
    ):
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
        )

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """Setup SQL validation input from workflow context."""
        # Extract SQL query from latest SQL context
        sql_query = ""
        context = getattr(workflow, 'context', None)
        sql_contexts = getattr(context, 'sql_contexts', None) if context else None
        if sql_contexts and len(sql_contexts) > 0:
            latest_sql_context = sql_contexts[-1]
            sql_query = getattr(latest_sql_context, 'sql_query', "") if latest_sql_context else ""

        # Get database dialect from agent config
        dialect = None
        if self.agent_config and hasattr(self.agent_config, "db_type"):
            dialect = self.agent_config.db_type

        # Create proper input object with all fields
        self.input = SQLValidateInput(
            sql_query=sql_query or "",
            dialect=dialect,
            check_table_existence=True,
            check_column_existence=True,
            check_dangerous_operations=True,
        )

        return {"success": True, "message": "SQL validation input setup complete"}

    async def run(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Run comprehensive SQL validation.

        Yields:
            ActionHistory: Progress and result actions
        """
        try:
            if not self.workflow or not self.workflow.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "No workflow or task available for SQL validation",
                    "sql_validate",
                    {"workflow_id": getattr(self.workflow, "id", "unknown") if self.workflow else "unknown"},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"SQL validation failed: {error_result.error_message}",
                    action_type="sql_validation",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            task = self.workflow.task

            # Step 1: Get SQL query to validate
            sql_query = getattr(self.input, "sql_query", "")
            if not sql_query:
                # Try to get from workflow context
                if self.workflow.context and self.workflow.context.sql_contexts:
                    sql_query = self.workflow.context.sql_contexts[-1].sql_query

            if not sql_query:
                yield ActionHistory(
                    action_id=f"{self.id}_no_sql",
                    role=ActionRole.TOOL,
                    messages="SQL validation skipped: No SQL query to validate",
                    action_type="sql_validation",
                    input={"task": task.task[:50] if task else ""},
                    status=ActionStatus.SUCCESS,
                    output={
                        "is_valid": True,
                        "reason": "No SQL query provided",
                        "skipped": True,
                    },
                )
                self.result = BaseResult(
                    success=True,
                    data={"is_valid": True, "reason": "no_sql", "skipped": True},
                )
                return

            # Step 2: Get database dialect
            dialect = DBType.SNOWFLAKE  # Default
            if self.agent_config and hasattr(self.agent_config, "db_type"):
                dialect = self.agent_config.db_type or DBType.SNOWFLAKE

            # Step 3: Run validation
            validation_result = await self._validate_sql(sql_query, dialect)

            # Step 4: Emit result
            if validation_result["is_valid"]:
                yield ActionHistory(
                    action_id=f"{self.id}_validation",
                    role=ActionRole.TOOL,
                    messages=f"SQL validation passed: {validation_result.get('summary', 'No issues found')}",
                    action_type="sql_validation",
                    input={
                        "task": task.task[:50] if task else "",
                        "sql_query": sql_query[:100],
                    },
                    status=ActionStatus.SUCCESS,
                    output=validation_result,
                )
                self.result = BaseResult(success=True, data=validation_result)
            else:
                self.last_action_status = ActionStatus.SOFT_FAILED
                if self.workflow:
                    if not hasattr(self.workflow, "metadata") or self.workflow.metadata is None:
                        self.workflow.metadata = {}
                    # Check if reflection retries are exhausted before skipping to reflect
                    # Priority: agent.yml configuration > constants default > environment variable fallback
                    if hasattr(self.workflow, "_global_config") and hasattr(self.workflow._global_config, "reflection_config"):
                        max_reflection_rounds = self.workflow._global_config.reflection_config.max_reflection_rounds
                    else:
                        max_reflection_rounds = get_env_int("MAX_REFLECTION_ROUNDS", DEFAULT_MAX_REFLECTION_ROUNDS)
                    current_reflection_round = getattr(self.workflow, "reflection_round", 0)
                    if current_reflection_round >= max_reflection_rounds:
                        # Reflection exhausted, proceed to output for final report
                        self.workflow.metadata["termination_status"] = WorkflowTerminationStatus.PROCEED_TO_OUTPUT
                        self.workflow.metadata["termination_reason"] = (
                            f"Max reflection rounds ({max_reflection_rounds}) exceeded after SQL validation failed"
                        )
                        logger.info(
                            f"SQL validation failed but reflection exhausted (round {current_reflection_round}/{max_reflection_rounds}), "
                            "proceeding to output"
                        )
                    else:
                        # Still have reflection retries available, skip to reflect
                        self.workflow.metadata["termination_status"] = WorkflowTerminationStatus.SKIP_TO_REFLECT
                        logger.info(
                            f"SQL validation failed, skipping to reflect (round {current_reflection_round}/{max_reflection_rounds})"
                        )
                yield ActionHistory(
                    action_id=f"{self.id}_validation",
                    role=ActionRole.TOOL,
                    messages=f"SQL validation failed: {validation_result.get('error_summary', 'Validation errors detected')}",
                    action_type="sql_validation",
                    input={
                        "task": task.task[:50] if task else "",
                        "sql_query": sql_query[:100],
                    },
                    status=ActionStatus.FAILED,
                    output=validation_result,
                )
                self.result = BaseResult(
                    success=False,
                    error=validation_result.get("error_summary", "SQL validation failed"),
                    data=validation_result,
                )

            logger.info(
                f"SQL validation completed: is_valid={validation_result['is_valid']}, "
                f"errors={len(validation_result.get('errors', []))}, "
                f"warnings={len(validation_result.get('warnings', []))}"
            )

        except Exception as e:
            logger.error(f"SQL validation failed: {e}")
            error_result = self.create_error_result(
                ErrorCode.NODE_EXECUTION_FAILED,
                f"SQL validation execution failed: {str(e)}",
                "sql_validate",
                {
                    "task_id": (
                        getattr(self.workflow.task, "id", "unknown")
                        if self.workflow and self.workflow.task
                        else "unknown"
                    )
                },
            )
            yield ActionHistory(
                action_id=f"{self.id}_error",
                role=ActionRole.TOOL,
                messages=f"SQL validation failed: {error_result.error_message}",
                action_type="sql_validation",
                input={},
                status=ActionStatus.FAILED,
                output={"error": error_result.error_message, "error_code": error_result.error_code},
            )
            self.result = BaseResult(success=False, error=str(e))

    async def _validate_sql(
        self,
        sql_query: str,
        dialect: str,
    ) -> Dict[str, Any]:
        """
        Comprehensive SQL validation.

        Returns:
            {
                "is_valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "fix_suggestions": List[Dict],
                "syntax_valid": bool,
                "tables_exist": bool,
                "columns_exist": bool,
                "has_dangerous_ops": bool,
                "summary": str,
                "error_summary": str,
            }
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "fix_suggestions": [],
            "syntax_valid": False,
            "tables_exist": True,
            "columns_exist": True,
            "has_dangerous_ops": False,
            "summary": "",
            "error_summary": "",
        }

        # 1. Syntax validation (reuse sql_utils.validate_and_suggest_sql_fixes)
        syntax_result = validate_and_suggest_sql_fixes(sql_query, dialect)
        result["syntax_valid"] = syntax_result["can_parse"]
        result["errors"].extend(syntax_result["errors"])
        result["warnings"].extend(syntax_result["warnings"])
        result["fix_suggestions"].extend(syntax_result["fix_suggestions"])

        # 2. Table/column validation (if schema metadata available)
        if self.workflow and self.workflow.context and self.workflow.context.table_schemas:
            table_col_result = await self._validate_tables_and_columns(
                sql_query, self.workflow.context.table_schemas, dialect
            )
            result["tables_exist"] = table_col_result["tables_exist"]
            result["columns_exist"] = table_col_result["columns_exist"]
            result["errors"].extend(table_col_result["errors"])
            result["warnings"].extend(table_col_result["warnings"])

        # 3. Dangerous operations check
        dangerous_result = self._check_dangerous_operations(sql_query)
        result["has_dangerous_ops"] = dangerous_result["has_dangerous"]
        result["warnings"].extend(dangerous_result["warnings"])

        # 4. Build summary
        if result["errors"]:
            result["error_summary"] = f"{len(result['errors'])} error(s) found"
        if result["warnings"]:
            if result["error_summary"]:
                result["error_summary"] += f", {len(result['warnings'])} warning(s)"
            else:
                result["error_summary"] = f"{len(result['warnings'])} warning(s) found"

        if result["syntax_valid"] and not result["errors"]:
            result["summary"] = "SQL syntax is valid"
            if result["warnings"]:
                result["summary"] += f" with {len(result['warnings'])} warning(s)"

        # 5. Final validity check
        result["is_valid"] = (
            result["syntax_valid"] and result["tables_exist"] and result["columns_exist"] and len(result["errors"]) == 0
        )

        return result

    async def _validate_tables_and_columns(
        self,
        sql_query: str,
        table_schemas: List[TableSchema],
        dialect: str,
    ) -> Dict[str, Any]:
        """
        Validate that tables and columns referenced in SQL exist in schema metadata.

        Returns:
            {
                "tables_exist": bool,
                "columns_exist": bool,
                "errors": List[str],
                "warnings": List[str],
            }
        """
        result = {
            "tables_exist": True,
            "columns_exist": True,
            "errors": [],
            "warnings": [],
        }

        if not table_schemas:
            return result

        # Extract table names from SQL
        from datus.utils.sql_utils import extract_table_names

        try:
            table_names = extract_table_names(sql_query, dialect=dialect)
        except Exception as e:
            logger.warning(f"Failed to extract table names from SQL: {e}")
            return result

        if not table_names:
            return result

        # Build schema lookup with normalized names
        schema_tables = set()
        for schema in table_schemas:
            table_name = (schema.table_name or "").lower()
            if table_name:
                schema_tables.add(table_name)
            database_name = (schema.database_name or "").lower()
            schema_name = (schema.schema_name or "").lower()
            catalog_name = (schema.catalog_name or "").lower()
            if database_name and schema_name:
                schema_tables.add(f"{database_name}.{schema_name}.{table_name}")
            if database_name:
                schema_tables.add(f"{database_name}..{table_name}")
            if catalog_name and database_name:
                schema_tables.add(f"{catalog_name}.{database_name}..{table_name}")

        # Check table existence
        missing_tables = []
        for table in table_names:
            table_lower = table.lower()
            table_parts = [part for part in table_lower.split(".") if part]
            base_table = table_parts[-1] if table_parts else table_lower
            if table_lower not in schema_tables and base_table not in schema_tables:
                missing_tables.append(table)

        if missing_tables:
            result["tables_exist"] = False
            result["errors"].append(f"Referenced tables not found in schema: {', '.join(missing_tables)}")

        # Column validation (basic but real: compare against DDL/column_comments)
        column_lookup = self._build_column_lookup(table_schemas, dialect)
        column_refs = self._extract_column_references(sql_query, dialect)

        missing_columns: List[str] = []
        for table_ref, col in column_refs:
            col_l = col.lower()
            if table_ref:
                table_l = table_ref.lower()
                if table_l in column_lookup and col_l not in column_lookup[table_l]:
                    missing_columns.append(f"{table_ref}.{col}")
            else:
                # Unqualified column: ensure it exists in any referenced table
                if not any(col_l in cols for cols in column_lookup.values()):
                    missing_columns.append(col)

        if missing_columns:
            result["columns_exist"] = False
            result["errors"].append(f"Referenced columns not found in schema: {', '.join(sorted(set(missing_columns)))}")

        return result

    def _build_column_lookup(self, table_schemas: List[TableSchema], dialect: str) -> Dict[str, Set[str]]:
        """
        Build a mapping of table_name -> set(column_names) using column_comments first,
        falling back to DDL parsing.
        """
        lookup: Dict[str, Set[str]] = {}
        for schema in table_schemas:
            table_key = (schema.table_name or "").lower()
            columns: Set[str] = set()

            # 1) Column comments JSON if available
            col_comments = getattr(schema, "column_comments", None)
            if col_comments:
                try:
                    parsed = json.loads(col_comments)
                    if isinstance(parsed, dict):
                        columns.update([c.lower() for c in parsed.keys()])
                    elif isinstance(parsed, list):
                        columns.update([str(item.get("name", "")).lower() for item in parsed if item.get("name")])
                except Exception:
                    logger.debug("Failed to parse column_comments JSON for table %s", table_key)

            # 2) Parse DDL if still empty
            if (not columns) and getattr(schema, "definition", None):
                try:
                    metadata = extract_enhanced_metadata_from_ddl(schema.definition, dialect=dialect, warn_on_invalid=False)
                    for col in metadata.get("columns", []):
                        name = col.get("name")
                        if name:
                            columns.add(str(name).lower())
                except Exception as exc:
                    logger.debug("DDL parse failed for table %s: %s", table_key, exc)

            if columns:
                lookup[table_key] = columns
        return lookup

    def _extract_column_references(self, sql_query: str, dialect: str) -> List[Tuple[Optional[str], str]]:
        """
        Extract (table, column) pairs referenced in the SQL using sqlglot.

        Unqualified columns will have table=None.
        """
        try:
            import sqlglot
            from sqlglot import exp
        except Exception:
            return []

        try:
            ast = sqlglot.parse_one(sql_query, read=dialect or "mysql")
        except Exception:
            return []

        refs: List[Tuple[Optional[str], str]] = []
        for col in ast.find_all(exp.Column):
            table = col.table
            name = col.name
            if name:
                refs.append((table, name))
        return refs

    def _check_dangerous_operations(self, sql_query: str) -> Dict[str, Any]:
        """
        Check for potentially dangerous SQL operations.

        Returns:
            {
                "has_dangerous": bool,
                "warnings": List[str],
            }
        """
        result = {
            "has_dangerous": False,
            "warnings": [],
        }

        sql_upper = sql_query.strip().upper()

        # Check for DELETE without WHERE
        if sql_upper.startswith("DELETE") and " WHERE " not in sql_upper:
            result["has_dangerous"] = True
            result["warnings"].append("DELETE statement without WHERE clause - will delete all rows")

        # Check for UPDATE without WHERE
        if sql_upper.startswith("UPDATE") and " WHERE " not in sql_upper:
            result["has_dangerous"] = True
            result["warnings"].append("UPDATE statement without WHERE clause - will update all rows")

        # Check for DROP TABLE
        if "DROP TABLE" in sql_upper:
            result["has_dangerous"] = True
            result["warnings"].append("DROP TABLE detected - this will permanently delete the table")

        # Check for TRUNCATE
        if sql_upper.startswith("TRUNCATE"):
            result["has_dangerous"] = True
            result["warnings"].append("TRUNCATE detected - this will delete all rows from the table")

        # Check for CREATE TABLE
        if sql_upper.startswith("CREATE") and "TABLE" in sql_upper:
            result["has_dangerous"] = True
            result["warnings"].append("CREATE TABLE detected - schema modification not allowed in text2sql workflow")

        # Check for ALTER TABLE
        if "ALTER TABLE" in sql_upper:
            result["has_dangerous"] = True
            result["warnings"].append("ALTER TABLE detected - schema modification not allowed")

        # Check for CREATE INDEX
        if "CREATE INDEX" in sql_upper:
            result["has_dangerous"] = True
            result["warnings"].append("CREATE INDEX detected - schema modification not allowed")

        # Check for DROP INDEX
        if "DROP INDEX" in sql_upper:
            result["has_dangerous"] = True
            result["warnings"].append("DROP INDEX detected - schema modification not allowed")

        return result

    def update_context(self, workflow: "Workflow") -> Dict[str, Any]:
        """
        Update workflow context with validation results and handle termination.

        If validation failed, sets termination_status to skip to reflect node.
        """
        try:
            if not self.result:
                return {"success": False, "message": "SQL validation produced no result"}

            # Safely get data, defaulting to empty dict if None
            data = self.result.data if self.result.data else {}

            # Store validation result in workflow metadata
            if not hasattr(workflow, "metadata") or workflow.metadata is None:
                workflow.metadata = {}
            workflow.metadata["sql_validation"] = data

            # If validation failed, signal workflow to skip to reflect
            is_valid = data.get("is_valid", False)
            if not is_valid:
                workflow.metadata["termination_status"] = WorkflowTerminationStatus.SKIP_TO_REFLECT

                logger.info(f"SQL validation failed, skipping to reflect: " f"errors={data.get('errors', [])}")

            return {
                "success": True,
                "message": f"SQL validation context updated: is_valid={is_valid}",
            }

        except Exception as e:
            logger.error(f"Failed to update SQL validation context: {str(e)}")
            return {"success": False, "message": f"SQL validation context update failed: {str(e)}"}

    def execute(self) -> BaseResult:
        """Execute SQL validation synchronously."""
        return execute_with_async_stream(self)

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute SQL validation with streaming support."""
        if action_history_manager:
            self.action_history_manager = action_history_manager
        async for action in self.run():
            yield action
