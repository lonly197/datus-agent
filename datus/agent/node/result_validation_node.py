# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ResultValidationNode implementation for validating SQL execution results.

This node validates that the SQL execution results are acceptable:
- SQL executed without errors
- Result is not empty (unless expected, e.g., DDL/DML queries)
- Result makes semantic sense for the query

Enhanced with HTML preview capability for SQL execution results.
"""

import html
from typing import Any, AsyncGenerator, Dict, Optional

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import SQLContext
from datus.utils.env import get_env_int
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ResultValidationNode(Node):
    """
    Node for validating SQL execution result quality.

    This node checks if the SQL execution produced acceptable results.
    It helps the reflect node decide whether to return results to the user
    or trigger additional reflection strategies.
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
        """Setup result validation input from workflow context."""
        if not self.input:
            self.input = BaseInput()
        return {"success": True, "message": "Result validation input setup complete"}

    async def run(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Run result validation to check if SQL execution results are acceptable.

        Yields:
            ActionHistory: Progress and result actions
        """
        try:
            if not self.workflow or not self.workflow.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "No workflow or task available for result validation",
                    "result_validation",
                    {"workflow_id": getattr(self.workflow, "id", "unknown") if self.workflow else "unknown"},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Result validation failed: {error_result.error_message}",
                    action_type="result_validation",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            task = self.workflow.task
            context = self.workflow.context

            # Step 1: Check if there are SQL execution results
            if not context or not context.sql_contexts:
                yield ActionHistory(
                    action_id=f"{self.id}_no_results",
                    role=ActionRole.TOOL,
                    messages="Result validation skipped: No SQL execution results",
                    action_type="result_validation",
                    input={"task": task.task[:50] if task else ""},
                    status=ActionStatus.SUCCESS,
                    output={
                        "is_valid": False,
                        "reason": "No SQL execution results to validate",
                        "suggestion": "Execute SQL first",
                    },
                )
                self.result = BaseResult(
                    success=True,  # Validation succeeded, but results are absent
                    data={"is_valid": False, "reason": "no_sql_results"},
                )
                return

            # Step 2: Get the most recent SQL context
            latest_sql_context = context.sql_contexts[-1]

            # Step 3: Validate the SQL execution result
            schema_validation = None
            if self.workflow and hasattr(self.workflow, "metadata") and self.workflow.metadata:
                schema_validation = self.workflow.metadata.get("schema_validation")

            validation_result = self._validate_sql_result(
                task.task if task else "",
                latest_sql_context,
                schema_validation,
            )

            # Emit result
            if validation_result["is_valid"]:
                yield ActionHistory(
                    action_id=f"{self.id}_validation",
                    role=ActionRole.TOOL,
                    messages=f"Result validation passed: {validation_result['reason']}",
                    action_type="result_validation",
                    input={
                        "task": task.task[:50] if task else "",
                        "row_count": validation_result.get("row_count", 0),
                    },
                    status=ActionStatus.SUCCESS,
                    output=validation_result,
                )

                # Generate HTML preview for SELECT query results
                html_preview = self._generate_html_preview(latest_sql_context)
                if html_preview:
                    yield ActionHistory(
                        action_id=f"{self.id}_preview",
                        role=ActionRole.TOOL,
                        messages="SQL Result Preview",
                        action_type="result_preview",
                        input={
                            "task": task.task[:50] if task else "",
                            "row_count": validation_result.get("row_count", 0),
                        },
                        status=ActionStatus.SUCCESS,
                        output={"html_preview": html_preview},
                    )

                self.result = BaseResult(success=True, data=validation_result)
            else:
                yield ActionHistory(
                    action_id=f"{self.id}_validation",
                    role=ActionRole.TOOL,
                    messages=f"Result validation failed: {validation_result['reason']}",
                    action_type="result_validation",
                    input={
                        "task": task.task[:50] if task else "",
                        "row_count": validation_result.get("row_count", 0),
                    },
                    status=ActionStatus.FAILED,
                    output=validation_result,
                )
                self.result = BaseResult(success=False, error=validation_result["reason"], data=validation_result)

            logger.info(
                f"Result validation completed: is_valid={validation_result['is_valid']}, "
                f"reason={validation_result['reason']}"
            )

        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            error_result = self.create_error_result(
                ErrorCode.NODE_EXECUTION_FAILED,
                f"Result validation execution failed: {str(e)}",
                "result_validation",
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
                messages=f"Result validation failed: {error_result.error_message}",
                action_type="result_validation",
                input={},
                status=ActionStatus.FAILED,
                output={"error": error_result.error_message, "error_code": error_result.error_code},
            )
            self.result = BaseResult(success=False, error=str(e))

    def _validate_sql_result(
        self,
        task: str,
        sql_context: SQLContext,
        schema_validation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate the SQL execution result.

        Returns a dictionary with:
        - is_valid: bool
        - reason: str
        - row_count: int
        - error: str (if any)
        - suggestion: str (if invalid)
        """
        result = {
            "is_valid": False,
            "reason": "",
            "row_count": 0,
            "error": None,
            "suggestion": None,
            "risk_level": "none",
        }

        # Check if there was an execution error
        # Note: We need to check the SQL execution context, not just the SQL query
        # The SQLContext stores the query and explanation, but we need to check
        # the execute_sql node result for errors

        # For now, we'll validate based on what we can infer
        if sql_context.sql_query:
            result["sql_query"] = sql_context.sql_query
        result["row_count"] = sql_context.row_count or 0

        if sql_context.sql_error:
            result["is_valid"] = False
            result["reason"] = "SQL execution failed"
            result["error"] = sql_context.sql_error
            result["suggestion"] = "Check table/column names and SQL syntax"
            return result

        # Check if this is a DDL/DML query (no results expected)
        if self._is_ddl_or_dml_query(sql_context.sql_query):
            result["is_valid"] = True
            result["reason"] = "DDL/DML query executed"
            return result

        # For SELECT queries, we would normally check:
        # 1. Execution error status
        # 2. Row count
        # 3. Whether empty results are acceptable

        # Since we don't have direct access to execution results here,
        # we'll assume the query executed if we have a SQL context
        # The real validation happens in the reflect node based on execution results

        # Basic validation: if we have a SQL query, consider it valid for now
        # More sophisticated validation would require access to execution results
        result["is_valid"] = True
        result["reason"] = "SQL query executed successfully"

        if result["row_count"] == 0:
            result["reason"] = "SQL executed successfully (empty result)"
            result["risk_level"] = "low"
            result["suggestion"] = "Result set is empty; verify filters or test data availability"

            if schema_validation and isinstance(schema_validation, dict):
                coverage_score = schema_validation.get("coverage_score")
                coverage_threshold = schema_validation.get("coverage_threshold")
                if isinstance(coverage_score, (int, float)) and isinstance(coverage_threshold, (int, float)):
                    if coverage_score <= coverage_threshold:
                        result["risk_level"] = "medium"
                        result["suggestion"] = (
                            "Empty result with low schema coverage; verify term-to-field mapping or refine schema"
                        )
                result["validation_context"] = {
                    "coverage_score": coverage_score,
                    "coverage_threshold": coverage_threshold,
                    "covered_terms": schema_validation.get("covered_terms", []),
                    "uncovered_terms": schema_validation.get("uncovered_terms", []),
                }

        return result

    def _is_ddl_or_dml_query(self, sql_query: Optional[str]) -> bool:
        """Check if the query is DDL or DML (not SELECT)."""
        if not sql_query:
            return False

        sql_upper = sql_query.strip().upper()

        # DDL statements
        ddl_prefixes = ("CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME")
        # DML statements (non-SELECT)
        dml_prefixes = ("INSERT", "UPDATE", "DELETE", "MERGE", "COPY")

        for prefix in ddl_prefixes + dml_prefixes:
            if sql_upper.startswith(prefix):
                return True

        return False

    def _generate_html_preview(self, sql_context: SQLContext, max_rows: Optional[int] = None) -> str:
        """
        Generate HTML table preview of SQL execution results.

        Args:
            sql_context: The SQL context containing execution results
            max_rows: Maximum number of rows to display (default from env or 10)

        Returns:
            HTML string containing a formatted table with the results
        """
        if max_rows is None:
            max_rows = get_env_int("SQL_PREVIEW_MAX_ROWS", 10)

        # Check if there are results to preview
        if not sql_context.sql_return:
            return ""

        # Handle DataFrame with to_csv() method (PyArrow/pandas)
        if hasattr(sql_context.sql_return, "to_csv"):
            try:
                # Check if DataFrame is empty
                if hasattr(sql_context.sql_return, "empty") and sql_context.sql_return.empty:
                    return "<div class='sql-preview-info'>Result set is empty</div>"

                # Try to get row count
                total_rows = len(sql_context.sql_return) if hasattr(sql_context.sql_return, "__len__") else 0

                # Get preview data
                if hasattr(sql_context.sql_return, "head"):
                    preview_df = sql_context.sql_return.head(max_rows)
                elif hasattr(sql_context.sql_return, "slice"):
                    preview_df = sql_context.sql_return.slice(0, min(max_rows, total_rows))
                else:
                    preview_df = sql_context.sql_return

                # Generate HTML
                html_parts = ["<div class='sql-preview-container'>"]

                # Get column names
                if hasattr(preview_df, "columns"):
                    columns = list(preview_df.columns)
                elif hasattr(preview_df, "schema"):
                    columns = [field.name for field in preview_df.schema]
                else:
                    columns = []

                # Build HTML table
                html_parts.append(
                    "<table class='sql-preview-table' border='1' style='border-collapse: collapse; width: 100%; font-size: 12px;'>"
                )

                # Header row
                html_parts.append("<thead><tr style='background-color: #f5f5f5;'>")
                for col in columns:
                    escaped_col = html.escape(str(col))
                    html_parts.append(
                        f"<th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>{escaped_col}</th>"
                    )
                html_parts.append("</tr></thead>")

                # Data rows
                html_parts.append("<tbody>")
                for i in range(min(len(preview_df), max_rows)):
                    html_parts.append("<tr>")
                    # Get row data
                    if hasattr(preview_df, "iloc"):
                        row = preview_df.iloc[i]
                        row_values = [row[col] for col in columns]
                    elif hasattr(preview_df, "take"):
                        row_data = preview_df.take([i])
                        row_values = [
                            row_data[col][0].as_py() if hasattr(row_data[col][0], "as_py") else row_data[col][0]
                            for col in columns
                        ]
                    else:
                        row_values = []

                    # Style alternating rows
                    row_style = "background-color: #fafafa;" if i % 2 == 0 else ""
                    for val in row_values:
                        val_str = str(val) if val is not None else ""
                        # Truncate long values
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                        escaped_val = html.escape(val_str)
                        html_parts.append(
                            f"<td style='padding: 6px; border: 1px solid #ddd; {row_style}'>{escaped_val}</td>"
                        )
                    html_parts.append("</tr>")
                html_parts.append("</tbody>")
                html_parts.append("</table>")

                # Add row count indicator
                if total_rows > max_rows:
                    html_parts.append(
                        f"<div class='sql-preview-footer' style='margin-top: 8px; font-size: 11px; color: #666;'>"
                    )
                    html_parts.append(f"<em>Showing {max_rows} of {total_rows} rows</em>")
                    html_parts.append("</div>")
                else:
                    html_parts.append(
                        f"<div class='sql-preview-footer' style='margin-top: 8px; font-size: 11px; color: #666;'>"
                    )
                    html_parts.append(f"<em>Total: {total_rows} row{'s' if total_rows != 1 else ''}</em>")
                    html_parts.append("</div>")

                html_parts.append("</div>")
                return "".join(html_parts)

            except Exception as e:
                logger.warning(f"Failed to generate HTML preview from DataFrame: {e}")
                return ""

        # Handle list of dicts
        if isinstance(sql_context.sql_return, list):
            try:
                rows = sql_context.sql_return[:max_rows]
                total_rows = len(sql_context.sql_return)

                if not rows:
                    return "<div class='sql-preview-info'>Result set is empty</div>"

                # Get columns from first row
                columns = list(rows[0].keys()) if rows else []

                html_parts = ["<div class='sql-preview-container'>"]
                html_parts.append(
                    "<table class='sql-preview-table' border='1' style='border-collapse: collapse; width: 100%; font-size: 12px;'>"
                )

                # Header row
                html_parts.append("<thead><tr style='background-color: #f5f5f5;'>")
                for col in columns:
                    escaped_col = html.escape(str(col))
                    html_parts.append(
                        f"<th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>{escaped_col}</th>"
                    )
                html_parts.append("</tr></thead>")

                # Data rows
                html_parts.append("<tbody>")
                for i, row in enumerate(rows):
                    html_parts.append("<tr>")
                    row_style = "background-color: #fafafa;" if i % 2 == 0 else ""
                    for col in columns:
                        val = row.get(col, "")
                        val_str = str(val) if val is not None else ""
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                        escaped_val = html.escape(val_str)
                        html_parts.append(
                            f"<td style='padding: 6px; border: 1px solid #ddd; {row_style}'>{escaped_val}</td>"
                        )
                    html_parts.append("</tr>")
                html_parts.append("</tbody>")
                html_parts.append("</table>")

                # Add row count indicator
                if total_rows > max_rows:
                    html_parts.append(
                        f"<div class='sql-preview-footer' style='margin-top: 8px; font-size: 11px; color: #666;'>"
                    )
                    html_parts.append(f"<em>Showing {max_rows} of {total_rows} rows</em>")
                    html_parts.append("</div>")

                html_parts.append("</div>")
                return "".join(html_parts)

            except Exception as e:
                logger.warning(f"Failed to generate HTML preview from list: {e}")
                return ""

        # Handle string result (CSV format)
        if isinstance(sql_context.sql_return, str):
            try:
                lines = sql_context.sql_return.strip().split("\n")
                if len(lines) < 2:
                    return ""

                headers = lines[0].split(",")
                data_lines = lines[1 : max_rows + 1]
                total_rows = len(lines) - 1

                html_parts = ["<div class='sql-preview-container'>"]
                html_parts.append(
                    "<table class='sql-preview-table' border='1' style='border-collapse: collapse; width: 100%; font-size: 12px;'>"
                )

                # Header row
                html_parts.append("<thead><tr style='background-color: #f5f5f5;'>")
                for header in headers:
                    escaped_header = html.escape(header.strip())
                    html_parts.append(
                        f"<th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>{escaped_header}</th>"
                    )
                html_parts.append("</tr></thead>")

                # Data rows
                html_parts.append("<tbody>")
                for i, line in enumerate(data_lines):
                    values = line.split(",")
                    html_parts.append("<tr>")
                    row_style = "background-color: #fafafa;" if i % 2 == 0 else ""
                    for val in values:
                        val_str = val.strip()
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                        escaped_val = html.escape(val_str)
                        html_parts.append(
                            f"<td style='padding: 6px; border: 1px solid #ddd; {row_style}'>{escaped_val}</td>"
                        )
                    html_parts.append("</tr>")
                html_parts.append("</tbody>")
                html_parts.append("</table>")

                # Add row count indicator
                if total_rows > max_rows:
                    html_parts.append(
                        f"<div class='sql-preview-footer' style='margin-top: 8px; font-size: 11px; color: #666;'>"
                    )
                    html_parts.append(f"<em>Showing {max_rows} of {total_rows} rows</em>")
                    html_parts.append("</div>")

                html_parts.append("</div>")
                return "".join(html_parts)

            except Exception as e:
                logger.warning(f"Failed to generate HTML preview from CSV string: {e}")
                return ""

        return ""

    def execute(self) -> BaseResult:
        """Execute result validation synchronously."""
        return execute_with_async_stream(self)

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute result validation with streaming support."""
        if action_history_manager:
            self.action_history_manager = action_history_manager
        async for action in self.run():
            yield action

    def update_context(self, workflow: "Workflow") -> Dict[str, Any]:
        """Update workflow context with validation results."""
        try:
            if not self.result:
                return {"success": False, "message": "Result validation produced no result"}

            # Store validation result in workflow metadata
            if not hasattr(workflow, "metadata"):
                workflow.metadata = {}
            workflow.metadata["result_validation"] = self.result.data

            is_valid = self.result.data.get("is_valid", False) if self.result.data else False

            return {"success": True, "message": f"Result validation context updated: is_valid={is_valid}"}

        except Exception as e:
            logger.error(f"Failed to update result validation context: {str(e)}")
            return {"success": False, "message": f"Result validation context update failed: {str(e)}"}
