# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SchemaDiscoveryNode implementation for discovering relevant schema and tables.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseResult
from datus.schemas.node_models import BaseInput
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SchemaDiscoveryNode(Node):
    """
    Node for discovering relevant schema and tables for a query.

    This node analyzes the query intent and discovers candidate tables/columns
    that may be relevant, populating the workflow context for downstream nodes.
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
        """
        Setup schema discovery input from workflow context.

        Args:
            workflow: Workflow instance containing context and task

        Returns:
            Dictionary with success status
        """
        if not self.input:
            self.input = BaseInput()

        return {"success": True, "message": "Schema discovery input setup complete"}

    async def run(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Run schema discovery to find relevant tables and columns.

        Yields:
            ActionHistory: Progress and result actions
        """
        try:
            if not self.workflow or not self.workflow.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "No workflow or task available for schema discovery",
                    "schema_discovery",
                    {"workflow_id": getattr(self.workflow, 'id', 'unknown') if self.workflow else 'unknown'}
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Schema discovery failed: {error_result.error_message}",
                    action_type="schema_discovery",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            task = self.workflow.task

            # Safely access intent from workflow metadata with comprehensive validation
            intent = "text2sql"  # Default fallback
            intent_confidence = 0.0

            if hasattr(self.workflow, 'metadata') and self.workflow.metadata:
                detected_intent = self.workflow.metadata.get("detected_intent")
                confidence = self.workflow.metadata.get("intent_confidence", 0.0)

                if detected_intent and isinstance(detected_intent, str) and confidence > 0.3:
                    intent = detected_intent
                    intent_confidence = confidence
                    logger.debug(f"Using detected intent from workflow metadata: {intent} (confidence: {confidence})")
                else:
                    logger.warning(f"Intent detection unreliable (intent: {detected_intent}, confidence: {confidence}), using default 'text2sql'")
            else:
                logger.warning("Workflow metadata not available, using default intent 'text2sql'")

            # Validate task has required attributes
            if not hasattr(task, 'task') or not task.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "Task has no content to analyze for schema discovery",
                    "schema_discovery",
                    {"task_id": getattr(task, 'id', 'unknown')}
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Schema discovery failed: {error_result.error_message}",
                    action_type="schema_discovery",
                    input={"task_content": getattr(task, 'task', '')},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            # Get candidate tables based on intent and task
            candidate_tables = await self._discover_candidate_tables(task, intent)

            # If we have candidate tables, load their schemas
            if candidate_tables:
                await self._load_table_schemas(task, candidate_tables)

            # Emit success action with results
            yield ActionHistory(
                action_id=f"{self.id}_schema_discovery",
                role=ActionRole.TOOL,
                messages=f"Schema discovery completed: found {len(candidate_tables)} candidate tables",
                action_type="schema_discovery",
                input={
                    "intent": intent,
                    "catalog": task.catalog_name,
                    "database": task.database_name,
                    "schema": task.schema_name,
                },
                status=ActionStatus.SUCCESS,
                output={
                    "candidate_tables": candidate_tables,
                    "table_count": len(candidate_tables),
                    "intent": intent,
                },
            )

            logger.info(
                f"Schema discovery completed: found {len(candidate_tables)} candidate tables for intent '{intent}'"
            )

        except Exception as e:
            logger.error(f"Schema discovery failed: {e}")
            error_result = self.create_error_result(
                ErrorCode.NODE_EXECUTION_FAILED,
                f"Schema discovery execution failed: {str(e)}",
                "schema_discovery",
                {"task_id": getattr(self.workflow.task, 'id', 'unknown') if self.workflow and self.workflow.task else 'unknown'}
            )
            yield ActionHistory(
                action_id=f"{self.id}_error",
                role=ActionRole.TOOL,
                messages=f"Schema discovery failed: {error_result.error_message}",
                action_type="schema_discovery",
                input={"intent": intent},
                status=ActionStatus.FAILED,
                output={"error": error_result.error_message, "error_code": error_result.error_code},
            )

    async def _discover_candidate_tables(self, task, intent: str) -> List[str]:
        """
        Discover candidate tables based on task and intent.

        Args:
            task: The SQL task
            intent: Detected intent type

        Returns:
            List of candidate table names
        """
        # For now, use a simple heuristic approach
        # In production, this could use semantic search or LLM-based table discovery

        candidate_tables = []

        # If tables are explicitly mentioned in the task, use those
        if hasattr(task, "tables") and task.tables:
            candidate_tables.extend(task.tables)

        # If intent is text2sql, try to extract table hints from the query text
        if intent == "text2sql" and hasattr(task, "task"):
            task_text = task.task.lower()

            # Simple keyword-based table discovery (can be enhanced with RAG)
            table_keywords = [
                # Common business tables
                "user",
                "users",
                "customer",
                "customers",
                "order",
                "orders",
                "product",
                "products",
                "sale",
                "sales",
                "transaction",
                "transactions",
                "employee",
                "employees",
                "department",
                "departments",
                # Add more as needed
            ]

            for keyword in table_keywords:
                if keyword in task_text:
                    # Convert singular/plural variations
                    table_name = keyword.rstrip("s")  # Simple singularization
                    if table_name not in candidate_tables:
                        candidate_tables.append(table_name)

        # If no candidates found, return empty list (downstream nodes will handle)
        return list(set(candidate_tables))  # Remove duplicates

    async def _load_table_schemas(self, task, candidate_tables: List[str]) -> None:
        """
        Load schemas for candidate tables and update workflow context.

        Args:
            task: The SQL task
            candidate_tables: List of table names to load schemas for
        """
        if not candidate_tables or not self.workflow:
            return

        try:
            # Use existing SchemaWithValueRAG to load table schemas
            if self.agent_config:
                rag = SchemaWithValueRAG(agent_config=self.agent_config)

                schemas, values = rag.search_tables(
                    tables=candidate_tables,
                    catalog_name=task.catalog_name or "",
                    database_name=task.database_name or "",
                    schema_name=task.schema_name or "",
                    dialect=task.database_type if hasattr(task, "database_type") else None,
                )

                if schemas:
                    # Update workflow context with discovered schemas
                    self.workflow.context.update_schema_and_values(schemas, values)
                    logger.debug(f"Loaded schemas for {len(schemas)} tables")
                else:
                    logger.warning(f"No schemas found for candidate tables: {candidate_tables}")

        except Exception as e:
            logger.warning(f"Failed to load table schemas: {e}")
            # Don't fail the entire node for schema loading issues


    def execute(self) -> BaseResult:
        """
        Execute schema discovery synchronously.

        Returns:
            BaseResult: The result of schema discovery execution
        """
        # execute_with_async_stream ensures self.result is properly set
        return execute_with_async_stream(self)

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute schema discovery with streaming support.

        Args:
            action_history_manager: Manager for tracking action history

        Yields:
            ActionHistory: Progress and result actions during execution
        """
        # Set the action_history_manager if provided
        if action_history_manager:
            self.action_history_manager = action_history_manager

        # Delegate to the existing run method
        async for action in self.run():
            yield action

    def update_context(self, workflow: "Workflow") -> Dict[str, Any]:
        """
        Update workflow context with schema discovery results.

        Args:
            workflow: The workflow instance to update

        Returns:
            Dict with success status and message
        """
        try:
            if not self.result or not self.result.success:
                return {
                    "success": False,
                    "message": "Schema discovery failed, cannot update context"
                }

            # If result has schema information, update workflow context
            if hasattr(self.result, 'output') and self.result.output:
                output = self.result.output

                # Update candidate tables if available
                if 'candidate_tables' in output:
                    # Store discovered tables in workflow metadata for downstream nodes
                    if not hasattr(workflow, 'metadata'):
                        workflow.metadata = {}
                    workflow.metadata['discovered_tables'] = output['candidate_tables']

                # Update table schemas if they were loaded
                if 'table_schemas' in output and workflow.context:
                    workflow.context.table_schemas.update(output.get('table_schemas', {}))
                    workflow.context.table_values.update(output.get('table_values', {}))

            return {
                "success": True,
                "message": f"Schema discovery context updated with {len(self.result.output.get('candidate_tables', []))} tables"
            }

        except Exception as e:
            logger.error(f"Failed to update schema discovery context: {str(e)}")
            return {
                "success": False,
                "message": f"Schema discovery context update failed: {str(e)}"
            }
