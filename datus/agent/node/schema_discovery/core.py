# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Core SchemaDiscoveryNode class.

This module contains the main SchemaDiscoveryNode class that orchestrates
the schema discovery process using all the refactored modules.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import (
    ActionHistory,
    ActionRole,
    ActionStatus,
)
from datus.schemas.base import BaseResult
from datus.schemas.node_models import BaseInput
from datus.utils.error_handling import LLMMixin, NodeExecutionResult
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.schema_discovery_metrics import (
    SchemaDiscoveryMetrics,
    global_metrics_collector,
)

from .discovery_engine import (
    discover_candidate_tables,
    finalize_candidates,
)
from .knowledge_enhancement import (
    combine_knowledge,
    llm_based_schema_matching,
    search_and_enhance_external_knowledge,
)
from .llm_discovery import llm_based_table_discovery
from .schema_loader import (
    ddl_fallback_and_retry,
    fallback_get_all_tables,
    filter_candidate_tables_for_db,
    get_all_database_tables,
    load_table_schemas,
    repair_metadata,
)
from .search_strategies import (
    context_based_discovery,
    keyword_table_discovery,
    semantic_table_discovery,
)
from .utils import (
    _contains_chinese,
    build_query_context,
    check_rerank_resources,
    extract_potential_tables_from_text,
    rewrite_fts_query_with_llm_wrapper,
)

logger = get_logger(__name__)


class SchemaDiscoveryNode(Node, LLMMixin):
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
        Node.__init__(
            self,
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
        )
        LLMMixin.__init__(self)
        self._rerank_check_cache: Dict[tuple, Dict[str, Any]] = {}
        self._metrics: Optional[SchemaDiscoveryMetrics] = None

    def _get_rerank_check(
        self, model_path: str, min_cpu_count: int, min_memory_gb: float
    ) -> Dict[str, Any]:
        """Get cached rerank resource check result."""
        cache_key = (model_path, min_cpu_count, min_memory_gb)
        cached = self._rerank_check_cache.get(cache_key)
        if cached is not None:
            return cached
        result = check_rerank_resources(model_path, min_cpu_count, min_memory_gb)
        self._rerank_check_cache[cache_key] = result
        return result

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
        if not self.workflow:
            logger.error("Workflow not initialized in SchemaDiscoveryNode")
            return

        try:
            if not self.workflow.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "No workflow or task available for schema discovery",
                    "schema_discovery",
                    {"workflow_id": getattr(self.workflow, "id", "unknown") if self.workflow else "unknown"},
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

            # Safely access intent from workflow metadata
            intent = "text2sql"  # Default fallback

            if hasattr(self.workflow, "metadata") and self.workflow.metadata:
                detected_intent = self.workflow.metadata.get("detected_intent")
                try:
                    confidence = float(self.workflow.metadata.get("intent_confidence", 0.0))
                except (ValueError, TypeError):
                    confidence = 0.0

                if detected_intent and isinstance(detected_intent, str) and confidence > 0.3:
                    intent = detected_intent
                    logger.debug(f"Using detected intent from workflow metadata: {intent} (confidence: {confidence})")
                else:
                    logger.warning(
                        f"Intent detection unreliable (intent: {detected_intent}, confidence: {confidence}), using default 'text2sql'"
                    )
            else:
                logger.warning("Workflow metadata not available, using default intent 'text2sql'")

            # Use clarified_task if available
            if hasattr(self.workflow, "metadata") and self.workflow.metadata:
                clarified_task = self.workflow.metadata.get("clarified_task")

                if clarified_task and clarified_task != task.task:
                    logger.info(
                        f"Using clarified task for schema discovery: "
                        f"'{task.task[:50]}...' → '{clarified_task[:50]}...'"
                    )
                    self._original_task_text = task.task
                    task.task = clarified_task

            # Validate task has required attributes
            if not hasattr(task, "task") or not task.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "Task has no content to analyze for schema discovery",
                    "schema_discovery",
                    {"task_id": getattr(task, "id", "unknown")},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Schema discovery failed: {error_result.error_message}",
                    action_type="schema_discovery",
                    input={"task_content": getattr(task, "task", "")},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            # Prepare helper functions
            async def llm_call_wrapper(**kwargs):
                return await self.llm_call_with_retry(**kwargs)

            async def llm_schema_matching_wrapper(query: str, matching_rate: str):
                return await llm_based_schema_matching(
                    query=query,
                    matching_rate=matching_rate,
                    agent_config=self.agent_config,
                    workflow=self.workflow,
                    model=self.model,
                )

            async def search_enhance_wrapper(user_query: str, subject_path):
                return await search_and_enhance_external_knowledge(
                    user_query=user_query,
                    subject_path=subject_path,
                    agent_config=self.agent_config,
                    workflow=self.workflow,
                )

            def combine_knowledge_wrapper(original: str, enhanced: str):
                return combine_knowledge(original, enhanced)

            def apply_progressive_wrapper(base_rate: str):
                return self._apply_progressive_matching(base_rate)

            def extract_tables_wrapper(text: str):
                return extract_potential_tables_from_text(text)

            def rewrite_fts_wrapper(query: str):
                return rewrite_fts_query_with_llm_wrapper(
                    query_text=query,
                    agent_config=self.agent_config,
                    model=self.model,
                )

            # Get candidate tables
            candidate_tables, candidate_details, discovery_stats = await discover_candidate_tables(
                task=task,
                intent=intent,
                workflow=self.workflow,
                agent_config=self.agent_config,
                metrics=self._metrics,
                llm_call_func=llm_call_wrapper,
                llm_schema_matching_func=llm_schema_matching_wrapper,
                search_enhance_func=search_enhance_wrapper,
                combine_knowledge_func=combine_knowledge_wrapper,
                apply_progressive_func=apply_progressive_wrapper,
                extract_tables_func=extract_tables_wrapper,
                rewrite_fts_func=rewrite_fts_wrapper,
                rerank_check_func=self._get_rerank_check,
            )

            # Load table schemas
            if candidate_tables:
                async def repair_metadata_wrapper(tables, storage, task):
                    return await repair_metadata(
                        table_names=tables,
                        schema_storage=storage,
                        task=task,
                        agent_config=self.agent_config,
                    )

                async def ddl_fallback_wrapper(tables, task):
                    return await ddl_fallback_and_retry(
                        candidate_tables=tables,
                        task=task,
                        agent_config=self.agent_config,
                        get_ddl_func=lambda tn, conn, t: get_ddl_with_fallbacks(tn, conn, t),
                    )

                await load_table_schemas(
                    task=task,
                    candidate_tables=candidate_tables,
                    agent_config=self.agent_config,
                    workflow=self.workflow,
                    repair_metadata_func=repair_metadata_wrapper,
                    ddl_fallback_func=ddl_fallback_wrapper,
                )
            else:
                # Fail-Fast: No tables found
                error_msg = f"No candidate tables found for task '{task.task[:50]}...'. Cannot generate SQL without schema context."
                logger.error(error_msg)

                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    error_msg,
                    "schema_discovery",
                    {"task_id": getattr(task, "id", "unknown")},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error_nofound",
                    role=ActionRole.TOOL,
                    messages=f"Schema discovery failed: {error_msg}\nTry explicitly mentioning table names or checking database connection.",
                    action_type="schema_discovery",
                    input={"intent": intent},
                    status=ActionStatus.FAILED,
                    output={"error": error_msg, "error_code": ErrorCode.NODE_EXECUTION_FAILED},
                )

                self.result = BaseResult(success=False, error=error_msg)
                return

            # Emit success action
            yield ActionHistory(
                action_id=f"{self.id}_schema_discovery",
                role=ActionRole.TOOL,
                messages=f"Schema discovery completed: found {len(candidate_tables)} candidate tables",
                action_type="schema_discovery",
                input={
                    "intent": intent,
                    "catalog": task.catalog_name,
                    "database": task.database_name,
                    "schema": task.schema_name or None,
                    "query_context": build_query_context(task),
                },
                status=ActionStatus.SUCCESS,
                output={
                    "candidate_tables": candidate_tables,
                    "table_count": len(candidate_tables),
                    "intent": intent,
                    "table_candidates": candidate_details,
                    "discovery_stats": discovery_stats,
                },
            )

            self.result = BaseResult(
                success=True,
                data={
                    "candidate_tables": candidate_tables,
                    "table_count": len(candidate_tables),
                    "intent": intent,
                    "table_candidates": candidate_details,
                    "discovery_stats": discovery_stats,
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
                messages=f"Schema discovery failed: {error_result.error_message}",
                action_type="schema_discovery",
                input={},
                status=ActionStatus.FAILED,
                output={"error": error_result.error_message, "error_code": error_result.error_code},
            )

        finally:
            # Restore original task
            original_task_text = getattr(self, "_original_task_text", None)
            if original_task_text is not None and self.workflow and self.workflow.task:
                self.workflow.task.task = original_task_text
                logger.info("Restored original task to prevent parameter pollution")
                self._original_task_text = None

    def execute(self) -> BaseResult:
        """Execute the node synchronously."""
        return execute_with_async_stream(self._run_async())

    async def execute_stream(self) -> AsyncGenerator[ActionHistory, None]:
        """Execute the node with streaming output."""
        async for action in self.run():
            yield action

    def update_context(self, workflow: "Workflow") -> Dict[str, Any]:
        """
        Update workflow context with discovered schemas.

        Args:
            workflow: Workflow instance to update

        Returns:
            Dictionary with update status
        """
        if not self.result or not self.result.success:
            return {"success": False, "message": "Node execution failed"}

        try:
            # Context is updated during run() via load_table_schemas
            return {"success": True, "message": "Context updated"}
        except Exception as e:
            logger.error(f"Failed to update context: {e}")
            return {"success": False, "error": str(e)}

    def _apply_progressive_matching(self, base_rate: str = "fast") -> str:
        """
        Apply progressive matching strategy based on reflection round.

        Args:
            base_rate: Base matching rate from configuration

        Returns:
            Adjusted matching rate
        """
        # Check if we're in a reflection round
        if self.workflow and hasattr(self.workflow, "metadata"):
            reflection_round = self.workflow.metadata.get("reflection_round", 0)
            if reflection_round > 0:
                logger.info(f"Progressive matching: using 'exhaustive' rate for reflection round {reflection_round}")
                return "exhaustive"

        return base_rate


async def get_ddl_with_fallbacks(
    table_name: str,
    connector,
    task,
) -> str:
    """
    Get DDL for a table with multiple fallback strategies.

    Args:
        table_name: Name of the table
        connector: Database connector
        task: SQL task

    Returns:
        DDL statement as string
    """
    try:
        # Try primary method
        if hasattr(connector, "get_ddl"):
            ddl = connector.get_ddl(
                table_name=table_name,
                catalog_name=task.catalog_name or "",
                database_name=task.database_name or "",
                schema_name=task.schema_name or "",
            )
            if ddl:
                return ddl
    except Exception as e:
        logger.debug(f"Primary DDL method failed for {table_name}: {e}")

    # Fallback to SHOW CREATE TABLE
    try:
        if hasattr(connector, "execute_sql"):
            result = connector.execute_sql(f"SHOW CREATE TABLE {table_name}")
            if result:
                return str(result)
    except Exception as e:
        logger.debug(f"SHOW CREATE TABLE failed for {table_name}: {e}")

    return ""
