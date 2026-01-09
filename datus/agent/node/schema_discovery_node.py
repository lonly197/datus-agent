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
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool.context_search import ContextSearchTools
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

            # Safely access intent from workflow metadata with comprehensive validation
            intent = "text2sql"  # Default fallback
            intent_confidence = 0.0

            if hasattr(self.workflow, "metadata") and self.workflow.metadata:
                detected_intent = self.workflow.metadata.get("detected_intent")
                confidence = self.workflow.metadata.get("intent_confidence", 0.0)

                if detected_intent and isinstance(detected_intent, str) and confidence > 0.3:
                    intent = detected_intent
                    intent_confidence = confidence
                    logger.debug(f"Using detected intent from workflow metadata: {intent} (confidence: {confidence})")
                else:
                    logger.warning(
                        f"Intent detection unreliable (intent: {detected_intent}, confidence: {confidence}), using default 'text2sql'"
                    )
            else:
                logger.warning("Workflow metadata not available, using default intent 'text2sql'")

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

            # Get candidate tables based on intent and task
            candidate_tables = await self._discover_candidate_tables(task, intent)

            # If we have candidate tables, load their schemas
            if candidate_tables:
                await self._load_table_schemas(task, candidate_tables)
            else:
                # Fail-Fast: If no tables found after all attempts, do not proceed
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

                # Mark node as failed
                self.result = BaseResult(success=False, error=error_msg)
                return

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

            # ✅ Fix: Set self.result to ensure node success
            self.result = BaseResult(
                success=True,
                data={
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
                {
                    "task_id": getattr(self.workflow.task, "id", "unknown")
                    if self.workflow and self.workflow.task
                    else "unknown"
                },
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

        # 1. If tables are explicitly mentioned in the task, use those (Highest Priority)
        if hasattr(task, "tables") and task.tables:
            candidate_tables.extend(task.tables)
            logger.info(f"Using explicitly specified tables: {task.tables}")

        # If intent is text2sql or sql (from intent analysis), try to discover tables
        if intent in ["text2sql", "sql"] and hasattr(task, "task"):
            task_text = task.task.lower()

            # --- Stage 1: Fast Cache/Keyword & Semantic (Hybrid Search) ---
            # 1. Semantic Search (High Priority)
            semantic_tables = await self._semantic_table_discovery(task.task)
            if semantic_tables:
                candidate_tables.extend(semantic_tables)
                logger.info(f"[Stage 1] Found {len(semantic_tables)} tables via semantic search")

            # 2. Keyword Matching (Medium Priority)
            # Optimization: Always run keyword search to improve recall (Hybrid Search)
            keyword_tables = self._keyword_table_discovery(task_text)
            if keyword_tables:
                candidate_tables.extend(keyword_tables)
                logger.info(f"[Stage 1] Found {len(keyword_tables)} tables via keyword matching")

            # Deduplicate
            candidate_tables = list(set(candidate_tables))

            # --- Stage 2: Deep Metadata Scan (Context Search) ---
            # If recall is low (e.g., 0 tables), try context search (Metrics, Reference SQL)
            if not candidate_tables:
                logger.info("[Stage 2] Stage 1 yielded 0 tables, initiating Context Search...")
                context_tables = await self._context_based_discovery(task.task)
                if context_tables:
                    candidate_tables.extend(context_tables)
                    logger.info(f"[Stage 2] Found {len(context_tables)} tables via context search")

            # Deduplicate
            candidate_tables = list(set(candidate_tables))

            # --- Stage 3: Full Structure Analysis (LLM Match - Placeholder/Future) ---
            # If still no tables, we could use LLM-based MatchSchemaTool here.
            # For now, we proceed to Fallback if still empty.

            # 4. Fallback: Get All Tables (Low Priority)
            # Only if absolutely no tables found from previous stages
            if not candidate_tables:
                logger.info("[Stage 3] No tables found, attempting Fallback (Get All Tables)...")
                fallback_tables = await self._fallback_get_all_tables(task)
                if fallback_tables:
                    candidate_tables.extend(fallback_tables)
                    logger.warning(f"[Stage 3] Used fallback: {len(fallback_tables)} tables")

        # If no candidates found, return empty list (downstream nodes will handle)
        return list(set(candidate_tables))  # Remove duplicates

    async def _context_based_discovery(self, query: str) -> List[str]:
        """
        Stage 2: Discover tables via Metrics and Reference SQL.
        """
        found_tables = []
        try:
            if not self.agent_config:
                return []

            context_search = ContextSearchTools(self.agent_config)

            # 1. Search Metrics
            if context_search.has_metrics:
                metric_result = context_search.search_metrics(query_text=query, top_n=3)
                if metric_result.success and metric_result.result:
                    # In a real impl, we would extract table names from metric definitions
                    # For now, we assume metric definitions might contain table references or hints
                    # This is a placeholder for extraction logic
                    pass

            # 2. Search Reference SQL
            # Reference SQL often contains valid table names
            ref_sql_result = context_search.search_reference_sql(query_text=query, top_n=3)
            if ref_sql_result.success and ref_sql_result.result:
                # Placeholder: Extract tables from SQL (would need sqlglot here)
                pass

        except Exception as e:
            logger.warning(f"Context-based discovery failed: {e}")

        return found_tables

    async def _semantic_table_discovery(self, query: str) -> List[str]:
        """
        Discover tables using semantic vector search.

        Args:
            query: User query text

        Returns:
            List of table names
        """
        try:
            if not self.agent_config:
                return []

            tables = []

            # 1. Use SchemaWithValueRAG for direct table semantic search
            rag = SchemaWithValueRAG(agent_config=self.agent_config)
            # Use search_similar to find tables based on vector similarity of their definitions
            schema_results, _ = rag.search_similar(query_text=query, top_n=5)

            if schema_results and len(schema_results) > 0:
                # Extract table names from arrow table
                found_tables = schema_results.column("table_name").to_pylist()
                if found_tables:
                    tables.extend(found_tables)
                    logger.info(f"Semantic search found tables via metadata: {found_tables}")

            # 2. Use ContextSearchTools for metrics/business logic search (complementary)
            context_search = ContextSearchTools(self.agent_config)
            if context_search.has_metrics:
                result = context_search.search_metrics(query_text=query, top_n=5)
                if result.success and result.result:
                    # Logic to extract tables from metrics would go here
                    # For now, we rely primarily on SchemaWithValueRAG
                    pass

            return list(set(tables))
        except Exception as e:
            logger.warning(f"Semantic table discovery failed: {e}")
            return []

    def _keyword_table_discovery(self, task_text: str) -> List[str]:
        """
        Discover tables using keyword matching.

        Args:
            task_text: Lowercase user query text

        Returns:
            List of table names
        """
        candidate_tables = []

        # Simple keyword-based table discovery
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
            # Chinese keywords support
            "用户",
            "user",
            "客户",
            "customer",
            "订单",
            "order",
            "产品",
            "product",
            "销售",
            "sale",
            "交易",
            "transaction",
            "员工",
            "employee",
            "部门",
            "department",
            "试驾",
            "test_drive",  # Based on user query
            "转化",
            "conversion",
            # Add more as needed
        ]

        # Iterate in pairs (keyword, table_name)
        # For English words, keyword == table_name (mostly)
        # For Chinese words, keyword maps to English table_name

        # Refined logic: the list above is mixed. Let's make it a mapping for better accuracy
        keyword_map = {
            "用户": "users",
            "user": "users",
            "客户": "customers",
            "customer": "customers",
            "订单": "orders",
            "order": "orders",
            "产品": "products",
            "product": "products",
            "销售": "sales",
            "sale": "sales",
            "交易": "transactions",
            "transaction": "transactions",
            "员工": "employees",
            "employee": "employees",
            "部门": "departments",
            "department": "departments",
            "试驾": "dwd_assign_dlr_clue_fact_di",  # Updated mapping based on log analysis
            "线索": "dwd_assign_dlr_clue_fact_di",
            "转化": "conversions",
        }

        # Check mapping first
        for keyword, table_name in keyword_map.items():
            if keyword in task_text:
                candidate_tables.append(table_name)
                # Also add singular/plural variations
                if table_name.endswith("s"):
                    candidate_tables.append(table_name[:-1])
                else:
                    candidate_tables.append(table_name + "s")

        # Fallback to the original list check for direct matches
        for keyword in table_keywords:
            if keyword in task_text and keyword not in keyword_map:
                # Assuming keyword itself is a potential table name
                candidate_tables.append(keyword)
                table_name = keyword.rstrip("s")
                if table_name != keyword:
                    candidate_tables.append(table_name)

        return candidate_tables

    async def _fallback_get_all_tables(self, task) -> List[str]:
        """
        Fallback: Get all tables from the database if no candidates found.

        Args:
            task: The SQL task

        Returns:
            List of all table names (limited to top N)
        """
        try:
            db_manager = db_manager_instance(self.agent_config.namespaces)
            connector = db_manager.get_conn(self.agent_config.current_namespace, task.database_name)

            if not connector:
                return []

            # Get all tables
            # Note: get_all_table_names might be expensive for large DBs
            # We should probably limit this or cache it
            all_tables = connector.get_all_table_names()

            # Limit to reasonable number to avoid context overflow
            MAX_TABLES = 50
            if len(all_tables) > MAX_TABLES:
                logger.warning(f"Too many tables ({len(all_tables)}), limiting to first {MAX_TABLES} for fallback")
                return all_tables[:MAX_TABLES]

            return all_tables

        except Exception as e:
            logger.warning(f"Fallback table discovery failed: {e}")
            return []

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
                return {"success": False, "message": "Schema discovery failed, cannot update context"}

            # If result has schema information, update workflow context
            if hasattr(self.result, "data") and self.result.data:
                output = self.result.data

                # Update candidate tables if available
                if "candidate_tables" in output:
                    # Store discovered tables in workflow metadata for downstream nodes
                    if not hasattr(workflow, "metadata"):
                        workflow.metadata = {}
                    workflow.metadata["discovered_tables"] = output["candidate_tables"]

                # Note: Table schemas are already loaded via _load_table_schemas() which calls
                # workflow.context.update_schema_and_values(). The schema loading happens
                # during the run() method before this update_context() is called.
                if "table_schemas" in output and workflow.context:
                    # Schemas were already loaded by _load_table_schemas(), log for verification
                    schema_count = len(workflow.context.table_schemas)
                    value_count = len(workflow.context.table_values)
                    logger.debug(f"Schema discovery context contains {schema_count} schemas and {value_count} values")

            return {
                "success": True,
                "message": f"Schema discovery context updated with {len(self.result.data.get('candidate_tables', []))} tables",
            }

        except Exception as e:
            logger.error(f"Failed to update schema discovery context: {str(e)}")
            return {"success": False, "message": f"Schema discovery context update failed: {str(e)}"}
