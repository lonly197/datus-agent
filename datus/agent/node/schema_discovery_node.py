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
from datus.configuration.business_term_config import (
    LLM_TABLE_DISCOVERY_CONFIG,
    get_business_term_mapping,
)
from datus.schemas.action_history import (
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)
from datus.schemas.base import BaseResult
from datus.schemas.node_models import BaseInput, Metric, ReferenceSql, TableSchema
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.tools.db_tools.db_manager import get_db_manager
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.utils.context_lock import safe_context_update
from datus.utils.error_handler import LLMMixin, NodeExecutionResult
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names

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

            if hasattr(self.workflow, "metadata") and self.workflow.metadata:
                detected_intent = self.workflow.metadata.get("detected_intent")
                confidence = self.workflow.metadata.get("intent_confidence", 0.0)

                if detected_intent and isinstance(detected_intent, str) and confidence > 0.3:
                    intent = detected_intent
                    logger.debug(f"Using detected intent from workflow metadata: {intent} (confidence: {confidence})")
                else:
                    logger.warning(
                        f"Intent detection unreliable (intent: {detected_intent}, confidence: {confidence}), using default 'text2sql'"
                    )
            else:
                logger.warning("Workflow metadata not available, using default intent 'text2sql'")

            # ✅ Use clarified_task if available (from IntentClarificationNode)
            # Check if intent clarification has occurred and use the clarified version
            if hasattr(self.workflow, "metadata") and self.workflow.metadata:
                clarified_task = self.workflow.metadata.get("clarified_task")
                original_task = self.workflow.metadata.get("original_task")

                if clarified_task and clarified_task != task.task:
                    logger.info(
                        f"Using clarified task for schema discovery: "
                        f"'{task.task[:50]}...' → '{clarified_task[:50]}...'"
                    )
                    # Temporarily use clarified task for schema discovery
                    original_task_text = task.task
                    task.task = clarified_task

                    # Store clarification metadata for logging
                    clarification_result = self.workflow.metadata.get("intent_clarification", {})
                    if clarification_result:
                        entities = clarification_result.get("entities", {})
                        corrections = clarification_result.get("corrections", {})
                        logger.info(
                            f"Intent clarification metadata: "
                            f"business_terms={entities.get('business_terms', [])}, "
                            f"typos_fixed={corrections.get('typos_fixed', [])}"
                        )

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

            # 3. LLM Inference (Stage 1.5) - Enhance recall for Chinese/Ambiguous queries
            llm_tables = await self._llm_based_table_discovery(task.task)
            if llm_tables:
                candidate_tables.extend(llm_tables)
                logger.info(f"[Stage 1.5] Found {len(llm_tables)} tables via LLM inference")

            # Deduplicate
            candidate_tables = list(set(candidate_tables))

            # --- Stage 2: Deep Metadata Scan (Context Search) ---
            # ✅ Optimized: More aggressive trigger condition (was <3, now <10)
            # This ensures better recall by using context search more frequently
            if len(candidate_tables) < 10:
                logger.info(f"[Stage 2] Tables found ({len(candidate_tables)}) below threshold (10), initiating Context Search...")
                context_tables = await self._context_based_discovery(task.task)
                if context_tables:
                    candidate_tables.extend(context_tables)
                    logger.info(f"[Stage 2] Found {len(context_tables)} tables via context search")

            # Deduplicate
            candidate_tables = list(set(candidate_tables))

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

        This method systematically searches for metrics and reference SQL,
        then updates the workflow context with discovered knowledge.
        """
        found_tables = []
        try:
            if not self.agent_config:
                return []

            context_search = ContextSearchTools(self.agent_config)

            # 1. Search Metrics - Systematic search (no conditional check)
            try:
                metric_result = context_search.search_metrics(query_text=query, top_n=5)
                if metric_result.success and metric_result.result:
                    # Convert results to Metric objects and update context
                    metrics = []
                    for item in metric_result.result:
                        if isinstance(item, dict):
                            metric = Metric(
                                name=item.get("name", ""), llm_text=item.get("llm_text", item.get("description", ""))
                            )
                            metrics.append(metric)

                    if metrics and self.workflow:

                        def update_metrics():
                            self.workflow.context.update_metrics(metrics)
                            return {"metrics_count": len(metrics)}

                        result = safe_context_update(
                            self.workflow.context,
                            update_metrics,
                            operation_name="update_metrics",
                        )

                        if result["success"]:
                            logger.info(f"Updated context with {len(metrics)} metrics")
                        else:
                            logger.warning(f"Metrics update failed: {result.get('error')}")
            except Exception as e:
                logger.warning(f"Metrics search failed: {e}")

            # 2. Search Reference SQL - Systematic search (always execute)
            try:
                ref_sql_result = context_search.search_reference_sql(query_text=query, top_n=5)
                if ref_sql_result.success and ref_sql_result.result:
                    # Convert results to ReferenceSql objects and update context
                    reference_sqls = []
                    for item in ref_sql_result.result:
                        if isinstance(item, dict):
                            ref_sql = ReferenceSql(
                                name=item.get("name", item.get("summary", ""))[:100],
                                sql=item.get("sql", ""),
                                comment=item.get("comment", ""),
                                summary=item.get("summary", ""),
                                tags=item.get("tags", ""),
                            )
                            reference_sqls.append(ref_sql)

                    if reference_sqls and self.workflow:

                        def update_reference_sqls():
                            self.workflow.context.update_reference_sqls(reference_sqls)
                            return {"ref_sql_count": len(reference_sqls)}

                        result = safe_context_update(
                            self.workflow.context,
                            update_reference_sqls,
                            operation_name="update_reference_sqls",
                        )

                        if result["success"]:
                            logger.info(f"Updated context with {len(reference_sqls)} reference SQLs")
                            # Extract table names from reference SQLs and populate found_tables
                            for ref_sql in reference_sqls:
                                if ref_sql.sql:
                                    try:
                                        tables = extract_table_names(ref_sql.sql)
                                        found_tables.extend(tables)
                                    except Exception as e:
                                        logger.debug(f"Failed to extract tables from SQL: {e}")
                        else:
                            logger.warning(f"Reference SQLs update failed: {result.get('error')}")
            except Exception as e:
                logger.warning(f"Reference SQL search failed: {e}")

        except Exception as e:
            logger.warning(f"Context-based discovery failed: {e}")

        return found_tables

    async def _update_context(self, schemas: List[TableSchema], task) -> None:
        """
        Update the workflow context with discovered schemas.
        """
        if not schemas:
            return

        # Add to schemas list
        context_schemas = task.schemas or []

        # Merge new schemas, avoiding duplicates
        existing_tables = {s.table_name for s in context_schemas}
        new_schemas = [s for s in schemas if s.table_name not in existing_tables]

        if new_schemas:
            context_schemas.extend(new_schemas)
            task.schemas = context_schemas

            # Also update tables list for simple access
            current_tables = task.tables or []
            new_table_names = [s.table_name for s in new_schemas]
            current_tables.extend(new_table_names)
            task.tables = list(set(current_tables))

            logger.info(f"Updated context with {len(new_schemas)} new schemas")

    async def _semantic_table_discovery(self, task_text: str) -> List[str]:
        """
        Discover tables using semantic vector search.

        ✅ Optimized: Added similarity threshold filtering to improve precision.

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
            # ✅ Optimized: Increase top_n to 20 and filter by similarity threshold
            schema_results, _ = rag.search_similar(query_text=task_text, top_n=20)

            if schema_results and len(schema_results) > 0:
                # ✅ Optimized: Filter by similarity threshold
                # LanceDB vector search returns _distance column (lower = more similar)
                # Convert distance to similarity: similarity = 1 / (1 + distance)
                similarity_threshold = 0.5  # Configurable threshold

                try:
                    # Check if _distance column exists
                    if "_distance" in schema_results.column_names:
                        distances = schema_results.column("_distance").to_pylist()
                        table_names = schema_results.column("table_name").to_pylist()

                        # Filter tables by similarity threshold
                        filtered_tables = []
                        for table_name, distance in zip(table_names, distances):
                            # Calculate similarity from distance (LanceDB uses Euclidean distance)
                            similarity = 1.0 / (1.0 + distance)
                            if similarity >= similarity_threshold:
                                filtered_tables.append(table_name)

                        if filtered_tables:
                            tables.extend(filtered_tables)
                            logger.info(
                                f"Semantic search found {len(filtered_tables)}/{len(table_names)} tables "
                                f"via metadata (threshold={similarity_threshold}): {filtered_tables}"
                            )
                        else:
                            logger.warning(
                                f"No tables passed similarity threshold ({similarity_threshold}). "
                                f"Using all {len(table_names)} results as fallback."
                            )
                            tables.extend(table_names)
                    else:
                        # No _distance column, use all results
                        found_tables = schema_results.column("table_name").to_pylist()
                        tables.extend(found_tables)
                        logger.info(f"Semantic search found tables via metadata (no distance filtering): {found_tables}")

                except Exception as filter_error:
                    logger.warning(f"Similarity filtering failed: {filter_error}. Using all results.")
                    found_tables = schema_results.column("table_name").to_pylist()
                    tables.extend(found_tables)

            # 2. Use ContextSearchTools for metrics/business logic search (complementary)
            context_search = ContextSearchTools(self.agent_config)
            if context_search.has_metrics:
                result = context_search.search_metrics(query_text=task_text, top_n=5)
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
        Discover tables using keyword matching and External Knowledge.

        Args:
            task_text: Lowercase user query text

        Returns:
            List of table names
        """
        candidate_tables = []

        # 1. Check hardcoded configuration
        from datus.configuration.business_term_config import TABLE_KEYWORD_PATTERNS

        for keyword, table_name in TABLE_KEYWORD_PATTERNS.items():
            if keyword in task_text:
                candidate_tables.append(table_name)
                if table_name.endswith("s"):
                    candidate_tables.append(table_name[:-1])
                else:
                    candidate_tables.append(table_name + "s")

                business_mappings = get_business_term_mapping(keyword)
                if business_mappings:
                    candidate_tables.extend(business_mappings)

        # 2. Check External Knowledge Store
        try:
            from datus.storage.cache import get_storage_cache_instance

            storage_cache = get_storage_cache_instance(self.agent_config)
            ext_knowledge = storage_cache.ext_knowledge_storage()

            # Search for relevant terms in the knowledge base
            results = ext_knowledge.search_knowledge(query_text=task_text, top_n=5)

            if results:
                for item in results:
                    explanation = item.get("explanation", "")
                    potential_tables = self._extract_potential_tables_from_text(explanation)
                    candidate_tables.extend(potential_tables)

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"External knowledge search failed (network): {e}")
            # Continue with hardcoded mappings as fallback
        except Exception as e:
            logger.warning(f"External knowledge search failed: {e}")
            # Continue with hardcoded mappings as fallback

        # Remove duplicates while preserving order
        seen = set()
        unique_tables = []
        for table in candidate_tables:
            if table not in seen:
                seen.add(table)
                unique_tables.append(table)

        return unique_tables

    def _extract_potential_tables_from_text(self, text: str) -> List[str]:
        """Helper to extract potential table names (snake_case words) from text."""
        import re

        # Simple regex for potential table names (snake_case, at least one underscore)
        return re.findall(r"\b[a-z]+_[a-z0-9_]+\b", text)

    async def _llm_based_table_discovery(self, query: str) -> List[str]:
        """
        Stage 1.5: Use LLM to infer potential English table names from Chinese query.

        ✅ Fixed: Uses LLMMixin for automatic retry with exponential backoff
        and configured prompt template from business_term_config.py
        """
        if not query:
            return []

        # ✅ Use configured prompt template
        prompt_template = LLM_TABLE_DISCOVERY_CONFIG.get("prompt_template", "")
        prompt = prompt_template.format(query=query)

        try:
            # ✅ Use LLMMixin with retry and caching
            cache_key = f"llm_table_discovery:{hash(query)}"
            max_retries = LLM_TABLE_DISCOVERY_CONFIG.get("max_retries", 3)

            response = await self.llm_call_with_retry(
                prompt=prompt,
                operation_name="table_discovery",
                cache_key=cache_key,
                max_retries=max_retries,
            )

            tables = response.get("tables", [])
            cleaned_tables = list(set([str(t) for t in tables if isinstance(t, (str, int, float))]))
            logger.info(f"LLM inferred potential tables: {cleaned_tables}")
            return cleaned_tables
        except NodeExecutionResult as e:
            logger.warning(f"LLM table discovery failed after retries: {e.error_message}")
            return []
        except Exception as e:
            logger.warning(f"LLM table discovery failed with unexpected error: {e}")
            return []

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
        Includes automatic metadata repair if definitions are missing.

        ✅ Fixed: Uses thread-safe context update to prevent race conditions.

        Args:
            task: The SQL task
            candidate_tables: List of table names to load schemas for
        """
        if not candidate_tables or not self.workflow:
            return

        try:
            # --- Repair Logic Start ---
            if self.agent_config:
                try:
                    from datus.storage.cache import get_storage_cache_instance

                    storage_cache = get_storage_cache_instance(self.agent_config)
                    schema_storage = storage_cache.schema_storage()

                    # Check for missing definitions
                    current_schemas = schema_storage.get_table_schemas(candidate_tables)
                    missing_tables = []
                    for i, schema in enumerate(current_schemas):
                        if not schema or not schema.definition or not schema.definition.strip():
                            missing_tables.append(candidate_tables[i])

                    if missing_tables:
                        logger.info(f"Found {len(missing_tables)} tables with missing metadata. Attempting repair...")
                        await self._repair_metadata(missing_tables, schema_storage, task)

                except Exception as e:
                    logger.warning(f"Metadata repair pre-check failed: {e}")
            # --- Repair Logic End ---

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
                    # ✅ Use thread-safe context update
                    def update_schemas():
                        self.workflow.context.update_schema_and_values(schemas, values)
                        return {"loaded_count": len(schemas)}

                    result = safe_context_update(
                        self.workflow.context,
                        update_schemas,
                        operation_name="load_table_schemas",
                    )

                    if result["success"]:
                        logger.debug(f"Loaded schemas for {result['result']['loaded_count']} tables")
                    else:
                        logger.warning(f"Context update failed: {result.get('error')}")
                else:
                    logger.warning(f"No schemas found for candidate tables: {candidate_tables}")

                    # DDL Fallback: If storage is empty, retrieve DDL from database and populate storage
                    if candidate_tables:
                        logger.info("Schema storage empty, attempting DDL retrieval fallback...")
                        await self._ddl_fallback_and_retry(candidate_tables, task)

        except Exception as e:
            logger.warning(f"Failed to load table schemas: {e}")
            # Don't fail the entire node for schema loading issues

    async def _repair_metadata(self, table_names: List[str], schema_storage, task) -> int:
        """
        Attempt to repair missing metadata by fetching DDL directly from the database.
        """
        repaired_count = 0
        try:
            db_manager = get_db_manager()
            # Use the configured connection for the current task
            connector = db_manager.get_conn(self.agent_config.current_namespace, task.database_name)

            if not connector:
                return 0

            for table in table_names:
                try:
                    # Try to fetch DDL using connector
                    ddl = None
                    if hasattr(connector, "get_table_ddl"):
                        ddl = connector.get_table_ddl(table)
                    elif hasattr(connector, "get_ddl"):
                        ddl = connector.get_ddl(table)

                    if ddl:
                        schema_storage.update_table_schema(table, ddl)
                        repaired_count += 1
                        logger.info(f"Repaired metadata for table: {table}")
                except Exception as e:
                    logger.debug(f"Failed to repair metadata for table {table}: {e}")

        except Exception as e:
            logger.error(f"Metadata repair process failed: {e}")

        return repaired_count

    async def _ddl_fallback_and_retry(self, candidate_tables: List[str], task) -> None:
        """
        Fallback to retrieve DDL from database and populate storage when SchemaStorage is empty.

        This method:
        1. Connects to the database using the configured connector
        2. Retrieves DDL for all tables (or candidate tables)
        3. Populates SchemaStorage with the retrieved DDL
        4. Retries schema search with populated storage

        Args:
            candidate_tables: List of table names to search for
            task: The current task with database connection info
        """
        try:
            # SchemaWithValueRAG is already imported at module level
            # Using local import here for clarity in this async context
            db_manager = get_db_manager()

            # Get the database connector for the current task
            connector = db_manager.get_conn(self.agent_config.current_namespace, task.database_name)

            if not connector:
                logger.warning(f"Could not get database connector for {task.database_name}")
                return

            # Initialize RAG for storage operations
            rag = SchemaWithValueRAG(agent_config=self.agent_config)
            schemas_to_store = []
            retrieved_tables = set()

            # Try to get DDL for tables using available connector methods
            if hasattr(connector, "get_tables_with_ddl"):
                # Best case: connector can get all tables with DDL at once
                logger.info("Using get_tables_with_ddl to retrieve all table DDLs...")
                tables_with_ddl = connector.get_tables_with_ddl()

                for tbl_info in tables_with_ddl:
                    table_name = tbl_info.get("name", "")
                    if table_name:
                        schemas_to_store.append({
                            "identifier": f"{task.catalog_name or ''}.{task.database_name}..{table_name}.table",
                            "catalog_name": task.catalog_name or "",
                            "database_name": task.database_name or "",
                            "schema_name": task.schema_name or "",
                            "table_name": table_name,
                            "table_type": "table",
                            "definition": tbl_info.get("ddl", ""),
                        })
                        retrieved_tables.add(table_name)

                logger.info(f"Retrieved DDL for {len(schemas_to_store)} tables from database")

            elif hasattr(connector, "get_table_ddl") or hasattr(connector, "get_ddl"):
                # Fallback: get DDL for each candidate table individually
                logger.info(f"Retrieving DDL for {len(candidate_tables)} candidate tables individually...")

                for table_name in candidate_tables:
                    try:
                        ddl = None
                        if hasattr(connector, "get_table_ddl"):
                            ddl = connector.get_table_ddl(table_name)
                        elif hasattr(connector, "get_ddl"):
                            ddl = connector.get_ddl(table_name)

                        if ddl:
                            schemas_to_store.append({
                                "identifier": f"{task.catalog_name or ''}.{task.database_name}..{table_name}.table",
                                "catalog_name": task.catalog_name or "",
                                "database_name": task.database_name or "",
                                "schema_name": task.schema_name or "",
                                "table_name": table_name,
                                "table_type": "table",
                                "definition": ddl,
                            })
                            retrieved_tables.add(table_name)
                    except Exception as e:
                        logger.debug(f"Failed to get DDL for table {table_name}: {e}")

                logger.info(f"Retrieved DDL for {len(schemas_to_store)} candidate tables")

            else:
                logger.warning("Connector does not support DDL retrieval methods")
                return

            # Store retrieved schemas in storage
            if schemas_to_store:
                logger.info(f"Populating SchemaStorage with {len(schemas_to_store)} schemas...")
                rag.store_batch(schemas_to_store, [])

                # Retry schema search with populated storage
                schemas, values = rag.search_tables(
                    tables=candidate_tables,
                    catalog_name=task.catalog_name or "",
                    database_name=task.database_name or "",
                    schema_name=task.schema_name or "",
                    dialect=task.database_type if hasattr(task, "database_type") else None,
                )

                if schemas:
                    # Update workflow context with retrieved schemas
                    def update_schemas():
                        self.workflow.context.update_schema_and_values(schemas, values)
                        return {"loaded_count": len(schemas)}

                    result = safe_context_update(
                        self.workflow.context,
                        update_schemas,
                        operation_name="ddl_fallback_load_schemas",
                    )

                    if result["success"]:
                        logger.info(f"✅ Successfully loaded {len(schemas)} schemas via DDL fallback")
                    else:
                        logger.warning(f"Context update failed after DDL fallback: {result.get('error')}")
                else:
                    logger.warning(f"DDL fallback retrieved schemas but still no match for candidates: {candidate_tables}")
            else:
                logger.warning("No schemas retrieved from database for fallback")

        except Exception as e:
            logger.warning(f"DDL fallback failed: {e}")

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

        ✅ Fixed: Uses thread-safe context update to prevent race conditions.

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

                # ✅ Thread-safe metadata update
                if "candidate_tables" in output:

                    def update_metadata():
                        # Store discovered tables in workflow metadata for downstream nodes
                        if not hasattr(workflow, "metadata"):
                            workflow.metadata = {}
                        workflow.metadata["discovered_tables"] = output["candidate_tables"]
                        return {"table_count": len(output["candidate_tables"])}

                    result = safe_context_update(
                        workflow,
                        update_metadata,
                        operation_name="update_discovered_tables",
                    )

                    if not result["success"]:
                        logger.warning(f"Metadata update failed: {result.get('error')}")

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
