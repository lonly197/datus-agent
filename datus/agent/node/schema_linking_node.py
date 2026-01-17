# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import AsyncGenerator, Dict, List, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput
from datus.schemas.node_models import TableSchema, TableValue
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.lineage_graph_tools.schema_lineage import SchemaLineageTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SchemaLinkingNode(Node):
    """
    @deprecated This node is deprecated. Use SchemaDiscoveryNode instead.

    SchemaLinkingNode functionality has been fully integrated into SchemaDiscoveryNode,
    which provides enhanced features including:
    - Progressive matching (semantic + keyword + context + fallback)
    - External knowledge enhancement
    - LLM-based schema matching
    - Multi-stage discovery pipeline

    This node is retained for backward compatibility only.
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[BaseInput] = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
        )
        self._table_schemas: List[TableSchema] = []
        self._table_values: List[TableValue] = []

    def execute(self):
        self.result = self._execute_schema_linking()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute schema linking with streaming support."""
        # Emit deprecation warning
        logger.warning(
            "SchemaLinkingNode is deprecated. Use SchemaDiscoveryNode instead. "
            "SchemaDiscoveryNode provides enhanced features including progressive matching, "
            "external knowledge enhancement, and LLM-based schema matching."
        )
        async for action in self._schema_linking_stream(action_history_manager):
            yield action

    def update_context(self, workflow: Workflow) -> Dict:
        """Update schema linking results to workflow context.

        When schema_linking is triggered by reflection, it may find additional
        tables that weren't discovered by the initial schema_discovery. This
        method merges the new schemas with existing ones.
        """
        result = self.result
        try:
            if len(workflow.context.table_schemas) == 0:
                # First schema linking - set directly
                workflow.context.table_schemas = result.table_schemas
                workflow.context.table_values = result.table_values
                logger.info(f"Schema linking set initial context with {len(result.table_schemas)} schemas")
            else:
                # Merge schemas: add only tables that don't already exist
                existing_table_names = {s.table_name for s in workflow.context.table_schemas}
                new_schemas = [s for s in result.table_schemas if s.table_name not in existing_table_names]
                new_values = [v for v in result.table_values if v.table_name not in existing_table_names]

                if new_schemas:
                    workflow.context.table_schemas.extend(new_schemas)
                    workflow.context.table_values.extend(new_values)
                    logger.info(f"Schema linking added {len(new_schemas)} new tables to existing context")
                else:
                    logger.debug("Schema linking found no new tables to add to context")

            return {"success": True, "message": "Updated schema linking context"}
        except Exception as e:
            logger.error(f"Failed to update schema linking context: {str(e)}")
            return {"success": False, "message": f"Schema linking context update failed: {str(e)}"}

    def setup_input(self, workflow: Workflow) -> Dict:
        logger.info("Setup schema linking input")

        # Search and enhance external knowledge before schema linking
        enhanced_external_knowledge = self._search_external_knowledge(
            workflow.task.task,  # User query
            workflow.task.subject_path,  # Subject hierarchy path
        )

        # Combine original and searched knowledge
        if enhanced_external_knowledge:
            original_knowledge = workflow.task.external_knowledge
            combined_knowledge = self._combine_knowledge(original_knowledge, enhanced_external_knowledge)
            workflow.task.external_knowledge = combined_knowledge
        if workflow.context and workflow.context.table_schemas:
            self._table_schemas = workflow.context.table_schemas
            self._table_values = workflow.context.table_values
            self.input = SchemaLinkingInput(
                input_text=workflow.task.task,
                matching_rate=self.agent_config.schema_linking_rate,
                database_type=workflow.task.database_type,
                database_name=workflow.task.database_name,
                sql_context=None,
                table_type=workflow.task.schema_linking_type,
            )
        else:
            # Setup schema linking input
            matching_rate = self.agent_config.schema_linking_rate
            matching_rates = ["fast", "medium", "slow", "from_llm"]
            start = matching_rates.index(matching_rate)
            final_matching_rate = matching_rates[min(start + workflow.reflection_round, len(matching_rates) - 1)]
            logger.debug(f"Final matching rate: {final_matching_rate}")
            next_input = SchemaLinkingInput(
                input_text=workflow.task.task,
                matching_rate=final_matching_rate,
                database_type=workflow.task.database_type,
                database_name=workflow.task.database_name,
                sql_context=None,
                table_type=workflow.task.schema_linking_type,
            )
            self.input = next_input
        return {"success": True, "message": "Schema and external knowledge prepared"}

    def _execute_schema_linking(self) -> SchemaLinkingResult:
        """Execute schema linking action to analyze database schema.
        Input:
            query - The input query to analyze.
        Returns:
            A validated SchemaLinkingResult containing table schemas and values.
        """
        if self._table_schemas:
            return SchemaLinkingResult(
                success=True,
                table_schemas=self._table_schemas,
                table_values=self._table_values,
                schema_count=0 if not self._table_schemas else len(self._table_schemas),
                value_count=0 if not self._table_values else len(self._table_values),
            )

        tool = None
        try:
            tool = SchemaLineageTool(agent_config=self.agent_config)

            # Execute with RAG tool
            result = tool.execute(self.input, self.model)
            if not result.success:
                logger.warning(f"Schema linking failed: {result.error}")
                return self._execute_schema_linking_fallback(tool)

            logger.info(f"Schema linking result: found {len(result.table_schemas)} tables")
            if len(result.table_schemas) > 0:
                return result

            logger.info("No tables found, using fallback method")
            return self._execute_schema_linking_fallback(tool)

        except Exception as e:
            logger.warning(f"Schema linking tool initialization/execution failed: {e}")
            if tool:
                return self._execute_schema_linking_fallback(tool)

            # If tool failed to initialize, try to initialize it with minimal dependencies for fallback
            # Note: This assumes SchemaLineageTool might fail due to storage issues but we still want fallback
            try:
                # Attempt to create tool bypassing storage init if possible, or just fail
                # Since we can't easily bypass __init__, we return failure if we can't create tool
                # But we can try to use the fallback logic directly if we could refactor.
                # For now, return error.
                return SchemaLinkingResult(
                    success=False,
                    error=f"Schema linking failed: {e}",
                    schema_count=0,
                    value_count=0,
                    table_schemas=[],
                    table_values=[],
                )
            except Exception:
                return SchemaLinkingResult(
                    success=False,
                    error=f"Schema linking failed: {e}",
                    schema_count=0,
                    value_count=0,
                    table_schemas=[],
                    table_values=[],
                )

    def _execute_schema_linking_fallback(self, tool: SchemaLineageTool) -> SchemaLinkingResult:
        # Fallback: directly get tables from current database
        logger.info("Get tables directly from database")
        try:
            # Get database connector through db_manager
            from datus.tools.db_tools.db_manager import get_db_manager

            db_manager = get_db_manager(self.agent_config.namespaces)

            # Get current namespace and database connection
            current_namespace = self.agent_config.current_namespace
            database_name = self.input.database_name if hasattr(self.input, "database_name") else ""

            # Get database connector
            connector = db_manager.get_conn(current_namespace, database_name)

            return tool.get_schems_by_db(connector=connector, input_param=self.input)

        except Exception as e:
            logger.warning(f"Schema linking failed: {e}")
            return SchemaLinkingResult(
                success=False,
                error=f"Schema linking failed: {e}",
                schema_count=0,
                value_count=0,
                table_schemas=[],
                table_values=[],
            )

    def _search_external_knowledge(self, user_query: str, subject_path: Optional[List[str]] = None) -> str:
        """Search for relevant external knowledge based on user query and subject path.

        Args:
            user_query: The user's natural language query
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])

        Returns:
            Formatted string of relevant knowledge entries, empty string if no results or error
        """
        try:
            # Use ContextSearchTools for external knowledge search (preferred over direct store access)
            context_search_tools = ContextSearchTools(self.agent_config)

            # Check if external knowledge is available
            if not context_search_tools.has_ext_knowledge:
                logger.debug("External knowledge store is empty or not available, skipping search")
                return ""

            # Execute semantic search
            # Convert subject_path to domain/layer1/layer2 format
            if subject_path and len(subject_path) > 0:
                domain = subject_path[0] if len(subject_path) > 0 else ""
                layer1 = subject_path[1] if len(subject_path) > 1 else ""
                layer2 = subject_path[2] if len(subject_path) > 2 else ""
            else:
                domain = ""
                layer1 = ""
                layer2 = ""

            search_result = context_search_tools.search_external_knowledge(
                query_text=user_query, domain=domain, layer1=layer1, layer2=layer2, top_n=5
            )

            # Format search results
            if search_result.success and search_result.result:
                knowledge_items = []
                for result in search_result.result:
                    terminology = result.get("terminology", "")
                    explanation = result.get("explanation", "")
                    if terminology and explanation:
                        knowledge_items.append(f"- {terminology}: {explanation}")

                if knowledge_items:
                    formatted_knowledge = "\n".join(knowledge_items)
                    logger.info(f"Found {len(knowledge_items)} relevant knowledge entries")
                    return formatted_knowledge

            logger.debug("No relevant external knowledge found")
            return ""

        except Exception as e:
            logger.warning(f"Failed to search external knowledge: {str(e)}")
            return ""

    def _combine_knowledge(self, original: str, enhanced: str) -> str:
        """Combine original knowledge and searched knowledge.

        Args:
            original: Original external knowledge from SqlTask
            enhanced: Knowledge retrieved from search

        Returns:
            Combined knowledge string
        """
        parts = []
        if original:
            parts.append(original)
        if enhanced:
            parts.append(f"Relevant Business Knowledge:\n{enhanced}")

        return "\n\n".join(parts)

    async def _schema_linking_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute schema linking with streaming support and action history tracking."""
        try:
            # External knowledge search action
            knowledge_action = ActionHistory(
                action_id="external_knowledge_search",
                role=ActionRole.WORKFLOW,
                messages="Searching external knowledge base for relevant business context",
                action_type="knowledge_search",
                input={
                    "query": self.input.input_text if hasattr(self.input, "input_text") else "",
                    "database_name": self.input.database_name if hasattr(self.input, "database_name") else "",
                },
                status=ActionStatus.PROCESSING,
            )
            yield knowledge_action

            # Execute external knowledge search
            try:
                enhanced_knowledge = self._search_external_knowledge(
                    self.input.input_text if hasattr(self.input, "input_text") else "",
                    None,  # subject_path - would need to get from workflow if available
                )
                knowledge_action.status = ActionStatus.SUCCESS
                knowledge_action.output = {
                    "knowledge_found": bool(enhanced_knowledge),
                    "knowledge_length": len(enhanced_knowledge) if enhanced_knowledge else 0,
                }
            except Exception as e:
                knowledge_action.status = ActionStatus.FAILED
                knowledge_action.output = {"error": str(e)}
                logger.warning(f"External knowledge search failed: {e}")

            # Schema linking action
            schema_action = ActionHistory(
                action_id="schema_linking",
                role=ActionRole.WORKFLOW,
                messages="Analyzing database schema and linking relevant tables",
                action_type="schema_linking",
                input={
                    "input_text": self.input.input_text if hasattr(self.input, "input_text") else "",
                    "database_name": self.input.database_name if hasattr(self.input, "database_name") else "",
                    "matching_rate": self.input.matching_rate if hasattr(self.input, "matching_rate") else "medium",
                },
                status=ActionStatus.PROCESSING,
            )
            yield schema_action

            # Execute schema linking - reuse existing logic
            try:
                result = self._execute_schema_linking()

                schema_action.status = ActionStatus.SUCCESS
                schema_action.output = {
                    "success": result.success,
                    "tables_found": len(result.table_schemas) if result.table_schemas else 0,
                    "values_found": len(result.table_values) if result.table_values else 0,
                }

                # Store result for later use
                self.result = result

            except Exception as e:
                schema_action.status = ActionStatus.FAILED
                schema_action.output = {"error": str(e)}
                logger.error(f"Schema linking error: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Schema linking streaming error: {str(e)}")
            raise
