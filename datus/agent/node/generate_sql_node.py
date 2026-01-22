# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, cast

from agents import Tool

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.prompts.gen_sql import get_sql_prompt
from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.schemas.node_models import (GenerateSQLInput, GenerateSQLResult,
                                       SQLContext, SqlTask, TableSchema,
                                       TableValue)
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.utils.constants import DBType
from datus.utils.exceptions import ErrorCode
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger
from datus.utils.time_utils import get_default_current_date
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)

# 配置选项：控制智能DDL包含策略
ENABLE_SMART_DDL_SELECTION = os.getenv("ENABLE_SMART_DDL_SELECTION", "true").lower() == "true"
DEFAULT_INCLUDE_SCHEMA_DDL = os.getenv("DEFAULT_INCLUDE_SCHEMA_DDL", "false").lower() == "true"


class GenerateSQLNode(Node):
    """
    Node responsible for generating SQL queries based on user intent and schema.
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[GenerateSQLInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[List[Tool]] = None,
    ):
        super().__init__(node_id, description, node_type, input_data, agent_config, tools)
        self._metadata_rag: Optional[SchemaWithValueRAG] = None

    @property
    def metadata_rag(self) -> SchemaWithValueRAG:
        """Lazy load the metadata RAG service."""
        if not self._metadata_rag:
            if not self.agent_config:
                raise ValueError("Agent config is required for SchemaWithValueRAG")
            self._metadata_rag = SchemaWithValueRAG(self.agent_config)
        return self._metadata_rag

    def _should_include_ddl(self, query: str, table_count: int) -> bool:
        """
        决定是否需要包含DDL信息的智能策略。

        Args:
            query: 用户查询文本
            table_count: 涉及的表数量

        Returns:
            True表示需要包含DDL，False表示只需要表名和注释
        """
        # 如果禁用了智能选择，使用默认设置
        if not ENABLE_SMART_DDL_SELECTION:
            return DEFAULT_INCLUDE_SCHEMA_DDL

        if not query:
            return DEFAULT_INCLUDE_SCHEMA_DDL

        query_lower = query.lower().strip()

        # 简单查询：单表，简单条件 - 不需要DDL
        simple_indicators = [
            # 显示/查询类
            query_lower.startswith(("显示", "show", "select", "查询", "find", "search")),
            # 简单条件
            " where " in query_lower and table_count <= 1,
            # 前N条记录
            any(phrase in query_lower for phrase in ["前", "top", "limit", "前10", "前5"]),
        ]

        if any(simple_indicators) and table_count <= 1:
            return False

        # 复杂查询：多表，复杂条件，聚合等 - 需要DDL
        complex_keywords = [
            # 连接操作
            " join ",
            " inner join ",
            " left join ",
            " right join ",
            " full join ",
            # 聚合操作
            " group by ",
            " having ",
            " count(",
            " sum(",
            " avg(",
            " max(",
            " min(",
            # 子查询
            " subquery",
            " exists ",
            " in (select",
            " (select",
            " cte ",
            " with ",
            # 排序和窗口函数
            " order by ",
            " over ",
            " partition by ",
            # 分析函数
            " rank()",
            " dense_rank()",
            " row_number()",
            " lag(",
            " lead(",
        ]

        if any(keyword in query_lower for keyword in complex_keywords):
            return True

        # 多表查询通常需要DDL
        if table_count > 1:
            return True

        # 默认不包含DDL，节省token
        return DEFAULT_INCLUDE_SCHEMA_DDL

    def execute(self) -> None:
        """Execute the SQL generation logic."""
        self.result = self._execute_generate_sql()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute SQL generation with streaming support.

        Args:
            action_history_manager: Manager for tracking action history.

        Yields:
            ActionHistory: Updates on execution progress.
        """
        async for action in self._generate_sql_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Prepare input data for SQL generation.

        Args:
            workflow: The workflow instance.

        Returns:
            Dict containing success status and suggestions.
        """
        if workflow.context.document_result:
            database_docs = "\n Reference documents:\n"
            for _, docs in workflow.context.document_result.docs.items():
                database_docs += "\n".join(docs) + "\n"
        else:
            database_docs = ""

        task = workflow.task
        if not task:
            return {"success": False, "message": "No task available in workflow"}

        # Use smart strategy to decide whether to include DDL
        query = task.task if hasattr(task, "task") else ""
        table_count = len(workflow.context.table_schemas) if workflow.context.table_schemas else 0
        include_ddl = self._should_include_ddl(query, table_count)

        logger.debug(f"Query complexity analysis: table_count={table_count}, include_ddl={include_ddl}")

        # Create input for the next step
        next_input = GenerateSQLInput(
            database_type=task.database_type,
            sql_task=task,
            table_schemas=workflow.context.table_schemas,
            data_details=workflow.context.table_values,
            metrics=workflow.context.metrics,
            contexts=workflow.context.sql_contexts,
            external_knowledge=task.external_knowledge,
            database_docs=database_docs,
            include_schema_ddl=include_ddl,
        )
        self.input = next_input
        return {"success": True, "message": "Schema appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Update SQL generation results to workflow context.

        Note: Schemas should already be loaded by schema_discovery and validated
        by schema_validation. This node only adds the generated SQL to the context.

        Args:
            workflow: The workflow instance.

        Returns:
            Dict containing success status.
        """
        result = cast(Optional[GenerateSQLResult], self.result)
        if not result:
            return {"success": False, "message": "No generation result available"}

        try:
            # Create new SQL context record and add to context
            new_record = SQLContext(
                sql_query=result.sql_query,
                explanation=result.explanation or "",
                reflection_strategy="",
                reflection_explanation="",
                sql_error="",
            )
            workflow.context.sql_contexts.append(new_record)

            # Log table usage for debugging
            if result.tables:
                logger.debug(f"SQL generation used tables: {result.tables}")

            return {"success": True, "message": "Updated SQL generation context"}
        except Exception as e:
            logger.error(f"Failed to update SQL generation context: {str(e)}")
            return {"success": False, "message": f"SQL generation context update failed: {str(e)}"}

    def _execute_generate_sql(self) -> GenerateSQLResult:
        """
        Execute SQL generation action to create SQL query.

        Combines input data from previous nodes into a structured format for SQL generation.
        The input data includes:
        - table_schemas: Database schema information from schema linking
        - data_details: Additional data context
        - metrics: Relevant metrics information
        - database: Database type information

        Returns:
            GenerateSQLResult containing the generated SQL query
        """
        if not self.model:
            return GenerateSQLResult(
                success=False,
                error="SQL generation model not provided",
                sql_query="",
                tables=[],
                explanation=None,
            )

        input_data = cast(GenerateSQLInput, self.input)
        if not input_data:
            return GenerateSQLResult(
                success=False,
                error="Input data is missing",
                sql_query="",
                tables=[],
                explanation=None,
            )

        try:
            logger.debug(f"Generate SQL input: {type(input_data)} {input_data}")
            return generate_sql(self.model, input_data)
        except Exception as e:
            logger.error(f"SQL generation execution error: {str(e)}")
            return GenerateSQLResult(success=False, error=str(e), sql_query="", tables=[], explanation=None)

    def _get_schema_and_values(
        self, sql_task: SqlTask, table_names: List[str]
    ) -> Tuple[List[TableSchema], List[TableValue]]:
        """Get table schemas and values using the schema lineage tool."""
        try:
            # Get the schema lineage tool instance
            input_data = cast(GenerateSQLInput, self.input)
            if not input_data:
                return [], []

            sql_connector = self._sql_connector(input_data.sql_task.database_name)
            if not sql_connector:
                return [], []

            catalog_name = sql_task.catalog_name or sql_connector.catalog_name
            database_name = sql_task.database_name or sql_connector.database_name
            schema_name = sql_task.schema_name or sql_connector.schema_name

            # Use the tool to get schemas and values
            logger.debug(f"Getting schemas and values for tables {table_names} from {database_name}")
            return self.metadata_rag.search_tables(
                tables=table_names,
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                dialect=sql_task.database_type,
            )
        except Exception as e:
            logger.warning(f"Failed to get schemas and values for tables {table_names}: {e}")
            return [], []  # Return empty lists if lookup fails

    async def _generate_sql_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate SQL with streaming support and action history tracking."""
        if not self.model:
            logger.error("Model not available for SQL generation")
            return

        input_data = cast(GenerateSQLInput, self.input)
        if not input_data:
            return

        try:
            # SQL generation preparation action
            prep_action = ActionHistory(
                action_id="sql_generation_prep",
                role=ActionRole.WORKFLOW,
                messages="Preparing SQL generation with schema and context information",
                action_type="sql_preparation",
                input={
                    "database_type": input_data.database_type if hasattr(input_data, "database_type") else "",
                    "table_count": (
                        len(input_data.table_schemas)
                        if hasattr(input_data, "table_schemas") and input_data.table_schemas
                        else 0
                    ),
                    "has_metrics": bool(hasattr(input_data, "metrics") and input_data.metrics),
                    "has_external_knowledge": bool(
                        hasattr(input_data, "external_knowledge") and input_data.external_knowledge
                    ),
                },
                status=ActionStatus.PROCESSING,
            )
            yield prep_action

            # Update preparation status
            try:
                prep_action.status = ActionStatus.SUCCESS
                prep_action.output = {
                    "preparation_complete": True,
                    "input_validated": True,
                }
            except Exception as e:
                prep_action.status = ActionStatus.FAILED
                prep_action.output = {"error": str(e)}
                logger.warning(f"SQL preparation failed: {e}")

            # SQL generation action
            generation_action = ActionHistory(
                action_id="sql_generation",
                role=ActionRole.WORKFLOW,
                messages="Generating SQL query based on schema and requirements",
                action_type="sql_generation",
                input={
                    "task_description": (
                        getattr(input_data.sql_task, "task", "") if hasattr(input_data, "sql_task") else ""
                    ),
                    "database_type": input_data.database_type if hasattr(input_data, "database_type") else "",
                },
                status=ActionStatus.PROCESSING,
            )
            yield generation_action

            # Execute SQL generation - reuse existing logic
            try:
                result = self._execute_generate_sql()

                generation_action.status = ActionStatus.SUCCESS
                generation_action.output = {
                    "success": result.success,
                    "sql_query": result.sql_query,
                    "tables_involved": result.tables if result.tables else [],
                    "has_explanation": bool(result.explanation),
                }

                # Store result for later use
                self.result = result

            except Exception as e:
                generation_action.status = ActionStatus.FAILED
                generation_action.output = {"error": str(e)}
                logger.error(f"SQL generation error: {str(e)}")
                raise

            # Yield the updated generation action with final status
            yield generation_action

        except Exception as e:
            logger.error(f"SQL generation streaming error: {str(e)}")
            raise


@optional_traceable()
def generate_sql(model: LLMBaseModel, input_data: GenerateSQLInput) -> GenerateSQLResult:
    """Generate SQL query using the provided model."""
    if not isinstance(input_data, GenerateSQLInput):
        raise TypeError("Input data must be a GenerateSQLInput instance")

    sql_query = ""
    try:
        # Format the prompt with schema list
        prompt = get_sql_prompt(
            database_type=input_data.database_type or DBType.SQLITE.value,
            table_schemas=input_data.table_schemas,
            data_details=input_data.data_details,
            metrics=input_data.metrics,
            question=input_data.sql_task.task,
            external_knowledge=input_data.external_knowledge,
            prompt_version=input_data.prompt_version,
            context=[sql_context.to_str() for sql_context in input_data.contexts],
            max_table_schemas_length=input_data.max_table_schemas_length,
            max_data_details_length=input_data.max_data_details_length,
            max_context_length=input_data.max_context_length,
            max_value_length=input_data.max_value_length,
            max_text_mark_length=input_data.max_text_mark_length,
            database_docs=input_data.database_docs,
            current_date=get_default_current_date(input_data.sql_task.current_date),
            date_ranges=getattr(input_data.sql_task, "date_ranges", ""),
            include_schema_ddl=input_data.include_schema_ddl,
        )

        logger.debug(f"Generated SQL prompt:  {type(model)}, {prompt}")
        # Generate SQL using the provided model
        sql_query = model.generate_with_json_output(prompt)
        logger.debug(f"Generated SQL: {sql_query}")

        # Parse the response using robust JSON utility
        if isinstance(sql_query, str):
            # Use llm_result2json to handle markdown, truncated JSON, and common format errors
            sql_query_dict = llm_result2json(sql_query, expected_type=dict)
            if sql_query_dict is None:
                logger.error(f"Failed to parse SQL query JSON: {sql_query[:200]}")
                return GenerateSQLResult(success=False, error="Invalid JSON format", sql_query=sql_query)
        else:
            sql_query_dict = sql_query

        # Return result as GenerateSQLResult
        if sql_query_dict and isinstance(sql_query_dict, dict):
            return GenerateSQLResult(
                success=True,
                error=None,
                sql_query=sql_query_dict.get("sql", ""),
                tables=sql_query_dict.get("tables", []),
                explanation=sql_query_dict.get("explanation"),
            )
        else:
            return GenerateSQLResult(success=False, error="sql generation failed, no result", sql_query=sql_query)
    except json.JSONDecodeError as e:
        logger.error(f"SQL json decode failed: {e}")
        return GenerateSQLResult(success=False, error=str(e), sql_query=str(sql_query))
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return GenerateSQLResult(success=False, error=str(e), sql_query="")
