# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unified Execution Event Manager for all execution modes.
统一执行事件管理器，处理所有执行模式的事件流转。
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict

try:
    from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
    from datus.utils.loggings import get_logger

    logger = get_logger(__name__)
except ImportError:
    # For standalone testing
    logger = None


class ExecutionContext:
    """Execution context for different scenarios."""

    def __init__(
        self,
        scenario: str,
        task_data: Dict[str, Any],
        agent_config: Any = None,
        model: Any = None,
        workflow_metadata: Dict[str, Any] = None,
    ):
        self.scenario = scenario  # "text2sql", "data_analysis", "sql_review", "smart_query", "deep_analysis"
        self.task_data = task_data
        self.agent_config = agent_config
        self.model = model
        self.workflow_metadata = workflow_metadata or {}
        self.execution_status = ExecutionStatus()
        self.start_time = time.time()


class ExecutionStatus:
    """Unified execution status tracker."""

    def __init__(self):
        self.phase = "idle"  # idle, planning, executing, completed, failed
        self.current_step = ""
        self.progress = 0.0
        self.errors = []
        self.warnings = []
        self.execution_stats = {}

    def update_phase(self, phase: str, current_step: str = ""):
        self.phase = phase
        self.current_step = current_step
        logger.info(f"Execution phase updated: {phase} - {current_step}")

    def add_error(self, error_type: str, message: str):
        self.errors.append({"type": error_type, "message": message, "timestamp": time.time()})

    def add_warning(self, warning_type: str, message: str):
        self.warnings.append({"type": warning_type, "message": message, "timestamp": time.time()})


class ExecutionEventManager:
    """
    Unified execution event manager for all execution modes.
    统一的执行事件管理器，处理所有执行模式的事件流转。
    """

    def __init__(self, action_history_manager: ActionHistoryManager):
        self.action_history_manager = action_history_manager
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._event_queue: asyncio.Queue[ActionHistory] = asyncio.Queue()

    async def start_execution(
        self,
        execution_id: str,
        scenario: str,
        task_data: Dict[str, Any],
        agent_config: Any = None,
        model: Any = None,
        workflow_metadata: Dict[str, Any] = None,
    ) -> ExecutionContext:
        """Start a new execution."""
        context = ExecutionContext(scenario, task_data, agent_config, model, workflow_metadata)
        self._active_executions[execution_id] = context

        # Create initial execution action
        action = ActionHistory.create_action(
            role=ActionRole.SYSTEM,
            action_type=f"execution_start_{scenario}",
            messages=f"Starting {scenario} execution",
            input_data={"execution_id": execution_id, "scenario": scenario, "task_data": task_data},
            status=ActionStatus.PROCESSING,
        )
        self.action_history_manager.add_action(action)
        await self._event_queue.put(action)

        return context

    async def update_execution_status(
        self, execution_id: str, phase: str, current_step: str = "", progress: float = None
    ):
        """Update execution status."""
        if execution_id not in self._active_executions:
            logger.warning(f"Execution {execution_id} not found")
            return

        context = self._active_executions[execution_id]
        context.execution_status.update_phase(phase, current_step)
        if progress is not None:
            context.execution_status.progress = progress

        # Create status update action
        action = ActionHistory.create_action(
            role=ActionRole.SYSTEM,
            action_type=f"execution_status_{phase}",
            messages=f"Execution {phase}: {current_step}",
            input_data={
                "execution_id": execution_id,
                "phase": phase,
                "current_step": current_step,
                "progress": progress,
            },
            status=ActionStatus.PROCESSING,
        )
        self.action_history_manager.add_action(action)
        await self._event_queue.put(action)

    async def record_tool_execution(
        self,
        execution_id: str,
        tool_name: str,
        tool_call_id: str,
        input_data: Dict[str, Any],
        result: Dict[str, Any] = None,
        error: str = None,
        execution_time: float = None,
    ):
        """Record tool execution."""
        success = result and result.get("success", False) if result else False

        action = ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type=f"tool_{tool_name}",
            messages=f"Tool {tool_name}: {'SUCCESS' if success else 'FAILED'}",
            input_data={
                "execution_id": execution_id,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "input": input_data,
                "execution_time": execution_time,
            },
            output_data={"result": result, "error": error} if result or error else None,
            status=ActionStatus.SUCCESS if success else ActionStatus.FAILED,
        )
        self.action_history_manager.add_action(action)
        await self._event_queue.put(action)

        # Update execution context
        if execution_id in self._active_executions:
            context = self._active_executions[execution_id]
            if error:
                context.execution_status.add_error("tool_error", f"{tool_name}: {error}")
            elif not success:
                context.execution_status.add_warning("tool_warning", f"{tool_name} execution incomplete")

    async def record_llm_interaction(
        self,
        execution_id: str,
        interaction_type: str,  # "planning", "reasoning", "generation", etc.
        prompt: str,
        response: str = None,
        error: str = None,
        execution_time: float = None,
        additional_data: Dict[str, Any] = None,
    ):
        """Record LLM interaction."""
        success = response is not None and error is None

        input_data = {
            "execution_id": execution_id,
            "interaction_type": interaction_type,
            "prompt_length": len(prompt) if prompt else 0,
            "execution_time": execution_time,
        }
        if additional_data:
            input_data["additional_data"] = additional_data

        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type=f"llm_{interaction_type}",
            messages=f"LLM {interaction_type}: {'SUCCESS' if success else 'FAILED'}",
            input_data=input_data,
            output_data={"response": response, "error": error} if response or error else None,
            status=ActionStatus.SUCCESS if success else ActionStatus.FAILED,
        )
        self.action_history_manager.add_action(action)
        await self._event_queue.put(action)

    async def complete_execution(self, execution_id: str, final_result: Any = None, error: str = None):
        """Complete execution."""
        if execution_id not in self._active_executions:
            logger.warning(f"Execution {execution_id} not found")
            return

        context = self._active_executions[execution_id]
        duration = time.time() - context.start_time

        success = error is None

        action = ActionHistory.create_action(
            role=ActionRole.SYSTEM,
            action_type=f"execution_complete_{context.scenario}",
            messages=f"Execution completed: {'SUCCESS' if success else 'FAILED'}",
            input_data={"execution_id": execution_id, "duration": duration},
            output_data={
                "final_result": final_result,
                "error": error,
                "execution_stats": context.execution_status.execution_stats,
            },
            status=ActionStatus.SUCCESS if success else ActionStatus.FAILED,
        )
        self.action_history_manager.add_action(action)
        await self._event_queue.put(action)

        # Clean up
        del self._active_executions[execution_id]

    async def get_events_stream(self) -> AsyncGenerator[ActionHistory, None]:
        """Get events stream."""
        while True:
            try:
                # Non-blocking get with timeout
                action = self._event_queue.get_nowait()
                yield action
            except asyncio.QueueEmpty:
                # Check if we should stop - only stop if no active executions AND queue is empty
                # We need to continue yielding events even after executions complete if there are queued events
                if not self._active_executions:
                    # Do one final check to ensure queue is truly empty
                    await asyncio.sleep(0.01)  # Small delay to let any pending puts complete
                    if self._event_queue.empty():
                        break
                else:
                    await asyncio.sleep(0.01)  # Small delay to avoid busy waiting

    def get_active_executions(self) -> Dict[str, ExecutionContext]:
        """Get active executions."""
        return self._active_executions.copy()


class BaseExecutionMode(ABC):
    """
    Base execution mode class.
    基础执行模式类。

    This class provides the foundation for different execution modes (Text2SQL, DataAnalysis, etc.).
    Each execution mode must register itself with the ExecutionEventManager before performing
    any operations to ensure proper event tracking and prevent "Execution not found" warnings.
    """

    def __init__(self, event_manager: ExecutionEventManager, context: ExecutionContext):
        self.event_manager = event_manager
        self.context = context
        self.execution_id = f"{context.scenario}_{int(time.time() * 1000)}"
        self._started = False

    async def start(self) -> None:
        """
        Register execution with ExecutionEventManager.

        This method must be called before execute() to ensure the execution is properly
        registered and can emit events. Subsequent calls are idempotent.

        The registration creates an execution start event and adds the execution to the
        active executions tracking, preventing "Execution X not found" warnings during
        status updates and tool execution events.
        """
        if self._started:
            return

        await self.event_manager.start_execution(
            execution_id=self.execution_id,
            scenario=self.context.scenario,
            task_data=self.context.task_data,
            agent_config=self.context.agent_config,
            model=self.context.model,
            workflow_metadata=self.context.workflow_metadata,
        )
        self._started = True
        logger.debug(f"Registered execution {self.execution_id} for scenario {self.context.scenario}")

    @abstractmethod
    async def execute(self) -> None:
        """Execute the specific mode."""

    async def _execute_with_error_handling(self, step_name: str, step_func):
        """Execute a step with error handling."""
        try:
            await self.event_manager.update_execution_status(self.execution_id, "executing", step_name)
            result = await step_func()
            return result
        except Exception as e:
            logger.error(f"Error in {step_name}: {e}")
            await self.event_manager.record_llm_interaction(
                self.execution_id, "error", f"Error in {step_name}", error=str(e)
            )
            raise


class Text2SQLExecutionMode(BaseExecutionMode):
    """Text2SQL execution mode."""

    async def execute(self) -> None:
        await self.event_manager.update_execution_status(self.execution_id, "planning", "Analyzing query intent")

        # Planning phase - analyze query intent
        intent_result = await self._analyze_query_intent()
        await self.event_manager.record_llm_interaction(
            self.execution_id,
            "intent_analysis",
            f"Analyze query: {self.context.task_data.get('task', '')}",
            f"Intent: {intent_result}",
        )

        # Schema linking phase
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Linking database schema")
        schema_result = await self._link_schema()
        await self.event_manager.record_tool_execution(
            self.execution_id, "schema_linking", "schema_link_call", {"intent": intent_result}, schema_result
        )

        # SQL generation phase
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Generating SQL")
        sql_result = await self._generate_sql(intent_result, schema_result)
        await self.event_manager.record_llm_interaction(
            self.execution_id, "sql_generation", f"Generate SQL for intent: {intent_result}", sql_result
        )

        # Preflight validation phase
        await self.event_manager.update_execution_status(self.execution_id, "preflight", "Validating SQL syntax")
        syntax_validation = await self._validate_sql_syntax(sql_result)
        if not syntax_validation.get("valid", False):
            await self.event_manager.record_llm_interaction(
                self.execution_id,
                "syntax_validation",
                f"Validate SQL syntax: {sql_result}",
                error=syntax_validation.get("error", "Syntax validation failed"),
            )
            # Check if we should continue on failure
            config = self.context.workflow_metadata or {}
            if not config.get("text2sql_preflight", {}).get("continue_on_failure", True):
                # Fail fast - don't continue with invalid SQL
                await self.event_manager.complete_execution(
                    self.execution_id,
                    {
                        "sql": sql_result,
                        "intent": intent_result,
                        "schema_info": schema_result,
                        "syntax_validation": syntax_validation,
                        "status": "failed",
                        "error": "SQL syntax validation failed",
                    },
                )
                return

        # Validation phase
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Validating SQL")
        validation_result = await self._validate_sql(sql_result)
        if not validation_result.get("valid", False):
            await self.event_manager.record_llm_interaction(
                self.execution_id,
                "sql_validation",
                f"Validate SQL: {sql_result}",
                error=validation_result.get("error", "Validation failed"),
            )
        else:
            await self.event_manager.record_llm_interaction(
                self.execution_id, "sql_validation", f"Validate SQL: {sql_result}", "SQL validation passed"
            )

        # Complete
        await self.event_manager.complete_execution(
            self.execution_id,
            {"sql": sql_result, "intent": intent_result, "schema_info": schema_result, "validation": validation_result},
        )

    async def _analyze_query_intent(self):
        """Analyze the user's query intent using LLM."""
        task = self.context.task_data.get("task", "")

        if self.context.model:
            try:
                intent_prompt = f"""Analyze the following natural language query and extract key information for SQL generation.

Query: {task}

Return a JSON object with the following structure:
{{
    "query_type": "SELECT|INSERT|UPDATE|DELETE|OTHER",
    "entities": ["list", "of", "mentioned", "entities", "tables"],
    "filters": ["list", "of", "filter", "conditions"],
    "aggregations": ["count", "sum", "avg", "etc"],
    "sort_requirements": ["order", "by", "clauses"],
    "temporal_aspects": ["dates", "time", "periods", "mentioned"]
}}

Be specific about table names, column names, and business logic. If uncertain, use general terms."""

                intent_result = await self.context.model.generate_with_json_output(intent_prompt)
                logger.info(f"Intent analysis result: {intent_result}")
                return intent_result

            except Exception as e:
                logger.warning(f"Intent analysis failed: {e}")
                # Fallback to basic analysis
                return {
                    "query_type": "SELECT",
                    "entities": [],
                    "filters": [],
                    "aggregations": [],
                    "sort_requirements": [],
                    "temporal_aspects": [],
                    "error": str(e),
                }

        # Fallback if no model available
        return {
            "query_type": "SELECT",
            "entities": [],
            "filters": [],
            "aggregations": [],
            "sort_requirements": [],
            "temporal_aspects": [],
        }

    async def _link_schema(self):
        """Link relevant database schema using search tools."""
        try:
            from datus.tools.func_tool.database import db_function_tool_instance

            if not self.context.agent_config:
                return {"tables": [], "columns": {}, "error": "No agent config available"}

            # Get DB tool instance
            db_tool = db_function_tool_instance(self.context.agent_config, self.context.agent_config.current_database)

            # Get intent from previous step (this assumes intent analysis was done first)
            intent = await self._analyze_query_intent()
            entities = intent.get("entities", [])

            if not entities:
                # Fallback: search based on the task description
                task_text = self.context.task_data.get("task", "")
                search_result = db_tool.search_table(
                    query_text=task_text,
                    catalog=self.context.agent_config.current_catalog or "",
                    database=self.context.agent_config.current_database or "",
                    schema_name=self.context.agent_config.current_schema or "",
                    top_n=5,
                )
            else:
                # Search for specific entities
                search_result = db_tool.search_table(
                    query_text=" ".join(entities),
                    catalog=self.context.agent_config.current_catalog or "",
                    database=self.context.agent_config.current_database or "",
                    schema_name=self.context.agent_config.current_schema or "",
                    top_n=5,
                )

            if search_result.success and hasattr(search_result, "result"):
                tables_info = []
                for table_result in search_result.result:
                    table_name = table_result.get("table_name", "")
                    if table_name:
                        # Get detailed schema for each table
                        describe_result = db_tool.describe_table(
                            table_name=table_name,
                            catalog=self.context.agent_config.current_catalog or "",
                            database=self.context.agent_config.current_database or "",
                            schema_name=self.context.agent_config.current_schema or "",
                        )

                        if describe_result.success and hasattr(describe_result, "result"):
                            tables_info.append({"table_name": table_name, "schema": describe_result.result})

                return {
                    "tables": tables_info,
                    "search_results": search_result.result if hasattr(search_result, "result") else [],
                    "intent": intent,
                }
            else:
                return {
                    "tables": [],
                    "columns": {},
                    "error": search_result.error if hasattr(search_result, "error") else "Search failed",
                }

        except Exception as e:
            logger.error(f"Schema linking failed: {e}")
            return {"tables": [], "columns": {}, "error": str(e)}

    async def _generate_sql(self, intent, schema_info):
        """Generate SQL based on intent and schema using LLM."""
        if not self.context.model:
            return "SELECT * FROM table"  # Fallback

        try:
            # Build comprehensive context for SQL generation
            task = self.context.task_data.get("task", "")
            tables_info = schema_info.get("tables", [])

            # Format schema information for prompt
            schema_context = ""
            if tables_info:
                schema_lines = []
                for table_info in tables_info:
                    table_name = table_info.get("table_name", "")
                    schema_data = table_info.get("schema", {})

                    schema_lines.append(f"Table: {table_name}")
                    if "columns" in schema_data:
                        columns = schema_data["columns"]
                        if isinstance(columns, list):
                            for col in columns:
                                if isinstance(col, dict):
                                    col_name = col.get("name", "")
                                    col_type = col.get("type", "")
                                    col_comment = col.get("comment", "")
                                    schema_lines.append(
                                        f"  - {col_name} ({col_type}) {f': {col_comment}' if col_comment else ''}"
                                    )
                    schema_lines.append("")  # Empty line between tables

                schema_context = "\n".join(schema_lines)

            # Build SQL generation prompt
            sql_prompt = f"""Generate accurate SQL based on the user's query and available schema information.

User Query: {task}

Available Tables and Columns:
{schema_context}

Query Intent Analysis:
- Type: {intent.get('query_type', 'SELECT')}
- Entities: {', '.join(intent.get('entities', []))}
- Filters: {', '.join(intent.get('filters', []))}
- Aggregations: {', '.join(intent.get('aggregations', []))}
- Sort Requirements: {', '.join(intent.get('sort_requirements', []))}
- Temporal Aspects: {', '.join(intent.get('temporal_aspects', []))}

Requirements:
1. Use only the tables and columns provided in the schema information
2. Ensure SQL syntax is correct for the target database
3. Include appropriate WHERE, JOIN, GROUP BY, ORDER BY clauses as needed
4. Use proper column names and table aliases
5. Return only the SQL query without explanation

Generate the SQL query:"""

            # Generate SQL using LLM
            sql_result = await self.context.model.generate(sql_prompt)
            generated_sql = sql_result.strip()

            # Clean up the SQL (remove markdown code blocks if present)
            if generated_sql.startswith("```"):
                # Extract SQL from markdown code block
                lines = generated_sql.split("\n")
                sql_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        sql_lines.append(line)
                generated_sql = "\n".join(sql_lines).strip()

            logger.info(f"Generated SQL: {generated_sql}")
            return generated_sql

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            # Return a basic fallback SQL
            fallback_tables = schema_info.get("tables", [])
            if fallback_tables:
                table_name = fallback_tables[0].get("table_name", "table")
                return f"SELECT * FROM {table_name} LIMIT 10"
            else:
                return "SELECT * FROM table"

    async def _validate_sql_syntax(self, sql: str):
        """Validate SQL syntax using DB tool."""
        try:
            # Import here to avoid circular imports
            from datus.tools.func_tool.database import db_function_tool_instance

            # Get DB tool instance
            if self.context.agent_config:
                db_tool = db_function_tool_instance(
                    self.context.agent_config, self.context.agent_config.current_database
                )
                result = db_tool.validate_sql_syntax(sql)

                if result.success:
                    return {
                        "valid": True,
                        "tables_referenced": result.result.get("tables_referenced", []),
                        "sql_type": result.result.get("sql_type", "unknown"),
                    }
                else:
                    return {"valid": False, "error": result.error}
            else:
                # Fallback to basic validation if no config
                return await self._validate_sql(sql)

        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}

    async def _validate_sql(self, sql: str):
        """Validate generated SQL."""
        try:
            import sqlglot

            parsed = sqlglot.parse_one(sql)
            return {"valid": True, "parsed": str(parsed)}
        except Exception as e:
            return {"valid": False, "error": str(e)}


class SQLReviewExecutionMode(BaseExecutionMode):
    """SQL Review execution mode."""

    def __init__(self, event_manager: ExecutionEventManager, context: ExecutionContext, chat_node=None):
        super().__init__(event_manager, context)
        self.chat_node = chat_node

    async def execute(self) -> None:
        await self.event_manager.update_execution_status(self.execution_id, "planning", "Initializing SQL review")

        # Extract SQL
        sql_query = self._extract_sql_from_task()

        # Syntax validation
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Validating SQL syntax")
        await self._validate_sql_syntax(sql_query)

        # Preflight tools execution
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Running preflight tools")
        await self._execute_preflight_tools(sql_query)

        # LLM analysis
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Performing LLM analysis")
        review_result = await self._perform_llm_analysis(sql_query)

        # Complete
        await self.event_manager.complete_execution(self.execution_id, review_result)

    def _extract_sql_from_task(self) -> str:
        """Extract SQL from task text."""
        if self.chat_node:
            return self.chat_node._extract_sql_from_task(self.context.task_data.get("task", ""))
        # Fallback implementation
        task_text = self.context.task_data.get("task", "")
        import re

        sql_match = re.search(r"```\s*sql\s*(.*?)\s*```", task_text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        return task_text.strip()

    async def _validate_sql_syntax(self, sql: str):
        """Validate SQL syntax."""
        if self.chat_node:
            validation_result = self.chat_node._validate_sql_syntax_comprehensive(sql)
            if not validation_result.get("valid", False):
                error_msg = validation_result.get("error", "Unknown syntax error")
                await self.event_manager.record_llm_interaction(
                    self.execution_id, "syntax_validation", f"Validate SQL: {sql}", error=error_msg
                )
            else:
                await self.event_manager.record_llm_interaction(
                    self.execution_id, "syntax_validation", f"Validate SQL: {sql}", "Syntax validation passed"
                )

    async def _execute_preflight_tools(self, sql: str):
        """Execute preflight tools."""
        if self.chat_node:
            # Use the existing preflight execution logic from ChatAgenticNode
            await self.chat_node.run_preflight_tools(sql, self.event_manager, self.execution_id)

    async def _perform_llm_analysis(self, sql: str):
        """Perform LLM analysis."""
        # This would integrate with the LLM analysis logic
        return {"review": "SQL analysis complete", "sql": sql}


class DataAnalysisExecutionMode(BaseExecutionMode):
    """Data Analysis execution mode."""

    async def execute(self) -> None:
        await self.event_manager.update_execution_status(self.execution_id, "planning", "Planning data analysis")

        # Data exploration phase
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Exploring data")
        exploration_result = await self._explore_data()
        await self.event_manager.record_tool_execution(
            self.execution_id,
            "data_exploration",
            "explore_call",
            {"task": self.context.task_data.get("task", "")},
            exploration_result,
        )

        # Analysis planning
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Planning analysis approach")
        analysis_plan = await self._plan_analysis(exploration_result)
        await self.event_manager.record_llm_interaction(
            self.execution_id,
            "analysis_planning",
            f"Plan analysis based on exploration: {exploration_result}",
            f"Analysis plan: {analysis_plan}",
        )

        # Analysis execution
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Executing analysis")
        analysis_result = await self._execute_analysis(analysis_plan)
        await self.event_manager.record_tool_execution(
            self.execution_id, "data_analysis", "analysis_call", {"plan": analysis_plan}, analysis_result
        )

        # Result formatting
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Formatting results")
        formatted_result = await self._format_results(analysis_result)
        await self.event_manager.record_llm_interaction(
            self.execution_id,
            "result_formatting",
            f"Format analysis results: {analysis_result}",
            f"Formatted results: {formatted_result}",
        )

        # Complete
        await self.event_manager.complete_execution(
            self.execution_id,
            {
                "exploration": exploration_result,
                "plan": analysis_plan,
                "analysis": analysis_result,
                "formatted_results": formatted_result,
            },
        )

    async def _explore_data(self):
        """Explore available data."""
        # This would query database for available tables, sample data, etc.
        return {"tables": [], "sample_data": {}, "data_quality": {}}

    async def _plan_analysis(self, exploration_result):
        """Plan the analysis approach."""
        if self.context.model:
            try:
                # This would use LLM to plan analysis
                return {"approach": "statistical_analysis", "metrics": [], "visualizations": []}
            except Exception as e:
                logger.warning(f"Analysis planning failed: {e}")
                return {"approach": "basic_summary"}
        return {"approach": "basic_summary"}

    async def _execute_analysis(self, analysis_plan):
        """Execute the planned analysis."""
        # This would run various analysis tools based on the plan
        return {"statistics": {}, "insights": [], "recommendations": []}

    async def _format_results(self, analysis_result):
        """Format analysis results for presentation."""
        # This would format results into charts, tables, narratives, etc.
        return {
            "summary": "Analysis completed",
            "key_findings": analysis_result.get("insights", []),
            "visualizations": [],
        }


class SmartQueryExecutionMode(BaseExecutionMode):
    """Smart Query execution mode for intelligent query recommendations."""

    async def execute(self) -> None:
        await self.event_manager.update_execution_status(
            self.execution_id, "planning", "Understanding query requirements"
        )

        # Requirements analysis
        requirements = await self._analyze_requirements()
        await self.event_manager.record_llm_interaction(
            self.execution_id,
            "requirements_analysis",
            f"Analyze requirements: {self.context.task_data.get('task', '')}",
            f"Requirements: {requirements}",
        )

        # Query generation
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Generating smart queries")
        queries = await self._generate_queries(requirements)
        await self.event_manager.record_llm_interaction(
            self.execution_id,
            "query_generation",
            f"Generate queries for requirements: {requirements}",
            f"Generated queries: {queries}",
        )

        # Query validation
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Validating queries")
        validated_queries = await self._validate_queries(queries)

        # Recommendations
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Generating recommendations")
        recommendations = await self._generate_recommendations(validated_queries)

        # Complete
        await self.event_manager.complete_execution(
            self.execution_id,
            {"requirements": requirements, "queries": validated_queries, "recommendations": recommendations},
        )

    async def _analyze_requirements(self):
        """Analyze user requirements for smart queries."""
        return {"analysis_type": "recommendation", "complexity": "medium"}

    async def _generate_queries(self, requirements):
        """Generate multiple query options."""
        return [{"sql": "SELECT * FROM table", "explanation": "Basic query"}]

    async def _validate_queries(self, queries):
        """Validate generated queries."""
        return queries

    async def _generate_recommendations(self, queries):
        """Generate recommendations based on queries."""
        return {"primary_query": queries[0] if queries else None, "alternatives": queries[1:]}


class DeepAnalysisExecutionMode(BaseExecutionMode):
    """Deep Analysis execution mode for comprehensive data insights."""

    async def execute(self) -> None:
        await self.event_manager.update_execution_status(self.execution_id, "planning", "Planning deep analysis")

        # Multi-dimensional analysis planning
        analysis_plan = await self._plan_deep_analysis()
        await self.event_manager.record_llm_interaction(
            self.execution_id,
            "deep_analysis_planning",
            f"Plan deep analysis: {self.context.task_data.get('task', '')}",
            f"Analysis plan: {analysis_plan}",
        )

        # Execute multiple analysis dimensions
        await self.event_manager.update_execution_status(
            self.execution_id, "executing", "Executing multi-dimensional analysis"
        )
        analysis_results = await self._execute_deep_analysis(analysis_plan)

        # Synthesize insights
        await self.event_manager.update_execution_status(self.execution_id, "executing", "Synthesizing insights")
        insights = await self._synthesize_insights(analysis_results)

        # Generate comprehensive report
        await self.event_manager.update_execution_status(
            self.execution_id, "executing", "Generating comprehensive report"
        )
        report = await self._generate_comprehensive_report(insights)

        # Complete
        await self.event_manager.complete_execution(
            self.execution_id,
            {"plan": analysis_plan, "results": analysis_results, "insights": insights, "report": report},
        )

    async def _plan_deep_analysis(self):
        """Plan comprehensive deep analysis."""
        return {
            "dimensions": ["trend", "correlation", "anomaly", "prediction"],
            "data_sources": [],
            "methodologies": ["statistical", "ml", "domain_expert"],
        }

    async def _execute_deep_analysis(self, analysis_plan):
        """Execute deep analysis across multiple dimensions."""
        return {"trend_analysis": {}, "correlation_analysis": {}, "anomaly_detection": {}}

    async def _synthesize_insights(self, analysis_results):
        """Synthesize insights from multiple analysis results."""
        return {"key_insights": [], "patterns": [], "recommendations": []}

    async def _generate_comprehensive_report(self, insights):
        """Generate comprehensive analysis report."""
        return {"executive_summary": "", "detailed_findings": {}, "conclusions": []}


# Factory function for execution modes
def create_execution_mode(
    scenario: str, event_manager: ExecutionEventManager, context: ExecutionContext, chat_node=None
) -> BaseExecutionMode:
    """Factory function to create execution mode based on scenario."""
    mode_classes = {
        "text2sql": Text2SQLExecutionMode,
        "sql_review": SQLReviewExecutionMode,
        "data_analysis": DataAnalysisExecutionMode,
        "smart_query": SmartQueryExecutionMode,
        "deep_analysis": DeepAnalysisExecutionMode,
    }

    mode_class = mode_classes.get(scenario)
    if not mode_class:
        raise ValueError(f"Unsupported execution scenario: {scenario}")

    if scenario == "sql_review":
        return mode_class(event_manager, context, chat_node)
    else:
        return mode_class(event_manager, context)
