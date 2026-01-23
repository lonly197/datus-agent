# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from agents import Tool

from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
from datus.prompts.reasoning_sql_with_mcp import get_reasoning_prompt
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.schemas.reason_sql_node_models import ReasoningInput, ReasoningResult
from datus.tools.llms_tools.mcp_stream_utils import base_mcp_stream
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import llm_result2json, llm_result2sql
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


def _has_builtin_sql_tools(tools: List[Tool]) -> bool:
    """Check if the tools list contains SQL execution tools."""
    tool_names = [t.name for t in tools]
    return any(name in tool_names for name in ["read_query", "execute_sql", "sql_query"])


def _get_read_query_tool(tools: List[Tool]) -> Optional[Tool]:
    """Get the read_query tool from the tools list."""
    for tool in tools:
        if tool.name in ["read_query", "execute_sql", "sql_query"]:
            return tool
    return None


async def reasoning_sql_with_builtin_tools_stream(
    model: LLMBaseModel,
    input_data: ReasoningInput,
    tools: List[Tool],
    tool_config: Dict[str, Any],
    action_history_manager: Optional[ActionHistoryManager] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """
    Generate SQL reasoning using builtin tools instead of MCP.

    This provides a fallback when no MCP servers are configured,
    using the builtin database tools (read_query, etc.) for iterative SQL reasoning.
    """
    if not isinstance(input_data, ReasoningInput):
        logger.error(f"Input type error: expected ReasoningInput, got {type(input_data)}")
        raise ValueError(f"Input must be a ReasoningInput instance, got {type(input_data)}")

    if action_history_manager is None:
        action_history_manager = ActionHistoryManager()

    max_turns = tool_config.get("max_turns", 10)

    # Find the read_query tool
    read_query_tool = _get_read_query_tool(tools)
    if not read_query_tool:
        logger.warning("No read_query tool found in builtin tools, falling back to single-pass generation")
        # Yield a warning action and then return
        yield ActionHistory(
            action_id="reasoning_fallback_warning",
            role=ActionRole.WORKFLOW,
            messages="No SQL execution tool available, using single-pass generation",
            action_type="reasoning_fallback",
            input={"reason": "no_read_query_tool"},
            output={"fallback": "single_pass"},
            status=ActionStatus.SUCCESS,
        )
        return

    # Setup initial prompt
    prompt = get_reasoning_prompt(
        database_type=input_data.get("database_type", "sqlite"),
        table_schemas=input_data.table_schemas,
        data_details=input_data.data_details,
        metrics=input_data.metrics,
        question=input_data.sql_task.task,
        context=[sql_context.to_str(input_data.max_sql_return_length) for sql_context in input_data.contexts],
        prompt_version=input_data.prompt_version,
        max_table_schemas_length=input_data.max_table_schemas_length,
        max_data_details_length=input_data.max_data_details_length,
        max_context_length=input_data.max_context_length,
        max_value_length=input_data.max_value_length,
        max_text_mark_length=input_data.max_text_mark_length,
        knowledge_content=input_data.external_knowledge,
    )

    # Get system instruction
    instruction = prompt_manager.get_raw_template("reasoning_system", input_data.prompt_version)

    # Track SQL contexts from tool executions
    sql_contexts = []
    current_turn = 0
    iteration_history = []

    # Emit setup action
    setup_action = ActionHistory(
        action_id="reasoning_builtin_setup",
        role=ActionRole.WORKFLOW,
        messages="Setting up builtin tools reasoning (no MCP servers configured)",
        action_type="reasoning_setup",
        input={
            "database_type": input_data.get("database_type"),
            "task": input_data.sql_task.task,
            "max_turns": max_turns,
            "builtin_tools_used": True,
        },
        status=ActionStatus.SUCCESS,
    )
    yield setup_action
    setup_action.end_time = datetime.now()

    logger.info(f"Starting builtin tools reasoning with {max_turns} max turns")

    while current_turn < max_turns:
        current_turn += 1
        iteration_history.append(f"\n=== Iteration {current_turn} ===")

        # Emit iteration start action
        iteration_action = ActionHistory(
            action_id=f"reasoning_iteration_{current_turn}",
            role=ActionRole.WORKFLOW,
            messages=f"Starting reasoning iteration {current_turn}/{max_turns}",
            action_type="reasoning_iteration",
            input={"iteration": current_turn, "max_turns": max_turns},
            status=ActionStatus.PROCESSING,
        )
        yield iteration_action

        try:
            # Generate with tools (single turn)
            exec_result = await model.generate_with_tools(
                prompt=prompt,
                mcp_servers={},  # No MCP servers, use builtin tools
                instruction=instruction,
                tools=tools,
                output_type=str,
                max_turns=1,  # Single turn per iteration
            )

            # Extract the generated SQL
            content = exec_result.get("content", "")
            content_dict = llm_result2json(content)

            if content_dict and content_dict.get("sql"):
                sql_query = content_dict.get("sql", "")
                explanation = content_dict.get("explanation", "")
            else:
                # Try to extract SQL from response
                sql_query = llm_result2sql(content) or ""
                explanation = ""

            if not sql_query:
                logger.info(f"No SQL generated in iteration {current_turn}, stopping reasoning")
                iteration_action.output = {
                    "success": False,
                    "reason": "no_sql_generated",
                    "iteration": current_turn,
                }
                iteration_action.status = ActionStatus.SUCCESS
                yield iteration_action
                break

            logger.info(f"Iteration {current_turn}: Generated SQL: {sql_query[:100]}...")

            # Execute the SQL using builtin tool
            tool_output_action = ActionHistory(
                action_id=f"reasoning_execute_{current_turn}",
                role=ActionRole.TOOL,
                messages=f"Executing SQL in iteration {current_turn}",
                action_type="read_query",
                input={"sql": sql_query, "iteration": current_turn},
                status=ActionStatus.PROCESSING,
            )
            yield tool_output_action

            # Call the read_query tool
            try:
                if hasattr(read_query_tool, "_func"):
                    # It's a wrapped function tool
                    result = read_query_tool._func(sql=sql_query)
                else:
                    # It's a standard Tool, use tool call
                    result = await model.call_tool(read_query_tool, {"sql": sql_query})

                # Check if result is successful
                if hasattr(result, 'success'):
                    success = result.success
                    result_data = result.result if success else str(result.error or "Unknown error")
                    error = None if success else result.error
                else:
                    # Handle raw result
                    success = True
                    result_data = str(result)
                    error = None

                tool_output_action.output = {
                    "success": success,
                    "result": str(result_data)[:500] if result_data else "",
                    "error": error,
                }
                tool_output_action.status = ActionStatus.SUCCESS if success else ActionStatus.FAILED
                tool_output_action.end_time = datetime.now()
                yield tool_output_action

                # Create SQL context
                sql_context = SQLContext(
                    sql_query=sql_query,
                    explanation=explanation,
                    sql_return=str(result_data) if success else "",
                    sql_error=str(error) if error else "",
                    row_count=0,
                )
                sql_contexts.append(sql_context)

                # Store in action history manager for later retrieval
                if not hasattr(action_history_manager, "sql_contexts"):
                    action_history_manager.sql_contexts = []
                action_history_manager.sql_contexts = sql_contexts

                # Check if we should continue or stop
                if success:
                    # Check if result looks reasonable (non-empty)
                    result_str = str(result_data).strip()
                    if result_str and len(result_str) > 0:
                        logger.info(f"SQL execution successful in iteration {current_turn}")
                        # Continue to let LLM decide if more iterations are needed
                        # Update prompt with execution result
                        iteration_history.append(f"SQL: {sql_query}")
                        iteration_history.append(f"Result: {result_str[:200]}...")

                        # Add result to prompt for next iteration
                        result_context = f"\nPrevious iteration {current_turn} result:\nSQL: {sql_query}\nResult: {result_str[:500]}"
                        if prompt[-1]["role"] == "user":
                            prompt[-1]["content"] += result_context
                        else:
                            prompt.append({"role": "user", "content": result_context})

                        iteration_action.output = {
                            "success": True,
                            "sql": sql_query,
                            "result_preview": str(result_data)[:200],
                            "iteration": current_turn,
                            "continue": True,
                        }
                        iteration_action.status = ActionStatus.SUCCESS
                        yield iteration_action
                        continue

                # If execution failed or result is empty, stop
                logger.info(f"Stopping reasoning after iteration {current_turn}")
                break

            except Exception as e:
                logger.error(f"Tool execution error in iteration {current_turn}: {e}")
                tool_output_action.output = {
                    "success": False,
                    "error": str(e),
                }
                tool_output_action.status = ActionStatus.FAILED
                tool_output_action.end_time = datetime.now()
                yield tool_output_action
                break

        except Exception as e:
            logger.error(f"Reasoning iteration {current_turn} error: {e}")
            iteration_action.output = {
                "success": False,
                "error": str(e),
                "iteration": current_turn,
            }
            iteration_action.status = ActionStatus.SUCCESS
            yield iteration_action
            break

    # Emit completion action
    completion_action = ActionHistory(
        action_id="reasoning_builtin_complete",
        role=ActionRole.WORKFLOW,
        messages=f"Builtin tools reasoning completed after {current_turn} iterations",
        action_type="reasoning_complete",
        input={"iterations": current_turn, "max_turns": max_turns, "sql_contexts_count": len(sql_contexts)},
        output={"sql_contexts": [ctx.sql_query for ctx in sql_contexts]},
        status=ActionStatus.SUCCESS,
    )
    yield completion_action

    logger.info(f"Builtin tools reasoning completed: {len(sql_contexts)} SQL contexts generated")


@optional_traceable()
async def reasoning_sql_with_mcp_stream(
    model: LLMBaseModel,
    input_data: ReasoningInput,
    tool_config: Dict[str, Any],
    tools: List[Tool],
    action_history_manager: Optional[ActionHistoryManager] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """
    Generate SQL reasoning with streaming support and action history tracking.

    This function supports both MCP-based and builtin tools-based reasoning:
    - If MCP servers are configured: Use MCP protocol for tool calls
    - If no MCP servers but builtin SQL tools available: Use builtin tools
    - Otherwise: Fall back to single-pass generation
    """
    if not isinstance(input_data, ReasoningInput):
        logger.error(f"Input type error: expected ReasoningInput, got {type(input_data)}")
        raise ValueError(f"Input must be a ReasoningInput instance, got {type(input_data)}")

    # If no action history manager provided, create one to track the final result
    if action_history_manager is None:
        action_history_manager = ActionHistoryManager()

    # Check if we have MCP servers configured (from tool_config)
    mcp_servers = tool_config.get("mcp_servers", {})

    # Check if we have builtin SQL tools available
    has_builtin_tools = _has_builtin_sql_tools(tools)

    logger.info(f"Reasoning SQL - MCP servers: {len(mcp_servers)}, Builtin tools: {has_builtin_tools}")

    # Decision: Use builtin tools if no MCP servers but builtin tools available
    if not mcp_servers and has_builtin_tools:
        logger.info("No MCP servers configured, using builtin tools for reasoning")
        # Emit info action about fallback
        yield ActionHistory(
            action_id="reasoning_fallback_info",
            role=ActionRole.WORKFLOW,
            messages="Using builtin database tools for SQL reasoning (MCP servers not configured)",
            action_type="reasoning_fallback_info",
            input={
                "mcp_servers_count": len(mcp_servers),
                "builtin_tools_used": True,
                "reason": "no_mcp_servers",
            },
            status=ActionStatus.SUCCESS,
        )

        # Use builtin tools for reasoning
        async for action in reasoning_sql_with_builtin_tools_stream(
            model=model,
            input_data=input_data,
            tools=tools,
            tool_config=tool_config,
            action_history_manager=action_history_manager,
        ):
            yield action
        return

    # If no MCP servers and no builtin tools, emit warning
    if not mcp_servers and not has_builtin_tools:
        logger.warning("No MCP servers and no builtin SQL tools available")
        yield ActionHistory(
            action_id="reasoning_no_tools_warning",
            role=ActionRole.WORKFLOW,
            messages="No tools available for SQL reasoning",
            action_type="reasoning_no_tools",
            input={"reason": "no_tools_available"},
            status=ActionStatus.SUCCESS,
        )
        return

    # Use MCP protocol for reasoning
    prompt = get_reasoning_prompt(
        database_type=input_data.get("database_type", "sqlite"),
        table_schemas=input_data.table_schemas,
        data_details=input_data.data_details,
        metrics=input_data.metrics,
        question=input_data.sql_task.task,
        context=[sql_context.to_str(input_data.max_sql_return_length) for sql_context in input_data.contexts],
        prompt_version=input_data.prompt_version,
        max_table_schemas_length=input_data.max_table_schemas_length,
        max_data_details_length=input_data.max_data_details_length,
        max_context_length=input_data.max_context_length,
        max_value_length=input_data.max_value_length,
        max_text_mark_length=input_data.max_text_mark_length,
        knowledge_content=input_data.external_knowledge,
    )

    async for action in base_mcp_stream(
        model=model,
        input_data=input_data,
        tool_config=tool_config,
        mcp_servers=mcp_servers,
        prompt=prompt,
        tools=tools,
        instruction_template="reasoning_system",
        action_history_manager=action_history_manager,
    ):
        yield action

    # After streaming completes, extract final result and add to SQLContext
    try:
        # Find the final message/result from action history
        final_message_action = None
        sql_contexts = []

        # Look for actions that contain SQL execution results
        for action in action_history_manager.actions:
            if action.action_type == "read_query" and action.status.value == "success":
                # This is a SQL execution result, create SQLContext from it
                from datus.schemas.node_models import SQLContext

                sql_input = action.input or {}
                sql_output = action.output or {}

                sql_context = SQLContext(
                    sql_query=sql_input.get("sql", ""),
                    explanation="",
                    sql_return=sql_output.get("result", ""),
                    sql_error=sql_output.get("error", ""),
                    row_count=0,
                )
                sql_contexts.append(sql_context)

            elif action.action_type == "message" and action.role.value == "assistant":
                # This could be the final reasoning result
                final_message_action = action

        # Extract the final SQL from the final message if available
        if final_message_action and final_message_action.output:
            raw_output = final_message_action.output.get("raw_output", "")
            if raw_output:
                try:
                    # Parse the final result to extract SQL
                    content_dict = llm_result2json(raw_output)
                    sql_query = content_dict.get("sql", "")

                    if sql_query:
                        # Create SQLContext with the final result SQL
                        from datus.schemas.node_models import SQLContext

                        final_sql_context = SQLContext(
                            sql_query=sql_query,
                            explanation=content_dict.get("explanation", ""),
                            sql_return="",  # Will be filled by execution
                            sql_error="",
                            row_count=0,
                        )
                        sql_contexts.append(final_sql_context)
                        logger.info(f"Added final result SQL to SQLContext: {sql_query[:100]}...")

                except Exception as e:
                    logger.debug(f"Could not parse final message as JSON: {e}")

        # Store sql_contexts in action history manager for later retrieval
        if not hasattr(action_history_manager, "sql_contexts"):
            action_history_manager.sql_contexts = []
        action_history_manager.sql_contexts.extend(sql_contexts)

    except Exception as e:
        logger.warning(f"Failed to extract final result SQL for SQLContext: {e}")
        # Don't fail the entire process, just log the warning


@optional_traceable()
def reasoning_sql_with_mcp(
    model: LLMBaseModel, input_data: ReasoningInput, tools: List[Tool], tool_config: Dict[str, Any]
) -> ReasoningResult:
    """Generate SQL via MCP, execute it, and return the execution result."""
    if not isinstance(input_data, ReasoningInput):
        logger.error(f"Input type error: expected ReasoningInput, got {type(input_data)}")
        raise ValueError(f"Input must be a ReasoningInput instance, got {type(input_data)}")

    instruction = prompt_manager.get_raw_template("reasoning_system", input_data.prompt_version)
    mcp_servers = tool_config.get("mcp_servers", {})
    # update to python 3.12 to enable structured output
    # output_type = tool_config.get(
    # "output_type", {"sql": str, "tables": list, "explanation": str})
    # tool_list =
    max_turns = tool_config.get("max_turns", 10)

    prompt = get_reasoning_prompt(
        database_type=input_data.get("database_type", DBType.SQLITE),
        table_schemas=input_data.table_schemas,
        data_details=input_data.data_details,
        metrics=input_data.metrics,
        question=input_data.sql_task.task,
        context=[sql_context.to_str(input_data.max_sql_return_length) for sql_context in input_data.contexts],
        prompt_version=input_data.prompt_version,
        max_table_schemas_length=input_data.max_table_schemas_length,
        max_data_details_length=input_data.max_data_details_length,
        max_context_length=input_data.max_context_length,
        max_value_length=input_data.max_value_length,
        max_text_mark_length=input_data.max_text_mark_length,
        knowledge_content=input_data.external_knowledge,
    )
    try:
        exec_result = asyncio.run(
            model.generate_with_tools(
                prompt=prompt,
                mcp_servers=mcp_servers,
                instruction=instruction,
                tools=tools,
                # if model is OpenAI, json_schema output is supported, use ReasoningSQLResponse
                output_type=str,
                max_turns=max_turns,
            )
        )

        logger.debug(f"Reasoning SQL execute result: {exec_result['content']}")

        # Try JSON parsing first
        content_dict = llm_result2json(exec_result["content"])
        if content_dict:
            # Successfully parsed JSON with meaningful SQL content
            logger.info(f"Successfully parsed JSON content: {content_dict}")
            reasoning_result = ReasoningResult(
                success=True,
                sql_query=content_dict.get("sql", ""),
                sql_return="",  # Remove the result from the return to avoid large data return
                sql_contexts=exec_result["sql_contexts"],
            )
            logger.info(
                f"Created ReasoningResult: success={reasoning_result.success}, sql_query={reasoning_result.sql_query}"
            )
            return reasoning_result

        # JSON parsing failed, try SQL extraction.
        # Some LLM can't follow the instruction well, try some failback
        extracted_sql = llm_result2sql(exec_result["content"])
        if extracted_sql:
            # Successfully extracted SQL from code blocks
            logger.info(f"Extract json format failed, but find a sql {extracted_sql} from response")
            return ReasoningResult(
                success=True,
                sql_query=extracted_sql,
                sql_return="",
                sql_contexts=exec_result["sql_contexts"],
            )

        # Both JSON and SQL extraction failed, raise exception
        response_content = exec_result["content"]
        response_preview = response_content[:20] if response_content else ""
        response_length = len(response_content) if response_content else 0
        logger.error(f"Extract json format/sql failed. len:{response_length}, resp:{response_preview}... ")
        raise DatusException(
            ErrorCode.MODEL_ILLEGAL_FORMAT_RESPONSE,
            message_args={"response_preview": response_preview, "response_length": response_length},
        )

    except DatusException:
        raise
    except Exception as e:
        # TODO : deal with exceed the max round
        error_msg = str(e)
        logger.error(f"Reasoning SQL with MCP failed: {e}")

        # Re-raise permission/tool-calling errors so fallback can handle them
        if any(indicator in error_msg.lower() for indicator in ["403", "forbidden", "not allowed", "permission"]):
            logger.info("Re-raising permission error for fallback handling")
            raise

        # Return failed result for other errors
        logger.error(f"Reasoning SQL failed: {e}")
        raise DatusException(
            ErrorCode.NODE_EXECUTION_FAILED,
            message=f"Reasoning SQL failed: {e}",
        )
