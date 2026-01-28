# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Event emission for plan mode hooks.

This module contains methods for emitting events to the frontend:
- Action history events
- Plan update events
- Tool error events
- Status messages
- SQL execution events
"""

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from datus.api.models import (
    ChatEvent,
    DeepResearchEventType,
    SqlExecutionErrorEvent,
    SqlExecutionProgressEvent,
    SqlExecutionResultEvent,
    SqlExecutionStartEvent,
)
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from .core import PlanModeHooksCore

logger = get_logger(__name__)


async def emit_action(hooks: "PlanModeHooksCore", action: ActionHistory):
    """Emit action using the appropriate method.

    Args:
        hooks: PlanModeHooks instance
        action: Action to emit
    """
    if hooks.execution_event_manager and hooks.current_execution_id:
        # Use unified execution event manager
        if hasattr(hooks.execution_event_manager, "_event_queue"):
            await hooks.execution_event_manager._event_queue.put(action)
    elif hooks.emit_queue:
        # Fallback to legacy emit_queue
        await hooks.emit_queue.put(action)
    elif hooks.action_history_manager:
        # Fallback to action history manager
        hooks.action_history_manager.add_action(action)


async def emit_plan_update_event(hooks: "PlanModeHooksCore"):
    """Emit plan update event to notify frontend.

    Args:
        hooks: PlanModeHooks instance
    """
    try:
        todo_list = hooks.todo_storage.get_todo_list()
        if not todo_list:
            return

        # Create plan update event
        plan_data = {
            "items": [item.model_dump() for item in todo_list.items],
            "total": len(todo_list.items),
            "completed": sum(1 for item in todo_list.items if item.status == "completed"),
            "pending": sum(1 for item in todo_list.items if item.status == "pending"),
            "phase": hooks.plan_phase,
            "mode": hooks.execution_mode,
        }

        # Emit via appropriate channel
        if hooks.action_history_manager:
            action = ActionHistory(
                id=str(uuid.uuid4()),
                role=ActionRole.ASSISTANT,
                action_type="plan_update",
                messages=f"Plan updated: {plan_data['completed']}/{plan_data['total']} completed",
                input=plan_data,
                status=ActionStatus.SUCCESS,
                timestamp=time.time(),
            )
            await emit_action(hooks, action)

        logger.debug(f"Emitted plan update event: {plan_data}")

    except Exception as e:
        logger.error(f"Failed to emit plan update event: {e}")


async def emit_tool_error_event(
    hooks: "PlanModeHooksCore",
    tool_name: str,
    error_message: str,
    error_info: Dict[str, Any],
    tool_params: Dict[str, Any],
):
    """Emit tool error event with detailed information.

    Args:
        hooks: PlanModeHooks instance
        tool_name: Name of tool that failed
        error_message: Error message
        error_info: Detailed error information
        tool_params: Parameters passed to tool
    """
    try:
        error_data = {
            "tool_name": tool_name,
            "error_type": error_info.get("error_type", "unknown"),
            "error_message": error_message,
            "suggestions": error_info.get("suggestions", []),
            "can_retry": error_info.get("can_retry", False),
            "retry_delay": error_info.get("retry_delay", 0),
            "auto_fix_available": error_info.get("auto_fix_available", False),
            "fallback_tool": error_info.get("fallback_tool"),
            "params": tool_params,
        }

        if hooks.action_history_manager:
            action = ActionHistory(
                id=str(uuid.uuid4()),
                role=ActionRole.TOOL,
                action_type=tool_name,
                messages=f"Tool error: {error_message}",
                input=tool_params,
                output={"error": error_data},
                status=ActionStatus.FAILURE,
                timestamp=time.time(),
            )
            await emit_action(hooks, action)

        logger.debug(f"Emitted tool error event: {error_data}")

    except Exception as e:
        logger.error(f"Failed to emit tool error event: {e}")


async def emit_status_message(hooks: "PlanModeHooksCore", message: str, level: str = "info"):
    """Emit status message to frontend.

    Args:
        hooks: PlanModeHooks instance
        message: Status message
        level: Message level (info, warning, error)
    """
    try:
        status_data = {
            "message": message,
            "level": level,
            "phase": hooks.plan_phase,
            "timestamp": time.time(),
        }

        if hooks.action_history_manager:
            action = ActionHistory(
                id=str(uuid.uuid4()),
                role=ActionRole.ASSISTANT,
                action_type="status_message",
                messages=message,
                input=status_data,
                status=ActionStatus.SUCCESS,
                timestamp=time.time(),
            )
            await emit_action(hooks, action)

        logger.debug(f"Emitted status message: {message}")

    except Exception as e:
        logger.error(f"Failed to emit status message: {e}")


async def emit_sql_execution_start(
    hooks: "PlanModeHooksCore",
    sql: str,
    execution_id: str,
    database: Optional[str] = None,
):
    """Emit SQL execution start event.

    Args:
        hooks: PlanModeHooks instance
        sql: SQL query being executed
        execution_id: Unique execution ID
        database: Optional database name
    """
    try:
        event = SqlExecutionStartEvent(
            execution_id=execution_id,
            sql=sql,
            database=database,
            timestamp=time.time(),
        )

        if hooks.emit_queue:
            await hooks.emit_queue.put(ChatEvent(type="sql_execution_start", data=event.model_dump()))

        logger.debug(f"Emitted SQL execution start event: {execution_id}")

    except Exception as e:
        logger.error(f"Failed to emit SQL execution start event: {e}")


async def emit_sql_execution_result(
    hooks: "PlanModeHooksCore",
    execution_id: str,
    result: Any,
    row_count: Optional[int] = None,
    execution_time: Optional[float] = None,
):
    """Emit SQL execution result event.

    Args:
        hooks: PlanModeHooks instance
        execution_id: Unique execution ID
        result: Query result
        row_count: Optional row count
        execution_time: Optional execution time in seconds
    """
    try:
        event = SqlExecutionResultEvent(
            execution_id=execution_id,
            result=result,
            row_count=row_count,
            execution_time=execution_time,
            timestamp=time.time(),
        )

        if hooks.emit_queue:
            await hooks.emit_queue.put(ChatEvent(type="sql_execution_result", data=event.model_dump()))

        logger.debug(f"Emitted SQL execution result event: {execution_id}")

    except Exception as e:
        logger.error(f"Failed to emit SQL execution result event: {e}")


async def emit_sql_execution_error(
    hooks: "PlanModeHooksCore",
    execution_id: str,
    error_message: str,
    error_code: Optional[str] = None,
):
    """Emit SQL execution error event.

    Args:
        hooks: PlanModeHooks instance
        execution_id: Unique execution ID
        error_message: Error message
        error_code: Optional error code
    """
    try:
        event = SqlExecutionErrorEvent(
            execution_id=execution_id,
            error_message=error_message,
            error_code=error_code,
            timestamp=time.time(),
        )

        if hooks.emit_queue:
            await hooks.emit_queue.put(ChatEvent(type="sql_execution_error", data=event.model_dump()))

        logger.debug(f"Emitted SQL execution error event: {execution_id}")

    except Exception as e:
        logger.error(f"Failed to emit SQL execution error event: {e}")


# Note: The complete implementation would extract these methods from PlanModeHooks:
# - _emit_action (lines ~1931-1942)
# - _emit_plan_update_event (lines ~5100-5150)
# - _emit_tool_error_event (lines ~5150-5220)
# - _emit_status_message (lines ~5220-5270)
# - SQL execution event helpers
# - Deep research event helpers
# - Progress update helpers
