# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Event converter for mapping ActionHistory to DeepResearchEvent format.
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator, Dict, List, Any, Optional
from datetime import datetime

from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from .models import (
    DeepResearchEvent, DeepResearchEventType, BaseEvent,
    ChatEvent, PlanUpdateEvent, ToolCallEvent, ToolCallResultEvent,
    CompleteEvent, ReportEvent, ErrorEvent, TodoItem, TodoStatus
)


class DeepResearchEventConverter:
    """Converts ActionHistory events to DeepResearchEvent format."""

    def __init__(self):
        self.plan_id = str(uuid.uuid4())
        self.tool_call_map: Dict[str, str] = {}  # action_id -> tool_call_id

    def convert_action_to_event(self, action: ActionHistory, seq_num: int) -> Optional[DeepResearchEvent]:
        """Convert ActionHistory to DeepResearchEvent."""

        timestamp = int(action.start_time.timestamp() * 1000)
        event_id = f"{action.action_id}_{seq_num}"

        # 1. Handle chat/assistant messages
        if action.role == ActionRole.ASSISTANT:
            if action.action_type in ["llm_generation", "message", "thinking", "raw_stream"]:
                content = ""
                if action.output and isinstance(action.output, dict):
                    content = (
                        action.output.get("content", "") or
                        action.output.get("response", "") or
                        action.output.get("raw_output", "") or
                        action.messages
                    )
                return ChatEvent(
                    id=event_id,
                    planId=self.plan_id,
                    timestamp=timestamp,
                    content=content
                )

            elif action.action_type == "chat_response":
                # Final chat response
                content = ""
                if action.output and isinstance(action.output, dict):
                    content = action.output.get("response", "") or action.output.get("content", "")
                return ChatEvent(
                    id=event_id,
                    planId=self.plan_id,
                    timestamp=timestamp,
                    content=content
                )

        # 2. Handle tool calls
        elif action.role == ActionRole.TOOL:
            tool_call_id = str(uuid.uuid4())
            self.tool_call_map[action.action_id] = tool_call_id

            return ToolCallEvent(
                id=event_id,
                planId=self.plan_id,
                timestamp=timestamp,
                toolCallId=tool_call_id,
                toolName=action.action_type,
                input=action.input if isinstance(action.input, dict) else {}
            )

        # 3. Handle tool results (when tool call completes)
        elif action.action_type == "tool_call_result" and action.output:
            # Find corresponding tool call using exact matching
            tool_call_id = None
            # First try to find by exact action_id match
            if action.action_id in self.tool_call_map:
                tool_call_id = self.tool_call_map[action.action_id]
            # If not found, try to find by action_id in input
            elif action.input and isinstance(action.input, dict):
                input_action_id = action.input.get("action_id")
                if input_action_id and input_action_id in self.tool_call_map:
                    tool_call_id = self.tool_call_map[input_action_id]

            if tool_call_id:
                return ToolCallResultEvent(
                    id=event_id,
                    planId=self.plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    data=action.output,
                    error=action.status == ActionStatus.FAILED
                )

        # 4. Handle plan updates (for plan mode)
        elif action.action_type == "plan_update" and action.output:
            todos = []
            if isinstance(action.output, dict) and "todos" in action.output:
                for todo_data in action.output["todos"]:
                    if isinstance(todo_data, dict):
                        todos.append(TodoItem(
                            id=todo_data.get("id", str(uuid.uuid4())),
                            content=todo_data.get("content", ""),
                            status=TodoStatus(todo_data.get("status", "pending"))
                        ))

            return PlanUpdateEvent(
                id=event_id,
                planId=self.plan_id,
                timestamp=timestamp,
                todos=todos
            )

        # 5. Handle completion
        elif action.action_type == "workflow_completion" and action.status == ActionStatus.SUCCESS:
            return CompleteEvent(
                id=event_id,
                planId=self.plan_id,
                timestamp=timestamp,
                content=action.messages
            )

        # 6. Handle errors
        elif action.status == ActionStatus.FAILED:
            return ErrorEvent(
                id=event_id,
                planId=self.plan_id,
                timestamp=timestamp,
                error=action.messages or "Unknown error"
            )

        # 7. Handle report generation
        elif action.action_type == "output_generation" and action.output:
            if isinstance(action.output, dict):
                report_url = action.output.get("report_url", "")
                report_data = action.output.get("html_content", "")
                if report_url:
                    return ReportEvent(
                        id=event_id,
                        planId=self.plan_id,
                        timestamp=timestamp,
                        url=report_url,
                        data=report_data
                    )

        return None

    async def convert_stream_to_events(
        self, action_stream: AsyncGenerator[ActionHistory, None]
    ) -> AsyncGenerator[str, None]:
        """Convert ActionHistory stream to DeepResearchEvent SSE stream."""

        seq_num = 0

        try:
            async for action in action_stream:
                seq_num += 1
                event = self.convert_action_to_event(action, seq_num)

                if event:
                    # Convert to JSON and yield as SSE data
                    event_json = event.model_dump_json()
                    yield f"data: {event_json}\n\n"

                    # 检查是否被取消
                    if asyncio.current_task().cancelled():
                        break
        except asyncio.CancelledError:
            logger.info("Event conversion stream was cancelled")
            raise
