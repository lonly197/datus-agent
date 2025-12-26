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

    def convert_action_to_event(self, action: ActionHistory, seq_num: int) -> List[DeepResearchEvent]:
        """Convert ActionHistory to DeepResearchEvent list."""
        
        timestamp = int(action.start_time.timestamp() * 1000)
        event_id = f"{action.action_id}_{seq_num}"
        events = []
        
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
                events.append(ChatEvent(
                    id=event_id,
                    planId=self.plan_id,
                    timestamp=timestamp,
                    content=content
                ))

            elif action.action_type == "chat_response":
                content = ""
                if action.output and isinstance(action.output, dict):
                    content = action.output.get("response", "") or action.output.get("content", "")
                events.append(ChatEvent(
                    id=event_id,
                    planId=self.plan_id,
                    timestamp=timestamp,
                    content=content
                ))

        # 2. Handle tool calls - 生成 ToolCallEvent 和 ToolCallResultEvent
        elif action.role == ActionRole.TOOL:
            tool_call_id = str(uuid.uuid4())
            self.tool_call_map[action.action_id] = tool_call_id
            
            # 生成 ToolCallEvent
            events.append(ToolCallEvent(
                id=f"{event_id}_call",
                planId=self.plan_id,
                timestamp=timestamp,
                toolCallId=tool_call_id,
                toolName=action.action_type,
                input=action.input if isinstance(action.input, dict) else {}
            ))
            
            # 如果有输出，同时生成 ToolCallResultEvent
            if action.output:
                events.append(ToolCallResultEvent(
                    id=f"{event_id}_result",
                    planId=self.plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    data=action.output,
                    error=action.status == ActionStatus.FAILED
                ))
            
            # 如果是计划工具，同时生成 PlanUpdateEvent
            if action.action_type in ["todo_write", "todo_update"] and action.output:
                todos = []
                if isinstance(action.output, dict):
                    # 从 todo_write 输出中提取计划信息
                    if "todo_list" in action.output:
                        todo_list_data = action.output["todo_list"]
                        if isinstance(todo_list_data, dict) and "items" in todo_list_data:
                            for todo_data in todo_list_data["items"]:
                                if isinstance(todo_data, dict):
                                    todos.append(TodoItem(
                                        id=todo_data.get("id", str(uuid.uuid4())),
                                        content=todo_data.get("content", ""),
                                        status=TodoStatus(todo_data.get("status", "pending"))
                                    ))
                    
                    # 从 todo_update 输出中提取完整计划信息
                    elif "todo_list" in action.output:
                        todo_list_data = action.output["todo_list"]
                        if isinstance(todo_list_data, dict) and "items" in todo_list_data:
                            for todo_data in todo_list_data["items"]:
                                if isinstance(todo_data, dict):
                                    todos.append(TodoItem(
                                        id=todo_data.get("id", str(uuid.uuid4())),
                                        content=todo_data.get("content", ""),
                                        status=TodoStatus(todo_data.get("status", "pending"))
                                    ))

                if todos:
                    events.append(PlanUpdateEvent(
                        id=f"{event_id}_plan",
                        planId=self.plan_id,
                        timestamp=timestamp,
                        todos=todos
                    ))

        # 3. Handle tool results (legacy support)
        elif action.action_type == "tool_call_result" and action.output:
            tool_call_id = None
            if action.action_id in self.tool_call_map:
                tool_call_id = self.tool_call_map[action.action_id]
            elif action.input and isinstance(action.input, dict):
                input_action_id = action.input.get("action_id")
                if input_action_id and input_action_id in self.tool_call_map:
                    tool_call_id = self.tool_call_map[input_action_id]

            if tool_call_id:
                events.append(ToolCallResultEvent(
                    id=event_id,
                    planId=self.plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    data=action.output,
                    error=action.status == ActionStatus.FAILED
                ))

        # 4. Handle plan updates (legacy support)
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

            events.append(PlanUpdateEvent(
                id=event_id,
                planId=self.plan_id,
                timestamp=timestamp,
                todos=todos
            ))

        # 5. Handle workflow completion (修复 CompleteEvent 处理)
        elif action.action_type == "workflow_completion" and action.status == ActionStatus.SUCCESS:
            events.append(CompleteEvent(
                id=event_id,
                planId=self.plan_id,
                timestamp=timestamp,
                content=action.messages
            ))

        # 6. Handle errors
        elif action.status == ActionStatus.FAILED:
            events.append(ErrorEvent(
                id=event_id,
                planId=self.plan_id,
                timestamp=timestamp,
                error=action.messages or "Unknown error"
            ))

        # 7. Handle report generation
        elif action.action_type == "output_generation" and action.output:
            if isinstance(action.output, dict):
                report_url = action.output.get("report_url", "")
                report_data = action.output.get("html_content", "")
                if report_url:
                    events.append(ReportEvent(
                        id=event_id,
                        planId=self.plan_id,
                        timestamp=timestamp,
                        url=report_url,
                        data=report_data
                    ))

        return events

    async def convert_stream_to_events(
        self, action_stream: AsyncGenerator[ActionHistory, None]
    ) -> AsyncGenerator[str, None]:
        """Convert ActionHistory stream to DeepResearchEvent SSE stream."""

        seq_num = 0

        try:
            async for action in action_stream:
                seq_num += 1
                events = self.convert_action_to_event(action, seq_num)

                for event in events:
                    # Convert to JSON and yield as SSE data
                    event_json = event.model_dump_json()
                    yield f"data: {event_json}\n\n"

                    # 检查是否被取消
                    if asyncio.current_task().cancelled():
                        break
        except asyncio.CancelledError:
            logger.info("Event conversion stream was cancelled")
            raise
