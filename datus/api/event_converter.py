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

    def _extract_plan_from_output(self, output: Any) -> Dict[str, Any]:
        """
        Try to find plan-related fields ('todo_list' or 'updated_item') inside
        a possibly nested output structure. Handles dicts and JSON strings and
        returns the first matching dict with keys 'todo_list' or 'updated_item',
        or {} if none found.
        """
        def try_parse(obj):
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, str):
                try:
                    return json.loads(obj)
                except Exception:
                    return None
            return None

        stack = [output]
        visited = set()
        max_depth = 6
        depth = 0
        while stack and depth < max_depth:
            depth += 1
            current = stack.pop()
            if current is None:
                continue
            # avoid looping on same object (use id where possible)
            try:
                cid = id(current)
            except Exception:
                cid = None
            if cid and cid in visited:
                continue
            if cid:
                visited.add(cid)

            parsed = try_parse(current)
            if isinstance(parsed, dict):
                # direct matches
                if "todo_list" in parsed and isinstance(parsed["todo_list"], dict):
                    return {"todo_list": parsed["todo_list"]}
                if "updated_item" in parsed and isinstance(parsed["updated_item"], dict):
                    return {"updated_item": parsed["updated_item"]}

                # common wrapper keys
                for k in ("raw_output", "result", "data", "output"):
                    if k in parsed:
                        child = try_parse(parsed[k])
                        if isinstance(child, dict):
                            stack.append(child)

                # push dict values for further traversal
                for v in parsed.values():
                    child = try_parse(v)
                    if isinstance(child, dict):
                        stack.append(child)

        return {}

    def _extract_callid_from_output(self, output: Any) -> Optional[str]:
        """
        Search nested output for common call id fields (action_id, call_id, callId, tool_call_id, toolCallId).
        Returns the first found string value or None.
        """
        def try_parse(obj):
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, str):
                try:
                    return json.loads(obj)
                except Exception:
                    return None
            return None

        keys_to_check = ("action_id", "call_id", "callId", "tool_call_id", "toolCallId", "id")
        stack = [output]
        visited = set()
        max_depth = 6
        depth = 0
        while stack and depth < max_depth:
            depth += 1
            current = stack.pop()
            if current is None:
                continue
            try:
                cid = id(current)
            except Exception:
                cid = None
            if cid and cid in visited:
                continue
            if cid:
                visited.add(cid)

            parsed = try_parse(current)
            if isinstance(parsed, dict):
                # check keys
                for k in keys_to_check:
                    if k in parsed:
                        v = parsed.get(k)
                        if isinstance(v, str) and v.strip():
                            return v
                # push wrapper keys and values
                for k in ("raw_output", "result", "data", "output"):
                    if k in parsed:
                        child = try_parse(parsed[k])
                        if isinstance(child, dict):
                            stack.append(child)
                for v in parsed.values():
                    child = try_parse(v)
                    if isinstance(child, dict):
                        stack.append(child)

        return None

    def _find_tool_call_id(self, action: ActionHistory) -> Optional[str]:
        """
        Try to determine the tool_call_id for a given action by:
          1) mapping action.action_id -> tool_call_id
          2) checking common id keys in action.input and mapping if present
          3) extracting call id from nested output
        Returns the tool_call_id string if found, else None.
        """
        # direct mapping by action_id
        if action.action_id in self.tool_call_map:
            return self.tool_call_map[action.action_id]

        # check input for possible action/call id that maps to stored call ids
        if action.input and isinstance(action.input, dict):
            for k in ("action_id", "call_id", "callId", "tool_call_id", "toolCallId", "id"):
                v = action.input.get(k)
                if isinstance(v, str) and v:
                    if v in self.tool_call_map:
                        return self.tool_call_map[v]
        # try to extract from nested output
        if action.output:
            candidate = self._extract_callid_from_output(action.output)
            if candidate:
                # if candidate is already a mapped key, return mapped value; else return candidate as-is
                if candidate in self.tool_call_map:
                    return self.tool_call_map[candidate]
                return candidate

        return None

    def convert_action_to_event(self, action: ActionHistory, seq_num: int) -> List[DeepResearchEvent]:
        """Convert ActionHistory to DeepResearchEvent list."""
        
        timestamp = int(action.start_time.timestamp() * 1000)
        event_id = f"{action.action_id}_{seq_num}"
        events = []
        
        # 1. Handle chat/assistant messages
        if action.role == ActionRole.ASSISTANT:
            if action.action_type in ["llm_generation", "message", "thinking", "raw_stream", "response"]:
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

        # 2. Handle tool calls - ToolCallEvent / ToolCallResultEvent and PlanUpdateEvent for plan tools
        elif action.role == ActionRole.TOOL:
            tool_call_id = str(uuid.uuid4())
            # store mapping for later result binding
            self.tool_call_map[action.action_id] = tool_call_id

            is_plan_tool = action.action_type in ["todo_write", "todo_update"]
            plan_data = {}
            if action.output:
                plan_data = self._extract_plan_from_output(action.output)

            # If this is a plan tool and we found plan data, emit PlanUpdateEvent first
            if is_plan_tool and plan_data:
                todos = []
                if "todo_list" in plan_data:
                    tlist = plan_data["todo_list"]
                    if isinstance(tlist, dict) and "items" in tlist:
                        for todo_data in tlist["items"]:
                            if isinstance(todo_data, dict):
                                todos.append(TodoItem(
                                    id=todo_data.get("id", str(uuid.uuid4())),
                                    content=todo_data.get("content", ""),
                                    status=TodoStatus(todo_data.get("status", "pending"))
                                ))
                elif "updated_item" in plan_data:
                    ui = plan_data["updated_item"]
                    if isinstance(ui, dict):
                        todos.append(TodoItem(
                            id=ui.get("id", str(uuid.uuid4())),
                            content=ui.get("content", ""),
                            status=TodoStatus(ui.get("status", "pending"))
                        ))

                if todos:
                    # Emit a ToolCallEvent for visibility (call start)
                    events.append(ToolCallEvent(
                        id=f"{event_id}_call",
                        planId=self.plan_id,
                        timestamp=timestamp,
                        toolCallId=tool_call_id,
                        toolName=action.action_type,
                        input=action.input if isinstance(action.input, dict) else {}
                    ))

                    # Emit ToolCallResultEvent with the raw output
                    if action.output:
                        events.append(ToolCallResultEvent(
                            id=f"{event_id}_result",
                            planId=self.plan_id,
                            timestamp=timestamp,
                            toolCallId=tool_call_id,
                            data=action.output,
                            error=action.status == ActionStatus.FAILED
                        ))

                    # Finally emit the PlanUpdateEvent to update plan UI state
                    events.append(PlanUpdateEvent(
                        id=f"{event_id}_plan",
                        planId=self.plan_id,
                        timestamp=timestamp,
                        todos=todos
                    ))
            else:
                # Normal tool call: emit call + result (if available)
                events.append(ToolCallEvent(
                    id=f"{event_id}_call",
                    planId=self.plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    toolName=action.action_type,
                    input=action.input if isinstance(action.input, dict) else {}
                ))
                if action.output:
                    events.append(ToolCallResultEvent(
                        id=f"{event_id}_result",
                        planId=self.plan_id,
                        timestamp=timestamp,
                        toolCallId=tool_call_id,
                        data=action.output,
                        error=action.status == ActionStatus.FAILED
                    ))

        # 3. Handle tool results (legacy support)
        elif action.action_type == "tool_call_result" and action.output:
            # Try to find a matching tool_call_id robustly
            tool_call_id = self._find_tool_call_id(action)

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
