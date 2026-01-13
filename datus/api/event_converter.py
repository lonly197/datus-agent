# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Event converter for mapping ActionHistory to DeepResearchEvent format.
"""

import asyncio
import hashlib
import json
import time
import uuid
from collections import deque
from typing import Any, AsyncGenerator, Dict, List, Optional

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

from .models import (
    ChatEvent,
    CompleteEvent,
    DeepResearchEvent,
    ErrorEvent,
    PlanUpdateEvent,
    ReportEvent,
    TodoItem,
    TodoStatus,
    ToolCallEvent,
    ToolCallResultEvent,
)


class DeepResearchEventConverter:
    """Converts ActionHistory events to DeepResearchEvent format."""

    # Define virtual steps for non-agentic workflows (Text2SQL)
    VIRTUAL_STEPS = [
        {"id": "step_intent", "content": "Analyze Query Intent", "node_types": ["intent_analysis"]},
        {
            "id": "step_schema",
            "content": "Discover Database Schema",
            "node_types": ["schema_discovery", "schema_linking", "schema_validation"],
        },
        {"id": "step_sql", "content": "Generate SQL Query", "node_types": ["generate_sql"]},
        {
            "id": "step_exec",
            "content": "Execute SQL & Verify",
            "node_types": ["execute_sql", "result_validation"],
        },
        {"id": "step_reflect", "content": "Self-Correction & Optimization", "node_types": ["reflect", "output"]},
    ]

    def __init__(self):
        self.plan_id = str(uuid.uuid4())
        self.tool_call_map: Dict[str, str] = {}  # action_id -> tool_call_id
        self.logger = get_logger(__name__)
        # small rolling cache to deduplicate near-identical assistant messages
        self._recent_assistant_hashes: "deque[str]" = deque(maxlen=50)

        # State for virtual plan management
        self.virtual_plan_emitted = False
        self.active_virtual_step_id = None
        self.completed_virtual_steps = set()

    def _get_virtual_step_id(self, node_type: str) -> Optional[str]:
        """Map node type to virtual step ID."""
        for step in self.VIRTUAL_STEPS:
            if node_type in step["node_types"]:
                return str(step["id"])
        return None

    def _generate_virtual_plan_update(self, current_node_type: Optional[str] = None) -> Optional[PlanUpdateEvent]:
        """Generate PlanUpdateEvent based on current progress."""
        todos = []
        current_step_id = self._get_virtual_step_id(current_node_type) if current_node_type else None

        # If we found a new active step, update our state
        if current_step_id:
            self.active_virtual_step_id = current_step_id

        # Determine the index of the current active step
        active_index = -1
        if self.active_virtual_step_id:
            for i, step in enumerate(self.VIRTUAL_STEPS):
                if str(step["id"]) == self.active_virtual_step_id:
                    active_index = i
                    break

        for i, step in enumerate(self.VIRTUAL_STEPS):
            status = TodoStatus.PENDING

            # Robust state machine logic based on linear order
            if active_index != -1:
                if i < active_index:
                    status = TodoStatus.COMPLETED
                elif i == active_index:
                    status = TodoStatus.IN_PROGRESS
                else:
                    status = TodoStatus.PENDING
            else:
                # If no active step yet (initialization), check completed set or default to pending
                if step["id"] in self.completed_virtual_steps:
                    status = TodoStatus.COMPLETED

            todos.append(TodoItem(id=str(step["id"]), content=str(step["content"]), status=status))

        return PlanUpdateEvent(id=str(uuid.uuid4()), planId=None, timestamp=int(time.time() * 1000), todos=todos)

    def _hash_text(self, s: str) -> str:
        try:
            return hashlib.sha1(s.strip().encode("utf-8")).hexdigest()
        except Exception:
            return ""

    def _generate_sql_summary(self, sql: str, result: str, row_count: int) -> str:
        """Generate a markdown summary report for SQL execution results.

        Args:
            sql: The SQL query that was executed
            result: The CSV result string from SQL execution
            row_count: Number of rows returned

        Returns:
            Markdown formatted summary report
        """
        lines = []

        # Header
        lines.append("## 📊 SQL执行结果摘要\n")

        # SQL overview
        lines.append("### SQL查询")
        lines.append(f"- **行数**: {row_count}")
        lines.append("- **状态**: ✅ 执行成功\n")

        # Result preview (first 5 rows if available)
        if result and result.strip():
            lines.append("### 结果预览")
            try:
                import pandas as pd
                from io import StringIO

                df = pd.read_csv(StringIO(result))
                preview = df.head(5).to_markdown(index=False)
                lines.append(preview)

                if len(df) > 5:
                    lines.append(f"\n*...还有 {len(df) - 5} 行数据*\n")
            except Exception:
                # If parsing fails, show raw result preview
                result_lines = result.strip().split("\n")[:6]
                lines.append("```")
                lines.extend(result_lines)
                lines.append("```")
                if len(result.strip().split("\n")) > 6:
                    lines.append("*...更多数据*\n")

        return "\n".join(lines)

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

    def _try_parse_json_like(self, obj: Any) -> Optional[Dict[str, Any]]:
        """
        Try to parse an object that may be a dict or a JSON string into a dict.
        Returns the dict if successful, otherwise None.
        """
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _extract_todo_id_from_action(self, action: ActionHistory) -> Optional[str]:
        """
        Extracts the todo_id from an ActionHistory object, looking in various places.
        This is crucial for correctly setting the planId for events related to specific todos.
        """
        # Try to extract from input arguments first
        if action.input and isinstance(action.input, dict):
            # Check for 'plan_id' directly in input (for preflight tools created by PreflightOrchestrator)
            if action.input.get("plan_id"):
                return action.input["plan_id"]
            # Check for 'todo_id' directly in input
            if action.input.get("todo_id"):
                return action.input["todo_id"]
            # Check for 'plan_id' in nested "input" field (for execution_event_manager records)
            if "input" in action.input and isinstance(action.input["input"], dict):
                nested_input = action.input["input"]
                if nested_input.get("plan_id"):
                    return nested_input["plan_id"]
                if nested_input.get("todo_id"):
                    return nested_input["todo_id"]
            # Check for 'arguments' field which might be a JSON string containing 'todo_id' or 'plan_id'
            if "arguments" in action.input and isinstance(action.input["arguments"], str):
                parsed_args = self._try_parse_json_like(action.input["arguments"])
                if isinstance(parsed_args, dict):
                    if parsed_args.get("plan_id"):
                        return parsed_args["plan_id"]
                    if parsed_args.get("todo_id"):
                        return parsed_args["todo_id"]

        # For plan_update actions, try to extract from output's 'updated_item' or 'todo_list'
        if action.action_type == "plan_update" and action.output:
            plan_data = self._extract_plan_from_output(action.output)
            if "updated_item" in plan_data and isinstance(plan_data["updated_item"], dict):
                return plan_data["updated_item"].get("id")
            if "todo_list" in plan_data and isinstance(plan_data["todo_list"], dict):
                # If it's a full todo_list, we might not have a single todo_id,
                # but if there's only one item, we can use its ID.
                items = plan_data["todo_list"].get("items")
                if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
                    return items[0].get("id")

        # For tool_call_result, try to find the original tool_call_id and then its todo_id
        if action.action_type == "tool_call_result" and action.input and isinstance(action.input, dict):
            original_action_id = action.input.get("action_id")
            if original_action_id and original_action_id in self.tool_call_map:
                # This is complex as we don't store todo_id in tool_call_map directly.
                # This would require looking up the original ActionHistory for the tool call.
                # For now, we rely on todo_id being in the tool call's input.
                pass

        return None

    def _find_tool_call_id(self, action: ActionHistory) -> Optional[str]:
        """
        Try to determine the tool_call_id for a given action by:
          1) mapping action.action_id -> tool_call_id
          2) checking common id keys in action.input (including parsing stringified 'arguments')
          3) extracting call id from nested output
        Returns the tool_call_id string if found, else None.
        """
        # direct mapping by action_id
        if action.action_id in self.tool_call_map:
            return self.tool_call_map[action.action_id]

        # Normalize and inspect input for candidate ids
        input_candidate = None
        if action.input:
            # If it's a dict, copy and try to parse common string fields like 'arguments'
            if isinstance(action.input, dict):
                input_candidate = dict(action.input)
                # parse 'arguments' if it's a json string
                if "arguments" in input_candidate and isinstance(input_candidate["arguments"], str):
                    parsed_args = self._try_parse_json_like(input_candidate["arguments"])
                    if isinstance(parsed_args, dict):
                        input_candidate.update(parsed_args)
            else:
                # try to parse string input
                parsed = self._try_parse_json_like(action.input)
                if isinstance(parsed, dict):
                    input_candidate = parsed

        if input_candidate and isinstance(input_candidate, dict):
            for k in ("tool_call_id", "toolCallId", "call_id", "callId", "action_id", "id", "todo_id", "todoId"):
                v = input_candidate.get(k)
                if isinstance(v, str) and v:
                    # if value maps to our stored map, return mapped id
                    if v in self.tool_call_map:
                        return self.tool_call_map[v]
                    # otherwise, return the candidate (best-effort)
                    return v

        # try to extract from nested output
        if action.output:
            candidate = self._extract_callid_from_output(action.output)
            if candidate:
                if candidate in self.tool_call_map:
                    return self.tool_call_map[candidate]
                return candidate

        return None

    def _is_internal_todo_update(self, action: ActionHistory) -> bool:
        """
        Check if this is an internal todo_update call from server executor that should be filtered.

        Args:
            action: The ActionHistory to check

        Returns:
            bool: True if this is an internal todo_update that should be filtered
        """
        # Internal todo_update calls from the server executor often have specific action_id patterns
        # or messages. We want to filter these out as ToolCallEvents.
        if action.action_type == "todo_update":
            # Check if the action_id starts with "server_call_"
            if action.action_id.startswith("server_call_"):
                return True
            # Additionally, check if the message indicates it's an internal update
            if action.messages and (
                "Server executor: starting todo" in action.messages
                or "Server executor: todo_in_progress" in action.messages
                or "Server executor: todo_completed" in action.messages
                or "Server executor: todo_complete failed" in action.messages
            ):
                return True
        return False

    def validate_event_flow(self, action_type: str, events: List[DeepResearchEvent]) -> bool:
        """Validate event flow completeness for critical actions.

        Args:
            action_type: The type of action being converted
            events: The generated events list

        Returns:
            bool: True if event flow is valid, False otherwise
        """
        if action_type in ["schema_discovery", "sql_execution"]:
            # Should have both ToolCallEvent and ToolCallResultEvent
            has_call = any(e.event == "tool_call" for e in events)
            has_result = any(e.event == "tool_call_result" for e in events)
            if not (has_call and has_result):
                self.logger.warning(f"Action {action_type} missing tool call/result events")
                return False
        return True

    def convert_action_to_event(self, action: ActionHistory, seq_num: int) -> List[DeepResearchEvent]:
        """Convert ActionHistory to DeepResearchEvent list."""

        timestamp = int(time.time() * 1000)
        event_id = f"{action.action_id}_{seq_num}"
        events: List[DeepResearchEvent] = []

        # Debug logging: track action conversion
        self.logger.debug(f"Converting action: {action.action_type}, role: {action.role}, status: {action.status}")

        # Extract todo_id if present (for plan-related events)
        todo_id = self._extract_todo_id_from_action(action)

        # Fallback to virtual plan ID if no explicit todo_id (for Text2SQL workflow)
        if not todo_id and self.active_virtual_step_id:
            todo_id = self.active_virtual_step_id

        # 1. Handle chat/assistant messages
        if action.role == ActionRole.ASSISTANT:
            # ChatEvent should only have planId when directly related to a specific todo item execution
            # For general assistant messages (thinking, planning, etc.), planId should be None
            chat_plan_id = todo_id if todo_id else None

            # Emit streaming token chunks ("raw_stream") always.
            # Emit intermediate "message" or "thinking" actions only when explicitly flagged
            # by the producer via an `emit_chat` boolean in action.output or action.input.
            emit_flag = False
            if action.output and isinstance(action.output, dict):
                emit_flag = bool(action.output.get("emit_chat"))
            if not emit_flag and action.input and isinstance(action.input, dict):
                emit_flag = bool(action.input.get("emit_chat"))

            # Only allow raw_stream unconditionally; allow message/thinking when flagged.
            if action.action_type == "raw_stream" or (action.action_type in ("message", "thinking") and emit_flag):
                content = ""
                if action.output and isinstance(action.output, dict):
                    content = (
                        action.output.get("content", "")
                        or action.output.get("response", "")
                        or action.output.get("raw_output", "")
                        or action.messages
                    )
                # Only send chat events if they have actual content or are important messages
                if content and content.strip():
                    # dedupe near-identical assistant messages
                    h = self._hash_text(content)
                    if h and h in self._recent_assistant_hashes:
                        # skip duplicate message
                        return []
                    if h:
                        self._recent_assistant_hashes.append(h)
                    events.append(ChatEvent(id=event_id, planId=chat_plan_id, timestamp=timestamp, content=content))

            elif action.action_type == "chat_response":
                content = ""
                if action.output and isinstance(action.output, dict):
                    content = action.output.get("response", "") or action.output.get("content", "")
                # Always send chat_response events as they are final responses
                if content or action.output:
                    h = self._hash_text(content or str(action.output))
                    if h and h in self._recent_assistant_hashes:
                        return []
                    if h:
                        self._recent_assistant_hashes.append(h)
                    events.append(ChatEvent(id=event_id, planId=chat_plan_id, timestamp=timestamp, content=content))

        # Handle SQL generation events
        elif action.action_type == "sql_generation" and action.status == ActionStatus.SUCCESS:
            sql_content = ""
            if action.output and isinstance(action.output, dict):
                sql = action.output.get("sql_query", "")
                if sql:
                    # Wrap as Markdown SQL code block
                    sql_content = f"```sql\n{sql}\n```"

            if sql_content:
                # SQL generation events usually don't have a specific planId unless tied to a todo
                sql_plan_id = todo_id if todo_id else None
                events.append(
                    ChatEvent(
                        id=event_id,
                        planId=sql_plan_id,
                        timestamp=timestamp,
                        content=sql_content,
                    )
                )

        # Handle Intent Analysis (convert to ChatEvent for visibility)
        elif action.action_type == "intent_analysis" and action.status == ActionStatus.SUCCESS:
            intent = "Unknown"
            confidence = 0.0
            if action.output and isinstance(action.output, dict):
                intent = action.output.get("intent", intent)
                confidence = action.output.get("confidence", confidence)

            content = f"🧐 **Intent Detected**: `{intent}` (Confidence: {confidence:.2f})"
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=todo_id,
                    timestamp=timestamp,
                    content=content,
                )
            )

        # Handle Schema Discovery (convert to ToolCallEvent)
        elif action.action_type == "schema_discovery":
            tool_call_id = str(uuid.uuid4())
            # Ensure input is a dict
            tool_input = {}
            if action.input and isinstance(action.input, dict):
                tool_input = action.input

            # Use virtual step ID as planId (fix for Text2SQL workflow)
            schema_plan_id = todo_id if todo_id else self.active_virtual_step_id

            events.append(
                ToolCallEvent(
                    id=f"{event_id}_call",
                    planId=schema_plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    toolName="schema_discovery",
                    input=tool_input,
                )
            )

            events.append(
                ToolCallResultEvent(
                    id=f"{event_id}_result",
                    planId=schema_plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    data=action.output,
                    error=action.status == ActionStatus.FAILED,
                )
            )

        # Handle Schema Linking
        elif action.action_type == "schema_linking" and action.status == ActionStatus.SUCCESS:
            tables_found = 0
            if action.output and isinstance(action.output, dict):
                tables_found = action.output.get("tables_found", 0)

            content = f"🔗 **Schema Linking**: Linked {tables_found} tables to the query context."
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=todo_id,
                    timestamp=timestamp,
                    content=content,
                )
            )

        # Handle Knowledge Search
        elif action.action_type == "knowledge_search" and action.status == ActionStatus.SUCCESS:
            knowledge_found = False
            if action.output and isinstance(action.output, dict):
                knowledge_found = action.output.get("knowledge_found", False)

            if knowledge_found:
                content = "📚 **Knowledge Search**: Found relevant external business knowledge."
                events.append(
                    ChatEvent(
                        id=event_id,
                        planId=todo_id,
                        timestamp=timestamp,
                        content=content,
                    )
                )

        # Handle SQL Execution (convert to ToolCallEvent)
        elif action.action_type == "sql_execution":
            tool_call_id = str(uuid.uuid4())
            tool_input = {}
            if action.input and isinstance(action.input, dict):
                tool_input = action.input

            # Use virtual step ID as planId (fix for Text2SQL workflow)
            exec_plan_id = todo_id if todo_id else self.active_virtual_step_id

            events.append(
                ToolCallEvent(
                    id=f"{event_id}_call",
                    planId=exec_plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    toolName="execute_sql",
                    input=tool_input,
                )
            )

            events.append(
                ToolCallResultEvent(
                    id=f"{event_id}_result",
                    planId=exec_plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    data=action.output,
                    error=action.status == ActionStatus.FAILED,
                )
            )

        # Handle Preflight Tool Execution
        elif action.action_type.startswith("preflight_"):
            # Extract tool name from action_type (e.g., "preflight_describe_table" -> "describe_table")
            tool_name = action.action_type.replace("preflight_", "", 1)

            # Extract tool_call_id from input or generate a new one
            tool_call_id = None
            if action.input and isinstance(action.input, dict):
                # Try to find tool_call_id in the input
                for key, value in action.input.items():
                    if "tool_call" in str(key).lower() or isinstance(value, str) and "preflight_" in value:
                        tool_call_id = value
                        break

            if not tool_call_id:
                tool_call_id = str(uuid.uuid4())

            # Create ToolCallEvent for processing status
            if action.status == ActionStatus.PROCESSING:
                tool_input = {}
                if action.input and isinstance(action.input, dict):
                    tool_input = action.input

                events.append(
                    ToolCallEvent(
                        id=f"{event_id}_call",
                        planId=todo_id,  # Use plan_id from _extract_todo_id_from_action
                        timestamp=timestamp,
                        toolCallId=tool_call_id,
                        toolName=tool_name,
                        input=tool_input,
                    )
                )

            # Create ToolCallResultEvent for completed/failed status
            if action.status in [ActionStatus.SUCCESS, ActionStatus.FAILED]:
                events.append(
                    ToolCallResultEvent(
                        id=f"{event_id}_result",
                        planId=todo_id,  # Use plan_id from _extract_todo_id_from_action
                        timestamp=timestamp,
                        toolCallId=tool_call_id,
                        data=action.output,
                        error=action.status == ActionStatus.FAILED,
                    )
                )

        # Handle Reflection Analysis
        elif action.action_type == "reflection_analysis" and action.status == ActionStatus.SUCCESS:
            strategy = "UNKNOWN"
            if action.output and isinstance(action.output, dict):
                strategy = action.output.get("strategy", strategy)

            content = f"🤔 **Reflection**: Analyzing results... Strategy: `{strategy}`"
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=todo_id,
                    timestamp=timestamp,
                    content=content,
                )
            )

        # 2. Handle tool calls - ToolCallEvent / ToolCallResultEvent and PlanUpdateEvent for plan tools
        elif action.role == ActionRole.TOOL:
            # 过滤掉内部的todo_update状态管理调用
            if action.action_type == "todo_update" and self._is_internal_todo_update(action):
                return []  # 不生成任何事件

            tool_call_id = str(uuid.uuid4())
            # store mapping for later result binding
            self.tool_call_map[action.action_id] = tool_call_id

            is_plan_tool = action.action_type in ["todo_write", "todo_update"]
            plan_data = {}
            if action.output:
                plan_data = self._extract_plan_from_output(action.output)

            # Build normalized input: parse stringified 'arguments' if present so fields like todo_id are exposed.
            normalized_input: Dict[str, Any] = {}
            if action.input:
                if isinstance(action.input, dict):
                    normalized_input = dict(action.input)
                    # parse 'arguments' field if it's a JSON string
                    if "arguments" in normalized_input and isinstance(normalized_input["arguments"], str):
                        parsed_args = self._try_parse_json_like(normalized_input["arguments"])
                        if isinstance(parsed_args, dict):
                            normalized_input.update(parsed_args)
                else:
                    parsed = self._try_parse_json_like(action.input)
                    if isinstance(parsed, dict):
                        normalized_input = parsed

            # Determine planId for tool events:
            # - For plan tools: use specific todo_id if available, otherwise None for todo_write (creates entire plan)
            # - For other tools: use todo_id if present (indicates tool is executing a specific todo)
            tool_plan_id = None
            if is_plan_tool:
                if action.action_type == "todo_update" and todo_id:
                    # todo_update operates on specific todo items
                    tool_plan_id = todo_id
                # todo_write creates the entire plan, so planId should be None
            elif todo_id:
                # Non-plan tools that are executing specific todos
                tool_plan_id = todo_id

            # If this is a plan tool and we found plan data, emit PlanUpdateEvent first
            if is_plan_tool and plan_data:
                todos = []
                if "todo_list" in plan_data:
                    tlist = plan_data["todo_list"]
                    if isinstance(tlist, dict) and "items" in tlist:
                        for todo_data in tlist["items"]:
                            if isinstance(todo_data, dict):
                                todos.append(
                                    TodoItem(
                                        id=todo_data.get("id", str(uuid.uuid4())),
                                        content=todo_data.get("content", ""),
                                        status=TodoStatus(todo_data.get("status", "pending")),
                                    )
                                )
                elif "updated_item" in plan_data:
                    ui = plan_data["updated_item"]
                    if isinstance(ui, dict):
                        todos.append(
                            TodoItem(
                                id=ui.get("id", str(uuid.uuid4())),
                                content=ui.get("content", ""),
                                status=TodoStatus(ui.get("status", "pending")),
                            )
                        )

                if todos:
                    # For plan tools, emit tool events AND plan update event
                    events.append(
                        ToolCallEvent(
                            id=f"{event_id}_call",
                            planId=tool_plan_id,
                            timestamp=timestamp,
                            toolCallId=tool_call_id,
                            toolName=action.action_type,
                            input=normalized_input or (action.input if isinstance(action.input, dict) else {}),
                        )
                    )

                    if action.output:
                        events.append(
                            ToolCallResultEvent(
                                id=f"{event_id}_result",
                                planId=tool_plan_id,
                                timestamp=timestamp,
                                toolCallId=tool_call_id,
                                data=action.output,
                                error=action.status == ActionStatus.FAILED,
                            )
                        )

                    # PlanUpdateEvent uses the specific todo_id for updated items, or None for plan creation
                    plan_update_plan_id = None
                    if action.action_type == "todo_update" and "updated_item" in plan_data:
                        ui = plan_data["updated_item"]
                        if isinstance(ui, dict) and ui.get("id"):
                            plan_update_plan_id = ui["id"]

                    events.append(
                        PlanUpdateEvent(
                            id=f"{event_id}_plan", planId=plan_update_plan_id, timestamp=timestamp, todos=todos
                        )
                    )
            else:
                # Normal tool call: emit call + result (if available)
                events.append(
                    ToolCallEvent(
                        id=f"{event_id}_call",
                        planId=tool_plan_id,
                        timestamp=timestamp,
                        toolCallId=tool_call_id,
                        toolName=action.action_type,
                        input=normalized_input or (action.input if isinstance(action.input, dict) else {}),
                    )
                )
                if action.output:
                    events.append(
                        ToolCallResultEvent(
                            id=f"{event_id}_result",
                            planId=tool_plan_id,
                            timestamp=timestamp,
                            toolCallId=tool_call_id,
                            data=action.output,
                            error=action.status == ActionStatus.FAILED,
                        )
                    )

        # 3. Handle tool results (legacy support)
        elif action.action_type == "tool_call_result" and action.output:
            # Try to find a matching tool_call_id robustly
            tool_call_id = self._find_tool_call_id(action)

            if tool_call_id:
                events.append(
                    ToolCallResultEvent(
                        id=event_id,
                        planId=self.plan_id,
                        timestamp=timestamp,
                        toolCallId=str(tool_call_id),
                        data=action.output,
                        error=action.status == ActionStatus.FAILED,
                    )
                )

        # 4. Handle plan updates
        elif action.action_type == "plan_update" and action.output:
            todos = []
            if isinstance(action.output, dict):
                # Handle both "todos" (legacy) and "todo_list" (new) formats
                todo_data_source = None
                if "todo_list" in action.output and isinstance(action.output["todo_list"], dict):
                    todo_data_source = action.output["todo_list"].get("items", [])
                elif "todos" in action.output and isinstance(action.output["todos"], list):
                    todo_data_source = action.output["todos"]

                if todo_data_source:
                    for todo_data in todo_data_source:
                        if isinstance(todo_data, dict):
                            todos.append(
                                TodoItem(
                                    id=todo_data.get("id", str(uuid.uuid4())),
                                    content=todo_data.get("content", ""),
                                    status=TodoStatus(todo_data.get("status", "pending")),
                                )
                            )

            # For plan_update events, planId should be None (global plan update)
            events.append(PlanUpdateEvent(id=event_id, planId=None, timestamp=timestamp, todos=todos))

        # 5. Handle workflow completion (修复 CompleteEvent 处理)
        elif action.action_type == "workflow_completion" and action.status == ActionStatus.SUCCESS:
            # Force complete all virtual steps
            if self.virtual_plan_emitted:
                final_todos = []
                for step in self.VIRTUAL_STEPS:
                    final_todos.append(
                        TodoItem(id=str(step["id"]), content=str(step["content"]), status=TodoStatus.COMPLETED)
                    )
                events.append(
                    PlanUpdateEvent(id=f"{event_id}_plan_final", planId=None, timestamp=timestamp, todos=final_todos)
                )

            events.append(
                CompleteEvent(
                    id=event_id,
                    planId=None,  # CompleteEvent should not have planId by default
                    timestamp=timestamp,
                    content=action.messages,
                )
            )

        # 6. Handle workflow initialization
        elif action.action_type == "workflow_init":
            # Convert workflow init to a ChatEvent to inform user
            content = f"🚀 **System Initialization**: {action.messages}"
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=None,
                    timestamp=timestamp,
                    content=content,
                )
            )

            # Emit initial virtual plan for Text2SQL workflow
            # We assume if it's workflow_init, we can initialize the virtual plan
            if not self.virtual_plan_emitted:
                plan_update = self._generate_virtual_plan_update()
                if plan_update:
                    events.append(plan_update)
                    self.virtual_plan_emitted = True

        # 7. Handle node execution
        elif action.action_type == "node_execution":
            # Update virtual plan if applicable
            node_type = None
            if action.input and isinstance(action.input, dict):
                node_type = action.input.get("node_type")

            if node_type:
                plan_update = self._generate_virtual_plan_update(node_type)
                if plan_update and plan_update.todos:
                    # Check if status actually changed to avoid spamming events
                    # (Simplified: always emit for now as _generate logic handles state)
                    events.append(plan_update)

            # Convert node execution to a status update (ChatEvent with specific formatting or just text)
            # We can use it to show what the agent is doing
            node_desc = "Unknown Node"
            if action.input and isinstance(action.input, dict):
                node_desc = action.input.get("description", node_desc)

            content = f"🔄 **Executing Step**: {node_desc}"
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=self.active_virtual_step_id,  # Use active step ID
                    timestamp=timestamp,
                    content=content,
                )
            )

        # 8. Handle errors
        elif action.status == ActionStatus.FAILED:
            # ErrorEvent should use todo_id if the error is related to a specific todo
            error_plan_id = todo_id if todo_id else None

            error_msg = action.messages or "Unknown error"

            # Enrich error message with details if available
            if action.output and isinstance(action.output, dict):
                details = []
                if action.output.get("error_code"):
                    details.append(f"Code: {action.output.get('error_code')}")

                # Check for recovery suggestions
                suggestions = action.output.get("recovery_suggestions")
                if suggestions and isinstance(suggestions, list):
                    suggestions_str = "\n".join([f"- {s}" for s in suggestions])
                    details.append(f"Suggestions:\n{suggestions_str}")

                if details:
                    error_msg += "\n\n" + "\n".join(details)

            events.append(ErrorEvent(id=event_id, planId=error_plan_id, timestamp=timestamp, error=error_msg))

        # 9. Handle report generation and SQL output
        elif action.action_type == "output_generation" and action.output:
            if isinstance(action.output, dict):
                # First, handle SQL output via ChatEvent (for text2sql workflow)
                sql_query = action.output.get("sql_query", "")
                sql_result = action.output.get("sql_result", "")
                sql_query_final = action.output.get("sql_query_final", "")
                sql_result_final = action.output.get("sql_result_final", "")
                row_count = action.output.get("row_count", 0)
                success = action.output.get("success", True)

                # Use final SQL if available, otherwise use generated SQL
                final_sql = sql_query_final if sql_query_final else sql_query

                # If SQL exists, send ChatEvent with SQL and summary
                if final_sql:
                    # First ChatEvent: The SQL code block
                    sql_content = f"```sql\n{final_sql}\n```"
                    events.append(
                        ChatEvent(
                            id=f"{event_id}_sql",
                            planId=todo_id if todo_id else None,
                            timestamp=timestamp,
                            content=sql_content,
                        )
                    )

                    # Second ChatEvent: Summary report (if execution succeeded)
                    if success and (sql_result or sql_result_final):
                        final_result = sql_result_final if sql_result_final else sql_result
                        summary = self._generate_sql_summary(final_sql, final_result, row_count)
                        events.append(
                            ChatEvent(
                                id=f"{event_id}_summary",
                                planId=todo_id if todo_id else None,
                                timestamp=timestamp,
                                content=summary,
                            )
                        )

                # Original ReportEvent handling (for HTML reports)
                report_url = action.output.get("report_url", "")
                report_data = action.output.get("html_content", "")
                # Create ReportEvent if we have either url or data
                if report_url or report_data:
                    # ReportEvent should use todo_id if related to a specific todo
                    report_plan_id = todo_id if todo_id else None
                    events.append(
                        ReportEvent(
                            id=event_id, planId=report_plan_id, timestamp=timestamp, url=report_url, data=report_data
                        )
                    )

        # Debug logging: track generated events
        for event in events:
            self.logger.debug(f"Generated event: {event.event}, planId: {event.planId}, id: {event.id}")

        # Validate event flow for critical actions
        self.validate_event_flow(action.action_type, events)

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
                    current_task = asyncio.current_task()
                    if current_task and current_task.cancelled():
                        break
        except asyncio.CancelledError:
            self.logger.info("Event conversion stream was cancelled")
            raise
