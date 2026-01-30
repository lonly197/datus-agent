# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Core event converter class.

This module provides the main DeepResearchEventConverter class that
orchestrates event conversion from ActionHistory to DeepResearchEvent format.
"""

import time
import uuid
from collections import deque
from typing import Any, Dict, List

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger
from datus.utils.plan_id import PlanIdManager

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
from .event_validation import (
    extract_todo_id_from_action,
    find_tool_call_id,
    get_unified_plan_id,
    is_internal_todo_update,
    extract_node_type_from_action,
    validate_event_flow,
    generate_event_id,
)
from .normalization import (
    normalize_node_type,
    normalize_tool_name,
    normalize_todo_status,
    hash_text,
)
from .sql_processing import (
    generate_sql_summary,
    format_diagnostic_report,
)
from .virtual_steps import VirtualStepManager, TodoStateManager
from .streaming import convert_stream_to_events as _convert_stream_to_events


class DeepResearchEventConverter:
    """Converts ActionHistory events to DeepResearchEvent format."""

    def __init__(self):
        """Initialize the event converter."""
        self.logger = get_logger(__name__)

        # Tool call tracking
        self.tool_call_map: Dict[str, str] = {}  # action_id -> tool_call_id

        # Deduplication cache for assistant messages
        self._recent_assistant_hashes: "deque[str]" = deque(maxlen=50)

        # Virtual plan management
        self.virtual_plan_id = PlanIdManager.new_plan_id()
        self._virtual_step_manager = VirtualStepManager(self.virtual_plan_id)
        # Backward-compatible alias (some code paths still reference this name)
        self.virtual_plan_emitted = self._virtual_step_manager.virtual_plan_emitted

        # Todo state management
        self._todo_state_manager = TodoStateManager()

        # Active todo tracking
        self.active_todo_item_id: str = None
        self.todo_item_action_map: Dict[str, str] = {}

    @property
    def virtual_plan_emitted(self) -> bool:
        return self._virtual_step_manager.virtual_plan_emitted

    @virtual_plan_emitted.setter
    def virtual_plan_emitted(self, value: bool) -> None:
        self._virtual_step_manager.virtual_plan_emitted = value

    # Virtual step management (delegates to VirtualStepManager)
    def _get_virtual_step_id(self, node_type: str) -> str:
        """Map node type to virtual step ID."""
        return self._virtual_step_manager.get_virtual_step_id(node_type)

    def _generate_virtual_plan_update(self, current_node_type: str = None) -> PlanUpdateEvent:
        """Generate PlanUpdateEvent based on current progress."""
        return self._virtual_step_manager.generate_virtual_plan_update(current_node_type)

    # Todo state management (delegates to TodoStateManager)
    def _update_todo_state(self, todos: List[TodoItem], replace_order: bool = False) -> None:
        """Update cached todo state."""
        self._todo_state_manager.update_todo_state(todos, replace_order)

    def _get_todo_state_list(self) -> List[TodoItem]:
        """Return todos in cached order."""
        return self._todo_state_manager.get_todo_state_list()

    # Normalization methods (delegates to normalization module)
    def _normalize_node_type(self, node_type: str) -> str:
        """Normalize action/node type."""
        return normalize_node_type(node_type)

    def _normalize_tool_name(self, action_type: str) -> str:
        """Normalize tool name."""
        return normalize_tool_name(action_type)

    def _normalize_todo_status(self, status: str) -> str:
        """Normalize TodoStatus strings."""
        return normalize_todo_status(status)

    def _hash_text(self, s: str) -> str:
        """Generate hash for text deduplication."""
        return hash_text(s)

    async def convert_stream_to_events(self, action_stream):
        """Backward-compatible wrapper for streaming conversion."""
        async for event in _convert_stream_to_events(action_stream, self):
            yield event

    # Event validation helpers (delegates to event_validation module)
    def _extract_plan_from_output(self, output: Any) -> Dict[str, Any]:
        """Extract plan information from action output."""
        from .event_validation import extract_plan_from_output
        return extract_plan_from_output(output)

    def _extract_callid_from_output(self, output: Any) -> str:
        """Extract call ID from action output."""
        from .event_validation import extract_callid_from_output
        return extract_callid_from_output(output)

    def _try_parse_json_like(self, obj: Any) -> Dict[str, Any]:
        """Try to parse object as JSON-like dict."""
        from .event_validation import try_parse_json_like
        return try_parse_json_like(obj)

    def _extract_todo_id_from_action(self, action: ActionHistory) -> str:
        """Extract todo ID from action metadata."""
        return extract_todo_id_from_action(action)

    def _get_unified_plan_id(self, action: ActionHistory, force_associate: bool = False) -> str:
        """Get unified plan ID for action events."""
        # 1) todo_id for agentic workflows
        todo_id = self._extract_todo_id_from_action(action)
        if todo_id:
            return todo_id

        # 2) explicit plan_id from metadata/input/output
        plan_id = get_unified_plan_id(action, False, None)
        if plan_id:
            return plan_id

        # 3) map to virtual step id for text2sql/preflight
        node_type = self._extract_node_type_from_action(action) or action.action_type
        virtual_step_id = self._get_virtual_step_id(node_type) if node_type else None
        if virtual_step_id:
            return virtual_step_id

        # 4) fallback to virtual plan when forced
        if force_associate:
            return self.virtual_plan_id
        return None

    def _find_tool_call_id(self, action: ActionHistory) -> str:
        """Find tool call ID for action."""
        return find_tool_call_id(action, self.tool_call_map)

    def _is_internal_todo_update(self, action: ActionHistory) -> bool:
        """Check if action is an internal todo update."""
        return is_internal_todo_update(action)

    def _extract_node_type_from_action(self, action: ActionHistory) -> str:
        """Extract node type from action."""
        return extract_node_type_from_action(action)

    def validate_event_flow(self, action_type: str, events: List[DeepResearchEvent]) -> bool:
        """Validate event flow for critical actions."""
        return validate_event_flow(action_type, events, self.logger)

    # SQL processing helpers (delegates to sql_processing module)
    def _generate_sql_summary(self, sql: str, result: str, row_count: int) -> str:
        """Generate SQL execution summary."""
        return generate_sql_summary(sql, result, row_count)

    def _format_diagnostic_report(self, report: Dict[str, Any]) -> str:
        """Format diagnostic report."""
        return format_diagnostic_report(report)

    def _generate_sql_generation_report(
        self,
        sql_query: str,
        sql_result: str,
        row_count: int,
        metadata: Optional[Dict[str, Any]] = None,
        table_schemas: Optional[List[Any]] = None,
    ) -> str:
        """Generate SQL report (fallback to summary for compatibility)."""
        return self._generate_sql_summary(sql_query, sql_result, row_count)

    # Note: The full convert_action_to_event method is very large (800+ lines)
    # and has been omitted here for brevity. In the original file, it contains
    # extensive logic for converting different action types to events.
    # The method delegates to many of the helper methods defined above.
    #
    # Key sections include:
    # - Failed action handling
    # - Plan update handling
    # - Chat/assistant message handling
    # - SQL generation/validation handling
    # - Schema discovery/validation handling
    # - Tool call handling
    # - Error handling
    # - Report generation
    # - Output generation
    #
    # For the complete implementation, see the original event_converter.py file.

        def convert_action_to_event(self, action: ActionHistory, seq_num: int) -> List[DeepResearchEvent]:
        """Convert ActionHistory to DeepResearchEvent list."""

        timestamp = int(time.time() * 1000)
        event_id = f"{action.action_id}_{seq_num}"
        events: List[DeepResearchEvent] = []

        # Debug logging: track action conversion
        self.logger.debug(f"Converting action: {action.action_type}, role: {action.role}, status: {action.status}")

        # Track failed virtual steps for proper ERROR status in PlanUpdateEvent
        if action.status == ActionStatus.FAILED:
            node_type = self._extract_node_type_from_action(action)
            virtual_step_id = self._get_virtual_step_id(node_type) if node_type else None

            if virtual_step_id:
                if virtual_step_id not in self._virtual_step_manager.failed_virtual_steps:
                    self._virtual_step_manager.failed_virtual_steps.add(virtual_step_id)
                    self.logger.warning(
                        f"🚨 Marking virtual step as FAILED: {virtual_step_id} (from node_type: {node_type})"
                    )

                    plan_update = self._generate_virtual_plan_update()
                    if plan_update:
                        events.append(plan_update)

        # Extract todo_id if present
        todo_id = self._extract_todo_id_from_action(action)

        # 0. Handle explicit plan updates early
        if action.action_type == "plan_update" and action.output:
            todos = []
            if isinstance(action.output, dict):
                todo_data_source = None
                if "todo_list" in action.output and isinstance(action.output["todo_list"], dict):
                    todo_data_source = action.output["todo_list"].get("items", [])
                elif "todos" in action.output and isinstance(action.output["todos"], list):
                    todo_data_source = action.output["todos"]

                if todo_data_source:
                    for todo_data in todo_data_source:
                        if isinstance(todo_data, dict):
                            todo_id = todo_data.get("id")
                            if not todo_id:
                                self.logger.warning("Skipping plan_update todo without id: %s", todo_data)
                                continue
                            todos.append(
                                TodoItem(
                                    id=str(todo_id),
                                    content=todo_data.get("content", ""),
                                    status=TodoStatus(self._normalize_todo_status(todo_data.get("status", "pending"))),
                                )
                            )

            if todos:
                self._update_todo_state(todos, replace_order=True)
                todos = self._get_todo_state_list() or todos

            plan_event_id = self.virtual_plan_id if action.role == ActionRole.WORKFLOW else event_id
            events.append(PlanUpdateEvent(id=plan_event_id, planId=None, timestamp=timestamp, todos=todos))
            return events

        # 1. Handle chat/assistant messages
        if action.role == ActionRole.ASSISTANT:
            chat_plan_id = self._get_unified_plan_id(action, force_associate=False)

            emit_flag = False
            if action.output and isinstance(action.output, dict):
                emit_flag = bool(action.output.get("emit_chat"))
            if not emit_flag and action.input and isinstance(action.input, dict):
                emit_flag = bool(action.input.get("emit_chat"))

            if action.action_type == "raw_stream" or (action.action_type in ("message", "thinking") and emit_flag):
                content = ""
                if action.output and isinstance(action.output, dict):
                    content = (
                        action.output.get("content", "")
                        or action.output.get("response", "")
                        or action.output.get("raw_output", "")
                        or action.messages
                    )
                if content and content.strip():
                    h = self._hash_text(content)
                    if h and h in self._recent_assistant_hashes:
                        return []
                    if h:
                        self._recent_assistant_hashes.append(h)
                    events.append(ChatEvent(id=event_id, planId=chat_plan_id, timestamp=timestamp, content=content))

            elif action.action_type == "chat_response":
                content = ""
                if action.output and isinstance(action.output, dict):
                    content = action.output.get("response", "") or action.output.get("content", "")
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
                    sql_content = f"```sql
{sql}
```"

            if sql_content:
                sql_plan_id = self._get_unified_plan_id(action, force_associate=True)
                events.append(
                    ChatEvent(
                        id=event_id,
                        planId=sql_plan_id,
                        timestamp=timestamp,
                        content=sql_content,
                    )
                )

        # Handle Intent Analysis
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
                    planId=self._get_unified_plan_id(action, force_associate=True),
                    timestamp=timestamp,
                    content=content,
                )
            )

        # Handle SQL Validation
        elif action.action_type == "sql_validation":
            validation_result = {}
            if action.output and isinstance(action.output, dict):
                validation_result = action.output

            is_valid = validation_result.get("is_valid", False)
            syntax_valid = validation_result.get("syntax_valid", False)
            tables_exist = validation_result.get("tables_exist", True)
            columns_exist = validation_result.get("columns_exist", True)
            has_dangerous_ops = validation_result.get("has_dangerous_ops", False)
            errors = validation_result.get("errors", [])
            warnings = validation_result.get("warnings", [])

            if is_valid:
                content_lines = [
                    "✅ **SQL验证通过**",
                    f"- 语法验证: {'✅ 通过' if syntax_valid else '❌ 失败'}",
                    f"- 表存在性: {'✅ 通过' if tables_exist else '❌ 失败'}",
                    f"- 列存在性: {'✅ 通过' if columns_exist else '❌ 失败'}",
                    f"- 危险操作: {'⚠️ 检测到' if has_dangerous_ops else '✅ 无危险操作'}",
                ]

                if warnings:
                    content_lines.append(f"
**警告** ({len(warnings)}):")
                    for warning in warnings[:3]:
                        content_lines.append(f"- {warning}")
                    if len(warnings) > 3:
                        content_lines.append(f"- ...还有 {len(warnings) - 3} 个警告")

                content = "
".join(content_lines)
            else:
                content_lines = [
                    "❌ **SQL验证失败**",
                    f"- 语法验证: {'✅ 通过' if syntax_valid else '❌ 失败'}",
                    f"- 表存在性: {'✅ 通过' if tables_exist else '❌ 失败'}",
                    f"- 列存在性: {'✅ 通过' if columns_exist else '❌ 失败'}",
                    f"- 危险操作: {'⚠️ 检测到' if has_dangerous_ops else '✅ 无危险操作'}",
                ]

                if errors:
                    content_lines.append(f"
**错误** ({len(errors)}):")
                    for error in errors[:3]:
                        content_lines.append(f"- {error}")
                    if len(errors) > 3:
                        content_lines.append(f"- ...还有 {len(errors) - 3} 个错误")

                if warnings:
                    content_lines.append(f"
**警告** ({len(warnings)}):")
                    for warning in warnings[:3]:
                        content_lines.append(f"- {warning}")
                    if len(warnings) > 3:
                        content_lines.append(f"- ...还有 {len(warnings) - 3} 个警告")

                content = "
".join(content_lines)

            events.append(
                ChatEvent(
                    id=event_id,
                    planId=self._get_unified_plan_id(action, force_associate=True),
                    timestamp=timestamp,
                    content=content,
                )
            )

        # Handle Schema Discovery
        elif action.action_type == "schema_discovery":
            tool_call_id = str(uuid.uuid4())
            tool_input = {}
            if action.input and isinstance(action.input, dict):
                tool_input = action.input

            schema_plan_id = self._get_unified_plan_id(action, force_associate=True)

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

            if action.status == ActionStatus.FAILED and action.output and isinstance(action.output, dict):
                diagnostic_report = action.output.get("diagnostic_report")
                if diagnostic_report:
                    report_content = self._format_diagnostic_report(diagnostic_report)
                    events.append(
                        ChatEvent(
                            id=f"{event_id}_diagnostic",
                            planId=schema_plan_id,
                            timestamp=timestamp,
                            content=report_content,
                        )
                    )

        # Handle Schema Validation
        elif action.action_type == "schema_validation":
            tool_call_id = str(uuid.uuid4())
            tool_input = {}
            if action.input and isinstance(action.input, dict):
                tool_input = action.input

            validation_plan_id = self._get_unified_plan_id(action, force_associate=True)

            events.append(
                ToolCallEvent(
                    id=f"{event_id}_call",
                    planId=validation_plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    toolName="schema_validation",
                    input=tool_input,
                )
            )

            events.append(
                ToolCallResultEvent(
                    id=f"{event_id}_result",
                    planId=validation_plan_id,
                    timestamp=timestamp,
                    toolCallId=tool_call_id,
                    data=action.output,
                    error=action.status == ActionStatus.FAILED,
                )
            )

            if action.status == ActionStatus.FAILED and action.output and isinstance(action.output, dict):
                diagnostic_report = action.output.get("diagnostic_report")
                if diagnostic_report:
                    report_content = self._format_diagnostic_report(diagnostic_report)
                    events.append(
                        ChatEvent(
                            id=f"{event_id}_diagnostic",
                            planId=validation_plan_id,
                            timestamp=timestamp,
                            content=report_content,
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
                    planId=self._get_unified_plan_id(action, force_associate=True),
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
                        planId=self._get_unified_plan_id(action, force_associate=True),
                        timestamp=timestamp,
                        content=content,
                    )
                )

        # Handle SQL Execution
        elif action.action_type == "sql_execution":
            tool_call_id = action.action_id
            tool_input = {}
            if action.input and isinstance(action.input, dict):
                tool_input = action.input

            exec_plan_id = self._get_unified_plan_id(action, force_associate=True)

            if action.status == ActionStatus.PROCESSING:
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

            if action.status in (ActionStatus.SUCCESS, ActionStatus.FAILED):
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
            tool_name = action.action_type.replace("preflight_", "", 1)

            tool_call_id = None
            if action.input and isinstance(action.input, dict):
                for key, value in action.input.items():
                    if "tool_call" in str(key).lower() or isinstance(value, str) and "preflight_" in value:
                        tool_call_id = value
                        break

            if not tool_call_id:
                tool_call_id = str(uuid.uuid4())

            preflight_plan_id = self._get_unified_plan_id(action, force_associate=True)

            if action.status == ActionStatus.PROCESSING:
                tool_input = {}
                if action.input and isinstance(action.input, dict):
                    tool_input = action.input

                events.append(
                    ToolCallEvent(
                        id=f"{event_id}_call",
                        planId=preflight_plan_id,
                        timestamp=timestamp,
                        toolCallId=tool_call_id,
                        toolName=tool_name,
                        input=tool_input,
                    )
                )

            if action.status in [ActionStatus.SUCCESS, ActionStatus.FAILED]:
                events.append(
                    ToolCallResultEvent(
                        id=f"{event_id}_result",
                        planId=preflight_plan_id,
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
                    planId=self._get_unified_plan_id(action, force_associate=True),
                    timestamp=timestamp,
                    content=content,
                )
            )

        # 2. Handle tool calls - ToolCallEvent / ToolCallResultEvent and PlanUpdateEvent for plan tools
        elif action.role == ActionRole.TOOL:
            if action.action_type == "tool_call_result" and action.output:
                tool_call_id = self._find_tool_call_id(action)
                if tool_call_id:
                    events.append(
                        ToolCallResultEvent(
                            id=event_id,
                            planId=self._get_unified_plan_id(action, force_associate=True),
                            timestamp=timestamp,
                            toolCallId=str(tool_call_id),
                            data=action.output,
                            error=action.status == ActionStatus.FAILED,
                        )
                    )
                return events

            if action.action_type == "todo_update" and self._is_internal_todo_update(action):
                return []

            tool_call_id = str(uuid.uuid4())
            self.tool_call_map[action.action_id] = tool_call_id

            is_plan_tool = action.action_type in ["todo_write", "todo_update"]
            normalized_tool_name = self._normalize_tool_name(action.action_type)
            plan_data = {}
            if action.output:
                plan_data = self._extract_plan_from_output(action.output)

            normalized_input: Dict[str, Any] = {}
            if action.input:
                if isinstance(action.input, dict):
                    normalized_input = dict(action.input)
                    if "arguments" in normalized_input and isinstance(normalized_input["arguments"], str):
                        parsed_args = self._try_parse_json_like(normalized_input["arguments"])
                        if isinstance(parsed_args, dict):
                            normalized_input.update(parsed_args)
                else:
                    parsed = self._try_parse_json_like(action.input)
                    if isinstance(parsed, dict):
                        normalized_input = parsed

            tool_plan_id = self._get_unified_plan_id(action, force_associate=True)

            if is_plan_tool and action.action_type == "todo_update":
                todo_id = None
                if "todo_id" in normalized_input:
                    todo_id = normalized_input["todo_id"]
                elif "updated_item" in plan_data:
                    ui = plan_data["updated_item"]
                    if isinstance(ui, dict) and ui.get("id"):
                        todo_id = ui["id"]
                        status = ui.get("status", "").lower()
                        if status == "in_progress":
                            self.active_todo_item_id = todo_id
                            self.logger.debug(f"Tracking active TodoItem: {todo_id}")
                        elif status in ("completed", "failed", "error"):
                            if self.active_todo_item_id == todo_id:
                                self.logger.debug(f"Clearing active TodoItem: {todo_id} (status={status})")
                                self.active_todo_item_id = None

            if is_plan_tool and plan_data:
                todos = []
                replace_order = False
                if "todo_list" in plan_data:
                    tlist = plan_data["todo_list"]
                    if isinstance(tlist, dict) and "items" in tlist:
                        replace_order = True
                        for todo_data in tlist["items"]:
                            if isinstance(todo_data, dict):
                                todo_id = todo_data.get("id")
                                if not todo_id:
                                    self.logger.warning("Skipping todo item without id in todo_list: %s", todo_data)
                                    continue
                                todos.append(
                                    TodoItem(
                                        id=str(todo_id),
                                        content=todo_data.get("content", ""),
                                        status=TodoStatus(self._normalize_todo_status(todo_data.get("status", "pending"))),
                                    )
                                )
                elif "updated_item" in plan_data:
                    ui = plan_data["updated_item"]
                    if isinstance(ui, dict):
                        todo_id = ui.get("id")
                        if not todo_id:
                            self.logger.warning("Skipping updated_item without id: %s", ui)
                        else:
                            todos.append(
                                TodoItem(
                                    id=str(todo_id),
                                    content=ui.get("content", ""),
                                    status=TodoStatus(self._normalize_todo_status(ui.get("status", "pending"))),
                                )
                            )

                if todos:
                    self._update_todo_state(todos, replace_order=replace_order)
                    todo_payload = self._get_todo_state_list() or todos

                    events.append(
                        ToolCallEvent(
                            id=f"{event_id}_call",
                            planId=tool_plan_id,
                            timestamp=timestamp,
                            toolCallId=tool_call_id,
                            toolName=normalized_tool_name,
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

                    plan_update_plan_id = None
                    if action.action_type == "todo_update" and "updated_item" in plan_data:
                        ui = plan_data["updated_item"]
                        if isinstance(ui, dict) and ui.get("id"):
                            plan_update_plan_id = ui["id"]

                    events.append(
                        PlanUpdateEvent(
                            id=f"{event_id}_plan", planId=plan_update_plan_id, timestamp=timestamp, todos=todo_payload
                        )
                    )
            else:
                events.append(
                    ToolCallEvent(
                        id=f"{event_id}_call",
                        planId=tool_plan_id,
                        timestamp=timestamp,
                        toolCallId=tool_call_id,
                        toolName=normalized_tool_name,
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

        # Handle workflow completion (always emit CompleteEvent)
        elif action.action_type == "workflow_completion":
            if self._virtual_step_manager.virtual_plan_emitted:
                final_todos = []
                for step in self.VIRTUAL_STEPS:
                    status = TodoStatus.ERROR if step["id"] in self._virtual_step_manager.failed_virtual_steps else TodoStatus.COMPLETED
                    final_todos.append(
                        TodoItem(id=str(step["id"]), content=str(step["content"]), status=status)
                    )
                events.append(
                    PlanUpdateEvent(id=f"{event_id}_plan_final", planId=None, timestamp=timestamp, todos=final_todos)
                )

            events.append(
                CompleteEvent(
                    id=event_id,
                    planId=None,
                    timestamp=timestamp,
                    content=action.messages or "",
                )
            )

        # Handle workflow initialization
        elif action.action_type == "workflow_init":
            content = f"🚀 **System Initialization**: {action.messages}"
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=None,
                    timestamp=timestamp,
                    content=content,
                )
            )

            if not self._virtual_step_manager.virtual_plan_emitted:
                plan_update = self._generate_virtual_plan_update()
                if plan_update:
                    events.append(plan_update)
                    self._virtual_step_manager.virtual_plan_emitted = True

        # Handle node execution
        elif action.action_type == "node_execution":
            node_type = None
            if action.input and isinstance(action.input, dict):
                node_type = action.input.get("node_type")

            if node_type:
                plan_update = self._generate_virtual_plan_update(node_type)
                if plan_update and plan_update.todos:
                    events.append(plan_update)

            node_desc = "Unknown Node"
            if action.input and isinstance(action.input, dict):
                node_desc = action.input.get("description", node_desc)

            content = f"🔄 **Executing Step**: {node_desc}"
            events.append(
                ChatEvent(
                    id=event_id,
                    planId=self._get_unified_plan_id(action, force_associate=True),
                    timestamp=timestamp,
                    content=content,
                )
            )

        # Handle errors
        elif action.status == ActionStatus.FAILED:
            error_plan_id = self._get_unified_plan_id(action, force_associate=False)

            error_msg = action.messages or "Unknown error"
            if action.output and isinstance(action.output, dict):
                details = []
                if action.output.get("error_code"):
                    details.append(f"Code: {action.output.get('error_code')}")

                suggestions = action.output.get("recovery_suggestions")
                if suggestions and isinstance(suggestions, list):
                    suggestions_str = "
".join([f"- {s}" for s in suggestions])
                    details.append(f"Suggestions:
{suggestions_str}")

                if details:
                    error_msg += "

" + "
".join(details)

            events.append(ErrorEvent(id=event_id, planId=error_plan_id, timestamp=timestamp, error=error_msg))

        # Handle report generation and SQL output
        elif action.action_type == "output_generation" and action.output:
            if isinstance(action.output, dict):
                sql_query = action.output.get("sql_query", "")
                sql_result = action.output.get("sql_result", "")
                sql_query_final = action.output.get("sql_query_final", "")
                sql_result_final = action.output.get("sql_result_final", "")
                row_count = action.output.get("row_count", 0)
                metadata = action.output.get("metadata", {})

                final_sql = sql_query_final if sql_query_final else sql_query

                if final_sql:
                    final_result = sql_result_final if sql_result_final else sql_result

                    if metadata:
                        has_valid_content = any(
                            bool(v) and not (k == "reflection_count" and v == 0)
                            for k, v in metadata.items()
                        )

                        if has_valid_content or "table_schemas" in metadata:
                            table_schemas = metadata.get("table_schemas")
                            report = self._generate_sql_generation_report(
                                sql_query=final_sql,
                                sql_result=final_result,
                                row_count=row_count,
                                metadata=metadata,
                                table_schemas=table_schemas,
                            )
                        else:
                            report = self._generate_sql_summary(final_sql, final_result, row_count)
                    else:
                        report = self._generate_sql_summary(final_sql, final_result, row_count)

                    events.append(
                        ChatEvent(
                            id=f"{event_id}_report",
                            planId=self._get_unified_plan_id(action, force_associate=True),
                            timestamp=timestamp,
                            content=report,
                        )
                    )

                report_url = action.output.get("report_url", "")
                report_data = action.output.get("html_content", "")
                if report_url or report_data:
                    report_plan_id = self._get_unified_plan_id(action, force_associate=True)
                    events.append(
                        ReportEvent(
                            id=event_id, planId=report_plan_id, timestamp=timestamp, url=report_url, data=report_data
                        )
                    )

        # Debug logging: track generated events
        for event in events:
            self.logger.debug(f"Generated event: {event.event}, planId: {event.planId}, id: {event.id}")

        self.validate_event_flow(action.action_type, events)

        return events
