# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datetime import datetime

from datus.api.event_converter import DeepResearchEventConverter
from datus.api.models import (
    ChatEvent,
    CompleteEvent,
    DeepResearchEventType,
    ErrorEvent,
    PlanUpdateEvent,
    TodoStatus,
    ToolCallEvent,
    ToolCallResultEvent,
)
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus


class TestDeepResearchEventConverter:
    def setup_method(self):
        """Setup test fixtures."""
        self.converter = DeepResearchEventConverter()

    def test_convert_assistant_message_to_chat_event(self):
        """Test conversion of assistant message to ChatEvent."""
        action = ActionHistory(
            action_id="test_action_1",
            role=ActionRole.ASSISTANT,
            action_type="raw_stream",
            messages="Test message",
            input={"test": "input"},
            output={"content": "Hello world", "response": "Hi there"},
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) >= 1
        event = events[0]
        assert isinstance(event, ChatEvent)
        assert event.event == DeepResearchEventType.CHAT
        assert event.content == "Hello world"  # Should prefer 'content' field
        # planId is None for general chat messages not associated with a todo
        assert event.planId is None

    def test_convert_tool_call_to_tool_call_event(self):
        """Test conversion of tool action to ToolCallEvent."""
        action = ActionHistory(
            action_id="tool_action_1",
            role=ActionRole.TOOL,
            action_type="schema_linking",
            messages="Calling schema tool",
            input={"table": "users", "database": "test"},
            output={},
            status=ActionStatus.PROCESSING,
            start_time=datetime.now(),
            end_time=None,
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) >= 1
        # Find the ToolCallEvent (may have ChatEvent too)
        event = next((e for e in events if isinstance(e, ToolCallEvent)), None)
        assert event is not None
        assert event.event == DeepResearchEventType.TOOL_CALL
        assert event.toolName == "schema_linking"
        assert event.input == {"table": "users", "database": "test"}
        assert event.toolCallId is not None

        # Check that tool_call_id is stored in the map
        assert action.action_id in self.converter.tool_call_map
        assert self.converter.tool_call_map[action.action_id] == event.toolCallId

    def test_convert_tool_result_to_tool_call_result_event(self):
        """Test conversion of tool result to ToolCallResultEvent."""
        # First create a tool call
        tool_call_action = ActionHistory(
            action_id="tool_call_123",
            role=ActionRole.TOOL,
            action_type="schema_linking",
            messages="Calling schema tool",
            input={"table": "users"},
            output={},
            status=ActionStatus.PROCESSING,
            start_time=datetime.now(),
            end_time=None,
        )

        # Convert tool call first
        tool_call_events = self.converter.convert_action_to_event(tool_call_action, 1)
        tool_call_event = next((e for e in tool_call_events if isinstance(e, ToolCallEvent)), None)
        assert tool_call_event is not None
        tool_call_id = tool_call_event.toolCallId

        # Now create the result action
        result_action = ActionHistory(
            action_id="tool_result_123",
            role=ActionRole.TOOL,
            action_type="tool_call_result",
            messages="Tool completed",
            input={"action_id": "tool_call_123"},
            output={"result": "success", "data": {"tables": ["users", "orders"]}},
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(result_action, 2)
        assert len(events) >= 1
        event = next((e for e in events if isinstance(e, ToolCallResultEvent)), None)
        assert event is not None
        assert event.event == DeepResearchEventType.TOOL_CALL_RESULT
        assert event.toolCallId == tool_call_id
        assert event.data == {"result": "success", "data": {"tables": ["users", "orders"]}}
        assert event.error is False

    def test_convert_plan_update_to_plan_update_event(self):
        """Test conversion of plan update action to PlanUpdateEvent."""
        action = ActionHistory(
            action_id="plan_update_1",
            role=ActionRole.TOOL,
            action_type="todo_write",
            messages="Updating execution plan",
            input={},
            output={
                "plan_data": {
                    "todo_list": {
                        "items": [
                            {"id": "task_1", "content": "Analyze schema", "status": "completed"},
                            {"id": "task_2", "content": "Generate SQL", "status": "in_progress"},
                        ]
                    }
                }
            },
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) >= 1
        event = next((e for e in events if isinstance(e, PlanUpdateEvent)), None)
        assert event is not None
        assert event.event == DeepResearchEventType.PLAN_UPDATE
        assert len(event.todos) == 2
        assert event.todos[0].id == "task_1"
        assert event.todos[0].status == TodoStatus.COMPLETED
        assert event.todos[1].status == TodoStatus.IN_PROGRESS

    def test_convert_workflow_completion_to_complete_event(self):
        """Test conversion of workflow completion to CompleteEvent."""
        action = ActionHistory(
            action_id="completion_1",
            role=ActionRole.WORKFLOW,
            action_type="workflow_completion",
            messages="Workflow completed successfully",
            input={},
            output={"final_result": "SQL generated"},
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) >= 1
        event = next((e for e in events if isinstance(e, CompleteEvent)), None)
        assert event is not None
        assert event.event == DeepResearchEventType.COMPLETE
        assert event.content == "Workflow completed successfully"

    def test_convert_failed_action_to_error_event(self):
        """Test conversion of failed action to ErrorEvent."""
        action = ActionHistory(
            action_id="failed_action_1",
            role=ActionRole.WORKFLOW,
            action_type="unknown",
            messages="Tool execution failed",
            input={},
            output={"error": "Tool execution failed"},
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) >= 1
        event = next((e for e in events if isinstance(e, ErrorEvent)), None)
        assert event is not None
        assert event.event == DeepResearchEventType.ERROR
        assert "Tool execution failed" in event.error

    def test_convert_unknown_action_returns_none(self):
        """Test that unknown actions return None."""
        action = ActionHistory(
            action_id="unknown_1",
            role=ActionRole.SYSTEM,
            action_type="unknown_type",
            messages="Unknown action",
            input={},
            output={},
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        # Unknown actions return empty list
        assert len(events) == 0

    def test_convert_stream_to_events_yields_sse_format(self):
        """Test that convert_stream_to_events yields proper SSE format."""

        async def mock_action_stream():
            yield ActionHistory(
                action_id="chat_1",
                role=ActionRole.ASSISTANT,
                action_type="llm_generation",
                messages="Hello",
                input={},
                output={"content": "Hello world"},
                status=ActionStatus.SUCCESS,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        # This would require async testing, but for now we'll test the synchronous parts
        # In a real test, we'd use pytest-asyncio or similar

    def test_plan_id_consistency(self):
        """Test that tool events have planId while general chat may not."""
        actions = [
            ActionHistory(
                action_id="tool_1",
                role=ActionRole.TOOL,
                action_type="schema_linking",
                messages="Tool call",
                input={"table": "users"},
                output={},
                status=ActionStatus.PROCESSING,
                start_time=datetime.now(),
                end_time=None,
            ),
        ]

        events = []
        for i, action in enumerate(actions, 1):
            action_events = self.converter.convert_action_to_event(action, i)
            if action_events:
                events.extend(action_events)

        # Tool events should always have planId
        assert len(events) >= 1
        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_events) >= 1
        assert tool_events[0].planId is not None

    def test_convert_syntax_error_to_error_event(self):
        """Test conversion of SQL syntax error to ErrorEvent."""
        action = ActionHistory(
            action_id="syntax_error_1",
            role=ActionRole.WORKFLOW,
            action_type="unknown",
            messages="SQL语句缺少基本的查询/修改操作关键词(SELECT, INSERT, UPDATE, DELETE等)",
            input={},
            output={
                "error": "SQL syntax validation failed: missing SELECT keyword",
            },
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) >= 1
        event = next((e for e in events if isinstance(e, ErrorEvent)), None)
        assert event is not None
        assert isinstance(event, ErrorEvent)
        assert event.event == DeepResearchEventType.ERROR
        assert "SQL语句缺少基本的查询" in event.error

    def test_convert_table_not_found_to_error_event(self):
        """Test conversion of table not found error to ErrorEvent."""
        action = ActionHistory(
            action_id="table_error_1",
            role=ActionRole.WORKFLOW,
            action_type="unknown",
            messages="Table 'nonexistent_table' not found",
            input={},
            output={
                "error": "Table 'nonexistent_table' not found or no DDL available",
            },
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) >= 1
        event = next((e for e in events if isinstance(e, ErrorEvent)), None)
        assert event is not None
        assert isinstance(event, ErrorEvent)
        assert event.event == DeepResearchEventType.ERROR
        assert "nonexistent_table" in event.error

    def test_convert_db_connection_error_to_error_event(self):
        """Test conversion of database connection error to ErrorEvent."""
        action = ActionHistory(
            action_id="db_error_1",
            role=ActionRole.WORKFLOW,
            action_type="unknown",
            messages="Database connection failed: Connection timeout",
            input={},
            output={
                "error": "Connection timeout after 30 seconds",
            },
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) >= 1
        event = next((e for e in events if isinstance(e, ErrorEvent)), None)
        assert event is not None
        assert isinstance(event, ErrorEvent)
        assert event.event == DeepResearchEventType.ERROR
        assert "Connection timeout" in event.error

    def test_convert_generic_tool_error_to_error_event(self):
        """Test conversion of generic tool error to ErrorEvent."""
        action = ActionHistory(
            action_id="generic_error_1",
            role=ActionRole.WORKFLOW,
            action_type="unknown",
            messages="Tool read_query failed: Permission denied",
            input={},
            output={
                "error": "Permission denied for table access",
            },
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) >= 1
        event = next((e for e in events if isinstance(e, ErrorEvent)), None)
        assert event is not None
        assert isinstance(event, ErrorEvent)
        assert event.event == DeepResearchEventType.ERROR
        assert "Permission denied" in event.error
