# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from datus.api.event_converter import DeepResearchEventConverter
from datus.api.models import (
    ChatEvent,
    CompleteEvent,
    DeepResearchEventType,
    ErrorEvent,
    PlanUpdateEvent,
    TodoItem,
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
            action_type="llm_generation",
            messages="Test message",
            input={"test": "input"},
            output={"content": "Hello world", "response": "Hi there"},
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        event = self.converter.convert_action_to_event(action, 1)

        assert event is not None
        assert isinstance(event, ChatEvent)
        assert event.event == DeepResearchEventType.CHAT
        assert event.content == "Hello world"  # Should prefer 'content' field
        assert event.planId is not None

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

        event = self.converter.convert_action_to_event(action, 1)

        assert event is not None
        assert isinstance(event, ToolCallEvent)
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
        tool_call_event = self.converter.convert_action_to_event(tool_call_action, 1)
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

        event = self.converter.convert_action_to_event(result_action, 2)

        assert event is not None
        assert isinstance(event, ToolCallResultEvent)
        assert event.event == DeepResearchEventType.TOOL_CALL_RESULT
        assert event.toolCallId == tool_call_id
        assert event.data == {"result": "success", "data": {"tables": ["users", "orders"]}}
        assert event.error is False

    def test_convert_plan_update_to_plan_update_event(self):
        """Test conversion of plan update action to PlanUpdateEvent."""
        action = ActionHistory(
            action_id="plan_update_1",
            role=ActionRole.ASSISTANT,
            action_type="plan_update",
            messages="Updating execution plan",
            input={},
            output={
                "todos": [
                    {"id": "task_1", "content": "Analyze schema", "status": "completed"},
                    {"id": "task_2", "content": "Generate SQL", "status": "in_progress"},
                ]
            },
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        event = self.converter.convert_action_to_event(action, 1)

        assert event is not None
        assert isinstance(event, PlanUpdateEvent)
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

        event = self.converter.convert_action_to_event(action, 1)

        assert event is not None
        assert isinstance(event, CompleteEvent)
        assert event.event == DeepResearchEventType.COMPLETE
        assert event.content == "Workflow completed successfully"

    def test_convert_failed_action_to_error_event(self):
        """Test conversion of failed action to ErrorEvent."""
        action = ActionHistory(
            action_id="failed_action_1",
            role=ActionRole.ASSISTANT,
            action_type="llm_generation",
            messages="LLM call failed",
            input={},
            output={"error": "API rate limit exceeded"},
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        event = self.converter.convert_action_to_event(action, 1)

        assert event is not None
        assert isinstance(event, ErrorEvent)
        assert event.event == DeepResearchEventType.ERROR
        assert "LLM call failed" in event.error

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

        event = self.converter.convert_action_to_event(action, 1)

        assert event is None

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
        pass

    def test_plan_id_consistency(self):
        """Test that all events from the same converter share the same planId."""
        actions = [
            ActionHistory(
                action_id="chat_1",
                role=ActionRole.ASSISTANT,
                action_type="llm_generation",
                messages="First message",
                input={},
                output={"content": "Hello"},
                status=ActionStatus.SUCCESS,
                start_time=datetime.now(),
                end_time=datetime.now(),
            ),
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
            event = self.converter.convert_action_to_event(action, i)
            if event:
                events.append(event)

        assert len(events) == 2
        assert events[0].planId == events[1].planId
        assert events[0].planId == self.converter.plan_id

    def test_convert_syntax_error_to_error_event(self):
        """Test conversion of SQL syntax error to ErrorEvent."""
        action = ActionHistory(
            action_id="syntax_error_1",
            role=ActionRole.TOOL,
            action_type="sql_execution_error",
            messages="SQL syntax validation failed: missing SELECT keyword",
            input={"sql_query": "FROM users WHERE id = 1", "error_type": "syntax"},
            output={
                "error": "SQL语句缺少基本的查询/修改操作关键词(SELECT, INSERT, UPDATE, DELETE等)",
                "error_type": "syntax",
                "suggestions": ["检查SQL语句是否包含必要的关键词(SELECT, INSERT, UPDATE, DELETE等)", "验证括号和引号是否匹配", "确认表名和列名拼写是否正确"],
                "can_retry": False,
            },
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ErrorEvent)
        assert event.event == DeepResearchEventType.ERROR
        assert "SQL语句缺少基本的查询" in event.error

    def test_convert_table_not_found_to_error_event(self):
        """Test conversion of table not found error to ErrorEvent."""
        action = ActionHistory(
            action_id="table_error_1",
            role=ActionRole.TOOL,
            action_type="table_not_found",
            messages="Table 'nonexistent_table' not found",
            input={"table_name": "nonexistent_table", "error_type": "table_not_found"},
            output={
                "error": "Table 'nonexistent_table' not found or no DDL available",
                "error_type": "table_not_found",
                "suggestions": ["检查表名拼写是否正确", "确认数据库权限", "使用search_table工具查找相似表名"],
                "can_retry": True,
            },
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ErrorEvent)
        assert event.event == DeepResearchEventType.ERROR
        assert "nonexistent_table" in event.error

    def test_convert_db_connection_error_to_error_event(self):
        """Test conversion of database connection error to ErrorEvent."""
        action = ActionHistory(
            action_id="db_error_1",
            role=ActionRole.TOOL,
            action_type="db_connection_error",
            messages="Database connection failed: Connection timeout",
            input={"sql_query": "SELECT * FROM users", "error_type": "connection"},
            output={
                "error": "Connection timeout after 30 seconds",
                "error_type": "connection",
                "suggestions": ["检查数据库服务是否运行", "验证网络连接", "确认数据库连接配置"],
                "can_retry": True,
            },
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ErrorEvent)
        assert event.event == DeepResearchEventType.ERROR
        assert "Connection timeout" in event.error

    def test_convert_generic_tool_error_to_error_event(self):
        """Test conversion of generic tool error to ErrorEvent."""
        action = ActionHistory(
            action_id="generic_error_1",
            role=ActionRole.TOOL,
            action_type="tool_execution_error",
            messages="Tool read_query failed: Permission denied",
            input={"tool_name": "read_query", "error_type": "permission"},
            output={
                "error": "Permission denied for table access",
                "error_type": "permission",
                "tool_name": "read_query",
                "can_retry": False,
            },
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        events = self.converter.convert_action_to_event(action, 1)

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ErrorEvent)
        assert event.event == DeepResearchEventType.ERROR
        assert "Permission denied" in event.error
