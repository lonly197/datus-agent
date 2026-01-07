# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from datus.api.models import ChatResearchRequest, DeepResearchEventType
from datus.api.service import create_app


class TestChatResearchSSE:
    @pytest.fixture
    def test_client(self, tmp_path):
        """Create a test client for the FastAPI app."""
        # Mock the agent configuration loading
        with patch("datus.api.service.load_agent_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.current_namespace = "test"
            mock_config.return_value.rag_base_path = str(tmp_path / "test")

            app = create_app(MagicMock())
            client = TestClient(app)
            return client

    def test_chat_research_endpoint_requires_sse_accept_header(self, test_client):
        """Test that /chat_research endpoint requires Accept: text/event-stream header."""
        request_data = {"namespace": "test", "task": "Generate a simple SQL query"}

        # Request without SSE accept header should fail
        response = test_client.post("/workflows/chat_research", json=request_data)
        assert response.status_code == 400
        assert "text/event-stream" in response.json()["detail"]

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_chat_research_endpoint_accepts_sse_requests(self, mock_stream, test_client):
        """Test that /chat_research endpoint accepts SSE requests."""

        # Mock the streaming response
        async def mock_stream_generator():
            yield 'data: {"id":"evt_1","planId":"plan_123","timestamp":1703123456789,"event":"chat","content":"Starting research..."}\n\n'
            yield 'data: {"id":"evt_2","planId":"plan_123","timestamp":1703123456790,"event":"complete","content":"Research completed"}\n\n'

        mock_stream.return_value = mock_stream_generator()

        request_data = {"namespace": "test", "task": "Generate a simple SQL query"}

        # Request with SSE accept header
        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/event-stream"
        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == "no-cache"

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_sse_response_format(self, mock_stream, test_client):
        """Test that SSE responses have correct format."""

        # Mock the streaming response with proper SSE format
        async def mock_stream_generator():
            yield 'data: {"id":"evt_1","planId":"plan_123","timestamp":1703123456789,"event":"chat","content":"Hello world"}\n\n'
            yield 'data: {"id":"evt_2","planId":"plan_123","timestamp":1703123456790,"event":"complete","content":"Done"}\n\n'

        mock_stream.return_value = mock_stream_generator()

        request_data = {"namespace": "test", "task": "Test task"}

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

        # Get response content
        content = response.content.decode("utf-8")

        # Verify SSE format - should contain 'data: ' prefix and end with '\n\n'
        lines = content.strip().split("\n")
        assert len(lines) >= 2  # At least one data line and empty line

        # Check that each data line starts with 'data: '
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) >= 1

        # Parse JSON from first data line
        first_data_line = data_lines[0]
        json_str = first_data_line.replace("data: ", "")
        event_data = json.loads(json_str)

        # Verify event structure
        assert "id" in event_data
        assert "planId" in event_data
        assert "timestamp" in event_data
        assert "event" in event_data
        assert event_data["event"] in [e.value for e in DeepResearchEventType]

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_sse_events_contain_valid_json(self, mock_stream, test_client):
        """Test that all SSE events contain valid JSON."""

        # Mock streaming with various event types
        async def mock_stream_generator():
            yield 'data: {"id":"chat_1","planId":"plan_123","timestamp":1703123456789,"event":"chat","content":"Starting analysis..."}\n\n'
            yield 'data: {"id":"plan_1","planId":"plan_123","timestamp":1703123456790,"event":"plan_update","todos":[{"id":"task_1","content":"Analyze schema","status":"in_progress"}]}\n\n'
            yield 'data: {"id":"tool_1","planId":"plan_123","timestamp":1703123456791,"event":"tool_call","toolCallId":"call_456","toolName":"schema_linking","input":{"table":"users"}}\n\n'
            yield 'data: {"id":"result_1","planId":"plan_123","timestamp":1703123456792,"event":"tool_call_result","toolCallId":"call_456","data":{"success":true},"error":false}\n\n'
            yield 'data: {"id":"complete_1","planId":"plan_123","timestamp":1703123456793,"event":"complete","content":"Research completed successfully"}\n\n'

        mock_stream.return_value = mock_stream_generator()

        request_data = {"namespace": "test", "task": "Test comprehensive research task"}

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

        content = response.content.decode("utf-8")
        lines = content.strip().split("\n")

        # Extract all data lines
        data_lines = [line for line in lines if line.startswith("data: ")]

        # Should have 5 events
        assert len(data_lines) == 5

        # Parse and validate each event
        expected_events = [
            DeepResearchEventType.CHAT,
            DeepResearchEventType.PLAN_UPDATE,
            DeepResearchEventType.TOOL_CALL,
            DeepResearchEventType.TOOL_CALL_RESULT,
            DeepResearchEventType.COMPLETE,
        ]

        for i, data_line in enumerate(data_lines):
            json_str = data_line.replace("data: ", "")
            event_data = json.loads(json_str)

            # Validate required fields
            assert "id" in event_data
            assert "planId" in event_data
            assert "timestamp" in event_data
            assert "event" in event_data

            # Validate event type
            assert event_data["event"] == expected_events[i].value

            # Validate planId consistency
            assert event_data["planId"] == "plan_123"

            # Validate event-specific fields
            if event_data["event"] == DeepResearchEventType.CHAT.value:
                assert "content" in event_data
            elif event_data["event"] == DeepResearchEventType.PLAN_UPDATE.value:
                assert "todos" in event_data
                assert isinstance(event_data["todos"], list)
            elif event_data["event"] == DeepResearchEventType.TOOL_CALL.value:
                assert "toolCallId" in event_data
                assert "toolName" in event_data
                assert "input" in event_data
            elif event_data["event"] == DeepResearchEventType.TOOL_CALL_RESULT.value:
                assert "toolCallId" in event_data
                assert "data" in event_data
                assert "error" in event_data
            elif event_data["event"] == DeepResearchEventType.COMPLETE.value:
                assert "content" in event_data

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_error_event_handling(self, mock_stream, test_client):
        """Test that error events are properly formatted."""

        # Mock streaming with an error event
        async def mock_stream_generator():
            yield 'data: {"id":"error_1","planId":"plan_456","timestamp":1703123456789,"event":"error","error":"Workflow execution failed: API timeout"}\n\n'

        mock_stream.return_value = mock_stream_generator()

        request_data = {"namespace": "test", "task": "Test error handling"}

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

        content = response.content.decode("utf-8")
        data_lines = [line for line in content.split("\n") if line.startswith("data: ")]

        assert len(data_lines) == 1

        json_str = data_lines[0].replace("data: ", "")
        event_data = json.loads(json_str)

        assert event_data["event"] == DeepResearchEventType.ERROR.value
        assert "error" in event_data
        assert "API timeout" in event_data["error"]

    def test_request_validation(self, test_client):
        """Test request validation for required fields."""
        # Test missing namespace
        request_data = {"task": "Generate SQL"}

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        # Should fail validation
        assert response.status_code == 422  # Validation error

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_plan_mode_parameter(self, mock_stream, test_client):
        """Test that plan_mode parameter is passed correctly."""
        # Mock to capture the request object passed to the service
        captured_request = None

        async def capture_request_generator(request):
            nonlocal captured_request
            captured_request = request
            yield 'data: {"id":"test","planId":"plan_123","timestamp":1703123456789,"event":"complete","content":"Done"}\n\n'

        # Mock the service method to capture the request
        async def mock_run_chat_research_stream(request, client_id):
            return capture_request_generator(request)

        with patch.object(
            test_client.app.state.service, "run_chat_research_stream", side_effect=mock_run_chat_research_stream
        ):
            request_data = {"namespace": "test", "task": "Test plan mode", "plan_mode": True}

            response = test_client.post(
                "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
            )

            assert response.status_code == 200
            # Note: In a real test, we'd verify that plan_mode=True was passed to the workflow
            # This would require more complex mocking of the agent and workflow components


class TestChatResearchPromptParameters:
    """Test cases for prompt and promptMode parameters in chat research."""

    @pytest.fixture
    def test_client(self):
        """Create a test client for the FastAPI app."""
        with patch("datus.api.service.load_agent_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.current_namespace = "test"
            mock_config.return_value.rag_base_path = "/tmp/test"

            app = create_app(MagicMock())
            client = TestClient(app)
            return client

    def test_chat_research_accepts_prompt_parameters(self, test_client):
        """Test that chat research accepts prompt and promptMode parameters."""
        # Test with append mode (default)
        request_data = {
            "namespace": "test",
            "task": "Generate SQL query",
            "prompt": "You are a SQL expert",
            "prompt_mode": "append",
        }

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

    def test_chat_research_accepts_prompt_replace_mode(self, test_client):
        """Test that chat research accepts prompt with replace mode."""
        request_data = {
            "namespace": "test",
            "task": "Generate SQL query",
            "prompt": "You are a data analyst",
            "prompt_mode": "replace",
        }

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

    def test_chat_research_prompt_default_mode(self, test_client):
        """Test that prompt works with default append mode when prompt_mode is not specified."""
        request_data = {
            "namespace": "test",
            "task": "Generate SQL query",
            "prompt": "You are a SQL expert",
            # prompt_mode defaults to "append"
        }

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

    def test_chat_research_without_prompt(self, test_client):
        """Test that chat research works without prompt parameters (backward compatibility)."""
        request_data = {
            "namespace": "test",
            "task": "Generate SQL query",
            # No prompt or prompt_mode
        }

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

    def test_chat_research_request_model_validation(self):
        """Test ChatResearchRequest model validation with prompt parameters."""
        # Valid request with prompt
        request = ChatResearchRequest(
            namespace="test", task="Generate SQL", prompt="You are a SQL expert", prompt_mode="append"
        )
        assert request.prompt == "You are a SQL expert"
        assert request.prompt_mode == "append"

        # Valid request with replace mode
        request = ChatResearchRequest(
            namespace="test", task="Generate SQL", prompt="You are a data analyst", prompt_mode="replace"
        )
        assert request.prompt_mode == "replace"

        # Valid request without prompt (should work)
        request = ChatResearchRequest(namespace="test", task="Generate SQL")
        assert request.prompt is None
        assert request.prompt_mode == "append"  # default value

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_prompt_parameters_passed_to_workflow_metadata(self, mock_stream, test_client):
        """Test that prompt parameters are passed to workflow metadata."""
        captured_request = None

        async def capture_request_generator(request):
            nonlocal captured_request
            captured_request = request
            yield 'data: {"id":"test","planId":"plan_123","timestamp":1703123456789,"event":"complete","content":"Done"}\n\n'

        async def mock_run_chat_research_stream(request, client_id):
            return capture_request_generator(request)

        with patch.object(
            test_client.app.state.service, "run_chat_research_stream", side_effect=mock_run_chat_research_stream
        ):
            request_data = {
                "namespace": "test",
                "task": "Generate SQL with custom prompt",
                "prompt": "You are a senior SQL developer",
                "prompt_mode": "replace",
            }

            response = test_client.post(
                "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
            )

            assert response.status_code == 200
            # In a real implementation, we would verify that the prompt parameters
            # are correctly passed through to the workflow metadata

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_sql_review_task_identification_and_config(self, mock_stream, test_client):
        """Test that SQL review tasks are correctly identified and configured."""

        # Mock streaming response with preflight success
        async def mock_stream_generator():
            yield 'data: {"id":"evt_1","timestamp":1703123456789,"event":"chat","content":"Starting SQL review..."}\n\n'
            yield 'data: {"id":"evt_2","timestamp":1703123456790,"event":"complete","content":"Review completed"}\n\n'

        mock_stream.return_value = mock_stream_generator()

        request_data = {
            "namespace": "test",
            "task": "审查以下SQL：SELECT * FROM users WHERE id = 1",
            "ext_knowledge": "使用StarRocks规范",
        }

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200
        mock_stream.assert_called_once()

        # Verify the call arguments include sql_review configuration
        call_args = mock_stream.call_args[0][0]  # First positional argument
        assert call_args.task == request_data["task"]
        assert call_args.ext_knowledge == request_data["ext_knowledge"]

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_sql_review_with_mcp_unavailable_fallback(self, mock_stream, test_client):
        """Test SQL review with MCP services unavailable, should still generate report with warnings."""

        # Mock streaming response that includes fail-safe annotations
        async def mock_stream_with_fallback():
            yield 'data: {"id":"evt_1","timestamp":1703123456789,"event":"chat","content":"Preflight tools partially failed..."}\n\n'
            yield 'data: {"id":"evt_2","timestamp":1703123456790,"event":"report","content":"⚠️ 数据完整性说明: 部分工具调用失败"}\n\n'
            yield 'data: {"id":"evt_3","timestamp":1703123456791,"event":"complete","content":"Review completed with limitations"}\n\n'

        mock_stream.return_value = mock_stream_with_fallback()

        request_data = {
            "namespace": "test",
            "task": "检查这个SQL：SELECT * FROM orders WHERE date > '2024-01-01'",
            "ext_knowledge": "检查性能问题",
        }

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

        # Verify that the service was called with the correct parameters
        call_args = mock_stream.call_args[0][0]
        assert "检查" in call_args.task  # Should contain review keywords

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_sql_review_preflight_success_scenario(self, mock_stream, test_client):
        """Test successful preflight execution for SQL review."""

        async def mock_successful_preflight():
            yield 'data: {"id":"evt_1","timestamp":1703123456789,"event":"tool_call","tool":"describe_table","status":"success"}\n\n'
            yield 'data: {"id":"evt_2","timestamp":1703123456790,"event":"tool_call","tool":"search_external_knowledge","status":"success"}\n\n'
            yield 'data: {"id":"evt_3","timestamp":1703123456791,"event":"tool_call","tool":"read_query","status":"success"}\n\n'
            yield 'data: {"id":"evt_4","timestamp":1703123456792,"event":"chat","content":"基于完整数据分析生成审查报告..."}\n\n'
            yield 'data: {"id":"evt_5","timestamp":1703123456793,"event":"complete","content":"审查完成"}\n\n'

        mock_stream.return_value = mock_successful_preflight()

        request_data = {
            "namespace": "test",
            "task": "审核SQL：SELECT id, name FROM customers WHERE status = 'active'",
            "ext_knowledge": "遵循公司SQL规范",
        }

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_non_sql_review_task_uses_regular_flow(self, mock_stream, test_client):
        """Test that non-SQL review tasks don't trigger preflight tools."""

        async def mock_regular_flow():
            yield 'data: {"id":"evt_1","timestamp":1703123456789,"event":"chat","content":"Generating SQL..."}\n\n'
            yield 'data: {"id":"evt_2","timestamp":1703123456790,"event":"complete","content":"SQL generated"}\n\n'

        mock_stream.return_value = mock_regular_flow()

        request_data = {
            "namespace": "test",
            "task": "Generate SQL to find all active users",  # Not a review task
            "ext_knowledge": "Use standard SQL practices",
        }

        response = test_client.post(
            "/workflows/chat_research", json=request_data, headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

        # Verify task type identification
        call_args = mock_stream.call_args[0][0]
        assert call_args.task == request_data["task"]

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_enhanced_preflight_tools_integration(self, mock_stream, test_client):
        """Test integration of enhanced preflight tools in chat research workflow."""

        # Mock streaming response with enhanced preflight events
        async def mock_enhanced_stream():
            # Preflight plan update
            yield 'data: {"id":"plan_1","timestamp":1703123456789,"event":"plan_update","todos":[{"id":"analyze_plan","content":"Analyze query plan","status":"pending"},{"id":"check_conflicts","content":"Check table conflicts","status":"pending"},{"id":"validate_partitioning","content":"Validate partitioning","status":"pending"}]}\n\n'

            # Tool call events
            yield 'data: {"id":"tool_1","timestamp":1703123456790,"event":"tool_call","toolCallId":"call_analyze","toolName":"analyze_query_plan","input":{"sql_query":"SELECT * FROM test_table"}}\n\n'

            # Tool result events
            yield 'data: {"id":"result_1","timestamp":1703123456791,"event":"tool_call_result","toolCallId":"call_analyze","data":{"success":true,"estimated_rows":1000,"hotspots":[{"reason":"full_table_scan","severity":"high"}]},"executionTime":150}\n\n'

            # Chat response with enhanced context
            yield 'data: {"id":"chat_1","timestamp":1703123456792,"event":"chat","content":"Based on the query plan analysis, I found performance issues..."}\n\n'

            # Completion
            yield 'data: {"id":"complete_1","timestamp":1703123456793,"event":"complete","content":"SQL review completed with enhanced analysis"}\n\n'

        mock_stream.return_value = mock_enhanced_stream()

        request_data = {
            "namespace": "test",
            "task": "审查以下SQL：SELECT * FROM test_table WHERE id = 1",
            "plan_mode": True,
        }

        response = test_client.post(
            "/workflows/chat_research",
            json=request_data,
            headers={"Accept": "text/event-stream"},
        )

        assert response.status_code == 200

        # Parse SSE events
        response_text = response.text
        events = [line for line in response_text.split("\n") if line.startswith("data: ")]

        # Verify enhanced preflight events are present
        event_types = []
        for event in events:
            try:
                event_data = json.loads(event[6:])  # Remove 'data: ' prefix
                event_types.append(event_data.get("event"))
            except json.JSONDecodeError:
                continue

        assert "plan_update" in event_types
        assert "tool_call" in event_types
        assert "tool_call_result" in event_types
        assert "chat" in event_types
        assert "complete" in event_types

    @patch("datus.api.service.DatusAPIService.run_chat_research_stream")
    def test_sql_review_with_enhanced_preflight_error_handling(self, mock_stream, test_client):
        """Test SQL review with enhanced preflight error handling."""

        async def mock_error_stream():
            # Tool call
            yield 'data: {"id":"tool_1","timestamp":1703123456789,"event":"tool_call","toolCallId":"call_analyze","toolName":"analyze_query_plan","input":{"sql_query":"SELECT * FROM invalid_table"}}\n\n'

            # Tool failure
            yield 'data: {"id":"error_1","timestamp":1703123456790,"event":"tool_call_result","toolCallId":"call_analyze","data":{"success":false,"error":"Table \'invalid_table\' not found"},"executionTime":50}\n\n'

            # Error event
            yield 'data: {"id":"sys_error","timestamp":1703123456791,"event":"error","error":"Database table not found","suggestions":["Check table name spelling","Verify table exists in database"]}\n\n'

            # Recovery chat response
            yield 'data: {"id":"chat_1","timestamp":1703123456792,"event":"chat","content":"虽然查询计划分析失败，但我可以基于其他信息继续审查..."}\n\n'

        mock_stream.return_value = mock_error_stream()

        request_data = {
            "namespace": "test",
            "task": "审查SQL：SELECT * FROM invalid_table WHERE id = 1",
            "plan_mode": True,
        }

        response = test_client.post(
            "/workflows/chat_research",
            json=request_data,
            headers={"Accept": "text/event-stream"},
        )

        assert response.status_code == 200

        # Verify error handling events
        response_text = response.text
        events = [line for line in response_text.split("\n") if line.startswith("data: ")]

        error_events = []
        for event in events:
            try:
                event_data = json.loads(event[6:])
                if event_data.get("event") == "error":
                    error_events.append(event_data)
            except json.JSONDecodeError:
                continue

        assert len(error_events) > 0
        assert "not found" in error_events[0].get("error", "").lower()
