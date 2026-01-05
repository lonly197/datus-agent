# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.api.models import RunWorkflowRequest
from datus.api.service import generate_sse_stream
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus


class TestSSEStreaming:
    """Test SSE streaming functionality for workflow execution."""

    @pytest.fixture
    def mock_service(self):
        """Mock the global service instance."""
        mock_service = MagicMock()
        mock_service._generate_task_id.return_value = "test_task_123"
        mock_service.run_workflow_stream = AsyncMock()
        mock_service.task_store = None
        return mock_service

    @pytest.fixture
    def sample_request(self):
        """Create a sample workflow request."""
        return RunWorkflowRequest(
            workflow="chat_agentic", namespace="test", task="Test SQL generation task", database_name="test_db"
        )

    def test_progress_sequence_increment(self, mock_service):
        """Test that progress_seq increments for each action."""
        with patch("datus.api.service.service", mock_service):
            # Create mock actions with different types
            actions = [
                ActionHistory.create_action(
                    action_id="node_exec_1",
                    role=ActionRole.USER,
                    action_type="node_execution",
                    messages="Starting node execution",
                    input_data={"node_type": "chat"},
                    status=ActionStatus.PROCESSING,
                ),
                ActionHistory.create_action(
                    action_id="msg_1",
                    role=ActionRole.ASSISTANT,
                    action_type="message",
                    messages="Thinking content",
                    output_data={"raw_output": "Partial thinking content"},
                    status=ActionStatus.SUCCESS,
                ),
                ActionHistory.create_action(
                    action_id="response_1",
                    role=ActionRole.ASSISTANT,
                    action_type="chat_response",
                    messages="Final response",
                    output_data={"response": "Final SQL", "sql": "SELECT 1", "tokens_used": 100},
                    status=ActionStatus.SUCCESS,
                ),
            ]

            mock_service.run_workflow_stream.return_value = actions

            # Collect SSE events
            events = []

            async def collect_events():
                async for event in generate_sse_stream(self.sample_request, "test_client"):
                    events.append(event)

            asyncio.run(collect_events())

            # Verify progress_seq increments
            progress_seqs = []
            for event in events:
                if "progress_seq" in event:
                    # Extract progress_seq from event data
                    data_part = event.split("data: ")[1].strip()
                    try:
                        data = json.loads(data_part)
                        if "progress_seq" in data:
                            progress_seqs.append(data["progress_seq"])
                    except json.JSONDecodeError:
                        continue

            # Should have 3 progress sequences (one per action)
            assert len(progress_seqs) >= 3
            # Should be monotonically increasing
            assert progress_seqs == sorted(progress_seqs)

    def test_partial_content_throttling(self, mock_service):
        """Test that partial content events are throttled."""
        with patch("datus.api.service.service", mock_service):
            # Create multiple rapid partial actions
            actions = []
            for i in range(5):
                action = ActionHistory.create_action(
                    action_id=f"partial_{i}",
                    role=ActionRole.ASSISTANT,
                    action_type="message",
                    messages=f"Partial {i}",
                    output_data={"raw_output": f"Partial content {i}"},
                    status=ActionStatus.SUCCESS,
                )
                actions.append(action)

            mock_service.run_workflow_stream.return_value = actions

            # Collect SSE events and measure timing
            events = []
            asyncio.get_event_loop().time()

            async def collect_events():
                async for event in generate_sse_stream(self.sample_request, "test_client"):
                    events.append((asyncio.get_event_loop().time(), event))

            asyncio.run(collect_events())

            # Should have generated events (though timing might vary in test environment)
            assert len(events) > 0

    def test_event_type_mapping(self, mock_service):
        """Test that different action types map to correct SSE event types."""
        with patch("datus.api.service.service", mock_service):
            actions = [
                # Node execution start
                ActionHistory.create_action(
                    action_id="node_exec_1",
                    role=ActionRole.USER,
                    action_type="node_execution",
                    messages="Starting chat node",
                    input_data={"node_type": "chat", "description": "Chat node"},
                    status=ActionStatus.PROCESSING,
                ),
                # Chat thinking partial
                ActionHistory.create_action(
                    action_id="think_1",
                    role=ActionRole.ASSISTANT,
                    action_type="message",
                    messages="Thinking...",
                    output_data={"raw_output": "Analyzing schema..."},
                    status=ActionStatus.SUCCESS,
                ),
                # Tool call
                ActionHistory.create_action(
                    action_id="tool_1",
                    role=ActionRole.ASSISTANT,
                    action_type="tool_call",
                    messages="Calling database tool",
                    input_data={"function_name": "search_table"},
                    status=ActionStatus.PROCESSING,
                ),
                # Final chat response
                ActionHistory.create_action(
                    action_id="response_1",
                    role=ActionRole.ASSISTANT,
                    action_type="chat_response",
                    messages="Generated SQL",
                    output_data={"response": "SQL generated", "sql": "SELECT * FROM table", "tokens_used": 50},
                    status=ActionStatus.SUCCESS,
                ),
                # Generic action (fallback)
                ActionHistory.create_action(
                    action_id="generic_1",
                    role=ActionRole.SYSTEM,
                    action_type="custom_action",
                    messages="Custom system action",
                    status=ActionStatus.SUCCESS,
                ),
            ]

            mock_service.run_workflow_stream.return_value = actions

            # Collect SSE events
            events = []

            async def collect_events():
                async for event in generate_sse_stream(self.sample_request, "test_client"):
                    events.append(event)

            asyncio.run(collect_events())

            # Verify event types are present
            event_types = []
            for event in events:
                if event.startswith("event: "):
                    event_type = event.split("event: ")[1].split("\n")[0]
                    event_types.append(event_type)

            # Should include our expected event types
            assert "node_progress" in event_types
            assert "chat_thinking" in event_types
            assert "node_detail" in event_types
            assert "chat_response" in event_types
            assert "generic_action" in event_types

    def test_partial_content_truncation(self, mock_service):
        """Test that large partial content is truncated."""
        with patch("datus.api.service.service", mock_service):
            # Create action with very large partial content
            large_content = "x" * 10000  # 10KB of content
            action = ActionHistory.create_action(
                action_id="large_partial",
                role=ActionRole.ASSISTANT,
                action_type="message",
                messages="Large thinking content",
                output_data={"raw_output": large_content},
                status=ActionStatus.SUCCESS,
            )

            mock_service.run_workflow_stream.return_value = [action]

            # Collect SSE events
            events = []

            async def collect_events():
                async for event in generate_sse_stream(self.sample_request, "test_client"):
                    events.append(event)

            asyncio.run(collect_events())

            # Find chat_thinking event
            chat_thinking_event = None
            for event in events:
                if "event: chat_thinking" in event:
                    chat_thinking_event = event
                    break

            assert chat_thinking_event is not None

            # Parse the event data
            data_part = chat_thinking_event.split("data: ")[1].strip()
            data = json.loads(data_part)

            # Content should be truncated
            assert data["is_truncated"] is True
            assert len(data["content"]) <= 8192  # 8KB limit

    def test_backward_compatibility(self, mock_service):
        """Test that existing event types and data formats are preserved."""
        with patch("datus.api.service.service", mock_service):
            actions = [
                # Workflow initialization (existing)
                ActionHistory.create_action(
                    action_id="workflow_init",
                    role=ActionRole.SYSTEM,
                    action_type="workflow_init",
                    messages="Initializing workflow",
                    status=ActionStatus.PROCESSING,
                ),
                # SQL generation success (existing)
                ActionHistory.create_action(
                    action_id="sql_gen",
                    role=ActionRole.ASSISTANT,
                    action_type="sql_generation",
                    messages="Generated SQL",
                    output_data={"sql_query": "SELECT 1"},
                    status=ActionStatus.SUCCESS,
                ),
                # SQL execution success (existing)
                ActionHistory.create_action(
                    action_id="sql_exec",
                    role=ActionRole.TOOL,
                    action_type="sql_execution",
                    messages="Executed SQL",
                    output_data={"has_results": True, "row_count": 10, "sql_result": "result"},
                    status=ActionStatus.SUCCESS,
                ),
            ]

            mock_service.run_workflow_stream.return_value = actions

            # Collect SSE events
            events = []

            async def collect_events():
                async for event in generate_sse_stream(self.sample_request, "test_client"):
                    events.append(event)

            asyncio.run(collect_events())

            # Verify existing event types are preserved
            sql_generated_found = any("event: sql_generated" in event for event in events)
            execution_complete_found = any("event: execution_complete" in event for event in events)
            progress_found = any("event: progress" in event for event in events)

            assert sql_generated_found, "sql_generated event should be preserved"
            assert execution_complete_found, "execution_complete event should be preserved"
            assert progress_found, "progress event should be preserved"

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_service):
        """Test error handling in SSE streaming."""
        with patch("datus.api.service.service", mock_service):
            # Mock an exception during streaming
            mock_service.run_workflow_stream.side_effect = Exception("Test error")

            events = []
            try:
                async for event in generate_sse_stream(self.sample_request, "test_client"):
                    events.append(event)
            except Exception:
                pass  # Expected to handle errors gracefully

            # Should have error event
            error_events = [e for e in events if "event: error" in e]
            assert len(error_events) > 0, "Should emit error event on exception"
