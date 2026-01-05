# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for Node async stream adaptor functionality.
"""

import asyncio

import pytest

from datus.agent.node.node import Node, _run_async_stream_to_result
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseResult


class DummyAsyncStreamNode(Node):
    """Test node that only implements execute_stream."""

    def __init__(self):
        # Minimal initialization for testing - don't call super().__init__ to avoid complexity
        self.type = NodeType.TYPE_CHAT
        self.result = None
        self.status = "pending"
        self.action_history_manager = None
        self.id = "test_node"
        self.description = "Test async stream node"

    def execute(self):
        """Not implemented - we only support async streaming."""
        # This should never be called since Node.run() should use the adaptor
        raise NotImplementedError("Use execute_stream instead")

    def setup_input(self, input_data):
        """Mock setup input."""

    def update_context(self, context):
        """Mock update context."""

    async def execute_stream(self, action_history_manager=None):
        """Mock async streaming execution."""
        # Yield some mock ActionHistory objects
        yield ActionHistory.create_action(
            role=ActionRole.SYSTEM,
            action_type="test_action",
            messages="Test message",
            input_data={"test": "input"},
            status=ActionStatus.PROCESSING,
        )
        yield ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="test_response",
            messages="Test response",
            input_data={"test": "input"},
            output_data={"test": "output"},
            status=ActionStatus.SUCCESS,
        )

    def _initialize(self):
        """Mock initialization."""

    def start(self):
        """Mock start."""

    def complete(self, result):
        """Mock complete."""
        self.result = result
        self.status = "completed"

    def fail(self, error_message):
        """Mock fail."""
        self.status = "failed"


class DummyFailingAsyncStreamNode(DummyAsyncStreamNode):
    """Test node that fails during async streaming."""

    def __init__(self):
        super().__init__()
        self.description = "Test failing async stream node"

    async def execute_stream(self, action_history_manager=None):
        """Mock failing async streaming execution."""
        yield ActionHistory.create_action(
            role=ActionRole.SYSTEM,
            action_type="failing_action",
            messages="About to fail",
            input_data={"test": "input"},
            status=ActionStatus.PROCESSING,
        )
        raise Exception("Test failure in async stream")


class TestNodeAsyncAdaptor:
    """Test cases for Node async stream adaptor."""

    def test_run_async_stream_to_result_success(self):
        """Test successful execution of async stream node."""
        node = DummyAsyncStreamNode()
        action_history_manager = ActionHistoryManager()
        node.action_history_manager = action_history_manager

        # Run the adaptor
        result = _run_async_stream_to_result(node)

        # Verify result
        assert isinstance(result, BaseResult)
        assert result.success is True

        # Verify ActionHistory was recorded
        actions = action_history_manager.get_actions()
        assert len(actions) == 2
        assert actions[0].action_type == "test_action"
        assert actions[1].action_type == "test_response"

    def test_run_async_stream_to_result_failure(self):
        """Test failed execution of async stream node."""
        node = DummyFailingAsyncStreamNode()
        action_history_manager = ActionHistoryManager()
        node.action_history_manager = action_history_manager

        # Run the adaptor
        result = _run_async_stream_to_result(node)

        # Verify result is error
        from datus.schemas.base import BaseResult

        assert isinstance(result, BaseResult)
        assert result.success is False
        assert "Test failure in async stream" in result.error

        # Verify partial ActionHistory was still recorded
        actions = action_history_manager.get_actions()
        assert len(actions) == 1  # Only the first action before failure
        assert actions[0].action_type == "failing_action"

    def test_run_async_stream_to_result_no_action_history_manager(self):
        """Test execution without action history manager."""
        node = DummyAsyncStreamNode()
        node.action_history_manager = None

        # Run the adaptor
        result = _run_async_stream_to_result(node)

        # Verify result
        assert isinstance(result, BaseResult)
        assert result.success is True

        # Verify no action history manager interaction attempted
        # (This should not raise any exceptions)

    @pytest.mark.asyncio
    async def test_node_run_uses_async_adaptor(self):
        """Test that Node.run() uses the async adaptor for streaming-only nodes."""
        node = DummyAsyncStreamNode()
        action_history_manager = ActionHistoryManager()
        node.action_history_manager = action_history_manager

        # Run the node
        node.run()

        # Verify node completed successfully
        assert node.status == "completed"
        assert isinstance(node.result, BaseResult)
        assert node.result.success is True

        # Verify ActionHistory was recorded during execution
        actions = action_history_manager.get_actions()
        assert len(actions) == 2

    def test_adaptor_creates_fresh_event_loop(self):
        """Test that adaptor creates a fresh event loop and cleans it up."""
        # This test verifies that the adaptor doesn't interfere with existing event loops
        node = DummyAsyncStreamNode()

        # Get initial event loop state
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists
            pass

        # Run adaptor
        result = _run_async_stream_to_result(node)

        # Verify success
        assert result.success is True

        # Verify event loop was created and closed properly
        # The adaptor should not leave dangling event loops
        # This is mainly a smoke test to ensure no exceptions were raised
