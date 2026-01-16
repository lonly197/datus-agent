# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Test suite for workflow termination mechanism validation.

This test suite validates the critical bug fix from Phase 4:
- Ensuring workflow_task.cancel() prevents background execution
- Verifying ErrorEvent and CompletedEvent are sent correctly
- Confirming SOFT_FAILED vs FAILED status handling
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datus.agent.workflow_runner import WorkflowRunner, WorkflowTerminationStatus
from datus.schemas.action_history import ActionStatus
from datus.configuration.node_type import NodeType
from datus.agent.node import Node


class TestWorkflowTerminationMechanism:
    """
    Test suite for validating workflow termination mechanism (Phase 4.5).

    Critical scenarios to test:
    1. Schema validation failure (no schemas) → HARD FAILED → terminate with error
    2. Node failure with reflect node → SOFT FAILED → skip to reflect
    3. Node failure without reflect node → HARD FAILED → terminate with error
    4. workflow_task.cancel() actually prevents background execution
    """

    @pytest.fixture
    def mock_agent_config(self):
        """Create mock agent configuration."""
        config = Mock()
        config.get_model = Mock(return_value="gpt-4")
        return config

    @pytest.fixture
    def mock_workflow(self):
        """Create mock workflow with no reflect node (hard failure scenario)."""
        workflow = Mock()
        workflow.task = Mock()
        workflow.task.task = "test query"
        workflow.context = Mock()
        workflow.context.table_schemas = []  # Empty schemas → hard failure
        workflow.nodes = []  # No reflect node
        workflow.metadata = {}
        return workflow

    @pytest.mark.asyncio
    async def test_schema_validation_hard_failure_terminates(self, mock_agent_config, mock_workflow):
        """
        Test Case 1: Schema validation with no schemas → HARD FAILED → immediate termination.

        Expected behavior:
        1. SchemaValidationNode sets last_action_status = FAILED (not SOFT_FAILED)
        2. _handle_node_failure() returns TERMINATE_WITH_ERROR
        3. _terminate_workflow() calls workflow_task.cancel()
        4. Background execution is prevented
        """
        from datus.agent.node.schema_validation_node import SchemaValidationNode

        # Create SchemaValidationNode
        node = SchemaValidationNode(
            node_id="test_schema_validation",
            description="Test Schema Validation",
            node_type=NodeType.TYPE_SCHEMA_VALIDATION,
            agent_config=mock_agent_config,
        )
        node.workflow = mock_workflow

        # Execute node
        result = []
        async for action in node.run():
            result.append(action)

        # Validate: Node returned FAILED status (not SOFT_FAILED)
        assert hasattr(node, 'last_action_status'), "Node should have last_action_status attribute"
        assert node.last_action_status == ActionStatus.FAILED, f"Expected FAILED, got {node.last_action_status}"

        # Validate: Result contains allow_reflection=False (hard failure)
        assert node.result is not None, "Node should have result"
        assert node.result.success == False, "Result should indicate failure"
        assert node.result.data.get("allow_reflection") == False, "Hard failure should not allow reflection"

        print("✅ Test 1 PASSED: Schema validation hard failure correctly sets FAILED status")

    @pytest.mark.asyncio
    async def test_workflow_runner_handles_hard_failure(self, mock_agent_config):
        """
        Test Case 2: WorkflowRunner correctly handles hard failure without reflect node.

        Expected behavior:
        1. _handle_node_failure() detects no reflect node reachable
        2. Returns TERMINATE_WITH_ERROR (not SKIP_TO_REFLECT)
        3. Triggers workflow_task.cancel()
        """
        from datus.agent.workflow_runner import WorkflowRunner
        from datus.agent.node.schema_validation_node import SchemaValidationNode
        import argparse

        # Create minimal args
        args = argparse.Namespace(
            workflow="text2sql",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
        )

        # Create WorkflowRunner
        runner = WorkflowRunner(
            args=args,
            agent_config=mock_agent_config,
            run_id="test_termination",
        )

        # Create mock workflow with no schemas
        workflow = Mock()
        workflow.task = Mock()
        workflow.task.task = "test query"
        workflow.context = Mock()
        workflow.context.table_schemas = []
        workflow.nodes = []  # No reflect node
        workflow.metadata = {}

        runner.workflow = workflow

        # Create failed node
        failed_node = SchemaValidationNode(
            node_id="test_schema_validation",
            description="Test Schema Validation",
            node_type=NodeType.TYPE_SCHEMA_VALIDATION,
            agent_config=mock_agent_config,
        )
        failed_node.workflow = workflow
        failed_node.last_action_status = ActionStatus.FAILED

        # Test _handle_node_failure
        termination_status = runner._handle_node_failure(failed_node)

        # Validate: Returns TERMINATE_WITH_ERROR
        assert termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR, \
            f"Expected TERMINATE_WITH_ERROR, got {termination_status}"

        print("✅ Test 2 PASSED: WorkflowRunner correctly returns TERMINATE_WITH_ERROR for hard failure")

    @pytest.mark.asyncio
    async def test_terminate_workflow_cancels_task(self, mock_agent_config):
        """
        Test Case 3: _terminate_workflow() actually cancels workflow_task.

        Expected behavior:
        1. workflow_task.cancel() is called
        2. Task is marked as cancelled (cancelled() returns True)
        3. Background execution is prevented
        """
        from datus.agent.workflow_runner import WorkflowRunner
        import argparse

        # Create minimal args
        args = argparse.Namespace(
            workflow="text2sql",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
        )

        # Create WorkflowRunner
        runner = WorkflowRunner(
            args=args,
            agent_config=mock_agent_config,
            run_id="test_termination",
        )

        # Create a mock workflow_task that can be cancelled
        async def dummy_workflow():
            await asyncio.sleep(10)
            return {"status": "completed"}

        # Create actual asyncio task
        runner.workflow_task = asyncio.create_task(dummy_workflow())

        # Verify task is not done yet
        assert not runner.workflow_task.done(), "Task should not be done initially"

        # Call _terminate_workflow with TERMINATE_WITH_ERROR
        runner._terminate_workflow(
            termination_status=WorkflowTerminationStatus.TERMINATE_WITH_ERROR,
            error_message="Test error message"
        )

        # Verify: Task was cancelled
        assert runner.workflow_task.cancelled(), "Task should be cancelled after _terminate_workflow"

        print("✅ Test 3 PASSED: _terminate_workflow correctly cancels workflow_task")

    @pytest.mark.asyncio
    async def test_soft_failure_with_reflect_node(self, mock_agent_config):
        """
        Test Case 4: Node failure with reflect node → SOFT FAILED → skip to reflect.

        Expected behavior:
        1. _handle_node_failure() detects reflect node is reachable
        2. Returns SKIP_TO_REFLECT (not TERMINATE_WITH_ERROR)
        3. Does NOT cancel workflow_task
        """
        from datus.agent.workflow_runner import WorkflowRunner
        from datus.agent.node.schema_validation_node import SchemaValidationNode
        import argparse

        # Create minimal args
        args = argparse.Namespace(
            workflow="text2sql",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
        )

        # Create WorkflowRunner
        runner = WorkflowRunner(
            args=args,
            agent_config=mock_agent_config,
            run_id="test_termination",
        )

        # Create workflow WITH reflect node (soft failure scenario)
        workflow = Mock()
        workflow.task = Mock()
        workflow.task.task = "test query"
        workflow.context = Mock()
        workflow.context.table_schemas = []  # No schemas → failure

        # Add reflect node to workflow
        reflect_node = Mock()
        reflect_node.type = NodeType.TYPE_REFLECT
        workflow.nodes = [reflect_node]
        workflow.metadata = {}

        runner.workflow = workflow

        # Create failed node with SOFT_FAILED status
        failed_node = SchemaValidationNode(
            node_id="test_schema_validation",
            description="Test Schema Validation",
            node_type=NodeType.TYPE_SCHEMA_VALIDATION,
            agent_config=mock_agent_config,
        )
        failed_node.workflow = workflow
        failed_node.last_action_status = ActionStatus.SOFT_FAILED

        # Test _handle_node_failure
        termination_status = runner._handle_node_failure(failed_node)

        # Validate: Returns SKIP_TO_REFLECT (not TERMINATE_WITH_ERROR)
        assert termination_status == WorkflowTerminationStatus.SKIP_TO_REFLECT, \
            f"Expected SKIP_TO_REFLECT, got {termination_status}"

        print("✅ Test 4 PASSED: Soft failure with reflect node correctly returns SKIP_TO_REFLECT")

    @pytest.mark.asyncio
    async def test_no_background_execution_after_termination(self, mock_agent_config):
        """
        Test Case 5: Verify no background execution continues after termination.

        This is the critical test for the bug fix:
        - Before Phase 4: workflow_task continued running in background after ErrorEvent
        - After Phase 4: workflow_task.cancel() should prevent background execution

        Expected behavior:
        1. Workflow starts execution
        2. Node fails (hard failure)
        3. workflow_task is cancelled
        4. No further nodes are executed
        5. Background execution is prevented
        """
        from datus.agent.workflow_runner import WorkflowRunner
        import argparse

        # Create minimal args
        args = argparse.Namespace(
            workflow="text2sql",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
        )

        # Create WorkflowRunner
        runner = WorkflowRunner(
            args=args,
            agent_config=mock_agent_config,
            run_id="test_no_background_execution",
        )

        # Create workflow with multiple nodes
        workflow = Mock()
        workflow.task = Mock()
        workflow.task.task = "test query"
        workflow.context = Mock()
        workflow.context.table_schemas = []

        # Create nodes: first will fail hard, second should NOT execute
        first_node = Mock()
        first_node.id = "first_node"
        first_node.description = "First Node (will fail)"
        first_node.type = NodeType.TYPE_SCHEMA_VALIDATION
        first_node.status = "failed"

        second_node = Mock()
        second_node.id = "second_node"
        second_node.description = "Second Node (should not execute)"
        second_node.type = NodeType.TYPE_GENERATE_SQL
        second_node.status = "pending"

        workflow.nodes = [first_node, second_node]  # No reflect node
        workflow.metadata = {}

        runner.workflow = workflow

        # Mock first_node to have last_action_status = FAILED
        first_node.last_action_status = ActionStatus.FAILED

        # Simulate workflow execution
        termination_status = runner._handle_node_failure(first_node)

        # Validate: Terminated with error
        assert termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR

        # Create actual workflow_task and cancel it
        async def dummy_background_task():
            # This simulates background execution that should be prevented
            await asyncio.sleep(5)
            second_node.status = "completed"  # This should NOT happen
            return {"status": "background_executed"}

        runner.workflow_task = asyncio.create_task(dummy_background_task())

        # Terminate workflow
        runner._terminate_workflow(
            termination_status=WorkflowTerminationStatus.TERMINATE_WITH_ERROR,
            error_message="Test termination"
        )

        # Wait a bit to ensure background task had time to start if it wasn't cancelled
        await asyncio.sleep(0.1)

        # Validate: Task was cancelled
        assert runner.workflow_task.cancelled(), "Background task should be cancelled"

        # Validate: Second node was NOT executed
        assert second_node.status == "pending", \
            "Second node should not execute after workflow termination"

        print("✅ Test 5 PASSED: Background execution is prevented after termination")


def run_tests():
    """
    Run all workflow termination tests.

    Usage:
        python -m tests.test_workflow_termination
    """
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
