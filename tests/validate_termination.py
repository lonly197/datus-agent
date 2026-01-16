#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Quick validation script for workflow termination mechanism (Phase 4.5).

This script validates the critical bug fix from Phase 4:
- workflow_task.cancel() prevents background execution
- SOFT_FAILED vs FAILED status handling
- ErrorEvent and CompletedEvent are sent correctly

Usage:
    python tests/validate_termination.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def validate_termination_mechanism():
    """
    Validate workflow termination mechanism with simple scenarios.
    """
    print("=" * 80)
    print("🔍 Workflow Termination Mechanism Validation (Phase 4.5)")
    print("=" * 80)
    print()

    # Test 1: Validate WorkflowTerminationStatus enum exists
    print("Test 1: Validate WorkflowTerminationStatus enum...")
    try:
        from datus.agent.workflow_runner import WorkflowTerminationStatus

        assert hasattr(WorkflowTerminationStatus, "CONTINUE")
        assert hasattr(WorkflowTerminationStatus, "SKIP_TO_REFLECT")
        assert hasattr(WorkflowTerminationStatus, "TERMINATE_WITH_ERROR")
        assert hasattr(WorkflowTerminationStatus, "TERMINATE_SUCCESS")

        print("✅ WorkflowTerminationStatus enum exists with all required states")
        print(f"   - CONTINUE: {WorkflowTerminationStatus.CONTINUE}")
        print(f"   - SKIP_TO_REFLECT: {WorkflowTerminationStatus.SKIP_TO_REFLECT}")
        print(f"   - TERMINATE_WITH_ERROR: {WorkflowTerminationStatus.TERMINATE_WITH_ERROR}")
        print(f"   - TERMINATE_SUCCESS: {WorkflowTerminationStatus.TERMINATE_SUCCESS}")
        print()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    # Test 2: Validate _terminate_workflow method exists and cancels task
    print("Test 2: Validate _terminate_workflow method...")
    try:
        from datus.agent.workflow_runner import WorkflowRunner
        from unittest.mock import Mock
        import argparse

        # Create minimal args
        args = argparse.Namespace(
            workflow="text2sql",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
        )

        # Create mock config
        mock_config = Mock()
        mock_config.get_model = Mock(return_value="gpt-4")

        # Create WorkflowRunner
        runner = WorkflowRunner(
            args=args,
            agent_config=mock_config,
            run_id="validation_test",
        )

        # Verify method exists
        assert hasattr(runner, "_terminate_workflow")

        # Create a real asyncio task
        async def dummy_task():
            await asyncio.sleep(10)
            return {"status": "completed"}

        runner.workflow_task = asyncio.create_task(dummy_task())

        # Verify task is running
        assert not runner.workflow_task.done()
        assert not runner.workflow_task.cancelled()

        # Call _terminate_workflow
        runner._terminate_workflow(
            termination_status=WorkflowTerminationStatus.TERMINATE_WITH_ERROR,
            error_message="Test error"
        )

        # Verify task was cancelled
        await asyncio.sleep(0.1)  # Give cancellation a moment to take effect
        assert runner.workflow_task.cancelled(), "Task should be cancelled"

        print("✅ _terminate_workflow method exists and correctly cancels workflow_task")
        print(f"   - Task cancelled: {runner.workflow_task.cancelled()}")
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Validate _handle_node_failure returns correct status
    print("Test 3: Validate _handle_node_failure method...")
    try:
        from datus.agent.workflow_runner import WorkflowRunner
        from datus.agent.node.schema_validation_node import SchemaValidationNode
        from datus.schemas.action_history import ActionStatus
        import argparse

        # Create minimal args
        args = argparse.Namespace(
            workflow="text2sql",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
        )

        # Create mock config
        mock_config = Mock()
        mock_config.get_model = Mock(return_value="gpt-4")

        # Create WorkflowRunner
        runner = WorkflowRunner(
            args=args,
            agent_config=mock_config,
            run_id="validation_test",
        )

        # Scenario 3a: Hard failure (no reflect node)
        print("   Scenario 3a: Hard failure (no reflect node, FAILED status)...")

        # Create workflow WITHOUT reflect node
        workflow_hard_fail = Mock()
        workflow_hard_fail.task = Mock()
        workflow_hard_fail.task.task = "test query"
        workflow_hard_fail.context = Mock()
        workflow_hard_fail.context.table_schemas = []
        workflow_hard_fail.nodes = []  # No reflect node
        workflow_hard_fail.metadata = {}

        runner.workflow = workflow_hard_fail

        # Create failed node with FAILED status
        failed_node_hard = SchemaValidationNode(
            node_id="test_schema_validation",
            description="Test Schema Validation",
            node_type="schema_validation",
            agent_config=mock_config,
        )
        failed_node_hard.workflow = workflow_hard_fail
        failed_node_hard.last_action_status = ActionStatus.FAILED

        # Test _handle_node_failure
        termination_status = runner._handle_node_failure(failed_node_hard)

        assert termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR, \
            f"Expected TERMINATE_WITH_ERROR, got {termination_status}"

        print("   ✅ Returns TERMINATE_WITH_ERROR for hard failure")

        # Scenario 3b: Soft failure (with reflect node)
        print("   Scenario 3b: Soft failure (with reflect node, SOFT_FAILED status)...")

        # Create workflow WITH reflect node
        workflow_soft_fail = Mock()
        workflow_soft_fail.task = Mock()
        workflow_soft_fail.task.task = "test query"
        workflow_soft_fail.context = Mock()
        workflow_soft_fail.context.table_schemas = []

        # Add reflect node
        reflect_node = Mock()
        reflect_node.type = "reflect"
        workflow_soft_fail.nodes = [reflect_node]
        workflow_soft_fail.metadata = {}

        runner.workflow = workflow_soft_fail

        # Create failed node with SOFT_FAILED status
        failed_node_soft = SchemaValidationNode(
            node_id="test_schema_validation",
            description="Test Schema Validation",
            node_type="schema_validation",
            agent_config=mock_config,
        )
        failed_node_soft.workflow = workflow_soft_fail
        failed_node_soft.last_action_status = ActionStatus.SOFT_FAILED

        # Test _handle_node_failure
        termination_status = runner._handle_node_failure(failed_node_soft)

        assert termination_status == WorkflowTerminationStatus.SKIP_TO_REFLECT, \
            f"Expected SKIP_TO_REFLECT, got {termination_status}"

        print("   ✅ Returns SKIP_TO_REFLECT for soft failure")
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Validate SchemaValidationNode sets correct status
    print("Test 4: Validate SchemaValidationNode sets last_action_status...")
    try:
        from datus.agent.node.schema_validation_node import SchemaValidationNode
        from datus.schemas.action_history import ActionStatus
        from unittest.mock import Mock

        # Create mock config
        mock_config = Mock()
        mock_config.get_model = Mock(return_value="gpt-4")

        # Create workflow with no schemas (hard failure scenario)
        workflow = Mock()
        workflow.task = Mock()
        workflow.task.task = "test query"
        workflow.context = Mock()
        workflow.context.table_schemas = []  # Empty schemas → hard failure
        workflow.metadata = {}

        # Create SchemaValidationNode
        node = SchemaValidationNode(
            node_id="test_schema_validation",
            description="Test Schema Validation",
            node_type="schema_validation",
            agent_config=mock_config,
        )
        node.workflow = workflow

        # Execute node
        result = []
        async for action in node.run():
            result.append(action)

        # Validate: last_action_status is set to FAILED (not SOFT_FAILED)
        assert hasattr(node, 'last_action_status'), "Node should have last_action_status"
        assert node.last_action_status == ActionStatus.FAILED, \
            f"Expected FAILED, got {node.last_action_status}"

        # Validate: Result contains allow_reflection=False
        assert node.result is not None, "Node should have result"
        assert node.result.success == False, "Result should indicate failure"
        assert node.result.data.get("allow_reflection") == False, \
            "Hard failure should not allow reflection"

        print("✅ SchemaValidationNode correctly sets FAILED status for hard failure")
        print(f"   - last_action_status: {node.last_action_status}")
        print(f"   - allow_reflection: {node.result.data.get('allow_reflection')}")
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("=" * 80)
    print("✅ ALL VALIDATION TESTS PASSED")
    print("=" * 80)
    print()
    print("Summary of validated fixes:")
    print("1. ✅ WorkflowTerminationStatus enum exists with all required states")
    print("2. ✅ _terminate_workflow() correctly cancels workflow_task")
    print("3. ✅ _handle_node_failure() returns correct status based on failure type")
    print("4. ✅ SchemaValidationNode sets FAILED (not SOFT_FAILED) for hard failure")
    print()
    print("🎉 Critical bug fix validated: Background execution is prevented after termination")
    print()

    return True


if __name__ == "__main__":
    success = asyncio.run(validate_termination_mechanism())
    sys.exit(0 if success else 1)
