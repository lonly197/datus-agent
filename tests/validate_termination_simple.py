#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Simple validation script for workflow termination mechanism (Phase 4.5).

This script validates the critical bug fix from Phase 4 without running
full node execution, focusing on the core termination logic.

Usage:
    python tests/validate_termination_simple.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def validate_termination_mechanism():
    """
    Validate workflow termination mechanism with simple code inspection.
    """
    print("=" * 80)
    print("🔍 Workflow Termination Mechanism Validation (Phase 4.5)")
    print("=" * 80)
    print()

    all_passed = True

    # Test 1: Validate WorkflowTerminationStatus enum exists
    print("Test 1: Validate WorkflowTerminationStatus enum...")
    try:
        from datus.agent.workflow_runner import WorkflowTerminationStatus

        required_states = [
            "CONTINUE",
            "SKIP_TO_REFLECT",
            "TERMINATE_WITH_ERROR",
            "TERMINATE_SUCCESS"
        ]

        for state in required_states:
            assert hasattr(WorkflowTerminationStatus, state), \
                f"Missing state: {state}"

        print("✅ WorkflowTerminationStatus enum exists with all required states")
        for state in required_states:
            value = getattr(WorkflowTerminationStatus, state)
            print(f"   - {state}: {value}")
        print()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 2: Validate _terminate_workflow method signature
    print("Test 2: Validate _terminate_workflow method...")
    try:
        from datus.agent.workflow_runner import WorkflowRunner
        import inspect

        # Check method exists
        assert hasattr(WorkflowRunner, "_terminate_workflow"), \
            "WorkflowRunner should have _terminate_workflow method"

        # Check method signature
        sig = inspect.signature(WorkflowRunner._terminate_workflow)
        params = list(sig.parameters.keys())

        assert "self" in params, "Method should have 'self' parameter"
        assert "termination_status" in params, "Method should have 'termination_status' parameter"
        assert "error_message" in params, "Method should have 'error_message' parameter"

        print("✅ _terminate_workflow method exists with correct signature")
        print(f"   - Parameters: {params}")
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 3: Validate _handle_node_failure method
    print("Test 3: Validate _handle_node_failure method...")
    try:
        from datus.agent.workflow_runner import WorkflowRunner
        import inspect

        # Check method exists
        assert hasattr(WorkflowRunner, "_handle_node_failure"), \
            "WorkflowRunner should have _handle_node_failure method"

        # Check method signature
        sig = inspect.signature(WorkflowRunner._handle_node_failure)
        params = list(sig.parameters.keys())

        assert "self" in params, "Method should have 'self' parameter"
        assert "current_node" in params, "Method should have 'current_node' parameter"

        print("✅ _handle_node_failure method exists with correct signature")
        print(f"   - Parameters: {params}")
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 4: Validate SchemaValidationNode has last_action_status attribute
    print("Test 4: Validate SchemaValidationNode has last_action_status...")
    try:
        from datus.agent.node.schema_validation_node import SchemaValidationNode
        import inspect

        # Check if __init__ sets last_action_status
        source = inspect.getsource(SchemaValidationNode.__init__)

        assert "last_action_status" in source, \
            "SchemaValidationNode.__init__ should set last_action_status"

        print("✅ SchemaValidationNode has last_action_status attribute")
        print("   - Found in __init__ method")
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 5: Validate _terminate_workflow contains workflow_task.cancel()
    print("Test 5: Validate _terminate_workflow contains cancellation logic...")
    try:
        from datus.agent.workflow_runner import WorkflowRunner
        import inspect

        # Get source code
        source = inspect.getsource(WorkflowRunner._terminate_workflow)

        # Check for critical components
        checks = [
            ("workflow_task", "Checks workflow_task attribute"),
            (".cancel()", "Calls cancel() method"),
            ("TERMINATE_WITH_ERROR", "Handles TERMINATE_WITH_ERROR status"),
        ]

        for check_str, description in checks:
            assert check_str in source, \
                f"_terminate_workflow should contain: {check_str} ({description})"
            print(f"   ✅ Contains {check_str}: {description}")

        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 6: Validate workflow_task tracking in __init__
    print("Test 6: Validate WorkflowRunner tracks workflow_task...")
    try:
        from datus.agent.workflow_runner import WorkflowRunner
        import inspect

        # Get source code
        source = inspect.getsource(WorkflowRunner.__init__)

        assert "workflow_task" in source, \
            "WorkflowRunner.__init__ should initialize workflow_task"

        print("✅ WorkflowRunner initializes workflow_task attribute")
        print("   - Found in __init__ method")
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 7: Validate integration with actual cancellation
    print("Test 7: Validate actual task cancellation behavior...")
    try:
        from datus.agent.workflow_runner import WorkflowRunner, WorkflowTerminationStatus
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

        # Create a real asyncio task
        async def dummy_task():
            await asyncio.sleep(10)
            return {"status": "completed"}

        runner.workflow_task = asyncio.create_task(dummy_task())

        # Verify task is running
        assert not runner.workflow_task.done(), "Task should be running"
        assert not runner.workflow_task.cancelled(), "Task should not be cancelled yet"

        # Call _terminate_workflow
        runner._terminate_workflow(
            termination_status=WorkflowTerminationStatus.TERMINATE_WITH_ERROR,
            error_message="Test error"
        )

        # Give cancellation a moment to take effect
        await asyncio.sleep(0.1)

        # Verify task was cancelled
        assert runner.workflow_task.cancelled(), "Task should be cancelled"

        print("✅ Actual task cancellation works correctly")
        print(f"   - Task cancelled: {runner.workflow_task.cancelled()}")
        print(f"   - Task done: {runner.workflow_task.done()}")
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Summary
    print("=" * 80)
    if all_passed:
        print("✅ ALL VALIDATION TESTS PASSED")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 80)
    print()

    if all_passed:
        print("Summary of validated fixes:")
        print("1. ✅ WorkflowTerminationStatus enum exists with all required states")
        print("2. ✅ _terminate_workflow() has correct signature")
        print("3. ✅ _handle_node_failure() has correct signature")
        print("4. ✅ SchemaValidationNode has last_action_status attribute")
        print("5. ✅ _terminate_workflow() contains cancellation logic")
        print("6. ✅ WorkflowRunner tracks workflow_task")
        print("7. ✅ Actual task cancellation works")
        print()
        print("🎉 Critical bug fix validated: Background execution is prevented after termination")
        print()
    else:
        print("Some tests failed. Please review the output above.")
        print()

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(validate_termination_mechanism())
    sys.exit(0 if success else 1)
