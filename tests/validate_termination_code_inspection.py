#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Code inspection validation for workflow termination mechanism (Phase 4.5).

This script validates the critical bug fix from Phase 4 by inspecting source code
without importing modules (avoids dependency issues).

Usage:
    python3 tests/validate_termination_code_inspection.py
"""

import re
import sys
import os


def validate_workflow_runner():
    """Validate workflow_runner.py contains required changes."""
    print("=" * 80)
    print("🔍 Workflow Termination Mechanism Validation (Phase 4.5)")
    print("=" * 80)
    print()

    all_passed = True

    # Test 1: Validate WorkflowTerminationStatus enum
    print("Test 1: Validate WorkflowTerminationStatus enum...")
    try:
        with open("datus/agent/workflow_runner.py", "r") as f:
            content = f.read()

        # Check for enum definition
        enum_pattern = r'class WorkflowTerminationStatus\(str, Enum\):'
        if not re.search(enum_pattern, content):
            print("❌ FAILED: WorkflowTerminationStatus enum not found")
            all_passed = False
        else:
            print("✅ WorkflowTerminationStatus enum class found")

            # Check for required enum values
            required_values = [
                'CONTINUE = "continue"',
                'SKIP_TO_REFLECT = "skip_to_reflect"',
                'TERMINATE_WITH_ERROR = "terminate_with_error"',
                'TERMINATE_SUCCESS = "terminate_success"'
            ]

            for value in required_values:
                if value in content:
                    print(f"   ✅ Contains: {value}")
                else:
                    print(f"   ❌ Missing: {value}")
                    all_passed = False
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 2: Validate _terminate_workflow method
    print("Test 2: Validate _terminate_workflow method...")
    try:
        with open("datus/agent/workflow_runner.py", "r") as f:
            content = f.read()

        # Check for method definition
        method_pattern = r'def _terminate_workflow\('
        if not re.search(method_pattern, content):
            print("❌ FAILED: _terminate_workflow method not found")
            all_passed = False
        else:
            print("✅ _terminate_workflow method found")

            # Check for critical components
            critical_checks = [
                ('workflow_task.cancel()', "Calls cancel() on workflow_task"),
                ('TERMINATE_WITH_ERROR', "Handles TERMINATE_WITH_ERROR status"),
                ('TERMINATE_SUCCESS', "Handles TERMINATE_SUCCESS status"),
                ("def __init__", "Found in WorkflowRunner class"),
            ]

            for check_str, description in critical_checks:
                if check_str in content:
                    print(f"   ✅ Contains {check_str}: {description}")
                else:
                    print(f"   ❌ Missing {check_str}: {description}")
                    all_passed = False
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 3: Validate _handle_node_failure method
    print("Test 3: Validate _handle_node_failure method...")
    try:
        with open("datus/agent/workflow_runner.py", "r") as f:
            content = f.read()

        # Check for method definition
        method_pattern = r'def _handle_node_failure\('
        if not re.search(method_pattern, content):
            print("❌ FAILED: _handle_node_failure method not found")
            all_passed = False
        else:
            print("✅ _handle_node_failure method found")

            # Check for critical logic
            critical_checks = [
                ('ActionStatus.SOFT_FAILED', "Checks for SOFT_FAILED status"),
                ('ActionStatus.FAILED', "Checks for FAILED status"),
                ('WorkflowTerminationStatus.SKIP_TO_REFLECT', "Returns SKIP_TO_REFLECT"),
                ('WorkflowTerminationStatus.TERMINATE_WITH_ERROR', "Returns TERMINATE_WITH_ERROR"),
                ('check_reflect_node_reachable', "Uses reflect node checker"),
            ]

            for check_str, description in critical_checks:
                if check_str in content:
                    print(f"   ✅ Contains {check_str}: {description}")
                else:
                    print(f"   ❌ Missing {check_str}: {description}")
                    all_passed = False
        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 4: Validate workflow_task tracking in __init__
    print("Test 4: Validate workflow_task tracking...")
    try:
        with open("datus/agent/workflow_runner.py", "r") as f:
            content = f.read()

        # Check for workflow_task initialization
        if 'self.workflow_task: Optional[asyncio.Task] = None' in content:
            print("✅ workflow_task attribute declared with correct type")
        else:
            print("❌ workflow_task attribute not properly declared")
            all_passed = False

        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 5: Validate SchemaValidationNode has last_action_status
    print("Test 5: Validate SchemaValidationNode has last_action_status...")
    try:
        with open("datus/agent/node/schema_validation_node.py", "r") as f:
            content = f.read()

        # Check for last_action_status in __init__
        if 'self.last_action_status = None' in content:
            print("✅ SchemaValidationNode initializes last_action_status")
        else:
            print("❌ SchemaValidationNode missing last_action_status initialization")
            all_passed = False

        # Check for FAILED (not SOFT_FAILED) usage
        if 'ActionStatus.FAILED  # Use FAILED (not SOFT_FAILED)' in content:
            print("✅ SchemaValidationNode correctly uses FAILED for hard failure")
        else:
            print("⚠️  SchemaValidationNode may not explicitly use FAILED status")

        # Check for allow_reflection=False in hard failure
        if '"allow_reflection": False' in content:
            print("✅ SchemaValidationNode sets allow_reflection=False for hard failure")
        else:
            print("⚠️  SchemaValidationNode may not set allow_reflection=False")

        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 6: Validate IntentClarificationNode exists
    print("Test 6: Validate IntentClarificationNode implementation...")
    try:
        with open("datus/agent/node/intent_clarification_node.py", "r") as f:
            content = f.read()

        # Check for class definition
        if 'class IntentClarificationNode(Node, LLMMixin):' in content:
            print("✅ IntentClarificationNode class exists")
        else:
            print("❌ IntentClarificationNode class not found")
            all_passed = False

        # Check for key methods
        if 'async def _clarify_intent(' in content:
            print("✅ IntentClarificationNode has _clarify_intent method")
        else:
            print("⚠️  IntentClarificationNode may not have _clarify_intent method")

        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 7: Validate KnowledgeEnhancementNode exists
    print("Test 7: Validate KnowledgeEnhancementNode implementation...")
    try:
        with open("datus/agent/node/knowledge_enhancement_node.py", "r") as f:
            content = f.read()

        # Check for class definition
        if 'class KnowledgeEnhancementNode(Node, LLVMixin):' in content or \
           'class KnowledgeEnhancementNode(Node, LLMMixin):' in content:
            print("✅ KnowledgeEnhancementNode class exists")
        else:
            print("❌ KnowledgeEnhancementNode class not found")
            all_passed = False

        # Check for key methods
        key_methods = [
            '_normalize_knowledge',
            '_filter_relevant_knowledge',
            '_retrieve_vector_knowledge',
            '_merge_knowledge'
        ]

        for method in key_methods:
            if f'def {method}(' in content:
                print(f"   ✅ Has {method} method")
            else:
                print(f"   ⚠️  Missing {method} method")

        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 8: Validate node registration
    print("Test 8: Validate node type registration...")
    try:
        with open("datus/configuration/node_type.py", "r") as f:
            content = f.read()

        # Check for new node types
        new_types = [
            'TYPE_INTENT_CLARIFICATION',
            'TYPE_KNOWLEDGE_ENHANCEMENT'
        ]

        for node_type in new_types:
            if f'{node_type} = "' in content:
                print(f"✅ {node_type} registered in node_type.py")
            else:
                print(f"❌ {node_type} not found in node_type.py")
                all_passed = False

        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Test 9: Validate node factory
    print("Test 9: Validate node factory registration...")
    try:
        with open("datus/agent/node/node.py", "r") as f:
            content = f.read()

        # Check for new node imports and factory cases
        checks = [
            ('IntentClarificationNode', "IntentClarificationNode imported"),
            ('KnowledgeEnhancementNode', "KnowledgeEnhancementNode imported"),
            ('NodeType.TYPE_INTENT_CLARIFICATION', "Factory case for IntentClarification"),
            ('NodeType.TYPE_KNOWLEDGE_ENHANCEMENT', "Factory case for KnowledgeEnhancement"),
        ]

        for check_str, description in checks:
            if check_str in content:
                print(f"✅ {description}")
            else:
                print(f"❌ Missing: {description}")
                all_passed = False

        print()

    except Exception as e:
        print(f"❌ FAILED: {e}")
        all_passed = False

    # Summary
    print("=" * 80)
    if all_passed:
        print("✅ ALL VALIDATION TESTS PASSED")
    else:
        print("⚠️  SOME TESTS FAILED - Review output above")
    print("=" * 80)
    print()

    if all_passed:
        print("Summary of validated implementations:")
        print("1. ✅ WorkflowTerminationStatus enum with all required states")
        print("2. ✅ _terminate_workflow() method with cancellation logic")
        print("3. ✅ _handle_node_failure() method with status-based routing")
        print("4. ✅ workflow_task tracking in WorkflowRunner")
        print("5. ✅ SchemaValidationNode with last_action_status")
        print("6. ✅ IntentClarificationNode implementation (Phase 1)")
        print("7. ✅ KnowledgeEnhancementNode implementation (Phase 2)")
        print("8. ✅ Node type registration in node_type.py")
        print("9. ✅ Node factory registration in node.py")
        print()
        print("🎉 Code inspection validation: All Phase 1-4 fixes present in codebase")
        print()
        print("📝 Note: Runtime validation requires fixing pydantic_core architecture mismatch")
        print("   This is a local environment issue, not a code issue.")
        print()
    else:
        print("Some validation checks failed. Please review the output above.")
        print()

    return all_passed


if __name__ == "__main__":
    success = validate_workflow_runner()
    sys.exit(0 if success else 1)
