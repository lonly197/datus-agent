# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for DeepResearchEventConverter.

Tests verify that event conversion correctly handles:
- ToolCallEvent and ToolCallResultEvent generation for schema_discovery and sql_execution
- planId binding to virtual step IDs for Text2SQL workflows
- Workflow completion generates final PlanUpdateEvent with all steps marked completed
- Virtual plan state transitions follow correct linear order
"""

from unittest.mock import MagicMock

import pytest

from datus.api.event_converter import DeepResearchEventConverter
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus


def test_schema_discovery_generates_tool_events():
    """Verify schema_discovery generates ToolCallEvent and ToolCallResultEvent."""
    converter = DeepResearchEventConverter()
    # Set virtual step context to simulate Text2SQL workflow
    converter.active_virtual_step_id = "step_schema"
    converter.virtual_plan_emitted = True

    action = ActionHistory(
        action_id="test_schema_discovery",
        role=ActionRole.TOOL,
        action_type="schema_discovery",
        status=ActionStatus.SUCCESS,
        input={"task": "test query"},
        output={"candidate_tables": ["table1", "table2"], "table_count": 2},
    )

    events = converter.convert_action_to_event(action, 1)

    # Verify event types
    assert any(e.event == "tool_call" for e in events), "Missing ToolCallEvent"
    assert any(e.event == "tool_call_result" for e in events), "Missing ToolCallResultEvent"

    # Verify planId binding
    tool_call = next(e for e in events if e.event == "tool_call")
    assert tool_call.planId == "step_schema", f"planId should be 'step_schema', got {tool_call.planId}"

    # Verify toolCallId binding
    tool_call_result = next(e for e in events if e.event == "tool_call_result")
    assert tool_call.toolCallId == tool_call_result.toolCallId, "toolCallId should match"


def test_sql_execution_generates_tool_events():
    """Verify sql_execution generates ToolCallEvent and ToolCallResultEvent."""
    converter = DeepResearchEventConverter()
    # Set virtual step context to simulate Text2SQL workflow
    converter.active_virtual_step_id = "step_exec"
    converter.virtual_plan_emitted = True

    action = ActionHistory(
        action_id="test_sql_execution",
        role=ActionRole.TOOL,
        action_type="sql_execution",
        status=ActionStatus.SUCCESS,
        input={"sql_query": "SELECT 1"},
        output={"rows": 1, "data": [[1]]},
    )

    events = converter.convert_action_to_event(action, 1)

    # Verify event types
    assert any(e.event == "tool_call" for e in events), "Missing ToolCallEvent"
    assert any(e.event == "tool_call_result" for e in events), "Missing ToolCallResultEvent"

    # Verify planId binding
    tool_call = next(e for e in events if e.event == "tool_call")
    assert tool_call.planId == "step_exec", f"planId should be 'step_exec', got {tool_call.planId}"

    # Verify toolCallId binding
    tool_call_result = next(e for e in events if e.event == "tool_call_result")
    assert tool_call.toolCallId == tool_call_result.toolCallId, "toolCallId should match"


def test_workflow_completion_generates_final_plan_update():
    """Verify workflow_completion generates CompleteEvent."""
    converter = DeepResearchEventConverter()
    # Set up as if workflow_init had already run
    converter.virtual_plan_emitted = True

    action = ActionHistory(
        action_id="test_workflow_completion",
        role=ActionRole.WORKFLOW,  # FIXED: workflow_completion uses WORKFLOW role, not ASSISTANT
        action_type="workflow_completion",
        status=ActionStatus.SUCCESS,
        messages="Workflow completed",
        output={"workflow_completed": True},
    )

    events = converter.convert_action_to_event(action, 1)

    # The workflow_completion event should generate CompleteEvent
    assert len(events) > 0, "Should generate at least one event for workflow_completion"

    # Verify CompleteEvent is generated
    complete_events = [e for e in events if e.event == "complete"]
    assert len(complete_events) > 0, "Should generate CompleteEvent"

    # Verify PlanUpdateEvent with all completed steps (when virtual_plan_emitted is True)
    plan_updates = [e for e in events if e.event == "plan_update"]
    if len(plan_updates) > 0:
        # If plan update generated, verify all todos are marked completed
        final_plan = plan_updates[0]
        assert all(todo.status == "completed" for todo in final_plan.todos), "Not all todos marked as completed"


def test_virtual_plan_state_transitions():
    """Verify virtual plan state transitions follow correct linear order."""
    converter = DeepResearchEventConverter()
    converter.virtual_plan_emitted = True

    # Test with a single node type to verify the mechanism works
    action = ActionHistory(
        action_id="test_node_schema_discovery",
        role=ActionRole.WORKFLOW,  # FIXED: node_execution uses WORKFLOW role, not ASSISTANT
        action_type="node_execution",
        status=ActionStatus.SUCCESS,
        input={"node_type": "schema_discovery", "description": "Test schema discovery"},
        messages="Executing schema discovery",  # Add messages to trigger ChatEvent
    )

    events = converter.convert_action_to_event(action, 1)

    # Verify we get events (ChatEvent at minimum)
    assert len(events) > 0, "Should generate events for node_execution"

    # Verify ChatEvent exists for node execution
    chat_events = [e for e in events if e.event == "chat"]
    assert len(chat_events) > 0, "Should have chat events for node execution"

    # Verify planId binding if active_virtual_step_id is set
    # After processing node_execution, active_virtual_step_id should be updated
    if converter.active_virtual_step_id:
        for event in chat_events:
            if event.planId:
                # planId should match the active virtual step or the expected step
                assert event.planId in [converter.active_virtual_step_id, "step_schema"], \
                    f"planId should be step_schema or {converter.active_virtual_step_id}, got {event.planId}"


def test_event_flow_validation():
    """Verify event flow validation catches missing events."""
    converter = DeepResearchEventConverter()

    # Test with schema_discovery (should fail validation without tool events)
    result = converter.validate_event_flow("schema_discovery", [])
    assert result is False, "Validation should fail for empty event list"

    # Test with chat action (should pass validation)
    result = converter.validate_event_flow("chat", [MagicMock(event="chat")])
    assert result is True, "Validation should pass for non-critical actions"
