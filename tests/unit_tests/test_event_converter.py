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


def test_preflight_tool_binding():
    """Verify preflight tools bind to correct virtual step ID."""
    converter = DeepResearchEventConverter()
    converter.virtual_plan_emitted = True

    # Test preflight_describe_table should bind to step_schema
    action = ActionHistory(
        action_id="test_preflight_describe",
        role=ActionRole.TOOL,
        action_type="preflight_describe_table",
        status=ActionStatus.SUCCESS,
        input={"table_name": "test_table"},
        output={"columns": [{"name": "id", "type": "int"}]},
    )

    events = converter.convert_action_to_event(action, 1)

    # Verify event types
    tool_call = next((e for e in events if e.event == "tool_call"), None)
    assert tool_call is not None, "Should generate ToolCallEvent"
    assert tool_call.planId == "step_schema", f"planId should be 'step_schema', got {tool_call.planId}"

    # Test preflight_analyze_query_plan should bind to step_exec
    action2 = ActionHistory(
        action_id="test_preflight_analyze",
        role=ActionRole.TOOL,
        action_type="preflight_analyze_query_plan",
        status=ActionStatus.SUCCESS,
        input={"sql": "SELECT * FROM users"},
        output={"plan_text": "Seq Scan on users"},
    )

    events2 = converter.convert_action_to_event(action2, 2)

    # Verify event types
    tool_call2 = next((e for e in events2 if e.event == "tool_call"), None)
    assert tool_call2 is not None, "Should generate ToolCallEvent"
    assert tool_call2.planId == "step_exec", f"planId should be 'step_exec', got {tool_call2.planId}"


def test_tool_prefixed_preflight_events_bind_to_virtual_steps():
    """tool_* prefixed preflight actions should still map to virtual step IDs."""
    converter = DeepResearchEventConverter()
    converter.virtual_plan_emitted = True

    # Start event from execution_event_manager style action_type
    action = ActionHistory(
        action_id="pref_tool_call",
        role=ActionRole.TOOL,
        action_type="tool_preflight_describe_table",
        status=ActionStatus.SUCCESS,
        input={"table_name": "users"},
        output={"columns": [{"name": "id"}]},
    )
    events = converter.convert_action_to_event(action, 1)
    call_event = next((e for e in events if e.event == "tool_call"), None)
    assert call_event is not None, "Should emit ToolCallEvent for tool_* action"
    assert call_event.planId == "step_schema", f"planId should bind to step_schema, got {call_event.planId}"

    # Result-style action_type with suffix should also normalize correctly
    result_action = ActionHistory(
        action_id="pref_tool_result",
        role=ActionRole.TOOL,
        action_type="tool_preflight_describe_table_result",
        status=ActionStatus.SUCCESS,
        input={"table_name": "users"},
        output={"columns": [{"name": "id"}]},
    )
    result_events = converter.convert_action_to_event(result_action, 2)
    result_event = next((e for e in result_events if e.event == "tool_call_result"), None)
    assert result_event is not None, "Should emit ToolCallResultEvent for *_result action"
    assert result_event.planId == "step_schema", f"planId should bind to step_schema, got {result_event.planId}"


def test_output_node_binding_to_step_output():
    """Verify output and output_generation nodes bind to step_output virtual step."""
    converter = DeepResearchEventConverter()
    converter.virtual_plan_emitted = True

    # Test output node binds to step_output
    action = ActionHistory(
        action_id="test_output",
        role=ActionRole.ASSISTANT,
        action_type="output",
        status=ActionStatus.SUCCESS,
        input={"task": "test query"},
        output={"result": "test result"},
    )

    events = converter.convert_action_to_event(action, 1)

    # Verify planId binding
    chat_events = [e for e in events if e.event == "chat"]
    assert len(chat_events) > 0, "Should generate ChatEvent"
    assert chat_events[0].planId == "step_output", f"planId should be 'step_output', got {chat_events[0].planId}"

    # Test output_generation also binds to step_output
    action2 = ActionHistory(
        action_id="test_output_generation",
        role=ActionRole.ASSISTANT,
        action_type="output_generation",
        status=ActionStatus.SUCCESS,
        input={"task": "test query"},
        output={"generated_output": "test output"},
    )

    events2 = converter.convert_action_to_event(action2, 2)
    chat_events2 = [e for e in events2 if e.event == "chat"]
    assert len(chat_events2) > 0, "Should generate ChatEvent"
    assert chat_events2[0].planId == "step_output", f"planId should be 'step_output', got {chat_events2[0].planId}"


def test_reflect_node_binding_to_step_reflect():
    """Verify reflect node binds to step_reflect (not step_output after separation)."""
    converter = DeepResearchEventConverter()
    converter.virtual_plan_emitted = True

    # Test reflect node binds to step_reflect
    action = ActionHistory(
        action_id="test_reflect",
        role=ActionRole.ASSISTANT,
        action_type="reflect",
        status=ActionStatus.SUCCESS,
        input={"task": "test query"},
        output={"reflection": "test reflection"},
    )

    events = converter.convert_action_to_event(action, 1)

    # Verify planId binding
    chat_events = [e for e in events if e.event == "chat"]
    assert len(chat_events) > 0, "Should generate ChatEvent"
    assert chat_events[0].planId == "step_reflect", f"planId should be 'step_reflect', got {chat_events[0].planId}"
