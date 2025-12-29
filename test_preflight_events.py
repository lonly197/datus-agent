#!/usr/bin/env python3
"""
Test script for preflight events functionality.
"""
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unittest.mock import AsyncMock, MagicMock
from datus.agent.node.chat_agentic_node import ChatAgenticNode, ExecutionStatus
from datus.api.models import DeepResearchEventType

async def test_preflight_events():
    """Test preflight events functionality."""
    print("Testing preflight events functionality...")

    # Create a mock ChatAgenticNode instance
    node = ChatAgenticNode(
        node_id="test_node",
        description="Test node",
        node_type="chat",
        agent_config=MagicMock()
    )

    # Mock emit_queue
    node.emit_queue = AsyncMock()

    # Test ExecutionStatus
    print("1. Testing ExecutionStatus...")
    status = ExecutionStatus()
    status.mark_syntax_validation(True)
    status.add_tool_execution("describe_table", True)
    status.add_error("test_error", "test message")

    assert status.syntax_validation_passed == True
    assert len(status.tools_executed) == 1
    assert len(status.errors_encountered) == 1
    print("✓ ExecutionStatus works correctly")

    # Test event sending methods
    print("2. Testing event sending methods...")

    # Mock workflow
    workflow = MagicMock()
    workflow.metadata = {"required_tool_sequence": ["describe_table", "read_query"]}

    # Test _send_preflight_plan_update
    await node._send_preflight_plan_update(workflow, ["describe_table", "read_query"])

    # Check if event was queued
    assert node.emit_queue.put.called
    call_args = node.emit_queue.put.call_args_list[0][0][0]
    assert call_args.event == DeepResearchEventType.PLAN_UPDATE
    assert len(call_args.todos) == 2
    print("✓ _send_preflight_plan_update works correctly")

    # Test _send_tool_call_event
    node.emit_queue.reset_mock()
    await node._send_tool_call_event(
        tool_name="describe_table",
        tool_call_id="test_call_123",
        input_data={"table": "test_table"}
    )

    call_args = node.emit_queue.put.call_args_list[0][0][0]
    assert call_args.event == DeepResearchEventType.TOOL_CALL
    assert call_args.toolCallId == "test_call_123"
    assert call_args.toolName == "describe_table"
    print("✓ _send_tool_call_event works correctly")

    # Test _send_tool_call_result_event
    node.emit_queue.reset_mock()
    await node._send_tool_call_result_event(
        tool_call_id="test_call_123",
        result={"success": True, "data": "test"},
        execution_time=1.5,
        cache_hit=False
    )

    call_args = node.emit_queue.put.call_args_list[0][0][0]
    assert call_args.event == DeepResearchEventType.TOOL_CALL_RESULT
    assert call_args.toolCallId == "test_call_123"
    assert call_args.data["success"] == True
    assert call_args.data["execution_time"] == 1.5
    print("✓ _send_tool_call_result_event works correctly")

    # Test error events
    node.emit_queue.reset_mock()
    await node._send_syntax_error_event("SELECT * FROM", "Missing FROM clause")

    call_args = node.emit_queue.put.call_args_list[0][0][0]
    assert call_args.event == DeepResearchEventType.ERROR
    assert "语法错误" in call_args.error
    print("✓ _send_syntax_error_event works correctly")

    print("\n🎉 All preflight events tests passed!")

if __name__ == "__main__":
    asyncio.run(test_preflight_events())
