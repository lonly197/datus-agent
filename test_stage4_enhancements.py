#!/usr/bin/env python3
"""
Test script for stage 4 error handling enhancements.
"""
import sys
import os
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datus.agent.node.chat_agentic_node import ExecutionStatus

def test_execution_status_enhancement():
    """Test enhanced ExecutionStatus functionality."""
    print("Testing ExecutionStatus enhancements...")

    status = ExecutionStatus()

    # Test basic functionality
    status.mark_syntax_validation(True)
    status.mark_preflight_complete(False)

    # Test enhanced add_tool_execution
    status.add_tool_execution("describe_table", True, 1.5)
    status.add_tool_execution("read_query", False, 0.8, "connection_error")
    status.add_error("connection_error", "Database connection failed")

    # Verify results
    assert status.syntax_validation_passed == True
    assert status.preflight_completed == False
    assert len(status.tools_executed) == 2
    assert len(status.errors_encountered) == 1

    # Check enhanced metadata
    tool_record = status.tools_executed[1]
    assert tool_record["execution_time"] == 0.8
    assert tool_record["error_type"] == "connection_error"

    print("✅ ExecutionStatus enhancements test passed")

async def test_error_event_methods():
    """Test error event sending methods (mocked)."""
    print("Testing error event methods...")

    from datus.agent.node.chat_agentic_node import ChatAgenticNode
    from unittest.mock import AsyncMock, MagicMock

    # Create a mock ChatAgenticNode instance
    node = ChatAgenticNode.__new__(ChatAgenticNode)
    node.emit_queue = AsyncMock()

    # Test error event methods
    await node._send_permission_error_event("SELECT * FROM test", "Access denied")
    await node._send_timeout_error_event("SELECT * FROM test", "Query timed out")
    await node._send_table_not_found_error_event("test_table", "Table does not exist")

    # Verify events were queued
    assert node.emit_queue.put.call_count == 3
    print("✅ Error event methods test passed")

async def test_error_dispatch():
    """Test error dispatch logic."""
    print("Testing error dispatch logic...")

    from datus.agent.node.chat_agentic_node import ChatAgenticNode
    from unittest.mock import AsyncMock, MagicMock

    # Create a mock ChatAgenticNode instance
    node = ChatAgenticNode.__new__(ChatAgenticNode)
    node.emit_queue = AsyncMock()

    # Mock error event methods
    node._send_permission_error_event = AsyncMock()
    node._send_timeout_error_event = AsyncMock()
    node._send_table_not_found_error_event = AsyncMock()
    node._send_db_connection_error_event = AsyncMock()

    # Test dispatch logic
    await node._dispatch_error_event("permission_error", "SELECT * FROM test", "Access denied", "describe_table", ["test_table"])
    await node._dispatch_error_event("timeout_error", "SELECT * FROM test", "Query timed out", "read_query", ["test_table"])
    await node._dispatch_error_event("table_not_found", "SELECT * FROM test", "Table not found", "describe_table", ["test_table"])
    await node._dispatch_error_event("connection_error", "SELECT * FROM test", "Connection failed", "read_query", ["test_table"])
    await node._dispatch_error_event("unknown_error", "SELECT * FROM test", "Unknown error", "unknown_tool", ["test_table"])

    # Verify correct methods were called
    assert node._send_permission_error_event.call_count == 1
    assert node._send_timeout_error_event.call_count == 1
    assert node._send_table_not_found_error_event.call_count == 1
    assert node._send_db_connection_error_event.call_count == 1

    print("✅ Error dispatch logic test passed")

async def main():
    """Run all tests."""
    print("Running Stage 4 enhancement tests...\n")

    test_execution_status_enhancement()
    await test_error_event_methods()
    await test_error_dispatch()

    print("\n🎉 All Stage 4 enhancement tests passed!")

if __name__ == "__main__":
    asyncio.run(main())
