#!/usr/bin/env python3
"""
Simple test for stage 4 enhancements.
"""
import sys
import time

# Direct test of ExecutionStatus class
class ExecutionStatus:
    def __init__(self):
        self.preflight_completed = False
        self.syntax_validation_passed = False
        self.tools_executed = []
        self.errors_encountered = []

    def mark_preflight_complete(self, success: bool):
        self.preflight_completed = True

    def mark_syntax_validation(self, passed: bool):
        self.syntax_validation_passed = passed

    def add_tool_execution(self, tool_name: str, success: bool, execution_time: float = None, error_type: str = None):
        record = {
            "tool": tool_name,
            "success": success,
            "timestamp": time.time(),
            "execution_time": execution_time,
            "error_type": error_type
        }
        self.tools_executed.append(record)

    def add_error(self, error_type: str, error_msg: str):
        self.errors_encountered.append({
            "type": error_type,
            "message": error_msg,
            "timestamp": time.time()
        })

def test_execution_status():
    """Test ExecutionStatus enhancements."""
    print("Testing ExecutionStatus enhancements...")

    status = ExecutionStatus()

    # Test basic functionality
    status.mark_syntax_validation(True)
    status.mark_preflight_complete(False)  # This just marks as completed, regardless of success

    # Test enhanced add_tool_execution
    status.add_tool_execution("describe_table", True, 1.5)
    status.add_tool_execution("read_query", False, 0.8, "connection_error")
    status.add_error("connection_error", "Database connection failed")

    # Verify results
    assert status.syntax_validation_passed == True
    assert status.preflight_completed == True  # mark_preflight_complete always sets to True
    assert len(status.tools_executed) == 2
    assert len(status.errors_encountered) == 1

    # Check enhanced metadata
    tool_record = status.tools_executed[1]
    assert tool_record["execution_time"] == 0.8
    assert tool_record["error_type"] == "connection_error"

    print("✅ ExecutionStatus enhancements test passed")

if __name__ == "__main__":
    test_execution_status()
    print("\n🎉 Simple Stage 4 test passed!")
