#!/usr/bin/env python3
"""
Test script for PreflightOrchestrator functionality.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

# Test basic imports
try:
    from datus.agent.node.preflight_orchestrator import PreflightToolResult
    print("✓ PreflightToolResult imported successfully")
except Exception as e:
    print(f"✗ Failed to import PreflightToolResult: {e}")
    sys.exit(1)


class MockAgentConfig:
    """Mock agent configuration for testing."""
    def __init__(self):
        self.current_catalog = "default_catalog"
        self.current_database = "test"
        self.current_schema = ""


class MockWorkflow:
    """Mock workflow for testing."""
    def __init__(self):
        self.task = MockTask()


class MockTask:
    """Mock task for testing."""
    def __init__(self):
        self.task = "统计每个月'首次试驾'到'下定'的平均转化周期（天数）"
        self.catalog_name = "default_catalog"
        self.database_name = "test"
        self.schema_name = ""


# Test PreflightToolResult
print("Testing PreflightToolResult...")

result = PreflightToolResult(
    tool_name="test_tool",
    success=True,
    result={"test": "data"},
    error=None,
    execution_time=1.5,
    cache_hit=False
)

print("✓ PreflightToolResult created successfully")

# Test to_dict method
result_dict = result.to_dict()
expected_keys = ["tool_name", "success", "result", "error", "execution_time", "cache_hit"]
if all(key in result_dict for key in expected_keys):
    print("✓ PreflightToolResult.to_dict() works correctly")
else:
    print("✗ PreflightToolResult.to_dict() missing keys")

print("Basic PreflightToolResult test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_preflight_orchestrator())