#!/usr/bin/env python3
"""
Test script for the fixes to SQL review issues.
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_check_table_exists_fix():
    """Test the fixed check_table_exists method."""
    from unittest.mock import MagicMock

    from datus.tools.func_tool.database import DBFuncTool

    # Create a mock DBFuncTool instance
    db_tool = DBFuncTool.__new__(DBFuncTool)
    db_tool.connector = MagicMock()

    # Mock get_tables to return a list of strings (as it should)
    db_tool.connector.get_tables.return_value = ["users", "orders", "products"]

    # Test table exists
    result = db_tool.check_table_exists("users")
    assert result.success == 1
    assert result.result["table_exists"] == True
    print("✅ Table exists check passed")

    # Test table doesn't exist with suggestions
    result = db_tool.check_table_exists("user")  # Should suggest "users"
    assert result.success == 1
    assert result.result["table_exists"] == False
    assert "users" in result.result["suggestions"]
    print("✅ Table not found with suggestions check passed")

    # Test with empty table list
    db_tool.connector.get_tables.return_value = []
    result = db_tool.check_table_exists("test_table")
    assert result.success == 1
    assert result.result["table_exists"] == False
    assert result.result["available_tables"] == []
    print("✅ Empty table list check passed")

    # Test exception handling
    db_tool.connector.get_tables.side_effect = Exception("Connection failed")
    result = db_tool.check_table_exists("test_table")
    assert result.success == 0
    assert "Connection failed" in result.error
    print("✅ Exception handling check passed")


if __name__ == "__main__":
    test_check_table_exists_fix()
    print("\n🎉 All fixes verified!")
