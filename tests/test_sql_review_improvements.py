#!/usr/bin/env python3
"""
Test script for SQL review improvements.
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_sql_extraction():
    """Test improved SQL extraction logic."""
    from datus.agent.node.chat_agentic_node import ChatAgenticNode

    # Create a mock instance to test the method
    node = ChatAgenticNode.__new__(ChatAgenticNode)

    # Test cases
    test_cases = [
        # Case 1: SQL with Chinese prefix
        {
            "input": """你是SQL审查专家。
待审查SQL：SELECT * FROM users WHERE id = 1;""",
            "expected": "SELECT * FROM users WHERE id = 1",
            "description": "Chinese prefix SQL",
        },
        # Case 2: SQL in backticks
        {
            "input": """审查这个SQL：
```sql
SELECT name, email FROM customers WHERE active = 1;
```""",
            "expected": "SELECT name, email FROM customers WHERE active = 1;",
            "description": "SQL in backticks",
        },
        # Case 3: Problematic case from logs (should be avoided)
        {
            "input": """待审查SQL：SELECT * FROM dwd_assign_dlr_clue_fact_di WHERE LEFT(clue_create_time, 10) = '2025-12-24' AND newest_original_clue_json LIKE '%线上%'""",
            "expected": "SELECT * FROM dwd_assign_dlr_clue_fact_di WHERE LEFT(clue_create_time, 10) = '2025-12-24' AND newest_original_clue_json LIKE '%线上%'",
            "description": "Clean SQL extraction",
        },
    ]

    print("Testing SQL extraction improvements...")

    for i, test_case in enumerate(test_cases, 1):
        result = node._extract_sql_from_task(test_case["input"])
        print(f"Test {i} ({test_case['description']}): {'✅ PASS' if result == test_case['expected'] else '❌ FAIL'}")
        if result != test_case["expected"]:
            print(f"  Expected: {test_case['expected']}")
            print(f"  Got:      {result}")

    print()


def test_invalid_content_detection():
    """Test invalid mixed content detection."""
    from datus.agent.node.chat_agentic_node import ChatAgenticNode

    node = ChatAgenticNode.__new__(ChatAgenticNode)

    test_cases = [
        # Should be invalid (Chinese mixed with SQL)
        {
            "input": "SELECT *禁止分区裁剪等 FROM table1",
            "expected": True,
            "description": "Chinese mixed with SQL keywords",
        },
        # Should be valid (clean SQL)
        {"input": "SELECT * FROM users WHERE id = 1", "expected": False, "description": "Clean SQL"},
        # Should be invalid (too many Chinese chars)
        {
            "input": "选择所有数据从用户表中获取WHERE条件是激活的用户状态",
            "expected": True,
            "description": "Too many Chinese characters",
        },
    ]

    print("Testing invalid content detection...")

    for i, test_case in enumerate(test_cases, 1):
        result = node._contains_invalid_mixed_content(test_case["input"])
        print(f"Test {i} ({test_case['description']}): {'✅ PASS' if result == test_case['expected'] else '❌ FAIL'}")

    print()


def test_tool_failure_impact_analysis():
    """Test tool failure impact analysis."""
    from datus.agent.node.chat_agentic_node import ChatAgenticNode

    node = ChatAgenticNode.__new__(ChatAgenticNode)

    test_cases = [
        {
            "tool_name": "describe_table",
            "result": {"success": False, "error": "Table not found"},
            "expected_contains": "表结构分析受限",
            "description": "describe_table failure",
        },
        {
            "tool_name": "read_query",
            "result": {"success": False, "error": "Syntax error detected"},
            "expected_contains": "语法验证失败",
            "description": "read_query syntax error",
        },
        {
            "tool_name": "search_external_knowledge",
            "result": {"success": False, "error": "Network timeout"},
            "expected_contains": "规则检索失败",
            "description": "knowledge search failure",
        },
        {
            "tool_name": "describe_table",
            "result": {"success": True},
            "expected_contains": "无影响",
            "description": "successful tool execution",
        },
    ]

    print("Testing tool failure impact analysis...")

    for i, test_case in enumerate(test_cases, 1):
        result = node._analyze_tool_failure_impact(test_case["tool_name"], test_case["result"])
        contains_expected = test_case["expected_contains"] in result
        print(f"Test {i} ({test_case['description']}): {'✅ PASS' if contains_expected else '❌ FAIL'}")
        if not contains_expected:
            print(f"  Expected to contain: {test_case['expected_contains']}")
            print(f"  Got: {result}")

    print()


if __name__ == "__main__":
    test_sql_extraction()
    test_invalid_content_detection()
    test_tool_failure_impact_analysis()
    print("🎉 All SQL review improvement tests completed!")
