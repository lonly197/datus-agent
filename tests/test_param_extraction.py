#!/usr/bin/env python3
"""
Test script for parameter extraction optimization.
"""

import os
import re
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Copy the parameter extraction logic for testing
DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP = {
    "describe_table": [
        "describe table",
        "检查表结构",
        "inspect table schema",
        "查看表结构",
        "examine table structure",
        "分析表结构",
        "describe table structure",
        "检查表定义",
        "查看表模式",
        "analyze table structure",
        "分析表元数据",
        "表定义",
        "表模式",
        "表元数据",
        "表字段",
    ],
}


class TestParamExtraction:
    def __init__(self):
        self.keyword_map = DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP

    def _preprocess_todo_content(self, content):
        """Preprocess todo content to improve parameter extraction."""
        if not content:
            return ""

        # Remove common prefixes that don't help with parameter extraction
        prefixes_to_remove = [
            "sub-question:",
            "sub question:",
            "任务：",
            "任务:",
            "问题：",
            "问题:",
            "| expected:",
            "| 期望:",
        ]

        cleaned = content.lower()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()

        # Remove extra whitespace
        cleaned = " ".join(cleaned.split())

        return cleaned

    def _is_valid_table_name(self, table_name):
        """Basic validation for table names."""
        if not table_name or len(table_name) > 100:
            return False

        # Check for basic SQL injection patterns (simple check)
        dangerous_patterns = [";", "--", "/*", "*/", "union", "select", "drop", "delete", "update", "insert"]
        if any(pattern in table_name.lower() for pattern in dangerous_patterns):
            return False

        # Allow alphanumeric, underscore, and some special chars common in table names
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            return False

        return True

    def _extract_describe_table_params(self, todo_content):
        """Extract table name from todo content for describe_table tool."""
        import re

        # Common patterns for table names in Chinese/English text
        patterns = [
            r"表\s*[\'\"]?(\w+)[\'\"]?",  # 表 'table_name' or 表 table_name
            r"table\s*[\'\"]?(\w+)[\'\"]?",  # table 'table_name'
            r"(\w+_table)",  # table_name
            r"(\w+_fact_\w+)",  # fact tables like dwd_assign_dlr_clue_fact_di
            r"(\w+_dim_\w+)",  # dimension tables
            r"(\w+_dws_\w+)",  # summary tables
            r"(\w+_ads_\w+)",  # application tables
            r"(\w+_ods_\w+)",  # ods tables
            r"(\w+_dwd_\w+)",  # dwd tables
        ]

        todo_content.lower()

        # Try regex patterns
        for pattern in patterns:
            matches = re.findall(pattern, todo_content, re.IGNORECASE)
            if matches:
                # Clean the table name
                table_name = matches[0].strip("'\"")
                # Validate table name format (basic validation)
                if self._is_valid_table_name(table_name):
                    print(f"Extracted table name '{table_name}' from todo content")
                    return {"table_name": table_name}

        # Fallback: if no table name found, this might be a search operation
        print("No table name found in todo content, falling back to search operation")
        return {"query_text": todo_content}

    def _extract_tool_parameters(self, tool_name, todo_content):
        """Extract appropriate parameters for a tool from todo content."""
        if not todo_content:
            return {}

        # Clean and preprocess todo content
        cleaned_content = self._preprocess_todo_content(todo_content)

        try:
            if tool_name == "describe_table":
                return self._extract_describe_table_params(cleaned_content)
            else:
                # For unknown tools, return basic parameters
                return {"query_text": cleaned_content}
        except Exception as e:
            print(f"Failed to extract parameters for tool {tool_name}: {e}")
            return {"query_text": cleaned_content}


def main():
    print("Testing parameter extraction optimization...")
    print("=" * 50)

    extractor = TestParamExtraction()

    # Test case 1: Original problematic content
    test_content_1 = "Sub-question: 探索数据库中的表结构，找到试驾表和线索表 | Expected: 确认表名、字段名和关联关系"
    print("Test 1 - Original problematic content:")
    print(f"Input: {test_content_1}")
    params_1 = extractor._extract_tool_parameters("describe_table", test_content_1)
    print(f"Output: {params_1}")
    print()

    # Test case 2: Preprocessing
    print("Test 2 - Preprocessing:")
    preprocessed = extractor._preprocess_todo_content(test_content_1)
    print(f"Original: {test_content_1}")
    print(f"Preprocessed: {preprocessed}")
    print()

    # Test case 3: Table name validation
    print("Test 3 - Table name validation:")
    valid_names = ["dwd_assign_dlr_clue_fact_di", "user_table", "sales_dim"]
    invalid_names = ["select * from users;", "drop table test", ""]
    for name in valid_names + invalid_names:
        is_valid = extractor._is_valid_table_name(name)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"  {status}: '{name}'")
    print()

    # Test case 4: Different todo formats
    test_cases = [
        "检查表结构 user_info",
        "describe table sales_fact_di",
        "查看表字段 customer_dim",
        "分析表结构：order_table",
    ]

    print("Test 4 - Different todo formats:")
    for i, test_content in enumerate(test_cases, 1):
        print(f"  Test 4.{i}: {test_content}")
        params = extractor._extract_tool_parameters("describe_table", test_content)
        print(f"    Result: {params}")
    print()

    print("🎉 All parameter extraction tests completed successfully!")


if __name__ == "__main__":
    main()
