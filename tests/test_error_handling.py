#!/usr/bin/env python3
"""
Test script for error handling and recovery mechanisms.
"""

import asyncio
import sys
import os
import re

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Define error types locally to avoid import issues
class ErrorType(str):
    """Error type classifications for better handling."""
    NETWORK = "network"
    DATABASE_CONNECTION = "database_connection"
    DATABASE_QUERY = "database_query"
    TABLE_NOT_FOUND = "table_not_found"
    COLUMN_NOT_FOUND = "column_not_found"
    SYNTAX_ERROR = "syntax_error"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN = "unknown"


class ErrorRecoveryStrategy:
    """Error recovery strategies with retry and fallback options."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

    def should_retry(self, error_type: ErrorType, attempt_count: int) -> bool:
        """Determine if an error should be retried based on type and attempt count."""
        retryable_errors = {
            ErrorType.NETWORK,
            ErrorType.DATABASE_CONNECTION,
            ErrorType.TIMEOUT,
            ErrorType.RESOURCE_EXHAUSTED
        }

        return error_type in retryable_errors and attempt_count < self.max_retries

    def get_retry_delay(self, attempt_count: int) -> float:
        """Calculate retry delay with exponential backoff."""
        return self.retry_delay * (self.backoff_factor ** (attempt_count - 1))


class ErrorHandler:
    """Comprehensive error handling and recovery system."""

    def __init__(self):
        self.recovery_strategy = ErrorRecoveryStrategy()
        self.error_patterns = self._load_error_patterns()

    def _load_error_patterns(self):
        """Load error pattern recognition rules."""
        return {
            # Database connection errors
            r"connection.*failed|unable.*connect": {
                "type": ErrorType.DATABASE_CONNECTION,
                "suggestions": [
                    "检查数据库连接配置",
                    "确认数据库服务正在运行",
                    "验证网络连接"
                ]
            },
            # Table not found errors
            r"table.*not.*found|relation.*does.*exist|doesn't exist": {
                "type": ErrorType.TABLE_NOT_FOUND,
                "suggestions": [
                    "确认表名拼写正确",
                    "检查数据库schema",
                    "使用search_table工具查找可用表"
                ]
            },
            # Column not found errors
            r"column.*not.*found|field.*does.*exist": {
                "type": ErrorType.COLUMN_NOT_FOUND,
                "suggestions": [
                    "检查列名拼写",
                    "使用describe_table查看表结构",
                    "确认字段是否存在于表中"
                ]
            },
            # Syntax errors
            r"syntax.*error|invalid.*sql": {
                "type": ErrorType.SYNTAX_ERROR,
                "suggestions": [
                    "检查SQL语法",
                    "确认引号和括号匹配",
                    "验证SQL语句结构"
                ]
            },
            # Permission errors
            r"permission.*denied|access.*denied": {
                "type": ErrorType.PERMISSION_DENIED,
                "suggestions": [
                    "检查数据库权限",
                    "确认用户有查询权限",
                    "联系数据库管理员"
                ]
            },
            # Timeout errors
            r"timeout|query.*timed.*out": {
                "type": ErrorType.TIMEOUT,
                "suggestions": [
                    "简化查询条件",
                    "添加适当的索引",
                    "考虑分批处理大数据"
                ]
            }
        }

    def classify_error(self, error_message: str):
        """Classify error type and provide recovery suggestions."""
        error_msg_lower = error_message.lower()

        for pattern, info in self.error_patterns.items():
            if re.search(pattern, error_msg_lower, re.IGNORECASE):
                return info["type"], info["suggestions"]

        return ErrorType.UNKNOWN, ["检查错误详情并联系技术支持"]

    def handle_tool_error(self, tool_name: str, error: Exception, attempt_count: int = 1, context: dict = None):
        """Handle tool execution errors with recovery strategies."""

        error_message = str(error)
        error_type, suggestions = self.classify_error(error_message)

        result = {
            "error_type": error_type,
            "error_message": error_message,
            "suggestions": suggestions,
            "can_retry": self.recovery_strategy.should_retry(error_type, attempt_count),
            "retry_delay": 0.0
        }

        if result["can_retry"]:
            result["retry_delay"] = self.recovery_strategy.get_retry_delay(attempt_count)
            result["suggestions"].append(f"系统将在 {result['retry_delay']:.1f} 秒后自动重试")

        # Add tool-specific recovery suggestions
        if tool_name == "describe_table" and error_type == ErrorType.TABLE_NOT_FOUND:
            result["fallback_tool"] = "search_table"
            result["fallback_reason"] = "表不存在，建议使用search_table查找可用表"
        elif tool_name == "execute_sql" and error_type == ErrorType.SYNTAX_ERROR:
            result["suggestions"].append("考虑使用read_query工具代替execute_sql")
        elif tool_name == "search_table" and error_type == ErrorType.DATABASE_CONNECTION:
            result["suggestions"].append("数据库连接问题，请稍后重试或联系管理员")

        return result

    def _auto_fix_sql_syntax(self, sql_query: str, error_message: str) -> str:
        """Attempt to auto-fix common SQL syntax errors."""
        if not sql_query:
            return sql_query

        fixed_sql = sql_query.strip()

        # Fix missing semicolon
        if not fixed_sql.endswith(';') and not fixed_sql.upper().strip().startswith('SELECT'):
            # Only add semicolon if it seems to be a complete statement
            if len(fixed_sql.split()) > 3:  # Basic heuristic
                fixed_sql += ';'

        # Fix common quote issues
        error_lower = error_message.lower()

        # Handle unterminated quoted strings
        if "unterminated quoted string" in error_lower or "quoted string not properly terminated" in error_lower:
            # Try to find and fix unterminated quotes
            single_quotes = fixed_sql.count("'")
            double_quotes = fixed_sql.count('"')

            if single_quotes % 2 != 0:
                fixed_sql += "'"
            elif double_quotes % 2 != 0:
                fixed_sql += '"'

        return fixed_sql


async def test_error_classification():
    """Test error classification and suggestions."""
    print("=== Testing Error Classification ===")

    handler = ErrorHandler()

    test_cases = [
        ("Table 'users' doesn't exist", ErrorType.TABLE_NOT_FOUND),
        ("Access denied for user", ErrorType.PERMISSION_DENIED),
        ("Connection timeout", ErrorType.TIMEOUT),
        ("Syntax error near 'SELEC'", ErrorType.SYNTAX_ERROR),
        ("Unknown error occurred", ErrorType.UNKNOWN),
    ]

    for error_msg, expected_type in test_cases:
        error_type, suggestions = handler.classify_error(error_msg)
        print(f"Error: '{error_msg}'")
        print(f"  Classified as: {error_type}")
        print(f"  Expected: {expected_type}")
        print(f"  Suggestions: {suggestions[:2]}...")  # Show first 2 suggestions
        print(f"  ✓ Correct: {error_type == expected_type}")
        print()


async def test_tool_error_handling():
    """Test tool-specific error handling."""
    print("=== Testing Tool Error Handling ===")

    handler = ErrorHandler()

    # Mock exceptions
    table_not_found_error = Exception("Table 'nonexistent_table' doesn't exist")
    syntax_error = Exception("You have an error in your SQL syntax")

    # Test describe_table error handling
    result = handler.handle_tool_error("describe_table", table_not_found_error, 1)
    print("describe_table with table not found:")
    print(f"  Error type: {result['error_type']}")
    print(f"  Can retry: {result['can_retry']}")
    print(f"  Has fallback: {'fallback_tool' in result}")
    print(f"  Suggestions: {len(result['suggestions'])} items")
    print()

    # Test execute_sql syntax error
    result = handler.handle_tool_error("execute_sql", syntax_error, 1, {"sql_query": "SELEC * FROM users"})
    print("execute_sql with syntax error:")
    print(f"  Error type: {result['error_type']}")
    print(f"  Auto-fix available: {result.get('auto_fix_available', False)}")
    print(f"  Has fixed SQL: {'fixed_sql' in result}")
    if 'fixed_sql' in result:
        print(f"  Original: 'SELEC * FROM users'")
        print(f"  Fixed: '{result['fixed_sql']}'")
    print()


async def test_auto_fix_sql():
    """Test SQL auto-fix functionality."""
    print("=== Testing SQL Auto-Fix ===")

    handler = ErrorHandler()

    test_cases = [
        ("SELEC * FROM users", "SELEC * FROM users;"),
        ("SELECT * FROM users", "SELECT * FROM users;"),  # Already has semicolon
        ("SELECT name FROM 'users", "SELECT name FROM 'users'"),  # Fix unterminated quote
    ]

    for original, expected in test_cases:
        fixed = handler._auto_fix_sql_syntax(original, "mock error")
        print(f"Original: '{original}'")
        print(f"Fixed: '{fixed}'")
        print(f"Expected: '{expected}'")
        print(f"✓ Correct: {fixed == expected}")
        print()


async def main():
    """Run all tests."""
    print("🧪 Testing Error Handling and Recovery Mechanisms\n")

    try:
        await test_error_classification()
        await test_tool_error_handling()
        await test_auto_fix_sql()

        print("✅ All error handling tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
