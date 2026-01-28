# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Error handling and recovery strategies for plan mode hooks."""

import re
from typing import Any, Dict, List, Tuple

from .types import ErrorType


class ErrorRecoveryStrategy:
    """Error recovery strategies with retry and fallback options."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, backoff_factor: float = 2.0):
        """Initialize recovery strategy.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

    def should_retry(self, error_type: ErrorType, attempt_count: int) -> bool:
        """Determine if an error should be retried based on type and attempt count.

        Args:
            error_type: Type of error that occurred
            attempt_count: Current attempt number

        Returns:
            True if error should be retried, False otherwise
        """
        retryable_errors = {
            ErrorType.NETWORK,
            ErrorType.DATABASE_CONNECTION,
            ErrorType.TIMEOUT,
            ErrorType.RESOURCE_EXHAUSTED,
        }

        return error_type in retryable_errors and attempt_count < self.max_retries

    def get_retry_delay(self, attempt_count: int) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            attempt_count: Current attempt number

        Returns:
            Delay in seconds before next retry
        """
        return self.retry_delay * (self.backoff_factor ** (attempt_count - 1))


class ErrorHandler:
    """Comprehensive error handling and recovery system."""

    def __init__(self):
        """Initialize error handler with recovery strategy and pattern matching."""
        self.recovery_strategy = ErrorRecoveryStrategy()
        self.error_patterns = self._load_error_patterns()

    def _load_error_patterns(self) -> Dict[str, Dict]:
        """Load error pattern recognition rules.

        Returns:
            Dict mapping regex patterns to error types and suggestions
        """
        return {
            # Database connection errors
            r"connection.*failed|unable.*connect": {
                "type": ErrorType.DATABASE_CONNECTION,
                "suggestions": ["检查数据库连接配置", "确认数据库服务正在运行", "验证网络连接"],
            },
            # Table not found errors
            r"table.*not.*found|relation.*does.*exist|doesn't exist": {
                "type": ErrorType.TABLE_NOT_FOUND,
                "suggestions": ["确认表名拼写正确", "检查数据库schema", "使用search_table工具查找可用表"],
            },
            # Column not found errors
            r"column.*not.*found|field.*does.*exist": {
                "type": ErrorType.COLUMN_NOT_FOUND,
                "suggestions": ["检查列名拼写", "使用describe_table查看表结构", "确认字段是否存在于表中"],
            },
            # Syntax errors
            r"syntax.*error|invalid.*sql": {
                "type": ErrorType.SYNTAX_ERROR,
                "suggestions": ["检查SQL语法", "确认引号和括号匹配", "验证SQL语句结构"],
            },
            # Permission errors
            r"permission.*denied|access.*denied": {
                "type": ErrorType.PERMISSION_DENIED,
                "suggestions": ["检查数据库权限", "确认用户有查询权限", "联系数据库管理员"],
            },
            # Timeout errors
            r"timeout|query.*timed.*out": {
                "type": ErrorType.TIMEOUT,
                "suggestions": ["简化查询条件", "添加适当的索引", "考虑分批处理大数据"],
            },
        }

    def classify_error(self, error_message: str) -> Tuple[ErrorType, List[str]]:
        """Classify error type and provide recovery suggestions.

        Args:
            error_message: Error message to classify

        Returns:
            Tuple of (error_type, suggestions)
        """
        error_msg_lower = error_message.lower()

        for pattern, info in self.error_patterns.items():
            if re.search(pattern, error_msg_lower, re.IGNORECASE):
                return info["type"], info["suggestions"]

        return ErrorType.UNKNOWN, ["检查错误详情并联系技术支持"]

    def handle_tool_error(
        self, tool_name: str, error: Exception, attempt_count: int = 1, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle tool execution errors with recovery strategies and auto-fix suggestions.

        Args:
            tool_name: Name of the tool that failed
            error: Exception that was raised
            attempt_count: Current attempt number
            context: Additional context about the error

        Returns:
            Dict with error handling information and recovery options
        """
        error_message = str(error)
        error_type, suggestions = self.classify_error(error_message)

        result = {
            "error_type": error_type,
            "error_message": error_message,
            "suggestions": suggestions,
            "can_retry": self.recovery_strategy.should_retry(error_type, attempt_count),
            "retry_delay": 0.0,
            "auto_fix_available": False,
            "auto_fix_suggestion": None,
        }

        if result["can_retry"]:
            result["retry_delay"] = self.recovery_strategy.get_retry_delay(attempt_count)
            result["suggestions"].append(f"系统将在 {result['retry_delay']:.1f} 秒后自动重试")

        # Add tool-specific recovery suggestions and auto-fix logic
        context = context or {}

        if tool_name == "describe_table":
            result.update(self._handle_describe_table_error(error_type, error_message, context))
        elif tool_name == "search_table":
            result.update(self._handle_search_table_error(error_type, error_message, context))
        elif tool_name == "execute_sql":
            result.update(self._handle_execute_sql_error(error_type, error_message, context))

        return result

    def _handle_describe_table_error(
        self, error_type: ErrorType, error_message: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle describe_table specific errors.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Error context

        Returns:
            Dict with tool-specific error handling info
        """
        result = {}

        if error_type == ErrorType.TABLE_NOT_FOUND:
            result["fallback_tool"] = "search_table"
            result["fallback_reason"] = "表不存在，建议使用search_table查找可用表"
            result["auto_fix_available"] = True
            result["auto_fix_suggestion"] = "自动切换到search_table工具查找相似表名"
        elif error_type == ErrorType.PERMISSION_DENIED:
            result["suggestions"].extend(["检查是否对该表有DESCRIBE权限", "尝试使用search_table获取基本表信息"])

        return result

    def _handle_search_table_error(
        self, error_type: ErrorType, error_message: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle search_table specific errors.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Error context

        Returns:
            Dict with tool-specific error handling info
        """
        result = {}

        if error_type == ErrorType.DATABASE_CONNECTION:
            result["suggestions"].extend(["检查数据库连接字符串", "确认数据库服务状态"])
            result["auto_fix_available"] = True
            result["auto_fix_suggestion"] = "简化搜索查询，使用基本关键词重试"
        elif error_type == ErrorType.TIMEOUT:
            result["suggestions"].extend(["缩短搜索关键词", "减少搜索结果数量"])
            result["auto_fix_available"] = True
            result["auto_fix_suggestion"] = "自动减少top_n参数并重试"

        return result

    def _handle_execute_sql_error(
        self, error_type: ErrorType, error_message: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle execute_sql specific errors with advanced auto-fix.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Error context

        Returns:
            Dict with tool-specific error handling info
        """
        result = {}

        if error_type == ErrorType.SYNTAX_ERROR:
            result["suggestions"].extend(["检查SQL语法，特别是引号和分号", "验证表名和列名是否正确"])
            result["auto_fix_available"] = True
            result["auto_fix_suggestion"] = "尝试自动修复常见语法错误"

            # Try to auto-fix common SQL syntax errors
            fixed_sql = self._auto_fix_sql_syntax(context.get("sql_query", ""), error_message)
            if fixed_sql != context.get("sql_query"):
                result["fixed_sql"] = fixed_sql

        elif error_type == ErrorType.PERMISSION_DENIED:
            result["suggestions"].extend(["检查是否有执行该SQL的权限", "确认用户角色和权限设置"])

        elif error_type == ErrorType.TABLE_NOT_FOUND:
            result["suggestions"].extend(["检查SQL中引用的表名是否存在", "使用search_table查找可用表名"])
            result["auto_fix_available"] = True
            result["auto_fix_suggestion"] = "查找相似的表名进行自动修正"

        return result

    def _auto_fix_sql_syntax(self, sql_query: str, error_message: str) -> str:
        """Attempt to auto-fix common SQL syntax errors.

        Args:
            sql_query: Original SQL query
            error_message: Error message containing hints about the issue

        Returns:
            Fixed SQL query if fixable, otherwise original query
        """
        if not sql_query:
            return sql_query

        fixed_sql = sql_query.strip()

        # Fix missing semicolon for SELECT statements
        if not fixed_sql.endswith(";") and fixed_sql.upper().strip().startswith("SELECT"):
            fixed_sql += ";"

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

        # Fix missing FROM clause (common in malformed queries)
        if "from" not in fixed_sql.lower() and "select" in fixed_sql.lower():
            # This is a very basic fix - in practice, would need more context
            pass

        return fixed_sql
