# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Type definitions for plan mode hooks system."""

from typing import Any, Dict


class TaskType:
    """Task type classifications for intelligent execution routing."""

    TOOL_EXECUTION = "tool_execution"  # Needs external tool calls
    LLM_ANALYSIS = "llm_analysis"  # Needs LLM reasoning/analysis
    HYBRID = "hybrid"  # May need both tools and analysis

    @classmethod
    def classify_task(cls, task_content: str) -> str:
        """Intelligent task type classification based on content analysis.

        Args:
            task_content: The task description to classify

        Returns:
            One of TOOL_EXECUTION, LLM_ANALYSIS, or HYBRID
        """
        if not task_content:
            return cls.HYBRID

        content_lower = task_content.lower()

        # Analysis keywords (higher priority)
        analysis_keywords = [
            "分析",
            "检查",
            "评估",
            "验证",
            "优化",
            "审查",
            "性能影响",
            "analyze",
            "check",
            "evaluate",
            "validate",
            "optimize",
            "review",
            "performance impact",
            "审查规则",
            "质量检查",
            "代码审查",
            "sql审查",
            "规则验证",
        ]

        # Tool execution keywords (higher priority)
        tool_keywords = [
            "执行",
            "运行",
            "查询",
            "搜索",
            "创建",
            "写入",
            "插入",
            "更新",
            "删除",
            "execute",
            "run",
            "query",
            "search",
            "create",
            "write",
            "insert",
            "update",
            "delete",
            "生成sql",
            "执行sql",
            "运行查询",
            "查找表",
            "搜索数据",
        ]

        analysis_score = sum(1 for kw in analysis_keywords if kw in content_lower)
        tool_score = sum(1 for kw in tool_keywords if kw in content_lower)

        # Boost analysis score for SQL review patterns
        if any(pattern in content_lower for pattern in ["sql审查", "质量检查", "规则验证", "性能评估"]):
            analysis_score += 2

        # Boost tool score for specific tool patterns
        if any(pattern in content_lower for pattern in ["表结构", "字段信息", "执行查询", "运行sql"]):
            tool_score += 2

        if analysis_score > tool_score:
            return cls.LLM_ANALYSIS
        elif tool_score > analysis_score:
            return cls.TOOL_EXECUTION
        else:
            return cls.HYBRID


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


class PlanningPhaseException(Exception):
    """Exception raised when trying to execute tools during planning phase."""

    pass


class UserCancelledException(Exception):
    """Exception raised when user explicitly cancels execution."""

    pass
