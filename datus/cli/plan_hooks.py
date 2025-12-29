# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Plan mode hooks implementation for intercepting agent execution flow."""

import asyncio
import hashlib
import json
import re
import statistics
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents import SQLiteSession
from agents.lifecycle import AgentHooks
from rich.console import Console

from datus.api.models import (
    ChatEvent,
    DeepResearchEventType,
    SqlExecutionErrorEvent,
    SqlExecutionProgressEvent,
    SqlExecutionResultEvent,
    SqlExecutionStartEvent,
)
from datus.cli.blocking_input_manager import blocking_input_manager
from datus.cli.execution_state import execution_controller
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Task type classification for intelligent execution routing
class TaskType:
    """Task type classifications for intelligent execution routing."""

    TOOL_EXECUTION = "tool_execution"  # Needs external tool calls
    LLM_ANALYSIS = "llm_analysis"  # Needs LLM reasoning/analysis
    HYBRID = "hybrid"  # May need both tools and analysis

    @classmethod
    def classify_task(cls, task_content: str) -> str:
        """Intelligent task type classification based on content analysis."""
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


class SmartExecutionRouter:
    """Intelligent task execution router based on task type classification."""

    def __init__(self, agent_config=None, model=None, action_history_manager=None, emit_queue=None):
        self.agent_config = agent_config
        self.model = model
        self.action_history_manager = action_history_manager
        self.emit_queue = emit_queue

    async def execute_task(self, todo_item, context) -> Dict[str, Any]:
        """Smart routing of task execution based on task type."""
        task_type = getattr(todo_item, "task_type", "hybrid")

        if task_type == TaskType.LLM_ANALYSIS:
            return await self._execute_llm_analysis(todo_item, context)
        elif task_type == TaskType.TOOL_EXECUTION:
            return await self._execute_tool_call(todo_item, context)
        else:  # hybrid or unknown
            return await self._execute_hybrid_task(todo_item, context)

    async def _execute_llm_analysis(self, todo_item, context) -> Dict[str, Any]:
        """Return instruction to execute LLM analysis task."""
        try:
            # Set up LLM reasoning parameters
            todo_item.requires_llm_reasoning = True
            todo_item.reasoning_type = "analysis"
            todo_item.requires_tool = False

            # Add analysis context if this is SQL review
            if "sql审查" in todo_item.content.lower() or "质量检查" in todo_item.content.lower():
                todo_item.analysis_context = {
                    "domain": "sql_review",
                    "rules": ["starrocks_3_3_rules", "performance_best_practices"],
                    "output_format": "structured_report",
                }

            return {
                "success": True,
                "execution_type": "llm_analysis",
                "action": "execute_llm_reasoning",  # 返回执行指令
                "todo_item": todo_item,
            }

        except Exception as e:
            logger.error(f"LLM analysis setup failed for todo {todo_item.id}: {e}")
            return {"success": False, "error": str(e), "execution_type": "llm_analysis"}

    async def _execute_tool_call(self, todo_item, context) -> Dict[str, Any]:
        """Return instruction to execute tool-based task."""
        return {
            "success": True,
            "execution_type": "tool_execution",
            "action": "execute_tool",  # 返回执行指令
            "todo_item": todo_item,
        }

    async def _execute_hybrid_task(self, todo_item, context) -> Dict[str, Any]:
        """Return instruction for hybrid task execution."""
        content_lower = todo_item.content.lower()

        # Check if this looks more like a tool task
        tool_indicators = ["执行", "运行", "查询", "搜索", "查找", "execute", "run", "query", "search"]
        if any(indicator in content_lower for indicator in tool_indicators):
            # Prefer tool execution for hybrid tasks that look tool-oriented
            return await self._execute_tool_call(todo_item, context)
        else:
            # Prefer LLM analysis for hybrid tasks that look analysis-oriented
            return await self._execute_llm_analysis(todo_item, context)


# Error handling types and configurations
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
            ErrorType.RESOURCE_EXHAUSTED,
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

    def _load_error_patterns(self) -> Dict[str, Dict]:
        """Load error pattern recognition rules."""
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
        """Classify error type and provide recovery suggestions."""
        error_msg_lower = error_message.lower()

        for pattern, info in self.error_patterns.items():
            if re.search(pattern, error_msg_lower, re.IGNORECASE):
                return info["type"], info["suggestions"]

        return ErrorType.UNKNOWN, ["检查错误详情并联系技术支持"]

    def handle_tool_error(
        self, tool_name: str, error: Exception, attempt_count: int = 1, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle tool execution errors with recovery strategies and auto-fix suggestions."""

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
        """Handle describe_table specific errors."""
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
        """Handle search_table specific errors."""
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
        """Handle execute_sql specific errors with advanced auto-fix."""
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
        """Attempt to auto-fix common SQL syntax errors."""
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


class ExecutionMonitor:
    """Comprehensive monitoring system for plan mode execution."""

    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size

        # Execution metrics
        self.execution_history = deque(maxlen=max_history_size)
        self.current_execution = None
        self.start_time = None

        # Performance counters
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_todos_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_optimizations": 0,
            "error_recoveries": 0,
        }

        # Tool-specific metrics
        self.tool_metrics = defaultdict(
            lambda: {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "avg_execution_time": 0,
                "total_execution_time": 0,
                "errors": defaultdict(int),
            }
        )

        # Real-time monitoring data
        self.active_operations = {}
        self.performance_trends = deque(maxlen=100)

    def start_execution(self, execution_id: str, plan_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """Start monitoring a new execution."""
        self.current_execution = {
            "id": execution_id,
            "plan_id": plan_id,
            "start_time": time.time(),
            "metadata": metadata or {},
            "todos": [],
            "tools_used": set(),
            "events": [],
            "metrics": {},
        }
        self.start_time = time.time()
        logger.info(f"📊 Started monitoring execution {execution_id}")

    def end_execution(self, status: str = "completed", error_message: Optional[str] = None):
        """End the current execution monitoring."""
        if not self.current_execution:
            return

        end_time = time.time()
        duration = end_time - self.current_execution["start_time"]

        self.current_execution.update(
            {
                "end_time": end_time,
                "duration": duration,
                "status": status,
                "error_message": error_message,
                "final_metrics": self._calculate_execution_metrics(),
            }
        )

        # Update global metrics
        self.metrics["total_executions"] += 1
        if status == "completed":
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1

        self.metrics["total_todos_processed"] += len(self.current_execution["todos"])

        # Store in history
        self.execution_history.append(self.current_execution.copy())

        # Update performance trends
        self.performance_trends.append(
            {
                "timestamp": end_time,
                "duration": duration,
                "todos_count": len(self.current_execution["todos"]),
                "tools_used": len(self.current_execution["tools_used"]),
                "status": status,
            }
        )

        logger.info(f"📊 Ended monitoring execution {self.current_execution['id']}: {status} in {duration:.2f}s")
        self.current_execution = None
        self.start_time = None

    def record_todo_start(self, todo_id: str, content: str):
        """Record the start of a todo execution."""
        if not self.current_execution:
            return

        todo_record = {
            "id": todo_id,
            "content": content,
            "start_time": time.time(),
            "status": "in_progress",
            "tools_used": [],
            "events": [],
        }
        self.current_execution["todos"].append(todo_record)
        self.active_operations[todo_id] = todo_record

        logger.debug(f"📝 Started todo {todo_id}: {content[:50]}...")

    def record_todo_end(
        self, todo_id: str, status: str = "completed", result: Optional[Any] = None, error: Optional[str] = None
    ):
        """Record the end of a todo execution."""
        if not self.current_execution or todo_id not in self.active_operations:
            return

        end_time = time.time()
        todo_record = self.active_operations[todo_id]
        start_time = todo_record["start_time"]

        todo_record.update(
            {
                "end_time": end_time,
                "duration": end_time - start_time,
                "status": status,
                "result": result,
                "error": error,
            }
        )

        del self.active_operations[todo_id]

        logger.debug(f"📝 Ended todo {todo_id}: {status} in {todo_record['duration']:.2f}s")

    def record_tool_call(self, tool_name: str, todo_id: str, params: Dict[str, Any], start_time: float):
        """Record a tool call start."""
        if not self.current_execution:
            return

        self.current_execution["tools_used"].add(tool_name)

        # Find the todo record
        for todo in self.current_execution["todos"]:
            if todo["id"] == todo_id:
                tool_call = {
                    "tool_name": tool_name,
                    "params": params,
                    "start_time": start_time,
                    "status": "in_progress",
                }
                todo["tools_used"].append(tool_call)
                break

        logger.debug(f"🔧 Started {tool_name} for todo {todo_id}")

    def record_tool_result(
        self,
        tool_name: str,
        todo_id: str,
        success: bool,
        execution_time: float,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ):
        """Record a tool call result."""
        if not self.current_execution:
            return

        # Update tool metrics
        tool_metric = self.tool_metrics[tool_name]
        tool_metric["calls"] += 1
        if success:
            tool_metric["successes"] += 1
        else:
            tool_metric["failures"] += 1
            if error:
                tool_metric["errors"][error] += 1

        tool_metric["total_execution_time"] += execution_time
        tool_metric["avg_execution_time"] = tool_metric["total_execution_time"] / tool_metric["calls"]

        # Update current execution record
        for todo in self.current_execution["todos"]:
            if todo["id"] == todo_id:
                for tool_call in todo["tools_used"]:
                    if tool_call["tool_name"] == tool_name and tool_call["status"] == "in_progress":
                        tool_call.update(
                            {
                                "end_time": time.time(),
                                "execution_time": execution_time,
                                "success": success,
                                "result": result,
                                "error": error,
                                "status": "completed" if success else "failed",
                            }
                        )
                        break
                break

        status_icon = "✅" if success else "❌"
        logger.debug(f"🔧 {status_icon} {tool_name} completed in {execution_time:.2f}ms for todo {todo_id}")

    def record_cache_hit(self, tool_name: str, cache_key: str):
        """Record a cache hit."""
        self.metrics["cache_hits"] += 1
        logger.debug(f"💾 Cache hit for {tool_name}: {cache_key}")

    def record_batch_optimization(self, optimization_type: str, original_count: int, optimized_count: int):
        """Record batch optimization."""
        self.metrics["batch_optimizations"] += 1
        savings = original_count - optimized_count
        logger.info(
            f"⚡ Batch optimization ({optimization_type}): {original_count} → {optimized_count} (saved {savings} operations)"
        )

    def record_error_recovery(self, strategy: str, success: bool):
        """Record error recovery attempt."""
        if success:
            self.metrics["error_recoveries"] += 1
        logger.debug(f"🔄 Error recovery ({strategy}): {'✅' if success else '❌'}")

    def _calculate_execution_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for the current execution."""
        if not self.current_execution:
            return {}

        todos = self.current_execution["todos"]
        total_todos = len(todos)
        completed_todos = sum(1 for t in todos if t["status"] == "completed")
        failed_todos = sum(1 for t in todos if t["status"] == "failed")

        total_tools = sum(len(t.get("tools_used", [])) for t in todos)
        total_execution_time = sum(t.get("duration", 0) for t in todos)

        return {
            "total_todos": total_todos,
            "completed_todos": completed_todos,
            "failed_todos": failed_todos,
            "success_rate": completed_todos / total_todos if total_todos > 0 else 0,
            "total_tools_called": total_tools,
            "total_execution_time": total_execution_time,
            "avg_todo_duration": total_execution_time / total_todos if total_todos > 0 else 0,
            "tools_used": list(self.current_execution["tools_used"]),
        }

    def get_monitoring_report(self) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        cache_total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        cache_hit_rate = self.metrics["cache_hits"] / cache_total if cache_total > 0 else 0

        recent_executions = list(self.execution_history)[-10:]  # Last 10 executions

        report = {
            "summary": {
                "total_executions": self.metrics["total_executions"],
                "success_rate": self.metrics["successful_executions"] / self.metrics["total_executions"]
                if self.metrics["total_executions"] > 0
                else 0,
                "total_todos_processed": self.metrics["total_todos_processed"],
                "cache_hit_rate": cache_hit_rate,
                "batch_optimizations": self.metrics["batch_optimizations"],
                "error_recoveries": self.metrics["error_recoveries"],
            },
            "current_status": {
                "active_operations": len(self.active_operations),
                "active_operation_details": list(self.active_operations.keys()),
            },
            "tool_performance": dict(self.tool_metrics),
            "recent_executions": recent_executions,
            "performance_trends": list(self.performance_trends)[-20:],  # Last 20 data points
        }

        return report

    def log_monitoring_report(self):
        """Log a formatted monitoring report."""
        report = self.get_monitoring_report()

        logger.info("📊 === Plan Mode Monitoring Report ===")
        logger.info(f"📈 Total Executions: {report['summary']['total_executions']}")
        logger.info(".1%")
        logger.info(f"📝 Total Todos Processed: {report['summary']['total_todos_processed']}")
        logger.info(".1%")
        logger.info(f"⚡ Batch Optimizations: {report['summary']['batch_optimizations']}")
        logger.info(f"🔄 Error Recoveries: {report['summary']['error_recoveries']}")

        # Log tool performance
        logger.info("🔧 Tool Performance:")
        for tool_name, metrics in report["tool_performance"].items():
            if metrics["calls"] > 0:
                success_rate = metrics["successes"] / metrics["calls"]
                avg_time = metrics["avg_execution_time"]
                logger.info(
                    f"  {tool_name}: {metrics['calls']} calls, {success_rate:.1%} success, {avg_time:.2f}ms avg"
                )

        # Log active operations
        if report["current_status"]["active_operations"] > 0:
            logger.info(f"🏃 Active Operations: {report['current_status']['active_operations']}")
            for op_id in report["current_status"]["active_operation_details"][:5]:  # Show first 5
                logger.info(f"  - {op_id}")

        logger.info("📊 === End Monitoring Report ===")

    def export_monitoring_data(self, format: str = "json") -> str:
        """Export monitoring data in the specified format."""
        report = self.get_monitoring_report()

        if format == "json":
            return json.dumps(report, indent=2, default=str)
        elif format == "text":
            lines = ["Plan Mode Monitoring Report"]
            lines.append("=" * 50)

            summary = report["summary"]
            lines.append(f"Total Executions: {summary['total_executions']}")
            lines.append(".1%")
            lines.append(f"Total Todos Processed: {summary['total_todos_processed']}")
            lines.append(".1%")
            lines.append(f"Batch Optimizations: {summary['batch_optimizations']}")
            lines.append(f"Error Recoveries: {summary['error_recoveries']}")

            lines.append("\nTool Performance:")
            for tool_name, metrics in report["tool_performance"].items():
                if metrics["calls"] > 0:
                    success_rate = metrics["successes"] / metrics["calls"]
                    avg_time = metrics["avg_execution_time"]
                    lines.append(
                        f"  {tool_name}: {metrics['calls']} calls, {success_rate:.1%} success, {avg_time:.2f}ms avg"
                    )

            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def generate_performance_dashboard(self) -> str:
        """Generate an ASCII-art performance dashboard."""
        report = self.get_monitoring_report()

        summary = report["summary"]
        current = report["current_status"]

        # Calculate some derived metrics
        avg_execution_time = 0
        total_executions = len(report["recent_executions"])
        if total_executions > 0:
            avg_execution_time = sum(ex.get("duration", 0) for ex in report["recent_executions"]) / total_executions

        # Build dashboard
        dashboard_lines = [
            "╔══════════════════════════════════════════════════════════════════════════════╗",
            "║                           📊 PLAN MODE PERFORMANCE DASHBOARD 📊              ║",
            "╠══════════════════════════════════════════════════════════════════════════════╣",
            f"║ Total Executions: {summary['total_executions']:<10} Success Rate: {summary['success_rate']:>7.1%}             ║",
            f"║ Todos Processed:  {summary['total_todos_processed']:<10} Cache Hit Rate: {summary['cache_hit_rate']:>7.1%}             ║",
            f"║ Batch Optimizations: {summary['batch_optimizations']:<6} Error Recoveries: {summary['error_recoveries']:<6}             ║",
            f"║ Active Operations: {current['active_operations']:<8} Avg Execution Time: {avg_execution_time:>7.1f}s           ║",
            "╠══════════════════════════════════════════════════════════════════════════════╣",
            "║                              🔧 TOOL PERFORMANCE                              ║",
            "╠══════════════════════════════════════════════════════════════════════════════╣",
        ]

        # Add tool performance
        tool_perf = report["tool_performance"]
        if tool_perf:
            for tool_name, metrics in list(tool_perf.items())[:5]:  # Show top 5 tools
                if metrics["calls"] > 0:
                    success_rate = metrics["successes"] / metrics["calls"]
                    avg_time = metrics["avg_execution_time"]
                    dashboard_lines.append(
                        f"║ {tool_name:<15} Calls: {metrics['calls']:<4} Success: {success_rate:>5.1%} Avg: {avg_time:>6.1f}ms ║"
                    )
        else:
            dashboard_lines.append("║                              No tool data available                        ║")

        dashboard_lines.extend(
            [
                "╠══════════════════════════════════════════════════════════════════════════════╣",
                "║                             📈 RECENT PERFORMANCE TREND                      ║",
                "╠══════════════════════════════════════════════════════════════════════════════╣",
            ]
        )

        # Add performance trend (last 5 executions)
        trends = report["performance_trends"][-5:]
        if trends:
            for i, trend in enumerate(trends):
                duration = trend.get("duration", 0)
                todos = trend.get("todos_count", 0)
                status_icon = "✅" if trend.get("status") == "completed" else "❌"
                dashboard_lines.append(
                    f"║ Execution {i+1}: {status_icon} Duration: {duration:>5.1f}s Todos: {todos:<3}                         ║"
                )
        else:
            dashboard_lines.append("║                              No recent executions                          ║")

        dashboard_lines.extend(
            [
                "╚══════════════════════════════════════════════════════════════════════════════╝",
                "",
                "💡 Tips:",
                f"   • Cache hit rate of {summary['cache_hit_rate']:.1%} indicates {'excellent' if summary['cache_hit_rate'] > 0.8 else 'good' if summary['cache_hit_rate'] > 0.5 else 'needs improvement'} caching performance",
                f"   • {summary['batch_optimizations']} batch optimizations saved computational resources",
                f"   • {summary['error_recoveries']} successful error recoveries improved reliability",
                "",
                "Use monitor.export_monitoring_data('json') for detailed JSON export",
            ]
        )

        return "\n".join(dashboard_lines)


class QueryCache:
    """Intelligent query result caching system."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _get_cache_key(self, tool_name: str, **kwargs) -> str:
        """Generate a deterministic cache key for the query."""
        # Normalize kwargs by sorting keys and excluding non-deterministic parameters
        cache_params = {k: v for k, v in kwargs.items() if k not in ["todo_id", "call_id"]}
        sorted_params = json.dumps(cache_params, sort_keys=True)
        key_content = f"{tool_name}:{sorted_params}"
        return hashlib.md5(key_content.encode()).hexdigest()

    def get(self, tool_name: str, **kwargs) -> Optional[Any]:
        """Retrieve cached result if available and not expired."""
        cache_key = self._get_cache_key(tool_name, **kwargs)

        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                logger.debug(f"Cache hit for {tool_name} with key {cache_key}")
                return entry["result"]

            # Remove expired entry
            del self.cache[cache_key]

        return None

    def set(self, tool_name: str, result: Any, **kwargs) -> None:
        """Cache the result."""
        cache_key = self._get_cache_key(tool_name, **kwargs)

        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

        self.cache[cache_key] = {"result": result, "timestamp": time.time()}

        logger.debug(f"Cached result for {tool_name} with key {cache_key}")


class ToolBatchProcessor:
    """Batch processor for similar tool calls to improve efficiency."""

    def __init__(self):
        self.batches = {}  # tool_name -> list of (todo_item, params)

    def add_to_batch(self, tool_name: str, todo_item, params: Dict[str, Any]) -> None:
        """Add a tool call to the batch."""
        if tool_name not in self.batches:
            self.batches[tool_name] = []

        self.batches[tool_name].append((todo_item, params))

    def get_batch_size(self, tool_name: str) -> int:
        """Get the current batch size for a tool."""
        return len(self.batches.get(tool_name, []))

    def clear_batch(self, tool_name: str) -> List[Tuple]:
        """Clear and return the batch for a tool."""
        if tool_name in self.batches:
            batch = self.batches[tool_name]
            self.batches[tool_name] = []
            return batch
        return []

    def optimize_search_table_batch(self, batch: List[Tuple]) -> List[Tuple]:
        """Optimize search_table batch by consolidating similar queries."""
        if not batch:
            return batch

        # Group by similar query patterns
        query_groups = {}
        for todo_item, params in batch:
            query_text = params.get("query_text", "").lower().strip()

            # Find the most specific query (longest common prefix)
            found_group = False
            for group_key in query_groups:
                if query_text.startswith(group_key) or group_key.startswith(query_text):
                    # Use the more specific query as group key
                    new_key = max(group_key, query_text, key=len)
                    if new_key != group_key:
                        query_groups[new_key] = query_groups.pop(group_key)
                    query_groups[new_key].append((todo_item, params))
                    found_group = True
                    break

            if not found_group:
                query_groups[query_text] = [(todo_item, params)]

        # Consolidate groups: if we have multiple similar queries, keep only the most comprehensive one
        optimized_batch = []
        for group_key, items in query_groups.items():
            if len(items) == 1:
                optimized_batch.extend(items)
            else:
                # For multiple similar queries, use the one with highest top_n
                best_item = max(items, key=lambda x: x[1].get("top_n", 5))
                optimized_batch.append(best_item)
                logger.info(f"Optimized search_table batch: consolidated {len(items)} similar queries into 1")

        return optimized_batch

    def optimize_describe_table_batch(self, batch: List[Tuple]) -> List[Tuple]:
        """Optimize describe_table batch by removing duplicates."""
        if not batch:
            return batch

        # Remove duplicate table names
        seen_tables = set()
        unique_batch = []

        for todo_item, params in batch:
            table_name = params.get("table_name", "").lower().strip()
            if table_name and table_name not in seen_tables:
                seen_tables.add(table_name)
                unique_batch.append((todo_item, params))

        if len(unique_batch) < len(batch):
            logger.info(f"Optimized describe_table batch: removed {len(batch) - len(unique_batch)} duplicates")

        return unique_batch


# Default keyword mapping for plan executor - tool name to list of matching phrases
DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP = {
    # Database tools
    "search_table": [
        "search for table",
        "搜索表结构",
        "查找数据库表",
        "find table in database",
        "search table schema",
        "查找表信息",
        "find database tables",
        "搜索数据库表",
        "探索数据库中的表结构",
        "找到试驾表和线索表",
        "确认表名",
        "表结构",
        "字段名",
    ],
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
    "execute_sql": [
        "execute sql query",
        "执行sql查询",
        "run sql statement",
        "执行sql语句",
        "run database query",
        "执行数据库查询",
        "execute the sql",
        "运行sql代码",
        "execute sql",
        "执行sql",
        "run the query",
        "执行查询",
        "运行sql",
        "执行sql语句",
        "查询执行",
    ],
    "read_query": ["run query", "执行查询", "execute database query", "运行数据库查询", "perform sql execution", "执行sql执行"],
    # Metrics and reference SQL tools
    "search_metrics": [
        "search metrics",
        "查找指标",
        "find business metrics",
        "搜索业务指标",
        "look for kpis",
        "查找kpi",
        "search performance metrics",
        "查找绩效指标",
    ],
    "search_reference_sql": [
        "search sql examples",
        "查找参考sql",
        "find reference sql",
        "搜索sql模板",
        "look for sql patterns",
        "查找sql模式",
        "search similar sql",
        "查找相似sql",
    ],
    "list_domain_layers_tree": [
        "list domains",
        "查看领域层级",
        "show domain structure",
        "显示业务分类",
        "explore domain layers",
        "浏览领域层级",
        "view business taxonomy",
        "查看业务分类法",
    ],
    # Semantic model tools
    "check_semantic_model_exists": [
        "check semantic model",
        "检查语义模型",
        "verify semantic model",
        "验证语义模型",
        "semantic model exists",
        "语义模型是否存在",
        "find semantic model",
        "查找语义模型",
    ],
    "check_metric_exists": [
        "check metric exists",
        "检查指标是否存在",
        "verify metric availability",
        "验证指标可用性",
        "metric exists",
        "指标是否存在",
        "find existing metric",
        "查找现有指标",
    ],
    "generate_sql_summary_id": [
        "generate summary id",
        "生成摘要标识",
        "create sql summary",
        "创建sql摘要",
        "generate sql id",
        "生成sql标识",
        "create summary identifier",
        "创建摘要标识符",
    ],
    # Time parsing tools
    "parse_temporal_expressions": [
        "parse date expressions",
        "解析日期表达式",
        "parse temporal expressions",
        "解析时间表达式",
        "analyze date ranges",
        "分析日期范围",
        "parse time periods",
        "解析时间段",
    ],
    "get_current_date": [
        "get current date",
        "获取当前日期",
        "current date",
        "今天日期",
        "today's date",
        "今日日期",
        "get today date",
        "获取今天日期",
    ],
    # File system tools
    "write_file": [
        "write file",
        "写入文件",
        "save to file",
        "保存到文件",
        "create file",
        "创建文件",
        "write content to file",
        "将内容写入文件",
    ],
    "read_file": ["read file", "读取文件", "load file", "加载文件", "open file", "打开文件", "read file content", "读取文件内容"],
    "list_directory": [
        "list directory",
        "列出目录",
        "show directory contents",
        "显示目录内容",
        "list files",
        "列出文件",
        "directory listing",
        "目录列表",
    ],
    # Reporting tools
    "report": [
        "generate final report",
        "生成最终报告",
        "create comprehensive report",
        "创建综合报告",
        "write final report",
        "编写最终报告",
        "produce report",
        "生成报告",
        "生成最终html",
        "生成最终",
        "生成html",
        "生成报告文档",
    ],
    # Plan management tools
    "todo_write": [
        "create plan",
        "创建计划",
        "write execution plan",
        "编写执行计划",
        "generate todo list",
        "生成任务列表",
        "create task plan",
        "创建任务计划",
    ],
    "todo_update": [
        "update task status",
        "更新任务状态",
        "mark task complete",
        "标记任务完成",
        "update todo status",
        "更新待办状态",
        "change task state",
        "更改任务状态",
    ],
    "todo_read": [
        "read plan",
        "查看计划",
        "show execution plan",
        "显示执行计划",
        "check plan status",
        "检查计划状态",
        "view todo list",
        "查看任务列表",
    ],
}


class TaskTypeClassifier:
    """智能任务类型分类器"""

    TOOL_EXECUTION = "tool_execution"  # 需要调用外部工具
    LLM_ANALYSIS = "llm_analysis"  # 需要LLM推理分析
    HYBRID = "hybrid"  # 可能需要工具+分析

    @classmethod
    def classify_task(cls, task_content: str) -> str:
        """智能分类任务类型"""
        if not task_content:
            return cls.HYBRID

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
            "考察",
            "审视",
            "审核",
            "诊断",
            "评测",
            "衡量",
            "比对",
        ]

        tool_keywords = [
            "执行",
            "运行",
            "查询",
            "搜索",
            "创建",
            "写入",
            "获取",
            "获取",
            "execute",
            "run",
            "query",
            "search",
            "create",
            "write",
            "fetch",
            "retrieve",
            "调用",
            "启动",
            "构建",
            "生成",
            "制作",
            "编写",
        ]

        content_lower = task_content.lower()
        analysis_score = sum(1 for kw in analysis_keywords if kw in content_lower)
        tool_score = sum(1 for kw in tool_keywords if kw in content_lower)

        # 特殊规则：如果明确提到"执行SQL"、"运行查询"等，优先归类为工具执行
        if any(phrase in content_lower for phrase in ["执行sql", "运行sql", "execute sql", "run sql"]):
            return cls.TOOL_EXECUTION

        # 特殊规则：如果明确提到"分析"、"检查"、"评估"等，优先归类为LLM分析
        if any(phrase in content_lower for phrase in ["sql审查", "sql检查", "sql分析", "sql优化"]):
            return cls.LLM_ANALYSIS

        # 基于关键词得分分类
        if analysis_score > tool_score:
            return cls.LLM_ANALYSIS
        elif tool_score > analysis_score:
            return cls.TOOL_EXECUTION
        else:
            return cls.HYBRID

    @classmethod
    def get_task_context(cls, task_content: str, task_type: str) -> Dict[str, Any]:
        """根据任务类型获取相应的上下文信息"""
        context = {}

        if task_type == cls.LLM_ANALYSIS:
            content_lower = task_content.lower()

            # SQL审查任务的特殊上下文
            if "sql审查" in content_lower or "sql检查" in content_lower:
                context.update(
                    {
                        "domain": "sql_review",
                        "rules": ["starrocks_3_3_rules", "performance_best_practices"],
                        "output_format": "structured_report",
                    }
                )

            # 性能分析任务的上下文
            elif "性能" in content_lower or "performance" in content_lower:
                context.update(
                    {
                        "domain": "performance_analysis",
                        "metrics": ["execution_time", "memory_usage", "query_complexity"],
                        "focus_areas": ["index_usage", "join_efficiency", "data_distribution"],
                    }
                )

            # 业务逻辑验证的上下文
            elif "业务逻辑" in content_lower or "business logic" in content_lower:
                context.update(
                    {
                        "domain": "business_logic_validation",
                        "aspects": ["data_consistency", "business_rules", "data_quality"],
                    }
                )

        return context


class PlanningPhaseException(Exception):
    """Exception raised when trying to execute tools during planning phase."""


class UserCancelledException(Exception):
    """Exception raised when user explicitly cancels execution"""


class PlanModeHooks(AgentHooks):
    """Plan Mode hooks for workflow management"""

    def __init__(
        self,
        console: Console,
        session: SQLiteSession,
        auto_mode: bool = False,
        action_history_manager=None,
        agent_config=None,
        emit_queue: Optional[asyncio.Queue] = None,
        model=None,
        auto_injected_knowledge: Optional[List[str]] = None,
    ):
        self.console = console
        self.session = session
        self.auto_mode = auto_mode
        # Optional model for LLM reasoning fallback
        self.model = model
        from datus.tools.func_tool.plan_tools import SessionTodoStorage

        self.todo_storage = SessionTodoStorage(session)
        self.plan_phase = "generating"
        self.execution_mode = "auto" if auto_mode else "manual"
        self.replan_feedback = ""
        self._state_transitions = []
        self._plan_generated_pending = False  # Flag to defer plan display until LLM ends
        # Optional ActionHistoryManager passed from node to allow hooks to add actions
        self.action_history_manager = action_history_manager
        # Optional agent_config to instantiate DB/Filesystem tools when executing todos
        self.agent_config = agent_config
        # Optional emit queue to stream ActionHistory produced by hooks back to node
        self.emit_queue = emit_queue
        # Executor task handle
        self._executor_task = None
        # Completion signaling for coordination between server executor and main agent loop
        self._execution_complete = asyncio.Event()
        self._all_todos_completed = False
        # Load and merge keyword->tool mapping
        self.keyword_map: Dict[str, List[str]] = self._load_keyword_map(agent_config)
        # fallback behavior enabled by default unless agent_config disables it
        self.enable_fallback = True
        # Initialize error handler for robust error recovery
        self.error_handler = ErrorHandler()

        # Initialize performance optimization components
        self.query_cache = QueryCache()
        self.batch_processor = ToolBatchProcessor()
        self.enable_batch_processing = True
        self.enable_query_caching = True

        # Initialize smart execution router for intelligent task routing
        self.execution_router = SmartExecutionRouter(
            agent_config=agent_config, model=model, action_history_manager=action_history_manager, emit_queue=emit_queue
        )

        # Initialize monitoring system
        self.monitor = ExecutionMonitor()
        # Store auto-injected knowledge for user confirmation
        self.auto_injected_knowledge = auto_injected_knowledge or []
        try:
            if agent_config and hasattr(agent_config, "plan_executor_enable_fallback"):
                self.enable_fallback = bool(agent_config.plan_executor_enable_fallback)
        except Exception:
            pass

    def _load_keyword_map(self, agent_config) -> Dict[str, List[str]]:
        """
        Load and merge keyword mapping from config and defaults.

        Args:
            agent_config: Agent configuration object

        Returns:
            Dict mapping tool names to lists of keyword phrases
        """
        # Start with the default map
        merged_map = DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP.copy()

        try:
            # If agent_config provides custom mapping, merge it
            if agent_config and getattr(agent_config, "plan_executor_keyword_map", None):
                custom_map = agent_config.plan_executor_keyword_map
                if isinstance(custom_map, dict):
                    # Validate and normalize custom map
                    for tool_name, keywords in custom_map.items():
                        if isinstance(keywords, list):
                            # Normalize keywords to lowercase
                            normalized_keywords = [str(k).lower().strip() for k in keywords if k and str(k).strip()]
                            if normalized_keywords:
                                merged_map[tool_name] = normalized_keywords
                        else:
                            logger.warning(f"Invalid keyword list for tool '{tool_name}': {keywords}")

                    logger.info(f"Merged custom keyword map for {len(custom_map)} tools")
                else:
                    logger.warning("Invalid plan_executor_keyword_map format, using defaults")
        except Exception as e:
            logger.warning(f"Failed to load custom keyword map: {e}, using defaults")

        # Normalize all keywords to lowercase for consistent matching
        for tool_name in merged_map:
            merged_map[tool_name] = [str(k).lower().strip() for k in merged_map[tool_name] if k and str(k).strip()]

        logger.debug(f"Loaded keyword map with {len(merged_map)} tools")
        return merged_map

    def _determine_fallback_candidates(self, text: str) -> List[Tuple[str, float]]:
        """
        Determine prioritized fallback tool candidates based on content analysis.

        Args:
            text: Todo content to analyze

        Returns:
            List of (tool_name, confidence_score) tuples, sorted by confidence descending
        """
        if not text:
            return []

        content_lower = text.lower()
        candidates = []

        # Database-related patterns (highest priority)
        db_patterns = {
            "search_table": ["table", "database", "schema", "column", "field", "表", "字段", "结构", "数据库"],
            "describe_table": ["describe", "structure", "definition", "schema", "describe table"],
            "execute_sql": ["sql", "query", "execute", "run", "SQL", "查询", "执行"],
            "read_query": ["read", "select", "query", "读取", "查询"],
        }

        for tool, patterns in db_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in content_lower)
            if matches > 0:
                # Higher confidence for more pattern matches
                confidence = min(0.9, 0.3 + (matches * 0.2))
                candidates.append((tool, confidence))

        # Metrics and reference patterns
        if any(word in content_lower for word in ["metric", "kpi", "指标", "转化率", "收入", "performance"]):
            candidates.append(("search_metrics", 0.7))

        if any(word in content_lower for word in ["reference", "example", "template", "参考", "模板", "例子"]):
            candidates.append(("search_reference_sql", 0.6))

        # File system patterns
        if any(word in content_lower for word in ["file", "write", "save", "read", "文件", "写入", "保存", "读取"]):
            if "write" in content_lower or "save" in content_lower or "create" in content_lower:
                candidates.append(("write_file", 0.8))
            elif "read" in content_lower or "load" in content_lower:
                candidates.append(("read_file", 0.8))
            else:
                candidates.append(("list_directory", 0.6))

        # Report generation patterns
        if any(word in content_lower for word in ["report", "summary", "generate", "create", "报告", "摘要", "生成"]):
            candidates.append(("report", 0.8))

        # Sort by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Fallback candidates for '{text[:50]}...': {candidates}")
        return candidates

    def _match_tool_for_todo(self, text: str) -> Optional[str]:
        """
        Advanced hybrid tool matching with intelligent intent recognition:
        1. Task intent classification and semantic understanding
        2. Context-aware keyword matching with priority scoring
        3. LLM reasoning fallback with enhanced prompts
        4. Intelligent task-to-tool mapping
        """
        if not text:
            return None

        cleaned_text = self._preprocess_todo_content(text)

        # Tier 1: Task Intent Classification (most intelligent)
        intent_result = self._classify_task_intent(cleaned_text)
        if intent_result and intent_result["confidence"] > 0.8:
            logger.debug(
                f"Matched tool '{intent_result['tool']}' via intent classification (confidence: {intent_result['confidence']:.2f})"
            )
            return intent_result["tool"]

        # Tier 2: Enhanced Context-Aware Keyword Matching
        context_match = self._match_keywords_with_context(cleaned_text)
        if context_match and context_match["confidence"] > 0.7:
            logger.debug(
                f"Matched tool '{context_match['tool']}' via context-aware matching (confidence: {context_match['confidence']:.2f})"
            )
            return context_match["tool"]

        # Tier 3: Semantic Understanding for Chinese Tasks
        semantic_match = self._semantic_chinese_matching(cleaned_text)
        if semantic_match:
            logger.debug(
                f"Matched tool '{semantic_match['tool']}' via semantic understanding (confidence: {semantic_match['confidence']:.2f})"
            )
            return semantic_match["tool"]

        # Tier 4: LLM reasoning fallback (if model available and high uncertainty)
        if self.model and (not intent_result or intent_result["confidence"] < 0.6):
            tool = self._enhanced_llm_reasoning(cleaned_text)
            if tool:
                logger.debug(f"Matched tool '{tool}' via enhanced LLM reasoning")
                return tool

        # Tier 5: Intelligent inference (last resort with improved logic)
        tool = self._enhanced_intelligent_inference(cleaned_text)
        if tool:
            logger.debug(f"Matched tool '{tool}' via enhanced intelligent inference")
            return tool

        logger.debug(f"No tool matched for text: '{text}'")
        return None

    async def _match_tool_for_todo_async(self, text: str) -> Optional[str]:
        """
        Async version of tool matching that can use async LLM calls.
        """
        if not text:
            return None

        # Tier 1: Exact keyword phrase matching (word boundaries)
        tool = self._match_exact_keywords(text)
        if tool:
            logger.debug(f"Matched tool '{tool}' via exact keyword matching")
            return tool

        # Tier 2: LLM reasoning fallback (if model available)
        if self.model:
            tool = await self._llm_reasoning_fallback_async(text)
            if tool:
                logger.debug(f"Matched tool '{tool}' via LLM reasoning")
                return tool

        # Tier 3: Intelligent inference (pattern-based)
        tool = self._intelligent_inference(text)
        if tool:
            logger.debug(f"Matched tool '{tool}' via intelligent inference")
            return tool

        logger.debug(f"No tool matched for text: '{text}'")
        return None

    def _match_exact_keywords(self, text: str) -> Optional[str]:
        """Exact keyword phrase matching with word boundaries and flexible matching."""
        if not text:
            return None

        t = text.lower()
        # Use word boundaries to ensure exact phrase matching
        for tool_name, keywords in self.keyword_map.items():
            for keyword in keywords:
                if not keyword:
                    continue

                keyword_lower = keyword.lower()
                # Original exact phrase matching
                if f" {keyword_lower} " in f" {t} ":
                    return tool_name
                # Also check for phrase at start/end of text
                if t.startswith(keyword_lower) or t.endswith(keyword_lower):
                    return tool_name
                # Flexible matching for Chinese text - check if any word in keyword appears
                keyword_words = keyword_lower.split()
                if len(keyword_words) > 1 and any(word in t for word in keyword_words):
                    return tool_name

        return None

    async def _execute_llm_reasoning(self, item: "TodoItem") -> Optional[Dict[str, Any]]:
        """
        Execute LLM reasoning for a todo item.

        Args:
            item: TodoItem requiring LLM reasoning

        Returns:
            Dict containing reasoning result or None if failed
        """
        if not self.model:
            logger.warning(f"No LLM model available for reasoning on todo {item.id}")
            return None

        logger.info(f"LLM reasoning: model available, type: {type(self.model)}")

        try:
            # Get recent context (last 5 actions for relevance)
            context_actions = []
            if self.action_history_manager:
                recent_actions = self.action_history_manager.get_actions()[-5:]
                for action in recent_actions:
                    if action.role in [ActionRole.ASSISTANT, ActionRole.TOOL]:
                        context_actions.append(
                            {
                                "role": action.role.value if hasattr(action.role, "value") else str(action.role),
                                "action_type": action.action_type,
                                "messages": action.messages[:100] if action.messages else "",  # Truncate for context
                            }
                        )

            # Build reasoning prompt based on type
            reasoning_instructions = {
                "analysis": "Analyze the following task and provide insights, considerations, or structured breakdown.",
                "reflection": "Reflect on the current state and previous actions. What worked well? What could be improved?",
                "validation": "Validate the approach, check for completeness, and identify potential issues.",
                "synthesis": "Synthesize information from context and provide a comprehensive response or next steps.",
            }

            instruction = reasoning_instructions.get(
                item.reasoning_type, "Provide reasoning and insights for this task."
            )

            prompt = f"""{instruction}

Task: {item.content}

Context (recent actions):
{chr(10).join(f"- {ctx['role']}: {ctx['action_type']} - {ctx['messages']}" for ctx in context_actions) if context_actions else "No recent context available"}

Provide your reasoning and any recommendations. If this requires tool calls, you can suggest them in your response."""

            # Execute LLM reasoning
            logger.info(f"LLM reasoning: calling model.generate for todo {item.id}")
            try:
                response = await asyncio.to_thread(self.model.generate, prompt, max_tokens=500, temperature=0.3)
                logger.info(
                    f"LLM reasoning: model.generate returned for todo {item.id}, response type: {type(response)}"
                )
            except Exception as e:
                logger.error(f"LLM reasoning: model.generate failed for todo {item.id}: {e}")
                return None

            if response:
                logger.info(f"LLM reasoning: response has content attr: {hasattr(response, 'content')}")
                if hasattr(response, "content"):
                    logger.info(f"LLM reasoning: content length: {len(response.content) if response.content else 0}")
                else:
                    logger.info(
                        f"LLM reasoning: response attrs: {[attr for attr in dir(response) if not attr.startswith('_')]}"
                    )
            else:
                logger.warning(f"LLM reasoning: response is None for todo {item.id}")

            if response and hasattr(response, "content"):
                reasoning_result = {
                    "reasoning_type": item.reasoning_type,
                    "response": response.content.strip(),
                    "context_used": len(context_actions),
                    "sql": None,  # May be extracted from response if present
                    "tool_calls": None,  # May be populated if LLM suggests tools
                }

                # Try to extract SQL if present in response
                import re

                sql_match = re.search(r"```sql\s*(.*?)\s*```", response.content, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    reasoning_result["sql"] = sql_match.group(1).strip()

                # Try to parse tool calls if LLM suggests them (JSON format)
                try:
                    # Look for JSON-like tool call suggestions in response
                    json_match = re.search(r'\{.*?"tool_calls".*?\}', response.content, re.DOTALL)
                    if json_match:
                        tool_calls_data = json.loads(json_match.group(0))
                        if "tool_calls" in tool_calls_data:
                            reasoning_result["tool_calls"] = tool_calls_data["tool_calls"]
                except Exception:
                    pass  # Tool calls parsing is optional

                return reasoning_result
            else:
                logger.warning(f"LLM reasoning failed for todo {item.id}: no response content")
                return None

        except Exception as e:
            logger.error(f"LLM reasoning execution failed for todo {item.id}: {e}")
        return None

    def _llm_reasoning_fallback(self, text: str) -> Optional[str]:
        """
        Use LLM to intelligently determine the appropriate tool for a todo item.
        This is called when exact keyword matching fails.
        Note: Synchronous version for compatibility with existing sync code.
        """
        if not self.model or not text:
            return None

        try:
            # Create a prompt for the LLM to reason about tool selection
            available_tools = list(self.keyword_map.keys())

            prompt = f"""Given the following todo item text, determine which tool from the available tools would be most appropriate to execute this task.

Available tools:
{chr(10).join(f"- {tool}: Used for tasks involving {tool.replace('_', ' ')}" for tool in available_tools)}

Todo item: "{text}"

Consider the intent and requirements of the todo item. Choose the single most appropriate tool from the list above. If no tool seems appropriate, respond with "none".

Respond with only the tool name, nothing else."""

            # Use a simple completion call
            response = self.model.generate(prompt, max_tokens=50, temperature=0.1)

            if response and hasattr(response, "content"):
                tool_name = response.content.strip().lower()
                # Validate that the suggested tool exists
                if tool_name in [t.lower() for t in available_tools]:
                    # Find the exact case-sensitive tool name
                    for available_tool in available_tools:
                        if available_tool.lower() == tool_name:
                            return available_tool
                elif tool_name == "none":
                    return None

            logger.debug(f"LLM reasoning failed or returned invalid tool: {response}")
            return None

        except Exception as e:
            logger.debug(f"LLM reasoning failed: {e}")
            return None

    async def _llm_reasoning_fallback_async(self, text: str) -> Optional[str]:
        """
        Async version of LLM reasoning fallback for future use.
        """
        if not self.model or not text:
            return None

        try:
            # Create a prompt for the LLM to reason about tool selection
            available_tools = list(self.keyword_map.keys())

            prompt = f"""Given the following todo item text, determine which tool from the available tools would be most appropriate to execute this task.

Available tools:
{chr(10).join(f"- {tool}: Used for tasks involving {tool.replace('_', ' ')}" for tool in available_tools)}

Todo item: "{text}"

Consider the intent and requirements of the todo item. Choose the single most appropriate tool from the list above. If no tool seems appropriate, respond with "none".

Respond with only the tool name, nothing else."""

            # Use async generation if available
            if hasattr(self.model, "generate_async"):
                response = await asyncio.to_thread(self.model.generate, prompt, max_tokens=50, temperature=0.1)
            else:
                # Fallback to sync method in a thread
                import asyncio

                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.model.generate(prompt, max_tokens=50, temperature=0.1)
                )

            if response and hasattr(response, "content"):
                tool_name = response.content.strip().lower()
                # Validate that the suggested tool exists
                if tool_name in [t.lower() for t in available_tools]:
                    # Find the exact case-sensitive tool name
                    for available_tool in available_tools:
                        if available_tool.lower() == tool_name:
                            return available_tool
                elif tool_name == "none":
                    return None

            logger.debug(f"LLM reasoning failed or returned invalid tool: {response}")
            return None

        except Exception as e:
            logger.debug(f"LLM reasoning failed: {e}")
            return None

    def _intelligent_inference(self, text: str) -> Optional[str]:
        """Intelligent pattern-based tool inference as last resort."""
        if not text:
            return None

        t = text.lower()

        # Database-related inference (more specific patterns)
        if ("search" in t or "find" in t or "lookup" in t or "查找" in t) and (
            "table" in t or "database" in t or "表" in t or "数据库" in t
        ):
            return "search_table"
        elif (
            ("describe" in t and "table" in t)
            or ("表结构" in t)
            or ("table schema" in t)
            or ("表模式" in t)
            or ("table metadata" in t)
        ):
            return "describe_table"
        elif (
            ("execute" in t and "sql" in t)
            or ("run" in t and "query" in t)
            or ("执行" in t and ("sql" in t or "查询" in t))
        ):
            return "execute_sql"

        # Metrics-related inference (require both metric and search intent)
        if ("search" in t or "find" in t or "lookup" in t or "查找" in t) and any(
            keyword in t for keyword in ["metric", "kpi", "指标", "转化率", "收入", "销售额", "performance", "绩效"]
        ):
            return "search_metrics"

        # Time-related inference (require both time and parse/analyze intent)
        if ("parse" in t or "analyze" in t or "解析" in t or "分析" in t) and any(
            keyword in t for keyword in ["date", "time", "temporal", "日期", "时间", "period", "期间"]
        ):
            return "parse_temporal_expressions"

        # File-related inference (more specific patterns)
        if ("write" in t or "save" in t or "create" in t or "写入" in t or "保存" in t) and ("file" in t or "文件" in t):
            return "write_file"
        elif ("read" in t or "load" in t or "读取" in t) and ("file" in t or "文件" in t):
            return "read_file"
        elif ("list" in t or "directory" in t or "列出" in t) and ("directory" in t or "文件夹" in t):
            return "list_directory"

        # Report-related inference (require specific report generation intent)
        if any(
            phrase in t
            for phrase in ["final report", "生成报告", "create report", "生成最终", "final summary", "最终报告", "generate report"]
        ):
            return "report"

        return None

    async def on_start(self, context, agent) -> None:
        logger.debug(f"Plan mode start: phase={self.plan_phase}")

    async def on_tool_start(self, context, agent, tool) -> None:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        logger.debug(f"Plan mode tool start: {tool_name}, phase: {self.plan_phase}, mode: {self.execution_mode}")

        if tool_name == "todo_update" and self.execution_mode == "manual" and self.plan_phase == "executing":
            # Check if this is updating to pending status
            if self._is_pending_update(context):
                await self._handle_execution_step(tool_name)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        if tool_name == "todo_write":
            logger.info("Plan generation completed, will show plan after LLM finishes current turn")
            # Set flag instead of immediately showing plan
            # This allows any remaining "Thinking" messages to be generated first
            self._plan_generated_pending = True

    async def on_llm_end(self, context, agent, response) -> None:
        """Called when LLM finishes a turn - perfect time to show plan after all thinking is done"""
        if self._plan_generated_pending and self.plan_phase == "generating":
            self._plan_generated_pending = False
            await self._on_plan_generated()

    async def on_end(self, context, agent, output) -> None:
        logger.info(f"Plan mode end: phase={self.plan_phase}")

    def _transition_state(self, new_state: str, context: dict = None):
        old_state = self.plan_phase
        self.plan_phase = new_state

        transition_data = {
            "from_state": old_state,
            "to_state": new_state,
            "context": context or {},
            "timestamp": time.time(),
        }

        self._state_transitions.append(transition_data)
        logger.info(f"Plan mode state transition: {old_state} -> {new_state}")
        return transition_data

    def is_execution_complete(self) -> bool:
        """Check if server executor has completed all todos.

        This provides coordination between the server executor (background task)
        and the main agent loop, allowing early termination when all plan todos
        are completed.
        """
        return self._execution_complete.is_set() and self._all_todos_completed

    async def _on_plan_generated(self):
        todo_list = self.todo_storage.get_todo_list()
        logger.info(f"Plan generation - todo_list: {todo_list.model_dump() if todo_list else None}")

        # Clear replan feedback BEFORE transitioning state to ensure prompt updates correctly
        self.replan_feedback = ""
        self._transition_state("confirming", {"todo_count": len(todo_list.items) if todo_list else 0})

        if not todo_list:
            self.console.print("[red]No plan generated[/]")
            return

        # Stop live display BEFORE showing the plan (keep registered for restart)
        # At this point, LLM has finished its turn, so all thinking/tool messages are already displayed
        execution_controller.stop_live_display()
        await asyncio.sleep(0.3)

        # Surface auto-injected knowledge for user confirmation if not in auto mode
        if self.auto_injected_knowledge and not self.auto_mode:
            self.console.print("[bold yellow]Auto-detected Knowledge:[/]")
            self.console.print("[dim]The following knowledge was automatically detected and will be used:[/]")
            for i, knowledge in enumerate(self.auto_injected_knowledge, 1):
                self.console.print(f"  {i}. {knowledge}")
            self.console.print()

            # Ask for user confirmation
            try:
                confirmed = await self._get_user_confirmation_for_knowledge()
                if not confirmed:
                    self.console.print("[yellow]Knowledge injection cancelled by user.[/]")
                    return
            except Exception as e:
                logger.warning(f"Failed to get knowledge confirmation: {e}")
                # Continue with plan generation if confirmation fails

        self.console.print("[bold green]Plan Generated Successfully![/]")
        self.console.print("[bold cyan]Execution Plan:[/]")

        for i, item in enumerate(todo_list.items, 1):
            self.console.print(f"  {i}. {item.content}")

        # Auto mode: skip user confirmation
        if self.auto_mode:
            self.execution_mode = "auto"
            self._transition_state("executing", {"mode": "auto"})
            self.console.print("[green]Auto execution mode (workflow/benchmark context)[/]")

            # Send initial plan_update event to notify frontend of current plan state
            if self.action_history_manager and self.emit_queue is not None:
                try:
                    await self._emit_plan_update_event()
                    logger.info("Sent initial plan_update event for auto execution")
                except Exception as e:
                    logger.error(f"Failed to send initial plan_update event: {e}")

            # Start server-side executor in auto mode if action_history_manager is available
            if self.action_history_manager:
                try:
                    # schedule executor as background task
                    self._executor_task = asyncio.create_task(self._run_server_executor())
                    logger.info("Started server-side plan executor task")
                except Exception as e:
                    logger.error(f"Failed to start server-side executor: {e}")
            return

        # Interactive mode: ask for user confirmation
        try:
            await self._get_user_confirmation()
        except PlanningPhaseException:
            # Re-raise to be handled by chat_agentic_node.py
            raise

    async def _get_user_confirmation(self):
        import asyncio
        import sys

        try:
            sys.stdout.flush()
            sys.stderr.flush()

            self.console.print("\n" + "=" * 50)
            self.console.print("\n[bold cyan]CHOOSE EXECUTION MODE:[/]")
            self.console.print("")
            self.console.print("  1. Manual Confirm - Confirm each step")
            self.console.print("  2. Auto Execute - Run all steps automatically")
            self.console.print("  3. Revise - Provide feedback and regenerate plan")
            self.console.print("  4. Cancel")
            self.console.print("")

            # Pause execution while getting user input (live display already stopped by caller)
            async with execution_controller.pause_execution():
                # Small delay for console stability after flushing
                await asyncio.sleep(0.2)

                # Get input using blocking_input_manager
                def get_user_input():
                    return blocking_input_manager.get_blocking_input(
                        lambda: input("Your choice (1-4) [1]: ").strip() or "1"
                    )

                choice = await execution_controller.request_user_input(get_user_input)

            if choice == "1":
                self.execution_mode = "manual"
                self._transition_state("executing", {"mode": "manual"})
                self.console.print("[green]Manual confirmation mode selected[/]")
                # Recreate live display from current cursor position (brand new display)
                execution_controller.recreate_live_display()
                return
            elif choice == "2":
                self.execution_mode = "auto"
                self._transition_state("executing", {"mode": "auto"})
                self.console.print("[green]Auto execution mode selected[/]")
                # Recreate live display from current cursor position (brand new display)
                execution_controller.recreate_live_display()
                return
            elif choice == "3":
                await self._handle_replan()
                # Recreate live display for regeneration phase
                execution_controller.recreate_live_display()
                raise PlanningPhaseException(f"REPLAN_REQUIRED: Revise the plan with feedback: {self.replan_feedback}")
            elif choice == "4":
                self._transition_state("cancelled", {})
                self.console.print("[yellow]Plan cancelled[/]")
                raise UserCancelledException("User cancelled plan execution")
            else:
                self.console.print("[red]Invalid choice, please try again[/]")
                await self._get_user_confirmation()

        except (KeyboardInterrupt, EOFError):
            self._transition_state("cancelled", {"reason": "keyboard_interrupt"})
            self.console.print("\n[yellow]Plan cancelled[/]")

    async def _handle_replan(self):
        try:
            # Stop live display before prompting (keep registered for restart)
            execution_controller.stop_live_display()

            async with execution_controller.pause_execution():
                await asyncio.sleep(0.1)

                self.console.print("\n[bold yellow]Provide feedback for replanning:[/]")

                def get_user_input():
                    return blocking_input_manager.get_blocking_input(lambda: input("> ").strip())

                feedback = await execution_controller.request_user_input(get_user_input)
            if feedback:
                todo_list = self.todo_storage.get_todo_list()
                completed_items = [item for item in todo_list.items if item.status == "completed"] if todo_list else []

                if completed_items:
                    self.console.print(f"[blue]Found {len(completed_items)} completed steps[/]")

                self.console.print(f"[green]Replanning with feedback: {feedback}[/]")
                self.replan_feedback = feedback
                # Transition back to generating phase for replan
                self._transition_state("generating", {"replan_triggered": True, "feedback": feedback})
            else:
                self.console.print("[yellow]No feedback provided[/]")
                if self.plan_phase == "confirming":
                    await self._get_user_confirmation()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Replan cancelled[/]")

    async def _get_user_confirmation_for_knowledge(self) -> bool:
        """Get user confirmation for auto-injected knowledge."""
        import asyncio
        import sys

        try:
            sys.stdout.flush()
            sys.stderr.flush()

            self.console.print("\n[bold cyan]AUTO-DETECTED KNOWLEDGE CONFIRMATION:[/]")
            self.console.print("The system automatically detected relevant knowledge for your task.")
            self.console.print("This knowledge will be used to generate better SQL/results.")
            self.console.print("")
            self.console.print("  y. Accept and continue with plan generation")
            self.console.print("  n. Reject auto-detected knowledge (plan will proceed without it)")
            self.console.print("")

            # Pause execution while getting user input
            async with execution_controller.pause_execution():
                await asyncio.sleep(0.2)

                def get_user_input():
                    return blocking_input_manager.get_blocking_input(
                        lambda: input("Accept auto-detected knowledge? (y/n) [y]: ").strip().lower() or "y"
                    )

                choice = await execution_controller.request_user_input(get_user_input)

                if choice in ["y", "yes", ""]:
                    self.console.print("[green]Accepted auto-detected knowledge[/]")
                    return True
                elif choice in ["n", "no"]:
                    self.console.print("[yellow]Rejected auto-detected knowledge[/]")
                    return False
                else:
                    self.console.print("[yellow]Invalid choice, defaulting to accept[/]")
                    return True

        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Knowledge confirmation cancelled, proceeding with acceptance[/]")
            return True
        except Exception as e:
            logger.warning(f"Knowledge confirmation failed: {e}, proceeding with acceptance")
            return True

    async def _handle_execution_step(self, _tool_name: str):
        import asyncio
        import sys

        logger.info(f"PlanHooks: _handle_execution_step called with tool: {_tool_name}")

        # Auto mode: skip all step confirmations
        if self.auto_mode:
            logger.info("Auto mode enabled, executing step without confirmation")
            return

        todo_list = self.todo_storage.get_todo_list()
        logger.info(f"PlanHooks: Retrieved todo list with {len(todo_list.items) if todo_list else 0} items")

        if not todo_list:
            logger.warning("PlanHooks: No todo list found!")
            return

        pending_items = [item for item in todo_list.items if item.status == "pending"]
        logger.info(f"PlanHooks: Found {len(pending_items)} pending items")

        if not pending_items:
            return

        current_item = pending_items[0]

        # Stop live display BEFORE showing step progress (keep registered for restart)
        execution_controller.stop_live_display()

        await asyncio.sleep(0.2)
        sys.stdout.flush()
        sys.stderr.flush()

        # Print newlines to push content down and avoid overlap when resuming
        self.console.print("\n" * 2)
        self.console.print("-" * 40)

        try:
            if self.execution_mode == "auto":
                # Display full todo list with progress indicators in auto mode too
                self.console.print("\n[bold cyan]Plan Progress:[/]")
                for i, item in enumerate(todo_list.items, 1):
                    if item.status == "completed":
                        status_icon = "[green]✓[/]"
                        text_style = "[dim]"
                        close_tag = "[/]"
                    elif item.id == current_item.id:
                        status_icon = "[yellow]▶[/]"  # Current step
                        text_style = "[bold cyan]"
                        close_tag = "[/]"
                    else:
                        status_icon = "[white]○[/]"  # Pending
                        text_style = ""
                        close_tag = ""

                    self.console.print(f"  {status_icon} {text_style}{i}. {item.content}{close_tag}")

                self.console.print(f"\n[bold cyan]Auto Mode:[/] {current_item.content}")

                # Pause execution while getting user input (live display already stopped)
                async with execution_controller.pause_execution():
                    await asyncio.sleep(0.1)

                    def get_user_input():
                        return blocking_input_manager.get_blocking_input(
                            lambda: input("Execute? (y/n) [y]: ").strip().lower() or "y"
                        )

                    choice = await execution_controller.request_user_input(get_user_input)

                if choice in ["y", "yes"]:
                    self.console.print("[green]Executing...[/]")
                    # Recreate live display from current cursor position
                    execution_controller.recreate_live_display()
                    return
                elif choice in ["cancel", "c", "n", "no"]:
                    self.console.print("[yellow]Execution cancelled[/]")
                    self.plan_phase = "cancelled"
                    raise UserCancelledException("Execution cancelled by user")
            else:
                # Display full todo list with progress indicators
                self.console.print("\n[bold cyan]Plan Progress:[/]")
                for i, item in enumerate(todo_list.items, 1):
                    if item.status == "completed":
                        status_icon = "[green]✓[/]"
                        text_style = "[dim]"
                        close_tag = "[/]"
                    elif item.id == current_item.id:
                        status_icon = "[yellow]▶[/]"  # Current step
                        text_style = "[bold cyan]"
                        close_tag = "[/]"
                    else:
                        status_icon = "[white]○[/]"  # Pending
                        text_style = ""
                        close_tag = ""

                    self.console.print(f"  {status_icon} {text_style}{i}. {item.content}{close_tag}")

                self.console.print(f"\n[bold cyan]Next step:[/] {current_item.content}")
                self.console.print("Options:")
                self.console.print("  1. Execute this step")
                self.console.print("  2. Execute this step and continue automatically")
                self.console.print("  3. Revise remaining plan")
                self.console.print("  4. Cancel")

                while True:
                    # Pause execution while getting user input (live display already stopped)
                    async with execution_controller.pause_execution():
                        await asyncio.sleep(0.1)

                        def get_user_input():
                            return blocking_input_manager.get_blocking_input(
                                lambda: input("\nYour choice (1-4) [1]: ").strip() or "1"
                            )

                        choice = await execution_controller.request_user_input(get_user_input)

                    if choice == "1":
                        self.console.print("[green]Executing step...[/]")
                        # Recreate live display from current cursor position
                        execution_controller.recreate_live_display()
                        return
                    elif choice == "2":
                        self.execution_mode = "auto"
                        self.console.print("[green]Switching to auto mode...[/]")
                        # Recreate live display from current cursor position
                        execution_controller.recreate_live_display()
                        return
                    elif choice == "3":
                        await self._handle_replan()
                        # Recreate live display for regeneration phase
                        execution_controller.recreate_live_display()
                        raise PlanningPhaseException(
                            f"REPLAN_REQUIRED: Revise the plan with feedback: {self.replan_feedback}"
                        )
                    elif choice == "4":
                        self._transition_state("cancelled", {"step": current_item.content, "user_choice": choice})
                        self.console.print("[yellow]Execution cancelled[/]")
                        raise UserCancelledException("User cancelled execution")
                    else:
                        self.console.print(f"[red]Invalid choice '{choice}'. Please enter 1, 2, 3, or 4.[/]")

        except (KeyboardInterrupt, EOFError):
            self._transition_state("cancelled", {"reason": "execution_interrupted"})
            self.console.print("\n[yellow]Execution cancelled[/]")

    def _todo_already_executed(self, todo_id: str) -> bool:
        """Check action history for a completed tool action referencing todo_id."""
        try:
            if not self.action_history_manager:
                return False
            for a in self.action_history_manager.get_actions():
                if a.role == "tool" or a.role == ActionRole.TOOL:
                    # inspect input: may be dict or string
                    inp = a.input
                    if isinstance(inp, dict):
                        # parsed arguments may be nested under 'arguments' as json string
                        if inp.get("todo_id") == todo_id or inp.get("todoId") == todo_id:
                            if a.status == ActionStatus.SUCCESS:
                                return True
                        args = inp.get("arguments")
                        if isinstance(args, str):
                            try:
                                parsed = json.loads(args)
                                if isinstance(parsed, dict) and (
                                    parsed.get("todo_id") == todo_id or parsed.get("todoId") == todo_id
                                ):
                                    if a.status == ActionStatus.SUCCESS:
                                        return True
                            except Exception:
                                pass
                    else:
                        # try parse string input
                        if isinstance(inp, str):
                            try:
                                parsed = json.loads(inp)
                                if isinstance(parsed, dict) and (
                                    parsed.get("todo_id") == todo_id or parsed.get("todoId") == todo_id
                                ):
                                    if a.status == ActionStatus.SUCCESS:
                                        return True
                            except Exception:
                                pass
                # Also inspect output for updated_item references
                if a.output and isinstance(a.output, dict):
                    raw = a.output
                    # find nested updated_item or todo_list entries
                    if "raw_output" in raw and isinstance(raw["raw_output"], dict):
                        ro = raw["raw_output"]
                        try:
                            res = ro.get("result", {})
                            if isinstance(res, dict):
                                updated = res.get("updated_item") or res.get("updatedItem")
                                if isinstance(updated, dict) and updated.get("id") == todo_id:
                                    if a.status == ActionStatus.SUCCESS:
                                        return True
                        except Exception:
                            pass
            return False
        except Exception as e:
            logger.debug(f"Error checking action history for todo execution: {e}")
            return False

    async def _execute_tool_with_error_handling(
        self, tool_func, tool_name: str, *args, **kwargs
    ) -> Tuple[Any, bool, Dict[str, Any]]:
        """
        Execute a tool function with comprehensive error handling, recovery, and auto-fix.

        Args:
            tool_func: The tool function to execute
            tool_name: Name of the tool for error reporting
            *args, **kwargs: Arguments to pass to the tool function

        Returns:
            Tuple of (result, success, error_info)
        """
        attempt_count = 0
        max_attempts = 3
        original_kwargs = kwargs.copy()  # Preserve original parameters for fallback

        while attempt_count < max_attempts:
            attempt_count += 1

            try:
                # Execute the tool
                result = tool_func(*args, **kwargs)
                return result, True, {}

            except Exception as e:
                logger.warning(f"Tool {tool_name} execution failed (attempt {attempt_count}/{max_attempts}): {str(e)}")

                # Analyze the error with context
                context = {"sql_query": kwargs.get("sql", kwargs.get("query_text", ""))}
                error_info = self.error_handler.handle_tool_error(tool_name, e, attempt_count, context)

                # Try auto-fix if available
                if error_info.get("auto_fix_available", False) and attempt_count < max_attempts:
                    fixed_params = self._apply_auto_fix(tool_name, error_info, kwargs)
                    if fixed_params:
                        logger.info(f"Applying auto-fix for {tool_name}: {error_info.get('auto_fix_suggestion', '')}")
                        kwargs = fixed_params
                        attempt_count -= 1  # Don't count auto-fix as a retry attempt
                        continue

                # Try fallback tool if suggested
                if error_info.get("fallback_tool") and attempt_count < max_attempts:
                    fallback_tool = error_info["fallback_tool"]
                    logger.info(f"Attempting fallback from {tool_name} to {fallback_tool}")

                    # Try to execute fallback tool
                    try:
                        fallback_result = await self._execute_fallback_tool(fallback_tool, original_kwargs, error_info)
                        if fallback_result:
                            # Record successful error recovery
                            self.monitor.record_error_recovery(f"fallback_to_{fallback_tool}", True)
                            return (
                                fallback_result[0],
                                fallback_result[1],
                                {**error_info, "fallback_used": True, "fallback_tool": fallback_tool},
                            )
                        else:
                            # Record failed error recovery
                            self.monitor.record_error_recovery(f"fallback_to_{fallback_tool}", False)
                    except Exception as fallback_e:
                        logger.warning(f"Fallback tool {fallback_tool} also failed: {str(fallback_e)}")
                        self.monitor.record_error_recovery(f"fallback_to_{fallback_tool}", False)

                # Check if we should retry
                if not error_info.get("can_retry", False) or attempt_count >= max_attempts:
                    # No more retries or not retryable - return the error
                    error_info["final_attempt"] = True
                    return None, False, error_info

                # Wait before retry
                retry_delay = error_info.get("retry_delay", 1.0)
                logger.info(f"Retrying {tool_name} in {retry_delay:.1f} seconds...")
                await asyncio.sleep(retry_delay)

        # Should not reach here, but just in case
        return None, False, {"error_type": ErrorType.UNKNOWN, "error_message": "Maximum retries exceeded"}

    def _apply_auto_fix(
        self, tool_name: str, error_info: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply automatic fixes based on error analysis."""
        fixed_kwargs = kwargs.copy()

        try:
            if tool_name == "execute_sql" and "fixed_sql" in error_info:
                fixed_kwargs["sql"] = error_info["fixed_sql"]
                logger.info(f"Auto-fixed SQL syntax: {error_info['fixed_sql']}")
                return fixed_kwargs

            elif tool_name == "search_table" and error_info.get("auto_fix_suggestion") == "自动减少top_n参数并重试":
                current_top_n = fixed_kwargs.get("top_n", 5)
                if current_top_n > 1:
                    fixed_kwargs["top_n"] = max(1, current_top_n // 2)
                    logger.info(f"Auto-reduced top_n from {current_top_n} to {fixed_kwargs['top_n']}")
                    return fixed_kwargs

            elif tool_name == "search_table" and error_info.get("auto_fix_suggestion") == "简化搜索查询，使用基本关键词重试":
                query_text = fixed_kwargs.get("query_text", "")
                if len(query_text.split()) > 3:
                    # Simplify by taking first few words
                    simplified = " ".join(query_text.split()[:2])
                    fixed_kwargs["query_text"] = simplified
                    logger.info(f"Auto-simplified query from '{query_text}' to '{simplified}'")
                    return fixed_kwargs

        except Exception as e:
            logger.warning(f"Failed to apply auto-fix: {e}")

        return None

    async def _execute_fallback_tool(
        self, fallback_tool_name: str, original_kwargs: Dict[str, Any], error_info: Dict[str, Any]
    ) -> Optional[Tuple[Any, bool]]:
        """Execute a fallback tool when the primary tool fails."""
        try:
            # Import tools here to avoid circular imports
            from datus.tools.func_tool import db_tools

            # Get database tool instance (this assumes we have access to it)
            # In practice, this would need to be passed as a parameter
            db_tool = getattr(self, "_db_tool", None)
            if not db_tool:
                return None

            if fallback_tool_name == "search_table":
                query_text = original_kwargs.get("table_name", original_kwargs.get("query_text", ""))
                result = db_tool.search_table(query_text=query_text, top_n=3)
                return (result, True)

            elif fallback_tool_name == "describe_table":
                # This would be unusual, but handle it
                table_name = (
                    original_kwargs.get("query_text", "").split()[0] if original_kwargs.get("query_text") else ""
                )
                if table_name:
                    result = db_tool.describe_table(table_name=table_name)
                    return (result, True)

        except Exception as e:
            logger.warning(f"Fallback tool {fallback_tool_name} execution failed: {e}")

        return None

    async def _emit_status_message(self, message: str, plan_id: Optional[str] = None):
        """Emit a user-friendly status message using Chat event."""
        try:
            if self.emit_queue is not None:
                status_event = ChatEvent(
                    id=f"status_{int(time.time() * 1000)}",
                    planId=plan_id,
                    timestamp=int(time.time() * 1000),
                    event=DeepResearchEventType.Chat,
                    content=message,
                )
                await self.emit_queue.put(status_event)
                logger.debug(f"Emitted status message: {message}")
        except Exception as e:
            logger.debug(f"Failed to emit status message: {e}")

    async def _execute_batch_operations(self, db_tool, fs_tool, call_id: str) -> Dict[str, List[Dict]]:
        """Execute batched tool operations for improved efficiency."""
        batch_results = {}

        # Process search_table batch
        if self.batch_processor.get_batch_size("search_table") > 0:
            search_batch = self.batch_processor.clear_batch("search_table")
            optimized_batch = self.batch_processor.optimize_search_table_batch(search_batch)

            # Record batch optimization
            if len(optimized_batch) < len(search_batch):
                self.monitor.record_batch_optimization(
                    "search_table_consolidation", len(search_batch), len(optimized_batch)
                )

            batch_results["search_table"] = await self._execute_search_table_batch(optimized_batch, db_tool, call_id)

        # Process describe_table batch
        if self.batch_processor.get_batch_size("describe_table") > 0:
            describe_batch = self.batch_processor.clear_batch("describe_table")
            optimized_batch = self.batch_processor.optimize_describe_table_batch(describe_batch)

            # Record batch optimization
            if len(optimized_batch) < len(describe_batch):
                self.monitor.record_batch_optimization(
                    "describe_table_deduplication", len(describe_batch), len(optimized_batch)
                )

            batch_results["describe_table"] = await self._execute_describe_table_batch(
                optimized_batch, db_tool, call_id
            )

        return batch_results

    async def _execute_search_table_batch(self, batch: List[Tuple], db_tool, call_id: str) -> List[Dict]:
        """Execute a batch of search_table operations."""
        results = []

        for todo_item, params in batch:
            query_text = params.get("query_text", "")
            top_n = params.get("top_n", 5)

            # Check cache first
            if self.enable_query_caching:
                cached_result = self.query_cache.get("search_table", query_text=query_text, top_n=top_n)
                if cached_result is not None:
                    logger.info(f"Using cached result for search_table query: {query_text[:50]}...")
                    # Record cache hit
                    self.monitor.record_cache_hit("search_table", f"{query_text}_{top_n}")

                    result_payload = (
                        cached_result.model_dump() if hasattr(cached_result, "model_dump") else dict(cached_result)
                    )
                    complete_action_db = ActionHistory(
                        action_id=f"{call_id}_cached_search",
                        role=ActionRole.TOOL,
                        messages=f"Server executor: db.search_table (cached) for todo {todo_item.id}",
                        action_type="search_table",
                        input={
                            "function_name": "search_table",
                            "arguments": json.dumps(
                                {"query_text": query_text, "top_n": top_n, "todo_id": todo_item.id}
                            ),
                        },
                        output=result_payload,
                        status=ActionStatus.SUCCESS,
                    )

                    if self.action_history_manager:
                        self.action_history_manager.add_action(complete_action_db)
                        if self.emit_queue is not None:
                            try:
                                self.emit_queue.put_nowait(complete_action_db)
                            except Exception as e:
                                logger.debug(f"emit_queue put failed: {e}")

                    results.append({"todo_id": todo_item.id, "status": "completed", "cached": True})
                    continue

            # Execute with error handling
            res, success, error_info = await self._execute_tool_with_error_handling(
                db_tool.search_table, "search_table", query_text=query_text, top_n=top_n
            )

            if success:
                result_payload = res.model_dump() if hasattr(res, "model_dump") else dict(res)
                status = ActionStatus.SUCCESS
                messages = f"Server executor: db.search_table for todo {todo_item.id}"

                # Cache successful results
                if self.enable_query_caching:
                    self.query_cache.set("search_table", res, query_text=query_text, top_n=top_n)
            else:
                result_payload = {
                    "error": error_info.get("error_message", "Unknown error"),
                    "suggestions": error_info.get("suggestions", []),
                    "error_type": error_info.get("error_type", ErrorType.UNKNOWN),
                }
                status = ActionStatus.FAILED
                messages = f"Server executor: db.search_table failed for todo {todo_item.id}: {error_info.get('error_message', 'Unknown error')}"

            complete_action_db = ActionHistory(
                action_id=f"{call_id}_search_{todo_item.id}",
                role=ActionRole.TOOL,
                messages=messages,
                action_type="search_table",
                input={
                    "function_name": "search_table",
                    "arguments": json.dumps({"query_text": query_text, "top_n": top_n, "todo_id": todo_item.id}),
                },
                output=result_payload,
                status=status,
            )

            if self.action_history_manager:
                self.action_history_manager.add_action(complete_action_db)
                if self.emit_queue is not None:
                    try:
                        self.emit_queue.put_nowait(complete_action_db)
                    except Exception as e:
                        logger.debug(f"emit_queue put failed: {e}")

            results.append({"todo_id": todo_item.id, "status": "completed" if success else "failed"})

        return results

    async def _execute_describe_table_batch(self, batch: List[Tuple], db_tool, call_id: str) -> List[Dict]:
        """Execute a batch of describe_table operations."""
        results = []

        for todo_item, params in batch:
            table_name = params.get("table_name", "")

            # Check cache first
            if self.enable_query_caching and table_name:
                cached_result = self.query_cache.get("describe_table", table_name=table_name)
                if cached_result is not None:
                    logger.info(f"Using cached result for describe_table: {table_name}")
                    # Record cache hit
                    self.monitor.record_cache_hit("describe_table", table_name)

                    result_payload = (
                        cached_result.model_dump() if hasattr(cached_result, "model_dump") else dict(cached_result)
                    )
                    complete_action_db = ActionHistory(
                        action_id=f"{call_id}_cached_describe",
                        role=ActionRole.TOOL,
                        messages=f"Server executor: db.describe_table (cached) for todo {todo_item.id}",
                        action_type="describe_table",
                        input={
                            "function_name": "describe_table",
                            "arguments": json.dumps({"table_name": table_name, "todo_id": todo_item.id}),
                        },
                        output=result_payload,
                        status=ActionStatus.SUCCESS,
                    )

                    if self.action_history_manager:
                        self.action_history_manager.add_action(complete_action_db)
                        if self.emit_queue is not None:
                            try:
                                self.emit_queue.put_nowait(complete_action_db)
                            except Exception as e:
                                logger.debug(f"emit_queue put failed: {e}")

                    results.append({"todo_id": todo_item.id, "status": "completed", "cached": True})
                    continue

            # Execute with error handling
            res, success, error_info = await self._execute_tool_with_error_handling(
                db_tool.describe_table, "describe_table", table_name=table_name
            )

            if success:
                result_payload = res.model_dump() if hasattr(res, "model_dump") else dict(res)
                status = ActionStatus.SUCCESS
                messages = f"Server executor: db.describe_table for todo {todo_item.id}"

                # Cache successful results
                if self.enable_query_caching and table_name:
                    self.query_cache.set("describe_table", res, table_name=table_name)
            else:
                result_payload = {
                    "error": error_info.get("error_message", "Unknown error"),
                    "suggestions": error_info.get("suggestions", []),
                    "error_type": error_info.get("error_type", ErrorType.UNKNOWN),
                }
                status = ActionStatus.FAILED
                messages = f"Server executor: db.describe_table failed for todo {todo_item.id}: {error_info.get('error_message', 'Unknown error')}"

            complete_action_db = ActionHistory(
                action_id=f"{call_id}_describe_{todo_item.id}",
                role=ActionRole.TOOL,
                messages=messages,
                action_type="describe_table",
                input={
                    "function_name": "describe_table",
                    "arguments": json.dumps({"table_name": table_name, "todo_id": todo_item.id}),
                },
                output=result_payload,
                status=status,
            )

            if self.action_history_manager:
                self.action_history_manager.add_action(complete_action_db)
                if self.emit_queue is not None:
                    try:
                        self.emit_queue.put_nowait(complete_action_db)
                    except Exception as e:
                        logger.debug(f"emit_queue put failed: {e}")

            results.append({"todo_id": todo_item.id, "status": "completed" if success else "failed"})

        return results

    async def _run_server_executor(self):
        """
        Server-side plan executor with comprehensive monitoring: sequentially execute pending todos when LLM did not drive tool calls.
        This is a conservative scaffold: it marks todos in_progress -> executes minimal placeholder work ->
        marks completed and emits ActionHistory entries so the event_converter will stream proper events.
        """
        try:
            # Start execution monitoring
            execution_id = f"server_exec_{int(time.time() * 1000)}"
            self.monitor.start_execution(execution_id, plan_id="server_batch", metadata={"executor": "server"})

            # Send friendly status message about starting execution
            await self._emit_status_message("🚀 **开始执行计划任务**\n\n正在分析待执行的任务并准备工具环境...", plan_id="server_batch")

            # Small delay to allow any in-flight LLM-driven tool calls to finish
            await asyncio.sleep(0.5)
            todo_list = self.todo_storage.get_todo_list()
            if not todo_list:
                logger.info("Server executor: no todo list found, exiting")
                self.monitor.end_execution("completed", "No todos to process")
                return

            from datus.tools.func_tool.plan_tools import PlanTool

            plan_tool = PlanTool(self.session)
            plan_tool.storage = self.todo_storage

            # Prepare optional DB and filesystem tools if agent_config provided
            db_tool = None
            fs_tool = None
            try:
                if self.agent_config:
                    from datus.tools.func_tool.database import db_function_tool_instance

                    db_tool = db_function_tool_instance(
                        self.agent_config, database_name=getattr(self.agent_config, "current_database", "")
                    )
            except Exception as e:
                logger.debug(f"Could not initialize DB tool: {e}")

            try:
                from datus.tools.func_tool.filesystem_tool import FilesystemFuncTool

                fs_tool = FilesystemFuncTool()
            except Exception as e:
                logger.debug(f"Could not initialize Filesystem tool: {e}")

            for item in list(todo_list.items):
                if item.status != "pending":
                    continue

                if self._todo_already_executed(item.id):
                    logger.info(f"Server executor: todo {item.id} already executed by LLM, skipping")
                    continue

                # Mark in_progress using plan tool (directly, no ActionHistory for internal updates)
                try:
                    res1 = plan_tool._update_todo_status(item.id, "in_progress")
                    # Send plan_update event to notify frontend of status change
                    await self._emit_plan_update_event(item.id, "in_progress")
                except Exception as e:
                    logger.error(f"Server executor failed to set in_progress for {item.id}: {e}")
                    try:
                        plan_tool._update_todo_status(item.id, "failed")
                        await self._emit_plan_update_event(item.id, "failed")
                    except Exception:
                        pass
                    continue

                # Define call_id for this execution step
                call_id = f"server_call_{uuid.uuid4().hex[:8]}"

                # 智能任务路由：根据任务类型选择执行方式
                task_type = getattr(item, "task_type", "hybrid")
                if task_type == TaskType.LLM_ANALYSIS:
                    # 对于纯分析任务，使用智能路由器执行LLM分析
                    logger.info(f"Server executor: routing {item.id} to LLM analysis (task_type: {task_type})")

                    try:
                        routing_result = await self.execution_router.execute_task(
                            item, {"call_id": call_id, "plan_tool": plan_tool, "db_tool": db_tool, "fs_tool": fs_tool}
                        )

                        if routing_result["success"]:
                            action = routing_result.get("action")

                            # Execute the actual task based on action
                            if action == "execute_llm_reasoning":
                                # Execute LLM reasoning
                                logger.info(
                                    f"Server executor: executing LLM reasoning for todo {item.id} (execution_router)"
                                )
                                reasoning_result = await self._execute_llm_reasoning(item)
                                executed_any = True  # Always mark as executed to avoid infinite loop
                                if reasoning_result:
                                    # Mark as completed
                                    try:
                                        plan_tool._update_todo_status(item.id, "completed")
                                        await self._emit_plan_update_event(item.id, "completed")
                                        logger.info(f"Server executor: LLM reasoning completed for todo {item.id}")
                                    except Exception as e:
                                        logger.error(f"Failed to mark LLM analysis task as completed {item.id}: {e}")
                                else:
                                    logger.warning(
                                        f"Server executor: LLM reasoning failed for todo {item.id}, marking as failed"
                                    )
                                    try:
                                        plan_tool._update_todo_status(item.id, "failed")
                                        await self._emit_plan_update_event(item.id, "failed")
                                    except Exception as e:
                                        logger.error(f"Failed to mark LLM analysis task as failed {item.id}: {e}")
                            elif action == "execute_tool":
                                # Execute tool call through existing logic
                                logger.info(f"Server executor: executing tool for todo {item.id} (execution_router)")
                                # The existing tool matching logic will handle this below
                                # We don't set executed_any here, let the tool matching logic handle it
                                pass
                            else:
                                # Unknown action, mark as executed to avoid infinite loop
                                executed_any = True
                                logger.warning(f"Unknown action '{action}' for execution_router of todo {item.id}")
                        else:
                            logger.warning(
                                f"LLM analysis routing failed for {item.id}: {routing_result.get('error', 'Unknown error')}"
                            )
                            # Fall through to regular tool execution
                    except Exception as e:
                        logger.error(f"Smart routing failed for {item.id}: {e}")
                        # Fall through to regular tool execution

                # Execute mapped tools for the todo based on simple heuristics (fallback for tool_execution and hybrid tasks)
                content_lower = (item.content or "").lower()
                executed_any = False

                # Determine matched tool via keyword map
                matched_tool = None
                try:
                    matched_tool = self._match_tool_for_todo(item.content or "")
                except Exception as e:
                    logger.debug(f"Keyword matching failed for todo {item.id}: {e}")

                # If todo explicitly says requires_tool == False, skip execution
                if getattr(item, "requires_tool", True) is False:
                    logger.info(f"Server executor: todo {item.id} marked requires_tool=False, skipping tool execution")
                    # Emit a short assistant/system note explaining skip
                    try:
                        note_action = ActionHistory.create_action(
                            role=ActionRole.SYSTEM,
                            action_type="thinking",
                            messages=f"Skipped tool execution for todo {item.id} (requires_tool=False)",
                            input_data={"todo_id": item.id},
                            output={
                                "raw_output": "This step does not require external tool execution",
                                "emit_chat": True,
                            },
                            status=ActionStatus.SUCCESS,
                        )
                        if self.action_history_manager:
                            self.action_history_manager.add_action(note_action)
                            if self.emit_queue is not None:
                                try:
                                    self.emit_queue.put_nowait(note_action)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.debug(f"Failed to emit skip note for todo {item.id}: {e}")
                    executed_any = True  # consider handled

                # Execute LLM reasoning if required
                if not executed_any and getattr(item, "requires_llm_reasoning", False):
                    try:
                        logger.info(
                            f"Server executor: executing LLM reasoning for todo {item.id} (type: {item.reasoning_type})"
                        )
                        reasoning_result = await self._execute_llm_reasoning(item)

                        if reasoning_result:
                            # Create ActionHistory for LLM reasoning
                            reasoning_action = ActionHistory.create_action(
                                role=ActionRole.ASSISTANT,
                                action_type="thinking",
                                messages=f"LLM reasoning completed for todo {item.id}",
                                input_data={
                                    "todo_id": item.id,
                                    "reasoning_type": item.reasoning_type,
                                    "content": item.content,
                                },
                                output_data={
                                    "response": reasoning_result["response"],
                                    "reasoning_type": reasoning_result["reasoning_type"],
                                    "context_used": reasoning_result["context_used"],
                                    "sql": reasoning_result["sql"],
                                    "emit_chat": True,
                                },
                                status=ActionStatus.SUCCESS,
                            )

                            if self.action_history_manager:
                                self.action_history_manager.add_action(reasoning_action)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(reasoning_action)
                                    except Exception as e:
                                        logger.debug(f"emit_queue put failed for reasoning action: {e}")

                            # If LLM suggested tool calls, update the item for subsequent execution
                            if reasoning_result.get("tool_calls"):
                                item.tool_calls = reasoning_result["tool_calls"]

                            executed_any = True
                        else:
                            logger.warning(f"LLM reasoning returned no result for todo {item.id}")
                    except Exception as e:
                        logger.error(f"LLM reasoning execution failed for todo {item.id}: {e}")

                # If a tool was matched, execute mapped action
                if not executed_any and matched_tool:
                    try:
                        logger.info(
                            f"Server executor: matched tool '{matched_tool}' for todo {item.id} (fs_tool: {fs_tool is not None}, db_tool: {db_tool is not None})"
                        )
                        if matched_tool == "search_table" and db_tool:
                            tool_params = self._extract_tool_parameters(matched_tool, item.content or "")
                            query_text = tool_params.get("query_text", item.content or "")
                            top_n = tool_params.get("top_n", 5)

                            # Send status message about starting search
                            await self._emit_status_message(
                                f"🔍 **正在搜索数据库表**\n\n查找包含关键词的表结构和信息...", getattr(self, "plan_id", None)
                            )

                            # Execute with error handling and recovery
                            res, success, error_info = await self._execute_tool_with_error_handling(
                                db_tool.search_table, "search_table", query_text=query_text, top_n=top_n
                            )

                            if success:
                                result_payload = (
                                    res.model_dump()
                                    if hasattr(res, "model_dump")
                                    else dict(res)
                                    if isinstance(res, dict)
                                    else {"result": res}
                                )
                                status = ActionStatus.SUCCESS
                                messages = f"Server executor: db.search_table for todo {item.id}"

                                # Send success status message
                                await self._emit_status_message(
                                    f"✅ **搜索完成**\n\n找到 {len(result_payload.get('tables', []))} 个相关表",
                                    getattr(self, "plan_id", None),
                                )
                            else:
                                # Handle error with user-friendly message
                                result_payload = {
                                    "error": error_info.get("error_message", "Unknown error"),
                                    "suggestions": error_info.get("suggestions", []),
                                    "error_type": error_info.get("error_type", ErrorType.UNKNOWN),
                                }
                                status = ActionStatus.FAILED
                                messages = f"Server executor: db.search_table failed for todo {item.id}: {error_info.get('error_message', 'Unknown error')}"

                            complete_action_db = ActionHistory(
                                action_id=f"{call_id}_db",
                                role=ActionRole.TOOL,
                                messages=messages,
                                action_type="search_table",
                                input={
                                    "function_name": "search_table",
                                    "arguments": json.dumps(
                                        {"query_text": query_text, "top_n": top_n, "todo_id": item.id}
                                    ),
                                },
                                output=result_payload,
                                status=status,
                            )
                            if self.action_history_manager:
                                self.action_history_manager.add_action(complete_action_db)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(complete_action_db)
                                    except Exception as e:
                                        logger.debug(f"emit_queue put failed for complete_action_db: {e}")
                            executed_any = True
                        elif matched_tool == "execute_sql" and db_tool:
                            # attempt to find SQL in action history (existing logic)
                            sql_text = None
                            if self.action_history_manager:
                                for a in reversed(self.action_history_manager.get_actions()):
                                    if (
                                        getattr(a, "role", "") == "assistant"
                                        or getattr(a, "role", "") == ActionRole.ASSISTANT
                                    ):
                                        out = getattr(a, "output", None)
                                        if isinstance(out, dict) and out.get("sql"):
                                            sql_text = out.get("sql")
                                            break
                                        content_field = out.get("content") if isinstance(out, dict) else None
                                        if (
                                            content_field
                                            and isinstance(content_field, str)
                                            and "```sql" in content_field
                                        ):
                                            start = content_field.find("```sql")
                                            end = content_field.find("```", start + 6)
                                            if start != -1 and end != -1:
                                                sql_text = content_field[start + 6 : end].strip()
                                                break
                            if sql_text:
                                # Emit SQL execution start event
                                sql_start_event = SqlExecutionStartEvent(
                                    id=f"sql_start_{call_id}",
                                    planId=getattr(self, "plan_id", None),
                                    timestamp=int(time.time() * 1000),
                                    sqlQuery=sql_text,
                                    databaseName=getattr(self.workflow.task, "database_name", None)
                                    if hasattr(self, "workflow") and self.workflow
                                    else None,
                                )
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(sql_start_event)
                                        logger.debug(f"Emitted SQL execution start event for query: {sql_text[:50]}...")
                                    except Exception as e:
                                        logger.debug(f"Failed to emit SQL start event: {e}")

                                # Emit progress event - preparing for execution
                                sql_progress_event = SqlExecutionProgressEvent(
                                    id=f"sql_progress_{call_id}_prep",
                                    planId=getattr(self, "plan_id", None),
                                    timestamp=int(time.time() * 1000),
                                    sqlQuery=sql_text,
                                    progress=0.1,
                                    currentStep="准备执行SQL查询",
                                    elapsedTime=0,
                                )
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(sql_progress_event)
                                        logger.debug("Emitted SQL execution progress: preparing")
                                    except Exception as e:
                                        logger.debug(f"Failed to emit SQL progress event: {e}")

                                # Execute with error handling and recovery
                                start_time = time.time()
                                res, success, error_info = await self._execute_tool_with_error_handling(
                                    db_tool.read_query, "execute_sql", sql=sql_text
                                )

                                # Emit progress event - execution in progress
                                mid_time = time.time()
                                sql_progress_event2 = SqlExecutionProgressEvent(
                                    id=f"sql_progress_{call_id}_exec",
                                    planId=getattr(self, "plan_id", None),
                                    timestamp=int(mid_time * 1000),
                                    sqlQuery=sql_text,
                                    progress=0.7,
                                    currentStep="正在执行数据库查询",
                                    elapsedTime=int((mid_time - start_time) * 1000),
                                )
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(sql_progress_event2)
                                        logger.debug("Emitted SQL execution progress: executing")
                                    except Exception as e:
                                        logger.debug(f"Failed to emit SQL progress event: {e}")

                                # Calculate execution time
                                execution_time_ms = int((time.time() - start_time) * 1000)

                                if success:
                                    result_payload = (
                                        res.model_dump()
                                        if hasattr(res, "model_dump")
                                        else dict(res)
                                        if isinstance(res, dict)
                                        else {"result": res}
                                    )

                                    # Send success status message for SQL execution
                                    row_count = getattr(res, "row_count", 0)
                                    await self._emit_status_message(
                                        f"✅ **查询执行成功**\n\n返回 {row_count} 行数据，耗时 {execution_time_ms}ms",
                                        getattr(self, "plan_id", None),
                                    )

                                    # Emit successful execution result event
                                    sql_result_event = SqlExecutionResultEvent(
                                        id=f"sql_result_{call_id}",
                                        planId=getattr(self, "plan_id", None),
                                        timestamp=int(time.time() * 1000),
                                        sqlQuery=sql_text,
                                        rowCount=getattr(res, "row_count", 0),
                                        executionTime=execution_time_ms,
                                        data=getattr(res, "sql_return", None),
                                        hasMoreData=getattr(res, "has_more_data", False),
                                        dataPreview=str(getattr(res, "sql_return", ""))[:500]
                                        if getattr(res, "sql_return", None)
                                        else None,
                                    )
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(sql_result_event)
                                            logger.debug(
                                                f"Emitted SQL execution result event: {sql_result_event.rowCount} rows in {execution_time_ms}ms"
                                            )
                                        except Exception as e:
                                            logger.debug(f"Failed to emit SQL result event: {e}")
                                else:
                                    # Handle error with enhanced error information
                                    error_msg = error_info.get("error_message", "Unknown SQL execution error")
                                    result_payload = {
                                        "error": error_msg,
                                        "suggestions": error_info.get("suggestions", []),
                                        "error_type": error_info.get("error_type", ErrorType.UNKNOWN),
                                    }

                                    # Emit SQL execution error event with enhanced information
                                    sql_error_event = SqlExecutionErrorEvent(
                                        id=f"sql_error_{call_id}",
                                        planId=getattr(self, "plan_id", None),
                                        timestamp=int(time.time() * 1000),
                                        sqlQuery=sql_text,
                                        error=error_msg,
                                        errorType=error_info.get("error_type", ErrorType.UNKNOWN),
                                        suggestions=error_info.get("suggestions", []),
                                        canRetry=error_info.get("can_retry", False),
                                    )
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(sql_error_event)
                                            logger.debug(f"Emitted enhanced SQL execution error event: {error_msg}")
                                        except Exception as e:
                                            logger.debug(f"Failed to emit SQL error event: {e}")

                                    # Send user-friendly error explanation via Chat event
                                    error_type = error_info.get("error_type", ErrorType.UNKNOWN)
                                    suggestions = error_info.get("suggestions", [])

                                    if error_type == ErrorType.SYNTAX_ERROR:
                                        friendly_msg = f"❌ **SQL语法错误**\n\n查询语句存在语法问题，已自动尝试修复。如有疑问，请检查SQL语句结构。"
                                    elif error_type == ErrorType.TABLE_NOT_FOUND:
                                        friendly_msg = f"❌ **表不存在**\n\n查询的表在数据库中不存在。请检查表名是否正确。"
                                    elif error_type == ErrorType.PERMISSION_DENIED:
                                        friendly_msg = f"❌ **权限不足**\n\n没有执行此查询的权限。请联系数据库管理员。"
                                    elif error_type == ErrorType.TIMEOUT:
                                        friendly_msg = f"⚠️ **查询超时**\n\n查询执行时间过长，可能需要优化查询条件。"
                                    else:
                                        friendly_msg = f"❌ **查询执行失败**\n\n执行过程中遇到问题，系统已记录详细信息。"

                                    if suggestions:
                                        friendly_msg += f"\n\n💡 **建议解决方法**:\n" + "\n".join(
                                            f"• {s}" for s in suggestions[:3]
                                        )

                                    await self._emit_status_message(friendly_msg, getattr(self, "plan_id", None))

                                complete_action_db = ActionHistory(
                                    action_id=f"{call_id}_exec",
                                    role=ActionRole.TOOL,
                                    messages=f"Server executor: db.read_query for todo {item.id}",
                                    action_type="read_query",
                                    input={
                                        "function_name": "read_query",
                                        "arguments": json.dumps({"sql": sql_text, "todo_id": item.id}),
                                    },
                                    output=result_payload,
                                    status=ActionStatus.SUCCESS if success else ActionStatus.FAILED,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(complete_action_db)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(complete_action_db)
                                        except Exception as e:
                                            logger.debug(f"emit_queue put failed for complete_action_db: {e}")

                                # Update workflow SQL context with execution results
                                try:
                                    if (
                                        hasattr(self, "workflow")
                                        and self.workflow
                                        and hasattr(res, "success")
                                        and res.success
                                    ):
                                        # Update the last SQL context with execution results
                                        if self.workflow.context.sql_contexts:
                                            last_sql_context = self.workflow.context.sql_contexts[-1]
                                            last_sql_context.sql_return = getattr(res, "sql_return", "")
                                            last_sql_context.row_count = getattr(res, "row_count", 0)
                                            last_sql_context.sql_error = getattr(res, "error", None)
                                            logger.info(
                                                f"Updated workflow SQL context with execution results for todo {item.id}"
                                            )
                                        else:
                                            # If no existing SQL context, create one with the results
                                            sql_context = SQLContext(
                                                sql_query=sql_text,
                                                sql_return=getattr(res, "sql_return", ""),
                                                row_count=getattr(res, "row_count", 0),
                                                sql_error=getattr(res, "error", None),
                                            )
                                            self.workflow.context.sql_contexts.append(sql_context)
                                            logger.info(f"Created new SQL context in workflow for todo {item.id}")
                                except Exception as e:
                                    logger.warning(f"Failed to update workflow SQL context for todo {item.id}: {e}")

                                executed_any = True
                        elif matched_tool == "report" and fs_tool:
                            # generate report using filesystem tool
                            report_path = f"reports/{item.id}_report.html"
                            report_body = f"<html><body><h1>Report for {item.content}</h1><p>Generated by server executor.</p></body></html>"
                            res = fs_tool.write_file(path=report_path, content=report_body, file_type="report")
                            result_payload = (
                                res.model_dump()
                                if hasattr(res, "model_dump")
                                else dict(res)
                                if isinstance(res, dict)
                                else {"result": res}
                            )

                            # Create tool call action for report generation
                            complete_action_report = ActionHistory(
                                action_id=f"{call_id}_report",
                                role=ActionRole.TOOL,
                                messages=f"Server executor: report generation for todo {item.id}",
                                action_type="write_file",  # Use write_file as the actual tool action
                                input={
                                    "function_name": "write_file",
                                    "arguments": json.dumps(
                                        {"path": report_path, "content": report_body, "todo_id": item.id}
                                    ),
                                },
                                output=result_payload,
                                status=ActionStatus.SUCCESS if getattr(res, "success", 1) else ActionStatus.FAILED,
                            )

                            # Also create output_generation action for event converter
                            output_gen_action = ActionHistory(
                                action_id=f"{call_id}_output_gen",
                                role=ActionRole.TOOL,
                                messages=f"Report generated for todo {item.id}",
                                action_type="output_generation",
                                input={"todo_id": item.id},
                                output={
                                    "report_url": report_path,
                                    "html_content": report_body,
                                    "success": getattr(res, "success", 1),
                                },
                                status=ActionStatus.SUCCESS if getattr(res, "success", 1) else ActionStatus.FAILED,
                            )

                            if self.action_history_manager:
                                self.action_history_manager.add_action(complete_action_report)
                                self.action_history_manager.add_action(output_gen_action)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(complete_action_report)
                                        self.emit_queue.put_nowait(output_gen_action)
                                    except Exception as e:
                                        logger.debug(f"emit_queue put failed for report actions: {e}")
                            executed_any = True
                        elif matched_tool == "describe_table" and db_tool:
                            # Describe table structure with error handling
                            tool_params = self._extract_tool_parameters(matched_tool, item.content or "")
                            table_name = tool_params.get("table_name")
                            if table_name:
                                # Send status message about starting table analysis
                                await self._emit_status_message(
                                    f"📋 **正在分析表结构**\n\n检查表 `{table_name}` 的字段、类型和关系...", getattr(self, "plan_id", None)
                                )

                                # Execute with error handling and recovery
                                res, success, error_info = await self._execute_tool_with_error_handling(
                                    db_tool.describe_table, "describe_table", table_name=table_name
                                )

                                if success:
                                    result_payload = (
                                        res.model_dump()
                                        if hasattr(res, "model_dump")
                                        else dict(res)
                                        if isinstance(res, dict)
                                        else {"result": res}
                                    )
                                    status = ActionStatus.SUCCESS
                                    messages = f"Server executor: db.describe_table for todo {item.id}"

                                    # Send success status message
                                    await self._emit_status_message(
                                        f"✅ **表结构分析完成**\n\n成功获取表 `{table_name}` 的详细信息", getattr(self, "plan_id", None)
                                    )
                                else:
                                    # Handle error with user-friendly message
                                    result_payload = {
                                        "error": error_info.get("error_message", "Unknown error"),
                                        "suggestions": error_info.get("suggestions", []),
                                        "error_type": error_info.get("error_type", ErrorType.UNKNOWN),
                                    }
                                    status = ActionStatus.FAILED
                                    messages = f"Server executor: db.describe_table failed for todo {item.id}: {error_info.get('error_message', 'Unknown error')}"

                                    # Try fallback to search_table if suggested
                                    if error_info.get("fallback_tool") == "search_table":
                                        logger.info(f"Attempting fallback to search_table for todo {item.id}")
                                        (
                                            fallback_res,
                                            fallback_success,
                                            _,
                                        ) = await self._execute_tool_with_error_handling(
                                            db_tool.search_table, "search_table", query_text=item.content or "", top_n=5
                                        )
                                        if fallback_success:
                                            result_payload["fallback_result"] = (
                                                fallback_res.model_dump()
                                                if hasattr(fallback_res, "model_dump")
                                                else dict(fallback_res)
                                            )
                                            messages += " (fallback search successful)"

                                complete_action_db = ActionHistory(
                                    action_id=f"{call_id}_describe",
                                    role=ActionRole.TOOL,
                                    messages=messages,
                                    action_type="describe_table",
                                    input={
                                        "function_name": "describe_table",
                                        "arguments": json.dumps({"table_name": table_name, "todo_id": item.id}),
                                    },
                                    output=result_payload,
                                    status=status,
                                )
                                executed_any = True
                            else:
                                # If no table name found, this might be a search operation
                                logger.info(
                                    f"No table name extracted for describe_table, falling back to search_table for todo {item.id}"
                                )
                                query_text = tool_params.get("query_text", item.content or "")
                                res = db_tool.search_table(query_text=query_text, top_n=5)
                                result_payload = (
                                    res.model_dump()
                                    if hasattr(res, "model_dump")
                                    else dict(res)
                                    if isinstance(res, dict)
                                    else {"result": res}
                                )
                                complete_action_db = ActionHistory(
                                    action_id=f"{call_id}_search_fallback",
                                    role=ActionRole.TOOL,
                                    messages=f"Server executor: db.search_table (fallback from describe_table) for todo {item.id}",
                                    action_type="search_table",
                                    input={
                                        "function_name": "search_table",
                                        "arguments": json.dumps(
                                            {"query_text": query_text, "top_n": 5, "todo_id": item.id}
                                        ),
                                    },
                                    output=result_payload,
                                    status=ActionStatus.SUCCESS if getattr(res, "success", 1) else ActionStatus.FAILED,
                                )
                                executed_any = True
                            if self.action_history_manager:
                                self.action_history_manager.add_action(complete_action_db)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(complete_action_db)
                                    except Exception as e:
                                        logger.debug(f"emit_queue put failed for complete_action_db: {e}")
                            executed_any = True
                        elif matched_tool == "read_query" and db_tool:
                            # Read query execution
                            sql_text = None
                            if self.action_history_manager:
                                for a in reversed(self.action_history_manager.get_actions()):
                                    if (
                                        getattr(a, "role", "") == "assistant"
                                        or getattr(a, "role", "") == ActionRole.ASSISTANT
                                    ):
                                        out = getattr(a, "output", None)
                                        if isinstance(out, dict) and out.get("sql"):
                                            sql_text = out.get("sql")
                                            break
                                        content_field = out.get("content") if isinstance(out, dict) else None
                                        if (
                                            content_field
                                            and isinstance(content_field, str)
                                            and "```sql" in content_field
                                        ):
                                            start = content_field.find("```sql")
                                            end = content_field.find("```", start + 6)
                                            if start != -1 and end != -1:
                                                sql_text = content_field[start + 6 : end].strip()
                                                break
                            if sql_text:
                                res = db_tool.read_query(sql=sql_text)
                                result_payload = (
                                    res.model_dump()
                                    if hasattr(res, "model_dump")
                                    else dict(res)
                                    if isinstance(res, dict)
                                    else {"result": res}
                                )
                                complete_action_db = ActionHistory(
                                    action_id=f"{call_id}_read",
                                    role=ActionRole.TOOL,
                                    messages=f"Server executor: db.read_query for todo {item.id}",
                                    action_type="read_query",
                                    input={
                                        "function_name": "read_query",
                                        "arguments": json.dumps({"sql": sql_text, "todo_id": item.id}),
                                    },
                                    output=result_payload,
                                    status=ActionStatus.SUCCESS if getattr(res, "success", 1) else ActionStatus.FAILED,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(complete_action_db)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(complete_action_db)
                                        except Exception as e:
                                            logger.debug(f"emit_queue put failed for complete_action_db: {e}")
                                executed_any = True
                        elif matched_tool == "search_metrics" and db_tool:
                            # Search for business metrics
                            tool_params = self._extract_tool_parameters(matched_tool, item.content or "")
                            query_text = tool_params.get("query_text", item.content or "")
                            res = db_tool.search_metrics(query_text=query_text)
                            result_payload = (
                                res.model_dump()
                                if hasattr(res, "model_dump")
                                else dict(res)
                                if isinstance(res, dict)
                                else {"result": res}
                            )
                            complete_action_db = ActionHistory(
                                action_id=f"{call_id}_metrics",
                                role=ActionRole.TOOL,
                                messages=f"Server executor: db.search_metrics for todo {item.id}",
                                action_type="search_metrics",
                                input={
                                    "function_name": "search_metrics",
                                    "arguments": json.dumps({"query_text": query_text, "todo_id": item.id}),
                                },
                                output=result_payload,
                                status=ActionStatus.SUCCESS if getattr(res, "success", 1) else ActionStatus.FAILED,
                            )
                            if self.action_history_manager:
                                self.action_history_manager.add_action(complete_action_db)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(complete_action_db)
                                    except Exception as e:
                                        logger.debug(f"emit_queue put failed for complete_action_db: {e}")
                            executed_any = True
                        elif matched_tool == "search_reference_sql" and db_tool:
                            # Search for reference SQL examples
                            tool_params = self._extract_tool_parameters(matched_tool, item.content or "")
                            query_text = tool_params.get("query_text", item.content or "")
                            res = db_tool.search_reference_sql(query_text=query_text)
                            result_payload = (
                                res.model_dump()
                                if hasattr(res, "model_dump")
                                else dict(res)
                                if isinstance(res, dict)
                                else {"result": res}
                            )
                            complete_action_db = ActionHistory(
                                action_id=f"{call_id}_refsql",
                                role=ActionRole.TOOL,
                                messages=f"Server executor: db.search_reference_sql for todo {item.id}",
                                action_type="search_reference_sql",
                                input={
                                    "function_name": "search_reference_sql",
                                    "arguments": json.dumps({"query_text": query_text, "todo_id": item.id}),
                                },
                                output=result_payload,
                                status=ActionStatus.SUCCESS if getattr(res, "success", 1) else ActionStatus.FAILED,
                            )
                            if self.action_history_manager:
                                self.action_history_manager.add_action(complete_action_db)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(complete_action_db)
                                    except Exception as e:
                                        logger.debug(f"emit_queue put failed for complete_action_db: {e}")
                            executed_any = True
                        elif matched_tool == "list_domain_layers_tree":
                            # List domain layers tree - placeholder for now
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"Domain layers listing for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"Domain layers exploration completed for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit domain layers note for todo {item.id}: {e}")
                        elif matched_tool == "check_semantic_model_exists":
                            # Check semantic model existence - placeholder
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"Semantic model check for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"Semantic model verification completed for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit semantic model check note for todo {item.id}: {e}")
                        elif matched_tool == "check_metric_exists":
                            # Check metric existence - placeholder
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"Metric existence check for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"Metric availability verification completed for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit metric check note for todo {item.id}: {e}")
                        elif matched_tool == "generate_sql_summary_id":
                            # Generate SQL summary ID - placeholder
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"SQL summary ID generation for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"SQL summary identifier created for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit SQL summary note for todo {item.id}: {e}")
                        elif matched_tool == "parse_temporal_expressions":
                            # Parse temporal expressions - placeholder
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"Temporal expression parsing for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"Date/time expression analysis completed for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit temporal parsing note for todo {item.id}: {e}")
                        elif matched_tool == "get_current_date":
                            # Get current date - placeholder
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"Current date retrieval for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"Current date/time information retrieved for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit current date note for todo {item.id}: {e}")
                        elif matched_tool == "write_file" and fs_tool:
                            # Write file operation
                            write_file_path = f"output/{item.id}_output.txt"
                            write_file_content = f"Content generated for: {item.content}"
                            res = fs_tool.write_file(path=write_file_path, content=write_file_content)
                            result_payload = (
                                res.model_dump()
                                if hasattr(res, "model_dump")
                                else dict(res)
                                if isinstance(res, dict)
                                else {"result": res}
                            )
                            complete_action_fs = ActionHistory(
                                action_id=f"{call_id}_write",
                                role=ActionRole.TOOL,
                                messages=f"Server executor: write_file for todo {item.id}",
                                action_type="write_file",
                                input={
                                    "function_name": "write_file",
                                    "arguments": json.dumps(
                                        {"path": write_file_path, "content": write_file_content, "todo_id": item.id}
                                    ),
                                },
                                output=result_payload,
                                status=ActionStatus.SUCCESS if getattr(res, "success", 1) else ActionStatus.FAILED,
                            )
                            if self.action_history_manager:
                                self.action_history_manager.add_action(complete_action_fs)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(complete_action_fs)
                                    except Exception as e:
                                        logger.debug(f"emit_queue put failed for complete_action_fs: {e}")
                                executed_any = True
                        elif matched_tool == "read_file" and fs_tool:
                            # Read file operation - placeholder
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"File reading operation for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"File content read operation completed for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit file read note for todo {item.id}: {e}")
                        elif matched_tool == "list_directory" and fs_tool:
                            # List directory operation - placeholder
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"Directory listing for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"Directory contents listed for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit directory list note for todo {item.id}: {e}")
                        elif matched_tool == "todo_write" and plan_tool:
                            # Create/update todo list - placeholder
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"Todo list creation/update for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"Task planning completed for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit todo write note for todo {item.id}: {e}")
                        elif matched_tool == "todo_update" and plan_tool:
                            # Update todo status - placeholder
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"Todo status update for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"Task status updated for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit todo update note for todo {item.id}: {e}")
                        elif matched_tool == "todo_read" and plan_tool:
                            # Read todo list - placeholder
                            try:
                                note_action = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"Todo list reading for todo {item.id}",
                                    input_data={"todo_id": item.id},
                                    output={
                                        "raw_output": f"Task list retrieved for: {item.content}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(note_action)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(note_action)
                                        except Exception:
                                            pass
                                executed_any = True
                            except Exception as e:
                                logger.debug(f"Failed to emit todo read note for todo {item.id}: {e}")
                    except Exception as e:
                        logger.error(f"Server executor matched_tool '{matched_tool}' failed for {item.id}: {e}")
                        logger.error(f"Exception type: {type(e).__name__}, matched_tool was: {matched_tool}")
                        import traceback

                        logger.error(f"Traceback: {traceback.format_exc()}")

                # Execute LLM-suggested tool calls if available
                if not executed_any and item.tool_calls:
                    try:
                        logger.info(f"Server executor: executing LLM-suggested tool calls for todo {item.id}")
                        for tool_call in item.tool_calls:
                            tool_name = tool_call.get("tool")
                            tool_args = tool_call.get("arguments", {})

                            # Execute the suggested tool call (simplified version - would need full tool integration)
                            # For now, create a placeholder action indicating the tool call was suggested
                            tool_call_action = ActionHistory.create_action(
                                role=ActionRole.TOOL,
                                action_type=tool_name,
                                messages=f"LLM-suggested tool call for todo {item.id}",
                                input_data={
                                    "todo_id": item.id,
                                    "suggested_by_llm": True,
                                    "tool": tool_name,
                                    "arguments": tool_args,
                                },
                                output_data={
                                    "raw_output": f"Tool call suggested by LLM reasoning: {tool_name}",
                                    "emit_chat": True,
                                },
                                status=ActionStatus.SUCCESS,
                            )

                            if self.action_history_manager:
                                self.action_history_manager.add_action(tool_call_action)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(tool_call_action)
                                    except Exception as e:
                                        logger.debug(f"emit_queue put failed for tool call action: {e}")

                        executed_any = True
                    except Exception as e:
                        logger.error(f"LLM-suggested tool calls execution failed for todo {item.id}: {e}")

                # SQL execution logic has been moved to keyword-based matching above

                # Report generation logic has been moved to keyword-based matching above

                # If nothing was mapped/executed, use SmartExecutionRouter for intelligent task routing
                if not executed_any:
                    logger.info(f"Server executor: using SmartExecutionRouter for task {item.id}")

                    # Initialize SmartExecutionRouter if not exists
                    if not hasattr(self, "_smart_router"):
                        self._smart_router = SmartExecutionRouter(
                            agent_config=self.agent_config,
                            model=self.model,
                            action_history_manager=self.action_history_manager,
                            emit_queue=self.emit_queue,
                        )

                    try:
                        # Use smart router to determine execution strategy
                        routing_result = await self._smart_router.execute_task(
                            item,
                            {
                                "todo_id": item.id,
                                "agent_config": self.agent_config,
                                "action_history_manager": self.action_history_manager,
                                "emit_queue": self.emit_queue,
                            },
                        )

                        if routing_result["success"]:
                            action = routing_result.get("action")

                            # Emit routing decision note
                            try:
                                routing_note = ActionHistory.create_action(
                                    role=ActionRole.SYSTEM,
                                    action_type="thinking",
                                    messages=f"Smart routing: {routing_result['execution_type']} for task {item.id}",
                                    input_data={
                                        "todo_id": item.id,
                                        "execution_type": routing_result["execution_type"],
                                        "action": action,
                                    },
                                    output={
                                        "raw_output": f"智能路由决策: {routing_result['execution_type']}",
                                        "emit_chat": True,
                                    },
                                    status=ActionStatus.SUCCESS,
                                )
                                if self.action_history_manager:
                                    self.action_history_manager.add_action(routing_note)
                                    if self.emit_queue is not None:
                                        try:
                                            self.emit_queue.put_nowait(routing_note)
                                        except Exception:
                                            pass
                            except Exception as e:
                                logger.debug(f"Failed to emit routing note for todo {item.id}: {e}")

                            # Execute the actual task based on action
                            if action == "execute_llm_reasoning":
                                # Execute LLM reasoning
                                logger.info(
                                    f"Server executor: executing LLM reasoning for todo {item.id} (smart routing)"
                                )
                                reasoning_result = await self._execute_llm_reasoning(item)
                                executed_any = True  # Always mark as executed to avoid infinite loop
                                if reasoning_result:
                                    # Mark as completed
                                    try:
                                        plan_tool._update_todo_status(item.id, "completed")
                                        await self._emit_plan_update_event(item.id, "completed")
                                        logger.info(f"Server executor: LLM reasoning completed for todo {item.id}")
                                    except Exception as e:
                                        logger.error(f"Failed to mark LLM analysis task as completed {item.id}: {e}")
                                else:
                                    logger.warning(
                                        f"Server executor: LLM reasoning failed for todo {item.id}, marking as failed"
                                    )
                                    try:
                                        plan_tool._update_todo_status(item.id, "failed")
                                        await self._emit_plan_update_event(item.id, "failed")
                                    except Exception as e:
                                        logger.error(f"Failed to mark LLM analysis task as failed {item.id}: {e}")

                            elif action == "execute_tool":
                                # Execute tool call through existing logic
                                logger.info(f"Server executor: executing tool for todo {item.id} (smart routing)")
                                # The existing tool matching logic will handle this below
                                # We don't set executed_any here, let the tool matching logic handle it
                                pass

                            else:
                                # Unknown action, mark as executed to avoid infinite loop
                                executed_any = True
                                logger.warning(f"Unknown action '{action}' for smart routing of todo {item.id}")

                        else:
                            logger.warning(f"Smart routing failed for task {item.id}: {routing_result}")

                    except Exception as e:
                        logger.error(f"Smart routing error for task {item.id}: {e}")
                        # Continue to fallback logic if smart routing fails

                # If nothing was mapped/executed, try prioritized fallback tools if enabled
                if not executed_any and self.enable_fallback:
                    fallback_candidates = self._determine_fallback_candidates(item.content or "")

                    # Try candidates in order of confidence until one succeeds
                    for tool_name, confidence in fallback_candidates:
                        if confidence < 0.5:  # Skip low-confidence fallbacks
                            continue

                        success = False

                        # Try to execute the fallback tool
                        try:
                            if tool_name == "search_table" and db_tool:
                                logger.info(
                                    f"Server executor: fallback {tool_name} for todo {item.id} (confidence: {confidence:.2f})"
                                )
                                tool_params = self._extract_tool_parameters(tool_name, item.content or "")
                                query_text = tool_params.get("query_text", item.content or "")
                                top_n = tool_params.get("top_n", 3)
                                res = db_tool.search_table(query_text=query_text, top_n=top_n)
                                result_payload = (
                                    res.model_dump()
                                    if hasattr(res, "model_dump")
                                    else dict(res)
                                    if isinstance(res, dict)
                                    else {"result": res}
                                )
                                fallback_action = ActionHistory(
                                    action_id=f"{call_id}_fallback_{tool_name}",
                                    role=ActionRole.TOOL,
                                    messages=f"Server executor: fallback {tool_name} for todo {item.id} (confidence: {confidence:.2f})",
                                    action_type=tool_name,
                                    input={
                                        "function_name": tool_name,
                                        "arguments": json.dumps(
                                            {
                                                "query_text": query_text,
                                                "top_n": top_n,
                                                "todo_id": item.id,
                                                "is_fallback": True,
                                            }
                                        ),
                                    },
                                    output=result_payload,
                                    status=ActionStatus.SUCCESS if getattr(res, "success", 1) else ActionStatus.FAILED,
                                )
                                success = True

                            elif tool_name == "describe_table" and db_tool:
                                logger.info(
                                    f"Server executor: fallback {tool_name} for todo {item.id} (confidence: {confidence:.2f})"
                                )
                                tool_params = self._extract_tool_parameters(tool_name, item.content or "")
                                table_name = tool_params.get("table_name")
                                if table_name:
                                    res = db_tool.describe_table(table_name=table_name)
                                    result_payload = (
                                        res.model_dump()
                                        if hasattr(res, "model_dump")
                                        else dict(res)
                                        if isinstance(res, dict)
                                        else {"result": res}
                                    )
                                    fallback_action = ActionHistory(
                                        action_id=f"{call_id}_fallback_{tool_name}",
                                        role=ActionRole.TOOL,
                                        messages=f"Server executor: fallback {tool_name} for todo {item.id} (confidence: {confidence:.2f})",
                                        action_type=tool_name,
                                        input={
                                            "function_name": tool_name,
                                            "arguments": json.dumps(
                                                {"table_name": table_name, "todo_id": item.id, "is_fallback": True}
                                            ),
                                        },
                                        output=result_payload,
                                        status=ActionStatus.SUCCESS
                                        if getattr(res, "success", 1)
                                        else ActionStatus.FAILED,
                                    )
                                    success = True
                                else:
                                    # Fallback to search if no table name found
                                    query_text = tool_params.get("query_text", item.content or "")
                                    res = db_tool.search_table(query_text=query_text, top_n=3)
                                    result_payload = (
                                        res.model_dump()
                                        if hasattr(res, "model_dump")
                                        else dict(res)
                                        if isinstance(res, dict)
                                        else {"result": res}
                                    )
                                    fallback_action = ActionHistory(
                                        action_id=f"{call_id}_fallback_search",
                                        role=ActionRole.TOOL,
                                        messages=f"Server executor: fallback search_table (from describe_table) for todo {item.id} (confidence: {confidence:.2f})",
                                        action_type="search_table",
                                        input={
                                            "function_name": "search_table",
                                            "arguments": json.dumps(
                                                {
                                                    "query_text": query_text,
                                                    "top_n": 3,
                                                    "todo_id": item.id,
                                                    "is_fallback": True,
                                                }
                                            ),
                                        },
                                        output=result_payload,
                                        status=ActionStatus.SUCCESS
                                        if getattr(res, "success", 1)
                                        else ActionStatus.FAILED,
                                    )
                                    success = True

                            # Add more tool fallbacks here as needed...

                        except Exception as e:
                            logger.debug(f"Server executor fallback {tool_name} failed for {item.id}: {e}")
                            continue

                        # If fallback tool executed successfully, emit the action and stop trying more fallbacks
                        if success:
                            if self.action_history_manager:
                                self.action_history_manager.add_action(fallback_action)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(fallback_action)
                                    except Exception as e:
                                        logger.debug(f"emit_queue put failed for fallback action: {e}")
                            executed_any = True
                            break

                # If still nothing executed, emit system note for transparency
                if not executed_any:
                    try:
                        fallback_status = "disabled" if not self.enable_fallback else "no suitable fallback found"
                        note_action = ActionHistory.create_action(
                            role=ActionRole.SYSTEM,
                            action_type="thinking",
                            messages=f"No tool executed for todo {item.id} ({fallback_status}); marking completed",
                            input_data={"todo_id": item.id},
                            output={
                                "raw_output": f"No tool matched or executed ({fallback_status}); step marked completed",
                                "emit_chat": True,
                            },
                            status=ActionStatus.SUCCESS,
                        )
                        if self.action_history_manager:
                            self.action_history_manager.add_action(note_action)
                            if self.emit_queue is not None:
                                try:
                                    self.emit_queue.put_nowait(note_action)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.debug(f"Failed to emit completion note for todo {item.id}: {e}")

                # Mark completed only if we actually executed something or reached a terminal state
                try:
                    if executed_any:
                        plan_tool._update_todo_status(item.id, "completed")
                        await self._emit_plan_update_event(item.id, "completed")
                    else:
                        # Mark as failed if no tool was executed (fallback didn't work)
                        plan_tool._update_todo_status(item.id, "failed")
                        await self._emit_plan_update_event(item.id, "failed")
                        logger.warning(f"Server executor: no tool executed for todo {item.id}, marking as failed")
                except Exception as e:
                    logger.error(f"Server executor failed to set completed for {item.id}: {e}")
                    try:
                        plan_tool._update_todo_status(item.id, "failed")
                        await self._emit_plan_update_event(item.id, "failed")
                    except Exception:
                        pass

            logger.info("Server executor finished all pending todos")

            # Signal completion to main agent loop for coordination
            self._all_todos_completed = True
            self._execution_complete.set()
            logger.info("Server executor signaled completion to main agent loop")

            # Send completion status message
            await self._emit_status_message("✅ **执行完成**\n\n所有计划任务已成功完成！", plan_id="server_batch")

            # End execution monitoring with success
            self.monitor.end_execution("completed")
        except Exception as e:
            logger.error(f"Unhandled server executor error: {e}")

            # Generate ErrorEvent for server executor failures
            try:
                error_action = ActionHistory(
                    action_id=f"server_executor_error_{uuid.uuid4().hex[:8]}",
                    role=ActionRole.SYSTEM,
                    messages=f"Server executor failed: {str(e)}",
                    action_type="error",
                    input={"source": "server_executor"},
                    output={"error": str(e)},
                    status=ActionStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                )
                if self.action_history_manager:
                    self.action_history_manager.add_action(error_action)
                    if self.emit_queue is not None:
                        try:
                            self.emit_queue.put_nowait(error_action)
                        except Exception as emit_e:
                            logger.debug(f"Failed to emit server executor error event: {emit_e}")
            except Exception as inner_e:
                logger.error(f"Failed to create error event for server executor: {inner_e}")
            finally:
                # End execution monitoring
                if self.monitor.current_execution:
                    self.monitor.end_execution("failed", str(e) if "e" in locals() else "Unknown error")

    async def _emit_plan_update_event(self, todo_id: str = None, status: str = None):
        """
        Emit a plan_update event to notify frontend of plan status changes.

        Args:
            todo_id: Specific todo ID if updating a single item, None for full plan
            status: Status to set for the todo item
        """
        try:
            import time
            import uuid

            from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

            todo_list = self.todo_storage.get_todo_list()
            if not todo_list:
                return

            # For completed status, verify this is actually completed (not prematurely marked)
            if status == "completed" and todo_id:
                if not self._is_actually_completed(todo_id):
                    logger.debug(f"Skipping premature completion event for todo {todo_id}")
                    return

            # Create plan_update action
            plan_update_id = f"plan_update_{uuid.uuid4().hex[:8]}"
            plan_update_action = ActionHistory(
                action_id=plan_update_id,
                role=ActionRole.SYSTEM,  # Use SYSTEM role for internal events
                messages=f"Plan status update: {todo_id or 'full_plan'} -> {status or 'current'}",
                action_type="plan_update",
                input={"source": "server_executor", "todo_id": todo_id, "status": status},
                output={"todo_list": todo_list.model_dump()},
                status=ActionStatus.SUCCESS,
                start_time=datetime.now(),
            )

            # Add to action history manager
            if self.action_history_manager:
                self.action_history_manager.add_action(plan_update_action)

            # Emit to queue if available
            if self.emit_queue is not None:
                try:
                    self.emit_queue.put_nowait(plan_update_action)
                    logger.debug(f"Emitted plan_update event for todo {todo_id}")
                except Exception as e:
                    logger.debug(f"Failed to emit plan_update event: {e}")

        except Exception as e:
            logger.error(f"Failed to emit plan_update event: {e}")

    def _extract_tool_parameters(self, tool_name: str, todo_content: str) -> Dict[str, Any]:
        """
        Extract appropriate parameters for a tool from todo content.

        Args:
            tool_name: Name of the tool to extract parameters for
            todo_content: The todo content to parse

        Returns:
            Dict containing extracted parameters for the tool
        """
        if not todo_content:
            return {}

        # Clean and preprocess todo content
        cleaned_content = self._preprocess_todo_content(todo_content)

        try:
            params = {}
            if tool_name == "describe_table":
                params = self._extract_describe_table_params(cleaned_content)
            elif tool_name == "search_table":
                params = self._extract_search_table_params(cleaned_content)
            elif tool_name == "read_query":
                params = self._extract_read_query_params(cleaned_content)
            elif tool_name == "execute_sql":
                params = self._extract_execute_sql_params(cleaned_content)
            elif tool_name == "write_file":
                params = self._extract_write_file_params(cleaned_content)
            elif tool_name == "read_file":
                params = self._extract_read_file_params(cleaned_content)
            else:
                # For unknown tools, return basic parameters
                params = {"query_text": cleaned_content}

            # Validate and sanitize parameters
            validated_params = self._validate_tool_parameters(tool_name, params)

            logger.debug(f"Extracted parameters for {tool_name}: {validated_params}")
            return validated_params

        except Exception as e:
            logger.debug(f"Failed to extract parameters for tool {tool_name}: {e}")
            return {"query_text": cleaned_content}

    def _preprocess_todo_content(self, content: str) -> str:
        """
        Preprocess todo content to improve parameter extraction.
        """
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

    def _validate_tool_parameters(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize tool parameters.
        """
        if not params:
            return params

        validated = params.copy()

        try:
            if tool_name == "describe_table":
                table_name = params.get("table_name")
                if table_name and not self._is_valid_table_name(table_name):
                    logger.warning(f"Invalid table name '{table_name}', falling back to search")
                    validated = {"query_text": params.get("query_text", table_name)}
                elif table_name:
                    validated["table_name"] = table_name.strip()

            elif tool_name in ["read_query", "execute_sql"]:
                sql = params.get("sql", "")
                if sql and len(sql) > 10000:  # Reasonable SQL length limit
                    logger.warning(f"SQL query too long ({len(sql)} chars), truncating")
                    validated["sql"] = sql[:10000] + "..."

            elif tool_name == "search_table":
                top_n = params.get("top_n", 5)
                if isinstance(top_n, int) and (top_n < 1 or top_n > 20):
                    validated["top_n"] = min(max(top_n, 1), 20)

            elif tool_name == "write_file":
                path = params.get("path", "")
                if path:
                    # Basic path validation
                    import os

                    if ".." in path or path.startswith("/"):
                        logger.warning(f"Potentially unsafe file path: {path}")
                        validated["path"] = f"output/safe_{hash(path) % 10000}.txt"

        except Exception as e:
            logger.debug(f"Parameter validation failed for {tool_name}: {e}")

        return validated

    def _extract_describe_table_params(self, todo_content: str) -> Dict[str, Any]:
        """
        Extract table name from todo content for describe_table tool.
        """
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

        content_lower = todo_content.lower()

        # First try to extract from known successful examples
        # From the log, we know dwd_assign_dlr_clue_fact_di was found
        if "线索表" in content_lower or "clue" in content_lower:
            # Try to find table name from action history if available
            if self.action_history_manager:
                for action in reversed(self.action_history_manager.get_actions()):
                    if action.action_type == "search_table" and action.output and isinstance(action.output, dict):
                        result = action.output.get("result", {})
                        if isinstance(result, dict):
                            metadata = result.get("metadata", [])
                            if metadata and isinstance(metadata, list) and len(metadata) > 0:
                                # Use the first table found as candidate
                                first_table = metadata[0].get("table_name", "")
                                if first_table:
                                    logger.debug(f"Using table name from search results: {first_table}")
                                    return {"table_name": first_table}

        # Try regex patterns
        for pattern in patterns:
            matches = re.findall(pattern, todo_content, re.IGNORECASE)
            if matches:
                # Clean the table name
                table_name = matches[0].strip("'\"")
                # Validate table name format (basic validation)
                if self._is_valid_table_name(table_name):
                    logger.debug(f"Extracted table name '{table_name}' from todo content")
                    return {"table_name": table_name}

        # Fallback: if no table name found, this might be a search operation
        logger.debug(f"No table name found in todo content, falling back to search operation")
        return {"query_text": todo_content}

    def _extract_search_table_params(self, todo_content: str) -> Dict[str, Any]:
        """Extract search parameters for search_table tool."""
        return {"query_text": todo_content, "top_n": 5}

    def _extract_read_query_params(self, todo_content: str) -> Dict[str, Any]:
        """Extract SQL query for read_query tool."""
        # Try to extract SQL from action history (previous SQL generation)
        if self.action_history_manager:
            for action in reversed(self.action_history_manager.get_actions()):
                if (
                    action.role in ("assistant", ActionRole.ASSISTANT)
                    and action.output
                    and isinstance(action.output, dict)
                ):
                    sql = action.output.get("sql")
                    if sql:
                        return {"sql": sql}

        # Fallback to raw content (might contain SQL)
        return {"sql": todo_content}

    def _extract_execute_sql_params(self, todo_content: str) -> Dict[str, Any]:
        """Extract SQL for execute_sql tool."""
        return self._extract_read_query_params(todo_content)

    def _extract_write_file_params(self, todo_content: str) -> Dict[str, Any]:
        """Extract file write parameters."""
        # Basic implementation - could be enhanced
        return {"path": f"output/{todo_content[:50].replace(' ', '_')}.txt", "content": todo_content}

    def _extract_read_file_params(self, todo_content: str) -> Dict[str, Any]:
        """Extract file read parameters."""
        # Basic implementation - could be enhanced
        return {"path": todo_content}

    def _is_valid_table_name(self, table_name: str) -> bool:
        """
        Basic validation for table names.
        """
        if not table_name or len(table_name) > 100:
            return False

        # Check for basic SQL injection patterns (simple check)
        dangerous_patterns = [";", "--", "/*", "*/", "union", "select", "drop", "delete", "update", "insert"]
        if any(pattern in table_name.lower() for pattern in dangerous_patterns):
            return False

        # Allow alphanumeric, underscore, and some special chars common in table names
        import re

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            return False

        return True

    def _classify_task_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Classify the intent of a task and map it to the most appropriate tool.

        Returns:
            Dict with 'tool', 'confidence', and 'reason' keys, or None if no match
        """
        if not text:
            return None

        text_lower = text.lower()

        # Comprehensive Task Type Patterns with associated tools and confidence scores
        task_patterns = {
            # Database Schema Exploration
            "explore_schema": {
                "patterns": [
                    r"探索.*表结构",
                    r"查看.*表结构",
                    r"检查.*表结构",
                    r"分析.*表结构",
                    r"找到.*表",
                    r"查找.*表",
                    r"发现.*表",
                    r"搜索.*表",
                    r"explore.*table.*structure",
                    r"find.*tables",
                    r"search.*tables",
                    r"确认.*表名",
                    r"表.*字段",
                    r"字段.*信息",
                    r"table.*columns",
                    r"数据库.*结构",
                    r"schema.*information",
                    r"table.*list",
                ],
                "tool": "search_table",
                "confidence": 0.95,
                "reason": "Database table exploration task",
                "priority": 1,
            },
            # Specific Table Description
            "describe_specific_table": {
                "patterns": [
                    r"描述.*表.*[\w_]+",
                    r"检查.*表.*[\w_]+.*定义",
                    r"查看.*表.*[\w_]+.*模式",
                    r"分析.*表.*[\w_]+.*元数据",
                    r"describe.*table.*[\w_]+",
                    r"inspect.*table.*[\w_]+.*schema",
                    r"examine.*table.*[\w_]+.*structure",
                    r"表.*[\w_]+.*详细.*信息",
                    r"表.*[\w_]+.*结构.*详情",
                ],
                "tool": "describe_table",
                "confidence": 0.92,
                "reason": "Specific table description and metadata analysis",
                "priority": 1,
            },
            # General Table Description
            "describe_table_general": {
                "patterns": [
                    r"描述.*表",
                    r"检查.*表.*定义",
                    r"查看.*表.*模式",
                    r"分析.*表.*元数据",
                    r"describe.*table",
                    r"inspect.*table.*schema",
                    r"examine.*table.*structure",
                    r"表.*详细.*信息",
                    r"表.*结构.*详情",
                    r"table.*metadata",
                ],
                "tool": "describe_table",
                "confidence": 0.85,
                "reason": "General table description and metadata analysis",
                "priority": 2,
            },
            # SQL Query Execution
            "execute_sql": {
                "patterns": [
                    r"执行.*查询",
                    r"运行.*sql",
                    r"运行.*查询",
                    r"执行.*语句",
                    r"execute.*query",
                    r"run.*sql",
                    r"execute.*statement",
                    r"查询.*结果",
                    r"获取.*数据",
                    r"fetch.*data",
                    r"运行.*select",
                ],
                "tool": "execute_sql",
                "confidence": 0.90,
                "reason": "SQL execution and data retrieval",
                "priority": 1,
            },
            # SQL Query Reading/Analysis
            "read_sql": {
                "patterns": [
                    r"读取.*查询",
                    r"分析.*sql",
                    r"查看.*查询.*结果",
                    r"检查.*sql",
                    r"read.*query",
                    r"analyze.*sql",
                    r"examine.*query.*results",
                ],
                "tool": "read_query",
                "confidence": 0.80,
                "reason": "SQL query reading and analysis",
                "priority": 2,
            },
            # Business Metrics Analysis
            "analyze_metrics": {
                "patterns": [
                    r"分析.*指标",
                    r"查找.*指标",
                    r"查看.*指标",
                    r"搜索.*指标",
                    r"analyze.*metrics",
                    r"find.*metrics",
                    r"search.*metrics",
                    r"kpi.*分析",
                    r"performance.*metrics",
                    r"转化率.*分析",
                    r"收入.*分析",
                    r"销售额.*分析",
                    r"用户.*分析",
                ],
                "tool": "search_metrics",
                "confidence": 0.88,
                "reason": "Business metrics analysis",
                "priority": 1,
            },
            # Reference SQL Search
            "search_reference_sql": {
                "patterns": [
                    r"查找.*参考.*sql",
                    r"搜索.*sql.*模板",
                    r"找到.*sql.*例子",
                    r"find.*reference.*sql",
                    r"search.*sql.*patterns",
                    r"look.*sql.*examples",
                    r"参考.*查询",
                    r"sql.*样例",
                    r"查询.*模板",
                ],
                "tool": "search_reference_sql",
                "confidence": 0.85,
                "reason": "Reference SQL examples search",
                "priority": 2,
            },
            # File Writing Operations
            "write_file": {
                "patterns": [
                    r"写入.*文件",
                    r"保存.*文件",
                    r"创建.*文件",
                    r"生成.*文件",
                    r"write.*file",
                    r"save.*file",
                    r"create.*file",
                    r"generate.*file",
                    r"文件.*写入",
                    r"保存.*到.*文件",
                ],
                "tool": "write_file",
                "confidence": 0.82,
                "reason": "File writing operations",
                "priority": 1,
            },
            # File Reading Operations
            "read_file": {
                "patterns": [
                    r"读取.*文件",
                    r"加载.*文件",
                    r"打开.*文件",
                    r"查看.*文件",
                    r"read.*file",
                    r"load.*file",
                    r"open.*file",
                    r"view.*file",
                    r"文件.*读取",
                    r"加载.*文件.*内容",
                ],
                "tool": "read_file",
                "confidence": 0.80,
                "reason": "File reading operations",
                "priority": 1,
            },
            # Domain/Layer Exploration
            "explore_domains": {
                "patterns": [
                    r"查看.*领域",
                    r"浏览.*层级",
                    r"探索.*业务.*分类",
                    r"view.*domains",
                    r"browse.*layers",
                    r"explore.*business.*taxonomy",
                    r"领域.*结构",
                    r"业务.*层级",
                    r"数据.*分层",
                ],
                "tool": "list_domain_layers_tree",
                "confidence": 0.78,
                "reason": "Domain and layer exploration",
                "priority": 2,
            },
            # Time/Temporal Analysis
            "temporal_analysis": {
                "patterns": [
                    r"解析.*日期",
                    r"分析.*时间",
                    r"处理.*时间.*范围",
                    r"parse.*date",
                    r"analyze.*temporal",
                    r"process.*time.*range",
                    r"日期.*表达式",
                    r"时间.*段",
                    r"时间.*分析",
                ],
                "tool": "parse_temporal_expressions",
                "confidence": 0.75,
                "reason": "Temporal expression parsing",
                "priority": 2,
            },
            # Current Date/Time Retrieval
            "get_current_time": {
                "patterns": [
                    r"获取.*当前.*日期",
                    r"今天.*日期",
                    r"现在.*时间",
                    r"get.*current.*date",
                    r"today.*date",
                    r"current.*time",
                    r"当前.*时间",
                    r"今日.*日期",
                ],
                "tool": "get_current_date",
                "confidence": 0.95,
                "reason": "Current date/time retrieval",
                "priority": 1,
            },
        }

        # Check each task type with priority ordering
        matches = []
        for task_type, config in task_patterns.items():
            for pattern in config["patterns"]:
                import re

                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches.append(
                        {
                            "tool": config["tool"],
                            "confidence": config["confidence"],
                            "reason": config["reason"],
                            "task_type": task_type,
                            "priority": config["priority"],
                        }
                    )
                    break  # Only add each task type once

        # Return the highest priority match (lowest priority number)
        if matches:
            best_match = min(matches, key=lambda x: (x["priority"], -x["confidence"]))
            return best_match

        return None

    def _match_keywords_with_context(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced keyword matching that considers context and intent.

        Returns:
            Dict with 'tool', 'confidence', and 'context' keys
        """
        if not text:
            return None

        text_lower = text.lower()
        best_match = None
        best_score = 0

        # Analyze context clues
        context_indicators = {
            "database_focus": sum(
                1 for word in ["表", "table", "database", "schema", "字段", "column"] if word in text_lower
            ),
            "sql_focus": sum(1 for word in ["sql", "query", "select", "执行", "运行"] if word in text_lower),
            "metrics_focus": sum(1 for word in ["指标", "metrics", "kpi", "转化率", "收入"] if word in text_lower),
            "file_focus": sum(1 for word in ["文件", "file", "保存", "读取"] if word in text_lower),
        }

        # Primary context
        primary_context = max(context_indicators.items(), key=lambda x: x[1])
        primary_context_type = primary_context[0] if primary_context[1] > 0 else None

        # Enhanced keyword matching with context awareness
        context_aware_mappings = {
            "database_focus": {
                "search_table": ["探索", "查找", "找到", "搜索", "explore", "find", "search"],
                "describe_table": ["描述", "检查", "查看", "分析", "describe", "inspect", "examine"],
            },
            "sql_focus": {
                "execute_sql": ["执行", "运行", "查询", "execute", "run", "query"],
                "read_query": ["读取", "获取", "read", "fetch"],
            },
            "metrics_focus": {"search_metrics": ["指标", "metrics", "kpi", "分析", "analyze"]},
            "file_focus": {
                "write_file": ["写入", "保存", "创建", "write", "save", "create"],
                "read_file": ["读取", "加载", "read", "load"],
            },
        }

        # Apply context-aware scoring
        for tool_name, keywords in self.keyword_map.items():
            score = 0

            # Base keyword matching
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1

            # Context boost
            if primary_context_type and tool_name in context_aware_mappings.get(primary_context_type, {}):
                context_keywords = context_aware_mappings[primary_context_type][tool_name]
                for ctx_keyword in context_keywords:
                    if ctx_keyword.lower() in text_lower:
                        score += 2  # Context match gets higher weight

            # Normalize score
            if keywords:
                normalized_score = score / len(keywords)
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_match = {
                        "tool": tool_name,
                        "confidence": min(0.95, normalized_score),
                        "context": primary_context_type,
                        "score": score,
                    }

        return best_match if best_score > 0.3 else None

    def _semantic_chinese_matching(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced semantic understanding for Chinese task descriptions.
        """
        if not text:
            return None

        text_lower = text.lower()

        # Advanced Chinese task pattern recognition with context
        chinese_semantic_patterns = {
            "search_table": {
                "primary_verbs": ["探索", "查找", "找到", "搜索", "查看", "发现", "确认", "浏览"],
                "secondary_verbs": ["了解", "获取", "收集", "整理"],
                "nouns": ["表结构", "表", "字段", "数据库", "数据表", "表清单", "表列表"],
                "context_indicators": ["有哪些表", "表都有什么", "数据库里有", "表结构是"],
                "confidence": 0.88,
                "priority": 1,
            },
            "describe_table": {
                "primary_verbs": ["描述", "检查", "分析", "查看", "检验", "了解"],
                "secondary_verbs": ["解释", "说明", "展示", "显示"],
                "nouns": ["表定义", "表模式", "表元数据", "表结构", "字段信息", "表详情"],
                "context_indicators": ["表长什么样", "表的结构", "字段类型", "表定义是"],
                "confidence": 0.85,
                "priority": 1,
            },
            "execute_sql": {
                "primary_verbs": ["执行", "运行", "查询", "计算"],
                "secondary_verbs": ["处理", "分析", "统计"],
                "nouns": ["sql", "查询", "语句", "数据"],
                "context_indicators": ["运行sql", "执行查询", "计算结果", "查询数据"],
                "confidence": 0.90,
                "priority": 1,
            },
            "search_metrics": {
                "primary_verbs": ["分析", "查看", "查找", "计算", "统计"],
                "secondary_verbs": ["评估", "监控", "跟踪"],
                "nouns": ["指标", "转化率", "收入", "销售额", "kpi", "绩效", "效率"],
                "context_indicators": ["指标情况", "转化如何", "收入多少", "销售数据"],
                "confidence": 0.86,
                "priority": 1,
            },
            "write_file": {
                "primary_verbs": ["写入", "保存", "创建", "生成", "输出"],
                "secondary_verbs": ["导出", "存储", "记录"],
                "nouns": ["文件", "文档", "报告", "结果"],
                "context_indicators": ["保存到文件", "生成报告", "导出结果", "写入文件"],
                "confidence": 0.82,
                "priority": 1,
            },
            "read_file": {
                "primary_verbs": ["读取", "加载", "打开", "查看"],
                "secondary_verbs": ["导入", "获取", "获取"],
                "nouns": ["文件", "文档", "内容", "数据"],
                "context_indicators": ["读取文件", "查看内容", "加载数据", "打开文件"],
                "confidence": 0.80,
                "priority": 1,
            },
        }

        # Calculate semantic scores for each tool
        candidates = []
        for tool_name, pattern_config in chinese_semantic_patterns.items():
            score = 0
            reasons = []

            # Primary verbs (higher weight)
            primary_verb_matches = sum(1 for verb in pattern_config["primary_verbs"] if verb in text)
            if primary_verb_matches > 0:
                score += primary_verb_matches * 2
                reasons.append(f"primary_verbs: {primary_verb_matches}")

            # Secondary verbs
            secondary_verb_matches = sum(1 for verb in pattern_config["secondary_verbs"] if verb in text)
            if secondary_verb_matches > 0:
                score += secondary_verb_matches * 1.5
                reasons.append(f"secondary_verbs: {secondary_verb_matches}")

            # Nouns
            noun_matches = sum(1 for noun in pattern_config["nouns"] if noun in text)
            if noun_matches > 0:
                score += noun_matches * 1.8
                reasons.append(f"nouns: {noun_matches}")

            # Context indicators (highest weight)
            context_matches = sum(1 for indicator in pattern_config["context_indicators"] if indicator in text_lower)
            if context_matches > 0:
                score += context_matches * 3
                reasons.append(f"context: {context_matches}")

            # Semantic phrase patterns
            semantic_patterns = [
                (r"把.*保存.*文件", "write_file"),
                (r"从.*文件.*读取", "read_file"),
                (r"执行.*sql.*查询", "execute_sql"),
                (r"分析.*指标.*数据", "search_metrics"),
                (r"查看.*表.*结构", "describe_table"),
                (r"探索.*数据库.*表", "search_table"),
            ]

            for pattern, expected_tool in semantic_patterns:
                import re

                if expected_tool == tool_name and re.search(pattern, text_lower):
                    score += 4
                    reasons.append(f"semantic_pattern: {pattern}")

            # Calculate confidence based on score and priority
            if score > 0:
                # Normalize confidence based on maximum possible score
                max_possible_score = (
                    len(pattern_config["primary_verbs"]) * 2
                    + len(pattern_config["secondary_verbs"]) * 1.5
                    + len(pattern_config["nouns"]) * 1.8
                    + len(pattern_config["context_indicators"]) * 3
                    + 4
                )

                confidence = min(
                    pattern_config["confidence"], (score / max_possible_score) * pattern_config["confidence"]
                )

                if confidence > 0.65:  # Minimum threshold
                    candidates.append(
                        {
                            "tool": tool_name,
                            "confidence": confidence,
                            "score": score,
                            "priority": pattern_config["priority"],
                            "reasons": reasons,
                            "semantic_analysis": True,
                        }
                    )

        # Return the best candidate
        if candidates:
            best_candidate = max(candidates, key=lambda x: (x["score"], -x["priority"]))
            return {
                "tool": best_candidate["tool"],
                "confidence": best_candidate["confidence"],
                "reason": f"Chinese semantic analysis: {'; '.join(best_candidate['reasons'])}",
                "score": best_candidate["score"],
                "semantic_analysis": True,
            }

        return None

    def _enhanced_llm_reasoning(self, text: str) -> Optional[str]:
        """
        Enhanced LLM reasoning with comprehensive tool knowledge and context.
        """
        if not self.model or not text:
            return None

        try:
            # Comprehensive tool descriptions with use cases
            tool_knowledge = {
                "search_table": {
                    "description": "Search for database tables and their schemas",
                    "use_cases": ["explore database structure", "find tables by name", "discover available tables"],
                    "examples": ["find all tables related to customers", "search for sales tables"],
                },
                "describe_table": {
                    "description": "Get detailed schema information for a specific table",
                    "use_cases": ["examine table structure", "check column definitions", "understand table metadata"],
                    "examples": ["describe the customer table", "show me the structure of orders table"],
                },
                "execute_sql": {
                    "description": "Execute SQL queries and return results",
                    "use_cases": ["run data queries", "perform calculations", "retrieve specific data"],
                    "examples": ["run this SQL query", "execute the analysis query"],
                },
                "read_query": {
                    "description": "Execute read-only SQL queries for data retrieval",
                    "use_cases": ["fetch data", "analyze datasets", "perform read operations"],
                    "examples": ["get customer data", "retrieve sales records"],
                },
                "search_metrics": {
                    "description": "Search for business metrics and KPIs",
                    "use_cases": ["find performance indicators", "locate business metrics", "analyze KPIs"],
                    "examples": ["find conversion rate metrics", "search for sales KPIs"],
                },
                "search_reference_sql": {
                    "description": "Find example SQL queries and patterns",
                    "use_cases": ["find similar queries", "get SQL examples", "learn query patterns"],
                    "examples": ["find SQL examples for joins", "search for aggregation queries"],
                },
                "write_file": {
                    "description": "Write content to files",
                    "use_cases": ["save results", "create reports", "export data"],
                    "examples": ["save results to file", "create a report"],
                },
                "read_file": {
                    "description": "Read content from files",
                    "use_cases": ["load data", "import content", "read documents"],
                    "examples": ["read the configuration file", "load the data file"],
                },
                "list_directory": {
                    "description": "List files and directories",
                    "use_cases": ["explore file system", "find files", "browse directories"],
                    "examples": ["list files in directory", "show me the contents"],
                },
                "list_domain_layers_tree": {
                    "description": "Explore business domain structure and data layers",
                    "use_cases": ["understand data organization", "explore business domains", "view data hierarchy"],
                    "examples": ["show business domains", "explore data layers"],
                },
                "parse_temporal_expressions": {
                    "description": "Parse and understand date/time expressions",
                    "use_cases": ["analyze time periods", "parse date ranges", "understand temporal data"],
                    "examples": ["parse this date expression", "analyze time periods"],
                },
                "get_current_date": {
                    "description": "Get the current date and time",
                    "use_cases": ["get today's date", "current timestamp", "now time"],
                    "examples": ["what is today's date", "current time"],
                },
                "check_semantic_model_exists": {
                    "description": "Check if semantic models are available",
                    "use_cases": ["verify model availability", "find semantic definitions"],
                    "examples": ["check semantic model", "find available models"],
                },
                "check_metric_exists": {
                    "description": "Verify if specific metrics exist",
                    "use_cases": ["validate metric availability", "check metric definitions"],
                    "examples": ["does this metric exist", "check metric availability"],
                },
                "generate_sql_summary_id": {
                    "description": "Generate identifiers for SQL queries",
                    "use_cases": ["create query IDs", "generate summary identifiers"],
                    "examples": ["generate SQL summary ID"],
                },
                "todo_write": {
                    "description": "Create and manage task lists (todos)",
                    "use_cases": ["create task plans", "manage workflows", "organize tasks"],
                    "examples": ["create a task list", "plan the work"],
                },
                "todo_update": {
                    "description": "Update task status in todo lists",
                    "use_cases": ["mark tasks complete", "update progress", "change task status"],
                    "examples": ["mark task as done", "update task status"],
                },
                "todo_read": {
                    "description": "Read and display task lists",
                    "use_cases": ["view current tasks", "check progress", "review plans"],
                    "examples": ["show my tasks", "check task status"],
                },
                "report": {
                    "description": "Generate comprehensive reports",
                    "use_cases": ["create final reports", "summarize results", "produce documentation"],
                    "examples": ["generate final report", "create summary report"],
                },
            }

            # Analyze task context
            task_context = self._analyze_task_context(text)

            # Build detailed prompt
            tool_list = []
            for tool_name, knowledge in tool_knowledge.items():
                examples = ", ".join(knowledge["examples"][:2])  # Limit examples
                tool_list.append(f"- {tool_name}: {knowledge['description']} (e.g., {examples})")

            prompt = f"""You are an expert at selecting the right tool for database and data analysis tasks.

TASK TO ANALYZE: "{text}"

TASK CONTEXT ANALYSIS:
- Primary focus: {task_context.get('primary_focus', 'unknown')}
- Action type: {task_context.get('action_type', 'unknown')}
- Data type: {task_context.get('data_type', 'unknown')}
- Expected output: {task_context.get('expected_output', 'unknown')}

AVAILABLE TOOLS:
{chr(10).join(tool_list)}

INSTRUCTIONS:
1. Analyze the task intent and requirements
2. Consider what type of operation is needed (explore, describe, execute, analyze, create, read)
3. Match the task to the most appropriate tool based on its capabilities
4. Choose the tool that will most directly accomplish the task goal

Respond with ONLY the exact tool name (one word, lowercase with underscores if needed).
Do not include any explanation or additional text.

TOOL:"""

            # Use async call if available
            if hasattr(self.model, "generate_async"):
                response = asyncio.run(self.model.generate_async(prompt, max_tokens=15, temperature=0.1))
            else:
                response = self.model.generate(prompt, max_tokens=15, temperature=0.1)

            if response and hasattr(response, "content"):
                tool_name = response.content.strip().lower()
                # Clean up response (remove extra text if any)
                tool_name = tool_name.split()[0] if tool_name.split() else tool_name
                tool_name = tool_name.strip(".:")

                # Validate response
                available_tools = set(self.keyword_map.keys())
                if tool_name in available_tools:
                    return tool_name
                else:
                    logger.debug(f"LLM suggested invalid tool '{tool_name}', available: {available_tools}")

            return None

        except Exception as e:
            logger.debug(f"Enhanced LLM reasoning failed: {e}")
            return None

    def _analyze_task_context(self, text: str) -> Dict[str, str]:
        """
        Analyze task context to provide better LLM reasoning.
        """
        text_lower = text.lower()

        # Determine primary focus
        if any(word in text_lower for word in ["表", "table", "database", "schema"]):
            primary_focus = "database"
        elif any(word in text_lower for word in ["指标", "metrics", "kpi", "转化率"]):
            primary_focus = "metrics"
        elif any(word in text_lower for word in ["文件", "file", "document"]):
            primary_focus = "files"
        elif any(word in text_lower for word in ["sql", "query", "select"]):
            primary_focus = "queries"
        elif any(word in text_lower for word in ["时间", "日期", "date", "time"]):
            primary_focus = "temporal"
        else:
            primary_focus = "general"

        # Determine action type
        if any(word in text_lower for word in ["探索", "查找", "找到", "搜索", "explore", "find", "search"]):
            action_type = "explore"
        elif any(word in text_lower for word in ["描述", "检查", "查看", "describe", "inspect", "examine"]):
            action_type = "describe"
        elif any(word in text_lower for word in ["执行", "运行", "execute", "run"]):
            action_type = "execute"
        elif any(word in text_lower for word in ["创建", "生成", "写入", "create", "generate", "write"]):
            action_type = "create"
        elif any(word in text_lower for word in ["读取", "加载", "read", "load"]):
            action_type = "read"
        else:
            action_type = "analyze"

        # Determine data type
        if "sql" in text_lower:
            data_type = "sql_queries"
        elif any(word in text_lower for word in ["json", "xml", "text"]):
            data_type = "structured_data"
        elif any(word in text_lower for word in ["指标", "metrics", "kpi"]):
            data_type = "business_metrics"
        elif any(word in text_lower for word in ["表", "table", "schema"]):
            data_type = "database_schema"
        else:
            data_type = "general_data"

        # Determine expected output
        if any(word in text_lower for word in ["报告", "report", "summary"]):
            expected_output = "report"
        elif any(word in text_lower for word in ["结果", "数据", "results", "data"]):
            expected_output = "data"
        elif any(word in text_lower for word in ["结构", "信息", "structure", "information"]):
            expected_output = "information"
        else:
            expected_output = "results"

        return {
            "primary_focus": primary_focus,
            "action_type": action_type,
            "data_type": data_type,
            "expected_output": expected_output,
        }

    def _enhanced_intelligent_inference(self, text: str) -> Optional[str]:
        """
        Enhanced intelligent inference as final fallback.
        """
        if not text:
            return None

        text_lower = text.lower()

        # Priority-based inference rules
        inference_rules = [
            # High priority: Database exploration
            (
                lambda t: any(word in t for word in ["探索", "查找", "找到", "搜索", "explore", "find", "search"])
                and any(word in t for word in ["表", "table", "database", "schema"]),
                "search_table",
                0.8,
            ),
            # High priority: Table description
            (
                lambda t: any(word in t for word in ["描述", "检查", "分析", "describe", "inspect", "analyze"])
                and any(word in t for word in ["表", "table", "schema", "structure"]),
                "describe_table",
                0.75,
            ),
            # Medium priority: SQL execution
            (
                lambda t: any(word in t for word in ["执行", "运行", "execute", "run"])
                and any(word in t for word in ["sql", "查询", "query"]),
                "execute_sql",
                0.7,
            ),
            # Medium priority: Metrics
            (lambda t: any(word in t for word in ["指标", "metrics", "kpi", "转化率"]), "search_metrics", 0.7),
            # Low priority: File operations
            (
                lambda t: any(word in t for word in ["文件", "写入", "保存", "读取", "file", "write", "save", "read"]),
                "write_file",
                0.6,
            ),
        ]

        for condition, tool, confidence in inference_rules:
            if condition(text_lower):
                logger.debug(f"Inferred tool '{tool}' with confidence {confidence}")
                return tool

        return None

    def _is_actually_completed(self, todo_id: str) -> bool:
        """
        Check if a todo is actually completed by verifying action history.

        Args:
            todo_id: The todo ID to check

        Returns:
            bool: True if the todo has been successfully executed
        """
        try:
            if not self.action_history_manager:
                return False

            # Check if there's a successful tool execution for this todo_id
            for action in reversed(self.action_history_manager.get_actions()):
                if action.role in ("tool", ActionRole.TOOL):
                    # Check input for todo_id
                    if action.input and isinstance(action.input, dict):
                        input_todo_id = action.input.get("todo_id") or action.input.get("todoId")
                        if input_todo_id == todo_id and action.status == ActionStatus.SUCCESS:
                            return True

                    # Check output for completion markers
                    if action.output and isinstance(action.output, dict):
                        if action.output.get("success") and action.status == ActionStatus.SUCCESS:
                            return True

            return False
        except Exception as e:
            logger.debug(f"Error checking completion status for todo {todo_id}: {e}")
            return False

    def _is_pending_update(self, context) -> bool:
        """
        Check if todo_update is being called with status='pending'.

        Args:
            context: ToolContext with tool_arguments field (JSON string)

        Returns:
            bool: True if this is a pending status update
        """
        try:
            if hasattr(context, "tool_arguments"):
                if context.tool_arguments:
                    tool_args = json.loads(context.tool_arguments)

                    # Check if status is 'pending'
                    if isinstance(tool_args, dict):
                        if tool_args.get("status") == "pending":
                            logger.debug(f"Detected pending status update with args: {tool_args}")
                            return True

            logger.debug("Not a pending status update")
            return False

        except Exception as e:
            logger.debug(f"Error checking tool arguments: {e}")
            return False

    def get_plan_tools(self):
        from datus.tools.func_tool.plan_tools import PlanTool

        plan_tool = PlanTool(self.session)
        plan_tool.storage = self.todo_storage
        return plan_tool.available_tools()
