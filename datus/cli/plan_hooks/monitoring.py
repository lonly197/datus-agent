# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Monitoring system for plan mode execution."""

import hashlib
import json
import re
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from .types import ErrorType

from datus.utils.loggings import get_logger


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
            # Preflight-specific metrics
            "preflight_calls": 0,
            "preflight_successes": 0,
            "preflight_failures": 0,
            "preflight_cache_hits": 0,
            "preflight_tool_calls": 0,
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

    def start_preflight(self, execution_id: str, tool_sequence: List[str]):
        """Start monitoring preflight tool execution."""
        if not self.current_execution or self.current_execution["id"] != execution_id:
            logger.warning(f"No active execution {execution_id} for preflight monitoring")
            return

        self.current_execution["preflight"] = {
            "start_time": time.time(),
            "tool_sequence": tool_sequence,
            "tool_results": {},
            "cache_hits": 0,
            "success_count": 0,
            "failure_count": 0,
        }
        self.metrics["preflight_calls"] += 1
        logger.info(f"📊 Started preflight monitoring for execution {execution_id} with {len(tool_sequence)} tools")

    def record_preflight_tool_call(
        self,
        execution_id: str,
        tool_name: str,
        success: bool,
        cache_hit: bool = False,
        execution_time: float = 0,
        error: str = None,
    ):
        """Record a preflight tool call result."""
        if not self.current_execution or self.current_execution["id"] != execution_id:
            return

        if "preflight" not in self.current_execution:
            return

        preflight = self.current_execution["preflight"]

        # Record tool result
        preflight["tool_results"][tool_name] = {
            "success": success,
            "cache_hit": cache_hit,
            "execution_time": execution_time,
            "error": error,
            "timestamp": time.time(),
        }

        # Update counters
        if success:
            preflight["success_count"] += 1
            self.metrics["preflight_successes"] += 1
        else:
            preflight["failure_count"] += 1
            self.metrics["preflight_failures"] += 1

        if cache_hit:
            preflight["cache_hits"] += 1
            self.metrics["preflight_cache_hits"] += 1

        # Enhanced metrics for specific tools
        self._record_enhanced_tool_metrics(tool_name, success, execution_time, cache_hit)

    def _record_enhanced_tool_metrics(self, tool_name: str, success: bool, execution_time: float, cache_hit: bool):
        """Record metrics specific to enhanced preflight tools."""
        # Initialize enhanced metrics if not exists
        if "enhanced_tools" not in self.metrics:
            self.metrics["enhanced_tools"] = {
                "analyze_query_plan": {"calls": 0, "successes": 0, "avg_time": 0, "cache_hits": 0},
                "check_table_conflicts": {"calls": 0, "successes": 0, "avg_time": 0, "cache_hits": 0},
                "validate_partitioning": {"calls": 0, "successes": 0, "avg_time": 0, "cache_hits": 0},
            }

        tool_metrics = self.metrics["enhanced_tools"].get(tool_name, {})
        if not tool_metrics:
            return

        # Update metrics
        tool_metrics["calls"] += 1
        if success:
            tool_metrics["successes"] += 1
        if cache_hit:
            tool_metrics["cache_hits"] += 1

        # Update average execution time
        if tool_metrics["calls"] == 1:
            tool_metrics["avg_time"] = execution_time
        else:
            # Running average
            prev_avg = tool_metrics["avg_time"]
            tool_metrics["avg_time"] = (prev_avg * (tool_metrics["calls"] - 1) + execution_time) / tool_metrics["calls"]

        # Log enhanced metrics periodically
        if tool_metrics["calls"] % 10 == 0:  # Log every 10 calls
            success_rate = (tool_metrics["successes"] / tool_metrics["calls"]) * 100
            cache_hit_rate = (tool_metrics["cache_hits"] / tool_metrics["calls"]) * 100
            logger.info(
                f"Enhanced tool '{tool_name}' metrics: "
                f"calls={tool_metrics['calls']}, "
                f"success_rate={success_rate:.1f}%, "
                f"cache_hit_rate={cache_hit_rate:.1f}%, "
                f"avg_time={tool_metrics['avg_time']:.2f}s"
            )

        self.metrics["preflight_tool_calls"] += 1

        # Update tool-specific metrics
        self.tool_metrics[tool_name]["calls"] += 1
        if success:
            self.tool_metrics[tool_name]["successes"] += 1
        else:
            self.tool_metrics[tool_name]["failures"] += 1
            if error:
                self.tool_metrics[tool_name]["errors"][error] += 1

        # Update execution time tracking
        if execution_time > 0:
            tool_metric = self.tool_metrics[tool_name]
            tool_metric["total_execution_time"] += execution_time
            tool_metric["avg_execution_time"] = tool_metric["total_execution_time"] / tool_metric["calls"]

        status = "SUCCESS" if success else "FAILED"
        cache_status = " (cache hit)" if cache_hit else ""
        logger.info(f"📊 Preflight tool {tool_name}: {status}{cache_status} in {execution_time:.3f}s")

    def end_preflight(self, execution_id: str) -> Dict[str, Any]:
        """End preflight monitoring and return summary."""
        if not self.current_execution or self.current_execution["id"] != execution_id:
            return {}

        if "preflight" not in self.current_execution:
            return {}

        preflight = self.current_execution["preflight"]
        end_time = time.time()
        duration = end_time - preflight["start_time"]

        summary = {
            "duration": duration,
            "total_tools": len(preflight["tool_sequence"]),
            "success_count": preflight["success_count"],
            "failure_count": preflight["failure_count"],
            "cache_hits": preflight["cache_hits"],
            "success_rate": (
                preflight["success_count"] / len(preflight["tool_sequence"]) if preflight["tool_sequence"] else 0
            ),
            "tool_results": preflight["tool_results"],
        }

        logger.info(
            f"📊 Preflight completed for {execution_id}: {preflight['success_count']}/{len(preflight['tool_sequence'])} tools successful in {duration:.2f}s"
        )
        return summary

    def get_preflight_status(self, execution_id: str) -> Dict[str, Any]:
        """Get current preflight status."""
        if not self.current_execution or self.current_execution["id"] != execution_id:
            return {"status": "no_active_execution"}

        if "preflight" not in self.current_execution:
            return {"status": "no_preflight_active"}

        preflight = self.current_execution["preflight"]
        return {
            "status": "active",
            "tools_completed": len(preflight["tool_results"]),
            "tools_total": len(preflight["tool_sequence"]),
            "success_count": preflight["success_count"],
            "failure_count": preflight["failure_count"],
            "cache_hits": preflight["cache_hits"],
        }

    def classify_failure_reason(self, tool_name: str, error: str) -> str:
        """Classify failure reason for better reporting.

        Args:
            tool_name: Name of tool that failed
            error: Error message

        Returns:
            Failure reason category: database_connection, timeout, permission,
            table_not_found, resource_not_found, syntax_error, or unknown
        """
        error_lower = error.lower()

        if "connection" in error_lower or "connect" in error_lower:
            return "database_connection"
        elif "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        elif "permission" in error_lower or "access denied" in error_lower:
            return "permission"
        elif "not found" in error_lower or "不存在" in error_lower:
            if tool_name in ["describe_table", "get_table_ddl", "check_table_exists"]:
                return "table_not_found"
            return "resource_not_found"
        elif "syntax" in error_lower or "parse" in error_lower:
            return "syntax_error"
        else:
            return "unknown"

    def _get_category_description(self, category: str) -> str:
        """Get Chinese description for failure category."""
        descriptions = {
            "database_connection": "数据库连接失败",
            "timeout": "查询超时",
            "permission": "权限不足",
            "table_not_found": "表不存在",
            "resource_not_found": "资源未找到",
            "syntax_error": "SQL语法错误",
            "unknown": "未知错误",
        }
        return descriptions.get(category, category)

    def generate_enhanced_fail_safe_annotation(self, execution_id: str) -> str:
        """Generate enhanced fail-safe report annotation with detailed failure info.

        Args:
            execution_id: Execution identifier

        Returns:
            Formatted annotation string with failure details and suggestions
        """
        if not self.current_execution or self.current_execution["id"] != execution_id:
            return ""

        if "preflight" not in self.current_execution:
            return ""

        preflight = self.current_execution["preflight"]
        tool_results = preflight.get("tool_results", {})

        if not tool_results:
            return ""

        # Classify failures
        failure_categories = {}
        critical_failures = []

        for tool_name, result in tool_results.items():
            if not result.get("success", True):
                error = result.get("error", "Unknown error")
                reason = self.classify_failure_reason(tool_name, error)

                if reason not in failure_categories:
                    failure_categories[reason] = []
                failure_categories[reason].append(tool_name)

                # Identify critical failures
                if reason in ["database_connection", "syntax_error"] or tool_name == "validate_sql_syntax":
                    critical_failures.append((tool_name, reason, error))

        # Generate annotation
        if not failure_categories:
            return ""

        parts = []

        # Critical failures section
        if critical_failures:
            parts.append("🚨 **关键失败 - 审查结果可靠性受严重影响**:")
            for tool, reason, error in critical_failures:
                parts.append(f"   - {tool} ({reason}): {error[:100]}...")

        # Failure details section
        if failure_categories:
            parts.append("\\n⚠️ **工具调用失败详情**:")
            for category, tools in failure_categories.items():
                desc = self._get_category_description(category)
                parts.append(f"   - {desc}: {', '.join(tools)}")

        # Suggestions section
        if "database_connection" in failure_categories:
            parts.append("\\n💡 **建议**: 检查数据库连接配置和网络连接")
        if "table_not_found" in failure_categories:
            parts.append("\\n💡 **建议**: 验证表名拼写和数据库权限")
        if "syntax_error" in failure_categories:
            parts.append("\\n💡 **建议**: 检查SQL语句语法，可能需要手动修正")
        if "permission" in failure_categories:
            parts.append("\\n💡 **建议**: 检查数据库用户权限配置")

        return "\\n".join(parts)

    def generate_fail_safe_report_annotation(self, execution_id: str) -> str:
        """Generate fail-safe report annotation for missing tool data."""
        if not self.current_execution or self.current_execution["id"] != execution_id:
            return ""

        if "preflight" not in self.current_execution:
            return ""

        preflight = self.current_execution["preflight"]
        failed_tools = [tool_name for tool_name, result in preflight["tool_results"].items() if not result["success"]]

        if not failed_tools:
            return ""

        annotation = f"""
⚠️ **数据完整性说明**: 以下工具调用失败，相关实证数据可能缺失:
{chr(10).join(f"- {tool}: {preflight['tool_results'][tool].get('error', '未知错误')}" for tool in failed_tools)}

**影响**: 审查报告基于LLM推理而非完整实证数据，结论准确性可能受影响。
**建议**: 检查数据库连接和外部知识库配置以恢复完整功能。
"""
        return annotation

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
        failed_todos = sum(1 for t in todos if t["status"] in ("failed", "error"))

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
                "success_rate": (
                    self.metrics["successful_executions"] / self.metrics["total_executions"]
                    if self.metrics["total_executions"] > 0
                    else 0
                ),
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
        logger.info(f"📈 Success Rate: {report['summary']['success_rate']:.1%}")
        logger.info(f"📝 Total Todos Processed: {report['summary']['total_todos_processed']}")
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
            success_rate = summary.get('success_rate', 0)
            if isinstance(success_rate, (int, float)):
                lines.append(f"Success Rate: {success_rate:.1%}")
            else:
                lines.append(f"Success Rate: {success_rate}")
            lines.append(f"Total Todos Processed: {summary['total_todos_processed']}")
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



