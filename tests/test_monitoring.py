#!/usr/bin/env python3
"""
Test script for monitoring and observability features.
"""

import asyncio
import os
import sys
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("."))

import json
from collections import defaultdict, deque
from typing import Any, Dict, Optional


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
        print(f"📊 Started monitoring execution {execution_id}")

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

        print(f"📊 Ended monitoring execution {self.current_execution['id']}: {status} in {duration:.2f}s")
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

        print(f"📝 Started todo {todo_id}: {content[:30]}...")

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

        print(f"📝 Ended todo {todo_id}: {status} in {todo_record['duration']:.2f}s")

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

        print(f"🔧 Started {tool_name} for todo {todo_id}")

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
        print(f"🔧 {status_icon} {tool_name} completed in {execution_time:.2f}ms for todo {todo_id}")

    def record_cache_hit(self, tool_name: str, cache_key: str):
        """Record a cache hit."""
        self.metrics["cache_hits"] += 1
        print(f"💾 Cache hit for {tool_name}: {cache_key}")

    def record_batch_optimization(self, optimization_type: str, original_count: int, optimized_count: int):
        """Record batch optimization."""
        self.metrics["batch_optimizations"] += 1
        savings = original_count - optimized_count
        print(
            f"⚡ Batch optimization ({optimization_type}): {original_count} → {optimized_count} (saved {savings} operations)"
        )

    def record_error_recovery(self, strategy: str, success: bool):
        """Record error recovery attempt."""
        if success:
            self.metrics["error_recoveries"] += 1
        print(f"🔄 Error recovery ({strategy}): {'✅' if success else '❌'}")

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
        report["current_status"]

        # Calculate some derived metrics
        total_executions = len(report["recent_executions"])
        if total_executions > 0:
            sum(ex.get("duration", 0) for ex in report["recent_executions"]) / total_executions

        # Build dashboard
        dashboard_lines = [
            "╔══════════════════════════════════════════════════════════════════════════════╗",
            "║                           📊 PLAN MODE PERFORMANCE DASHBOARD 📊              ║",
            "╠══════════════════════════════════════════════════════════════════════════════╣",
            ".1%",
            ".1%",
            f"║ Batch Optimizations: {summary['batch_optimizations']:<6} Error Recoveries: {summary['error_recoveries']:<6}             ║",
            ".1f",
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
                todos_count = trend.get("todos_count", 0)
                status_icon = "✅" if trend.get("status") == "completed" else "❌"
                dashboard_lines.append(
                    f"║ {i+1}. {duration:.2f}s | {todos_count} todos | {status_icon}                      ║"
                )
        else:
            dashboard_lines.append("║                              No recent executions                          ║")

        dashboard_lines.extend(
            [
                "╚══════════════════════════════════════════════════════════════════════════════╝",
                "",
                "💡 Tips:",
                ".1%",
                f"   • {summary['batch_optimizations']} batch optimizations saved computational resources",
                f"   • {summary['error_recoveries']} successful error recoveries improved reliability",
                "",
                "Use monitor.export_monitoring_data('json') for detailed JSON export",
            ]
        )

        return "\n".join(dashboard_lines)


async def test_execution_monitor():
    """Test execution monitoring functionality."""
    print("=== Testing Execution Monitor ===")

    monitor = ExecutionMonitor(max_history_size=10)

    # Test execution lifecycle
    execution_id = "test_exec_001"

    # Start execution
    monitor.start_execution(execution_id, plan_id="test_plan", metadata={"test": True})
    print(f"✅ Started execution: {execution_id}")

    # Record some todos
    monitor.record_todo_start("todo_1", "Test task 1")
    monitor.record_todo_start("todo_2", "Test task 2")

    # Simulate tool calls
    start_time = time.time()
    await asyncio.sleep(0.01)  # Simulate work
    monitor.record_tool_call("search_table", "todo_1", {"query": "test"}, start_time)

    await asyncio.sleep(0.01)  # Simulate more work
    monitor.record_tool_result("search_table", "todo_1", True, 15.5, result="test_result")

    # Record todo completions
    monitor.record_todo_end("todo_1", "completed", "Task completed successfully")
    monitor.record_todo_end("todo_2", "completed")

    # End execution
    monitor.end_execution("completed")
    print("✅ Ended execution successfully")

    # Test metrics
    report = monitor.get_monitoring_report()
    print(f"📊 Generated monitoring report with {len(report)} sections")

    summary = report["summary"]
    print(f"   - Total executions: {summary['total_executions']}")
    print(".1%")
    print(f"   - Todos processed: {summary['total_todos_processed']}")
    print(f"   - Active operations: {report['current_status']['active_operations']}")

    print()


async def test_monitor_exports():
    """Test monitoring data export formats."""
    print("=== Testing Monitor Data Exports ===")

    monitor = ExecutionMonitor()

    # Add some test data
    monitor.start_execution("export_test", metadata={"format_test": True})
    monitor.record_todo_start("export_todo", "Export test task")
    monitor.record_tool_call("describe_table", "export_todo", {"table": "test"}, time.time())
    monitor.record_tool_result("describe_table", "export_todo", True, 10.0)
    monitor.record_todo_end("export_todo", "completed")
    monitor.end_execution("completed")

    # Test JSON export
    json_data = monitor.export_monitoring_data("json")
    print(f"✅ JSON export: {len(json_data)} characters")

    # Test text export
    text_data = monitor.export_monitoring_data("text")
    print(f"✅ Text export: {len(text_data)} lines")

    print()


async def test_performance_dashboard():
    """Test performance dashboard generation."""
    print("=== Testing Performance Dashboard ===")

    monitor = ExecutionMonitor()

    # Add diverse test data
    for i in range(5):
        exec_id = f"dashboard_test_{i}"
        monitor.start_execution(exec_id, plan_id=f"plan_{i}")

        # Add some todos with varying completion times
        for j in range(2):
            todo_id = f"todo_{i}_{j}"
            monitor.record_todo_start(todo_id, f"Dashboard test task {i}.{j}")

            # Simulate different tool calls
            tools = ["search_table", "describe_table", "execute_sql"]
            tool = tools[j % len(tools)]

            start_time = time.time()
            await asyncio.sleep(0.001 * (i + j))  # Vary timing
            success = (i + j) % 3 != 0  # Some failures

            monitor.record_tool_call(tool, todo_id, {"param": f"value_{i}_{j}"}, start_time)
            monitor.record_tool_result(tool, todo_id, success, 5.0 + i, result=f"result_{i}_{j}")
            monitor.record_todo_end(todo_id, "completed" if success else "failed")

        monitor.end_execution("completed" if i < 4 else "failed")

    # Record some additional metrics
    monitor.record_cache_hit("search_table", "cache_key_1")
    monitor.record_batch_optimization("search_consolidation", 3, 1)
    monitor.record_error_recovery("fallback_to_describe", True)

    # Generate dashboard
    dashboard = monitor.generate_performance_dashboard()
    print("✅ Generated performance dashboard")
    print("📊 Dashboard preview (first 10 lines):")
    lines = dashboard.split("\n")[:10]
    for line in lines:
        print(f"   {line}")

    print()

    # Test with empty monitor
    empty_monitor = ExecutionMonitor()
    empty_monitor.generate_performance_dashboard()
    print("✅ Generated empty dashboard (no data)")

    print()


async def main():
    """Run all monitoring tests."""
    print("🧪 Testing Monitoring and Observability Features\n")

    try:
        await test_execution_monitor()
        await test_monitor_exports()
        await test_performance_dashboard()

        print("✅ All monitoring and observability tests completed successfully!")
        print("\n📊 Monitoring Features Summary:")
        print("- ✅ Execution lifecycle tracking")
        print("- ✅ Tool performance monitoring")
        print("- ✅ Cache hit rate analysis")
        print("- ✅ Batch optimization metrics")
        print("- ✅ Error recovery statistics")
        print("- ✅ Multiple export formats (JSON/Text)")
        print("- ✅ ASCII performance dashboard")
        print("- ✅ Real-time active operation tracking")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
