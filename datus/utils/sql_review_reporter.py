# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""
SQL Review Report Generator

Generates comprehensive review reports that combine:
- Empirical findings from preflight tools
- SQL validation results
- LLM-based analysis
- Clear confidence scoring
"""

from typing import Any, Dict, List
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SQLReviewReporter:
    """Generate comprehensive SQL review reports."""

    def generate_review_report(
        self,
        sql: str,
        preflight_results: Dict[str, Any],
        llm_analysis: Dict[str, Any],
        validation_results: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive review report.

        Args:
            sql: The SQL query being reviewed
            preflight_results: Results from preflight tool execution
            llm_analysis: Analysis from LLM
            validation_results: SQL validation results

        Returns:
            Structured report with sections:
            - overview: High-level summary
            - empirical_findings: Results from tools (high confidence)
            - inferences: LLM-based analysis (medium confidence)
            - sql_validation: Syntax and structure validation
            - recommendations: Prioritized action items
        """
        report = {
            "sql_query": sql,
            "overview": self._generate_overview(preflight_results, validation_results),
            "tool_execution_summary": self._summarize_tool_execution(preflight_results),
            "empirical_findings": self._extract_empirical_findings(preflight_results),
            "inferences": llm_analysis,
            "sql_validation": validation_results or {},
            "recommendations": self._generate_recommendations(preflight_results, validation_results, llm_analysis),
            "confidence_scores": self._calculate_confidence_scores(preflight_results),
        }

        return report

    def _generate_overview(
        self, preflight_results: Dict[str, Any], validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate high-level overview."""
        # Extract tool execution statistics
        executed = preflight_results.get("tools_executed", [])
        failed = preflight_results.get("tools_failed", [])

        total_tools = len(executed) + len(failed)
        successful_tools = sum(1 for t in executed if t.get("success"))
        failed_tools = len(failed)

        # Check SQL validation
        is_valid_sql = True
        critical_issues = 0

        if validation_results:
            is_valid_sql = validation_results.get("is_valid", True)
            critical_issues = sum(
                1 for sug in validation_results.get("fix_suggestions", [])
                if sug.get("severity") == "critical"
            )

        # Determine overall status
        if not is_valid_sql or critical_issues > 0:
            overall_status = "fail"
        elif failed_tools > 0:
            overall_status = "warning"
        else:
            overall_status = "pass"

        # Check for critical tool failures
        critical_failure = False
        for tool_failure in failed:
            if tool_failure.get("is_critical"):
                critical_failure = True
                break

        if critical_failure:
            overall_status = "fail"

        return {
            "overall_status": overall_status,
            "sql_valid": is_valid_sql,
            "tools_successful": f"{successful_tools}/{total_tools}",
            "tools_failed": failed_tools,
            "critical_issues": critical_issues,
            "critical_tool_failure": critical_failure,
            "summary": self._generate_summary_text(overall_status, is_valid_sql, critical_issues, failed_tools, critical_failure),
        }

    def _generate_summary_text(
        self, status: str, is_valid: bool, critical: int, failed: int, critical_failure: bool
    ) -> str:
        """Generate human-readable summary."""
        if status == "fail":
            if critical_failure:
                return f"❌ SQL审查未通过：关键工具失败，无法完成完整审查"
            elif not is_valid_sql:
                return f"❌ SQL审查未通过：发现{critical}个严重语法错误"
            else:
                return f"❌ SQL审查未通过：发现{critical}个严重问题"
        elif status == "warning":
            return f"⚠️ SQL审查通过警告：{failed}个分析工具失败，建议人工复核"
        else:
            return "✅ SQL审查通过：未发现严重问题"

    def _summarize_tool_execution(self, preflight_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize which tools succeeded/failed."""
        executed = preflight_results.get("tools_executed", [])
        failed = preflight_results.get("tools_failed", [])

        return {
            "successful": [t["tool_name"] for t in executed if t.get("success")],
            "failed": [t["tool_name"] for t in failed],
            "failed_with_errors": [
                {"tool": t["tool_name"], "error": t.get("error", "Unknown")}
                for t in failed
            ],
            "execution_time_total": preflight_results.get("total_execution_time", 0),
            "cache_hits": preflight_results.get("cache_hits", 0),
            "sql_validation_passed": preflight_results.get("sql_validation_passed", False),
        }

    def _extract_empirical_findings(self, preflight_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract empirical findings from tool results.

        These are high-confidence findings based on actual tool execution,
        not LLM inference.
        """
        findings = {
            "query_plan_analysis": None,
            "table_conflicts": None,
            "partitioning_validation": None,
            "table_structure": None,
        }

        # Extract from context if available
        # Note: These would be populated from actual preflight_results during execution

        return findings

    def _generate_recommendations(
        self,
        preflight_results: Dict[str, Any],
        validation_results: Dict[str, Any],
        llm_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations."""
        recommendations = []

        # Priority 1: SQL syntax fixes (critical)
        if validation_results:
            for fix in validation_results.get("fix_suggestions", []):
                severity = fix.get("severity", "info")
                priority = "critical" if severity == "critical" else "high" if severity == "error" else "medium"

                recommendations.append({
                    "priority": priority,
                    "category": "syntax_error",
                    "issue": fix["description"],
                    "action": fix.get("suggestion", ""),
                    "example": fix.get("example", ""),
                    "source": "sql_validation"
                })

        # Priority 2: Performance hotspots from query plan
        # (This would be extracted from preflight_results when available)

        # Priority 3: LLM suggestions
        if llm_analysis:
            # Extract LLM recommendations
            pass

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 5))

        return recommendations

    def _calculate_confidence_scores(self, preflight_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate confidence scores for different report sections.

        Returns:
            Dict mapping section names to confidence scores (0.0 to 1.0)
        """
        scores = {
            "sql_validation": 1.0,  # Direct parsing, high confidence
            "tool_execution": 0.9,  # Actual tool results, high confidence
            "performance_analysis": 0.7,  # Based on query plan, medium-high
            "business_logic": 0.5,  # LLM inference, medium confidence
        }

        # Adjust scores based on tool failures
        failed = preflight_results.get("tools_failed", [])
        if failed:
            # Reduce tool execution confidence based on failure rate
            total_tools = len(preflight_results.get("tools_executed", [])) + len(failed)
            failure_rate = len(failed) / total_tools if total_tools > 0 else 0
            scores["tool_execution"] *= (1 - failure_rate * 0.2)
            scores["performance_analysis"] *= 0.8  # Reduce if tools failed

        # Check for critical failures
        critical_failure = any(t.get("is_critical") for t in failed)
        if critical_failure:
            scores["tool_execution"] *= 0.5
            scores["performance_analysis"] *= 0.6

        # Round to 2 decimal places
        return {k: round(v, 2) for k, v in scores.items()}

    def format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format the report as markdown."""
        lines = []

        # Title
        lines.append("# 📋 SQL审查报告\n")

        # Overview
        overview = report["overview"]
        lines.append(f"## 审查概览")
        lines.append(f"**状态**: {overview['summary']}\n")
        lines.append(f"- **SQL有效性**: {'✅ 通过' if overview['sql_valid'] else '❌ 未通过'}")
        lines.append(f"- **工具执行**: {overview['tools_successful']}")
        lines.append(f"- **失败工具**: {overview['tools_failed']}")
        lines.append(f"- **严重问题**: {overview['critical_issues']}\n")

        # SQL Validation
        if report.get("sql_validation"):
            validation = report["sql_validation"]
            lines.append("## 🔍 SQL语法验证")

            if validation.get("errors"):
                lines.append(f"**解析错误**:")
                for error in validation["errors"]:
                    lines.append(f"- ❌ {error}")
                lines.append("")

            if validation.get("warnings"):
                lines.append(f"**警告**:")
                for warning in validation["warnings"][:10]:  # Limit to first 10
                    lines.append(f"- ⚠️ {warning}")
                lines.append("")

            if validation.get("fix_suggestions"):
                lines.append(f"**修复建议** ({len(validation['fix_suggestions'])}条):")
                for fix in validation["fix_suggestions"][:10]:  # Limit to first 10
                    severity_emoji = {
                        "critical": "🔴",
                        "error": "🟠",
                        "warning": "🟡",
                        "info": "ℹ️"
                    }.get(fix.get("severity", "info"), "ℹ️")

                    lines.append(f"- {severity_emoji} **{fix['description']}**")
                    if fix.get("suggestion"):
                        lines.append(f"  - 建议: {fix['suggestion']}")
                lines.append("")

        # Tool Execution Summary
        tool_summary = report["tool_execution_summary"]
        lines.append("## 🔧 工具执行摘要")
        lines.append(f"- **成功**: {', '.join(tool_summary['successful']) if tool_summary['successful'] else '无'}")
        lines.append(f"- **失败**: {', '.join(tool_summary['failed']) if tool_summary['failed'] else '无'}")
        lines.append(f"- **总耗时**: {tool_summary['execution_time_total']:.2f}秒")
        lines.append(f"- **缓存命中**: {tool_summary['cache_hits']}次\n")

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            lines.append("## 💡 改进建议")

            # Group by priority
            by_priority = {}
            for rec in recommendations:
                priority = rec["priority"]
                if priority not in by_priority:
                    by_priority[priority] = []
                by_priority[priority].append(rec)

            priority_order = ["critical", "high", "medium", "low", "info"]
            priority_emoji = {
                "critical": "🔴 关键",
                "high": "🟠 重要",
                "medium": "🟡 中等",
                "low": "🔵 较低",
                "info": "ℹ️ 提示"
            }

            for priority in priority_order:
                if priority in by_priority:
                    lines.append(f"\n### {priority_emoji[priority]}优先级")
                    for rec in by_priority[priority][:5]:  # Limit to 5 per priority
                        lines.append(f"- **{rec['issue']}**")
                        if rec.get("action"):
                            lines.append(f"  - 行动: {rec['action']}")
                        if rec.get("example"):
                            lines.append(f"  - 示例: {rec['example']}")
            lines.append("")

        # Confidence Scores
        scores = report.get("confidence_scores", {})
        if scores:
            lines.append("## 📊 置信度评分")
            for section, score in scores.items():
                percentage = score * 100
                bar_length = int(percentage / 10)
                bar = "█" * bar_length + "░" * (10 - bar_length)
                section_name = {
                    "sql_validation": "SQL验证",
                    "tool_execution": "工具执行",
                    "performance_analysis": "性能分析",
                    "business_logic": "业务逻辑"
                }.get(section, section)

                lines.append(f"- **{section_name}**: {bar} {percentage:.0f}%")
            lines.append("")

        return "\n".join(lines)
