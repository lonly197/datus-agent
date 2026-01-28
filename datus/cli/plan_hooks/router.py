# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Routing and task classification for plan mode execution."""

from typing import Any, Dict

from .types import TaskType


class SmartExecutionRouter:
    """Intelligent task execution router based on task type classification."""

    def __init__(self, agent_config=None, model=None, action_history_manager=None, emit_queue=None):
        """Initialize the execution router.

        Args:
            agent_config: Agent configuration object
            model: LLM model for reasoning tasks
            action_history_manager: Manager for action history
            emit_queue: Queue for emitting events
        """
        self.agent_config = agent_config
        self.model = model
        self.action_history_manager = action_history_manager
        self.emit_queue = emit_queue

    async def execute_task(self, todo_item, context) -> Dict[str, Any]:
        """Smart routing of task execution based on task type.

        Args:
            todo_item: The todo item to execute
            context: Execution context

        Returns:
            Dict with execution results and action instructions
        """
        task_type = getattr(todo_item, "task_type", "hybrid")

        if task_type == TaskType.LLM_ANALYSIS:
            return await self._execute_llm_analysis(todo_item, context)
        elif task_type == TaskType.TOOL_EXECUTION:
            return await self._execute_tool_call(todo_item, context)
        else:  # hybrid or unknown
            return await self._execute_hybrid_task(todo_item, context)

    async def _execute_llm_analysis(self, todo_item, context) -> Dict[str, Any]:
        """Return instruction to execute LLM analysis task.

        Args:
            todo_item: Todo item requiring LLM analysis
            context: Execution context

        Returns:
            Dict with execution instructions
        """
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
            from datus.utils.loggings import get_logger

            logger = get_logger(__name__)
            logger.error(f"LLM analysis setup failed for todo {todo_item.id}: {e}")
            return {"success": False, "error": str(e), "execution_type": "llm_analysis"}

    async def _execute_tool_call(self, todo_item, context) -> Dict[str, Any]:
        """Return instruction to execute tool-based task.

        Args:
            todo_item: Todo item requiring tool execution
            context: Execution context

        Returns:
            Dict with execution instructions
        """
        return {
            "success": True,
            "execution_type": "tool_execution",
            "action": "execute_tool",  # 返回执行指令
            "todo_item": todo_item,
        }

    async def _execute_hybrid_task(self, todo_item, context) -> Dict[str, Any]:
        """Return instruction for hybrid task execution.

        Args:
            todo_item: Todo item that may need both tools and analysis
            context: Execution context

        Returns:
            Dict with execution instructions
        """
        content_lower = todo_item.content.lower()

        # Check if this looks more like a tool task
        tool_indicators = ["执行", "运行", "查询", "搜索", "查找", "execute", "run", "query", "search"]
        if any(indicator in content_lower for indicator in tool_indicators):
            # Prefer tool execution for hybrid tasks that look tool-oriented
            return await self._execute_tool_call(todo_item, context)
        else:
            # Prefer LLM analysis for hybrid tasks that look analysis-oriented
            return await self._execute_llm_analysis(todo_item, context)


class TaskTypeClassifier:
    """智能任务类型分类器"""

    TOOL_EXECUTION = "tool_execution"  # 需要调用外部工具
    LLM_ANALYSIS = "llm_analysis"  # 需要LLM推理分析
    HYBRID = "hybrid"  # 可能需要工具+分析

    @classmethod
    def classify_task(cls, task_content: str) -> str:
        """智能分类任务类型

        Args:
            task_content: 任务内容描述

        Returns:
            任务类型: TOOL_EXECUTION, LLM_ANALYSIS, 或 HYBRID
        """
        if not task_content:
            return cls.HYBRID

        content_lower = task_content.lower()

        # === NEW: SQL review tool-first patterns ===
        # These patterns indicate database tool usage even with analysis keywords
        # Check these BEFORE general analysis keywords to prioritize tool execution
        sql_review_tool_patterns = [
            # Table structure requires describe_table
            "检查表结构",
            "表结构",
            "列信息",
            "分区",
            "分区键",
            "describe table",
            "表字段",
            "表定义",
            "表schema",
            # Execution plan requires execute_sql
            "执行计划",
            "explain",
            "查询计划",
            "执行路径",
            # Verification requires execute_sql
            "验证业务逻辑",
            "数据一致性",
            "运行sql",
            "执行sql",
            "运行查询",
            "执行查询",
        ]

        # Check for explicit tool patterns first
        for pattern in sql_review_tool_patterns:
            if pattern in content_lower:
                return cls.HYBRID  # Hybrid: analysis + tool execution

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

        analysis_score = sum(1 for kw in analysis_keywords if kw in content_lower)
        tool_score = sum(1 for kw in tool_keywords if kw in content_lower)

        # 特殊规则：如果明确提到"执行SQL"、"运行查询"等，优先归类为工具执行
        if any(phrase in content_lower for phrase in ["执行sql", "运行sql", "execute sql", "run sql"]):
            return cls.TOOL_EXECUTION

        # 特殊规则：如果明确提到"分析"、"检查"、"评估"等，优先归类为LLM分析
        # NOTE: This check comes AFTER sql_review_tool_patterns, so specific database
        # operation patterns will be caught first and classified as HYBRID
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
        """根据任务类型获取相应的上下文信息

        Args:
            task_content: 任务内容描述
            task_type: 任务类型

        Returns:
            包含任务上下文信息的字典
        """
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
