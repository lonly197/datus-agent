# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tool matching and LLM reasoning logic for plan mode hooks.

This module contains methods extracted from PlanModeHooks for matching
todo items to appropriate tools using:
- Exact keyword matching
- Context-aware matching
- LLM reasoning fallback
- Chinese semantic matching
- Intelligent inference
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.schemas.action_history import ActionHistory, ActionRole

logger = get_logger(__name__)

# DEFAULT keyword mapping for plan executor
DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP: Dict[str, List[str]] = {
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
    # ... (remaining mappings would be extracted from original file)
}


class ToolMatcher:
    """Advanced tool matching with multiple strategies."""

    def __init__(self, keyword_map: Dict[str, List[str]], model=None):
        """Initialize tool matcher.

        Args:
            keyword_map: Mapping of tool names to keyword phrases
            model: Optional LLM model for reasoning fallback
        """
        self.keyword_map = keyword_map
        self.model = model

    def match_tool(self, text: str) -> Optional[str]:
        """Match a todo item to the appropriate tool.

        This implements a multi-tier matching strategy:
        1. Task intent classification
        2. Context-aware keyword matching
        3. Chinese semantic matching
        4. LLM reasoning fallback
        5. Intelligent inference

        Args:
            text: Todo content to match

        Returns:
            Matched tool name or None
        """
        # Implementation would be extracted from PlanModeHooks._match_tool_for_todo
        # For now, this is a placeholder
        return None

    async def match_tool_async(self, text: str) -> Optional[str]:
        """Async version of tool matching.

        Args:
            text: Todo content to match

        Returns:
            Matched tool name or None
        """
        # Implementation would be extracted from PlanModeHooks._match_tool_for_todo_async
        return None

    def _match_exact_keywords(self, text: str) -> Optional[str]:
        """Exact keyword phrase matching with word boundaries."""
        # Implementation from original file
        return None

    async def _llm_reasoning_fallback(self, text: str) -> Optional[str]:
        """Use LLM to determine appropriate tool when keyword matching fails."""
        # Implementation from original file
        return None

    def _intelligent_inference(self, text: str) -> Optional[str]:
        """Pattern-based tool inference as last resort."""
        # Implementation from original file
        return None


# Note: The complete implementation would extract the following methods
# from PlanModeHooks and organize them here:
#
# - _match_tool_for_todo (lines ~2042-2093)
# - _match_tool_for_todo_async (lines ~2095-2122)
# - _match_exact_keywords (lines ~2124-2148)
# - _execute_llm_reasoning (lines ~2150-2270)
# - _llm_reasoning_fallback (lines ~2272-2315)
# - _llm_reasoning_fallback_async (lines ~2317-2366)
# - _intelligent_inference (lines ~2368-2432)
# - _enhanced_llm_reasoning (lines ~5600-5720)
# - _enhanced_intelligent_inference (lines ~5722-5850)
# - _preprocess_todo_content (lines ~5852-5870)
# - _classify_task_intent (lines ~5872-5930)
# - _match_keywords_with_context (lines ~5932-6050)
# - _semantic_chinese_matching (lines ~6052-6150)
# - _analyze_task_context (lines ~5670-5750)
#
# Plus DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP constant (lines ~1451-1671)
