# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Virtual steps management for non-agentic workflows (Text2SQL).

This module defines the virtual steps that represent high-level stages
in the Text2SQL workflow and manages their state.
"""

from typing import List, Optional

from .models import PlanUpdateEvent, TodoItem, TodoStatus
from .normalization import normalize_node_type


# Define virtual steps for non-agentic workflows (Text2SQL)
VIRTUAL_STEPS = [
    {
        "id": "step_intent",
        "content": "分析查询意图",
        "node_types": ["intent_analysis", "intent_clarification"],
    },
    {
        "id": "step_schema",
        "content": "发现数据库模式",
        "node_types": [
            "schema_discovery",
            "schema_validation",
            # Preflight tools for schema discovery
            "preflight_search_table",
            "preflight_describe_table",
            "preflight_get_table_ddl",
            "preflight_search_reference_sql",
            "preflight_parse_temporal_expressions",
            "preflight_check_table_exists",
        ],
    },
    {"id": "step_sql", "content": "生成SQL查询", "node_types": ["generate_sql", "sql_generation"]},
    {
        "id": "step_exec",
        "content": "执行SQL并验证结果",
        "node_types": [
            "execute_sql",
            "sql_execution",
            "sql_validate",
            "result_validation",
            # Preflight tools for SQL execution and validation
            "preflight_validate_sql_syntax",
            "preflight_analyze_query_plan",
            "preflight_check_table_conflicts",
            "preflight_validate_partitioning",
        ],
    },
    {"id": "step_reflect", "content": "自我纠正与优化", "node_types": ["reflect", "reflection_analysis"]},
    {"id": "step_output", "content": "生成结果报告", "node_types": ["output", "output_generation"]},
]


class VirtualStepManager:
    """Manages virtual plan state for non-agentic workflows."""

    def __init__(self, virtual_plan_id: str):
        """Initialize virtual step manager.

        Args:
            virtual_plan_id: Fixed plan ID for all PlanUpdateEvents
        """
        self.virtual_plan_id = virtual_plan_id
        self.virtual_plan_emitted = False
        self.active_virtual_step_id: Optional[str] = None
        self.completed_virtual_steps: set[str] = set()
        # Track failed virtual steps for proper ERROR status in PlanUpdateEvent
        self.failed_virtual_steps: set[str] = set()

    def get_virtual_step_id(self, node_type: str) -> Optional[str]:
        """Map node type to virtual step ID.

        Args:
            node_type: The raw node type from action

        Returns:
            Virtual step ID if found, None otherwise
        """
        normalized_type = normalize_node_type(node_type)
        for step in VIRTUAL_STEPS:
            if normalized_type in step["node_types"]:
                return str(step["id"])
        return None

    def generate_virtual_plan_update(self, current_node_type: Optional[str] = None) -> Optional[PlanUpdateEvent]:
        """Generate PlanUpdateEvent based on current progress.

        Status priority: ERROR > COMPLETED > IN_PROGRESS > PENDING
        This ensures failed steps are never incorrectly marked as COMPLETED.

        Args:
            current_node_type: Current node type being executed

        Returns:
            PlanUpdateEvent if virtual plan should be emitted, None otherwise
        """
        import time

        current_step_id = self.get_virtual_step_id(current_node_type) if current_node_type else None
        if current_step_id:
            self.active_virtual_step_id = current_step_id

        # Find active step index
        active_index = -1
        if self.active_virtual_step_id:
            active_index = next(
                (i for i, step in enumerate(VIRTUAL_STEPS)
                 if str(step["id"]) == self.active_virtual_step_id),
                -1
            )

        # Build todos with status priority
        todos = []
        for i, step in enumerate(VIRTUAL_STEPS):
            step_id = str(step["id"])

            if step_id in self.failed_virtual_steps:
                status = TodoStatus.ERROR
            elif step_id in self.completed_virtual_steps:
                status = TodoStatus.COMPLETED
            elif active_index != -1:
                status = (
                    TodoStatus.COMPLETED if i < active_index else
                    TodoStatus.IN_PROGRESS if i == active_index else
                    TodoStatus.PENDING
                )
            else:
                status = TodoStatus.PENDING

            todos.append(TodoItem(
                id=step_id,
                content=str(step["content"]),
                status=status
            ))

        return PlanUpdateEvent(
            id=self.virtual_plan_id,
            planId=None,
            timestamp=int(time.time() * 1000),
            todos=todos
        )


class TodoStateManager:
    """Manages TodoItem state for agentic workflows."""

    def __init__(self):
        """Initialize todo state manager."""
        # Cached plan todos for emitting full PlanUpdateEvent payloads
        self._todo_state: dict = {}
        self._todo_order: List[str] = []

    def update_todo_state(self, todos: List[TodoItem], replace_order: bool = False) -> None:
        """Update cached todo state for full PlanUpdateEvent emission.

        Args:
            todos: List of TodoItem objects to cache
            replace_order: If True, replace todo order with new list
        """
        if replace_order:
            self._todo_order = [todo.id for todo in todos]
        for todo in todos:
            if todo.id not in self._todo_order:
                self._todo_order.append(todo.id)
            self._todo_state[todo.id] = todo

    def get_todo_state_list(self) -> List[TodoItem]:
        """Return todos in cached order when available.

        Returns:
            List of TodoItem objects in cached order
        """
        if not self._todo_state:
            return []
        if not self._todo_order:
            return list(self._todo_state.values())
        return [self._todo_state[todo_id] for todo_id in self._todo_order if todo_id in self._todo_state]
