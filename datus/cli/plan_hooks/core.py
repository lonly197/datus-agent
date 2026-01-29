# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Core PlanModeHooks lifecycle management.

This module contains the core PlanModeHooks class with lifecycle methods:
- Initialization and cleanup
- Agent lifecycle hooks (on_start, on_end, on_tool_*, on_llm_*)
- State management
- Configuration loading
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agents import SQLiteSession
from agents.lifecycle import AgentHooks
from rich.console import Console

from datus.cli.blocking_input_manager import blocking_input_manager
from datus.cli.execution_state import execution_controller
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.schemas.action_history import ActionHistoryManager

logger = get_logger(__name__)


class PlanModeHooksCore(AgentHooks):
    """Core lifecycle management for plan mode hooks.

    This class handles the agent lifecycle events and state management.
    The execution logic is separated into other modules.
    """

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
        """Initialize PlanModeHooks.

        Args:
            console: Rich console for output
            session: SQLite session for todo storage
            auto_mode: If True, skip user confirmations
            action_history_manager: Manager for action history
            agent_config: Agent configuration
            emit_queue: Queue for emitting events
            model: LLM model for reasoning
            auto_injected_knowledge: List of auto-detected knowledge
        """
        self.console = console
        self.session = session
        self.auto_mode = auto_mode
        self.model = model
        self.auto_injected_knowledge = auto_injected_knowledge or []

        from datus.tools.func_tool.plan_tools import SessionTodoStorage

        self.todo_storage = SessionTodoStorage(session)
        self.plan_phase = "generating"
        self.execution_mode = "auto" if auto_mode else "manual"
        self.replan_feedback = ""
        self._state_transitions = []
        self._plan_generated_pending = False

        self.action_history_manager = action_history_manager
        self.agent_config = agent_config
        self.emit_queue = emit_queue
        self.execution_event_manager = None
        self.current_execution_id = None

        self._executor_task = None
        self._execution_complete = asyncio.Event()
        self._all_todos_completed = False

        # Load keyword map
        self.keyword_map: Dict[str, List[str]] = self._load_keyword_map(agent_config)
        self.enable_fallback = True

        # Initialize components
        from .error_handling import ErrorHandler
        from .monitoring import ExecutionMonitor
        from .optimization import QueryCache, ToolBatchProcessor
        from .router import SmartExecutionRouter

        self.error_handler = ErrorHandler()
        self.query_cache = QueryCache()
        self.batch_processor = ToolBatchProcessor()
        self.enable_batch_processing = True
        self.enable_query_caching = True
        self.execution_router = SmartExecutionRouter(
            agent_config=agent_config, model=model, action_history_manager=action_history_manager, emit_queue=emit_queue
        )
        self.monitor = ExecutionMonitor()

        try:
            if agent_config and hasattr(agent_config, "plan_executor_enable_fallback"):
                self.enable_fallback = bool(agent_config.plan_executor_enable_fallback)
        except Exception:
            pass

    def set_execution_event_manager(self, event_manager, execution_id: str):
        """Set the unified execution event manager.

        Args:
            event_manager: The event manager instance
            execution_id: Current execution ID
        """
        self.execution_event_manager = event_manager
        self.current_execution_id = execution_id

    async def cleanup(self):
        """Clean up resources and cancel pending tasks."""
        if self._executor_task and not self._executor_task.done():
            self._executor_task.cancel()
            try:
                await self._executor_task
            except asyncio.CancelledError:
                pass
            self._executor_task = None

        self._execution_complete.set()

    def _load_keyword_map(self, agent_config) -> Dict[str, List[str]]:
        """Load and merge keyword mapping from config and defaults.

        Args:
            agent_config: Agent configuration object

        Returns:
            Dict mapping tool names to lists of keyword phrases
        """
        from .matching import DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP

        merged_map = DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP.copy()

        try:
            if agent_config and getattr(agent_config, "plan_executor_keyword_map", None):
                custom_map = agent_config.plan_executor_keyword_map
                if isinstance(custom_map, dict):
                    for tool_name, keywords in custom_map.items():
                        if isinstance(keywords, list):
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

        for tool_name in merged_map:
            merged_map[tool_name] = [str(k).lower().strip() for k in merged_map[tool_name] if k and str(k).strip()]

        logger.debug(f"Loaded keyword map with {len(merged_map)} tools")
        return merged_map

    async def on_start(self, context, agent) -> None:
        """Called when agent starts execution.

        Args:
            context: Agent context
            agent: Agent instance
        """
        logger.debug(f"Plan mode start: phase={self.plan_phase}")

    async def on_tool_start(self, context, agent, tool) -> None:
        """Called before tool execution.

        Args:
            context: Agent context
            agent: Agent instance
            tool: Tool being executed
        """
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        logger.debug(f"Plan mode tool start: {tool_name}, phase: {self.plan_phase}, mode: {self.execution_mode}")

        # Delegate to execution module
        if tool_name == "todo_update" and self.execution_mode == "manual" and self.plan_phase == "executing":
            from .execution import handle_execution_step

            if self._is_pending_update(context):
                await handle_execution_step(self, tool_name)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Called after tool execution.

        Args:
            context: Agent context
            agent: Agent instance
            tool: Tool that was executed
            result: Tool execution result
        """
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        if tool_name == "todo_write":
            logger.info("Plan generation completed, will show plan after LLM finishes current turn")
            self._plan_generated_pending = True

    async def on_llm_end(self, context, agent, response) -> None:
        """Called when LLM finishes a turn.

        Args:
            context: Agent context
            agent: Agent instance
            response: LLM response
        """
        if self._plan_generated_pending and self.plan_phase == "generating":
            self._plan_generated_pending = False
            from .execution import on_plan_generated

            await on_plan_generated(self)

    async def on_end(self, context, agent, output) -> None:
        """Called when agent finishes execution.

        Args:
            context: Agent context
            agent: Agent instance
            output: Agent output
        """
        logger.info(f"Plan mode end: phase={self.plan_phase}")

    def _transition_state(self, new_state: str, context: dict = None):
        """Transition to a new state.

        Args:
            new_state: New state name
            context: Optional context about the transition

        Returns:
            Dict with transition data
        """
        import time

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

        Returns:
            True if execution is complete
        """
        return self._execution_complete.is_set() and self._all_todos_completed

    def _is_pending_update(self, context) -> bool:
        """Check if todo_update is setting status to pending.

        Args:
            context: Tool execution context

        Returns:
            True if this is a pending update
        """
        # Implementation from original file
        return False


# TODO: Complete PlanModeHooksCore implementation
# Pending migration from PlanModeHooks:
# - Lifecycle hook methods
# - State management methods
# - Configuration loading
