# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Plan mode hooks implementation for intercepting agent execution flow."""

import asyncio
import time
from datetime import datetime
from typing import Optional

from agents import SQLiteSession
from agents.lifecycle import AgentHooks
from rich.console import Console
from datus.cli.blocking_input_manager import blocking_input_manager
from datus.cli.execution_state import execution_controller
from datus.utils.loggings import get_logger
import json
import uuid
from datus.schemas.action_history import ActionRole, ActionStatus, ActionHistory
from typing import Dict, List, Optional

logger = get_logger(__name__)


class PlanningPhaseException(Exception):
    """Exception raised when trying to execute tools during planning phase."""


class UserCancelledException(Exception):
    """Exception raised when user explicitly cancels execution"""


class PlanModeHooks(AgentHooks):
    """Plan Mode hooks for workflow management"""

    def __init__(self, console: Console, session: SQLiteSession, auto_mode: bool = False, action_history_manager=None, agent_config=None, emit_queue: Optional[asyncio.Queue] = None, model=None):
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
        # load keyword->tool mapping from agent_config if provided, otherwise use default
        self.keyword_map: Dict[str, List[str]] = {}
        # Comprehensive keyword mapping with exact phrase matching
        default_map: Dict[str, List[str]] = {
            # Database tools
            "search_table": [
                "search for table", "搜索表结构", "查找数据库表", "find table in database",
                "search table schema", "查找表信息", "find database tables", "搜索数据库表"
            ],
            "describe_table": [
                "describe table", "检查表结构", "inspect table schema", "查看表结构",
                "examine table structure", "分析表结构", "describe table structure",
                "检查表定义", "查看表模式", "analyze table structure", "分析表元数据"
            ],
            "execute_sql": [
                "execute sql query", "执行sql查询", "run sql statement", "执行sql语句",
                "run database query", "执行数据库查询", "execute the sql", "运行sql代码",
                "execute sql", "执行sql", "run the query", "执行查询"
            ],
            "read_query": [
                "run query", "执行查询", "execute database query", "运行数据库查询",
                "perform sql execution", "执行sql执行"
            ],

            # Metrics and reference SQL tools
            "search_metrics": [
                "search metrics", "查找指标", "find business metrics", "搜索业务指标",
                "look for kpis", "查找kpi", "search performance metrics", "查找绩效指标"
            ],
            "search_reference_sql": [
                "search sql examples", "查找参考sql", "find reference sql", "搜索sql模板",
                "look for sql patterns", "查找sql模式", "search similar sql", "查找相似sql"
            ],
            "list_domain_layers_tree": [
                "list domains", "查看领域层级", "show domain structure", "显示业务分类",
                "explore domain layers", "浏览领域层级", "view business taxonomy", "查看业务分类法"
            ],

            # Semantic model tools
            "check_semantic_model_exists": [
                "check semantic model", "检查语义模型", "verify semantic model", "验证语义模型",
                "semantic model exists", "语义模型是否存在", "find semantic model", "查找语义模型"
            ],
            "check_metric_exists": [
                "check metric exists", "检查指标是否存在", "verify metric availability", "验证指标可用性",
                "metric exists", "指标是否存在", "find existing metric", "查找现有指标"
            ],
            "generate_sql_summary_id": [
                "generate summary id", "生成摘要标识", "create sql summary", "创建sql摘要",
                "generate sql id", "生成sql标识", "create summary identifier", "创建摘要标识符"
            ],

            # Time parsing tools
            "parse_temporal_expressions": [
                "parse date expressions", "解析日期表达式", "parse temporal expressions", "解析时间表达式",
                "analyze date ranges", "分析日期范围", "parse time periods", "解析时间段"
            ],
            "get_current_date": [
                "get current date", "获取当前日期", "current date", "今天日期",
                "today's date", "今日日期", "get today date", "获取今天日期"
            ],

            # File system tools
            "write_file": [
                "write file", "写入文件", "save to file", "保存到文件",
                "create file", "创建文件", "write content to file", "将内容写入文件"
            ],
            "read_file": [
                "read file", "读取文件", "load file", "加载文件",
                "open file", "打开文件", "read file content", "读取文件内容"
            ],
            "list_directory": [
                "list directory", "列出目录", "show directory contents", "显示目录内容",
                "list files", "列出文件", "directory listing", "目录列表"
            ],

            # Reporting tools
            "report": [
                "generate final report", "生成最终报告", "create comprehensive report", "创建综合报告",
                "write final report", "编写最终报告", "produce report", "生成报告",
                "生成最终html", "生成最终", "生成html", "生成报告文档"
            ],

            # Plan management tools
            "todo_write": [
                "create plan", "创建计划", "write execution plan", "编写执行计划",
                "generate todo list", "生成任务列表", "create task plan", "创建任务计划"
            ],
            "todo_update": [
                "update task status", "更新任务状态", "mark task complete", "标记任务完成",
                "update todo status", "更新待办状态", "change task state", "更改任务状态"
            ],
            "todo_read": [
                "read plan", "查看计划", "show execution plan", "显示执行计划",
                "check plan status", "检查计划状态", "view todo list", "查看任务列表"
            ],
        }
        try:
            if agent_config and getattr(agent_config, "plan_executor_keyword_map", None):
                # agent_config may provide mapping as dict
                self.keyword_map = agent_config.plan_executor_keyword_map
            else:
                self.keyword_map = default_map
        except Exception:
            self.keyword_map = default_map
        # fallback behavior enabled by default unless agent_config disables it
        self.enable_fallback = True
        try:
            if agent_config and hasattr(agent_config, "plan_executor_enable_fallback"):
                self.enable_fallback = bool(agent_config.plan_executor_enable_fallback)
        except Exception:
            pass

    def _match_tool_for_todo(self, text: str) -> Optional[str]:
        """
        Hybrid tool matching with three-tier approach:
        1. Exact keyword phrase matching (fast and reliable)
        2. LLM reasoning fallback (intelligent but slower)
        3. Intelligent inference (last resort)
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
            tool = self._llm_reasoning_fallback(text)
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
        """Exact keyword phrase matching with word boundaries."""
        if not text:
            return None

        t = text.lower()
        # Use word boundaries to ensure exact phrase matching
        for tool_name, keywords in self.keyword_map.items():
            for keyword in keywords:
                if keyword and f" {keyword.lower()} " in f" {t} ":
                    return tool_name
                # Also check for phrase at start/end of text
                if keyword and (t.startswith(keyword.lower()) or t.endswith(keyword.lower())):
                    return tool_name

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

            if response and hasattr(response, 'content'):
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
            if hasattr(self.model, 'generate_async'):
                response = await self.model.generate_async(prompt, max_tokens=50, temperature=0.1)
            else:
                # Fallback to sync method in a thread
                import asyncio
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.model.generate(prompt, max_tokens=50, temperature=0.1)
                )

            if response and hasattr(response, 'content'):
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
        if ("search" in t or "find" in t or "lookup" in t or "查找" in t) and ("table" in t or "database" in t or "表" in t or "数据库" in t):
            return "search_table"
        elif ("describe" in t and "table" in t) or ("表结构" in t) or ("table schema" in t) or ("表模式" in t) or ("table metadata" in t):
            return "describe_table"
        elif ("execute" in t and "sql" in t) or ("run" in t and "query" in t) or ("执行" in t and ("sql" in t or "查询" in t)):
            return "execute_sql"

        # Metrics-related inference (require both metric and search intent)
        if ("search" in t or "find" in t or "lookup" in t or "查找" in t) and any(keyword in t for keyword in ["metric", "kpi", "指标", "转化率", "收入", "销售额", "performance", "绩效"]):
            return "search_metrics"

        # Time-related inference (require both time and parse/analyze intent)
        if ("parse" in t or "analyze" in t or "解析" in t or "分析" in t) and any(keyword in t for keyword in ["date", "time", "temporal", "日期", "时间", "period", "期间"]):
            return "parse_temporal_expressions"

        # File-related inference (more specific patterns)
        if ("write" in t or "save" in t or "create" in t or "写入" in t or "保存" in t) and ("file" in t or "文件" in t):
            return "write_file"
        elif ("read" in t or "load" in t or "读取" in t) and ("file" in t or "文件" in t):
            return "read_file"
        elif ("list" in t or "directory" in t or "列出" in t) and ("directory" in t or "文件夹" in t):
            return "list_directory"

        # Report-related inference (require specific report generation intent)
        if any(phrase in t for phrase in ["final report", "生成报告", "create report", "生成最终", "final summary", "最终报告", "generate report"]):
            return "report"

        return None
        # Executor task handle
        self._executor_task = None

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
                                if isinstance(parsed, dict) and (parsed.get("todo_id") == todo_id or parsed.get("todoId") == todo_id):
                                    if a.status == ActionStatus.SUCCESS:
                                        return True
                            except Exception:
                                pass
                    else:
                        # try parse string input
                        if isinstance(inp, str):
                            try:
                                parsed = json.loads(inp)
                                if isinstance(parsed, dict) and (parsed.get("todo_id") == todo_id or parsed.get("todoId") == todo_id):
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

    async def _run_server_executor(self):
        """
        Server-side plan executor: sequentially execute pending todos when LLM did not drive tool calls.
        This is a conservative scaffold: it marks todos in_progress -> executes minimal placeholder work ->
        marks completed and emits ActionHistory entries so the event_converter will stream proper events.
        """
        try:
            # Small delay to allow any in-flight LLM-driven tool calls to finish
            await asyncio.sleep(0.5)
            todo_list = self.todo_storage.get_todo_list()
            if not todo_list:
                logger.info("Server executor: no todo list found, exiting")
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

                    db_tool = db_function_tool_instance(self.agent_config, database_name=getattr(self.agent_config, "current_database", ""))
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

                # Execute mapped tools for the todo based on simple heuristics.
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
                            output={"raw_output": "This step does not require external tool execution", "emit_chat": True},
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

                # If a tool was matched, execute mapped action
                if not executed_any and matched_tool:
                    try:
                        logger.info(f"Server executor: matched tool '{matched_tool}' for todo {item.id}")
                        if matched_tool == "search_table" and db_tool:
                            res = db_tool.search_table(query_text=item.content, top_n=5)
                            result_payload = res.model_dump() if hasattr(res, "model_dump") else dict(res) if isinstance(res, dict) else {"result": res}
                            complete_action_db = ActionHistory(
                                action_id=f"{call_id}_db",
                                role=ActionRole.TOOL,
                                messages=f"Server executor: db.search_table for todo {item.id}",
                                action_type="search_table",
                                input={"function_name": "search_table", "arguments": json.dumps({"query_text": item.content, "top_n": 5, "todo_id": item.id})},
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
                        elif matched_tool == "execute_sql" and db_tool:
                            # attempt to find SQL in action history (existing logic)
                            sql_text = None
                            if self.action_history_manager:
                                for a in reversed(self.action_history_manager.get_actions()):
                                    if getattr(a, "role", "") == "assistant" or getattr(a, "role", "") == ActionRole.ASSISTANT:
                                        out = getattr(a, "output", None)
                                        if isinstance(out, dict) and out.get("sql"):
                                            sql_text = out.get("sql")
                                            break
                                        content_field = out.get("content") if isinstance(out, dict) else None
                                        if content_field and isinstance(content_field, str) and "```sql" in content_field:
                                            start = content_field.find("```sql")
                                            end = content_field.find("```", start + 6)
                                            if start != -1 and end != -1:
                                                sql_text = content_field[start + 6 : end].strip()
                                                break
                            if sql_text:
                                res = db_tool.read_query(sql=sql_text)
                                result_payload = res.model_dump() if hasattr(res, "model_dump") else dict(res) if isinstance(res, dict) else {"result": res}
                                complete_action_db = ActionHistory(
                                    action_id=f"{call_id}_exec",
                                    role=ActionRole.TOOL,
                                    messages=f"Server executor: db.read_query for todo {item.id}",
                                    action_type="read_query",
                                    input={"function_name": "read_query", "arguments": json.dumps({"sql": sql_text, "todo_id": item.id})},
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
                        elif matched_tool == "report" and fs_tool:
                            # generate report using filesystem tool
                            report_path = f"reports/{item.id}_report.html"
                            report_body = f"<html><body><h1>Report for {item.content}</h1><p>Generated by server executor.</p></body></html>"
                            res = fs_tool.write_file(path=report_path, content=report_body, file_type="report")
                            result_payload = res.model_dump() if hasattr(res, "model_dump") else dict(res) if isinstance(res, dict) else {"result": res}
                            complete_action_report = ActionHistory(
                                action_id=f"{call_id}_report",
                                role=ActionRole.TOOL,
                                messages=f"Server executor: report generation for todo {item.id}",
                                action_type="report",
                                input={"function_name": "report", "arguments": json.dumps({"path": report_path, "todo_id": item.id})},
                                output=result_payload,
                                status=ActionStatus.SUCCESS if getattr(res, "success", 1) else ActionStatus.FAILED,
                            )
                            if self.action_history_manager:
                                self.action_history_manager.add_action(complete_action_report)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(complete_action_report)
                                    except Exception as e:
                                        logger.debug(f"emit_queue put failed for complete_action_report: {e}")
                            executed_any = True
                    except Exception as e:
                        logger.error(f"Server executor matched_tool '{matched_tool}' failed for {item.id}: {e}")

                # SQL execution logic has been moved to keyword-based matching above

                # Report generation logic has been moved to keyword-based matching above

                # If nothing was mapped/executed, try selective fallback search_table if enabled and clearly database-related
                if not executed_any:
                    content_lower = (item.content or "").lower()
                    # Only fallback for todos that clearly involve database/schema work
                    db_related_keywords = ["table", "column", "schema", "database", "sql", "query", "数据", "表", "字段", "结构"]
                    has_db_keywords = any(keyword in content_lower for keyword in db_related_keywords)

                    if db_tool and self.enable_fallback and has_db_keywords:
                        try:
                            logger.info(f"Server executor: selective fallback search_table for todo {item.id} (database-related)")
                            res = db_tool.search_table(query_text=item.content, top_n=3)
                            result_payload = res.model_dump() if hasattr(res, "model_dump") else dict(res) if isinstance(res, dict) else {"result": res}
                            fallback_action = ActionHistory(
                                action_id=f"{call_id}_fallback",
                                role=ActionRole.TOOL,
                                messages=f"Server executor: fallback db.search_table for todo {item.id}",
                                action_type="search_table",
                                input={"function_name": "search_table", "arguments": json.dumps({"query_text": item.content, "top_n": 3, "todo_id": item.id})},
                                output=result_payload,
                                status=ActionStatus.SUCCESS if getattr(res, "success", 1) else ActionStatus.FAILED,
                            )
                            if self.action_history_manager:
                                self.action_history_manager.add_action(fallback_action)
                                if self.emit_queue is not None:
                                    try:
                                        self.emit_queue.put_nowait(fallback_action)
                                    except Exception:
                                        pass
                            executed_any = True
                        except Exception as e:
                            logger.debug(f"Server executor fallback search_table failed for {item.id}: {e}")
                    else:
                        # no fallback available or not database-related; emit short system note for transparency
                        try:
                            reason = "not database-related" if not has_db_keywords else "fallback disabled"
                            note_action = ActionHistory.create_action(
                                role=ActionRole.SYSTEM,
                                action_type="thinking",
                                messages=f"No tool executed for todo {item.id} ({reason}); marking completed",
                                input_data={"todo_id": item.id},
                                output={"raw_output": f"No tool matched ({reason}); step marked completed", "emit_chat": True},
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
                            logger.debug(f"Failed to emit system note for todo {item.id}: {e}")

                # Mark completed
                try:
                    plan_tool._update_todo_status(item.id, "completed")
                    await self._emit_plan_update_event(item.id, "completed")
                except Exception as e:
                    logger.error(f"Server executor failed to set completed for {item.id}: {e}")
                    try:
                        plan_tool._update_todo_status(item.id, "failed")
                        await self._emit_plan_update_event(item.id, "failed")
                    except Exception:
                        pass

            logger.info("Server executor finished all pending todos")
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
                    end_time=datetime.now()
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

    async def _emit_plan_update_event(self, todo_id: str = None, status: str = None):
        """
        Emit a plan_update event to notify frontend of plan status changes.

        Args:
            todo_id: Specific todo ID if updating a single item, None for full plan
            status: Status to set for the todo item
        """
        try:
            from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
            import uuid
            import time

            todo_list = self.todo_storage.get_todo_list()
            if not todo_list:
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
                start_time=datetime.now()
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

    def _is_pending_update(self, context) -> bool:
        """
        Check if todo_update is being called with status='pending'.

        Args:
            context: ToolContext with tool_arguments field (JSON string)

        Returns:
            bool: True if this is a pending status update
        """
        try:
            import json

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
