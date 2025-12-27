# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Plan mode hooks implementation for intercepting agent execution flow."""

import asyncio
import time

from agents import SQLiteSession
from agents.lifecycle import AgentHooks
from rich.console import Console
from datus.cli.blocking_input_manager import blocking_input_manager
from datus.cli.execution_state import execution_controller
from datus.utils.loggings import get_logger
import json
import uuid
from datus.schemas.action_history import ActionRole, ActionStatus, ActionHistory

logger = get_logger(__name__)


class PlanningPhaseException(Exception):
    """Exception raised when trying to execute tools during planning phase."""


class UserCancelledException(Exception):
    """Exception raised when user explicitly cancels execution"""


class PlanModeHooks(AgentHooks):
    """Plan Mode hooks for workflow management"""

    def __init__(self, console: Console, session: SQLiteSession, auto_mode: bool = False, action_history_manager=None, agent_config=None, emit_queue: Optional[asyncio.Queue] = None):
        self.console = console
        self.session = session
        self.auto_mode = auto_mode
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

                # Create a server-initiated tool call action (start)
                call_id = f"server_call_{uuid.uuid4().hex[:8]}"
                start_action = ActionHistory(
                    action_id=call_id,
                    role=ActionRole.TOOL,
                    messages=f"Server executor: starting todo {item.content}",
                    action_type="todo_update",
                    input={"function_name": "todo_update", "arguments": json.dumps({"todo_id": item.id, "status": "in_progress"})},
                    status=ActionStatus.PROCESSING,
                )
                # Add start action to history (will be converted to ToolCallEvent)
                if self.action_history_manager:
                    self.action_history_manager.add_action(start_action)
                    # also emit to node stream if emit_queue provided
                    if self.emit_queue is not None:
                        try:
                            self.emit_queue.put_nowait(start_action)
                        except Exception as e:
                            logger.debug(f"emit_queue put failed for start_action: {e}")

                # Mark in_progress using plan tool
                try:
                    res1 = plan_tool._update_todo_status(item.id, "in_progress")
                    # Build result action
                    result_payload = res1.model_dump() if hasattr(res1, "model_dump") else dict(res1) if isinstance(res1, dict) else {"result": res1}
                    complete_action = ActionHistory(
                        action_id=call_id,
                        role=ActionRole.TOOL,
                        messages=f"Server executor: todo_in_progress {item.id}",
                        action_type="todo_update",
                        input={"function_name": "todo_update", "arguments": json.dumps({"todo_id": item.id, "status": "in_progress"})},
                        output=result_payload,
                        status=ActionStatus.SUCCESS,
                    )
                    if self.action_history_manager:
                        self.action_history_manager.add_action(complete_action)
                        if self.emit_queue is not None:
                            try:
                                self.emit_queue.put_nowait(complete_action)
                            except Exception as e:
                                logger.debug(f"emit_queue put failed for complete_action: {e}")
                except Exception as e:
                    logger.error(f"Server executor failed to set in_progress for {item.id}: {e}")
                    fail_action = ActionHistory(
                        action_id=call_id,
                        role=ActionRole.TOOL,
                        messages=f"Server executor: todo_in_progress failed {item.id}: {e}",
                        action_type="todo_update",
                        input={"function_name": "todo_update", "arguments": json.dumps({"todo_id": item.id, "status": "in_progress"})},
                        output={"success": 0, "error": str(e)},
                        status=ActionStatus.FAILED,
                    )
                    if self.action_history_manager:
                        self.action_history_manager.add_action(fail_action)
                        if self.emit_queue is not None:
                            try:
                                self.emit_queue.put_nowait(fail_action)
                            except Exception as e:
                                logger.debug(f"emit_queue put failed for fail_action: {e}")
                    # mark todo failed in storage
                    try:
                        plan_tool._update_todo_status(item.id, "failed")
                    except Exception:
                        pass
                    continue

                # Execute mapped tools for the todo based on simple heuristics.
                content_lower = (item.content or "").lower()
                executed_any = False

                # 1) Search table semantics
                if db_tool and ("search" in content_lower or "搜索" in content_lower or "表结构" in content_lower):
                    try:
                        # call search_table with todo content as query_text
                        logger.info(f"Server executor: calling db_tool.search_table for todo {item.id}")
                        res = db_tool.search_table(query_text=item.content, top_n=5)
                        # create action representing the db tool result
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
                    except Exception as e:
                        logger.error(f"Server executor db_tool.search_table failed for {item.id}: {e}")

                # 2) Execute SQL if action history contains generated SQL and todo requests execution
                if db_tool and ("execute sql" in content_lower or "执行sql" in content_lower or ("执行" in content_lower and "sql" in content_lower)):
                    try:
                        # Find latest SQL in action history
                        sql_text = None
                        if self.action_history_manager:
                            for a in reversed(self.action_history_manager.get_actions()):
                                if getattr(a, "role", "") == "assistant" or getattr(a, "role", "") == ActionRole.ASSISTANT:
                                    out = getattr(a, "output", None)
                                    if isinstance(out, dict) and out.get("sql"):
                                        sql_text = out.get("sql")
                                        break
                                    # fallback: look into messages/content string for code block
                                    content_field = out.get("content") if isinstance(out, dict) else None
                                    if content_field and isinstance(content_field, str) and "```sql" in content_field:
                                        # crude extraction
                                        start = content_field.find("```sql")
                                        end = content_field.find("```", start + 6)
                                        if start != -1 and end != -1:
                                            sql_text = content_field[start + 6 : end].strip()
                                            break

                        if sql_text:
                            logger.info(f"Server executor: executing SQL for todo {item.id}")
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
                    except Exception as e:
                        logger.error(f"Server executor db_tool.read_query failed for {item.id}: {e}")

                # 3) Generate final report file if filesystem tool available
                if fs_tool and ("生成最终html" in content_lower or "生成最终" in content_lower or "生成html" in content_lower or "生成报告" in content_lower or "write" in content_lower):
                    try:
                        report_path = f"reports/{item.id}_report.html"
                        # try to extract assistant content as report body
                        report_body = f"<html><body><h1>Report for {item.content}</h1><p>Generated by server executor.</p></body></html>"
                        # attempt to use last assistant content if available
                        if self.action_history_manager:
                            for a in reversed(self.action_history_manager.get_actions()):
                                if getattr(a, "role", "") == "assistant" or getattr(a, "role", "") == ActionRole.ASSISTANT:
                                    out = getattr(a, "output", None)
                                    if isinstance(out, dict):
                                        body = out.get("response") or out.get("content")
                                        if body and isinstance(body, str):
                                            report_body = body
                                            break

                        res = fs_tool.write_file(path=report_path, content=report_body, file_type="report")
                        result_payload = res.model_dump() if hasattr(res, "model_dump") else dict(res) if isinstance(res, dict) else {"result": res}
                        complete_action_fs = ActionHistory(
                            action_id=f"{call_id}_write",
                            role=ActionRole.TOOL,
                            messages=f"Server executor: write_file for todo {item.id}",
                            action_type="write_file",
                            input={"function_name": "write_file", "arguments": json.dumps({"path": report_path, "todo_id": item.id})},
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
                    except Exception as e:
                        logger.error(f"Server executor fs_tool.write_file failed for {item.id}: {e}")

                # If nothing was mapped/executed, just sleep briefly (no-op) — still mark completed below
                if not executed_any:
                    await asyncio.sleep(0.1)

                # Mark completed
                try:
                    res2 = plan_tool._update_todo_status(item.id, "completed")
                    result_payload2 = res2.model_dump() if hasattr(res2, "model_dump") else dict(res2) if isinstance(res2, dict) else {"result": res2}
                    complete_action2 = ActionHistory(
                        action_id=f"{call_id}_done",
                        role=ActionRole.TOOL,
                        messages=f"Server executor: todo_completed {item.id}",
                        action_type="todo_update",
                        input={"function_name": "todo_update", "arguments": json.dumps({"todo_id": item.id, "status": "completed"})},
                        output=result_payload2,
                        status=ActionStatus.SUCCESS,
                    )
                    if self.action_history_manager:
                        self.action_history_manager.add_action(complete_action2)
                        if self.emit_queue is not None:
                            try:
                                self.emit_queue.put_nowait(complete_action2)
                            except Exception as e:
                                logger.debug(f"emit_queue put failed for complete_action2: {e}")
                except Exception as e:
                    logger.error(f"Server executor failed to set completed for {item.id}: {e}")
                    fail_action2 = ActionHistory(
                        action_id=f"{call_id}_done",
                        role=ActionRole.TOOL,
                        messages=f"Server executor: todo_complete failed {item.id}: {e}",
                        action_type="todo_update",
                        input={"function_name": "todo_update", "arguments": json.dumps({"todo_id": item.id, "status": "completed"})},
                        output={"success": 0, "error": str(e)},
                        status=ActionStatus.FAILED,
                    )
                    if self.action_history_manager:
                        self.action_history_manager.add_action(fail_action2)
                        if self.emit_queue is not None:
                            try:
                                self.emit_queue.put_nowait(fail_action2)
                            except Exception as e:
                                logger.debug(f"emit_queue put failed for fail_action2: {e}")

            logger.info("Server executor finished all pending todos")
        except Exception as e:
            logger.error(f"Unhandled server executor error: {e}")

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
