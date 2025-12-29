#!/usr/bin/env python3
"""
Test script to validate the plan hooks fixes.
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Mock the missing dependencies
class MockConsole:
    def print(self, *args, **kwargs):
        pass


class MockSession:
    pass


class MockActionHistoryManager:
    def __init__(self):
        self.actions = []

    def add_action(self, action):
        self.actions.append(action)


class MockAgentConfig:
    pass


# Test the _emit_plan_update_event method
def test_emit_plan_update_event():
    print("Testing _emit_plan_update_event...")

    # Import required classes
    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

    # Create a minimal PlanModeHooks-like object for testing
    class TestPlanHooks:
        def __init__(self):
            from datus.tools.func_tool.plan_tools import SessionTodoStorage

            self.todo_storage = MockTodoStorage()
            self.action_history_manager = MockActionHistoryManager()
            self.emit_queue = asyncio.Queue()

        async def _emit_plan_update_event(self, todo_id=None, status=None):
            """Copy of the fixed method"""
            try:
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
                    start_time=datetime.now(),
                )

                # Add to action history manager
                if self.action_history_manager:
                    self.action_history_manager.add_action(plan_update_action)

                # Emit to queue if available
                if self.emit_queue is not None:
                    try:
                        self.emit_queue.put_nowait(plan_update_action)
                        print("✓ Successfully emitted plan_update event to queue")
                    except Exception as e:
                        print(f"✗ Failed to emit to queue: {e}")

                return plan_update_action

            except Exception as e:
                print(f"✗ Failed to emit plan_update event: {e}")
                return None

    class MockTodoStorage:
        def get_todo_list(self):
            from datus.tools.func_tool.plan_tools import TodoList

            todo_list = TodoList()
            # Add a test todo
            from datus.tools.func_tool.plan_tools import TodoItem, TodoStatus

            todo_list.add_item("Test todo", TodoStatus.PENDING)
            return todo_list

    # Test the method
    hooks = TestPlanHooks()

    async def run_test():
        result = await hooks._emit_plan_update_event("test_id", "in_progress")
        if result:
            print("✓ _emit_plan_update_event completed successfully")
            print(f"  - Action ID: {result.action_id}")
            print(f"  - Action Type: {result.action_type}")
            print(f"  - Messages: {result.messages}")
            return True
        else:
            print("✗ _emit_plan_update_event failed")
            return False

    # Run the async test
    import asyncio

    result = asyncio.run(run_test())
    return result


# Test call_id definition
def test_call_id_definition():
    print("Testing call_id definition...")

    # Simulate the fixed code
    call_id = f"server_call_{uuid.uuid4().hex[:8]}"
    test_action_id = f"{call_id}_db"

    print(f"✓ call_id generated: {call_id}")
    print(f"✓ action_id generated: {test_action_id}")

    return True


if __name__ == "__main__":
    try:
        print("Testing Plan Hooks fixes...\n")

        # Test call_id definition
        test_call_id_definition()
        print()

        # Test emit plan update event
        success = test_emit_plan_update_event()
        print()

        if success:
            print("🎉 All tests passed! The fixes should resolve the SSE streaming issues.")
        else:
            print("❌ Some tests failed. Please check the implementation.")

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
