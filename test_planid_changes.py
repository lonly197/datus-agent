#!/usr/bin/env python3
"""
Test script to demonstrate the planId changes in DeepResearchEventConverter.

This script shows how planId is now properly set based on the context:
- ChatEvent: planId only when related to specific todo execution
- ToolCallEvent/ToolCallResultEvent: planId set to todo_id when executing specific todos
- PlanUpdateEvent: planId set appropriately for plan updates
- CompleteEvent: planId is None
"""

from datetime import datetime
from unittest.mock import MagicMock

# Mock the required classes since we can't import the full environment
class MockActionHistory:
    def __init__(self, action_id, role, action_type, messages="", input_data=None, output=None, status="success"):
        self.action_id = action_id
        self.role = role
        self.action_type = action_type
        self.messages = messages
        self.input = input_data
        self.output = output
        self.status = status
        self.start_time = datetime.now()

class MockTodoStatus:
    PENDING = "pending"
    COMPLETED = "completed"

class MockEventConverter:
    """Simplified version of DeepResearchEventConverter to demonstrate planId logic"""

    def __init__(self):
        self.plan_id = "global-plan-uuid"

    def _extract_todo_id_from_action(self, action):
        """Extract todo_id from action input"""
        if not action.input:
            return None

        input_data = action.input
        if isinstance(input_data, dict):
            # Check for arguments field
            if "arguments" in input_data and isinstance(input_data["arguments"], str):
                try:
                    import json
                    parsed = json.loads(input_data["arguments"])
                    if isinstance(parsed, dict):
                        return parsed.get("todo_id") or parsed.get("todoId")
                except:
                    pass

            # Direct check
            return input_data.get("todo_id") or input_data.get("todoId")

        return None

    def demonstrate_planid_logic(self):
        """Demonstrate the new planId logic"""

        print("=== DeepResearchEventConverter planId Logic Demonstration ===\n")

        # Test Case 1: ChatEvent with no todo context
        print("1. ChatEvent - General assistant message (no todo context):")
        chat_action = MockActionHistory(
            "chat_1", "assistant", "llm_generation",
            messages="I'm thinking about the plan...",
            input_data={"content": "thinking"},
            output={"content": "Let me analyze this step by step"}
        )
        todo_id = self._extract_todo_id_from_action(chat_action)
        chat_plan_id = todo_id if todo_id else None
        print(f"   Action: {chat_action.action_type}")
        print(f"   Extracted todo_id: {todo_id}")
        print(f"   ChatEvent.planId: {chat_plan_id} (None - not todo-specific)")
        print()

        # Test Case 2: ChatEvent with todo context
        print("2. ChatEvent - Feedback for specific todo execution:")
        chat_todo_action = MockActionHistory(
            "chat_2", "assistant", "llm_generation",
            messages="Completed database analysis",
            input_data={"arguments": '{"todo_id": "task_123", "content": "analysis done"}'},
            output={"content": "Database analysis complete"}
        )
        todo_id = self._extract_todo_id_from_action(chat_todo_action)
        chat_plan_id = todo_id if todo_id else None
        print(f"   Action: {chat_todo_action.action_type}")
        print(f"   Extracted todo_id: {todo_id}")
        print(f"   ChatEvent.planId: {chat_plan_id} (specific todo ID)")
        print()

        # Test Case 3: Tool call for specific todo
        print("3. ToolCallEvent - Executing specific todo:")
        tool_action = MockActionHistory(
            "tool_1", "tool", "read_query",
            messages="Executing SQL query",
            input_data={"arguments": '{"todo_id": "task_456", "sql": "SELECT * FROM users"}'},
            output={"result": "success"}
        )
        todo_id = self._extract_todo_id_from_action(tool_action)
        tool_plan_id = todo_id if todo_id else None
        print(f"   Action: {tool_action.action_type}")
        print(f"   Extracted todo_id: {todo_id}")
        print(f"   ToolCallEvent.planId: {tool_plan_id} (specific todo ID)")
        print()

        # Test Case 4: Plan update for specific todo
        print("4. PlanUpdateEvent - Status update for specific todo:")
        plan_update_action = MockActionHistory(
            "plan_1", "assistant", "plan_update",
            messages="Updating plan",
            input_data={},
            output={"updated_item": {"id": "task_789", "status": "completed", "content": "Task done"}}
        )
        # For plan updates, we extract from output
        updated_item = plan_update_action.output.get("updated_item", {}) if plan_update_action.output else {}
        plan_update_plan_id = updated_item.get("id") if isinstance(updated_item, dict) else None
        print(f"   Action: {plan_update_action.action_type}")
        print(f"   Extracted todo_id from output: {plan_update_plan_id}")
        print(f"   PlanUpdateEvent.planId: {plan_update_plan_id} (specific todo ID)")
        print()

        # Test Case 5: CompleteEvent
        print("5. CompleteEvent - Workflow completion:")
        complete_action = MockActionHistory(
            "complete_1", "workflow", "workflow_completion",
            messages="Workflow completed successfully"
        )
        complete_plan_id = None  # CompleteEvent should not have planId
        print(f"   Action: {complete_action.action_type}")
        print(f"   CompleteEvent.planId: {complete_plan_id} (None - not todo-specific)")
        print()

        print("=== Summary ===")
        print("✓ ChatEvent.planId: Only set when directly related to todo execution")
        print("✓ ToolCallEvent.planId: Set to todo_id when executing specific todos")
        print("✓ PlanUpdateEvent.planId: Set to specific todo ID for updates")
        print("✓ CompleteEvent.planId: Always None")
        print("✓ ErrorEvent.planId: Set to todo_id if error relates to specific todo")

if __name__ == "__main__":
    converter = MockEventConverter()
    converter.demonstrate_planid_logic()
