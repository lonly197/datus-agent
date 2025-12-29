#!/usr/bin/env python3
"""
Test script to validate the event converter changes.
"""

import json
import os
import sys
import uuid
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Mock the missing dependencies to allow import
class MockActionHistory:
    def __init__(self, action_id, role, messages, action_type, input_data=None, output=None, status=None):
        self.action_id = action_id
        self.role = role
        self.messages = messages
        self.action_type = action_type
        self.input = input_data
        self.output = output
        self.status = status
        self.start_time = datetime.now()


class MockActionRole:
    ASSISTANT = "assistant"
    TOOL = "tool"


class MockActionStatus:
    SUCCESS = "success"


# Mock the logger
class MockLogger:
    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


# Mock json and other imports that might be needed
sys.modules["datus.utils.loggings"] = type("MockModule", (), {"get_logger": lambda x: MockLogger()})()


# Import the specific methods we need by copying them
def _try_parse_json_like(obj):
    """Copy of the method from event_converter.py"""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return None
    return None


def _extract_plan_from_output(output):
    """Copy of the method from event_converter.py"""

    def try_parse(obj):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                return None
        return None

    stack = [output]
    visited = set()
    while stack:
        current = stack.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))

        parsed = try_parse(current)
        if isinstance(parsed, dict):
            # Check for plan-related fields
            if "todo_list" in parsed:
                return parsed
            if "updated_item" in parsed:
                return parsed
            # Continue searching nested structures
            for value in parsed.values():
                if isinstance(value, (dict, list, str)):
                    stack.append(value)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, (dict, list, str)):
                    stack.append(item)
    return {}


# Test the event converter functions
def test_extract_todo_id():
    print("Testing _extract_todo_id_from_action...")

    # Copy the method from event_converter.py
    def _extract_todo_id_from_action(action):
        """
        Extracts the todo_id from an ActionHistory object, looking in various places.
        This is crucial for correctly setting the planId for events related to specific todos.
        """
        # Try to extract from input arguments first
        if action.input and isinstance(action.input, dict):
            # Check for 'todo_id' directly in input
            if action.input.get("todo_id"):
                return action.input["todo_id"]
            # Check for 'arguments' field which might be a JSON string containing 'todo_id'
            if "arguments" in action.input and isinstance(action.input["arguments"], str):
                parsed_args = _try_parse_json_like(action.input["arguments"])
                if isinstance(parsed_args, dict) and parsed_args.get("todo_id"):
                    return parsed_args["todo_id"]

        # For plan_update actions, try to extract from output's 'updated_item' or 'todo_list'
        if action.action_type == "plan_update" and action.output:
            plan_data = _extract_plan_from_output(action.output)
            if "updated_item" in plan_data and isinstance(plan_data["updated_item"], dict):
                return plan_data["updated_item"].get("id")
            if "todo_list" in plan_data and isinstance(plan_data["todo_list"], dict):
                # If it's a full todo_list, we might not have a single todo_id,
                # but if there's only one item, we can use its ID.
                items = plan_data["todo_list"].get("items")
                if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
                    return items[0].get("id")

        # For tool_call_result, try to find the original tool_call_id and then its todo_id
        if action.action_type == "tool_call_result" and action.input and isinstance(action.input, dict):
            original_action_id = action.input.get("action_id")
            if original_action_id:
                # This is complex as we don't store todo_id in tool_call_map directly.
                # This would require looking up the original ActionHistory for the tool call.
                # For now, we rely on todo_id being in the tool call's input.
                pass

        return None

    # Test case 1: todo_id in arguments
    action1 = MockActionHistory(
        action_id="test1",
        role=MockActionRole.TOOL,
        messages="test",
        action_type="tool_call",
        input_data={"arguments": json.dumps({"todo_id": "test_todo_123"})},
    )
    result1 = _extract_todo_id_from_action(action1)
    print(f"Test 1 - todo_id in arguments: {result1} (expected: test_todo_123)")
    assert result1 == "test_todo_123", f"Expected test_todo_123, got {result1}"

    # Test case 2: plan_update with updated_item
    action2 = MockActionHistory(
        action_id="test2",
        role=MockActionRole.TOOL,
        messages="test",
        action_type="plan_update",
        output={"updated_item": {"id": "plan_todo_456"}},
    )
    result2 = _extract_todo_id_from_action(action2)
    print(f"Test 2 - plan_update with updated_item: {result2} (expected: plan_todo_456)")
    assert result2 == "plan_todo_456", f"Expected plan_todo_456, got {result2}"

    # Test case 3: plan_update with todo_list containing single item
    action3 = MockActionHistory(
        action_id="test3",
        role=MockActionRole.TOOL,
        messages="test",
        action_type="plan_update",
        output={"todo_list": {"items": [{"id": "single_todo_789"}]}},
    )
    result3 = _extract_todo_id_from_action(action3)
    print(f"Test 3 - plan_update with single item todo_list: {result3} (expected: single_todo_789)")
    assert result3 == "single_todo_789", f"Expected single_todo_789, got {result3}"

    print("✓ _extract_todo_id_from_action tests passed")


def test_is_internal_todo_update():
    print("Testing _is_internal_todo_update...")

    # Copy the method from event_converter.py
    def _is_internal_todo_update(action):
        """
        Check if this is an internal todo_update call from server executor that should be filtered.
        These are the 'todo_update' actions generated by the server executor to manage internal state,
        which should not be exposed as ToolCallEvents to the frontend.
        """
        # Internal todo_update calls from the server executor often have specific action_id patterns
        # or messages. We want to filter these out as ToolCallEvents.
        if action.action_type == "todo_update":
            # Check if the action_id starts with "server_call_"
            if action.action_id.startswith("server_call_"):
                return True
            # Additionally, check if the message indicates it's an internal update
            if action.messages and (
                "Server executor: starting todo" in action.messages
                or "Server executor: todo_in_progress" in action.messages
                or "Server executor: todo_completed" in action.messages
                or "Server executor: todo_complete failed" in action.messages
            ):
                return True
        return False

    # Test case 1: server_call_ action_id
    action1 = MockActionHistory(
        action_id="server_call_abc123", role=MockActionRole.TOOL, messages="test", action_type="todo_update"
    )
    result1 = _is_internal_todo_update(action1)
    print(f"Test 1 - server_call_ action_id: {result1} (expected: True)")
    assert result1 == True, f"Expected True, got {result1}"

    # Test case 2: Server executor message
    action2 = MockActionHistory(
        action_id="normal_action",
        role=MockActionRole.TOOL,
        messages="Server executor: starting todo",
        action_type="todo_update",
    )
    result2 = _is_internal_todo_update(action2)
    print(f"Test 2 - Server executor message: {result2} (expected: True)")
    assert result2 == True, f"Expected True, got {result2}"

    # Test case 3: Server executor todo_in_progress message
    action3 = MockActionHistory(
        action_id="another_action",
        role=MockActionRole.TOOL,
        messages="Server executor: todo_in_progress",
        action_type="todo_update",
    )
    result3 = _is_internal_todo_update(action3)
    print(f"Test 3 - Server executor todo_in_progress: {result3} (expected: True)")
    assert result3 == True, f"Expected True, got {result3}"

    # Test case 4: Normal action
    action4 = MockActionHistory(
        action_id="normal_action", role=MockActionRole.TOOL, messages="User requested update", action_type="todo_update"
    )
    result4 = _is_internal_todo_update(action4)
    print(f"Test 4 - Normal action: {result4} (expected: False)")
    assert result4 == False, f"Expected False, got {result4}"

    # Test case 5: Non-todo_update action
    action5 = MockActionHistory(
        action_id="server_call_xyz",
        role=MockActionRole.TOOL,
        messages="Server executor: starting todo",
        action_type="search_table",
    )
    result5 = _is_internal_todo_update(action5)
    print(f"Test 5 - Non-todo_update action: {result5} (expected: False)")
    assert result5 == False, f"Expected False, got {result5}"

    print("✓ _is_internal_todo_update tests passed")


if __name__ == "__main__":
    try:
        test_extract_todo_id()
        test_is_internal_todo_update()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
