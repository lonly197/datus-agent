import json

from datus.tools.func_tool.plan_tools import PlanTool, SessionTodoStorage, TodoStatus


def test_todo_write_preserves_requires_tool():
    # Create PlanTool with dummy session (SessionTodoStorage does not use session for logic)
    plan_tool = PlanTool(session=None)

    # Prepare todos JSON with requires_tool flags
    todos = [
        {"content": "Step that needs a tool", "status": "pending", "requires_tool": True},
        {"content": "Step that does not need a tool", "status": "pending", "requires_tool": False},
        {"content": "Already done step", "status": "completed", "requires_tool": True},
    ]

    result = plan_tool.todo_write(json.dumps(todos))
    assert result is not None
    assert "todo_list" in result.result

    todo_list = plan_tool.storage.get_todo_list()
    assert todo_list is not None
    # Find items by content
    needs_tool = next((t for t in todo_list.items if t.content.startswith("Step that needs")), None)
    no_tool = next((t for t in todo_list.items if t.content.startswith("Step that does not")), None)
    done_step = next((t for t in todo_list.items if t.content.startswith("Already done")), None)

    assert needs_tool is not None and needs_tool.requires_tool is True
    assert no_tool is not None and no_tool.requires_tool is False
    assert done_step is not None and done_step.status == TodoStatus.COMPLETED


