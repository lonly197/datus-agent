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


def test_todo_write_preserves_llm_reasoning_fields():
    # Create PlanTool with dummy session
    plan_tool = PlanTool(session=None)

    # Prepare todos JSON with LLM reasoning fields
    todos = [
        {
            "content": "Analyze the business requirements",
            "status": "pending",
            "requires_tool": False,
            "requires_llm_reasoning": True,
            "reasoning_type": "analysis"
        },
        {
            "content": "Generate the SQL query",
            "status": "pending",
            "requires_tool": True,
            "requires_llm_reasoning": False
        },
        {
            "content": "Execute the query",
            "status": "pending",
            "requires_tool": True,
            "tool_calls": [{"tool": "execute_sql", "arguments": {"query": "SELECT * FROM users"}}]
        }
    ]

    result = plan_tool.todo_write(json.dumps(todos))
    assert result is not None
    assert "todo_list" in result.result

    todo_list = plan_tool.storage.get_todo_list()
    assert todo_list is not None

    # Find items by content
    analysis_item = next((t for t in todo_list.items if "Analyze" in t.content), None)
    sql_item = next((t for t in todo_list.items if "Generate" in t.content), None)
    execute_item = next((t for t in todo_list.items if "Execute" in t.content), None)

    # Check LLM reasoning fields
    assert analysis_item is not None
    assert analysis_item.requires_llm_reasoning is True
    assert analysis_item.reasoning_type == "analysis"

    # Check normal tool execution
    assert sql_item is not None
    assert sql_item.requires_llm_reasoning is False
    assert sql_item.requires_tool is True

    # Check tool_calls field
    assert execute_item is not None
    assert execute_item.tool_calls == [{"tool": "execute_sql", "arguments": {"query": "SELECT * FROM users"}}]


