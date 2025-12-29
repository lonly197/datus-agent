import pytest
from rich.console import Console

from datus.cli.plan_hooks import PlanModeHooks


class DummyAgentConfig:
    def __init__(self, keyword_map=None, enable_fallback=True):
        self.plan_executor = {"keyword_tool_map": keyword_map} if keyword_map else {}
        self.plan_executor_keyword_map = keyword_map
        self.plan_executor_enable_fallback = enable_fallback


class DummySession:
    pass


def test_match_tool_for_todo_default():
    console = Console()
    session = DummySession()
    # default mapping is used when agent_config lacks custom mapping
    hooks = PlanModeHooks(
        console=console,
        session=session,
        auto_mode=False,
        action_history_manager=None,
        agent_config=None,
        emit_queue=None,
        model=None,
    )
    assert hooks._match_tool_for_todo("Search the table structure of ods_xxx") == "search_table"
    assert hooks._match_tool_for_todo("Please generate report") == "report"


def test_match_tool_for_todo_custom_map():
    console = Console()
    session = DummySession()
    custom_map = {"inspect_schema": ["inspect", "inspect table"], "execute_sql": ["run my sql"]}
    agent_cfg = DummyAgentConfig(keyword_map=custom_map)
    hooks = PlanModeHooks(
        console=console,
        session=session,
        auto_mode=False,
        action_history_manager=None,
        agent_config=agent_cfg,
        emit_queue=None,
        model=None,
    )
    assert hooks._match_tool_for_todo("Please inspect table foo") == "inspect_schema"
    assert hooks._match_tool_for_todo("will run my sql now") == "execute_sql"


def test_determine_fallback_candidates():
    console = Console()
    session = DummySession()
    hooks = PlanModeHooks(
        console=console,
        session=session,
        auto_mode=False,
        action_history_manager=None,
        agent_config=None,
        emit_queue=None,
        model=None,
    )

    # Test database-related content
    candidates = hooks._determine_fallback_candidates("Find all users in the customer table")
    tool_names = [name for name, _ in candidates]
    assert "search_table" in tool_names

    # Test SQL-related content
    candidates = hooks._determine_fallback_candidates("Execute the SQL query to get results")
    tool_names = [name for name, _ in candidates]
    assert "execute_sql" in tool_names

    # Test file-related content
    candidates = hooks._determine_fallback_candidates("Save the results to a file")
    tool_names = [name for name, _ in candidates]
    assert "write_file" in tool_names

    # Test report-related content
    candidates = hooks._determine_fallback_candidates("Generate a final report")
    tool_names = [name for name, _ in candidates]
    assert "report" in tool_names


def test_keyword_map_merging():
    console = Console()
    session = DummySession()

    # Test with custom config that extends defaults
    custom_map = {"custom_tool": ["custom action", "special task"]}
    agent_cfg = DummyAgentConfig(keyword_map=custom_map)
    hooks = PlanModeHooks(
        console=console,
        session=session,
        auto_mode=False,
        action_history_manager=None,
        agent_config=agent_cfg,
        emit_queue=None,
        model=None,
    )

    # Should have both default tools and custom tools
    assert "search_table" in hooks.keyword_map
    assert "custom_tool" in hooks.keyword_map
    assert hooks.keyword_map["custom_tool"] == ["custom action", "special task"]
