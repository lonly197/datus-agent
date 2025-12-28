import pytest

from datus.cli.plan_hooks import PlanModeHooks
from rich.console import Console


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
    hooks = PlanModeHooks(console=console, session=session, auto_mode=False, action_history_manager=None, agent_config=None, emit_queue=None)
    assert hooks._match_tool_for_todo("Search the table structure of ods_xxx") == "search_table"
    assert hooks._match_tool_for_todo("Please generate report") == "report"


def test_match_tool_for_todo_custom_map():
    console = Console()
    session = DummySession()
    custom_map = {"inspect_schema": ["inspect", "inspect table"], "execute_sql": ["run my sql"]}
    agent_cfg = DummyAgentConfig(keyword_map=custom_map)
    hooks = PlanModeHooks(console=console, session=session, auto_mode=False, action_history_manager=None, agent_config=agent_cfg, emit_queue=None)
    assert hooks._match_tool_for_todo("Please inspect table foo") == "inspect_schema"
    assert hooks._match_tool_for_todo("will run my sql now") == "execute_sql"


