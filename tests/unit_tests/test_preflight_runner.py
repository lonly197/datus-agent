from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.schemas.action_history import ActionHistoryManager
from datus.schemas.node_models import SqlTask


class TestPreflightRunner:
    """Test preflight tool runner functionality."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow with required tool sequence."""
        workflow = MagicMock()
        workflow.metadata = {"required_tool_sequence": ["describe_table", "search_external_knowledge", "read_query"]}
        workflow.task = SqlTask(
            task="审查SQL：SELECT * FROM test_table WHERE id = 1",
            database_type="starrocks",
            database_name="test",
            catalog_name="default_catalog",
        )
        workflow.context = MagicMock()
        workflow.context.table_schemas = []
        workflow.context.sql_contexts = []
        return workflow

    @pytest.fixture
    def mock_action_history_manager(self):
        """Create a mock action history manager."""
        return MagicMock(spec=ActionHistoryManager)

    @pytest.fixture
    def chat_node(self):
        """Create a ChatAgenticNode instance with mocked dependencies."""
        with patch("datus.agent.node.chat_agentic_node.GenSQLAgenticNode.__init__", return_value=None):
            node = ChatAgenticNode(node_id="test_node", description="Test node", node_type="chat")

            # Mock required attributes
            node.db_func_tool = MagicMock()
            node.context_search_tools = MagicMock()
            node.plan_hooks = MagicMock()
            node.plan_hooks.monitor = MagicMock()

            return node

    @pytest.mark.asyncio
    async def test_run_preflight_tools_no_sequence(self, chat_node, mock_workflow, mock_action_history_manager):
        """Test preflight runner with no required tool sequence."""
        mock_workflow.metadata = {}

        result = await chat_node.run_preflight_tools(mock_workflow, mock_action_history_manager)

        assert result is True
        mock_action_history_manager.add_action.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_preflight_tools_success(self, chat_node, mock_workflow, mock_action_history_manager):
        """Test successful preflight tool execution."""
        # Mock successful tool executions
        chat_node._execute_preflight_tool = AsyncMock(
            side_effect=[
                {"success": True, "result": {"table_name": "test_table", "columns": []}},
                {"success": True, "result": [{"terminology": "test", "explanation": "test explanation"}]},
                {"success": True, "result": "SELECT * FROM test_table"},
            ]
        )

        chat_node._inject_tool_result_into_context = MagicMock()

        result = await chat_node.run_preflight_tools(mock_workflow, mock_action_history_manager)

        assert result is True
        assert chat_node._execute_preflight_tool.call_count == 3
        chat_node._inject_tool_result_into_context.assert_called()

    @pytest.mark.asyncio
    async def test_run_preflight_tools_partial_failure(self, chat_node, mock_workflow, mock_action_history_manager):
        """Test preflight tool execution with partial failures."""
        # Mock tool executions with one failure
        chat_node._execute_preflight_tool = AsyncMock(
            side_effect=[
                {"success": True, "result": {"table_name": "test_table", "columns": []}},
                {"success": False, "error": "Knowledge search failed"},
                {"success": True, "result": "SELECT * FROM test_table"},
            ]
        )

        chat_node._inject_tool_result_into_context = MagicMock()

        result = await chat_node.run_preflight_tools(mock_workflow, mock_action_history_manager)

        assert result is False  # Should return False due to failure
        assert chat_node._execute_preflight_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_preflight_tool_describe_table(self, chat_node):
        """Test describe_table tool execution."""
        chat_node.db_func_tool.describe_table = MagicMock(
            return_value=MagicMock(__dict__={"success": True, "table_name": "test_table", "columns": []})
        )

        result = await chat_node._execute_preflight_tool(
            "describe_table", "SELECT * FROM test_table", ["test_table"], "catalog", "database", "schema"
        )

        assert result["success"] is True
        assert "table_name" in result
        chat_node.db_func_tool.describe_table.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_preflight_tool_search_knowledge(self, chat_node):
        """Test search_external_knowledge tool execution."""
        chat_node.context_search_tools.search_external_knowledge = MagicMock(
            return_value=MagicMock(
                __dict__={"success": True, "result": [{"terminology": "test", "explanation": "explanation"}]}
            )
        )

        result = await chat_node._execute_preflight_tool(
            "search_external_knowledge", "SELECT * FROM test_table", ["test_table"], "catalog", "database", "schema"
        )

        assert result["success"] is True
        chat_node.context_search_tools.search_external_knowledge.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_preflight_tool_read_query(self, chat_node):
        """Test read_query tool execution."""
        chat_node.db_func_tool.read_query = MagicMock(
            return_value=MagicMock(__dict__={"success": True, "result": "query result"})
        )

        result = await chat_node._execute_preflight_tool(
            "read_query", "SELECT * FROM test_table", ["test_table"], "catalog", "database", "schema"
        )

        assert result["success"] is True
        chat_node.db_func_tool.read_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_preflight_tool_get_ddl(self, chat_node):
        """Test get_table_ddl tool execution."""
        chat_node.db_func_tool.get_table_ddl = MagicMock(
            return_value=MagicMock(__dict__={"success": True, "result": {"definition": "CREATE TABLE test_table ..."}})
        )

        result = await chat_node._execute_preflight_tool(
            "get_table_ddl", "SELECT * FROM test_table", ["test_table"], "catalog", "database", "schema"
        )

        assert result["success"] is True
        chat_node.db_func_tool.get_table_ddl.assert_called_once()

    def test_extract_sql_from_task(self, chat_node):
        """Test SQL extraction from task text."""
        # Test backtick extraction
        task_text = "审查SQL：```sql\nSELECT * FROM table\n```"
        result = chat_node._extract_sql_from_task(task_text)
        assert result == "SELECT * FROM table"

        # Test SELECT statement extraction
        task_text = "检查这个查询：SELECT id, name FROM users WHERE active = 1;"
        result = chat_node._extract_sql_from_task(task_text)
        assert result == "SELECT id, name FROM users WHERE active = 1"

        # Test fallback to full text
        task_text = "Some random text without SQL"
        result = chat_node._extract_sql_from_task(task_text)
        assert result == task_text

    def test_parse_table_names_from_sql(self, chat_node):
        """Test table name parsing from SQL."""
        sql = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id WHERE u.active = 1"
        result = chat_node._parse_table_names_from_sql(sql)

        assert "users" in result
        assert "orders" in result

    def test_inject_tool_result_into_context(self, chat_node, mock_workflow):
        """Test tool result injection into workflow context."""
        # Test describe_table result injection
        result = {"success": True, "table_name": "test_table", "columns": [{"name": "id", "type": "int"}]}
        chat_node._inject_tool_result_into_context(mock_workflow, "describe_table", result)

        mock_workflow.context.table_schemas.append.assert_called_once()

        # Test search_external_knowledge result injection
        result = {"success": True, "result": [{"terminology": "test", "explanation": "explanation"}]}
        chat_node._inject_tool_result_into_context(mock_workflow, "search_external_knowledge", result)

        # Should update external_knowledge
        assert "StarRocks审查规则:" in mock_workflow.task.external_knowledge

        # Test read_query result injection
        result = {"success": True, "result": "query output", "row_count": 10}
        chat_node._inject_tool_result_into_context(mock_workflow, "read_query", result)

        mock_workflow.context.sql_contexts.append.assert_called_once()

        # Test get_table_ddl result injection
        result = {"success": True, "result": {"definition": "CREATE TABLE test ..."}}
        chat_node._inject_tool_result_into_context(mock_workflow, "get_table_ddl", result)

        assert "表DDL信息:" in mock_workflow.task.external_knowledge
