# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for PreflightOrchestrator.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from datus.agent.node.preflight_orchestrator import PreflightOrchestrator, PreflightToolResult
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager


class TestPreflightOrchestrator:
    """Test cases for PreflightOrchestrator."""

    @pytest.fixture
    def mock_agent_config(self):
        """Mock agent configuration."""
        config = MagicMock(spec=AgentConfig)
        config.current_catalog = "default_catalog"
        config.current_database = "test_db"
        config.current_schema = "public"
        return config

    @pytest.fixture
    def mock_plan_hooks(self):
        """Mock plan hooks with QueryCache."""
        hooks = MagicMock()
        hooks.enable_query_caching = True
        hooks.query_cache = MagicMock()
        hooks.query_cache.get = MagicMock(return_value=None)  # No cache hits by default
        hooks.query_cache.set = MagicMock()
        return hooks

    @pytest.fixture
    def orchestrator(self, mock_agent_config, mock_plan_hooks):
        """Create PreflightOrchestrator instance."""
        return PreflightOrchestrator(
            agent_config=mock_agent_config,
            plan_hooks=mock_plan_hooks
        )

    @pytest.fixture
    def mock_workflow(self):
        """Mock workflow with task."""
        workflow = MagicMock()
        workflow.task = MagicMock()
        workflow.task.catalog_name = "test_catalog"
        workflow.task.database_name = "test_db"
        workflow.task.schema_name = "public"
        workflow.task.task = "Show me all users"
        workflow.context = type('Context', (), {})()
        workflow.context.preflight_results = []
        return workflow

    @pytest.fixture
    def action_history_manager(self):
        """Create action history manager."""
        return ActionHistoryManager()

    def test_preflight_tool_result_creation(self):
        """Test PreflightToolResult creation and serialization."""
        result = PreflightToolResult(
            tool_name="search_table",
            success=True,
            result={"tables": ["users", "orders"]},
            execution_time=1.5,
            cache_hit=False
        )

        assert result.tool_name == "search_table"
        assert result.success is True
        assert result.result == {"tables": ["users", "orders"]}
        assert result.execution_time == 1.5
        assert result.cache_hit is False

        # Test serialization
        dict_result = result.to_dict()
        assert dict_result["tool_name"] == "search_table"
        assert dict_result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_search_table_success(self, orchestrator):
        """Test successful search_table execution."""
        with patch.object(orchestrator, '_get_db_func_tool') as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.search_table.return_value = MagicMock(
                result=[{"table_name": "users"}]
            )
            mock_get_tool.return_value = mock_tool

            result = await orchestrator._execute_search_table(
                query="users", catalog="test", database="db", schema="public"
            )

            assert result["success"] is True
            assert result["tables"] == [{"table_name": "users"}]
            mock_tool.search_table.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_describe_table_success(self, orchestrator):
        """Test successful describe_table execution."""
        with patch.object(orchestrator, '_get_db_func_tool') as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.describe_table.return_value = MagicMock(
                result={"columns": [{"name": "id", "type": "INTEGER"}]}
            )
            mock_get_tool.return_value = mock_tool

            result = await orchestrator._execute_describe_table(
                table_names=["users"], catalog="test", database="db", schema="public"
            )

            assert result["success"] is True
            assert len(result["tables_described"]) == 1
            mock_tool.describe_table.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_search_reference_sql_success(self, orchestrator):
        """Test successful search_reference_sql execution."""
        with patch.object(orchestrator, '_get_context_search_tools') as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.search_reference_sql.return_value = MagicMock(
                result=[{"sql": "SELECT * FROM users", "description": "Get all users"}]
            )
            mock_get_tool.return_value = mock_tool

            result = await orchestrator._execute_search_reference_sql("users")

            assert result["success"] is True
            assert len(result["reference_sqls"]) == 1
            mock_tool.search_reference_sql.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_parse_temporal_expressions_success(self, orchestrator):
        """Test successful parse_temporal_expressions execution."""
        with patch.object(orchestrator, '_get_date_parsing_tools') as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool.extract_and_parse_dates.return_value = MagicMock(
                result=[{"original": "last month", "parsed": "2024-11-01"}]
            )
            mock_get_tool.return_value = mock_tool

            result = await orchestrator._execute_parse_temporal_expressions("Show data from last month")

            assert result["success"] is True
            assert len(result["temporal_expressions"]) == 1
            mock_tool.extract_and_parse_dates.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_preflight_tools_with_cache_hit(self, orchestrator, mock_workflow, action_history_manager, mock_plan_hooks):
        """Test preflight execution with cache hit."""
        # Setup cache hit
        mock_plan_hooks.query_cache.get.return_value = {
            "success": True,
            "tables": [{"table_name": "users"}]
        }

        # Collect actions
        actions = []
        async for action in orchestrator.run_preflight_tools(
            workflow=mock_workflow,
            action_history_manager=action_history_manager,
            required_tools=["search_table"]
        ):
            actions.append(action)

        # Verify cache was checked
        mock_plan_hooks.query_cache.get.assert_called_once()

        # Verify actions were yielded
        assert len(actions) >= 2  # At least tool start and result actions

        # Verify results were injected into context
        assert hasattr(mock_workflow.context, 'preflight_results')
        assert len(mock_workflow.context.preflight_results) > 0

    @pytest.mark.asyncio
    async def test_run_preflight_tools_with_cache_miss(self, orchestrator, mock_workflow, action_history_manager, mock_plan_hooks):
        """Test preflight execution with cache miss."""
        # Setup cache miss
        mock_plan_hooks.query_cache.get.return_value = None

        # Mock successful tool execution
        with patch.object(orchestrator, '_execute_search_table') as mock_execute:
            mock_execute.return_value = {"success": True, "tables": ["users"]}

            # Collect actions
            actions = []
            async for action in orchestrator.run_preflight_tools(
                workflow=mock_workflow,
                action_history_manager=action_history_manager,
                required_tools=["search_table"]
            ):
                actions.append(action)

            # Verify cache was checked and set
            mock_plan_hooks.query_cache.get.assert_called_once()
            mock_plan_hooks.query_cache.set.assert_called_once()

            # Verify tool was executed
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_preflight_tools_with_failure(self, orchestrator, mock_workflow, action_history_manager, mock_plan_hooks):
        """Test preflight execution with tool failure."""
        # Setup cache miss
        mock_plan_hooks.query_cache.get.return_value = None

        # Mock failed tool execution
        with patch.object(orchestrator, '_execute_search_table') as mock_execute:
            mock_execute.side_effect = Exception("Tool failed")

            # Collect actions
            actions = []
            async for action in orchestrator.run_preflight_tools(
                workflow=mock_workflow,
                action_history_manager=action_history_manager,
                required_tools=["search_table"]
            ):
                actions.append(action)

            # Verify failure was handled gracefully
            assert len(actions) >= 2  # Actions were still yielded

            # Verify result contains error
            assert len(mock_workflow.context.preflight_results) > 0
            result = mock_workflow.context.preflight_results[0]
            assert result["success"] is False
            assert "error" in result

    def test_inject_preflight_results_into_context(self, orchestrator, mock_workflow):
        """Test injecting preflight results into workflow context."""
        results = [
            PreflightToolResult("search_table", True, {"tables": ["users"]}),
            PreflightToolResult("describe_table", True, {"tables_described": [{"table_name": "users"}]})
        ]

        orchestrator._inject_preflight_results_into_context(mock_workflow, results)

        # Verify structured data extraction
        assert hasattr(mock_workflow.context, 'schema_info')
        assert hasattr(mock_workflow.context, 'reference_sqls')
        assert hasattr(mock_workflow.context, 'temporal_expressions')

        # Verify preflight results
        assert len(mock_workflow.context.preflight_results) == 2
