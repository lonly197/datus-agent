# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for Text2SQLExecutionMode.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.agent.node.execution_event_manager import ExecutionContext, Text2SQLExecutionMode
from datus.configuration.agent_config import AgentConfig


class TestText2SQLExecutionMode:
    """Test cases for Text2SQLExecutionMode."""

    @pytest.fixture
    def mock_agent_config(self):
        """Mock agent configuration."""
        config = MagicMock(spec=AgentConfig)
        config.current_catalog = "default_catalog"
        config.current_database = "test_db"
        config.current_schema = "public"
        config.db_type = "sqlite"
        return config

    @pytest.fixture
    def mock_model(self):
        """Mock LLM model."""
        model = MagicMock()
        model.generate_with_json_output = AsyncMock()
        return model

    @pytest.fixture
    def execution_context(self, mock_agent_config, mock_model):
        """Create execution context for testing."""
        return ExecutionContext(
            scenario="text2sql",
            task_data={"task": "Show me all users from the users table"},
            agent_config=mock_agent_config,
            model=mock_model,
            workflow_metadata={},
        )

    @pytest.fixture
    def execution_mode(self, execution_context):
        """Create Text2SQLExecutionMode instance."""
        return Text2SQLExecutionMode(event_manager=MagicMock(), context=execution_context)

    @pytest.mark.asyncio
    async def test_analyze_query_intent_success(self, execution_mode, mock_model):
        """Test successful query intent analysis."""
        # Setup mock response
        mock_model.generate_with_json_output.return_value = {
            "query_type": "SELECT",
            "entities": ["users"],
            "filters": [],
            "aggregations": [],
            "sort_requirements": [],
            "temporal_aspects": [],
        }

        result = await execution_mode._analyze_query_intent()

        assert result["query_type"] == "SELECT"
        assert result["entities"] == ["users"]
        mock_model.generate_with_json_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_query_intent_fallback(self, execution_mode):
        """Test fallback when model is not available."""
        execution_mode.context.model = None

        result = await execution_mode._analyze_query_intent()

        assert result["query_type"] == "SELECT"
        assert result["entities"] == []

    @pytest.mark.asyncio
    @patch("datus.agent.node.execution_event_manager.db_function_tool_instance")
    async def test_link_schema_success(self, mock_db_tool_func, execution_mode, mock_agent_config):
        """Test successful schema linking."""
        # Setup mocks
        mock_db_tool = MagicMock()
        mock_db_tool.search_table.return_value = MagicMock(
            success=True,
            result=[{"table_name": "users", "relevance_score": 0.9}, {"table_name": "orders", "relevance_score": 0.7}],
        )
        mock_db_tool.describe_table.return_value = MagicMock(
            success=True, result={"columns": [{"name": "id", "type": "INTEGER"}]}
        )
        mock_db_tool_func.return_value = mock_db_tool

        # Mock _analyze_query_intent to return entities
        execution_mode._analyze_query_intent = AsyncMock(return_value={"entities": ["users"]})

        result = await execution_mode._link_schema()

        assert result["tables"] is not None
        assert len(result["tables"]) > 0
        mock_db_tool.search_table.assert_called_once()
        mock_db_tool.describe_table.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_sql_success(self, execution_mode, mock_model):
        """Test successful SQL generation."""
        # Setup mock
        mock_model.generate.return_value = "SELECT id, name FROM users WHERE active = 1;"

        intent = {"query_type": "SELECT", "entities": ["users"]}
        schema_info = {"tables": [{"table_name": "users", "schema": {"columns": [{"name": "id", "type": "INTEGER"}]}}]}

        result = await execution_mode._generate_sql(intent, schema_info)

        assert "SELECT" in result
        assert "FROM users" in result
        mock_model.generate.assert_called_once()

    @pytest.mark.asyncio
    @patch("datus.agent.node.execution_event_manager.db_function_tool_instance")
    async def test_validate_sql_syntax_success(self, mock_db_tool_func, execution_mode):
        """Test successful SQL syntax validation."""
        # Setup mock
        mock_db_tool = MagicMock()
        mock_db_tool.validate_sql_syntax.return_value = MagicMock(
            success=True, result={"tables_referenced": ["users"], "sql_type": "SELECT"}
        )
        mock_db_tool_func.return_value = mock_db_tool

        result = await execution_mode._validate_sql_syntax("SELECT * FROM users")

        assert result["valid"] is True
        assert result["tables_referenced"] == ["users"]
        mock_db_tool.validate_sql_syntax.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_sql_syntax_failure(self, execution_mode):
        """Test SQL syntax validation failure."""
        # Mock the db_function_tool_instance to raise an exception
        with patch("datus.agent.node.execution_event_manager.db_function_tool_instance") as mock_func:
            mock_func.side_effect = Exception("Database error")

            result = await execution_mode._validate_sql_syntax("INVALID SQL")

            assert result["valid"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_full_flow(self, execution_mode, mock_model):
        """Test the complete execution flow."""
        # Setup mocks
        execution_mode._analyze_query_intent = AsyncMock(return_value={"query_type": "SELECT", "entities": ["users"]})
        execution_mode._link_schema = AsyncMock(
            return_value={"tables": [{"table_name": "users", "schema": {"columns": []}}]}
        )
        execution_mode._generate_sql = AsyncMock(return_value="SELECT * FROM users")
        execution_mode._validate_sql_syntax = AsyncMock(return_value={"valid": True, "tables_referenced": ["users"]})

        # Mock event manager
        mock_event_manager = MagicMock()
        execution_mode.event_manager = mock_event_manager

        # Execute the mode (it records events through event manager)
        await execution_mode.execute()
        mock_event_manager.complete_execution.assert_called_once()

        # Verify the final result
        call_args = mock_event_manager.complete_execution.call_args
        result = call_args[1]  # kwargs
        assert "sql" in result
        assert result["sql"] == "SELECT * FROM users"

    @pytest.mark.asyncio
    async def test_start_registers_execution(self, execution_mode, mock_agent_config, mock_model):
        """Test that start() method registers execution with ExecutionEventManager."""
        # Create mock event manager
        mock_event_manager = MagicMock()
        execution_mode.event_manager = mock_event_manager

        # Mock start_execution to return a context
        mock_context = ExecutionContext(
            scenario="text2sql",
            task_data={"task": "test task"},
            agent_config=mock_agent_config,
            model=mock_model,
        )
        mock_event_manager.start_execution = AsyncMock(return_value=mock_context)

        # Call start method
        await execution_mode.start()

        # Verify start_execution was called with correct parameters
        mock_event_manager.start_execution.assert_called_once_with(
            execution_id=execution_mode.execution_id,
            scenario="text2sql",
            task_data={"task": "Show me all users from the users table"},
            agent_config=mock_agent_config,
            model=mock_model,
            workflow_metadata={},
        )

        # Verify execution is marked as started
        assert execution_mode._started is True

        # Test idempotency - calling start again should not call start_execution again
        mock_event_manager.start_execution.reset_mock()
        await execution_mode.start()
        mock_event_manager.start_execution.assert_not_called()

    @pytest.mark.asyncio
    async def test_execution_manager_integration(self, execution_mode):
        """Test integration with ExecutionEventManager after start()."""
        from datus.agent.node.execution_event_manager import ExecutionEventManager
        from datus.schemas.action_history import ActionHistoryManager

        # Create real event manager and action history manager
        action_history_manager = ActionHistoryManager()
        event_manager = ExecutionEventManager(action_history_manager)

        # Replace mock with real event manager
        execution_mode.event_manager = event_manager

        # Call start - this should register the execution
        await execution_mode.start()

        # Verify execution is registered
        assert execution_mode.execution_id in event_manager._active_executions

        # Test that update_execution_status no longer produces warnings
        # (This would previously log "Execution X not found")
        with patch("datus.agent.node.execution_event_manager.logger") as mock_logger:
            await event_manager.update_execution_status(execution_mode.execution_id, "executing", "test step")
            # Should not call warning about execution not found
            mock_logger.warning.assert_not_called()
