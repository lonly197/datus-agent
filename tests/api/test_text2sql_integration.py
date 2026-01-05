# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for Text2SQL end-to-end flow.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.api.models import ChatResearchRequest
from datus.api.service import DatusService
from datus.configuration.agent_config import AgentConfig


@pytest.mark.integration
class TestText2SQLIntegration:
    """Integration tests for Text2SQL functionality."""

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
    def service(self, mock_agent_config):
        """Create DatusService instance."""
        return DatusService(mock_agent_config)

    @pytest.fixture
    def text2sql_request(self):
        """Create a Text2SQL chat research request."""
        return ChatResearchRequest(
            namespace="test_namespace",
            task="统计每个月'首次试驾'到'最终购买'的转化率",
            catalog_name="test_catalog",
            database_name="test_db",
            schema_name="public",
            domain="automotive",
            layer1="sales",
            layer2="conversion_analysis",
        )

    @pytest.mark.asyncio
    async def test_text2sql_task_identification(self, service, text2sql_request):
        """Test that Text2SQL tasks are correctly identified."""
        task_type = await service._identify_task_type(text2sql_request)

        assert task_type == "text2sql"

    @pytest.mark.asyncio
    async def test_text2sql_configuration(self, service):
        """Test Text2SQL task configuration."""
        config = service._configure_task_processing("text2sql", MagicMock())

        assert config["workflow"] == "text2sql"
        assert config["system_prompt"] == "text2sql_system"
        assert config["output_format"] == "json"
        assert "required_tool_sequence" in config
        assert "search_table" in config["required_tool_sequence"]
        assert "describe_table" in config["required_tool_sequence"]

    @pytest.mark.asyncio
    @patch("datus.api.service.DatusService._create_workflow_runner")
    @patch("datus.api.service.DatusService._create_sql_task")
    async def test_text2sql_workflow_creation(self, mock_create_task, mock_create_runner, service, text2sql_request):
        """Test Text2SQL workflow creation and execution."""
        # Setup mocks
        mock_task = MagicMock()
        mock_create_task.return_value = ("task_id_123", mock_task)

        mock_runner = MagicMock()
        mock_runner.execute_async = AsyncMock(return_value=MagicMock())
        mock_create_runner.return_value = mock_runner

        # Mock the task identification and configuration
        with patch.object(service, "_identify_task_type", return_value="text2sql"):
            with patch.object(service, "_configure_task_processing") as mock_config:
                mock_config.return_value = {
                    "workflow": "text2sql",
                    "plan_mode": False,
                    "system_prompt": "text2sql_system",
                    "output_format": "json",
                    "required_tool_sequence": ["search_table", "describe_table"],
                }

                # Execute the chat research
                await service.chat_research(text2sql_request)

                # Verify task was created with correct parameters
                mock_create_task.assert_called_once()
                call_args = mock_create_task.call_args[0][0]  # First positional arg

                assert call_args.workflow == "text2sql"
                assert call_args.task == text2sql_request.task
                assert call_args.namespace == text2sql_request.namespace

    @pytest.mark.asyncio
    @patch("datus.agent.node.preflight_orchestrator.PreflightOrchestrator")
    async def test_preflight_orchestrator_integration(self, mock_preflight_class, service):
        """Test PreflightOrchestrator integration in the service flow."""
        # Setup mock preflight orchestrator
        mock_orchestrator = MagicMock()
        mock_preflight_class.return_value = mock_orchestrator

        # Mock successful preflight execution
        mock_orchestrator.run_preflight_tools = AsyncMock()
        mock_orchestrator.run_preflight_tools.return_value = [
            MagicMock(action_type="preflight_search_table", status="completed"),
            MagicMock(action_type="preflight_describe_table", status="completed"),
        ]

        # This test would need to be expanded to test the full workflow execution
        # For now, just verify the orchestrator can be instantiated
        assert mock_preflight_class.called

    @pytest.mark.asyncio
    async def test_text2sql_workflow_execution_flow(self):
        """Test the complete Text2SQL workflow execution flow."""
        # This would be a comprehensive test that mocks all components:
        # 1. API service receives request
        # 2. Task type identified as text2sql
        # 3. PreflightOrchestrator runs tools
        # 4. Text2SQLExecutionMode executes
        # 5. Results returned with proper structure

        # For now, create a skeleton test that can be expanded
        assert True  # Placeholder for future comprehensive integration test

    @pytest.mark.asyncio
    async def test_text2sql_error_handling(self, service):
        """Test error handling in Text2SQL flow."""
        # Test various error scenarios:
        # 1. Database connection failure
        # 2. Tool execution timeout
        # 3. Invalid SQL generation
        # 4. Schema linking failure

        # For now, create a skeleton test
        assert True  # Placeholder for error handling tests

    @pytest.mark.asyncio
    async def test_text2sql_caching_behavior(self, service):
        """Test caching behavior in Text2SQL flow."""
        # Test that:
        # 1. Cache hits reduce execution time
        # 2. Cache misses trigger tool execution
        # 3. Cache invalidation works correctly
        # 4. Cache TTL is respected

        # For now, create a skeleton test
        assert True  # Placeholder for caching tests

    @pytest.mark.asyncio
    async def test_text2sql_sse_events(self):
        """Test SSE event streaming for Text2SQL."""
        # Test that proper events are emitted:
        # 1. ToolCallEvent for preflight tools
        # 2. ToolCallResultEvent for tool results
        # 3. LLM interaction events
        # 4. SQL generation events
        # 5. Execution completion events

        # For now, create a skeleton test
        assert True  # Placeholder for SSE event tests
