from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.agent.node.schema_discovery_node import SchemaDiscoveryNode
from datus.schemas.node_models import BaseInput, SqlTask


class TestSchemaDiscoveryNodeV3:
    @pytest.fixture
    def mock_workflow(self):
        workflow = MagicMock()
        workflow.task = SqlTask(
            id="test_task",
            task="统计每个月首次试驾的转化率",
            database_type="sqlite",
            catalog_name="test_catalog",
            database_name="test_db",
            schema_name="public",
            subject_path=[],
            external_knowledge="",
        )
        workflow.context = MagicMock()
        return workflow

    @pytest.fixture
    def mock_agent_config(self):
        config = MagicMock()
        return config

    @pytest.fixture
    def schema_node(self, mock_agent_config):
        node = SchemaDiscoveryNode(
            node_id="schema_node",
            description="Test schema discovery",
            node_type="schema_discovery",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )
        node.model = MagicMock()
        return node

    @pytest.mark.asyncio
    async def test_llm_based_discovery_called(self, schema_node, mock_workflow):
        """Verify that LLM-based discovery is called and integrated."""
        schema_node.workflow = mock_workflow

        # Mock methods
        schema_node._semantic_table_discovery = AsyncMock(return_value=[])
        schema_node._keyword_table_discovery = MagicMock(return_value=["users"])  # Stage 1 finds 1 table
        schema_node._context_based_discovery = AsyncMock(return_value=["orders"])  # Stage 2 finds another
        schema_node._load_table_schemas = AsyncMock()

        # Mock LLM response
        schema_node.model.generate_with_json_output.return_value = {"tables": ["test_drive"]}

        # Run
        actions = []
        async for action in schema_node.run():
            actions.append(action)

        # Verify LLM was called with correct prompt structure
        args, _ = schema_node.model.generate_with_json_output.call_args
        assert "Translate business concepts into English database terms" in args[0]

        # Verify results merged
        # We expect: "users" (keyword), "test_drive" (LLM), "orders" (Context - because < 3 tables found initially)
        # Total 3 tables.
        result_tables = schema_node.result.data["candidate_tables"]
        assert "test_drive" in result_tables
        assert "users" in result_tables
        assert "orders" in result_tables

    @pytest.mark.asyncio
    async def test_stage2_trigger_condition(self, schema_node, mock_workflow):
        """Verify Stage 2 runs when candidate count is low."""
        schema_node.workflow = mock_workflow

        # Scenario: Stage 1 + LLM find only 1 table.
        schema_node._semantic_table_discovery = AsyncMock(return_value=[])
        schema_node._keyword_table_discovery = MagicMock(return_value=["t1"])
        schema_node.model.generate_with_json_output.return_value = {"tables": []}

        schema_node._context_based_discovery = AsyncMock(return_value=["t2"])
        schema_node._load_table_schemas = AsyncMock()

        await schema_node.run().__anext__()  # Run until yield

        # Stage 2 should have been called because 1 < 3
        schema_node._context_based_discovery.assert_called()

    @pytest.mark.asyncio
    async def test_stage2_skip_condition(self, schema_node, mock_workflow):
        """Verify Stage 2 is skipped when candidate count is sufficient."""
        schema_node.workflow = mock_workflow

        # Scenario: Stage 1 + LLM find 5 tables.
        schema_node._semantic_table_discovery = AsyncMock(return_value=[])
        schema_node._keyword_table_discovery = MagicMock(return_value=["t1", "t2", "t3"])
        schema_node.model.generate_with_json_output.return_value = {"tables": ["t4", "t5"]}

        schema_node._context_based_discovery = AsyncMock(return_value=["t6"])  # Should NOT be called
        schema_node._load_table_schemas = AsyncMock()

        async for _ in schema_node.run():
            pass

        # Stage 2 should NOT have been called because 5 >= 3
        schema_node._context_based_discovery.assert_not_called()
