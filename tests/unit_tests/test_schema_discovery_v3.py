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

    @pytest.mark.asyncio
    async def test_fallback_get_all_tables_uses_correct_db_manager(self, schema_node, mock_workflow):
        """Verify that _fallback_get_all_tables uses get_db_manager() not db_manager_instance()."""
        import pyarrow as pa
        from unittest.mock import patch
        from datus.tools.db_tools.db_manager import get_db_manager

        schema_node.workflow = mock_workflow

        # Mock the database connector and get_db_manager
        mock_connector = MagicMock()
        mock_connector.get_all_table_names.return_value = ["table1", "table2", "table3"]

        mock_db_manager = MagicMock()
        mock_db_manager.get_conn.return_value = mock_connector

        with patch("datus.agent.node.schema_discovery_node.get_db_manager", return_value=mock_db_manager):
            result = await schema_node._fallback_get_all_tables(mock_workflow.task)

        # Verify correct function was called and results are returned
        mock_db_manager.get_conn.assert_called_once()
        mock_connector.get_all_table_names.assert_called_once()
        assert result == ["table1", "table2", "table3"]

    @pytest.mark.asyncio
    async def test_load_table_schemas_handles_pyarrow_table_correctly(self, schema_node, mock_workflow, mock_agent_config):
        """Verify that _load_table_schemas correctly converts PyArrow Table to list of dicts."""
        import pyarrow as pa

        schema_node.workflow = mock_workflow

        # Mock the storage cache to return a PyArrow Table (simulating real behavior)
        mock_schema_storage = MagicMock()

        # Create a PyArrow Table with definition column
        arrow_table = pa.table({
            "table_name": ["table1", "table2", "table3"],
            "definition": ["CREATE TABLE table1...", "", None]  # One empty, one None
        })

        mock_schema_storage.get_table_schemas.return_value = arrow_table

        mock_storage_cache = MagicMock()
        mock_storage_cache.schema_storage.return_value = mock_schema_storage

        with patch("datus.agent.node.schema_discovery_node.get_storage_cache_instance", return_value=mock_storage_cache):
            # Mock SchemaWithValueRAG to avoid actual schema loading
            with patch("datus.agent.node.schema_discovery_node.SchemaWithValueRAG") as mock_rag:
                mock_rag.return_value.search_tables.return_value = ([], [])

                # This should not raise an AttributeError
                await schema_node._load_table_schemas(mock_workflow.task, ["table1", "table2", "table3"])

        # Verify get_table_schemas was called
        mock_schema_storage.get_table_schemas.assert_called_once_with(["table1", "table2", "table3"])

    @pytest.mark.asyncio
    async def test_load_table_schemas_identifies_missing_definitions(self, schema_node, mock_workflow, mock_agent_config):
        """Verify that _load_table_schemas correctly identifies tables with missing DDL definitions."""
        import pyarrow as pa

        schema_node.workflow = mock_workflow

        # Mock the storage cache
        mock_schema_storage = MagicMock()

        # Create a PyArrow Table with some missing definitions
        arrow_table = pa.table({
            "table_name": ["table1", "table2", "table3", "table4"],
            "definition": ["CREATE TABLE table1...", "", None, "   "]  # table2, table3, table4 missing
        })

        mock_schema_storage.get_table_schemas.return_value = arrow_table

        mock_storage_cache = MagicMock()
        mock_storage_cache.schema_storage.return_value = mock_schema_storage

        # Mock _repair_metadata to verify it gets called with correct tables
        schema_node._repair_metadata = AsyncMock()

        with patch("datus.agent.node.schema_discovery_node.get_storage_cache_instance", return_value=mock_storage_cache):
            with patch("datus.agent.node.schema_discovery_node.SchemaWithValueRAG") as mock_rag:
                mock_rag.return_value.search_tables.return_value = ([], [])

                await schema_node._load_table_schemas(mock_workflow.task, ["table1", "table2", "table3", "table4"])

        # Verify _repair_metadata was called with tables that have missing definitions
        schema_node._repair_metadata.assert_called_once()
        repair_call_args = schema_node._repair_metadata.call_args[0]
        missing_tables = repair_call_args[0]  # First positional argument

        # Should include table2 (empty string), table3 (None), and table4 (whitespace only)
        # But NOT table1 (has valid definition)
        assert "table2" in missing_tables
        assert "table3" in missing_tables
        assert "table4" in missing_tables
        assert "table1" not in missing_tables
