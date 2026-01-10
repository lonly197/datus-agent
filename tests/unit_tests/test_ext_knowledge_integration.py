from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.schema_discovery_node import SchemaDiscoveryNode
from datus.agent.node.schema_validation_node import SchemaValidationNode
from datus.schemas.node_models import BaseInput, SqlTask, TableSchema
from datus.tools.db_tools.db_manager import get_db_manager


class TestExtKnowledgeIntegration:
    @pytest.fixture
    def mock_agent_config(self):
        return MagicMock()

    @pytest.fixture
    def mock_storage_cache(self):
        cache = MagicMock()
        ext_knowledge = MagicMock()
        cache.ext_knowledge_storage.return_value = ext_knowledge
        return cache

    @pytest.fixture
    def schema_node(self, mock_agent_config):
        node = SchemaDiscoveryNode(
            node_id="schema_node",
            description="Test schema discovery",
            node_type="schema_discovery",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )
        return node

    def test_keyword_discovery_with_ext_knowledge(self, schema_node, mock_storage_cache):
        """Verify keyword discovery queries ExtKnowledgeStore."""
        # Setup mocks - ensure the ext_knowledge mock is properly configured
        ext_knowledge = mock_storage_cache.ext_knowledge_storage()

        # Mock KB results - properly configure the return value
        ext_knowledge.search_knowledge.return_value = [
            {"explanation": "This refers to the user_orders table.", "terminology": "query"}
        ]

        # Mock helper to extract table names
        schema_node._extract_potential_tables_from_text = MagicMock(return_value=["user_orders"])

        # Patch at module level to avoid real cache instantiation
        with patch("datus.storage.cache.get_storage_cache_instance") as mock_get_cache:
            mock_get_cache.return_value = mock_storage_cache

            # Run
            tables = schema_node._keyword_table_discovery("query")

            # Verify search_knowledge was called with correct parameters
            ext_knowledge.search_knowledge.assert_called_once_with(query_text="query", top_n=5)
            assert "user_orders" in tables

    @pytest.mark.asyncio
    async def test_metadata_repair(self, schema_node, mock_storage_cache):
        """Verify metadata repair logic attempts to fetch DDL."""
        # Setup mocks
        mock_schema_storage = MagicMock()
        mock_storage_cache.schema_storage.return_value = mock_schema_storage

        # Mock DB connector directly
        mock_connector = MagicMock()
        mock_connector.get_table_ddl.return_value = "CREATE TABLE t1..."

        # Mock DB manager to return our connector
        mock_db_manager = MagicMock()
        mock_db_manager.get_conn.return_value = mock_connector

        # We need to mock get_db_manager BEFORE it's called in _repair_metadata
        # The issue is that get_db_manager() creates a real instance that tries to initialize
        # So we'll mock the DBManager class itself
        with (
            patch("datus.storage.cache.get_storage_cache_instance") as mock_get_cache,
            patch("datus.tools.db_tools.db_manager.DBManager") as MockDBManager,
        ):
            # Create a mock DBManager instance
            mock_instance = MagicMock()
            mock_instance.get_conn = MagicMock(return_value=mock_connector)
            MockDBManager.return_value = mock_instance

            mock_get_cache.return_value = mock_storage_cache
            schema_node.agent_config.current_namespace = "test"

            task = SqlTask(id="1", task="test", database_name="db")

            # Verify _repair_metadata directly
            count = await schema_node._repair_metadata(["t1"], mock_schema_storage, task)

            assert count == 1
            mock_connector.get_table_ddl.assert_called_with("t1")
            mock_schema_storage.update_table_schema.assert_called()

    def test_extract_potential_tables_from_text(self, schema_node):
        """Verify table name extraction from explanation text."""
        # Test with snake_case table names
        text1 = "This data comes from user_orders table and order_items table."
        tables1 = schema_node._extract_potential_tables_from_text(text1)
        assert "user_orders" in tables1
        assert "order_items" in tables1

        # Test with no table names
        text2 = "This is just plain text with no table names."
        tables2 = schema_node._extract_potential_tables_from_text(text2)
        assert len(tables2) == 0

        # Test with complex table names
        text3 = "Query dwd_assign_dlr_clue_fact_di for test drive data."
        tables3 = schema_node._extract_potential_tables_from_text(text3)
        assert "dwd_assign_dlr_clue_fact_di" in tables3


class TestSchemaValidationKnowledgeIntegration:
    """Test ExtKnowledgeStore integration in SchemaValidationNode."""

    @pytest.fixture
    def mock_agent_config(self):
        return MagicMock()

    @pytest.fixture
    def mock_storage_cache(self):
        cache = MagicMock()
        ext_knowledge = MagicMock()
        cache.ext_knowledge_storage.return_value = ext_knowledge
        return cache

    @pytest.fixture
    def validation_node(self, mock_agent_config):
        node = SchemaValidationNode(
            node_id="validation_node",
            description="Test schema validation",
            node_type="schema_validation",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )
        return node

    def test_check_schema_coverage_with_ext_knowledge(self, validation_node, mock_storage_cache):
        """Verify schema validation uses ExtKnowledgeStore for term mapping."""
        # Setup mock schemas
        schemas = [
            TableSchema(
                table_name="test_drive",
                database_name="db",
                definition="CREATE TABLE test_drive (id INT, dealer_code VARCHAR)",
            )
        ]

        # Setup ExtKnowledgeStore mock
        ext_knowledge = mock_storage_cache.ext_knowledge_storage()
        ext_knowledge.search_knowledge.return_value = [
            {"terminology": "试驾", "explanation": "This refers to test_drive table with dealer_code column"}
        ]

        with patch("datus.agent.node.schema_validation_node.get_storage_cache_instance") as mock_get_cache:
            mock_get_cache.return_value = mock_storage_cache

            # Test with Chinese term that should map via ExtKnowledgeStore
            result = validation_node._check_schema_coverage(schemas, ["试驾", "test"])

            # Verify ExtKnowledgeStore was queried
            ext_knowledge.search_knowledge.assert_called()

            # Verify coverage - "试驾" should be covered via knowledge base
            assert result["coverage_score"] > 0
            assert "试驾" in result["covered_terms"]

    def test_check_schema_coverage_fallback_to_config(self, validation_node, mock_storage_cache):
        """Verify fallback to hardcoded config when ExtKnowledgeStore fails."""
        schemas = [TableSchema(table_name="orders", database_name="db", definition="CREATE TABLE orders (id INT)")]

        # Setup ExtKnowledgeStore to raise exception
        ext_knowledge = mock_storage_cache.ext_knowledge_storage()
        ext_knowledge.search_knowledge.side_effect = Exception("KB unavailable")

        with patch("datus.agent.node.schema_validation_node.get_storage_cache_instance") as mock_get_cache:
            mock_get_cache.return_value = mock_storage_cache

            # Should fallback to hardcoded config and still work
            result = validation_node._check_schema_coverage(schemas, ["订单"])

            # Should have some coverage from hardcoded mappings
            assert "coverage_score" in result
