from unittest.mock import MagicMock, patch

import pytest

from datus.cli.plan_hooks import QueryCache, ToolBatchProcessor


class TestQueryCache:
    """Test QueryCache functionality."""

    @pytest.fixture
    def cache(self):
        """Create a QueryCache instance."""
        return QueryCache(max_size=10, ttl_seconds=300)

    def test_cache_key_generation(self, cache):
        """Test cache key generation."""
        key1 = cache._get_cache_key("describe_table", table_name="users", database="test")
        key2 = cache._get_cache_key("describe_table", table_name="users", database="test")
        key3 = cache._get_cache_key("describe_table", table_name="orders", database="test")

        assert key1 == key2
        assert key1 != key3

    def test_cache_set_and_get(self, cache):
        """Test basic cache set and get operations."""
        # Set a value
        cache.set("describe_table", {"success": True, "columns": []}, table_name="users")

        # Get the value
        result = cache.get("describe_table", table_name="users")

        assert result is not None
        assert result["success"] is True

    def test_cache_expiry(self, cache):
        """Test cache expiry."""
        # Set a value with very short TTL
        expired_cache = QueryCache(max_size=10, ttl_seconds=0.001)
        expired_cache.set("describe_table", {"success": True}, table_name="users")

        import time

        time.sleep(0.01)  # Wait for expiry

        result = expired_cache.get("describe_table", table_name="users")
        assert result is None

    def test_cache_max_size(self, cache):
        """Test cache size limits."""
        # Fill cache beyond max size
        for i in range(15):  # max_size is 10
            cache.set("describe_table", {"id": i}, table_name=f"table_{i}")

        # Should only keep the most recent entries
        assert len(cache.cache) <= 10

    def test_cache_excludes_non_deterministic_params(self, cache):
        """Test that non-deterministic parameters are excluded from cache key."""
        key1 = cache._get_cache_key("read_query", sql="SELECT * FROM users", todo_id="123")
        key2 = cache._get_cache_key("read_query", sql="SELECT * FROM users", todo_id="456")

        # Keys should be the same despite different todo_id
        assert key1 == key2


class TestToolBatchProcessor:
    """Test ToolBatchProcessor functionality."""

    @pytest.fixture
    def batch_processor(self):
        """Create a ToolBatchProcessor instance."""
        return ToolBatchProcessor()

    def test_add_to_batch(self, batch_processor):
        """Test adding items to batch."""
        todo_item = MagicMock()
        params = {"table_name": "users"}

        batch_processor.add_to_batch("describe_table", todo_item, params)

        assert "describe_table" in batch_processor.batches
        assert len(batch_processor.batches["describe_table"]) == 1

    def test_get_batch_size(self, batch_processor):
        """Test getting batch size."""
        assert batch_processor.get_batch_size("describe_table") == 0

        todo_item = MagicMock()
        batch_processor.add_to_batch("describe_table", todo_item, {"table": "users"})

        assert batch_processor.get_batch_size("describe_table") == 1

    def test_clear_batch(self, batch_processor):
        """Test clearing batch."""
        todo_item = MagicMock()
        batch_processor.add_to_batch("describe_table", todo_item, {"table": "users"})

        batch = batch_processor.clear_batch("describe_table")

        assert len(batch) == 1
        assert batch_processor.get_batch_size("describe_table") == 0

    def test_clear_empty_batch(self, batch_processor):
        """Test clearing non-existent batch."""
        batch = batch_processor.clear_batch("non_existent_tool")

        assert batch == []

    def test_optimize_search_table_batch(self, batch_processor):
        """Test search_table batch optimization."""
        # Create mock todo items and params
        todo1 = MagicMock()
        params1 = {"query_text": "find user information"}

        todo2 = MagicMock()
        params2 = {"query_text": "find user data and details"}

        todo3 = MagicMock()
        params3 = {"query_text": "find order information"}

        batch = [(todo1, params1), (todo2, params2), (todo3, params3)]

        optimized = batch_processor.optimize_search_table_batch(batch)

        # Should group similar queries
        assert len(optimized) == 3  # All queries are different enough

    def test_optimize_search_table_batch_empty(self, batch_processor):
        """Test optimizing empty batch."""
        optimized = batch_processor.optimize_search_table_batch([])

        assert optimized == []


class TestPreflightCacheIntegration:
    """Test preflight tool execution with cache integration."""

    @pytest.fixture
    def mock_plan_hooks(self):
        """Create mock plan hooks with cache."""
        hooks = MagicMock()
        hooks.enable_query_caching = True
        hooks.query_cache = QueryCache(max_size=10, ttl_seconds=300)
        hooks.monitor = MagicMock()
        return hooks

    @pytest.mark.asyncio
    async def test_preflight_tool_caching(self, mock_plan_hooks):
        """Test that preflight tools use cache correctly."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        with patch("datus.agent.node.chat_agentic_node.GenSQLAgenticNode.__init__", return_value=None):
            node = ChatAgenticNode("test", "test", "chat")
            node.plan_hooks = mock_plan_hooks
            node.db_func_tool = MagicMock()

            # Mock successful describe_table result
            mock_result = MagicMock()
            mock_result.__dict__ = {"success": True, "table_name": "users", "columns": []}
            node.db_func_tool.describe_table = MagicMock(return_value=mock_result)

            # First call should execute and cache
            result1 = await node._execute_preflight_tool(
                "describe_table", "SELECT * FROM users", ["users"], "", "test", ""
            )

            assert result1["success"] is True
            assert node.db_func_tool.describe_table.call_count == 1

            # Second call should use cache
            result2 = await node._execute_preflight_tool(
                "describe_table", "SELECT * FROM users", ["users"], "", "test", ""
            )

            assert result2["success"] is True
            # Should still be 1 call since second was cached
            assert node.db_func_tool.describe_table.call_count == 1

    @pytest.mark.asyncio
    async def test_preflight_tool_cache_disabled(self):
        """Test preflight tools when cache is disabled."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        with patch("datus.agent.node.chat_agentic_node.GenSQLAgenticNode.__init__", return_value=None):
            node = ChatAgenticNode("test", "test", "chat")
            node.plan_hooks = None  # No plan hooks means no caching
            node.db_func_tool = MagicMock()

            # Mock successful describe_table result
            mock_result = MagicMock()
            mock_result.__dict__ = {"success": True, "table_name": "users", "columns": []}
            node.db_func_tool.describe_table = MagicMock(return_value=mock_result)

            # Both calls should execute (no caching)
            await node._execute_preflight_tool("describe_table", "SELECT * FROM users", ["users"], "", "test", "")

            await node._execute_preflight_tool("describe_table", "SELECT * FROM users", ["users"], "", "test", "")

            assert node.db_func_tool.describe_table.call_count == 2
