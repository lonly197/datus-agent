import asyncio
import threading
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from datus.agent.cancellation_registry import (
    clear_cancelled,
    clear_cancelled_sync,
    is_cancelled,
    is_cancelled_sync,
    mark_cancelled,
    mark_cancelled_sync,
)
from datus.agent.error_handling import unified_error_handler
from datus.agent.intent_detection import IntentDetector
from datus.utils.exceptions import DatusException, ErrorCode


class TestCodeQualityFixes(unittest.IsolatedAsyncioTestCase):

    def test_cancellation_registry_threading(self):
        """Test thread safety of cancellation registry."""
        task_id = "test_task_thread"
        
        def worker():
            mark_cancelled_sync(task_id)
            
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        
        self.assertTrue(is_cancelled_sync(task_id))
        clear_cancelled_sync(task_id)
        self.assertFalse(is_cancelled_sync(task_id))

    async def test_cancellation_registry_async(self):
        """Test async/sync interoperability."""
        task_id = "test_task_async"
        await mark_cancelled(task_id)
        self.assertTrue(is_cancelled_sync(task_id))
        self.assertTrue(await is_cancelled(task_id))
        
        await clear_cancelled(task_id)
        self.assertFalse(await is_cancelled(task_id))

    def test_error_handling_decorator(self):
        """Test unified error handling decorator."""
        
        class MockNode:
            id = "mock_node"
            type = "MockNode"
            
            @unified_error_handler("MockNode", "test_op")
            def failing_method(self):
                raise ValueError("Test error")
                
            @unified_error_handler("MockNode", "test_op")
            def datus_failing_method(self):
                raise DatusException(ErrorCode.NODE_EXECUTION_FAILED, "Datus error")

        node = MockNode()
        result = node.failing_method()
        self.assertFalse(result.success)
        self.assertIn("Test error", result.error)
        
        result_datus = node.datus_failing_method()
        self.assertFalse(result_datus.success)
        self.assertIn("Datus error", result_datus.error)

    def test_intent_detection_regex_safety(self):
        """Test intent detection with long string."""
        detector = IntentDetector()
        long_text = "SELECT " + "a" * 10005 + " FROM table"
        # Should not crash or hang
        is_sql, _ = detector.detect_sql_intent_by_keyword(long_text)
        # It truncates at 10000, so " FROM table" might be lost if it's at the end
        # "SELECT " is at start. 
        # The logic: if len > 10000, truncate.
        # "SELECT " + 10005 chars -> truncated.
        # But let's check if it handles it gracefully.
        self.assertIsInstance(is_sql, bool)

if __name__ == "__main__":
    unittest.main()
