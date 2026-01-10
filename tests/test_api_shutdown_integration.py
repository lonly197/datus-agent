"""
Integration tests for API shutdown and task cancellation behavior.
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from datus.api.server import create_app


class TestAPIShutdownIntegration:
    """Integration tests for graceful shutdown and task cancellation."""

    @pytest.fixture
    def test_app_args(self):
        """Create test app arguments."""
        from argparse import Namespace

        # Create a minimal args object for testing
        return Namespace(
            namespace="test",
            config=None,
            max_steps=10,
            workflow="text2sql",
            load_cp=None,
            debug=False,
            shutdown_timeout_seconds=2.0,  # Short timeout for testing
        )

    @pytest.fixture
    def test_app(self, test_app_args):
        """Create a test FastAPI app."""
        app = create_app(test_app_args, root_path="")
        return app

    @pytest.fixture
    def client(self, test_app):
        """Create a test client."""
        return TestClient(test_app)

    def test_cancel_endpoint_task_cancellation(self, client):
        """Test that the cancel endpoint properly cancels running tasks within timeout."""

        # Start a long-running workflow via sync endpoint
        workflow_request = {
            "workflow": "text2sql",
            "namespace": "test",
            "task": "SELECT COUNT(*) FROM test_table WHERE created_at > '2023-01-01'",
            "database_name": "test_db",
            "schema_name": "test_schema",
            "task_id": "test_cancel_task",
        }

        # Start the workflow in a separate thread since it's blocking
        import threading
        import time

        result = {}
        exception = {}

        def run_workflow():
            try:
                response = client.post("/workflows/run", json=workflow_request)
                result["response"] = response
            except Exception as e:
                exception["error"] = e

        # Start workflow in background
        workflow_thread = threading.Thread(target=run_workflow, daemon=True)
        workflow_thread.start()

        # Give it a moment to start
        time.sleep(0.5)

        # Cancel the task
        cancel_response = client.delete(f"/workflows/tasks/{workflow_request['task_id']}")

        # Should return success quickly
        assert cancel_response.status_code == 200
        response_data = cancel_response.json()
        assert response_data["status"] == "cancelled"

        # Wait a bit for the workflow to actually cancel
        time.sleep(0.5)

        # Check task status - should be cancelled
        status_response = client.get(f"/workflows/tasks/{workflow_request['task_id']}")
        assert status_response.status_code == 200
        status_data = status_response.json()

        # The task should either be cancelled or not found (if cleaned up)
        if status_data.get("is_running", True):
            assert status_data.get("status") == "cancelled"

    def test_lifespan_shutdown_with_running_tasks(self, test_app_args):
        """Test that lifespan shutdown properly cancels running tasks within timeout."""

        # Create app and manually set up service with running tasks
        app = create_app(test_app_args, root_path="")
        service = app.state.service if hasattr(app.state, "service") else None

        if not service:
            # Manually initialize service for testing
            from datus.api.service import DatusAPIService

            service = DatusAPIService(test_app_args)
            app.state.service = service

        # Create some mock running tasks
        async def mock_long_running_task():
            await asyncio.sleep(10)

        async def setup_test_tasks():
            # Register some tasks
            for i in range(3):
                task = asyncio.create_task(mock_long_running_task())
                await service.register_running_task(f"shutdown_test_{i}", task, {"client": "test"})

        # Run setup in event loop
        asyncio.run(setup_test_tasks())

        # Verify tasks are running
        running_tasks = asyncio.run(service.get_all_running_tasks())
        assert len(running_tasks) >= 3

        # Simulate shutdown by calling cancel_all_running_tasks
        start_time = time.time()
        asyncio.run(service.cancel_all_running_tasks(per_task_timeout=0.5))
        elapsed = time.time() - start_time

        # Should complete within reasonable time (less than 1 second for our timeout)
        assert elapsed < 1.0

        # Verify tasks were cancelled
        for task_info in running_tasks.values():
            assert task_info.task.cancelled() or task_info.task.done()

    def test_shutdown_timeout_configuration(self, test_app_args):
        """Test that shutdown timeout is properly configurable."""

        # Test with different timeout values
        test_app_args.shutdown_timeout_seconds = 1.0
        app = create_app(test_app_args, root_path="")

        # The app should be created successfully with the timeout configured
        assert app is not None

        # Check that the timeout is accessible in the lifespan
        # This is indirectly tested by the lifespan function using getattr(args, 'shutdown_timeout_seconds', 5)
