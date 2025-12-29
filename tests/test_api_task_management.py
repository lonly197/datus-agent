"""
Tests for task management API endpoints.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.api.models import ChatResearchRequest, RunWorkflowRequest
from datus.api.service import DatusAPIService, RunningTask


class TestTaskManagement:
    """Test task management functionality."""

    @pytest.fixture
    def service(self):
        """Create a test service instance."""
        args = MagicMock()
        service = DatusAPIService(args)
        service.task_store = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_running_tasks_registry(self, service):
        """Test running tasks registry operations."""
        # Test registering a task
        task = asyncio.create_task(asyncio.sleep(1))
        meta = {"type": "test", "client": "test_client"}

        await service.register_running_task("test_task", task, meta)

        # Test getting the task
        running_task = await service.get_running_task("test_task")
        assert running_task is not None
        assert running_task.task_id == "test_task"
        assert running_task.status == "running"
        assert running_task.meta == meta

        # Test getting all tasks
        all_tasks = await service.get_all_running_tasks()
        assert "test_task" in all_tasks

        # Test unregistering
        await service.unregister_running_task("test_task")
        running_task = await service.get_running_task("test_task")
        assert running_task is None

        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_task_cancellation(self, service):
        """Test task cancellation functionality."""

        # Create a long-running task
        async def long_running():
            await asyncio.sleep(10)

        task = asyncio.create_task(long_running())
        await service.register_running_task("cancel_test", task)

        # Verify task is running
        running_task = await service.get_running_task("cancel_test")
        assert running_task.status == "running"
        assert not task.cancelled()
        assert not task.done()

        # Cancel the task
        task.cancel()

        # Wait for cancellation to complete
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify task was cancelled and is done
        assert task.cancelled()
        assert task.done()

        # Verify the task is still registered but marked as cancelled
        # (Note: the service doesn't automatically update status on cancellation)
        running_task_after = await service.get_running_task("cancel_test")
        assert running_task_after is not None
        assert running_task_after.task is task

        # Clean up registry
        await service.unregister_running_task("cancel_test")

        # Verify task is no longer registered
        running_task_final = await service.get_running_task("cancel_test")
        assert running_task_final is None

    def test_running_task_dataclass(self):
        """Test RunningTask dataclass."""
        from datetime import datetime

        from datus.api.service import RunningTask

        task = asyncio.create_task(asyncio.sleep(1))
        created_at = datetime.now()
        meta = {"test": "data"}

        running_task = RunningTask(task_id="test", task=task, created_at=created_at, status="running", meta=meta)

        assert running_task.task_id == "test"
        assert running_task.status == "running"
        assert running_task.created_at == created_at
        assert running_task.meta == meta

        # Clean up
        task.cancel()
        try:
            task.result()
        except asyncio.CancelledError:
            pass
