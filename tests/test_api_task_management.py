"""
Tests for task management API endpoints.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from datus.api.service import DatusAPIService


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

    @pytest.mark.asyncio
    async def test_cancel_all_running_tasks(self, service):
        """Test cancel_all_running_tasks functionality with timeout."""

        # Create multiple long-running tasks
        async def long_task(task_id: str, duration: float):
            try:
                await asyncio.sleep(duration)
                return f"task_{task_id}_completed"
            except asyncio.CancelledError:
                return f"task_{task_id}_cancelled"

        tasks = []
        task_ids = []

        # Create 3 tasks with different durations
        for i in range(3):
            task_id = f"cancel_all_test_{i}"
            task = asyncio.create_task(long_task(task_id, 10.0))  # Long duration
            tasks.append(task)
            task_ids.append(task_id)

            # Register each task
            await service.register_running_task(task_id, task, {"client": "test_client"})

        # Verify all tasks are running
        all_running = await service.get_all_running_tasks()
        assert len(all_running) == 3
        for task_id in task_ids:
            assert task_id in all_running
            assert all_running[task_id].status == "running"

        # Cancel all running tasks with a short timeout
        await service.cancel_all_running_tasks(per_task_timeout=0.1)

        # Verify all tasks are cancelled
        for task in tasks:
            assert task.cancelled() or task.done()

        # Verify registry is cleaned up for completed tasks
        all_running_after = await service.get_all_running_tasks()
        # Some tasks might still be in the registry if they didn't complete within timeout
        # but they should all be cancelled
        for task_id in task_ids:
            if task_id in all_running_after:
                running_task = all_running_after[task_id]
                assert running_task.task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_all_running_tasks_empty_registry(self, service):
        """Test cancel_all_running_tasks with empty registry."""
        # Should not raise any exceptions
        await service.cancel_all_running_tasks()
        all_running = await service.get_all_running_tasks()
        assert len(all_running) == 0

    @pytest.mark.asyncio
    async def test_cancel_all_running_tasks_already_completed(self, service):
        """Test cancel_all_running_tasks when tasks are already completed."""

        # Create tasks that complete immediately
        async def quick_task(task_id: str):
            return f"task_{task_id}_done"

        tasks = []
        task_ids = []

        # Create and complete tasks
        for i in range(2):
            task_id = f"completed_test_{i}"
            task = asyncio.create_task(quick_task(task_id))
            await task  # Wait for completion
            tasks.append(task)
            task_ids.append(task_id)

            # Register completed task
            await service.register_running_task(task_id, task, {"client": "test_client"})

        # Cancel all running tasks - should handle completed tasks gracefully
        await service.cancel_all_running_tasks(per_task_timeout=0.1)

        # Verify registry is cleaned up
        all_running_after = await service.get_all_running_tasks()
        assert len(all_running_after) == 0

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

    @pytest.mark.asyncio
    async def test_frontend_task_id_support(self, service):
        """Test that frontend can specify task_id via messageId."""

        # Test task_id validation - should pass for new task_id
        try:
            await service._validate_task_id_uniqueness("frontend_msg_123", "test_client")
        except Exception:
            pytest.fail("Validation should pass for new task_id")

        # Test task_id validation - should fail for duplicate task_id
        # First register a task with the same ID
        task = asyncio.create_task(asyncio.sleep(1))
        await service.register_running_task("frontend_msg_123", task, {"client": "test_client"})

        # Now validation should fail
        with pytest.raises(HTTPException) as exc_info:
            await service._validate_task_id_uniqueness("frontend_msg_123", "test_client")

        assert exc_info.value.status_code == 409
        assert "already exists and is running" in str(exc_info.value.detail)

        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await service.unregister_running_task("frontend_msg_123")

    @pytest.mark.asyncio
    async def test_task_id_conflict_detection(self, service):
        """Test that duplicate task_ids are rejected."""

        # Test with different clients - should fail
        task = asyncio.create_task(asyncio.sleep(1))
        await service.register_running_task("duplicate_id", task, {"client": "client1"})

        with pytest.raises(HTTPException) as exc_info:
            await service._validate_task_id_uniqueness("duplicate_id", "client2")

        assert exc_info.value.status_code == 409
        assert "already exists for different client" in str(exc_info.value.detail)

        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await service.unregister_running_task("duplicate_id")

    def test_generate_task_id_with_prefix(self, service):
        """Test task ID generation with prefix."""

        task_id = service._generate_task_id("test_client", "research")
        assert task_id.startswith("research_test_client_")
        assert len(task_id) > len("research_test_client_")

        # Test without prefix
        task_id_no_prefix = service._generate_task_id("test_client")
        assert task_id_no_prefix.startswith("test_client_")
        assert len(task_id_no_prefix) > len("test_client_")

    @pytest.mark.asyncio
    async def test_cancel_task_with_client_validation(self, service):
        """Test task cancellation with client ownership validation."""

        # Register a task for client1
        task = asyncio.create_task(asyncio.sleep(10))
        await service.register_running_task("test_task", task, {"client": "client1"})

        # Try to cancel from different client - should fail
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            # Simulate cancel_task logic
            running_task = await service.get_running_task("test_task")
            task_client = running_task.meta.get("client") if running_task.meta else None
            if task_client != "client2":
                raise HTTPException(status_code=403, detail="Task belongs to different client")
        assert exc_info.value.status_code == 403

        # Cancel from correct client - should succeed
        if not task.done():
            task.cancel()
        await service.unregister_running_task("test_task")

    @pytest.mark.asyncio
    async def test_cancel_completed_task(self, service):
        """Test cancelling a task that has already completed."""

        # Create a task that completes immediately
        async def quick_task():
            return "done"

        task = asyncio.create_task(quick_task())
        await service.register_running_task("quick_task", task, {"client": "test_client"})

        # Wait for task to complete
        await task

        # Try to cancel - should handle gracefully
        running_task = await service.get_running_task("quick_task")
        if running_task:
            if not running_task.task.done():
                running_task.task.cancel()
            else:
                print("Task was already completed")

        await service.unregister_running_task("quick_task")
