import sys
from unittest.mock import MagicMock

# Mock modules that might cause import errors in this environment
sys.modules["pydantic"] = MagicMock()
sys.modules["datus.configuration.agent_config"] = MagicMock()
sys.modules["datus.configuration.node_type"] = MagicMock()

import asyncio

import pytest

from datus.api.service import RunningTask, Service
from datus.utils.async_utils import await_cancellable, ensure_not_cancelled


@pytest.mark.asyncio
async def test_await_cancellable():
    # Test normal completion
    async def fast_task():
        return "done"

    result = await await_cancellable(fast_task(), timeout=1.0)
    assert result == "done"

    # Test timeout
    async def slow_task():
        await asyncio.sleep(2.0)
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        await await_cancellable(slow_task(), timeout=0.1)

    # Test cancellation propagation
    task = asyncio.create_task(await_cancellable(slow_task(), timeout=5.0))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_all_running_tasks():
    service = Service()
    # Mock dependencies normally injected
    service.agent_config = MagicMock()

    # Manually setup running_tasks if not init by default
    # Service.__init__ calls self.running_tasks = {}
    if not hasattr(service, "running_tasks"):
        service.running_tasks = {}
        service.running_tasks_lock = asyncio.Lock()

    # Create dummy tasks
    async def dummy_task(name, delay):
        try:
            await asyncio.sleep(delay)
            return name
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)  # Simulate cleanup
            raise

    t1 = asyncio.create_task(dummy_task("t1", 5.0))
    t2 = asyncio.create_task(dummy_task("t2", 5.0))

    service.register_running_task("1", t1, "t1")
    service.register_running_task("2", t2, "t2")

    assert "1" in service.running_tasks
    assert "2" in service.running_tasks

    # Cancel all
    await service.cancel_all_running_tasks(wait_timeout=1.0)

    assert t1.cancelled() or t1.done()
    assert t2.cancelled() or t2.done()

    # Verify registry cleanup
    assert "1" not in service.running_tasks
    assert "2" not in service.running_tasks


@pytest.mark.asyncio
async def test_ensure_not_cancelled():
    async def check_task():
        ensure_not_cancelled()
        await asyncio.sleep(0.1)
        ensure_not_cancelled()
        return "ok"

    t = asyncio.create_task(check_task())
    await asyncio.sleep(0.05)
    t.cancel()

    with pytest.raises(asyncio.CancelledError):
        await t
