import asyncio
from typing import Awaitable, Optional, TypeVar

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def ensure_not_cancelled():
    """
    Check if the current task has been cancelled.
    If so, raise asyncio.CancelledError.
    """
    current_task = asyncio.current_task()
    if current_task and current_task.cancelled():
        raise asyncio.CancelledError()


async def await_cancellable(coro: Awaitable[T], timeout: Optional[float] = None) -> T:
    """
    Await a coroutine, allowing it to be cancelled and optionally timing out.
    Does NOT shield the coroutine from cancellation.

    Args:
        coro: The coroutine to await.
        timeout: Optional timeout in seconds.

    Returns:
        The result of the coroutine.

    Raises:
        asyncio.CancelledError: If the task is cancelled.
        asyncio.TimeoutError: If the timeout is reached.
    """
    # ensure_not_cancelled() # Check before starting

    if timeout:
        return await asyncio.wait_for(coro, timeout=timeout)
    else:
        return await coro


def run_async(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine synchronously.

    This is useful for calling async functions from sync contexts (e.g., CLI commands,
    sync tools). It handles event loop management automatically.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If we are already in an event loop, we cannot block it.
        # We run the coroutine in a separate thread with its own event loop.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        # No running loop, just run it
        return asyncio.run(coro)
