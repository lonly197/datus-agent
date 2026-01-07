import asyncio
from typing import TypeVar, Optional, Awaitable
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
