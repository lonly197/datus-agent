# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Task cancellation registry for cooperative cancellation across async boundaries.

This module provides a simple registry for tracking task cancellation status
that can be accessed from different parts of the application without tight coupling.
"""

import threading
from typing import Dict

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Module-level registry for task cancellation status
_cancellation_registry: Dict[str, bool] = {}
# Use threading.Lock for thread safety across both sync and async contexts
_registry_lock = threading.Lock()


async def is_cancelled(task_id: str) -> bool:
    """
    Check if a task has been cancelled.

    Args:
        task_id: The task ID to check

    Returns:
        True if the task has been cancelled, False otherwise
    """
    with _registry_lock:
        return _cancellation_registry.get(task_id, False)


async def mark_cancelled(task_id: str) -> None:
    """
    Mark a task as cancelled.

    Args:
        task_id: The task ID to mark as cancelled
    """
    with _registry_lock:
        _cancellation_registry[task_id] = True
        logger.info(f"Task {task_id} marked as cancelled in registry")


async def clear_cancelled(task_id: str) -> None:
    """
    Clear the cancellation status for a task.

    Args:
        task_id: The task ID to clear
    """
    with _registry_lock:
        _cancellation_registry.pop(task_id, None)
        logger.debug(f"Task {task_id} cancellation status cleared from registry")


def is_cancelled_sync(task_id: str) -> bool:
    """
    Synchronously check if a task has been cancelled.

    This is a non-async version for use in synchronous contexts.

    Args:
        task_id: The task ID to check

    Returns:
        True if the task has been cancelled, False otherwise
    """
    with _registry_lock:
        return _cancellation_registry.get(task_id, False)


def mark_cancelled_sync(task_id: str) -> None:
    """
    Synchronously mark a task as cancelled.

    This is a non-async version for use in synchronous contexts.

    Args:
        task_id: The task ID to mark as cancelled
    """
    with _registry_lock:
        _cancellation_registry[task_id] = True
        logger.info(f"Task {task_id} marked as cancelled in registry (sync)")


def clear_cancelled_sync(task_id: str) -> None:
    """
    Synchronously clear the cancellation status for a task.

    This is a non-async version for use in synchronous contexts.

    Args:
        task_id: The task ID to clear
    """
    with _registry_lock:
        _cancellation_registry.pop(task_id, None)
        logger.debug(f"Task {task_id} cancellation status cleared from registry (sync)")
