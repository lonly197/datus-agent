# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Thread-safe context update utilities to prevent race conditions.

This module provides utilities for safely updating workflow context
from multiple nodes without race conditions.
"""

import threading
from typing import Any, Callable, Dict, List, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ContextUpdateLock:
    """
    Reentrant lock for context updates.

    This ensures that context updates are atomic and prevents
    race conditions when multiple nodes try to update the same
    context concurrently.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._local_lock = threading.RLock()
                    cls._instance._update_stack = []
        return cls._instance

    def acquire(self):
        """Acquire the context update lock."""
        return self._local_lock.acquire()

    def release(self):
        """Release the context update lock."""
        return self._local_lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def safe_context_update(
    context: Any,
    update_func: Callable[[], Dict[str, Any]],
    operation_name: str = "context_update",
) -> Dict[str, Any]:
    """
    Safely update context with automatic locking.

    Args:
        context: The workflow context object
        update_func: Function that performs the actual update
        operation_name: Name for logging/debugging

    Returns:
        Result of the update function

    Example:
        def add_tables():
            context.table_schemas.extend(new_schemas)
            return {"added": len(new_schemas)}

        result = safe_context_update(context, add_tables, "schema_addition")
    """
    lock = ContextUpdateLock()

    try:
        with lock:
            logger.debug(f"Acquired context lock for {operation_name}")
            result = update_func()
            logger.debug(f"Context update completed: {operation_name} - {result}")
            return {"success": True, "operation": operation_name, "result": result}
    except Exception as e:
        logger.error(f"Context update failed for {operation_name}: {e}")
        return {"success": False, "operation": operation_name, "error": str(e)}


def atomic_context_merge(
    context: Any,
    updates: Dict[str, Any],
    merge_strategy: str = "update",
) -> Dict[str, Any]:
    """
    Atomically merge updates into context.

    Args:
        context: The workflow context object
        updates: Dictionary of updates to apply
        merge_strategy: How to merge ('update', 'replace', or 'extend')

    Returns:
        Success status and details
    """
    lock = ContextUpdateLock()

    def _merge():
        if merge_strategy == "replace":
            # Direct replacement
            for key, value in updates.items():
                setattr(context, key, value)
        elif merge_strategy == "extend":
            # Extend lists
            for key, value in updates.items():
                if hasattr(context, key):
                    existing = getattr(context, key)
                    if isinstance(existing, list) and isinstance(value, list):
                        existing.extend(value)
                    else:
                        setattr(context, key, value)
                else:
                    setattr(context, key, value)
        else:  # update (default)
            # Update attributes
            for key, value in updates.items():
                if hasattr(context, key):
                    current = getattr(context, key)
                    if isinstance(current, dict) and isinstance(value, dict):
                        current.update(value)
                    elif isinstance(current, list) and isinstance(value, list):
                        current.extend(value)
                    else:
                        setattr(context, key, value)
                else:
                    setattr(context, key, value)

        return {"updated_keys": list(updates.keys())}

    try:
        with lock:
            result = _merge()
            logger.debug(f"Atomic context merge completed: {result}")
            return {"success": True, **result}
    except Exception as e:
        logger.error(f"Atomic context merge failed: {e}")
        return {"success": False, "error": str(e)}


class ContextSnapshot:
    """
    Create and restore snapshots of context state.

    Useful for rolling back changes if an operation fails.
    """

    def __init__(self, context: Any):
        self.context = context
        self._snapshot: Optional[Dict[str, Any]] = None

    def capture(self) -> None:
        """Capture current context state."""
        lock = ContextUpdateLock()
        with lock:
            self._snapshot = {}
            # Capture all public attributes
            for attr in dir(self.context):
                if not attr.startswith('_'):
                    try:
                        value = getattr(self.context, attr)
                        if not callable(value):
                            # Deep copy for mutable types
                            import copy
                            self._snapshot[attr] = copy.deepcopy(value)
                    except Exception:
                        # Skip attributes that can't be copied
                        pass

            logger.debug(f"Context snapshot captured: {len(self._snapshot)} attributes")

    def restore(self) -> bool:
        """Restore context from snapshot."""
        if self._snapshot is None:
            logger.warning("No snapshot to restore")
            return False

        lock = ContextUpdateLock()
        try:
            with lock:
                for attr, value in self._snapshot.items():
                    setattr(self.context, attr, value)

                logger.debug(f"Context snapshot restored: {len(self._snapshot)} attributes")
                return True
        except Exception as e:
            logger.error(f"Failed to restore context snapshot: {e}")
            return False


def with_context_rollback(context: Any) -> Callable:
    """
    Decorator for functions that should rollback on failure.

    Args:
        context: The workflow context object

    Example:
        @with_context_rollback(workflow.context)
        def risky_operation():
            # Do something that might fail
            context.table_schemas.extend(new_schemas)
            if some_condition:
                raise ValueError("Something went wrong")
            # If exception raised, context will be restored
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            snapshot = ContextSnapshot(context)
            snapshot.capture()

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.warning(f"Operation failed, restoring context: {e}")
                snapshot.restore()
                raise

        return wrapper

    return decorator
