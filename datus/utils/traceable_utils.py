# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Literal, Type, Union

from openai import AsyncOpenAI, OpenAI

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

HAS_LANGSMITH = False
try:
    from langsmith.client import RUN_TYPE_T

    HAS_LANGSMITH = True
except ImportError:
    RUN_TYPE_T = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


def create_openai_client(
    cls: Type[Union[OpenAI, AsyncOpenAI]],
    api_key: str,
    base_url: str,
    default_headers: Union[dict[str, str], None] = None,
    timeout: float = 300.0,  # 5-minute timeout for all requests
) -> Union[OpenAI, AsyncOpenAI]:
    # Create client with timeout settings and disable built-in retries
    # We handle retries at the application level for better control
    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout,  # Set timeout for all requests
        "max_retries": 0,  # Disable built-in retries, we handle retries at app level
    }

    if default_headers:
        client_kwargs["default_headers"] = default_headers

    client = cls(**client_kwargs)

    # Additional check: try to disable retries on the underlying httpx client if possible
    if hasattr(client, '_client'):
        httpx_client = client._client
        if hasattr(httpx_client, '_transport'):
            transport = httpx_client._transport
            if hasattr(transport, 'retries'):
                transport.retries = 0
                logger.debug("Disabled retries on httpx transport")
        elif hasattr(httpx_client, '_pool'):
            # For older httpx versions
            pool = httpx_client._pool
            if hasattr(pool, '_retries'):
                pool._retries = 0
                logger.debug("Disabled retries on httpx pool")

    logger.debug(f"Created OpenAI client with max_retries=0 for {base_url}")
    return client
    if not HAS_LANGSMITH:
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except ImportError:
        logger.warning("langsmith wrapper not available")
        return client


def optional_traceable(name: str = "", run_type: RUN_TYPE_T = "chain"):
    def decorator(func):
        if not HAS_LANGSMITH:
            return func
        try:
            from langsmith import traceable

            # Use provided run_name or fallback to function name
            trace_name = name or getattr(func, "__name__", "agent_operation")

            # Directly apply the traceable decorator to the original function and return it
            return traceable(name=trace_name, run_type=run_type)(func)
        except ImportError:
            # If langsmith is not available, just return the original function
            return func

    return decorator
