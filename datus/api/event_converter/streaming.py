# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Streaming utilities for SSE event conversion.

This module provides functions to convert ActionHistory streams to
Server-Sent Events (SSE) format for real-time event delivery.
"""

import asyncio
from typing import AsyncGenerator

from datus.schemas.action_history import ActionHistory
from datus.utils.loggings import get_logger


async def convert_stream_to_events(
    action_stream: AsyncGenerator[ActionHistory, None],
    converter
) -> AsyncGenerator[str, None]:
    """Convert ActionHistory stream to DeepResearchEvent SSE stream.

    This function takes a stream of ActionHistory objects and converts
    them to Server-Sent Events (SSE) format for real-time delivery to clients.

    Args:
        action_stream: Async generator of ActionHistory objects
        converter: DeepResearchEventConverter instance

    Yields:
        SSE formatted event strings (data: {json}\\n\\n)
    """
    logger = get_logger(__name__)
    seq_num = 0

    try:
        async for action in action_stream:
            seq_num += 1
            events = converter.convert_action_to_event(action, seq_num)

            for event in events:
                # Convert to JSON and yield as SSE data
                event_json = event.model_dump_json()
                yield f"data: {event_json}\n\n"

                # Check for cancellation
                current_task = asyncio.current_task()
                if current_task and current_task.cancelled():
                    logger.info("Event conversion stream cancelled")
                    break

    except asyncio.CancelledError:
        logger.info("Event conversion stream was cancelled")
    except Exception as e:
        logger.error(f"Error in event conversion stream: {e}", exc_info=True)
        raise
