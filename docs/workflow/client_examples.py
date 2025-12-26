# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Client examples for Datus Agent API - Chat Research with Prompt Parameter

This file demonstrates how to use the new `prompt` and `promptMode` parameters
in the `/workflows/chat_research` endpoint to customize AI agent behavior.
"""

import asyncio
import json
import httpx


async def chat_research_with_prompt_append():
    """
    Example: Using prompt with 'append' mode (default behavior)

    This appends the custom prompt to the default system prompt,
    allowing you to add specific instructions without replacing the core behavior.
    """
    async with httpx.AsyncClient() as client:
        # First, authenticate to get access token
        auth_response = await client.post(
            "http://localhost:8000/auth/token",
            json={
                "client_id": "your_client_id",
                "client_secret": "your_client_secret",
                "grant_type": "client_credentials"
            }
        )
        auth_data = auth_response.json()
        access_token = auth_data["access_token"]

        # Now make the chat research request with custom prompt
        async with client.stream(
            "POST",
            "http://localhost:8000/workflows/chat_research",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            },
            json={
                "namespace": "your_database_namespace",
                "task": "Generate SQL for monthly revenue analysis",
                "prompt": "You are a senior data analyst specializing in e-commerce metrics. Always provide detailed explanations of your SQL queries and suggest optimizations for performance.",
                "prompt_mode": "append"  # This is the default, can be omitted
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])  # Remove "data: " prefix
                    print(f"Event: {event_data.get('event')}")
                    if event_data.get('event') == 'chat':
                        print(f"Message: {event_data.get('content')}")


async def chat_research_with_prompt_replace():
    """
    Example: Using prompt with 'replace' mode

    This completely replaces the default system prompt with your custom prompt,
    giving you full control over the AI agent's behavior and instructions.
    """
    async with httpx.AsyncClient() as client:
        # Authentication (same as above)
        auth_response = await client.post(
            "http://localhost:8000/auth/token",
            json={
                "client_id": "your_client_id",
                "client_secret": "your_client_secret",
                "grant_type": "client_credentials"
            }
        )
        auth_data = auth_response.json()
        access_token = auth_data["access_token"]

        # Chat research request with prompt replacement
        async with client.stream(
            "POST",
            "http://localhost:8000/workflows/chat_research",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            },
            json={
                "namespace": "your_database_namespace",
                "task": "Analyze customer retention patterns",
                "prompt": """You are an expert customer analytics specialist with 10+ years of experience.
Your task is to provide comprehensive customer behavior analysis using advanced SQL techniques.

Guidelines:
- Always use window functions for trend analysis
- Include statistical measures (percentiles, standard deviations)
- Provide business insights alongside technical results
- Suggest actionable recommendations based on the data
- Use CTEs for complex multi-step analysis
- Optimize queries for large datasets""",
                "prompt_mode": "replace"
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])
                    print(f"Event: {event_data.get('event')}")
                    if event_data.get('event') == 'chat':
                        print(f"Message: {event_data.get('content')}")


async def chat_research_without_custom_prompt():
    """
    Example: Traditional usage without custom prompt

    This uses the default system behavior without any custom prompt modifications.
    """
    async with httpx.AsyncClient() as client:
        # Authentication
        auth_response = await client.post(
            "http://localhost:8000/auth/token",
            json={
                "client_id": "your_client_id",
                "client_secret": "your_client_secret",
                "grant_type": "client_credentials"
            }
        )
        auth_data = auth_response.json()
        access_token = auth_data["access_token"]

        # Traditional request without prompt parameters
        async with client.stream(
            "POST",
            "http://localhost:8000/workflows/chat_research",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            },
            json={
                "namespace": "your_database_namespace",
                "task": "Show me user registration trends",
                # No prompt or prompt_mode specified - uses defaults
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])
                    print(f"Event: {event_data.get('event')}")
                    if event_data.get('event') == 'chat':
                        print(f"Message: {event_data.get('content')}")


if __name__ == "__main__":
    print("Choose an example to run:")
    print("1. Chat research with prompt append (default)")
    print("2. Chat research with prompt replace")
    print("3. Chat research without custom prompt")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        asyncio.run(chat_research_with_prompt_append())
    elif choice == "2":
        asyncio.run(chat_research_with_prompt_replace())
    elif choice == "3":
        asyncio.run(chat_research_without_custom_prompt())
    else:
        print("Invalid choice. Please run again and choose 1, 2, or 3.")
