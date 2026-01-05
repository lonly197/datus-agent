#!/usr/bin/env python3
"""
Simple test script to validate task_id functionality without full imports.
"""
import asyncio
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Mock the problematic imports to focus on our logic
class MockAgent:
    pass


class MockWorkflowRunner:
    pass


# Monkey patch to avoid import issues
import datus.agent

datus.agent.Agent = MockAgent
datus.agent.WorkflowRunner = MockWorkflowRunner

# Mock the problematic node imports
import datus.agent.node

datus.agent.node.Node = type("MockNode", (), {})

from unittest.mock import MagicMock

from fastapi import HTTPException

# Now import our service
from datus.api.service import DatusAPIService


async def test_task_id_functionality():
    """Test our task_id functionality."""
    print("Testing task_id functionality...")

    # Create service instance
    args = MagicMock()
    service = DatusAPIService(args)

    # Test 1: Task ID generation
    print("\n1. Testing task ID generation:")
    task_id = service._generate_task_id("test_client", "research")
    print(f"   Generated task_id: {task_id}")
    assert task_id.startswith("research_test_client_"), f"Expected prefix, got: {task_id}"
    print("   ✓ Task ID generation works")

    task_id_no_prefix = service._generate_task_id("test_client")
    print(f"   Generated task_id (no prefix): {task_id_no_prefix}")
    assert task_id_no_prefix.startswith("test_client_"), f"Expected client prefix, got: {task_id_no_prefix}"
    print("   ✓ Task ID generation without prefix works")

    # Test 2: New task_id validation should pass
    print("\n2. Testing new task_id validation:")
    try:
        await service._validate_task_id_uniqueness("new_task_123", "client1")
        print("   ✓ New task_id validation passed (as expected)")
    except Exception as e:
        print(f"   ✗ New task_id validation failed unexpectedly: {e}")
        return False

    # Test 3: Duplicate task_id validation should fail
    print("\n3. Testing duplicate task_id validation:")
    task = asyncio.create_task(asyncio.sleep(1))
    await service.register_running_task("duplicate_task", task, {"client": "client1"})

    try:
        await service._validate_task_id_uniqueness("duplicate_task", "client1")
        print("   ✗ Duplicate task_id validation should have failed")
        task.cancel()
        await service.unregister_running_task("duplicate_task")
        return False
    except HTTPException as e:
        if e.status_code == 409 and "already exists and is running" in str(e.detail):
            print("   ✓ Duplicate task_id validation correctly failed with 409 status")
        else:
            print(f"   ✗ Duplicate task_id validation failed with wrong error: {e.status_code} - {e.detail}")
            task.cancel()
            await service.unregister_running_task("duplicate_task")
            return False
    except Exception as e:
        print(f"   ✗ Duplicate task_id validation failed with unexpected error: {e}")
        task.cancel()
        await service.unregister_running_task("duplicate_task")
        return False

    # Clean up
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await service.unregister_running_task("duplicate_task")

    # Test 4: Different client with same task_id should fail
    print("\n4. Testing different client with same task_id:")
    task2 = asyncio.create_task(asyncio.sleep(1))
    await service.register_running_task("shared_task", task2, {"client": "client1"})

    try:
        await service._validate_task_id_uniqueness("shared_task", "client2")
        print("   ✗ Different client task_id validation should have failed")
        task2.cancel()
        await service.unregister_running_task("shared_task")
        return False
    except HTTPException as e:
        if e.status_code == 409 and "already exists for different client" in str(e.detail):
            print("   ✓ Different client task_id validation correctly failed")
        else:
            print(f"   ✗ Different client task_id validation failed with wrong error: {e}")
            task2.cancel()
            await service.unregister_running_task("shared_task")
            return False
    except Exception as e:
        print(f"   ✗ Different client task_id validation failed with unexpected error: {e}")
        task2.cancel()
        await service.unregister_running_task("shared_task")
        return False

    # Clean up
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        pass
    await service.unregister_running_task("shared_task")

    print("\n✅ All task_id functionality tests passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_task_id_functionality())
    if success:
        print("\n🎉 Implementation is working correctly!")
    else:
        print("\n❌ Implementation has issues!")
        sys.exit(1)
