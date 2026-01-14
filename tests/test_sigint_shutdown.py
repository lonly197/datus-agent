#!/usr/bin/env python3
"""
Test script to verify Ctrl+C graceful shutdown behavior.

This script simulates a user pressing Ctrl+C while the server is running
and verifies that graceful shutdown is triggered correctly.
"""

import asyncio
import signal
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_sigint_handler():
    """Test that SIGINT triggers graceful shutdown."""
    print("=" * 60)
    print("Testing SIGINT (Ctrl+C) Graceful Shutdown")
    print("=" * 60)

    # Import after path setup
    import argparse

    from datus.api.server import _run_server_async
    from datus.api.service import create_app

    # Create minimal args
    args = argparse.Namespace(
        host="127.0.0.1",
        port=8765,  # Use non-standard port for testing
        reload=False,
        workers=1,
        debug=True,
        namespace="test",
        config=None,
        max_steps=20,
        workflow="fixed",
        load_cp=None,
        root_path="",
        shutdown_timeout=5.0,
    )

    agent_args = argparse.Namespace(
        namespace=args.namespace,
        config=args.config,
        max_steps=args.max_steps,
        workflow=args.workflow,
        load_cp=args.load_cp,
        debug=args.debug,
        shutdown_timeout_seconds=args.shutdown_timeout,
    )

    # Start server in background
    server_task = asyncio.create_task(_run_server_async(args, agent_args))

    # Wait a bit for server to start
    print("Waiting for server to start...")
    await asyncio.sleep(2)

    # Simulate Ctrl+C by sending SIGINT to ourselves
    print("\n" + "=" * 60)
    print("Simulating Ctrl+C (SIGINT)...")
    print("=" * 60)

    # Send SIGINT to current process
    try:
        # This should trigger our graceful shutdown handler
        import os
        os.kill(os.getpid(), signal.SIGINT)

        # Wait for shutdown to complete
        print("Waiting for graceful shutdown to complete...")
        await asyncio.wait_for(server_task, timeout=10.0)
        print("\n✓ Server shut down gracefully!")
        print("=" * 60)
        return True
    except asyncio.TimeoutError:
        print("\n✗ Server did not shut down within timeout!")
        print("=" * 60)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        return False
    except Exception as e:
        print(f"\n✗ Error during shutdown: {e}")
        print("=" * 60)
        return False


async def test_sigterm_handler():
    """Test that SIGTERM also triggers graceful shutdown."""
    print("\n" + "=" * 60)
    print("Testing SIGTERM Graceful Shutdown")
    print("=" * 60)

    import argparse

    from datus.api.server import _run_server_async

    args = argparse.Namespace(
        host="127.0.0.1",
        port=8766,  # Different port
        reload=False,
        workers=1,
        debug=True,
        namespace="test",
        config=None,
        max_steps=20,
        workflow="fixed",
        load_cp=None,
        root_path="",
        shutdown_timeout=5.0,
    )

    agent_args = argparse.Namespace(
        namespace=args.namespace,
        config=args.config,
        max_steps=args.max_steps,
        workflow=args.workflow,
        load_cp=args.load_cp,
        debug=args.debug,
        shutdown_timeout_seconds=args.shutdown_timeout,
    )

    server_task = asyncio.create_task(_run_server_async(args, agent_args))

    print("Waiting for server to start...")
    await asyncio.sleep(2)

    print("\n" + "=" * 60)
    print("Simulating SIGTERM...")
    print("=" * 60)

    try:
        import os
        os.kill(os.getpid(), signal.SIGTERM)

        print("Waiting for graceful shutdown to complete...")
        await asyncio.wait_for(server_task, timeout=10.0)
        print("\n✓ Server shut down gracefully!")
        print("=" * 60)
        return True
    except asyncio.TimeoutError:
        print("\n✗ Server did not shut down within timeout!")
        print("=" * 60)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        return False
    except Exception as e:
        print(f"\n✗ Error during shutdown: {e}")
        print("=" * 60)
        return False


def main():
    """Run all signal handler tests."""
    print("\n" + "=" * 60)
    print("Datus API - Signal Handler Test Suite")
    print("=" * 60)

    results = []

    # Test SIGINT
    try:
        result = asyncio.run(test_sigint_handler())
        results.append(("SIGINT (Ctrl+C)", result))
    except Exception as e:
        print(f"SIGINT test failed with exception: {e}")
        results.append(("SIGINT (Ctrl+C)", False))

    # Give ports time to release
    time.sleep(1)

    # Test SIGTERM
    try:
        result = asyncio.run(test_sigterm_handler())
        results.append(("SIGTERM", result))
    except Exception as e:
        print(f"SIGTERM test failed with exception: {e}")
        results.append(("SIGTERM", False))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result for _, result in results)
    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
