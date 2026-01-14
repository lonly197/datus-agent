#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Datus Agent FastAPI server startup script.
Supports foreground and daemon (background) modes with start/stop/restart/status.
"""
import argparse
import atexit
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import uvicorn

from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)


def _default_paths() -> Tuple[Path, Path]:
    """Return default pid and log file paths."""
    from datus.utils.path_manager import get_path_manager

    path_manager = get_path_manager()
    pid_file = path_manager.pid_file_path("datus-agent-api")
    log_file = Path("logs") / "datus-agent-api.log"  # Use logs/ directory like other modules
    return pid_file, log_file


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        pid_text = pid_file.read_text().strip()
        return int(pid_text) if pid_text else None
    except Exception:
        return None


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def _write_pid_file(pid_file: Path, pid: int) -> None:
    _ensure_parent_dir(pid_file)
    pid_file.write_text(str(pid))


def _remove_pid_file(pid_file: Path) -> None:
    try:
        if pid_file.exists():
            pid_file.unlink()
    except Exception:
        pass


def _redirect_stdio(log_file: Path) -> None:
    _ensure_parent_dir(log_file)
    # Open files
    si = open(os.devnull, "rb", buffering=0)
    so = open(log_file, "ab", buffering=0)
    se = open(log_file, "ab", buffering=0)
    # Duplicate fds
    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())


def _daemon_worker(args: argparse.Namespace, agent_args: argparse.Namespace, pid_file: Path, log_file: Path) -> None:
    """Worker function that runs in the daemon process."""
    # Set process session and umask
    os.setsid()
    os.umask(0)

    # Configure logging for daemon process
    configure_logging(args.debug, log_dir="logs", console_output=False)

    # Redirect stdio
    _redirect_stdio(log_file)

    # Write PID file
    _write_pid_file(pid_file, os.getpid())

    def _cleanup(*_args):
        _remove_pid_file(pid_file)

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_a: (_cleanup(), os._exit(0)))

    # Run the actual server
    _run_server(args, agent_args)


def _stop(pid_file: Path, timeout_seconds: float = 10.0) -> int:
    pid = _read_pid(pid_file)
    if not pid:
        logger.info("No PID file found; server not running?")
        return 0
    if not _is_process_running(pid):
        logger.info("Stale PID file found; process not running. Cleaning up.")
        _remove_pid_file(pid_file)
        return 0

    logger.info(f"Stopping server (pid={pid}) ...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _remove_pid_file(pid_file)
        return 0

    import time

    start = time.time()
    while time.time() - start < timeout_seconds:
        if not _is_process_running(pid):
            _remove_pid_file(pid_file)
            logger.info("Stopped.")
            return 0
        time.sleep(0.2)

    # Fallback to SIGKILL
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _remove_pid_file(pid_file)
    logger.info("Force killed.")
    return 0


def _status(pid_file: Path) -> int:
    pid = _read_pid(pid_file)
    if pid and _is_process_running(pid):
        print(f"running (pid={pid})")
        return 0
    print("stopped")
    return 1


def _build_agent_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        namespace=args.namespace,
        config=args.config,
        max_steps=args.max_steps,
        workflow=args.workflow,
        load_cp=args.load_cp,
        debug=args.debug,
        shutdown_timeout_seconds=args.shutdown_timeout,
    )


async def _run_server_async(args: argparse.Namespace, agent_args: argparse.Namespace) -> None:
    """Run the server with custom signal handling for graceful shutdown."""
    import asyncio

    from datus.api.service import create_app

    # Create the app
    app = create_app(agent_args, root_path=args.root_path)

    # Create a Server instance instead of using uvicorn.run()
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="debug" if args.debug else "info",
        access_log=True,
    )
    server = uvicorn.Server(config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    # Handler for SIGINT (Ctrl+C) and SIGTERM
    def handle_signal(sig, frame):
        """Handle shutdown signals gracefully."""
        signal_name = signal.Signals(sig).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown (timeout={args.shutdown_timeout}s)...")
        shutdown_event.set()

        # Trigger server shutdown
        if server.should_exit:
            return
        server.should_exit = True

    # Register signal handlers
    try:
        loop.add_signal_handler(signal.SIGINT, handle_signal, signal.SIGINT, None)
        loop.add_signal_handler(signal.SIGTERM, handle_signal, signal.SIGTERM, None)
        logger.info("Signal handlers registered for graceful shutdown (SIGINT, SIGTERM)")
    except NotImplementedError:
        # add_signal_handler not available on this platform
        # Fall back to standard signal handling (will use uvicorn's default)
        logger.warning("asyncio signal handlers not supported on this platform, using default uvicorn signal handling")

    # Serve with graceful shutdown
    try:
        await server.serve()
    except KeyboardInterrupt:
        # Fallback: if KeyboardInterrupt still gets through, handle gracefully
        logger.info("KeyboardInterrupt caught, initiating graceful shutdown...")
        if server.should_exit:
            return
        server.should_exit = True
        await server.shutdown()


def _run_server(args: argparse.Namespace, agent_args: argparse.Namespace) -> None:
    """Run the server with proper signal handling for foreground mode."""
    import asyncio

    # Use asyncio.run to properly handle the async server
    try:
        asyncio.run(_run_server_async(args, agent_args))
    except KeyboardInterrupt:
        # This should not normally happen due to our signal handlers,
        # but keep it as a safety net
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


def main():
    """Main entry point for starting the Datus Agent API server."""
    # Set multiprocessing start method to avoid potential issues
    if hasattr(multiprocessing, "set_start_method"):
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # Start method can only be set once
            pass
    parser = argparse.ArgumentParser(description="Start Datus Agent FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--namespace",
        type=str,
        help="Namespace of databases or benchmark",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: conf/agent.yml > ~/.datus/conf/agent.yml)",
    )
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum workflow steps")
    parser.add_argument("--workflow", type=str, default="fixed", help="Workflow plan type")
    parser.add_argument("--load_cp", type=str, help="Load workflow from checkpoint file")

    # Daemon control
    parser.add_argument(
        "--action",
        choices=["start", "stop", "restart", "status"],
        default="start",
        help="Daemon action (default: start)",
    )
    parser.add_argument("--daemon", action="store_true", help="Run in background as a daemon")
    parser.add_argument(
        "--pid-file",
        type=str,
        help="PID file path (default: ~/.datus/run/datus-agent-api.pid)",
    )
    parser.add_argument(
        "--daemon-log-file",
        type=str,
        help="Daemon log file path (default: logs/datus-agent-api.log)",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default="",
        help="Root path for the API (e.g., '/sql-api' for proxy setups)",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=5.0,
        help="Maximum time in seconds to wait for running tasks to cancel during shutdown (default: 5.0)",
    )

    args = parser.parse_args()

    # Resolve defaults for pid/log
    default_pid, default_log = _default_paths()
    pid_file = Path(args.pid_file) if args.pid_file else default_pid
    log_file = Path(args.daemon_log_file) if args.daemon_log_file else default_log

    if args.action in {"status", "stop"}:
        configure_logging(args.debug)
        if args.action == "status":
            raise SystemExit(_status(pid_file))
        if args.action == "stop":
            raise SystemExit(_stop(pid_file))

    if args.action == "restart":
        configure_logging(args.debug)
        _stop(pid_file)
        # fall-through to start

    # Start
    if args.daemon and args.reload:
        print("--daemon mode is mutually exclusive with --reload. Remove --reload.", file=sys.stderr)
        raise SystemExit(2)

    if args.daemon:
        if (pid := _read_pid(pid_file)) and _is_process_running(pid):
            print(f"Already running (pid={pid})", file=sys.stderr)
            raise SystemExit(0)

        configure_logging(args.debug, log_dir="logs", console_output=False)
        logger.info(
            f"Starting Datus Agent API server (daemon) on {args.host}:{args.port} | "
            f"Workers: {args.workers}, Debug: {args.debug}"
        )

        # Create daemon process using multiprocessing
        agent_args = _build_agent_args(args)
        daemon_process = multiprocessing.Process(
            target=_daemon_worker, args=(args, agent_args, pid_file, log_file), daemon=False
        )
        daemon_process.start()

        # Give the daemon a moment to start
        time.sleep(0.5)

        # Check if the daemon started successfully
        if daemon_process.is_alive():
            print(f"Daemon started successfully (pid={daemon_process.pid})")
            # Force exit main process immediately - don't wait for daemon
            os._exit(0)
        else:
            print("Failed to start daemon process", file=sys.stderr)
            daemon_process.join()  # Clean up failed process
            os._exit(1)
    else:
        # Foreground mode - normal logging setup with console output
        configure_logging(args.debug)

    # Foreground run (existing behavior)
    logger.info(f"Starting Datus Agent API server on {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}, Reload: {args.reload}, Debug: {args.debug}")
    logger.info(f"Agent config - Namespace: {args.namespace}, Config: {args.config}")
    agent_args = _build_agent_args(args)
    _run_server(args, agent_args)


if __name__ == "__main__":
    main()
