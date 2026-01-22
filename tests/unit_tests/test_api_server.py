
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import argparse
import os
from pathlib import Path
from datus.api.server import _daemon_worker, _run_server_async

class TestApiServer(unittest.IsolatedAsyncioTestCase):
    @patch('datus.api.server.os')
    @patch('datus.api.server.configure_logging')
    @patch('datus.api.server._redirect_stdio')
    @patch('datus.api.server._write_pid_file')
    @patch('datus.api.server.atexit')
    @patch('datus.api.server.signal')
    @patch('datus.api.server._run_server')
    def test_daemon_worker_security(self, mock_run, mock_signal, mock_atexit, mock_pid, mock_redir, mock_log, mock_os):
        """Test daemon worker security settings."""
        args = MagicMock()
        agent_args = MagicMock()
        pid_file = Path("test.pid")
        log_file = Path("test.log")

        _daemon_worker(args, agent_args, pid_file, log_file)

        # Verify umask is secure (0o022)
        mock_os.umask.assert_called_with(0o022)
        # Verify setsid is called
        mock_os.setsid.assert_called()

    @patch('datus.api.server.uvicorn')
    @patch('datus.api.service.create_app')
    async def test_server_config_security(self, mock_create_app, mock_uvicorn):
        """Test uvicorn server configuration."""
        # Setup async mock for server.serve
        mock_server_instance = mock_uvicorn.Server.return_value
        mock_server_instance.serve = AsyncMock()

        args = MagicMock()
        args.host = "127.0.0.1"
        args.port = 8000
        args.reload = False
        args.workers = 1
        args.debug = False
        agent_args = MagicMock()

        # We need to run the async function
        await _run_server_async(args, agent_args)

        # Check Config call args
        call_args = mock_uvicorn.Config.call_args
        self.assertIsNotNone(call_args)
        kwargs = call_args[1]
        
        # Verify server_header is False
        self.assertFalse(kwargs.get("server_header"))

if __name__ == "__main__":
    unittest.main()
