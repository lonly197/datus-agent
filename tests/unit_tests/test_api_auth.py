
import os
import unittest
import secrets
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException

from datus.api.auth import AuthService, DEFAULT_JWT_CONFIG

class TestAuthService(unittest.TestCase):
    def setUp(self):
        # Configuration for testing
        self.test_clients = {"test_client": "test_secret_123"}
        self.test_jwt_config = {
            "secret_key": "test-jwt-secret",
            "algorithm": "HS256",
            "expiration_hours": 1
        }

    def create_mocked_service(self):
        # Helper to create service with mocked config
        with patch('datus.api.auth.load_auth_config') as mock_load:
            mock_load.return_value = {
                "clients": self.test_clients,
                "jwt": self.test_jwt_config
            }
            return AuthService()

    def test_validate_client_credentials_success(self):
        """Test successful client validation."""
        service = self.create_mocked_service()
        self.assertTrue(
            service.validate_client_credentials("test_client", "test_secret_123")
        )

    def test_validate_client_credentials_failure(self):
        """Test failed client validation."""
        service = self.create_mocked_service()
        # Wrong secret
        self.assertFalse(
            service.validate_client_credentials("test_client", "wrong_secret")
        )
        # Wrong client
        self.assertFalse(
            service.validate_client_credentials("wrong_client", "test_secret_123")
        )
        # Empty inputs
        self.assertFalse(
            service.validate_client_credentials("", "")
        )

    def test_token_generation_and_validation(self):
        """Test full token lifecycle."""
        service = self.create_mocked_service()
        token_data = service.generate_access_token("test_client")
        token = token_data["access_token"]
        
        payload = service.validate_token(token)
        self.assertEqual(payload["client_id"], "test_client")
        
        # Check expiration
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        self.assertGreater(exp, datetime.now(timezone.utc))

    def test_token_expiration(self):
        """Test that expired tokens raise HTTPException."""
        service = self.create_mocked_service()
        # Create an expired token manually
        expired_payload = {
            "client_id": "test_client",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
        }
        token = jwt.encode(
            expired_payload, 
            self.test_jwt_config["secret_key"], 
            algorithm=self.test_jwt_config["algorithm"]
        )
        
        with self.assertRaises(HTTPException) as cm:
            service.validate_token(token)
        self.assertEqual(cm.exception.detail, "Token has expired")

    def test_invalid_token(self):
        """Test that invalid tokens raise HTTPException."""
        service = self.create_mocked_service()
        with self.assertRaises(HTTPException) as cm:
            service.validate_token("invalid.token.string")
        self.assertEqual(cm.exception.detail, "Invalid token")

    def test_default_security(self):
        """Test that default configuration is secure (random key)."""
        # No patching of load_auth_config here, so it runs real code
        
        # We need to mock path_manager to avoid file not found warnings or actual file loading
        # We patch where it is IMPORTED or used.
        # Since load_auth_config imports it inside, we patch datus.utils.path_manager.get_path_manager
        
        with patch('datus.utils.path_manager.get_path_manager') as mock_pm:
            # Mock the path object returned
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            
            mock_pm.return_value.auth_config_path.return_value = mock_path
            
            auth_service = AuthService()
            
            # Should have empty clients by default
            self.assertEqual(auth_service.clients, {})
            
            # Should have a secret key (randomly generated)
            self.assertTrue(auth_service.jwt_secret)
            # Should not be the old hardcoded string (just in case)
            self.assertNotEqual(auth_service.jwt_secret, "your-secret-key-change-in-production")
            
            # Default key length should be substantial (hex string of 32 bytes = 64 chars)
            self.assertEqual(len(DEFAULT_JWT_CONFIG["secret_key"]), 64)

if __name__ == "__main__":
    unittest.main()
