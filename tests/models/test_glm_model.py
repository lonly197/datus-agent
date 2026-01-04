import os
from unittest.mock import MagicMock, patch

import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv

from datus.configuration.agent_config import AgentConfig, ModelConfig
from datus.models.glm_model import GlmModel
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

logger = get_logger(__name__)
set_tracing_disabled(True)


@pytest.fixture
def agent_config() -> AgentConfig:
    load_dotenv()
    config = load_acceptance_config()
    # Inject GLM config for testing
    config.models["glm"] = ModelConfig(
        type="glm",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="mock_key",
        model="glm-4",
    )
    return config


class TestGlmModel:
    """Test suite for the GlmModel class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, agent_config):
        """Set up test environment before each test method."""
        self.config = agent_config.models["glm"]
        self.model = GlmModel(model_config=self.config)

    def test_init(self):
        """Test initialization."""
        assert self.model.model_name == "glm-4"
        assert self.model.base_url == "https://open.bigmodel.cn/api/paas/v4/"
        assert self.model.api_key == "mock_key"
        assert self.model.max_tokens() == 128000

    def test_get_api_key_from_env(self):
        """Test getting API key from environment."""
        # Create a config without API key
        config = ModelConfig(
            type="glm",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key="",
            model="glm-4",
        )
        
        # Test with GLM_API_KEY
        with patch.dict(os.environ, {"GLM_API_KEY": "env_key_1"}):
            model = GlmModel(model_config=config)
            assert model.api_key == "env_key_1"
            
        # Test with ZHIPUAI_API_KEY
        with patch.dict(os.environ, {"GLM_API_KEY": "", "ZHIPUAI_API_KEY": "env_key_2"}):
            model = GlmModel(model_config=config)
            assert model.api_key == "env_key_2"
            
        # Test missing key
        with patch.dict(os.environ, {"GLM_API_KEY": "", "ZHIPUAI_API_KEY": ""}):
            with pytest.raises(ValueError, match="GLM API key must be provided"):
                GlmModel(model_config=config)

    @patch("datus.models.openai_compatible.OpenAI")
    def test_generate(self, mock_openai):
        """Test basic text generation functionality."""
        # Mock the client response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Hello from GLM"))]
        mock_client.chat.completions.create.return_value = mock_completion
        
        # We need to patch the client on the instance because it's initialized in __init__
        self.model.client = mock_client
        
        result = self.model.generate("Hello", temperature=0.5, max_tokens=100)
        
        assert result == "Hello from GLM"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "glm-4"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    def test_token_count(self):
        """Test token counting."""
        text = "Hello world"
        # Implementation is int(len(prompt) * 0.3 + 0.5)
        expected = int(len(text) * 0.3 + 0.5)
        assert self.model.token_count(text) == expected
