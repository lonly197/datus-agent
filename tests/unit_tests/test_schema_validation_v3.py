from unittest.mock import MagicMock

import pytest

from datus.agent.node.schema_validation_node import SchemaValidationNode
from datus.schemas.node_models import BaseInput


class TestSchemaValidationNodeV3:
    @pytest.fixture
    def mock_agent_config(self):
        return MagicMock()

    @pytest.fixture
    def validation_node(self, mock_agent_config):
        node = SchemaValidationNode(
            node_id="validation_node",
            description="Test schema validation",
            node_type="schema_validation",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )
        node.model = MagicMock()
        return node

    def test_extract_query_terms_prompt(self, validation_node):
        """Verify the prompt contains fine-grained breakdown instructions."""
        query = "统计每个月首次试驾的平均转化周期"
        validation_node.model.generate_with_json_output.return_value = {
            "terms": ["统计", "每个月", "首次", "试驾", "平均", "转化", "周期"]
        }

        terms = validation_node._extract_query_terms(query)

        # Verify prompt content
        args, _ = validation_node.model.generate_with_json_output.call_args
        prompt = args[0]

        assert "Break down compound business terms into atomic concepts" in prompt
        assert 'e.g., "首次试驾" -> "首次", "试驾"' in prompt
        assert "Examples:" in prompt
        assert query in prompt

        # Verify output
        assert "首次" in terms
        assert "试驾" in terms
