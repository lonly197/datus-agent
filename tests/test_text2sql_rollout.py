#!/usr/bin/env python3
"""
Text2SQL Rollout Verification Script
Tests the unified text2sql workflow implementation
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_text2sql_workflow():
    """Test the complete text2sql workflow integration"""
    print("🚀 Starting Text2SQL workflow verification...")

    try:
        # Test 1: Import verification
        print("\n📦 Testing imports...")
        from datus.api.service import DatusAPIService
        from datus.configuration.node_type import NodeType

        print("✅ All imports successful")

        # Test 2: Node type constants
        print("\n🏷️  Testing node type constants...")
        assert hasattr(NodeType, "TYPE_INTENT_ANALYSIS"), "TYPE_INTENT_ANALYSIS missing"
        assert hasattr(NodeType, "TYPE_SCHEMA_DISCOVERY"), "TYPE_SCHEMA_DISCOVERY missing"
        assert NodeType.TYPE_INTENT_ANALYSIS == "intent_analysis"
        assert NodeType.TYPE_SCHEMA_DISCOVERY == "schema_discovery"
        print("✅ Node type constants verified")

        # Test 3: Service functionality
        print("\n🔧 Testing service functionality...")
        service = DatusAPIService.__new__(DatusAPIService)

        # Test workflow normalization
        assert service._normalize_workflow_name("nl2sql") == "text2sql"
        assert service._normalize_workflow_name("text2sql_standard") == "text2sql"
        assert service._normalize_workflow_name("text2sql") == "text2sql"
        print("✅ Workflow normalization works")

        # Test task type identification
        assert service._identify_task_type("Show me sales data") == "text2sql"
        assert service._identify_task_type("审查SQL查询") == "sql_review"
        assert service._identify_task_type("分析用户行为") == "data_analysis"
        print("✅ Task type identification works")

        # Test text2sql configuration
        config = service._configure_task_processing("text2sql", None)
        assert config["workflow"] == "text2sql"
        assert config["system_prompt"] == "text2sql_system"
        assert "required_tool_sequence" in config
        assert len(config["required_tool_sequence"]) >= 4  # At least 4 tools
        print("✅ Text2SQL configuration works")

        # Test 4: Workflow file verification
        print("\n📋 Testing workflow configuration...")
        import yaml

        with open("datus/agent/workflow.yml", "r") as f:
            wf_config = yaml.safe_load(f)

        workflows = wf_config.get("workflow", {})
        assert "text2sql" in workflows, "text2sql workflow missing"
        text2sql_steps = workflows["text2sql"]
        expected_steps = [
            "intent_analysis",
            "schema_discovery",
            "sql_generation",
            "syntax_validation",
            "execution_preview",
            "output",
        ]
        assert text2sql_steps == expected_steps, f"Unexpected steps: {text2sql_steps}"
        print("✅ Workflow configuration verified")

        # Test 5: Prompt verification
        print("\n📝 Testing prompt templates...")
        prompt_file = Path("datus/prompts/prompt_templates/text2sql_system_1.0.j2")
        assert prompt_file.exists(), "text2sql_system prompt template missing"
        with open(prompt_file, "r") as f:
            content = f.read()
            assert "Text-to-SQL expert" in content, "Prompt content incorrect"
        print("✅ Prompt template verified")

        # Test 6: Backward compatibility
        print("\n🔄 Testing backward compatibility...")
        # Verify legacy names are mapped correctly
        legacy_mappings = {"nl2sql": "text2sql", "text2sql_standard": "text2sql"}
        for legacy, expected in legacy_mappings.items():
            result = service._normalize_workflow_name(legacy)
            assert result == expected, f"Legacy mapping failed: {legacy} -> {result}"
        print("✅ Backward compatibility verified")

        print("\n🎉 All verification tests passed!")
        print("\n📊 Summary:")
        print("- ✅ Workflow unification: nl2sql/text2sql_standard → text2sql")
        print("- ✅ New nodes: intent_analysis, schema_discovery")
        print("- ✅ API integration: normalization and configuration")
        print("- ✅ Backward compatibility: legacy names supported")
        print("- ✅ Configuration: workflow, prompts, and tools")
        print("\n🚀 Text2SQL workflow is ready for production!")

        return True

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_text2sql_workflow())
    sys.exit(0 if success else 1)
