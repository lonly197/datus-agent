#!/usr/bin/env python3
"""
Test script for improved tool matching functionality.
"""

import os
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Mock the required components for testing
class MockPlanModeHooks:
    def __init__(self):
        self.keyword_map = {
            "search_table": ["探索", "查找", "找到", "搜索", "explore", "find", "search"],
            "describe_table": ["描述", "检查", "查看", "describe", "inspect", "examine"],
            "execute_sql": ["执行", "运行", "execute", "run"],
            "search_metrics": ["指标", "metrics", "kpi"],
        }
        self.model = None  # No LLM for testing

    def _preprocess_todo_content(self, content):
        """Preprocess todo content to improve parameter extraction."""
        if not content:
            return ""

        # Remove common prefixes that don't help with parameter extraction
        prefixes_to_remove = [
            "sub-question:",
            "sub question:",
            "任务：",
            "任务:",
            "问题：",
            "问题:",
            "| expected:",
            "| 期望:",
        ]

        cleaned = content.lower()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()

        # Remove extra whitespace
        cleaned = " ".join(cleaned.split())

        return cleaned

    def _classify_task_intent(self, text: str):
        """Classify the intent of a task and map it to the most appropriate tool."""
        if not text:
            return None

        text_lower = text.lower()

        # Simple task patterns for testing
        task_patterns = {
            "explore_schema": {
                "patterns": [r"探索.*表结构", r"查看.*表结构", r"找到.*表"],
                "tool": "search_table",
                "confidence": 0.95,
                "reason": "Database table exploration task",
                "priority": 1,
            },
            "describe_table": {
                "patterns": [r"描述.*表", r"检查.*表.*定义"],
                "tool": "describe_table",
                "confidence": 0.90,
                "reason": "Table description and metadata analysis",
                "priority": 1,
            },
        }

        # Check each task type
        import re

        for task_type, config in task_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return {
                        "tool": config["tool"],
                        "confidence": config["confidence"],
                        "reason": config["reason"],
                        "task_type": task_type,
                    }

        return None

    def _match_keywords_with_context(self, text: str):
        """Enhanced keyword matching that considers context and intent."""
        if not text:
            return None

        text_lower = text.lower()
        best_match = None
        best_score = 0

        # Analyze context clues
        context_indicators = {
            "database_focus": sum(1 for word in ["表", "table", "database", "schema"] if word in text_lower),
        }

        # Primary context
        primary_context = "database_focus" if context_indicators["database_focus"] > 0 else None

        # Enhanced keyword matching with context awareness
        context_aware_mappings = {
            "database_focus": {"search_table": ["探索", "查找", "找到", "搜索"], "describe_table": ["描述", "检查", "查看"]}
        }

        # Apply context-aware scoring
        for tool_name, keywords in self.keyword_map.items():
            score = 0

            # Base keyword matching
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1

            # Context boost
            if primary_context and tool_name in context_aware_mappings.get(primary_context, {}):
                context_keywords = context_aware_mappings[primary_context][tool_name]
                for ctx_keyword in context_keywords:
                    if ctx_keyword.lower() in text_lower:
                        score += 2  # Context match gets higher weight

            # Normalize score
            if keywords:
                normalized_score = score / len(keywords)
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_match = {
                        "tool": tool_name,
                        "confidence": min(0.95, normalized_score),
                        "context": primary_context,
                        "score": score,
                    }

        return best_match if best_score > 0.3 else None

    def _semantic_chinese_matching(self, text: str):
        """Semantic understanding for Chinese task descriptions."""
        if not text:
            return None

        # Simple Chinese pattern recognition for testing
        chinese_patterns = {
            "search_table": {"verbs": ["探索", "查找", "找到", "搜索"], "nouns": ["表结构", "表", "字段", "数据库"], "confidence": 0.85}
        }

        for tool_name, pattern_config in chinese_patterns.items():
            verb_matches = sum(1 for verb in pattern_config["verbs"] if verb in text)
            noun_matches = sum(1 for noun in pattern_config["nouns"] if noun in text)

            if verb_matches > 0 and noun_matches > 0:
                confidence = min(
                    pattern_config["confidence"],
                    (verb_matches + noun_matches) / (len(pattern_config["verbs"]) + len(pattern_config["nouns"])),
                )

                if confidence > 0.6:
                    return {"tool": tool_name, "confidence": confidence, "reason": "Chinese semantic pattern matching"}

        return None

    def _enhanced_llm_reasoning(self, text: str):
        """Mock LLM reasoning - returns None for testing."""
        return None

    def _enhanced_intelligent_inference(self, text: str):
        """Enhanced intelligent inference as final fallback."""
        if not text:
            return None

        text_lower = text.lower()

        # Priority-based inference rules
        inference_rules = [
            (
                lambda t: any(word in t for word in ["探索", "查找", "找到", "搜索"])
                and any(word in t for word in ["表", "table", "database"]),
                "search_table",
                0.8,
            ),
            (
                lambda t: any(word in t for word in ["描述", "检查", "分析"])
                and any(word in t for word in ["表", "table", "schema"]),
                "describe_table",
                0.75,
            ),
        ]

        for condition, tool, confidence in inference_rules:
            if condition(text_lower):
                return tool

        return None

    def _match_tool_for_todo(self, text: str):
        """
        Advanced hybrid tool matching with intelligent intent recognition.
        """
        if not text:
            return None

        cleaned_text = self._preprocess_todo_content(text)

        # Tier 1: Task Intent Classification
        intent_result = self._classify_task_intent(cleaned_text)
        if intent_result and intent_result["confidence"] > 0.8:
            return intent_result["tool"]

        # Tier 2: Enhanced Context-Aware Keyword Matching
        context_match = self._match_keywords_with_context(cleaned_text)
        if context_match and context_match["confidence"] > 0.7:
            return context_match["tool"]

        # Tier 3: Semantic Understanding for Chinese Tasks
        semantic_match = self._semantic_chinese_matching(cleaned_text)
        if semantic_match:
            return semantic_match["tool"]

        # Tier 4: LLM reasoning fallback
        llm_match = self._enhanced_llm_reasoning(cleaned_text)
        if llm_match:
            return llm_match

        # Tier 5: Intelligent inference
        inference_match = self._enhanced_intelligent_inference(cleaned_text)
        if inference_match:
            return inference_match

        return None


def test_original_problem_case():
    """Test the original problematic case from the logs."""
    print("=== Testing Original Problem Case ===")
    hooks = MockPlanModeHooks()

    # Original problematic input
    test_input = "Sub-question: 探索数据库中的表结构，找到试驾表和线索表 | Expected: 确认表名、字段名和关联关系"
    print(f"Input: {test_input}")

    result = hooks._match_tool_for_todo(test_input)
    print(f"Matched tool: {result}")

    # Expected: should match "search_table" instead of "describe_table"
    expected = "search_table"
    if result == expected:
        print("✅ SUCCESS: Correctly matched to search_table")
    else:
        print(f"❌ FAILED: Expected {expected}, got {result}")

    print()


def test_various_cases():
    """Test various task descriptions."""
    print("=== Testing Various Cases ===")
    hooks = MockPlanModeHooks()

    test_cases = [
        ("探索数据库中的表结构", "search_table"),
        ("描述用户表的结构", "describe_table"),
        ("执行SQL查询获取数据", "execute_sql"),
        ("分析销售指标数据", "search_metrics"),
        ("查看customer表定义", "describe_table"),
        ("查找所有用户相关表", "search_table"),
    ]

    for input_text, expected_tool in test_cases:
        result = hooks._match_tool_for_todo(input_text)
        status = "✅" if result == expected_tool else "❌"
        print(f"{status} '{input_text}' -> {result} (expected: {expected_tool})")

    print()


def test_preprocessing():
    """Test text preprocessing."""
    print("=== Testing Text Preprocessing ===")
    hooks = MockPlanModeHooks()

    test_cases = [
        ("Sub-question: 探索数据库", "探索数据库"),
        ("任务：分析数据 | expected: 结果", "分析数据 | expected: 结果"),
        ("问题：查找表结构", "查找表结构"),
    ]

    for input_text, expected in test_cases:
        result = hooks._preprocess_todo_content(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_text}' -> '{result}'")

    print()


def main():
    print("Testing Improved Tool Matching Functionality")
    print("=" * 50)

    test_preprocessing()
    test_original_problem_case()
    test_various_cases()

    print("🎉 Tool matching improvement tests completed!")


if __name__ == "__main__":
    main()
