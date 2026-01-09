# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SchemaValidationNode implementation for validating schema sufficiency.

This node validates that the schemas discovered by schema_discovery_node
are sufficient for generating SQL for the given query. It checks for:
- Minimum schema requirements (at least one table)
- Schema completeness (tables have column definitions)
- Basic query-schema alignment (keywords in query match schema)
"""

import re
from typing import Any, AsyncGenerator, Dict, List, Optional

import jieba

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import TableSchema
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SchemaValidationNode(Node):
    """
    Node for validating schema sufficiency before SQL generation.

    This node checks if the discovered schemas contain sufficient information
    to generate SQL for the given query. If schemas are insufficient, it
    provides actionable feedback for reflection strategies.
    """

    # Business term mapping: Chinese → English schema terms
    # This enables semantic matching between Chinese business terminology
    # and English database schema names
    BUSINESS_TERM_MAPPING = {
        # Test drive related
        "试驾": ["test_drive", "test_drive_date", "trial_drive", "testdrive"],
        "首次": ["first", "initial", "first_time"],
        "线索": ["clue", "clue_id", "lead", "lead_id"],
        # Order related
        "下定": ["order", "order_date", "booking", "book"],
        "订单": ["order", "order_id", "orders"],
        "转化": ["conversion", "convert", "transform"],
        # Time related
        "周期": ["cycle", "period", "duration"],
        "天数": ["days", "date_diff", "day_count"],
        "月份": ["month", "monthly", "mth"],
        # Statistics
        "统计": ["count", "sum", "avg", "calculate", "compute"],
        "平均": ["avg", "average", "mean"],
    }

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[BaseInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
    ):
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
        )

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """Setup schema validation input from workflow context."""
        if not self.input:
            self.input = BaseInput()
        return {"success": True, "message": "Schema validation input setup complete"}

    async def run(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Run schema validation to check if schemas are sufficient for SQL generation.

        Yields:
            ActionHistory: Progress and result actions
        """
        try:
            if not self.workflow or not self.workflow.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "No workflow or task available for schema validation",
                    "schema_validation",
                    {"workflow_id": getattr(self.workflow, "id", "unknown") if self.workflow else "unknown"},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Schema validation failed: {error_result.error_message}",
                    action_type="schema_validation",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            task = self.workflow.task
            context = self.workflow.context

            # Step 1: Check if schemas were discovered
            if not context or not context.table_schemas:
                no_schemas_result = {
                    "is_sufficient": False,
                    "error": "No schemas discovered",
                    "missing_tables": ["all"],
                    "suggestions": ["Trigger schema_linking to discover tables"],
                    "allow_reflection": True,  # Allow reflection to recover
                }
                yield ActionHistory(
                    action_id=f"{self.id}_no_schemas",
                    role=ActionRole.TOOL,
                    messages="Schema validation failed: No schemas discovered",
                    action_type="schema_validation",
                    input={"task": task.task[:50] if task else ""},
                    status=ActionStatus.SOFT_FAILED,  # Use SOFT_FAILED to allow reflection
                    output=no_schemas_result,
                )
                self.result = BaseResult(
                    success=False,
                    error="No schemas discovered for validation",
                    data=no_schemas_result,
                )
                return

            # Step 2: Validate schema completeness
            table_count = len(context.table_schemas)
            missing_definitions = []
            for schema in context.table_schemas:
                if not schema.definition or schema.definition.strip() == "":
                    missing_definitions.append(schema.table_name)

            # Step 3: Check basic query-schema alignment
            # Extract key terms from the query
            query_lower = task.task.lower() if task else ""
            query_terms = self._extract_query_terms(query_lower)

            # Check if schemas contain relevant columns/tables
            schema_coverage = self._check_schema_coverage(context.table_schemas, query_terms)

            # Step 4: Determine if schemas are sufficient
            is_sufficient = (
                table_count > 0
                and len(missing_definitions) == 0
                and schema_coverage["coverage_score"] > 0.3  # At least 30% coverage
            )

            # Build validation result
            validation_result = {
                "is_sufficient": is_sufficient,
                "table_count": table_count,
                "missing_definitions": missing_definitions,
                "query_terms": query_terms,
                "coverage_score": schema_coverage["coverage_score"],
                "covered_terms": schema_coverage["covered_terms"],
                "uncovered_terms": schema_coverage["uncovered_terms"],
            }

            # Add recommendations if insufficient
            if not is_sufficient:
                if missing_definitions:
                    validation_result["suggestions"] = [
                        f"Load full DDL for tables: {', '.join(missing_definitions[:5])}"
                    ]
                elif schema_coverage["coverage_score"] < 0.3:
                    validation_result["suggestions"] = [
                        "Trigger schema_linking to find more relevant tables",
                        f"Consider tables matching terms: {', '.join(schema_coverage['uncovered_terms'][:5])}",
                    ]
                    validation_result["missing_tables"] = schema_coverage["uncovered_terms"][:5]

                # Set allow_reflection flag to enable workflow continuation
                validation_result["allow_reflection"] = True

            # Emit result
            if is_sufficient:
                yield ActionHistory(
                    action_id=f"{self.id}_validation",
                    role=ActionRole.TOOL,
                    messages=f"Schema validation passed: {table_count} tables with sufficient coverage",
                    action_type="schema_validation",
                    input={
                        "task": task.task[:50] if task else "",
                        "table_count": table_count,
                    },
                    status=ActionStatus.SUCCESS,
                    output=validation_result,
                )
                self.result = BaseResult(success=True, data=validation_result)
            else:
                yield ActionHistory(
                    action_id=f"{self.id}_validation",
                    role=ActionRole.TOOL,
                    messages=f"Schema validation failed: Insufficient schema coverage (score: {schema_coverage['coverage_score']:.2f})",
                    action_type="schema_validation",
                    input={
                        "task": task.task[:50] if task else "",
                        "table_count": table_count,
                    },
                    status=ActionStatus.SOFT_FAILED,  # Use SOFT_FAILED to allow reflection
                    output=validation_result,
                )
                self.result = BaseResult(
                    success=False, error="Schema validation failed: Insufficient coverage", data=validation_result
                )

            logger.info(
                f"Schema validation completed: is_sufficient={is_sufficient}, "
                f"table_count={table_count}, coverage_score={schema_coverage['coverage_score']:.2f}"
            )

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            error_result = self.create_error_result(
                ErrorCode.NODE_EXECUTION_FAILED,
                f"Schema validation execution failed: {str(e)}",
                "schema_validation",
                {
                    "task_id": getattr(self.workflow.task, "id", "unknown")
                    if self.workflow and self.workflow.task
                    else "unknown"
                },
            )
            yield ActionHistory(
                action_id=f"{self.id}_error",
                role=ActionRole.TOOL,
                messages=f"Schema validation failed: {error_result.error_message}",
                action_type="schema_validation",
                input={},
                status=ActionStatus.FAILED,
                output={"error": error_result.error_message, "error_code": error_result.error_code},
            )
            self.result = BaseResult(success=False, error=str(e))

    def _extract_query_terms(self, query: str) -> List[str]:
        """
        Extract key terms from the query for schema matching.

        Uses jieba for Chinese word segmentation and regex for English.
        This hybrid approach ensures proper tokenization for mixed-language queries.
        """
        # Common SQL and business terms to look for (stop words)
        stop_words = {
            # English stop words
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "when",
            "where",
            "how",
            "why",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "also",
            "now",
            "here",
            "there",
            "then",
            "once",
            "get",
            "got",
            "make",
            "go",
            "see",
            "know",
            "take",
            "come",
            "think",
            "look",
            "want",
            "give",
            "use",
            "find",
            "tell",
            "ask",
            "work",
            "seem",
            "feel",
            "try",
            "leave",
            "call",
            "show",
            # Chinese stop words (statistics terms that don't map to schema)
            "每个",
            "查询",
            "分析",
            "数据",
            "表",
        }

        # Detect if query contains Chinese characters
        has_chinese = any("\u4e00" <= char <= "\u9fff" for char in query)

        if has_chinese:
            # Use jieba for Chinese word segmentation
            words = jieba.lcut(query)
            # Filter out stop words and short terms
            terms = [w for w in words if w not in stop_words and len(w) > 1]
        else:
            # Use regex for English
            words = re.findall(r"\b\w+\b", query)
            # Filter out stop words and short words
            terms = [w for w in words if w.lower() not in stop_words and len(w) > 2]

        return terms

    def _check_schema_coverage(self, schemas: List[TableSchema], query_terms: List[str]) -> Dict[str, Any]:
        """
        Check how well the schemas cover the query terms.

        Uses semantic matching via business term mapping to enable Chinese-to-English
        schema matching. This allows queries with Chinese business terminology to
        match English database schema names.
        """
        if not query_terms:
            return {"coverage_score": 1.0, "covered_terms": [], "uncovered_terms": []}

        covered = []
        uncovered = []

        # Build a set of all schema terms (table names, column names)
        schema_terms = set()
        for schema in schemas:
            # Add table name
            schema_terms.add(schema.table_name.lower())

            # Add column names from definition
            if schema.definition:
                # Extract column names from CREATE TABLE statement
                columns = re.findall(r"`(\w+)`", schema.definition)
                schema_terms.update([c.lower() for c in columns])

        # Check each query term with semantic mapping
        for term in query_terms:
            # Direct match (case-insensitive)
            if term.lower() in schema_terms:
                covered.append(term)
                continue

            # Semantic match via business term mapping (Chinese → English)
            if term in self.BUSINESS_TERM_MAPPING:
                english_terms = self.BUSINESS_TERM_MAPPING[term]
                # Check if any of the mapped English terms are in the schema
                if any(eng_term.lower() in schema_terms for eng_term in english_terms):
                    covered.append(term)
                    continue

            # Partial match for compound terms (e.g., "首次试驾" contains "试驾")
            found_partial = False
            for schema_term in schema_terms:
                if term in schema_term or schema_term in term:
                    covered.append(term)
                    found_partial = True
                    break

            if not found_partial:
                uncovered.append(term)

        coverage_score = len(covered) / len(query_terms) if query_terms else 1.0

        return {
            "coverage_score": coverage_score,
            "covered_terms": covered,
            "uncovered_terms": uncovered,
        }

    def execute(self) -> BaseResult:
        """Execute schema validation synchronously."""
        return execute_with_async_stream(self)

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute schema validation with streaming support."""
        if action_history_manager:
            self.action_history_manager = action_history_manager
        async for action in self.run():
            yield action

    def update_context(self, workflow: "Workflow") -> Dict[str, Any]:
        """Update workflow context with validation results."""
        try:
            if not self.result or not self.result.success:
                return {"success": False, "message": "Schema validation failed, cannot update context"}

            # Store validation result in workflow metadata
            if not hasattr(workflow, "metadata"):
                workflow.metadata = {}
            workflow.metadata["schema_validation"] = self.result.data

            return {
                "success": True,
                "message": f"Schema validation context updated with result: {self.result.data.get('is_sufficient', False)}",
            }

        except Exception as e:
            logger.error(f"Failed to update schema validation context: {str(e)}")
            return {"success": False, "message": f"Schema validation context update failed: {str(e)}"}
