# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Core SchemaValidationNode implementation.

This module contains the main SchemaValidationNode class that orchestrates
schema validation workflow, including prerequisite checks, schema coverage
validation, and result generation.
"""

import hashlib
from typing import Any, AsyncGenerator, Dict, List, Optional

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.configuration.business_term_config import LLM_TERM_EXTRACTION_CONFIG, SCHEMA_VALIDATION_CONFIG
from datus.schemas.action_history import (
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)
from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import TableSchema
from datus.utils.error_handler import LLMMixin, NodeExecutionResult
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

from .diagnostics import DiagnosticReporter
from .term_utils import TermProcessor
from .validators import CoverageValidator

logger = get_logger(__name__)


class SchemaValidationNode(Node, LLMMixin):
    """
    Node for validating schema sufficiency before SQL generation.

    This node checks if the discovered schemas contain sufficient information
    to generate SQL for the given query. If schemas are insufficient, it
    provides actionable feedback for reflection strategies.

    ✅ Fixed: Uses LLMMixin for LLM retry and centralized business term config.
    ✅ Enhanced: Sets last_action_status for workflow termination logic.
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[BaseInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
    ):
        Node.__init__(
            self,
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
        )
        LLMMixin.__init__(self)
        self.last_action_status = None  # Track last action status for workflow runner

        # Initialize helper classes
        self.term_processor = TermProcessor()
        self.diagnostic_reporter = DiagnosticReporter()

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
            # Pre-validation: Ensure workflow and task exist
            async for action in self._validate_workflow_prerequisites():
                yield action
                if action.status == ActionStatus.FAILED:
                    return

            task = self.workflow.task
            context = self.workflow.context

            # Step 1: Check if schemas were discovered (hard failure)
            if not context or not context.table_schemas:
                yield await self._handle_no_schemas_failure(task)
                return

            # Step 2-4: Validate schemas and check coverage
            validation_result = await self._validate_schema_coverage(task, context)

            # Set result based on validation
            hard_block = not validation_result["is_sufficient"] and (
                validation_result["coverage_score"] < 0.2
                or len(validation_result.get("critical_terms_uncovered", [])) > 0
            )
            self.last_action_status = (
                ActionStatus.SUCCESS
                if validation_result["is_sufficient"]
                else (ActionStatus.FAILED if hard_block else ActionStatus.SOFT_FAILED)
            )

            self.result = BaseResult(
                success=validation_result["is_sufficient"],
                error=None
                if validation_result["is_sufficient"]
                else "Schema validation failed: Insufficient coverage",
                data=validation_result,
            )

            # Persist validation snapshot for downstream nodes / reflection
            if self.workflow is not None:
                if not hasattr(self.workflow, "metadata") or self.workflow.metadata is None:
                    self.workflow.metadata = {}
                self.workflow.metadata["schema_validation"] = validation_result
                if hard_block:
                    from datus.agent.workflow_status import WorkflowTerminationStatus

                    self.workflow.metadata["termination_status"] = WorkflowTerminationStatus.SKIP_TO_REFLECT
                    self.workflow.metadata["termination_reason"] = "schema_insufficient_coverage"

            # Emit validation result
            yield self._create_validation_action(validation_result, task, hard_block)

            logger.info(
                f"Schema validation completed: is_sufficient={validation_result['is_sufficient']}, "
                f"table_count={validation_result['table_count']}, "
                f"coverage_score={validation_result['coverage_score']:.2f}"
            )

        except Exception as e:
            yield self._handle_execution_error(e)

    async def _validate_workflow_prerequisites(self) -> AsyncGenerator[ActionHistory, None]:
        """Check if workflow and task are available for validation."""
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
                messages=f"Schema validation failed: {error_result.error}",
                action_type="schema_validation",
                input={},
                status=ActionStatus.FAILED,
                output={"error": error_result.error, "error_code": error_result.error_code},
            )
            return
        # Yield a success marker to indicate validation passed
        yield ActionHistory(
            action_id=f"{self.id}_prereq_ok",
            role=ActionRole.TOOL,
            messages="Workflow prerequisites validated",
            action_type="schema_validation_prereq",
            input={},
            status=ActionStatus.SUCCESS,
            output={},
        )

    async def _handle_no_schemas_failure(self, task) -> ActionHistory:
        """
        Handle critical failure when no schemas are discovered.

        This is a HARD failure - no reflection possible. Sets last_action_status
        to FAILED and generates comprehensive diagnostic report.
        """
        self.last_action_status = ActionStatus.FAILED

        # Gather diagnostic information
        diagnostics = self.diagnostic_reporter.gather_diagnostics(task, self.agent_config, self.workflow)

        # Log enhanced error messages
        self.diagnostic_reporter.log_schema_failure(diagnostics)

        # Generate and store diagnostic report
        diagnostic_report = self.diagnostic_reporter.create_diagnostic_report(diagnostics)
        self.diagnostic_reporter.store_diagnostic_report(self.workflow, diagnostic_report)

        # Build failure result
        no_schemas_result = {
            "is_sufficient": False,
            "error": "No schemas discovered",
            "missing_tables": ["all"],
            "suggestions": [
                "Re-run migration with --import-schemas flag",
                "Or run: python -m datus.storage.schema_metadata.local_init",
            ],
            "allow_reflection": False,  # Unrecoverable - no reflection
            "diagnostic_report": diagnostic_report,
            "diagnostics_provided": True,
        }

        self.result = BaseResult(
            success=False,
            error="No schemas discovered for validation",
            data=no_schemas_result,
        )

        return ActionHistory(
            action_id=f"{self.id}_no_schemas",
            role=ActionRole.TOOL,
            messages="Schema validation failed: No schemas discovered",
            action_type="schema_validation",
            input={"task": task.task[:50] if task else ""},
            status=ActionStatus.FAILED,
            output=no_schemas_result,
        )

    async def _validate_schema_coverage(self, task, context) -> Dict[str, Any]:
        """
        Validate schema completeness and coverage against query terms.

        Returns validation result with coverage score and recommendations.
        """
        # Initialize validator
        coverage_validator = CoverageValidator(self.agent_config, self.workflow)

        # Check schema completeness
        table_count = len(context.table_schemas)
        missing_definitions = coverage_validator.find_missing_definitions(context.table_schemas)

        # Extract query terms and check coverage
        query_text = task.task if task else ""
        if self.workflow and hasattr(self.workflow, "metadata") and self.workflow.metadata:
            clarified_task = self.workflow.metadata.get("clarified_task")
            if clarified_task:
                query_text = clarified_task
        intent_terms = self._extract_intent_terms_from_metadata()
        query_terms = await self._extract_query_terms(query_text, protected_terms=intent_terms)
        query_terms = self.term_processor.merge_terms(intent_terms, query_terms)
        schema_coverage = coverage_validator.check_schema_coverage(context.table_schemas, query_terms)

        # Calculate dynamic coverage threshold
        coverage_threshold = self.term_processor.calculate_coverage_threshold(
            query_terms, SCHEMA_VALIDATION_CONFIG
        )
        logger.info(f"Using dynamic coverage threshold: {coverage_threshold:.2f} for {len(query_terms)} query terms")

        invalid_definitions = schema_coverage.get("invalid_tables", [])
        if invalid_definitions:
            missing_definitions = sorted(set(missing_definitions + invalid_definitions))

        critical_terms = intent_terms
        critical_covered = self.term_processor.intersection_terms(
            critical_terms, schema_coverage.get("covered_terms", [])
        )
        critical_uncovered = self.term_processor.difference_terms(critical_terms, critical_covered)
        critical_threshold = self.term_processor.calculate_critical_threshold(critical_terms)
        critical_coverage_score = (
            len(critical_covered) / len(critical_terms) if critical_terms else 1.0
        )

        # Determine if schemas are sufficient
        is_sufficient = (
            table_count > 0
            and len(missing_definitions) == 0
            and schema_coverage["coverage_score"] > coverage_threshold
            and critical_coverage_score >= critical_threshold
        )

        # Build validation result
        validation_result = {
            "is_sufficient": is_sufficient,
            "table_count": table_count,
            "missing_definitions": missing_definitions,
            "query_terms": query_terms,
            "coverage_score": schema_coverage["coverage_score"],
            "coverage_threshold": coverage_threshold,
            "covered_terms": schema_coverage["covered_terms"],
            "uncovered_terms": schema_coverage["uncovered_terms"],
            "critical_terms": critical_terms,
            "critical_terms_covered": critical_covered,
            "critical_terms_uncovered": critical_uncovered,
            "critical_coverage_score": critical_coverage_score,
            "critical_coverage_threshold": critical_threshold,
            "term_evidence": schema_coverage.get("term_evidence", {}),
            "table_coverage": schema_coverage.get("table_coverage", {}),
            "invalid_definitions": invalid_definitions,
        }

        # Add recommendations if insufficient
        if not is_sufficient:
            self.diagnostic_reporter.add_insufficient_schema_recommendations(validation_result, schema_coverage)
            self.last_action_status = ActionStatus.SOFT_FAILED

        return validation_result

    def _extract_intent_terms_from_metadata(self) -> List[str]:
        """Extract business terms, metrics, and dimensions from intent clarification metadata."""
        if not self.workflow or not hasattr(self.workflow, "metadata") or not self.workflow.metadata:
            return []
        clarification = self.workflow.metadata.get("intent_clarification", {})
        if not isinstance(clarification, dict):
            return []
        entities = clarification.get("entities", {})
        if not isinstance(entities, dict):
            return []
        terms: List[str] = []
        for key in ("business_terms", "metrics", "dimensions"):
            values = entities.get(key, []) or []
            if isinstance(values, list):
                terms.extend([str(v) for v in values if isinstance(v, (str, int, float))])
        return self.term_processor.dedupe_terms(terms)

    async def _extract_query_terms(self, query: str, protected_terms: Optional[List[str]] = None) -> List[str]:
        """
        Extract key terms from the query for schema matching using LLM.

        ✅ Fixed: Uses LLMMixin for automatic retry and configured prompt template.
        """
        if not query:
            return []

        # ✅ Use configured prompt template
        prompt_template = LLM_TERM_EXTRACTION_CONFIG.get("prompt_template", "")
        prompt = prompt_template.format(query=query)

        try:
            # ✅ Use LLMMixin with retry and caching
            # Use stable SHA256 hash instead of built-in hash() for cross-process consistency
            query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
            cache_key = f"llm_term_extraction:{query_hash}"
            max_retries = LLM_TERM_EXTRACTION_CONFIG.get("max_retries", 3)
            cache_enabled = LLM_TERM_EXTRACTION_CONFIG.get("cache_enabled", True)
            cache_ttl_seconds = LLM_TERM_EXTRACTION_CONFIG.get("cache_ttl_seconds") if cache_enabled else None

            response = await self.llm_call_with_retry(
                prompt=prompt,
                operation_name="term_extraction",
                cache_key=cache_key if cache_enabled else None,
                max_retries=max_retries,
                cache_ttl_seconds=cache_ttl_seconds,
            )

            terms = response.get("terms", [])

            # Ensure all terms are strings and remove duplicates
            cleaned_terms = list(set([str(t) for t in terms if isinstance(t, (str, int, float))]))
            cleaned_terms = self.term_processor.filter_generic_terms(cleaned_terms, protected_terms=protected_terms)

            logger.info(f"LLM extracted terms for query '{query}': {cleaned_terms}")
            return cleaned_terms

        except NodeExecutionResult as e:
            logger.warning(f"LLM term extraction failed after retries: {e.error_message}. Fallback to regex.")
            return self.term_processor.fallback_term_extraction(query, protected_terms=protected_terms)
        except Exception as e:
            logger.warning(f"LLM term extraction failed with unexpected error: {e}. Fallback to regex.")
            return self.term_processor.fallback_term_extraction(query, protected_terms=protected_terms)

    def _create_validation_action(
        self, validation_result: Dict[str, Any], task, hard_block: bool = False
    ) -> ActionHistory:
        """Create appropriate action based on validation result."""
        table_count = validation_result["table_count"]
        coverage_score = validation_result["coverage_score"]
        candidate_tables_count = 0
        if self.workflow and hasattr(self.workflow, "metadata") and self.workflow.metadata:
            candidate_tables_count = len(self.workflow.metadata.get("discovered_tables", []))
        query_context = self._build_query_context(task)

        if validation_result["is_sufficient"]:
            return ActionHistory(
                action_id=f"{self.id}_validation",
                role=ActionRole.TOOL,
                messages=f"Schema validation passed: {table_count} tables with sufficient coverage",
                action_type="schema_validation",
                input={
                    "task": task.task[:50] if task else "",
                    "table_count": table_count,
                    "candidate_tables_count": candidate_tables_count,
                    "catalog": getattr(task, "catalog_name", ""),
                    "database": getattr(task, "database_name", ""),
                    "schema": getattr(task, "schema_name", "") or None,
                    "coverage_threshold": validation_result.get("coverage_threshold"),
                    "query_context": query_context,
                },
                status=ActionStatus.SUCCESS,
                output=validation_result,
            )
        else:
            return ActionHistory(
                action_id=f"{self.id}_validation",
                role=ActionRole.TOOL,
                messages=(
                    "Schema validation hard-blocked: insufficient coverage"
                    if hard_block
                    else f"Schema validation failed: Insufficient schema coverage (score: {coverage_score:.2f})"
                ),
                action_type="schema_validation",
                input={
                    "task": task.task[:50] if task else "",
                    "table_count": table_count,
                    "candidate_tables_count": candidate_tables_count,
                    "catalog": getattr(task, "catalog_name", ""),
                    "database": getattr(task, "database_name", ""),
                    "schema": getattr(task, "schema_name", "") or None,
                    "coverage_threshold": validation_result.get("coverage_threshold"),
                    "query_context": query_context,
                },
                status=ActionStatus.FAILED if hard_block else ActionStatus.SOFT_FAILED,
                output=validation_result,
            )

    def _build_query_context(self, task) -> Dict[str, Any]:
        """Build query context from task and workflow metadata."""
        context: Dict[str, Any] = {"task": task.task if task else ""}
        if self.workflow and hasattr(self.workflow, "metadata") and self.workflow.metadata:
            clarified_task = self.workflow.metadata.get("clarified_task")
            if clarified_task:
                context["clarified_task"] = clarified_task
            clarification = self.workflow.metadata.get("intent_clarification", {})
            if isinstance(clarification, dict):
                entities = clarification.get("entities", {})
                if entities:
                    context["business_terms"] = entities.get("business_terms", [])
                    context["dimensions"] = entities.get("dimensions", [])
                    context["metrics"] = entities.get("metrics", [])
                    context["time_range"] = entities.get("time_range")
        return context

    def _handle_execution_error(self, error: Exception) -> ActionHistory:
        """Handle unexpected execution errors."""
        logger.error(f"Schema validation failed: {error}")
        error_result = self.create_error_result(
            ErrorCode.NODE_EXECUTION_FAILED,
            f"Schema validation execution failed: {str(error)}",
            "schema_validation",
            {
                "task_id": (
                    getattr(self.workflow.task, "id", "unknown") if self.workflow and self.workflow.task else "unknown"
                )
            },
        )

        self.result = BaseResult(success=False, error=str(error))

        return ActionHistory(
            action_id=f"{self.id}_error",
            role=ActionRole.TOOL,
            messages=f"Schema validation failed: {error_result.error}",
            action_type="schema_validation",
            input={},
            status=ActionStatus.FAILED,
            output={"error": error_result.error, "error_code": error_result.error_code},
        )

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
            if not self.result:
                return {"success": False, "message": "Schema validation failed, cannot update context"}

            # Store validation result in workflow metadata
            if not hasattr(workflow, "metadata"):
                workflow.metadata = {}
            workflow.metadata["schema_validation"] = self.result.data

            if not self.result.success:
                return {"success": False, "message": "Schema validation failed, stored validation result"}

            return {
                "success": True,
                "message": f"Schema validation context updated with result: {self.result.data.get('is_sufficient', False)}",
            }

        except Exception as e:
            logger.error(f"Failed to update schema validation context: {str(e)}")
            return {"success": False, "message": f"Schema validation context update failed: {str(e)}"}
