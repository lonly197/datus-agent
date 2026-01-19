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
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.configuration.business_term_config import (
    LLM_TERM_EXTRACTION_CONFIG,
    SCHEMA_VALIDATION_CONFIG,
    get_schema_term_mapping,
)
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import TableSchema
from datus.storage.cache import get_storage_cache_instance
from datus.utils.error_handler import LLMMixin, NodeExecutionResult
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

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
                    messages=f"Schema validation failed: {error_result.error}",
                    action_type="schema_validation",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error, "error_code": error_result.error_code},
                )
                return

            task = self.workflow.task
            context = self.workflow.context

            # Step 1: Check if schemas were discovered
            if not context or not context.table_schemas:
                # CRITICAL: No schemas available - this is a HARD failure
                # Set last_action_status to FAILED (not SOFT_FAILED)
                self.last_action_status = ActionStatus.FAILED

                # Get database information for diagnostics
                database_name = getattr(task, "database_name", "unknown")
                namespace = getattr(self.agent_config, "current_namespace", "unknown") if self.agent_config else "unknown"
                candidate_tables_count = len(context.get("candidate_tables", [])) if context else 0

                # Enhanced logging before hard termination
                logger.error("")
                logger.error("=" * 80)
                logger.error("SCHEMA VALIDATION FAILED: NO SCHEMAS DISCOVERED")
                logger.error("=" * 80)
                logger.error("")
                logger.error("Root Cause Analysis:")
                logger.error("  • LanceDB schema storage is empty (no schema metadata found)")
                logger.error("  • DDL fallback also failed to retrieve schemas from database")
                logger.error(f"  • Found {candidate_tables_count} candidate tables, but none matched stored schemas")
                logger.error("")
                logger.error("Possible Causes:")
                logger.error("  1. Schema import was not run after LanceDB v1 migration")
                logger.error("  2. Database connection parameters are incorrect")
                logger.error("  3. Database is empty (no tables exist)")
                logger.error("  4. Namespace/database name mismatch")
                logger.error("")
                logger.error("Immediate Actions:")
                logger.error("  1. Re-run migration with schema import:")
                logger.error(f"     python -m datus.storage.schema_metadata.migrate_v0_to_v1 \\")
                logger.error(f"       --config=<config_path> --namespace={namespace} \\")
                logger.error(f"       --import-schemas")
                logger.error("")
                logger.error("  2. Or run schema import separately:")
                logger.error(f"     python -m datus.storage.schema_metadata.local_init \\")
                logger.error(f"       --config=<config_path> --namespace={namespace}")
                logger.error("")
                logger.error("  3. Verify database connection:")
                logger.error(f"     python -c \"")
                logger.error(f"       from datus.tools.db_tools.db_manager import get_db_manager")
                logger.error(f"       db = get_db_manager().get_conn('{namespace}', '{database_name}')")
                logger.error(f"       print('Tables:', len(db.get_tables_with_ddl()))")
                logger.error(f"     \"")
                logger.error("=" * 80)
                logger.error("")

                # Generate diagnostic report before termination
                diagnostic_report = {
                    "report_type": "Schema Discovery Failure Report",
                    "timestamp": datetime.now().isoformat(),
                    "task": task.task if task else "unknown",
                    "database_name": database_name,
                    "namespace": namespace,
                    "candidate_tables_count": candidate_tables_count,
                    "sections": [
                        {
                            "title": "1. Schema Discovery Status",
                            "status": "FAILED",
                            "findings": {
                                "lancedb_storage": "empty - no schema metadata found",
                                "ddl_fallback": "returned 0 tables from database",
                                "candidate_tables_found": candidate_tables_count,
                            }
                        },
                        {
                            "title": "2. Root Cause Analysis",
                            "possible_causes": [
                                "Schema import was not run after LanceDB v1 migration",
                                "Database connection parameters are incorrect",
                                "Database is empty (no tables exist)",
                                "Namespace/database name mismatch"
                            ]
                        },
                        {
                            "title": "3. Immediate Actions Required",
                            "steps": [
                                "Re-run migration with --import-schemas flag",
                                "Or run schema import separately",
                                "Verify database connection and table existence"
                            ],
                            "commands": [
                                f"python -m datus.storage.schema_metadata.migrate_v0_to_v1 --config=<config> --namespace={namespace} --import-schemas",
                                f"python -c \"from datus.tools.db_tools.db_manager import get_db_manager; db = get_db_manager().get_conn('{namespace}', '{database_name}'); print(f'Tables: {{len(db.get_tables_with_ddl())}}')\""
                            ]
                        },
                        {
                            "title": "4. SQL Generated (May Contain Hallucinated Tables)",
                            "sql_query": context.get("sql_query", "No SQL generated") if context else "No SQL generated",
                            "warning": "SQL was generated without schema context - table names may be incorrect"
                        },
                        {
                            "title": "5. Next Steps",
                            "recommendations": [
                                "Import schema metadata using one of the commands above",
                                "Re-run text2sql workflow after schema import completes",
                                "Contact administrator if schema import fails"
                            ]
                        }
                    ]
                }

                # Store report in workflow metadata for event converter
                if self.workflow:
                    if not hasattr(self.workflow, "metadata"):
                        self.workflow.metadata = {}
                    self.workflow.metadata["schema_discovery_failure_report"] = diagnostic_report

                no_schemas_result = {
                    "is_sufficient": False,
                    "error": "No schemas discovered",
                    "missing_tables": ["all"],
                    "suggestions": [
                        "Re-run migration with --import-schemas flag",
                        "Or run: python -m datus.storage.schema_metadata.local_init",
                    ],
                    "allow_reflection": False,  # No reflection - this is unrecoverable
                    "diagnostic_report": diagnostic_report,
                    "diagnostics_provided": True
                }
                yield ActionHistory(
                    action_id=f"{self.id}_no_schemas",
                    role=ActionRole.TOOL,
                    messages="Schema validation failed: No schemas discovered",
                    action_type="schema_validation",
                    input={"task": task.task[:50] if task else ""},
                    status=ActionStatus.FAILED,  # Use FAILED (not SOFT_FAILED)
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
            query_terms = await self._extract_query_terms(query_lower)

            # Check if schemas contain relevant columns/tables
            schema_coverage = self._check_schema_coverage(context.table_schemas, query_terms)

            # Step 4: Determine if schemas are sufficient using dynamic threshold
            coverage_threshold = self._calculate_coverage_threshold(query_terms)
            logger.info(
                f"Using dynamic coverage threshold: {coverage_threshold:.2f} for {len(query_terms)} query terms"
            )

            is_sufficient = (
                table_count > 0
                and len(missing_definitions) == 0
                and schema_coverage["coverage_score"] > coverage_threshold
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
                        "Use enhanced schema_discovery (with progressive matching and LLM inference) to find matching tables",
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
                # Insufficient but some schemas exist - SOFT failure with reflection
                self.last_action_status = ActionStatus.SOFT_FAILED

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
                    "task_id": (
                        getattr(self.workflow.task, "id", "unknown")
                        if self.workflow and self.workflow.task
                        else "unknown"
                    )
                },
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
            self.result = BaseResult(success=False, error=str(e))

    async def _extract_query_terms(self, query: str) -> List[str]:
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
            cache_key = f"llm_term_extraction:{hash(query)}"
            max_retries = LLM_TERM_EXTRACTION_CONFIG.get("max_retries", 3)

            response = await self.llm_call_with_retry(
                prompt=prompt,
                operation_name="term_extraction",
                cache_key=cache_key,
                max_retries=max_retries,
            )

            terms = response.get("terms", [])

            # Ensure all terms are strings and remove duplicates
            cleaned_terms = list(set([str(t) for t in terms if isinstance(t, (str, int, float))]))

            logger.info(f"LLM extracted terms for query '{query}': {cleaned_terms}")
            return cleaned_terms

        except NodeExecutionResult as e:
            logger.warning(f"LLM term extraction failed after retries: {e.error_message}. Fallback to regex.")
            return self._fallback_term_extraction(query)
        except Exception as e:
            logger.warning(f"LLM term extraction failed with unexpected error: {e}. Fallback to regex.")
            return self._fallback_term_extraction(query)

    def _fallback_term_extraction(self, query: str) -> List[str]:
        """Fallback to simple regex split if LLM fails."""
        words = re.findall(r"\b\w+\b", query)
        # Simple stop word filtering for fallback
        stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "are"}
        return [w for w in words if w.lower() not in stop_words and len(w) > 1]

    def _calculate_coverage_threshold(self, query_terms: List[str]) -> float:
        """
        Calculate dynamic threshold based on query complexity.

        ✅ Fixed: Uses configured thresholds from SCHEMA_VALIDATION_CONFIG.
        """
        term_count = len(query_terms)
        thresholds = SCHEMA_VALIDATION_CONFIG.get("coverage_thresholds", {})

        # Simple queries (≤3 terms) require higher coverage
        if term_count <= thresholds.get("simple", {}).get("max_terms", 3):
            return thresholds.get("simple", {}).get("threshold", 0.5)
        # Medium queries (4-6 terms) use standard coverage
        elif term_count <= thresholds.get("medium", {}).get("max_terms", 6):
            return thresholds.get("medium", {}).get("threshold", 0.3)
        # Complex queries (>6 terms) allow lower coverage
        else:
            return thresholds.get("complex", {}).get("threshold", 0.2)

    def _check_schema_coverage(self, schemas: List[TableSchema], query_terms: List[str]) -> Dict[str, Any]:
        """
        Check how well the schemas cover the query terms.

        ✅ Fixed: Uses centralized business term mapping from config for semantic matching.
        Also uses External Knowledge for terminology lookup.

        Uses semantic matching via business term mapping to enable Chinese-to-English
        schema matching. This allows queries with Chinese business terminology to
        match English database schema names.
        """
        if not query_terms:
            logger.debug("No query terms provided, returning perfect coverage")
            return {"coverage_score": 1.0, "covered_terms": [], "uncovered_terms": []}

        logger.debug(f"Checking schema coverage for {len(query_terms)} query terms: {query_terms}")

        # Initialize ExtKnowledgeStore if available
        ext_knowledge = None
        if self.agent_config:
            try:
                storage_cache = get_storage_cache_instance(self.agent_config)
                ext_knowledge = storage_cache.ext_knowledge_storage()
            except Exception:
                pass

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

        logger.debug(f"Built schema terms set with {len(schema_terms)} unique terms from {len(schemas)} schemas")

        # Check each query term with semantic mapping
        for term in query_terms:
            # Direct match (case-insensitive)
            if term.lower() in schema_terms:
                logger.debug(f"Term '{term}' matched directly (case-insensitive)")
                covered.append(term)
                continue

            # ✅ Semantic match via centralized business term mapping (Chinese → English)
            english_terms = get_schema_term_mapping(term)
            if english_terms:
                # Check if any of the mapped English terms are in the schema
                if any(eng_term.lower() in schema_terms for eng_term in english_terms):
                    logger.debug(f"Term '{term}' matched via semantic mapping: {english_terms}")
                    covered.append(term)
                    continue
                else:
                    logger.debug(f"Term '{term}' has mapping {english_terms} but no match found in schema")

            # External Knowledge Store Match
            is_covered_by_kb = False
            if ext_knowledge:
                try:
                    # Search for the term in knowledge base
                    results = ext_knowledge.search_knowledge(query_text=term, top_n=3)
                    for item in results:
                        if item.get("terminology", "").lower() == term.lower():
                            explanation = item.get("explanation", "").lower()
                            # Check if explanation contains any table names present in schemas
                            for schema in schemas:
                                if schema and schema.table_name.lower() in explanation:
                                    covered.append(term)
                                    is_covered_by_kb = True
                                    logger.info(
                                        f"Term '{term}' covered by ExtKnowledge mapping to table '{schema.table_name}'"
                                    )
                                    break
                        if is_covered_by_kb:
                            break
                except Exception as e:
                    logger.warning(f"ExtKnowledge check failed for term '{term}': {e}")

            if is_covered_by_kb:
                continue

            # Partial match for compound terms (e.g., "首次试驾" contains "试驾")
            if SCHEMA_VALIDATION_CONFIG.get("enable_partial_matching", True):
                found_partial = False
                for schema_term in schema_terms:
                    if term in schema_term or schema_term in term:
                        logger.debug(f"Term '{term}' matched partially with '{schema_term}'")
                        covered.append(term)
                        found_partial = True
                        break

                if not found_partial:
                    logger.debug(f"Term '{term}' not found in schema (uncovered)")
                    uncovered.append(term)
            else:
                logger.debug(f"Term '{term}' not found in schema (uncovered)")
                uncovered.append(term)

        coverage_score = len(covered) / len(query_terms) if query_terms else 1.0

        logger.info(
            f"Schema coverage result: score={coverage_score:.2f}, "
            f"covered={len(covered)}/{len(query_terms)}, "
            f"covered_terms={covered}, "
            f"uncovered_terms={uncovered}"
        )

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
