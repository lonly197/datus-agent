# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Diagnostic reporting and error handling for schema validation.

This module provides comprehensive diagnostic capabilities for schema
validation failures, including structured reports and actionable recommendations.
"""

from datetime import datetime
from typing import Any, Dict, List

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DiagnosticReporter:
    """
    Handles diagnostic reporting for schema validation failures.

    Provides methods to gather diagnostic information, log failures,
    create structured reports, and handle execution errors.
    """

    @staticmethod
    def gather_diagnostics(task, agent_config, workflow) -> Dict[str, Any]:
        """
        Collect diagnostic information for schema failure.

        Args:
            task: Workflow task object
            agent_config: Agent configuration
            workflow: Workflow object with metadata

        Returns:
            Dictionary with diagnostic information
        """
        database_name = getattr(task, "database_name", "unknown")
        namespace = getattr(agent_config, "current_namespace", "unknown") if agent_config else "unknown"

        # Get candidate_tables from workflow metadata
        candidate_tables = []
        if workflow and hasattr(workflow, "metadata") and workflow.metadata:
            candidate_tables = workflow.metadata.get("discovered_tables", [])

        return {
            "database_name": database_name,
            "namespace": namespace,
            "candidate_tables_count": len(candidate_tables),
            "task": task.task if task else "unknown",
        }

    @staticmethod
    def log_schema_failure(diagnostics: Dict[str, Any]) -> None:
        """
        Log comprehensive schema failure diagnostics.

        Args:
            diagnostics: Diagnostic information dictionary
        """
        logger.error("")
        logger.error("=" * 80)
        logger.error("SCHEMA VALIDATION FAILED: NO SCHEMAS DISCOVERED")
        logger.error("=" * 80)
        logger.error("")
        logger.error("Root Cause Analysis:")
        logger.error("  • LanceDB schema storage is empty (no schema metadata found)")
        logger.error("  • DDL fallback also failed to retrieve schemas from database")
        logger.error(
            f"  • Found {diagnostics['candidate_tables_count']} candidate tables, but none matched stored schemas"
        )
        logger.error("")
        logger.error("Possible Causes:")
        logger.error("  1. Schema import was not run after LanceDB v1 migration")
        logger.error("  2. Database connection parameters are incorrect")
        logger.error("  3. Database is empty (no tables exist)")
        logger.error("  4. Namespace/database name mismatch")
        logger.error("")
        logger.error("Immediate Actions:")
        logger.error(f"  1. Re-run migration with schema import:")
        logger.error(f"     python -m datus.storage.schema_metadata.migrate_v0_to_v1 \\")
        logger.error(f"       --config=<config_path> --namespace={diagnostics['namespace']} \\")
        logger.error(f"       --import-schemas")
        logger.error("")
        logger.error("  2. Or run schema import separately:")
        logger.error(f"     python -m datus.storage.schema_metadata.local_init \\")
        logger.error(f"       --config=<config_path> --namespace={diagnostics['namespace']}")
        logger.error("")
        logger.error("  3. Verify database connection:")
        logger.error(f'     python -c "')
        logger.error(f"       from datus.tools.db_tools.db_manager import get_db_manager")
        logger.error(
            f"       db = get_db_manager().get_conn('{diagnostics['namespace']}', '{diagnostics['database_name']}')"
        )
        logger.error(f"       print('Tables:', len(db.get_tables_with_ddl()))")
        logger.error(f'     "')
        logger.error("=" * 80)
        logger.error("")

    @staticmethod
    def create_diagnostic_report(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create structured diagnostic report for schema failure.

        Args:
            diagnostics: Diagnostic information dictionary

        Returns:
            Structured diagnostic report with sections
        """
        return {
            "report_type": "Schema Discovery Failure Report",
            "timestamp": datetime.now().isoformat(),
            "task": diagnostics["task"],
            "database_name": diagnostics["database_name"],
            "namespace": diagnostics["namespace"],
            "candidate_tables_count": diagnostics["candidate_tables_count"],
            "sections": [
                {
                    "title": "1. Schema Discovery Status",
                    "status": "FAILED",
                    "findings": {
                        "lancedb_storage": "empty - no schema metadata found",
                        "ddl_fallback": "returned 0 tables from database",
                        "candidate_tables_found": diagnostics["candidate_tables_count"],
                    },
                },
                {
                    "title": "2. Root Cause Analysis",
                    "possible_causes": [
                        "Schema import was not run after LanceDB v1 migration",
                        "Database connection parameters are incorrect",
                        "Database is empty (no tables exist)",
                        "Namespace/database name mismatch",
                    ],
                },
                {
                    "title": "3. Immediate Actions Required",
                    "steps": [
                        "Re-run migration with --import-schemas flag",
                        "Or run schema import separately",
                        "Verify database connection and table existence",
                    ],
                    "commands": [
                        f"python -m datus.storage.schema_metadata.migrate_v0_to_v1 --config=<config> --namespace={diagnostics['namespace']} --import-schemas",
                        f"python -c \"from datus.tools.db_tools.db_manager import get_db_manager; db = get_db_manager().get_conn('{diagnostics['namespace']}', '{diagnostics['database_name']}'); print(f'Tables: {{len(db.get_tables_with_ddl())}}')\"",
                    ],
                },
                {
                    "title": "4. SQL Generated (May Contain Hallucinated Tables)",
                    "sql_query": "No SQL generated (validation failed before SQL generation)",
                    "warning": "SQL was generated without schema context - table names may be incorrect",
                },
                {
                    "title": "5. Next Steps",
                    "recommendations": [
                        "Import schema metadata using one of the commands above",
                        "Re-run text2sql workflow after schema import completes",
                        "Contact administrator if schema import fails",
                    ],
                },
            ],
        }

    @staticmethod
    def store_diagnostic_report(workflow, report: Dict[str, Any]) -> None:
        """
        Store diagnostic report in workflow metadata.

        Args:
            workflow: Workflow object to store report in
            report: Diagnostic report dictionary
        """
        if workflow:
            if not hasattr(workflow, "metadata"):
                workflow.metadata = {}
            workflow.metadata["schema_discovery_failure_report"] = report

    @staticmethod
    def add_insufficient_schema_recommendations(
        validation_result: Dict[str, Any], schema_coverage: Dict[str, Any]
    ) -> None:
        """
        Add actionable recommendations when schema validation fails.

        Args:
            validation_result: Validation result dictionary to update
            schema_coverage: Schema coverage information
        """
        if validation_result["missing_definitions"]:
            validation_result["suggestions"] = [
                f"Load full DDL for tables: {', '.join(validation_result['missing_definitions'][:5])}"
            ]
        elif validation_result.get("critical_terms_uncovered"):
            validation_result["suggestions"] = [
                "Ensure schema includes key business terms before SQL generation",
                f"Uncovered critical terms: {', '.join(validation_result['critical_terms_uncovered'][:5])}",
            ]
            validation_result["missing_tables"] = validation_result["critical_terms_uncovered"][:5]
        elif schema_coverage["coverage_score"] < 0.3:
            validation_result["suggestions"] = [
                "Use enhanced schema_discovery (with progressive matching and LLM inference) to find matching tables",
                f"Consider tables matching terms: {', '.join(schema_coverage['uncovered_terms'][:5])}",
            ]
            validation_result["missing_tables"] = schema_coverage["uncovered_terms"][:5]

        # Enable workflow reflection for recovery
        validation_result["allow_reflection"] = True
