# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Schema coverage validation logic.

This module provides validation capabilities for checking schema completeness
and coverage against query terms, including semantic matching via business
term mappings and external knowledge stores.
"""

import re
from typing import Any, Dict, List, Optional

from datus.configuration.business_term_config import get_schema_term_mapping
from datus.configuration.business_term_config import SCHEMA_VALIDATION_CONFIG
from datus.schemas.node_models import TableSchema
from datus.storage.cache import get_storage_cache_instance
from datus.utils.sql_utils import (
    extract_enhanced_metadata_from_ddl,
    parse_dialect,
    sanitize_ddl_for_storage,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CoverageValidator:
    """
    Validates schema coverage against query terms.

    Provides methods to check if discovered schemas contain sufficient
    information to generate SQL for a given query, including semantic
    matching via business term mappings and external knowledge stores.
    """

    def __init__(self, agent_config, workflow):
        """
        Initialize CoverageValidator.

        Args:
            agent_config: Agent configuration object
            workflow: Workflow object with metadata
        """
        self.agent_config = agent_config
        self.workflow = workflow

    @staticmethod
    def find_missing_definitions(schemas: List[TableSchema]) -> List[str]:
        """
        Find tables with missing or empty DDL definitions.

        Args:
            schemas: List of table schemas to check

        Returns:
            List of table names with missing definitions
        """
        return [schema.table_name for schema in schemas if not schema.definition or schema.definition.strip() == ""]

    def check_schema_coverage(self, schemas: List[TableSchema], query_terms: List[str]) -> Dict[str, Any]:
        """
        Check how well the schemas cover the query terms.

        Uses semantic matching via business term mapping to enable Chinese-to-English
        schema matching. This allows queries with Chinese business terminology to
        match English database schema names.

        Also uses External Knowledge for terminology lookup and extracts Chinese DDL
        comments for better Chinese query matching.

        Args:
            schemas: List of table schemas to check
            query_terms: List of query terms to match

        Returns:
            Dictionary with coverage score, covered/uncovered terms, and evidence
        """
        if not query_terms:
            logger.debug("No query terms provided, returning perfect coverage")
            return {
                "coverage_score": 1.0,
                "covered_terms": [],
                "uncovered_terms": [],
                "term_evidence": {},
                "table_coverage": {},
            }

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
        term_evidence: Dict[str, Dict[str, Any]] = {}
        invalid_tables: List[str] = []

        # Build a set of all schema terms (table names, column names, AND comments)
        schema_terms = set()
        schema_tokens = set()
        comment_terms = {}  # Track which schema provided each comment term
        term_sources: Dict[str, List[Dict[str, Any]]] = {}

        def add_term_source(term: str, table_name: str, source: str, column_name: Optional[str] = None) -> None:
            """Add a term source to the tracking dictionaries."""
            term_lower = term.lower()
            schema_terms.add(term_lower)
            for token in self._tokenize_schema_term(term_lower):
                schema_tokens.add(token)
            term_sources.setdefault(term_lower, [])
            term_sources[term_lower].append(
                {
                    "table": table_name,
                    "source": source,
                    "column": column_name,
                }
            )

        dialect = parse_dialect(
            getattr(self.workflow.task, "database_type", "") if self.workflow and self.workflow.task else ""
        )

        for schema in schemas:
            # Add table name
            add_term_source(schema.table_name, schema.table_name, "table_name")

            # Add column names from definition
            if schema.definition:
                definition = sanitize_ddl_for_storage(schema.definition)
                parsed_metadata = extract_enhanced_metadata_from_ddl(
                    definition,
                    dialect=dialect,
                    warn_on_invalid=False,
                )

                # Extract column names from parsed metadata
                parsed_columns = [col.get("name") for col in parsed_metadata.get("columns", []) if col.get("name")]
                if not parsed_columns:
                    parsed_columns = re.findall(r"`(\w+)`", definition)
                if not parsed_columns:
                    invalid_tables.append(schema.table_name)
                for col_name in parsed_columns:
                    add_term_source(col_name, schema.table_name, "column_name", column_name=col_name)

                # Extract table comment
                table_comment = parsed_metadata.get("table", {}).get("comment", "")
                if table_comment:
                    add_term_source(table_comment, schema.table_name, "table_comment")
                    comment_terms[table_comment.lower()] = f"table:{schema.table_name}"
                    logger.debug(f"Extracted table comment '{table_comment}' from {schema.table_name}")

                # Extract column comments
                for col in parsed_metadata.get("columns", []):
                    col_comment = col.get("comment", "")
                    col_name = col.get("name") or "unknown"
                    if col_comment:
                        add_term_source(
                            col_comment,
                            schema.table_name,
                            "column_comment",
                            column_name=col_name,
                        )
                        comment_terms[col_comment.lower()] = f"column:{schema.table_name}.{col_name}"
                        logger.debug(
                            f"Extracted column comment '{col_comment}' from {schema.table_name}.{col_name}"
                        )

        logger.debug(f"Built schema terms set with {len(schema_terms)} unique terms from {len(schemas)} schemas")

        # Check each query term with semantic mapping
        for term in query_terms:
            term_lower = term.lower()

            # Direct match (case-insensitive) - including comments
            if term_lower in schema_terms:
                # Log if matched via comment
                if term_lower in comment_terms:
                    logger.debug(f"Term '{term}' matched via {comment_terms[term_lower]} comment")
                else:
                    logger.debug(f"Term '{term}' matched directly (case-insensitive)")
                covered.append(term)
                term_evidence[term] = {
                    "match_type": "direct",
                    "matched_terms": [term_lower],
                    "sources": term_sources.get(term_lower, []),
                }
                continue

            # Semantic match via centralized business term mapping (Chinese → English)
            english_terms = get_schema_term_mapping(term)
            if english_terms:
                # Check if any of the mapped English terms are in the schema
                matched_english = []
                for eng in english_terms:
                    eng_lower = eng.lower()
                    if eng_lower in schema_terms:
                        matched_english.append(eng)
                        continue
                    if eng_lower in schema_tokens:
                        matched_english.append(eng)
                        continue
                    if any(eng_lower in schema_term for schema_term in schema_terms):
                        matched_english.append(eng)
                if matched_english:
                    logger.debug(f"Term '{term}' matched via semantic mapping: {english_terms}")
                    covered.append(term)
                    matched_sources = []
                    for eng_term in matched_english:
                        eng_term_lower = eng_term.lower()
                        matched_sources.extend(term_sources.get(eng_term_lower, []))
                        for schema_term, sources in term_sources.items():
                            if eng_term_lower in schema_term:
                                matched_sources.extend(sources)
                    term_evidence[term] = {
                        "match_type": "semantic",
                        "matched_terms": matched_english,
                        "sources": matched_sources,
                    }
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
                                    term_evidence[term] = {
                                        "match_type": "external_knowledge",
                                        "matched_terms": [term_lower],
                                        "sources": [{"table": schema.table_name, "source": "external_knowledge"}],
                                    }
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
                        term_evidence[term] = {
                            "match_type": "partial",
                            "matched_terms": [schema_term],
                            "sources": term_sources.get(schema_term, []),
                        }
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

        table_coverage: Dict[str, List[str]] = {}
        for term, evidence in term_evidence.items():
            for source in evidence.get("sources", []):
                table = source.get("table")
                if not table:
                    continue
                table_coverage.setdefault(table, [])
                if term not in table_coverage[table]:
                    table_coverage[table].append(term)

        return {
            "coverage_score": coverage_score,
            "covered_terms": covered,
            "uncovered_terms": uncovered,
            "term_evidence": term_evidence,
            "table_coverage": table_coverage,
            "invalid_tables": invalid_tables,
        }

    @staticmethod
    def _tokenize_schema_term(term: str) -> List[str]:
        """
        Tokenize a schema term into individual components.

        Splits on non-alphanumeric characters to handle compound terms.

        Args:
            term: Schema term to tokenize

        Returns:
            List of tokens
        """
        if not term:
            return []
        tokens = re.split(r"[^a-zA-Z0-9]+", term.lower())
        return [t for t in tokens if t]
