#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
LLM-based final refinement for business terms.

Goal: remove low-quality terms that harm schema discovery.
"""

import hashlib
from typing import Dict, Iterable, List, Optional, Set, Tuple

from datus.models.base import LLMBaseModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class LLMTermRefiner:
    """Use LLM to filter out low-quality business terms."""

    def __init__(
        self,
        agent_config=None,
        batch_size: int = 80,
        cache_enabled: bool = True,
    ):
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, List[str]] = {}

        self.llm_model = None
        if agent_config:
            try:
                self.llm_model = LLMBaseModel.create_model(agent_config=agent_config)
                logger.info("LLM term refiner initialized")
            except Exception as e:
                logger.warning(f"LLM term refiner init failed: {e}")

    def refine(self, business_terms: Dict) -> Dict:
        """Refine term_to_table / term_to_schema / table_keywords."""
        if not self.llm_model:
            logger.warning("LLM term refiner disabled (no model).")
            return business_terms

        term_to_table = business_terms.get("term_to_table", {})
        term_to_schema = business_terms.get("term_to_schema", {})
        table_keywords = business_terms.get("table_keywords", {})

        term_to_table = self._filter_term_map("table_terms", term_to_table)
        term_to_schema = self._filter_term_map("schema_terms", term_to_schema)
        table_keywords = self._filter_term_map("table_keywords", table_keywords)

        return {
            "term_to_table": term_to_table,
            "term_to_schema": term_to_schema,
            "table_keywords": table_keywords,
            "_stats": business_terms.get("_stats", {}),
        }

    def _filter_term_map(self, category: str, term_map: Dict) -> Dict:
        terms = list(term_map.keys())
        if not terms:
            return term_map

        keep_terms: Set[str] = set()
        for batch in self._batch(terms, self.batch_size):
            kept = self._llm_filter_terms(category, batch, term_map)
            keep_terms.update(kept)

        filtered = {t: term_map[t] for t in terms if t in keep_terms}
        logger.info(f"LLM final filter ({category}): {len(filtered)}/{len(terms)} kept")
        return filtered

    def _llm_filter_terms(self, category: str, terms: List[str], term_map: Dict) -> List[str]:
        cache_key = self._cache_key(category, terms)
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        samples = self._build_samples(terms, term_map, max_values=3)
        prompt = self._build_prompt(category, samples)

        try:
            response = self.llm_model.generate_with_json_output(prompt)
            keep = response.get("keep", []) if isinstance(response, dict) else []
            keep = [t for t in keep if t in terms]
        except Exception as e:
            logger.warning(f"LLM final filter failed ({category}): {e}")
            keep = terms  # fallback: keep all

        if self.cache_enabled:
            self._cache[cache_key] = keep

        return keep

    def _build_samples(self, terms: List[str], term_map: Dict, max_values: int = 3) -> List[Dict]:
        samples = []
        for term in terms:
            values = term_map.get(term, [])
            if isinstance(values, dict):
                values = list(values.keys())
            elif isinstance(values, set):
                values = list(values)
            elif not isinstance(values, list):
                values = [str(values)]
            samples.append({"term": term, "values": values[:max_values]})
        return samples

    def _build_prompt(self, category: str, samples: List[Dict]) -> str:
        return f"""You are a data warehouse analyst. Filter business terms for Text2SQL schema discovery.

Category: {category}

Rules (keep only high-quality terms):
1. Keep concise business entities, metrics, dimensions, or domain concepts.
2. Remove sentence fragments, functional words, generic words (e.g., 信息/时间/是否/从...).
3. Remove overly technical tokens (id/code/status/time) unless clearly business concepts.
4. Prefer terms users would naturally type in queries.

Input (term + sample mappings):
{samples}

Return JSON ONLY:
{{"keep": ["term1", "term2", ...]}}
"""

    def _batch(self, items: List[str], batch_size: int) -> Iterable[List[str]]:
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def _cache_key(self, category: str, terms: List[str]) -> str:
        text = category + ":" + "|".join(terms)
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
