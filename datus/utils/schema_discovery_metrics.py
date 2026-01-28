# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema Discovery Metrics and Monitoring

This module provides comprehensive metrics tracking for schema discovery:
- Search stage hit rates and accuracy
- External knowledge retrieval success rates
- Query processing performance
- End-to-end discovery statistics

Metrics are structured for easy export to monitoring systems (Prometheus, etc.)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SearchStage(Enum):
    """Schema discovery search stages."""
    EXPLICIT = "explicit"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    LLM_MATCHING = "llm_matching"
    EXTERNAL_KNOWLEDGE = "external_knowledge"
    CONTEXT_SEARCH = "context_search"
    FALLBACK = "fallback"


@dataclass
class StageMetrics:
    """Metrics for a single search stage."""
    stage: SearchStage
    # Hit tracking
    attempted: bool = False
    success: bool = False
    tables_found: int = 0
    tables_returned: int = 0
    
    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Accuracy (if ground truth available)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self):
        """Start timing the stage."""
        self.start_time = time.time()
        self.attempted = True
    
    def end(self, success: bool = True, tables_found: int = 0, tables_returned: int = 0):
        """End timing and record results."""
        self.end_time = time.time()
        self.success = success
        self.tables_found = tables_found
        self.tables_returned = tables_returned
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate (0-1)."""
        if not self.attempted:
            return 0.0
        return 1.0 if self.tables_found > 0 else 0.0
    
    @property
    def precision(self) -> Optional[float]:
        """Calculate precision if accuracy data available."""
        total = self.true_positives + self.false_positives
        if total == 0:
            return None
        return self.true_positives / total
    
    @property
    def recall(self) -> Optional[float]:
        """Calculate recall if accuracy data available."""
        total = self.true_positives + self.false_negatives
        if total == 0:
            return None
        return self.true_positives / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/export."""
        return {
            "stage": self.stage.value,
            "attempted": self.attempted,
            "success": self.success,
            "tables_found": self.tables_found,
            "tables_returned": self.tables_returned,
            "hit_rate": self.hit_rate,
            "duration_ms": self.duration_ms,
            "precision": self.precision,
            "recall": self.recall,
            **self.metadata,
        }


@dataclass
class ExternalKnowledgeMetrics:
    """Metrics for external knowledge retrieval."""
    attempted: bool = False
    success: bool = False
    
    # Search results
    entries_found: int = 0
    entries_used: int = 0
    
    # Timing
    duration_ms: Optional[float] = None
    
    # Search parameters
    query_text: str = ""
    domain: str = ""
    layer1: str = ""
    layer2: str = ""
    
    # Error tracking
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attempted": self.attempted,
            "success": self.success,
            "entries_found": self.entries_found,
            "entries_used": self.entries_used,
            "duration_ms": self.duration_ms,
            "success_rate": self.entries_used / max(self.entries_found, 1),
            "has_error": bool(self.error_type),
        }


@dataclass
class SemanticSearchMetrics:
    """Detailed metrics for semantic search stage."""
    # Query info
    original_query: str = ""
    enhanced_query: str = ""
    
    # Search configuration
    similarity_threshold: float = 0.0
    top_n_requested: int = 0
    top_n_returned: int = 0
    
    # Results breakdown
    vector_hits: int = 0
    fts_hits: int = 0
    rerank_hits: int = 0
    final_tables: int = 0
    
    # Hybrid search weights
    vector_weight: float = 0.6
    fts_weight: float = 0.3
    rerank_weight: float = 0.0
    
    # Chinese query specific
    is_chinese: bool = False
    chinese_ratio: float = 0.0
    complexity: str = ""
    has_bilingual_rewrite: bool = False
    
    # Timing
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query[:100] if self.original_query else "",
            "enhanced_query_length": len(self.enhanced_query),
            "similarity_threshold": self.similarity_threshold,
            "top_n_requested": self.top_n_requested,
            "top_n_returned": self.top_n_returned,
            "vector_hits": self.vector_hits,
            "fts_hits": self.fts_hits,
            "rerank_hits": self.rerank_hits,
            "final_tables": self.final_tables,
            "is_chinese": self.is_chinese,
            "chinese_ratio": self.chinese_ratio,
            "complexity": self.complexity,
            "has_bilingual_rewrite": self.has_bilingual_rewrite,
            "duration_ms": self.duration_ms,
            "vector_hit_rate": self.vector_hits / max(self.top_n_requested, 1),
            "fts_hit_rate": self.fts_hits / max(self.top_n_requested, 1),
        }


@dataclass
class SchemaDiscoveryMetrics:
    """
    Complete metrics for a schema discovery execution.
    
    This class tracks all stages of schema discovery and provides
    aggregated statistics for monitoring and debugging.
    """
    
    # Identification
    workflow_id: str = ""
    query_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    # Overall timing
    total_start_time: Optional[float] = None
    total_end_time: Optional[float] = None
    total_duration_ms: Optional[float] = None
    
    # Query info
    query_text: str = ""
    query_language: str = ""  # "zh", "en", "mixed"
    is_chinese: bool = False
    
    # Stage metrics
    stages: Dict[SearchStage, StageMetrics] = field(default_factory=dict)
    
    # Detailed stage metrics
    semantic_metrics: Optional[SemanticSearchMetrics] = None
    external_knowledge_metrics: Optional[ExternalKnowledgeMetrics] = None
    
    # Final results
    total_candidate_tables: int = 0
    final_table_count: int = 0
    table_sources: List[str] = field(default_factory=list)
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize stage metrics."""
        for stage in SearchStage:
            if stage not in self.stages:
                self.stages[stage] = StageMetrics(stage=stage)
    
    def start_discovery(self, query_text: str = ""):
        """Start tracking schema discovery."""
        self.total_start_time = time.time()
        self.query_text = query_text
        self.is_chinese = self._detect_chinese(query_text)
        self.query_language = self._classify_language(query_text)
    
    def end_discovery(self, final_tables: List[str] = None):
        """End tracking schema discovery."""
        self.total_end_time = time.time()
        if self.total_start_time:
            self.total_duration_ms = (self.total_end_time - self.total_start_time) * 1000
        if final_tables:
            self.final_table_count = len(final_tables)
            self.table_sources = list(set(self.table_sources))
    
    def start_stage(self, stage: SearchStage):
        """Start timing a search stage."""
        if stage not in self.stages:
            self.stages[stage] = StageMetrics(stage=stage)
        self.stages[stage].start()
    
    def end_stage(self, stage: SearchStage, success: bool = True, 
                  tables_found: int = 0, tables_returned: int = 0,
                  metadata: Dict[str, Any] = None):
        """End timing a search stage."""
        if stage in self.stages:
            self.stages[stage].end(success, tables_found, tables_returned)
            if metadata:
                self.stages[stage].metadata.update(metadata)
    
    def record_external_knowledge(self, metrics: ExternalKnowledgeMetrics):
        """Record external knowledge retrieval metrics."""
        self.external_knowledge_metrics = metrics
    
    def record_semantic_search(self, metrics: SemanticSearchMetrics):
        """Record semantic search detailed metrics."""
        self.semantic_metrics = metrics
    
    def record_error(self, stage: SearchStage, error: Exception, context: Dict[str, Any] = None):
        """Record an error during discovery."""
        error_info = {
            "stage": stage.value if stage else "unknown",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
        }
        if context:
            error_info.update(context)
        self.errors.append(error_info)
    
    @staticmethod
    def _detect_chinese(text: str) -> bool:
        """Detect if text contains Chinese characters."""
        if not text:
            return False
        for char in text:
            code_point = ord(char)
            if (0x4E00 <= code_point <= 0x9FFF) or (0x3400 <= code_point <= 0x4DBF):
                return True
        return False
    
    @staticmethod
    def _classify_language(text: str) -> str:
        """Classify query language."""
        if not text:
            return "unknown"
        
        has_chinese = False
        has_english = False
        
        for char in text:
            code_point = ord(char)
            if (0x4E00 <= code_point <= 0x9FFF) or (0x3400 <= code_point <= 0x4DBF):
                has_chinese = True
            elif char.isascii() and char.isalpha():
                has_english = True
        
        if has_chinese and has_english:
            return "mixed"
        elif has_chinese:
            return "zh"
        elif has_english:
            return "en"
        return "unknown"
    
    # Aggregation properties
    @property
    def overall_hit_rate(self) -> float:
        """Calculate overall hit rate across all stages."""
        attempted_stages = [s for s in self.stages.values() if s.attempted]
        if not attempted_stages:
            return 0.0
        successful_stages = [s for s in attempted_stages if s.tables_found > 0]
        return len(successful_stages) / len(attempted_stages)
    
    @property
    def external_knowledge_success_rate(self) -> Optional[float]:
        """Get external knowledge retrieval success rate."""
        if not self.external_knowledge_metrics or not self.external_knowledge_metrics.attempted:
            return None
        return 1.0 if self.external_knowledge_metrics.success else 0.0
    
    @property
    def stage_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all stages."""
        return {
            stage.value: metrics.to_dict()
            for stage, metrics in self.stages.items()
            if metrics.attempted
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary for logging/export."""
        return {
            "workflow_id": self.workflow_id,
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "total_duration_ms": self.total_duration_ms,
            "query_info": {
                "query_length": len(self.query_text),
                "query_language": self.query_language,
                "is_chinese": self.is_chinese,
            },
            "summary": {
                "overall_hit_rate": self.overall_hit_rate,
                "final_table_count": self.final_table_count,
                "total_errors": len(self.errors),
            },
            "stages": self.stage_summary,
            "external_knowledge": self.external_knowledge_metrics.to_dict() if self.external_knowledge_metrics else None,
            "semantic_search": self.semantic_metrics.to_dict() if self.semantic_metrics else None,
            "errors": self.errors,
        }
    
    def log_summary(self, logger):
        """Log a formatted summary of metrics."""
        logger.info("=" * 80)
        logger.info("SCHEMA DISCOVERY METRICS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Query Language: {self.query_language}")
        logger.info(f"Total Duration: {self.total_duration_ms:.2f}ms" if self.total_duration_ms else "N/A")
        logger.info(f"Final Tables: {self.final_table_count}")
        logger.info(f"Overall Hit Rate: {self.overall_hit_rate:.2%}")
        
        logger.info("-" * 40)
        logger.info("STAGE BREAKDOWN")
        logger.info("-" * 40)
        
        for stage, metrics in self.stages.items():
            if metrics.attempted:
                status = "✓" if metrics.success else "✗"
                logger.info(
                    f"{status} {stage.value:20s} | "
                    f"found: {metrics.tables_found:3d} | "
                    f"duration: {metrics.duration_ms:8.2f}ms | "
                    f"hit_rate: {metrics.hit_rate:.0%}"
                )
        
        if self.external_knowledge_metrics and self.external_knowledge_metrics.attempted:
            logger.info("-" * 40)
            logger.info("EXTERNAL KNOWLEDGE")
            logger.info("-" * 40)
            ek = self.external_knowledge_metrics
            logger.info(f"Success: {ek.success}")
            logger.info(f"Entries Found: {ek.entries_found}")
            logger.info(f"Entries Used: {ek.entries_used}")
            logger.info(f"Duration: {ek.duration_ms:.2f}ms" if ek.duration_ms else "N/A")
        
        if self.errors:
            logger.info("-" * 40)
            logger.info(f"ERRORS ({len(self.errors)})")
            logger.info("-" * 40)
            for error in self.errors:
                logger.warning(f"  [{error['stage']}] {error['error_type']}: {error['error_message'][:80]}")
        
        logger.info("=" * 80)


# Global metrics collector for tracking across multiple queries
class MetricsCollector:
    """Collects metrics across multiple schema discovery executions."""
    
    def __init__(self):
        self.metrics_history: List[SchemaDiscoveryMetrics] = []
        self.max_history: int = 1000
    
    def record(self, metrics: SchemaDiscoveryMetrics):
        """Record metrics for aggregation."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
    
    def get_aggregated_stats(self, window_size: int = 100) -> Dict[str, Any]:
        """Get aggregated statistics over recent executions."""
        recent = self.metrics_history[-window_size:]
        if not recent:
            return {}
        
        total_queries = len(recent)
        chinese_queries = sum(1 for m in recent if m.is_chinese)
        
        # Stage success rates
        stage_stats = {}
        for stage in SearchStage:
            stage_attempts = [m for m in recent if m.stages[stage].attempted]
            stage_success = [m for m in stage_attempts if m.stages[stage].success]
            stage_hits = [m for m in stage_attempts if m.stages[stage].tables_found > 0]
            
            stage_stats[stage.value] = {
                "attempted": len(stage_attempts),
                "success": len(stage_success),
                "success_rate": len(stage_success) / len(stage_attempts) if stage_attempts else 0,
                "hit_rate": len(stage_hits) / len(stage_attempts) if stage_attempts else 0,
                "avg_duration_ms": sum(m.stages[stage].duration_ms or 0 for m in stage_attempts) / len(stage_attempts) if stage_attempts else 0,
            }
        
        # External knowledge stats
        ek_attempts = [m for m in recent if m.external_knowledge_metrics and m.external_knowledge_metrics.attempted]
        ek_success = [m for m in ek_attempts if m.external_knowledge_metrics.success]
        
        return {
            "total_queries": total_queries,
            "chinese_queries": chinese_queries,
            "chinese_ratio": chinese_queries / total_queries if total_queries else 0,
            "avg_total_duration_ms": sum(m.total_duration_ms or 0 for m in recent) / total_queries,
            "avg_final_tables": sum(m.final_table_count for m in recent) / total_queries,
            "overall_hit_rate": sum(m.overall_hit_rate for m in recent) / total_queries,
            "stages": stage_stats,
            "external_knowledge": {
                "attempted": len(ek_attempts),
                "success": len(ek_success),
                "success_rate": len(ek_success) / len(ek_attempts) if ek_attempts else 0,
            },
            "error_rate": sum(len(m.errors) for m in recent) / total_queries if total_queries else 0,
        }


# Global collector instance
global_metrics_collector = MetricsCollector()
