# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Knowledge Base Bootstrap Module.

Handles initialization of knowledge base storage components including:
- Metadata schema storage
- Semantic metrics storage
- Document storage
- External knowledge storage
- Reference SQL storage
"""

import argparse
import os
from typing import Any, Dict, Optional

from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionStatus
from datus.storage.ext_knowledge.ext_knowledge_init import init_ext_knowledge
from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from datus.storage.metric.metrics_init import init_semantic_yaml_metrics, init_success_story_metrics
from datus.storage.metric.store import SemanticMetricsRAG
from datus.storage.reference_sql import ReferenceSqlRAG
from datus.storage.reference_sql.reference_sql_init import init_reference_sql
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.storage.schema_metadata.benchmark_init import init_snowflake_schema
from datus.storage.schema_metadata.benchmark_init_bird import init_dev_schema
from datus.storage.schema_metadata.local_init import init_local_schema
from datus.storage.sub_agent_kb_bootstrap import SubAgentBootstrapper
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class KnowledgeBaseBootstrapper:
    """
    Manages initialization and bootstrapping of knowledge base components.

    This class handles the setup of various knowledge storage systems
    including schema metadata, metrics, documents, and reference SQL.
    """

    VALID_COMPONENTS = {"metadata", "metrics", "document", "ext_knowledge", "reference_sql"}

    def __init__(self, agent_config: AgentConfig, args: argparse.Namespace):
        """
        Initialize the bootstrapper.

        Args:
            agent_config: Agent configuration
            args: Command line arguments
        """
        self.global_config = agent_config
        self.args = args
        self.metadata_store: Optional[SchemaWithValueRAG] = None
        self.metrics_store: Optional[SemanticMetricsRAG] = None
        self.ext_knowledge_store: Optional[ExtKnowledgeStore] = None
        self.reference_sql_store: Optional[ReferenceSqlRAG] = None
        self.storage_modules: Dict[str, Any] = {}

    def _safe_delete_directory(self, dir_path: str, context: str = "") -> None:
        """Safely delete a directory with path validation."""
        abs_path = os.path.abspath(dir_path)
        project_root = os.path.abspath(self.global_config.rag_storage_path())

        if not abs_path.startswith(project_root):
            raise ValueError(f"Refusing to delete directory outside project root: {abs_path}")

        if not os.path.exists(abs_path):
            logger.debug(f"Directory does not exist, skipping deletion: {abs_path}")
            return

        try:
            import shutil
            shutil.rmtree(abs_path)
            logger.info(f"Deleted directory: {context} ({abs_path})")
        except Exception as e:
            logger.error(f"Failed to delete directory {abs_path}: {e}")
            raise

    def bootstrap(self, selected_components: Optional[list] = None) -> Dict[str, Any]:
        """
        Bootstrap knowledge base components.

        Args:
            selected_components: List of components to bootstrap.
                If None, uses args.components.

        Returns:
            Bootstrap result dictionary
        """
        if selected_components is None:
            selected_components = self.args.components

        # Validate component names
        invalid = set(selected_components) - self.VALID_COMPONENTS
        if invalid:
            raise ValueError(
                f"Invalid bootstrap components: {invalid}. Valid components: {sorted(self.VALID_COMPONENTS)}"
            )

        kb_update_strategy = self.args.kb_update_strategy
        benchmark_platform = self.args.benchmark
        pool_size = 4 if not self.args.pool_size else self.args.pool_size
        dir_path = self.global_config.rag_storage_path()

        # Parse subject_tree from command line if provided
        subject_tree = None
        if hasattr(self.args, "subject_tree") and self.args.subject_tree:
            subject_tree = [s.strip() for s in self.args.subject_tree.split(",") if s.strip()]
            logger.info(f"Using predefined subject_tree categories: {subject_tree}")

        results = {}

        for component in selected_components:
            if component == "metadata":
                result = self._bootstrap_metadata(dir_path, kb_update_strategy, benchmark_platform, pool_size)
                return result

            elif component == "metrics":
                result = self._bootstrap_metrics(dir_path, kb_update_strategy, subject_tree)
                return result

            elif component == "document":
                from datus.storage.document.store import document_store
                self.storage_modules = {"document_store": document_store(self.global_config.rag_storage_path())}

            elif component == "ext_knowledge":
                result = self._bootstrap_ext_knowledge(dir_path, kb_update_strategy, pool_size)
                return result

            elif component == "reference_sql":
                result = self._bootstrap_reference_sql(dir_path, kb_update_strategy, pool_size, subject_tree)
                return result

            results[component] = True

        # Initialize success story storage (always created)
        success_story_path = os.path.join("storage", "success_story")
        if not os.path.exists(success_story_path):
            os.makedirs(success_story_path)
        results["success_story"] = True

        logger.info(f"Knowledge base components initialized successfully: {', '.join(selected_components)}")
        return {
            "status": "success",
            "message": "Knowledge base initialized",
            "components": results,
        }

    def _bootstrap_metadata(
        self,
        dir_path: str,
        kb_update_strategy: str,
        benchmark_platform: Optional[str],
        pool_size: int,
    ) -> Dict[str, Any]:
        """Bootstrap metadata component."""
        if kb_update_strategy == "check":
            if not os.path.exists(dir_path):
                raise ValueError("metadata is not built, please run bootstrap_kb with overwrite strategy first")
            else:
                self.global_config.check_init_storage_config("database")
                self.metadata_store = SchemaWithValueRAG(self.global_config)
                return {
                    "status": "success",
                    "message": f"current metadata is already built, "
                    f"dir_path={dir_path},"
                    f"schema_size={self.metadata_store.get_schema_size()}, "
                    f"value_size={self.metadata_store.get_value_size()}",
                }

        if kb_update_strategy == "overwrite":
            self.global_config.save_storage_config("database")
            schema_metadata_path = os.path.join(dir_path, "schema_metadata.lance")
            self._safe_delete_directory(schema_metadata_path, "schema_metadata")
            schema_value_path = os.path.join(dir_path, "schema_value.lance")
            self._safe_delete_directory(schema_value_path, "schema_value")
        else:
            self.global_config.check_init_storage_config("database")
        self.metadata_store = SchemaWithValueRAG(self.global_config)

        from datus.agent.bootstrap.sub_agent_refresher import refresh_scoped_agents

        if not benchmark_platform:
            self.check_db()
            init_local_schema(
                self.metadata_store,
                self.global_config,
                self.db_manager,
                kb_update_strategy,
                table_type=self.args.schema_linking_type,
                init_catalog_name=self.args.catalog or "",
                init_database_name=self.args.database_name or "",
                pool_size=pool_size,
            )
        elif benchmark_platform == "spider2":
            benchmark_path = self.global_config.benchmark_path(benchmark_platform)
            init_snowflake_schema(
                self.metadata_store,
                benchmark_path,
                kb_update_strategy,
                pool_size=pool_size,
            )
        elif benchmark_platform == "bird_dev":
            self.check_db()
            benchmark_path = self.global_config.benchmark_path(benchmark_platform)
            init_dev_schema(
                self.metadata_store,
                self.db_manager,
                self.global_config.current_namespace,
                benchmark_path,
                kb_update_strategy,
                pool_size=pool_size,
            )
        elif benchmark_platform == "bird_critic":
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=f"Unsupported benchmark platform: {benchmark_platform}",
            )
        else:
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                f"Unsupported benchmark platform: {benchmark_platform}",
            )

        result = {
            "status": "success",
            "message": f"metadata bootstrap completed, "
            f"schema_size={self.metadata_store.get_schema_size()}, "
            f"value_size={self.metadata_store.get_value_size()}",
        }
        refresh_scoped_agents(self.global_config, "metadata", kb_update_strategy)
        return result

    def _bootstrap_metrics(
        self,
        dir_path: str,
        kb_update_strategy: str,
        subject_tree: Optional[list],
    ) -> Dict[str, Any]:
        """Bootstrap metrics component."""
        semantic_model_path = os.path.join(dir_path, "semantic_model.lance")
        metrics_path = os.path.join(dir_path, "metrics.lance")
        if kb_update_strategy == "overwrite":
            self._safe_delete_directory(semantic_model_path, "semantic_model")
            self._safe_delete_directory(metrics_path, "metrics")
            self.global_config.save_storage_config("metric")
        else:
            self.global_config.check_init_storage_config("metric")

        # Initialize metrics using unified SemanticAgenticNode approach
        if hasattr(self.args, "semantic_yaml") and self.args.semantic_yaml:
            successful, error_message = init_semantic_yaml_metrics(self.args.semantic_yaml, self.global_config)
        else:
            successful, error_message = init_success_story_metrics(self.args, self.global_config, subject_tree)

        # Create metrics_store for statistics
        from datus.agent.bootstrap.sub_agent_refresher import refresh_scoped_agents

        if successful:
            self.metrics_store = SemanticMetricsRAG(self.global_config)
            result = {
                "status": "success",
                "message": f"metrics bootstrap completed,"
                f"semantic_model_size={self.metrics_store.get_semantic_model_size()}, "
                f"metrics_size={self.metrics_store.get_metrics_size()}",
                "error": error_message,
            }
            refresh_scoped_agents(self.global_config, "metrics", kb_update_strategy)
        else:
            result = {
                "status": "failed",
                "message": error_message,
            }
        return result

    def _bootstrap_ext_knowledge(
        self,
        dir_path: str,
        kb_update_strategy: str,
        pool_size: int,
    ) -> Dict[str, Any]:
        """Bootstrap external knowledge component."""
        ext_knowledge_path = os.path.join(dir_path, "ext_knowledge.lance")
        if kb_update_strategy == "overwrite":
            self._safe_delete_directory(ext_knowledge_path, "ext_knowledge")
            self.global_config.save_storage_config("ext_knowledge")
        else:
            self.global_config.check_init_storage_config("ext_knowledge")
        self.ext_knowledge_store = ExtKnowledgeStore(dir_path)
        init_ext_knowledge(
            self.ext_knowledge_store, self.args, build_mode=kb_update_strategy, pool_size=pool_size
        )
        return {
            "status": "success",
            "message": f"ext_knowledge bootstrap completed, "
            f"knowledge_size={self.ext_knowledge_store.table_size()}",
        }

    def _bootstrap_reference_sql(
        self,
        dir_path: str,
        kb_update_strategy: str,
        pool_size: int,
        subject_tree: Optional[list],
    ) -> Dict[str, Any]:
        """Bootstrap reference SQL component."""
        reference_sql_path = os.path.join(dir_path, "reference_sql.lance")
        if kb_update_strategy == "overwrite":
            self._safe_delete_directory(reference_sql_path, "reference_sql")
            self.global_config.save_storage_config("reference_sql")
        else:
            self.global_config.check_init_storage_config("reference_sql")

        self.reference_sql_store = ReferenceSqlRAG(self.global_config)
        result = init_reference_sql(
            self.reference_sql_store,
            self.args,
            self.global_config,
            build_mode=kb_update_strategy,
            pool_size=pool_size,
            subject_tree=subject_tree,
        )

        from datus.agent.bootstrap.sub_agent_refresher import refresh_scoped_agents

        if isinstance(result, dict) and result.get("status") != "error":
            refresh_scoped_agents(self.global_config, "reference_sql", kb_update_strategy)
        return result

    def check_db(self) -> Dict[str, Any]:
        """Validate database connectivity."""
        from datus.tools.db_tools.db_manager import get_db_manager

        logger.info("Checking database connectivity")
        namespace = self.global_config.current_namespace
        if namespace in self.global_config.namespaces:
            connections = get_db_manager(self.global_config.namespaces).get_connections(namespace)
            if not connections:
                logger.warning(f"No connections found for {namespace}")
                return {"status": "error", "message": f"No connections found for {namespace}"}
            if isinstance(connections, dict):
                for name, conn in connections.items():
                    conn.test_connection()
                    logger.info(f"Database connection test successful for {name}")
            else:
                connections.test_connection()
                logger.info(f"Database connection test successful {namespace}")
            return {"status": "success", "message": "Database connection test successful"}
        else:
            logger.error(f"Database connection test failed: {namespace} not found in namespaces")
            return {"status": "error", "message": f"{namespace} not found in namespaces"}

    @property
    def db_manager(self):
        """Lazy import db_manager."""
        from datus.tools.db_tools.db_manager import get_db_manager
        return get_db_manager(self.global_config.namespaces)
