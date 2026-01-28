# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import asyncio
import os
from typing import Any, AsyncGenerator, Dict, Optional

from datus.agent.workflow_runner import WorkflowRunner
from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.node_models import SqlTask
from datus.tools.db_tools.db_manager import DBManager, get_db_manager
from datus.utils.async_utils import ensure_not_cancelled
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class Agent:
    """
    Main entry point for the SQL Agent system.
    Handles initialization, workflow management, and execution coordination.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        agent_config: AgentConfig,
        db_manager: Optional[DBManager] = None,
    ):
        """
        Initialize the Agent with configuration parameters.

        Args:
            args: Command line arguments and configuration
            agent_config: Pre-loaded agent configuration
            db_manager: Optional database manager instance
        """
        self.args = args
        self.global_config = agent_config
        if db_manager:
            self.db_manager = db_manager
        else:
            self.db_manager = get_db_manager(self.global_config.namespaces)

        self.tools: Dict[str, Any] = {}
        self.storage_modules: Dict[str, bool] = {}
        self.metadata_store = None
        self.metrics_store = None
        self._check_storage_modules()

    def _initialize_model(self) -> LLMBaseModel:
        llm_model = LLMBaseModel.create_model(model_name="default", agent_config=self.global_config)
        logger.info(f"Using model type: {llm_model.model_config.type}, model name: {llm_model.model_config.model}")
        return llm_model

    def _check_storage_modules(self):
        """Check if storage modules exist and initialize them if needed."""
        if os.path.exists(os.path.join("storage", "schema_metadata")):
            self.storage_modules["schema_metadata"] = True
        if os.path.exists(os.path.join("storage", "metric_store")):
            self.storage_modules["metric_store"] = True
        if os.path.exists(os.path.join("storage", "document")):
            self.storage_modules["document"] = True
        if os.path.exists(os.path.join("storage", "success_story")):
            self.storage_modules["success_story"] = True
        logger.info(f"Storage modules initialized: {list(self.storage_modules.keys())}")

    def create_workflow_runner(
        self, check_db: bool = True, run_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowRunner:
        """Create a workflow runner that can safely execute in isolation."""
        return WorkflowRunner(
            self.args,
            self.global_config,
            pre_run_callable=self.check_db if check_db else None,
            run_id=run_id,
            metadata=metadata,
        )

    def run(
        self,
        sql_task: Optional[SqlTask] = None,
        check_storage: bool = False,
        check_db: bool = True,
        run_id: Optional[str] = None,
    ) -> dict:
        """Execute a workflow synchronously via a dedicated runner."""
        runner = self.create_workflow_runner(check_db=check_db, run_id=run_id)
        return runner.run(sql_task=sql_task, check_storage=check_storage)

    async def run_stream(
        self,
        sql_task: Optional[SqlTask] = None,
        check_storage: bool = False,
        action_history_manager: Optional[ActionHistoryManager] = None,
        task_id: Optional[str] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute a workflow with streaming progress updates."""
        runner = self.create_workflow_runner()
        async for action in runner.run_stream(
            sql_task=sql_task,
            check_storage=check_storage,
            action_history_manager=action_history_manager,
            task_id=task_id,
        ):
            yield action

    async def run_stream_with_metadata(
        self, sql_task: SqlTask, metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute workflow with streaming and custom metadata."""
        try:
            runner = self.create_workflow_runner(metadata=metadata)
            ensure_not_cancelled()
            async for action in runner.run_stream(sql_task=sql_task):
                yield action
                ensure_not_cancelled()
        except asyncio.CancelledError:
            logger.info("Agent stream execution was cancelled")
            raise

    def check_db(self) -> Dict[str, Any]:
        """Validate database connectivity."""
        logger.info("Checking database connectivity")
        namespace = self.global_config.current_namespace
        if namespace in self.global_config.namespaces:
            connections = self.db_manager.get_connections(namespace)
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

    def probe_llm(self):
        """Test LLM model connectivity."""
        logger.info("Testing LLM model connectivity")
        try:
            llm_model = LLMBaseModel.create_model(model_name="default", agent_config=self.global_config)
            logger.info(
                f"Using model type: {llm_model.model_config.type}, " f"model name: {llm_model.model_config.model}"
            )
            response = llm_model.generate("Hello, can you hear me?")
            logger.info("LLM model test successful")
            return {
                "status": "success",
                "message": "LLM model test successful",
                "response": response,
            }
        except Exception as e:
            logger.error(f"LLM model test failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def bootstrap_kb(self) -> Dict[str, Any]:
        """Initialize knowledge base storage components."""
        from datus.agent.bootstrap import KnowledgeBaseBootstrapper

        logger.info("Initializing knowledge base components")
        bootstrapper = KnowledgeBaseBootstrapper(self.global_config, self.args)
        return bootstrapper.bootstrap()

    def benchmark(self) -> Dict[str, Any]:
        """Run benchmark tasks."""
        from datus.agent.benchmark import BenchmarkEngine

        logger.info("Benchmarking begins")
        engine = BenchmarkEngine(self.global_config, self.args, check_db_fn=self.check_db)
        return engine.run()

    def evaluation(self, log_summary: bool = True) -> Dict[str, Any]:
        """Evaluate the benchmarking."""
        benchmark_platform = self.args.benchmark
        if benchmark_platform in ("semantic_layer", "bird_critic"):
            return {
                "status": "failed",
                "message": "Benchmark bird_critic and semantic_layer evaluation is not supported at the moment",
            }

        from datus.utils.benchmark_utils import evaluate_benchmark_and_report

        run_id = getattr(self.args, "run_id", None)
        summary_report_file = getattr(self.args, "summary_report_file", None)
        evaluation_result = evaluate_benchmark_and_report(
            agent_config=self.global_config,
            benchmark_platform=benchmark_platform,
            target_task_ids=self.args.task_ids,
            output_file=self.args.output_file,
            log_summary=log_summary,
            run_id=run_id,
            summary_report_file=summary_report_file,
        )
        return {
            "status": evaluation_result.get("status"),
            "generated_time": evaluation_result.get("generated_time"),
            "message": evaluation_result.get("error"),
        }

    def generate_dataset(self) -> Dict[str, Any]:
        """Generate dataset from trajectory files."""
        from datus.agent.dataset import TrajectoryDatasetGenerator

        logger.info("Generating dataset from trajectory files")
        generator = TrajectoryDatasetGenerator(self.global_config, self.args)
        return generator.generate()

    def benchmark_bird_critic(self):
        """Placeholder for BIRD critic benchmark."""
        pass
