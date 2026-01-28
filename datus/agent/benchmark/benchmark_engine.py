# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Benchmark Engine Module.

Handles benchmark execution for evaluating agent performance
on various SQL generation and semantic layer tasks.
"""

import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Set

from datus.configuration.agent_config import AgentConfig, BenchmarkConfig
from datus.schemas.node_models import SqlTask
from datus.tools.db_tools.db_manager import get_db_manager
from datus.utils.benchmark_utils import load_benchmark_tasks
from datus.utils.loggings import get_logger
from datus.utils.time_utils import format_duration_human

logger = get_logger(__name__)


class BenchmarkEngine:
    """
    Engine for executing benchmark tasks and evaluating agent performance.

    Supports multiple benchmark platforms including Spider, BIRD, and
    semantic layer benchmarks.
    """

    def __init__(self, agent_config: AgentConfig, args: argparse.Namespace, check_db_fn=None):
        """
        Initialize the benchmark engine.

        Args:
            agent_config: Agent configuration
            args: Command line arguments
            check_db_fn: Optional database check function
        """
        self.global_config = agent_config
        self.args = args
        self._check_db_fn = check_db_fn

    def run(self) -> Dict[str, Any]:
        """Execute the benchmark."""
        logger.info("Benchmarking begins")
        benchmark_platform = self.args.benchmark
        benchmark_path = self.global_config.benchmark_path(benchmark_platform)

        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Benchmark_path not found: {benchmark_path}")

        target_task_ids = getattr(self.args, "benchmark_task_ids", [])
        target_task_ids = set(target_task_ids) if target_task_ids else None

        import time
        from datetime import datetime

        # Generate a shared run_id for this benchmark run
        benchmark_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Benchmark run_id: {benchmark_run_id}")

        start = time.perf_counter()
        if benchmark_platform == "semantic_layer":
            self.global_config.check_init_storage_config("metric")
            result = self.run_semantic_layer(benchmark_path, target_task_ids, run_id=benchmark_run_id)
        else:
            self.global_config.check_init_storage_config("database")
            self.global_config.check_init_storage_config("metric")
            result = self.run_standard(benchmark_platform, target_task_ids, run_id=benchmark_run_id)
        end = time.perf_counter()

        time_spends = end - start
        result["time_spends"] = format_duration_human(time_spends)
        result["time_spends_seconds"] = str(time_spends)
        result["run_id"] = benchmark_run_id
        return result

    def run_standard(
        self, benchmark_platform: str, target_task_ids: Optional[Set[str]] = None, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run standard benchmark (Spider, BIRD, etc.).

        Args:
            benchmark_platform: Name of the benchmark platform
            target_task_ids: Optional set of task IDs to run
            run_id: Optional run identifier

        Returns:
            Benchmark result dictionary
        """
        _, conn = get_db_manager(self.global_config.namespaces).first_conn_with_name(
            self.global_config.current_namespace
        )
        if self._check_db_fn:
            self._check_db_fn()

        benchmark_config = self.global_config.benchmark_config(benchmark_platform)
        max_workers = getattr(self.args, "max_workers", 1) or 1
        logger.info(f"Loaded tasks from {benchmark_platform} benchmark")

        task_id_key = benchmark_config.question_id_key or "_task_id"
        completed_tasks = {}

        def run_single_task(task_id: str, config: BenchmarkConfig, task_item: Dict[str, Any]):
            """Execute a single benchmark task."""
            task = task_item.get(config.question_key)
            if not task:
                logger.warning(
                    f"The question content was not obtained through {config.question_key}, "
                    "please check your benchmark configuration."
                )
                return task_id, ""
            database_name = task_item.get(config.db_key) or conn.database_name or ""
            logger.info(f"start benchmark with {task_id}: {task}")
            use_tables = None if not config.use_tables_key else task_item.get(config.use_tables_key)

            # Use hierarchical save directory structure
            output_dir = self.global_config.get_save_run_dir(run_id) if run_id else self.global_config.output_dir

            from datus.agent.agent import Agent

            agent = Agent(self.args, self.global_config)
            result = agent.run(
                SqlTask(
                    id=task_id,
                    database_type=conn.dialect,
                    task=task,
                    database_name=database_name,
                    output_dir=output_dir,
                    current_date=self.args.current_date,
                    tables=use_tables,
                    external_knowledge=(
                        ""
                        if not config.ext_knowledge_key
                        else task_item.get(config.ext_knowledge_key, "")
                    ),
                    schema_linking_type="full",
                ),
                check_storage=False,
                check_db=False,
                run_id=run_id,
            )
            logger.info(f"Finish benchmark with {task_id}, file saved in {output_dir}/{task_id}.csv.")
            return task_id, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for task_item in load_benchmark_tasks(self.global_config, benchmark_platform):
                raw_task_id = task_item.get(task_id_key)
                if raw_task_id in (None, ""):
                    logger.warning(f"Task id {raw_task_id} was not found, please check your benchmark configuration.")
                    continue
                else:
                    task_id = str(raw_task_id)
                task_item[task_id_key] = task_id
                if not target_task_ids or task_id in target_task_ids:
                    f = executor.submit(run_single_task, task_id, benchmark_config, task_item)
                    future_to_task[f] = task_item

            # Wait for completion
            for future in as_completed(future_to_task):
                task_item = future_to_task[future]
                try:
                    task_id, _ = future.result()
                    logger.debug(f"Task {task_id} completed successfully")
                    completed_tasks[task_id] = True
                except Exception as exc:
                    task_id = task_item.get(task_id_key) or task_item.get("_task_id")
                    if task_id is None:
                        task_id = f"unknown_{len(future_to_task)}"
                    logger.error(f"Task {task_id} generated an exception: {exc}")
                    completed_tasks[task_id] = False

        logger.info("Benchmark execution completed.")
        return {"status": "success", "message": "Benchmark tasks executed successfully", "completed": completed_tasks}

    def run_semantic_layer(
        self, benchmark_path: str, target_task_ids: Optional[Set[str]] = None, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run semantic layer benchmark.

        Args:
            benchmark_path: Path to benchmark data
            target_task_ids: Optional set of task IDs to run
            run_id: Optional run identifier

        Returns:
            Benchmark result dictionary
        """
        task_file = self.args.testing_set
        self._check_benchmark_file(task_file)

        # Clean up previous execution results to avoid interference
        self._cleanup_benchmark_output_paths(benchmark_path)

        tasks = []
        with open(task_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line_no, row in enumerate(reader, 1):
                logger.debug(f"line {line_no}: {row}")
                if "question" in row and "sql" in row and row["question"].strip() and row["sql"].strip():
                    task_data = {"question_id": line_no, "question": row["question"].strip(), "sql": row["sql"].strip()}
                    # Check if ext_knowledge column exists
                    if "external_knowledge" in row and row["external_knowledge"].strip():
                        task_data["external_knowledge"] = row["external_knowledge"].strip()
                    tasks.append(task_data)

        logger.info(f"Loaded {len(tasks)} tasks from semantic_layer benchmark")

        metric_meta = self.global_config.current_metric_meta(self.args.metric_meta)

        from datus.agent.agent import Agent

        agent = Agent(self.args, self.global_config)

        for task in tasks:
            task_id = str(task["question_id"])
            if target_task_ids and task_id not in target_task_ids:
                continue

            question = task["question"]
            logger.info(f"start benchmark with {task_id}: {question}")
            current_db_config = self.global_config.current_db_config()

            # Merge external knowledge from file with metric_meta
            combined_ext_knowledge = metric_meta.ext_knowledge
            if "external_knowledge" in task and task["external_knowledge"]:
                if combined_ext_knowledge:
                    combined_ext_knowledge = f"{combined_ext_knowledge}\n\n{task['external_knowledge']}"
                else:
                    combined_ext_knowledge = task["external_knowledge"]

            # Use hierarchical save directory structure
            output_dir = self.global_config.get_save_run_dir(run_id) if run_id else self.global_config.output_dir

            if metric_meta.subject_path and metric_meta.subject_path.strip():
                subject_path = [c.strip() for c in metric_meta.subject_path.split("/") if c.strip()]
            else:
                subject_path = None

            agent.run(
                SqlTask(
                    id=task_id,
                    database_type=current_db_config.type,
                    task=question,
                    database_name=current_db_config.database,
                    schema_name=current_db_config.schema,
                    subject_path=subject_path,
                    output_dir=output_dir,
                    external_knowledge=combined_ext_knowledge,
                    current_date=self.args.current_date,
                ),
                run_id=run_id,
            )

            logger.info(f"Finish benchmark with {task_id}, file saved in {output_dir}/{task_id}.csv.")

        return {"status": "success", "message": "Semantic layer benchmark tasks executed successfully"}

    def _check_benchmark_file(self, file_path: str):
        """Check if benchmark file exists."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Benchmarking task file not found, file_path={file_path}")

    def _cleanup_benchmark_output_paths(self, benchmark_path: str):
        """Clean up previous benchmark execution results."""
        current_namespace = self.global_config.current_namespace

        # Clean up namespace directory in output directory
        output_dir = self.global_config.output_dir
        namespace_dir = os.path.join(output_dir, current_namespace)

        # Safety check: ensure we're only deleting within output directory
        if namespace_dir and os.path.exists(namespace_dir):
            namespace_abs = os.path.abspath(namespace_dir)
            output_abs = os.path.abspath(output_dir)
            if not namespace_abs.startswith(output_abs):
                raise ValueError(f"Namespace directory outside output root: {namespace_dir}")

            logger.info(f"Cleaning up namespace directory: {namespace_dir}")
            try:
                shutil.rmtree(namespace_dir)
                logger.info(f"Successfully removed namespace directory: {namespace_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean namespace directory {namespace_dir}: {e}")

        # Clean up gold directory (which contains exec_result)
        gold_path = os.path.join(benchmark_path, "gold")
        if gold_path and os.path.exists(gold_path):
            gold_abs = os.path.abspath(gold_path)
            benchmark_abs = os.path.abspath(benchmark_path)
            if not gold_abs.startswith(benchmark_abs):
                raise ValueError(f"Gold directory outside benchmark path: {gold_path}")

            logger.info(f"Cleaning up gold directory: {gold_path}")
            try:
                shutil.rmtree(gold_path)
                logger.info(f"Successfully removed gold directory: {gold_path}")
            except Exception as e:
                logger.warning(f"Failed to clean gold directory {gold_path}: {e}")
