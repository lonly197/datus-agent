# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Dataset Generation Module.

Generates training datasets from workflow trajectory files
for fine-tuning and evaluation purposes.
"""

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from datus.utils.json_utils import to_str
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class TrajectoryDatasetGenerator:
    """
    Generates training datasets from workflow trajectory files.

    Extracts SQL contexts, prompts, and other metadata from
    workflow execution logs for dataset creation.
    """

    def __init__(self, agent_config, args: argparse.Namespace):
        """
        Initialize the dataset generator.

        Args:
            agent_config: Agent configuration
            args: Command line arguments
        """
        self.global_config = agent_config
        self.args = args

    def generate(self) -> Dict[str, Any]:
        """
        Generate dataset from trajectory files.

        Returns:
            Generation result dictionary
        """
        logger.info("Generating dataset from trajectory files")

        trajectory_dir = self.args.trajectory_dir
        dataset_name = self.args.dataset_name
        output_format = getattr(self.args, "format", "json")
        benchmark_task_ids = getattr(self.args, "benchmark_task_ids", None)

        if not os.path.exists(trajectory_dir):
            raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

        # Parse benchmark_task_ids if provided
        allowed_task_ids = None
        if benchmark_task_ids:
            allowed_task_ids = [task_id.strip() for task_id in benchmark_task_ids.split(",")]
            logger.info(f"Filtering by task IDs: {allowed_task_ids}")

        # Find all trajectory YAML files
        trajectory_files = glob.glob(os.path.join(trajectory_dir, "*_*.yaml"))
        logger.info(f"Found {len(trajectory_files)} trajectory files")

        dataset_data = []

        for trajectory_file in trajectory_files:
            try:
                # Extract task_id from filename (e.g., "0_1750662901.yaml" -> "0")
                filename = os.path.basename(trajectory_file)
                task_id = filename.split("_")[0]

                # Filter by task_id if benchmark_task_ids is provided
                if allowed_task_ids and task_id not in allowed_task_ids:
                    logger.debug(f"Skipping trajectory file {filename} (task_id {task_id} not in allowed list)")
                    continue

                logger.info(f"Processing trajectory file: {filename}")

                # Load trajectory YAML file
                with open(trajectory_file, "r", encoding="utf-8") as f:
                    trajectory_data = yaml.safe_load(f)

                # Extract sql_contexts from the workflow
                sql_contexts = None
                first_sql_node_id = None

                if "workflow" in trajectory_data and "nodes" in trajectory_data["workflow"]:
                    for node in trajectory_data["workflow"]["nodes"]:
                        if node.get("type") in ["reasoning", "generate_sql"]:
                            if "result" in node and "sql_contexts" in node["result"]:
                                sql_contexts = node["result"]["sql_contexts"]
                                first_sql_node_id = node["id"]
                                break

                if not sql_contexts or not first_sql_node_id:
                    logger.warning(f"No sql_contexts found in {filename}")
                    continue

                # Load node details from the corresponding node file
                node_file = os.path.join(trajectory_dir, task_id, f"{first_sql_node_id}.yml")
                if not os.path.exists(node_file):
                    logger.warning(f"Node file not found: {node_file}")
                    continue

                with open(node_file, "r", encoding="utf-8") as f:
                    node_data = yaml.safe_load(f)

                # Extract required fields
                user_prompt = node_data.get("user_prompt", "")
                system_prompt = node_data.get("system_prompt", "")
                reason_content = node_data.get("reason_content", [])
                output_content = node_data.get("output_content", "")

                # Create dataset entry
                dataset_entry = {
                    "task_id": task_id,
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "reason_content": reason_content,
                    "sql_contexts": sql_contexts,
                    "output_content": output_content,
                }

                dataset_data.append(dataset_entry)
                logger.info(f"Successfully processed {filename}")

            except Exception as e:
                logger.error(f"Error processing {trajectory_file}: {str(e)}")
                continue

        # Save dataset to file based on format
        output_file = self._save_dataset(dataset_data, dataset_name, output_format)

        filter_info = f" (filtered by task IDs: {allowed_task_ids})" if allowed_task_ids else ""
        logger.info(f"Dataset generated successfully: {output_file}")
        logger.info(f"Total entries: {len(dataset_data)}{filter_info}")

        return {
            "status": "success",
            "message": f"Dataset generated successfully: {output_file}",
            "total_entries": len(dataset_data),
            "output_file": output_file,
            "format": output_format,
            "filtered_task_ids": allowed_task_ids,
        }

    def _save_dataset(
        self,
        dataset_data: list,
        dataset_name: str,
        output_format: str,
    ) -> str:
        """
        Save dataset to file in specified format.

        Args:
            dataset_data: List of dataset entries
            dataset_name: Base name for output file
            output_format: Output format (json, parquet)

        Returns:
            Output file path
        """
        # Validate dataset_name to prevent path traversal attacks
        if dataset_name != Path(dataset_name).name:
            raise ValueError(f"Invalid dataset_name '{dataset_name}': must be a simple filename without path separators")

        if output_format == "json":
            output_file = f"{dataset_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset_data, f, ensure_ascii=False, indent=2)
        elif output_format == "parquet":
            output_file = self._save_parquet(dataset_data, dataset_name)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        return output_file

    def _save_parquet(self, dataset_data: list, dataset_name: str) -> str:
        """
        Save dataset to parquet format.

        Args:
            dataset_data: List of dataset entries
            dataset_name: Base name for output file

        Returns:
            Output file path
        """
        try:
            import pandas as pd

            output_file = f"{dataset_name}.parquet"

            # Convert the dataset to a pandas DataFrame
            df_data = []
            for entry in dataset_data:
                df_entry = {
                    "user_prompt": entry["user_prompt"],
                    "system_prompt": entry["system_prompt"],
                    "reason_content": to_str(entry["reason_content"]),
                    "sql_contexts": to_str(entry["sql_contexts"]),
                    "output_content": entry["output_content"],
                }
                df_data.append(df_entry)

            df = pd.DataFrame(df_data)
            df.to_parquet(output_file, index=False)
            return output_file

        except ImportError:
            logger.error(
                "pandas is required for parquet format. Please install it with: pip install pandas pyarrow"
            )
            raise ImportError("pandas is required for parquet format")
