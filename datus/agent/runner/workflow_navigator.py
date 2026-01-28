# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Workflow Navigation.

Handles node jumping logic for reflect and output nodes
during workflow execution.
"""

from typing import Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class WorkflowNavigator:
    """Handles workflow node navigation and jumping logic."""

    def __init__(self, workflow=None):
        self.workflow = workflow

    def set_workflow(self, workflow):
        """Set the workflow reference."""
        self.workflow = workflow

    def jump_to_reflect_node(self) -> bool:
        """
        Jump directly to the next reflect node in the workflow execution path.

        When a node soft-fails, we want to skip all intermediate nodes and go
        directly to the reflect node for recovery.

        Returns:
            True if successfully jumped to a reflect node, False otherwise
        """
        if not self.workflow or not self.workflow.nodes:
            return False

        # Find the next reflect node in the execution order
        current_idx = self.workflow.current_node_index
        if current_idx is None:
            return False

        # Search forward in node_order for the first reflect node
        # Limit search range to prevent excessive iteration
        max_search_range = min(len(self.workflow.node_order), current_idx + 100)
        nodes_checked = 0

        for i in range(current_idx + 1, max_search_range):
            node_id = self.workflow.node_order[i]
            node = self.workflow.nodes.get(node_id)

            # Validate node_order and nodes are in sync
            if not node:
                logger.warning(f"Node {node_id} in node_order but not in nodes dict, skipping")
                nodes_checked += 1
                continue

            if node.type == "reflect" and node.status not in ["completed", "skipped"]:
                # Jump directly to this reflect node
                self.workflow.current_node_index = i
                logger.info(
                    f"Jumping to reflect node '{node_id}' at index {i} for recovery"
                )
                return True

            nodes_checked += 1

        logger.warning(
            f"No reflect node found in execution path for recovery (checked {nodes_checked} nodes)"
        )
        return False

    def jump_to_output_node(self) -> bool:
        """
        Jump directly to the output node in the workflow execution path.

        When strategies are exhausted or reasoning fails, we want to skip
        all remaining nodes and go directly to the output node for report generation.

        Returns:
            True if successfully jumped to the output node, False otherwise
        """
        if not self.workflow or not self.workflow.nodes:
            return False

        # Find the output node
        output_node_id = None
        for node_id, node in self.workflow.nodes.items():
            if node.type == "output":
                output_node_id = node_id
                break

        if not output_node_id:
            logger.warning("No output node found in workflow")
            return False

        # Find output node index in node_order
        current_idx = self.workflow.current_node_index
        if current_idx is None:
            return False

        # Search forward in node_order for the output node
        max_search_range = min(len(self.workflow.node_order), current_idx + 100)

        for i in range(current_idx + 1, max_search_range):
            node_id = self.workflow.node_order[i]
            if node_id == output_node_id:
                # Jump directly to output node
                self.workflow.current_node_index = i
                logger.info(
                    f"Jumping to output node '{output_node_id}' at index {i} for report generation"
                )
                return True

        # If output node is before current position (shouldn't happen normally),
        # try to add it to the end
        if output_node_id not in self.workflow.node_order:
            self.workflow.node_order.append(output_node_id)
            self.workflow.current_node_index = len(self.workflow.node_order) - 1
            logger.info(
                f"Added output node to node_order at index {self.workflow.current_node_index}"
            )
            return True

        logger.warning(f"Output node not found after current position in node_order")
        return False

    def find_output_node(self):
        """Find and return the output node."""
        if not self.workflow or not self.workflow.nodes:
            return None

        for node in self.workflow.nodes.values():
            if node.type == "output":
                return node
        return None

    def find_output_node_index(self, output_node_id: str) -> Optional[int]:
        """Find the index of the output node in node_order."""
        if not self.workflow:
            return None

        current_idx = self.workflow.current_node_index
        if current_idx is None:
            return None

        max_search_range = min(len(self.workflow.node_order), current_idx + 100)

        for i in range(current_idx + 1, max_search_range):
            node_id = self.workflow.node_order[i]
            if node_id == output_node_id:
                return i

        return None
