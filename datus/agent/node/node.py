# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from agents import Tool

from datus.agent.error_handling import ErrorHandlerMixin
from datus.agent.error_handling import NodeErrorResult as AgentNodeErrorResult
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.models.base import LLMBaseModel
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from datus.schemas.date_parser_node_models import DateParserInput, DateParserResult
from datus.schemas.fix_node_models import FixInput
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput, GenSQLNodeResult
from datus.schemas.node_models import (
    BaseInput,
    BaseResult,
    ExecuteSQLInput,
    ExecuteSQLResult,
    GenerateSQLInput,
    GenerateSQLResult,
    OutputInput,
    OutputResult,
    ReflectionResult,
)
from datus.schemas.reason_sql_node_models import ReasoningResult
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.utils.error_handling import NodeErrorResult as UtilsNodeErrorResult
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datus.agent.workflow import Workflow


def _run_async_stream_to_result(node: "Node") -> BaseResult:
    """
    Run node.execute_stream() to completion and materialize result.

    This adaptor allows synchronous runner code (Node.run()) to execute nodes
    that only implement the async streaming interface (execute_stream).

    The adaptor handles event loop management to avoid conflicts with existing
    running loops (e.g., pytest-asyncio) by running the async operation in a
    separate thread with its own event loop.

    ActionHistory events yielded by execute_stream are forwarded to the node's
    action_history_manager for proper event tracking.
    """
    import concurrent.futures

    def _run_async_in_thread():
        """Run the async operation in a separate thread to avoid event loop conflicts."""

        async def _consume_stream():
            """Consume the async generator and collect ActionHistory events."""
            last_action = None
            async for action in node.execute_stream(
                node.action_history_manager if hasattr(node, "action_history_manager") else None
            ):
                last_action = action  # Keep track of the last action
                try:
                    # Forward ActionHistory to action_history_manager if available
                    if hasattr(node, "action_history_manager") and node.action_history_manager:
                        node.action_history_manager.add_action(action)
                except Exception as e:
                    logger.debug(f"Failed to record action in adaptor: {e}")
                    # Continue processing - don't fail the whole execution
            return last_action

        # Use asyncio.run for clean event loop management
        return asyncio.run(_consume_stream())

    try:
        # Run async operation in a thread to avoid conflicts with pytest's event loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_async_in_thread)
            last_action = future.result(timeout=60)  # 60 second timeout

        # Infer result from the last ActionHistory and node state
        from datus.schemas.action_history import ActionStatus
        from datus.schemas.base import BaseResult

        # If node has already set self.result (like IntentAnalysisNode), use it
        if hasattr(node, "result") and node.result is not None:
            return node.result

        # Otherwise, infer from the last ActionHistory
        if last_action and last_action.status == ActionStatus.SUCCESS:
            return BaseResult(success=True)
        elif last_action and last_action.status == ActionStatus.FAILED:
            error_msg = (
                last_action.output.get("error", "Stream execution failed")
                if last_action.output
                else "Stream execution failed"
            )
            return BaseResult(success=False, error=error_msg)
        else:
            # No action or unknown status, assume success for backward compatibility
            return BaseResult(success=True)

    except Exception as e:
        # Convert async exceptions to error result for consistent error handling
        from datus.schemas.base import BaseResult

        return BaseResult(success=False, error=str(e))


def execute_with_async_stream(node: "Node") -> BaseResult:
    """
    Helper method for nodes that need to execute via async stream.

    This ensures self.result is properly set, which is required for Node.run()
    success checking. Use this instead of directly calling _run_async_stream_to_result()
    in execute() methods.

    Args:
        node: The node instance to execute

    Returns:
        BaseResult: Execution result with self.result properly set on the node
    """
    node.result = _run_async_stream_to_result(node)
    return node.result


class Node(ErrorHandlerMixin, ABC):
    """
    Represents a single node in a workflow.
    """

    @classmethod
    def new_instance(
        cls,
        node_id: str,
        description: str,
        node_type: str,
        input_data: BaseInput = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[List[Tool]] = None,
        node_name: Optional[str] = None,
    ):
        from datus.agent.node import (
            BeginNode,
            ChatAgenticNode,
            CompareNode,
            DateParserNode,
            DocSearchNode,
            ExecuteSQLNode,
            FixNode,
            GenerateSQLNode,
            GenSQLAgenticNode,
            HitlNode,
            IntentAnalysisNode,
            OutputNode,
            ParallelNode,
            ReasonSQLNode,
            ReflectNode,
            ResultValidationNode,
            SchemaDiscoveryNode,
            SchemaLinkingNode,
            SchemaValidationNode,
            SearchMetricsNode,
            SelectionNode,
            SubworkflowNode,
        )

        if node_type == NodeType.TYPE_SCHEMA_LINKING:
            return SchemaLinkingNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_GENERATE_SQL:
            return GenerateSQLNode(node_id, description, node_type, input_data, agent_config, tools)
        elif node_type == NodeType.TYPE_EXECUTE_SQL:
            return ExecuteSQLNode(node_id, description, node_type, input_data, agent_config, tools)
        elif node_type == NodeType.TYPE_REASONING:
            return ReasonSQLNode(node_id, description, node_type, input_data, agent_config, tools)
        elif node_type == NodeType.TYPE_DOC_SEARCH:
            return DocSearchNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_OUTPUT:
            return OutputNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_FIX:
            return FixNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_REFLECT:
            return ReflectNode(node_id, description, node_type, input_data, agent_config, tools)
        elif node_type == NodeType.TYPE_HITL:
            return HitlNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_BEGIN:
            return BeginNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_SEARCH_METRICS:
            return SearchMetricsNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_INTENT_ANALYSIS:
            return IntentAnalysisNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_SCHEMA_DISCOVERY:
            return SchemaDiscoveryNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_SCHEMA_VALIDATION:
            return SchemaValidationNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_RESULT_VALIDATION:
            return ResultValidationNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_PARALLEL:
            return ParallelNode(node_id, description, node_type, input_data, agent_config, tools)
        elif node_type == NodeType.TYPE_SELECTION:
            return SelectionNode(node_id, description, node_type, input_data, agent_config, tools)
        elif node_type == NodeType.TYPE_SUBWORKFLOW:
            return SubworkflowNode(node_id, description, node_type, input_data, agent_config, tools)
        elif node_type == NodeType.TYPE_COMPARE:
            return CompareNode(node_id, description, node_type, input_data, agent_config, tools)
        elif node_type == NodeType.TYPE_DATE_PARSER:
            return DateParserNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_CHAT:
            return ChatAgenticNode(node_id, description, node_type, input_data, agent_config, tools)
        elif node_type == NodeType.TYPE_GENSQL:
            return GenSQLAgenticNode(node_id, description, node_type, input_data, agent_config, tools, node_name)
        else:
            raise ValueError(f"Invalid node type: {node_type}")

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: BaseInput = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Initialize a node with its metadata.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of node (e.g., sql_generation, data_validation)
            input_data: Input data for the node
        """
        if node_type not in NodeType.ACTION_TYPES and node_type not in NodeType.CONTROL_TYPES:
            raise ValueError(f"Invalid node type: {node_type}")

        self.id = node_id
        self.description = description
        self.type = node_type
        self.input = input_data
        self.status = "pending"  # pending, running, completed, failed
        self.result = None
        self.start_time = None
        self.end_time = None
        self.dependencies = []  # IDs of nodes that must complete before this one
        self.metadata = {}
        self.agent_config = agent_config
        self.model = None
        self.tools = tools

    def _initialize(self):
        """Initialize the model for this node"""
        model_name = None
        nodes_config = {}

        # Check if model is already initialized (e.g., by AgenticNode or subworkflow config)
        if self.model and isinstance(self.model, LLMBaseModel):
            # Model already initialized, skip re-initialization
            logger.debug(f"Model already initialized for node {self.type}, skipping _initialize")
            return
        elif self.model:
            # self.model contains a model name string
            model_name = self.model
        else:
            # Fall back to agent config
            nodes_config = self.agent_config.nodes
            if self.type in nodes_config:
                node_config = nodes_config[self.type]
                model_name = node_config.model
                node_input = node_config.input
                # If self.input is None, use node_input directly
                if self.input is None:
                    self.input = node_input
                # Otherwise, apply non-None values from node_input as defaults
                elif node_input is not None:
                    for attr, value in node_input.__dict__.items():
                        if value is not None:
                            setattr(self.input, attr, value)

        llm_model = LLMBaseModel.create_model(model_name=model_name, agent_config=self.agent_config)
        logger.info(
            f"Initializing model type: {llm_model.model_config.type}"
            f", model name {llm_model.model_config.model} for node {self.type}"
        )

        if (
            hasattr(llm_model, "set_context")
            and hasattr(self, "workflow")
            and self.workflow
            and llm_model.model_config.save_llm_trace
        ):
            llm_model.set_context(workflow=self.workflow, current_node=self)

        self.model = llm_model

    @abstractmethod
    def update_context(self, workflow: "Workflow") -> Dict:
        pass

    @abstractmethod
    def setup_input(self, workflow: "Workflow") -> Dict:
        pass

    def start(self):
        """
        Mark the node as started.
        """
        self.status = "running"
        self.start_time = time.time()

    def complete(self, result: BaseResult):
        """
        Mark the node as completed with a result.

        Args:
            result: The result of the node execution
        """
        final_status = "completed" if result.success else "failed"
        logger.debug(f"Node.complete: type={self.type}, result.success={result.success}, final_status={final_status}")
        self.status = final_status
        self.result = result
        self.end_time = time.time()

    def fail(self, error: str = None):
        """
        Mark the node as failed with an error message.

        Args:
            error: The error message explaining the failure
        """
        self.status = "failed"
        if error:
            self.result = BaseResult(success=False, error=error)
        self.end_time = time.time()

    @abstractmethod
    def execute(self) -> BaseResult:
        pass

    @abstractmethod
    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the node with streaming support.

        Each subclass should implement this method to call their corresponding _stream() method.

        Args:
            action_history_manager: Manager for tracking action history

        Yields:
            ActionHistory: Progress updates during node execution
        """

    def run(self):
        """Execute the node based on its type and update status."""
        try:
            self._initialize()
            self.start()

            if self.type in NodeType.ACTION_TYPES or self.type in NodeType.CONTROL_TYPES:
                # Check if node prefers async streaming over sync execute
                # This allows sync runner (Node.run) to execute async streaming nodes like ChatAgenticNode
                if hasattr(self, "execute_stream") and callable(getattr(self, "execute_stream")):
                    # Try to call execute, if it throws NotImplementedError, use async stream adaptor
                    try:
                        self.execute()
                    except NotImplementedError:
                        # execute not implemented, use async stream adaptor to run execute_stream
                        self.result = _run_async_stream_to_result(self)
                else:
                    # No execute_stream available, use regular execute
                    self.execute()

                # REFLECT type always completes successfully, others check result
                logger.debug(
                    f"Node.run checking result: type={self.type}, result_type={type(self.result)}, "
                    f"result_is_not_None={self.result is not None}, "
                    f"result_success={getattr(self.result, 'success', 'N/A')}"
                )
                if self.type == NodeType.TYPE_REFLECT or (self.result is not None and self.result.success):
                    logger.info(f"Node.run calling complete for {self.type}")
                    self.complete(self.result)
                else:
                    # Enhanced error handling for failed results
                    if isinstance(self.result, (UtilsNodeErrorResult, AgentNodeErrorResult)):
                        # Already a standardized error result
                        self.fail(self.result.error_message)
                    else:
                        # Legacy error result, wrap it
                        error_msg = f"{self.type} node execution failed: {self.result}"
                        logger.error(error_msg)
                        self.fail(error_msg)
            else:
                error_msg = f"Invalid node type: {self.type}"
                logger.error(error_msg)
                self.fail(error_msg)
        except Exception as e:
            # Use unified error handling for unexpected exceptions
            error_result = self.create_error_result(
                ErrorCode.NODE_EXECUTION_FAILED,
                f"Unexpected error during {self.type} execution: {str(e)}",
                "node_execution",
            )
            logger.error(f"Node execution failed: {error_result.error}", exc_info=True)
            self.fail(error_result.error)
            self.result = error_result

        return self.result

    async def run_async(self):
        return self.run()

    async def run_stream(self, action_history_manager: Optional[ActionHistoryManager] = None):
        """Execute the node with streaming support based on its type and update status."""
        try:
            self._initialize()
            self.start()

            if self.type in NodeType.ACTION_TYPES or self.type in NodeType.CONTROL_TYPES:
                # Execute with streaming and collect results
                async for action in self.execute_stream(action_history_manager):
                    yield action

                # REFLECT type always completes successfully, others check result
                if self.type == NodeType.TYPE_REFLECT or (self.result is not None and self.result.success):
                    self.complete(self.result)
                else:
                    logger.error(f"{self.type} node execution failed: {self.result}")
                    self.fail(f"{self.type} node execution failed: {self.result}")
            else:
                raise ValueError(f"Invalid node type: {self.type}")
        except Exception as e:
            logger.error(f"Node streaming execution failed: {str(e)}")
            self.fail(str(e))

    def add_dependency(self, node_id: str):
        """
        Add a dependency to this node.

        Args:
            node_id: ID of the node that must complete before this one
        """
        if node_id not in self.dependencies:
            self.dependencies.append(node_id)

    def _sql_connector(self, database_name: str = "") -> BaseSqlConnector:
        return db_manager_instance(self.agent_config.namespaces).get_conn(
            self.agent_config.current_namespace,
            database_name,
        )

    def to_dict(self) -> Dict:
        """
        Convert the node to a dictionary representation.

        Returns:
            Dictionary representation of the node
        """
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type,
            "input": (
                dict(self.input) if isinstance(self.input, BaseInput) else self.input
            ),  # try to use BaseInput for all input data
            "status": self.status,
            "result": dict(self.result) if isinstance(self.result, BaseResult) else self.result,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any], agent_config: Optional[AgentConfig] = None) -> Node:
        """Create a Node instance from dictionary representation."""
        # Convert input data based on a node type
        input_data = node_dict["input"]
        if isinstance(input_data, dict):
            try:
                if node_dict["type"] == NodeType.TYPE_SCHEMA_LINKING:
                    input_data = SchemaLinkingInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_GENERATE_SQL:
                    input_data = GenerateSQLInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_EXECUTE_SQL:
                    input_data = ExecuteSQLInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_OUTPUT:
                    input_data = OutputInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_FIX:
                    input_data = FixInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_DATE_PARSER:
                    input_data = DateParserInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_CHAT:
                    input_data = ChatNodeInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_GENSQL:
                    input_data = GenSQLNodeInput(**input_data)
            except Exception as e:
                logger.warning(f"Failed to convert input data for {node_dict['type']}: {e}")
                input_data = None

        # Create node instance
        node = cls.new_instance(
            node_id=node_dict["id"],
            description=node_dict["description"],
            node_type=node_dict["type"],
            input_data=input_data,
            agent_config=agent_config,
        )

        # Convert result data based on node type
        result_data = node_dict["result"]
        if isinstance(result_data, dict):
            try:
                # TODO: use factory pattern to create the result data
                if node_dict["type"] == NodeType.TYPE_SCHEMA_LINKING:
                    result_data = SchemaLinkingResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_GENERATE_SQL:
                    result_data = GenerateSQLResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_EXECUTE_SQL:
                    result_data = ExecuteSQLResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_OUTPUT:
                    result_data = OutputResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_REFLECT:
                    result_data = ReflectionResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_REASONING:
                    result_data = ReasoningResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_DATE_PARSER:
                    result_data = DateParserResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_CHAT:
                    result_data = ChatNodeResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_GENSQL:
                    result_data = GenSQLNodeResult(**result_data)
                elif "success" in result_data:
                    result_data = BaseResult(**result_data)
            except Exception as e:
                logger.warning(f"Failed to convert result data for {node_dict['type']}: {e}")
                result_data = None

        # Set additional attributes
        node.status = node_dict["status"]
        node.result = result_data
        node.start_time = node_dict["start_time"]
        node.end_time = node_dict["end_time"]
        node.dependencies = node_dict["dependencies"]
        node.metadata = node_dict["metadata"]

        return node

    def _create_error_result(
        self,
        error_code: ErrorCode,
        error_message: str,
        operation: str,
        error_details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ) -> NodeErrorResult:
        """
        Create a standardized error result for this node.

        Args:
            error_code: The error code
            error_message: Human-readable error message
            operation: The operation that failed
            error_details: Additional error details
            retryable: Whether the error is retryable

        Returns:
            NodeErrorResult with comprehensive error information
        """
        from datus.utils.error_handling import _create_node_error_result

        return _create_node_error_result(self, error_code, error_message, operation, error_details, retryable)

    def _summarize_input(self) -> Dict[str, Any]:
        """
        Generate a summary of node input for error context.

        Returns:
            Dictionary with input summary
        """
        if not self.input:
            return {}

        input_obj = self.input
        summary = {}

        # Extract common fields that are useful for debugging
        common_fields = ["sql_query", "task", "database_name", "table_schemas", "task_id"]
        for field in common_fields:
            if hasattr(input_obj, field):
                value = getattr(input_obj, field)
                if isinstance(value, str) and len(value) > 100:
                    summary[field] = value[:100] + "..."
                elif isinstance(value, list) and len(value) > 3:
                    summary[field] = f"{len(value)} items"
                else:
                    summary[field] = value

        return summary
