# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import asyncio
import csv
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from datus.agent.agent import Agent
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SqlTask
from datus.storage.task import TaskStore
from datus.utils.loggings import get_logger

from ..utils.json_utils import to_str
from .auth import auth_service, get_current_client
from .event_converter import DeepResearchEventConverter
from .models import (
    ChatResearchRequest,
    DeepResearchEventType,
    ErrorEvent,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    Mode,
    RunWorkflowRequest,
    RunWorkflowResponse,
    TokenResponse,
)

logger = get_logger(__name__)

_form_required = Form(...)
_form_client_id = Form(...)
_form_client_secret = Form(...)
_form_grant_type = Form(...)
_depends_get_current_client = Depends(get_current_client)


@dataclass
class RunningTask:
    """Represents a currently running task."""

    task_id: str
    task: asyncio.Task[Any]
    created_at: datetime
    status: str  # 'running' | 'cancelled' | 'completed' | 'failed'
    meta: Optional[Dict[str, Any]] = None


class DatusAPIService:
    """Main service class for Datus Agent API."""

    def __init__(self, args: argparse.Namespace):
        self.agents: Dict[str, Agent] = {}
        self.agent_config = None
        self.args = args
        self.task_store = None
        self.running_tasks: Dict[str, RunningTask] = {}
        self.running_tasks_lock = asyncio.Lock()

    async def register_running_task(
        self, task_id: str, task: asyncio.Task[Any], meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a running task in the registry."""
        async with self.running_tasks_lock:
            self.running_tasks[task_id] = RunningTask(
                task_id=task_id, task=task, created_at=datetime.now(), status="running", meta=meta
            )

    async def unregister_running_task(self, task_id: str) -> None:
        """Unregister a task from the registry."""
        async with self.running_tasks_lock:
            self.running_tasks.pop(task_id, None)

    async def get_running_task(self, task_id: str) -> Optional[RunningTask]:
        """Get a running task by ID."""
        async with self.running_tasks_lock:
            return self.running_tasks.get(task_id)

    async def get_all_running_tasks(self) -> Dict[str, RunningTask]:
        """Get all running tasks."""
        async with self.running_tasks_lock:
            return dict(self.running_tasks)

    async def _run_streaming_workflow_task(self, req: RunWorkflowRequest, current_client: str, task_id: str) -> None:
        """Run a streaming workflow task and handle completion."""
        try:
            # The actual streaming work is handled by generate_sse_stream
            # This task just ensures proper cleanup - it will be cancelled when the client disconnects
            await asyncio.sleep(0)  # Allow task registration to complete

            # If we get here, the streaming completed successfully
            logger.info(f"Streaming workflow task {task_id} completed successfully")
            if self.task_store:
                self.task_store.update_task(task_id, status="completed")

        except asyncio.CancelledError:
            logger.info(f"Streaming workflow task {task_id} was cancelled")
            if self.task_store:
                self.task_store.update_task(task_id, status="cancelled")
            raise
        except Exception as e:
            logger.error(f"Streaming workflow task {task_id} failed: {e}")
            if self.task_store:
                self.task_store.update_task(task_id, status="failed")
            raise
        finally:
            # Clean up the task from registry when done
            await self.unregister_running_task(task_id)

    async def _generate_registered_stream(self, task_id: str):
        """Generate stream for a registered task."""
        try:
            # Get the task request from the registry
            running_task = await self.get_running_task(task_id)
            if not running_task or not running_task.meta:
                raise ValueError(f"No registered task found for {task_id}")

            req_data = running_task.meta.get("request")
            current_client = running_task.meta.get("client")

            if not req_data or not current_client:
                raise ValueError(f"Invalid task metadata for {task_id}")

            # Reconstruct the request
            from .models import RunWorkflowRequest

            req = RunWorkflowRequest(**req_data)

            # Generate the stream
            async for event in generate_sse_stream(req, current_client):
                yield event

        except Exception as e:
            logger.error(f"Error in registered stream for task {task_id}: {e}")
            yield f"event: error\ndata: {to_str({'error': str(e)})}\n\n"

    async def _run_chat_research_task(self, req: ChatResearchRequest, current_client: str, task_id: str) -> None:
        """Run a chat research task and handle completion."""
        try:
            # The actual work is handled by _generate_chat_research_stream
            # This task just ensures proper cleanup - it will be cancelled when the client disconnects
            await asyncio.sleep(0)  # Allow task registration to complete

            # If we get here, the streaming completed successfully
            logger.info(f"Chat research task {task_id} completed successfully")
            if self.task_store:
                self.task_store.update_task(task_id, status="completed")

        except asyncio.CancelledError:
            logger.info(f"Chat research task {task_id} was cancelled")
            if self.task_store:
                self.task_store.update_task(task_id, status="cancelled")
            raise
        except Exception as e:
            logger.error(f"Chat research task {task_id} failed: {e}")
            if self.task_store:
                self.task_store.update_task(task_id, status="failed")
            raise
        finally:
            # Clean up the task from registry when done
            await self.unregister_running_task(task_id)

    async def _generate_chat_research_stream(self, task_id: str):
        """Generate chat research stream for a registered task."""
        try:
            # Get the task request from the registry
            running_task = await self.get_running_task(task_id)
            if not running_task or not running_task.meta:
                raise ValueError(f"No registered task found for {task_id}")

            req_data = running_task.meta.get("request")
            current_client = running_task.meta.get("client")

            if not req_data or not current_client:
                raise ValueError(f"Invalid task metadata for {task_id}")

            # Reconstruct the request
            from .models import ChatResearchRequest

            req = ChatResearchRequest(**req_data)

            # Generate the stream
            async for event in self.run_chat_research_stream(req, current_client):
                yield event

        except Exception as e:
            logger.error(f"Error in chat research stream for task {task_id}: {e}")
            yield f"data: {to_str({'error': str(e)})}\n\n"

    async def _run_sync_workflow_task(self, req: RunWorkflowRequest, current_client: str, task_id: str):
        """Run a synchronous workflow task and return result."""
        try:
            # Run the synchronous workflow
            result = await self.run_workflow(req, current_client)

            # If we get here, the workflow completed successfully
            logger.info(f"Synchronous workflow task {task_id} completed successfully")
            if self.task_store:
                self.task_store.update_task(task_id, status="completed")

            return result

        except asyncio.CancelledError:
            logger.info(f"Synchronous workflow task {task_id} was cancelled")
            if self.task_store:
                self.task_store.update_task(task_id, status="cancelled")
            raise
        except Exception as e:
            logger.error(f"Synchronous workflow task {task_id} failed: {e}")
            if self.task_store:
                self.task_store.update_task(task_id, status="failed")
            raise
        finally:
            # Clean up the task from registry when done
            await self.unregister_running_task(task_id)

    async def initialize(self):
        """Initialize the service with default configurations."""
        try:
            # Load default agent configuration
            self.agent_config = load_agent_config(**vars(self.args))
            logger.info("Agent configuration loaded successfully")

            # Initialize task store
            task_db_path = os.path.join(self.agent_config.rag_base_path, "task")
            self.task_store = TaskStore(task_db_path)
            logger.info("Task store initialized successfully")

            # Clean up old tasks on startup
            cleaned_count = self.task_store.cleanup_old_tasks(hours=24)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old task records on startup")
        except Exception as e:
            logger.error(f"Failed to load agent configuration: {e}")
            self.agent_config = None

    def _parse_csv_to_list(self, csv_string: str) -> List[Dict[str, Any]]:
        """Parse CSV string to list of dictionaries."""
        try:
            if not csv_string or not csv_string.strip():
                return []

            reader = csv.DictReader(StringIO(csv_string.strip()))
            return [dict(row) for row in reader]
        except Exception as e:
            logger.warning(f"Failed to parse CSV data: {e}")
            return []

    def _generate_task_id(self, client_id: str, prefix: str = "") -> str:
        """Generate task ID using client_id, optional prefix and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        if prefix:
            return f"{prefix}_{client_id}_{timestamp}"
        return f"{client_id}_{timestamp}"

    async def _validate_task_id_uniqueness(self, task_id: str, current_client: str) -> None:
        """Validate that the task_id is unique for the current client.

        Raises:
            HTTPException: If task_id already exists for this client
        """
        # Check running tasks
        running_task = await self.get_running_task(task_id)
        if running_task:
            # Check if it's owned by the same client
            task_client = running_task.meta.get("client") if running_task.meta else None
            if task_client == current_client:
                raise HTTPException(
                    status_code=409, detail=f"Task {task_id} already exists and is running for client {current_client}"
                )
            else:
                raise HTTPException(status_code=409, detail=f"Task {task_id} already exists for different client")

        # Check completed tasks in database (last 24 hours)
        if self.task_store:
            task_data = self.task_store.get_task(task_id)
            if task_data:
                # Check if task was created by same client recently
                task_client = task_data.get("client")
                if task_client == current_client:
                    created_at = task_data.get("created_at")
                    if created_at:
                        # Convert to timestamp if needed
                        if isinstance(created_at, str):
                            try:
                                created_ts = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
                            except (ValueError, TypeError):
                                created_ts = 0
                        else:
                            created_ts = created_at

                        # Check if task was created within last 24 hours
                        cutoff_time = datetime.now().timestamp() - (24 * 60 * 60)
                        if created_ts > cutoff_time:
                            raise HTTPException(
                                status_code=409,
                                detail=f"Task {task_id} was recently executed by client {current_client}. Please use a different task_id.",
                            )

    def get_agent(self, namespace: str) -> Agent:
        """Get or create an agent for the specified namespace."""
        if namespace not in self.agents:
            if not self.agent_config:
                raise HTTPException(status_code=500, detail="Agent configuration not available")

            self.agent_config.current_namespace = namespace
            # Create agent instance
            self.agents[namespace] = Agent(self.args, self.agent_config)
            logger.info(f"Created new agent for namespace: {namespace}")

        return self.agents[namespace]

    async def _create_sql_task(self, request: RunWorkflowRequest, task_id: str, agent: Agent) -> SqlTask:
        """Create SQL task from request parameters."""
        # Load default metric_meta (will return default values if not configured)
        metric_meta = agent.global_config.current_metric_meta("default")
        external_knowledge = metric_meta.ext_knowledge
        domain = metric_meta.domain
        layer1 = metric_meta.layer1
        layer2 = metric_meta.layer2

        # Override with request parameters if provided
        if request.domain is not None:
            domain = request.domain
        if request.layer1 is not None:
            layer1 = request.layer1
        if request.layer2 is not None:
            layer2 = request.layer2
        if request.ext_knowledge is not None:
            external_knowledge = request.ext_knowledge

        # Auto-inject external knowledge in plan_mode when not explicitly provided
        if (
            request.plan_mode
            and request.ext_knowledge is None
            and getattr(agent.global_config, "plan_mode_auto_inject_ext_knowledge", True)
        ):
            try:
                # Import here to avoid circular imports
                from datus.agent.intent_detection import detect_sql_intent
                from datus.storage.ext_knowledge.store import ExtKnowledgeStore

                # Get model for LLM fallback (use target model from config)
                model = None
                try:
                    model = agent.global_config.get_model()
                except Exception as e:
                    logger.warning(f"Could not get model for intent detection: {e}")

                # Detect SQL intent
                intent_result = await detect_sql_intent(
                    text=request.task,
                    model=model,
                    keyword_threshold=getattr(agent.global_config, "intent_detector_keyword_threshold", 1),
                    llm_confidence_threshold=getattr(
                        agent.global_config, "intent_detector_llm_confidence_threshold", 0.7
                    ),
                )

                # If SQL-related intent detected, search for relevant external knowledge
                if intent_result.intent in ["sql_generation", "sql_review", "sql_related"]:
                    try:
                        ext_knowledge_store = ExtKnowledgeStore(agent.global_config.rag_storage_path())

                        # Check if store has knowledge
                        if ext_knowledge_store.table_size() > 0:
                            search_results = ext_knowledge_store.search_knowledge(
                                query_text=request.task,
                                domain=domain or "",
                                layer1=layer1 or "",
                                layer2=layer2 or "",
                                top_n=5,
                            )

                            # Build knowledge string from results
                            if len(search_results) > 0:
                                knowledge_items = []
                                for result in search_results:
                                    terminology = result.get("terminology", "")
                                    explanation = result.get("explanation", "")
                                    if terminology and explanation:
                                        knowledge_items.append(f"- {terminology}: {explanation}")

                                if knowledge_items:
                                    auto_injected_knowledge = "\n".join(knowledge_items)
                                    if external_knowledge:
                                        external_knowledge = f"{external_knowledge}\n\nAuto-detected relevant knowledge:\n{auto_injected_knowledge}"
                                    else:
                                        external_knowledge = (
                                            f"Auto-detected relevant knowledge:\n{auto_injected_knowledge}"
                                        )

                                    logger.info(
                                        f"Auto-injected {len(knowledge_items)} knowledge items for task: {request.task[:50]}..."
                                    )

                                    # Add metadata to indicate auto-injection occurred
                                    if not hasattr(request, "_auto_injected_knowledge"):
                                        request._auto_injected_knowledge = []
                                    request._auto_injected_knowledge.extend(knowledge_items)

                    except Exception as e:
                        logger.warning(f"Failed to auto-inject external knowledge: {e}")

            except Exception as e:
                logger.warning(f"Intent detection failed: {e}")

        return SqlTask(
            id=task_id,
            task=request.task,
            database_type=agent.global_config.db_type,  # Add database type from agent global config
            catalog_name=request.catalog_name or "",
            database_name=request.database_name or "default",
            schema_name=request.schema_name or "",
            domain=domain,
            layer1=layer1,
            layer2=layer2,
            external_knowledge=external_knowledge,
            output_dir=agent.global_config.output_dir,
            current_date=request.current_date,
        )

    def _create_response(
        self,
        task_id: str,
        request: RunWorkflowRequest,
        status: str,
        sql_query: str = None,
        query_results: list = None,
        metadata: dict = None,
        error: str = None,
        execution_time: float = None,
    ) -> RunWorkflowResponse:
        """Create standardized workflow response."""
        return RunWorkflowResponse(
            task_id=task_id,
            status=status,
            workflow=request.workflow,
            sql=sql_query,
            result=query_results,
            metadata=metadata,
            error=error,
            execution_time=execution_time,
        )

    async def run_workflow(self, request: RunWorkflowRequest, client_id: str = None) -> RunWorkflowResponse:
        """Execute a workflow synchronously and return results."""
        task_id = request.task_id or self._generate_task_id(client_id or "unknown")
        start_time = time.time()

        try:
            # Initialize task tracking in database
            if self.task_store:
                self.task_store.create_task(task_id, request.task)

            # Get agent for the namespace
            agent = self.get_agent(request.namespace)

            # Create SQL task
            sql_task = await self._create_sql_task(request, task_id, agent)

            # Execute workflow synchronously using isolated runner
            runner = agent.create_workflow_runner()
            result = runner.run(sql_task)
            workflow = runner.workflow
            execution_time = time.time() - start_time

            if result and result.get("status") == "completed":
                # Extract SQL and results from the workflow
                sql_query = None
                query_results = None

                # Get the last SQL context from workflow using the correct method
                try:
                    if workflow:
                        last_sql_context = workflow.get_last_sqlcontext()
                        sql_query = last_sql_context.sql_query
                        query_results_raw = last_sql_context.sql_return

                        # Convert CSV string to list of dictionaries for API response
                        if query_results_raw and isinstance(query_results_raw, str):
                            query_results = self._parse_csv_to_list(query_results_raw)
                        else:
                            query_results = None

                        # Update task in database (store as string)
                        if self.task_store:
                            self.task_store.update_task(
                                task_id,
                                sql_query=sql_query,
                                sql_result=str(query_results_raw) if query_results_raw else "",
                            )
                except Exception as e:
                    logger.warning(f"Could not extract SQL context for task {task_id}: {e}")
                    # Continue without SQL data

                # Update task status to completed
                if self.task_store:
                    self.task_store.update_task(task_id, status="completed")

                return self._create_response(
                    task_id=task_id,
                    request=request,
                    status="completed",
                    sql_query=sql_query,
                    query_results=query_results,
                    metadata=result,
                    execution_time=execution_time,
                )
            else:
                # Update task status to failed
                if self.task_store:
                    self.task_store.update_task(task_id, status="failed")

                return self._create_response(
                    task_id=task_id,
                    request=request,
                    status="failed",
                    metadata=result,
                    error="Workflow execution failed",
                    execution_time=execution_time,
                )

        except Exception as e:
            logger.error(f"Error executing workflow {task_id}: {e}")
            # Update task status to failed
            if self.task_store:
                self.task_store.update_task(task_id, status="failed")
            return self._create_response(
                task_id=task_id, request=request, status="error", error=str(e), execution_time=time.time() - start_time
            )

    async def run_workflow_stream(
        self, request: RunWorkflowRequest, client_id: str = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute a workflow with streaming support and yield progress updates."""
        task_id = request.task_id or self._generate_task_id(client_id or "unknown")

        try:
            # Get agent for the namespace
            agent = self.get_agent(request.namespace)

            # Create SQL task
            sql_task = await self._create_sql_task(request, task_id, agent)

            # Create action history manager for tracking
            action_history_manager = ActionHistoryManager()

            # Execute workflow with streaming
            async for action in agent.run_stream(sql_task, action_history_manager=action_history_manager):
                yield action

        except Exception as e:
            logger.error(f"Error executing streaming workflow {task_id}: {e}")
            # Yield error action
            error_action = ActionHistory(
                action_id="workflow_error",
                role=ActionRole.WORKFLOW,
                messages=f"Workflow execution failed: {str(e)}",
                action_type="error",
                input={"task_id": task_id},
                status=ActionStatus.FAILED,
                output={"error": str(e)},
            )
            yield error_action

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Record user feedback for a task."""
        try:
            if not self.task_store:
                raise HTTPException(status_code=500, detail="Task store not initialized")

            # Record the feedback by updating the user_feedback field
            recorded_data = self.task_store.record_feedback(task_id=request.task_id, status=request.status.value)

            return FeedbackResponse(
                task_id=recorded_data["task_id"], acknowledged=True, recorded_at=recorded_data["recorded_at"]
            )

        except Exception as e:
            logger.error(f"Error recording feedback for task {request.task_id}: {e}")
            return FeedbackResponse(
                task_id=request.task_id,
                acknowledged=False,
                recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

    def _identify_task_type(self, task_text: str) -> str:
        """识别任务类型"""
        task_lower = task_text.lower()

        # SQL审查任务特征
        review_keywords = [
            "审查",
            "review",
            "检查",
            "check",
            "审核",
            "audit",
            "质量",
            "quality",
            "评估",
            "evaluate",
            "分析sql",
            "analyze sql",
        ]
        if any(keyword in task_lower for keyword in review_keywords):
            return "sql_review"

        # 数据分析任务特征
        analysis_keywords = [
            "分析",
            "analysis",
            "对比",
            "compare",
            "趋势",
            "trend",
            "统计",
            "statistics",
            "汇总",
            "summary",
            "报告",
            "report",
        ]
        if any(keyword in task_lower for keyword in analysis_keywords):
            return "data_analysis"

        # 默认Text2SQL
        return "text2sql"

    def _configure_task_processing(self, task_type: str, request: ChatResearchRequest) -> dict:
        """根据任务类型配置处理参数"""

        if task_type == "sql_review":
            # SQL审查：不使用plan模式，使用专门的审查提示词
            return {
                "workflow": "chat_agentic",  # 不使用plan模式
                "plan_mode": False,
                "auto_execute_plan": False,
                "system_prompt": "sql_review",  # 专门的SQL审查提示词（会自动添加_system后缀）
                "output_format": "markdown",  # 指定输出格式
                "required_tool_sequence": [
                    "describe_table",  # 获取表结构信息
                    "search_external_knowledge",  # 检索StarRocks审查规则
                    "read_query",  # 执行查询验证SQL正确性
                    "get_table_ddl",  # 获取表DDL定义
                ],
            }
        elif task_type == "data_analysis":
            # 数据分析：使用plan模式
            return {
                "workflow": "chat_agentic_plan",
                "plan_mode": True,
                "auto_execute_plan": True,
                "system_prompt": "plan_mode",
                "output_format": "json",
            }
        else:  # text2sql
            # Text2SQL：标准模式
            return {
                "workflow": "chat_agentic",
                "plan_mode": False,
                "auto_execute_plan": False,
                "system_prompt": "chat_system",
                "output_format": "json",
            }

    async def run_chat_research_stream(
        self, request: ChatResearchRequest, client_id: str = None
    ) -> AsyncGenerator[str, None]:
        """Execute chat research workflow with DeepResearchEvent streaming."""

        task_id = f"research_{client_id}_{int(time.time())}"

        try:
            # Get agent for the namespace
            agent = self.get_agent(request.namespace)

            # 智能识别任务类型并配置处理参数
            task_type = self._identify_task_type(request.task)
            task_config = self._configure_task_processing(task_type, request)

            # 使用任务配置的工作流名称
            workflow_name = task_config["workflow"]
            sql_task = await self._create_sql_task(
                RunWorkflowRequest(
                    workflow=workflow_name,
                    namespace=request.namespace,
                    catalog_name=request.catalog_name,
                    database_name=request.database_name,
                    schema_name=request.schema_name,
                    task=request.task,
                    current_date=request.current_date,
                    domain=request.domain,
                    layer1=request.layer1,
                    layer2=request.layer2,
                    ext_knowledge=request.ext_knowledge,
                    plan_mode=request.plan_mode,
                    mode="async",
                ),
                task_id,
                agent,
            )

            # Enable plan mode in workflow metadata
            workflow_metadata = {
                "plan_mode": task_config["plan_mode"],
                "auto_execute_plan": task_config["auto_execute_plan"],
                "system_prompt": task_config["system_prompt"].replace("_system", "")
                if task_config["system_prompt"].endswith("_system")
                else task_config["system_prompt"],
                "output_format": task_config["output_format"],
                "task_type": task_type,
                "prompt": request.prompt,
                "prompt_mode": request.prompt_mode,
            }

            # Add required tool sequence if specified (for sql_review tasks)
            if "required_tool_sequence" in task_config:
                workflow_metadata["required_tool_sequence"] = task_config["required_tool_sequence"]

            # Initialize task tracking
            if self.task_store:
                self.task_store.create_task(task_id, request.task)

            # Create event converter
            converter = DeepResearchEventConverter()

            # Execute workflow with streaming and convert events
            async for event_data in converter.convert_stream_to_events(
                agent.run_stream_with_metadata(sql_task, metadata=workflow_metadata)
            ):
                yield event_data

            # Update task status
            if self.task_store:
                self.task_store.update_task(task_id, status="completed")

        except asyncio.CancelledError:
            # 重新抛出 CancelledError，让 asyncio 正确处理取消
            logger.info(f"Chat research task {task_id} was cancelled")
            if self.task_store:
                self.task_store.update_task(task_id, status="cancelled")
            raise
        except Exception as e:
            logger.error(f"Chat research error for task {task_id}: {e}")

            # Enhance error message with user-friendly suggestions
            error_message, suggestions = self._enhance_error_message(str(e))

            error_event = ErrorEvent(
                id=f"error_{int(time.time() * 1000)}",
                timestamp=int(time.time() * 1000),
                event=DeepResearchEventType.ERROR,
                error=error_message,
            )

            # Add suggestions as additional data if available
            if suggestions:
                enhanced_error = error_event.model_dump()
                enhanced_error["suggestions"] = suggestions
                yield f"data: {json.dumps(enhanced_error)}\n\n"
            else:
                yield f"data: {error_event.model_dump_json()}\n\n"

            if self.task_store:
                self.task_store.update_task(task_id, status="failed")

    async def health_check(self) -> HealthResponse:
        """Perform health check on the service."""
        try:
            # Check default agent if available
            database_status = {}
            llm_status = "unknown"

            if self.agent_config:
                # Create a temporary agent for health check using service configuration
                temp_agent = Agent(self.args, self.agent_config)

                # Check database connectivity
                db_check = temp_agent.check_db()
                database_status[self.agent_config.current_namespace] = db_check.get("status", "unknown")

                # Check LLM connectivity
                llm_check = temp_agent.probe_llm()
                llm_status = llm_check.get("status", "unknown")

            return HealthResponse(
                status="healthy", version="1.0.0", database_status=database_status, llm_status=llm_status
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy", version="1.0.0", database_status={"error": str(e)}, llm_status="error"
            )

    def _enhance_error_message(self, error_message: str) -> Tuple[str, List[str]]:
        """
        Enhance error messages with user-friendly descriptions and suggestions.

        Args:
            error_message: The original error message

        Returns:
            Tuple of (enhanced_message, suggestions_list)
        """
        error_lower = error_message.lower()
        suggestions = []

        # Database connection errors
        if any(keyword in error_lower for keyword in ["connection", "connect", "network", "timeout"]):
            enhanced_msg = "数据库连接出现问题"
            suggestions = ["检查数据库服务是否正在运行", "确认网络连接是否正常", "验证数据库连接配置", "稍后重试操作"]

        # Table/column not found errors
        elif any(keyword in error_lower for keyword in ["table", "column", "relation", "does not exist"]):
            enhanced_msg = "查询的表或列不存在"
            suggestions = ["检查表名和列名是否正确", "确认数据库schema", "使用搜索功能查找可用表"]

        # Permission errors
        elif any(keyword in error_lower for keyword in ["permission", "access", "denied", "privilege"]):
            enhanced_msg = "数据库访问权限不足"
            suggestions = ["检查数据库用户权限", "确认有查询相关表的权限", "联系数据库管理员"]

        # SQL syntax errors
        elif any(keyword in error_lower for keyword in ["syntax", "invalid sql", "parse error"]):
            enhanced_msg = "SQL语法错误"
            suggestions = ["检查SQL语句语法", "确认引号和括号匹配", "验证SQL语句结构"]

        # Resource exhausted errors
        elif any(keyword in error_lower for keyword in ["resource", "memory", "disk", "quota"]):
            enhanced_msg = "系统资源不足"
            suggestions = ["减少查询的数据量", "优化查询条件", "分批处理大数据"]

        else:
            enhanced_msg = "执行过程中出现错误"
            suggestions = ["检查输入参数是否正确", "确认系统配置", "联系技术支持"]

        return enhanced_msg, suggestions

    async def cancel_all_running_tasks(self):
        """Cancel all currently running tasks during shutdown."""
        if not hasattr(self, "running_tasks_lock") or not hasattr(self, "running_tasks"):
            return

        async with self.running_tasks_lock:
            tasks_to_cancel = list(self.running_tasks.values())

        if tasks_to_cancel:
            logger.info(f"Cancelling {len(tasks_to_cancel)} running tasks during shutdown")
            for running_task in tasks_to_cancel:
                try:
                    task = running_task.task
                    if not task.done():
                        task.cancel()
                        logger.debug(f"Cancelled task: {running_task.task_id}")
                except Exception as e:
                    logger.warning(f"Error cancelling task {running_task.task_id}: {e}")

            # Wait a bit for tasks to cancel
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[t.task for t in tasks_to_cancel if not t.task.done()], return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not cancel within timeout")
            except Exception as e:
                logger.warning(f"Error waiting for task cancellation: {e}")

        logger.info("Task cancellation completed")


# Global service instance - will be initialized with command line args
service = None


async def generate_sse_stream(req: RunWorkflowRequest, current_client: str):
    """Generate Server-Sent Events stream for workflow execution."""
    import asyncio
    import json
    import time as time_module

    task_id = req.task_id or service._generate_task_id(current_client)
    start_time = time_module.time()
    progress_seq = 0  # Progress sequence counter for ordering

    # Throttling for partial streaming events (max 10 per second)
    last_partial_time = 0
    min_partial_interval = 0.1  # 100ms minimum interval between partial events

    try:
        # Initialize task tracking in database
        if service.task_store:
            service.task_store.create_task(task_id, req.task)

        # Send started event
        yield f"event: started\ndata: {to_str({'task_id': task_id, 'client': current_client})}\n\n"
        await asyncio.sleep(0)  # Allow event loop to flush

        # Execute workflow with streaming
        sql_query = None

        async for action in service.run_workflow_stream(req, current_client):
            # Check if task was cancelled
            running_task = await service.get_running_task(task_id)
            if running_task and running_task.status == "cancelled":
                logger.info(f"Task {task_id} was cancelled, stopping stream")
                break
            progress_seq += 1

            # Map different action types to SSE events with prioritized matching

            # 1. Handle specific high-priority business logic events first
            if action.action_type == "sql_generation" and action.status == "success":
                if action.output and "sql_query" in action.output:
                    sql_query = action.output["sql_query"]
                    # Update task in database
                    if service.task_store:
                        service.task_store.update_task(task_id, sql_query=sql_query)
                    yield f"event: sql_generated\ndata: {to_str({'sql': sql_query, 'progress_seq': progress_seq})}\n\n"

            elif action.action_type == "sql_execution" and action.status == "success":
                output = action.output or {}
                if output.get("has_results"):
                    sql_result = output.get("sql_result", "")
                    # Update task in database
                    if service.task_store:
                        service.task_store.update_task(task_id, sql_result=str(sql_result))
                    result_data = {
                        "row_count": output.get("row_count", 0),
                        "sql_result": sql_result,
                        "progress_seq": progress_seq,
                    }
                    yield f"event: execution_complete\ndata: {to_str(result_data)}\n\n"

            elif action.action_type == "output_generation" and action.status == "success":
                output = action.output or {}
                output_data = {
                    "output_generated": output.get("output_generated", True),
                    "sql_query": output.get("sql_query", ""),
                    "sql_result": output.get("sql_result", ""),
                    "progress_seq": progress_seq,
                }
                yield f"event: output_ready\ndata: {to_str(output_data)}\n\n"

            elif action.action_type == "workflow_completion":
                logger.info(
                    f"Workflow completion action: {action}, action_type: {action.action_type}, status: {action.status}"
                )
                if action.status == "success":
                    # Update task status to completed
                    if service.task_store:
                        service.task_store.update_task(task_id, status="completed")
                    execution_time_ms = int((time_module.time() - start_time) * 1000)
                    yield f"event: done\ndata: {json.dumps({'exec_time_ms': execution_time_ms, 'progress_seq': progress_seq})}\n\n"
                elif action.status == "failed":
                    # Update task status to failed
                    if service.task_store:
                        service.task_store.update_task(task_id, status="failed")
                    error_msg = (action.output or {}).get("error", "Unknown error")
                    yield f"event: error\ndata: {to_str({'error': error_msg, 'progress_seq': progress_seq})}\n\n"
                # For status="processing", do nothing and wait for final status

            elif action.status == "failed":
                error_msg = (action.output or {}).get("error", "Action failed")
                yield f"event: error\ndata: {to_str({'error': error_msg, 'action_id': action.action_id, 'progress_seq': progress_seq})}\n\n"

            # 2. Handle workflow and node progress events
            elif action.action_id == "workflow_initialization":
                progress_data = {
                    "action": "initialization",
                    "status": action.status,
                    "message": action.messages,
                    "progress_seq": progress_seq,
                }
                yield f"event: progress\ndata: {to_str(progress_data)}\n\n"

            elif action.action_id.startswith("node_execution_"):
                node_info = action.input or {}
                node_data = {
                    "action": "node_execution",
                    "status": action.status,
                    "node_type": node_info.get("node_type", ""),
                    "description": node_info.get("description", ""),
                    "message": action.messages,
                    "progress_seq": progress_seq,
                }
                yield f"event: node_progress\ndata: {to_str(node_data)}\n\n"

            # 3. Handle chat-specific streaming events
            elif action.action_type == "chat_response" and action.status == ActionStatus.SUCCESS:
                output = action.output or {}
                chat_data = {
                    "response": output.get("response", ""),
                    "sql": output.get("sql", ""),
                    "tokens_used": output.get("tokens_used", 0),
                    "progress_seq": progress_seq,
                }
                yield f"event: chat_response\ndata: {to_str(chat_data)}\n\n"

            elif action.role == ActionRole.ASSISTANT and action.action_type in (
                "message",
                "llm_generation",
                "thinking",
            ):
                # Handle chat thinking and streaming content
                output = action.output or {}
                partial_content = ""
                is_partial = False

                # Extract partial content from various possible fields
                if "raw_output" in output:
                    partial_content = output.get("raw_output", "")
                    is_partial = True
                elif "content" in output:
                    partial_content = output.get("content", "")
                    is_partial = True
                elif "partial" in output:
                    partial_content = output.get("partial", "")
                    is_partial = True

                if is_partial and partial_content:
                    # Throttle partial events to avoid overwhelming clients (max 10 per second)
                    current_time = time_module.time()
                    time_since_last_partial = current_time - last_partial_time
                    if time_since_last_partial < min_partial_interval:
                        await asyncio.sleep(min_partial_interval - time_since_last_partial)
                        current_time = time_module.time()

                    last_partial_time = current_time

                    # Truncate large partials to avoid overwhelming clients (8KB limit)
                    max_partial_size = 8192
                    is_truncated = len(partial_content) > max_partial_size
                    if is_truncated:
                        partial_content = partial_content[:max_partial_size]

                    stream_data = {
                        "action_type": action.action_type,
                        "content": partial_content,
                        "is_truncated": is_truncated,
                        "node_type": "chat",
                        "progress_seq": progress_seq,
                    }
                    yield f"event: chat_thinking\ndata: {to_str(stream_data)}\n\n"
                else:
                    # Regular chat message without partial content
                    detail_data = {
                        "action_type": action.action_type,
                        "status": action.status,
                        "message": action.messages,
                        "node_type": "chat",
                        "progress_seq": progress_seq,
                    }
                    yield f"event: node_detail\ndata: {to_str(detail_data)}\n\n"

            # 4. Handle expanded node-specific progress for streaming operations
            elif action.action_type in [
                "schema_linking",
                "sql_preparation",
                "sql_generation",
                "sql_execution",
                "output_preparation",
                "output_generation",
                "llm_generation",
                "tool_call",
                "function_call",
                "message",
                "thinking",
                "raw_stream",
                "response",
            ]:
                output = action.output or {}
                detail_data = {
                    "action_type": action.action_type,
                    "status": action.status,
                    "message": action.messages,
                    "progress_seq": progress_seq,
                }

                # Include partial output if available
                if "raw_output" in output:
                    partial_content = output.get("raw_output", "")
                    if partial_content:
                        # Throttle partial events to avoid overwhelming clients
                        current_time = time_module.time()
                        time_since_last_partial = current_time - last_partial_time
                        if time_since_last_partial < min_partial_interval:
                            await asyncio.sleep(min_partial_interval - time_since_last_partial)
                            current_time = time_module.time()

                        last_partial_time = current_time

                        max_partial_size = 8192
                        is_truncated = len(partial_content) > max_partial_size
                        if is_truncated:
                            partial_content = partial_content[:max_partial_size]
                        detail_data["partial_output"] = partial_content
                        detail_data["is_truncated"] = is_truncated
                        yield f"event: node_stream\ndata: {to_str(detail_data)}\n\n"
                    else:
                        yield f"event: node_detail\ndata: {to_str(detail_data)}\n\n"
                else:
                    yield f"event: node_detail\ndata: {to_str(detail_data)}\n\n"

            # 5. Generic fallback for any ActionHistory not matched above
            else:
                # Extract serializable subset to avoid issues with complex objects
                serializable_action = {
                    "action_id": action.action_id,
                    "role": str(action.role) if hasattr(action, "role") else None,
                    "action_type": action.action_type,
                    "messages": action.messages,
                    "status": str(action.status) if hasattr(action, "status") else None,
                    "progress_seq": progress_seq,
                }

                # Include basic input/output info if available and serializable
                if action.input and isinstance(action.input, dict):
                    serializable_action["input_keys"] = list(action.input.keys())
                if action.output and isinstance(action.output, dict):
                    serializable_action["output_keys"] = list(action.output.keys())

                yield f"event: generic_action\ndata: {to_str(serializable_action)}\n\n"

            # Allow event loop to flush network buffer (non-blocking)
            await asyncio.sleep(0)

    except Exception as e:
        logger.error(f"SSE stream error for task {task_id}: {e}")
        yield f"event: error\ndata: {json.dumps({'error': str(e), 'progress_seq': progress_seq})}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle."""
    global service
    args = getattr(app.state, "agent_args", None)
    service = DatusAPIService(args)

    # Startup
    await service.initialize()
    logger.info("Datus API Service started")
    yield
    # Shutdown
    logger.info("Datus API Service shutting down")

    # Cancel all running tasks asynchronously to avoid blocking shutdown
    if service:
        # Create background task for cancellation to prevent blocking lifespan shutdown
        asyncio.create_task(service.cancel_all_running_tasks())
        logger.info("Task cancellation initiated in background")


def create_app(agent_args: argparse.Namespace) -> FastAPI:
    """Create FastAPI app with agent args."""
    app = FastAPI(
        title="Datus Agent API",
        description="FastAPI service for Datus Agent workflow execution",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.agent_args = agent_args

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Route handlers with decorators
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        return {"message": "Datus Agent API", "version": "1.0.0", "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        """Health check endpoint (no authentication required)."""
        return await service.health_check()

    @app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
    async def authenticate(
        client_id: str = _form_client_id, client_secret: str = _form_client_secret, grant_type: str = _form_grant_type
    ) -> TokenResponse:
        """OAuth2 client credentials token endpoint."""
        if grant_type != "client_credentials":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid grant_type. Must be 'client_credentials'"
            )

        if not auth_service.validate_client_credentials(client_id, client_secret):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid client credentials")

        token_data = auth_service.generate_access_token(client_id)
        return TokenResponse(**token_data)

    @app.post("/workflows/run", tags=["workflows"])
    async def run_workflow(
        req: RunWorkflowRequest, request: Request, current_client: str = _depends_get_current_client
    ):
        """Execute a workflow based on the request parameters."""
        try:
            logger.info(f"Workflow request from client: {current_client}, mode: {req.mode}")

            # Generate or validate task_id (use frontend messageId if provided)
            if req.task_id:
                # Validate uniqueness for frontend-provided task_id
                await service._validate_task_id_uniqueness(req.task_id, current_client)
                task_id = req.task_id
                logger.info(f"Using frontend-provided task_id: {task_id}")
            else:
                # Generate new task_id if not provided
                task_id = service._generate_task_id(current_client)
                logger.info(f"Generated new task_id: {task_id}")

            # Check if client accepts server-sent events for async mode
            if req.mode == Mode.ASYNC:
                accept_header = request.headers.get("accept", "")
                if "text/event-stream" not in accept_header:
                    raise HTTPException(
                        status_code=400, detail="For async mode, Accept header must include 'text/event-stream'"
                    )

                # Create and register the streaming task
                stream_task = asyncio.create_task(service._run_streaming_workflow_task(req, current_client, task_id))
                await service.register_running_task(
                    task_id,
                    stream_task,
                    meta={"type": "workflow_stream", "request": req.model_dump(), "client": current_client},
                )

                # Return streaming response
                return StreamingResponse(
                    service._generate_registered_stream(task_id),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                    },
                )
            else:
                # Synchronous mode - run in background task for cancellation support
                sync_task = asyncio.create_task(service._run_sync_workflow_task(req, current_client, task_id))
                await service.register_running_task(
                    task_id,
                    sync_task,
                    meta={"type": "workflow_sync", "request": req.model_dump(), "client": current_client},
                )

                # Wait for completion and return result
                try:
                    return await sync_task
                except asyncio.CancelledError:
                    if service.task_store:
                        service.task_store.update_task(task_id, status="cancelled")
                    raise

        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/workflows/chat_research", tags=["workflows"])
    async def run_chat_research(
        req: ChatResearchRequest, request: Request, current_client: str = _depends_get_current_client
    ):
        """Execute chat research workflow with DeepResearchEvent streaming."""
        try:
            logger.info(f"Chat research request from client: {current_client}")

            # Check if client accepts server-sent events
            accept_header = request.headers.get("accept", "")
            if "text/event-stream" not in accept_header:
                raise HTTPException(status_code=400, detail="Accept header must include 'text/event-stream'")

            # Generate or validate task_id (use frontend messageId if provided)
            if req.task_id:
                # Validate uniqueness for frontend-provided task_id
                await service._validate_task_id_uniqueness(req.task_id, current_client)
                task_id = req.task_id
                logger.info(f"Using frontend-provided task_id: {task_id}")
            else:
                # Generate new task_id if not provided
                task_id = service._generate_task_id(current_client, "research")
                logger.info(f"Generated new task_id: {task_id}")

            # Create and register the streaming task
            stream_task = asyncio.create_task(service._run_chat_research_task(req, current_client, task_id))
            await service.register_running_task(
                task_id,
                stream_task,
                meta={"type": "chat_research", "request": req.model_dump(), "client": current_client},
            )

            # Return streaming response
            return StreamingResponse(
                service._generate_chat_research_stream(task_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                },
            )

        except Exception as e:
            logger.error(f"Chat research execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/workflows/feedback", response_model=FeedbackResponse, tags=["workflows"])
    async def record_feedback(req: FeedbackRequest, current_client: str = _depends_get_current_client):
        """Record user feedback for a task."""
        try:
            logger.info(f"Feedback request from client: {current_client} for task: {req.task_id}")
            return await service.record_feedback(req)
        except Exception as e:
            logger.error(f"Feedback recording error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/workflows/tasks", tags=["workflows"])
    async def list_tasks(current_client: str = _depends_get_current_client):
        """List all running and recent tasks."""
        try:
            logger.info(f"Task list request from client: {current_client}")

            # Get running tasks
            running_tasks = await service.get_all_running_tasks()

            # Get recent tasks from database (last 24 hours)
            recent_tasks = []
            if service.task_store:
                # Get all tasks and filter recent ones
                all_tasks = service.task_store.get_all_feedback()
                cutoff_time = datetime.now().timestamp() - (24 * 60 * 60)  # 24 hours ago

                for task in all_tasks:
                    if task.get("created_at"):
                        # Convert string timestamp to float if needed
                        created_ts = task["created_at"]
                        if isinstance(created_ts, str):
                            try:
                                created_ts = datetime.fromisoformat(created_ts.replace("Z", "+00:00")).timestamp()
                            except (ValueError, TypeError):
                                continue

                        if created_ts > cutoff_time:
                            recent_tasks.append(task)

            return {
                "running_tasks": [
                    {
                        "task_id": rt.task_id,
                        "status": rt.status,
                        "created_at": rt.created_at.isoformat(),
                        "type": rt.meta.get("type", "unknown") if rt.meta else "unknown",
                        "client": rt.meta.get("client") if rt.meta else None,
                    }
                    for rt in running_tasks.values()
                ],
                "recent_tasks": recent_tasks,
            }
        except Exception as e:
            logger.error(f"Task list error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/workflows/tasks/{task_id}", tags=["workflows"])
    async def get_task(task_id: str, current_client: str = _depends_get_current_client):
        """Get details for a specific task."""
        try:
            logger.info(f"Task detail request from client: {current_client} for task: {task_id}")

            # Check running tasks first
            running_task = await service.get_running_task(task_id)
            if running_task:
                return {
                    "task_id": running_task.task_id,
                    "status": running_task.status,
                    "created_at": running_task.created_at.isoformat(),
                    "type": running_task.meta.get("type", "unknown") if running_task.meta else "unknown",
                    "client": running_task.meta.get("client") if running_task.meta else None,
                    "is_running": True,
                    "request": running_task.meta.get("request") if running_task.meta else None,
                }

            # Check database for completed tasks
            if service.task_store:
                task_data = service.task_store.get_task(task_id)
                if task_data:
                    return {
                        "task_id": task_data["task_id"],
                        "status": task_data["status"],
                        "created_at": task_data["created_at"],
                        "is_running": False,
                        "sql_query": task_data.get("sql_query"),
                        "sql_result": task_data.get("sql_result"),
                    }

            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Task detail error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/workflows/tasks/{task_id}", tags=["workflows"])
    async def cancel_task(task_id: str, current_client: str = _depends_get_current_client):
        """Cancel a running task."""
        try:
            logger.info(f"Task cancel request from client: {current_client} for task: {task_id}")

            # Check if task is running
            running_task = await service.get_running_task(task_id)
            if running_task:
                # Cancel the asyncio task
                running_task.task.cancel()
                running_task.status = "cancelled"

                # Update database
                if service.task_store:
                    service.task_store.update_task(task_id, status="cancelled")

                return {"task_id": task_id, "status": "cancelled", "message": "Task cancellation requested"}

            # Check if task exists in database
            if service.task_store:
                task_data = service.task_store.get_task(task_id)
                if task_data and task_data["status"] == "running":
                    # Task exists but not in memory (maybe different worker)
                    service.task_store.update_task(task_id, status="cancelled")
                    return {
                        "task_id": task_id,
                        "status": "cancelled",
                        "message": "Task cancellation requested (best-effort)",
                    }

            raise HTTPException(status_code=404, detail=f"Running task {task_id} not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Task cancel error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app
