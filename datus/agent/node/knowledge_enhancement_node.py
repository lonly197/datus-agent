# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
KnowledgeEnhancementNode implementation for unified knowledge processing.

This node unifies handling of:
1. User-provided knowledge (ext_knowledge string or ext_knowledges array)
2. Automatically retrieved knowledge from vector stores (ext_knowledge_store, reference_sql, metrics)
3. Intelligent filtering when knowledge is too long
4. Knowledge merging and formatting for downstream nodes
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput, BaseResult
from datus.storage.cache import get_storage_cache_instance
from datus.utils.error_handler import LLMMixin
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class KnowledgeEnhancementNode(Node, LLMMixin):
    """
    Node for knowledge enhancement: unified processing of user and retrieved knowledge.

    This node:
    1. Normalizes knowledge input (ext_knowledge string or ext_knowledges array)
    2. Filters relevant knowledge using LLM when knowledge is too long
    3. Automatically retrieves vector knowledge (ext_knowledge_store, reference_sql, metrics)
    4. Merges and formats knowledge for downstream nodes
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[BaseInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        max_knowledge_length: int = 5000,
        vector_search_top_n: int = 5,
    ):
        Node.__init__(
            self,
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
        )
        LLMMixin.__init__(self)
        self.max_knowledge_length = max_knowledge_length
        self.vector_search_top_n = vector_search_top_n

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """Setup knowledge enhancement input from workflow context."""
        if not self.input:
            self.input = BaseInput()
        return {"success": True, "message": "Knowledge enhancement input setup complete"}

    async def run(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Run knowledge enhancement to unify and enrich knowledge.

        Yields:
            ActionHistory: Progress and result actions
        """
        try:
            if not self.workflow or not self.workflow.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "No workflow or task available for knowledge enhancement",
                    "knowledge_enhancement",
                    {"workflow_id": getattr(self.workflow, "id", "unknown") if self.workflow else "unknown"},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Knowledge enhancement failed: {error_result.error}",
                    action_type="knowledge_enhancement",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error, "error_code": error_result.error_code},
                )
                return

            task = self.workflow.task
            context = self.workflow.context

            # Step 1: Normalize knowledge input (handle array and separator)
            normalized_knowledge = self._normalize_knowledge(task, context)

            # Step 2: Filter if too long using LLM
            if len(normalized_knowledge) > self.max_knowledge_length:
                logger.info(
                    f"Knowledge length ({len(normalized_knowledge)}) exceeds threshold ({self.max_knowledge_length}), filtering..."
                )
                normalized_knowledge = await self._filter_relevant_knowledge(
                    task_text=task.task,
                    schemas=context.table_schemas if context else None,
                    knowledge=normalized_knowledge,
                )

            # Step 3: Retrieve vector knowledge automatically
            retrieved_knowledge = await self._retrieve_vector_knowledge(
                task_text=task.task,
                schemas=context.table_schemas if context else None,
                context=context,
            )

            # Step 4: Merge knowledge
            enhanced_knowledge = self._merge_knowledge(
                user_knowledge=normalized_knowledge,
                retrieved_knowledge=retrieved_knowledge,
            )

            # Step 5: Update task and context
            if hasattr(task, "external_knowledge"):
                task.external_knowledge = enhanced_knowledge
            if context and hasattr(context, "external_knowledge"):
                context.external_knowledge = enhanced_knowledge

            # Step 6: Log statistics
            self._log_knowledge_sources(
                user_knowledge=normalized_knowledge,
                retrieved_knowledge=retrieved_knowledge,
                enhanced_knowledge=enhanced_knowledge,
            )

            # Emit success action
            yield ActionHistory(
                action_id=f"{self.id}_enhancement",
                role=ActionRole.TOOL,
                messages=f"Knowledge enhancement completed: {len(enhanced_knowledge)} characters",
                action_type="knowledge_enhancement",
                input={
                    "task": task.task[:50] if task else "",
                    "user_knowledge_length": len(normalized_knowledge),
                },
                status=ActionStatus.SUCCESS,
                output={
                    "enhanced_knowledge_length": len(enhanced_knowledge),
                    "retrieved_ext_knowledge_count": len(retrieved_knowledge.get("ext_knowledge", [])),
                    "retrieved_reference_sql_count": len(retrieved_knowledge.get("reference_sql", [])),
                    "retrieved_metrics_count": len(retrieved_knowledge.get("metrics", [])),
                },
            )

            self.result = BaseResult(
                success=True,
                data={
                    "enhanced_knowledge": enhanced_knowledge,
                    "statistics": {
                        "user_knowledge_length": len(normalized_knowledge),
                        "retrieved_ext_knowledge": len(retrieved_knowledge.get("ext_knowledge", [])),
                        "retrieved_reference_sql": len(retrieved_knowledge.get("reference_sql", [])),
                        "retrieved_metrics": len(retrieved_knowledge.get("metrics", [])),
                        "enhanced_knowledge_length": len(enhanced_knowledge),
                    },
                },
            )

            logger.info(
                f"Knowledge enhancement completed: user={len(normalized_knowledge)} chars, "
                f"retrieved={sum(len(v) for v in retrieved_knowledge.values())} items, "
                f"enhanced={len(enhanced_knowledge)} chars"
            )

        except Exception as e:
            logger.error(f"Knowledge enhancement failed: {e}")
            error_result = self.create_error_result(
                ErrorCode.NODE_EXECUTION_FAILED,
                f"Knowledge enhancement execution failed: {str(e)}",
                "knowledge_enhancement",
                {
                    "task_id": (
                        getattr(self.workflow.task, "id", "unknown")
                        if self.workflow and self.workflow.task
                        else "unknown"
                    )
                },
            )
            yield ActionHistory(
                action_id=f"{self.id}_error",
                role=ActionRole.TOOL,
                messages=f"Knowledge enhancement failed: {error_result.error}",
                action_type="knowledge_enhancement",
                input={},
                status=ActionStatus.FAILED,
                output={"error": error_result.error, "error_code": error_result.error_code},
            )
            self.result = BaseResult(success=False, error=str(e))

    def _normalize_knowledge(
        self, task, context: Optional["Context"]
    ) -> str:
        """
        Normalize knowledge input: handle ext_knowledges array and | separator.

        Args:
            task: SqlTask instance
            context: Workflow context

        Returns:
            Normalized knowledge string
        """
        # 1. Check ext_knowledges array parameter
        if context and hasattr(context, "workflow_metadata") and context.workflow_metadata:
            ext_knowledges = context.workflow_metadata.get("ext_knowledges")
            if ext_knowledges and isinstance(ext_knowledges, list):
                return "\n".join([f"- {item}" for item in ext_knowledges])

        # 2. Handle ext_knowledge string with | separator
        ext_knowledge = getattr(task, "external_knowledge", None) or ""
        if "|" in ext_knowledge:
            items = [item.strip() for item in ext_knowledge.split("|")]
            return "\n".join([f"- {item}" for item in items])

        return ext_knowledge

    async def _filter_relevant_knowledge(
        self, task_text: str, schemas: Optional[List], knowledge: str
    ) -> str:
        """
        Use LLM to filter relevant knowledge when knowledge is too long.

        Args:
            task_text: User query text
            schemas: List of table schemas
            knowledge: Knowledge text to filter

        Returns:
            Filtered knowledge text
        """
        table_names = []
        if schemas:
            table_names = [s.table_name for s in schemas]

        prompt = f"""你是一个专业的数据分析助手。请从以下业务知识中筛选出与用户查询直接相关的内容。

用户查询：{task_text}

可用表：{", ".join(table_names) if table_names else "无"}

业务知识（过多）：
{knowledge}

请仅输出与用户查询相关的业务知识，每行一条，格式：
- 相关知识1
- 相关知识2
...

重要：仅输出相关的知识，不要包含其他解释。"""

        try:
            cache_key = f"knowledge_filter:{hash(task_text + knowledge)}"
            response = await self.llm_call_with_retry(
                prompt=prompt,
                operation_name="knowledge_filter",
                cache_key=cache_key,
                max_retries=2,
            )

            filtered_knowledge = response.get("text", knowledge).strip()
            logger.info(f"LLM filtered knowledge: {len(knowledge)} → {len(filtered_knowledge)} chars")
            return filtered_knowledge

        except Exception as e:
            logger.warning(f"LLM knowledge filtering failed: {e}. Using original knowledge.")
            return knowledge

    async def _retrieve_vector_knowledge(
        self, task_text: str, schemas: Optional[List], context: Optional["Context"]
    ) -> Dict[str, List[Dict]]:
        """
        Automatically retrieve vector knowledge from multiple sources.

        Args:
            task_text: User query text
            schemas: List of table schemas
            context: Workflow context

        Returns:
            Dictionary with retrieved knowledge from different sources
        """
        retrieved = {
            "ext_knowledge": [],
            "reference_sql": [],
            "metrics": [],
        }

        if not self.agent_config:
            return retrieved

        try:
            storage_cache = get_storage_cache_instance(self.agent_config)
        except Exception as e:
            logger.warning(f"Failed to get storage cache: {e}")
            return retrieved

        # 1. Retrieve external knowledge
        try:
            ext_knowledge_store = storage_cache.ext_knowledge_storage()
            results = ext_knowledge_store.search_knowledge(
                query_text=task_text,
                top_n=self.vector_search_top_n,
            )
            retrieved["ext_knowledge"] = results
            logger.info(
                f"Retrieved {len(results)} external knowledge: {[r.get('terminology', 'N/A') for r in results]}"
            )
        except Exception as e:
            logger.warning(f"External knowledge retrieval failed: {e}")

        # 2. Retrieve reference SQL
        try:
            from datus.tools.context_search_tools import ContextSearchTools

            search_tools = ContextSearchTools(self.agent_config)
            results = search_tools.search_reference_sql(
                query_text=task_text,
                top_n=self.vector_search_top_n,
            )
            if hasattr(results, "data"):
                retrieved["reference_sql"] = results.data
                logger.info(f"Retrieved {len(results.data)} reference SQL")
        except Exception as e:
            logger.warning(f"Reference SQL retrieval failed: {e}")

        # 3. Retrieve metrics (based on table names)
        if schemas:
            try:
                from datus.tools.metric_rag_tools import SemanticMetricsRAG

                metric_rag = SemanticMetricsRAG(self.agent_config)
                for schema in schemas[:3]:  # Limit to 3 tables
                    results = metric_rag.search_metrics(
                        query_text=task_text,
                        table_name=schema.table_name,
                        top_n=3,
                    )
                    if results:
                        retrieved["metrics"].extend(results)
                logger.info(f"Retrieved {len(retrieved['metrics'])} metrics")
            except Exception as e:
                logger.warning(f"Metrics retrieval failed: {e}")

        return retrieved

    def _merge_knowledge(
        self, user_knowledge: str, retrieved_knowledge: Dict[str, List[Dict]]
    ) -> str:
        """
        Merge user knowledge and retrieved knowledge into formatted output.

        Args:
            user_knowledge: User-provided knowledge
            retrieved_knowledge: Retrieved knowledge from vector stores

        Returns:
            Merged and formatted knowledge text
        """
        parts = []

        # 1. User-provided business knowledge
        if user_knowledge and user_knowledge.strip():
            parts.append(f"## 用户提供的业务知识\n{user_knowledge}")

        # 2. Retrieved external knowledge
        if retrieved_knowledge.get("ext_knowledge"):
            parts.append("\n## 相关业务术语")
            for item in retrieved_knowledge["ext_knowledge"]:
                terminology = item.get("terminology", "")
                explanation = item.get("explanation", "")
                parts.append(f"- **{terminology}**: {explanation}")

        # 3. Retrieved reference SQL
        if retrieved_knowledge.get("reference_sql"):
            parts.append("\n## 相关SQL案例")
            for item in retrieved_knowledge["reference_sql"][:3]:  # Limit to 3 examples
                name = item.get("name", "")
                sql = item.get("sql", "")
                summary = item.get("summary", "")
                parts.append(f"### {name}\n{summary}\n```sql\n{sql}\n```")

        # 4. Retrieved metrics
        if retrieved_knowledge.get("metrics"):
            parts.append("\n## 相关指标")
            seen_metrics = set()
            for item in retrieved_knowledge["metrics"]:
                metric_name = item.get("metric_name", "")
                if metric_name and metric_name not in seen_metrics:
                    definition = item.get("definition", "")
                    parts.append(f"- **{metric_name}**: {definition}")
                    seen_metrics.add(metric_name)

        return "\n".join(parts)

    def _log_knowledge_sources(
        self,
        user_knowledge: str,
        retrieved_knowledge: Dict,
        enhanced_knowledge: str,
    ):
        """Log knowledge sources and statistics."""
        logger.info(
            f"""
知识增强统计：
- 用户知识长度: {len(user_knowledge)} 字符
- 检索外部知识: {len(retrieved_knowledge.get('ext_knowledge', []))} 条
- 检索参考SQL: {len(retrieved_knowledge.get('reference_sql', []))} 条
- 检索指标: {len(retrieved_knowledge.get('metrics', []))} 条
- 增强后总长度: {len(enhanced_knowledge)} 字符
"""
        )

    def execute(self) -> BaseResult:
        """Execute knowledge enhancement synchronously."""
        return execute_with_async_stream(self)

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute knowledge enhancement with streaming support."""
        if action_history_manager:
            self.action_history_manager = action_history_manager
        async for action in self.run():
            yield action

    def update_context(self, workflow: "Workflow") -> Dict[str, Any]:
        """Update workflow context with enhancement results."""
        try:
            if not self.result or not self.result.success:
                return {"success": False, "message": "Knowledge enhancement failed, cannot update context"}

            # Store enhancement result in workflow metadata
            if not hasattr(workflow, "metadata") or workflow.metadata is None:
                workflow.metadata = {}
            workflow.metadata["knowledge_enhancement"] = self.result.data

            return {
                "success": True,
                "message": f"Knowledge enhancement context updated: {len(self.result.data.get('enhanced_knowledge', ''))} chars",
            }

        except Exception as e:
            logger.error(f"Failed to update knowledge enhancement context: {str(e)}")
            return {"success": False, "message": f"Knowledge enhancement context update failed: {str(e)}"}
