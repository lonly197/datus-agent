# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, Optional, Type, get_type_hints

from pydantic import BaseModel, create_model

from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.schemas.compare_node_models import CompareInput
from datus.schemas.date_parser_node_models import DateParserInput
from datus.schemas.doc_search_node_models import DocSearchInput
from datus.schemas.fix_node_models import FixInput
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
from datus.schemas.node_models import BaseInput, ExecuteSQLInput, GenerateSQLInput, OutputInput, ReflectionInput
from datus.schemas.parallel_node_models import ParallelInput, SelectionInput
from datus.schemas.reason_sql_node_models import ReasoningInput
from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.schemas.search_metrics_node_models import SearchMetricsInput
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput
from datus.schemas.subworkflow_node_models import SubworkflowInput


class NodeType:
    # Registry for node type to input class mapping
    _input_class_registry: Dict[str, Type[BaseInput]] = {}

    # Workflow control node types
    TYPE_BEGIN = "start"
    # TYPE_EVALUATE = "evaluate"
    TYPE_HITL = "hitl"
    TYPE_REFLECT = "reflect"
    TYPE_PARALLEL = "parallel"
    TYPE_SELECTION = "selection"
    TYPE_SUBWORKFLOW = "subworkflow"

    # Control node types list
    CONTROL_TYPES = [TYPE_BEGIN, TYPE_HITL, TYPE_REFLECT, TYPE_PARALLEL, TYPE_SELECTION, TYPE_SUBWORKFLOW]

    # SQL workflow action types
    TYPE_SCHEMA_LINKING = "schema_linking"  # For database schema analysis
    TYPE_GENERATE_SQL = "generate_sql"  # For SQL query generation
    TYPE_EXECUTE_SQL = "execute_sql"  # For SQL query execution
    TYPE_OUTPUT = "output"  # For result presentation
    TYPE_REASONING = "reasoning"  # For result presentation
    TYPE_DOC_SEARCH = "doc_search"  # For document search
    TYPE_FIX = "fix"  # For fixing the SQL query
    TYPE_SEARCH_METRICS = "search_metrics"  # For search metrics
    TYPE_COMPARE = "compare"  # For comparing SQL with expectations
    TYPE_DATE_PARSER = "date_parser"  # For parsing temporal expressions
    TYPE_INTENT_ANALYSIS = "intent_analysis"  # For analyzing query intent
    TYPE_SCHEMA_DISCOVERY = "schema_discovery"  # For discovering relevant schema
    TYPE_SCHEMA_VALIDATION = "schema_validation"  # For validating schema sufficiency
    TYPE_RESULT_VALIDATION = "result_validation"  # For validating result quality

    # Agentic node types
    TYPE_CHAT = "chat"  # For conversational AI interactions
    TYPE_GENSQL = "gensql"  # For SQL generation with conversational AI
    TYPE_SEMANTIC = "semantic"  # For semantic model generation
    TYPE_SQL_SUMMARY = "sql_summary"  # For SQL summary generation

    ACTION_TYPES = [
        TYPE_SCHEMA_LINKING,
        TYPE_GENERATE_SQL,
        TYPE_EXECUTE_SQL,
        TYPE_OUTPUT,
        TYPE_REASONING,
        TYPE_DOC_SEARCH,
        TYPE_FIX,
        TYPE_SEARCH_METRICS,
        TYPE_COMPARE,
        TYPE_DATE_PARSER,
        TYPE_INTENT_ANALYSIS,
        TYPE_SCHEMA_DISCOVERY,
        TYPE_SCHEMA_VALIDATION,
        TYPE_RESULT_VALIDATION,
        TYPE_CHAT,
        TYPE_GENSQL,
        TYPE_SEMANTIC,
        TYPE_SQL_SUMMARY,
    ]

    NODE_TYPE_DESCRIPTIONS = {
        TYPE_BEGIN: "Beginning of the workflow",
        TYPE_SCHEMA_LINKING: "Understand the query and find related schemas",
        TYPE_GENERATE_SQL: "Generate SQL query",
        TYPE_EXECUTE_SQL: "Execute SQL query",
        TYPE_REFLECT: "evaluation and self-reflection",
        TYPE_OUTPUT: "Return the results to the user",
        TYPE_REASONING: "Reasoning analysis",
        TYPE_DOC_SEARCH: "Search related documents",
        TYPE_HITL: "Human in the loop",
        TYPE_FIX: "Fix the SQL query",
        TYPE_SEARCH_METRICS: "Search metrics",
        TYPE_PARALLEL: "Execute child nodes in parallel",
        TYPE_SELECTION: "Select best result from multiple candidates",
        TYPE_SUBWORKFLOW: "Execute a nested workflow",
        TYPE_COMPARE: "Compare SQL with expectations",
        TYPE_DATE_PARSER: "Parse temporal expressions in queries",
        TYPE_INTENT_ANALYSIS: "Analyze query intent using heuristics and optional LLM fallback",
        TYPE_SCHEMA_DISCOVERY: "Discover relevant schema and tables for the query",
        TYPE_SCHEMA_VALIDATION: "Validate schema sufficiency for SQL generation",
        TYPE_RESULT_VALIDATION: "Validate SQL execution result quality",
        TYPE_CHAT: "Conversational AI interactions with tool calling",
        TYPE_GENSQL: "SQL generation with conversational AI and tool calling",
        TYPE_SEMANTIC: "Semantic model generation with conversational AI",
        TYPE_SQL_SUMMARY: "SQL summary generation with conversational AI",
    }

    @classmethod
    def get_description(cls, node_type: str) -> str:
        return cls.NODE_TYPE_DESCRIPTIONS.get(node_type, f"Unknown node type: {node_type} for workflow")

    @classmethod
    def register_input_class(cls, node_type: str, input_class: Type[BaseInput]) -> None:
        """
        Register an input class for a node type.

        This allows extending node types without modifying the NodeType class,
        following the Open/Closed Principle.

        Args:
            node_type: The node type identifier
            input_class: The input class to register

        Example:
            NodeType.register_input_class("custom_node", CustomNodeInput)
        """
        cls._input_class_registry[node_type] = input_class

    @classmethod
    def _get_default_input_classes(cls) -> Dict[str, Type[BaseInput]]:
        """
        Get the default mapping of node types to input classes.

        This method is called once to initialize the registry with built-in
        node types. Custom node types can be added using register_input_class().

        Returns:
            Dictionary mapping node type strings to input classes
        """
        return {
            cls.TYPE_SCHEMA_LINKING: SchemaLinkingInput,
            cls.TYPE_GENERATE_SQL: GenerateSQLInput,
            cls.TYPE_EXECUTE_SQL: ExecuteSQLInput,
            cls.TYPE_REFLECT: ReflectionInput,
            cls.TYPE_REASONING: ReasoningInput,
            cls.TYPE_OUTPUT: OutputInput,
            cls.TYPE_FIX: FixInput,
            cls.TYPE_DOC_SEARCH: DocSearchInput,
            cls.TYPE_SEARCH_METRICS: SearchMetricsInput,
            cls.TYPE_PARALLEL: ParallelInput,
            cls.TYPE_SELECTION: SelectionInput,
            cls.TYPE_SUBWORKFLOW: SubworkflowInput,
            cls.TYPE_COMPARE: CompareInput,
            cls.TYPE_DATE_PARSER: DateParserInput,
            cls.TYPE_INTENT_ANALYSIS: BaseInput,
            cls.TYPE_SCHEMA_DISCOVERY: BaseInput,
            cls.TYPE_SCHEMA_VALIDATION: BaseInput,
            cls.TYPE_RESULT_VALIDATION: BaseInput,
            cls.TYPE_CHAT: ChatNodeInput,
            cls.TYPE_GENSQL: GenSQLNodeInput,
            cls.TYPE_SEMANTIC: SemanticNodeInput,
            cls.TYPE_SQL_SUMMARY: SqlSummaryNodeInput,
        }

    @classmethod
    def type_input(
        cls, node_type: str, input_data: dict, ignore_require_check: bool = False
    ) -> BaseInput:
        """
        Create an input instance for the given node type.

        This method uses a registry pattern to map node types to their input
        classes, making it extensible without modifying this method.

        Args:
            node_type: The type of node
            input_data: Dictionary of input data
            ignore_require_check: If True, make all fields optional

        Returns:
            An instance of the appropriate input class

        Raises:
            NotImplementedError: If node_type is not registered

        Example:
            input_obj = NodeType.type_input("generate_sql", {"task": "..."})
        """
        # Initialize registry on first call
        if not cls._input_class_registry:
            cls._input_class_registry = cls._get_default_input_classes()

        # Look up input class in registry
        input_data_cls = cls._input_class_registry.get(node_type)

        if input_data_cls is None:
            raise NotImplementedError(
                f"Node type '{node_type}' not implemented. "
                f"Available types: {list(cls._input_class_registry.keys())}"
            )

        if ignore_require_check:
            input_data_cls = cls.make_optional_model(input_data_cls)

        return input_data_cls(**input_data)

    # By default, Pydantic v2 validates required fields, but since we are using it as a config,
    # we don't need that strict validation. Therefore, we introduce this to relax the checks.
    @classmethod
    def make_optional_model(cls, base_model: type[BaseModel], name_suffix: str = "_Relaxed") -> type[BaseModel]:
        """
        Create a relaxed version of a Pydantic model with all fields optional.

        Args:
            base_model: The base model to relax
            name_suffix: Suffix to add to the generated model name

        Returns:
            A new model class with all fields optional
        """
        # Get field types from class annotations
        type_hints = get_type_hints(base_model)

        fields = {name: (Optional[typ], None) for name, typ in type_hints.items()}

        new_model = create_model(base_model.__name__ + name_suffix, __base__=base_model, **fields)
        return new_model
