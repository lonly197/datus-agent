#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Test cases for schema DDL optimization fix.

Tests the new functionality that reduces token consumption by only sending
table names and comments to LLM in SQL generation phase, instead of full DDL.
"""

import os
import unittest
from unittest.mock import Mock, patch
from typing import List

from datus.schemas.node_models import TableSchema, GenerateSQLInput, SqlTask
from datus.prompts.gen_sql import get_sql_prompt
from datus.agent.node.generate_sql_node import GenerateSQLNode
from datus.agent.workflow import Workflow, WorkflowContext
from datus.schemas.node_models import TableValue


class TestSchemaDDLOptimization(unittest.TestCase):
    """Test cases for schema DDL optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_table_schemas = [
            TableSchema(
                identifier="test_table_1",
                catalog_name="test_catalog",
                table_name="users",
                database_name="test_db",
                schema_name="public",
                definition="CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255) NOT NULL, email VARCHAR(255) UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);",
                table_type="table"
            ),
            TableSchema(
                identifier="test_table_2",
                catalog_name="test_catalog",
                table_name="orders",
                database_name="test_db",
                schema_name="public",
                definition="CREATE TABLE orders (id INT PRIMARY KEY, user_id INT NOT NULL, order_date DATE NOT NULL, total_amount DECIMAL(10,2) NOT NULL, status VARCHAR(50) NOT NULL, FOREIGN KEY (user_id) REFERENCES users(id));",
                table_type="table"
            )
        ]

        self.sample_data_details = [
            TableValue(
                table_name="users",
                columns=["id", "name", "email"],
                values=[(1, "Alice", "alice@example.com"), (2, "Bob", "bob@example.com")]
            )
        ]

        self.sql_task = SqlTask(
            id="test_task",
            database_type="sqlite",
            task="显示用户表的前10行",
            catalog_name="test_catalog",
            database_name="test_db",
            schema_name="public"
        )

    def test_table_schema_to_prompt_include_ddl_true(self):
        """Test TableSchema.to_prompt() with include_ddl=True returns full DDL."""
        schema = self.sample_table_schemas[0]

        result = schema.to_prompt("sqlite", include_ddl=True)

        # Should return full DDL
        self.assertIn("CREATE TABLE", result)
        self.assertIn("users", result)
        self.assertIn("INT PRIMARY KEY", result)
        self.assertIn("VARCHAR(255)", result)

    def test_table_schema_to_prompt_include_ddl_false(self):
        """Test TableSchema.to_prompt() with include_ddl=False returns only table info."""
        schema = self.sample_table_schemas[0]

        result = schema.to_prompt("sqlite", include_ddl=False)

        # Should return only table name
        self.assertIn("Table: users", result)
        # Should NOT include DDL details
        self.assertNotIn("CREATE TABLE", result)
        self.assertNotIn("PRIMARY KEY", result)

    def test_get_sql_prompt_include_schema_ddl_false(self):
        """Test get_sql_prompt() with include_schema_ddl=False reduces token usage."""
        prompt = get_sql_prompt(
            database_type="sqlite",
            table_schemas=self.sample_table_schemas,
            data_details=[],
            metrics=[],
            question="显示用户表的前10行",
            include_schema_ddl=False
        )

        # Should have system and user messages
        self.assertEqual(len(prompt), 2)
        user_content = prompt[1]["content"]

        # Should only contain table names, not full DDL
        self.assertIn("Table: users", user_content)
        self.assertIn("Table: orders", user_content)
        # Should NOT contain DDL details
        self.assertNotIn("CREATE TABLE", user_content)
        self.assertNotIn("PRIMARY KEY", user_content)
        self.assertNotIn("FOREIGN KEY", user_content)

    def test_get_sql_prompt_include_schema_ddl_true(self):
        """Test get_sql_prompt() with include_schema_ddl=True includes full DDL."""
        prompt = get_sql_prompt(
            database_type="sqlite",
            table_schemas=self.sample_table_schemas,
            data_details=[],
            metrics=[],
            question="统计每个用户的订单数量和总金额",
            include_schema_ddl=True
        )

        # Should have system and user messages
        self.assertEqual(len(prompt), 2)
        user_content = prompt[1]["content"]

        # Should contain full DDL
        self.assertIn("CREATE TABLE", user_content)
        self.assertIn("users", user_content)
        self.assertIn("orders", user_content)
        self.assertIn("PRIMARY KEY", user_content)
        self.assertIn("FOREIGN KEY", user_content)

    def test_query_complexity_analysis_simple_query(self):
        """Test query complexity analysis for simple queries."""
        node = GenerateSQLNode(
            node_id="test_node",
            description="Test GenerateSQLNode",
            node_type="generate_sql"
        )

        # Simple queries should NOT include DDL
        simple_queries = [
            "显示用户表的前10行",
            "show users",
            "select * from users",
            "查询所有订单",
            "find orders where id = 1",
        ]

        for query in simple_queries:
            with self.subTest(query=query):
                result = node._should_include_ddl(query, 1)
                self.assertFalse(result, f"Simple query '{query}' should not include DDL")

    def test_query_complexity_analysis_complex_query(self):
        """Test query complexity analysis for complex queries."""
        node = GenerateSQLNode(
            node_id="test_node",
            description="Test GenerateSQLNode",
            node_type="generate_sql"
        )

        # Complex queries should include DDL
        complex_queries = [
            "统计每个用户的订单数量和总金额",
            "join users and orders",
            "select count(*) from orders group by user_id",
            "select u.name, sum(o.total_amount) from users u join orders o on u.id = o.user_id",
            "with cte as (select * from users) select * from cte",
        ]

        for query in complex_queries:
            with self.subTest(query=query):
                result = node._should_include_ddl(query, 2)
                self.assertTrue(result, f"Complex query '{query}' should include DDL")

    def test_query_complexity_analysis_multi_table(self):
        """Test query complexity analysis for multi-table queries."""
        node = GenerateSQLNode(
            node_id="test_node",
            description="Test GenerateSQLNode",
            node_type="generate_sql"
        )

        # Multi-table queries should include DDL
        result = node._should_include_ddl("显示用户和订单信息", 2)
        self.assertTrue(result, "Multi-table query should include DDL")

        result = node._should_include_ddl("compare data", 3)
        self.assertTrue(result, "Query with 3 tables should include DDL")

    def test_generate_sql_input_include_schema_ddl_parameter(self):
        """Test GenerateSQLInput accepts include_schema_ddl parameter."""
        input_data = GenerateSQLInput(
            database_type="sqlite",
            sql_task=self.sql_task,
            table_schemas=self.sample_table_schemas,
            include_schema_ddl=False
        )

        self.assertFalse(input_data.include_schema_ddl)

        input_data_with_ddl = GenerateSQLInput(
            database_type="sqlite",
            sql_task=self.sql_task,
            table_schemas=self.sample_table_schemas,
            include_schema_ddl=True
        )

        self.assertTrue(input_data_with_ddl.include_schema_ddl)

    @patch.dict(os.environ, {'ENABLE_SMART_DDL_SELECTION': 'false', 'DEFAULT_INCLUDE_SCHEMA_DDL': 'true'})
    def test_configuration_override(self):
        """Test that configuration settings override smart selection."""
        node = GenerateSQLNode(
            node_id="test_node",
            description="Test GenerateSQLNode",
            node_type="generate_sql"
        )

        # When smart selection is disabled and default is True
        result = node._should_include_ddl("显示用户表的前10行", 1)
        self.assertTrue(result, "Should use configuration default when smart selection is disabled")

    @patch.dict(os.environ, {'ENABLE_SMART_DDL_SELECTION': 'false', 'DEFAULT_INCLUDE_SCHEMA_DDL': 'false'})
    def test_configuration_override_false(self):
        """Test that configuration settings override smart selection (False case)."""
        node = GenerateSQLNode(
            node_id="test_node",
            description="Test GenerateSQLNode",
            node_type="generate_sql"
        )

        # When smart selection is disabled and default is False
        result = node._should_include_ddl("统计每个用户的订单数量", 2)
        self.assertFalse(result, "Should use configuration default when smart selection is disabled")

    def test_token_usage_comparison(self):
        """Test that token usage is significantly reduced with DDL optimization."""
        # Test without DDL optimization (old behavior)
        prompt_without_ddl = get_sql_prompt(
            database_type="sqlite",
            table_schemas=self.sample_table_schemas,
            data_details=[],
            metrics=[],
            question="显示用户表的前10行",
            include_schema_ddl=False
        )

        # Test with DDL optimization (new behavior)
        prompt_with_ddl = get_sql_prompt(
            database_type="sqlite",
            table_schemas=self.sample_table_schemas,
            data_details=[],
            metrics=[],
            question="统计每个用户的订单数量和总金额",
            include_schema_ddl=True
        )

        # Both should be valid prompts
        self.assertEqual(len(prompt_without_ddl), 2)
        self.assertEqual(len(prompt_with_ddl), 2)

        # Simple query should have shorter content without DDL
        simple_content = prompt_without_ddl[1]["content"]
        complex_content = prompt_with_ddl[1]["content"]

        # Complex query with DDL should be longer than simple query without DDL
        self.assertGreater(len(complex_content), len(simple_content))

        # Simple query without DDL should be concise
        self.assertLess(len(simple_content), 500, "Simple query without DDL should be very concise")

    def test_table_comment_handling(self):
        """Test that table comments are included when available."""
        schema_with_comment = TableSchema(
            identifier="test_table_3",
            catalog_name="test_catalog",
            table_name="products",
            database_name="test_db",
            schema_name="public",
            definition="CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(255) NOT NULL);",
            table_type="table"
        )

        # Add table_comment attribute
        schema_with_comment.table_comment = "Product information table"

        result_without_ddl = schema_with_comment.to_prompt("sqlite", include_ddl=False)

        # Should include table comment
        self.assertIn("Table: products", result_without_ddl)
        self.assertIn("Product information table", result_without_ddl)
        # Should NOT include DDL
        self.assertNotIn("CREATE TABLE", result_without_ddl)

    def test_empty_query_handling(self):
        """Test that empty queries are handled gracefully."""
        node = GenerateSQLNode(
            node_id="test_node",
            description="Test GenerateSQLNode",
            node_type="generate_sql"
        )

        # Empty query should use default behavior
        result = node._should_include_ddl("", 1)
        self.assertFalse(result, "Empty query should use default (no DDL)")

        result = node._should_include_ddl(None, 1)
        self.assertFalse(result, "None query should use default (no DDL)")


if __name__ == "__main__":
    unittest.main()