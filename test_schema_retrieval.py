#!/usr/bin/env python3
"""
Schema Retrieval Test Script for Datus-Agent

This script tests whether the migration successfully imported DDL metadata
into the vector database and verifies RAG/Keyword retrieval functionality.

Test Scenario:
- Production StarRocks data warehouse
- DDL from: ods_ddl.sql, dws_ddl.sql, etc.
- Test query: Sales lead analysis (销售线索)
- Focus: Schema discovery stage only

Usage:
    python test_schema_retrieval.py --config=path/to/agent.yml --namespace=your_namespace
"""

import argparse
import json
import sys
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add project to path
sys.path.insert(0, '/Users/lonlyhuang/workspace/git/Datus-agent')

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich import print as rprint
except ImportError:
    # Fallback if rich not installed
    def print(*args, **kwargs):
        print(*args, **kwargs)
    Console = None
    Table = None
    Panel = None
    Syntax = None
    rprint = print


class SchemaRetrievalTester:
    """Test schema retrieval functionality after migration."""

    def __init__(self, config_path: str, namespace: str):
        self.config_path = config_path
        self.namespace = namespace
        self.console = Console() if Console else None

        # Initialize components
        try:
            from datus.configuration.agent_config_loader import load_agent_config
            from datus.storage.schema_metadata import SchemaWithValueRAG

            self.config = load_agent_config(config_path)
            self.config.current_namespace = namespace

            self.storage = SchemaWithValueRAG(self.config)
            self.schema_store = self.storage.schema_store

            self.console and self.console.print("[green]✅ Successfully initialized components[/green]")
        except Exception as e:
            self.console and self.console.print(f"[red]❌ Failed to initialize: {e}[/red]")
            raise

    def print_header(self, text: str):
        """Print a formatted header."""
        if self.console:
            self.console.print("\n" + "=" * 80)
            self.console.print(f"[bold cyan]{text}[/bold cyan]")
            self.console.print("=" * 80 + "\n")
        else:
            print("\n" + "=" * 80)
            print(text)
            print("=" * 80 + "\n")

    def print_section(self, text: str):
        """Print a section header."""
        if self.console:
            self.console.print(f"\n[bold yellow]{text}[/bold yellow]")
            self.console.print("-" * 80)
        else:
            print(f"\n{text}")
            print("-" * 80)

    def test_database_connection(self) -> bool:
        """Test if database is accessible."""
        self.print_section("Test 1: Database Connection")

        try:
            # Test vector database connection
            self.schema_store._ensure_table_ready()
            table_names = self.schema_store.db.table_names()

            if 'schema_metadata' in table_names:
                self.console and self.console.print("[green]✅ Vector database accessible[/green]")
                self.console and self.console.print(f"   Tables: {table_names}")
            else:
                self.console and self.console.print("[red]❌ schema_metadata table not found[/red]")
                return False

            return True
        except Exception as e:
            self.console and self.console.print(f"[red]❌ Database connection failed: {e}[/red]")
            return False

    def test_migration_success(self) -> Tuple[bool, int, int]:
        """Test if migration was successful."""
        self.print_section("Test 2: Migration Success")

        try:
            from collections import Counter

            # Get all records
            all_data = self.schema_store._search_all(
                where=None,
                select_fields=["metadata_version", "table_name", "table_comment"]
            )

            if not all_data or len(all_data) == 0:
                self.console and self.console.print("[red]❌ No data found in vector database[/red]")
                return False, 0, 0

            records = all_data.to_pylist()
            total_count = len(records)

            # Check version distribution
            version_counts = Counter(
                row.get("metadata_version", 0) for row in records
            )

            v0_count = version_counts.get(0, 0)
            v1_count = version_counts.get(1, 0)

            # Display results
            if self.console:
                table = Table(title="Migration Status")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Total Records", str(total_count))
                table.add_row("v0 (Legacy) Records", str(v0_count))
                table.add_row("v1 (Enhanced) Records", str(v1_count))

                migration_pct = (v1_count / total_count * 100) if total_count > 0 else 0
                table.add_row("Migration Progress", f"{migration_pct:.1f}%")

                self.console.print(table)
            else:
                print(f"Total Records: {total_count}")
                print(f"v0 (Legacy) Records: {v0_count}")
                print(f"v1 (Enhanced) Records: {v1_count}")
                print(f"Migration Progress: {(v1_count / total_count * 100):.1f}%")

            # Determine success
            if v1_count > 0:
                self.console and self.console.print("\n[green]✅ Migration successful - v1 records found[/green]")
                return True, total_count, v1_count
            else:
                self.console and self.console.print("\n[red]❌ Migration failed - no v1 records[/red]")
                return False, total_count, v0_count

        except Exception as e:
            self.console and self.console.print(f"[red]❌ Migration check failed: {e}[/red]")
            return False, 0, 0

    def test_comment_extraction(self, limit: int = 10) -> bool:
        """Test if table and column comments are extracted."""
        self.print_section("Test 3: Comment Information Extraction")

        try:
            # Get sample records
            all_data = self.schema_store._search_all(
                where=None,
                select_fields=[
                    "table_name",
                    "table_comment",
                    "column_comments",
                    "business_tags"
                ],
                limit=limit
            )

            if not all_data or len(all_data) == 0:
                self.console and self.console.print("[red]❌ No records to check[/red]")
                return False

            records = all_data.to_pylist()

            # Analyze comments
            tables_with_comment = 0
            tables_with_column_comments = 0
            tables_with_business_tags = 0

            if self.console:
                table = Table(title="Comment Extraction Results")
                table.add_column("Table Name", style="cyan")
                table.add_column("Table Comment", style="green")
                table.add_column("Has Column Comments", style="yellow")
                table.add_column("Business Tags", style="blue")

                for row in records:
                    table_name = row.get('table_name', 'N/A')
                    table_comment = row.get('table_comment', '')
                    column_comments = row.get('column_comments', '{}')
                    business_tags = row.get('business_tags', [])

                    # Check if comment exists (non-empty and not default)
                    has_table_comment = bool(table_comment and table_comment.strip())
                    has_column_comments = bool(column_comments and column_comments.strip() and column_comments != '{}')
                    has_business_tags = bool(business_tags)

                    if has_table_comment:
                        tables_with_comment += 1
                    if has_column_comments:
                        tables_with_column_comments += 1
                    if has_business_tags:
                        tables_with_business_tags += 1

                    # Truncate long comments for display
                    display_comment = (table_comment[:50] + "...") if len(table_comment) > 50 else table_comment
                    if not display_comment:
                        display_comment = "[dim]None[/dim]"

                    display_tags = ", ".join(business_tags[:3]) if business_tags else "[dim]None[/dim]"
                    if len(str(business_tags)) > 30:
                        display_tags = display_tags + "..."

                    has_col_comments = "✅" if has_column_comments else "❌"

                    table.add_row(
                        table_name,
                        display_comment,
                        has_col_comments,
                        display_tags
                    )

                self.console.print(table)
            else:
                for row in records:
                    table_name = row.get('table_name', 'N/A')
                    table_comment = row.get('table_comment', '')
                    column_comments = row.get('column_comments', '{}')
                    business_tags = row.get('business_tags', [])

                    has_table_comment = bool(table_comment and table_comment.strip())
                    has_column_comments = bool(column_comments and column_comments.strip() and column_comments != '{}')
                    has_business_tags = bool(business_tags)

                    if has_table_comment:
                        tables_with_comment += 1
                    if has_column_comments:
                        tables_with_column_comments += 1
                    if has_business_tags:
                        tables_with_business_tags += 1

                    print(f"\n{table_name}:")
                    print(f"  Table Comment: {table_comment[:100] if table_comment else 'None'}")
                    print(f"  Has Column Comments: {has_column_comments}")
                    print(f"  Business Tags: {business_tags[:5] if business_tags else 'None'}")

            # Calculate percentages
            total_tables = len(records)
            comment_pct = (tables_with_comment / total_tables * 100) if total_tables > 0 else 0
            column_comment_pct = (tables_with_column_comments / total_tables * 100) if total_tables > 0 else 0
            tags_pct = (tables_with_business_tags / total_tables * 100) if total_tables > 0 else 0

            # Summary
            if self.console:
                summary = Panel(
                    f"[bold]Tables with Table Comments:[/bold] {tables_with_comment}/{total_tables} ({comment_pct:.1f}%)\n"
                    f"[bold]Tables with Column Comments:[/bold] {tables_with_column_comments}/{total_tables} ({column_comment_pct:.1f}%)\n"
                    f"[bold]Tables with Business Tags:[/bold] {tables_with_business_tags}/{total_tables} ({tags_pct:.1f}%)",
                    title="Comment Extraction Summary",
                    border_style="green" if comment_pct > 50 else "yellow"
                )
                self.console.print(summary)
            else:
                print(f"\nSummary:")
                print(f"  Tables with Table Comments: {tables_with_comment}/{total_tables} ({comment_pct:.1f}%)")
                print(f"  Tables with Column Comments: {tables_with_column_comments}/{total_tables} ({column_comment_pct:.1f}%)")
                print(f"  Tables with Business Tags: {tables_with_business_tags}/{total_tables} ({tags_pct:.1f}%)")

            # Determine success (at least 50% of tables should have comments)
            success = comment_pct >= 50

            if success:
                self.console and self.console.print("\n[green]✅ Comment extraction successful[/green]")
            else:
                self.console and self.console.print("\n[yellow]⚠️ Low comment extraction rate[/yellow]")

            return success

        except Exception as e:
            self.console and self.console.print(f"[red]❌ Comment extraction test failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False

    def test_keyword_search(self, query: str, top_n: int = 20) -> bool:
        """Test keyword-based schema discovery."""
        self.print_section(f"Test 4: Keyword Search - '{query}'")

        try:
            # Simulate keyword matching
            # In production, this would use schema_discovery_node's keyword matching logic

            # For now, do a simple text search in table names and comments
            all_data = self.schema_store._search_all(
                where=None,
                select_fields=["table_name", "table_comment", "business_tags"]
            )

            if not all_data or len(all_data) == 0:
                self.console and self.console.print("[red]❌ No data for keyword search[/red]")
                return False

            records = all_data.to_pylist()

            # Keyword matching
            matched_tables = []
            query_lower = query.lower()

            for record in records:
                table_name = record.get('table_name', '').lower()
                table_comment = record.get('table_comment', '').lower()
                business_tags = ' '.join(record.get('business_tags', [])).lower()

                # Check if query keywords match
                if (query_lower in table_name or
                    query_lower in table_comment or
                    query_lower in business_tags):
                    matched_tables.append({
                        'table_name': record.get('table_name'),
                        'table_comment': record.get('table_comment', ''),
                        'business_tags': record.get('business_tags', []),
                        'relevance': self._calculate_relevance(query_lower, table_name, table_comment, business_tags)
                    })

            # Sort by relevance
            matched_tables.sort(key=lambda x: x['relevance'], reverse=True)

            # Take top N
            matched_tables = matched_tables[:top_n]

            if self.console:
                table = Table(title=f"Keyword Search Results ({len(matched_tables)} matches)")
                table.add_column("Table Name", style="cyan")
                table.add_column("Relevance", style="green", justify="right")
                table.add_column("Comment", style="yellow")
                table.add_column("Tags", style="blue")

                for table in matched_tables:
                    table_name = table['table_name']
                    relevance = f"{table['relevance']:.2f}"
                    comment = table['table_comment'][:60] if table['table_comment'] else "[dim]None[/dim]"
                    tags = ", ".join(table['business_tags'][:3]) if table['business_tags'] else "[dim]None[/dim]"

                    table.add_row(table_name, relevance, comment, tags)

                self.console.print(table)
            else:
                print(f"\nFound {len(matched_tables)} matching tables:")
                for table in matched_tables[:10]:
                    print(f"\n  {table['table_name']} (relevance: {table['relevance']:.2f})")
                    print(f"    Comment: {table['table_comment'][:100] if table['table_comment'] else 'None'}")
                    print(f"    Tags: {table['business_tags'][:5] if table['business_tags'] else 'None'}")

            if matched_tables:
                self.console and self.console.print(f"\n[green]✅ Keyword search found {len(matched_tables)} tables[/green]")
                return True
            else:
                self.console and self.console.print("\n[yellow]⚠️ No tables matched the keyword search[/yellow]")
                return False

        except Exception as e:
            self.console and self.console.print(f"[red]❌ Keyword search failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False

    def test_rag_search(self, query: str, top_n: int = 20) -> bool:
        """Test RAG-based schema discovery."""
        self.print_section(f"Test 5: RAG (Semantic) Search - '{query}'")

        try:
            # Use vector similarity search
            results = self.storage.search_similar(
                query_text=query,
                top_n=top_n
            )

            if not results:
                self.console and self.console.print("[yellow]⚠️ No results from RAG search[/yellow]")
                return False

            if self.console:
                table = Table(title=f"RAG Search Results ({len(results)} results)")
                table.add_column("Table Name", style="cyan")
                table.add_column("Similarity", style="green", justify="right")
                table.add_column("Table Comment", style="yellow")
                table.add_column("Business Tags", style="blue")

                for result in results:
                    # Handle both dict and object results
                    table_name = result.get('table_name', result.table_name if hasattr(result, 'table_name') else 'N/A')
                    similarity = result.get('similarity', getattr(result, 'similarity', 0))
                    table_comment = result.get('table_comment', getattr(result, 'table_comment', ''))
                    business_tags = result.get('business_tags', getattr(result, 'business_tags', []))

                    relevance_pct = (similarity * 100) if similarity else 0

                    comment = table_comment[:60] if table_comment else "[dim]None[/dim]"
                    tags = ", ".join(business_tags[:3]) if business_tags else "[dim]None[/dim]"

                    table.add_row(
                        table_name,
                        f"{relevance_pct:.1f}%",
                        comment,
                        tags
                    )

                self.console.print(table)
            else:
                print(f"\nFound {len(results)} relevant tables:")
                for result in results[:10]:
                    table_name = result.get('table_name', 'N/A')
                    similarity = result.get('similarity', 0)
                    relevance_pct = (similarity * 100) if similarity else 0
                    table_comment = result.get('table_comment', '')
                    business_tags = result.get('business_tags', [])

                    print(f"\n  {table_name} (similarity: {relevance_pct:.1f}%)")
                    print(f"    Comment: {table_comment[:100] if table_comment else 'None'}")
                    print(f"    Tags: {business_tags[:5] if business_tags else 'None'}")

            self.console and self.console.print(f"\n[green]✅ RAG search returned {len(results)} results[/green]")
            return True

        except Exception as e:
            self.console and self.console.print(f"[red]❌ RAG search failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False

    def _calculate_relevance(self, query: str, table_name: str, comment: str, tags: str) -> float:
        """Calculate simple relevance score for keyword matching."""
        score = 0.0

        # Exact match in table name gets highest score
        if query in table_name:
            score += 10.0

        # Match in comment
        if query in comment:
            score += 5.0

        # Match in tags
        if query in tags:
            score += 3.0

        # Partial matches
        for word in query.split():
            if word in table_name:
                score += 2.0
            if word in comment:
                score += 1.0
            if word in tags:
                score += 0.5

        return score

    def test_sales_lead_scenario(self) -> bool:
        """Test the specific sales lead analysis scenario."""
        self.print_section("Test 6: Sales Lead Analysis Scenario")

        # Test queries related to sales leads
        test_queries = [
            "销售线索",
            "线索",
            "销售",
            "客户",
            "潜在客户",
            "商机"
        ]

        all_passed = True

        for query in test_queries:
            self.console and self.console.print(f"\n[bold]Testing query: {query}[/bold]")

            # Test keyword search
            keyword_results = self._test_keyword_for_query(query)

            # Test RAG search
            rag_results = self._test_rag_for_query(query)

            if keyword_results or rag_results:
                self.console and self.console.print(f"[green]  ✅ Found results for '{query}'[/green]")
            else:
                self.console and self.console.print(f"[yellow]  ⚠️ No results for '{query}'[/yellow]")
                all_passed = False

        return all_passed

    def _test_keyword_for_query(self, query: str) -> bool:
        """Test keyword search for a specific query."""
        try:
            all_data = self.schema_store._search_all(
                where=None,
                select_fields=["table_name", "table_comment", "business_tags"]
            )

            if not all_data:
                return False

            records = all_data.to_pylist()
            query_lower = query.lower()

            for record in records:
                table_name = record.get('table_name', '').lower()
                table_comment = record.get('table_comment', '').lower()
                business_tags = ' '.join(record.get('business_tags', [])).lower()

                if (query_lower in table_name or
                    query_lower in table_comment or
                    query_lower in business_tags):
                    return True

            return False
        except Exception:
            return False

    def _test_rag_for_query(self, query: str) -> bool:
        """Test RAG search for a specific query."""
        try:
            results = self.storage.search_similar(
                query_text=query,
                top_n=5
            )
            return len(results) > 0
        except Exception:
            return False

    def run_all_tests(self):
        """Run all schema retrieval tests."""
        self.print_header("Schema Retrieval Test Suite")

        results = {}

        # Test 1: Database Connection
        results['database'] = self.test_database_connection()

        if not results['database']:
            self.console and self.console.print("\n[red]❌ Cannot proceed without database connection[/red]")
            return results

        # Test 2: Migration Success
        migration_success, total_count, v1_count = self.test_migration_success()
        results['migration'] = migration_success

        if not migration_success:
            self.console and self.console.print("\n[red]❌ Migration not successful, skipping remaining tests[/red]")
            return results

        # Test 3: Comment Extraction
        results['comments'] = self.test_comment_extraction()

        # Test 4: Keyword Search
        results['keyword'] = self.test_keyword_search("销售线索")

        # Test 5: RAG Search
        results['rag'] = self.test_rag_search("销售线索")

        # Test 6: Sales Lead Scenario
        results['scenario'] = self.test_sales_lead_scenario()

        # Final Summary
        self.print_header("Test Summary")

        if self.console:
            table = Table(title="Overall Results")
            table.add_column("Test", style="cyan")
            table.add_column("Status", style="green")

            for test_name, passed in results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                test_display = test_name.replace('_', ' ').title()
                table.add_row(test_display, status)

            self.console.print(table)

            # Overall assessment
            passed_count = sum(1 for v in results.values() if v)
            total_tests = len(results)

            if passed_count == total_tests:
                self.console.print("\n[bold green]✅ ALL TESTS PASSED[/bold green]")
                self.console.print("[green]Schema retrieval is working correctly![/green]")
            elif passed_count >= total_tests * 0.7:
                self.console.print(f"\n[bold yellow]⚠️ {passed_count}/{total_tests} TESTS PASSED[/bold yellow]")
                self.console.print("[yellow]Some issues detected, but core functionality works[/yellow]")
            else:
                self.console.print(f"\n[bold red]❌ {passed_count}/{total_tests} TESTS PASSED[/bold red]")
                self.console.print("[red]Significant issues detected[/red]")
        else:
            print("\nTest Results:")
            for test_name, passed in results.items():
                status = "PASS" if passed else "FAIL"
                test_display = test_name.replace('_', ' ').title()
                print(f"  {test_display}: {status}")

            passed_count = sum(1 for v in results.values() if v)
            total_tests = len(results)
            print(f"\nPassed: {passed_count}/{total_tests}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Test schema retrieval after migration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with namespace
  python test_schema_retrieval.py --config=conf/agent.yml --namespace=my_namespace

  # Test without namespace
  python test_schema_retrieval.py --config=conf/agent.yml
        """
    )

    parser.add_argument(
        '--config',
        required=True,
        help='Path to agent configuration file (agent.yml)'
    )

    parser.add_argument(
        '--namespace',
        help='Namespace for the database (optional)'
    )

    parser.add_argument(
        '--query',
        default='销售线索',
        help='Query to test (default: "销售线索")'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of results to return (default: 20)'
    )

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        tester = SchemaRetrievalTester(args.config, args.namespace)
        results = tester.run_all_tests()

        # Exit with error code if any test failed
        if not all(results.values()):
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
