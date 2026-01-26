# Subject Tree Taxonomy

## Overview

The subject tree is a hierarchical taxonomy that organizes Metrics and Reference SQL. It is stored in
`subject_tree.db` alongside LanceDB data and is used for filtering and aggregation.

## Why It Matters

- Provides consistent categorization across metrics and SQL references
- Enables subject-scoped search and retrieval
- Supports incremental growth without reclassifying older entries

## Structure

A subject path is a list of strings that represents the hierarchy:

```
["Sales", "Revenue", "Daily"]
```

Paths are persisted in `subject_tree.db` and referenced by `subject_node_id` in LanceDB.

## Where It Is Used

- Metrics: `MetricStorage` resolves `subject_path` into the subject tree
- Reference SQL: `ReferenceSqlStorage` resolves `subject_path` into the subject tree

## Providing a Predefined Tree

You can lock the taxonomy during bootstrap using `--subject_tree`:

```bash
datus-agent bootstrap-kb \
    --namespace your_namespace \
    --components metrics reference_sql \
    --subject_tree "Sales/Revenue/Daily,Sales/Analytics/Trends"
```

If `--subject_tree` is not provided, the system reuses existing subject paths in storage or creates new ones as needed.
