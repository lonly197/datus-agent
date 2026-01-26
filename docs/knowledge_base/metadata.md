# Schema Metadata Intelligence

## Introduction

The metadata module is primarily used to enable LLMs to quickly match possible related table definition information and sample data based on user questions.

When you use the `bootstrap-kb` command, we initialize table/view (and optional materialized view) DDL plus sample rows from
the target data source into LanceDB.

This module contains two types of information: **table definition** (`schema_metadata.lance`) and **sample data**
(`schema_value.lance`).

## Data Structure of Table Definition

| Field Name | Explanation |
|------------|-------------|
| `identifier` | Unique key composed from catalog/database/schema/table/type |
| `catalog_name` | Top-level container (only populated for dialects that support catalogs) |
| `database_name` | Database name (populated when the dialect supports databases) |
| `schema_name` | Schema name (populated when the dialect supports schemas) |
| `table_type` | `table`, `view`, or `mv` (materialized view) |
| `table_name` | Name of the table/view/materialized view |
| `definition` | DDL statement for the table/view/materialized view; may be augmented with parsed comments |
| `table_comment` | Table-level comment extracted from DDL (when available) |
| `column_comments` | JSON string map of column comments |
| `column_enums` | JSON string map of enumerated values parsed from comments |
| `business_tags` | List of inferred business tags |
| `row_count` | Row count if available from the connector/metadata extractor |
| `sample_statistics` | JSON string of basic column statistics when extracted |
| `relationship_metadata` | JSON string of inferred relationships (e.g., foreign keys, join paths) |
| `metadata_version` | Metadata schema version (0 legacy, 1 enhanced) |
| `last_updated` | Unix timestamp of last update |

> `catalog_name`, `database_name`, and `schema_name` are populated based on the database dialect. For example,
Snowflake/StarRocks/BigQuery populate catalogs, while SQLite leaves catalog/database/schema empty.

## Data Structure of Sample Data

| Field Name | Explanation |
|------------|-------------|
| `catalog_name` | Same as above |
| `database_name` | Same as above |
| `schema_name` | Same as above |
| `table_type` | Same as above |
| `table_name` | Same as above |
| `sample_rows` | Sample data for the current table/view/mv (stored as CSV text) |
| `identifier` | Same as above |

## How to Build

You can build it using the `datus-agent bootstrap-kb` command (the default component is `metadata`):

```bash
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components metadata \
    --kb_update_strategy [check/overwrite/incremental]
```

### Command Line Parameter Description

- `--namespace`: The key corresponding to your database configuration
- `--kb_update_strategy`: Execution strategy, there are three options:
  - `check`: Check the number of data entries currently constructed
  - `overwrite`: Fully overwrite existing data
  - `incremental`: Incremental update: if existing data has changed, update it and append non-existent data
- `--schema_linking_type`: Limit to `table`, `view`, `mv`, or `full` (default `full`)
- `--catalog`: Optional catalog name (for catalog-aware databases)
- `--database_name`: Optional database name (for database-aware databases)

## Usage Examples

### Check Current Status
```bash
datus-agent bootstrap-kb --namespace my_database --kb_update_strategy check
```

### Full Rebuild
```bash
datus-agent bootstrap-kb --namespace my_database --kb_update_strategy overwrite
```

### Incremental Update
```bash
datus-agent bootstrap-kb --namespace my_database --kb_update_strategy incremental
```

## Best Practices

### Database Configuration
- Ensure your database namespace is properly configured in `agent.yml`
- Verify database connectivity before running bootstrap commands
- Use appropriate credentials with read access to system tables

### Update Strategy Selection
- Use `check` to verify current state without making changes
- Use `overwrite` for initial setup or when schema has changed significantly
- Use `incremental` for regular updates to capture new tables and changes

### Performance Considerations
- Large databases may take time to process during initial bootstrap
- Consider running during off-peak hours for production databases
- Monitor disk space as metadata is stored locally in LanceDB

## Troubleshooting

### Common Issues
- **Permission errors**: Ensure database user has access to system/information schema tables
- **Connection timeouts**: Check network connectivity and database availability
- **Large result sets**: Consider filtering to specific schemas if database is very large

### Verification
After bootstrap completion, verify the metadata was captured correctly:
- Check LanceDB storage directory for populated files
- Test search functionality through the CLI
- Verify sample data represents actual table contents
