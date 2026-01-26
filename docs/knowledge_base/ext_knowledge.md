# External Knowledge Intelligence

## Overview

The External Knowledge component ingests curated business knowledge from a CSV file and stores it in LanceDB for
retrieval during query understanding, schema validation, and reasoning.

## Usage

### Basic Command

```bash
datus-agent bootstrap-kb \
    --namespace your_namespace \
    --components ext_knowledge \
    --ext_knowledge path/to/knowledge.csv \
    --kb_update_strategy overwrite
```

### Key Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--namespace` | ✅ | Database namespace | `analytics_db` |
| `--components` | ✅ | Components to initialize | `ext_knowledge` |
| `--ext_knowledge` | ✅ | CSV file path | `conf/knowledge.csv` |
| `--kb_update_strategy` | ✅ | Update strategy | `overwrite`/`incremental` |
| `--pool_size` | ❌ | Concurrent processing threads | `4` |

## CSV Format

The CSV is read directly into the store. Keep columns consistent and include a subject path for organization.
The exact schema is flexible, but should at least include a subject path and a knowledge term.

Example:

```csv
subject_path,terminology,description
Sales/Revenue,GMV,"Gross merchandise value before refunds"
Finance/Accounting,NetRevenue,"Revenue after refunds and chargebacks"
```

## Behavior Notes

- `incremental` mode skips entries already stored (based on a hash of subject path + terminology).
- If `--ext_knowledge` is omitted, the store is created but remains empty.
- Entries are indexed for vector search and are retrievable during workflow nodes that pull external knowledge.
