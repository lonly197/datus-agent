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

### Sample Data Example

The repo includes a small example CSV at `sample_data/ext_knowledge_sample.csv`:

```bash
datus-agent bootstrap-kb \
    --namespace your_namespace \
    --components ext_knowledge \
    --ext_knowledge sample_data/ext_knowledge_sample.csv \
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

The loader expects the following required columns:

```csv
subject_path,name,terminology,explanation
Sales/Revenue,GMV,gmv,Gross merchandise value before refunds.
```

## Behavior Notes

- `incremental` mode skips entries already stored (based on a hash of subject path + terminology).
- If `--ext_knowledge` is omitted, the store is created but remains empty.
- Entries are indexed for vector search and are retrievable during workflow nodes that pull external knowledge.
