# Document Store Intelligence

## Overview

The Document component stores chunked documents in LanceDB for retrieval. It is designed for general document search
and can be used to inject relevant passages into the workflow.

## Usage

### Basic Command

```bash
datus-agent bootstrap-kb \
    --namespace your_namespace \
    --components document \
    --kb_update_strategy overwrite
```

### Key Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--namespace` | ✅ | Database namespace | `analytics_db` |
| `--components` | ✅ | Components to initialize | `document` |
| `--kb_update_strategy` | ✅ | Update strategy | `overwrite`/`incremental` |

## Stored Fields

The document store persists chunk-level metadata:

- `title`: Document title
- `hierarchy`: Hierarchical path (e.g., section/subsection)
- `keywords`: Keyword list
- `language`: Document language
- `chunk_text`: Chunk content used for embeddings

## Notes

- Document ingestion is handled by the caller; bootstrap initializes the store and indices.
- Search is vector-based over `chunk_text`.
