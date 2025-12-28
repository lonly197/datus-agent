# Auto-inject External Knowledge in Plan Mode

## Overview

The auto-inject external knowledge feature automatically detects SQL-related intent in user requests and injects relevant external knowledge (such as StarRocks SQL review rules) when the frontend omits the `ext_knowledge` parameter. This feature only activates in `plan_mode` for safety and user control.

## How It Works

### 1. Intent Detection Pipeline

When a `chat_research` request arrives with `plan_mode=true` but no `ext_knowledge`:

1. **Keyword-based Detection (Fast)**: Scans for SQL-related keywords and patterns
2. **LLM Classification (Fallback)**: If keyword detection is uncertain, uses LLM to classify intent
3. **Knowledge Search**: If SQL intent is detected, searches the `ExtKnowledgeStore` for relevant rules
4. **Injection**: Formats found knowledge and injects it into the workflow

### 2. User Confirmation

- **Interactive Mode**: Shows auto-detected knowledge and asks for user confirmation before proceeding
- **Auto Mode**: Automatically applies detected knowledge without user interaction

## Configuration

Add these optional settings to your `agent.yml`:

```yaml
agent:
  # Enable/disable auto-injection (default: true)
  plan_mode_auto_inject_ext_knowledge: true

  # Intent detection thresholds
  intent_detector:
    keyword_threshold: 1        # Min keyword matches for SQL detection
    llm_confidence_threshold: 0.7  # Min LLM confidence to accept classification
```

## API Usage

### Without Auto-injection (Traditional)
```json
{
  "namespace": "test",
  "task": "Generate SQL for user analysis",
  "ext_knowledge": "Use StarRocks best practices",
  "plan_mode": true
}
```

### With Auto-injection (New)
```json
{
  "namespace": "test",
  "task": "Generate SQL for user analysis",  // No ext_knowledge provided
  "plan_mode": true                          // Auto-injection enabled
}
```

The system will automatically detect SQL intent and inject relevant knowledge.

## Detection Logic

### Keyword Detection
The system looks for these indicators:

**Keywords:**
- SQL terms: `sql`, `select`, `join`, `where`, `partition`
- Database terms: `starrocks`, `hive`, `spark`, `snowflake`
- Business terms: `试驾`, `线索`, `转化`, `下定`, `用户`, `订单`

**Patterns:**
- `SELECT ... FROM ...`
- `INSERT INTO ...`
- `Date functions`: `date_format`, `datediff`, `date_add`

### LLM Classification
If keyword detection is uncertain, the LLM classifies into:
- `sql_generation`: Creating new SQL queries
- `sql_review`: Analyzing existing SQL
- `metadata_query`: Schema/table information requests
- `other`: Non-SQL tasks

## Examples

### Example 1: SQL Generation Task
```
User: "从 ODS 试驾表和线索表关联，统计每个月'首次试驾'到'下定'的平均转化周期"

Auto-detected knowledge:
- partition_pruning_basic: 强制要求：大表查询必须指定分区字段过滤
- join_optimization: Join顺序：小表前置，避免大表先参与Join
- comment_requirement: 建表语句及字段必须包含中文COMMENT
```

### Example 2: SQL Review Task
```
User: "审查以下SQL：SELECT * FROM dwd_assign_dlr_clue_fact_di"

Auto-detected knowledge:
- forbid_select_star: 严格禁止使用SELECT *
- partition_pruning_basic: 强制要求：大表查询必须指定分区字段过滤
- naming_convention: 表名必须遵循分层命名规范
```

## User Experience

### Interactive Mode (`auto_execute_plan: false`)
```
Plan Generated Successfully!

Auto-detected Knowledge:
The following knowledge was automatically detected and will be used:
  1. - partition_pruning_basic: 强制要求：大表查询必须指定分区字段过滤
  2. - forbid_select_star: 严格禁止使用SELECT *

AUTO-DETECTED KNOWLEDGE CONFIRMATION:
Accept auto-detected knowledge? (y/n) [y]: y
Accepted auto-detected knowledge

Execution Plan:
  1. Analyze table schema
  2. Generate optimized SQL
  3. Execute and validate results
```

### Auto Mode (`auto_execute_plan: true`)
```
Auto-detected Knowledge:
The following knowledge was automatically detected and will be used:
  1. - partition_pruning_basic: 强制要求：大表查询必须指定分区字段过滤
  2. - forbid_select_star: 严格禁止使用SELECT *

Auto execution mode (workflow/benchmark context)
[Processing plan...]
```

## Safety Features

1. **Plan Mode Only**: Only activates when `plan_mode=true` to ensure user awareness
2. **User Confirmation**: Interactive mode requires explicit user approval
3. **Fallback Handling**: If detection fails, continues without external knowledge
4. **Logging**: All auto-injection actions are logged for debugging

## Troubleshooting

### No Knowledge Injected
- Check that `ExtKnowledgeStore` has data: `datus bootstrap_kb --components ext_knowledge`
- Verify `plan_mode_auto_inject_ext_knowledge` is `true` in config
- Check logs for detection failures

### Wrong Knowledge Detected
- Adjust `keyword_threshold` or `llm_confidence_threshold` in config
- Provide explicit `ext_knowledge` in API request to override
- Report false positives for keyword list updates

### Performance Issues
- Keyword detection is fast (< 1ms)
- LLM fallback only triggers for uncertain cases
- Consider increasing `keyword_threshold` to reduce LLM calls

## Implementation Details

### Core Components
- `datus/agent/intent_detection.py`: Hybrid intent detection logic
- `datus/api/service.py`: Auto-injection in `_create_sql_task`
- `datus/tools/func_tool/context_search.py`: External knowledge search tools
- `datus/cli/plan_hooks.py`: User confirmation UI

### Data Flow
1. API request → Intent detection → Knowledge search → Injection
2. Chat node → Plan hooks → User confirmation → Plan execution
3. Schema linking → Enhanced context → LLM generation

## Future Enhancements

- **Domain-specific Models**: Fine-tuned intent classifiers for different business domains
- **Knowledge Ranking**: Score and rank detected knowledge by relevance
- **Context Learning**: Learn from user feedback to improve detection accuracy
- **Multi-language Support**: Enhanced detection for different language patterns
