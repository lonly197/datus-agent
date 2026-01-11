#!/bin/bash

# SQL Review Task Test Script
# This script sends a SQL review request to the Datus API and saves the response

# Configuration
API_URL="http://localhost:8080/workflows/chat_research"
NAMESPACE="test"
CATALOG_NAME="default_catalog"
DATABASE_NAME="test"
OUTPUT_DIR="./test_output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# SQL Review Task
TASK="审查：SELECT * FROM dwd_assign_dlr_clue_fact_di WHERE LEFT(clue_create_time, 10) = '2025-12-24' AND newest_original_clue_json LIKE '%线上%';"

# External knowledge for SQL review
EXTERNAL_KNOWLEDGE="使用StarRocks 3.3 SQL审查规则"

# Generate timestamp for output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/sql_review_${TIMESTAMP}.log"
MESSAGES_FILE="$OUTPUT_DIR/sql_review_${TIMESTAMP}_messages.txt"

echo "Starting SQL Review Task Test..."
echo "Task: $TASK"
echo "Output will be saved to:"
echo "  - Log: $LOG_FILE"
echo "  - Messages: $MESSAGES_FILE"
echo ""

# Send request and capture output
curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d "{
    \"namespace\": \"$NAMESPACE\",
    \"catalog_name\": \"$CATALOG_NAME\",
    \"database_name\": \"$DATABASE_NAME\",
    \"task\": \"$TASK\",
    \"external_knowledge\": \"$EXTERNAL_KNOWLEDGE\",
    \"execution_mode\": \"review_sql\"
  }" \
  2>&1 | tee "$MESSAGES_FILE" | tee "$LOG_FILE"

echo ""
echo "Test completed!"
echo "Full log saved to: $LOG_FILE"
echo "Messages saved to: $MESSAGES_FILE"
