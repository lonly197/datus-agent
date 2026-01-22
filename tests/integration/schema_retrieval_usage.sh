#!/bin/bash
# Example usage of schema retrieval test script

echo "=============================================="
echo "Schema Retrieval Test - Example Usage"
echo "=============================================="
echo ""

# Example 1: Basic test with namespace
echo "Example 1: Testing with namespace"
echo "-----------------------------------"
cat << 'EOF'
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.yml \
    --namespace=analytics_db
EOF
echo ""

# Example 2: Test with custom query
echo "Example 2: Testing with custom query"
echo "-----------------------------------"
cat << 'EOF'
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.yml \
    --namespace=analytics_db \
    --query="销售转化率" \
    --top-n=30
EOF
echo ""

# Example 3: Production environment test
echo "Example 3: Production environment test"
echo "-----------------------------------"
cat << 'EOF'
python tests/integration/test_schema_retrieval_script.py \
    --config=/Users/lonlyhuang/workspace/myway/Examples/datus-docs/conf/agent.yml \
    --namespace=starrocks_prod \
    --query="销售线索" \
    --top-n=50
EOF
echo ""

# Example 4: Test specific business scenarios
echo "Example 4: Test specific business scenarios"
echo "-----------------------------------"
cat << 'EOF'
# Test customer conversion
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.yml \
    --namespace=analytics \
    --query="客户转化"

# Test sales performance
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.yml \
    --namespace=analytics \
    --query="销售业绩"

# Test lead analysis
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.yml \
    --namespace=analytics \
    --query="线索分析"
EOF
echo ""

# Example 5: Verify migration before testing
echo "Example 5: Verify migration before testing"
echo "-----------------------------------"
cat << 'EOF'
# Step 1: Run migration
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
    --config=conf/agent.yml \
    --namespace=analytics_db \
    --extract-relationships=true \
    --import-schemas \
    --force

# Step 2: Verify migration
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.yml \
    --namespace=analytics_db
EOF
echo ""

# Example 6: Troubleshooting
echo "Example 6: Troubleshooting common issues"
echo "-----------------------------------"
cat << 'EOF'
# Issue 1: Migration not successful
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
    --config=conf/agent.yml \
    --namespace=analytics_db \
    --force

# Issue 2: Check database connection
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.yml \
    --namespace=analytics_db

# Issue 3: Low comment extraction rate
# Check if DDL files contain COMMENT
grep -i "COMMENT" /path/to/ods_ddl.sql | head -10

# Issue 4: RAG search returns no results
# Verify data import
python -c "
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata import SchemaStorage

config = load_agent_config('conf/agent.yml')
storage = SchemaStorage(db_path=config.rag_storage_path())
print(f'Total tables: {len(storage._search_all(where=None))}')
"
EOF
echo ""

# Example 7: Automated testing script
echo "Example 7: Automated testing script"
echo "-----------------------------------"
cat << 'EOF'
#!/bin/bash
# Automated schema retrieval test

CONFIG_FILE=$1
NAMESPACE=$2
QUERY=${3:-"销售线索"}

if [ -z "$CONFIG_FILE" ] || [ -z "$NAMESPACE" ]; then
    echo "Usage: $0 <config_file> <namespace> [query]"
    exit 1
fi

echo "Running schema retrieval test..."
echo "Config: $CONFIG_FILE"
echo "Namespace: $NAMESPACE"
echo "Query: $QUERY"
echo ""

python tests/integration/test_schema_retrieval_script.py \
    --config="$CONFIG_FILE" \
    --namespace="$NAMESPACE" \
    --query="$QUERY"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "❌ Some tests failed. Check the output above."
    exit 1
fi
EOF
echo ""

# Example 8: Test different scenarios
echo "Example 8: Test different business scenarios"
echo "-----------------------------------"
cat << 'EOF'
#!/bin/bash
# Test multiple business scenarios

QUERIES=(
    "销售线索"
    "客户转化"
    "销售业绩"
    "商机"
    "潜在客户"
)

for QUERY in "${QUERIES[@]}"; do
    echo "Testing: $QUERY"
    python tests/integration/test_schema_retrieval_script.py \
        --config=conf/agent.yml \
        --namespace=analytics \
        --query="$QUERY"
    echo ""
done
EOF
echo ""

echo "=============================================="
echo "For more details, see:"
echo "  - tests/integration/SCHEMA_RETRIEVAL.md (full documentation)"
echo "  - QUICKSTART_SCHEMA_TEST.md (quick start guide)"
echo "=============================================="
