#!/bin/bash
# Verification script for schema retrieval test implementation

echo "================================================================"
echo "Schema Retrieval Test - Implementation Verification"
echo "================================================================"
echo ""

# Check files exist
echo "Checking files..."
echo "-----------------------------------"

FILES=(
    "test_schema_retrieval.py:Test script"
    "TEST_SCHEMA_RETRIEVAL.md:Full documentation"
    "QUICKSTART_SCHEMA_TEST.md:Quick start guide"
    "test_example_usage.sh:Usage examples"
    "SCHEMA_TEST_SUMMARY.md:Implementation summary"
    "MIGRATION_OUTPUT_IMPROVEMENTS.md:Migration improvements doc"
)

all_exist=true
for item in "${FILES[@]}"; do
    IFS=':' read -r file desc <<< "$item"
    if [ -f "/Users/lonlyhuang/workspace/git/Datus-agent/$file" ]; then
        echo "✅ $file ($desc)"
    else
        echo "❌ $file (MISSING)"
        all_exist=false
    fi
done

echo ""

# Check permissions
echo "Checking permissions..."
echo "-----------------------------------"

if [ -x "/Users/lonlyhuang/workspace/git/Datus-agent/test_schema_retrieval.py" ]; then
    echo "✅ test_schema_retrieval.py is executable"
else
    echo "⚠️  test_schema_retrieval.py is not executable"
fi

if [ -x "/Users/lonlyhuang/workspace/git/Datus-agent/test_example_usage.sh" ]; then
    echo "✅ test_example_usage.sh is executable"
else
    echo "⚠️  test_example_usage.sh is not executable"
fi

echo ""

# Check syntax
echo "Checking Python syntax..."
echo "-----------------------------------"

if python -m py_compile /Users/lonlyhuang/workspace/git/Datus-agent/test_schema_retrieval.py 2>/dev/null; then
    echo "✅ test_schema_retrieval.py syntax is valid"
else
    echo "❌ test_schema_retrieval.py has syntax errors"
    all_exist=false
fi

echo ""

# Check help functionality
echo "Checking help functionality..."
echo "-----------------------------------"

if python /Users/lonlyhuang/workspace/git/Datus-agent/test_schema_retrieval.py --help > /dev/null 2>&1; then
    echo "✅ Help command works"
else
    echo "❌ Help command failed"
fi

echo ""

# Check imports
echo "Checking Python imports..."
echo "-----------------------------------"

python -c "
import sys
sys.path.insert(0, '/Users/lonlyhuang/workspace/git/Datus-agent')

# Check rich
try:
    from rich.console import Console
    print('✅ rich library available')
except ImportError:
    print('⚠️  rich library not available (optional)')

# Check argparse (standard library)
try:
    import argparse
    print('✅ argparse available')
except ImportError:
    print('❌ argparse not available')

# Check pathlib (standard library)
try:
    from pathlib import Path
    print('✅ pathlib available')
except ImportError:
    print('❌ pathlib not available')
"

echo ""

# Check migration script
echo "Checking migration script modifications..."
echo "-----------------------------------"

if grep -q "print_final_migration_report" /Users/lonlyhuang/workspace/git/Datus-agent/datus/storage/schema_metadata/migrate_v0_to_v1.py; then
    echo "✅ Migration script has print_final_migration_report function"
else
    echo "❌ Migration script missing print_final_migration_report function"
    all_exist=false
fi

if grep -q "finally:" /Users/lonlyhuang/workspace/git/Datus-agent/datus/storage/schema_metadata/migrate_v0_to_v1.py; then
    echo "✅ Migration script has finally block"
else
    echo "❌ Migration script missing finally block"
    all_exist=false
fi

echo ""

# Summary
echo "================================================================"
echo "Verification Summary"
echo "================================================================"
echo ""

if [ "$all_exist" = true ]; then
    echo "✅ All components verified successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run migration script:"
    echo "   python -m datus.storage.schema_metadata.migrate_v0_to_v1 \\"
    echo "     --config=conf/agent.yml \\"
    echo "     --namespace=your_namespace \\"
    echo "     --import-schemas --force"
    echo ""
    echo "2. Test schema retrieval:"
    echo "   python test_schema_retrieval.py \\"
    echo "     --config=conf/agent.yml \\"
    echo "     --namespace=your_namespace"
    echo ""
    echo "3. Read documentation:"
    echo "   - QUICKSTART_SCHEMA_TEST.md (quick start)"
    echo "   - TEST_SCHEMA_RETRIEVAL.md (full guide)"
    echo "   - SCHEMA_TEST_SUMMARY.md (implementation)"
    echo ""
    exit 0
else
    echo "❌ Some components failed verification"
    echo "Please check the errors above"
    echo ""
    exit 1
fi
