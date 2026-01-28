#!/usr/bin/env python3
"""
验证 business_terms.yml 是否正确生成和加载
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datus.configuration.business_term_config import (
    BUSINESS_TERM_TO_TABLE_MAPPING,
    BUSINESS_TERM_TO_SCHEMA_MAPPING,
    TABLE_KEYWORD_PATTERNS,
)


def main():
    print("=" * 60)
    print("BUSINESS TERMS VERIFICATION")
    print("=" * 60)
    
    # 统计信息
    print(f"\n✓ Table mappings: {len(BUSINESS_TERM_TO_TABLE_MAPPING)}")
    print(f"✓ Schema mappings: {len(BUSINESS_TERM_TO_SCHEMA_MAPPING)}")
    print(f"✓ Table keywords: {len(TABLE_KEYWORD_PATTERNS)}")
    
    # 检查关键业务术语
    key_terms = [
        "线索", "试驾", "订单", "客户", 
        "有效线索", "到店", "转化", "漏斗",
        "铂智3X", "首触", "战败", "退订"
    ]
    
    print("\n关键业务术语检查:")
    for term in key_terms:
        tables = BUSINESS_TERM_TO_TABLE_MAPPING.get(term, [])
        schemas = BUSINESS_TERM_TO_SCHEMA_MAPPING.get(term, [])
        if tables or schemas:
            print(f"  ✓ '{term}': {len(tables)} tables, {len(schemas)} schemas")
        else:
            print(f"  ✗ '{term}': NOT FOUND")
    
    # 检查垃圾条目
    print("\n质量检查:")
    suspicious = [
        k for k in BUSINESS_TERM_TO_TABLE_MAPPING.keys() 
        if k.startswith('_') or len(k) < 2 or k in ['id', 'code', 'name', 'key', 'engine']
    ]
    if suspicious:
        print(f"  ⚠ Found {len(suspicious)} suspicious entries: {suspicious[:5]}")
    else:
        print("  ✓ No suspicious entries found")
    
    # 示例查询
    print("\n示例查询测试:")
    test_queries = ["有效线索", "试驾", "订单实绩"]
    for query in test_queries:
        tables = BUSINESS_TERM_TO_TABLE_MAPPING.get(query, [])
        print(f"  '{query}' -> {tables[:3] if tables else 'NOT FOUND'}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
