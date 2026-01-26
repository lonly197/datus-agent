#!/bin/bash
# 备份存储目录脚本

# 使用 Python 获取存储路径
DB_PATH=$(python3 -c "
import yaml
with open('conf/agent.yml', 'r') as f:
    config = yaml.safe_load(f)
    print(config['agent']['storage']['base_path'])
")

echo "检测到存储路径: $DB_PATH"
cp -r "$DB_PATH" "$DB_PATH.backup_v0_$(date +%Y%m%d_%H%M%S)"
echo "备份完成: $DB_PATH.backup_v0_$(date +%Y%m%d_%H%M%S)"
