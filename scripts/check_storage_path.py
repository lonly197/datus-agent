#!/usr/bin/env python3
"""检查存储路径配置脚本"""
import yaml

with open('conf/agent.yml', 'r') as f:
    config = yaml.safe_load(f)
    print('Storage path:', config['agent']['storage']['base_path'])
