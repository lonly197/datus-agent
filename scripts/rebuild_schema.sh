#!/bin/bash
# 重建 Schema 脚本 - 清空后重新导入
# 用法: bash scripts/rebuild_schema.sh [--namespace=<name>] [--config=<path>]

set -e

# 默认配置
CONFIG_FILE="conf/agent.yml"
NAMESPACE=""
FORCE_CLEAR=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace=*)
            NAMESPACE="${1#*=}"
            ;;
        --config=*)
            CONFIG_FILE="${1#*=}"
            ;;
        --clear)
            FORCE_CLEAR="--clear"
            ;;
        --help|-h)
            echo "用法: bash scripts/rebuild_schema.sh [选项]"
            echo "选项:"
            echo "  --namespace=<name>  指定命名空间"
            echo "  --config=<path>     配置文件路径 (默认: conf/agent.yml)"
            echo "  --clear             清除现有数据"
            echo "  --help, -h          显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
    shift
done

# 检查命名空间是否指定
if [ -z "$NAMESPACE" ]; then
    echo "错误: 必须指定 --namespace 参数"
    echo "使用 --help 查看用法"
    exit 1
fi

echo "=========================================="
echo "Schema 重建脚本"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "命名空间: $NAMESPACE"
echo "清除数据: $([ -n "$FORCE_CLEAR" ] && echo '是' || echo '否')"
echo "=========================================="

# 执行重建
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
  --config="$CONFIG_FILE" \
  --namespace="$NAMESPACE" \
  --import-schemas \
  --import-only \
  $FORCE_CLEAR \
  --force

echo ""
echo "重建完成！"
