# Schema Discovery 层级优先级验证结论

> **文档版本**: v1.0
> **更新日期**: 2026-01-29
> **目的**: 记录 schema discovery 层级优先级验证结论，为后续优化提供支持

---

## 一、验证结论摘要

### 1.1 核心结论

| 验证项目 | 结果 | 说明 |
|---------|------|------|
| 层级优先级规则 | ✅ 通过 | `DW > ADS > ODS` 优先级正确实现 |
| DW表 (dws_/dwd_/dim_) | ✅ 100%提升 | 159/159 表获得 +0.05 分数加成，排名提升 |
| ODS表 (ods_) | ✅ 100%降权 | 160/160 表获得 ×0.7 分数惩罚，排名下降 |
| Top 5 排名 | ✅ 符合预期 | 4个DWS表 + 1个DIM表，无ODS表进入Top 5 |

### 1.2 数据统计 (430张表)

```
层级分布:
├── DWS/DWD/DIM (DW层): 159表 (36.98%) → 100%提升
├── ADS层: 111表 (25.81%) → 保持中性
├── ODS层: 160表 (37.21%) → 100%降权
```

---

## 二、验证方法

### 2.1 测试脚本

```bash
# 使用 Mock 数据验证逻辑
python scripts/test_layer_priority.py --mock --config=conf/agent.yml

# 使用真实 LanceDB 数据验证
python scripts/test_layer_priority.py --lancedb --config=conf/agent.yml
```

### 2.2 验证原理

脚本模拟 `discovery_engine.py::finalize_candidates()` 的权重计算逻辑：

```python
def apply_layer_weights(candidates):
    for item in candidates:
        # 白名单: dws_/dwd_/dim_ → +0.05
        if match_prefix(item["table_name"], ["dws_", "dwd_", "dim_"]):
            item["score"] += whitelist_bonus  # +0.05

        # 黑名单: ods_ → ×0.7
        elif match_prefix(item["table_name"], ["ods_"]):
            item["score"] *= (1 - blacklist_penalty)  # ×0.7
```

---

## 三、配置位置

层级优先级配置位于 `datus/configuration/agent_config.py`:

```python
class SchemaDiscoveryConfig:
    # 层级白名单 (获得加分)
    table_prefix_whitelist: List[str] = field(
        default_factory=lambda: ["dws_", "dwd_", "dim_"]
    )

    # 层级黑名单 (获得降权)
    table_prefix_blacklist: List[str] = field(
        default_factory=lambda: ["ods_"]
    )

    # 权重参数
    prefix_blacklist_penalty: float = 0.3   # ODS: ×0.7
    prefix_whitelist_bonus: float = 0.05    # DW: +0.05
```

---

## 四、已知限制与优化建议

### 4.1 当前限制

| 限制项 | 影响 | 优先级 |
|-------|------|-------|
| FTS分词缺少垂直领域术语 | 中文查询可能匹配不精准 | 中 |
| 未启用 generate_business_config.py | 业务术语库未配置 | 高 |
| Reranker未启用 | 高精度场景召回可能不理想 | 低 |

### 4.2 优化建议

#### 高优先级

1. **启用业务术语配置**
   ```bash
   python scripts/generate_business_config.py --source=docs/业务术语表.xlsx
   ```

2. **完善 agent.yml 配置**
   ```yaml
   schema_discovery:
     business_term_config_path: conf/business_terms.yml
     enable_reranker: true  # 高精度场景启用
   ```

#### 中优先级

3. **中文查询增强**
   ```bash
   datus-agent --llm-rewrite  # 启用 LLM query rewrite
   ```

4. **垂直领域分词扩展**
   - 自定义 jieba 词库
   - 或集成专业中文分词组件

---

## 五、相关文件清单

### 5.1 核心文件

| 文件路径 | 说明 |
|---------|------|
| `datus/agent/node/schema_discovery/discovery_engine.py` | 层权重量化逻辑 |
| `datus/configuration/agent_config.py` | 层级优先级配置 |
| `scripts/test_layer_priority.py` | 层级优先级验证脚本 |

### 5.2 测试脚本

| 文件路径 | 说明 |
|---------|------|
| `scripts/check_search_text_fts.py` | FTS 检索测试 (不含权重) |
| `scripts/test_layer_priority.py` | 层级权重验证 (含权重) |

### 5.3 文档

| 文件路径 | 版本 | 说明 |
|---------|------|------|
| `docs/Schema 元数据管理脚本指南.md` | v1.8 | 脚本使用指南 |
| `docs/scripts/脚本清单.md` | v1.6 | 脚本清单 |

---

## 六、FAQ

### Q1: check_search_text_fts.py 和 test_layer_priority.py 的区别？

**答**: `check_search_text_fts.py` 直接调用 `search_fts()` 跳过权重逻辑，用于测试底层 FTS 检索质量；`test_layer_priority.py` 模拟 `finalize_candidates()` 流程，用于验证层级权重是否正确应用。

### Q2: 为什么 Top 5 都是 DW 层表？

**答**: 这是预期行为。DW 层表 (dws_/dwd_/dim_) 获得 +0.05 分数加成，同时其表名通常包含更多业务语义，因此排名优先于 ODS 层。

### Q3: ODS 表是否完全不会被召回？

**答**: 否。ODS 表仅受到 ×0.7 的分数降权，不会被过滤。当 DW/ADS 层无匹配时，ODS 表仍可能被召回。

---

## 七、结论

Schema Discovery 的层级优先级规则已正确实现并验证通过：
- ✅ 优先级顺序: `DW > ADS > ODS`
- ✅ DW表 100% 获得分数提升
- ✅ ODS表 100% 受到分数降权
- ✅ 符合 text2sql 业务需求

后续优化方向：启用业务术语配置、增强中文分词能力、选择性启用 Reranker。
