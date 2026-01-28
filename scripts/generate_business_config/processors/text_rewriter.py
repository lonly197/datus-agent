#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
LLM-powered text rewriting for business term extraction.

将人类可读的Excel指标定义改写为便于检索的业务术语。
"""

import json
from typing import Dict, List, Optional

from datus.models.base import LLMBaseModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class LLMTextRewriter:
    """LLM文本改写器 - 将人类可读的指标定义改写为检索友好的术语"""

    def __init__(self, agent_config=None, use_llm: bool = False):
        self.use_llm = use_llm
        self.llm_model = None

        if use_llm and agent_config:
            try:
                self.llm_model = LLMBaseModel.create_model(agent_config=agent_config)
                logger.info("LLM文本改写器初始化成功")
            except Exception as e:
                logger.warning(f"LLM模型初始化失败: {e}，将使用规则模式")
                self.use_llm = False

    def rewrite_metric_definition(
        self,
        metric_name: str,
        business_definition: str,
        calc_logic: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """改写指标定义为检索友好的术语
        
        Args:
            metric_name: 指标名称
            business_definition: 业务定义
            calc_logic: 计算公式（可选）
            
        Returns:
            Dict包含:
                - core_concepts: 核心概念列表
                - search_terms: 检索关键词列表
                - synonyms: 同义词映射
        """
        if not self.use_llm or not self.llm_model:
            return self._rule_based_rewrite(metric_name, business_definition)

        prompt = f"""你是一个数据仓库业务分析师。请将以下指标定义改写为便于检索的业务术语。

## 输入
指标名称: {metric_name}
业务定义: {business_definition}
{"计算公式: " + calc_logic if calc_logic else ""}

## 任务
1. 提取核心业务概念（2-6个）
2. 生成检索关键词（包括同义词、相关术语）
3. 识别指标类型（数量/比率/占比/金额等）

## 输出格式
返回JSON:
{{
  "core_concepts": ["概念1", "概念2", ...],
  "search_terms": ["关键词1", "关键词2", ...],
  "synonyms": {{"术语": ["同义词1", "同义词2"]}},
  "metric_type": "数量|比率|占比|金额|次数|天数|时长",
  "business_entities": ["实体1", "实体2"]
}}

注意：
- core_concepts应该是该指标最核心的2-6个业务概念
- search_terms应该包含指标名称的各个组成部分及其同义词
- 所有术语应该是中文，便于业务人员搜索"""

        try:
            response = self.llm_model.generate_with_json_output(prompt)
            if isinstance(response, dict):
                return {
                    "core_concepts": response.get("core_concepts", []),
                    "search_terms": response.get("search_terms", []),
                    "synonyms": response.get("synonyms", {}),
                    "metric_type": response.get("metric_type", "未知"),
                    "business_entities": response.get("business_entities", []),
                }
        except Exception as e:
            logger.debug(f"LLM改写失败 '{metric_name}': {e}")

        return self._rule_based_rewrite(metric_name, business_definition)

    def _rule_based_rewrite(self, metric_name: str, business_definition: str) -> Dict[str, List[str]]:
        """基于规则的改写（LLM不可用时使用）"""
        from ..shared import extract_clean_keywords, METRIC_SUFFIXES

        # 提取指标名称中的核心概念
        core_concepts = extract_clean_keywords(metric_name, min_length=2, max_length=10)

        # 从业务定义中提取关键词
        def_keywords = extract_clean_keywords(business_definition, min_length=2, max_length=10)

        # 去除指标后缀得到核心概念
        clean_name = metric_name
        for suffix in METRIC_SUFFIXES:
            if clean_name.endswith(suffix) and len(clean_name) > len(suffix) + 2:
                core = clean_name[:-len(suffix)]
                if core and core not in core_concepts:
                    core_concepts.insert(0, core)
                break

        # 合并并去重
        all_terms = list(dict.fromkeys(core_concepts + def_keywords))

        return {
            "core_concepts": core_concepts[:6],
            "search_terms": all_terms[:15],
            "synonyms": {},
            "metric_type": self._detect_metric_type(metric_name),
            "business_entities": [],
        }

    def _detect_metric_type(self, metric_name: str) -> str:
        """检测指标类型"""
        if any(s in metric_name for s in ["率", "占比", "比例", "百分比"]):
            return "比率"
        elif any(s in metric_name for s in ["金额", "价格", "费用", "收入"]):
            return "金额"
        elif any(s in metric_name for s in ["天数", "时长", "周期", "时间"]):
            return "时长"
        elif any(s in metric_name for s in ["次数", "频次", "频率"]):
            return "次数"
        elif any(s in metric_name for s in ["数", "量", "数量"]):
            return "数量"
        return "其他"

    def rewrite_field_definition(
        self,
        field_name: str,
        field_cn: str,
        field_definition: str
    ) -> Dict[str, List[str]]:
        """改写字段定义为检索友好的术语
        
        Args:
            field_name: 字段英文名
            field_cn: 字段中文名
            field_definition: 字段定义
            
        Returns:
            Dict包含检索关键词
        """
        if not self.use_llm or not self.llm_model:
            return self._rule_based_field_rewrite(field_cn, field_definition)

        prompt = f"""请将以下数据仓库字段定义改写为便于检索的业务术语。

## 输入
字段英文名: {field_name}
字段中文名: {field_cn}
业务定义: {field_definition}

## 任务
1. 提取核心业务概念（中文）
2. 生成检索关键词列表
3. 识别相关实体（线索、客户、订单等）

## 输出格式
返回JSON:
{{
  "core_concepts": ["概念1", "概念2"],
  "search_terms": ["关键词1", "关键词2"],
  "related_entities": ["实体1", "实体2"],
  "business_scenarios": ["场景1", "场景2"]
}}"""

        try:
            response = self.llm_model.generate_with_json_output(prompt)
            if isinstance(response, dict):
                return {
                    "core_concepts": response.get("core_concepts", []),
                    "search_terms": response.get("search_terms", []),
                    "related_entities": response.get("related_entities", []),
                    "business_scenarios": response.get("business_scenarios", []),
                }
        except Exception as e:
            logger.debug(f"LLM字段改写失败 '{field_name}': {e}")

        return self._rule_based_field_rewrite(field_cn, field_definition)

    def _rule_based_field_rewrite(self, field_cn: str, field_definition: str) -> Dict[str, List[str]]:
        """基于规则的字段改写"""
        from ..shared import extract_clean_keywords

        concepts = extract_clean_keywords(field_cn, min_length=2, max_length=8)
        def_terms = extract_clean_keywords(field_definition, min_length=2, max_length=8)

        all_terms = list(dict.fromkeys(concepts + def_terms))

        return {
            "core_concepts": concepts[:4],
            "search_terms": all_terms[:12],
            "related_entities": [],
            "business_scenarios": [],
        }

    def batch_rewrite_metrics(
        self,
        metrics: List[Dict],
        batch_size: int = 10
    ) -> List[Dict]:
        """批量改写指标定义
        
        Args:
            metrics: 指标列表，每个指标包含 name, definition, calc_logic
            batch_size: 批处理大小
            
        Returns:
            包含改写结果的指标列表
        """
        results = []

        for i, metric in enumerate(metrics):
            try:
                rewritten = self.rewrite_metric_definition(
                    metric.get("name", ""),
                    metric.get("definition", ""),
                    metric.get("calc_logic")
                )
                results.append({
                    **metric,
                    "rewritten": rewritten,
                })

                if (i + 1) % batch_size == 0:
                    logger.info(f"已改写 {i + 1}/{len(metrics)} 个指标")

            except Exception as e:
                logger.warning(f"改写指标失败 '{metric.get('name')}': {e}")
                results.append(metric)

        return results
