#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SSE 合规性测试脚本

验证 /workflows/chat_research 接口的 SSE 输出是否符合
ChatBot接收信息响应结构定义.ts 的要求。

使用方法:
    cd /path/to/Datus-agent
    source .venv/bin/activate
    python tests/test_sse_compliance.py
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datus.api.event_converter import DeepResearchEventConverter
from datus.api.models import DeepResearchEventType
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus


class SSEComplianceChecker:
    """SSE 事件合规性检查器"""

    def __init__(self):
        self.events_collected: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def validate_base_event(self, event_data: Dict[str, Any]) -> bool:
        """验证基础事件结构"""
        required_fields = ["id", "planId", "timestamp", "event"]

        for field in required_fields:
            if field not in event_data:
                self.errors.append(f"Missing required field '{field}' in event: {event_data}")
                return False

        # 验证 event 类型
        if event_data["event"] not in [e.value for e in DeepResearchEventType]:
            self.errors.append(
                f"Invalid event type '{event_data['event']}'. Valid types: {[e.value for e in DeepResearchEventType]}"
            )
            return False

        # 验证 timestamp 是数字
        if not isinstance(event_data["timestamp"], (int, float)):
            self.errors.append(f"Timestamp must be numeric, got {type(event_data['timestamp'])}")
            return False

        return True

    def validate_chat_event(self, event_data: Dict[str, Any]) -> bool:
        """验证 ChatEvent"""
        if "content" not in event_data:
            self.errors.append("ChatEvent missing required 'content' field")
            return False
        return True

    def validate_plan_update_event(self, event_data: Dict[str, Any]) -> bool:
        """验证 PlanUpdateEvent"""
        if "todos" not in event_data:
            self.errors.append("PlanUpdateEvent missing required 'todos' field")
            return False

        if not isinstance(event_data["todos"], list):
            self.errors.append("PlanUpdateEvent 'todos' must be an array")
            return False

        # 验证 todos 结构
        for i, todo in enumerate(event_data["todos"]):
            if not isinstance(todo, dict):
                self.errors.append(f"Todo item {i} must be an object")
                continue

            required_todo_fields = ["id", "content", "status"]
            for field in required_todo_fields:
                if field not in todo:
                    self.errors.append(f"Todo item {i} missing required field '{field}'")

            # 验证 status
            if "status" in todo and todo["status"] not in ["pending", "in_progress", "completed"]:
                self.errors.append(f"Invalid todo status '{todo['status']}'. Valid: pending, in_progress, completed")

        return True

    def validate_tool_call_event(self, event_data: Dict[str, Any]) -> bool:
        """验证 ToolCallEvent"""
        required_fields = ["toolCallId", "toolName", "input"]

        for field in required_fields:
            if field not in event_data:
                self.errors.append(f"ToolCallEvent missing required field '{field}'")
                return False

        return True

    def validate_tool_call_result_event(self, event_data: Dict[str, Any]) -> bool:
        """验证 ToolCallResultEvent"""
        required_fields = ["toolCallId", "data", "error"]

        for field in required_fields:
            if field not in event_data:
                self.errors.append(f"ToolCallResultEvent missing required field '{field}'")
                return False

        if not isinstance(event_data["error"], bool):
            self.errors.append("ToolCallResultEvent 'error' field must be boolean")

        return True

    def validate_complete_event(self, event_data: Dict[str, Any]) -> bool:
        """验证 CompleteEvent"""
        # content 是可选的
        return True

    def validate_error_event(self, event_data: Dict[str, Any]) -> bool:
        """验证 ErrorEvent"""
        if "error" not in event_data:
            self.errors.append("ErrorEvent missing required 'error' field")
            return False
        return True

    def validate_event(self, event_data: Dict[str, Any]) -> bool:
        """验证单个事件"""
        # 先验证基础结构
        if not self.validate_base_event(event_data):
            return False

        # 根据事件类型验证具体结构
        event_type = event_data["event"]

        if event_type == DeepResearchEventType.CHAT.value:
            return self.validate_chat_event(event_data)
        elif event_type == DeepResearchEventType.PLAN_UPDATE.value:
            return self.validate_plan_update_event(event_data)
        elif event_type == DeepResearchEventType.TOOL_CALL.value:
            return self.validate_tool_call_event(event_data)
        elif event_type == DeepResearchEventType.TOOL_CALL_RESULT.value:
            return self.validate_tool_call_result_event(event_data)
        elif event_type == DeepResearchEventType.COMPLETE.value:
            return self.validate_complete_event(event_data)
        elif event_type == DeepResearchEventType.ERROR.value:
            return self.validate_error_event(event_data)
        elif event_type == DeepResearchEventType.REPORT.value:
            return True  # ReportEvent 结构较简单，暂不验证

        return True

    def validate_events_sequence(self) -> bool:
        """验证事件序列的逻辑一致性"""
        # 检查是否有 CompleteEvent 作为最后一个事件
        if not self.events_collected:
            self.errors.append("No events collected")
            return False

        last_event = self.events_collected[-1]
        if last_event["event"] != DeepResearchEventType.COMPLETE.value:
            self.errors.append("Last event should be CompleteEvent")
            return False

        # 检查 planId 一致性
        plan_ids = set()
        for event in self.events_collected:
            if "planId" in event:
                plan_ids.add(event["planId"])

        if len(plan_ids) > 1:
            self.errors.append(f"Multiple planIds found: {plan_ids}. All events should share the same planId")
            return False

        # 检查 ToolCallEvent 和 ToolCallResultEvent 的 toolCallId 匹配
        tool_calls = {}
        tool_results = {}

        for event in self.events_collected:
            if event["event"] == DeepResearchEventType.TOOL_CALL.value:
                tool_call_id = event["toolCallId"]
                tool_calls[tool_call_id] = event
            elif event["event"] == DeepResearchEventType.TOOL_CALL_RESULT.value:
                tool_call_id = event["toolCallId"]
                tool_results[tool_call_id] = event

        # 检查是否有未匹配的 toolCallId
        for call_id in tool_calls:
            if call_id not in tool_results:
                self.errors.append(f"ToolCallEvent with toolCallId '{call_id}' has no matching ToolCallResultEvent")

        for result_id in tool_results:
            if result_id not in tool_calls:
                self.errors.append(f"ToolCallResultEvent with toolCallId '{result_id}' has no matching ToolCallEvent")

        return True

    def analyze_events(self, events: List[Dict[str, Any]]) -> bool:
        """分析收集到的事件"""
        self.events_collected = events
        self.errors = []

        print(f"📊 收到 {len(events)} 个事件")

        # 统计事件类型
        event_types = {}
        for event in events:
            event_type = event.get("event", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1

        print("📈 事件类型统计:")
        for event_type, count in event_types.items():
            print(f"   {event_type}: {count}")

        # 验证每个事件
        valid_count = 0
        for i, event in enumerate(events):
            if self.validate_event(event):
                valid_count += 1
            else:
                print(f"❌ 事件 {i} 验证失败: {event}")

        # 验证事件序列
        sequence_valid = self.validate_events_sequence()

        print(f"✅ 有效事件: {valid_count}/{len(events)}")
        print(f"✅ 序列验证: {'通过' if sequence_valid else '失败'}")

        if self.errors:
            print("\n❌ 发现错误:")
            for error in self.errors:
                print(f"   - {error}")

        return len(self.errors) == 0 and sequence_valid


async def test_chat_research_sse_mock():
    """使用 mock 数据测试 SSE 事件合规性"""

    checker = SSEComplianceChecker()

    print("🚀 开始测试 SSE 事件合规性 (使用模拟数据)")

    # 创建模拟的 ActionHistory 对象，模拟实际的聊天研究流程
    actions = [
        # 1. 初始聊天事件
        ActionHistory(
            action_id="chat_1",
            role=ActionRole.ASSISTANT,
            action_type="llm_generation",
            messages="开始分析用户需求",
            input={},
            output={"content": "正在分析'首次试驾'到'下定'的平均转化周期需求..."},
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        ),
        # 2. 计划更新事件 - 生成执行计划 (使用 todo_write 工具)
        ActionHistory(
            action_id="plan_1",
            role=ActionRole.TOOL,
            action_type="todo_write",
            messages="生成执行计划",
            input={
                "function_name": "todo_write",
                "arguments": '{"todos_json": "[{\\"content\\": \\"理解业务需求：分析\'首次试驾\'到\'下定\'的平均转化周期\\", \\"status\\": \\"pending\\"}, {\\"content\\": \\"搜索相关表结构：试驾表和线索表\\", \\"status\\": \\"pending\\"}, {\\"content\\": \\"分析表字段和关联关系\\", \\"status\\": \\"pending\\"}, {\\"content\\": \\"设计SQL逻辑：识别首次试驾时间、下定时间\\", \\"status\\": \\"pending\\"}, {\\"content\\": \\"计算转化周期（天数）并按月统计\\", \\"status\\": \\"pending\\"}, {\\"content\\": \\"编写完整SQL代码并添加详细注释\\", \\"status\\": \\"pending\\"}]"}',
            },
            output={
                "success": 1,
                "error": None,
                "result": {
                    "message": "Successfully created todo list",
                    "todo_list": {
                        "items": [
                            {
                                "id": "task_1",
                                "content": "理解业务需求：分析'首次试驾'到'下定'的平均转化周期",
                                "status": "completed",
                            },
                            {"id": "task_2", "content": "搜索相关表结构：试驾表和线索表", "status": "in_progress"},
                            {"id": "task_3", "content": "分析表字段和关联关系", "status": "pending"},
                            {"id": "task_4", "content": "设计SQL逻辑：识别首次试驾时间、下定时间", "status": "pending"},
                            {"id": "task_5", "content": "计算转化周期（天数）并按月统计", "status": "pending"},
                            {"id": "task_6", "content": "编写完整SQL代码并添加详细注释", "status": "pending"},
                        ]
                    },
                },
            },
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        ),
        # 3. 单个工具调用事件 (包含输入和输出)
        ActionHistory(
            action_id="tool_call_1",
            role=ActionRole.TOOL,
            action_type="schema_linking",
            messages="调用数据库工具搜索表结构",
            input={"function_name": "search_table", "table": "trial_drive_table"},
            output={
                "success": True,
                "result": {
                    "message": "Successfully searched table structure",
                    "data": {"tables": ["ods_trial_drive", "ods_clue"]},
                },
            },
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        ),
        # 4. 最终的聊天响应 - 包含生成的 SQL
        ActionHistory(
            action_id="final_response",
            role=ActionRole.ASSISTANT,
            action_type="chat_response",
            messages="生成最终 SQL 代码",
            input={},
            output={
                "response": "已成功生成 SQL 代码来统计每月'首次试驾'到'下定'的平均转化周期",
                "sql": "SELECT * FROM table",  # 简化的 SQL
                "tokens_used": 150,
            },
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        ),
        # 5. 工作流完成事件
        ActionHistory(
            action_id="workflow_complete",
            role=ActionRole.WORKFLOW,
            action_type="workflow_completion",
            messages="工作流执行完成",
            input={},
            output={"final_result": "SQL generation completed successfully"},
            status=ActionStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now(),
        ),
    ]

    # 使用事件转换器将 ActionHistory 转换为 DeepResearchEvent
    converter = DeepResearchEventConverter()
    events = []

    for i, action in enumerate(actions, 1):
        event_result = converter.convert_action_to_event(action, i)
        if event_result:
            # convert_action_to_event 现在返回列表
            if isinstance(event_result, list):
                for event in event_result:
                    # 将事件对象转换为字典格式
                    event_dict = event.model_dump()
                    events.append(event_dict)
                    event_type = event_dict["event"]
                    if hasattr(event_type, "value"):
                        event_type = event_type.value
                    print(f"📨 转换事件: {event_type} (ID: {event_dict.get('id', 'unknown')})")
            else:
                # 单个事件对象
                event_dict = event_result.model_dump()
                events.append(event_dict)
                print(f"📨 转换事件: {event_dict['event']} (ID: {event_dict.get('id', 'unknown')})")

    print(f"\n📊 总共转换了 {len(events)} 个事件")

    # 分析事件合规性
    return checker.analyze_events(events)


async def main():
    """主函数"""
    print("=" * 60)
    print("SSE 合规性测试 - 验证 ChatBot 响应结构定义")
    print("=" * 60)

    print("使用模拟数据测试事件转换和合规性")
    print("这将验证 DeepResearchEventConverter 是否正确生成符合 TypeScript 定义的事件")
    print()

    # 运行测试
    success = await test_chat_research_sse_mock()

    print("\n" + "=" * 60)
    if success:
        print("🎉 SSE 合规性测试通过!")
        print("✅ 所有事件都符合 ChatBot接收信息响应结构定义.ts 的要求")
        print("✅ DeepResearchEventConverter 工作正常")
    else:
        print("❌ SSE 合规性测试失败!")
        print("请检查上述错误信息并修复代码")

    return success


if __name__ == "__main__":
    # 运行异步主函数
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
