# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
中文本地化模块，用于 Streamlit Web 界面

提供简体中文翻译和本地化功能。
所有 UI 文本都通过 t() 函数调用获取，确保界面显示为中文。
"""

from typing import Dict, Optional

# 中文本地化字典 - 简体中文 (zh_CN)
LOCALE_ZH_CN: Dict[str, str] = {
# 页面配置
"page_title": "AI Agent",

    # 主标题和描述
    "title_main": "🤖 AI Agent",
    "title_subagent": "🤖 AI Agent - {subagent}",
    "caption_main": "智能助手",
    "caption_subagent": "专用的 {subagent} 子代理，用于 SQL 生成 - 自然语言转 SQL",

    # 侧边栏
    "sidebar_title": "AI Agent",
    "sidebar_subagent_title": "🤖 当前子代理",
    "sidebar_subagent_info": "**{subagent}** (GenSQL 模式)",
    "sidebar_namespace_title": "🏷️ 当前命名空间",
    "sidebar_namespace_info": "**{namespace}**",
    "sidebar_model_title": "🤖 聊天模型",
    "sidebar_session_title": "💬 会话",
    "sidebar_history_title": "📚 会话历史",
    "sidebar_debug_title": "🔍 调试信息",

    # 按钮
    "button_clear_chat": "🗑️ 清空聊天",
    "button_load_session": "🔗 加载会话",
    "button_use_subagent": "🚀 使用 {subagent}",
    "button_save_success": "👍 成功",
    "button_download": "⏬ 下载",
    "button_configure_chart": "⚙️ 配置图表",

    # Tabs
    "tab_generated_sql": "🔧 生成的 SQL",
    "tab_execute_result": "📊 执行结果",
    "tab_chart": "📈 可视化",

    # 提示信息
    "config_loaded": "✅ 配置已加载！",
    "config_failed": "❌ 配置加载失败",
    "model_changed": "模型已更改为: {model}",
    "no_sessions": "暂无保存的会话",
    "showing_sessions": "显示 {count} 个最近的会话",
    "loading_config": "⚠️ 正在加载配置...",
    "viewing_shared_session": "📖 查看共享会话（只读）- ID: {id}...",
    "something_wrong": "⚠️ 出现问题，请尝试重启。",
    "config_description": "配置文件包含数据库连接、模型设置等。",
    "no_active_session": "未找到活跃会话。无法保存成功案例。",
    "unsafe_subagent": "不安全的子代理名称。",
    "success_saved": "✅ 成功案例已保存！会话链接: {link}",
    "save_failed": "保存成功案例失败: {error}",
    "session_not_found": "会话 {id} 未找到或无数据。",
    "session_no_messages": "会话 {id} 无消息可显示。",
    "session_load_failed": "加载会话失败: {error}",
    "config_load_failed": "加载配置失败: {e}",
    "db_not_initialized": "数据库连接器未初始化。请先配置代理。",
    "excel_generation_failed": "生成 Excel 失败: {error}",

    # 会话项
    "session_expander": "📝 {sid}...",
    "session_created": "**创建时间:** {date}",
    "session_messages": "**消息数:** {count}",
    "session_latest": "**最新消息:** {msg}",

    # 子代理
    "subagents_expander": "🔧 访问专用子代理",
    "subagents_available": "**可用的专用子代理:**",
    "subagents_description": "**{name} 子代理**: `{url}`",
    "subagents_tip": "💡 **提示**: 收藏子代理 URL 以便直接访问！",

    # 执行详情
    "execution_expander": "🔍 查看完整执行详情 ({count} 步)",
    "execution_trace": "完整执行跟踪，包含所有中间步骤",

    # 图表和数据
    "no_data_return": "无数据返回",
    "chart_failed": "图表建议失败: {error}",
    "chart_empty": "数据为空，无法生成图表。",
    "chart_select_type": "**选择图表类型和轴映射**",
    "chart_configure_hint": "请点击上面的 '⚙️ 配置图表' 按钮选择至少一个 Y 轴指标。",
    "chart_pie_warning": "饼图只能选择一个指标（Y 轴）。",

    # 执行详情
    "action_input": "**输入:**",
    "action_no_input": "(无输入)",
    "action_output": "**输出:**",
    "action_no_output": "(无输出)",
    "action_started_duration": "⏱️ 开始时间: {time} | 持续时间: {duration}s",
    "action_started": "⏱️ 开始时间: {time}",

    # 控制台输出
    "console_error_webchat_not_found": "❌ 错误: 在 {path} 未找到 Web 聊天机器人",
    "console_starting_web_interface": "🚀 启动 Datus Web 界面...",
    "console_using_namespace": "🔗 使用命名空间: {ns}",
    "console_using_config": "⚙️ 使用配置: {config}",
    "console_using_database": "📚 使用数据库: {db}",
    "console_server_started": "🌐 服务器启动在 http://{host}:{port}",
    "console_press_ctrl_c": "⏹️ 按 Ctrl+C 停止服务器",
    "console_web_server_stopped": "🛑 Web 服务器已停止",
    "console_web_interface_failed": "❌ 启动 Web 界面失败: {error}",

    # 响应
    "ai_response": "### 💬 AI 响应",
    "response_error": "抱歉，无法生成有效响应。请检查执行详情以获取更多信息。",

    # 调试信息标签
    "debug_expander": "调试详情",
    "debug_query_params": "查询参数:",
    "debug_startup_subagent": "启动子代理:",
    "debug_current_subagent": "当前子代理:",
    "debug_session_id": "会话 ID:",
    "debug_has_current_node": "有 current_node:",
    "debug_has_chat_node": "有 chat_node:",
}


def t(key: str, default: Optional[str] = None) -> str:
    """
    获取本地化文本的翻译函数。

    Args:
        key: 翻译键
        default: 默认值，如果找不到翻译则返回此值或 key 本身

    Returns:
        翻译后的文本
    """
    return LOCALE_ZH_CN.get(key, default or key)
