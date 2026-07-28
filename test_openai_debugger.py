import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# 导入待测模块（从核心模块导入，避免 tkinter 依赖）
from openai_debugger_core import (
    APIConfig, LogEntry, SpeedCalculator, ConversationHistory,
    format_message, build_request_body, parse_sse_line, get_preset_actions,
    APIClient
)


class TestAPIConfig:
    def test_config_initialization(self):
        config = APIConfig(
            base_url="http://localhost:8000/v1",
            api_key="sk-test123",
            model="gpt-3.5-turbo"
        )
        assert config.base_url == "http://localhost:8000/v1"
        assert config.api_key == "sk-test123"
        assert config.model == "gpt-3.5-turbo"
        assert config.timeout == 30

    def test_config_get_headers(self):
        config = APIConfig(base_url="http://test", api_key="sk-key", model="test")
        headers = config.get_headers()
        assert headers["Authorization"] == "Bearer sk-key"
        assert headers["Content-Type"] == "application/json"

    def test_config_custom_headers(self):
        config = APIConfig(base_url="http://test", api_key="sk-key", model="test")
        config.add_custom_header("X-Custom-Header", "MyValue")
        headers = config.get_headers()
        assert headers["X-Custom-Header"] == "MyValue"


class TestMessageFormatter:
    def test_format_system_message(self):
        msg = format_message("system", "You are a helper.")
        assert msg == {"role": "system", "content": "You are a helper."}

    def test_format_user_message(self):
        msg = format_message("user", "Hello")
        assert msg == {"role": "user", "content": "Hello"}

    def test_format_assistant_message(self):
        msg = format_message("assistant", "Hi there")
        assert msg == {"role": "assistant", "content": "Hi there"}


class TestPresetActions:
    def test_get_preset_actions_count(self):
        actions = get_preset_actions()
        assert len(actions) >= 4

    def test_connectivity_action(self):
        actions = {a['name']: a for a in get_preset_actions()}
        assert "测试联通性" in actions
        conn_action = actions["测试联通性"]
        assert conn_action['method'] == "GET"
        assert "/models" in conn_action['endpoint'] or conn_action['endpoint'] == ""

    def test_say_hello_action(self):
        actions = {a['name']: a for a in get_preset_actions()}
        assert "Say Hello" in actions
        hello_action = actions["Say Hello"]
        assert hello_action['method'] == "POST"
        assert "messages" in hello_action['body']
        assert hello_action['body']['messages'][0]['content'] == "Hello"

    def test_model_info_action(self):
        actions = {a['name']: a for a in get_preset_actions()}
        assert "获取模型信息" in actions


class TestSSEParser:
    def test_parse_single_chunk(self):
        line = 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        result = parse_sse_line(line)
        assert result is not None
        assert result['choices'][0]['delta']['content'] == "Hello"

    def test_parse_done_chunk(self):
        line = 'data: [DONE]'
        result = parse_sse_line(line)
        assert result is None

    def test_parse_invalid_json(self):
        line = 'data: {invalid json}'
        result = parse_sse_line(line)
        assert result is None

    def test_parse_empty_line(self):
        assert parse_sse_line("") is None
        assert parse_sse_line(": ping") is None


class TestSpeedCalculator:
    def test_speed_calculation(self):
        calc = SpeedCalculator()
        start_time = time.time()
        for i in range(10):
            calc.add_token(start_time + (i * 0.01))
        
        speed = calc.get_speed()
        assert speed > 0
        assert speed <= 150

    def test_speed_reset(self):
        calc = SpeedCalculator()
        calc.add_token(time.time())
        calc.add_token(time.time())
        assert calc.get_speed() > 0
        calc.reset()
        assert calc.get_speed() == 0.0


class TestConversationHistory:
    def test_add_message(self):
        history = ConversationHistory()
        history.add("user", "Hi")
        history.add("assistant", "Hello")
        assert len(history.get_messages()) == 2

    def test_clear_history(self):
        history = ConversationHistory()
        history.add("user", "Hi")
        history.clear()
        assert len(history.get_messages()) == 0

    def test_get_messages_for_api(self):
        history = ConversationHistory()
        history.add("user", "Test")
        msgs = history.get_messages()
        assert msgs[0] == {"role": "user", "content": "Test"}


class TestRequestBuilder:
    def test_build_chat_request(self):
        messages = [{"role": "user", "content": "Hi"}]
        params = {"temperature": 0.7, "max_tokens": 100}
        body = build_request_body(messages, params)
        
        assert "model" in body
        assert "messages" in body
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 100
        assert body["stream"] is True

    def test_build_request_with_custom_params(self):
        messages = []
        params = {"temperature": 0.9, "top_p": 0.5, "stream": False}
        body = build_request_body(messages, params)
        assert body["top_p"] == 0.5
        assert body["stream"] is False


class TestLogEntry:
    def test_log_entry_creation(self):
        entry = LogEntry("INFO", "Request sent", {"url": "http://test"})
        assert entry.level == "INFO"
        assert entry.message == "Request sent"
        assert entry.timestamp is not None

    def test_log_to_string(self):
        entry = LogEntry("ERROR", "Connection failed")
        log_str = entry.to_string()
        assert "ERROR" in log_str
        assert "Connection failed" in log_str


@pytest.mark.asyncio
async def test_mock_api_client_chat():
    """测试 SSE 解析和流式处理逻辑"""
    # 直接测试 parse_sse_line 函数来验证流式处理能力
    lines = [
        b'data: {"choices": [{"delta": {"content": "Hel"}}]}',
        b'data: {"choices": [{"delta": {"content": "lo"}}]}',
        b'data: [DONE]',
    ]
    
    results = []
    for line in lines:
        parsed = parse_sse_line(line.decode('utf-8'))
        if parsed:
            results.append(parsed)
    
    assert len(results) == 2
    assert results[0]['choices'][0]['delta']['content'] == "Hel"
    assert results[1]['choices'][0]['delta']['content'] == "lo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
