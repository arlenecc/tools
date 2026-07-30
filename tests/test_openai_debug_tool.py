"""测试核心模块"""
import pytest
from datetime import datetime
import time
import json
import os

# 导入被测试模块
from src.log_entry import LogEntry
from src.logger import Logger
from src.config_manager import Config, ConfigManager
from src.message_history import Message, MessageHistory
from src.speed_calculator import SpeedCalculator, SpeedStats
from src.api_client import APIClient, ModelInfo, ChatResponse


class TestLogEntry:
    """测试日志条目"""

    def test_create_log_entry(self):
        entry = LogEntry(level="INFO", message="Test message")
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert isinstance(entry.timestamp, datetime)

    def test_log_entry_to_string(self):
        entry = LogEntry(level="DEBUG", message="Debug info")
        str_repr = str(entry)
        assert "[DEBUG]" in str_repr
        assert "Debug info" in str_repr

    def test_log_entry_with_details(self):
        entry = LogEntry(level="ERROR", message="Error occurred", details="Stack trace")
        str_repr = str(entry)
        assert "Stack trace" in str_repr

    def test_log_entry_to_dict(self):
        entry = LogEntry(level="INFO", message="Test")
        data = entry.to_dict()
        assert data["level"] == "INFO"
        assert data["message"] == "Test"

    def test_log_entry_from_dict(self):
        data = {
            "timestamp": "2024-01-01T12:00:00.000000",
            "level": "WARNING",
            "message": "Warning msg",
            "details": None
        }
        entry = LogEntry.from_dict(data)
        assert entry.level == "WARNING"
        assert entry.message == "Warning msg"


class TestLogger:
    """测试日志管理器"""

    def test_logger_add_entry(self):
        logger = Logger()
        entry = logger.info("Test info")
        assert len(logger.get_entries()) == 1
        assert entry.level == "INFO"

    def test_logger_multiple_levels(self):
        logger = Logger()
        logger.debug("Debug")
        logger.info("Info")
        logger.warning("Warning")
        logger.error("Error")
        assert len(logger.get_entries()) == 4

    def test_logger_clear(self):
        logger = Logger()
        logger.info("Test")
        logger.clear()
        assert len(logger.get_entries()) == 0

    def test_logger_max_entries(self):
        logger = Logger(max_entries=5)
        for i in range(10):
            logger.info(f"Message {i}")
        assert len(logger.get_entries()) == 5

    def test_logger_callback(self):
        logger = Logger()
        received = []

        def callback(entry):
            received.append(entry)

        logger.register_callback(callback)
        logger.info("Test")
        assert len(received) == 1
        assert received[0].message == "Test"


class TestConfig:
    """测试配置"""

    def test_config_default_values(self):
        config = Config()
        assert config.base_url == "http://localhost:11434/v1"
        assert config.api_key == ""
        assert config.model == ""

    def test_config_to_dict(self):
        config = Config(base_url="http://test.com", api_key="key123")
        data = config.to_dict()
        assert data["base_url"] == "http://test.com"
        assert data["api_key"] == "key123"

    def test_config_from_dict(self):
        data = {"base_url": "http://example.com", "api_key": "", "model": "", "window_width": 800, "window_height": 600}
        config = Config.from_dict(data)
        assert config.base_url == "http://example.com"


class TestConfigManager:
    """测试配置管理器"""

    def test_config_manager_load_default(self, tmp_path):
        config_file = tmp_path / "config.json"
        manager = ConfigManager(str(config_file))
        config = manager.load()
        assert isinstance(config, Config)

    def test_config_manager_save_and_load(self, tmp_path):
        config_file = tmp_path / "config.json"
        manager = ConfigManager(str(config_file))
        config = Config(base_url="http://saved.com", api_key="saved_key")
        assert manager.save(config)
        
        # 重新加载
        manager2 = ConfigManager(str(config_file))
        loaded = manager2.load()
        assert loaded.base_url == "http://saved.com"
        assert loaded.api_key == "saved_key"


class TestMessage:
    """测试消息"""

    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        msg = Message(role="assistant", content="Hi there")
        data = msg.to_dict()
        assert data["role"] == "assistant"
        assert data["content"] == "Hi there"

    def test_message_to_api_format(self):
        msg = Message(role="user", content="Test")
        api_fmt = msg.to_api_format()
        assert api_fmt == {"role": "user", "content": "Test"}


class TestMessageHistory:
    """测试对话历史"""

    def test_add_user_message(self):
        history = MessageHistory()
        msg = history.add_user_message("Hello")
        assert msg.role == "user"
        assert len(history) == 1

    def test_add_assistant_message(self):
        history = MessageHistory()
        msg = history.add_assistant_message("Hi", model="gpt-4")
        assert msg.role == "assistant"
        assert msg.model == "gpt-4"

    def test_get_api_messages(self):
        history = MessageHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi")
        api_msgs = history.get_api_messages()
        assert len(api_msgs) == 2
        assert api_msgs[0] == {"role": "user", "content": "Hello"}

    def test_clear_history(self):
        history = MessageHistory()
        history.add_user_message("Test")
        history.clear()
        assert len(history) == 0

    def test_max_messages_limit(self):
        history = MessageHistory(max_messages=3)
        for i in range(5):
            history.add_user_message(f"Msg {i}")
        assert len(history) == 3


class TestSpeedCalculator:
    """测试速度计算器"""

    def test_speed_start_stop(self):
        calc = SpeedCalculator()
        calc.start()
        time.sleep(0.1)
        stats = calc.stop()
        assert stats.elapsed_time >= 0.1
        assert stats.total_tokens == 0

    def test_speed_add_token(self):
        calc = SpeedCalculator()
        calc.start()
        calc.add_token()
        time.sleep(0.05)
        calc.add_token()
        stats = calc.stop()
        assert stats.total_tokens == 2

    def test_speed_reset(self):
        calc = SpeedCalculator()
        calc.start()
        calc.add_token()
        calc.reset()
        stats = calc.get_current_stats()
        assert stats.total_tokens == 0

    def test_instant_speed(self):
        calc = SpeedCalculator()
        calc.start()
        calc.add_token()
        time.sleep(0.05)
        calc.add_token()
        instant = calc.get_instant_speed()
        assert instant > 0


class TestModelInfo:
    """测试模型信息"""

    def test_model_info_from_dict(self):
        data = {"id": "gpt-4", "name": "GPT-4", "created": 1234567890}
        model = ModelInfo.from_dict(data)
        assert model.id == "gpt-4"
        assert model.name == "GPT-4"


class TestAPIClient:
    """测试 API 客户端"""

    def test_client_initialization(self):
        client = APIClient("http://test.com/v1", "test_key")
        assert client.base_url == "http://test.com/v1"
        assert client.api_key == "test_key"

    def test_client_set_base_url(self):
        client = APIClient("http://initial.com")
        client.set_base_url("http://updated.com/v1")
        assert client.base_url == "http://updated.com/v1"

    def test_client_set_api_key(self):
        client = APIClient("http://test.com")
        client.set_api_key("new_key")
        assert client.api_key == "new_key"

    def test_fetch_models_error_handling(self):
        client = APIClient("http://invalid-url-test-12345.com", "")
        models, error = client.fetch_models()
        assert models == []
        assert error is not None

    def test_chat_completion_error_handling(self):
        client = APIClient("http://invalid-url-test-12345.com", "")
        response, error = client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="test"
        )
        assert response is None
        assert error is not None
