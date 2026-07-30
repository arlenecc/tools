"""测试 GUI 组件（不含实际 GUI 渲染）"""
import pytest
import sys
import os

from src.config_manager import Config, ConfigManager
from src.message_history import MessageHistory
from src.speed_calculator import SpeedCalculator
from src.logger import Logger


class TestConfigPanel:
    """测试配置面板相关功能"""

    def test_config_default_values(self):
        config = Config()
        assert config.base_url == "http://localhost:11434/v1"
        assert config.api_key == ""

    def test_config_save_load(self, tmp_path):
        config_file = tmp_path / "test_config.json"
        manager = ConfigManager(str(config_file))
        config = Config(base_url="http://test.com/v1", api_key="test_key", model="gpt-4")
        manager.save(config)
        
        manager2 = ConfigManager(str(config_file))
        loaded = manager2.load()
        assert loaded.base_url == "http://test.com/v1"
        assert loaded.api_key == "test_key"
        assert loaded.model == "gpt-4"


class TestChatArea:
    """测试对话区域相关功能"""

    def test_message_history_add_messages(self):
        history = MessageHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi there")
        
        assert len(history) == 2
        assert history.get_messages()[0].role == "user"
        assert history.get_messages()[1].role == "assistant"

    def test_message_history_api_format(self):
        history = MessageHistory()
        history.add_user_message("Test message")
        
        api_msgs = history.get_api_messages()
        assert len(api_msgs) == 1
        assert api_msgs[0] == {"role": "user", "content": "Test message"}

    def test_message_history_clear(self):
        history = MessageHistory()
        history.add_user_message("Msg 1")
        history.add_user_message("Msg 2")
        history.clear()
        assert len(history) == 0


class TestLogArea:
    """测试日志区域相关功能"""

    def test_logger_add_entries(self):
        logger = Logger()
        logger.info("Info message")
        logger.error("Error message")
        logger.debug("Debug message")
        
        entries = logger.get_entries()
        assert len(entries) == 3
        assert entries[0].level == "INFO"
        assert entries[1].level == "ERROR"

    def test_logger_clear(self):
        logger = Logger()
        logger.info("Test")
        logger.clear()
        assert len(logger.get_entries()) == 0


class TestSpeedDisplay:
    """测试速度显示相关功能"""

    def test_speed_calculator_basic(self):
        calc = SpeedCalculator()
        calc.start()
        import time
        time.sleep(0.05)
        calc.add_token()
        time.sleep(0.05)
        calc.add_token()
        
        stats = calc.stop()
        assert stats.total_tokens == 2
        assert stats.elapsed_time >= 0.1

    def test_speed_calculator_reset(self):
        calc = SpeedCalculator()
        calc.start()
        calc.add_token()
        calc.reset()
        
        stats = calc.get_current_stats()
        assert stats.total_tokens == 0
        assert stats.tokens_per_second == 0.0


class TestMainWindow:
    """测试主窗口功能（简化版）"""

    def test_main_window_module_exists(self):
        """测试主窗口模块文件存在"""
        import os
        assert os.path.exists("src/main.py")

    def test_worker_threads_module_exists(self):
        """测试工作线程模块文件存在"""
        import os
        assert os.path.exists("src/main.py")
        # 检查文件内容包含相关类定义
        with open("src/main.py", "r") as f:
            content = f.read()
            assert "class ModelFetchWorker" in content
            assert "class ChatWorker" in content
            assert "class MainWindow" in content
