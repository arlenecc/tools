"""
Test cases for OpenAI Debug Tool
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)


class TestConfigManager:
    """Tests for ConfigManager class"""
    
    def test_config_manager_creation(self):
        """Test that ConfigManager can be created"""
        from config_manager import ConfigManager
        config = ConfigManager()
        assert config is not None
    
    def test_config_manager_default_values(self):
        """Test ConfigManager has correct default values"""
        from config_manager import ConfigManager
        config = ConfigManager()
        assert config.base_url == ""
        assert config.api_key == ""
        assert config.model == ""
    
    def test_config_manager_set_values(self):
        """Test setting values in ConfigManager"""
        from config_manager import ConfigManager
        config = ConfigManager()
        config.base_url = "https://api.example.com"
        config.api_key = "test-key"
        config.model = "gpt-4"
        
        assert config.base_url == "https://api.example.com"
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"
    
    def test_config_manager_save_load(self, tmp_path):
        """Test saving and loading configuration"""
        from config_manager import ConfigManager
        config_file = tmp_path / "config.json"
        
        config = ConfigManager(str(config_file))
        config.base_url = "https://api.example.com"
        config.api_key = "test-key"
        config.model = "gpt-4"
        config.save()
        
        # Load into new instance
        config2 = ConfigManager(str(config_file))
        config2.load()
        
        assert config2.base_url == "https://api.example.com"
        assert config2.api_key == "test-key"
        assert config2.model == "gpt-4"


class TestAPIClient:
    """Tests for APIClient class"""
    
    def test_api_client_creation(self):
        """Test that APIClient can be created"""
        from api_client import APIClient
        client = APIClient("https://api.example.com", "test-key")
        assert client is not None
        assert client.base_url == "https://api.example.com"
        assert client.api_key == "test-key"
    
    def test_api_client_get_models_success(self):
        """Test successful model list retrieval"""
        from api_client import APIClient
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4"},
                {"id": "gpt-3.5-turbo"},
                {"id": "claude-3"}
            ]
        }
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            client = APIClient("https://api.example.com", "test-key")
            models = client.get_models()
            
            assert len(models) == 3
            assert "gpt-4" in models
            assert "gpt-3.5-turbo" in models
            assert "claude-3" in models
            mock_get.assert_called_once()
    
    def test_api_client_get_models_empty(self):
        """Test model list retrieval with empty response"""
        from api_client import APIClient
        
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            client = APIClient("https://api.example.com", "test-key")
            models = client.get_models()
            
            assert len(models) == 0
    
    def test_api_client_get_models_error(self):
        """Test model list retrieval with error"""
        from api_client import APIClient
        
        with patch('requests.get', side_effect=Exception("Connection error")):
            client = APIClient("https://api.example.com", "test-key")
            models = client.get_models()
            
            assert models is None
    
    def test_api_client_chat_completion_stream(self):
        """Test streaming chat completion"""
        from api_client import APIClient
        
        # Mock streaming response - lines should be bytes
        mock_line1 = b'data: {"choices":[{"delta":{"content":"Hello"}}]}'
        mock_line2 = b'data: {"choices":[{"delta":{"content":" world"}}]}'
        mock_line3 = b'data: [DONE]'
        
        mock_response = Mock()
        mock_response.iter_lines.return_value = [mock_line1, mock_line2, mock_line3]
        mock_response.raise_for_status = Mock()
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            client = APIClient("https://api.example.com", "test-key")
            
            messages = [{"role": "user", "content": "Hi"}]
            chunks = []
            
            for chunk in client.chat_completion_stream("gpt-4", messages):
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0] == "Hello"
            assert chunks[1] == " world"
    
    def test_api_client_chat_completion_stream_error(self):
        """Test streaming chat completion with error"""
        from api_client import APIClient
        
        with patch('requests.post', side_effect=Exception("API error")):
            client = APIClient("https://api.example.com", "test-key")
            
            messages = [{"role": "user", "content": "Hi"}]
            chunks = list(client.chat_completion_stream("gpt-4", messages))
            
            assert len(chunks) == 0


class TestMessageHistory:
    """Tests for MessageHistory class"""
    
    def test_message_history_creation(self):
        """Test that MessageHistory can be created"""
        from message_history import MessageHistory
        history = MessageHistory()
        assert history is not None
        assert len(history.messages) == 0
    
    def test_message_history_add_message(self):
        """Test adding messages to history"""
        from message_history import MessageHistory
        history = MessageHistory()
        
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there!")
        
        assert len(history.messages) == 2
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "Hello"
        assert history.messages[1]["role"] == "assistant"
        assert history.messages[1]["content"] == "Hi there!"
    
    def test_message_history_clear(self):
        """Test clearing message history"""
        from message_history import MessageHistory
        history = MessageHistory()
        
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there!")
        assert len(history.messages) == 2
        
        history.clear()
        assert len(history.messages) == 0
    
    def test_message_history_get_messages(self):
        """Test getting all messages"""
        from message_history import MessageHistory
        history = MessageHistory()
        
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi")
        
        messages = history.get_messages()
        assert len(messages) == 2
        assert messages == history.messages


class TestSpeedCalculator:
    """Tests for SpeedCalculator class"""
    
    def test_speed_calculator_creation(self):
        """Test that SpeedCalculator can be created"""
        from speed_calculator import SpeedCalculator
        calc = SpeedCalculator()
        assert calc is not None
    
    def test_speed_calculator_reset(self):
        """Test resetting the calculator"""
        from speed_calculator import SpeedCalculator
        calc = SpeedCalculator()
        
        calc.start_time = 100.0
        calc.total_tokens = 100
        
        calc.reset()
        
        assert calc.start_time is None
        assert calc.total_tokens == 0
    
    def test_speed_calculator_start(self):
        """Test starting the calculator"""
        from speed_calculator import SpeedCalculator
        calc = SpeedCalculator()
        
        calc.start()
        assert calc.start_time is not None
    
    def test_speed_calculator_add_tokens(self):
        """Test adding tokens"""
        from speed_calculator import SpeedCalculator
        calc = SpeedCalculator()
        
        calc.add_tokens(10)
        assert calc.total_tokens == 10
        
        calc.add_tokens(5)
        assert calc.total_tokens == 15
    
    def test_speed_calculator_get_speed(self):
        """Test calculating speed"""
        from speed_calculator import SpeedCalculator
        
        calc = SpeedCalculator()
        calc.start_time = time.time() - 1.0  # 1 second ago
        calc.total_tokens = 100
        
        speed = calc.get_speed()
        assert speed > 0  # Should be positive (around 100 tokens/sec)


class TestLogEntry:
    """Tests for LogEntry class"""
    
    def test_log_entry_creation(self):
        """Test that LogEntry can be created"""
        from log_entry import LogEntry
        entry = LogEntry("INFO", "Test message")
        assert entry is not None
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.timestamp is not None
    
    def test_log_entry_to_string(self):
        """Test converting log entry to string"""
        from log_entry import LogEntry
        entry = LogEntry("ERROR", "Something went wrong")
        
        str_repr = str(entry)
        assert "ERROR" in str_repr
        assert "Something went wrong" in str_repr


class TestLogger:
    """Tests for Logger class"""
    
    def test_logger_creation(self):
        """Test that Logger can be created"""
        from logger import Logger
        logger = Logger()
        assert logger is not None
        assert len(logger.entries) == 0
    
    def test_logger_add_entry(self):
        """Test adding log entries"""
        from logger import Logger
        logger = Logger()
        
        logger.info("Test info")
        logger.error("Test error")
        logger.warning("Test warning")
        
        assert len(logger.entries) == 3
        assert logger.entries[0].level == "INFO"
        assert logger.entries[1].level == "ERROR"
        assert logger.entries[2].level == "WARNING"
    
    def test_logger_clear(self):
        """Test clearing logs"""
        from logger import Logger
        logger = Logger()
        
        logger.info("Test")
        assert len(logger.entries) == 1
        
        logger.clear()
        assert len(logger.entries) == 0
    
    def test_logger_get_entries(self):
        """Test getting all entries"""
        from logger import Logger
        logger = Logger()
        
        logger.info("Test 1")
        logger.info("Test 2")
        
        entries = logger.get_entries()
        assert len(entries) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
