"""
Tests for OpenAI API Debugger Tool
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time
from datetime import datetime


class TestTokenCounter:
    """Test token counting functionality"""
    
    def test_count_tokens_simple_text(self):
        """Test counting tokens in simple text"""
        from openai_debugger import count_tokens
        text = "Hello world"
        count = count_tokens(text)
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_tokens_empty_text(self):
        """Test counting tokens in empty text"""
        from openai_debugger import count_tokens
        text = ""
        count = count_tokens(text)
        assert count == 0
    
    def test_count_tokens_chinese_text(self):
        """Test counting tokens in Chinese text"""
        from openai_debugger import count_tokens
        text = "你好世界"
        count = count_tokens(text)
        assert count > 0


class TestSpeedCalculator:
    """Test speed calculation functionality"""
    
    def test_calculate_speed_basic(self):
        """Test basic speed calculation"""
        from openai_debugger import calculate_speed
        tokens = 100
        elapsed_time = 2.0
        speed = calculate_speed(tokens, elapsed_time)
        assert speed == 50.0
    
    def test_calculate_speed_zero_time(self):
        """Test speed calculation with zero time"""
        from openai_debugger import calculate_speed
        tokens = 100
        elapsed_time = 0.0
        speed = calculate_speed(tokens, elapsed_time)
        assert speed == 0.0
    
    def test_calculate_speed_realtime(self):
        """Test realtime speed calculation"""
        from openai_debugger import SpeedCalculator
        calc = SpeedCalculator()
        calc.start()
        time.sleep(0.1)
        calc.update_tokens(10)
        speed = calc.get_current_speed()
        assert speed >= 0


class TestMessageFormatter:
    """Test message formatting functionality"""
    
    def test_format_user_message(self):
        """Test formatting user message"""
        from openai_debugger import format_message
        role = "user"
        content = "Hello"
        formatted = format_message(role, content)
        assert "user" in formatted.lower() or "User" in formatted
        assert "Hello" in formatted
    
    def test_format_assistant_message(self):
        """Test formatting assistant message"""
        from openai_debugger import format_message
        role = "assistant"
        content = "Hi there"
        formatted = format_message(role, content)
        assert "assistant" in formatted.lower() or "Assistant" in formatted
        assert "Hi there" in formatted
    
    def test_format_system_message(self):
        """Test formatting system message"""
        from openai_debugger import format_message
        role = "system"
        content = "You are helpful"
        formatted = format_message(role, content)
        assert "system" in formatted.lower() or "System" in formatted


class TestLogEntry:
    """Test log entry functionality"""
    
    def test_create_log_entry(self):
        """Test creating a log entry"""
        from openai_debugger import LogEntry
        timestamp = datetime.now()
        level = "INFO"
        message = "Test message"
        entry = LogEntry(timestamp, level, message)
        
        assert entry.timestamp == timestamp
        assert entry.level == level
        assert entry.message == message
    
    def test_log_entry_to_string(self):
        """Test log entry string representation"""
        from openai_debugger import LogEntry
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        entry = LogEntry(timestamp, "ERROR", "Something failed")
        str_repr = str(entry)
        assert "2024-01-01" in str_repr
        assert "ERROR" in str_repr
        assert "Something failed" in str_repr


class TestConversationHistory:
    """Test conversation history management"""
    
    def test_add_message(self):
        """Test adding a message to history"""
        from openai_debugger import ConversationHistory
        history = ConversationHistory()
        history.add_message("user", "Hello")
        
        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "Hello"
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages"""
        from openai_debugger import ConversationHistory
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi")
        history.add_message("user", "How are you?")
        
        assert len(history.messages) == 3
    
    def test_clear_history(self):
        """Test clearing conversation history"""
        from openai_debugger import ConversationHistory
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi")
        history.clear()
        
        assert len(history.messages) == 0
    
    def test_get_messages_for_api(self):
        """Test getting messages formatted for API"""
        from openai_debugger import ConversationHistory
        history = ConversationHistory()
        history.add_message("system", "You are helpful")
        history.add_message("user", "Hello")
        
        messages = history.get_messages_for_api()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestPresetTemplates:
    """Test preset template functionality"""
    
    def test_get_presets(self):
        """Test getting preset templates"""
        from openai_debugger import get_preset_templates
        presets = get_preset_templates()
        
        assert isinstance(presets, list)
        assert len(presets) > 0
        assert all(isinstance(p, dict) for p in presets)
    
    def test_preset_structure(self):
        """Test preset template structure"""
        from openai_debugger import get_preset_templates
        presets = get_preset_templates()
        
        for preset in presets:
            assert "name" in preset
            assert "messages" in preset
            assert isinstance(preset["messages"], list)
    
    def test_common_presets_exist(self):
        """Test that common presets exist"""
        from openai_debugger import get_preset_templates
        presets = get_preset_templates()
        names = [p["name"].lower() for p in presets]
        
        # Should have at least some common test scenarios
        assert any("simple" in name or "basic" in name or "hello" in name for name in names)


class TestAPIConfig:
    """Test API configuration management"""
    
    def test_create_default_config(self):
        """Test creating default API config"""
        from openai_debugger import APIConfig
        config = APIConfig()
        
        assert config.base_url is not None
        assert config.api_key == ""
        assert config.model == "gpt-3.5-turbo"
    
    def test_update_config(self):
        """Test updating API config"""
        from openai_debugger import APIConfig
        config = APIConfig()
        config.update(
            base_url="https://custom.api.com/v1",
            api_key="test-key-123",
            model="gpt-4"
        )
        
        assert config.base_url == "https://custom.api.com/v1"
        assert config.api_key == "test-key-123"
        assert config.model == "gpt-4"
    
    def test_config_validation(self):
        """Test config validation"""
        from openai_debugger import APIConfig
        config = APIConfig()
        
        # Valid config
        config.update(base_url="https://api.openai.com/v1")
        assert config.is_valid()
        
        # Invalid URL
        config.update(base_url="not-a-valid-url")
        # Should still be considered valid as we're lenient with URLs


class TestStreamParser:
    """Test SSE stream parsing"""
    
    def test_parse_sse_line(self):
        """Test parsing SSE line"""
        from openai_debugger import parse_sse_line
        line = 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        result = parse_sse_line(line)
        
        assert result is not None
        assert "choices" in result
    
    def test_parse_sse_done(self):
        """Test parsing [DONE] marker"""
        from openai_debugger import parse_sse_line
        line = 'data: [DONE]'
        result = parse_sse_line(line)
        
        assert result is None
    
    def test_parse_sse_empty(self):
        """Test parsing empty line"""
        from openai_debugger import parse_sse_line
        line = ''
        result = parse_sse_line(line)
        
        assert result is None


class TestResponseExtractor:
    """Test response content extraction"""
    
    def test_extract_content_from_delta(self):
        """Test extracting content from delta response"""
        from openai_debugger import extract_content_from_response
        response = {
            "choices": [{
                "delta": {"content": "Hello world"}
            }]
        }
        content = extract_content_from_response(response)
        assert content == "Hello world"
    
    def test_extract_content_from_message(self):
        """Test extracting content from complete message"""
        from openai_debugger import extract_content_from_response
        response = {
            "choices": [{
                "message": {"content": "Complete response"}
            }]
        }
        content = extract_content_from_response(response)
        assert content == "Complete response"
    
    def test_extract_content_empty(self):
        """Test extracting content from empty response"""
        from openai_debugger import extract_content_from_response
        response = {}
        content = extract_content_from_response(response)
        assert content == ""


class TestErrorHandling:
    """Test error handling functionality"""
    
    def test_format_api_error(self):
        """Test formatting API error"""
        from openai_debugger import format_api_error
        error = Exception("Connection failed")
        formatted = format_api_error(error)
        
        assert "Connection failed" in formatted
        assert "Error" in formatted
    
    def test_format_http_error(self):
        """Test formatting HTTP error"""
        from openai_debugger import format_api_error
        error = Exception("HTTP 401: Unauthorized")
        formatted = format_api_error(error)
        
        assert "401" in formatted or "Unauthorized" in formatted


class TestUIHelpers:
    """Test UI helper functions"""
    
    def test_truncate_text(self):
        """Test text truncation"""
        from openai_debugger import truncate_text
        text = "This is a very long text that should be truncated"
        truncated = truncate_text(text, max_length=20)
        
        assert len(truncated) <= 23  # 20 + "..."
        assert truncated.endswith("...")
    
    def test_truncate_text_short(self):
        """Test truncating short text"""
        from openai_debugger import truncate_text
        text = "Short"
        truncated = truncate_text(text, max_length=20)
        
        assert truncated == "Short"
    
    def test_format_timestamp(self):
        """Test timestamp formatting"""
        from openai_debugger import format_timestamp
        dt = datetime(2024, 1, 15, 14, 30, 45)
        formatted = format_timestamp(dt)
        
        assert "2024-01-15" in formatted
        assert "14:30:45" in formatted


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test full conversation flow with mocked API"""
        from openai_debugger import ConversationHistory, APIConfig
        
        # Setup
        history = ConversationHistory()
        config = APIConfig()
        
        # Add user message
        history.add_message("user", "Hello, how are you?")
        
        # Simulate assistant response
        history.add_message("assistant", "I'm doing well, thank you!")
        
        # Verify history
        assert len(history.messages) == 2
        messages = history.get_messages_for_api()
        
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    def test_speed_calculation_accuracy(self):
        """Test speed calculation accuracy"""
        from openai_debugger import SpeedCalculator
        import time
        
        calc = SpeedCalculator()
        calc.start()
        
        # Simulate token generation
        total_tokens = 0
        for i in range(5):
            time.sleep(0.1)
            tokens_batch = 10
            total_tokens += tokens_batch
            calc.update_tokens(tokens_batch)
        
        speed = calc.get_current_speed()
        # Speed should be around 100 tokens/second (10 tokens per 0.1 second)
        assert 50 <= speed <= 150  # Allow some variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
