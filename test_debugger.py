"""
Tests for OpenAI Debugger Tool
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import json


class TestOpenAIDebugger:
    """Test cases for the OpenAI Debugger tool"""

    def test_config_initialization(self):
        """Test that configuration can be initialized with default values"""
        from debugger_core import DebuggerConfig
        
        config = DebuggerConfig()
        
        assert config.api_base == "http://localhost:11434/v1"
        assert config.api_key == ""
        assert config.model == ""
        assert config.max_tokens == 2048
        assert config.temperature == 0.7

    def test_config_custom_values(self):
        """Test configuration with custom values"""
        from debugger_core import DebuggerConfig
        
        config = DebuggerConfig(
            api_base="https://api.openai.com/v1",
            api_key="test-key-123",
            model="gpt-4",
            max_tokens=4096,
            temperature=0.5
        )
        
        assert config.api_base == "https://api.openai.com/v1"
        assert config.api_key == "test-key-123"
        assert config.model == "gpt-4"
        assert config.max_tokens == 4096
        assert config.temperature == 0.5

    def test_message_creation(self):
        """Test message object creation"""
        from debugger_core import Message
        
        msg = Message(role="user", content="Hello, world!")
        
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.timestamp is not None

    def test_message_to_dict(self):
        """Test converting message to dictionary"""
        from debugger_core import Message
        
        msg = Message(role="assistant", content="Hi there!")
        msg_dict = msg.to_dict()
        
        assert msg_dict["role"] == "assistant"
        assert msg_dict["content"] == "Hi there!"
        assert "timestamp" in msg_dict

    def test_conversation_history(self):
        """Test conversation history management"""
        from debugger_core import ConversationHistory
        
        history = ConversationHistory()
        
        # Add messages
        history.add_message("user", "First message")
        history.add_message("assistant", "First response")
        history.add_message("user", "Second message")
        
        assert len(history.messages) == 3
        assert history.messages[0].role == "user"
        assert history.messages[1].role == "assistant"
        assert history.messages[2].role == "user"

    def test_clear_conversation(self):
        """Test clearing conversation history"""
        from debugger_core import ConversationHistory
        
        history = ConversationHistory()
        history.add_message("user", "Test message")
        
        assert len(history.messages) == 1
        
        history.clear()
        
        assert len(history.messages) == 0

    def test_get_messages_for_api(self):
        """Test getting messages formatted for API call"""
        from debugger_core import ConversationHistory
        
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi")
        
        messages = history.get_messages_for_api()
        
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_api_client_initialization(self):
        """Test API client initialization"""
        from debugger_core import APIClient, DebuggerConfig
        
        config = DebuggerConfig(
            api_base="http://test.com/v1",
            api_key="test-key",
            model="test-model"
        )
        
        client = APIClient(config)
        
        assert client.config.api_base == "http://test.com/v1"
        assert client.config.api_key == "test-key"
        assert client.config.model == "test-model"

    @pytest.mark.asyncio
    async def test_streaming_response_parsing(self):
        """Test parsing streaming response chunks"""
        from debugger_core import ResponseParser
        
        parser = ResponseParser()
        
        # Simulate SSE data lines
        chunk1 = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        chunk2 = 'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        chunk3 = 'data: [DONE]\n\n'
        
        content1 = parser.parse_chunk(chunk1)
        content2 = parser.parse_chunk(chunk2)
        content3 = parser.parse_chunk(chunk3)
        
        assert content1 == "Hello"
        assert content2 == " world"
        assert content3 == ""

    def test_token_speed_calculation(self):
        """Test token speed calculation"""
        from debugger_core import TokenSpeedCalculator
        
        calculator = TokenSpeedCalculator()
        
        # Simulate receiving tokens over time
        calculator.start()
        
        # Add some tokens
        for i in range(10):
            calculator.add_token("token")
        
        speed = calculator.get_speed()
        
        # Speed should be greater than 0
        assert speed >= 0

    def test_log_entry_creation(self):
        """Test log entry creation"""
        from debugger_core import LogEntry, LogLevel
        
        entry = LogEntry(
            level=LogLevel.INFO,
            message="Test log message",
            details={"key": "value"}
        )
        
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test log message"
        assert entry.details == {"key": "value"}
        assert entry.timestamp is not None

    def test_log_manager_add_entry(self):
        """Test adding log entries"""
        from debugger_core import LogManager, LogLevel
        
        manager = LogManager()
        
        manager.add_log(LogLevel.INFO, "Info message")
        manager.add_log(LogLevel.ERROR, "Error message", {"error": "test"})
        
        logs = manager.get_logs()
        
        assert len(logs) == 2
        assert logs[0].level == LogLevel.INFO
        assert logs[1].level == LogLevel.ERROR

    def test_log_manager_filter_by_level(self):
        """Test filtering logs by level"""
        from debugger_core import LogManager, LogLevel
        
        manager = LogManager()
        
        manager.add_log(LogLevel.DEBUG, "Debug message")
        manager.add_log(LogLevel.INFO, "Info message")
        manager.add_log(LogLevel.ERROR, "Error message")
        
        error_logs = manager.get_logs(level_filter=LogLevel.ERROR)
        
        assert len(error_logs) == 1
        assert error_logs[0].level == LogLevel.ERROR

    def test_log_manager_clear(self):
        """Test clearing logs"""
        from debugger_core import LogManager, LogLevel
        
        manager = LogManager()
        manager.add_log(LogLevel.INFO, "Test message")
        
        assert len(manager.get_logs()) == 1
        
        manager.clear()
        
        assert len(manager.get_logs()) == 0

    def test_preset_test_cases(self):
        """Test preset test cases availability"""
        from debugger_core import get_preset_test_cases
        
        presets = get_preset_test_cases()
        
        assert len(presets) > 0
        
        # Check structure of preset
        preset = presets[0]
        assert "name" in preset
        assert "messages" in preset
        assert "params" in preset

    def test_request_builder_basic(self):
        """Test building basic request payload"""
        from debugger_core import RequestBuilder, ConversationHistory
        
        history = ConversationHistory()
        history.add_message("user", "Test message")
        
        builder = RequestBuilder(
            messages=history.get_messages_for_api(),
            model="gpt-4",
            max_tokens=1024,
            temperature=0.7,
            stream=True
        )
        
        payload = builder.build()
        
        assert "model" in payload
        assert "messages" in payload
        assert "max_tokens" in payload
        assert "temperature" in payload
        assert "stream" in payload
        assert payload["model"] == "gpt-4"
        assert payload["stream"] is True

    def test_request_builder_with_system_message(self):
        """Test building request with system message"""
        from debugger_core import RequestBuilder
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        builder = RequestBuilder(
            messages=messages,
            model="gpt-4",
            max_tokens=1024,
            temperature=0.7,
            stream=True
        )
        
        payload = builder.build()
        
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"

    def test_error_handling_invalid_json(self):
        """Test error handling for invalid JSON responses"""
        from debugger_core import ResponseParser
        
        parser = ResponseParser()
        
        # Invalid JSON should return empty string or handle gracefully
        invalid_chunk = 'data: {invalid json}\n\n'
        
        try:
            content = parser.parse_chunk(invalid_chunk)
            # Should not raise exception
            assert isinstance(content, str)
        except json.JSONDecodeError:
            pytest.fail("Should handle invalid JSON gracefully")

    def test_conversation_export(self):
        """Test exporting conversation to JSON"""
        from debugger_core import ConversationHistory
        import json
        
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there!")
        
        exported = history.export_to_json()
        
        # Should be valid JSON
        data = json.loads(exported)
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_conversation_import(self):
        """Test importing conversation from JSON"""
        from debugger_core import ConversationHistory
        import json
        
        json_data = '''
        {
            "messages": [
                {"role": "user", "content": "Imported message"},
                {"role": "assistant", "content": "Imported response"}
            ]
        }
        '''
        
        history = ConversationHistory.import_from_json(json_data)
        
        assert len(history.messages) == 2
        assert history.messages[0].content == "Imported message"


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations"""

    async def test_mock_streaming_call(self):
        """Test mocking a streaming API call"""
        from debugger_core import APIClient, DebuggerConfig, ConversationHistory
        
        config = DebuggerConfig(
            api_base="http://test.com/v1",
            api_key="test-key",
            model="test-model"
        )
        
        client = APIClient(config)
        
        # Mock the actual HTTP call
        mock_chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
            b'data: [DONE]\n\n'
        ]
        
        async def mock_stream(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk
        
        with patch.object(client, '_make_request', side_effect=mock_stream):
            full_response = ""
            async for chunk in client.send_message("Test message", ConversationHistory()):
                if chunk:
                    full_response += chunk
            
            assert full_response == "Hello world"

    async def test_timeout_handling(self):
        """Test timeout handling in API calls"""
        from debugger_core import APIClient, DebuggerConfig
        
        config = DebuggerConfig(
            api_base="http://test.com/v1",
            api_key="test-key",
            model="test-model"
        )
        
        client = APIClient(config, timeout=1.0)
        
        assert client.timeout == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
