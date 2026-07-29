"""
Test cases for OpenAI Debug Tool
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json
from datetime import datetime


class TestOpenAIClient:
    """Tests for the OpenAI API client module"""
    
    def test_client_initialization(self):
        """Test that client can be initialized with required parameters"""
        from openai_debug_tool import OpenAIClient
        
        client = OpenAIClient(
            base_url="http://localhost:8000",
            api_key="test-key",
            model="gpt-3.5-turbo"
        )
        
        assert client.base_url == "http://localhost:8000"
        assert client.api_key == "test-key"
        assert client.model == "gpt-3.5-turbo"
    
    def test_client_default_values(self):
        """Test client default values"""
        from openai_debug_tool import OpenAIClient
        
        client = OpenAIClient()
        
        assert client.base_url == "https://api.openai.com/v1"
        assert client.api_key == ""
        assert client.model == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing available models"""
        from openai_debug_tool import OpenAIClient, APIError
        
        client = OpenAIClient(
            base_url="http://test-server",
            api_key="test-key",
            model="test-model"
        )
        
        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "data": [
                {"id": "gpt-3.5-turbo"},
                {"id": "gpt-4"},
                {"id": "claude-3"}
            ]
        })
        
        # Create a proper async mock for the httpx client
        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.aclose = AsyncMock()
        
        # Set _client directly (bypass __aenter__ which tries to create real httpx client)
        client._client = mock_http_client
        
        models = await client.list_models()
        
        assert len(models) == 3
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models
        assert "claude-3" in models
        
        # Verify the get method was called with correct URL
        mock_http_client.get.assert_called_once_with("http://test-server/models")
    
    @pytest.mark.asyncio
    async def test_chat_completion_streaming(self):
        """Test streaming chat completion"""
        from openai_debug_tool import OpenAIClient
        
        client = OpenAIClient(
            base_url="http://test-server",
            api_key="test-key",
            model="test-model"
        )
        
        # Mock response chunks
        mock_chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" world"}}]}',
            b'data: [DONE]'
        ]
        
        # Create async iterator for chunks
        async def async_iter_lines():
            for chunk in mock_chunks:
                yield chunk.decode('utf-8')
        
        mock_response = Mock()
        mock_response.aiter_lines = async_iter_lines
        mock_response.raise_for_status = Mock()
        
        mock_stream_cm = Mock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        
        mock_client = AsyncMock()
        mock_client.stream = Mock(return_value=mock_stream_cm)
        mock_client.aclose = AsyncMock()
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            async with client:
                messages = [{"role": "user", "content": "Hi"}]
                chunks = []
                
                async for chunk in client.chat_completion_stream(messages):
                    chunks.append(chunk)
                
                assert len(chunks) == 2
                assert chunks[0] == "Hello"
                assert chunks[1] == " world"
    
    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming(self):
        """Test non-streaming chat completion"""
        from openai_debug_tool import OpenAIClient
        
        client = OpenAIClient(
            base_url="http://test-server",
            api_key="test-key",
            model="test-model"
        )
        
        mock_response_data = {
            "choices": [{
                "message": {"content": "Hello world"}
            }]
        }
        
        mock_response = Mock()
        mock_response.json = Mock(return_value=mock_response_data)
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            async with client:
                messages = [{"role": "user", "content": "Hi"}]
                result = await client.chat_completion(messages, stream=False)
                
                assert result == "Hello world"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in API calls"""
        from openai_debug_tool import OpenAIClient, APIError
        
        client = OpenAIClient(
            base_url="http://test-server",
            api_key="test-key",
            model="test-model"
        )
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection error"))
        mock_client.aclose = AsyncMock()
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            async with client:
                messages = [{"role": "user", "content": "Hi"}]
                
                with pytest.raises(APIError):
                    await client.chat_completion(messages, stream=False)


class TestConversationManager:
    """Tests for conversation management"""
    
    def test_add_message(self):
        """Test adding messages to conversation"""
        from openai_debug_tool import ConversationManager
        
        manager = ConversationManager()
        manager.add_message("user", "Hello")
        
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "user"
        assert manager.messages[0]["content"] == "Hello"
    
    def test_add_assistant_message(self):
        """Test adding assistant messages"""
        from openai_debug_tool import ConversationManager
        
        manager = ConversationManager()
        manager.add_message("assistant", "Hi there!")
        
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "assistant"
    
    def test_clear_conversation(self):
        """Test clearing conversation history"""
        from openai_debug_tool import ConversationManager
        
        manager = ConversationManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        manager.clear()
        
        assert len(manager.messages) == 0
    
    def test_get_messages(self):
        """Test getting conversation messages"""
        from openai_debug_tool import ConversationManager
        
        manager = ConversationManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        
        messages = manager.get_messages()
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


class TestTokenCounter:
    """Tests for token counting utilities"""
    
    def test_count_tokens_simple(self):
        """Test basic token counting"""
        from openai_debug_tool import count_tokens
        
        text = "Hello world"
        count = count_tokens(text)
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_tokens_empty(self):
        """Test token counting with empty string"""
        from openai_debug_tool import count_tokens
        
        count = count_tokens("")
        assert count == 0
    
    def test_count_tokens_messages(self):
        """Test token counting for message list"""
        from openai_debug_tool import count_tokens_messages
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        count = count_tokens_messages(messages)
        assert count > 0
        assert isinstance(count, int)


class TestSpeedCalculator:
    """Tests for inference speed calculation"""
    
    def test_calculate_speed(self):
        """Test tokens per second calculation"""
        from openai_debug_tool import SpeedCalculator
        
        calc = SpeedCalculator()
        calc.start()
        
        # Simulate receiving tokens
        for i in range(10):
            calc.add_token()
        
        calc.stop()
        speed = calc.get_speed()
        
        assert speed >= 0
        assert isinstance(speed, float)
    
    def test_reset_calculator(self):
        """Test resetting speed calculator"""
        from openai_debug_tool import SpeedCalculator
        
        calc = SpeedCalculator()
        calc.start()
        calc.add_token()
        calc.reset()
        
        assert calc.token_count == 0
        assert calc.start_time is None


class TestLogEntry:
    """Tests for log entry creation"""
    
    def test_create_request_log(self):
        """Test creating request log entry"""
        from openai_debug_tool import LogEntry, LogLevel
        
        entry = LogEntry.create_request_log(
            url="http://test.com",
            method="POST",
            headers={"Content-Type": "application/json"},
            body={"messages": []}
        )
        
        assert entry.level == LogLevel.INFO
        assert "Request" in entry.message
        assert entry.timestamp is not None
    
    def test_create_response_log(self):
        """Test creating response log entry"""
        from openai_debug_tool import LogEntry, LogLevel
        
        entry = LogEntry.create_response_log(
            status_code=200,
            body={"choices": []}
        )
        
        assert entry.level == LogLevel.INFO
        assert "Response" in entry.message
    
    def test_create_error_log(self):
        """Test creating error log entry"""
        from openai_debug_tool import LogEntry, LogLevel
        
        entry = LogEntry.create_error_log("Connection failed")
        
        assert entry.level == LogLevel.ERROR
        assert "Error" in entry.message


class TestConfigManager:
    """Tests for configuration management"""
    
    def test_save_config(self, tmp_path):
        """Test saving configuration"""
        from openai_debug_tool import ConfigManager
        
        config_file = tmp_path / "config.json"
        manager = ConfigManager(str(config_file))
        
        manager.save_config({
            "base_url": "http://test.com",
            "api_key": "test-key",
            "model": "gpt-4"
        })
        
        assert config_file.exists()
        
        with open(config_file) as f:
            saved_config = json.load(f)
        
        assert saved_config["base_url"] == "http://test.com"
        assert saved_config["api_key"] == "test-key"
        assert saved_config["model"] == "gpt-4"
    
    def test_load_config(self, tmp_path):
        """Test loading configuration"""
        from openai_debug_tool import ConfigManager
        
        config_file = tmp_path / "config.json"
        
        config_data = {
            "base_url": "http://loaded.com",
            "api_key": "loaded-key",
            "model": "gpt-3.5-turbo"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        manager = ConfigManager(str(config_file))
        loaded_config = manager.load_config()
        
        assert loaded_config["base_url"] == "http://loaded.com"
        assert loaded_config["api_key"] == "loaded-key"
    
    def test_load_nonexistent_config(self, tmp_path):
        """Test loading non-existent config file"""
        from openai_debug_tool import ConfigManager
        
        config_file = tmp_path / "nonexistent.json"
        manager = ConfigManager(str(config_file))
        
        config = manager.load_config()
        
        assert config == {}


class TestUIComponents:
    """Tests for UI component utilities (headless testing)"""
    
    def test_format_speed_display(self):
        """Test speed display formatting"""
        from openai_debug_tool import format_speed
        
        speed = 15.678
        formatted = format_speed(speed)
        
        assert "tokens/s" in formatted
        assert "15.68" in formatted or "15.7" in formatted or "16" in formatted
    
    def test_format_timestamp(self):
        """Test timestamp formatting"""
        from openai_debug_tool import format_timestamp
        
        now = datetime.now()
        formatted = format_timestamp(now)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
    
    def test_truncate_text(self):
        """Test text truncation"""
        from openai_debug_tool import truncate_text
        
        long_text = "A" * 100
        truncated = truncate_text(long_text, max_length=50)
        
        assert len(truncated) <= 53  # 50 + "..."
        assert truncated.endswith("...")
        
        short_text = "Short"
        truncated_short = truncate_text(short_text, max_length=50)
        assert truncated_short == short_text


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test complete conversation flow"""
        from openai_debug_tool import (
            OpenAIClient, 
            ConversationManager, 
            SpeedCalculator
        )
        
        # Setup
        client = OpenAIClient(
            base_url="http://test-server",
            api_key="test-key",
            model="test-model"
        )
        conv_manager = ConversationManager()
        speed_calc = SpeedCalculator()
        
        # Add user message
        conv_manager.add_message("user", "Hello")
        
        # Mock API response
        mock_response_data = {
            "choices": [{
                "message": {"content": "Hi! How can I help?"}
            }]
        }
        
        mock_response = Mock()
        mock_response.json = Mock(return_value=mock_response_data)
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            async with client:
                # Get response
                messages = conv_manager.get_messages()
                response = await client.chat_completion(messages, stream=False)
                
                # Add assistant response
                conv_manager.add_message("assistant", response)
                
                # Verify
                assert len(conv_manager.messages) == 2
                assert conv_manager.messages[1]["content"] == "Hi! How can I help?"
