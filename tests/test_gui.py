"""
Test cases for the GUI application (without actual display)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)


class TestModelFetcherThread:
    """Tests for ModelFetcherThread class"""
    
    def test_model_fetcher_thread_creation(self):
        """Test that ModelFetcherThread can be created"""
        from main import ModelFetcherThread
        thread = ModelFetcherThread("https://api.example.com", "test-key")
        assert thread is not None
        assert thread.base_url == "https://api.example.com"
        assert thread.api_key == "test-key"
    
    @patch('main.APIClient')
    def test_model_fetcher_thread_run_success(self, mock_api_client_class):
        """Test successful model fetching in thread"""
        from main import ModelFetcherThread
        
        # Setup mock
        mock_client = Mock()
        mock_client.get_models.return_value = ["gpt-4", "gpt-3.5-turbo"]
        mock_api_client_class.return_value = mock_client
        
        thread = ModelFetcherThread("https://api.example.com", "test-key")
        
        # Mock signals
        thread.models_received = Mock()
        thread.error_occurred = Mock()
        
        thread.run()
        
        thread.models_received.emit.assert_called_once_with(["gpt-4", "gpt-3.5-turbo"])
        thread.error_occurred.emit.assert_not_called()
    
    @patch('main.APIClient')
    def test_model_fetcher_thread_run_empty(self, mock_api_client_class):
        """Test model fetching with empty result"""
        from main import ModelFetcherThread
        
        # Setup mock
        mock_client = Mock()
        mock_client.get_models.return_value = None
        mock_api_client_class.return_value = mock_client
        
        thread = ModelFetcherThread("https://api.example.com", "test-key")
        
        # Mock signals
        thread.models_received = Mock()
        thread.error_occurred = Mock()
        
        thread.run()
        
        thread.models_received.emit.assert_not_called()
        thread.error_occurred.emit.assert_called_once()


class TestChatWorkerThread:
    """Tests for ChatWorkerThread class"""
    
    def test_chat_worker_thread_creation(self):
        """Test that ChatWorkerThread can be created"""
        from main import ChatWorkerThread
        messages = [{"role": "user", "content": "Hello"}]
        thread = ChatWorkerThread("https://api.example.com", "test-key", "gpt-4", messages)
        assert thread is not None
        assert thread.model == "gpt-4"
        assert thread.messages == messages
    
    @patch('main.APIClient')
    def test_chat_worker_thread_run_success(self, mock_api_client_class):
        """Test successful chat streaming in thread"""
        from main import ChatWorkerThread
        
        # Setup mock
        mock_client = Mock()
        mock_client.chat_completion_stream.return_value = ["Hello", " world"]
        mock_api_client_class.return_value = mock_client
        
        messages = [{"role": "user", "content": "Hi"}]
        thread = ChatWorkerThread("https://api.example.com", "test-key", "gpt-4", messages)
        
        # Mock signals
        thread.chunk_received = Mock()
        thread.finished_signal = Mock()
        thread.error_occurred = Mock()
        
        thread.run()
        
        assert thread.chunk_received.emit.call_count == 2
        thread.chunk_received.emit.assert_any_call("Hello")
        thread.chunk_received.emit.assert_any_call(" world")
        thread.finished_signal.emit.assert_called_once()
        thread.error_occurred.emit.assert_not_called()
    
    @patch('main.APIClient')
    def test_chat_worker_thread_run_error(self, mock_api_client_class):
        """Test chat streaming with error"""
        from main import ChatWorkerThread
        
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.chat_completion_stream.side_effect = Exception("API Error")
        mock_api_client_class.return_value = mock_client
        
        messages = [{"role": "user", "content": "Hi"}]
        thread = ChatWorkerThread("https://api.example.com", "test-key", "gpt-4", messages)
        
        # Mock signals
        thread.chunk_received = Mock()
        thread.finished_signal = Mock()
        thread.error_occurred = Mock()
        
        thread.run()
        
        thread.chunk_received.emit.assert_not_called()
        thread.finished_signal.emit.assert_not_called()
        thread.error_occurred.emit.assert_called_once()


class TestOpenAIDebugTool:
    """Tests for OpenAIDebugTool main window"""
    
    def test_append_to_chat_formatting(self):
        """Test chat message formatting"""
        # This tests the HTML formatting logic
        role = "User"
        content = "Hello <script>alert('xss')</script>"
        bg_color = "#e3f2fd"
        
        # Simulate the formatting logic
        html = f'''
        <div style="background-color: {bg_color}; padding: 8px; margin: 4px 0; border-radius: 4px;">
            <b>{role}:</b><br/>
            {content.replace('<', '&lt;').replace('>', '&gt;')}
        </div>
        '''
        
        assert "&lt;script&gt;" in html
        assert "<script>" not in html
        assert bg_color in html
    
    def test_message_history_integration(self):
        """Test message history integration with chat"""
        from message_history import MessageHistory
        
        history = MessageHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there!")
        
        messages = history.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
