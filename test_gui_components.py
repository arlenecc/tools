"""
Test GUI components and button functionality
"""
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock


class TestGUIComponents(unittest.TestCase):
    """Test GUI component initialization and button bindings"""
    
    def test_import_gui_module(self):
        """Test that the GUI module can be imported"""
        try:
            # Mock tkinter before importing
            sys.modules['tkinter'] = MagicMock()
            sys.modules['tkinter.ttk'] = MagicMock()
            sys.modules['tkinter.scrolledtext'] = MagicMock()
            
            from openai_debug_tool import OpenAIDebugToolGUI
            
            # Verify class exists
            self.assertTrue(hasattr(OpenAIDebugToolGUI, '__init__'))
            self.assertTrue(hasattr(OpenAIDebugToolGUI, 'setup_gui'))
            self.assertTrue(hasattr(OpenAIDebugToolGUI, 'send_message'))
            self.assertTrue(hasattr(OpenAIDebugToolGUI, 'clear_conversation'))
            self.assertTrue(hasattr(OpenAIDebugToolGUI, 'stop_generation'))
            self.assertTrue(hasattr(OpenAIDebugToolGUI, 'save_configuration'))
            
        except ImportError as e:
            self.fail(f"Failed to import GUI module: {e}")
    
    def test_handler_methods_exist(self):
        """Test that all event handler methods exist"""
        sys.modules['tkinter'] = MagicMock()
        sys.modules['tkinter.ttk'] = MagicMock()
        sys.modules['tkinter.scrolledtext'] = MagicMock()
        
        from openai_debug_tool import OpenAIDebugToolGUI
        
        app = OpenAIDebugToolGUI.__new__(OpenAIDebugToolGUI)
        
        # Check all required handler methods exist
        self.assertTrue(hasattr(app, '_on_enter_key'))
        self.assertTrue(hasattr(app, '_on_shift_enter'))
        self.assertTrue(callable(getattr(app, '_on_enter_key')))
        self.assertTrue(callable(getattr(app, '_on_shift_enter')))
    
    def test_on_enter_key_logic(self):
        """Test Enter key handler logic"""
        sys.modules['tkinter'] = MagicMock()
        sys.modules['tkinter.ttk'] = MagicMock()
        sys.modules['tkinter.scrolledtext'] = MagicMock()
        
        from openai_debug_tool import OpenAIDebugToolGUI
        
        app = OpenAIDebugToolGUI.__new__(OpenAIDebugToolGUI)
        app.send_message = Mock()
        
        # Mock event without Shift (should send message)
        event_no_shift = Mock()
        event_no_shift.state = 0  # No modifiers
        
        result = app._on_enter_key(event_no_shift)
        
        # Should call send_message and return "break"
        app.send_message.assert_called_once()
        self.assertEqual(result, "break")
    
    def test_on_enter_key_with_shift(self):
        """Test Enter key handler with Shift pressed"""
        sys.modules['tkinter'] = MagicMock()
        sys.modules['tkinter.ttk'] = MagicMock()
        sys.modules['tkinter.scrolledtext'] = MagicMock()
        
        from openai_debug_tool import OpenAIDebugToolGUI
        
        app = OpenAIDebugToolGUI.__new__(OpenAIDebugToolGUI)
        app.send_message = Mock()
        
        # Mock event with Shift (0x1 = Shift)
        event_with_shift = Mock()
        event_with_shift.state = 0x1  # Shift pressed
        
        result = app._on_enter_key(event_with_shift)
        
        # Should NOT call send_message and return None
        app.send_message.assert_not_called()
        self.assertIsNone(result)
    
    def test_on_enter_key_with_control(self):
        """Test Enter key handler with Control pressed"""
        sys.modules['tkinter'] = MagicMock()
        sys.modules['tkinter.ttk'] = MagicMock()
        sys.modules['tkinter.scrolledtext'] = MagicMock()
        
        from openai_debug_tool import OpenAIDebugToolGUI
        
        app = OpenAIDebugToolGUI.__new__(OpenAIDebugToolGUI)
        app.send_message = Mock()
        
        # Mock event with Control (0x4 = Control)
        event_with_control = Mock()
        event_with_control.state = 0x4  # Control pressed
        
        result = app._on_enter_key(event_with_control)
        
        # Should NOT call send_message and return None
        app.send_message.assert_not_called()
        self.assertIsNone(result)
    
    def test_on_shift_enter(self):
        """Test Shift+Enter handler"""
        sys.modules['tkinter'] = MagicMock()
        sys.modules['tkinter.ttk'] = MagicMock()
        sys.modules['tkinter.scrolledtext'] = MagicMock()
        
        from openai_debug_tool import OpenAIDebugToolGUI
        
        app = OpenAIDebugToolGUI.__new__(OpenAIDebugToolGUI)
        
        event = Mock()
        result = app._on_shift_enter(event)
        
        # Should return None to allow default newline behavior
        self.assertIsNone(result)
    
    def test_core_classes_available(self):
        """Test that core classes are available and properly defined"""
        from openai_debug_tool import (
            OpenAIClient,
            ConversationManager,
            SpeedCalculator,
            ConfigManager,
            LogEntry,
            LogLevel
        )
        
        # Test instantiation
        client = OpenAIClient()
        self.assertIsNotNone(client)
        
        conv_mgr = ConversationManager()
        self.assertIsNotNone(conv_mgr)
        
        speed_calc = SpeedCalculator()
        self.assertIsNotNone(speed_calc)
        
        config_mgr = ConfigManager()
        self.assertIsNotNone(config_mgr)
        
        # Test LogEntry creation
        log_entry = LogEntry.create_request_log("url", "POST", {}, {})
        self.assertIsNotNone(log_entry)
        self.assertEqual(log_entry.level, LogLevel.INFO)


if __name__ == '__main__':
    unittest.main()
