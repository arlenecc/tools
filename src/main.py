"""
Main GUI Application for OpenAI Debug Tool
"""
import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox,
    QSplitter, QGroupBox, QScrollArea, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from config_manager import ConfigManager
from api_client import APIClient
from message_history import MessageHistory
from speed_calculator import SpeedCalculator
from logger import Logger


class ModelFetcherThread(QThread):
    """Thread for fetching model list from API"""
    models_received = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, base_url, api_key):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
    
    def run(self):
        try:
            client = APIClient(self.base_url, self.api_key)
            models = client.get_models()
            if models:
                self.models_received.emit(models)
            else:
                self.error_occurred.emit("No models returned or failed to fetch")
        except Exception as e:
            self.error_occurred.emit(str(e))


class ChatWorkerThread(QThread):
    """Thread for streaming chat completion"""
    chunk_received = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, base_url, api_key, model, messages):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.messages = messages
    
    def run(self):
        try:
            client = APIClient(self.base_url, self.api_key)
            for chunk in client.chat_completion_stream(self.model, self.messages):
                self.chunk_received.emit(chunk)
            self.finished_signal.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))


class OpenAIDebugTool(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.history = MessageHistory()
        self.app_logger = Logger()
        self.speed_calc = SpeedCalculator()
        self.chat_thread = None
        self.current_response = ""
        
        self.init_ui()
        self.load_config()
        self.app_logger.info("Application started")
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("OpenAI Debug Tool")
        self.setMinimumSize(1200, 800)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for top/bottom sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)
        
        # Top section - Configuration and Chat
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        # Left panel - Configuration
        config_group = self.create_config_panel()
        top_layout.addWidget(config_group, stretch=1)
        
        # Right panel - Chat
        chat_group = self.create_chat_panel()
        top_layout.addWidget(chat_group, stretch=2)
        
        splitter.addWidget(top_widget)
        
        # Bottom section - Logs
        log_group = self.create_log_panel()
        splitter.addWidget(log_group)
        
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_config_panel(self):
        """Create configuration panel"""
        group = QGroupBox("Configuration")
        layout = QVBoxLayout(group)
        
        # Base URL
        url_layout = QHBoxLayout()
        url_label = QLabel("Base URL:")
        url_label.setMinimumWidth(80)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://api.openai.com/v1")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)
        
        # API Key
        key_layout = QHBoxLayout()
        key_label = QLabel("API Key:")
        key_label.setMinimumWidth(80)
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("sk-...")
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.key_input)
        layout.addLayout(key_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setMinimumWidth(80)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(200)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch Models")
        self.fetch_btn.clicked.connect(self.fetch_models)
        self.save_btn = QPushButton("Save Config")
        self.save_btn.clicked.connect(self.save_config)
        btn_layout.addWidget(self.fetch_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        
        # Spacer
        layout.addStretch()
        
        return group
    
    def create_chat_panel(self):
        """Create chat panel"""
        group = QGroupBox("Chat")
        layout = QVBoxLayout(group)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText("Type your message here...")
        input_layout.addWidget(self.input_field)
        
        # Send/Clear buttons
        btn_layout = QVBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_chat)
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.clear_btn)
        input_layout.addLayout(btn_layout)
        layout.addLayout(input_layout)
        
        # Speed display
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed:")
        self.speed_display = QLabel("0.0 tokens/s")
        self.speed_display.setStyleSheet("font-weight: bold; color: #0066cc;")
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_display)
        speed_layout.addStretch()
        layout.addLayout(speed_layout)
        
        return group
    
    def create_log_panel(self):
        """Create log panel"""
        group = QGroupBox("Debug Logs")
        layout = QVBoxLayout(group)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setMaximumHeight(200)
        layout.addWidget(self.log_display)
        
        # Clear logs button
        clear_log_btn = QPushButton("Clear Logs")
        clear_log_btn.clicked.connect(self.clear_logs)
        layout.addWidget(clear_log_btn)
        
        return group
    
    def load_config(self):
        """Load saved configuration"""
        self.config.load()
        self.url_input.setText(self.config.base_url)
        self.key_input.setText(self.config.api_key)
        if self.config.model:
            self.model_combo.setCurrentText(self.config.model)
        self.app_logger.info("Configuration loaded")
    
    def save_config(self):
        """Save current configuration"""
        self.config.base_url = self.url_input.text().strip()
        self.config.api_key = self.key_input.text().strip()
        self.config.model = self.model_combo.currentText().strip()
        self.config.save()
        self.app_logger.info("Configuration saved")
        self.statusBar().showMessage("Configuration saved", 3000)
    
    def fetch_models(self):
        """Fetch available models from API"""
        base_url = self.url_input.text().strip()
        api_key = self.key_input.text().strip()
        
        if not base_url or not api_key:
            QMessageBox.warning(self, "Warning", "Please enter Base URL and API Key")
            return
        
        self.app_logger.info(f"Fetching models from {base_url}")
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("Fetching...")
        
        self.model_thread = ModelFetcherThread(base_url, api_key)
        self.model_thread.models_received.connect(self.on_models_received)
        self.model_thread.error_occurred.connect(self.on_fetch_error)
        self.model_thread.start()
    
    def on_models_received(self, models):
        """Handle received models list"""
        self.model_combo.clear()
        for model in models:
            self.model_combo.addItem(model)
        
        self.app_logger.info(f"Found {len(models)} models")
        self.statusBar().showMessage(f"Found {len(models)} models", 3000)
        self.fetch_btn.setEnabled(True)
        self.fetch_btn.setText("Fetch Models")
    
    def on_fetch_error(self, error_msg):
        """Handle fetch error"""
        self.app_logger.error(f"Failed to fetch models: {error_msg}")
        QMessageBox.warning(self, "Error", f"Failed to fetch models:\n{error_msg}")
        self.fetch_btn.setEnabled(True)
        self.fetch_btn.setText("Fetch Models")
    
    def send_message(self):
        """Send message to API"""
        user_input = self.input_field.toPlainText().strip()
        if not user_input:
            return
        
        base_url = self.url_input.text().strip()
        api_key = self.key_input.text().strip()
        model = self.model_combo.currentText().strip()
        
        if not base_url or not api_key or not model:
            QMessageBox.warning(self, "Warning", 
                              "Please fill in Base URL, API Key, and Model")
            return
        
        # Add user message to history and display
        self.history.add_message("user", user_input)
        self.append_to_chat("User", user_input, "#e3f2fd")
        
        # Clear input
        self.input_field.clear()
        
        # Disable send button during request
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Sending...")
        
        # Start streaming
        self.current_response = ""
        self.speed_calc.start()
        
        self.chat_thread = ChatWorkerThread(
            base_url, api_key, model, self.history.get_messages()
        )
        self.chat_thread.chunk_received.connect(self.on_chunk_received)
        self.chat_thread.finished_signal.connect(self.on_chat_finished)
        self.chat_thread.error_occurred.connect(self.on_chat_error)
        self.chat_thread.start()
        
        self.app_logger.info(f"Sent message to {model}")
    
    def on_chunk_received(self, chunk):
        """Handle received chunk from streaming response"""
        self.current_response += chunk
        self.speed_calc.add_tokens(len(chunk) // 4)  # Approximate token count
        
        # Update speed display
        speed = self.speed_calc.get_speed()
        self.speed_display.setText(f"{speed:.1f} tokens/s")
        
        # Append to chat (update last assistant message)
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Find and update the assistant response
        self.append_to_chat("Assistant", self.current_response, "#f5f5f5", update_last=True)
    
    def on_chat_finished(self):
        """Handle chat completion finished"""
        self.history.add_message("assistant", self.current_response)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        self.app_logger.info("Response completed")
        
        # Final speed update
        speed = self.speed_calc.get_speed()
        self.speed_display.setText(f"{speed:.1f} tokens/s")
    
    def on_chat_error(self, error_msg):
        """Handle chat error"""
        self.app_logger.error(f"Chat error: {error_msg}")
        QMessageBox.warning(self, "Error", f"Chat error:\n{error_msg}")
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        self.speed_calc.reset()
    
    def append_to_chat(self, role, content, bg_color, update_last=False):
        """Append message to chat display"""
        cursor = self.chat_display.textCursor()
        
        if update_last:
            # Remove previous assistant content and add new
            # Simple approach: just append with marker
            pass
        
        # Format message
        html = f'''
        <div style="background-color: {bg_color}; padding: 8px; margin: 4px 0; border-radius: 4px;">
            <b>{role}:</b><br/>
            {content.replace('<', '&lt;').replace('>', '&gt;')}
        </div>
        '''
        
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html)
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
    
    def clear_chat(self):
        """Clear chat history"""
        self.history.clear()
        self.chat_display.clear()
        self.speed_calc.reset()
        self.speed_display.setText("0.0 tokens/s")
        self.app_logger.info("Chat cleared")
    
    def clear_logs(self):
        """Clear log display"""
        self.log_display.clear()
        self.app_logger.clear()
    
    def add_log_entry(self, entry):
        """Add log entry to display"""
        self.log_display.append(str(entry))
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )
    
    def closeEvent(self, event):
        """Handle window close"""
        self.save_config()
        self.app_logger.info("Application closed")
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = OpenAIDebugTool()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
