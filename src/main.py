"""OpenAI 调试工具 - PyQt6 GUI"""
import sys
from typing import Optional, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox,
    QSplitter, QGroupBox, QFormLayout, QMessageBox, QFrame,
    QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor

from .config_manager import ConfigManager, Config
from .api_client import APIClient, ModelInfo
from .message_history import MessageHistory
from .speed_calculator import SpeedCalculator
from .logger import Logger
from .log_entry import LogEntry


class ModelFetchWorker(QThread):
    """模型获取工作线程"""
    finished = pyqtSignal(list, str)  # models, error

    def __init__(self, client: APIClient):
        super().__init__()
        self.client = client

    def run(self):
        models, error = self.client.fetch_models()
        self.finished.emit(models, error or "")


class ChatWorker(QThread):
    """聊天工作线程"""
    token_received = pyqtSignal(str)
    finished = pyqtSignal(str, str)  # full_content, error
    speed_update = pyqtSignal(float)

    def __init__(self, client: APIClient, messages: List, model: str, logger=None):
        super().__init__()
        self.client = client
        self.messages = messages
        self.model = model
        self.logger = logger
        self._full_content = ""

    def run(self):
        self._full_content = ""
        for content, error in self.client.chat_completion_stream(self.messages, self.model, self.logger):
            if error:
                self.finished.emit("", error)
                return
            if content:
                self._full_content += content
                self.token_received.emit(content)
        self.finished.emit(self._full_content, "")


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        self.client = APIClient(self.config.base_url, self.config.api_key)
        self.history = MessageHistory()
        self.logger = Logger()
        self.speed_calc = SpeedCalculator()
        self.current_worker: Optional[ChatWorker] = None

        self._init_ui()
        self._load_config_to_ui()
        self.logger.register_callback(self._on_log_entry)

    def _init_ui(self):
        """初始化 UI - 新布局：顶部配置 (1/5)，左侧对话 (2/3)，右侧日志 (1/3)"""
        self.setWindowTitle("OpenAI API 调试工具")
        self.setMinimumSize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 顶部配置面板 (约 1/5 高度，横向占满)
        config_group = self._create_config_panel()
        config_group.setMaximumHeight(180)
        main_layout.addWidget(config_group, 0)

        # 下部区域 (水平分割：左侧对话 + 右侧日志)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.setStretchFactor(0, 2)  # 对话区域占 2/3
        bottom_splitter.setStretchFactor(1, 1)  # 日志区域占 1/3
        main_layout.addWidget(bottom_splitter, 1)

        # 左侧对话区域
        chat_widget = self._create_chat_area()
        bottom_splitter.addWidget(chat_widget)

        # 右侧日志区域
        log_widget = self._create_log_area()
        bottom_splitter.addWidget(log_widget)

        # 设置初始比例
        QTimer.singleShot(100, lambda: self._set_splitter_sizes(bottom_splitter))

    def _set_splitter_sizes(self, splitter: QSplitter):
        """设置分割器初始大小"""
        total_width = splitter.width()
        if total_width > 0:
            splitter.setSizes([int(total_width * 0.67), int(total_width * 0.33)])

    def _create_config_panel(self) -> QGroupBox:
        group = QGroupBox("服务配置")
        layout = QFormLayout()

        # Base URL
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("http://localhost:11434/v1")
        layout.addRow("Base URL:", self.url_input)

        # API Key
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("可选")
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addRow("API Key:", self.key_input)

        # 模型选择
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setPlaceholderText("选择或输入模型名称")
        model_layout.addWidget(self.model_combo, 1)

        self.fetch_btn = QPushButton("获取模型列表")
        self.fetch_btn.clicked.connect(self._fetch_models)
        model_layout.addWidget(self.fetch_btn)
        layout.addRow("模型:", model_layout)

        group.setLayout(layout)
        return group

    def _create_chat_area(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 消息显示区
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 14))  # 字体加大到 14pt
        # 设置对话区域背景为深灰色，与输入框一致
        self.chat_display.setStyleSheet("""
            QTextEdit { 
                background-color: #2b2b2b; 
                color: #e0e0e0;
                border: 1px solid #555555;
                padding: 5px;
            }
        """)
        layout.addWidget(self.chat_display)

        # 状态栏
        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.speed_label = QLabel("速度：0.0 tokens/s")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.speed_label)
        layout.addLayout(status_layout)

        # 输入区
        input_layout = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText("输入消息... (Ctrl+Enter 发送)")
        self.input_field.setStyleSheet("""
            QTextEdit { 
                background-color: #2b2b2b; 
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
            }
        """)
        self.input_field.installEventFilter(self)
        input_layout.addWidget(self.input_field, 1)

        send_btn = QPushButton("发送")
        send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(send_btn)

        clear_btn = QPushButton("清空对话")
        clear_btn.clicked.connect(self._clear_chat)
        input_layout.addWidget(clear_btn)

        layout.addLayout(input_layout)
        return widget

    def _create_log_area(self) -> QGroupBox:
        group = QGroupBox("调试日志")
        layout = QVBoxLayout()

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 14))  # 字体加大到 14pt
        # 设置日志区域背景色为深灰色，与整体主题一致
        self.log_display.setStyleSheet("""
            QTextEdit { 
                background-color: #2b2b2b; 
                color: #e0e0e0;
                border: 1px solid #555555;
                padding: 5px;
            }
        """)
        layout.addWidget(self.log_display)

        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self._clear_log)
        layout.addWidget(clear_log_btn)

        group.setLayout(layout)
        return group

    def _load_config_to_ui(self):
        self.url_input.setText(self.config.base_url)
        self.key_input.setText(self.config.api_key)
        if self.config.model:
            self.model_combo.setCurrentText(self.config.model)

    def _save_config_from_ui(self):
        self.config.base_url = self.url_input.text().strip()
        self.config.api_key = self.key_input.text().strip()
        self.config.model = self.model_combo.currentText().strip()
        self.config_manager.save(self.config)

    def _fetch_models(self):
        self._save_config_from_ui()
        self.client.set_base_url(self.config.base_url)
        self.client.set_api_key(self.config.api_key)

        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("获取中...")
        self.logger.info("正在获取模型列表...", f"URL: {self.config.base_url}/models")

        self.worker = ModelFetchWorker(self.client)
        self.worker.finished.connect(self._on_models_fetched)
        self.worker.start()

    def _on_models_fetched(self, models: List[ModelInfo], error: str):
        self.fetch_btn.setEnabled(True)
        self.fetch_btn.setText("获取模型列表")

        if error:
            self.logger.error("获取模型列表失败", error)
            QMessageBox.warning(self, "错误", f"获取模型列表失败:\n{error}")
            return

        self.model_combo.clear()
        if not models:
            self.logger.warning("未找到任何模型")
            self.model_combo.addItem("无可用模型")
            return

        self.logger.info(f"找到 {len(models)} 个模型")
        for model in models:
            self.model_combo.addItem(model.id)

        # 如果有之前选择的模型，尝试选中
        if self.config.model:
            idx = self.model_combo.findText(self.config.model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

    def _send_message(self):
        user_text = self.input_field.toPlainText().strip()
        if not user_text:
            return

        if self.current_worker and self.current_worker.isRunning():
            return

        self._save_config_from_ui()
        self.client.set_base_url(self.config.base_url)
        self.client.set_api_key(self.config.api_key)

        model = self.model_combo.currentText().strip()
        if not model or model == "无可用模型":
            QMessageBox.warning(self, "警告", "请先选择或输入模型名称")
            return

        # 添加用户消息
        self.history.add_user_message(user_text)
        self._append_chat("user", user_text)
        self.input_field.clear()

        self.status_label.setText("正在思考...")
        self.speed_calc.reset()
        self.speed_calc.start()
        self.logger.info(f"发送请求到模型: {model}", f"消息数: {len(self.history)}")

        messages = self.history.get_api_messages()
        self.current_worker = ChatWorker(self.client, messages, model, self.logger)
        self.current_worker.token_received.connect(self._on_token)
        self.current_worker.finished.connect(self._on_chat_finished)
        self.current_worker.start()

    def _on_token(self, text: str):
        """处理接收到的 token - 实时更新显示"""
        self.speed_calc.add_token()
        stats = self.speed_calc.get_current_stats()
        self.speed_label.setText(f"速度：{stats.tokens_per_second:.1f} tokens/s")

        # 初始化或追加到当前 assistant 消息块
        if not hasattr(self, "_current_assistant_block"):
            self._current_assistant_block = ""
        
        self._current_assistant_block += text
        
        # 更新历史中的最后一条消息
        if self.history._messages and self.history._messages[-1].role == "assistant":
            self.history._messages[-1].content = self._current_assistant_block
        else:
            # 如果是第一条 assistant 消息，添加到历史
            self.history.add_assistant_message(self._current_assistant_block, self.config.model)
        
        # 刷新对话显示 - 清空并重新渲染所有消息
        self.chat_display.clear()
        for msg in self.history.get_messages():
            self._append_chat(msg.role, msg.content)
        
        # 更新状态栏显示思考过程（最后 50 个字符）
        preview = self._current_assistant_block[-50:] if len(self._current_assistant_block) > 50 else self._current_assistant_block
        self.status_label.setText(f"正在生成：{preview}...")

    def _on_chat_finished(self, content: str, error: str):
        self.current_worker = None
        self.speed_calc.stop()
        
        if error:
            self.logger.error("聊天请求失败", error)
            self._append_chat("system", f"错误：{error}")
            self.status_label.setText("错误")
        else:
            if content:
                if not hasattr(self, '_current_assistant_block') or not self._current_assistant_block:
                    self.history.add_assistant_message(content, self.config.model)
                    self._append_chat("assistant", content)
                else:
                    self.history._messages[-1].content = self._current_assistant_block
                self.logger.info("收到响应", f"长度：{len(content)}")
            self.status_label.setText("就绪")
        
        stats = self.speed_calc.stop()
        self.speed_label.setText(f"速度：{stats.tokens_per_second:.1f} tokens/s (共 {stats.total_tokens} tokens)")
        self._current_assistant_block = None

    def _append_chat(self, role: str, content: str):
        color_map = {"user": "#0066cc", "assistant": "#339933", "system": "#cc3300"}
        name_map = {"user": "👤 用户", "assistant": "🤖 AI", "system": "⚠️ 系统"}
        color = color_map.get(role, "#666666")
        name = name_map.get(role, role)

        html = f'<div style="margin: 8px 0;"><b style="color: {color};">{name}</b><br/>{content.replace(chr(10), "<br/>")}</div>'
        self.chat_display.append(html)
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def _clear_chat(self):
        self.history.clear()
        self.chat_display.clear()
        self.status_label.setText("已清空")
        self.speed_label.setText("速度：0.0 tokens/s")
        self.logger.info("对话历史已清空")
        self._current_assistant_block = None

    def _on_log_entry(self, entry: LogEntry):
        time_str = entry.timestamp.strftime("%H:%M:%S")
        self.log_display.append(f"[{time_str}] [{entry.level}] {entry.message}")
        if entry.details:
            self.log_display.append(f"  → {entry.details}")
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

    def _clear_log(self):
        self.logger.clear()
        self.log_display.clear()

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent
        
        if obj == self.input_field and event.type() == QEvent.Type.KeyPress:
            key_event = event
            if key_event.key() == Qt.Key.Key_Return and key_event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        self._save_config_from_ui()
        self.client.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
