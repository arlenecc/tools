import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox, QSplitter, QMessageBox, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMetaObject, QMutex, QMutexLocker
from PyQt6.QtGui import QFont, QColor, QTextCursor, QPalette

from .config_manager import Config, ConfigManager
from .api_client import APIClient
from .message_history import MessageHistory
from .speed_calculator import SpeedCalculator
from .logger import Logger

class StreamWorker(QThread):
    token_received = pyqtSignal(str); response_finished = pyqtSignal(); error_occurred = pyqtSignal(str); log_signal = pyqtSignal(str)
    def __init__(self, client, messages, model):
        super().__init__(); self.client = client; self.messages = messages; self.model = model; self._stop_flag = False
    def run(self):
        try:
            self.log_signal.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 开始流式请求...")
            for content, error in self.client.chat_completion_stream(self.messages, self.model):
                if self._stop_flag: break
                if error:
                    self.log_signal.emit(f"[{datetime.now().strftime('%H:%M:%S')}] {error}")
                    self.error_occurred.emit(error)
                    return
                if content:
                    self.token_received.emit(content)
            self.log_signal.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 响应结束")
            self.response_finished.emit()
        except Exception as e:
            err = f"错误: {str(e)}"
            self.log_signal.emit(f"[{datetime.now().strftime('%H:%M:%S')}] {err}")
            self.error_occurred.emit(err)
    def stop(self): self._stop_flag = True

class OpenAIDebugTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager(); self.api_client = APIClient(); self.message_history = MessageHistory()
        self.speed_calculator = SpeedCalculator(); self.logger = Logger()
        self.stream_worker = None; self.current_reply = ""; self.reply_mutex = QMutex()
        self._msg_data = []; self._collapsed = False
        self._init_ui(); self._load_config(); self._setup_connections()

    def _init_ui(self):
        self.setWindowTitle("OpenAI 调试工具 (修复版)"); self.setGeometry(100, 100, 1400, 900)
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QVBoxLayout(main_widget); main_layout.setSpacing(10); main_layout.setContentsMargins(10,10,10,10)
        
        # 顶部配置
        cfg = QGroupBox("配置"); cfg.setStyleSheet("QGroupBox{color:#e0e0e0;border:1px solid #555;border-radius:5px;margin-top:5px;padding-top:2px;font-weight:bold;} QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}")
        cfg_l = QHBoxLayout(cfg); cfg_l.setSpacing(8); cfg_l.setContentsMargins(5,2,5,2)
        self.url_in = QLineEdit(); self.url_in.setPlaceholderText("Base URL"); self.url_in.setMinimumWidth(200)
        self.key_in = QLineEdit(); self.key_in.setPlaceholderText("API Key"); self.key_in.setEchoMode(QLineEdit.EchoMode.Password); self.key_in.setMinimumWidth(200)
        self.mod_in = QComboBox(); self.mod_in.setEditable(True); self.mod_in.setPlaceholderText("模型"); self.mod_in.setMinimumWidth(150)
        self.get_btn = QPushButton("获取模型"); self.get_btn.setStyleSheet("QPushButton{background:#3a7bd5;color:white;border:none;padding:6px 12px;border-radius:4px;font-weight:bold;} QPushButton:hover{background:#2c5aa0;}")
        cfg_l.addWidget(QLabel("URL:")); cfg_l.addWidget(self.url_in)
        cfg_l.addWidget(QLabel("Key:")); cfg_l.addWidget(self.key_in)
        cfg_l.addWidget(QLabel("Model:")); cfg_l.addWidget(self.mod_in)
        cfg_l.addWidget(self.get_btn); cfg_l.addStretch()

        # 中间分割
        split = QSplitter(Qt.Orientation.Horizontal); split.setHandleWidth(2); split.setStyleSheet("QSplitter::handle{background:#555;}")
        
        # 左侧对话
        left_w = QWidget(); left_l = QVBoxLayout(left_w); left_l.setContentsMargins(0,0,0,0); left_l.setSpacing(5)
        chat_h = QHBoxLayout()
        chat_lbl = QLabel("对话"); chat_lbl.setStyleSheet("color:#e0e0e0;font-weight:bold;font-size:12px;")
        self.think_btn = QPushButton("折叠思考过程"); self.think_btn.setStyleSheet("QPushButton{background:#6c757d;color:white;border:none;padding:4px 8px;border-radius:3px;font-size:11px;} QPushButton:hover{background:#5a6268;}")
        self.think_btn.clicked.connect(self._toggle_thinking)
        chat_h.addWidget(chat_lbl); chat_h.addStretch(); chat_h.addWidget(self.think_btn)
        left_l.addLayout(chat_h)
        self.chat_out = QTextEdit(); self.chat_out.setReadOnly(True); self.chat_out.setFont(QFont("Consolas", 14))
        self.chat_out.setStyleSheet("QTextEdit{background:#2b2b2b;color:#e0e0e0;border:1px solid #555;border-radius:4px;padding:5px;}")
        left_l.addWidget(self.chat_out, 85)
        in_l = QHBoxLayout(); self.in_field = QLineEdit(); self.in_field.setPlaceholderText("输入消息..."); self.in_field.setFont(QFont("Arial", 14))
        self.in_field.setStyleSheet("QLineEdit{background:#2b2b2b;color:#e0e0e0;border:1px solid #555;border-radius:4px;padding:8px;}")
        self.in_field.returnPressed.connect(self._on_send)
        self.send_btn = QPushButton("发送"); self.send_btn.setStyleSheet("QPushButton{background:#28a745;color:white;border:none;padding:6px 15px;border-radius:4px;font-weight:bold;font-size:14px;} QPushButton:disabled{background:#555;color:#888;}")
        self.send_btn.clicked.connect(self._on_send)
        self.clr_btn = QPushButton("清空"); self.clr_btn.setStyleSheet("QPushButton{background:#dc3545;color:white;border:none;padding:6px 12px;border-radius:4px;font-weight:bold;}")
        self.clr_btn.clicked.connect(self._clear_chat)
        in_l.addWidget(self.in_field); in_l.addWidget(self.send_btn); in_l.addWidget(self.clr_btn)
        left_l.addLayout(in_l, 15)

        # 右侧日志
        right_w = QWidget(); right_l = QVBoxLayout(right_w); right_l.setContentsMargins(0,0,0,0); right_l.setSpacing(5)
        log_h = QHBoxLayout(); log_lbl = QLabel("实时日志"); log_lbl.setStyleSheet("color:#e0e0e0;font-weight:bold;font-size:12px;")
        self.clr_log_btn = QPushButton("清空"); self.clr_log_btn.setStyleSheet("QPushButton{background:#6c757d;color:white;border:none;padding:4px 8px;border-radius:3px;font-size:11px;}")
        self.clr_log_btn.clicked.connect(self._clear_logs)
        log_h.addWidget(log_lbl); log_h.addStretch(); log_h.addWidget(self.clr_log_btn)
        self.log_out = QTextEdit(); self.log_out.setReadOnly(True); self.log_out.setFont(QFont("Consolas", 14))
        self.log_out.setStyleSheet("QTextEdit{background:#1e1e1e;color:#00ff00;border:1px solid #555;border-radius:4px;padding:5px;}")
        right_l.addLayout(log_h); right_l.addWidget(self.log_out)

        split.addWidget(left_w); split.addWidget(right_w); split.setStretchFactor(0, 2); split.setStretchFactor(1, 1)

        # 底部状态
        self.status_lbl = QLabel("就绪"); self.status_lbl.setStyleSheet("color:#aaa;padding:5px;background:#2b2b2b;border-radius:3px;")

        main_layout.addWidget(cfg, 10); main_layout.addWidget(split, 90); main_layout.addWidget(self.status_lbl)

    def _load_config(self):
        c = self.config_manager.load()
        self.url_in.setText(c.base_url); self.key_in.setText(c.api_key); self.mod_in.setCurrentText(c.model)

    def _setup_connections(self):
        self.get_btn.clicked.connect(self._get_models)
        self.logger.log_signal.connect(self._append_log)

    def _append_log(self, msg):
        cur = self.log_out.textCursor(); cur.movePosition(QTextCursor.MoveOperation.End); cur.insertText(msg+"\n")
        self.log_out.setTextCursor(cur); self.log_out.ensureCursorVisible()

    def _get_models(self):
        url = self.url_in.text().strip(); key = self.key_in.text().strip()
        if not url: QMessageBox.warning(self, "警告", "请填写 Base URL"); return
        self.status_lbl.setText("获取中..."); self.get_btn.setEnabled(False)
        self.api_client.set_base_url(url); self.api_client.set_api_key(key)
        try:
            models, error = self.api_client.fetch_models()
            if error:
                QMessageBox.critical(self, "错误", error)
                return
            model_ids = [m.id for m in models]
            self.mod_in.clear()
            if model_ids:
                self.mod_in.addItems(model_ids)
            self.status_lbl.setText(f"找到 {len(model_ids)} 个模型" if model_ids else "无模型")
        except Exception as e: QMessageBox.critical(self, "错误", str(e))
        finally: self.get_btn.setEnabled(True)

    def _on_send(self):
        txt = self.in_field.text().strip()
        if not txt: return
        url = self.url_in.text().strip(); key = self.key_in.text().strip(); mod = self.mod_in.currentText().strip()
        if not url or not key or not mod: QMessageBox.warning(self, "警告", "请填写完整配置"); return
        
        self._append_msg("user", txt); self.in_field.clear()
        self.send_btn.setEnabled(False); self.in_field.setEnabled(False)
        self.status_lbl.setText("思考中..."); self.current_reply = ""; self.speed_calculator.start()
        self.message_history.add_message("user", txt)
        self.api_client.set_base_url(url); self.api_client.set_api_key(key)
        
        self.stream_worker = StreamWorker(self.api_client, self.message_history.get_api_messages(), mod)
        self.stream_worker.token_received.connect(self._on_token)
        self.stream_worker.response_finished.connect(self._on_finish)
        self.stream_worker.error_occurred.connect(self._on_err)
        self.stream_worker.log_signal.connect(self._append_log)
        self.stream_worker.start()

    def _on_token(self, token):
        with QMutexLocker(self.reply_mutex): self.current_reply += token
        spd = self.speed_calculator.add_token()
        disp = self.current_reply[-50:] if len(self.current_reply)>50 else self.current_reply
        self.status_lbl.setText(f"⚡ {spd:.1f} t/s | ...{disp}")
        cur = self.chat_out.textCursor(); cur.movePosition(QTextCursor.MoveOperation.End); cur.insertText(token)
        self.chat_out.setTextCursor(cur); self.chat_out.ensureCursorVisible()

    def _on_finish(self):
        self.send_btn.setEnabled(True); self.in_field.setEnabled(True); self.in_field.setFocus()
        with QMutexLocker(self.reply_mutex): final = self.current_reply
        if final: self.message_history.add_message("assistant", final)
        self.status_lbl.setText(f"完成 (Tokens: {self.speed_calculator.get_current_stats().total_tokens})")
        self.speed_calculator.reset()

    def _on_err(self, msg):
        self.send_btn.setEnabled(True); self.in_field.setEnabled(True); self.status_lbl.setText("错误")
        QMessageBox.critical(self, "错误", msg)

    def _append_msg(self, role, content):
        self._msg_data.append({'role': role, 'content': content})
        self._rebuild_chat()

    def _clear_chat(self):
        self.chat_out.clear(); self.message_history.clear()
        self._msg_data.clear(); self.status_lbl.setText("已清空")

    def _clear_logs(self): self.log_out.clear()

    def _rebuild_chat(self):
        html = ''
        for msg in self._msg_data:
            role, content = msg['role'], msg['content']
            color = "#4da6ff" if role == "user" else "#00cc66"
            name = "User" if role == "user" else "AI"
            c = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace(chr(10), '<br>')
            html += f'<div style="margin:10px 0;"><span style="color:{color};font-weight:bold;">{name}:</span><br>'
            if role == "assistant" and self._collapsed:
                html += '<span style="color:#888;font-style:italic;">[思考过程已折叠 - 点击按钮展开]</span>'
            else:
                html += f'<span style="color:#e0e0e0;">{c}</span>'
            html += '</div>'
        self.chat_out.setHtml(html)
        cur = self.chat_out.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        self.chat_out.setTextCursor(cur)
        self.chat_out.ensureCursorVisible()

    def _toggle_thinking(self):
        self._collapsed = not self._collapsed
        self.think_btn.setText("展开思考过程" if self._collapsed else "折叠思考过程")
        self._rebuild_chat()

    def closeEvent(self, event):
        if self.stream_worker and self.stream_worker.isRunning(): self.stream_worker.stop(); self.stream_worker.wait()
        self.config_manager.save(Config(base_url=self.url_in.text(), api_key=self.key_in.text(), model=self.mod_in.currentText()))
        event.accept()

def main():
    app = QApplication(sys.argv); app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(53,53,53)); pal.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    pal.setColor(QPalette.ColorRole.Base, QColor(25,25,25)); pal.setColor(QPalette.ColorRole.AlternateBase, QColor(53,53,53))
    pal.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white); pal.setColor(QPalette.ColorRole.Button, QColor(53,53,53))
    pal.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white); pal.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.black)
    pal.setColor(QPalette.ColorRole.Link, QColor(42,130,218)); pal.setColor(QPalette.ColorRole.Highlight, QColor(42,130,218))
    pal.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(pal)
    win = OpenAIDebugTool(); win.show(); sys.exit(app.exec())

if __name__ == "__main__": main()
