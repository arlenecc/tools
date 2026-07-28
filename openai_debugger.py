#!/usr/bin/env python3
"""
OpenAI API 调试工具 - 图形界面版本
支持 macOS 平台，具有实时速度显示、日志记录、预设动作等功能
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
import json
import time
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
import aiohttp


# ==================== 数据类 ====================

@dataclass
class APIConfig:
    """API 配置管理"""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    timeout: int = 30
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    def get_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        headers.update(self.custom_headers)
        return headers
    
    def add_custom_header(self, key: str, value: str):
        self.custom_headers[key] = value
    
    def remove_custom_header(self, key: str):
        if key in self.custom_headers:
            del self.custom_headers[key]


@dataclass
class LogEntry:
    """日志条目"""
    level: str
    message: str
    details: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_string(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        detail_str = json.dumps(self.details, ensure_ascii=False, indent=2) if self.details else ""
        return f"[{ts}] [{self.level}] {self.message}\n{detail_str}".strip()


class SpeedCalculator:
    """实时推理速度计算器"""
    
    def __init__(self, window_size: int = 50):
        self.token_times: List[float] = []
        self.window_size = window_size
        self.total_tokens = 0
        
    def add_token(self, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        self.token_times.append(timestamp)
        self.total_tokens += 1
        # 保持窗口大小
        if len(self.token_times) > self.window_size:
            self.token_times.pop(0)
    
    def get_speed(self) -> float:
        if len(self.token_times) < 2:
            return 0.0
        time_span = self.token_times[-1] - self.token_times[0]
        if time_span <= 0:
            return 0.0
        return (len(self.token_times) - 1) / time_span
    
    def reset(self):
        self.token_times = []
        self.total_tokens = 0


class ConversationHistory:
    """对话历史管理"""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
    
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def clear(self):
        self.messages = []
    
    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages.copy()
    
    def get_last_message(self) -> Optional[Dict[str, str]]:
        if self.messages:
            return self.messages[-1]
        return None


# ==================== 功能函数 ====================

def format_message(role: str, content: str) -> Dict[str, str]:
    """格式化消息"""
    return {"role": role, "content": content}


def build_request_body(
    messages: List[Dict[str, str]], 
    params: Dict[str, Any],
    model: str = "gpt-3.5-turbo"
) -> Dict[str, Any]:
    """构建请求体"""
    body = {
        "model": model,
        "messages": messages,
        "temperature": params.get("temperature", 0.7),
        "max_tokens": params.get("max_tokens", 1024),
        "stream": params.get("stream", True)
    }
    
    if "top_p" in params and params["top_p"] is not None:
        body["top_p"] = params["top_p"]
    if "frequency_penalty" in params and params["frequency_penalty"] is not None:
        body["frequency_penalty"] = params["frequency_penalty"]
    if "presence_penalty" in params and params["presence_penalty"] is not None:
        body["presence_penalty"] = params["presence_penalty"]
    
    return body


def parse_sse_line(line: str) -> Optional[Dict]:
    """解析 SSE 行"""
    if not line.startswith("data: "):
        return None
    if line == "data: [DONE]":
        return None
    
    data_part = line[6:]  # 移除 "data: "
    try:
        return json.loads(data_part)
    except json.JSONDecodeError:
        return None


def get_preset_actions() -> List[Dict[str, Any]]:
    """获取预设动作列表"""
    return [
        {
            "name": "测试联通性",
            "description": "测试服务器是否可达",
            "method": "GET",
            "endpoint": "/models",
            "body": None,
            "headers": {}
        },
        {
            "name": "获取模型信息",
            "description": "获取可用模型列表",
            "method": "GET",
            "endpoint": "/models",
            "body": None,
            "headers": {}
        },
        {
            "name": "Say Hello",
            "description": "发送简单的问候消息",
            "method": "POST",
            "endpoint": "/chat/completions",
            "body": {
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "temperature": 0.7,
                "max_tokens": 50,
                "stream": True
            },
            "headers": {}
        },
        {
            "name": "测试正常返回",
            "description": "测试模型正常响应能力",
            "method": "POST",
            "endpoint": "/chat/completions",
            "body": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "请简要介绍一下你自己。"}
                ],
                "temperature": 0.7,
                "max_tokens": 200,
                "stream": True
            },
            "headers": {}
        },
        {
            "name": "自定义请求",
            "description": "用户自定义请求内容和参数",
            "method": "POST",
            "endpoint": "/chat/completions",
            "body": {
                "messages": [],
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": True
            },
            "headers": {},
            "editable": True
        }
    ]


# ==================== API 客户端 ====================

class APIClient:
    """异步 API 客户端"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def send_request(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[Dict] = None,
        timeout: int = 30
    ) -> AsyncGenerator[Dict, None]:
        """发送 HTTP 请求（支持流式）"""
        session = await self.get_session()
        
        try:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=body if method in ["POST", "PUT", "PATCH"] else None,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield {"error": f"HTTP {response.status}: {error_text}"}
                    return
                
                # 检查是否是流式响应
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type or body.get("stream", False):
                    async for line in response.content.iter_lines():
                        if line:
                            parsed = parse_sse_line(line.decode('utf-8'))
                            if parsed:
                                yield parsed
                else:
                    result = await response.json()
                    yield result
                    
        except asyncio.TimeoutError:
            yield {"error": "Request timeout"}
        except aiohttp.ClientError as e:
            yield {"error": f"Connection error: {str(e)}"}
        except Exception as e:
            yield {"error": f"Unexpected error: {str(e)}"}
    
    async def send_chat_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any]
    ) -> AsyncGenerator[Dict, None]:
        """专门用于聊天补全的请求"""
        async for chunk in self.send_request(url, "POST", headers, body):
            yield chunk


# ==================== GUI 应用 ====================

class OpenAIDebuggerApp:
    """主 GUI 应用"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OpenAI API Debugger")
        self.root.geometry("1200x800")
        
        # 状态变量
        self.config = APIConfig()
        self.history = ConversationHistory()
        self.speed_calc = SpeedCalculator()
        self.api_client = APIClient()
        self.is_streaming = False
        self.logs: List[LogEntry] = []
        
        # 请求参数
        self.params = {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": True
        }
        
        # 创建 UI
        self._create_menu()
        self._create_ui()
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="导出日志", command=self._export_logs)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_close)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self._show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _create_ui(self):
        """创建主界面"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 配置区域
        self._create_config_section(main_frame)
        
        # 预设动作区域
        self._create_preset_section(main_frame)
        
        # 对话区域
        self._create_chat_section(main_frame)
        
        # 日志区域
        self._create_log_section(main_frame)
        
        # 状态栏
        self._create_status_bar(main_frame)
    
    def _create_config_section(self, parent):
        """创建配置区域"""
        config_frame = ttk.LabelFrame(parent, text="API 配置", padding="5")
        config_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        
        # 第一行：Base URL
        ttk.Label(config_frame, text="Base URL:").grid(row=0, column=0, sticky="w")
        self.url_var = tk.StringVar(value=self.config.base_url)
        url_entry = ttk.Entry(config_frame, textvariable=self.url_var, width=50)
        url_entry.grid(row=0, column=1, sticky="ew", padx=5)
        
        # 第二行：API Key
        ttk.Label(config_frame, text="API Key:").grid(row=1, column=0, sticky="w")
        self.key_var = tk.StringVar(value=self.config.api_key)
        key_entry = ttk.Entry(config_frame, textvariable=self.key_var, width=50, show="*")
        key_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        
        # 第三行：Model
        ttk.Label(config_frame, text="Model:").grid(row=2, column=0, sticky="w")
        self.model_var = tk.StringVar(value=self.config.model)
        model_entry = ttk.Entry(config_frame, textvariable=self.model_var, width=50)
        model_entry.grid(row=2, column=1, sticky="ew", padx=5)
        
        # 第四行：自定义 Headers
        ttk.Label(config_frame, text="Custom Headers:").grid(row=3, column=0, sticky="nw")
        self.headers_text = scrolledtext.ScrolledText(config_frame, width=50, height=3)
        self.headers_text.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        self.headers_text.insert("1.0", "{}")
        
        # 参数配置
        param_frame = ttk.LabelFrame(config_frame, text="请求参数", padding="5")
        param_frame.grid(row=0, column=2, rowspan=4, sticky="nsew", padx=10)
        
        # Temperature
        ttk.Label(param_frame, text="Temperature:").grid(row=0, column=0, sticky="w")
        self.temp_var = tk.DoubleVar(value=self.params["temperature"])
        temp_spin = ttk.Spinbox(param_frame, from_=0, to=2, increment=0.1, 
                               textvariable=self.temp_var, width=10)
        temp_spin.grid(row=0, column=1, sticky="w", padx=5)
        
        # Max Tokens
        ttk.Label(param_frame, text="Max Tokens:").grid(row=1, column=0, sticky="w")
        self.max_tokens_var = tk.IntVar(value=self.params["max_tokens"])
        max_tokens_spin = ttk.Spinbox(param_frame, from_=1, to=8192, increment=100,
                                     textvariable=self.max_tokens_var, width=10)
        max_tokens_spin.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        # Stream
        self.stream_var = tk.BooleanVar(value=self.params["stream"])
        stream_check = ttk.Checkbutton(param_frame, text="Stream", 
                                       variable=self.stream_var)
        stream_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        
        config_frame.columnconfigure(1, weight=1)
    
    def _create_preset_section(self, parent):
        """创建预设动作区域"""
        preset_frame = ttk.LabelFrame(parent, text="预设动作 (双击执行)", padding="5")
        preset_frame.grid(row=1, column=0, sticky="ns", pady=(0, 5), padx=(0, 5))
        
        # 动作列表
        self.preset_listbox = tk.Listbox(preset_frame, width=25, height=10)
        self.preset_listbox.grid(row=0, column=0, sticky="ns")
        
        scrollbar = ttk.Scrollbar(preset_frame, orient="vertical", 
                                 command=self.preset_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.preset_listbox.config(yscrollcommand=scrollbar.set)
        
        # 填充预设动作
        self.preset_actions = get_preset_actions()
        for action in self.preset_actions:
            self.preset_listbox.insert(tk.END, action["name"])
        
        # 绑定双击事件
        self.preset_listbox.bind("<Double-Button-1>", self._on_preset_double_click)
        
        # 动作描述
        self.preset_desc_var = tk.StringVar()
        desc_label = ttk.Label(preset_frame, textvariable=self.preset_desc_var, 
                              wraplength=200)
        desc_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        
        # 绑定选择事件
        self.preset_listbox.bind("<<ListboxSelect>>", self._on_preset_select)
        
        preset_frame.rowconfigure(0, weight=1)
    
    def _create_chat_section(self, parent):
        """创建对话区域"""
        chat_frame = ttk.LabelFrame(parent, text="对话", padding="5")
        chat_frame.grid(row=1, column=1, sticky="nsew", pady=(0, 5))
        
        # 对话历史显示
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, 
                                                     state='disabled', height=15)
        self.chat_display.grid(row=0, column=0, columnspan=2, sticky="nsew")
        
        # 输入区域
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=4, width=50)
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 发送按钮
        send_btn = ttk.Button(input_frame, text="发送", command=self._send_message)
        send_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 清空按钮
        clear_btn = ttk.Button(chat_frame, text="清空对话", command=self._clear_chat)
        clear_btn.grid(row=2, column=0, sticky="w", pady=(5, 0))
        
        # 速度显示
        self.speed_var = tk.StringVar(value="Speed: 0.0 tokens/s")
        speed_label = ttk.Label(chat_frame, textvariable=self.speed_var, 
                               foreground="green")
        speed_label.grid(row=2, column=1, sticky="e", pady=(5, 0))
        
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
    
    def _create_log_section(self, parent):
        """创建日志区域"""
        log_frame = ttk.LabelFrame(parent, text="交互日志", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        
        # 日志显示
        self.log_display = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, 
                                                    state='disabled', height=8)
        self.log_display.grid(row=0, column=0, sticky="nsew")
        
        # 日志控制
        control_frame = ttk.Frame(log_frame)
        control_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        ttk.Button(control_frame, text="清空日志", 
                  command=self._clear_logs).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="导出日志", 
                  command=self._export_logs).pack(side=tk.LEFT, padx=5)
        
        # 日志级别过滤
        ttk.Label(control_frame, text="过滤:").pack(side=tk.LEFT, padx=(10, 5))
        self.log_filter_var = tk.StringVar(value="ALL")
        filter_combo = ttk.Combobox(control_frame, textvariable=self.log_filter_var,
                                   values=["ALL", "INFO", "WARNING", "ERROR"],
                                   width=10)
        filter_combo.pack(side=tk.LEFT)
        filter_combo.bind("<<ComboboxSelected>>", self._apply_log_filter)
        
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def _create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, expand=True)
    
    def _on_preset_select(self, event):
        """预设动作选择事件"""
        selection = self.preset_listbox.curselection()
        if selection:
            index = selection[0]
            action = self.preset_actions[index]
            self.preset_desc_var.set(action["description"])
    
    def _on_preset_double_click(self, event):
        """预设动作双击执行"""
        selection = self.preset_listbox.curselection()
        if selection:
            index = selection[0]
            action = self.preset_actions[index]
            self._execute_preset_action(action)
    
    def _execute_preset_action(self, action: Dict):
        """执行预设动作"""
        self.status_var.set(f"执行：{action['name']}")
        
        # 更新配置
        self.config.base_url = self.url_var.get().rstrip('/')
        self.config.api_key = self.key_var.get()
        self.config.model = self.model_var.get()
        
        # 解析自定义 headers
        try:
            headers_json = self.headers_text.get("1.0", "end-1c").strip()
            if headers_json:
                custom_headers = json.loads(headers_json)
                for k, v in custom_headers.items():
                    self.config.add_custom_header(k, v)
        except json.JSONDecodeError:
            self._add_log("WARNING", "Custom headers JSON 格式错误")
        
        # 构建 URL
        url = f"{self.config.base_url}{action['endpoint']}"
        
        # 准备请求体
        body = action['body']
        if body and 'messages' in body and not body['messages']:
            # 如果是空消息，使用当前输入
            user_input = self.input_text.get("1.0", "end-1c").strip()
            if user_input:
                body['messages'] = [format_message("user", user_input)]
        
        # 执行请求
        if action['method'] == "GET":
            self._send_get_request(url, action['name'])
        else:
            self._send_post_request(url, body, action['name'])
    
    def _send_get_request(self, url: str, action_name: str):
        """发送 GET 请求"""
        headers = self.config.get_headers()
        
        async def run():
            try:
                async for result in self.api_client.send_request(
                    url=url, method="GET", headers=headers, timeout=self.config.timeout
                ):
                    self.root.after(0, self._handle_response, result, action_name)
            except Exception as e:
                self.root.after(0, self._handle_error, str(e))
        
        self._run_async(run)
    
    def _send_post_request(self, url: str, body: Optional[Dict], action_name: str):
        """发送 POST 请求"""
        headers = self.config.get_headers()
        
        if body is None:
            body = {}
        
        # 应用当前参数
        if 'temperature' not in body:
            body['temperature'] = self.temp_var.get()
        if 'max_tokens' not in body:
            body['max_tokens'] = self.max_tokens_var.get()
        body['stream'] = self.stream_var.get()
        if 'model' not in body or not body['model']:
            body['model'] = self.config.model
        
        self._add_log("INFO", f"发送请求到 {url}", {"body": body})
        
        async def run():
            try:
                if body.get('stream', False):
                    self.is_streaming = True
                    self.speed_calc.reset()
                    full_content = ""
                    
                    async for chunk in self.api_client.send_chat_request(
                        url=url, headers=headers, body=body
                    ):
                        if 'error' in chunk:
                            self.root.after(0, self._handle_error, chunk['error'])
                            break
                        
                        # 提取内容
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                full_content += content
                                self.speed_calc.add_token()
                                self.root.after(0, self._append_response, content)
                                self.root.after(0, self._update_speed)
                    
                    if full_content:
                        self.history.add("assistant", full_content)
                        self.root.after(0, self._add_log, "INFO", "响应完成", 
                                      {"content_length": len(full_content)})
                    
                    self.is_streaming = False
                    self.root.after(0, self.status_var.set, "就绪")
                else:
                    # 非流式请求
                    async for result in self.api_client.send_request(
                        url=url, method="POST", headers=headers, body=body,
                        timeout=self.config.timeout
                    ):
                        self.root.after(0, self._handle_response, result, action_name)
            except Exception as e:
                self.root.after(0, self._handle_error, str(e))
        
        self._run_async(run)
    
    def _send_message(self):
        """发送用户消息"""
        if self.is_streaming:
            return
        
        user_input = self.input_text.get("1.0", "end-1c").strip()
        if not user_input:
            return
        
        # 添加到历史
        self.history.add("user", user_input)
        self._append_message("user", user_input)
        self.input_text.delete("1.0", tk.END)
        
        # 构建请求
        self.config.base_url = self.url_var.get().rstrip('/')
        self.config.api_key = self.key_var.get()
        self.config.model = self.model_var.get()
        
        url = f"{self.config.base_url}/chat/completions"
        body = build_request_body(
            self.history.get_messages(),
            {
                "temperature": self.temp_var.get(),
                "max_tokens": self.max_tokens_var.get(),
                "stream": self.stream_var.get()
            },
            self.config.model
        )
        
        self._send_post_request(url, body, "Chat")
    
    def _run_async(self, coro):
        """在后台线程运行异步代码"""
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(coro())
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
    
    def _handle_response(self, result: Dict, action_name: str):
        """处理响应"""
        if 'error' in result:
            self._handle_error(result['error'])
            return
        
        self._add_log("INFO", f"{action_name} 成功", {"result": result})
        
        # 显示结果
        if 'data' in result and isinstance(result['data'], list):
            # 模型列表
            models = [m.get('id', 'unknown') for m in result['data']]
            self._append_message("system", f"可用模型：{', '.join(models)}")
        elif 'choices' in result:
            # 聊天完成（非流式）
            content = result['choices'][0]['message']['content']
            self.history.add("assistant", content)
            self._append_message("assistant", content)
        else:
            # 其他响应
            self._append_message("system", json.dumps(result, indent=2, ensure_ascii=False))
    
    def _handle_error(self, error: str):
        """处理错误"""
        self._add_log("ERROR", error)
        self._append_message("system", f"错误：{error}")
        self.status_var.set("错误")
        self.is_streaming = False
    
    def _append_message(self, role: str, content: str):
        """追加消息到对话显示"""
        self.chat_display.config(state='normal')
        
        # 颜色标记
        color_map = {
            "user": "blue",
            "assistant": "black",
            "system": "gray"
        }
        
        self.chat_display.insert(tk.END, f"\n{role.upper()}:\n", (role,))
        self.chat_display.tag_configure(role, foreground=color_map.get(role, "black"))
        self.chat_display.insert(tk.END, f"{content}\n")
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def _append_response(self, content: str):
        """追加流式响应内容"""
        self.chat_display.config(state='normal')
        
        # 检查最后一行是否是 assistant
        last_index = self.chat_display.index("end-1c")
        prev_line = self.chat_display.get("end-2c", "end-1c")
        
        if "ASSISTANT:" not in prev_line:
            self.chat_display.insert(tk.END, f"\nASSISTANT:\n", ("assistant",))
            self.chat_display.tag_configure("assistant", foreground="black")
        
        self.chat_display.insert(tk.END, content)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def _update_speed(self):
        """更新速度显示"""
        speed = self.speed_calc.get_speed()
        self.speed_var.set(f"Speed: {speed:.1f} tokens/s")
    
    def _clear_chat(self):
        """清空对话"""
        self.history.clear()
        self.chat_display.config(state='normal')
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state='disabled')
        self.speed_calc.reset()
        self.speed_var.set("Speed: 0.0 tokens/s")
    
    def _add_log(self, level: str, message: str, details: Optional[Dict] = None):
        """添加日志"""
        entry = LogEntry(level, message, details)
        self.logs.append(entry)
        self._display_log(entry)
    
    def _display_log(self, entry: LogEntry):
        """显示日志"""
        filter_level = self.log_filter_var.get()
        if filter_level != "ALL" and entry.level != filter_level:
            return
        
        self.log_display.config(state='normal')
        log_str = entry.to_string() + "\n" + "-" * 50 + "\n"
        self.log_display.insert(tk.END, log_str)
        self.log_display.config(state='disabled')
        self.log_display.see(tk.END)
    
    def _apply_log_filter(self, event=None):
        """应用日志过滤"""
        self.log_display.config(state='normal')
        self.log_display.delete("1.0", tk.END)
        self.log_display.config(state='disabled')
        
        for entry in self.logs:
            self._display_log(entry)
    
    def _clear_logs(self):
        """清空日志"""
        self.logs = []
        self.log_display.config(state='normal')
        self.log_display.delete("1.0", tk.END)
        self.log_display.config(state='disabled')
    
    def _export_logs(self):
        """导出日志"""
        file_path = simpledialog.askstring("导出日志", "请输入文件名:", 
                                          initialvalue="openai_debug_logs.txt")
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for entry in self.logs:
                    f.write(entry.to_string() + "\n" + "=" * 50 + "\n")
            messagebox.showinfo("成功", f"日志已导出到 {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败：{str(e)}")
    
    def _show_about(self):
        """显示关于对话框"""
        messagebox.showinfo("关于", 
                           "OpenAI API Debugger\n\n"
                           "一个用于调试 OpenAI 标准接口的图形化工具\n"
                           "支持实时速度显示、日志记录、预设动作等功能")
    
    def _on_close(self):
        """关闭窗口"""
        async def cleanup():
            await self.api_client.close()
        
        # 同步调用清理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(cleanup())
        finally:
            loop.close()
        
        self.root.destroy()


def main():
    """主入口"""
    root = tk.Tk()
    
    # 设置样式
    style = ttk.Style()
    style.theme_use('clam')  # 使用更现代的主题
    
    app = OpenAIDebuggerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
