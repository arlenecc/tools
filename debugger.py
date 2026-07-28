"""
OpenAI API Debugger Tool
A GUI tool for debugging OpenAI-compatible API endpoints

Usage:
    python debugger.py              # Launch GUI (requires tkinter)
    python debugger.py --cli        # Launch CLI mode (no GUI required)
    python debugger.py --help       # Show help
"""
import asyncio
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from collections.abc import AsyncGenerator
from enum import Enum
import argparse
import sys

# Try to import tkinter, fall back to CLI mode if not available
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Note: tkinter not available. GUI mode disabled. Use --cli for command-line mode.")

import threading


# ============================================================================
# Data Classes and Enums
# ============================================================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class LogEntry:
    """Represents a log entry"""
    level: LogLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "details": self.details
        }


@dataclass
class Message:
    """Represents a chat message"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DebuggerConfig:
    """Configuration for the debugger"""
    api_base: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = True
    timeout: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
            "timeout": self.timeout
        }


# ============================================================================
# Core Components
# ============================================================================

class ConversationHistory:
    """Manages conversation history"""
    
    def __init__(self):
        self.messages: List[Message] = []
    
    def add_message(self, role: str, content: str) -> Message:
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        return msg
    
    def clear(self):
        self.messages = []
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]
    
    def export_to_json(self) -> str:
        data = {"messages": [m.to_dict() for m in self.messages]}
        return json.dumps(data, indent=2)
    
    @classmethod
    def import_from_json(cls, json_data: str) -> 'ConversationHistory':
        data = json.loads(json_data)
        history = cls()
        for msg_data in data.get("messages", []):
            msg = Message(
                role=msg_data["role"],
                content=msg_data["content"]
            )
            history.messages.append(msg)
        return history


class LogManager:
    """Manages log entries"""
    
    def __init__(self, max_logs: int = 1000):
        self.logs: List[LogEntry] = []
        self.max_logs = max_logs
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
    
    def add_log(self, level: LogLevel, message: str, details: Optional[Dict] = None):
        with self._lock:
            entry = LogEntry(level=level, message=message, details=details)
            self.logs.append(entry)
            
            # Trim old logs if needed
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(entry)
                except Exception:
                    pass
    
    def get_logs(self, level_filter: Optional[LogLevel] = None) -> List[LogEntry]:
        with self._lock:
            if level_filter is None:
                return list(self.logs)
            return [log for log in self.logs if log.level == level_filter]
    
    def clear(self):
        with self._lock:
            self.logs = []
    
    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)


class TokenSpeedCalculator:
    """Calculates token generation speed"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.token_count: int = 0
        self.tokens: List[str] = []
        self._lock = threading.Lock()
    
    def start(self):
        with self._lock:
            self.start_time = time.time()
            self.token_count = 0
            self.tokens = []
    
    def add_token(self, token: str):
        with self._lock:
            if self.start_time is None:
                self.start_time = time.time()
            self.token_count += 1
            self.tokens.append(token)
    
    def get_speed(self) -> float:
        with self._lock:
            if self.start_time is None or self.token_count == 0:
                return 0.0
            
            elapsed = time.time() - self.start_time
            if elapsed <= 0:
                return 0.0
            
            return self.token_count / elapsed
    
    def get_total_tokens(self) -> int:
        with self._lock:
            return self.token_count
    
    def reset(self):
        with self._lock:
            self.start_time = None
            self.token_count = 0
            self.tokens = []


class ResponseParser:
    """Parses streaming API responses"""
    
    def parse_chunk(self, chunk: str) -> str:
        """Parse a SSE chunk and extract content"""
        try:
            if not chunk.startswith('data: '):
                return ""
            
            data = chunk[6:].strip()
            
            if data == '[DONE]':
                return ""
            
            parsed = json.loads(data)
            choices = parsed.get('choices', [])
            
            if not choices:
                return ""
            
            delta = choices[0].get('delta', {})
            content = delta.get('content', '')
            
            return content if content else ""
        except (json.JSONDecodeError, KeyError, IndexError):
            return ""


class RequestBuilder:
    """Builds API request payloads"""
    
    def __init__(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = True
    ):
        self.messages = messages
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream
    
    def build(self) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream
        }
        return payload


def get_preset_test_cases() -> List[Dict[str, Any]]:
    """Returns preset test cases for quick testing"""
    return [
        {
            "name": "Simple Hello",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "params": {
                "max_tokens": 100,
                "temperature": 0.7
            }
        },
        {
            "name": "Code Generation",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
            ],
            "params": {
                "max_tokens": 500,
                "temperature": 0.3
            }
        },
        {
            "name": "Translation",
            "messages": [
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": "Translate 'Hello, world!' to French."}
            ],
            "params": {
                "max_tokens": 100,
                "temperature": 0.5
            }
        },
        {
            "name": "JSON Output",
            "messages": [
                {"role": "system", "content": "Output only valid JSON."},
                {"role": "user", "content": "Generate a JSON object with name, age, and city fields."}
            ],
            "params": {
                "max_tokens": 200,
                "temperature": 0.2
            }
        },
        {
            "name": "Long Context",
            "messages": [
                {"role": "user", "content": "Summarize the following text in one sentence:\n\n" + "Lorem ipsum dolor sit amet. " * 50}
            ],
            "params": {
                "max_tokens": 100,
                "temperature": 0.5
            }
        }
    ]


# ============================================================================
# API Client
# ============================================================================

class APIClient:
    """Async HTTP client for OpenAI-compatible APIs"""
    
    def __init__(self, config: DebuggerConfig, timeout: Optional[int] = None):
        self.config = config
        self.timeout = timeout or config.timeout
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            import aiohttp
            headers = {
                "Content-Type": "application/json"
            }
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _make_request(self, url: str, payload: Dict[str, Any]):
        """Make HTTP request and yield chunks"""
        session = await self._get_session()
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API Error: {response.status} - {error_text}")
            
            async for line in response.content:
                yield line
    
    async def send_message(self, user_message: str, conversation: ConversationHistory) -> AsyncGenerator[str, None]:
        """Send a message and stream the response"""
        parser = ResponseParser()
        
        # Add user message to conversation
        conversation.add_message("user", user_message)
        
        # Build request
        builder = RequestBuilder(
            messages=conversation.get_messages_for_api(),
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=self.config.stream
        )
        
        payload = builder.build()
        url = f"{self.config.api_base}/chat/completions"
        
        parser = ResponseParser()
        full_response = ""
        
        try:
            async for chunk in self._make_request(url, payload):
                chunk_str = chunk.decode('utf-8')
                content = parser.parse_chunk(chunk_str)
                
                if content:
                    full_response += content
                    yield content
        finally:
            # Add assistant response to conversation
            if full_response:
                conversation.add_message("assistant", full_response)
    
    async def send_message_no_stream(self, user_message: str, conversation: ConversationHistory) -> str:
        """Send a message without streaming"""
        # Add user message to conversation
        conversation.add_message("user", user_message)
        
        # Build request with stream=False
        builder = RequestBuilder(
            messages=conversation.get_messages_for_api(),
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=False
        )
        
        payload = builder.build()
        url = f"{self.config.api_base}/chat/completions"
        
        session = await self._get_session()
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API Error: {response.status} - {error_text}")
            
            data = await response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Add assistant response to conversation
            if content:
                conversation.add_message("assistant", content)
            
            return content


# Fix: Import AsyncGenerator properly
from typing import AsyncGenerator


# Re-define APIClient.send_message with proper import
class APIClient:
    """Async HTTP client for OpenAI-compatible APIs"""
    
    def __init__(self, config: DebuggerConfig, timeout: Optional[int] = None):
        self.config = config
        self.timeout = timeout or config.timeout
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            import aiohttp
            headers = {
                "Content-Type": "application/json"
            }
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _make_request(self, url: str, payload: Dict[str, Any]):
        """Make HTTP request and yield chunks"""
        session = await self._get_session()
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API Error: {response.status} - {error_text}")
            
            async for line in response.content:
                yield line
    
    async def send_message(self, user_message: str, conversation: ConversationHistory) -> AsyncGenerator[str, None]:
        """Send a message and stream the response"""
        parser = ResponseParser()
        
        # Add user message to conversation
        conversation.add_message("user", user_message)
        
        # Build request
        builder = RequestBuilder(
            messages=conversation.get_messages_for_api(),
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=self.config.stream
        )
        
        payload = builder.build()
        url = f"{self.config.api_base}/chat/completions"
        
        full_response = ""
        
        try:
            async for chunk in self._make_request(url, payload):
                chunk_str = chunk.decode('utf-8')
                content = parser.parse_chunk(chunk_str)
                
                if content:
                    full_response += content
                    yield content
        finally:
            # Add assistant response to conversation
            if full_response:
                conversation.add_message("assistant", full_response)
    
    async def send_message_no_stream(self, user_message: str, conversation: ConversationHistory) -> str:
        """Send a message without streaming"""
        # Add user message to conversation
        conversation.add_message("user", user_message)
        
        # Build request with stream=False
        builder = RequestBuilder(
            messages=conversation.get_messages_for_api(),
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=False
        )
        
        payload = builder.build()
        url = f"{self.config.api_base}/chat/completions"
        
        session = await self._get_session()
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API Error: {response.status} - {error_text}")
            
            data = await response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Add assistant response to conversation
            if content:
                conversation.add_message("assistant", content)
            
            return content


# ============================================================================
# GUI Application (only defined if tkinter is available)
# ============================================================================

if TKINTER_AVAILABLE:
    class OpenAIDebuggerGUI:
        """Main GUI application for OpenAI API debugging"""
        
        def __init__(self, root: tk.Tk):
            self.root = root
            self.root.title("OpenAI API Debugger")
            self.root.geometry("1200x800")
            
            # Initialize components
            self.config = DebuggerConfig()
            self.conversation = ConversationHistory()
            self.log_manager = LogManager()
            self.speed_calculator = TokenSpeedCalculator()
            self.api_client: Optional[APIClient] = None
            self.is_generating = False
            self.event_loop = None
            
            # Setup event loop for async operations
            self._setup_event_loop()
            
            # Build UI
            self._build_ui()
            
            # Register log callback
            self.log_manager.register_callback(self._on_log_entry)
            
            # Initial log
            self.log_manager.add_log(LogLevel.INFO, "Debugger initialized")
    
    def _setup_event_loop(self):
        """Setup asyncio event loop in a separate thread"""
        def run_loop():
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_forever()
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
    
    def _run_async(self, coro):
        """Run async coroutine in the event loop"""
        future = asyncio.run_coroutine_threadsafe(coro, self.event_loop)
        return future.result()
    
    def _build_ui(self):
        """Build the main UI"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Config panel
        self._build_config_panel(main_frame, 0)
        
        # Chat panel
        self._build_chat_panel(main_frame, 1)
        
        # Log panel
        self._build_log_panel(main_frame, 2)
        
        # Status bar
        self._build_status_bar(main_frame, 3)
    
    def _build_config_panel(self, parent, row):
        """Build configuration panel"""
        config_frame = ttk.LabelFrame(parent, text="Configuration", padding="5")
        config_frame.grid(row=row, column=0, sticky="ew", pady=(0, 5))
        config_frame.columnconfigure(1, weight=1)
        
        # API Base URL
        ttk.Label(config_frame, text="API Base:").grid(row=0, column=0, sticky="w", padx=5)
        self.api_base_var = tk.StringVar(value=self.config.api_base)
        api_base_entry = ttk.Entry(config_frame, textvariable=self.api_base_var, width=50)
        api_base_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        
        # API Key
        ttk.Label(config_frame, text="API Key:").grid(row=1, column=0, sticky="w", padx=5)
        self.api_key_var = tk.StringVar(value=self.config.api_key)
        api_key_entry = ttk.Entry(config_frame, textvariable=self.api_key_var, show="*", width=50)
        api_key_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        
        # Model
        ttk.Label(config_frame, text="Model:").grid(row=2, column=0, sticky="w", padx=5)
        self.model_var = tk.StringVar(value=self.config.model)
        model_entry = ttk.Entry(config_frame, textvariable=self.model_var, width=50)
        model_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        
        # Max Tokens
        ttk.Label(config_frame, text="Max Tokens:").grid(row=3, column=0, sticky="w", padx=5)
        self.max_tokens_var = tk.IntVar(value=self.config.max_tokens)
        max_tokens_spin = ttk.Spinbox(config_frame, from_=1, to=32768, textvariable=self.max_tokens_var, width=20)
        max_tokens_spin.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        
        # Temperature
        ttk.Label(config_frame, text="Temperature:").grid(row=4, column=0, sticky="w", padx=5)
        self.temperature_var = tk.DoubleVar(value=self.config.temperature)
        temp_scale = ttk.Scale(config_frame, from_=0.0, to=2.0, variable=self.temperature_var, length=200)
        temp_scale.grid(row=4, column=1, sticky="w", padx=5, pady=2)
        self.temp_value_label = ttk.Label(config_frame, text=f"{self.config.temperature:.2f}")
        self.temp_value_label.grid(row=4, column=2, sticky="w", padx=5)
        temp_scale.configure(command=lambda v: self.temp_value_label.configure(text=f"{float(v):.2f}"))
        
        # Top P
        ttk.Label(config_frame, text="Top P:").grid(row=5, column=0, sticky="w", padx=5)
        self.top_p_var = tk.DoubleVar(value=self.config.top_p)
        top_p_scale = ttk.Scale(config_frame, from_=0.0, to=1.0, variable=self.top_p_var, length=200)
        top_p_scale.grid(row=5, column=1, sticky="w", padx=5, pady=2)
        self.top_p_value_label = ttk.Label(config_frame, text=f"{self.config.top_p:.2f}")
        self.top_p_value_label.grid(row=5, column=2, sticky="w", padx=5)
        top_p_scale.configure(command=lambda v: self.top_p_value_label.configure(text=f"{float(v):.2f}"))
        
        # Stream checkbox
        self.stream_var = tk.BooleanVar(value=self.config.stream)
        stream_check = ttk.Checkbutton(config_frame, text="Stream Response", variable=self.stream_var)
        stream_check.grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        # Preset test cases
        ttk.Label(config_frame, text="Quick Tests:").grid(row=0, column=3, sticky="w", padx=(20, 5))
        self.preset_var = tk.StringVar()
        presets = get_preset_test_cases()
        preset_names = [p["name"] for p in presets]
        preset_combo = ttk.Combobox(config_frame, textvariable=self.preset_var, values=preset_names, width=30, state="readonly")
        preset_combo.grid(row=1, column=3, sticky="ew", padx=5, pady=2)
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)
        
        # Action buttons
        btn_frame = ttk.Frame(config_frame)
        btn_frame.grid(row=2, column=3, rowspan=5, sticky="ns", padx=(20, 5))
        
        ttk.Button(btn_frame, text="Apply Config", command=self._apply_config).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Clear Chat", command=self._clear_chat).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Export Chat", command=self._export_chat).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Import Chat", command=self._import_chat).pack(fill="x", pady=2)
    
    def _build_chat_panel(self, parent, row):
        """Build chat panel"""
        chat_frame = ttk.LabelFrame(parent, text="Chat", padding="5")
        chat_frame.grid(row=row, column=0, sticky="nsew", pady=(0, 5))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=20, font=("Courier", 10))
        self.chat_display.grid(row=0, column=0, sticky="nsew")
        self.chat_display.configure(state='disabled')
        
        # Input frame
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        input_frame.columnconfigure(0, weight=1)
        
        # Message input
        self.message_input = scrolledtext.ScrolledText(input_frame, height=4, font=("Courier", 10))
        self.message_input.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.message_input.bind("<Control-Return>", lambda e: self._send_message())
        
        # Send button
        self.send_button = ttk.Button(input_frame, text="Send (Ctrl+Enter)", command=self._send_message)
        self.send_button.grid(row=0, column=1)
        
        # Stop button
        self.stop_button = ttk.Button(input_frame, text="Stop", command=self._stop_generation, state='disabled')
        self.stop_button.grid(row=0, column=2, padx=(5, 0))
    
    def _build_log_panel(self, parent, row):
        """Build log panel"""
        log_frame = ttk.LabelFrame(parent, text="Logs", padding="5")
        log_frame.grid(row=row, column=0, sticky="nsew", pady=(0, 5), ipady=100)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)
        
        # Log display
        self.log_display = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8, font=("Courier", 9))
        self.log_display.grid(row=0, column=0, sticky="nsew")
        self.log_display.configure(state='disabled')
        
        # Log controls
        ctrl_frame = ttk.Frame(log_frame)
        ctrl_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        ttk.Button(ctrl_frame, text="Clear Logs", command=self._clear_logs).pack(side="left")
        
        # Filter options
        ttk.Label(ctrl_frame, text="Filter:").pack(side="left", padx=(20, 5))
        self.log_filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(ctrl_frame, textvariable=self.log_filter_var, values=["All", "DEBUG", "INFO", "WARNING", "ERROR"], width=10, state="readonly")
        filter_combo.pack(side="left")
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_logs())
    
    def _build_status_bar(self, parent, row):
        """Build status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=row, column=0, sticky="ew")
        
        # Token speed
        ttk.Label(status_frame, text="Token Speed:").pack(side="left", padx=5)
        self.speed_label = ttk.Label(status_frame, text="0.0 tokens/s", font=("Courier", 10))
        self.speed_label.pack(side="left", padx=5)
        
        # Total tokens
        ttk.Label(status_frame, text="Total Tokens:").pack(side="left", padx=(20, 5))
        self.tokens_label = ttk.Label(status_frame, text="0", font=("Courier", 10))
        self.tokens_label.pack(side="left", padx=5)
        
        # Status
        self.status_label = ttk.Label(status_frame, text="Ready", relief="sunken", anchor="w")
        self.status_label.pack(side="right", fill="x", expand=True, padx=5)
    
    def _apply_config(self):
        """Apply configuration changes"""
        self.config.api_base = self.api_base_var.get()
        self.config.api_key = self.api_key_var.get()
        self.config.model = self.model_var.get()
        self.config.max_tokens = self.max_tokens_var.get()
        self.config.temperature = self.temperature_var.get()
        self.config.top_p = self.top_p_var.get()
        self.config.stream = self.stream_var.get()
        
        self.log_manager.add_log(LogLevel.INFO, "Configuration applied", self.config.to_dict())
        messagebox.showinfo("Success", "Configuration applied successfully!")
    
    def _clear_chat(self):
        """Clear chat history"""
        self.conversation.clear()
        self.chat_display.configure(state='normal')
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.configure(state='disabled')
        self.log_manager.add_log(LogLevel.INFO, "Chat cleared")
    
    def _export_chat(self):
        """Export chat to file"""
        from tkinter import filedialog
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(self.conversation.export_to_json())
                self.log_manager.add_log(LogLevel.INFO, f"Chat exported to {filepath}")
                messagebox.showinfo("Success", f"Chat exported to {filepath}")
            except Exception as e:
                self.log_manager.add_log(LogLevel.ERROR, f"Export failed: {str(e)}")
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def _import_chat(self):
        """Import chat from file"""
        from tkinter import filedialog
        
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    json_data = f.read()
                self.conversation = ConversationHistory.import_from_json(json_data)
                self._refresh_chat_display()
                self.log_manager.add_log(LogLevel.INFO, f"Chat imported from {filepath}")
                messagebox.showinfo("Success", f"Chat imported from {filepath}")
            except Exception as e:
                self.log_manager.add_log(LogLevel.ERROR, f"Import failed: {str(e)}")
                messagebox.showerror("Error", f"Failed to import: {str(e)}")
    
    def _on_preset_selected(self, event):
        """Handle preset selection"""
        preset_name = self.preset_var.get()
        presets = get_preset_test_cases()
        
        for preset in presets:
            if preset["name"] == preset_name:
                # Load messages
                self.conversation.clear()
                self._refresh_chat_display()
                
                # Apply params
                params = preset.get("params", {})
                if "max_tokens" in params:
                    self.max_tokens_var.set(params["max_tokens"])
                if "temperature" in params:
                    self.temperature_var.set(params["temperature"])
                
                # Display messages
                for msg in preset.get("messages", []):
                    self.conversation.add_message(msg["role"], msg["content"])
                
                self._refresh_chat_display()
                self.log_manager.add_log(LogLevel.INFO, f"Preset loaded: {preset_name}")
                break
    
    def _send_message(self):
        """Send message to API"""
        if self.is_generating:
            return
        
        message = self.message_input.get(1.0, tk.END).strip()
        if not message:
            return
        
        if not self.config.model:
            messagebox.showwarning("Warning", "Please specify a model name")
            return
        
        # Clear input
        self.message_input.delete(1.0, tk.END)
        
        # Update UI
        self.is_generating = True
        self.send_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status_label.configure(text="Generating...")
        
        # Reset speed calculator
        self.speed_calculator.reset()
        self.speed_calculator.start()
        
        # Start async generation
        def generate():
            try:
                self.api_client = APIClient(self.config)
                
                if self.config.stream:
                    # Streaming mode
                    future = asyncio.run_coroutine_threadsafe(
                        self._stream_generation(message),
                        self.event_loop
                    )
                    future.result()
                else:
                    # Non-streaming mode
                    future = asyncio.run_coroutine_threadsafe(
                        self._non_stream_generation(message),
                        self.event_loop
                    )
                    response = future.result()
                    self.root.after(0, lambda: self._append_to_chat("assistant", response))
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.root.after(0, lambda: self._append_to_chat("assistant", error_msg))
                self.root.after(0, lambda: self.log_manager.add_log(LogLevel.ERROR, str(e)))
            finally:
                self.root.after(0, self._generation_complete)
        
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
    
    async def _stream_generation(self, message: str):
        """Stream generation with real-time updates"""
        full_response = ""
        
        async for chunk in self.api_client.send_message(message, self.conversation):
            self.speed_calculator.add_token(chunk)
            full_response += chunk
            
            # Update UI in main thread
            def update(chunk_text):
                self._append_to_chat_partial(chunk_text)
                self._update_speed_display()
            
            self.root.after(0, update, chunk)
        
        return full_response
    
    async def _non_stream_generation(self, message: str) -> str:
        """Non-streaming generation"""
        return await self.api_client.send_message_no_stream(message, self.conversation)
    
    def _stop_generation(self):
        """Stop current generation"""
        self.is_generating = False
        self._generation_complete()
        self.log_manager.add_log(LogLevel.WARNING, "Generation stopped by user")
    
    def _generation_complete(self):
        """Handle generation completion"""
        self.is_generating = False
        self.send_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.status_label.configure(text="Ready")
        self._update_speed_display()
    
    def _append_to_chat(self, role: str, content: str):
        """Append complete message to chat"""
        self.chat_display.configure(state='normal')
        
        prefix = "👤 You: " if role == "user" else "🤖 Assistant: "
        self.chat_display.insert(tk.END, f"\n{prefix}\n{content}\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')
    
    def _append_to_chat_partial(self, content: str):
        """Append partial content for streaming"""
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, content)
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')
    
    def _refresh_chat_display(self):
        """Refresh entire chat display"""
        self.chat_display.configure(state='normal')
        self.chat_display.delete(1.0, tk.END)
        
        for msg in self.conversation.messages:
            prefix = "👤 You: " if msg.role == "user" else "🤖 Assistant: "
            self.chat_display.insert(tk.END, f"\n{prefix}\n{msg.content}\n")
        
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')
    
    def _update_speed_display(self):
        """Update speed display"""
        speed = self.speed_calculator.get_speed()
        total = self.speed_calculator.get_total_tokens()
        
        self.speed_label.configure(text=f"{speed:.1f} tokens/s")
        self.tokens_label.configure(text=str(total))
    
    def _on_log_entry(self, entry: LogEntry):
        """Handle new log entry"""
        self.root.after(0, self._refresh_logs)
    
    def _refresh_logs(self):
        """Refresh log display"""
        self.log_display.configure(state='normal')
        self.log_display.delete(1.0, tk.END)
        
        filter_level = self.log_filter_var.get()
        
        if filter_level == "All":
            logs = self.log_manager.get_logs()
        else:
            try:
                level = LogLevel[filter_level]
                logs = self.log_manager.get_logs(level_filter=level)
            except KeyError:
                logs = self.log_manager.get_logs()
        
        for log in logs[-100:]:  # Show last 100 logs
            timestamp = log.timestamp.strftime("%H:%M:%S")
            self.log_display.insert(tk.END, f"[{timestamp}] [{log.level.value}] {log.message}\n")
            
            if log.details:
                details_str = json.dumps(log.details, indent=2)
                self.log_display.insert(tk.END, f"  {details_str}\n")
        
        self.log_display.see(tk.END)
        self.log_display.configure(state='disabled')
    
    def _clear_logs(self):
        """Clear logs"""
        self.log_manager.clear()
        self._refresh_logs()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="OpenAI API Debugger Tool - Debug OpenAI-compatible API endpoints"
    )
    parser.add_argument(
        "--cli", 
        action="store_true",
        help="Run in command-line mode (no GUI)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="API Base URL (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--key",
        type=str,
        default="",
        help="API Key"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model name (default: gpt-3.5-turbo)"
    )
    
    args = parser.parse_args()
    
    if args.cli or not TKINTER_AVAILABLE:
        run_cli_mode(args)
    else:
        run_gui_mode()


def run_cli_mode(args):
    """Run the debugger in command-line mode"""
    print("=" * 60)
    print("OpenAI API Debugger - CLI Mode")
    print("=" * 60)
    print(f"API URL: {args.url}")
    print(f"Model: {args.model}")
    print("-" * 60)
    
    # Create config
    config = DebuggerConfig(
        api_base=args.url,
        api_key=args.key if args.key else "sk-test-key",
        model=args.model
    )
    
    # Create client and conversation
    client = APIClient(config)
    conversation = ConversationHistory()
    
    async def run_test():
        print("\nSending test request...")
        
        try:
            response_text = ""
            async for chunk in client.send_message("Say hello in one sentence.", conversation):
                response_text += chunk
                print(chunk, end="", flush=True)
            
            print("\n\n" + "-" * 60)
            print("Response received successfully!")
            return True
        except Exception as e:
            print(f"\nError: {e}")
            return False
    
    # Run async test
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)


def run_gui_mode():
    """Run the debugger in GUI mode"""
    root = tk.Tk()
    
    # Set theme
    style = ttk.Style()
    try:
        style.theme_use('clam')  # Modern look
    except:
        pass
    
    app = OpenAIDebuggerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
