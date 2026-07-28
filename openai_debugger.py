"""
OpenAI API Debugger Tool
A GUI tool for debugging OpenAI-compatible API services with real-time streaming,
speed calculation, and conversation management.
"""
import json
import time
import threading
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import re

# Tkinter is optional for testing
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# ============================================================================
# Core Data Classes
# ============================================================================

@dataclass
class LogEntry:
    """Represents a log entry"""
    timestamp: datetime
    level: str
    message: str
    
    def __str__(self) -> str:
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {self.level}: {self.message}"


@dataclass
class APIConfig:
    """API configuration settings"""
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = True
    
    def update(self, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return bool(self.base_url)


@dataclass
class SpeedCalculator:
    """Calculate token generation speed in real-time"""
    start_time: float = 0.0
    total_tokens: int = 0
    token_batches: List[tuple] = field(default_factory=list)
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.total_tokens = 0
        self.token_batches = []
    
    def update_tokens(self, tokens: int):
        """Update token count"""
        current_time = time.time()
        self.total_tokens += tokens
        self.token_batches.append((current_time, tokens))
        
        # Keep only last 2 seconds of data for realtime calculation
        cutoff = current_time - 2.0
        self.token_batches = [(t, n) for t, n in self.token_batches if t > cutoff]
    
    def get_current_speed(self) -> float:
        """Get current tokens/second"""
        if not self.token_batches or len(self.token_batches) < 2:
            if self.start_time > 0 and self.total_tokens > 0:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    return self.total_tokens / elapsed
            return 0.0
        
        # Calculate speed from recent batches
        first_time = self.token_batches[0][0]
        last_time = self.token_batches[-1][0]
        elapsed = last_time - first_time
        
        if elapsed <= 0:
            return 0.0
        
        recent_tokens = sum(n for _, n in self.token_batches)
        return recent_tokens / elapsed


# ============================================================================
# Utility Functions
# ============================================================================

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    if not text:
        return 0
    
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    
    # Fallback: estimate ~4 characters per token
    return max(1, len(text) // 4)


def calculate_speed(tokens: int, elapsed_time: float) -> float:
    """Calculate tokens per second"""
    if elapsed_time <= 0:
        return 0.0
    return tokens / elapsed_time


def format_message(role: str, content: str) -> str:
    """Format a message for display"""
    role_display = role.capitalize()
    return f"{role_display}: {content}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_timestamp(dt: datetime) -> str:
    """Format datetime as string"""
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse Server-Sent Events line"""
    if not line.startswith('data: '):
        return None
    
    data = line[6:].strip()
    if data == '[DONE]':
        return None
    
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def extract_content_from_response(response: Dict[str, Any]) -> str:
    """Extract content from API response"""
    if not response or 'choices' not in response:
        return ""
    
    choices = response['choices']
    if not choices:
        return ""
    
    choice = choices[0]
    
    # Try delta first (streaming)
    if 'delta' in choice and choice['delta']:
        return choice['delta'].get('content', '')
    
    # Try message (non-streaming)
    if 'message' in choice and choice['message']:
        return choice['message'].get('content', '')
    
    return ""


def format_api_error(error: Exception) -> str:
    """Format API error for display"""
    error_str = str(error)
    return f"Error: {error_str}"


def get_preset_templates() -> List[Dict[str, Any]]:
    """Get preset conversation templates"""
    return [
        {
            "name": "Simple Hello",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        },
        {
            "name": "System Instruction",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What can you do?"}
            ]
        },
        {
            "name": "Code Review",
            "messages": [
                {"role": "system", "content": "You are a code review expert."},
                {"role": "user", "content": "Please review this Python code:\n\ndef hello():\n    print('world')"}
            ]
        },
        {
            "name": "Translation Test",
            "messages": [
                {"role": "system", "content": "You are a translator."},
                {"role": "user", "content": "Translate 'Hello world' to Chinese"}
            ]
        },
        {
            "name": "JSON Response",
            "messages": [
                {"role": "system", "content": "Always respond in JSON format."},
                {"role": "user", "content": "Give me a sample user object"}
            ]
        }
    ]


# ============================================================================
# Conversation Management
# ============================================================================

class ConversationHistory:
    """Manage conversation history"""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to history"""
        self.messages.append({"role": role, "content": content})
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for API request"""
        return self.messages.copy()
    
    def get_last_message(self) -> Optional[Dict[str, str]]:
        """Get the last message"""
        if self.messages:
            return self.messages[-1]
        return None


# ============================================================================
# API Client
# ============================================================================

class APIClient:
    """Async API client for OpenAI-compatible services"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = None
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                             on_token: Callable[[str], None] = None,
                             on_complete: Callable[[str], None] = None,
                             on_error: Callable[[Exception], None] = None):
        """Send chat completion request"""
        import aiohttp
        
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": self.config.stream
        }
        
        full_response = ""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
                    if self.config.stream:
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            parsed = parse_sse_line(line)
                            if parsed:
                                content = extract_content_from_response(parsed)
                                if content:
                                    full_response += content
                                    if on_token:
                                        on_token(content)
                        
                        if on_complete:
                            on_complete(full_response)
                    else:
                        data = await response.json()
                        content = extract_content_from_response(data)
                        full_response = content
                        if on_complete:
                            on_complete(full_response)
                            
        except Exception as e:
            if on_error:
                on_error(e)
            raise


# ============================================================================
# GUI Application
# ============================================================================

class OpenAIDebuggerApp:
    """Main application GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OpenAI API Debugger")
        self.root.geometry("1200x800")
        
        # State
        self.config = APIConfig()
        self.history = ConversationHistory()
        self.speed_calc = SpeedCalculator()
        self.is_streaming = False
        self.logs: List[LogEntry] = []
        
        # Setup UI
        self._setup_styles()
        self._create_menu()
        self._create_layout()
        self._load_presets()
        
        # Logging
        self._add_log("INFO", "Application started")
    
    def _setup_styles(self):
        """Configure UI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'bg': '#f0f0f0',
            'user_msg': '#e3f2fd',
            'assistant_msg': '#f5f5f5',
            'system_msg': '#fff3e0',
            'error': '#ffebee',
            'success': '#e8f5e9'
        }
    
    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Conversation", command=self._new_conversation)
        file_menu.add_command(label="Export Logs", command=self._export_logs)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="API Configuration", command=self._show_config_dialog)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_layout(self):
        """Create main layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Configuration & Presets
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.grid(row=0, column=0, rowspan=3, sticky="ns", padx=(0, 5))
        left_panel.grid_propagate(False)
        
        self._create_config_panel(left_panel)
        
        # Right panel - Chat & Logs
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        self._create_chat_panel(right_panel)
        
        # Bottom panel - Logs
        bottom_panel = ttk.Frame(main_frame)
        bottom_panel.grid(row=2, column=1, sticky="nsew", pady=(5, 0))
        bottom_panel.columnconfigure(0, weight=1)
        bottom_panel.rowconfigure(1, weight=1)
        
        self._create_log_panel(bottom_panel)
    
    def _create_config_panel(self, parent):
        """Create configuration panel"""
        config_frame = ttk.LabelFrame(parent, text="API Configuration", padding="10")
        config_frame.pack(fill="x", pady=(0, 10))
        
        # Base URL
        ttk.Label(config_frame, text="Base URL:").grid(row=0, column=0, sticky="w", pady=2)
        self.url_var = tk.StringVar(value=self.config.base_url)
        url_entry = ttk.Entry(config_frame, textvariable=self.url_var, width=35)
        url_entry.grid(row=0, column=1, sticky="ew", pady=2)
        
        # API Key
        ttk.Label(config_frame, text="API Key:").grid(row=1, column=0, sticky="w", pady=2)
        self.key_var = tk.StringVar(value=self.config.api_key)
        key_entry = ttk.Entry(config_frame, textvariable=self.key_var, show="*", width=35)
        key_entry.grid(row=1, column=1, sticky="ew", pady=2)
        
        # Model
        ttk.Label(config_frame, text="Model:").grid(row=2, column=0, sticky="w", pady=2)
        self.model_var = tk.StringVar(value=self.config.model)
        model_entry = ttk.Entry(config_frame, textvariable=self.model_var, width=35)
        model_entry.grid(row=2, column=1, sticky="ew", pady=2)
        
        # Temperature
        ttk.Label(config_frame, text="Temperature:").grid(row=3, column=0, sticky="w", pady=2)
        self.temp_var = tk.DoubleVar(value=self.config.temperature)
        temp_scale = ttk.Scale(config_frame, from_=0.0, to=2.0, variable=self.temp_var, orient="horizontal")
        temp_scale.grid(row=3, column=1, sticky="ew", pady=2)
        self.temp_label = ttk.Label(config_frame, text=f"{self.config.temperature:.1f}")
        self.temp_label.grid(row=3, column=2, padx=5)
        temp_scale.configure(command=lambda v: self.temp_label.config(text=f"{float(v):.1f}"))
        
        # Max Tokens
        ttk.Label(config_frame, text="Max Tokens:").grid(row=4, column=0, sticky="w", pady=2)
        self.tokens_var = tk.IntVar(value=self.config.max_tokens)
        tokens_spinbox = ttk.Spinbox(config_frame, from_=1, to=32768, textvariable=self.tokens_var, width=10)
        tokens_spinbox.grid(row=4, column=1, sticky="w", pady=2)
        
        # Stream toggle
        self.stream_var = tk.BooleanVar(value=self.config.stream)
        stream_check = ttk.Checkbutton(config_frame, text="Stream Response", variable=self.stream_var)
        stream_check.grid(row=5, column=0, columnspan=2, sticky="w", pady=5)
        
        # Save button
        save_btn = ttk.Button(config_frame, text="Apply Settings", command=self._apply_settings)
        save_btn.grid(row=6, column=0, columnspan=3, pady=10)
        
        # Presets
        presets_frame = ttk.LabelFrame(parent, text="Quick Actions", padding="10")
        presets_frame.pack(fill="both", expand=True)
        
        self.preset_listbox = tk.Listbox(presets_frame, height=8)
        self.preset_listbox.pack(fill="x", pady=(0, 5))
        self.preset_listbox.bind('<<ListboxSelect>>', self._on_preset_select)
        
        load_btn = ttk.Button(presets_frame, text="Load Preset", command=self._load_preset)
        load_btn.pack(fill="x")
        
        # Speed display
        speed_frame = ttk.LabelFrame(parent, text="Performance", padding="10")
        speed_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(speed_frame, text="Tokens/sec:").grid(row=0, column=0, sticky="w")
        self.speed_label = ttk.Label(speed_frame, text="0.0", font=("Courier", 16, "bold"))
        self.speed_label.grid(row=0, column=1, sticky="e")
        
        ttk.Label(speed_frame, text="Total Tokens:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.total_tokens_label = ttk.Label(speed_frame, text="0", font=("Courier", 12))
        self.total_tokens_label.grid(row=1, column=1, sticky="e", pady=(5, 0))
    
    def _create_chat_panel(self, parent):
        """Create chat conversation panel"""
        # Header
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Label(header_frame, text="Conversation", font=("Arial", 12, "bold")).pack(side="left")
        
        clear_btn = ttk.Button(header_frame, text="Clear", command=self._clear_conversation)
        clear_btn.pack(side="right")
        
        # Chat display
        chat_frame = ttk.LabelFrame(parent, text="Messages", padding="5")
        chat_frame.grid(row=1, column=0, sticky="nsew")
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, state='disabled',
            font=("Consolas", 11), bg='#fafafa'
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew")
        
        # Configure tags for different message types
        self.chat_display.tag_configure('user', background=self.colors['user_msg'])
        self.chat_display.tag_configure('assistant', background=self.colors['assistant_msg'])
        self.chat_display.tag_configure('system', background=self.colors['system_msg'])
        self.chat_display.tag_configure('error', background=self.colors['error'])
        
        # Input area
        input_frame = ttk.Frame(parent)
        input_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        input_frame.columnconfigure(0, weight=1)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=4, font=("Consolas", 11))
        self.input_text.grid(row=0, column=0, sticky="ew")
        self.input_text.bind('<Shift-Return>', lambda e: None)  # Allow newlines
        self.input_text.bind('<Return>', self._on_enter_press)
        
        # Buttons
        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=1, column=0, sticky="e", pady=(5, 0))
        
        send_btn = ttk.Button(btn_frame, text="Send", command=self._send_message)
        send_btn.pack(side="left", padx=(0, 5))
        
        self.send_btn = send_btn  # Reference for enabling/disabling
        
        stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_generation)
        stop_btn.pack(side="left")
        self.stop_btn = stop_btn
        stop_btn.configure(state='disabled')
    
    def _create_log_panel(self, parent):
        """Create log panel"""
        log_frame = ttk.LabelFrame(parent, text="Interaction Logs", padding="5")
        log_frame.grid(row=0, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        
        # Log controls
        control_frame = ttk.Frame(log_frame)
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Label(control_frame, text="Filter:").pack(side="left")
        self.log_filter_var = tk.StringVar(value="ALL")
        filter_combo = ttk.Combobox(
            control_frame, 
            textvariable=self.log_filter_var,
            values=["ALL", "INFO", "SUCCESS", "ERROR", "REQUEST", "RESPONSE"],
            width=12,
            state="readonly"
        )
        filter_combo.pack(side="left", padx=(5, 10))
        filter_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_logs())
        
        clear_logs_btn = ttk.Button(control_frame, text="Clear Logs", command=self._clear_logs)
        clear_logs_btn.pack(side="right")
        
        # Log display
        self.log_display = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, state='disabled',
            font=("Consolas", 9), bg='#ffffff'
        )
        self.log_display.grid(row=1, column=0, sticky="nsew")
        
        # Configure log colors
        self.log_display.tag_configure('INFO', foreground='blue')
        self.log_display.tag_configure('SUCCESS', foreground='green')
        self.log_display.tag_configure('ERROR', foreground='red')
        self.log_display.tag_configure('REQUEST', foreground='purple')
        self.log_display.tag_configure('RESPONSE', foreground='orange')
    
    def _load_presets(self):
        """Load preset templates into listbox"""
        presets = get_preset_templates()
        for preset in presets:
            self.preset_listbox.insert(tk.END, preset["name"])
    
    def _add_log(self, level: str, message: str):
        """Add a log entry"""
        entry = LogEntry(datetime.now(), level, message)
        self.logs.append(entry)
        self._refresh_logs()
    
    def _refresh_logs(self):
        """Refresh log display"""
        self.log_display.configure(state='normal')
        self.log_display.delete(1.0, tk.END)
        
        filter_level = self.log_filter_var.get()
        
        for entry in self.logs:
            if filter_level == "ALL" or entry.level == filter_level:
                self.log_display.insert(tk.END, str(entry) + "\n", entry.level)
        
        self.log_display.configure(state='disabled')
        self.log_display.see(tk.END)
    
    def _apply_settings(self):
        """Apply configuration settings"""
        self.config.update(
            base_url=self.url_var.get(),
            api_key=self.key_var.get(),
            model=self.model_var.get(),
            temperature=self.temp_var.get(),
            max_tokens=self.tokens_var.get(),
            stream=self.stream_var.get()
        )
        self._add_log("INFO", f"Settings applied: {self.config.model} @ {self.config.base_url}")
        messagebox.showinfo("Success", "Settings applied successfully!")
    
    def _on_preset_select(self, event):
        """Handle preset selection"""
        pass  # Selection handled by load button
    
    def _load_preset(self):
        """Load selected preset"""
        selection = self.preset_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a preset")
            return
        
        index = selection[0]
        presets = get_preset_templates()
        preset = presets[index]
        
        self.history.clear()
        self.chat_display.configure(state='normal')
        self.chat_display.delete(1.0, tk.END)
        
        for msg in preset["messages"]:
            self.history.add_message(msg["role"], msg["content"])
            self._display_message(msg["role"], msg["content"])
        
        self._add_log("INFO", f"Preset loaded: {preset['name']}")
    
    def _display_message(self, role: str, content: str):
        """Display a message in chat"""
        self.chat_display.configure(state='normal')
        
        # Format and insert message
        formatted = f"{role.upper()}: {content}\n\n"
        self.chat_display.insert(tk.END, formatted, role)
        
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)
    
    def _on_enter_press(self, event):
        """Handle Enter key press"""
        if not event.state & 0x1:  # Shift not pressed
            self._send_message()
            return 'break'
        return None
    
    def _send_message(self):
        """Send message to API"""
        if self.is_streaming:
            return
        
        message = self.input_text.get(1.0, tk.END).strip()
        if not message:
            return
        
        # Add to history and display
        self.history.add_message("user", message)
        self._display_message("user", message)
        
        # Clear input
        self.input_text.delete(1.0, tk.END)
        
        # Prepare for response
        self.is_streaming = True
        self.send_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.speed_calc.start()
        
        # Log request
        self._add_log("REQUEST", f"Sending to {self.config.model}: {truncate_text(message, 50)}")
        
        # Start async request
        self._run_async_request()
    
    def _run_async_request(self):
        """Run async API request in thread"""
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                client = APIClient(self.config)
                messages = self.history.get_messages_for_api()
                
                def on_token(token: str):
                    self.root.after(0, lambda: self._on_token_received(token))
                
                def on_complete(response: str):
                    self.root.after(0, lambda: self._on_complete(response))
                
                def on_error(error: Exception):
                    self.root.after(0, lambda: self._on_error(error))
                
                loop.run_until_complete(
                    client.chat_completion(
                        messages,
                        on_token=on_token,
                        on_complete=on_complete,
                        on_error=on_error
                    )
                )
            except Exception as e:
                self.root.after(0, lambda: self._on_error(e))
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        
        # Start speed update loop
        self._update_speed_display()
    
    def _on_token_received(self, token: str):
        """Handle received token"""
        self.speed_calc.update_tokens(count_tokens(token))
        
        # Append to chat
        self.chat_display.configure(state='normal')
        
        # Check if we need to start a new tag
        last_msg = self.history.get_last_message()
        if not last_msg or last_msg["role"] != "assistant":
            self.history.add_message("assistant", token)
            self.chat_display.insert(tk.END, f"\nASSISTANT: {token}", 'assistant')
        else:
            # Update existing message
            self.history.messages[-1]["content"] += token
            self.chat_display.insert(tk.END, token, 'assistant')
        
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)
        
        # Update total tokens
        self.total_tokens_label.config(text=str(self.speed_calc.total_tokens))
    
    def _on_complete(self, response: str):
        """Handle completion"""
        self.is_streaming = False
        self.send_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        
        speed = self.speed_calc.get_current_speed()
        self._add_log("SUCCESS", f"Response complete ({self.speed_calc.total_tokens} tokens, {speed:.1f} tok/s)")
        self._add_log("RESPONSE", f"Complete: {truncate_text(response, 50)}")
    
    def _on_error(self, error: Exception):
        """Handle error"""
        self.is_streaming = False
        self.send_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        
        error_msg = format_api_error(error)
        self._add_log("ERROR", error_msg)
        
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"\nERROR: {error_msg}\n", 'error')
        self.chat_display.configure(state='disabled')
    
    def _update_speed_display(self):
        """Update speed display periodically"""
        if self.is_streaming:
            speed = self.speed_calc.get_current_speed()
            self.speed_label.config(text=f"{speed:.1f}")
            self.root.after(100, self._update_speed_display)
    
    def _stop_generation(self):
        """Stop generation (note: actual stream stopping requires more complex implementation)"""
        self.is_streaming = False
        self.send_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self._add_log("INFO", "Generation stopped by user")
    
    def _clear_conversation(self):
        """Clear conversation"""
        self.history.clear()
        self.chat_display.configure(state='normal')
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.configure(state='disabled')
        self._add_log("INFO", "Conversation cleared")
    
    def _new_conversation(self):
        """Start new conversation"""
        self._clear_conversation()
        self.speed_calc = SpeedCalculator()
        self.speed_label.config(text="0.0")
        self.total_tokens_label.config(text="0")
    
    def _clear_logs(self):
        """Clear logs"""
        self.logs = []
        self._refresh_logs()
        self._add_log("INFO", "Logs cleared")
    
    def _export_logs(self):
        """Export logs to file"""
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    for entry in self.logs:
                        f.write(str(entry) + "\n")
                self._add_log("SUCCESS", f"Logs exported to {filepath}")
            except Exception as e:
                self._add_log("ERROR", f"Failed to export logs: {e}")
    
    def _show_config_dialog(self):
        """Show configuration dialog"""
        # Already shown in main window, could extend with advanced options
        messagebox.showinfo("Info", "API configuration is available in the left panel")
    
    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "OpenAI API Debugger\n\n"
            "A tool for debugging OpenAI-compatible API services.\n\n"
            "Features:\n"
            "- Real-time streaming responses\n"
            "- Token generation speed tracking\n"
            "- Conversation history management\n"
            "- Preset templates for quick testing\n"
            "- Detailed interaction logs"
        )
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    app = OpenAIDebuggerApp()
    app.run()


if __name__ == "__main__":
    main()
