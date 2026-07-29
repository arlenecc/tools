"""
OpenAI Debug Tool - A GUI tool for debugging OpenAI-compatible API endpoints
Cross-platform compatible (macOS, Windows, Linux)
"""
import asyncio
import json
import time
import threading
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncGenerator, Callable
from pathlib import Path

import httpx


# ============================================================================
# Configuration and Constants
# ============================================================================

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-3.5-turbo"
CONFIG_FILE = Path.home() / ".openai_debug_tool" / "config.json"


# ============================================================================
# Enums and Data Classes
# ============================================================================

class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


@dataclass
class LogEntry:
    """Represents a log entry for API communication"""
    timestamp: datetime
    level: LogLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create_request_log(cls, url: str, method: str, headers: Dict, body: Dict) -> 'LogEntry':
        return cls(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message=f"Request: {method} {url}",
            details={"headers": headers, "body": body}
        )
    
    @classmethod
    def create_response_log(cls, status_code: int, body: Dict) -> 'LogEntry':
        return cls(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message=f"Response: Status {status_code}",
            details={"status_code": status_code, "body": body}
        )
    
    @classmethod
    def create_error_log(cls, error_message: str) -> 'LogEntry':
        return cls(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message=f"Error: {error_message}",
            details=None
        )
    
    @classmethod
    def create_debug_log(cls, message: str, details: Optional[Dict] = None) -> 'LogEntry':
        return cls(
            timestamp=datetime.now(),
            level=LogLevel.DEBUG,
            message=message,
            details=details
        )
    
    @classmethod
    def create_warning_log(cls, message: str, details: Optional[Dict] = None) -> 'LogEntry':
        return cls(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            message=message,
            details=details
        )


# ============================================================================
# Custom Exceptions
# ============================================================================

class APIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


# ============================================================================
# Core Classes
# ============================================================================

class OpenAIClient:
    """Client for interacting with OpenAI-compatible APIs"""
    
    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = "", model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    def _get_endpoint_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"
    
    async def chat_completion(self, messages: List[Dict], stream: bool = False, **kwargs) -> str:
        """Send a chat completion request"""
        if not self._client:
            raise APIError("Client not initialized. Use async context manager.")
        
        url = self._get_endpoint_url("/chat/completions")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if data.get("choices") and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                raise APIError("No choices in response", response.status_code)
                
        except httpx.HTTPStatusError as e:
            raise APIError(f"HTTP Error: {e.response.status_code}", e.response.status_code)
        except Exception as e:
            raise APIError(f"Request failed: {str(e)}")
    
    async def chat_completion_stream(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat completion response"""
        if not self._client:
            raise APIError("Client not initialized. Use async context manager.")
        
        url = self._get_endpoint_url("/chat/completions")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        try:
            async with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if data.get("choices") and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPStatusError as e:
            raise APIError(f"HTTP Error: {e.response.status_code}", e.response.status_code)
        except Exception as e:
            raise APIError(f"Stream failed: {str(e)}")
    
    async def list_models(self) -> List[str]:
        """List available models from the API"""
        if not self._client:
            raise APIError("Client not initialized. Use async context manager.")
        
        url = self._get_endpoint_url("/models")
        
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            data = response.json()
            
            models = []
            if "data" in data:
                for model_info in data["data"]:
                    model_id = model_info.get("id", "")
                    if model_id:
                        models.append(model_id)
            
            return models
            
        except httpx.HTTPStatusError as e:
            raise APIError(f"HTTP Error: {e.response.status_code}", e.response.status_code)
        except Exception as e:
            raise APIError(f"Failed to list models: {str(e)}")


class ConversationManager:
    """Manages conversation history"""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation"""
        self.messages.append({"role": role, "content": content})
    
    def clear(self):
        """Clear all messages"""
        self.messages = []
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages"""
        return self.messages.copy()
    
    def remove_last(self):
        """Remove the last message"""
        if self.messages:
            self.messages.pop()


class SpeedCalculator:
    """Calculates tokens per second during streaming"""
    
    def __init__(self):
        self.token_count = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.token_count = 0
        self.end_time = None
    
    def add_token(self):
        """Increment token count"""
        self.token_count += 1
    
    def stop(self):
        """Stop timing"""
        self.end_time = time.time()
    
    def reset(self):
        """Reset calculator"""
        self.token_count = 0
        self.start_time = None
        self.end_time = None
    
    def get_speed(self) -> float:
        """Get current tokens per second"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        elapsed = end - self.start_time
        
        if elapsed <= 0:
            return 0.0
        
        return self.token_count / elapsed


class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_path: str = str(CONFIG_FILE)):
        self.config_path = Path(config_path)
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}


# ============================================================================
# Utility Functions
# ============================================================================

def count_tokens(text: str) -> int:
    """Estimate token count for text (simple approximation)"""
    if not text:
        return 0
    # Simple approximation: ~4 characters per token
    return max(1, len(text) // 4)


def count_tokens_messages(messages: List[Dict]) -> int:
    """Estimate total tokens for message list"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")
        # Add overhead for role
        total += count_tokens(content) + count_tokens(role) + 4
    return total


def format_speed(speed: float) -> str:
    """Format speed for display"""
    return f"{speed:.2f} tokens/s"


def format_timestamp(dt: datetime) -> str:
    """Format datetime for display"""
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


# ============================================================================
# GUI Application (using tkinter for cross-platform compatibility)
# ============================================================================

class OpenAIDebugToolGUI:
    """Main GUI application for OpenAI Debug Tool"""
    
    def __init__(self):
        self.root = None
        self.conversation_manager = ConversationManager()
        self.speed_calculator = SpeedCalculator()
        self.config_manager = ConfigManager()
        self.log_entries: List[LogEntry] = []
        self.is_streaming = False
        self.current_client: Optional[OpenAIClient] = None
        
        # Load saved configuration
        self.saved_config = self.config_manager.load_config()
    
    def setup_gui(self):
        """Initialize the GUI"""
        import tkinter as tk
        from tkinter import ttk, scrolledtext, messagebox
        
        self.root = tk.Tk()
        self.root.title("OpenAI Debug Tool")
        self.root.geometry("1200x800")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === TOP: Configuration Section ===
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Configure grid weights for responsive layout - make column 1 expandable
        config_frame.grid_columnconfigure(1, weight=1)
        config_frame.grid_columnconfigure(2, weight=0)  # Button column doesn't expand
        
        # Base URL
        ttk.Label(config_frame, text="Base URL:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.url_var = tk.StringVar(value=self.saved_config.get("base_url", DEFAULT_BASE_URL))
        self.url_entry = ttk.Entry(config_frame, textvariable=self.url_var, font=("TkDefaultFont", 10))
        self.url_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # API Key
        ttk.Label(config_frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.api_key_var = tk.StringVar(value=self.saved_config.get("api_key", ""))
        self.api_key_entry = ttk.Entry(config_frame, textvariable=self.api_key_var, show="*", font=("TkDefaultFont", 10))
        self.api_key_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Model - now using Combobox for model selection
        ttk.Label(config_frame, text="Model:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value=self.saved_config.get("model", DEFAULT_MODEL))
        self.model_combo = ttk.Combobox(config_frame, textvariable=self.model_var, font=("TkDefaultFont", 10), state="readonly")
        self.model_combo.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Get Models button
        get_models_btn = ttk.Button(config_frame, text="Get Models", command=self.get_models)
        get_models_btn.grid(row=2, column=2, padx=5, pady=5)
        
        # Save config button
        save_btn = ttk.Button(config_frame, text="Save Config", command=self.save_configuration)
        save_btn.grid(row=3, column=1, sticky=tk.E, padx=5, pady=5)
        
        # === MIDDLE: Chat and Logs (side by side) ===
        # Create paned window for resizable sections
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Chat interface
        left_frame = ttk.Frame(paned, padding="5")
        paned.add(left_frame, weight=2)
        
        # Right panel - Logs
        right_frame = ttk.Frame(paned, padding="5")
        paned.add(right_frame, weight=1)
        
        # === LEFT PANEL: Chat ===
        # Chat display
        chat_frame = ttk.LabelFrame(left_frame, text="Conversation", padding="5")
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED, font=("TkDefaultFont", 10))
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for different message types
        self.chat_display.tag_configure("user", foreground="blue")
        self.chat_display.tag_configure("assistant", foreground="green")
        self.chat_display.tag_configure("system", foreground="orange")
        self.chat_display.tag_configure("error", foreground="red")
        
        # Input section
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Input text area with explicit width and proper packing
        ttk.Label(input_frame, text="Message:").pack(anchor=tk.W)
        self.input_text = scrolledtext.ScrolledText(
            input_frame, 
            height=5, 
            wrap=tk.WORD, 
            font=("TkDefaultFont", 10),
            width=50  # Explicit width
        )
        self.input_text.pack(fill=tk.X, expand=True, pady=(5, 5))
        
        # Button row
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.send_btn = ttk.Button(button_frame, text="Send (Enter)", command=self.send_message, width=15)
        self.send_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_generation, width=10)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_btn.state(['disabled'])
        
        self.clear_btn = ttk.Button(button_frame, text="Clear Chat", command=self.clear_conversation, width=12)
        self.clear_btn.pack(side=tk.LEFT)
        
        # Status bar
        status_frame = ttk.Frame(left_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.speed_label = ttk.Label(status_frame, text="Speed: 0.00 tokens/s", relief=tk.SUNKEN, anchor=tk.W)
        self.speed_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.token_label = ttk.Label(status_frame, text="Tokens: 0", relief=tk.SUNKEN, anchor=tk.E)
        self.token_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        # === RIGHT PANEL: Logs ===
        log_frame = ttk.LabelFrame(right_frame, text="API Communication Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_display = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state=tk.DISABLED, font=("TkDefaultFont", 9))
        self.log_display.pack(fill=tk.BOTH, expand=True)
        
        # Log tags
        self.log_display.tag_configure("info", foreground="black")
        self.log_display.tag_configure("warning", foreground="orange")
        self.log_display.tag_configure("error", foreground="red")
        self.log_display.tag_configure("debug", foreground="gray")
        
        # Log controls
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.pack(fill=tk.X, pady=(5, 0))
        
        clear_log_btn = ttk.Button(log_control_frame, text="Clear Log", command=self.clear_log)
        clear_log_btn.pack(side=tk.LEFT)
        
        export_log_btn = ttk.Button(log_control_frame, text="Export Log", command=self.export_log)
        export_log_btn.pack(side=tk.RIGHT)
        
        # Bind Enter key to send
        # Shift+Enter for newline, Enter alone to send
        self.input_text.bind("<Return>", self._on_enter_key)
        self.input_text.bind("<Shift-Return>", self._on_shift_enter)
        self.input_text.bind("<Control-Return>", lambda e: self.send_message())
        # Mac Command+Enter support
        try:
            self.input_text.bind("<Command-Return>", lambda e: self.send_message())
        except:
            pass  # Command binding may not be available on all platforms
        
        # Protocol handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _on_shift_enter(self, event):
        """Handle Shift+Enter - allow newline"""
        return None  # Allow default behavior (newline)
    
    def _on_enter_key(self, event):
        """Handle Enter key press - send message unless Shift is held"""
        # Check if Shift is pressed using the state flags
        # 0x1 = Shift, 0x4 = Control, 0x8 = Alt/Meta
        shift_pressed = bool(event.state & 0x1)
        control_pressed = bool(event.state & 0x4)
        
        if shift_pressed or control_pressed:
            return None  # Allow default behavior (newline for Shift, nothing special for Control)
        else:
            self.send_message()
            return "break"  # Prevent default newline
    
    def save_configuration(self):
        """Save current configuration"""
        config = {
            "base_url": self.url_var.get(),
            "api_key": self.api_key_var.get(),
            "model": self.model_var.get()
        }
        self.config_manager.save_config(config)
        self.add_log_entry(LogEntry.create_debug_log("Configuration saved"))
    
    async def get_models_async(self) -> List[str]:
        """Async method to fetch models from API"""
        import tkinter as tk
        
        base_url = self.url_var.get().strip()
        api_key = self.api_key_var.get().strip()
        
        if not base_url:
            self.add_error_message("Base URL is required")
            self.add_log_entry(LogEntry.create_error_log("Cannot fetch models: Base URL is empty"))
            return []
        
        self.add_log_entry(LogEntry.create_debug_log(f"Fetching models from: {base_url}"))
        
        try:
            async with OpenAIClient(base_url, api_key, "") as client:
                # Log request
                self.add_log_entry(LogEntry.create_request_log(
                    url=f"{base_url}/models",
                    method="GET",
                    headers={"Authorization": "Bearer ***" if api_key else "", "Content-Type": "application/json"},
                    body={}
                ))
                
                models = await client.list_models()
                
                # Log response
                self.add_log_entry(LogEntry.create_response_log(
                    status_code=200,
                    body={"models": models, "count": len(models)}
                ))
                
                self.add_log_entry(LogEntry.create_debug_log(f"Found {len(models)} models"))
                
                return models
                
        except APIError as e:
            self.add_log_entry(LogEntry.create_error_log(f"API Error fetching models: {str(e)} (status: {e.status_code})"))
            self.add_error_message(f"Failed to fetch models: {str(e)}")
            return []
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            self.add_log_entry(LogEntry.create_error_log(f"Exception fetching models: {str(e)}\n{error_trace}"))
            self.add_error_message(f"Unexpected error: {str(e)}")
            return []
    
    def get_models(self):
        """Fetch available models from API and populate dropdown"""
        import tkinter as tk
        
        # Disable button during fetch
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.LabelFrame):
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, ttk.Button) and grandchild.cget("text") == "Get Models":
                                grandchild.state(['disabled'])
        
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            models = loop.run_until_complete(self.get_models_async())
            loop.close()
            
            # Update UI in main thread
            if models:
                self.root.after(0, lambda: self.model_combo.configure(values=models))
            
            # Re-enable button
            def enable_button():
                for widget in self.root.winfo_children():
                    if isinstance(widget, ttk.Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, ttk.LabelFrame):
                                for grandchild in child.winfo_children():
                                    if isinstance(grandchild, ttk.Button) and grandchild.cget("text") == "Get Models":
                                        grandchild.state(['!disabled'])
            
            self.root.after(0, enable_button)
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
    
    def add_log_entry(self, entry: LogEntry):
        """Add entry to log display"""
        import tkinter as tk
        
        self.log_entries.append(entry)
        
        self.log_display.configure(state=tk.NORMAL)
        timestamp_str = format_timestamp(entry.timestamp)
        level_str = entry.level.value
        
        self.log_display.insert(tk.END, f"[{timestamp_str}] [{level_str}] {entry.message}\n", 
                               entry.level.value.lower())
        
        if entry.details:
            details_str = json.dumps(entry.details, indent=2)
            self.log_display.insert(tk.END, f"{details_str}\n\n", "debug")
        
        self.log_display.see(tk.END)
        self.log_display.configure(state=tk.DISABLED)
    
    def add_chat_message(self, role: str, content: str):
        """Add message to chat display"""
        import tkinter as tk
        
        self.chat_display.configure(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "system")
        self.chat_display.insert(tk.END, f"{role.capitalize()}: ", role)
        self.chat_display.insert(tk.END, f"{content}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)
    
    def add_error_message(self, error: str):
        """Add error message to chat"""
        import tkinter as tk
        
        self.chat_display.configure(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] Error: {error}\n\n", "error")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)
    
    def clear_conversation(self):
        """Clear conversation history"""
        import tkinter as tk
        
        self.conversation_manager.clear()
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self.update_speed_display(0.0)
        self.token_label.configure(text="Tokens: 0")
    
    def clear_log(self):
        """Clear log display"""
        import tkinter as tk
        
        self.log_entries = []
        self.log_display.configure(state=tk.NORMAL)
        self.log_display.delete(1.0, tk.END)
        self.log_display.configure(state=tk.DISABLED)
    
    def export_log(self):
        """Export log to file"""
        import tkinter.filedialog as fd
        
        file_path = fd.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            log_data = [
                {
                    "timestamp": format_timestamp(entry.timestamp),
                    "level": entry.level.value,
                    "message": entry.message,
                    "details": entry.details
                }
                for entry in self.log_entries
            ]
            
            with open(file_path, 'w') as f:
                json.dump(log_data, f, indent=2)
    
    def update_speed_display(self, speed: float):
        """Update speed label"""
        self.speed_label.configure(text=f"Speed: {format_speed(speed)}")
    
    async def send_message_async(self):
        """Async message sending logic"""
        import tkinter as tk
        
        try:
            user_input = self.input_text.get(1.0, tk.END).strip()
            if not user_input:
                self.add_log_entry(LogEntry.create_debug_log("No input provided"))
                return
            
            self.add_log_entry(LogEntry.create_debug_log(f"Sending message: {user_input[:100]}..."))
            
            # Clear input
            self.input_text.delete(1.0, tk.END)
            
            # Add user message to conversation
            self.conversation_manager.add_message("user", user_input)
            self.add_chat_message("user", user_input)
            
            # Update token count
            total_tokens = count_tokens_messages(self.conversation_manager.get_messages())
            self.token_label.configure(text=f"Tokens: {total_tokens}")
            
            # Prepare client
            base_url = self.url_var.get().strip()
            api_key = self.api_key_var.get().strip()
            model = self.model_var.get().strip()
            
            if not base_url:
                self.add_error_message("Base URL is required")
                self.add_log_entry(LogEntry.create_error_log("Base URL is empty"))
                return
            
            if not model:
                self.add_error_message("Model name is required")
                self.add_log_entry(LogEntry.create_error_log("Model name is empty"))
                return
            
            self.add_log_entry(LogEntry.create_debug_log(
                f"Connecting to: {base_url}", 
                {"model": model, "api_key_set": bool(api_key)}
            ))
            
            self.is_streaming = True
            self.send_btn.state(['disabled'])
            self.stop_btn.state(['!disabled'])
            
            assistant_content = ""
            self.speed_calculator.reset()
            self.speed_calculator.start()
            
            async with OpenAIClient(base_url, api_key, model) as client:
                messages = self.conversation_manager.get_messages()
                
                # Log request with full details
                request_body = {"model": model, "messages": messages, "stream": True}
                self.add_log_entry(LogEntry.create_request_log(
                    url=f"{base_url}/chat/completions",
                    method="POST",
                    headers={"Authorization": "Bearer ***" if api_key else "", "Content-Type": "application/json"},
                    body=request_body
                ))
                
                self.add_log_entry(LogEntry.create_debug_log("Starting stream request..."))
                
                # Stream response
                response_text = ""
                token_count = 0
                
                try:
                    async for chunk in client.chat_completion_stream(messages):
                        if not self.is_streaming:
                            self.add_log_entry(LogEntry.create_debug_log("Streaming stopped by user"))
                            break
                        
                        response_text += chunk
                        token_count += 1
                        self.speed_calculator.add_token()
                        
                        # Update display incrementally
                        if token_count == 1:
                            self.add_chat_message("assistant", chunk)
                        else:
                            # Append to last message
                            self.chat_display.configure(state=tk.NORMAL)
                            self.chat_display.insert(tk.END, chunk)
                            self.chat_display.see(tk.END)
                            self.chat_display.configure(state=tk.DISABLED)
                        
                        # Update speed periodically
                        if token_count % 5 == 0:
                            speed = self.speed_calculator.get_speed()
                            self.update_speed_display(speed)
                    
                    self.speed_calculator.stop()
                    final_speed = self.speed_calculator.get_speed()
                    self.update_speed_display(final_speed)
                    
                    if response_text:
                        self.conversation_manager.add_message("assistant", response_text)
                        
                        # Log response
                        self.add_log_entry(LogEntry.create_response_log(
                            status_code=200,
                            body={"content": response_text[:500] + "..." if len(response_text) > 500 else response_text, "tokens": token_count}
                        ))
                        
                        self.add_log_entry(LogEntry.create_debug_log(
                            f"Response complete: {token_count} tokens at {final_speed:.2f} tokens/s"
                        ))
                        
                        # Update final token count
                        total_tokens = count_tokens_messages(self.conversation_manager.get_messages())
                        self.token_label.configure(text=f"Tokens: {total_tokens}")
                    else:
                        self.add_log_entry(LogEntry.create_warning_log("Empty response received"))
                        
                except Exception as stream_error:
                    self.add_log_entry(LogEntry.create_error_log(f"Stream error: {str(stream_error)}"))
                    raise
                    
        except APIError as e:
            self.add_error_message(str(e))
            self.add_log_entry(LogEntry.create_error_log(f"API Error: {str(e)} (status: {e.status_code})"))
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            self.add_error_message(f"Unexpected error: {str(e)}")
            self.add_log_entry(LogEntry.create_error_log(f"Exception: {str(e)}\n{error_trace}"))
        finally:
            self.is_streaming = False
            self.send_btn.state(['!disabled'])
            self.stop_btn.state(['disabled'])
    
    def send_message(self):
        """Send message (wrapper for async)"""
        if self.is_streaming:
            return
        
        # Run async function in separate thread
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_message_async())
            loop.close()
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
    
    def stop_generation(self):
        """Stop current generation"""
        self.is_streaming = False
        self.speed_calculator.stop()
    
    def on_closing(self):
        """Handle window closing"""
        # Save configuration
        self.save_configuration()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.setup_gui()
        self.root.mainloop()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    app = OpenAIDebugToolGUI()
    app.run()


if __name__ == "__main__":
    main()
