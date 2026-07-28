"""
OpenAI API Debugger - Core Module
Contains all core logic without GUI dependencies
"""
import asyncio
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, AsyncGenerator
from enum import Enum
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
