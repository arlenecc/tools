#!/usr/bin/env python3
"""
OpenAI API 调试工具 - 核心逻辑模块
支持 macOS 平台，具有实时速度显示、日志记录、预设动作等功能
此模块不包含 GUI 依赖，可独立测试
"""

import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


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
        self.session: Optional[Any] = None
    
    async def get_session(self) -> Any:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is not available")
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
        if not AIOHTTP_AVAILABLE:
            yield {"error": "aiohttp is not available"}
            return
            
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
                if "text/event-stream" in content_type or (body and body.get("stream", False)):
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
        except Exception as e:
            yield {"error": f"Connection error: {str(e)}"}
    
    async def send_chat_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any]
    ) -> AsyncGenerator[Dict, None]:
        """专门用于聊天补全的请求"""
        async for chunk in self.send_request(url, "POST", headers, body):
            yield chunk


# 导出所有公共接口
__all__ = [
    'APIConfig',
    'LogEntry', 
    'SpeedCalculator',
    'ConversationHistory',
    'format_message',
    'build_request_body',
    'parse_sse_line',
    'get_preset_actions',
    'APIClient'
]
