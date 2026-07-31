"""对话历史管理"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class Message:
    """对话消息"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    model: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            model=data.get("model")
        )

    def to_api_format(self) -> Dict[str, str]:
        """转换为 API 格式"""
        return {"role": self.role, "content": self.content}


class MessageHistory:
    """对话历史管理器"""

    def __init__(self, max_messages: int = 100):
        self._messages: List[Message] = []
        self._max_messages = max_messages

    def add_message(self, role: str, content: str, model: Optional[str] = None) -> Message:
        """添加消息"""
        message = Message(role=role, content=content, model=model)
        self._messages.append(message)
        # 限制消息数量
        while len(self._messages) > self._max_messages:
            self._messages.pop(0)
        return message

    def add_user_message(self, content: str) -> Message:
        return self.add_message("user", content)

    def add_assistant_message(self, content: str, model: Optional[str] = None) -> Message:
        return self.add_message("assistant", content, model)

    def add_system_message(self, content: str) -> Message:
        return self.add_message("system", content)

    def get_messages(self) -> List[Message]:
        """获取所有消息"""
        return self._messages.copy()

    def get_api_messages(self) -> List[Dict[str, str]]:
        """获取 API 格式的消息列表"""
        return [msg.to_api_format() for msg in self._messages]

    def clear(self) -> None:
        """清空历史"""
        self._messages.clear()

    def remove_last(self) -> Optional[Message]:
        """移除最后一条消息"""
        if self._messages:
            return self._messages.pop()
        return None

    def __len__(self) -> int:
        return len(self._messages)

    def to_list(self) -> List[dict]:
        """转换为列表"""
        return [msg.to_dict() for msg in self._messages]

    @classmethod
    def from_list(cls, data: List[dict], max_messages: int = 100) -> "MessageHistory":
        """从列表创建"""
        history = cls(max_messages=max_messages)
        for item in data:
            history._messages.append(Message.from_dict(item))
        return history
