"""日志条目类"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime = field(default_factory=datetime.now)
    level: str = "INFO"
    message: str = ""
    details: Optional[str] = None

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S.%f")[:-3]
        base = f"[{time_str}] [{self.level}] {self.message}"
        if self.details:
            base += f"\n{self.details}"
        return base

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "details": self.details
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LogEntry":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=data["level"],
            message=data["message"],
            details=data.get("details")
        )
