"""日志管理器"""
from typing import List, Callable
from collections import deque
from .log_entry import LogEntry


class Logger:
    """线程安全的日志管理器"""

    def __init__(self, max_entries: int = 1000):
        self._entries: deque[LogEntry] = deque(maxlen=max_entries)
        self._callbacks: List[Callable[[LogEntry], None]] = []

    def add_entry(self, entry: LogEntry) -> None:
        """添加日志条目"""
        self._entries.append(entry)
        for callback in self._callbacks:
            try:
                callback(entry)
            except Exception:
                pass

    def log(self, message: str, level: str = "INFO", details: str = None) -> LogEntry:
        """记录日志"""
        entry = LogEntry(level=level, message=message, details=details)
        self.add_entry(entry)
        return entry

    def info(self, message: str, details: str = None) -> LogEntry:
        return self.log(message, "INFO", details)

    def debug(self, message: str, details: str = None) -> LogEntry:
        return self.log(message, "DEBUG", details)

    def warning(self, message: str, details: str = None) -> LogEntry:
        return self.log(message, "WARNING", details)

    def error(self, message: str, details: str = None) -> LogEntry:
        return self.log(message, "ERROR", details)

    def get_entries(self) -> List[LogEntry]:
        """获取所有日志条目"""
        return list(self._entries)

    def clear(self) -> None:
        """清空日志"""
        self._entries.clear()

    def register_callback(self, callback: Callable[[LogEntry], None]) -> None:
        """注册回调函数"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[LogEntry], None]) -> None:
        """注销回调函数"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
