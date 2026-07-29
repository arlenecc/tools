"""
Log Entry for OpenAI Debug Tool
"""
from datetime import datetime


class LogEntry:
    """Represents a single log entry"""
    
    def __init__(self, level, message):
        """Initialize log entry
        
        Args:
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            message: Log message
        """
        self.level = level
        self.message = message
        self.timestamp = datetime.now()
    
    def __str__(self):
        """String representation of log entry"""
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] [{self.level}] {self.message}"
