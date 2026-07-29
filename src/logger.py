"""
Logger for OpenAI Debug Tool
"""
from log_entry import LogEntry


class Logger:
    """Application logger with in-memory storage"""
    
    def __init__(self, max_entries=1000):
        """Initialize logger
        
        Args:
            max_entries: Maximum number of entries to keep in memory
        """
        self.entries = []
        self.max_entries = max_entries
    
    def _add_entry(self, level, message):
        """Add a log entry
        
        Args:
            level: Log level
            message: Log message
        """
        entry = LogEntry(level, message)
        self.entries.append(entry)
        
        # Trim if exceeding max entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def info(self, message):
        """Add info level log
        
        Args:
            message: Log message
        """
        self._add_entry("INFO", message)
    
    def warning(self, message):
        """Add warning level log
        
        Args:
            message: Log message
        """
        self._add_entry("WARNING", message)
    
    def error(self, message):
        """Add error level log
        
        Args:
            message: Log message
        """
        self._add_entry("ERROR", message)
    
    def debug(self, message):
        """Add debug level log
        
        Args:
            message: Log message
        """
        self._add_entry("DEBUG", message)
    
    def clear(self):
        """Clear all log entries"""
        self.entries = []
    
    def get_entries(self):
        """Get all log entries
        
        Returns:
            list: List of LogEntry objects
        """
        return self.entries.copy()
    
    def get_entries_str(self):
        """Get all log entries as strings
        
        Returns:
            list: List of formatted log strings
        """
        return [str(entry) for entry in self.entries]
