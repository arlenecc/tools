"""
Speed Calculator for OpenAI Debug Tool
"""
import time


class SpeedCalculator:
    """Calculates token generation speed"""
    
    def __init__(self):
        """Initialize speed calculator"""
        self.start_time = None
        self.total_tokens = 0
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.total_tokens = 0
    
    def reset(self):
        """Reset the calculator"""
        self.start_time = None
        self.total_tokens = 0
    
    def add_tokens(self, count):
        """Add token count
        
        Args:
            count: Number of tokens to add
        """
        self.total_tokens += count
    
    def get_speed(self):
        """Get current tokens per second
        
        Returns:
            float: Tokens per second, or 0 if not started
        """
        if self.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        
        return self.total_tokens / elapsed
