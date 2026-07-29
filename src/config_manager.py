"""
Configuration Manager for OpenAI Debug Tool
"""
import json
import os


class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_file=None):
        """Initialize configuration manager
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or "config.json"
        self.base_url = ""
        self.api_key = ""
        self.model = ""
        
        if os.path.exists(self.config_file):
            self.load()
    
    def save(self):
        """Save configuration to file"""
        config_data = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
    
    def load(self):
        """Load configuration from file"""
        if not os.path.exists(self.config_file):
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.base_url = config_data.get("base_url", "")
            self.api_key = config_data.get("api_key", "")
            self.model = config_data.get("model", "")
        except (json.JSONDecodeError, IOError):
            pass
    
    def is_valid(self):
        """Check if configuration is valid for API calls
        
        Returns:
            bool: True if base_url and api_key are set
        """
        return bool(self.base_url and self.api_key)
