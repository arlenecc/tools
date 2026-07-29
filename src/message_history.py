"""
Message History for OpenAI Debug Tool
"""


class MessageHistory:
    """Manages conversation message history"""
    
    def __init__(self):
        """Initialize message history"""
        self.messages = []
    
    def add_message(self, role, content):
        """Add a message to history
        
        Args:
            role: Role of the message sender ('user', 'assistant', 'system')
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content
        })
    
    def clear(self):
        """Clear all messages from history"""
        self.messages = []
    
    def get_messages(self):
        """Get all messages
        
        Returns:
            list: List of message dictionaries
        """
        return self.messages.copy()
    
    def remove_last_assistant(self):
        """Remove the last assistant message if exists"""
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
