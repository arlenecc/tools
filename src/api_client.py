"""
API Client for OpenAI Debug Tool
"""
import requests
import json


class APIClient:
    """Client for interacting with OpenAI-compatible APIs"""
    
    def __init__(self, base_url, api_key):
        """Initialize API client
        
        Args:
            base_url: Base URL of the API endpoint
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_models(self):
        """Get list of available models
        
        Returns:
            list: List of model IDs, or None if error occurred
        """
        try:
            url = f"{self.base_url}/models"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return models
        except Exception as e:
            return None
    
    def chat_completion_stream(self, model, messages):
        """Stream chat completion
        
        Args:
            model: Model ID to use
            messages: List of message dictionaries
            
        Yields:
            str: Content chunks from the streaming response
        """
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "stream": True
            }
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            return
