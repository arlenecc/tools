"""OpenAI API 客户端"""
import json
import requests
from typing import List, Dict, Optional, Generator, Tuple
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """模型信息"""
    id: str
    name: Optional[str] = None
    created: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        return cls(
            id=data.get("id", ""),
            name=data.get("name"),
            created=data.get("created")
        )


@dataclass
class ChatResponse:
    """聊天响应"""
    content: str
    model: str
    usage: Optional[Dict] = None
    finish_reason: Optional[str] = None


class APIClient:
    """OpenAI API 客户端"""

    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers["Content-Type"] = "application/json"

    def set_base_url(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        else:
            self._session.headers.pop("Authorization", None)

    def fetch_models(self) -> Tuple[List[ModelInfo], Optional[str]]:
        """获取模型列表"""
        url = f"{self.base_url}/models"
        try:
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            models = []
            for item in data.get("data", []):
                models.append(ModelInfo.from_dict(item))
            return models, None
        except requests.exceptions.RequestException as e:
            return [], f"请求失败：{str(e)}"
        except json.JSONDecodeError as e:
            return [], f"JSON 解析失败：{str(e)}"
        except Exception as e:
            return [], f"未知错误：{str(e)}"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = True
    ) -> Tuple[Optional[ChatResponse], Optional[str]]:
        """非流式聊天完成"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        try:
            response = self._session.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            return ChatResponse(
                content=message.get("content", ""),
                model=data.get("model", model),
                usage=data.get("usage"),
                finish_reason=choice.get("finish_reason")
            ), None
        except requests.exceptions.RequestException as e:
            return None, f"请求失败：{str(e)}"
        except json.JSONDecodeError as e:
            return None, f"JSON 解析失败：{str(e)}"
        except Exception as e:
            return None, f"未知错误：{str(e)}"

    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        logger=None
    ) -> Generator[Tuple[Optional[str], Optional[str]], None, None]:
        """流式聊天完成，yield (content_chunk, error)"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        if logger:
            logger.debug("发送请求", f"POST {url}\nPayload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        try:
            if logger:
                logger.debug("等待响应...")
            response = self._session.post(url, json=payload, timeout=120, stream=True)
            
            if logger:
                logger.debug("收到响应", f"Status: {response.status_code}")
            
            response.raise_for_status()
            
            token_count = 0
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if logger:
                    logger.debug("SSE 数据", f"Raw: {decoded[:100]}..." if len(decoded) > 100 else f"Raw: {decoded}")
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str.strip() == "[DONE]":
                        if logger:
                            logger.info("流式传输完成", f"总 token 数：{token_count}")
                        break
                    try:
                        data = json.loads(data_str)
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        finish_reason = choice.get("finish_reason")
                        if content:
                            token_count += 1
                            yield content, None
                        elif finish_reason:
                            if logger:
                                logger.debug("完成原因", f"finish_reason: {finish_reason}")
                            yield None, None
                    except json.JSONDecodeError as e:
                        if logger:
                            logger.warning("JSON 解析失败", str(e))
                        continue
        except requests.exceptions.RequestException as e:
            error_msg = f"请求失败：{str(e)}"
            if logger:
                logger.error("HTTP 请求错误", error_msg)
            yield None, error_msg
        except Exception as e:
            error_msg = f"未知错误：{str(e)}"
            if logger:
                logger.error("未知错误", error_msg)
            yield None, error_msg

    def close(self) -> None:
        """关闭会话"""
        self._session.close()
