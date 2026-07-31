"""配置管理器"""
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """应用配置"""
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = ""
    window_width: int = 1000
    window_height: int = 700

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(Path.home(), ".openai_debug_tool", "config.json")
        self._config_path = config_path
        self._config: Optional[Config] = None

    @property
    def config(self) -> Config:
        if self._config is None:
            self._config = self.load()
        return self._config

    def load(self) -> Config:
        """加载配置"""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return Config.from_dict(data)
        except Exception:
            pass
        return Config()

    def save(self, config: Config) -> bool:
        """保存配置"""
        try:
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            self._config = config
            return True
        except Exception:
            return False

    def get_config_path(self) -> str:
        """获取配置文件路径"""
        return self._config_path
