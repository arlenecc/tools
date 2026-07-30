"""推理速度计算器"""
import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class SpeedStats:
    """速度统计"""
    tokens_per_second: float = 0.0
    total_tokens: int = 0
    elapsed_time: float = 0.0


class SpeedCalculator:
    """推理速度计算器"""

    def __init__(self):
        self._start_time: Optional[float] = None
        self._token_count: int = 0
        self._last_token_time: Optional[float] = None
        self._instant_tps: float = 0.0

    def start(self) -> None:
        """开始计时"""
        self._start_time = time.time()
        self._token_count = 0
        self._last_token_time = None
        self._instant_tps = 0.0

    def add_token(self) -> float:
        """添加一个 token，返回瞬时速度"""
        current_time = time.time()
        self._token_count += 1

        if self._last_token_time is not None:
            delta = current_time - self._last_token_time
            if delta > 0:
                self._instant_tps = 1.0 / delta
        else:
            self._instant_tps = 0.0

        self._last_token_time = current_time
        return self._instant_tps

    def stop(self) -> SpeedStats:
        """停止计算并返回统计"""
        if self._start_time is None:
            return SpeedStats()

        elapsed = time.time() - self._start_time
        tps = self._token_count / elapsed if elapsed > 0 else 0.0

        return SpeedStats(
            tokens_per_second=tps,
            total_tokens=self._token_count,
            elapsed_time=elapsed
        )

    def reset(self) -> None:
        """重置计算器"""
        self._start_time = None
        self._token_count = 0
        self._last_token_time = None
        self._instant_tps = 0.0

    def get_instant_speed(self) -> float:
        """获取瞬时速度"""
        return self._instant_tps

    def get_current_stats(self) -> SpeedStats:
        """获取当前统计（不中断）"""
        if self._start_time is None:
            return SpeedStats()

        elapsed = time.time() - self._start_time
        tps = self._token_count / elapsed if elapsed > 0 else 0.0

        return SpeedStats(
            tokens_per_second=tps,
            total_tokens=self._token_count,
            elapsed_time=elapsed
        )
