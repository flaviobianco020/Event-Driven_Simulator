from __future__ import annotations
from collections import deque
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..traffic.packet import Packet


class QueueManager:
    def __init__(self, max_size: int = 100, service_rate: float = 10.0) -> None:
        self.buffer: deque["Packet"] = deque()
        self.max_size = max_size
        self.service_rate = service_rate        # packets per time unit
        self.drop_events = 0
        self._server_free_at: float = 0.0
        self._total_wait: float = 0.0
        self._served: int = 0

    @property
    def queue_occupancy(self) -> float:
        return len(self.buffer) / self.max_size if self.max_size > 0 else 0.0

    @property
    def waiting_time(self) -> float:
        return self._total_wait / self._served if self._served > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self.buffer)

    def enqueue(self, packet: "Packet") -> bool:
        if len(self.buffer) >= self.max_size:
            self.drop(packet)
            return False
        self.buffer.append(packet)
        return True

    def dequeue(self) -> Optional["Packet"]:
        return self.buffer.popleft() if self.buffer else None

    def drop(self, packet: "Packet") -> None:
        self.drop_events += 1

    def next_service_completion(self, current_time: float, size_ratio: float = 1.0) -> float:
        service_duration = size_ratio / self.service_rate
        start = max(current_time, self._server_free_at)
        self._server_free_at = start + service_duration
        return self._server_free_at

    def record_wait(self, enqueue_time: float, dequeue_time: float) -> None:
        self._total_wait += dequeue_time - enqueue_time
        self._served += 1
