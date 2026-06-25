from __future__ import annotations
import heapq
from typing import Optional
from .event import Event


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []

    def schedule(self, event: Event) -> None:
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[Event]:
        return self._heap[0] if self._heap else None

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)
