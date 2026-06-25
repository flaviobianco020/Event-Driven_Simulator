from __future__ import annotations
from typing import Callable, Union
from .event import Event
from .event_queue import EventQueue
from .handler import EventHandler

HandlerType = Union[EventHandler, Callable[["Event", "SimulationEngine"], None]]


class SimulationEngine:
    def __init__(self, end_time: float = float("inf")) -> None:
        self.end_time = end_time
        self.current_time: float = 0.0
        self._queue = EventQueue()
        self._handlers: dict[str, list[HandlerType]] = {}
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, event: Event) -> None:
        if event.time < self.current_time:
            raise ValueError(
                f"Cannot schedule event at t={event.time:.4f} before current t={self.current_time:.4f}"
            )
        self._queue.schedule(event)

    def register(
        self, event_type: str, handler: HandlerType
    ) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        while self._running and not self._queue.is_empty():
            event = self._queue.pop()
            if event.time > self.end_time:
                break
            self.current_time = event.time
            self._dispatch(event)
        self._running = False

    def _dispatch(self, event: Event) -> None:
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            return
        for h in handlers:
            if isinstance(h, EventHandler):
                h.handle(event, self)
            else:
                h(event, self)
