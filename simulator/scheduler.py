from __future__ import annotations
import heapq
from typing import Callable, Optional
from .event import Event, EventType

HandlerFn = Callable[[Event, "EventScheduler"], None]


class EventScheduler:
    def __init__(self) -> None:
        self._future_events: list[Event] = []
        self.simulation_time: float = 0.0
        self._handlers: dict[EventType, list[HandlerFn]] = {}
        self._running = False

    def schedule_event(self, event: Event) -> None:
        heapq.heappush(self._future_events, event)

    def extract_next_event(self) -> Optional[Event]:
        return heapq.heappop(self._future_events) if self._future_events else None

    def advance_time(self, t: float) -> None:
        self.simulation_time = t

    def execute_handler(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            handler(event, self)

    def register(self, event_type: EventType, handler: HandlerFn) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def run(self, end_time: float = float("inf")) -> None:
        self._running = True
        while self._running and self._future_events:
            event = heapq.heappop(self._future_events)
            if event.simulation_time > end_time:
                break
            self.advance_time(event.simulation_time)
            self.execute_handler(event)
        self._running = False

    def run_until(self, end_time: float) -> None:
        """
        Resumable slice execution (Phase 3 / MARL): processes all events with
        simulation_time <= end_time and LEAVES later events in the queue.

        Unlike run(), which pops the first out-of-range event before breaking
        (losing it), this peeks before popping — repeated calls with increasing
        end_time reproduce exactly the same trajectory as a single run().
        """
        self._running = True
        while self._running and self._future_events:
            if self._future_events[0].simulation_time > end_time:
                break
            event = heapq.heappop(self._future_events)
            self.advance_time(event.simulation_time)
            self.execute_handler(event)
        self._running = False
        # il tempo avanza fino alla fine della slice anche senza eventi
        if self.simulation_time < end_time:
            self.advance_time(end_time)

    def stop(self) -> None:
        self._running = False

    def is_empty(self) -> bool:
        return len(self._future_events) == 0
