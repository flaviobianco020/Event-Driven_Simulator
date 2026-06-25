from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import SimulationEngine
    from .event import Event


class EventHandler(ABC):
    @abstractmethod
    def handle(self, event: "Event", engine: "SimulationEngine") -> None: ...
