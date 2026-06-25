from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import itertools

_seq = itertools.count()


@dataclass(order=True)
class Event:
    time: float
    priority: int = field(default=0)
    _seq: int = field(default_factory=lambda: next(_seq), compare=True)
    event_type: str = field(default="", compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)

    def __repr__(self) -> str:
        return f"Event(t={self.time:.4f}, type={self.event_type!r}, payload={self.payload})"
