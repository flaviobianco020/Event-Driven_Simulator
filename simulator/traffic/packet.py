from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import itertools

if TYPE_CHECKING:
    from .flow import Flow
    from ..network.link import Link

_pkt_counter = itertools.count()


@dataclass
class Packet:
    size: int
    priority: int
    creation_time: float
    flow: Optional["Flow"] = None
    id: int = field(default_factory=lambda: next(_pkt_counter))
    enqueue_time: float = field(default=0.0)
    path: list["Link"] = field(default_factory=list, repr=False)
    hop: int = field(default=0)

    def __repr__(self) -> str:
        fid = self.flow.id if self.flow else "?"
        return f"Packet(id={self.id}, size={self.size}B, pri={self.priority}, flow={fid})"
