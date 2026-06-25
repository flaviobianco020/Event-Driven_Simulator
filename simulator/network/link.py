from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node


@dataclass
class Link:
    source: "Node"
    target: "Node"
    capacity: float         # packets per time unit
    delay: float            # propagation delay in time units
    active: bool = True

    def __repr__(self) -> str:
        return f"Link({self.source.id}→{self.target.id}, cap={self.capacity}, delay={self.delay})"

    def __hash__(self) -> int:
        return hash((self.source.id, self.target.id))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Link)
            and self.source.id == other.source.id
            and self.target.id == other.target.id
        )
