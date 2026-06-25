from __future__ import annotations
from dataclasses import dataclass, field
from .queue_manager import QueueManager
from .congestion import CongestionStateMachine


@dataclass
class Node:
    id: str
    queues: list[QueueManager] = field(default_factory=lambda: [QueueManager(max_size=1000, service_rate=1000.0)])
    state_machine: CongestionStateMachine = field(default_factory=CongestionStateMachine)

    @property
    def default_queue(self) -> QueueManager:
        return self.queues[0]

    def __repr__(self) -> str:
        return f"Node({self.id})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.id == other.id
