from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import itertools

if TYPE_CHECKING:
    from .traffic.packet import Packet
    from .network.node import Node
    from .network.link import Link


class EventType(Enum):
    # Traffic
    FLOW_START = auto()
    FLOW_STOP = auto()
    PACKET_GENERATE = auto()
    PACKET_ARRIVAL = auto()
    # Queue
    QUEUE_ENQUEUE = auto()
    QUEUE_DEQUEUE = auto()
    PACKET_DROP = auto()
    # Transmission
    PACKET_TRANSMIT = auto()
    PACKET_DELIVER = auto()
    # Network
    LINK_RATE_CHANGE = auto()
    LINK_FAILURE = auto()
    LINK_RECOVERY = auto()
    # Monitoring
    METRIC_SAMPLE = auto()
    STATE_UPDATE = auto()


_seq_counter = itertools.count()


@dataclass(order=True)
class Event:
    simulation_time: float
    _seq: int = field(default_factory=lambda: next(_seq_counter), compare=True)
    type: EventType = field(default=EventType.METRIC_SAMPLE, compare=False)
    packet: Optional["Packet"] = field(default=None, compare=False)
    node: Optional["Node"] = field(default=None, compare=False)
    link: Optional["Link"] = field(default=None, compare=False)
    metadata: dict = field(default_factory=dict, compare=False)

    def __repr__(self) -> str:
        return f"Event(t={self.simulation_time:.4f}, type={self.type.name})"
