from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import itertools
import random

if TYPE_CHECKING:
    from ..network.node import Node

_flow_counter = itertools.count()


class FlowModel(Enum):
    CBR = "cbr"
    POISSON = "poisson"
    BURSTY = "bursty"
    PERIODIC_TELEMETRY = "periodic_telemetry"
    VIDEO = "video"
    CONTROL = "control"


@dataclass
class TrafficClass:
    packet_size_distribution: tuple[int, int]   # (min_bytes, max_bytes)
    priority_level: int
    latency_sensitivity: bool = False
    compression_sensitivity: bool = False

    def sample_size(self, rng: random.Random) -> int:
        lo, hi = self.packet_size_distribution
        return lo if lo == hi else rng.randint(lo, hi)


@dataclass
class Flow:
    model: FlowModel
    traffic_class: TrafficClass
    source: "Node"
    destination: "Node"
    rate: float
    id: int = field(default_factory=lambda: next(_flow_counter))
    active: bool = True

    def inter_arrival(self, rng: random.Random) -> float:
        if self.model in (FlowModel.POISSON, FlowModel.VIDEO):
            return rng.expovariate(self.rate)
        if self.model == FlowModel.BURSTY:
            if rng.random() < 0.1:
                return rng.expovariate(self.rate * 5)
            return rng.expovariate(self.rate * 0.5)
        return 1.0 / self.rate

    def __repr__(self) -> str:
        return f"Flow(id={self.id}, {self.model.value}, {self.source.id}→{self.destination.id}, rate={self.rate})"
