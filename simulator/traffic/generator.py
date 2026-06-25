from __future__ import annotations
import random
from .flow import Flow, FlowModel, TrafficClass
from .packet import Packet


class TrafficGenerator:
    def __init__(self) -> None:
        self.classes: list[TrafficClass] = []
        self._flows: list[Flow] = []

    def add_class(self, tc: TrafficClass) -> "TrafficGenerator":
        self.classes.append(tc)
        return self

    def add_flow(self, flow: Flow) -> "TrafficGenerator":
        self._flows.append(flow)
        return self

    def generate_flows(self) -> list[Flow]:
        return list(self._flows)

    def create_packet(self, flow: Flow, current_time: float, rng: random.Random) -> Packet:
        return Packet(
            size=flow.traffic_class.sample_size(rng),
            priority=flow.traffic_class.priority_level,
            creation_time=current_time,
            flow=flow,
        )
