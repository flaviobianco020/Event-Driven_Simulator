from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network.node import Node


@dataclass
class MetricSnapshot:
    time: float
    throughput: float
    packet_delivery_ratio: float
    end_to_end_latency: float
    queue_occupancy: float
    drop_count: int
    compression_ratio: float = 1.0
    congestion_state_transitions: int = 0


class MetricsEngine:
    def __init__(self) -> None:
        self._generated = 0
        self._delivered = 0
        self._dropped = 0
        self._total_latency = 0.0
        self._state_transitions = 0
        self._snapshots: list[MetricSnapshot] = []
        self._last_snapshot_time = 0.0
        self._delivered_since_last = 0

    def record_generated(self) -> None:
        self._generated += 1

    def record_delivered(self, latency: float) -> None:
        self._delivered += 1
        self._delivered_since_last += 1
        self._total_latency += latency

    def record_dropped(self) -> None:
        self._dropped += 1

    def record_state_transition(self) -> None:
        self._state_transitions += 1

    def collect_throughput(self, current_time: float) -> float:
        dt = current_time - self._last_snapshot_time
        return self._delivered_since_last / dt if dt > 0 else 0.0

    def collect_packet_delivery_ratio(self) -> float:
        return self._delivered / self._generated if self._generated > 0 else 0.0

    def collect_end_to_end_latency(self) -> float:
        return self._total_latency / self._delivered if self._delivered > 0 else 0.0

    def collect_queue_occupancy(self, nodes: list["Node"]) -> float:
        occs = [q.queue_occupancy for n in nodes for q in n.queues]
        return sum(occs) / len(occs) if occs else 0.0

    def collect_fairness(self) -> float:
        return 1.0  # Jain's index — placeholder for Phase 2

    def collect_compression_ratio(self) -> float:
        return 1.0  # placeholder for Phase 2

    def collect_congestion_state_transitions(self) -> int:
        return self._state_transitions

    def sample(self, current_time: float, nodes: list["Node"]) -> MetricSnapshot:
        snap = MetricSnapshot(
            time=current_time,
            throughput=self.collect_throughput(current_time),
            packet_delivery_ratio=self.collect_packet_delivery_ratio(),
            end_to_end_latency=self.collect_end_to_end_latency(),
            queue_occupancy=self.collect_queue_occupancy(nodes),
            drop_count=self._dropped,
            congestion_state_transitions=self._state_transitions,
        )
        self._snapshots.append(snap)
        self._last_snapshot_time = current_time
        self._delivered_since_last = 0
        return snap

    @property
    def snapshots(self) -> list[MetricSnapshot]:
        return list(self._snapshots)

    @property
    def total_generated(self) -> int:
        return self._generated

    @property
    def total_delivered(self) -> int:
        return self._delivered

    @property
    def total_dropped(self) -> int:
        return self._dropped
