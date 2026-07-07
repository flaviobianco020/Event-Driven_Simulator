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
    fairness: float = 1.0
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
        self._delivered_per_flow: dict[int, int] = {}   # flow_id → delivered count
        # compression tracking (Phase 2)
        self._bytes_original: int = 0
        self._bytes_compressed: int = 0

    def record_generated(self) -> None:
        self._generated += 1

    def record_delivered(self, latency: float, flow_id: int | None = None) -> None:
        self._delivered += 1
        self._delivered_since_last += 1
        self._total_latency += latency
        if flow_id is not None:
            self._delivered_per_flow[flow_id] = self._delivered_per_flow.get(flow_id, 0) + 1

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
        """Jain's Fairness Index: J = (Σxi)² / (n·Σxi²)  ∈ (0,1], 1=perfectly fair."""
        counts = list(self._delivered_per_flow.values())
        n = len(counts)
        if n == 0:
            return 1.0
        s1 = sum(counts)
        s2 = sum(c * c for c in counts)
        return (s1 * s1) / (n * s2) if s2 > 0 else 1.0

    def record_compression(self, compressed_size: int, original_size: int) -> None:
        self._bytes_original += original_size
        self._bytes_compressed += compressed_size

    def collect_compression_ratio(self) -> float:
        """Returns original/compressed — matches paper convention (6.1× = 6.1)."""
        if self._bytes_compressed == 0:
            return 1.0
        return self._bytes_original / self._bytes_compressed

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
            fairness=self.collect_fairness(),
            compression_ratio=self.collect_compression_ratio(),
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

    # ── Phase 3 (MARL) read-only accessors ────────────────────────────────────
    # The MAPPO environment computes per-window deltas of these counters to
    # build the reward r_t = PDR − λd·drop − λl·lat/lat_max + λf·J (doc §6).

    @property
    def total_latency(self) -> float:
        """Cumulative sum of end-to-end latencies of all delivered packets."""
        return self._total_latency

    @property
    def delivered_per_flow(self) -> dict[int, int]:
        """Copy of the per-flow delivered-packet counters (flow_id → count)."""
        return dict(self._delivered_per_flow)
