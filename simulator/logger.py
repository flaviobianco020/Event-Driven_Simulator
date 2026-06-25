from __future__ import annotations
from enum import IntEnum
from typing import TextIO, Optional
import sys


class LogLevel(IntEnum):
    NONE    = 0   # no output
    SUMMARY = 1   # only metrics + state transitions + link events
    INFO    = 2   # + packet drops + flow start/stop
    DEBUG   = 3   # every single event


class TraceLogger:
    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        file: Optional[TextIO] = None,
        also_stdout: bool = True,
    ) -> None:
        self.level = level
        self._file = file
        self._also_stdout = also_stdout
        self._event_count: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core write
    # ------------------------------------------------------------------

    def _write(self, line: str) -> None:
        if self._also_stdout:
            print(line)
        if self._file:
            self._file.write(line + "\n")

    def _log(self, min_level: LogLevel, t: float, tag: str, msg: str) -> None:
        if self.level < min_level:
            return
        self._event_count[tag] = self._event_count.get(tag, 0) + 1
        self._write(f"[{t:9.4f}]  {tag:<22} {msg}")

    # ------------------------------------------------------------------
    # Per-event methods (called from Simulator handlers)
    # ------------------------------------------------------------------

    def flow_start(self, t: float, flow) -> None:
        self._log(LogLevel.INFO, t, "FLOW_START",
                  f"flow={flow.id}  {flow.model.value}  "
                  f"{flow.source.id}→{flow.destination.id}  rate={flow.rate} pkt/s")

    def flow_stop(self, t: float, flow) -> None:
        self._log(LogLevel.SUMMARY, t, "FLOW_STOP",
                  f"flow={flow.id}  {flow.model.value}  {flow.source.id}→{flow.destination.id}")

    def link_rate_change(self, t: float, link, new_rate: float) -> None:
        self._log(LogLevel.SUMMARY, t, "LINK_RATE_CHANGE",
                  f"link={link.source.id}→{link.target.id}  new_rate={new_rate} pkt/s")

    def packet_generate(self, t: float, pkt, flow) -> None:
        self._log(LogLevel.DEBUG, t, "PACKET_GENERATE",
                  f"pkt={pkt.id}  flow={flow.id}  size={pkt.size}B  pri={pkt.priority}")

    def queue_enqueue(self, t: float, pkt, node) -> None:
        q = node.default_queue
        self._log(LogLevel.DEBUG, t, "QUEUE_ENQUEUE",
                  f"pkt={pkt.id}  node={node.id}  "
                  f"queue={q.size}/{q.max_size} ({q.queue_occupancy*100:.0f}%)  "
                  f"state={node.state_machine.current_state.name}")

    def queue_full_drop(self, t: float, pkt, node) -> None:
        q = node.default_queue
        self._log(LogLevel.INFO, t, "QUEUE_DROP(full)",
                  f"pkt={pkt.id}  node={node.id}  pri={pkt.priority}  "
                  f"queue={q.size}/{q.max_size} FULL")

    def congestion_drop(self, t: float, pkt, node) -> None:
        self._log(LogLevel.INFO, t, "QUEUE_DROP(congestion)",
                  f"pkt={pkt.id}  node={node.id}  pri={pkt.priority}  "
                  f"state={node.state_machine.current_state.name}")

    def queue_dequeue(self, t: float, pkt, node, wait: float) -> None:
        q = node.default_queue
        self._log(LogLevel.DEBUG, t, "QUEUE_DEQUEUE",
                  f"pkt={pkt.id}  node={node.id}  "
                  f"wait={wait*1000:.1f}ms  "
                  f"queue={q.size}/{q.max_size}  "
                  f"state={node.state_machine.current_state.name}")

    def state_update(self, t: float, node, old_state, new_state) -> None:
        self._log(LogLevel.INFO, t, "STATE_UPDATE",
                  f"node={node.id}  {old_state.name} → {new_state.name}")

    def packet_transmit(self, t: float, pkt, link) -> None:
        self._log(LogLevel.DEBUG, t, "PACKET_TRANSMIT",
                  f"pkt={pkt.id}  link={link.source.id}→{link.target.id}  "
                  f"delay={link.delay*1000:.1f}ms")

    def packet_arrival(self, t: float, pkt, node) -> None:
        self._log(LogLevel.DEBUG, t, "PACKET_ARRIVAL",
                  f"pkt={pkt.id}  node={node.id}  hop={pkt.hop}")

    def packet_deliver(self, t: float, pkt, latency: float) -> None:
        self._log(LogLevel.DEBUG, t, "PACKET_DELIVER",
                  f"pkt={pkt.id}  flow={pkt.flow.id if pkt.flow else '?'}  "
                  f"latency={latency*1000:.1f}ms")

    def link_failure(self, t: float, link) -> None:
        self._log(LogLevel.SUMMARY, t, "LINK_FAILURE",
                  f"link={link.source.id}→{link.target.id}")

    def link_recovery(self, t: float, link) -> None:
        self._log(LogLevel.SUMMARY, t, "LINK_RECOVERY",
                  f"link={link.source.id}→{link.target.id}")

    def metric_sample(self, t: float, snap, router_state: str) -> None:
        self._log(
            LogLevel.SUMMARY, t, "METRIC_SAMPLE",
            f"tput={snap.throughput:5.2f} pkt/s  "
            f"PDR={snap.packet_delivery_ratio:.3f}  "
            f"lat={snap.end_to_end_latency*1000:7.2f}ms  "
            f"q_occ={snap.queue_occupancy:.3f}  "
            f"fairness={snap.fairness:.3f}  "
            f"drops={snap.drop_count:4d}  "
            f"state={router_state}",
        )

    # ------------------------------------------------------------------
    # Summary at end of simulation
    # ------------------------------------------------------------------

    def print_summary(self, metrics, topology) -> None:
        if self.level == LogLevel.NONE:
            return
        sep = "─" * 68
        self._write(f"\n{sep}")
        self._write("  EVENT COUNTS")
        self._write(sep)
        for tag, count in sorted(self._event_count.items(), key=lambda x: -x[1]):
            self._write(f"  {tag:<26} {count:>6}")
        self._write(sep)
        self._write("  FINAL METRICS")
        self._write(sep)
        self._write(f"  Generated          {metrics.total_generated:>8}")
        self._write(f"  Delivered          {metrics.total_delivered:>8}")
        self._write(f"  Dropped            {metrics.total_dropped:>8}")
        self._write(f"  PDR                {metrics.collect_packet_delivery_ratio():>8.3f}")
        self._write(f"  Avg latency        {metrics.collect_end_to_end_latency()*1000:>7.1f} ms")
        self._write(f"  State transitions  {metrics.collect_congestion_state_transitions():>8}")
        for node in topology.nodes:
            if node.id == "router" or node.queues[0].drop_events > 0:
                q = node.default_queue
                self._write(
                    f"  Node [{node.id:<8}]   "
                    f"drops={q.drop_events}  "
                    f"avg_wait={q.waiting_time*1000:.1f}ms  "
                    f"state={node.state_machine.current_state.name}"
                )
        self._write(sep)
