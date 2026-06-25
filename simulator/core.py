from __future__ import annotations
import random
from typing import Optional
from .config import ConfigurationManager
from .event import Event, EventType
from .scheduler import EventScheduler
from .network.topology import NetworkTopology
from .traffic.generator import TrafficGenerator
from .metrics import MetricsEngine
from .logger import TraceLogger, LogLevel


class Simulator:
    def __init__(
        self,
        config: ConfigurationManager,
        topology: NetworkTopology,
        generator: TrafficGenerator,
        metrics: MetricsEngine,
        end_time: float = 100.0,
        metric_interval: float = 10.0,
        logger: Optional[TraceLogger] = None,
    ) -> None:
        self.config = config
        self.topology = topology
        self.generator = generator
        self.metrics = metrics
        self.end_time = end_time
        self.metric_interval = metric_interval
        self.scheduler = EventScheduler()
        self._rng = random.Random(config.random_seed)
        self._log = logger or TraceLogger(level=LogLevel.NONE)
        self._register_handlers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        for flow in self.generator.generate_flows():
            self.scheduler.schedule_event(Event(
                simulation_time=0.0,
                type=EventType.FLOW_START,
                metadata={"flow": flow},
            ))
        self.scheduler.schedule_event(Event(
            simulation_time=self.metric_interval,
            type=EventType.METRIC_SAMPLE,
        ))
        self.scheduler.run(end_time=self.end_time)

    def stop(self) -> None:
        self.scheduler.stop()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        reg = self.scheduler.register
        reg(EventType.FLOW_START,      self._on_flow_start)
        reg(EventType.PACKET_GENERATE, self._on_packet_generate)
        reg(EventType.QUEUE_ENQUEUE,   self._on_queue_enqueue)
        reg(EventType.QUEUE_DEQUEUE,   self._on_queue_dequeue)
        reg(EventType.PACKET_DROP,     self._on_packet_drop)
        reg(EventType.PACKET_TRANSMIT, self._on_packet_transmit)
        reg(EventType.PACKET_ARRIVAL,  self._on_packet_arrival)
        reg(EventType.PACKET_DELIVER,  self._on_packet_deliver)
        reg(EventType.METRIC_SAMPLE,   self._on_metric_sample)
        reg(EventType.FLOW_STOP,       self._on_flow_stop)
        reg(EventType.LINK_FAILURE,    self._on_link_failure)
        reg(EventType.LINK_RECOVERY,   self._on_link_recovery)
        reg(EventType.LINK_RATE_CHANGE, self._on_link_rate_change)
        reg(EventType.STATE_UPDATE,    self._on_state_update)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_flow_stop(self, event: Event, scheduler: EventScheduler) -> None:
        flow = event.metadata.get("flow")
        if flow:
            flow.active = False
            self._log.flow_stop(event.simulation_time, flow)

    def _on_flow_start(self, event: Event, scheduler: EventScheduler) -> None:
        flow = event.metadata["flow"]
        self._log.flow_start(event.simulation_time, flow)
        scheduler.schedule_event(Event(
            simulation_time=event.simulation_time,
            type=EventType.PACKET_GENERATE,
            metadata={"flow": flow},
        ))

    def _on_packet_generate(self, event: Event, scheduler: EventScheduler) -> None:
        flow = event.metadata["flow"]
        t = event.simulation_time
        if not flow.active:
            return

        path = self.topology.get_path(flow.source.id, flow.destination.id)
        if not path:
            return

        pkt = self.generator.create_packet(flow, t, self._rng)
        pkt.path = path
        pkt.hop = 0
        self.metrics.record_generated()
        self._log.packet_generate(t, pkt, flow)

        scheduler.schedule_event(Event(
            simulation_time=t,
            type=EventType.QUEUE_ENQUEUE,
            packet=pkt,
            node=path[0].source,
        ))

        inter = flow.inter_arrival(self._rng)
        scheduler.schedule_event(Event(
            simulation_time=t + inter,
            type=EventType.PACKET_GENERATE,
            metadata={"flow": flow},
        ))

    def _on_queue_enqueue(self, event: Event, scheduler: EventScheduler) -> None:
        pkt = event.packet
        node = event.node
        t = event.simulation_time
        queue = node.default_queue

        if node.state_machine.current_state.value >= 4 and pkt.priority > 0:
            self._log.congestion_drop(t, pkt, node)
            scheduler.schedule_event(Event(
                simulation_time=t,
                type=EventType.PACKET_DROP,
                packet=pkt,
                node=node,
            ))
            return

        pkt.enqueue_time = t
        success = queue.enqueue(pkt)
        if not success:
            self._log.queue_full_drop(t, pkt, node)
            scheduler.schedule_event(Event(
                simulation_time=t,
                type=EventType.PACKET_DROP,
                packet=pkt,
                node=node,
            ))
            return

        self._log.queue_enqueue(t, pkt, node)
        dequeue_t = queue.next_service_completion(t)
        scheduler.schedule_event(Event(
            simulation_time=dequeue_t,
            type=EventType.QUEUE_DEQUEUE,
            node=node,
        ))

    def _on_queue_dequeue(self, event: Event, scheduler: EventScheduler) -> None:
        node = event.node
        t = event.simulation_time
        queue = node.default_queue

        pkt = queue.dequeue()
        if pkt is None:
            return

        wait = t - pkt.enqueue_time
        queue.record_wait(pkt.enqueue_time, t)
        self._log.queue_dequeue(t, pkt, node, wait)

        old_state = node.state_machine.current_state
        changed = node.state_machine.update(queue.queue_occupancy, t)
        if changed:
            self.metrics.record_state_transition()
            self._log.state_update(t, node, old_state, node.state_machine.current_state)
            scheduler.schedule_event(Event(
                simulation_time=t,
                type=EventType.STATE_UPDATE,
                node=node,
                metadata={"state": node.state_machine.current_state},
            ))

        if pkt.hop >= len(pkt.path):
            scheduler.schedule_event(Event(
                simulation_time=t,
                type=EventType.PACKET_DELIVER,
                packet=pkt,
                node=node,
            ))
            return

        link = pkt.path[pkt.hop]
        scheduler.schedule_event(Event(
            simulation_time=t,
            type=EventType.PACKET_TRANSMIT,
            packet=pkt,
            link=link,
        ))

    def _on_packet_transmit(self, event: Event, scheduler: EventScheduler) -> None:
        link = event.link
        pkt = event.packet
        t = event.simulation_time

        if not link.active:
            self._log.queue_full_drop(t, pkt, link.source)
            scheduler.schedule_event(Event(
                simulation_time=t,
                type=EventType.PACKET_DROP,
                packet=pkt,
                node=link.source,
            ))
            return

        self._log.packet_transmit(t, pkt, link)
        scheduler.schedule_event(Event(
            simulation_time=t + link.delay,
            type=EventType.PACKET_ARRIVAL,
            packet=pkt,
            node=link.target,
        ))

    def _on_packet_arrival(self, event: Event, scheduler: EventScheduler) -> None:
        pkt = event.packet
        node = event.node
        t = event.simulation_time

        pkt.hop += 1
        self._log.packet_arrival(t, pkt, node)

        if pkt.hop >= len(pkt.path):
            scheduler.schedule_event(Event(
                simulation_time=t,
                type=EventType.PACKET_DELIVER,
                packet=pkt,
                node=node,
            ))
        else:
            scheduler.schedule_event(Event(
                simulation_time=t,
                type=EventType.QUEUE_ENQUEUE,
                packet=pkt,
                node=node,
            ))

    def _on_packet_deliver(self, event: Event, scheduler: EventScheduler) -> None:
        pkt = event.packet
        t = event.simulation_time
        latency = t - pkt.creation_time
        self.metrics.record_delivered(latency)
        self._log.packet_deliver(t, pkt, latency)

    def _on_packet_drop(self, event: Event, scheduler: EventScheduler) -> None:
        self.metrics.record_dropped()

    def _on_metric_sample(self, event: Event, scheduler: EventScheduler) -> None:
        t = event.simulation_time
        snap = self.metrics.sample(t, self.topology.nodes)
        try:
            router_state = self.topology.get_node("router").state_machine.current_state.name
        except KeyError:
            router_state = "N/A"
        self._log.metric_sample(t, snap, router_state)
        if t + self.metric_interval <= self.end_time:
            scheduler.schedule_event(Event(
                simulation_time=t + self.metric_interval,
                type=EventType.METRIC_SAMPLE,
            ))

    def _on_link_rate_change(self, event: Event, scheduler: EventScheduler) -> None:
        link = event.link
        new_rate = event.metadata.get("new_rate")
        if link and new_rate is not None:
            link.capacity = new_rate
            link.target.default_queue.service_rate = new_rate
            self._log.link_rate_change(event.simulation_time, link, new_rate)

    def _on_link_failure(self, event: Event, scheduler: EventScheduler) -> None:
        if event.link:
            event.link.active = False
            self._log.link_failure(event.simulation_time, event.link)

    def _on_link_recovery(self, event: Event, scheduler: EventScheduler) -> None:
        if event.link:
            event.link.active = True
            self._log.link_recovery(event.simulation_time, event.link)

    def _on_state_update(self, event: Event, scheduler: EventScheduler) -> None:
        pass  # hook for Phase 2 control layer
