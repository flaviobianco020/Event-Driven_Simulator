import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from simulator import (
    Event, EventType, EventScheduler,
    NetworkTopology, Node, Link, QueueManager,
    CongestionStateMachine, CongestionState,
    Packet, Flow, FlowModel, TrafficClass, TrafficGenerator,
    MetricsEngine, ConfigurationManager, Simulator,
)


# ── EventType ────────────────────────────────────────────────────────────────

def test_event_type_enum_values():
    required = {
        "FLOW_START", "FLOW_STOP", "PACKET_GENERATE", "PACKET_ARRIVAL",
        "QUEUE_ENQUEUE", "QUEUE_DEQUEUE", "PACKET_DROP",
        "PACKET_TRANSMIT", "PACKET_DELIVER",
        "LINK_RATE_CHANGE", "LINK_FAILURE", "LINK_RECOVERY",
        "METRIC_SAMPLE", "STATE_UPDATE",
    }
    assert required == {e.name for e in EventType}


# ── Event ordering ────────────────────────────────────────────────────────────

def test_event_order_by_time():
    e1 = Event(simulation_time=2.0, type=EventType.METRIC_SAMPLE)
    e2 = Event(simulation_time=1.0, type=EventType.METRIC_SAMPLE)
    assert e2 < e1

def test_event_order_fifo_same_time():
    e1 = Event(simulation_time=1.0, type=EventType.FLOW_START)
    e2 = Event(simulation_time=1.0, type=EventType.FLOW_STOP)
    assert e1 < e2


# ── EventScheduler ────────────────────────────────────────────────────────────

def test_scheduler_processes_in_time_order():
    s = EventScheduler()
    seen = []
    s.register(EventType.METRIC_SAMPLE, lambda e, sc: seen.append(e.simulation_time))
    for t in [3.0, 1.0, 2.0, 0.5]:
        s.schedule_event(Event(simulation_time=t, type=EventType.METRIC_SAMPLE))
    s.run(end_time=5.0)
    assert seen == sorted(seen)

def test_scheduler_respects_end_time():
    s = EventScheduler()
    seen = []
    s.register(EventType.METRIC_SAMPLE, lambda e, sc: seen.append(e.simulation_time))
    for t in [1.0, 3.0, 5.0, 7.0]:
        s.schedule_event(Event(simulation_time=t, type=EventType.METRIC_SAMPLE))
    s.run(end_time=5.0)
    assert all(t <= 5.0 for t in seen)
    assert 7.0 not in seen

def test_scheduler_stop():
    s = EventScheduler()
    seen = []
    def handler(e, sc):
        seen.append(e.simulation_time)
        if e.simulation_time >= 2.0:
            sc.stop()
    s.register(EventType.METRIC_SAMPLE, handler)
    for t in [1.0, 2.0, 3.0, 4.0]:
        s.schedule_event(Event(simulation_time=t, type=EventType.METRIC_SAMPLE))
    s.run()
    assert 3.0 not in seen


# ── CongestionStateMachine ────────────────────────────────────────────────────

def test_congestion_normal():
    csm = CongestionStateMachine()
    assert csm.evaluate(0.3) == CongestionState.NORMAL

def test_congestion_transitions():
    csm = CongestionStateMachine()
    assert csm.evaluate(0.55) == CongestionState.HEADER_COMPRESSION
    assert csm.evaluate(0.72) == CongestionState.DELTA_COMPRESSION
    assert csm.evaluate(0.87) == CongestionState.INCREMENTAL_COMPRESSION
    assert csm.evaluate(0.97) == CongestionState.DROP_LOW_PRIORITY

def test_congestion_update_records_history():
    csm = CongestionStateMachine()
    changed = csm.update(0.6, sim_time=5.0)
    assert changed
    assert csm.current_state == CongestionState.HEADER_COMPRESSION
    assert len(csm.history) == 1
    assert csm.history[0] == (5.0, CongestionState.HEADER_COMPRESSION)

def test_congestion_no_transition_if_same():
    csm = CongestionStateMachine()
    csm.update(0.6)
    changed = csm.update(0.55)  # still HEADER_COMPRESSION
    assert not changed


# ── QueueManager ──────────────────────────────────────────────────────────────

def make_pkt(priority=0) -> Packet:
    return Packet(size=100, priority=priority, creation_time=0.0)

def test_queue_enqueue_dequeue():
    q = QueueManager(max_size=5, service_rate=10.0)
    p = make_pkt()
    assert q.enqueue(p)
    assert q.size == 1
    assert q.dequeue() is p

def test_queue_full_drops():
    q = QueueManager(max_size=2, service_rate=10.0)
    q.enqueue(make_pkt())
    q.enqueue(make_pkt())
    assert not q.enqueue(make_pkt())
    assert q.drop_events == 1

def test_queue_occupancy():
    q = QueueManager(max_size=10, service_rate=10.0)
    for _ in range(5):
        q.enqueue(make_pkt())
    assert q.queue_occupancy == pytest.approx(0.5)

def test_queue_service_completion_monotonic():
    q = QueueManager(max_size=100, service_rate=10.0)
    t1 = q.next_service_completion(0.0)
    t2 = q.next_service_completion(0.0)
    t3 = q.next_service_completion(0.0)
    assert t1 < t2 < t3


# ── NetworkTopology ───────────────────────────────────────────────────────────

def test_single_bottleneck_topology():
    topo = NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0)
    node_ids = {n.id for n in topo.nodes}
    assert "router" in node_ids
    assert "dst" in node_ids
    assert "src0" in node_ids and "src1" in node_ids

def test_topology_get_path_single_bottleneck():
    topo = NetworkTopology.single_bottleneck(n_sources=1)
    path = topo.get_path("src0", "dst")
    assert len(path) == 2
    assert path[0].source.id == "src0"
    assert path[0].target.id == "router"
    assert path[1].source.id == "router"
    assert path[1].target.id == "dst"

def test_topology_get_path_not_found():
    topo = NetworkTopology.single_bottleneck(n_sources=1)
    path = topo.get_path("dst", "src0")
    assert path == []

def test_topology_link_failure():
    topo = NetworkTopology.single_bottleneck(n_sources=1)
    link = topo.get_link("router", "dst")
    link.active = False
    assert not link.active


# ── Full Simulation ───────────────────────────────────────────────────────────

def _build_sim(end_time=50.0, rate_video=8.0, rate_control=5.0):
    config = ConfigurationManager(random_seed=99)
    topo = NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0, queue_size=20)
    src0, src1, dst = topo.get_node("src0"), topo.get_node("src1"), topo.get_node("dst")
    video_cls   = TrafficClass((1400, 1500), priority_level=1, latency_sensitivity=True)
    control_cls = TrafficClass((100, 100),   priority_level=0)
    gen = (
        TrafficGenerator()
        .add_flow(Flow(FlowModel.POISSON, video_cls,   src0, dst, rate_video))
        .add_flow(Flow(FlowModel.CONTROL, control_cls, src1, dst, rate_control))
    )
    metrics = MetricsEngine()
    sim = Simulator(config, topo, gen, metrics, end_time=end_time, metric_interval=end_time + 1)
    return sim, metrics, topo

def test_simulation_delivers_packets():
    sim, metrics, _ = _build_sim(end_time=20.0)
    sim.run()
    assert metrics.total_delivered > 0

def test_simulation_generates_packets():
    sim, metrics, _ = _build_sim(end_time=20.0)
    sim.run()
    assert metrics.total_generated > 0

def test_simulation_drops_under_overload():
    sim, metrics, _ = _build_sim(end_time=50.0, rate_video=8.0, rate_control=5.0)
    sim.run()
    assert metrics.total_dropped > 0

def test_simulation_pdr_under_one_when_overloaded():
    sim, metrics, _ = _build_sim(end_time=50.0, rate_video=8.0, rate_control=5.0)
    sim.run()
    assert metrics.collect_packet_delivery_ratio() < 1.0

def test_simulation_congestion_state_transitions():
    sim, metrics, topo = _build_sim(end_time=50.0)
    sim.run()
    assert metrics.collect_congestion_state_transitions() > 0

def test_simulation_link_failure_increases_drops():
    sim1, m1, topo1 = _build_sim(end_time=50.0)
    sim1.run()
    drops_normal = m1.total_dropped

    sim2, m2, topo2 = _build_sim(end_time=50.0)
    link = topo2.get_link("router", "dst")
    sim2.scheduler.schedule_event(Event(
        simulation_time=10.0, type=EventType.LINK_FAILURE, link=link
    ))
    sim2.scheduler.schedule_event(Event(
        simulation_time=40.0, type=EventType.LINK_RECOVERY, link=link
    ))
    sim2.run()
    assert m2.total_dropped > drops_normal

def test_metrics_engine_records_correctly():
    m = MetricsEngine()
    m.record_generated()
    m.record_generated()
    m.record_delivered(0.01, flow_id=0)
    m.record_dropped()
    assert m.total_generated == 2
    assert m.total_delivered == 1
    assert m.total_dropped == 1
    assert m.collect_packet_delivery_ratio() == pytest.approx(0.5)
    assert m.collect_end_to_end_latency() == pytest.approx(0.01)

def test_fairness_single_flow():
    m = MetricsEngine()
    for _ in range(10):
        m.record_delivered(0.01, flow_id=0)
    assert m.collect_fairness() == pytest.approx(1.0)

def test_fairness_perfectly_equal():
    m = MetricsEngine()
    for _ in range(5):
        m.record_delivered(0.01, flow_id=0)
        m.record_delivered(0.01, flow_id=1)
    assert m.collect_fairness() == pytest.approx(1.0)

def test_fairness_unequal():
    m = MetricsEngine()
    for _ in range(9):
        m.record_delivered(0.01, flow_id=0)
    m.record_delivered(0.01, flow_id=1)   # flow 1 gets only 1 out of 10
    # J = (10)^2 / (2 * (81+1)) = 100/164 ≈ 0.61
    assert m.collect_fairness() < 1.0
    assert m.collect_fairness() > 0.0

def test_link_rate_change_affects_source_queue():
    topo = NetworkTopology.single_bottleneck(n_sources=1, bottleneck_capacity=10.0)
    link = topo.get_link("router", "dst")
    router = topo.get_node("router")
    assert router.default_queue.service_rate == pytest.approx(10.0)
    link.capacity = 4.0
    router.default_queue.service_rate = 4.0
    assert router.default_queue.service_rate == pytest.approx(4.0)
