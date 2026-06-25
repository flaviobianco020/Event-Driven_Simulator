"""
Phase 1 — 6 canonical congestion scenarios (PDF §4.5)

Usage:
    python3 examples/scenarios.py [1-6] [summary|info|debug]

    1  single_bottleneck    — basic overload, queue fills, drops occur
    2  flash_crowd          — sudden traffic surge at t=20
    3  bandwidth_degradation — link capacity halved at t=30
    4  link_failure_recovery — link down t=30, up t=50
    5  persistent_overload  — sustained overload for full simulation
    6  mixed_telemetry_video — 3 traffic classes: CONTROL + TELEMETRY + VIDEO
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator import (
    ConfigurationManager, NetworkTopology, TrafficGenerator, MetricsEngine,
    Simulator, Flow, FlowModel, TrafficClass, Event, EventType,
    TraceLogger, LogLevel,
)

LEVELS = {"summary": LogLevel.SUMMARY, "info": LogLevel.INFO, "debug": LogLevel.DEBUG}

# ── shared traffic classes ─────────────────────────────────────────────────────

def _video_class()     -> TrafficClass:
    return TrafficClass((1400, 1500), priority_level=2, latency_sensitivity=True,  compression_sensitivity=True)

def _telemetry_class() -> TrafficClass:
    return TrafficClass((200,  300),  priority_level=1, latency_sensitivity=False, compression_sensitivity=True)

def _control_class()   -> TrafficClass:
    return TrafficClass((100,  100),  priority_level=0, latency_sensitivity=True,  compression_sensitivity=False)


def _run(sim: Simulator, logger: TraceLogger, title: str) -> None:
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")
    sim.run()
    logger.print_summary(sim.metrics, sim.topology)


# ── 1. Single bottleneck congestion ───────────────────────────────────────────

def scenario_single_bottleneck(logger: TraceLogger) -> None:
    config   = ConfigurationManager(random_seed=1)
    topology = NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0, queue_size=20)
    src0, src1, dst = topology.get_node("src0"), topology.get_node("src1"), topology.get_node("dst")
    gen = (
        TrafficGenerator()
        .add_flow(Flow(FlowModel.POISSON, _video_class(),   src0, dst, rate=8.0))
        .add_flow(Flow(FlowModel.CONTROL, _control_class(), src1, dst, rate=5.0))
    )
    sim = Simulator(config, topology, gen, MetricsEngine(), end_time=60.0, metric_interval=10.0, logger=logger)
    _run(sim, logger, "Scenario 1 — Single Bottleneck Congestion  (load=13 > cap=10)")


# ── 2. Flash crowd traffic surge ──────────────────────────────────────────────

def scenario_flash_crowd(logger: TraceLogger) -> None:
    config   = ConfigurationManager(random_seed=2)
    topology = NetworkTopology.single_bottleneck(n_sources=3, bottleneck_capacity=10.0, queue_size=20)
    src0, src1, src2 = topology.get_node("src0"), topology.get_node("src1"), topology.get_node("src2")
    dst = topology.get_node("dst")

    normal_flow = Flow(FlowModel.POISSON, _video_class(), src0, dst, rate=4.0)
    surge_flow  = Flow(FlowModel.BURSTY,  _video_class(), src1, dst, rate=6.0)
    ctrl_flow   = Flow(FlowModel.CONTROL, _control_class(), src2, dst, rate=2.0)

    gen = TrafficGenerator().add_flow(normal_flow).add_flow(ctrl_flow)
    sim = Simulator(config, topology, gen, MetricsEngine(), end_time=80.0, metric_interval=10.0, logger=logger)

    # Flash crowd: extra surge flow starts at t=20, stops at t=50
    sim.scheduler.schedule_event(Event(simulation_time=20.0, type=EventType.FLOW_START, metadata={"flow": surge_flow}))
    sim.scheduler.schedule_event(Event(simulation_time=50.0, type=EventType.FLOW_STOP,  metadata={"flow": surge_flow}))
    gen.add_flow(surge_flow)   # register so path can be computed on FLOW_START

    _run(sim, logger, "Scenario 2 — Flash Crowd Traffic Surge  (surge flow: t=20→50)")


# ── 3. Sudden bandwidth degradation ──────────────────────────────────────────

def scenario_bandwidth_degradation(logger: TraceLogger) -> None:
    config   = ConfigurationManager(random_seed=3)
    topology = NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0, queue_size=20)
    src0, src1, dst = topology.get_node("src0"), topology.get_node("src1"), topology.get_node("dst")
    gen = (
        TrafficGenerator()
        .add_flow(Flow(FlowModel.POISSON, _video_class(),   src0, dst, rate=7.0))
        .add_flow(Flow(FlowModel.CONTROL, _control_class(), src1, dst, rate=2.0))
    )
    sim = Simulator(config, topology, gen, MetricsEngine(), end_time=80.0, metric_interval=10.0, logger=logger)
    bottleneck = topology.get_link("router", "dst")
    # At t=30 bandwidth drops from 10→4 pkt/s; at t=60 restores to 10
    sim.scheduler.schedule_event(Event(simulation_time=30.0, type=EventType.LINK_RATE_CHANGE,
                                       link=bottleneck, metadata={"new_rate": 4.0}))
    sim.scheduler.schedule_event(Event(simulation_time=60.0, type=EventType.LINK_RATE_CHANGE,
                                       link=bottleneck, metadata={"new_rate": 10.0}))
    _run(sim, logger, "Scenario 3 — Sudden Bandwidth Degradation  (10→4 pkt/s at t=30, restored t=60)")


# ── 4. Link failure and recovery ──────────────────────────────────────────────

def scenario_link_failure_recovery(logger: TraceLogger) -> None:
    config   = ConfigurationManager(random_seed=4)
    topology = NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0, queue_size=20)
    src0, src1, dst = topology.get_node("src0"), topology.get_node("src1"), topology.get_node("dst")
    gen = (
        TrafficGenerator()
        .add_flow(Flow(FlowModel.POISSON, _video_class(),   src0, dst, rate=6.0))
        .add_flow(Flow(FlowModel.CONTROL, _control_class(), src1, dst, rate=3.0))
    )
    sim = Simulator(config, topology, gen, MetricsEngine(), end_time=90.0, metric_interval=10.0, logger=logger)
    link = topology.get_link("router", "dst")
    sim.scheduler.schedule_event(Event(simulation_time=30.0, type=EventType.LINK_FAILURE,  link=link))
    sim.scheduler.schedule_event(Event(simulation_time=55.0, type=EventType.LINK_RECOVERY, link=link))
    _run(sim, logger, "Scenario 4 — Link Failure & Recovery  (down t=30, up t=55)")


# ── 5. Persistent overload ────────────────────────────────────────────────────

def scenario_persistent_overload(logger: TraceLogger) -> None:
    config   = ConfigurationManager(random_seed=5)
    topology = NetworkTopology.single_bottleneck(n_sources=3, bottleneck_capacity=10.0, queue_size=20)
    src0, src1, src2 = (topology.get_node(f"src{i}") for i in range(3))
    dst = topology.get_node("dst")
    gen = (
        TrafficGenerator()
        .add_flow(Flow(FlowModel.POISSON, _video_class(),     src0, dst, rate=7.0))
        .add_flow(Flow(FlowModel.POISSON, _telemetry_class(), src1, dst, rate=5.0))
        .add_flow(Flow(FlowModel.CONTROL, _control_class(),   src2, dst, rate=3.0))
    )
    # Total load = 15 pkt/s >> capacity 10 → persistent overload throughout
    sim = Simulator(config, topology, gen, MetricsEngine(), end_time=100.0, metric_interval=10.0, logger=logger)
    _run(sim, logger, "Scenario 5 — Persistent Overload  (load=15 >> cap=10, no recovery)")


# ── 6. Mixed telemetry and video traffic ──────────────────────────────────────

def scenario_mixed_traffic(logger: TraceLogger) -> None:
    config   = ConfigurationManager(random_seed=6)
    topology = NetworkTopology.single_bottleneck(n_sources=3, bottleneck_capacity=10.0, queue_size=30)
    src0, src1, src2 = (topology.get_node(f"src{i}") for i in range(3))
    dst = topology.get_node("dst")
    gen = (
        TrafficGenerator()
        .add_flow(Flow(FlowModel.VIDEO,              _video_class(),     src0, dst, rate=5.0))
        .add_flow(Flow(FlowModel.PERIODIC_TELEMETRY, _telemetry_class(), src1, dst, rate=4.0))
        .add_flow(Flow(FlowModel.CONTROL,            _control_class(),   src2, dst, rate=2.0))
    )
    # Load ≈ 11 > cap=10 — moderate overload; priority=0 (control) should be protected
    sim = Simulator(config, topology, gen, MetricsEngine(), end_time=80.0, metric_interval=10.0, logger=logger)
    _run(sim, logger, "Scenario 6 — Mixed Telemetry & Video  (VIDEO pri=2, TELEMETRY pri=1, CONTROL pri=0)")


# ── main ──────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "1": scenario_single_bottleneck,
    "2": scenario_flash_crowd,
    "3": scenario_bandwidth_degradation,
    "4": scenario_link_failure_recovery,
    "5": scenario_persistent_overload,
    "6": scenario_mixed_traffic,
}

if __name__ == "__main__":
    which    = sys.argv[1] if len(sys.argv) > 1 else "all"
    lv_name  = sys.argv[2] if len(sys.argv) > 2 else "summary"
    level    = LEVELS.get(lv_name, LogLevel.SUMMARY)
    logger   = TraceLogger(level=level, also_stdout=True)

    if which == "all":
        for fn in SCENARIOS.values():
            fn(logger)
    elif which in SCENARIOS:
        SCENARIOS[which](logger)
    else:
        print(f"Usage: python3 scenarios.py [1-6|all] [summary|info|debug]")
