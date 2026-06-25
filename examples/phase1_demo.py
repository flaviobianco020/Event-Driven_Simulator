"""
Phase 1 Demo — Single Bottleneck Congestion

Usage:
    python3 examples/phase1_demo.py              # default: SUMMARY log
    python3 examples/phase1_demo.py summary      # metrics + state changes + link events
    python3 examples/phase1_demo.py info         # + drops
    python3 examples/phase1_demo.py debug        # every single event
    python3 examples/phase1_demo.py debug file   # debug + save to logs/trace.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator import (
    ConfigurationManager, NetworkTopology, TrafficGenerator, MetricsEngine,
    Simulator, Flow, FlowModel, TrafficClass, Event, EventType,
    TraceLogger, LogLevel,
)

SIM_TIME = 100.0
METRIC_INTERVAL = 10.0

LEVELS = {
    "none":    LogLevel.NONE,
    "summary": LogLevel.SUMMARY,
    "info":    LogLevel.INFO,
    "debug":   LogLevel.DEBUG,
}


def main() -> None:
    level_name = sys.argv[1].lower() if len(sys.argv) > 1 else "summary"
    level = LEVELS.get(level_name, LogLevel.SUMMARY)
    save_to_file = len(sys.argv) > 2 and sys.argv[2] == "file"

    log_file = None
    if save_to_file:
        os.makedirs("logs", exist_ok=True)
        log_file = open("logs/trace.log", "w")
        print(f"→ Saving trace to logs/trace.log")

    logger = TraceLogger(level=level, file=log_file, also_stdout=True)

    config = ConfigurationManager(random_seed=42)

    topology = NetworkTopology.single_bottleneck(
        n_sources=2,
        bottleneck_capacity=10.0,
        bottleneck_delay=0.005,
        queue_size=20,
    )

    src0 = topology.get_node("src0")
    src1 = topology.get_node("src1")
    dst  = topology.get_node("dst")

    video_class   = TrafficClass((1400, 1500), priority_level=1,
                                 latency_sensitivity=True, compression_sensitivity=True)
    control_class = TrafficClass((100, 100),   priority_level=0,
                                 latency_sensitivity=True)

    generator = (
        TrafficGenerator()
        .add_flow(Flow(FlowModel.POISSON, video_class,   src0, dst, rate=8.0))
        .add_flow(Flow(FlowModel.CONTROL, control_class, src1, dst, rate=5.0))
    )

    metrics = MetricsEngine()

    sim = Simulator(
        config=config,
        topology=topology,
        generator=generator,
        metrics=metrics,
        end_time=SIM_TIME,
        metric_interval=METRIC_INTERVAL,
        logger=logger,
    )

    bottleneck_link = topology.get_link("router", "dst")
    sim.scheduler.schedule_event(Event(
        simulation_time=40.0, type=EventType.LINK_FAILURE, link=bottleneck_link
    ))
    sim.scheduler.schedule_event(Event(
        simulation_time=55.0, type=EventType.LINK_RECOVERY, link=bottleneck_link
    ))

    print(f"{'='*68}")
    print(f"  EDS Phase 1 — Single Bottleneck  [log={level_name.upper()}]")
    print(f"  Bottleneck : {bottleneck_link}")
    print(f"  Flows      : {generator.generate_flows()}")
    print(f"{'='*68}")

    sim.run()

    logger.print_summary(metrics, topology)

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
