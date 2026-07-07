# Event-Driven Network Simulator — Phase 1

> MS Thesis: *Towards Agentic Networks: An Architecture for Autonomous Congestion Management*

Phase 1 delivers a fully functional discrete-event network simulator that models packet-level traffic, queue dynamics, and congestion state machines. Phases 2–4 will add a rule-based control layer, MARL, and an agentic AI layer on top of this foundation.

---

## Architecture

```
simulator/
├── core.py             # Simulator — orchestrates all event handlers
├── config.py           # ConfigurationManager — seed + parameter store
├── event.py            # Event, EventType (14 types)
├── scheduler.py        # EventScheduler — min-heap priority queue
├── metrics.py          # MetricsEngine, MetricSnapshot
├── logger.py           # TraceLogger, LogLevel
├── network/
│   ├── congestion.py   # CongestionStateMachine, CongestionState
│   ├── queue_manager.py# QueueManager — FIFO tail-drop with HOL-blocking
│   ├── link.py         # Link — directed edge with capacity and delay
│   ├── node.py         # Node — holds queues + state machine
│   └── topology.py     # NetworkTopology — BFS routing, factory methods
└── traffic/
    ├── packet.py       # Packet — carries path[] and hop for routing
    ├── flow.py         # Flow, FlowModel (6 types), TrafficClass
    └── generator.py    # TrafficGenerator

examples/
├── phase1_demo.py      # Single bottleneck with link failure/recovery
└── scenarios.py        # 6 canonical congestion scenarios (PDF §4.5)

tests/
└── test_phase1.py      # 29 unit + integration tests
```

---

## Installation

Requires Python 3.10+. No external dependencies beyond `pytest` for testing.

```bash
git clone <repo>
cd Event-Driven_Simulator
pip install pytest          # only needed to run tests
```

---

## Quick Start

```python
from simulator import (
    ConfigurationManager, NetworkTopology, TrafficGenerator, MetricsEngine,
    Simulator, Flow, FlowModel, TrafficClass, TraceLogger, LogLevel,
)

config   = ConfigurationManager(random_seed=42)
topology = NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0, queue_size=20)
src0, src1, dst = topology.get_node("src0"), topology.get_node("src1"), topology.get_node("dst")

video_cls   = TrafficClass((1400, 1500), priority_level=2, latency_sensitivity=True)
control_cls = TrafficClass((100, 100),   priority_level=0)

gen = (
    TrafficGenerator()
    .add_flow(Flow(FlowModel.POISSON, video_cls,   src0, dst, rate=8.0))
    .add_flow(Flow(FlowModel.CONTROL, control_cls, src1, dst, rate=5.0))
)

logger = TraceLogger(level=LogLevel.SUMMARY, also_stdout=True)
sim    = Simulator(config, topology, gen, MetricsEngine(), end_time=60.0,
                   metric_interval=10.0, logger=logger)
sim.run()
logger.print_summary(sim.metrics, sim.topology)
```

---

## Running the Examples

### Interactive demo (single bottleneck + link failure at t=40)

```bash
python3 examples/phase1_demo.py [summary|info|debug] [logfile]
```

### Six canonical congestion scenarios

```bash
python3 examples/scenarios.py [1-6|all] [summary|info|debug]
```

| # | Scenario | Description |
|---|----------|-------------|
| 1 | Single bottleneck | Load 13 > cap 10 → steady-state drops |
| 2 | Flash crowd | Surge flow joins at t=20, leaves at t=50 |
| 3 | Bandwidth degradation | Cap 10→4 at t=30, restored at t=60 |
| 4 | Link failure & recovery | Link down t=30, up t=55 |
| 5 | Persistent overload | Load 15 >> cap 10, no relief |
| 6 | Mixed traffic classes | VIDEO pri=2, TELEMETRY pri=1, CONTROL pri=0 |

---

## Running the Tests

```bash
python3 -m pytest tests/test_phase1.py -v
```

All 29 tests should pass. The suite covers:

- `EventType` enum completeness (14 types)
- `Event` ordering: by time, then FIFO within same timestamp
- `EventScheduler`: time order, end-time boundary, `stop()` mid-run
- `CongestionStateMachine`: all 5 state thresholds, history recording, idempotent transitions
- `QueueManager`: enqueue/dequeue, tail-drop, occupancy, monotonic service completion
- `NetworkTopology`: node/link presence, BFS path correctness, unreachable returns `[]`
- Full simulation: packet delivery, drops under overload, PDR < 1 when congested
- `LINK_FAILURE` / `LINK_RECOVERY` events increase drop count
- Jain's Fairness Index: 1.0 for one flow, 1.0 for equal flows, < 1.0 for skewed flows
- `LINK_RATE_CHANGE` applies to the source (bottleneck) node's queue service rate

---

## Core Components

### EventScheduler

Min-heap priority queue. Events are ordered by `simulation_time`, with a monotonic sequence counter as the tiebreaker (FIFO within the same timestamp).

```python
s = EventScheduler()
s.register(EventType.METRIC_SAMPLE, my_handler)   # handler(event, scheduler)
s.schedule_event(Event(simulation_time=5.0, type=EventType.METRIC_SAMPLE))
s.run(end_time=100.0)
```

### EventType

14 domain-specific types covering the full packet lifecycle:

```
FLOW_START  FLOW_STOP  PACKET_GENERATE  PACKET_ARRIVAL
QUEUE_ENQUEUE  QUEUE_DEQUEUE  PACKET_DROP
PACKET_TRANSMIT  PACKET_DELIVER
LINK_RATE_CHANGE  LINK_FAILURE  LINK_RECOVERY
METRIC_SAMPLE  STATE_UPDATE
```

### NetworkTopology

Three built-in factory methods:

```python
# Two sources feeding a shared bottleneck router → dst
NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0,
                                   bottleneck_delay=0.005, queue_size=20)

# Linear chain: node0 → node1 → … → nodeN
NetworkTopology.multi_hop(n_hops=4, capacity=100.0, delay=0.002, queue_size=50)

# Bidirectional N×M grid
NetworkTopology.mesh(rows=3, cols=3, capacity=100.0, delay=0.001, queue_size=50)
```

Path computation uses BFS; the path (list of `Link` objects) is stored directly in each `Packet` so no global routing table is needed.

### CongestionStateMachine

Five states driven by queue occupancy thresholds:

| Occupancy | State |
|-----------|-------|
| < 50% | `NORMAL` |
| 50–70% | `HEADER_COMPRESSION` |
| 70–85% | `DELTA_COMPRESSION` |
| 85–95% | `INCREMENTAL_COMPRESSION` |
| ≥ 95% | `DROP_LOW_PRIORITY` |

In `DROP_LOW_PRIORITY` state, packets with `priority > 0` are dropped before entering the queue (protecting priority-0 control traffic).

### QueueManager

FIFO tail-drop queue with HOL-blocking via `next_service_completion(t)`:

```python
q = QueueManager(max_size=20, service_rate=10.0)   # 10 pkt/s
q.enqueue(pkt)          # returns False and increments drop_events if full
q.dequeue()             # returns next packet or None
q.queue_occupancy       # float in [0, 1]
q.next_service_completion(current_time)   # schedules dequeue, tracks _server_free_at
```

`service_rate` is updated by `LINK_RATE_CHANGE` events (applied to the source node of the link, which is the bottleneck).

### Traffic Models

| FlowModel | Inter-arrival |
|-----------|--------------|
| `CBR` | fixed = 1/rate |
| `POISSON` | exponential(1/rate) |
| `BURSTY` | exponential with burst multiplier |
| `PERIODIC_TELEMETRY` | fixed |
| `VIDEO` | exponential(1/rate) |
| `CONTROL` | fixed = 1/rate |

`TrafficClass` encodes `packet_size_distribution`, `priority_level`, `latency_sensitivity`, and `compression_sensitivity`.

### MetricsEngine

Collected per simulation run:

| Metric | Method |
|--------|--------|
| Throughput | `collect_throughput(t)` |
| Packet Delivery Ratio | `collect_packet_delivery_ratio()` |
| Mean end-to-end latency | `collect_end_to_end_latency()` |
| Queue occupancy (avg across all nodes) | `collect_queue_occupancy(nodes)` |
| Jain's Fairness Index | `collect_fairness()` |
| Congestion state transitions | `collect_congestion_state_transitions()` |

**Jain's Fairness Index** is computed per-flow at delivery time:

```
J = (Σ xᵢ)² / (n · Σ xᵢ²)   ∈ (0, 1],  1 = perfectly fair
```

Periodic snapshots (`MetricSnapshot`) are stored in `metrics.snapshots` for post-run analysis.

### TraceLogger

```python
logger = TraceLogger(
    level=LogLevel.SUMMARY,   # NONE | SUMMARY | INFO | DEBUG
    also_stdout=True,
    file=open("logs/trace.log", "w"),  # optional
)
```

Log levels control verbosity:

| Level | Events logged |
|-------|--------------|
| `SUMMARY` | METRIC_SAMPLE, LINK_FAILURE/RECOVERY, FLOW_STOP, LINK_RATE_CHANGE |
| `INFO` | + STATE_UPDATE, QUEUE_DROP(full), QUEUE_DROP(congestion), FLOW_START |
| `DEBUG` | + QUEUE_ENQUEUE/DEQUEUE, PACKET_GENERATE/TRANSMIT/ARRIVAL/DELIVER |

`logger.print_summary(metrics, topology)` prints event counts and final metrics at end of run.

---

## Injecting External Events

Any event can be pre-scheduled before `sim.run()`:

```python
from simulator import Event, EventType

link = topology.get_link("router", "dst")

# Link failure at t=30, recovery at t=55
sim.scheduler.schedule_event(Event(simulation_time=30.0, type=EventType.LINK_FAILURE,  link=link))
sim.scheduler.schedule_event(Event(simulation_time=55.0, type=EventType.LINK_RECOVERY, link=link))

# Bandwidth degradation at t=40
sim.scheduler.schedule_event(Event(simulation_time=40.0, type=EventType.LINK_RATE_CHANGE,
                                   link=link, metadata={"new_rate": 4.0}))

# Dynamic flow injection
sim.scheduler.schedule_event(Event(simulation_time=20.0, type=EventType.FLOW_START,
                                   metadata={"flow": surge_flow}))
sim.scheduler.schedule_event(Event(simulation_time=50.0, type=EventType.FLOW_STOP,
                                   metadata={"flow": surge_flow}))
```

---

## Phase 2 Hook

`Simulator._on_state_update` is intentionally a no-op:

```python
def _on_state_update(self, event: Event, scheduler: EventScheduler) -> None:
    pass  # hook for Phase 2 control layer
```

Phase 2 will register a control agent here to react to `STATE_UPDATE` events with compression and routing decisions.

---

## Phase 3 — MAPPO (Multi-Agent Reinforcement Learning)

Implementazione fedele al documento tecnico *"MAPPO — Fase 3 EDS"*
(`generate_mappo_doc.py`): il controllo della congestione diventa una policy
appresa con Multi-Agent Proximal Policy Optimization (CTDE).

| Componente | File | Specifica (doc) |
|---|---|---|
| Actor π(a\|o,θ) | `simulator/marl/networks.py` | 7→LayerNorm→64→Tanh→64→Tanh→3→Softmax (Tab. 5) |
| Critic V(s,φ) | `simulator/marl/networks.py` | (7N+4)→LayerNorm→128→Tanh→128→Tanh→1 (Tab. 6, N=1: 18.177 param) |
| GAE + rollout | `simulator/marl/buffer.py` | γ=0.99, λ=0.95, T=2048 (§3.3, Tab. 4) |
| Update PPO-CLIP | `simulator/marl/mappo.py` | ε=0.2, K=10 epoch, minibatch 256, lr 3e-4/1e-3, grad clip 10 (Tab. 4) |
| Ambiente Dec-POMDP | `simulator/marl/env.py` | obs dim=7 (Tab. 7), azioni {ESCALATE, MAINTAIN, DE-ESCALATE} (Tab. 8), reward = PDR − 0.3·drop − 0.2·lat/2s + 0.2·Jain (Tab. 9), Δt=1 s |
| Controller deploy | `simulator/marl/controller.py` | sostituto di `RuleBasedController.react()` (§7.1); solo Actor a runtime |
| Pipeline training | `examples/train_mappo.py` | 500 episodi, scenario random 1–6 da 100 s, eval argmax ogni 50, export JSON (Tab. 10) |

```bash
pip install numpy                      # unica dipendenza (solo Fase 3)
python3 examples/train_mappo.py        # training completo (500 episodi)
python3 examples/train_mappo.py --quick   # smoke test
python3 -m pytest tests/test_phase3.py -v # 19 test (incl. verifica gradienti)
```

Il checkpoint `checkpoints/mappo_best.json` contiene i pesi in JSON puro:
l'Actor si carica senza PyTorch (`MARLController.from_checkpoint(...)`) ed è
pronto per il deploy nel container router dell'emulatore ContainerLab
(repo `eds-containerlab`), come previsto dalla Tabella 10 del documento.
