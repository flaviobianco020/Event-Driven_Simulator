"""
env.py — EDSMarlEnv: il simulatore EDS come Dec-POMDP steppabile (doc §1, §7).

Ad ogni step (delta_t = 1 s simulato, doc §6) l'ambiente:
  1. applica le azioni {ESCALATE, MAINTAIN, DE-ESCALATE} degli agenti
     chiamando node.state_machine.transition(new_state) (doc §7.1, Tabella 8);
  2. fa avanzare il simulatore di 1 s (EventScheduler.run_until);
  3. costruisce le osservazioni locali o_i (dim=7, doc Tabella 7) e lo stato
     globale s (dim=7N+4, doc Tabella 6) per il Critic centralizzato;
  4. calcola il reward condiviso dalla finestra di metriche (doc §6):
         r_t = PDR_t − 0.3·drop_rate_t − 0.2·(lat_t / 2 s) + 0.2·J_t

La macchina a stati dei nodi-agente e' sostituita da
AgentControlledStateMachine: mantiene l'EWMA dell'occupancy (feature o_i[0],
"gia' disponibile in Phase 2", doc Tabella 7) ma NON transisce mai da sola —
le transizioni sono decise esclusivamente dalla policy MAPPO.

Gli scenari 1-6 replicano ESATTAMENTE examples/scenarios.py (stessi flussi,
rate, classi, eventi e seed di default); in training l'end_time e' forzato
a 100 s (doc Tabella 10, riga Training).
"""
from __future__ import annotations

import numpy as np

from ..config import ConfigurationManager
from ..core import Simulator
from ..event import Event, EventType
from ..metrics import MetricsEngine
from ..network.congestion import (
    PHASE2_EWMA_ALPHA,
    CongestionState,
    CongestionStateMachine,
)
from ..network.topology import NetworkTopology
from ..traffic.flow import Flow, FlowModel, TrafficClass
from ..traffic.generator import TrafficGenerator

# ── costanti (doc §5.3, §6) ───────────────────────────────────────────────────
OBS_DIM = 7
N_ACTIONS = 3
ESCALATE, MAINTAIN, DEESCALATE = 0, 1, 2       # doc Tabella 8

DT = 1.0             # passo di osservazione: 1 s simulato (doc §6)
T_MAX_STATE = 30.0   # normalizzazione feature t_stato/T_max (doc Tabella 7)
LAT_MAX = 2.0        # soglia latenza per traffico di controllo (doc Tabella 9)
LAMBDA_DROP = 0.3    # peso drop rate (doc Tabella 9)
LAMBDA_LAT = 0.2     # peso latenza (doc Tabella 9)
LAMBDA_FAIR = 0.2    # peso Jain (doc Tabella 9)


class AgentControlledStateMachine(CongestionStateMachine):
    """
    CSM passiva: update() aggiorna solo l'EWMA (alpha = 0.125, Phase 2) senza
    mai transire. Le transizioni avvengono solo via transition(), invocata
    dall'ambiente quando l'agente MAPPO sceglie ESCALATE/DE-ESCALATE.
    """

    def __init__(self) -> None:
        super().__init__(ewma_alpha=PHASE2_EWMA_ALPHA)
        self.last_transition_time: float = 0.0

    def update(self, occupancy: float, sim_time: float = 0.0) -> bool:
        self._ewma = (1.0 - self._alpha) * self._ewma + self._alpha * occupancy
        return False

    def transition(self, new_state: CongestionState, sim_time: float = 0.0) -> bool:
        changed = super().transition(new_state, sim_time)
        if changed:
            self.last_transition_time = sim_time
        return changed


# ── fabbrica scenari (specchio di examples/scenarios.py) ──────────────────────

def _video_class() -> TrafficClass:
    return TrafficClass((1400, 1500), priority_level=2,
                        latency_sensitivity=True, compression_sensitivity=True)


def _telemetry_class() -> TrafficClass:
    return TrafficClass((200, 300), priority_level=1,
                        latency_sensitivity=False, compression_sensitivity=True)


def _control_class() -> TrafficClass:
    return TrafficClass((100, 100), priority_level=0,
                        latency_sensitivity=True, compression_sensitivity=False)


def _build_scenario(scenario: int, seed: int):
    """
    Restituisce (topology, generator, extra_events, end_time_canonico).
    Flussi, rate, classi ed eventi identici a examples/scenarios.py.
    """
    if scenario == 1:
        topo = NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0, queue_size=20)
        gen = (TrafficGenerator()
               .add_flow(Flow(FlowModel.POISSON, _video_class(),   topo.get_node("src0"), topo.get_node("dst"), rate=8.0))
               .add_flow(Flow(FlowModel.CONTROL, _control_class(), topo.get_node("src1"), topo.get_node("dst"), rate=5.0)))
        return topo, gen, [], 60.0

    if scenario == 2:
        topo = NetworkTopology.single_bottleneck(n_sources=3, bottleneck_capacity=10.0, queue_size=20)
        dst = topo.get_node("dst")
        surge = Flow(FlowModel.BURSTY, _video_class(), topo.get_node("src1"), dst, rate=6.0)
        gen = (TrafficGenerator()
               .add_flow(Flow(FlowModel.POISSON, _video_class(),   topo.get_node("src0"), dst, rate=4.0))
               .add_flow(Flow(FlowModel.CONTROL, _control_class(), topo.get_node("src2"), dst, rate=2.0)))
        # identico a examples/scenarios.py: FLOW_START extra a t=20 e FLOW_STOP
        # a t=50; il surge, registrato nel generator, riceve anche il
        # FLOW_START a t=0 esattamente come nello scenario canonico
        events = [
            Event(simulation_time=20.0, type=EventType.FLOW_START, metadata={"flow": surge}),
            Event(simulation_time=50.0, type=EventType.FLOW_STOP,  metadata={"flow": surge}),
        ]
        gen.add_flow(surge)
        return topo, gen, events, 80.0

    if scenario == 3:
        topo = NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0, queue_size=20)
        gen = (TrafficGenerator()
               .add_flow(Flow(FlowModel.POISSON, _video_class(),   topo.get_node("src0"), topo.get_node("dst"), rate=7.0))
               .add_flow(Flow(FlowModel.CONTROL, _control_class(), topo.get_node("src1"), topo.get_node("dst"), rate=5.0)))
        bott = topo.get_link("router", "dst")
        events = [
            Event(simulation_time=30.0, type=EventType.LINK_RATE_CHANGE, link=bott, metadata={"new_rate": 4.0}),
            Event(simulation_time=60.0, type=EventType.LINK_RATE_CHANGE, link=bott, metadata={"new_rate": 10.0}),
        ]
        return topo, gen, events, 90.0

    if scenario == 4:
        topo = NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0, queue_size=20)
        gen = (TrafficGenerator()
               .add_flow(Flow(FlowModel.POISSON, _video_class(),   topo.get_node("src0"), topo.get_node("dst"), rate=6.0))
               .add_flow(Flow(FlowModel.CONTROL, _control_class(), topo.get_node("src1"), topo.get_node("dst"), rate=3.0)))
        link = topo.get_link("router", "dst")
        events = [
            Event(simulation_time=30.0, type=EventType.LINK_FAILURE,  link=link),
            Event(simulation_time=55.0, type=EventType.LINK_RECOVERY, link=link),
        ]
        return topo, gen, events, 90.0

    if scenario == 5:
        topo = NetworkTopology.single_bottleneck(n_sources=3, bottleneck_capacity=10.0, queue_size=20)
        dst = topo.get_node("dst")
        gen = (TrafficGenerator()
               .add_flow(Flow(FlowModel.POISSON, _video_class(),     topo.get_node("src0"), dst, rate=7.0))
               .add_flow(Flow(FlowModel.POISSON, _telemetry_class(), topo.get_node("src1"), dst, rate=5.0))
               .add_flow(Flow(FlowModel.CONTROL, _control_class(),   topo.get_node("src2"), dst, rate=3.0)))
        return topo, gen, [], 100.0

    if scenario == 6:
        topo = NetworkTopology.single_bottleneck(n_sources=3, bottleneck_capacity=10.0, queue_size=30)
        dst = topo.get_node("dst")
        gen = (TrafficGenerator()
               .add_flow(Flow(FlowModel.VIDEO,              _video_class(),     topo.get_node("src0"), dst, rate=5.0))
               .add_flow(Flow(FlowModel.PERIODIC_TELEMETRY, _telemetry_class(), topo.get_node("src1"), dst, rate=4.0))
               .add_flow(Flow(FlowModel.CONTROL,            _control_class(),   topo.get_node("src2"), dst, rate=2.0)))
        return topo, gen, [], 80.0

    raise ValueError(f"Scenario sconosciuto: {scenario} (validi: 1-6)")


# ─────────────────────────────── ambiente ────────────────────────────────────

class EDSMarlEnv:
    """
    Ambiente Dec-POMDP a passo fisso costruito sopra il Simulator EDS.

    Agenti = nodi router (nel single bottleneck: il solo "router", N=1).
    API in stile gym:  obs, state = reset();  obs, state, r, done, info = step(a).
    """

    def __init__(self, scenario: int, seed: int = 42,
                 end_time: float | None = None,
                 agent_node_ids: list[str] | None = None,
                 stability_penalty: float = 0.0,
                 compression_cost: float = 0.0) -> None:
        self.scenario = scenario
        self.seed = seed
        # penalita' opzionale per ogni cambio di stato (reward shaping):
        # r_t -= stability_penalty * (numero di transizioni nel passo).
        # 0.0 = reward esatto del documento (Tabella 9). Non influenza la
        # policy in deploy: agisce solo sul segnale di training.
        self.stability_penalty = float(stability_penalty)
        # costo opzionale del LIVELLO di compressione (reward shaping):
        # r_t -= compression_cost * (stato_medio / 4).
        # Modella il costo (fedelta'/CPU) della compressione, assente nel
        # reward del documento: senza, comprimere e' gratis e la policy non ha
        # incentivo a tornare a NORMAL quando la congestione sparisce.
        # 0.0 = comportamento invariato.
        self.compression_cost = float(compression_cost)
        topo, gen, extra_events, canonical_end = _build_scenario(scenario, seed)
        self.end_time = float(end_time if end_time is not None else canonical_end)
        self.topology = topo
        self.agent_ids = agent_node_ids or ["router"]
        self.n_agents = len(self.agent_ids)
        self.state_dim = OBS_DIM * self.n_agents + 4

        self.metrics = MetricsEngine()
        self.sim = Simulator(
            ConfigurationManager(random_seed=seed), topo, gen, self.metrics,
            end_time=self.end_time, metric_interval=10.0,
        )
        # sostituisce le CSM dei nodi-agente: il controllo passa a MAPPO
        self._nodes = [topo.get_node(nid) for nid in self.agent_ids]
        for node in self._nodes:
            node.state_machine = AgentControlledStateMachine()

        for ev in extra_events:
            self.sim.scheduler.schedule_event(ev)

        self.t = 0.0
        self._started = False
        # riferimenti finestra per reward/osservazioni
        self._prev = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0,
                      "per_flow": {}, "served": [0] * self.n_agents}

    # ── osservazioni (doc Tabella 7) ─────────────────────────────────────────

    def _window_deltas(self) -> dict:
        m = self.metrics
        cur_pf = m.delivered_per_flow
        d = {
            "gen": m.total_generated - self._prev["gen"],
            "del": m.total_delivered - self._prev["del"],
            "drop": m.total_dropped - self._prev["drop"],
            "lat": m.total_latency - self._prev["lat"],
            "per_flow": {fid: cur_pf.get(fid, 0) - self._prev["per_flow"].get(fid, 0)
                         for fid in cur_pf},
            "served": [n.default_queue._served - s0
                       for n, s0 in zip(self._nodes, self._prev["served"])],
        }
        return d

    def _commit_window(self) -> None:
        m = self.metrics
        self._prev = {
            "gen": m.total_generated, "del": m.total_delivered,
            "drop": m.total_dropped, "lat": m.total_latency,
            "per_flow": m.delivered_per_flow,
            "served": [n.default_queue._served for n in self._nodes],
        }

    def _observe(self, deltas: dict) -> tuple[np.ndarray, np.ndarray]:
        """Costruisce (obs (N,7), stato_globale (7N+4,))."""
        gen_w = max(deltas["gen"], 1)
        drop_rate = min(deltas["drop"] / gen_w, 1.0)

        obs = np.zeros((self.n_agents, OBS_DIM))
        utils = []
        for i, node in enumerate(self._nodes):
            smachine: AgentControlledStateMachine = node.state_machine  # type: ignore[assignment]
            q = node.default_queue
            buf = q.buffer
            n_buf = len(buf)
            hi = sum(1 for p in buf if p.priority == 0) / n_buf if n_buf else 0.0
            lo = sum(1 for p in buf if p.priority == 2) / n_buf if n_buf else 0.0
            link_util = min(deltas["served"][i] / (q.service_rate * DT), 1.0) \
                if q.service_rate > 0 else 0.0
            t_in_state = min((self.t - smachine.last_transition_time) / T_MAX_STATE, 1.0)
            obs[i] = [
                min(max(smachine.ewma_occupancy, 0.0), 1.0),  # ewma_occ
                smachine.current_state.value / 4.0,           # stato / 4
                hi,                                           # hi_pri_ratio (CONTROL)
                lo,                                           # lo_pri_ratio (VIDEO)
                drop_rate,                                    # drop rate finestra
                link_util,                                    # utilizzo link
                t_in_state,                                   # t_stato / T_max
            ]
            utils.append(link_util)

        # stato globale: concat(o_1..o_N) + [link_util, rate_p0, rate_p1, rate_p2]
        cap = self._nodes[0].default_queue.service_rate or 1.0
        rates = {0: 0.0, 1: 0.0, 2: 0.0}
        for flow in self.sim.generator.generate_flows():
            if flow.active:
                rates[flow.traffic_class.priority_level] = \
                    rates.get(flow.traffic_class.priority_level, 0.0) + flow.rate
        extras = np.array([
            float(np.mean(utils)),
            min(rates.get(0, 0.0) / cap, 2.0),
            min(rates.get(1, 0.0) / cap, 2.0),
            min(rates.get(2, 0.0) / cap, 2.0),
        ])
        state = np.concatenate([obs.reshape(-1), extras])
        return obs, state

    # ── reward (doc §6, Tabella 9) ───────────────────────────────────────────

    def _reward(self, deltas: dict) -> float:
        """Reward base del documento (§6). L'eventuale penalita' di stabilita'
        viene sottratta in step(), non qui, perche' dipende dall'azione."""
        gen_w = deltas["gen"]
        pdr = deltas["del"] / gen_w if gen_w > 0 else 1.0
        drop = min(deltas["drop"] / gen_w, 1.0) if gen_w > 0 else 0.0
        lat = (deltas["lat"] / deltas["del"]) if deltas["del"] > 0 else 0.0
        lat_norm = min(lat / LAT_MAX, 1.0)
        counts = [c for c in deltas["per_flow"].values() if c > 0]
        if counts:
            s1, s2 = sum(counts), sum(c * c for c in counts)
            jain = (s1 * s1) / (len(counts) * s2) if s2 > 0 else 1.0
        else:
            jain = 1.0
        return pdr - LAMBDA_DROP * drop - LAMBDA_LAT * lat_norm + LAMBDA_FAIR * jain

    # ── API gym-like ─────────────────────────────────────────────────────────

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        if self._started:
            raise RuntimeError("EDSMarlEnv non e' riutilizzabile: crearne uno nuovo per episodio.")
        self._started = True
        # schedulazione iniziale identica a Simulator.run()
        for flow in self.sim.generator.generate_flows():
            self.sim.scheduler.schedule_event(Event(
                simulation_time=0.0, type=EventType.FLOW_START,
                metadata={"flow": flow}))
        self.sim.scheduler.schedule_event(Event(
            simulation_time=self.sim.metric_interval, type=EventType.METRIC_SAMPLE))
        deltas = self._window_deltas()
        return self._observe(deltas)

    def step(self, actions: np.ndarray | list[int]
             ) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """
        actions: (N,) in {0=ESCALATE, 1=MAINTAIN, 2=DE-ESCALATE} (doc Tabella 8).
        Restituisce (obs, stato_globale, reward, done, info).
        """
        # 1. applica le azioni (doc §7.1: node.state_machine.transition)
        n_transitions = 0
        for node, a in zip(self._nodes, np.asarray(actions, dtype=int)):
            cur = node.state_machine.current_state.value
            if a == ESCALATE:
                new = min(cur + 1, CongestionState.DROP_LOW_PRIORITY.value)
            elif a == DEESCALATE:
                new = max(cur - 1, CongestionState.NORMAL.value)
            else:
                new = cur
            if new != cur:
                changed = node.state_machine.transition(CongestionState(new), self.t)
                if changed:
                    n_transitions += 1
                    self.metrics.record_state_transition()
                    self.sim.scheduler.schedule_event(Event(
                        simulation_time=self.t, type=EventType.STATE_UPDATE,
                        node=node, metadata={"state": node.state_machine.current_state}))

        # 2. avanza il simulatore di delta_t = 1 s
        self.t += DT
        self.sim.scheduler.run_until(self.t)

        # 3-4. osservazioni + reward dalla finestra, meno gli shaping opzionali:
        #   - penalita' di stabilita' (per transizione)
        #   - costo del livello di compressione (stato medio degli agenti / 4)
        deltas = self._window_deltas()
        mean_state = sum(n.state_machine.current_state.value
                         for n in self._nodes) / len(self._nodes)
        reward = (self._reward(deltas)
                  - self.stability_penalty * n_transitions
                  - self.compression_cost * (mean_state / 4.0))
        obs, state = self._observe(deltas)
        self._commit_window()

        done = self.t >= self.end_time - 1e-9
        info = {"t": self.t, "deltas": deltas, "transitions": n_transitions,
                "states": [n.state_machine.current_state.name for n in self._nodes]}
        return obs, state, reward, done, info

    # ── metriche finali per la valutazione (doc Tabella 10) ──────────────────

    def summary(self) -> dict:
        m = self.metrics
        return {
            "pdr": m.collect_packet_delivery_ratio(),
            "latency": m.collect_end_to_end_latency(),
            "fairness": m.collect_fairness(),
            "compression_ratio": m.collect_compression_ratio(),
            "transitions": m.collect_congestion_state_transitions(),
            "generated": m.total_generated,
            "delivered": m.total_delivered,
            "dropped": m.total_dropped,
        }
