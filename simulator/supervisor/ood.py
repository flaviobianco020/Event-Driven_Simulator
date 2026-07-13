"""
ood.py — Scenari FUORI DISTRIBUZIONE per la valutazione M3 (Fase 4).

Il criterio (piano Fase 4 §7): lo scenario deve essere genuinamente fuori dal
training della policy MAPPO (i 6 scenari canonici) e un supervisore che ragiona
deve poter plausibilmente aiutare. Due assi di novita':

  video_flood — "mix di traffico inedito" (primario, scelto per la tesi):
      3 flussi TUTTI VIDEO, zero CONTROL e zero TELEMETRY, carico 17 pkt/s.
      Perche' e' OOD: (a) in training il traffico CONTROL e' SEMPRE presente →
      le feature osservate ratio_high_pri/ratio_low_pri restano inchiodate a
      0/1, un input mai visto; (b) il VIDEO comprime peggio negli stati alti
      (fallback a delta 0.667: niente diff semantico sul binario) → l'efficacia
      dell'azione appresa cambia; (c) carico 17 > massimo addestrato (15).

  pulsed — carico a onda quadra (secondario, asse temporale):
      flusso base 4 pkt/s + surge VIDEO 16 pkt/s acceso/spento ogni 10 s.
      In training il carico e' stazionario o cambia una volta (flash crowd,
      degrado); un'alternanza rapida e periodica non e' mai stata vista.

Implementazione ADDITIVA: OODMarlEnv e' una sottoclasse di EDSMarlEnv che
bypassa _build_scenario accettando (topologia, generatore, eventi, durata)
espliciti. Nessuna modifica a simulator/marl/.
"""
from __future__ import annotations

from .. import (
    ConfigurationManager, Event, EventType, Flow, FlowModel,
    MetricsEngine, NetworkTopology, Simulator, TrafficClass, TrafficGenerator,
)
from ..marl.env import OBS_DIM, AgentControlledStateMachine, EDSMarlEnv


# stesse classi di traffico degli scenari canonici (marl/env.py)
def _video_class() -> TrafficClass:
    return TrafficClass((1400, 1500), priority_level=2,
                        latency_sensitivity=True, compression_sensitivity=True)


def _telemetry_class() -> TrafficClass:
    return TrafficClass((200, 300), priority_level=1,
                        latency_sensitivity=False, compression_sensitivity=True)


def _control_class() -> TrafficClass:
    return TrafficClass((100, 100), priority_level=0,
                        latency_sensitivity=True, compression_sensitivity=False)


class OODMarlEnv(EDSMarlEnv):
    """
    EDSMarlEnv con scenario esplicito (non dal catalogo 1-6). Replica il
    costruttore del padre saltando _build_scenario: stessa API reset/step/summary,
    cosi' i runner M2/M3 funzionano senza modifiche.
    """

    def __init__(self, topo, gen, extra_events, end_time: float,
                 seed: int = 42, name: str = "ood",
                 agent_node_ids: list[str] | None = None) -> None:
        # NB: replica di EDSMarlEnv.__init__ senza la chiamata a _build_scenario.
        self.scenario = name
        self.seed = seed
        self.stability_penalty = 0.0
        self.end_time = float(end_time)
        self.topology = topo
        self.agent_ids = agent_node_ids or ["router"]
        self.n_agents = len(self.agent_ids)
        self.state_dim = OBS_DIM * self.n_agents + 4

        self.metrics = MetricsEngine()
        self.sim = Simulator(
            ConfigurationManager(random_seed=seed), topo, gen, self.metrics,
            end_time=self.end_time, metric_interval=10.0,
        )
        self._nodes = [topo.get_node(nid) for nid in self.agent_ids]
        for node in self._nodes:
            node.state_machine = AgentControlledStateMachine()
        for ev in extra_events:
            self.sim.scheduler.schedule_event(ev)

        self.t = 0.0
        self._started = False
        self._prev = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0,
                      "per_flow": {}, "served": [0] * self.n_agents}


def build_video_flood(seed: int = 42, end_time: float | None = None) -> OODMarlEnv:
    """Mix inedito: solo VIDEO, carico 17 pkt/s, nessuna classe protetta in coda."""
    topo = NetworkTopology.single_bottleneck(n_sources=3, bottleneck_capacity=10.0,
                                             queue_size=20)
    dst = topo.get_node("dst")
    gen = (TrafficGenerator()
           .add_flow(Flow(FlowModel.POISSON, _video_class(), topo.get_node("src0"), dst, rate=6.0))
           .add_flow(Flow(FlowModel.VIDEO,   _video_class(), topo.get_node("src1"), dst, rate=6.0))
           .add_flow(Flow(FlowModel.BURSTY,  _video_class(), topo.get_node("src2"), dst, rate=5.0)))
    return OODMarlEnv(topo, gen, [], end_time or 100.0, seed=seed, name="video_flood")


def build_pulsed(seed: int = 42, end_time: float | None = None) -> OODMarlEnv:
    """Onda quadra: base 4 pkt/s + surge VIDEO 16 pkt/s on/off ogni 10 s."""
    end = end_time or 100.0
    topo = NetworkTopology.single_bottleneck(n_sources=2, bottleneck_capacity=10.0,
                                             queue_size=20)
    dst = topo.get_node("dst")
    base = Flow(FlowModel.POISSON, _video_class(), topo.get_node("src0"), dst, rate=4.0)
    surge = Flow(FlowModel.VIDEO, _video_class(), topo.get_node("src1"), dst, rate=16.0)
    gen = TrafficGenerator().add_flow(base).add_flow(surge)

    # surge attivo negli intervalli [0,10), [20,30), [40,50) ... → STOP/START alternati
    events = []
    t, on = 10.0, False
    while t < end:
        events.append(Event(simulation_time=t,
                            type=EventType.FLOW_START if on else EventType.FLOW_STOP,
                            metadata={"flow": surge}))
        on = not on
        t += 10.0
    return OODMarlEnv(topo, gen, events, end, seed=seed, name="pulsed")


def build_capacity_collapse(seed: int = 42, end_time: float | None = None) -> OODMarlEnv:
    """
    Collasso di capacita': mix a 3 classi (carico 15, come lo scenario 5) ma il
    collo di bottiglia crolla a 2 pkt/s a t=20 IN MODO PERMANENTE.

    Perche' e' OOD e perche' MAPPO deve fallire qui: in training il minimo visto
    e' 4 pkt/s, transitorio (scenario 3). A cap=2 anche la compressione massima
    "civile" (stato 3, ~2x) lascia il sistema sovraccarico ~3.7x: la coda satura
    e i drop passivi colpiscono INDISCRIMINATAMENTE, incluso il traffico CONTROL
    (priorita' 0). L'unico rimedio strutturale e' lo stato 4 (scarto attivo delle
    priorita' basse PRIMA della coda), che la strategia appresa park-at-3 non
    adotta stabilmente. Metrica chiave: consegna del flusso CONTROL
    (env.control_flow_id + env.control_expected).
    """
    return _capacity_scenario(seed, end_time or 100.0, drop_to=2.0, onset=20.0,
                              recover_at=None, name="capacity_collapse")


def _capacity_scenario(seed, end, drop_to, onset, recover_at, name):
    """Costruttore condiviso: mix a 3 classi (con CONTROL) + eventi di link.
    recover_at=None → collasso PERMANENTE; altrimenti il link torna a 10 a quell'istante
    (degrado TRANSITORIO di durata recover_at-onset)."""
    topo = NetworkTopology.single_bottleneck(n_sources=3, bottleneck_capacity=10.0,
                                             queue_size=20)
    dst = topo.get_node("dst")
    ctrl_flow = Flow(FlowModel.CONTROL, _control_class(), topo.get_node("src2"), dst, rate=3.0)
    gen = (TrafficGenerator()
           .add_flow(Flow(FlowModel.POISSON, _video_class(),     topo.get_node("src0"), dst, rate=7.0))
           .add_flow(Flow(FlowModel.POISSON, _telemetry_class(), topo.get_node("src1"), dst, rate=5.0))
           .add_flow(ctrl_flow))
    bottleneck = topo.get_link("router", "dst")
    events = [Event(simulation_time=onset, type=EventType.LINK_RATE_CHANGE,
                    link=bottleneck, metadata={"new_rate": drop_to})]
    if recover_at is not None:
        events.append(Event(simulation_time=recover_at, type=EventType.LINK_RATE_CHANGE,
                            link=bottleneck, metadata={"new_rate": 10.0}))
    env = OODMarlEnv(topo, gen, events, end, seed=seed, name=name)
    env.control_flow_id = ctrl_flow.id
    env.control_expected = 3.0 * end
    return env


def build_transient_degradation(seed: int = 42, end_time: float | None = None,
                                drop_to: float = 2.0, onset: float = 30.0,
                                duration: float = 40.0) -> OODMarlEnv:
    """
    Degrado TRANSITORIO controllato: stesso mix del collasso, ma il link crolla a
    `drop_to` all'istante `onset` e RECUPERA (torna a 10) dopo `duration` secondi.
    A t=onset e' indistinguibile dal collasso permanente; la differenza emerge solo
    aspettando. Serve a trovare il CONFINE dell'agente: se `duration` supera la
    finestra d'attesa dell'agente (~60 s), l'agente non vede il recupero e lo
    scambia per collasso → il floor di osservabilita' riemerge a scala piu' lunga.
    """
    return _capacity_scenario(seed, end_time or 200.0, drop_to=drop_to, onset=onset,
                              recover_at=onset + duration, name="transient_degradation")


def build_demand_spike(seed: int = 42, end_time: float | None = None,
                       onset: float = 30.0, duration: float = 40.0,
                       surge_rate: float = 16.0) -> OODMarlEnv:
    """
    Picco di DOMANDA (non calo di capacita'): il link resta a 10 pkt/s per tutto
    l'episodio, ma un surge VIDEO si accende in [onset, onset+duration] e spinge il
    carico oltre la capacita' → congestione → sintomo CRITICO identico al collasso.
    La differenza rispetto al collasso e' nella CAUSA, osservabile SUBITO: qui la
    capacita' del link e' NORMALE (carico alto), nel collasso e' BASSA. Un agente
    che interroga la capacita' (query_link_capacity) li distingue senza aspettare —
    la durata del surge diventa irrilevante (confine abbattuto per questa coppia).
    """
    end = end_time or 200.0
    topo = NetworkTopology.single_bottleneck(n_sources=4, bottleneck_capacity=10.0,
                                             queue_size=20)
    dst = topo.get_node("dst")
    ctrl_flow = Flow(FlowModel.CONTROL, _control_class(), topo.get_node("src2"), dst, rate=3.0)
    surge = Flow(FlowModel.VIDEO, _video_class(), topo.get_node("src3"), dst, rate=surge_rate)
    # NB: il surge NON e' nel generatore (quindi non parte a t=0); lo accendiamo con
    # FLOW_START a onset. Evita il bug per cui FLOW_START non riattiva un flusso gia'
    # fermato (core.py::_on_flow_start non rimette flow.active=True).
    gen = (TrafficGenerator()
           .add_flow(Flow(FlowModel.POISSON, _video_class(),     topo.get_node("src0"), dst, rate=7.0))
           .add_flow(Flow(FlowModel.POISSON, _telemetry_class(), topo.get_node("src1"), dst, rate=5.0))
           .add_flow(ctrl_flow))
    # surge ON in [onset, onset+duration], poi OFF. La capacita' del link NON cambia.
    events = [Event(simulation_time=onset, type=EventType.FLOW_START, metadata={"flow": surge}),
              Event(simulation_time=onset + duration, type=EventType.FLOW_STOP, metadata={"flow": surge})]
    env = OODMarlEnv(topo, gen, events, end, seed=seed, name="demand_spike")
    env.control_flow_id = ctrl_flow.id
    env.control_expected = 3.0 * end
    return env


OOD_SCENARIOS = {
    "video_flood": build_video_flood,
    "pulsed": build_pulsed,
    "capacity_collapse": build_capacity_collapse,
}
