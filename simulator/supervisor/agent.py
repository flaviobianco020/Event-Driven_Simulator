"""
agent.py — Fase 4b: l'LLM come AGENTE (non decisore one-shot).

Motivazione (risultato Fase 4a): un decisore one-shot non distingue collasso da
transitorio perche' l'informazione discriminante — se il sistema RECUPERA — e'
nel FUTURO, non nell'input (floor di osservabilita'). Un AGENTE lo batte perche'
puo' AGIRE per procurarsi quell'informazione: aspettare e ri-osservare, prima di
decidere.

Il ciclo dell'agente (percepisci → ragiona → usa tool → osserva → ripeti) gira
sul percorso LENTO. Il percorso veloce resta deterministico (MAPPO). L'agente non
sceglie lo stato di compressione: sceglie fra TOOL vagliati:
  - query_diagnostics()          percezione: stato/traiettoria correnti
  - wait_and_observe(n)          IL tool chiave: avanza n finestre e osserva se
                                 il sistema recupera (rivela il futuro)
  - trigger_reconfigure(class)   azione strategica: attiva la protezione delle
                                 priorita' alte (stato 4) — solo se CRITICO
  - conclude(diagnosis)          chiude: collasso_permanente / transitorio / incerto

Sicurezza (guardrail sui tool): reconfigure ammesso solo se CRITICO; numero di
passi limitato; wait limitato. L'LLM sceglie QUALE tool, non emette azioni grezze.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .controller import SupervisorController
from ..network.congestion import CongestionState
from ..marl import ESCALATE, MAINTAIN, DEESCALATE

MAX_AGENT_STEPS = 6          # anti-loop: passi di ragionamento per episodio
MAX_WAIT_WINDOWS = 4         # cap su wait_and_observe

AGENT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "tool": {"type": "string",
                 "enum": ["query_diagnostics", "wait_and_observe",
                          "trigger_reconfigure", "conclude"]},
        "n_windows": {"type": "integer"},
        "diagnosis": {"type": "string",
                      "enum": ["collasso_permanente", "transitorio", "incerto"]},
    },
    "required": ["reasoning", "tool"],
}

AGENT_SYSTEM_PROMPT = (
    "Sei un agente operatore di rete. Il controllo veloce e' automatico; tu intervieni "
    "sul percorso lento quando qualcosa non va. Il tuo compito NON e' scegliere la "
    "compressione, ma capire se un regime critico e' un COLLASSO PERMANENTE (serve "
    "intervento: proteggere le priorita' alte) o un TRANSITORIO (recupera da solo: non "
    "intervenire). Attenzione: all'inizio i due casi sembrano IDENTICI. NON tirare a "
    "indovinare: INDAGA. Usa wait_and_observe per vedere se il sistema recupera, poi "
    "decidi. Strumenti: query_diagnostics (stato attuale), wait_and_observe(n) (aspetta "
    "e ri-osserva), trigger_reconfigure (proteggi le priorita' alte, SOLO se resta "
    "critico), conclude(diagnosis). Ragiona un passo alla volta."
)


def _to_action(cur, tgt):
    return ESCALATE if cur < tgt else (DEESCALATE if cur > tgt else MAINTAIN)


@dataclass
class AgentSession:
    """Guida la simulazione: il percorso veloce avanza a finestre, i tool dell'agente
    osservano e agiscono su di esso."""
    env: object
    mappo: object
    window_s: float = 30.0
    protect_target: int = 4          # stato che protegge le priorita' alte
    active_target: int | None = None  # se attivo, forza lo stato (reconfigure)
    traj: list = field(default_factory=list)
    done: bool = False
    _obs: object = None
    reconfigured: bool = False
    reconfigure_blocked: int = 0
    nominal_capacity: float = 0.0

    def reset(self):
        self._obs, _ = self.env.reset()
        # capacita' nominale del collo di bottiglia (prima di qualunque guasto)
        self.nominal_capacity = self.env.topology.get_link("router", "dst").capacity

    def query_link_capacity(self) -> dict:
        """
        Sensore della CAUSA (non del sintomo): capacita' corrente del collo di
        bottiglia. Capacita' BASSA → calo strutturale (collasso); capacita' NORMALE
        con sistema critico → il problema e' la domanda (transitorio). Distingue i
        due modi di guasto SENZA aspettare — abbatte il confine temporale per la
        coppia collasso-capacita' vs picco-di-domanda.
        """
        cap = self.env.topology.get_link("router", "dst").capacity
        return {"capacity": cap, "nominal": self.nominal_capacity,
                "capacity_dropped": cap < self.nominal_capacity - 1e-9}

    def _advance_one_window(self) -> dict:
        """Avanza una finestra del percorso veloce; ritorna le metriche di finestra."""
        from examples.run_m1_explainer import _window_metrics  # riuso locale
        acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}
        t0 = self.env.t
        while not self.done and (self.env.t - t0) < self.window_s - 1e-9:
            cur = self.env._nodes[0].state_machine.current_state.value
            if self.active_target is not None:
                actions = [_to_action(cur, self.active_target)]
            else:
                actions = self.mappo.act(self._obs)
            self._obs, _s, _r, self.done, info = self.env.step(actions)
            d = info["deltas"]
            acc["gen"] += d["gen"]; acc["del"] += d["del"]
            acc["drop"] += d["drop"]; acc["lat"] += d["lat"]
            acc["trans"] += info["transitions"]
            self.traj.append(CongestionState[info["states"][0]].value)
        return _window_metrics(acc, self.window_s, self.env.metrics.collect_compression_ratio())

    def _health(self, metrics: dict) -> str:
        return SupervisorController.assess(metrics, self.traj)["health"]

    # ── TOOL ─────────────────────────────────────────────────────────────────────
    def query_diagnostics(self) -> dict:
        m = self._advance_one_window()      # una finestra per "misurare" lo stato attuale
        return {"health": self._health(m), "compressione_massima": self._maxed(),
                "traiettoria": self.traj[-8:]}

    def wait_and_observe(self, n: int) -> dict:
        n = max(1, min(int(n), MAX_WAIT_WINDOWS))
        m = {}
        for _ in range(n):
            if self.done:
                break
            m = self._advance_one_window()
        return {"finestre_attese": n, "health_dopo_attesa": self._health(m) if m else "fine",
                "compressione_massima": self._maxed(), "traiettoria": self.traj[-8:]}

    def trigger_reconfigure(self, current_health: str) -> dict:
        # guardrail: reconfigure ammesso SOLO se il sistema e' ancora critico
        if current_health != "CRITICO":
            self.reconfigure_blocked += 1
            return {"applicato": False, "motivo": "guardrail: sistema non critico, "
                    "riconfigurazione rifiutata"}
        self.active_target = self.protect_target
        self.reconfigured = True
        return {"applicato": True, "azione": f"protezione priorita' alte (stato "
                f"{self.protect_target}) attiva"}

    def _maxed(self) -> bool:
        tail = self.traj[-8:]
        return bool(tail) and sum(s >= 3 for s in tail) >= len(tail) * 0.5

    def finish(self):
        """Esaurito il ragionamento, porta l'episodio a fine."""
        while not self.done:
            self._advance_one_window()


def run_agent_episode(env, mappo, backend, window_s: float = 30.0,
                      verbose: bool = False) -> dict:
    """
    Guida un episodio con l'agente. Ritorna KPI finali + diagnosi + storico tool.
    backend.decide(context, system, user, schema) sceglie il tool (constrained).
    """
    from examples.run_m3_ood import _kpis
    sess = AgentSession(env=env, mappo=mappo, window_s=window_s)
    sess.reset()

    # avanza fino al primo segnale di guaio (prima finestra critica)
    last = {"health": "SANO"}
    while not sess.done:
        m = sess._advance_one_window()
        last = {"health": sess._health(m), "compressione_massima": sess._maxed(),
                "traiettoria": sess.traj[-8:]}
        if last["health"] == "CRITICO":
            break

    history = []
    diagnosis = "incerto"
    self_concluded = False
    obs = last
    for step in range(MAX_AGENT_STEPS):
        if sess.done:
            break
        user = _build_agent_prompt(obs, history)
        try:
            call = backend.decide({"obs": obs}, AGENT_SYSTEM_PROMPT, user,
                                  schema=AGENT_TOOL_SCHEMA)
        except Exception as exc:  # noqa: BLE001 — fail-safe: chiudi senza agire
            call = {"tool": "conclude", "diagnosis": "incerto",
                    "reasoning": f"backend errore ({exc})"}
        tool = call.get("tool", "conclude")

        if tool == "query_diagnostics":
            res = sess.query_diagnostics(); obs = res
        elif tool == "wait_and_observe":
            res = sess.wait_and_observe(call.get("n_windows", 2)); obs = res
            obs["health"] = res["health_dopo_attesa"]
        elif tool == "trigger_reconfigure":
            res = sess.trigger_reconfigure(obs.get("health", "?"))
            obs = dict(obs); obs["gia_intervenuto"] = True   # segnala: hai agito → concludi
        else:  # conclude
            diagnosis = call.get("diagnosis", "incerto")
            self_concluded = True
            history.append({"tool": tool, "diagnosis": diagnosis,
                            "reasoning": call.get("reasoning", "")})
            break

        history.append({"tool": tool, "reasoning": call.get("reasoning", ""), "result": res})
        if verbose:
            print(f"    [step {step}] {tool}({call.get('n_windows','') if tool=='wait_and_observe' else ''}) "
                  f"→ {res}")

    # scaffolding del control-flow: se l'agente non ha concluso (loop), inferisci la
    # diagnosi dallo stato — la conoscenza c'e' gia' nelle azioni fatte.
    if not self_concluded:
        if sess.reconfigured:
            diagnosis = "collasso_permanente"
        elif obs.get("health") == "SANO":
            diagnosis = "transitorio"

    sess.finish()
    out = _kpis(sess.env)
    out.update({"diagnosis": diagnosis, "self_concluded": self_concluded,
                "reconfigured": sess.reconfigured,
                "reconfigure_blocked": sess.reconfigure_blocked, "history": history})
    return out


def _build_agent_prompt(obs: dict, history: list) -> str:
    already_waited = any(h.get("tool") == "wait_and_observe" for h in history)
    hist = ""
    if history:
        hist = "\nAzioni gia' fatte:\n" + "\n".join(
            f"  - {h['tool']}: {h.get('result', h.get('diagnosis',''))}" for h in history)
    # regole di TERMINAZIONE esplicite (l'SLM tende a fare loop di wait senza concludere)
    if obs.get("gia_intervenuto"):
        step_hint = ("Hai GIA' riconfigurato. Ora DEVI chiamare conclude con "
                     "diagnosis=collasso_permanente. Non aspettare oltre.")
    elif already_waited and obs.get("health") == "SANO":
        step_hint = ("Hai gia' atteso ed e' tornato SANO → e' un TRANSITORIO. DEVI "
                     "chiamare conclude con diagnosis=transitorio. Non intervenire.")
    elif already_waited and obs.get("health") == "CRITICO":
        step_hint = ("Hai gia' atteso e resta CRITICO → e' un COLLASSO. DEVI chiamare "
                     "trigger_reconfigure (una sola volta). Non aspettare ancora.")
    else:
        step_hint = ("Non hai ancora indagato: USA wait_and_observe per vedere se recupera.")
    return (
        f"SITUAZIONE ATTUALE (simbolica):\n"
        f"  - salute: {obs.get('health','?')}\n"
        f"  - compressione gia' massima: {obs.get('compressione_massima','?')}\n"
        f"  - traiettoria stati recenti: {obs.get('traiettoria', [])}\n"
        f"{hist}\n"
        f"COSA FARE ORA: {step_hint}\n"
        "Scegli UN solo tool. Concludi appena hai la risposta: non ripetere wait inutilmente."
    )
