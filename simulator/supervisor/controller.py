"""
controller.py — Il SupervisorController (Fase 4): il ragionatore sul percorso lento.

DISACCOPPIATO dal motore: non importa ne' modifica core.py. Riceve una finestra
di metriche + la traiettoria di stati del MAPPO, costruisce il prompt, interroga
il backend LLM (constrained decoding), fa validare la decisione dal guardrail, e
ritorna lo stato eventualmente imposto al percorso veloce (o None).

Il driver (esempio: examples/run_supervisor.py, o l'hook _controller_tick
dell'emulatore) e' responsabile di:
  * chiamare tick() su cadenza lenta (~30 s) o su anomalia;
  * applicare `verdict.effective_state` al compressore, se presente;
  * loggare `decision.justification` (deliverable spiegabilita').

Questo isolamento e' voluto: lo scheletro Fase 4 e' additivo e scartabile senza
toccare le Fasi 1-3.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .actions import Action, Decision, decision_from_dict
from .backend import LLMBackend, MockBackend
from .guardrail import Guardrail, GuardrailVerdict


SYSTEM_PROMPT = (
    "Sei il supervisore di un controllore di congestione di rete. Un agente veloce "
    "(policy appresa) decide ogni secondo quanto comprimere il traffico. Tu intervieni "
    "solo su cadenza lenta e solo quando serve. Ricevi metriche aggregate (mai il "
    "contenuto dei pacchetti). Scegli UNA azione fra: endorse (tutto ok), override_state "
    "(forza uno stato di compressione 0-4 per una finestra breve, quando l'agente e' "
    "chiaramente in difficolta' o fuori dalla sua distribuzione di addestramento), "
    "flag_retrain (regime anomalo, segnala per ri-addestramento), explain (solo "
    "spiegazione), coordinate (messaggio a un peer). Stati: 0=nessuna compressione, "
    "1=header, 2=delta, 3=incrementale (max senza scarti), 4=scarta bassa priorita'. "
    "Preferisci endorse: intervieni solo con motivo. Giustifica SEMPRE in una frase."
)


@dataclass
class SupervisorLog:
    """Traccia delle decisioni per l'analisi di spiegabilita'."""
    entries: list = field(default_factory=list)

    def add(self, t: float, decision: Decision, verdict: GuardrailVerdict) -> None:
        self.entries.append({
            "t": t, "action": decision.action.value,
            "justification": decision.justification,
            "approved": verdict.approved, "reason": verdict.reason,
            "effective_state": verdict.effective_state,
        })


class SupervisorController:
    def __init__(self, backend: LLMBackend | None = None,
                 guardrail: Guardrail | None = None,
                 tick_interval: float = 30.0):
        self.backend = backend or MockBackend()
        self.guardrail = guardrail or Guardrail()
        self.tick_interval = tick_interval
        self.log = SupervisorLog()

    @staticmethod
    def _flag(value: float, good: float, bad: float, higher_better: bool) -> str:
        """Etichetta qualitativa pre-calcolata, cosi' il modello non deve dedurre la direzione."""
        if higher_better:
            return "BUONO" if value >= good else ("CRITICO" if value < bad else "DEGRADATO")
        return "BUONO" if value <= good else ("CRITICO" if value > bad else "ELEVATO")

    def _build_user_prompt(self, metrics: dict, state_traj: list) -> str:
        recent = state_traj[-10:] if state_traj else []
        pdr = metrics.get("pdr", 0.0)
        lat = metrics.get("latency_ms", 0.0)
        drop = metrics.get("drop_rate", 0.0)
        util = metrics.get("link_util", 0.0)
        trans = metrics.get("transitions", 0)
        compr = metrics.get("compression", 1.0)
        # Direzione e flag pre-calcolati: il 3B narra, non deduce (fix scivolone PDR).
        return (
            "Legenda direzione: PDR e compressione = piu' ALTO e' meglio; "
            "latenza, drop_rate, utilizzo_link, transizioni = piu' BASSO e' meglio.\n"
            "Finestra metriche correnti (con valutazione gia' calcolata):\n"
            f"  PDR={pdr:.3f}          [{self._flag(pdr, 0.90, 0.70, True)}]   (consegna; <0.90 = degrado)\n"
            f"  latenza={lat:.0f}ms      [{self._flag(lat, 300, 800, False)}]   (<300 buona, >800 critica)\n"
            f"  drop_rate={drop:.3f}     [{self._flag(drop, 0.05, 0.15, False)}]   (scarti; >0.15 alto)\n"
            f"  utilizzo_link={util:.2f}   [{self._flag(util, 0.85, 0.97, False)}]\n"
            f"  transizioni_finestra={trans}   compressione={compr:.2f}x\n"
            f"Stati recenti dell'agente veloce (0=nessuna..4=scarto, ultimi 10 tick): {recent}\n"
            "Decidi l'azione. Nella giustificazione cita i valori con la loro valutazione "
            "(BUONO/DEGRADATO/CRITICO), senza invertirne la direzione."
        )

    def tick(self, t: float, metrics: dict, state_traj: list) -> GuardrailVerdict:
        """
        Un passo del supervisore. Ritorna il verdetto del guardrail; il chiamante
        applica verdict.effective_state al percorso veloce se presente.
        """
        context = {"metrics": metrics, "state_trajectory": state_traj}
        user_prompt = self._build_user_prompt(metrics, state_traj)
        try:
            raw = self.backend.decide(context, SYSTEM_PROMPT, user_prompt)
            decision = decision_from_dict(raw)
        except Exception as exc:  # noqa: BLE001 — fail-safe: mai bloccare il percorso veloce
            decision = Decision(action=Action.ENDORSE,
                                justification=f"backend errore ({exc}); fallback endorse")
        verdict = self.guardrail.review(decision, metrics, t)
        self.log.add(t, decision, verdict)
        return verdict

    def current_override(self, t: float) -> int | None:
        """Stato attualmente imposto (o None). Il driver lo passa al compressore."""
        return self.guardrail.active_state(t)
