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
    def assess(metrics: dict, recent_states: list | None = None) -> dict:
        """
        Valutazione DETERMINISTICA della salute del sistema (nessun LLM).
        La decisione a soglia e' aritmetica banale: la fa un if, non un modello
        da 3B (che non sa confrontare 0.997 con 0.85). L'LLM poi SPIEGA questo
        verdetto — usa la sua forza (linguaggio), non la sua debolezza (calcolo).

        PRINCIPIO "FIRST, DO NO HARM" (esito sperimentale M3): il target e'
        sempre lo stato 3 (compressione massima SENZA scarti attivi), mai il 4.
        Una regola di escalation automatica a 4 e' stata provata e RIMOSSA: nei
        collassi di capacita' non scattava (bloccata dal PDR floor del guardrail)
        e sui degradi transitori forzava scarti attivi devastanti (scenario 3:
        drop +189, PDR -0.175). La policy appresa usa lo stato 4 da sola quando
        serve; il supervisore non deve mai imporlo su base di soglie statiche.

        `recent_states` e' accettato per contesto/prompt ma non altera il target.
        Ritorna {health, recommended_action, target_state, reason}.
        """
        pdr = metrics.get("pdr", 1.0)
        drop = metrics.get("drop_rate", 0.0)
        # salute misurata su PDR e drop; l'utilizzo_link alto e' normale (collo saturo).
        if pdr < 0.85 or drop > 0.15:
            health = "CRITICO" if (pdr < 0.70 or drop > 0.30) else "DEGRADATO"
            return {"health": health, "recommended_action": "override_state",
                    "target_state": 3,
                    "reason": f"PDR {pdr:.3f} (soglia 0.85) / drop {drop:.3f} (soglia 0.15)"}
        return {"health": "SANO", "recommended_action": "endorse", "target_state": None,
                "reason": f"PDR {pdr:.3f} e drop {drop:.3f} entro le soglie"}

    def _build_user_prompt(self, metrics: dict, state_traj: list) -> str:
        recent = state_traj[-10:] if state_traj else []
        a = self.assess(metrics, state_traj)
        pdr = metrics.get("pdr", 0.0)
        lat = metrics.get("latency_ms", 0.0)
        drop = metrics.get("drop_rate", 0.0)
        util = metrics.get("link_util", 0.0)
        trans = metrics.get("transitions", 0)
        compr = metrics.get("compression", 1.0)
        # Il verdetto e' GIA' calcolato: il modello NON decide ne' confronta numeri,
        # deve solo SPIEGARLO all'operatore e notare eventuali anomalie nella traiettoria.
        return (
            f"VALUTAZIONE (gia' calcolata, non ricalcolarla): sistema {a['health']}. "
            f"Azione = {a['recommended_action']}. Motivo: {a['reason']}.\n"
            "Metriche di contesto (PDR piu' alto e' meglio; latenza e drop piu' bassi meglio; "
            "utilizzo_link vicino a 1.0 e' normale, il collo di bottiglia e' saturo per natura):\n"
            f"  PDR={pdr:.3f}   latenza={lat:.0f}ms   drop_rate={drop:.3f}\n"
            f"  utilizzo_link={util:.2f}   transizioni_finestra={trans}   compressione={compr:.2f}x\n"
            f"Stati recenti dell'agente veloce (0=nessuna..4=scarto, ultimi 10 tick): {recent}\n"
            "Il tuo compito: conferma l'azione della valutazione e SPIEGALA in una frase "
            "all'operatore, descrivendo la situazione e cosa fa la traiettoria di stati. "
            "Non invertire la direzione delle metriche. Segnala se la traiettoria mostra "
            "un'anomalia (es. oscillazioni) che la regola a soglia non coglie."
        )

    def tick(self, t: float, metrics: dict, state_traj: list) -> GuardrailVerdict:
        """
        Un passo del supervisore. Ritorna il verdetto del guardrail; il chiamante
        applica verdict.effective_state al percorso veloce se presente.

        SEPARAZIONE decisione/spiegazione:
          * l'AZIONE (endorse/override + stato target) e' DETERMINISTICA (assess),
            non affidata all'aritmetica di un modello piccolo;
          * l'LLM fornisce la SPIEGAZIONE in linguaggio naturale, e puo' elevare a
            flag_retrain se rileva un'anomalia oltre la regola a soglia.
        """
        a = self.assess(metrics, state_traj)
        context = {"metrics": metrics, "state_trajectory": state_traj}
        user_prompt = self._build_user_prompt(metrics, state_traj)
        try:
            raw = self.backend.decide(context, SYSTEM_PROMPT, user_prompt)
            llm = decision_from_dict(raw)
            justification = llm.justification
            # L'unica azione LLM che puo' scavalcare la regola: flag_retrain
            # (segnalazione di anomalia genuina non colta dalla soglia).
            llm_flags_anomaly = (llm.action == Action.FLAG_RETRAIN)
        except Exception as exc:  # noqa: BLE001 — fail-safe: mai bloccare il percorso veloce
            justification = f"backend errore ({exc}); spiegazione non disponibile"
            llm_flags_anomaly = False

        # Decisione di CONTROLLO = 100% deterministica. L'LLM fornisce SOLO la
        # spiegazione (justification). Nota: sia l'azione (override/endorse) sia
        # il target dello stato vengono da assess() — MAI dall'output dell'LLM
        # (vedi simulator/supervisor/DECISION_RATIONALE.md). L'unico segnale che
        # l'LLM puo' alzare e' flag_retrain: monitoraggio/OOD, NON un'azione di
        # controllo (non tocca il percorso veloce).
        if a["recommended_action"] == "override_state":
            decision = Decision(action=Action.OVERRIDE_STATE, justification=justification,
                                target_state=a["target_state"], hold_seconds=30.0)
        elif llm_flags_anomaly:
            decision = Decision(action=Action.FLAG_RETRAIN, justification=justification)
        else:
            decision = Decision(action=Action.ENDORSE, justification=justification)

        verdict = self.guardrail.review(decision, metrics, t)
        self.log.add(t, decision, verdict)
        return verdict

    def current_override(self, t: float) -> int | None:
        """Stato attualmente imposto (o None). Il driver lo passa al compressore."""
        return self.guardrail.active_state(t)
