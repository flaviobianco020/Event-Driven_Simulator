"""
actions.py — Lo spazio di azione VINCOLATO del supervisore LLM (Fase 4).

Il supervisore non decide per-pacchetto: sceglie una di 5 azioni supervisorie
su un tick lento (~30 s). Lo spazio e' minuscolo e chiuso apposta — con il
constrained decoding (schema JSON) anche un modello piccolo (1.5-3B) emette
sempre un'azione valida; puo' solo sbagliare QUALE, non il formato.

Le azioni mappano la struttura CTDE della Fase 3 su chiamate a strumento:
    attore MAPPO           -> agente LLM locale
    azione discreta {0,1,2}-> chiamata a strumento (qui sotto)
    ricompensa scalare     -> feedback metrico + giustificazione in linguaggio naturale
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    """Le 5 azioni supervisorie (vedi diagramma Fase 4)."""
    ENDORSE = "endorse"              # MAPPO ok, nessun intervento (caso comune)
    OVERRIDE_STATE = "override_state"  # forza uno stato di compressione per una finestra
    FLAG_RETRAIN = "flag_retrain"    # segnala regime fuori distribuzione (retrain offline)
    EXPLAIN = "explain"              # sola giustificazione in linguaggio naturale
    COORDINATE = "coordinate"        # (mesh) messaggio a un peer router


# Stati di compressione ammessi per override_state (specchio della macchina a
# stati della congestione: 0=NORMAL .. 4=DROP_LOW_PRIORITY).
VALID_STATES = (0, 1, 2, 3, 4)

# Durata massima (secondi) di un override: reversibile e limitato nel tempo.
MAX_OVERRIDE_SECONDS = 120.0


@dataclass
class Decision:
    """L'output strutturato del supervisore per un tick."""
    action: Action
    justification: str                       # spiegazione in linguaggio naturale (sempre)
    target_state: int | None = None          # per OVERRIDE_STATE
    hold_seconds: float | None = None         # per OVERRIDE_STATE
    peer: str | None = None                  # per COORDINATE
    message: str | None = None               # per COORDINATE
    raw: dict = field(default_factory=dict)  # payload grezzo del modello (diagnostica)

    def is_control(self) -> bool:
        """True se l'azione ha effetto sul percorso veloce (solo OVERRIDE_STATE)."""
        return self.action == Action.OVERRIDE_STATE


# Schema JSON per il constrained decoding (Ollama `format`, o grammatica GBNF).
# Il modello DEVE produrre un oggetto conforme: questo garantisce output valido
# anche con modelli piccoli. La logica di validazione semantica sta nel guardrail.
DECISION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [a.value for a in Action]},
        "justification": {"type": "string"},
        "target_state": {"type": "integer", "enum": list(VALID_STATES)},
        "hold_seconds": {"type": "number"},
        "peer": {"type": "string"},
        "message": {"type": "string"},
    },
    "required": ["action", "justification"],
    "additionalProperties": False,
}


def decision_from_dict(d: dict) -> Decision:
    """Costruisce una Decision dal dizionario del modello (post constrained-decoding)."""
    return Decision(
        action=Action(d["action"]),
        justification=str(d.get("justification", "")),
        target_state=d.get("target_state"),
        hold_seconds=d.get("hold_seconds"),
        peer=d.get("peer"),
        message=d.get("message"),
        raw=d,
    )
