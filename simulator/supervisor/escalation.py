"""
escalation.py — Fase 4 / System-2 escalation: l'LLM decide SUL CASO AMBIGUO.

Motivazione (risultato M3): la regola a soglia e' cieca dove PDR/drop non
bastano a distinguere due regimi opposti — l'oscillazione transitoria (che si
stabilizza da sola) e il collasso strutturale (dove serve scartare le priorita'
basse per proteggere il traffico di controllo). Stessi due numeri, decisione
opposta. La soglia forza lo stato 3 in entrambi → nel collasso affossa il
controllo (control_del 0.46 vs 1.00 possibile).

Idea: NON chiedere all'LLM di confrontare numeri (sua debolezza, cfr.
DECISION_RATIONALE.md). Chiedergli di:
  1. CLASSIFICARE il regime da una descrizione SIMBOLICA + la traiettoria di
     stati (pattern temporale, non aritmetica) — la sua forza;
  2. SCEGLIERE fra azioni candidate GIA' VAGLIATE dal codice (non emettere uno
     stato grezzo) — il guardrail resta sovrano.

L'escalation scatta SOLO sul caso che la soglia non risolve (critico +
compressione gia' massima + persistente). Il 95% delle decisioni resta
deterministico (System 1). Modello: sopra il floor aritmetico (7B / Haiku).
"""
from __future__ import annotations

# Azioni candidate pre-vagliate (target assoluto di stato + descrizione qualitativa).
# La conoscenza di dominio sta nelle descrizioni; all'LLM resta il match situazione→azione.
CANDIDATES = [
    ("A", None, "Mantieni l'agente veloce, nessun intervento (adatto se e' "
                "un'oscillazione transitoria che si stabilizzera' da sola)."),
    ("B", 3,    "Forza la compressione massima SENZA scarti attivi (stato 3). "
                "Adatto a un degrado gestibile: massimizza la compressione ma "
                "non protegge nessuna classe se la coda satura."),
    ("C", 4,    "Forza lo scarto attivo delle priorita' basse (stato 4). "
                "Sacrifica il throughput a bassa priorita' (es. video) per "
                "PROTEGGERE il traffico ad alta priorita' (controllo) dai drop "
                "indiscriminati di coda piena. Adatto a un collasso strutturale."),
]
_CHOICE_TO_TARGET = {name: tgt for name, tgt, _ in CANDIDATES}

ESCALATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "regime": {"type": "string",
                   "enum": ["oscillazione_transitoria", "collasso_strutturale",
                            "novita_sconosciuta"]},
        "choice": {"type": "string", "enum": ["A", "B", "C"]},
        "justification": {"type": "string"},
    },
    "required": ["regime", "choice", "justification"],
}

ESCALATION_SYSTEM_PROMPT = (
    "Sei il supervisore di secondo livello di un controllo di congestione. "
    "Intervieni SOLO su un caso ambiguo che la regola automatica non sa risolvere: "
    "il sistema e' critico e comprime gia' al massimo da piu' finestre, eppure non "
    "recupera. Devi (1) classificare il regime e (2) scegliere UNA azione candidata. "
    "NON ti servono calcoli: ragiona sul PATTERN della traiettoria di stati e sulla "
    "descrizione qualitativa. Regola di dominio: se la compressione e' gia' massima e "
    "il sistema resta critico a lungo, non e' un'oscillazione — e' un collasso "
    "strutturale, e l'unico rimedio utile e' proteggere le priorita' alte scartando "
    "le basse. Giustifica in una frase."
)


def should_escalate(assessment: dict, recent_states: list, min_compress_ratio: float = 0.5) -> bool:
    """
    Vero SOLO nel caso ambiguo: verdetto CRITICO + compressione gia' massima
    (traiettoria recente prevalentemente >= 3). La persistenza (critico da piu'
    finestre) la impone il chiamante. E' esattamente lo scenario dove PDR/drop
    non distinguono oscillazione da collasso.
    """
    if assessment.get("health") != "CRITICO" or not recent_states:
        return False
    tail = recent_states[-10:]
    return sum(s >= 3 for s in tail) >= len(tail) * min_compress_ratio


def _symbolic_context(recent_states: list, windows_critical: int) -> str:
    """Descrizione QUALITATIVA (niente numeri grezzi: evita il floor aritmetico)."""
    tail = recent_states[-10:]
    maxed = sum(s >= 3 for s in tail) >= len(tail) * 0.5
    return (
        "SITUAZIONE (simbolica, non ricalcolare numeri):\n"
        f"  - regime: CRITICO da {windows_critical} finestre consecutive (persistente).\n"
        f"  - compressione: {'GIA MASSIMA (stati alti)' if maxed else 'non ancora massima'}.\n"
        f"  - il rimedio civile (stato 3) e' gia' in atto ma il sistema NON recupera.\n"
        f"  - traiettoria stati recenti (0=nessuna .. 4=scarto attivo): {tail}\n"
    )


def build_escalation_prompt(recent_states: list, windows_critical: int) -> str:
    lines = [_symbolic_context(recent_states, windows_critical),
             "\nAZIONI CANDIDATE (scegline UNA con la lettera):"]
    for name, _tgt, desc in CANDIDATES:
        lines.append(f"  {name}) {desc}")
    lines.append("\nClassifica il regime (oscillazione_transitoria / collasso_strutturale / "
                 "novita_sconosciuta) e scegli l'azione. Ricorda: compressione gia' massima "
                 "+ critico persistente = collasso, non oscillazione.")
    return "\n".join(lines)


def escalate_decision(backend, recent_states: list, windows_critical: int) -> dict:
    """
    Interroga il modello (schema ESCALATION) per classificare e scegliere.
    Ritorna {regime, choice, target_state, justification, ok}. Fail-safe:
    su errore → choice B (rimedio civile, comportamento della soglia).
    """
    prompt = build_escalation_prompt(recent_states, windows_critical)
    context = {"recent_states": recent_states, "windows_critical": windows_critical}
    try:
        raw = backend.decide(context, ESCALATION_SYSTEM_PROMPT, prompt,
                             schema=ESCALATION_JSON_SCHEMA)
        choice = raw.get("choice", "B")
        if choice not in _CHOICE_TO_TARGET:
            choice = "B"
        return {"regime": raw.get("regime", "?"), "choice": choice,
                "target_state": _CHOICE_TO_TARGET[choice],
                "justification": raw.get("justification", ""), "ok": True}
    except Exception as exc:  # noqa: BLE001 — fail-safe
        return {"regime": "errore", "choice": "B", "target_state": 3,
                "justification": f"backend errore ({exc}); fallback rimedio civile", "ok": False}
