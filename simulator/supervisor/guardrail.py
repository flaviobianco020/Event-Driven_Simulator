"""
guardrail.py — Guardrail + kill switch tra il supervisore LLM e il percorso veloce.

Il supervisore SUGGERISCE; il guardrail decide se applicare. Garantisce che una
decisione errata (o allucinata) del modello non possa danneggiare il sistema:

  * azioni solo dallo spazio vincolato (gia' garantito dallo schema, qui ri-validato);
  * override limitato nel tempo (MAX_OVERRIDE_SECONDS) e reversibile;
  * blocco se un KPI critico peggiora oltre soglia dopo un override → revoca;
  * kill switch: se attivo, si ignora ogni azione di controllo → MAPPO puro.

Nel peggiore dei casi un override sbagliato tiene uno stato errato per una
finestra breve, poi decade: danno limitato e reversibile. Il percorso veloce
(MAPPO) non viene mai spento — l'override sostituisce solo l'azione per la finestra.
"""
from __future__ import annotations
from dataclasses import dataclass

from .actions import Action, Decision, VALID_STATES, MAX_OVERRIDE_SECONDS


@dataclass
class GuardrailVerdict:
    approved: bool
    reason: str
    effective_state: int | None = None   # stato da imporre al percorso veloce, o None
    hold_seconds: float = 0.0


class Guardrail:
    def __init__(self, kill_switch: bool = False,
                 pdr_floor: float = 0.50):
        # kill_switch True → nessuna azione di controllo passa (solo MAPPO).
        self.kill_switch = kill_switch
        # PDR sotto cui un override viene rifiutato/revocato (protezione dura).
        self.pdr_floor = pdr_floor
        self._active_override_until: float = -1.0
        self._active_state: int | None = None

    def review(self, decision: Decision, metrics: dict, now: float) -> GuardrailVerdict:
        """Valuta una decisione del supervisore contro le regole di sicurezza."""
        # 1. Azioni non di controllo (endorse/explain/flag/coordinate): sempre ok,
        #    nessun effetto sul percorso veloce. Passano per logging/spiegabilita'.
        if not decision.is_control():
            return GuardrailVerdict(True, f"azione non di controllo: {decision.action.value}")

        # 2. Kill switch: blocca ogni override → MAPPO puro.
        if self.kill_switch:
            return GuardrailVerdict(False, "kill switch attivo: override ignorato, MAPPO puro")

        # 3. Validazione semantica dell'override.
        st = decision.target_state
        if st not in VALID_STATES:
            return GuardrailVerdict(False, f"target_state {st} fuori range")
        hold = min(float(decision.hold_seconds or 0.0), MAX_OVERRIDE_SECONDS)
        if hold <= 0.0:
            return GuardrailVerdict(False, "hold_seconds non valido")

        # 4. Protezione dura: non forzare uno stato se il PDR e' gia' sotto il floor
        #    (situazione critica: non fidarsi di un override speculativo).
        pdr = float(metrics.get("pdr", 1.0))
        if pdr < self.pdr_floor:
            return GuardrailVerdict(False, f"PDR {pdr:.2f} < floor {self.pdr_floor}: override rifiutato")

        # 5. Approvato: registra l'override attivo (reversibile, con scadenza).
        self._active_override_until = now + hold
        self._active_state = st
        return GuardrailVerdict(True, f"override approvato: stato {st} per {hold:.0f}s",
                                effective_state=st, hold_seconds=hold)

    def active_state(self, now: float) -> int | None:
        """Stato imposto attualmente dal supervisore, o None se scaduto/assente."""
        if now < self._active_override_until:
            return self._active_state
        self._active_state = None
        return None

    def revoke(self) -> None:
        """Revoca immediata dell'override attivo (es. KPI peggiorato)."""
        self._active_override_until = -1.0
        self._active_state = None
