"""
Test M2 (Fase 4): il supervisore acquisisce il controllo del percorso veloce.

Verifica: traduzione override assoluto → azione relativa; applicazione reale
dell'override al nodo (stato forzato al target durante la finestra); ritorno
del controllo a MAPPO alla scadenza; meccanismo di revoca su KPI critico;
equivalenza col comportamento M1 quando il kill switch e' attivo.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "mappo_best_stab.json")
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT),
                                reason="checkpoint canonico assente")

from run_m2_supervisor import (  # noqa: E402
    run_m2, override_to_action, maybe_revoke,
)
from simulator.marl import ESCALATE, MAINTAIN, DEESCALATE  # noqa: E402
from simulator.supervisor import Guardrail  # noqa: E402
from simulator.supervisor.actions import Action, Decision  # noqa: E402


class TestOverrideTranslation:
    def test_below_target_escalates(self):
        assert override_to_action(1, 3) == ESCALATE

    def test_above_target_deescalates(self):
        assert override_to_action(4, 3) == DEESCALATE

    def test_at_target_maintains(self):
        assert override_to_action(3, 3) == MAINTAIN


class TestRevocation:
    def _armed_guardrail(self) -> Guardrail:
        g = Guardrail()
        d = Decision(Action.OVERRIDE_STATE, "x", target_state=3, hold_seconds=60)
        v = g.review(d, {"pdr": 0.9}, now=0.0)
        assert v.approved
        return g

    def test_revokes_when_pdr_below_floor(self):
        g = self._armed_guardrail()
        assert maybe_revoke(g, {"pdr": 0.30}, t=10.0) is True
        assert g.active_state(10.0) is None          # override rimosso

    def test_keeps_override_when_pdr_ok(self):
        g = self._armed_guardrail()
        assert maybe_revoke(g, {"pdr": 0.90}, t=10.0) is False
        assert g.active_state(10.0) == 3

    def test_noop_without_active_override(self):
        g = Guardrail()
        assert maybe_revoke(g, {"pdr": 0.10}, t=0.0) is False


class TestM2Integration:
    def test_kill_switch_equals_m1(self):
        res = run_m2(scenario=3, seed=42, ckpt=CKPT, kill_switch=True, verbose=False)
        assert res["override_steps"] == 0 and res["override_windows"] == 0

    def test_override_is_actually_applied(self):
        # scenario 3: a t=60 assess → override (PDR crolla). Con controllo attivo
        # il supervisore deve comandare per almeno una finestra.
        res = run_m2(scenario=3, seed=42, ckpt=CKPT, kill_switch=False, verbose=False)
        assert res["override_windows"] >= 1
        assert res["override_steps"] >= 10           # ~una finestra da 30 tick
        # durante l'override lo stato deve convergere allo stato IMPOSTO dal
        # guardrail (3 o 4 a seconda della regola di escalation) e tenerlo.
        imposed = next(e["effective_state"] for e in res["supervisor_log"]
                       if e["effective_state"] is not None)
        traj = res["trajectory"]
        window = traj[60:90]                          # la finestra sotto override
        assert any(window[i:i + 5] == [imposed] * 5 for i in range(len(window) - 5)), \
            f"attesa tenuta stabile sullo stato imposto {imposed}, traiettoria: {window}"

    def test_mappo_resumes_after_override(self):
        # dopo la scadenza dell'override il controllo torna a MAPPO:
        # override_steps deve restare molto sotto la durata dell'episodio (90 tick)
        res = run_m2(scenario=3, seed=42, ckpt=CKPT, kill_switch=False, verbose=False)
        assert res["override_steps"] < 80
