"""
Smoke test dello scheletro Fase 4 (supervisore LLM). Verifica solo la meccanica
disaccoppiata — nessun modello richiesto (usa MockBackend). Il valore aggiunto
(spiegabilita', recupero OOD) sara' valutato in M3, non qui.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.supervisor import (
    SupervisorController, Guardrail, MockBackend, Action,
)
from simulator.supervisor.actions import (
    Decision, decision_from_dict, VALID_STATES, MAX_OVERRIDE_SECONDS,
)


class TestActions:
    def test_decision_from_dict_minimal(self):
        d = decision_from_dict({"action": "endorse", "justification": "ok"})
        assert d.action == Action.ENDORSE and not d.is_control()

    def test_override_is_control(self):
        d = decision_from_dict({"action": "override_state", "target_state": 3,
                                "hold_seconds": 30, "justification": "x"})
        assert d.is_control() and d.target_state == 3


class TestGuardrail:
    def test_endorse_always_approved(self):
        g = Guardrail()
        v = g.review(Decision(Action.ENDORSE, "ok"), {"pdr": 0.9}, now=0.0)
        assert v.approved and v.effective_state is None

    def test_override_approved_sets_state(self):
        g = Guardrail()
        d = Decision(Action.OVERRIDE_STATE, "x", target_state=3, hold_seconds=30)
        v = g.review(d, {"pdr": 0.9}, now=0.0)
        assert v.approved and v.effective_state == 3
        assert g.active_state(now=10.0) == 3      # dentro la finestra
        assert g.active_state(now=100.0) is None  # scaduta

    def test_kill_switch_blocks_override(self):
        g = Guardrail(kill_switch=True)
        d = Decision(Action.OVERRIDE_STATE, "x", target_state=3, hold_seconds=30)
        v = g.review(d, {"pdr": 0.9}, now=0.0)
        assert not v.approved

    def test_pdr_floor_rejects_override(self):
        g = Guardrail(pdr_floor=0.5)
        d = Decision(Action.OVERRIDE_STATE, "x", target_state=3, hold_seconds=30)
        v = g.review(d, {"pdr": 0.3}, now=0.0)     # PDR sotto il floor
        assert not v.approved

    def test_hold_capped(self):
        g = Guardrail()
        d = Decision(Action.OVERRIDE_STATE, "x", target_state=3, hold_seconds=9999)
        v = g.review(d, {"pdr": 0.9}, now=0.0)
        assert v.hold_seconds <= MAX_OVERRIDE_SECONDS

    def test_invalid_state_rejected(self):
        g = Guardrail()
        d = Decision(Action.OVERRIDE_STATE, "x", target_state=9, hold_seconds=30)
        v = g.review(d, {"pdr": 0.9}, now=0.0)
        assert not v.approved

    def test_revoke(self):
        g = Guardrail()
        d = Decision(Action.OVERRIDE_STATE, "x", target_state=2, hold_seconds=60)
        g.review(d, {"pdr": 0.9}, now=0.0)
        assert g.active_state(now=5.0) == 2
        g.revoke()
        assert g.active_state(now=5.0) is None


class TestController:
    def test_mock_endorses_when_calm(self):
        ctrl = SupervisorController(backend=MockBackend(drop_threshold=0.15))
        v = ctrl.tick(0.0, {"pdr": 0.98, "drop_rate": 0.01}, [2, 2, 3])
        assert ctrl.log.entries[-1]["action"] == "endorse"
        assert ctrl.current_override(0.0) is None

    def test_mock_overrides_on_high_drop(self):
        ctrl = SupervisorController(backend=MockBackend(drop_threshold=0.15))
        v = ctrl.tick(0.0, {"pdr": 0.7, "drop_rate": 0.30}, [1, 0, 3])
        assert v.approved and ctrl.current_override(0.0) == 3

    def test_backend_error_falls_back_to_endorse(self):
        class Broken(MockBackend):
            def decide(self, *a, **k): raise RuntimeError("boom")
        ctrl = SupervisorController(backend=Broken())
        v = ctrl.tick(0.0, {"pdr": 0.9, "drop_rate": 0.0}, [])
        # fail-safe: il percorso veloce non viene mai bloccato
        assert ctrl.log.entries[-1]["action"] == "endorse"
        assert ctrl.current_override(0.0) is None

    def test_log_accumulates(self):
        ctrl = SupervisorController(backend=MockBackend())
        for t in range(3):
            ctrl.tick(float(t), {"pdr": 0.9, "drop_rate": 0.0}, [])
        assert len(ctrl.log.entries) == 3
