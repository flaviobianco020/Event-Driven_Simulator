"""
Test System-2 escalation (Fase 4): l'LLM decide sul caso ambiguo.
Meccanica pura (nessun modello): trigger, mappatura scelta→stato, fail-safe.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.supervisor import escalation as esc


class TestTrigger:
    def test_escalates_on_critical_and_compressing(self):
        a = {"health": "CRITICO"}
        assert esc.should_escalate(a, [3, 4, 3, 3, 4, 3, 3, 3, 4, 3]) is True

    def test_no_escalation_when_not_compressing(self):
        # critico ma la policy e' ancora bassa → non e' il caso ambiguo (rimedio civile basta)
        a = {"health": "CRITICO"}
        assert esc.should_escalate(a, [0, 1, 1, 2, 1, 0, 1, 2, 1, 1]) is False

    def test_no_escalation_when_healthy(self):
        assert esc.should_escalate({"health": "SANO"}, [3, 3, 3, 3, 3]) is False

    def test_no_escalation_when_degraded(self):
        # DEGRADATO (non critico) resta a System 1
        assert esc.should_escalate({"health": "DEGRADATO"}, [3, 3, 3, 3, 3]) is False


class TestCandidates:
    def test_choice_maps_to_state(self):
        assert dict((n, t) for n, t, _ in esc.CANDIDATES) == {"A": None, "B": 3, "C": 4}

    def test_prompt_is_symbolic_no_raw_numbers(self):
        # il prompt NON deve contenere metriche numeriche grezze (evita il floor aritmetico)
        p = esc.build_escalation_prompt([4, 4, 3, 4, 4], windows_critical=3)
        assert "0.549" not in p and "PDR=" not in p
        assert "collasso" in p.lower() and "A)" in p and "C)" in p


class TestEscalateDecision:
    def test_reads_llm_choice(self):
        class B:
            def decide(self, ctx, sp, up, schema=None):
                return {"regime": "collasso_strutturale", "choice": "C",
                        "justification": "collasso"}
        d = esc.escalate_decision(B(), [4, 4, 4], 3)
        assert d["choice"] == "C" and d["target_state"] == 4 and d["ok"]

    def test_invalid_choice_falls_back_to_civil(self):
        class B:
            def decide(self, ctx, sp, up, schema=None):
                return {"regime": "?", "choice": "Z", "justification": "x"}
        d = esc.escalate_decision(B(), [4, 4, 4], 3)
        assert d["choice"] == "B" and d["target_state"] == 3

    def test_backend_error_is_failsafe(self):
        class B:
            def decide(self, *a, **k): raise RuntimeError("boom")
        d = esc.escalate_decision(B(), [4, 4, 4], 3)
        assert d["ok"] is False and d["target_state"] == 3   # rimedio civile
