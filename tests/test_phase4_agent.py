"""
Test agente (Fase 4b): il ciclo percepisci→indaga→agisci e i guardrail sui tool.
La meccanica pura non richiede modello (PolicyBackend deterministico).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

from simulator.supervisor.agent import (
    AGENT_TOOL_SCHEMA, MAX_WAIT_WINDOWS, run_agent_episode,
)

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "mappo_best_stab.json")


class TestToolSchema:
    def test_tools_present(self):
        tools = AGENT_TOOL_SCHEMA["properties"]["tool"]["enum"]
        assert set(tools) == {"query_diagnostics", "wait_and_observe",
                              "trigger_reconfigure", "conclude"}

    def test_diagnosis_options(self):
        diag = AGENT_TOOL_SCHEMA["properties"]["diagnosis"]["enum"]
        assert "collasso_permanente" in diag and "transitorio" in diag


class TestPolicyBackend:
    """L'agente deterministico: indaga → se resta critico interviene, altrimenti no."""

    def _backend(self):
        from run_agent import PolicyBackend
        return PolicyBackend()

    def test_first_action_is_investigate(self):
        b = self._backend()
        call = b.decide({"obs": {"health": "CRITICO"}}, "", "")
        assert call["tool"] == "wait_and_observe"

    def test_reconfigures_if_still_critical(self):
        b = self._backend()
        b.decide({"obs": {"health": "CRITICO"}}, "", "")           # wait
        call = b.decide({"obs": {"health": "CRITICO"}}, "", "")    # dopo attesa: ancora critico
        assert call["tool"] == "trigger_reconfigure"

    def test_concludes_transient_if_recovered(self):
        b = self._backend()
        b.decide({"obs": {"health": "CRITICO"}}, "", "")           # wait
        call = b.decide({"obs": {"health": "SANO"}}, "", "")       # recuperato
        assert call["tool"] == "conclude" and call["diagnosis"] == "transitorio"


@pytest.mark.skipif(not os.path.exists(CKPT), reason="checkpoint canonico assente")
class TestAgentDiscrimination:
    """Il test chiave: l'agente distingue permanente da transitorio INDAGANDO."""

    def _run(self, make_env):
        from run_agent import PolicyBackend
        from simulator.marl import MARLController
        return run_agent_episode(make_env(), MARLController.from_checkpoint(CKPT),
                                 PolicyBackend())

    def test_permanent_collapse_triggers_intervention(self):
        from simulator.supervisor.ood import build_capacity_collapse
        r = self._run(lambda: build_capacity_collapse(seed=42, end_time=200.0))
        assert r["diagnosis"] == "collasso_permanente" and r["reconfigured"] is True
        assert r["control_del"] > 0.85          # controllo protetto

    def test_transient_does_not_intervene(self):
        from simulator.marl import EDSMarlEnv
        r = self._run(lambda: EDSMarlEnv(3, seed=42, end_time=200.0))
        assert r["diagnosis"] == "transitorio" and r["reconfigured"] is False
        assert r["pdr"] > 0.85                   # nessun danno da intervento errato

    def test_short_transient_no_intervention(self):
        # transitorio piu' CORTO della finestra d'attesa → l'agente vede il recupero
        from simulator.supervisor.ood import build_transient_degradation
        r = self._run(lambda: build_transient_degradation(seed=42, end_time=260.0,
                                                          drop_to=2.0, onset=30.0, duration=40.0))
        assert r["diagnosis"] == "transitorio" and r["reconfigured"] is False

    def test_long_transient_is_the_known_boundary(self):
        # transitorio piu' LUNGO dell'attesa → l'agente non vede ancora il recupero e
        # lo scambia per collasso. E' il floor di osservabilita' che riemerge (limite noto).
        from simulator.supervisor.ood import build_transient_degradation
        r = self._run(lambda: build_transient_degradation(seed=42, end_time=260.0,
                                                          drop_to=2.0, onset=30.0, duration=100.0))
        assert r["reconfigured"] is True   # documenta il confine, non un bug

    def test_reconfigure_guardrail_blocks_when_healthy(self):
        # se il sistema non e' critico, la riconfigurazione va rifiutata dal guardrail
        from simulator.supervisor.agent import AgentSession
        from simulator.marl import EDSMarlEnv, MARLController
        sess = AgentSession(env=EDSMarlEnv(1, seed=42, end_time=60.0),
                            mappo=MARLController.from_checkpoint(CKPT))
        sess.reset()
        res = sess.trigger_reconfigure(current_health="SANO")
        assert res["applicato"] is False


@pytest.mark.skipif(not os.path.exists(CKPT), reason="checkpoint canonico assente")
class TestMonitorRetract:
    """Rete di sicurezza reversibile: dopo un intervento, ritira quando la causa
    e' risolta → il confine sul transitorio lungo ha costo LIMITATO, non permanente."""

    def _run(self, make_env, monitor):
        from simulator.marl import MARLController
        from run_agent import PolicyBackend
        return run_agent_episode(make_env(), MARLController.from_checkpoint(CKPT),
                                 PolicyBackend(), monitor=monitor)

    def test_retract_on_transient_recovery(self):
        from simulator.supervisor.ood import build_transient_degradation
        mk = lambda: build_transient_degradation(seed=42, end_time=300.0, drop_to=2.0,
                                                 onset=30.0, duration=120.0)
        no_mon = self._run(mk, monitor=False)
        mon = self._run(mk, monitor=True)
        assert mon["retracted"] is True and no_mon["retracted"] is False
        assert mon["pdr"] > no_mon["pdr"] + 0.2      # danno fortemente ridotto

    def test_no_retract_on_permanent_collapse(self):
        from simulator.supervisor.ood import _capacity_scenario
        mk = lambda: _capacity_scenario(42, 300.0, drop_to=2.0, onset=20.0,
                                        recover_at=None, name="c")
        mon = self._run(mk, monitor=True)
        assert mon["retracted"] is False and mon["control_del"] > 0.85  # resta protetto


@pytest.mark.skipif(not os.path.exists(CKPT), reason="checkpoint canonico assente")
class TestCauseSensor:
    """Il sensore della CAUSA (query_link_capacity) distingue i modi di guasto
    (calo di capacita' vs picco di domanda) senza aspettare → abbatte il confine."""

    def _session_at_t60(self, env):
        from simulator.supervisor.agent import AgentSession
        from simulator.marl import MARLController
        sess = AgentSession(env=env, mappo=MARLController.from_checkpoint(CKPT))
        sess.reset()
        while sess.env.t < 60 and not sess.done:
            sess._advance_one_window()
        return sess

    def test_capacity_collapse_reads_low(self):
        from simulator.supervisor.ood import _capacity_scenario
        sess = self._session_at_t60(_capacity_scenario(42, 200.0, drop_to=2.0, onset=20.0,
                                                        recover_at=None, name="c"))
        q = sess.query_link_capacity()
        assert q["capacity_dropped"] is True and q["capacity"] == 2.0

    def test_demand_spike_reads_normal(self):
        # anche con un surge LUNGO (che romperebbe l'agente ad attesa), la capacita'
        # resta nominale → il sensore-causa dice 'domanda' istantaneamente.
        from simulator.supervisor.ood import build_demand_spike
        sess = self._session_at_t60(build_demand_spike(seed=42, end_time=260.0,
                                                       onset=30.0, duration=120.0))
        q = sess.query_link_capacity()
        assert q["capacity_dropped"] is False and q["capacity"] == 10.0
