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

    def test_reconfigure_guardrail_blocks_when_healthy(self):
        # se il sistema non e' critico, la riconfigurazione va rifiutata dal guardrail
        from simulator.supervisor.agent import AgentSession
        from simulator.marl import EDSMarlEnv, MARLController
        sess = AgentSession(env=EDSMarlEnv(1, seed=42, end_time=60.0),
                            mappo=MARLController.from_checkpoint(CKPT))
        sess.reset()
        res = sess.trigger_reconfigure(current_health="SANO")
        assert res["applicato"] is False
