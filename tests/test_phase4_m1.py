"""
Test M1 (Fase 4): explainer read-only sul loop MAPPO reale.

Verifica l'integrazione runner (examples/run_m1_explainer.run_m1) con il
MockBackend: cadenza dei tick lenti, garanzia read-only (nessun override
applicato nemmeno quando suggerito), struttura del log di spiegabilita'.
Richiede il checkpoint canonico committato (checkpoints/mappo_best_stab.json).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "mappo_best_stab.json")

pytestmark = pytest.mark.skipif(not os.path.exists(CKPT),
                                reason="checkpoint canonico assente")

from run_m1_explainer import run_m1  # noqa: E402


def test_m1_runs_and_ticks_on_schedule():
    # window 10s, episodio troncato a 25s → tick lenti a t=10, 20 e 25 (done)
    res = run_m1(scenario=1, seed=42, ckpt=CKPT, window_s=10.0,
                 end_time=25.0, verbose=False)
    assert res["n_slow_ticks"] == 3
    assert len(res["supervisor_log"]) == 3
    for e in res["supervisor_log"]:
        assert e["action"] and e["justification"]        # spiegazione sempre presente


def test_m1_is_strictly_read_only():
    # scenario 3 completo: il degrado di banda fa suggerire un override al mock,
    # ma il kill switch lo blocca e nessuno stato viene mai imposto.
    res = run_m1(scenario=3, seed=42, ckpt=CKPT, window_s=30.0, verbose=False)
    log = res["supervisor_log"]
    overrides = [e for e in log if e["action"] == "override_state"]
    assert overrides, "sullo scenario 3 il mock deve suggerire almeno un override"
    for e in overrides:
        assert e["approved"] is False                   # bloccato dal kill switch
    for e in log:
        assert e["effective_state"] is None             # mai imposto nulla


def test_m1_summary_consistent_with_phase3():
    # il percorso veloce resta MAPPO puro: il PDR dell'episodio deve essere
    # nell'intorno dei valori Fase 3 noti per lo scenario 1 (≈0.99).
    res = run_m1(scenario=1, seed=42, ckpt=CKPT, window_s=30.0, verbose=False)
    assert res["summary"]["pdr"] > 0.95
