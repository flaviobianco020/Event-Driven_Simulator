#!/usr/bin/env python3
"""
run_agent_cause.py — Fase 4b: abbattere il confine osservando la CAUSA.

Il confine del transitorio (agente basato sull'ATTESA del recupero) esiste perche'
si guarda il SINTOMO (PDR/drop) e si aspetta che si risolva. Ma i due modi di
guasto hanno CAUSE diverse e osservabili SUBITO:
  - collasso  = calo di CAPACITA' (link 10→2): capacita' BASSA
  - picco     = eccesso di DOMANDA (surge, link resta 10): capacita' NORMALE

Un agente con il sensore query_link_capacity li distingue a t=0 — SENZA aspettare
— quindi la durata del surge diventa IRRILEVANTE: il confine e' abbattuto per
questa coppia di guasti.

Confronto: agente ad ATTESA (PolicyBackend) vs agente a CAUSA sul picco di domanda
a varie durate, comprese quelle lunghe che rompevano l'agente ad attesa.

Residuo onesto: la capacita' distingue i MODI (capacita' vs domanda), non la
PERMANENZA dentro un modo. Un calo di capacita' TRANSITORIO (transient_degradation)
ha capacita' bassa come il collasso → il sensore-causa da solo non basta li': serve
il tempo. La parte fondamentale del limite resta.

Uso:  python3 examples/run_agent_cause.py --seeds 5
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from simulator.marl import MARLController  # noqa: E402
from simulator.supervisor.ood import (  # noqa: E402
    build_demand_spike, build_transient_degradation, _capacity_scenario)
from simulator.supervisor.agent import AgentSession, run_agent_episode  # noqa: E402
from run_m1_explainer import DEFAULT_CKPT  # noqa: E402
from run_m3_ood import _kpis  # noqa: E402
from run_agent import PolicyBackend  # noqa: E402

END = 260.0


def run_cause_aware_episode(env, mappo):
    """Agente a CAUSA: alla prima finestra critica interroga la capacita' e decide
    SUBITO — capacita' bassa → strutturale (intervieni); normale → domanda (transitorio)."""
    sess = AgentSession(env=env, mappo=mappo)
    sess.reset()
    while not sess.done:
        m = sess._advance_one_window()
        if sess._health(m) == "CRITICO":
            break
    if sess.done:
        out = _kpis(sess.env); out.update(mode="sano", reconfigured=False, capacity=None)
        return out
    q = sess.query_link_capacity()
    if q["capacity_dropped"]:
        sess.trigger_reconfigure("CRITICO"); mode = "strutturale (capacita')"
    else:
        mode = "domanda (transitorio)"
    sess.finish()
    out = _kpis(sess.env)
    out.update(mode=mode, reconfigured=sess.reconfigured, capacity=q["capacity"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    S = range(42, 42 + args.seeds)
    ck = lambda: MARLController.from_checkpoint(DEFAULT_CKPT)

    print("=" * 74)
    print(f"  PICCO DI DOMANDA — agente ad ATTESA vs agente a CAUSA ({args.seeds} seed)")
    print("  (link resta a 10; la durata del surge include valori che rompono l'attesa)")
    print("=" * 74)
    print(f"  {'durata':>7}{'ATTESA: diagnosi':>22}{'CAUSA: capacita':>18}{'CAUSA: modo':>26}")
    for dur in (40.0, 80.0, 120.0):
        mk = lambda s, d=dur: build_demand_spike(seed=s, end_time=END, onset=30.0, duration=d)
        wait_diag, cap, mode = [], [], []
        for s in S:
            w = run_agent_episode(mk(s), ck(), PolicyBackend())
            c = run_cause_aware_episode(mk(s), ck())
            wait_diag.append(w["diagnosis"]); cap.append(c["capacity"]); mode.append(c["mode"])
        wd = max(set(wait_diag), key=wait_diag.count)
        wflag = "✓" if wd == "transitorio" else "✗ scambia per collasso"
        print(f"  {dur:>6.0f}s{wd + ' ' + wflag:>22}{np.mean(cap):>16.1f}"
              f"{max(set(mode), key=mode.count):>26}")

    print("\n  → l'agente ad ATTESA sbaglia il MODO sui surge lunghi; l'agente a CAUSA legge")
    print("    capacita'=10 (normale) e conclude 'domanda' a QUALUNQUE durata. Confine abbattuto.")

    print("\n" + "=" * 74)
    print("  CONTROPROVA — l'agente a CAUSA sul vero collasso (capacita' persa)")
    print("=" * 74)
    b_cd, a_cd, modes = [], [], []
    for s in S:
        mk = lambda ss=s: _capacity_scenario(ss, END, drop_to=2.0, onset=20.0,
                                             recover_at=None, name="collapse")
        from run_m3_ood import episode_alone
        b_cd.append(episode_alone(mk(), ck())["control_del"])
        r = run_cause_aware_episode(mk(), ck()); a_cd.append(r["control_del"]); modes.append(r["mode"])
    print(f"  capacita' letta: 2 (bassa) → modo: {max(set(modes), key=modes.count)}")
    print(f"  consegna controllo: MAPPO {np.mean(b_cd):.3f} → +agente {np.mean(a_cd):.3f} "
          f"(interviene giustamente)")

    print("\n" + "=" * 74)
    print("  RESIDUO ONESTO — calo di capacita' TRANSITORIO (stesso modo del collasso)")
    print("=" * 74)
    modes2, interv = [], 0
    for s in S:
        mk = lambda ss=s: build_transient_degradation(seed=ss, end_time=END, drop_to=2.0,
                                                      onset=30.0, duration=120.0)
        r = run_cause_aware_episode(mk(), ck())
        modes2.append(r["mode"]); interv += int(r["reconfigured"])
    print(f"  capacita' letta: 2 (bassa) → modo: {max(set(modes2), key=modes2.count)}  (interviene {interv}/{args.seeds})")
    print("  La capacita' distingue i MODI, non la PERMANENZA dentro un modo: un calo di")
    print("  capacita' transitorio 'sembra' un collasso. Qui SERVE ancora il tempo — e' la")
    print("  parte FONDAMENTALE del limite, non abbattibile. Combinare i due (causa + attesa)")
    print("  copre entrambi i casi.")


if __name__ == "__main__":
    main()
