#!/usr/bin/env python3
"""
run_agent_recover.py — Fase 4b: rete di sicurezza reversibile (monitora-e-ritira).

Il confine del transitorio (§robustezza) non sparisce: l'agente non puo' sapere al
momento della decisione se un degrado lungo recuperera'. Ma puo' rendere l'errore
ECONOMICO: dopo un intervento, RI-VALUTA a ogni finestra e RITIRA l'override appena
la salute recupera. Il collasso vero non recupera (resta protetto); il transitorio
lungo recupera (override ritirato, MAPPO riprende) → il danno e' limitato alla
finestra pre-recupero, non permanente.

Confronto sul transitorio LUNGO (durata 120s, oltre la finestra d'attesa):
  - senza monitor: l'agente scambia per collasso e tiene lo stato 4 fino a fine → DANNO
  - con monitor:   ritira all'atto del recupero → danno limitato
Controprova sul collasso permanente: con monitor NON ritira (non recupera) → resta
protetto.

Backend policy (deterministico). Uso:  python3 examples/run_agent_recover.py --seeds 5
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from simulator.marl import MARLController  # noqa: E402
from simulator.supervisor.ood import build_transient_degradation, _capacity_scenario  # noqa: E402
from simulator.supervisor.agent import run_agent_episode  # noqa: E402
from run_m1_explainer import DEFAULT_CKPT  # noqa: E402
from run_m3_ood import episode_alone  # noqa: E402
from run_agent import PolicyBackend  # noqa: E402

END = 300.0


def _run(make_env, seed, monitor):
    return run_agent_episode(make_env(seed), MARLController.from_checkpoint(DEFAULT_CKPT),
                             PolicyBackend(), monitor=monitor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    S = range(42, 42 + args.seeds)

    print("=" * 70)
    print(f"  TRANSITORIO LUNGO (durata 120s, oltre l'attesa) — {args.seeds} seed")
    print("=" * 70)
    tr = lambda s: build_transient_degradation(seed=s, end_time=END, drop_to=2.0,
                                               onset=30.0, duration=120.0)
    base = [episode_alone(tr(s), MARLController.from_checkpoint(DEFAULT_CKPT))["pdr"] for s in S]
    no_mon = [_run(tr, s, monitor=False) for s in S]
    mon = [_run(tr, s, monitor=True) for s in S]
    print(f"  {'braccio':<28}{'PDR':>8}{'ritirato':>12}{'':>4}")
    print(f"  {'MAPPO solo':<28}{np.mean(base):>8.3f}{'—':>12}")
    print(f"  {'agente senza monitor':<28}{np.mean([r['pdr'] for r in no_mon]):>8.3f}"
          f"{'0/'+str(args.seeds):>12}   ▽ danno permanente")
    print(f"  {'agente CON monitor':<28}{np.mean([r['pdr'] for r in mon]):>8.3f}"
          f"{str(sum(r['retracted'] for r in mon))+'/'+str(args.seeds):>12}   △ danno limitato")
    rt = np.mean([r["retract_t"] for r in mon if r["retracted"]]) if any(r["retracted"] for r in mon) else 0
    print(f"\n  recupero del link a t=150s; override ritirato in media a t={rt:.0f}s "
          f"→ danno confinato alla finestra pre-recupero.")

    print("\n" + "=" * 70)
    print("  CONTROPROVA — collasso PERMANENTE (non recupera) con monitor")
    print("=" * 70)
    ck = lambda s: _capacity_scenario(s, END, drop_to=2.0, onset=20.0, recover_at=None, name="c")
    b_cd = [episode_alone(ck(s), MARLController.from_checkpoint(DEFAULT_CKPT))["control_del"] for s in S]
    m_cd = [_run(ck, s, monitor=True) for s in S]
    print(f"  consegna controllo: MAPPO {np.mean(b_cd):.3f} → +agente(monitor) "
          f"{np.mean([r['control_del'] for r in m_cd]):.3f}  "
          f"(ritirato {sum(r['retracted'] for r in m_cd)}/{args.seeds} → resta protetto)")
    print("\n  Il confine non sparisce, ma il suo COSTO si': l'agente non deve indovinare al")
    print("  momento — corregge quando arriva l'evidenza (il recupero e' auto-evidente).")


if __name__ == "__main__":
    main()
