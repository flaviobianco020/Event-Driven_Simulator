#!/usr/bin/env python3
"""
run_agent_robustness.py — Fase 4b: dove l'agente REGGE e dove si ROMPE.

Non solo "piu' seed": due sweep che caratterizzano i limiti onestamente.

  STUDIO 1 — SEVERITA' del collasso (link crolla a 2/3/4/5 pkt/s, permanente).
    Domanda: l'agente diagnostica «permanente» e protegge il controllo a ogni
    gravita'? Atteso: si' (un permanente resta critico comunque, l'attesa lo rivela).

  STUDIO 2 — DURATA del transitorio (degrado che recupera dopo 20..120 s).
    LA prova chiave. L'agente aspetta ~60 s per confermare. Un transitorio piu'
    lungo dell'attesa non e' ancora recuperato quando l'agente decide → lo scambia
    per collasso, interviene, DANNEGGIA. E' il floor di osservabilita' che riemerge
    a scala piu' lunga: aspettare risolve i transitori PIU' CORTI della finestra
    d'osservazione, non quelli piu' lunghi. Lo sweep trova il confine.

Backend 'policy' (deterministico): stesse azioni dell'agente LLM, senza attesa
Ollama. N seed per configurazione.

Uso:  python3 examples/run_agent_robustness.py --seeds 5
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from simulator.marl import MARLController  # noqa: E402
from simulator.supervisor.ood import _capacity_scenario, build_transient_degradation  # noqa: E402
from simulator.supervisor.agent import run_agent_episode  # noqa: E402
from run_m1_explainer import DEFAULT_CKPT  # noqa: E402
from run_m3_ood import episode_alone  # noqa: E402
from run_agent import PolicyBackend  # noqa: E402

END = 260.0


def _agent(make_env, seed):
    return run_agent_episode(make_env(seed), MARLController.from_checkpoint(DEFAULT_CKPT),
                             PolicyBackend())


def _alone(make_env, seed):
    return episode_alone(make_env(seed), MARLController.from_checkpoint(DEFAULT_CKPT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = ap.parse_args()
    S = range(42, 42 + args.seeds)

    print("=" * 72)
    print(f"  STUDIO 1 — SEVERITA' del collasso permanente ({args.seeds} seed)")
    print("  (atteso: diagnosi 'permanente' + controllo protetto a ogni gravita')")
    print("=" * 72)
    print(f"  {'cap':>5}{'diagn. perm.':>14}{'ctrl MAPPO':>12}{'ctrl +ag':>10}{'Δ ctrl':>9}")
    for cap in (2.0, 3.0, 4.0, 5.0):
        mk = lambda s, c=cap: _capacity_scenario(s, END, drop_to=c, onset=20.0,
                                                 recover_at=None, name="collapse")
        perm = base = ag = 0
        b_cd, a_cd = [], []
        for s in S:
            r = _agent(mk, s)
            perm += int(r["diagnosis"] == "collasso_permanente")
            a_cd.append(r["control_del"]); b_cd.append(_alone(mk, s)["control_del"])
        print(f"  {cap:>5.0f}{perm:>10}/{args.seeds:<3}{np.mean(b_cd):>12.3f}"
              f"{np.mean(a_cd):>10.3f}{np.mean(a_cd)-np.mean(b_cd):>+9.3f}")

    print("\n" + "=" * 72)
    print(f"  STUDIO 2 — DURATA del transitorio (recupera dopo D secondi, {args.seeds} seed)")
    print("  (atteso: D corta → 'transitorio', nessun intervento; D lunga → confine)")
    print("=" * 72)
    print(f"  {'durata':>7}{'diagn. transit.':>17}{'intervenuti':>13}{'PDR MAPPO':>11}{'PDR +ag':>10}{'':>3}")
    for dur in (20.0, 40.0, 60.0, 80.0, 100.0, 120.0):
        mk = lambda s, d=dur: build_transient_degradation(seed=s, end_time=END,
                                                          drop_to=2.0, onset=30.0, duration=d)
        transit = interv = 0
        b_pdr, a_pdr = [], []
        for s in S:
            r = _agent(mk, s)
            transit += int(r["diagnosis"] == "transitorio")
            interv += int(r["reconfigured"])
            a_pdr.append(r["pdr"]); b_pdr.append(_alone(mk, s)["pdr"])
        harm = np.mean(a_pdr) - np.mean(b_pdr)
        flag = "  ✓ corretto" if interv == 0 else ("  ▽ DANNO" if harm < -0.02 else "  ~ intervento")
        print(f"  {dur:>6.0f}s{transit:>13}/{args.seeds:<3}{interv:>10}/{args.seeds:<3}"
              f"{np.mean(b_pdr):>11.3f}{np.mean(a_pdr):>10.3f}{flag}")

    print("\n  Lettura: il confine e' dove l'agente smette di vedere il recupero entro la sua")
    print("  finestra d'attesa. Sotto il confine e' robusto; sopra, il floor di osservabilita'")
    print("  riemerge — mitigabile allungando l'attesa (a costo di reattivita').")


if __name__ == "__main__":
    main()
