#!/usr/bin/env python3
"""
compare_phase2_phase3.py — Confronto diretto Fase 2 (baseline) vs Fase 3 (MAPPO).

Esegue i 6 scenari canonici sotto DUE controller diversi, a parita' di tutto
il resto (stessa topologia, stessi flussi, stesso seed → stesso traffico):

  * FASE 2  (baseline)  — CongestionStateMachine con EWMA (alpha=0.125) +
                          isteresi asimmetrica (escalation 1.5s, cooldown 4.5s),
                          transizioni automatiche guidate dall'occupancy
                          (eFRAC, doc Fase 2).
  * FASE 3  (MAPPO)     — policy Actor appresa: la stessa macchina a stati e'
                          pilotata dalle azioni {ESCALATE, MAINTAIN, DE-ESCALATE}
                          decise dalla rete neurale (checkpoint JSON).

L'UNICA differenza tra i due run e' CHI decide le transizioni di stato: il
meccanismo di enqueue/compressione/drop e il traffico sono identici, quindi
ogni differenza nelle metriche e' imputabile alla policy.

Per robustezza statistica media su piu' seed (default 5). KPI confrontati
(doc Tabella 10): PDR, latenza end-to-end, Jain Fairness, compression ratio,
numero di transizioni di stato.

Uso:
    python3 examples/compare_phase2_phase3.py
    python3 examples/compare_phase2_phase3.py --seeds 10
    python3 examples/compare_phase2_phase3.py --ckpt checkpoints/mappo_best.json
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from simulator import ConfigurationManager, MetricsEngine, Simulator  # noqa: E402
from simulator.marl import EDSMarlEnv, load_checkpoint  # noqa: E402
from simulator.marl.env import _build_scenario  # noqa: E402

SCENARIOS = (1, 2, 3, 4, 5, 6)
SCEN_NAMES = {
    1: "single bottleneck", 2: "flash crowd", 3: "bandwidth degr.",
    4: "link fail/recov.", 5: "persistent overload", 6: "mixed traffic",
}
KPIS = ("pdr", "latency", "fairness", "compression_ratio", "transitions")


def _summ(metrics: MetricsEngine) -> dict:
    return {
        "pdr": metrics.collect_packet_delivery_ratio(),
        "latency": metrics.collect_end_to_end_latency(),
        "fairness": metrics.collect_fairness(),
        "compression_ratio": metrics.collect_compression_ratio(),
        "transitions": metrics.collect_congestion_state_transitions(),
    }


def run_phase2(scenario: int, seed: int) -> dict:
    """Baseline Fase 2: transizioni automatiche EWMA + isteresi."""
    topo, gen, events, end_time = _build_scenario(scenario, seed)
    metrics = MetricsEngine()
    sim = Simulator(
        ConfigurationManager(random_seed=seed), topo, gen, metrics,
        end_time=end_time, metric_interval=10.0, enable_phase2=True,
    )
    for ev in events:
        sim.scheduler.schedule_event(ev)
    sim.run()
    return _summ(metrics)


def run_phase3(scenario: int, seed: int, actor) -> dict:
    """Fase 3: la policy MAPPO pilota le transizioni (deterministica)."""
    rng = np.random.default_rng(seed)
    env = EDSMarlEnv(scenario, seed=seed)      # end_time canonico dello scenario
    obs, _state = env.reset()
    done = False
    while not done:
        actions, _ = actor.act(obs, rng, deterministic=True)
        obs, _state, _r, done, _info = env.step(actions)
    return env.summary()


def _fmt(kpi: str, v: float) -> str:
    if kpi == "pdr" or kpi == "fairness":
        return f"{v*100:6.2f}%" if kpi == "pdr" else f"{v:6.3f}"
    if kpi == "latency":
        return f"{v*1000:7.1f}ms"
    if kpi == "compression_ratio":
        return f"{v:5.2f}x"
    return f"{v:5.0f}"


def _delta_str(kpi: str, p2: float, p3: float) -> str:
    """Freccia + segno: ↑ = Fase 3 meglio, ↓ = peggio (verso corretto per KPI)."""
    # per la latenza e le transizioni "meno e' meglio"; per gli altri "piu' e' meglio"
    lower_better = kpi in ("latency", "transitions")
    diff = p3 - p2
    if abs(diff) < 1e-9:
        return "  ="
    better = (diff < 0) if lower_better else (diff > 0)
    if kpi == "pdr" or kpi == "fairness":
        mag = f"{abs(diff)*100:.1f}{'%' if kpi=='pdr' else ''}"
    elif kpi == "latency":
        mag = f"{abs(diff)*1000:.0f}ms"
    elif kpi == "compression_ratio":
        mag = f"{abs(diff):.2f}x"
    else:
        mag = f"{abs(diff):.0f}"
    arrow = "▲" if better else "▼"
    return f"{arrow}{mag}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Confronto Fase 2 vs Fase 3 (MAPPO)")
    ap.add_argument("--seeds", type=int, default=5,
                    help="numero di seed su cui mediare (default 5)")
    ap.add_argument("--ckpt", default="checkpoints/mappo_best.json",
                    help="checkpoint MAPPO da valutare")
    args = ap.parse_args()

    ckpt = args.ckpt
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(os.path.dirname(__file__), "..", ckpt)
    if not os.path.exists(ckpt):
        print(f"ERRORE: checkpoint non trovato: {ckpt}")
        print("Esegui prima:  python3 examples/train_mappo.py")
        sys.exit(1)

    actor, _critic, meta = load_checkpoint(ckpt)
    seeds = list(range(1000, 1000 + args.seeds))

    print("=" * 78)
    print("  CONFRONTO  Fase 2 (baseline EWMA+isteresi)  vs  Fase 3 (MAPPO)")
    print(f"  checkpoint: {os.path.basename(ckpt)}  "
          f"(episodio {meta.get('episode','?')}, "
          f"eval_reward {meta.get('eval_reward', float('nan')):.4f})")
    print(f"  media su {len(seeds)} seed  |  ▲ = Fase 3 meglio, ▼ = peggio")
    print("=" * 78)

    agg = {"p2": {k: [] for k in KPIS}, "p3": {k: [] for k in KPIS}}

    for sc in SCENARIOS:
        p2 = {k: [] for k in KPIS}
        p3 = {k: [] for k in KPIS}
        for sd in seeds:
            r2 = run_phase2(sc, sd)
            r3 = run_phase3(sc, sd, actor)
            for k in KPIS:
                p2[k].append(r2[k])
                p3[k].append(r3[k])
        m2 = {k: float(np.mean(p2[k])) for k in KPIS}
        m3 = {k: float(np.mean(p3[k])) for k in KPIS}
        for k in KPIS:
            agg["p2"][k].append(m2[k])
            agg["p3"][k].append(m3[k])

        print(f"\n  Scenario {sc} — {SCEN_NAMES[sc]}")
        print(f"    {'KPI':<18}{'Fase 2':>11}{'Fase 3':>11}{'Δ':>10}")
        rows = [
            ("PDR", "pdr"), ("Latenza", "latency"),
            ("Jain fairness", "fairness"),
            ("Compression", "compression_ratio"),
            ("Transizioni", "transitions"),
        ]
        for label, k in rows:
            print(f"    {label:<18}{_fmt(k, m2[k]):>11}{_fmt(k, m3[k]):>11}"
                  f"{_delta_str(k, m2[k], m3[k]):>10}")

    # ── media globale su tutti gli scenari ────────────────────────────────────
    print("\n" + "=" * 78)
    print("  MEDIA GLOBALE (tutti gli scenari)")
    print(f"    {'KPI':<18}{'Fase 2':>11}{'Fase 3':>11}{'Δ':>10}")
    wins = 0
    for label, k in [("PDR", "pdr"), ("Latenza", "latency"),
                     ("Jain fairness", "fairness"),
                     ("Compression", "compression_ratio"),
                     ("Transizioni", "transitions")]:
        g2 = float(np.mean(agg["p2"][k]))
        g3 = float(np.mean(agg["p3"][k]))
        d = _delta_str(k, g2, g3)
        if d.startswith("▲"):
            wins += 1
        print(f"    {label:<18}{_fmt(k, g2):>11}{_fmt(k, g3):>11}{d:>10}")
    print("=" * 78)

    # verdetto principale: PDR globale (obiettivo primario, doc Tabella 9)
    g2_pdr = float(np.mean(agg["p2"]["pdr"]))
    g3_pdr = float(np.mean(agg["p3"]["pdr"]))
    print(f"\n  Verdetto: Fase 3 vince su {wins}/5 KPI globali.")
    if g3_pdr > g2_pdr + 0.005:
        print(f"  → MAPPO consegna piu' pacchetti: PDR {g2_pdr*100:.2f}% "
              f"→ {g3_pdr*100:.2f}%  (+{(g3_pdr-g2_pdr)*100:.2f} punti).")
    elif g3_pdr < g2_pdr - 0.005:
        print(f"  → La baseline Fase 2 consegna piu' pacchetti "
              f"({g2_pdr*100:.2f}% vs {g3_pdr*100:.2f}%): "
              f"la policy va allenata piu' a lungo.")
    else:
        print(f"  → PDR sostanzialmente pari ({g2_pdr*100:.2f}% vs "
              f"{g3_pdr*100:.2f}%); guarda gli altri KPI per il trade-off.")
    print()


if __name__ == "__main__":
    main()
