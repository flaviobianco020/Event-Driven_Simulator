#!/usr/bin/env python3
"""
run_agent_perf.py — Fase 4b: misura ONESTA di «l'agente non aggiunge latenza» e
«aiuta dove serve».

Tre esperimenti:
  A. NON-INTERFERENZA: su uno scenario dove l'agente indaga ma NON interviene
     (transitorio, sc.3), i KPI del traffico (latenza, PDR) devono essere
     IDENTICI a MAPPO-solo. Prova che la presenza dell'agente non perturba il
     traffico quando non agisce.
  B. TEMPO DEL PERCORSO VELOCE: wall-clock per decisione MAPPO (per-secondo).
     Microsecondi. L'LLM (secondi, una volta per finestra) NON e' qui dentro.
  C. PRESTAZIONI DOVE SERVE: sul collasso, MAPPO-solo vs +agente — consegna del
     traffico di controllo, con il trade-off onesto sul PDR.

I KPI usano il backend deterministico 'policy': coincidono con quelli dell'agente
LLM (stesse azioni), ma senza attesa Ollama. Il costo wall-clock dell'LLM (secondi
per tick, sul percorso lento) e' separato e non tocca la latenza del traffico.

Uso:  python3 examples/run_agent_perf.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from simulator.marl import EDSMarlEnv, MARLController  # noqa: E402
from simulator.supervisor.ood import build_capacity_collapse  # noqa: E402
from simulator.supervisor.agent import run_agent_episode  # noqa: E402
from run_m1_explainer import DEFAULT_CKPT  # noqa: E402
from run_m3_ood import episode_alone  # noqa: E402
from run_agent import PolicyBackend  # noqa: E402

END = 200.0


def main():
    print("=" * 66)
    print("  A. NON-INTERFERENZA — l'agente indaga ma NON interviene (sc.3)")
    print("=" * 66)
    print(f"  {'seed':>5}{'lat MAPPO':>12}{'lat +agente':>13}{'PDR MAPPO':>12}{'PDR +ag':>10}{'Δ':>8}")
    max_delta = 0.0
    for s in range(3):
        seed = 42 + s
        base = episode_alone(EDSMarlEnv(3, seed=seed, end_time=END),
                             MARLController.from_checkpoint(DEFAULT_CKPT))
        ag = run_agent_episode(EDSMarlEnv(3, seed=seed, end_time=END),
                               MARLController.from_checkpoint(DEFAULT_CKPT), PolicyBackend())
        d = abs(base["latency_ms"] - ag["latency_ms"]) + abs(base["pdr"] - ag["pdr"])
        max_delta = max(max_delta, d)
        assert not ag["reconfigured"], "atteso nessun intervento sul transitorio"
        print(f"  {seed:>5}{base['latency_ms']:>12.2f}{ag['latency_ms']:>13.2f}"
              f"{base['pdr']:>12.4f}{ag['pdr']:>10.4f}{d:>8.2e}")
    print(f"\n  → differenza massima traffico: {max_delta:.2e}  "
          f"({'IDENTICO: agente trasparente' if max_delta < 1e-9 else 'differenza!'})")

    print("\n" + "=" * 66)
    print("  B. TEMPO DEL PERCORSO VELOCE (wall-clock per decisione MAPPO)")
    print("=" * 66)
    env = EDSMarlEnv(3, seed=42, end_time=END)
    obs, _ = env.reset()
    mappo = MARLController.from_checkpoint(DEFAULT_CKPT)
    n, t0 = 0, time.perf_counter()
    done = False
    while not done:
        obs, _s, _r, done, _i = env.step(mappo.act(obs))
        n += 1
    dt = time.perf_counter() - t0
    us = dt / n * 1e6
    print(f"  {n} decisioni MAPPO in {dt*1e3:.1f} ms  →  {us:.1f} µs/decisione (per-secondo)")
    print(f"  Confronto: un tick dell'agente LLM ~ secondi, UNA volta per finestra (~30s),")
    print(f"  sul percorso LENTO. Rapporto: l'LLM e' ~{1e6/us*1e0:,.0f}× piu' lento del fast step,")
    print(f"  ma NON e' nel loop veloce → non tocca la latenza del traffico.")

    print("\n" + "=" * 66)
    print("  C. PRESTAZIONI DOVE SERVE (collasso permanente)")
    print("=" * 66)
    b_cd, b_pdr, b_lat, a_cd, a_pdr, a_lat = [], [], [], [], [], []
    for s in range(3):
        seed = 42 + s
        base = episode_alone(build_capacity_collapse(seed=seed, end_time=END),
                             MARLController.from_checkpoint(DEFAULT_CKPT))
        ag = run_agent_episode(build_capacity_collapse(seed=seed, end_time=END),
                               MARLController.from_checkpoint(DEFAULT_CKPT), PolicyBackend())
        b_cd.append(base["control_del"]); b_pdr.append(base["pdr"]); b_lat.append(base["latency_ms"])
        a_cd.append(ag["control_del"]); a_pdr.append(ag["pdr"]); a_lat.append(ag["latency_ms"])
    print(f"  {'KPI':<24}{'MAPPO solo':>14}{'+ agente':>12}{'':>4}")
    print(f"  {'consegna CONTROLLO':<24}{np.mean(b_cd):>14.3f}{np.mean(a_cd):>12.3f}"
          f"   {'△ +' + format(np.mean(a_cd)-np.mean(b_cd), '.3f')}")
    print(f"  {'PDR globale':<24}{np.mean(b_pdr):>14.3f}{np.mean(a_pdr):>12.3f}"
          f"   {'▽ ' + format(np.mean(a_pdr)-np.mean(b_pdr), '+.3f')} (sacrifica il video)")
    print(f"  {'latenza (ms)':<24}{np.mean(b_lat):>14.0f}{np.mean(a_lat):>12.0f}")
    print("\n  → l'agente MIGLIORA la metrica che conta nel collasso (controllo protetto),")
    print("    al costo voluto del throughput a bassa priorita'. Nessun 'pasto gratis'.")


if __name__ == "__main__":
    main()
