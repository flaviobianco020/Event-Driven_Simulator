#!/usr/bin/env python3
"""
run_agent_deployed.py — Fase 4b: l'agente COME SE FOSSE IN DEPLOYMENT.

Differenza dai runner batch: li' l'agente GUIDAVA la simulazione (wait_and_observe
AVANZAVA il tempo). In deployment il percorso veloce (MAPPO) gira DA SOLO in tempo
reale su un thread; l'agente vive su un thread SEPARATO, osserva metriche DAL VIVO
e posta un override in modo asincrono — senza mai bloccare il loop veloce.

Questo harness emula il deployment in-process:
  - Thread VELOCE: MAPPO decide ogni 1 s (simulato), a passo di wall-clock (tick_wall),
    applica l'override condiviso se presente, aggiorna le metriche di finestra vive.
  - Thread AGENTE: dorme sul percorso lento; alla prima finestra critica emula il
    COSTO D'INFERENZA LLM (una sleep di ~1 s durante la quale il loop veloce continua
    a girare), poi decide via sensore-causa (capacita' bassa → strutturale → protegge;
    normale → domanda, niente intervento) e monitora per RITIRARE al recupero.
  - Handoff thread-safe: solo il thread veloce tocca l'env; l'agente legge copie e
    scrive override_target sotto lock.

Prova due proprieta' di deployment:
  1. NON-BLOCCO: l'intervallo fra i tick veloci resta ~tick_wall anche mentre l'agente
     "pensa" per ~1 s → l'LLM non e' nel loop veloce.
  2. INTERVENTO/RITIRO DAL VIVO: override applicato e ritirato in tempo reale sul
     flusso di metriche, non su un episodio pre-registrato.

Uso:  python3 examples/run_agent_deployed.py
"""
import sys, os, time, threading, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from simulator.marl import (MARLController, ESCALATE, MAINTAIN, DEESCALATE)  # noqa: E402
from simulator.network.congestion import CongestionState  # noqa: E402
from simulator.supervisor.controller import SupervisorController  # noqa: E402
from simulator.supervisor.ood import build_transient_degradation  # noqa: E402
from run_m1_explainer import _window_metrics, DEFAULT_CKPT  # noqa: E402


def _to_action(cur, tgt):
    return ESCALATE if cur < tgt else (DEESCALATE if cur > tgt else MAINTAIN)


class Deployment:
    """Stato condiviso fra thread veloce e thread agente."""
    def __init__(self, env, mappo, window_s=30.0):
        self.env, self.mappo, self.window_s = env, mappo, window_s
        self.lock = threading.Lock()
        self.override_target = None          # scritto dall'agente, letto dal veloce
        self.metrics = None                  # ultime metriche di finestra (vive)
        self.capacity = None
        self.nominal = 0.0
        self.done = False
        self.tick_wall = []                  # timestamp wall di ogni tick veloce
        self.log = []                        # azioni dell'agente (wall, sim, testo)

    # ── thread VELOCE ─────────────────────────────────────────────────────────────
    def fast_loop(self, tick_wall_s):
        obs, _ = self.env.reset()
        self.nominal = self.env.topology.get_link("router", "dst").capacity
        acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}
        while not self.done:
            cur = self.env._nodes[0].state_machine.current_state.value
            with self.lock:
                tgt = self.override_target
            actions = [_to_action(cur, tgt)] if tgt is not None else self.mappo.act(obs)
            obs, _s, _r, done, info = self.env.step(actions)
            d = info["deltas"]
            acc["gen"] += d["gen"]; acc["del"] += d["del"]
            acc["drop"] += d["drop"]; acc["lat"] += d["lat"]; acc["trans"] += info["transitions"]
            if info["t"] % self.window_s < 1e-9:
                m = _window_metrics(acc, self.window_s, self.env.metrics.collect_compression_ratio())
                with self.lock:
                    self.metrics = m
                    self.capacity = self.env.topology.get_link("router", "dst").capacity
                acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}
            self.tick_wall.append(time.perf_counter())
            if done:
                with self.lock:
                    self.done = True
                break
            time.sleep(tick_wall_s)           # ritmo di tempo reale

    # ── thread AGENTE (percorso lento, asincrono) ─────────────────────────────────
    def agent_loop(self, period_wall_s, llm_latency_s):
        state = "idle"
        while True:
            time.sleep(period_wall_s)
            with self.lock:
                if self.done:
                    break
                m, cap, nom, t = self.metrics, self.capacity, self.nominal, self.env.t
            if m is None:
                continue
            health = SupervisorController.assess(m)["health"]
            if state == "idle" and health == "CRITICO":
                time.sleep(llm_latency_s)     # COSTO D'INFERENZA: il veloce intanto gira
                if cap is not None and cap < nom - 1e-9:
                    with self.lock:
                        self.override_target = 4
                    self.log.append((time.perf_counter(), t, "intervieni (capacita' bassa → strutturale)"))
                    state = "protecting"
                else:
                    self.log.append((time.perf_counter(), t, "domanda (capacita' normale) → nessun intervento"))
            elif state == "protecting":
                if cap is not None and cap >= nom - 1e-9:   # causa risolta → ritira
                    with self.lock:
                        self.override_target = None
                    self.log.append((time.perf_counter(), t, "RITIRA (capacita' ripristinata)"))
                    state = "idle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick-wall", type=float, default=0.012, help="wall-clock per tick veloce (s)")
    ap.add_argument("--agent-period", type=float, default=0.25, help="wall-clock fra check agente (s)")
    ap.add_argument("--llm-latency", type=float, default=1.0, help="costo d'inferenza LLM emulato (s)")
    ap.add_argument("--duration", type=float, default=120.0, help="durata degrado transitorio (s)")
    args = ap.parse_args()

    env = build_transient_degradation(seed=42, end_time=300.0, drop_to=2.0,
                                      onset=30.0, duration=args.duration)
    dep = Deployment(env, MARLController.from_checkpoint(DEFAULT_CKPT))

    print(f"  DEPLOYMENT — loop veloce (1s/tick @ {args.tick_wall*1e3:.0f}ms wall) + agente async")
    print(f"  degrado transitorio: link 10→2 a t=30s, recupero a t={30+args.duration:.0f}s\n")
    t0 = time.perf_counter()
    ft = threading.Thread(target=dep.fast_loop, args=(args.tick_wall,))
    at = threading.Thread(target=dep.agent_loop, args=(args.agent_period, args.llm_latency), daemon=True)
    ft.start(); at.start()
    ft.join(); at.join(timeout=2.0)
    wall = time.perf_counter() - t0

    # 1. NON-BLOCCO: intervallo fra tick veloci
    dt = np.diff(dep.tick_wall)
    print("  ── proprieta' 1: NON-BLOCCO ──")
    print(f"  {len(dep.tick_wall)} tick veloci in {wall:.1f}s wall")
    print(f"  intervallo fra tick: medio {dt.mean()*1e3:.1f}ms  max {dt.max()*1e3:.1f}ms")
    print(f"  costo d'inferenza agente: {args.llm_latency*1e3:.0f}ms")
    ok = dt.max() < args.llm_latency * 0.5
    print(f"  → max intervallo ({dt.max()*1e3:.0f}ms) << inferenza ({args.llm_latency*1e3:.0f}ms): "
          f"{'il loop veloce NON si ferma mentre l’agente pensa ✓' if ok else 'BLOCCO!'}")

    # 2. INTERVENTO/RITIRO dal vivo
    print("\n  ── proprieta' 2: INTERVENTO E RITIRO DAL VIVO ──")
    for wclk, tsim, msg in dep.log:
        print(f"  [sim t={tsim:5.0f}s | wall {wclk-t0:4.1f}s]  {msg}")

    s = env.summary()
    print(f"\n  KPI finali: PDR {s['pdr']:.3f}  latenza {s['latency']*1e3:.0f}ms  drop {s['dropped']}")
    print("  (override applicato e ritirato in tempo reale sul flusso vivo, non su episodio batch)")


if __name__ == "__main__":
    main()
