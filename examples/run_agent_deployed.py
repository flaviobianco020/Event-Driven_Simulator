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
from simulator.supervisor.controller import SupervisorController, SYSTEM_PROMPT  # noqa: E402
from simulator.supervisor import OllamaBackend, MockBackend  # noqa: E402
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
        self.llm_latencies = []              # latenze reali d'inferenza (s)
        self.explanations = []               # spiegazioni prodotte dall'SLM

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

    def _infer(self, backend, m, cap, nom, llm_latency_s):
        """Costo d'inferenza REALE: l'SLM spiega la situazione (suo ruolo genuino).
        Ritorna (latenza_s, spiegazione). Se backend None → sleep emulata."""
        t0 = time.perf_counter()
        if backend is None:
            time.sleep(llm_latency_s)
            return llm_latency_s, "[emulato]"
        health = SupervisorController.assess(m)["health"]
        user = (f"Situazione (gia' valutata): sistema {health}. Capacita' del collo di "
                f"bottiglia {cap:.0f} (nominale {nom:.0f}); PDR {m.get('pdr',0):.3f}, "
                f"drop {m.get('drop_rate',0):.3f}. Spiega in UNA frase se e' un guasto "
                f"strutturale (capacita' persa) o un eccesso di domanda (capacita' normale), "
                f"e l'azione opportuna.")
        try:
            raw = backend.decide({"metrics": m}, SYSTEM_PROMPT, user)
            expl = raw.get("justification", "")
        except Exception as exc:  # noqa: BLE001
            expl = f"[backend errore: {exc}]"
        return time.perf_counter() - t0, expl

    # ── thread AGENTE (percorso lento, asincrono) ─────────────────────────────────
    def agent_loop(self, period_wall_s, backend, llm_latency_s):
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
                lat, expl = self._infer(backend, m, cap, nom, llm_latency_s)  # il veloce intanto gira
                self.llm_latencies.append(lat)
                self.explanations.append((t, expl))
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
    ap.add_argument("--tick-wall", type=float, default=None,
                    help="wall-clock per tick veloce (s); default 0.15 con ollama, 0.012 emulato")
    ap.add_argument("--agent-period", type=float, default=0.25, help="wall-clock fra check agente (s)")
    ap.add_argument("--backend", choices=["ollama", "emulated"], default="ollama",
                    help="ollama = inferenza SLM REALE sul thread agente; emulated = sleep")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--llm-latency", type=float, default=1.0, help="costo emulato se --backend emulated (s)")
    ap.add_argument("--duration", type=float, default=120.0, help="durata degrado transitorio (s)")
    args = ap.parse_args()

    env = build_transient_degradation(seed=42, end_time=300.0, drop_to=2.0,
                                      onset=30.0, duration=args.duration)
    dep = Deployment(env, MARLController.from_checkpoint(DEFAULT_CKPT))
    backend = OllamaBackend(model=args.model, timeout=60.0) if args.backend == "ollama" else None
    # clock: con Ollama la sim deve durare piu' dell'inferenza reale (secondi)
    tick_wall = args.tick_wall if args.tick_wall is not None else (0.15 if backend else 0.012)

    label = f"Ollama {args.model} (inferenza REALE)" if backend else f"emulato ({args.llm_latency*1e3:.0f}ms)"
    print(f"  DEPLOYMENT — loop veloce (1s/tick @ {tick_wall*1e3:.0f}ms wall) + agente async")
    print(f"  backend agente: {label}")
    print(f"  degrado transitorio: link 10→2 a t=30s, recupero a t={30+args.duration:.0f}s")
    if backend:   # scalda il modello: la 1a chiamata carica i pesi in RAM (non e' inferenza)
        print("  scaldo il modello (caricamento pesi, escluso dalla misura)...", flush=True)
        try:
            backend.decide({}, SYSTEM_PROMPT, "Rispondi 'ok'.")
        except Exception as exc:  # noqa: BLE001
            print(f"  ATTENZIONE: warmup fallito ({exc}) — Ollama attivo? modello scaricato?")
    print()
    t0 = time.perf_counter()
    ft = threading.Thread(target=dep.fast_loop, args=(tick_wall,))
    at = threading.Thread(target=dep.agent_loop, args=(args.agent_period, backend, args.llm_latency),
                          daemon=True)
    ft.start(); at.start()
    ft.join(); at.join(timeout=30.0)      # attende l'eventuale inferenza in volo
    wall = time.perf_counter() - t0

    # 1. NON-BLOCCO: intervallo fra tick veloci vs latenza d'inferenza REALE
    dt = np.diff(dep.tick_wall)
    real_lat = max(dep.llm_latencies) if dep.llm_latencies else 0.0
    print("  ── proprieta' 1: NON-BLOCCO ──")
    print(f"  {len(dep.tick_wall)} tick veloci in {wall:.1f}s wall")
    print(f"  intervallo fra tick: medio {dt.mean()*1e3:.1f}ms  max {dt.max()*1e3:.1f}ms")
    print(f"  latenza d'inferenza SLM misurata: max {real_lat*1e3:.0f}ms "
          f"({len(dep.llm_latencies)} chiamate)")
    ok = real_lat > 0 and dt.max() < real_lat * 0.5
    print(f"  → max intervallo tick ({dt.max()*1e3:.0f}ms) << inferenza SLM ({real_lat*1e3:.0f}ms): "
          f"{'il loop veloce NON si ferma mentre l’SLM pensa ✓' if ok else '(controlla)'}")

    # 2. INTERVENTO/RITIRO + SPIEGAZIONI dal vivo
    print("\n  ── proprieta' 2: INTERVENTO/RITIRO E SPIEGAZIONE DAL VIVO ──")
    for wclk, tsim, msg in dep.log:
        print(f"  [sim t={tsim:5.0f}s | wall {wclk-t0:4.1f}s]  {msg}")
    for tsim, expl in dep.explanations:
        print(f"    [SLM @ sim t={tsim:.0f}s] «{expl}»")

    s = env.summary()
    print(f"\n  KPI finali: PDR {s['pdr']:.3f}  latenza {s['latency']*1e3:.0f}ms  drop {s['dropped']}")
    print("  (override e spiegazione prodotti in tempo reale sul flusso vivo, LLM fuori dal loop veloce)")


if __name__ == "__main__":
    main()
