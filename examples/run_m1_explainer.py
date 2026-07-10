#!/usr/bin/env python3
"""
run_m1_explainer.py — Fase 4, Milestone M1: explainer READ-ONLY sul loop MAPPO reale.

Fa girare la policy MAPPO canonica (Fase 3, checkpoint JSON) su uno scenario del
simulatore e, ogni finestra lenta (~30 s simulati), interroga il supervisore LLM
con le metriche aggregate della finestra + la traiettoria di stati. Il supervisore
produce azione suggerita + giustificazione in linguaggio naturale.

M1 = ZERO autorita' di controllo:
  * Guardrail con kill_switch=True → ogni override e' rifiutato a monte;
  * il loop, comunque, non applica MAI l'override al percorso veloce.
Il deliverable e' il LOG DI SPIEGABILITA': cio' che la policy neurale da sola
non puo' dare. (L'autorita' di controllo arriva in M2; la valutazione OOD in M3.)

Uso:
    python3 examples/run_m1_explainer.py                          # mock, scenario 3
    python3 examples/run_m1_explainer.py --scenario 5 --backend ollama \
        --model qwen2.5:3b-instruct
    python3 examples/run_m1_explainer.py --backend anthropic      # Haiku (tetto)

Output: log a console + JSON (default logs/m1_scenario<k>_<backend>.json) con
una entry per tick lento: finestra metriche, stati recenti, azione, giustificazione.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from simulator.marl import EDSMarlEnv, MARLController  # noqa: E402
from simulator.network.congestion import CongestionState  # noqa: E402
from simulator.supervisor import (  # noqa: E402
    SupervisorController, Guardrail, MockBackend, OllamaBackend, AnthropicBackend,
)

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..",
                            "checkpoints", "mappo_best_stab.json")

# capacita' del collo di bottiglia negli scenari canonici (pkt/s) — per link_util
BOTTLENECK_CAPACITY = 10.0


def make_backend(name: str, model: str):
    if name == "ollama":
        return OllamaBackend(model=model)
    if name == "anthropic":
        return AnthropicBackend()
    return MockBackend()


def _window_metrics(acc: dict, window_s: float, compression: float) -> dict:
    """Aggrega i deltas per-tick accumulati in metriche di finestra per il supervisore."""
    gen, dl, drop, lat = acc["gen"], acc["del"], acc["drop"], acc["lat"]
    pdr = dl / gen if gen > 0 else 1.0
    return {
        "pdr": pdr,
        "latency_ms": (lat / dl * 1000.0) if dl > 0 else 0.0,
        "drop_rate": min(drop / gen, 1.0) if gen > 0 else 0.0,
        # ratio cumulativa dal MetricsEngine (per-finestra richiederebbe byte windowed)
        "compression": compression,
        "link_util": min((dl / window_s) / BOTTLENECK_CAPACITY, 1.0),
        "transitions": acc["trans"],
    }


def run_m1(scenario: int, seed: int, ckpt: str, backend_name: str = "mock",
           model: str = "qwen2.5:3b-instruct", window_s: float = 30.0,
           end_time: float | None = None, verbose: bool = True) -> dict:
    """
    Esegue M1 su uno scenario. Ritorna {"supervisor_log", "summary", "n_slow_ticks"}.
    Importabile dai test (verbose=False, end_time corto).
    """
    mappo = MARLController.from_checkpoint(ckpt)          # percorso veloce (greedy)
    supervisor = SupervisorController(
        backend=make_backend(backend_name, model),
        guardrail=Guardrail(kill_switch=True),            # M1: read-only garantito
        tick_interval=window_s,
    )

    env = EDSMarlEnv(scenario, seed=seed, end_time=end_time)
    obs, _state = env.reset()

    acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}
    traj: list[int] = []          # traiettoria stati (valori 0-4), tutta la run
    done = False
    n_slow = 0

    if verbose:
        print(f"  M1 explainer — scenario {scenario}, backend {supervisor.backend.name}, "
              f"finestra {window_s:.0f}s, checkpoint {os.path.basename(ckpt)}")
        print("-" * 78)

    while not done:
        actions = mappo.act(obs)                          # MAPPO decide (ogni 1 s)
        obs, _state, _r, done, info = env.step(actions)

        d = info["deltas"]
        acc["gen"] += d["gen"]; acc["del"] += d["del"]
        acc["drop"] += d["drop"]; acc["lat"] += d["lat"]
        acc["trans"] += info["transitions"]
        traj.append(CongestionState[info["states"][0]].value)

        # tick lento del supervisore a fine finestra (e a fine episodio)
        if info["t"] % window_s < 1e-9 or done:
            metrics = _window_metrics(acc, window_s,
                                      env.metrics.collect_compression_ratio())
            verdict = supervisor.tick(info["t"], metrics, traj)
            n_slow += 1
            # M1: mai applicare l'override (e il kill switch gia' lo blocca)
            assert supervisor.current_override(info["t"]) is None, \
                "M1 e' read-only: nessun override deve mai essere attivo"

            if verbose:
                e = supervisor.log.entries[-1]
                print(f"[t={info['t']:5.0f}s]  PDR={metrics['pdr']:.3f}  "
                      f"lat={metrics['latency_ms']:6.0f}ms  drop={metrics['drop_rate']:.3f}  "
                      f"stati(ultimi 10)={traj[-10:]}")
                print(f"    azione suggerita: {e['action']}"
                      + ("  [BLOCCATA: read-only]" if e['action'] == 'override_state' else ""))
                print(f"    «{e['justification']}»")
                print("-" * 78)
            acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}

    summary = env.summary()
    if verbose:
        print(f"\n  Episodio: PDR={summary['pdr']:.3f}  lat={summary['latency']*1000:.0f}ms  "
              f"drop={summary['dropped']}  trans={summary['transitions']}")
        print(f"  {n_slow} tick lenti, {len(supervisor.log.entries)} decisioni loggate.")

    return {"supervisor_log": supervisor.log.entries, "summary": summary,
            "n_slow_ticks": n_slow, "backend": supervisor.backend.name,
            "scenario": scenario, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=3, choices=range(1, 7),
                    help="scenario canonico (default 3: bandwidth degradation)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--backend", choices=["mock", "ollama", "anthropic"], default="mock")
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--window", type=float, default=30.0, help="finestra tick lento (s)")
    ap.add_argument("--out", default=None, help="file JSON di log (default logs/m1_...)")
    args = ap.parse_args()

    result = run_m1(args.scenario, args.seed, args.ckpt,
                    backend_name=args.backend, model=args.model,
                    window_s=args.window)

    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "logs",
        f"m1_scenario{args.scenario}_{args.backend}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"  log salvato: {out}")


if __name__ == "__main__":
    main()
