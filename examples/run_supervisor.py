#!/usr/bin/env python3
"""
run_supervisor.py — Demo dello scheletro Fase 4 (supervisore LLM sul percorso lento).

Fa girare il SupervisorController su una traccia di metriche simulata, mostrando
il ciclo: metriche → decisione LLM → guardrail → stato imposto + giustificazione.
Con il MockBackend (default) gira SENZA modelli installati.

Uso:
    python3 examples/run_supervisor.py                 # MockBackend (nessuna dipendenza)
    python3 examples/run_supervisor.py --backend ollama --model qwen2.5:3b
    python3 examples/run_supervisor.py --backend anthropic   # Claude Haiku (tetto)

M1 (questa demo): explainer read-only + override vincolato dal guardrail. La
valutazione OOD (M3) confrontera' MAPPO-solo vs MAPPO+supervisore su scenari
non visti in addestramento.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.supervisor import (  # noqa: E402
    SupervisorController, Guardrail, MockBackend, OllamaBackend, AnthropicBackend,
)

# Traccia di metriche di esempio: finestra tranquilla → anomalia (drop) → recupero.
# Simula cio' che il driver reale leggerebbe dal MetricsEngine / da tc.
TRACE = [
    #  t     pdr   lat_ms  drop  compr  util  trans   stati recenti
    (0.0,  {"pdr": 0.98, "latency_ms": 160, "drop_rate": 0.01, "compression": 1.4, "link_util": 0.7, "transitions": 2}, [2, 2, 3, 3]),
    (30.0, {"pdr": 0.97, "latency_ms": 180, "drop_rate": 0.02, "compression": 1.5, "link_util": 0.8, "transitions": 3}, [3, 3, 3, 2]),
    (60.0, {"pdr": 0.71, "latency_ms": 950, "drop_rate": 0.28, "compression": 1.1, "link_util": 0.99, "transitions": 41}, [1, 3, 0, 2]),  # anomalia OOD: oscilla + drop
    (90.0, {"pdr": 0.90, "latency_ms": 400, "drop_rate": 0.06, "compression": 1.5, "link_util": 0.85, "transitions": 8}, [3, 3, 3, 3]),   # recupero
]


def make_backend(args):
    if args.backend == "ollama":
        return OllamaBackend(model=args.model)
    if args.backend == "anthropic":
        return AnthropicBackend()
    return MockBackend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "ollama", "anthropic"], default="mock")
    ap.add_argument("--model", default="qwen2.5:3b")
    args = ap.parse_args()

    ctrl = SupervisorController(backend=make_backend(args), guardrail=Guardrail(),
                               tick_interval=30.0)
    print(f"  Supervisore Fase 4 — backend: {ctrl.backend.name}")
    print("-" * 78)
    for t, metrics, states in TRACE:
        verdict = ctrl.tick(t, metrics, states)
        imposed = ctrl.current_override(t)
        d = ctrl.log.entries[-1]
        print(f"[t={t:5.0f}s]  azione={d['action']:<14}  approvato={verdict.approved}")
        print(f"            stato imposto al percorso veloce: {imposed}")
        print(f"            «{d['justification']}»")
        print(f"            guardrail: {verdict.reason}")
        print("-" * 78)

    print(f"\n  {len(ctrl.log.entries)} decisioni registrate (log spiegabilita').")


if __name__ == "__main__":
    main()
