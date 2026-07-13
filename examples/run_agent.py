#!/usr/bin/env python3
"""
run_agent.py — Fase 4b: l'agente batte il floor di osservabilita' INDAGANDO.

Test di discriminazione a livello AGENTE. Due scenari che a t=60 sembrano
identici ma divergono nel tempo:
  - capacity_collapse (guasto PERMANENTE): l'agente deve diagnosticare
    'collasso_permanente' e intervenire (proteggere le priorita' alte).
  - scenario 3 (degrado TRANSITORIO che recupera): l'agente deve diagnosticare
    'transitorio' e NON intervenire.

La differenza col decisore one-shot (Fase 4a): l'agente puo' usare
wait_and_observe per VEDERE se il sistema recupera — cioe' AGISCE per procurarsi
l'informazione che al one-shot mancava. Risolve l'ambiguita' raccogliendo, non
indovinando.

Backend:
  --backend policy   agente deterministico (indaga→aspetta→decidi): dimostra il
                     MECCANISMO senza LLM.
  --backend ollama --model qwen2.5:3b   l'agente LLM vero (sceglie i tool da solo).

Uso:
    python3 examples/run_agent.py --backend policy
    python3 examples/run_agent.py --backend ollama --model qwen2.5:3b --verbose
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from simulator.marl import EDSMarlEnv, MARLController  # noqa: E402
from simulator.supervisor.ood import build_capacity_collapse  # noqa: E402
from simulator.supervisor.agent import run_agent_episode  # noqa: E402
from simulator.supervisor import OllamaBackend, AnthropicBackend  # noqa: E402
from run_m1_explainer import DEFAULT_CKPT  # noqa: E402
from run_m3_ood import episode_alone  # noqa: E402

END = 200.0   # episodio piu' lungo: lascia all'agente il tempo di INDAGARE


class PolicyBackend:
    """Agente deterministico: indaga (wait) → se resta critico interviene, altrimenti
    conclude transitorio. Dimostra che RACCOGLIERE info risolve l'ambiguita'."""
    name = "policy"

    def __init__(self):
        self.waited = False
        self.reconfigured = False

    def decide(self, context, system, user, schema=None):
        health = context["obs"].get("health", "?")
        if not self.waited:
            self.waited = True
            return {"tool": "wait_and_observe", "n_windows": 2,
                    "reasoning": "indago: aspetto per vedere se recupera da solo."}
        if health == "CRITICO" and not self.reconfigured:
            self.reconfigured = True
            return {"tool": "trigger_reconfigure",
                    "reasoning": "resta critico dopo l'attesa → collasso permanente."}
        if self.reconfigured:
            return {"tool": "conclude", "diagnosis": "collasso_permanente",
                    "reasoning": "protezione priorita' alte attivata."}
        return {"tool": "conclude", "diagnosis": "transitorio",
                "reasoning": "tornato sano dopo l'attesa → nessun intervento."}


def make_backend(name, model, timeout):
    if name == "policy":
        return PolicyBackend()
    if name == "ollama":
        return OllamaBackend(model=model, timeout=timeout)
    if name == "anthropic":
        return AnthropicBackend()
    raise ValueError(name)


SCENARIOS = {
    "collasso permanente": (lambda seed: build_capacity_collapse(seed=seed, end_time=END),
                            "collasso_permanente", True),   # atteso: diagnosi, deve intervenire
    "transitorio (sc.3)": (lambda seed: EDSMarlEnv(3, seed=seed, end_time=END),
                           "transitorio", False),           # atteso: diagnosi, NON intervenire
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["policy", "ollama", "anthropic"], default="policy")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"  AGENTE — test di discriminazione (backend {args.backend}, {args.seeds} seed)")
    print("  L'agente puo' INDAGARE (wait_and_observe) prima di decidere.\n")

    for label, (make_env, expected_diag, should_act) in SCENARIOS.items():
        diags, acts, cds, pdrs, correct, self_conc = [], [], [], [], 0, 0
        for s in range(args.seeds):
            seed = 42 + s
            backend = make_backend(args.backend, args.model, args.timeout)  # stato pulito per episodio
            mappo = MARLController.from_checkpoint(args.ckpt)
            if args.verbose and s == 0:
                print(f"  ▶ {label} (seed {seed}):")
            r = run_agent_episode(make_env(seed), mappo, backend,
                                  verbose=args.verbose and s == 0)
            diags.append(r["diagnosis"]); acts.append(r["reconfigured"])
            pdrs.append(r["pdr"]); cds.append(r.get("control_del"))
            self_conc += int(r.get("self_concluded", False))
            ok = (r["diagnosis"] == expected_diag) and (r["reconfigured"] == should_act)
            correct += ok

        cd_txt = (f"control_del={np.mean([c for c in cds if c is not None]):.3f}  "
                  if any(c is not None for c in cds) else "")
        maj_diag = max(set(diags), key=diags.count)
        print(f"  ── {label} ──")
        print(f"     atteso: diagnosi={expected_diag}, intervieni={should_act}")
        print(f"     agente: diagnosi={maj_diag}, intervenuto={sum(acts)}/{len(acts)}  "
              f"{cd_txt}PDR={np.mean(pdrs):.3f}")
        print(f"     concluso da solo (non scaffold): {self_conc}/{args.seeds}")
        print(f"     CORRETTO: {correct}/{args.seeds}  "
              f"{'✓' if correct == args.seeds else '✗'}\n")


if __name__ == "__main__":
    main()
