#!/usr/bin/env python3
"""
run_m2_supervisor.py — Fase 4, Milestone M2: il supervisore ACQUISISCE il controllo.

Differenza da M1 (explainer read-only): il kill switch e' spento e l'override
approvato dal guardrail viene APPLICATO al percorso veloce. Meccanica non
invasiva (zero modifiche a marl/env.py): l'override assoluto ("stato k") viene
tradotto dal driver in azioni relative {ESCALATE, MAINTAIN, DEESCALATE} che
sostituiscono l'azione MAPPO finche' la finestra di override e' attiva; alla
scadenza (o revoca) MAPPO riprende il controllo.

Sicurezza (invariata dal piano):
  * decisione DETERMINISTICA (assess: PDR/drop a soglia) — l'LLM spiega soltanto;
  * override limitato nel tempo (max 120 s) e reversibile;
  * REVOCA: se durante un override il PDR di finestra scende sotto il floor del
    guardrail, l'override viene revocato e MAPPO riprende subito;
  * --kill-switch riporta al comportamento M1 (read-only).

Uso:
    python3 examples/run_m2_supervisor.py --scenario 3                # mock
    python3 examples/run_m2_supervisor.py --scenario 3 --backend ollama
    python3 examples/run_m2_supervisor.py --scenario 3 --compare      # solo-MAPPO vs +supervisore
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from simulator.marl import (  # noqa: E402
    EDSMarlEnv, MARLController, ESCALATE, MAINTAIN, DEESCALATE,
)
from simulator.network.congestion import CongestionState  # noqa: E402
from simulator.supervisor import (  # noqa: E402
    SupervisorController, Guardrail,
)
from run_m1_explainer import make_backend, _window_metrics, DEFAULT_CKPT  # noqa: E402


def override_to_action(current_state: int, target_state: int) -> int:
    """
    Traduce un override assoluto (porta il nodo allo stato k) nell'azione
    relativa del percorso veloce. Un passo per tick: la macchina a stati
    converge al target in <=4 tick e poi lo tiene (MAINTAIN).
    """
    if current_state < target_state:
        return ESCALATE
    if current_state > target_state:
        return DEESCALATE
    return MAINTAIN


def maybe_revoke(guardrail: Guardrail, window_metrics: dict, t: float) -> bool:
    """
    Revoca dell'override attivo se il KPI critico e' peggiorato sotto il floor:
    l'intervento non sta aiutando (o peggiora) → MAPPO riprende subito.
    """
    if guardrail.active_state(t) is None:
        return False
    if window_metrics.get("pdr", 1.0) < guardrail.pdr_floor:
        guardrail.revoke()
        return True
    return False


def run_m2(scenario: int, seed: int, ckpt: str, backend_name: str = "mock",
           model: str = "qwen2.5:3b", window_s: float = 30.0,
           end_time: float | None = None, kill_switch: bool = False,
           verbose: bool = True) -> dict:
    """
    Esegue M2. Ritorna dict con summary episodio, log supervisore, statistiche
    override (finestre, step sotto override, revoche). Importabile dai test.
    """
    mappo = MARLController.from_checkpoint(ckpt)
    guardrail = Guardrail(kill_switch=kill_switch)
    supervisor = SupervisorController(
        backend=make_backend(backend_name, model),
        guardrail=guardrail, tick_interval=window_s,
    )

    env = EDSMarlEnv(scenario, seed=seed, end_time=end_time)
    obs, _state = env.reset()

    acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}
    traj: list[int] = []
    done = False
    override_steps = 0        # step (1 s) in cui l'azione e' del supervisore
    override_windows = 0      # finestre con override approvato
    revocations = 0

    if verbose:
        mode = "KILL SWITCH (read-only)" if kill_switch else "CONTROLLO ATTIVO"
        print(f"  M2 supervisore — scenario {scenario}, backend {supervisor.backend.name}, "
              f"modo: {mode}")
        print("-" * 78)

    while not done:
        cur = env._nodes[0].state_machine.current_state.value
        target = supervisor.current_override(env.t)
        if target is not None:
            actions = [override_to_action(cur, target)]     # supervisore comanda
            override_steps += 1
        else:
            actions = mappo.act(obs)                        # MAPPO comanda

        obs, _state, _r, done, info = env.step(actions)

        d = info["deltas"]
        acc["gen"] += d["gen"]; acc["del"] += d["del"]
        acc["drop"] += d["drop"]; acc["lat"] += d["lat"]
        acc["trans"] += info["transitions"]
        traj.append(CongestionState[info["states"][0]].value)

        if info["t"] % window_s < 1e-9 or done:
            metrics = _window_metrics(acc, window_s,
                                      env.metrics.collect_compression_ratio())
            # 1. revoca se l'override attivo non sta proteggendo il KPI
            if maybe_revoke(guardrail, metrics, info["t"]):
                revocations += 1
                if verbose:
                    print(f"[t={info['t']:5.0f}s]  ⚠ REVOCA override (PDR {metrics['pdr']:.3f} "
                          f"< floor {guardrail.pdr_floor}) — MAPPO riprende")
            # 2. tick del supervisore (decisione deterministica + spiegazione LLM)
            verdict = supervisor.tick(info["t"], metrics, traj)
            if verdict.approved and verdict.effective_state is not None:
                override_windows += 1
            if verbose:
                e = supervisor.log.entries[-1]
                imposed = supervisor.current_override(info["t"])
                print(f"[t={info['t']:5.0f}s]  PDR={metrics['pdr']:.3f}  "
                      f"drop={metrics['drop_rate']:.3f}  stati={traj[-6:]}")
                print(f"    azione={e['action']}  applicata={verdict.approved and e['action']=='override_state'}"
                      f"  stato_imposto={imposed}")
                print(f"    «{e['justification']}»")
                print("-" * 78)
            acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}

    summary = env.summary()
    if verbose:
        print(f"\n  Episodio: PDR={summary['pdr']:.3f}  lat={summary['latency']*1000:.0f}ms  "
              f"drop={summary['dropped']}  trans={summary['transitions']}")
        print(f"  Override: {override_windows} finestre, {override_steps} step "
              f"sotto controllo supervisore, {revocations} revoche.")

    return {"summary": summary, "supervisor_log": supervisor.log.entries,
            "override_steps": override_steps, "override_windows": override_windows,
            "revocations": revocations, "trajectory": traj,
            "scenario": scenario, "seed": seed}


def run_mappo_alone(scenario: int, seed: int, ckpt: str,
                    end_time: float | None = None) -> dict:
    """Baseline: stesso episodio senza supervisore (Fase 3 pura)."""
    mappo = MARLController.from_checkpoint(ckpt)
    env = EDSMarlEnv(scenario, seed=seed, end_time=end_time)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _s, _r, done, _i = env.step(mappo.act(obs))
    return env.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=3, choices=range(1, 7))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--backend", choices=["mock", "ollama", "anthropic"], default="mock")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--kill-switch", action="store_true",
                    help="modalita' M1: supervisore read-only")
    ap.add_argument("--compare", action="store_true",
                    help="confronta MAPPO-solo vs MAPPO+supervisore (stesso seed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run_m2(args.scenario, args.seed, args.ckpt,
                    backend_name=args.backend, model=args.model,
                    window_s=args.window, kill_switch=args.kill_switch)

    if args.compare:
        base = run_mappo_alone(args.scenario, args.seed, args.ckpt)
        sup = result["summary"]
        print("\n" + "=" * 60)
        print(f"  CONFRONTO (scenario {args.scenario}, seed {args.seed})")
        print("=" * 60)
        print(f"  {'KPI':<16}{'MAPPO solo':>14}{'+ supervisore':>15}")
        print(f"  {'PDR':<16}{base['pdr']:>14.3f}{sup['pdr']:>15.3f}")
        print(f"  {'latenza (ms)':<16}{base['latency']*1000:>14.0f}{sup['latency']*1000:>15.0f}")
        print(f"  {'drop':<16}{base['dropped']:>14}{sup['dropped']:>15}")
        print(f"  {'transizioni':<16}{base['transitions']:>14}{sup['transitions']:>15}")
        print("=" * 60)

    out = args.out or os.path.join(os.path.dirname(__file__), "..", "logs",
                                   f"m2_scenario{args.scenario}_{args.backend}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"  log salvato: {out}")


if __name__ == "__main__":
    main()
