#!/usr/bin/env python3
"""
run_m3_escalation.py — Fase 4: l'LLM decide SUL CASO AMBIGUO (System-2 escalation).

Testa l'ipotesi: dove la soglia e' cieca (collasso vs oscillazione, stessi PDR/drop)
un ragionatore che classifica il regime e sceglie fra azioni vagliate RECUPERA il
traffico di controllo che la soglia affossa.

Tre bracci sul capacity_collapse:
  1. MAPPO solo            — nessun supervisore.
  2. soglia (System 1)     — override deterministico (forza stato 3): il
                             comportamento attuale di M2/M3.
  3. escalation (S1 + S2)  — sul caso ambiguo (critico + compressione gia'
                             massima) inoltra all'LLM che classifica e sceglie
                             fra {mantieni, stato 3, stato 4}. Applica la scelta
                             bypassando il PDR floor (l'azione protettiva serve
                             PROPRIO sotto il floor), time-boxed e reversibile.

Backend escalation:
  --backend oracle  → stub che sceglie sempre C (stato 4): prova che la plumbing
                      raggiunge il SOFFITTO (control_del ~1.0). NON e' un risultato
                      di ragionamento, e' il tetto raggiungibile.
  --backend ollama --model qwen2.5:7b  → il test vero: l'LLM ragiona da solo.

Metrica chiave: control_del (consegna del traffico di controllo, priorita' alta).

Uso:
    python3 examples/run_m3_escalation.py --backend oracle --seeds 5
    python3 examples/run_m3_escalation.py --backend ollama --model qwen2.5:7b --seeds 5
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from simulator.marl import (EDSMarlEnv, MARLController,  # noqa: E402
                            ESCALATE, MAINTAIN, DEESCALATE)
from simulator.network.congestion import CongestionState  # noqa: E402
from simulator.supervisor.controller import SupervisorController  # noqa: E402
from simulator.supervisor.ood import build_capacity_collapse  # noqa: E402
from simulator.supervisor import escalation as esc  # noqa: E402
from simulator.supervisor import OllamaBackend, AnthropicBackend, MockBackend  # noqa: E402
from run_m1_explainer import make_backend, _window_metrics, DEFAULT_CKPT  # noqa: E402


def build_backend(name, model, timeout):
    """Come make_backend ma con timeout esplicito per Ollama (7B lento su poca RAM)."""
    if name == "oracle":
        return OracleEscalationBackend()
    if name == "ollama":
        return OllamaBackend(model=model, timeout=timeout)
    if name == "anthropic":
        return AnthropicBackend()
    return MockBackend()
from run_m3_ood import episode_alone, episode_supervised, _kpis  # noqa: E402


class OracleEscalationBackend:
    """Sceglie sempre C (stato 4): misura il SOFFITTO raggiungibile dalla plumbing."""
    name = "oracle"

    def decide(self, context, system_prompt, user_prompt, schema=None):
        return {"regime": "collasso_strutturale", "choice": "C",
                "justification": "[oracle] collasso: proteggo le priorita' alte (stato 4)."}


def _to_action(cur, tgt):
    return ESCALATE if cur < tgt else (DEESCALATE if cur > tgt else MAINTAIN)


def episode_escalation(env, mappo, backend, window_s=30.0, persist=1, verbose=False):
    """Braccio 3: soglia + escalation LLM sul caso ambiguo. Ritorna KPI + log."""
    obs, _ = env.reset()
    acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}
    traj = []
    done = False
    windows_critical = 0
    active_target = None          # stato imposto dall'escalation (o None)
    esc_steps = 0
    log = []

    while not done:
        cur = env._nodes[0].state_machine.current_state.value
        if active_target is not None:
            actions = [_to_action(cur, active_target)]
            esc_steps += 1
        else:
            actions = mappo.act(obs)
        obs, _s, _r, done, info = env.step(actions)

        d = info["deltas"]
        acc["gen"] += d["gen"]; acc["del"] += d["del"]
        acc["drop"] += d["drop"]; acc["lat"] += d["lat"]
        acc["trans"] += info["transitions"]
        traj.append(CongestionState[info["states"][0]].value)

        if info["t"] % window_s < 1e-9 or done:
            metrics = _window_metrics(acc, window_s, env.metrics.collect_compression_ratio())
            a = SupervisorController.assess(metrics, traj)
            if a["health"] == "CRITICO":
                windows_critical += 1
            else:
                windows_critical = 0
                active_target = None            # regime rientrato → rilascia (reversibile)

            # escalation SOLO sul caso ambiguo, confermato da >= persist finestre
            if (active_target is None and windows_critical >= persist
                    and esc.should_escalate(a, traj)):
                dec = esc.escalate_decision(backend, traj, windows_critical)
                active_target = dec["target_state"]     # puo' essere None (scelta A)
                log.append({"t": info["t"], **dec})
                if verbose:
                    print(f"[t={info['t']:4.0f}s] ESCALATION → regime={dec['regime']} "
                          f"choice={dec['choice']} target={dec['target_state']}")
                    print(f"    «{dec['justification']}»")
            acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}

    out = _kpis(env)
    out["esc_steps"] = esc_steps
    out["esc_log"] = log
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["oracle", "ollama", "anthropic", "mock"],
                    default="oracle")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--persist", type=int, default=1,
                    help="finestre critiche consecutive prima di escalare")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="timeout Ollama in s (il 7B su poca RAM e' lento a caricare)")
    ap.add_argument("--scenario", default="collapse",
                    help="'collapse' (OOD, default) oppure un canonico 1-6 "
                         "(es. 3 = degrado TRANSITORIO: test di discriminazione — "
                         "l'LLM deve NON scegliere C qui)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # factory scenario: collasso OOD o canonico 1-6
    if args.scenario == "collapse":
        make_env = lambda seed: build_capacity_collapse(seed=seed)
        label = "capacity_collapse (OOD)"
    else:
        sc = int(args.scenario)
        make_env = lambda seed: EDSMarlEnv(sc, seed=seed)
        label = f"scenario {sc} (canonico — discriminazione)"

    backend = build_backend(args.backend, args.model, args.timeout)
    print(f"  M3-escalation — {label}, backend {getattr(backend,'name',args.backend)}, "
          f"{args.seeds} seed\n")

    arms = {"MAPPO solo": [], "soglia (S1)": [], "escalation (S1+S2)": []}
    for s in range(args.seeds):
        seed = 42 + s
        m = MARLController.from_checkpoint(args.ckpt)
        arms["MAPPO solo"].append(episode_alone(make_env(seed), m))
        m = MARLController.from_checkpoint(args.ckpt)
        arms["soglia (S1)"].append(episode_supervised(make_env(seed),
                                                       m, "mock", args.model, args.window))
        m = MARLController.from_checkpoint(args.ckpt)
        arms["escalation (S1+S2)"].append(
            episode_escalation(make_env(seed), m, backend,
                               args.window, args.persist, verbose=args.verbose and s == 0))

    has_cd = "control_del" in arms["MAPPO solo"][0]
    cd_hdr = "control_del" if has_cd else "(no ctrl)"
    print(f"  {'braccio':<22}{cd_hdr:>13}{'PDR':>9}{'drop':>9}")
    print("  " + "-" * 52)
    base_pdr = np.mean([r["pdr"] for r in arms["MAPPO solo"]])
    base = np.mean([r["control_del"] for r in arms["MAPPO solo"]]) if has_cd else base_pdr
    for name, rows in arms.items():
        pdr = np.mean([r["pdr"] for r in rows])
        drop = np.mean([r["dropped"] for r in rows])
        key = np.array([r["control_del"] for r in rows]) if has_cd else np.array([r["pdr"] for r in rows])
        cd_str = f"{key.mean():>8.3f}±{key.std():<4.2f}" if has_cd else f"{'—':>13}"
        mark = "" if name == "MAPPO solo" else ("  △" if key.mean() >= base - 1e-9 else "  ▽ DANNO")
        print(f"  {name:<22}{cd_str}{pdr:>9.3f}{drop:>9.0f}{mark}")
    print("  " + "-" * 52)
    if not has_cd:
        print("  (discriminazione: escalation deve NON danneggiare il PDR → l'LLM ha "
              "scelto A/B, non C)")
    # esempio di ragionamento (primo seed con log)
    for r in arms["escalation (S1+S2)"]:
        if r["esc_log"]:
            e = r["esc_log"][0]
            print(f"\n  esempio escalation: regime={e['regime']} choice={e['choice']} "
                  f"target={e['target_state']}\n    «{e['justification']}»")
            break


if __name__ == "__main__":
    main()
