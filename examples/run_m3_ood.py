#!/usr/bin/env python3
"""
run_m3_ood.py — Fase 4, Milestone M3: valutazione FUORI DISTRIBUZIONE.

Il risultato quantitativo della fase: su uno scenario mai visto in addestramento
(mix di traffico inedito), la policy MAPPO congelata degrada; il supervisore
(decisione deterministica a soglia + hold, con spiegazione LLM) recupera?

Protocollo:
  * per ogni seed: stesso episodio due volte — braccio A (MAPPO solo, Fase 3
    pura) e braccio B (MAPPO + supervisore con controllo attivo, M2);
  * gruppo di CONTROLLO in-distribution (scenario 3 canonico) con lo stesso
    protocollo: il divario deve essere piccolo dove MAPPO e' addestrato e
    grande dove non lo e' — altrimenti il supervisore sta solo correggendo
    la policy ovunque, non i casi OOD;
  * KPI misurati dal simulatore (indipendenti dal reward): PDR, latenza,
    drop, transizioni. Media ± deviazione su N seed.

Il backend di default e' mock: la DECISIONE e' deterministica (assess), quindi
i KPI non dipendono dal modello LLM — solo le spiegazioni. Per il log di
spiegabilita' con SLM reale usare --backend ollama.

Uso:
    python3 examples/run_m3_ood.py                          # video_flood, 5 seed
    python3 examples/run_m3_ood.py --ood pulsed --seeds 10
    python3 examples/run_m3_ood.py --no-control             # salta il gruppo di controllo
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from simulator.marl import EDSMarlEnv, MARLController  # noqa: E402
from simulator.network.congestion import CongestionState  # noqa: E402
from simulator.supervisor import SupervisorController, Guardrail  # noqa: E402
from simulator.supervisor.ood import OOD_SCENARIOS  # noqa: E402
from run_m1_explainer import make_backend, _window_metrics, DEFAULT_CKPT  # noqa: E402
from run_m2_supervisor import override_to_action, maybe_revoke  # noqa: E402

KPIS = ["pdr", "latency_ms", "dropped", "transitions"]


def _kpis(env) -> dict:
    """KPI dal summary + (se lo scenario la espone) consegna del flusso CONTROL."""
    s = env.summary()
    out = {"pdr": s["pdr"], "latency_ms": s["latency"] * 1000.0,
           "dropped": s["dropped"], "transitions": s["transitions"]}
    fid = getattr(env, "control_flow_id", None)
    if fid is not None:
        delivered = env.metrics.delivered_per_flow.get(fid, 0)
        out["control_del"] = delivered / max(getattr(env, "control_expected", 1.0), 1.0)
    return out


def episode_alone(env, mappo: MARLController) -> dict:
    """Braccio A: Fase 3 pura."""
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _s, _r, done, _i = env.step(mappo.act(obs))
    return _kpis(env)


def episode_supervised(env, mappo: MARLController, backend_name: str,
                       model: str, window_s: float = 30.0) -> dict:
    """Braccio B: M2 (supervisore con controllo attivo). Stesso loop di run_m2."""
    guardrail = Guardrail()
    supervisor = SupervisorController(backend=make_backend(backend_name, model),
                                      guardrail=guardrail, tick_interval=window_s)
    obs, _ = env.reset()
    acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}
    traj: list[int] = []
    done = False
    override_steps = 0
    while not done:
        cur = env._nodes[0].state_machine.current_state.value
        target = supervisor.current_override(env.t)
        if target is not None:
            actions = [override_to_action(cur, target)]
            override_steps += 1
        else:
            actions = mappo.act(obs)
        obs, _s, _r, done, info = env.step(actions)

        d = info["deltas"]
        acc["gen"] += d["gen"]; acc["del"] += d["del"]
        acc["drop"] += d["drop"]; acc["lat"] += d["lat"]
        acc["trans"] += info["transitions"]
        traj.append(CongestionState[info["states"][0]].value)

        if info["t"] % window_s < 1e-9 or done:
            metrics = _window_metrics(acc, window_s,
                                      env.metrics.collect_compression_ratio())
            maybe_revoke(guardrail, metrics, info["t"])
            supervisor.tick(info["t"], metrics, traj)
            acc = {"gen": 0, "del": 0, "drop": 0, "lat": 0.0, "trans": 0}

    out = _kpis(env)
    out["override_steps"] = override_steps
    out["supervisor_log"] = supervisor.log.entries
    return out


def run_block(env_factory, label: str, seeds: int, ckpt: str,
              backend_name: str, model: str, verbose: bool = True) -> dict:
    """Esegue i due bracci su N seed per una famiglia di episodi."""
    alone, sup = [], []
    for s in range(seeds):
        seed = 42 + s
        mappo = MARLController.from_checkpoint(ckpt)
        alone.append(episode_alone(env_factory(seed), mappo))
        mappo = MARLController.from_checkpoint(ckpt)     # rng pulito per parita'
        sup.append(episode_supervised(env_factory(seed), mappo, backend_name, model))

    def agg(rows, k):
        v = np.array([r[k] for r in rows], dtype=float)
        return v.mean(), v.std()

    kpis = KPIS + (["control_del"] if "control_del" in alone[0] else [])
    if verbose:
        print(f"\n  {label}  ({seeds} seed, media ± dev.std)")
        print(f"  {'KPI':<16}{'MAPPO solo':>18}{'+ supervisore':>18}{'Δ':>10}")
        print("  " + "-" * 62)
        for k in kpis:
            am, asd = agg(alone, k)
            sm, ssd = agg(sup, k)
            delta = sm - am
            better = (delta > 0) if k in ("pdr", "control_del") else (delta < 0)
            mark = "▲" if better else ("=" if abs(delta) < 1e-9 else "▼")
            print(f"  {k:<16}{am:>11.3f}±{asd:<5.2f}{sm:>11.3f}±{ssd:<5.2f}{mark}{delta:>+9.3f}")
        ov = np.mean([r["override_steps"] for r in sup])
        print(f"  {'step override':<16}{'—':>18}{ov:>18.1f}")
    return {"alone": alone, "supervised": sup}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", choices=list(OOD_SCENARIOS), default="video_flood")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--backend", choices=["mock", "ollama", "anthropic"], default="mock")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--no-control", action="store_true",
                    help="salta il gruppo di controllo in-distribution (scenario 3)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"  M3 — valutazione OOD «{args.ood}»  (backend {args.backend}; "
          f"la decisione e' deterministica, i KPI non dipendono dal modello LLM)")

    results = {"ood": {}, "control": {}}
    builder = OOD_SCENARIOS[args.ood]
    results["ood"] = run_block(lambda s: builder(seed=s),
                               f"FUORI DISTRIBUZIONE — {args.ood}",
                               args.seeds, args.ckpt, args.backend, args.model)

    if not args.no_control:
        results["control"] = run_block(lambda s: EDSMarlEnv(3, seed=s),
                                       "CONTROLLO in-distribution — scenario 3",
                                       args.seeds, args.ckpt, args.backend, args.model)

    out = args.out or os.path.join(os.path.dirname(__file__), "..", "logs",
                                   f"m3_{args.ood}_{args.backend}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # i log del supervisore sono voluminosi: si salvano solo i KPI per seed
    slim = {blk: {arm: [{k: r[k] for k in KPIS + ["control_del", "override_steps"] if k in r}
                        for r in rows]
                  for arm, rows in d.items()}
            for blk, d in results.items() if d}
    with open(out, "w") as fh:
        json.dump({"ood_scenario": args.ood, "seeds": args.seeds, "results": slim},
                  fh, indent=2, ensure_ascii=False)
    print(f"\n  log salvato: {out}")


if __name__ == "__main__":
    main()
