#!/usr/bin/env python3
"""
mappo_trace.py - Registra la traccia di osservazione della policy MAPPO nel
SIMULATORE, con lo STESSO schema dell'emulatore (`eds-mappo-observation-trace-v1`).

Serve a MISURARE lo scarto sim-to-real feature per feature: si esegue lo stesso
scenario con lo stesso checkpoint nei due mondi (qui il simulatore; sull'emulatore
`scenarios.py --mappo CKPT --mappo-trace ...`) e si confrontano le tracce con
examples/trace_diff.py. Le due feature approssimate sull'emulatore
(high_priority_ratio, low_priority_ratio) sono quelle da tenere d'occhio.

Uso:
    python3 examples/mappo_trace.py <checkpoint.json> <scenario 1-6> <out.json>
    # es: python3 examples/mappo_trace.py checkpoints/mappo_delta05.json 3 logs/trace_sim_s3.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from simulator.marl import EDSMarlEnv, load_checkpoint  # noqa: E402

# Nomi identici a MAPPO_TRACE_FEATURES / MAPPO_ACTION_NAMES dell'emulatore.
FEATURES = (
    "ewma_occupancy", "congestion_state", "high_priority_ratio",
    "low_priority_ratio", "drop_rate", "link_utilisation", "time_in_state",
)
ACTION_NAMES = ("ESCALATE", "MAINTAIN", "DEESCALATE")


def main():
    if len(sys.argv) < 4:
        print("Uso: mappo_trace.py <checkpoint.json> <scenario 1-6> <out.json>")
        sys.exit(1)
    ckpt, scenario, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    actor, _, meta = load_checkpoint(ckpt)
    min_state_dwell = float(meta.get("min_state_dwell", 0.0))

    rng = np.random.default_rng(4242)
    env = EDSMarlEnv(scenario, seed=1000 + scenario)   # deploy: nessuno shaping nel reward
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        t = env.t                          # tempo della decisione (prima dello step)
        a, _ = actor.act(obs, rng, deterministic=True)
        probs = actor.probs(obs)[0]        # policy dell'unico agente (single bottleneck)
        obs_vec = list(map(float, np.atleast_2d(obs)[0]))
        action_id = int(a[0])
        # maschera: tutta True finche' il checkpoint non dichiara min_state_dwell
        # (coerente con l'emulatore, che con dwell=0 non maschera nulla)
        action_mask = [True, True, True]
        obs, _state, _r, done, _info = env.step(a)
        rows.append({
            "t": t,
            "observation": {name: obs_vec[i] for i, name in enumerate(FEATURES)},
            "action": ACTION_NAMES[action_id],
            "action_id": action_id,
            "action_probabilities": [float(p) for p in probs],
            "action_mask": action_mask,
            "state_after": env._nodes[0].state_machine.current_state.value,
        })

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({
            "schema": "eds-mappo-observation-trace-v1",
            "source": "simulator",
            "scenario": scenario,
            "features": list(FEATURES),
            "checkpoint": os.path.abspath(ckpt),
            "min_state_dwell": min_state_dwell,
            "rows": rows,
        }, fh, indent=2)
    print(f"traccia simulatore salvata: {os.path.abspath(out)}  ({len(rows)} passi)")


if __name__ == "__main__":
    main()
