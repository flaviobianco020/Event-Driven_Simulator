#!/usr/bin/env python3
"""
sim_scenario_probe.py - Esegue UNO scenario nel simulatore con una policy MAPPO
e stampa le metriche nello STESSO formato dell'emulatore, per il confronto
sim-vs-emulatore.

Uso:
    python3 examples/sim_scenario_probe.py <checkpoint.json> [scenario]
    # es: python3 examples/sim_scenario_probe.py checkpoints/mappo_last_stab_cc.json 1
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from simulator.marl import EDSMarlEnv, load_checkpoint  # noqa: E402
from simulator.network.congestion import CongestionState  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Uso: sim_scenario_probe.py <checkpoint.json> [scenario 1-6]")
        sys.exit(1)
    ckpt = sys.argv[1]
    scenario = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    actor, _, meta = load_checkpoint(ckpt)

    rng = np.random.default_rng(4242)
    env = EDSMarlEnv(scenario, seed=1000 + scenario)   # deploy: nessuno shaping nel reward
    obs, _ = env.reset()
    state_time = {s.name: 0.0 for s in CongestionState}
    done = False
    while not done:
        a, _ = actor.act(obs, rng, deterministic=True)
        obs, _, _, done, info = env.step(a)
        state_time[info["states"][0]] += 1.0   # 1 s per passo

    s = env.summary()
    print("-" * 66)
    print(f"  SIMULATORE - scenario {scenario}  |  checkpoint: {os.path.basename(ckpt)}")
    print(f"  cc={meta.get('compression_cost',0)}  stab={meta.get('stability_penalty',0)}  "
          f"ep={meta.get('episode','?')}")
    print("-" * 66)
    print(f"  Pacchetti generati .............. {s['generated']}")
    print(f"  Pacchetti consegnati ............ {s['delivered']}")
    print(f"  Packet Delivery Ratio ........... {s['pdr']*100:.2f}%")
    print(f"  Latenza end-to-end .............. {s['latency']*1000:.2f} ms")
    print(f"  Drop totali ..................... {s['dropped']}")
    print(f"  Transizioni stato congestione ... {s['transitions']}")
    print(f"  Fairness (Jain) ................. {s['fairness']:.3f}")
    print(f"  Compression ratio ............... {s['compression_ratio']:.3f}x")
    st = "  ".join(f"{k[:4]}={v:.0f}s" for k, v in state_time.items() if v > 0)
    print(f"  Tempo per stato ................. {st}")
    print("-" * 66)


if __name__ == "__main__":
    main()
