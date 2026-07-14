#!/usr/bin/env python3
"""
probe_compression.py - Mostra l'effetto del costo di compressione (--compression-cost).

Esegue due checkpoint in modalita' deterministica su tutti i 6 scenari e
confronta QUANTO comprimono: frazione di tempo in NORMAL, stato di
compressione medio, compression ratio e PDR. L'ipotesi da verificare:

    con il costo di compressione la policy comprime SOLO sotto congestione
    (piu' tempo in NORMAL, stato medio piu' basso) SENZA perdere PDR.

Uso:
    python3 examples/probe_compression.py \
        checkpoints/mappo_best_stab.json checkpoints/mappo_best_stab_cc.json
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from simulator.marl import EDSMarlEnv, load_checkpoint  # noqa: E402

SCENARIOS = (1, 2, 3, 4, 5, 6)
NAMES = {1: "single bottleneck", 2: "flash crowd", 3: "bandwidth degr.",
         4: "link fail/recov.", 5: "persistent overload", 6: "mixed traffic"}


def probe(actor, seed_base=7000):
    """Per ogni scenario: PDR, compression, stato medio, %tempo in NORMAL."""
    rng = np.random.default_rng(seed_base)
    rows = {}
    for sc in SCENARIOS:
        env = EDSMarlEnv(sc, seed=seed_base + sc)   # deploy: nessuno shaping nel reward
        obs, _ = env.reset()
        states = []
        done = False
        while not done:
            a, _ = actor.act(obs, rng, deterministic=True)
            obs, _, _, done, info = env.step(a)
            states.append(env._nodes[0].state_machine.current_state.value)
        s = env.summary()
        arr = np.array(states)
        rows[sc] = {
            "pdr": s["pdr"],
            "compr": s["compression_ratio"],
            "mean_state": float(arr.mean()),
            "pct_normal": float((arr == 0).mean()),
        }
    return rows


def main():
    if len(sys.argv) < 3:
        print("Uso: probe_compression.py <ckpt_senza_costo> <ckpt_con_costo>")
        sys.exit(1)
    a0, _, m0 = load_checkpoint(sys.argv[1])
    a1, _, m1 = load_checkpoint(sys.argv[2])
    r0 = probe(a0)
    r1 = probe(a1)

    print("=" * 84)
    print("  EFFETTO DEL COSTO DI COMPRESSIONE")
    print(f"  A = senza costo   (cc={m0.get('compression_cost', 0)})   {os.path.basename(sys.argv[1])}")
    print(f"  B = con costo     (cc={m1.get('compression_cost', 0)})   {os.path.basename(sys.argv[2])}")
    print("=" * 84)
    hdr = (f"  {'scenario':<22}{'PDR A':>8}{'PDR B':>8}   "
           f"{'compr A':>8}{'compr B':>8}   {'stato A':>8}{'stato B':>8}   "
           f"{'%NORM A':>8}{'%NORM B':>8}")
    print(hdr)
    for sc in SCENARIOS:
        x, y = r0[sc], r1[sc]
        print(f"  {sc} {NAMES[sc]:<20}"
              f"{x['pdr']*100:7.1f}%{y['pdr']*100:7.1f}%   "
              f"{x['compr']:7.2f}x{y['compr']:7.2f}x   "
              f"{x['mean_state']:8.2f}{y['mean_state']:8.2f}   "
              f"{x['pct_normal']*100:7.0f}%{y['pct_normal']*100:7.0f}%")

    def avg(r, k):
        return float(np.mean([r[sc][k] for sc in SCENARIOS]))
    print("  " + "-" * 82)
    print(f"  {'MEDIA':<22}"
          f"{avg(r0,'pdr')*100:7.1f}%{avg(r1,'pdr')*100:7.1f}%   "
          f"{avg(r0,'compr'):7.2f}x{avg(r1,'compr'):7.2f}x   "
          f"{avg(r0,'mean_state'):8.2f}{avg(r1,'mean_state'):8.2f}   "
          f"{avg(r0,'pct_normal')*100:7.0f}%{avg(r1,'pct_normal')*100:7.0f}%")
    print("=" * 84)
    print("  Atteso con il costo (B): stato medio piu' basso e piu' tempo in")
    print("  NORMAL, con PDR sostanzialmente invariato → comprime solo se serve.")


if __name__ == "__main__":
    main()
