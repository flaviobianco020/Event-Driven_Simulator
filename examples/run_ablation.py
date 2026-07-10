#!/usr/bin/env python3
"""
run_ablation.py — Fase 4 / M3-prep: ablation sulla DIMENSIONE del supervisore SLM.

Domanda: "qual e' il supervisore piu' piccolo sufficiente?" (tesi, cfr. Belcak
et al. 2024). Poiche' la DECISIONE e' deterministica (assess()), la dimensione
del modello incide SOLO sulla qualita' della SPIEGAZIONE in linguaggio naturale.
Questo runner esegue lo stesso scenario con piu' modelli Ollama e li mette a
confronto fianco a fianco sul tick d'anomalia.

Prerequisito: server Ollama attivo + modelli scaricati, es.:
    ollama pull qwen2.5:0.5b
    ollama pull qwen2.5:1.5b
    ollama pull qwen2.5:3b
    ollama pull qwen2.5:7b        # ~4.7GB, tirato su 8GB RAM

Uso:
    python3 examples/run_ablation.py --scenario 3
    python3 examples/run_ablation.py --scenario 3 \
        --models qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b,qwen2.5:7b
    python3 examples/run_ablation.py --scenario 3 --anomaly-tick 60

Metrica automatica (euristica): "errore di direzione PDR" — l'explanation menziona
il PDR come alto/elevato quando la valutazione lo dice basso (o viceversa). E' il
difetto che il 3B mostrava; l'ablation verifica se i modelli piu' grandi lo evitano.
La qualita' fine resta un giudizio umano (per la tesi) sulle spiegazioni raccolte.
"""
import sys, os, json, argparse, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_m1_explainer import run_m1, DEFAULT_CKPT  # noqa: E402
from simulator.supervisor.controller import SupervisorController  # noqa: E402

DEFAULT_MODELS = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"]

# parole che indicano "alto/elevato" e "basso": per l'euristica di direzione PDR
_HIGH = re.compile(r"\b(alt[oi]|elevat[oi]|sopra)\b", re.I)
_LOW = re.compile(r"\b(bass[oi]|inferior[ei]|sotto|crollat[oi]|ridott[oi])\b", re.I)


def _pdr_direction_error(justification: str, pdr_is_low: bool) -> bool:
    """
    Euristica: cerca 'PDR' e la parola di direzione piu' vicina. Errore se la
    valutazione dice PDR basso ma il testo lo descrive alto (o viceversa).
    Approssimativa — segnala il difetto tipico dei modelli piccoli.
    """
    for m in re.finditer(r"pdr", justification, re.I):
        # finestra bidirezionale: la parola di direzione puo' precedere o seguire "PDR"
        window = justification[max(0, m.start() - 45): m.start() + 45]
        says_high = bool(_HIGH.search(window))
        says_low = bool(_LOW.search(window))
        if pdr_is_low and says_high and not says_low:
            return True
        if (not pdr_is_low) and says_low and not says_high:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=3, choices=range(1, 7))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--anomaly-tick", type=float, default=60.0,
                    help="istante (s) su cui confrontare le spiegazioni (default 60)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {}
    print(f"  ABLATION supervisore — scenario {args.scenario}, {len(models)} modelli")
    print(f"  (la decisione e' deterministica; varia solo la spiegazione)\n")

    for model in models:
        print(f"  ▶ {model} ...", end=" ", flush=True)
        try:
            res = run_m1(args.scenario, args.seed, args.ckpt,
                         backend_name="ollama", model=model, verbose=False)
            results[model] = res["supervisor_log"]
            # un errore backend lascia 'backend errore' nella giustificazione
            broken = any("backend errore" in e["justification"] for e in res["supervisor_log"])
            print("ERRORE (modello scaricato?)" if broken else "ok")
        except Exception as exc:  # noqa: BLE001
            print(f"ERRORE: {exc}")
            results[model] = None

    # confronto sul tick d'anomalia
    print("\n" + "=" * 78)
    print(f"  CONFRONTO SPIEGAZIONI @ t={args.anomaly_tick:.0f}s  (scenario {args.scenario})")
    print("=" * 78)
    for model in models:
        log = results.get(model)
        if not log:
            print(f"\n  [{model}]  (non disponibile)")
            continue
        entry = min(log, key=lambda e: abs(e["t"] - args.anomaly_tick))
        # verita' deterministica del PDR a quel tick per l'euristica di direzione
        # (ricaviamo dallo scenario ri-valutando: il tick d'anomalia ha PDR basso)
        pdr_low = entry["action"] == "override_state"  # override ⇔ PDR<0.85 o drop alto
        err = _pdr_direction_error(entry["justification"], pdr_low)
        print(f"\n  [{model}]  azione={entry['action']}  "
              f"{'⚠ errore direzione PDR' if err else '✓ direzione ok'}")
        print(f"    «{entry['justification']}»")

    # tabella riassuntiva errori di direzione su tutti i tick
    print("\n" + "=" * 78)
    print("  ERRORI DI DIREZIONE PDR (su tutti i tick lenti)")
    print("=" * 78)
    print(f"  {'modello':<18}{'tick':>6}{'errori direzione PDR':>24}")
    for model in models:
        log = results.get(model)
        if not log:
            print(f"  {model:<18}{'—':>6}{'non disponibile':>24}")
            continue
        n_err = sum(_pdr_direction_error(e["justification"],
                                         e["action"] == "override_state") for e in log)
        print(f"  {model:<18}{len(log):>6}{n_err:>18} / {len(log)}")

    out = args.out or os.path.join(os.path.dirname(__file__), "..", "logs",
                                   f"ablation_scenario{args.scenario}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"scenario": args.scenario, "models": models, "results": results},
                  fh, indent=2, ensure_ascii=False)
    print(f"\n  log salvato: {out}")


if __name__ == "__main__":
    main()
