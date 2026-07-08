#!/usr/bin/env python3
"""
train_mappo.py — Pipeline di training MAPPO, Fase 3 (doc Tabella 4 e 10).

  Training    : M = 500 episodi; ad ogni episodio uno scenario random tra i
                6 canonici, end_time = 100 s simulati (delta_t = 1 s → ~100
                passi/episodio). Update PPO ogni T = 2048 passi raccolti.
  Valutazione : ogni 50 episodi, esecuzione deterministica (argmax) su tutti
                e 6 gli scenari canonici. Metriche: PDR, latenza, Jain,
                compression_ratio, transizioni.
  Export      : checkpoint JSON (Actor+Critic) al miglioramento del reward
                medio di valutazione → checkpoints/mappo_best.json
                (+ mappo_last.json a fine training).

Uso:
    python3 examples/train_mappo.py                     # run completa (500 ep)
    python3 examples/train_mappo.py --episodes 50       # run ridotta
    python3 examples/train_mappo.py --quick             # smoke test (~1 min)
    python3 examples/train_mappo.py --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from simulator.marl import (  # noqa: E402
    Actor, Critic, EDSMarlEnv, MAPPOTrainer, ROLLOUT_T, RolloutBuffer,
    save_checkpoint,
)

N_EPISODES = 500        # doc Tabella 4, step 5
EPISODE_END_TIME = 100.0  # doc Tabella 10, riga Training
EVAL_EVERY = 50         # doc Tabella 10, riga Valutazione
SCENARIOS = (1, 2, 3, 4, 5, 6)
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def evaluate(actor: Actor, seed: int = 10_000) -> dict:
    """
    Esecuzione deterministica (argmax) sui 6 scenari canonici con i loro
    end_time originali. Restituisce metriche per scenario + reward medio.
    """
    rng = np.random.default_rng(seed)
    per_scenario: dict[int, dict] = {}
    rewards = []
    for sc in SCENARIOS:
        env = EDSMarlEnv(sc, seed=seed + sc)
        obs, _state = env.reset()
        done, ep_rew, steps = False, 0.0, 0
        while not done:
            actions, _ = actor.act(obs, rng, deterministic=True)
            obs, _state, r, done, _info = env.step(actions)
            ep_rew += r
            steps += 1
        summ = env.summary()
        summ["mean_step_reward"] = ep_rew / max(steps, 1)
        per_scenario[sc] = summ
        rewards.append(summ["mean_step_reward"])
    return {"per_scenario": per_scenario,
            "mean_reward": float(np.mean(rewards))}


def print_eval(ev: dict, episode: int) -> None:
    print(f"\n  ── Valutazione @ep {episode} "
          f"(deterministica, 6 scenari) ─────────────────────")
    print(f"  {'sc':>2}  {'PDR':>7}  {'lat(ms)':>8}  {'Jain':>6}  "
          f"{'compr':>6}  {'trans':>5}  {'r/step':>7}")
    for sc, s in ev["per_scenario"].items():
        print(f"  {sc:>2}  {s['pdr']*100:6.2f}%  {s['latency']*1000:8.2f}  "
              f"{s['fairness']:6.3f}  {s['compression_ratio']:5.2f}x  "
              f"{s['transitions']:>5}  {s['mean_step_reward']:7.3f}")
    print(f"  reward medio di valutazione: {ev['mean_reward']:.4f}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Training MAPPO Fase 3 EDS")
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 10 episodi, rollout 256")
    ap.add_argument("--stability-penalty", type=float, default=0.0,
                    help="penalita' di reward per ogni transizione di stato "
                         "(reward shaping, es. 0.03). 0 = reward del documento.")
    args = ap.parse_args()

    episodes = 10 if args.quick else args.episodes
    rollout_t = 256 if args.quick else ROLLOUT_T
    stab = args.stability_penalty

    # suffisso distinto per i run con reward shaping: non sovrascrive
    # il checkpoint "vanilla" (reward del documento)
    suffix = "_stab" if stab > 0 else ""
    best_path = os.path.join(CKPT_DIR, f"mappo_best{suffix}.json")
    last_path = os.path.join(CKPT_DIR, f"mappo_last{suffix}.json")

    os.makedirs(CKPT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n_agents = 1  # single bottleneck: l'unico agente e' il router (doc Tab. 6)
    actor = Actor(seed=args.seed)
    critic = Critic(n_agents=n_agents, seed=args.seed + 1)
    trainer = MAPPOTrainer(actor, critic, seed=args.seed + 2)
    buffer = RolloutBuffer(n_agents, capacity=rollout_t)

    print("=" * 68)
    print("  MAPPO — Fase 3 EDS  (Multi-Agent Proximal Policy Optimization)")
    print(f"  Actor {actor.n_params} parametri | Critic {critic.n_params} "
          f"parametri (N={n_agents})")
    print(f"  episodi={episodes}  rollout={rollout_t}  "
          f"eval ogni {EVAL_EVERY} episodi")
    if stab > 0:
        print(f"  stability penalty = {stab} per transizione (reward shaping)")
    print("=" * 68)

    best_reward = -np.inf
    ep_rewards: list[float] = []
    n_updates = 0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        scenario = int(rng.choice(SCENARIOS))
        env = EDSMarlEnv(scenario, seed=int(rng.integers(1, 2**31)),
                         end_time=EPISODE_END_TIME, stability_penalty=stab)
        obs, state = env.reset()
        done, ep_rew, steps = False, 0.0, 0

        while not done:
            actions, logp = actor.act(obs, rng, deterministic=False)
            value = float(critic.value(state[None, :])[0])
            next_obs, next_state, reward, done, _info = env.step(actions)
            buffer.add(obs, actions, logp, state, value, reward, done)
            ep_rew += reward
            steps += 1
            obs, state = next_obs, next_state

            if buffer.full:
                bootstrap = 0.0 if done else float(critic.value(state[None, :])[0])
                stats = trainer.update(buffer, bootstrap)
                n_updates += 1
                print(f"  [update {n_updates:>3}] ep {ep:>3}  "
                      f"r_medio={stats['mean_reward']:6.3f}  "
                      f"pg_loss={stats['pg_loss']:7.4f}  "
                      f"v_loss={stats['critic_loss']:7.4f}  "
                      f"KL={stats['approx_kl']:6.4f}  "
                      f"clip={stats['clipfrac']*100:4.1f}%  "
                      f"H={stats['entropy']:5.3f}")

        ep_rewards.append(ep_rew / max(steps, 1))
        if ep % 10 == 0:
            recent = float(np.mean(ep_rewards[-10:]))
            print(f"  ep {ep:>3}/{episodes}  scenario={scenario}  "
                  f"r/step (ultimi 10 ep) = {recent:.4f}")

        if ep % EVAL_EVERY == 0 or ep == episodes:
            ev = evaluate(actor)
            print_eval(ev, ep)
            if ev["mean_reward"] > best_reward:
                best_reward = ev["mean_reward"]
                save_checkpoint(best_path, actor, critic,
                                meta={"episode": ep,
                                      "eval_reward": best_reward,
                                      "seed": args.seed,
                                      "stability_penalty": stab})
                print(f"  ✔ nuovo best (r={best_reward:.4f}) → {best_path}")

    save_checkpoint(last_path, actor, critic,
                    meta={"episode": episodes, "seed": args.seed,
                          "stability_penalty": stab})
    dt = time.time() - t0
    print("=" * 68)
    print(f"  Training completato: {episodes} episodi, {n_updates} update PPO "
          f"in {dt:.0f}s")
    print(f"  Best eval reward: {best_reward:.4f}  ({best_path})")
    print("=" * 68)


if __name__ == "__main__":
    main()
