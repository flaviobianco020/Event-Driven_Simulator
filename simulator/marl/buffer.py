"""
buffer.py — Rollout buffer con Generalized Advantage Estimation (doc §3.3).

Accumula T = 2048 passi (doc Tabella 4, step 1) anche attraversando piu'
episodi (un episodio = 100 s simulati ≈ 100 passi con delta_t = 1 s, quindi
un rollout copre ~20 episodi). Il flag done marca i confini episodio: il
bootstrap GAE non attraversa mai un reset.

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_hat_t = sum_l (gamma*lambda)^l * delta_{t+l}      (gamma=0.99, lambda=0.95)

Il value target del Critic e' il lambda-return  R_t = A_hat_t + V(s_t),
forma generalizzata del bootstrap TD(0) "V_target = r + gamma V(s')" citato
nel documento §4.2 (per lambda→0 coincidono).
"""
from __future__ import annotations

import numpy as np

GAMMA = 0.99      # fattore di sconto (doc Tabella 1)
LAMBDA = 0.95     # mixing bias-varianza GAE (doc §3.3)
ROLLOUT_T = 2048  # passi per rollout (doc Tabella 4)


class RolloutBuffer:
    """
    Memorizza per ogni passo t (condiviso da N agenti, parameter sharing):
      obs      (N, 7)   osservazioni locali o_i
      actions  (N,)     azioni campionate a_i
      logp     (N,)     log pi_old(a_i | o_i)
      state    (7N+4,)  stato globale s_t per il Critic
      value    float    V(s_t)
      reward   float    reward condiviso r_t
      done     bool     True se t chiude l'episodio
    """

    def __init__(self, n_agents: int, capacity: int = ROLLOUT_T) -> None:
        self.n_agents = n_agents
        self.capacity = capacity
        self.clear()

    def clear(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.logp: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def full(self) -> bool:
        return len(self) >= self.capacity

    def add(self, obs: np.ndarray, actions: np.ndarray, logp: np.ndarray,
            state: np.ndarray, value: float, reward: float, done: bool) -> None:
        self.obs.append(np.asarray(obs, dtype=float))
        self.actions.append(np.asarray(actions, dtype=int))
        self.logp.append(np.asarray(logp, dtype=float))
        self.states.append(np.asarray(state, dtype=float))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def compute_gae(self, bootstrap_value: float,
                    gamma: float = GAMMA, lam: float = LAMBDA
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Restituisce (advantages (T,), value_targets (T,)).

        bootstrap_value = V(s_T) dello stato successivo all'ultimo passo del
        buffer; ignorato (azzerato) se l'ultimo passo chiude un episodio.
        """
        T = len(self)
        rewards = np.asarray(self.rewards)
        values = np.asarray(self.values)
        dones = np.asarray(self.dones, dtype=bool)

        adv = np.zeros(T)
        gae = 0.0
        next_value = float(bootstrap_value)
        for t in range(T - 1, -1, -1):
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
            next_value = values[t]
        returns = adv + values
        return adv, returns

    def stacked(self) -> dict[str, np.ndarray]:
        """Tensori impilati: obs (T,N,7), actions (T,N), logp (T,N), states (T,7N+4)."""
        return {
            "obs": np.stack(self.obs),
            "actions": np.stack(self.actions),
            "logp": np.stack(self.logp),
            "states": np.stack(self.states),
        }
