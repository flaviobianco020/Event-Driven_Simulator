"""
mappo.py — MAPPOTrainer: update PPO-CLIP multi-agente (doc §3.2, §4.2, Tabella 4).

Loss Actor (per agente i, parameter sharing):
    L_i(theta) = E[ min( r_i * A_hat,  clip(r_i, 1-eps, 1+eps) * A_hat ) ]
    r_i(theta) = pi_theta(a_i|o_i) / pi_theta_old(a_i|o_i)
con bonus di entropia (coefficiente c_e, pratica standard PPO/MAPPO,
Schulman et al. 2017 §5; Yu et al. 2022).

Loss Critic:
    L_critic(phi) = E[ (V_phi(s_t) - V_target_t)^2 ]

Iperparametri esatti dal documento (Tabella 4):
    K = 10 epoch, minibatch = 256, eps = 0.2,
    lr_actor = 3e-4, lr_critic = 1e-3, max_grad_norm = 10.
"""
from __future__ import annotations

import numpy as np

from .buffer import RolloutBuffer
from .networks import Actor, Adam, Critic

CLIP_EPS = 0.2          # doc §3.2
K_EPOCHS = 10           # doc Tabella 4
MINIBATCH = 256         # doc Tabella 4
LR_ACTOR = 3e-4         # doc Tabella 4
LR_CRITIC = 1e-3        # doc Tabella 4
MAX_GRAD_NORM = 10.0    # doc Tabella 4
ENTROPY_COEF = 0.01     # standard PPO (Schulman et al. 2017, c2)


class MAPPOTrainer:
    def __init__(self, actor: Actor, critic: Critic,
                 clip_eps: float = CLIP_EPS, k_epochs: int = K_EPOCHS,
                 minibatch: int = MINIBATCH, entropy_coef: float = ENTROPY_COEF,
                 lr_actor: float = LR_ACTOR, lr_critic: float = LR_CRITIC,
                 seed: int = 0) -> None:
        self.actor = actor
        self.critic = critic
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.minibatch = minibatch
        self.entropy_coef = entropy_coef
        self.opt_actor = Adam(actor.net.params, lr_actor, max_grad_norm=MAX_GRAD_NORM)
        self.opt_critic = Adam(critic.net.params, lr_critic, max_grad_norm=MAX_GRAD_NORM)
        self._rng = np.random.default_rng(seed)

    # ── loss + gradienti (backprop manuale) ──────────────────────────────────

    def _actor_minibatch_step(self, obs: np.ndarray, actions: np.ndarray,
                              logp_old: np.ndarray, adv: np.ndarray) -> dict:
        """Un passo di gradient ascent su L_CLIP per un minibatch flat (B, 7)."""
        B = len(actions)
        probs = self.actor.probs(obs)                      # forward (cache attiva)
        idx = np.arange(B)
        logp = np.log(probs[idx, actions] + 1e-12)
        ratio = np.exp(logp - logp_old)

        surr1 = ratio * adv
        clipped = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surr2 = clipped * adv
        pg_loss = -np.minimum(surr1, surr2).mean()

        entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        loss = pg_loss - self.entropy_coef * entropy.mean()

        # dL/dlogp: il ramo clippato ha gradiente nullo rispetto a theta
        active = (surr1 <= surr2).astype(float)            # min = ramo non clippato
        dlogp = -(active * adv * ratio) / B
        # dlogp/dlogits = onehot(a) − probs   (softmax + log-lik categorica)
        d_logits = probs * (-dlogp)[:, None]
        d_logits[idx, actions] += dlogp
        # entropia: dH/dz_k = −p_k (log p_k + H); L include −c_e·mean(H)
        dH = -probs * (np.log(probs + 1e-12) + entropy[:, None])
        d_logits += -(self.entropy_coef / B) * dH

        grads = self.actor.net.backward(d_logits)
        self.opt_actor.step(grads)

        approx_kl = float((logp_old - logp).mean())
        clipfrac = float((np.abs(ratio - 1.0) > self.clip_eps).mean())
        return {"pg_loss": float(pg_loss), "entropy": float(entropy.mean()),
                "approx_kl": approx_kl, "clipfrac": clipfrac,
                "loss": float(loss)}

    def _critic_minibatch_step(self, states: np.ndarray,
                               targets: np.ndarray) -> float:
        """Un passo di gradient descent su L_critic = mean((V − V_target)^2)."""
        v = self.critic.value(states)
        err = v - targets
        loss = float((err * err).mean())
        d_out = (2.0 * err / len(err))[:, None]
        grads = self.critic.net.backward(d_out)
        self.opt_critic.step(grads)
        return loss

    # ── update completo su un rollout ────────────────────────────────────────

    def update(self, buffer: RolloutBuffer, bootstrap_value: float) -> dict:
        """
        Esegue K epoch di update Actor e Critic sul rollout raccolto
        (doc Tabella 4, step 2-4), poi svuota il buffer.
        """
        adv, returns = buffer.compute_gae(bootstrap_value)
        data = buffer.stacked()
        T, N = data["actions"].shape

        # appiattisce (T, N) → (T·N): parameter sharing, il vantaggio del
        # passo t (calcolato dal Critic centralizzato) e' condiviso dagli N agenti
        obs_f = data["obs"].reshape(T * N, -1)
        act_f = data["actions"].reshape(T * N)
        logp_f = data["logp"].reshape(T * N)
        adv_f = np.repeat(adv, N)

        # normalizzazione dei vantaggi (pratica standard MAPPO, Yu et al. 2022)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        stats: dict[str, float] = {}
        n_actor = T * N
        for _ in range(self.k_epochs):
            perm = self._rng.permutation(n_actor)
            for lo in range(0, n_actor, self.minibatch):
                mb = perm[lo:lo + self.minibatch]
                stats = self._actor_minibatch_step(
                    obs_f[mb], act_f[mb], logp_f[mb], adv_f[mb])

        critic_loss = 0.0
        for _ in range(self.k_epochs):
            perm = self._rng.permutation(T)
            for lo in range(0, T, self.minibatch):
                mb = perm[lo:lo + self.minibatch]
                critic_loss = self._critic_minibatch_step(
                    data["states"][mb], returns[mb])

        stats["critic_loss"] = critic_loss
        stats["mean_reward"] = float(np.mean(buffer.rewards))
        buffer.clear()
        return stats
