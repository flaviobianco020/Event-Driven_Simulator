"""
networks.py — Reti neurali Actor e Critic di MAPPO (Fase 3, doc §5).

Implementazione NumPy pura con backpropagation manuale: nessuna dipendenza
da PyTorch/TensorFlow, cosi' l'Actor addestrato puo' essere caricato anche
nei container Alpine dell'emulatore ContainerLab (deploy, doc Tabella 10)
dove installare torch non e' pratico.

Architetture ESATTE dal documento MAPPO (Tabelle 5 e 6):

  Actor : Input(7) → LayerNorm → Linear(7→64) → Tanh
                   → Linear(64→64) → Tanh → Linear(64→3) → Softmax
  Critic: Input(7N+4) → LayerNorm → Linear(→128) → Tanh
                      → Linear(128→128) → Tanh → Linear(128→1)

LayerNorm senza parametri apprendibili (la formula dei parametri in Tabella 5,
"7x64+64+64x64+64+64x3+3", non include gain/bias di LayerNorm; per il Critic
N=1 la Tabella 6 da' 11*128+128+128*128+128+128+1 = 18.177, che torna esatto
solo con LayerNorm non-affine).

Ottimizzatore: Adam (pratica standard PPO, Schulman et al. 2017), learning
rate Actor 3e-4 e Critic 1e-3 come da Tabella 4, gradient clipping con
max_norm=10 come da Tabella 4 (step 3).
"""
from __future__ import annotations

import json
import numpy as np

_LN_EPS = 1e-5


# ─────────────────────────── inizializzazione ────────────────────────────────

def _orthogonal(shape: tuple[int, int], gain: float, rng: np.random.Generator) -> np.ndarray:
    """Init ortogonale (trick standard del paper MAPPO, Yu et al. 2022)."""
    a = rng.standard_normal(shape)
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else vt
    return gain * q[: shape[0], : shape[1]]


# ─────────────────────────────── MLP core ────────────────────────────────────

class _MLPCore:
    """
    LayerNorm → Linear → Tanh → Linear → Tanh → Linear   (backprop manuale).

    forward() salva le attivazioni in cache; backward(d_out) restituisce i
    gradienti di tutti i parametri per l'ultimo batch passato in forward().
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 rng: np.random.Generator, out_gain: float) -> None:
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.params: dict[str, np.ndarray] = {
            "W1": _orthogonal((in_dim, hidden), np.sqrt(2.0), rng),
            "b1": np.zeros(hidden),
            "W2": _orthogonal((hidden, hidden), np.sqrt(2.0), rng),
            "b2": np.zeros(hidden),
            "W3": _orthogonal((hidden, out_dim), out_gain, rng),
            "b3": np.zeros(out_dim),
        }
        self._cache: dict[str, np.ndarray] = {}

    @property
    def n_params(self) -> int:
        return sum(p.size for p in self.params.values())

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, in_dim) → out: (B, out_dim). Cache per backward()."""
        p = self.params
        mu = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + _LN_EPS)
        xhat = (x - mu) * inv_std                     # LayerNorm non-affine
        z1 = xhat @ p["W1"] + p["b1"]
        h1 = np.tanh(z1)
        z2 = h1 @ p["W2"] + p["b2"]
        h2 = np.tanh(z2)
        out = h2 @ p["W3"] + p["b3"]
        self._cache = {"xhat": xhat, "inv_std": inv_std, "h1": h1, "h2": h2}
        return out

    def backward(self, d_out: np.ndarray) -> dict[str, np.ndarray]:
        """d_out: (B, out_dim) = dL/d(out). Restituisce dL/d(param)."""
        p, c = self.params, self._cache
        xhat, inv_std, h1, h2 = c["xhat"], c["inv_std"], c["h1"], c["h2"]

        grads = {
            "W3": h2.T @ d_out,
            "b3": d_out.sum(axis=0),
        }
        dh2 = d_out @ p["W3"].T
        dz2 = dh2 * (1.0 - h2 * h2)                   # d tanh
        grads["W2"] = h1.T @ dz2
        grads["b2"] = dz2.sum(axis=0)
        dh1 = dz2 @ p["W2"].T
        dz1 = dh1 * (1.0 - h1 * h1)
        grads["W1"] = xhat.T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        # LayerNorm backward (non-affine):
        # dx = inv_std * (dxhat − mean(dxhat) − xhat * mean(dxhat⊙xhat))
        dxhat = dz1 @ p["W1"].T
        m1 = dxhat.mean(axis=1, keepdims=True)
        m2 = (dxhat * xhat).mean(axis=1, keepdims=True)
        _dx = inv_std * (dxhat - m1 - xhat * m2)      # non usato (input = dati)
        return grads


# ─────────────────────────────── ottimizzatore ───────────────────────────────

class Adam:
    """Adam standard (Kingma & Ba 2015) con global-norm gradient clipping."""

    def __init__(self, params: dict[str, np.ndarray], lr: float,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 max_grad_norm: float = 10.0) -> None:
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self._m = {k: np.zeros_like(v) for k, v in params.items()}
        self._v = {k: np.zeros_like(v) for k, v in params.items()}
        self._t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        # gradient clipping con max_norm=10 (doc Tabella 4, step 3)
        total = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        if total > self.max_grad_norm:
            scale = self.max_grad_norm / (total + 1e-12)
            grads = {k: g * scale for k, g in grads.items()}
        self._t += 1
        bc1 = 1.0 - self.b1 ** self._t
        bc2 = 1.0 - self.b2 ** self._t
        for k, g in grads.items():
            self._m[k] = self.b1 * self._m[k] + (1 - self.b1) * g
            self._v[k] = self.b2 * self._v[k] + (1 - self.b2) * (g * g)
            m_hat = self._m[k] / bc1
            v_hat = self._v[k] / bc2
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ──────────────────────────────── Actor ──────────────────────────────────────

class Actor:
    """
    Policy pi(a|o, theta) — doc Tabella 5. Condivisa tra tutti gli agenti
    (parameter sharing, doc §2). Azioni: 0=ESCALATE, 1=MAINTAIN, 2=DE-ESCALATE.
    """

    OBS_DIM = 7
    N_ACTIONS = 3

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # out_gain 0.01: logits iniziali ~uniformi → esplorazione ampia all'avvio
        self.net = _MLPCore(self.OBS_DIM, 64, self.N_ACTIONS, rng, out_gain=0.01)

    @property
    def n_params(self) -> int:
        return self.net.n_params

    def probs(self, obs: np.ndarray) -> np.ndarray:
        """obs: (B, 7) → probabilita' (B, 3) via softmax (stabile)."""
        logits = self.net.forward(np.atleast_2d(obs))
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def act(self, obs: np.ndarray, rng: np.random.Generator,
            deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        obs: (N, 7) → (azioni (N,), log-prob (N,)).
        deterministic=True → argmax (valutazione, doc Tabella 10);
        altrimenti campionamento stocastico dalla policy.
        """
        p = self.probs(obs)
        if deterministic:
            a = p.argmax(axis=1)
        else:
            cum = p.cumsum(axis=1)
            u = rng.random((p.shape[0], 1))
            a = (u > cum).sum(axis=1)
        logp = np.log(p[np.arange(len(a)), a] + 1e-12)
        return a, logp


# ──────────────────────────────── Critic ─────────────────────────────────────

class Critic:
    """
    V(s_global, phi) centralizzato (CTDE, doc §4.1) — doc Tabella 6.
    Input dim = 7N+4; per single bottleneck (N=1) dim=11, parametri 18.177.
    Usato solo in training: sparisce in produzione (doc Tabella 10).
    """

    def __init__(self, n_agents: int = 1, seed: int = 1) -> None:
        self.n_agents = n_agents
        self.state_dim = 7 * n_agents + 4
        rng = np.random.default_rng(seed)
        self.net = _MLPCore(self.state_dim, 128, 1, rng, out_gain=1.0)

    @property
    def n_params(self) -> int:
        return self.net.n_params

    def value(self, state: np.ndarray) -> np.ndarray:
        """state: (B, 7N+4) → V(s): (B,)."""
        return self.net.forward(np.atleast_2d(state))[:, 0]


# ───────────────────────────── checkpoint JSON ───────────────────────────────

def save_checkpoint(path: str, actor: Actor, critic: Critic | None = None,
                    meta: dict | None = None) -> None:
    """Esporta i pesi in JSON (doc Tabella 10, riga Export)."""
    blob: dict = {
        "format": "eds-mappo-v1",
        "actor": {k: v.tolist() for k, v in actor.net.params.items()},
        "meta": meta or {},
    }
    if critic is not None:
        blob["critic"] = {k: v.tolist() for k, v in critic.net.params.items()}
        blob["n_agents"] = critic.n_agents
    with open(path, "w") as fh:
        json.dump(blob, fh)


def load_checkpoint(path: str) -> tuple[Actor, Critic | None, dict]:
    """Ricarica un checkpoint salvato con save_checkpoint()."""
    with open(path) as fh:
        blob = json.load(fh)
    actor = Actor()
    for k, v in blob["actor"].items():
        actor.net.params[k] = np.asarray(v, dtype=float)
    critic = None
    if "critic" in blob:
        critic = Critic(n_agents=blob.get("n_agents", 1))
        for k, v in blob["critic"].items():
            critic.net.params[k] = np.asarray(v, dtype=float)
    return actor, critic, blob.get("meta", {})
