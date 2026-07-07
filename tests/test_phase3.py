"""
test_phase3.py — Test unitari Fase 3 (MAPPO).

Copre: architetture (conteggio parametri Tabelle 5-6), softmax valida,
correttezza dei gradienti manuali (confronto con differenze finite), GAE,
loss PPO-CLIP, semantica delle azioni, funzione di reward, ambiente
Dec-POMDP, checkpoint JSON round-trip, smoke test di training.

Esecuzione:  python3 -m pytest tests/test_phase3.py -q
        oppure  python3 tests/test_phase3.py
"""
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from simulator.marl import (
    Actor, Critic, EDSMarlEnv, MAPPOTrainer, MARLController, RolloutBuffer,
    ESCALATE, MAINTAIN, DEESCALATE, OBS_DIM,
    load_checkpoint, save_checkpoint,
)
from simulator.marl.env import AgentControlledStateMachine, LAT_MAX
from simulator.network.congestion import CongestionState


# ── architetture (doc Tabelle 5 e 6) ─────────────────────────────────────────

class TestNetworks(unittest.TestCase):
    def test_actor_param_count(self):
        # formula doc Tabella 5: 7*64+64 + 64*64+64 + 64*3+3 = 4867
        # (la caption riporta "4.931" ma la valutazione della formula e' 4867)
        self.assertEqual(Actor().n_params, 7 * 64 + 64 + 64 * 64 + 64 + 64 * 3 + 3)
        self.assertEqual(Actor().n_params, 4867)

    def test_critic_param_count_n1(self):
        # doc Tabella 6 (N=1, input dim=11): 11*128+128 + 128*128+128 + 128+1 = 18177
        self.assertEqual(Critic(n_agents=1).n_params, 18177)
        self.assertEqual(Critic(n_agents=1).state_dim, 11)

    def test_actor_probs_valid(self):
        a = Actor(seed=3)
        p = a.probs(np.random.default_rng(0).random((16, OBS_DIM)))
        self.assertEqual(p.shape, (16, 3))
        self.assertTrue(np.all(p > 0))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)

    def test_actor_deterministic_is_argmax(self):
        a = Actor(seed=4)
        obs = np.random.default_rng(1).random((8, OBS_DIM))
        acts, _ = a.act(obs, np.random.default_rng(2), deterministic=True)
        np.testing.assert_array_equal(acts, a.probs(obs).argmax(axis=1))

    def test_critic_output_shape(self):
        c = Critic(n_agents=1, seed=5)
        v = c.value(np.random.default_rng(3).random((10, 11)))
        self.assertEqual(v.shape, (10,))


# ── gradienti manuali vs differenze finite ────────────────────────────────────

class TestGradients(unittest.TestCase):
    """Verifica il backprop manuale contro il gradiente numerico."""

    def _numeric_grad(self, param: np.ndarray, loss_fn, eps: float = 1e-6):
        g = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            ix = it.multi_index
            orig = param[ix]
            param[ix] = orig + eps
            lp = loss_fn()
            param[ix] = orig - eps
            lm = loss_fn()
            param[ix] = orig
            g[ix] = (lp - lm) / (2 * eps)
            it.iternext()
        return g

    def test_critic_gradient_matches_numeric(self):
        c = Critic(n_agents=1, seed=7)
        rng = np.random.default_rng(11)
        states = rng.random((6, 11))
        targets = rng.random(6)

        def loss_fn():
            v = c.value(states)
            return float(((v - targets) ** 2).mean())

        v = c.value(states)
        d_out = (2.0 * (v - targets) / len(targets))[:, None]
        analytic = c.net.backward(d_out)
        for key in ("W3", "b3", "W2", "b2", "W1", "b1"):
            numeric = self._numeric_grad(c.net.params[key], loss_fn)
            np.testing.assert_allclose(analytic[key], numeric,
                                       rtol=1e-4, atol=1e-6,
                                       err_msg=f"gradiente {key} errato")

    def test_actor_gradient_matches_numeric(self):
        actor = Actor(seed=8)
        trainer = MAPPOTrainer(actor, Critic(seed=9), entropy_coef=0.01)
        rng = np.random.default_rng(12)
        B = 5
        obs = rng.random((B, OBS_DIM))
        actions = rng.integers(0, 3, B)
        adv = rng.standard_normal(B)
        # logp_old vicino ma diverso da logp corrente → alcuni ratio clippati
        logp_now = np.log(actor.probs(obs)[np.arange(B), actions])
        logp_old = logp_now + rng.uniform(-0.3, 0.3, B)

        def loss_fn():
            p = actor.probs(obs)
            lp = np.log(p[np.arange(B), actions] + 1e-12)
            ratio = np.exp(lp - logp_old)
            s1 = ratio * adv
            s2 = np.clip(ratio, 0.8, 1.2) * adv
            ent = -(p * np.log(p + 1e-12)).sum(axis=1)
            return float(-np.minimum(s1, s2).mean() - 0.01 * ent.mean())

        # gradiente analitico: replica _actor_minibatch_step senza step Adam
        probs = actor.probs(obs)
        idx = np.arange(B)
        logp = np.log(probs[idx, actions] + 1e-12)
        ratio = np.exp(logp - logp_old)
        s1 = ratio * adv
        s2 = np.clip(ratio, 0.8, 1.2) * adv
        active = (s1 <= s2).astype(float)
        dlogp = -(active * adv * ratio) / B
        d_logits = probs * (-dlogp)[:, None]
        d_logits[idx, actions] += dlogp
        entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        dH = -probs * (np.log(probs + 1e-12) + entropy[:, None])
        d_logits += -(0.01 / B) * dH
        analytic = actor.net.backward(d_logits)

        for key in ("W3", "b3", "W2", "b2", "W1", "b1"):
            numeric = self._numeric_grad(actor.net.params[key], loss_fn)
            np.testing.assert_allclose(analytic[key], numeric,
                                       rtol=1e-3, atol=1e-6,
                                       err_msg=f"gradiente {key} errato")
        del trainer  # inutilizzato oltre la costruzione


# ── GAE (doc §3.3) ────────────────────────────────────────────────────────────

class TestGAE(unittest.TestCase):
    def test_gae_hand_computed(self):
        buf = RolloutBuffer(n_agents=1, capacity=8)
        o = np.zeros((1, OBS_DIM)); s = np.zeros(11)
        # 3 passi, nessun done, bootstrap V=0.5
        data = [(1.0, 0.2), (0.5, 0.3), (2.0, 0.1)]   # (reward, value)
        for r, v in data:
            buf.add(o, np.array([1]), np.array([0.0]), s, v, r, False)
        gamma, lam = 0.99, 0.95
        adv, ret = buf.compute_gae(0.5, gamma, lam)

        d2 = 2.0 + gamma * 0.5 - 0.1
        d1 = 0.5 + gamma * 0.1 - 0.3
        d0 = 1.0 + gamma * 0.3 - 0.2
        a2 = d2
        a1 = d1 + gamma * lam * a2
        a0 = d0 + gamma * lam * a1
        np.testing.assert_allclose(adv, [a0, a1, a2], rtol=1e-12)
        np.testing.assert_allclose(ret, [a0 + 0.2, a1 + 0.3, a2 + 0.1], rtol=1e-12)

    def test_gae_done_blocks_bootstrap(self):
        buf = RolloutBuffer(n_agents=1, capacity=4)
        o = np.zeros((1, OBS_DIM)); s = np.zeros(11)
        buf.add(o, np.array([1]), np.array([0.0]), s, 0.4, 1.0, True)
        adv, _ = buf.compute_gae(bootstrap_value=99.0)   # ignorato: done=True
        self.assertAlmostEqual(adv[0], 1.0 - 0.4, places=12)


# ── ambiente (doc §5.3, §6, §7) ───────────────────────────────────────────────

class TestEnv(unittest.TestCase):
    def test_reset_shapes(self):
        env = EDSMarlEnv(1, seed=1)
        obs, state = env.reset()
        self.assertEqual(obs.shape, (1, OBS_DIM))
        self.assertEqual(state.shape, (11,))

    def test_obs_in_range(self):
        env = EDSMarlEnv(5, seed=2)
        obs, _ = env.reset()
        for _ in range(15):
            obs, _, r, done, _ = env.step([MAINTAIN])
            self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 1.0),
                            f"osservazione fuori [0,1]: {obs}")
            self.assertTrue(math.isfinite(r))

    def test_action_semantics(self):
        env = EDSMarlEnv(1, seed=3)
        env.reset()
        node = env._nodes[0]
        self.assertEqual(node.state_machine.current_state, CongestionState.NORMAL)
        env.step([ESCALATE])
        self.assertEqual(node.state_machine.current_state,
                         CongestionState.HEADER_COMPRESSION)
        env.step([DEESCALATE])
        self.assertEqual(node.state_machine.current_state, CongestionState.NORMAL)
        env.step([DEESCALATE])   # gia' a NORMAL: resta (clamp inferiore)
        self.assertEqual(node.state_machine.current_state, CongestionState.NORMAL)
        for _ in range(6):       # clamp superiore a DROP_LOW_PRIORITY
            env.step([ESCALATE])
        self.assertEqual(node.state_machine.current_state,
                         CongestionState.DROP_LOW_PRIORITY)

    def test_agent_csm_never_self_transitions(self):
        sm = AgentControlledStateMachine()
        for t in range(50):
            sm.update(1.0, float(t))   # occupancy sempre 100%
        self.assertEqual(sm.current_state, CongestionState.NORMAL)
        self.assertGreater(sm.ewma_occupancy, 0.9)

    def test_episode_terminates(self):
        env = EDSMarlEnv(1, seed=4, end_time=10.0)
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, _, done, _ = env.step([MAINTAIN])
            steps += 1
        self.assertEqual(steps, 10)

    def test_reward_formula(self):
        env = EDSMarlEnv(1, seed=5)
        deltas = {"gen": 100, "del": 80, "drop": 20, "lat": 80 * 0.5,
                  "per_flow": {0: 40, 1: 40}, "served": [0]}
        # PDR=0.8, drop=0.2, lat=0.5s → 0.25 norm., Jain=1
        expected = 0.8 - 0.3 * 0.2 + 0.2 * 1.0 - 0.2 * (0.5 / LAT_MAX)
        self.assertAlmostEqual(env._reward(deltas), expected, places=12)

    def test_all_scenarios_build_and_step(self):
        for sc in range(1, 7):
            env = EDSMarlEnv(sc, seed=sc, end_time=5.0)
            obs, state = env.reset()
            self.assertEqual(state.shape, (11,))
            for _ in range(5):
                obs, state, r, done, _ = env.step([MAINTAIN])
                self.assertTrue(math.isfinite(r))


# ── checkpoint / controller ───────────────────────────────────────────────────

class TestCheckpoint(unittest.TestCase):
    def test_roundtrip(self):
        actor, critic = Actor(seed=20), Critic(seed=21)
        obs = np.random.default_rng(6).random((4, OBS_DIM))
        st = np.random.default_rng(7).random((4, 11))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.json")
            save_checkpoint(path, actor, critic, meta={"episode": 3})
            a2, c2, meta = load_checkpoint(path)
        np.testing.assert_allclose(a2.probs(obs), actor.probs(obs), rtol=1e-12)
        np.testing.assert_allclose(c2.value(st), critic.value(st), rtol=1e-12)
        self.assertEqual(meta["episode"], 3)

    def test_marl_controller_acts(self):
        ctrl = MARLController(Actor(seed=22))
        acts = ctrl.act(np.random.default_rng(8).random((1, OBS_DIM)))
        self.assertEqual(acts.shape, (1,))
        self.assertIn(int(acts[0]), (0, 1, 2))


# ── smoke test training ───────────────────────────────────────────────────────

class TestTrainingSmoke(unittest.TestCase):
    def test_update_changes_params_and_is_finite(self):
        rng = np.random.default_rng(30)
        actor, critic = Actor(seed=30), Critic(seed=31)
        trainer = MAPPOTrainer(actor, critic, k_epochs=2, minibatch=16, seed=32)
        buf = RolloutBuffer(n_agents=1, capacity=64)

        env = EDSMarlEnv(1, seed=33, end_time=100.0)
        obs, state = env.reset()
        done = False
        while not buf.full and not done:
            a, lp = actor.act(obs, rng)
            v = float(critic.value(state[None, :])[0])
            obs2, state2, r, done, _ = env.step(a)
            buf.add(obs, a, lp, state, v, r, done)
            obs, state = obs2, state2

        w_before = actor.net.params["W1"].copy()
        stats = trainer.update(buf, bootstrap_value=0.0)
        self.assertFalse(np.allclose(actor.net.params["W1"], w_before),
                         "l'update PPO non ha modificato i pesi dell'Actor")
        for k, v in stats.items():
            self.assertTrue(math.isfinite(v), f"stat {k} non finita: {v}")
        self.assertEqual(len(buf), 0, "il buffer deve essere svuotato dopo l'update")


if __name__ == "__main__":
    unittest.main(verbosity=2)
