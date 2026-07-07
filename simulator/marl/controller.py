"""
controller.py — MARLController: la policy MAPPO addestrata come controller EDS.

Sostituto drop-in di RuleBasedController (doc §7.1): stessa interfaccia
react(event, scheduler), piu' il metodo act() che interroga l'Actor.

In esecuzione (Decentralized Execution, doc §4.1) serve SOLO l'Actor: il
Critic centralizzato "sparisce in produzione" (doc Tabella 10). Il checkpoint
JSON prodotto da examples/train_mappo.py e' autosufficiente e non richiede
PyTorch — puo' essere copiato cosi' com'e' nel container router
dell'emulatore ContainerLab.

Uso nel simulatore:
    actor, _, meta = load_checkpoint("checkpoints/mappo_best.json")
    ctrl = MARLController(actor)
    sim._controller = ctrl          # rimpiazza RuleBasedController
    action = ctrl.act(obs)          # 0=ESCALATE, 1=MAINTAIN, 2=DE-ESCALATE
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..network.congestion import CongestionState
from .networks import Actor, load_checkpoint

if TYPE_CHECKING:
    from ..event import Event
    from ..scheduler import EventScheduler


class MARLController:
    """Controller Fase 3: decisioni prese dalla policy Actor appresa."""

    def __init__(self, actor: Actor, deterministic: bool = True,
                 seed: int = 0) -> None:
        self.actor = actor
        self.deterministic = deterministic
        self._rng = np.random.default_rng(seed)
        # stessa contabilita' di RuleBasedController (compatibilita' API)
        self.state_entry_counts: dict[CongestionState, int] = {
            s: 0 for s in CongestionState
        }

    @classmethod
    def from_checkpoint(cls, path: str, deterministic: bool = True,
                        seed: int = 0) -> "MARLController":
        actor, _, _ = load_checkpoint(path)
        return cls(actor, deterministic=deterministic, seed=seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (N, 7) osservazioni locali → azioni (N,) in {0, 1, 2}.
        deterministic=True → argmax (deploy/valutazione, doc Tabella 10).
        """
        actions, _ = self.actor.act(np.atleast_2d(obs), self._rng,
                                    deterministic=self.deterministic)
        return actions

    def react(self, event: "Event", scheduler: "EventScheduler") -> None:
        """Hook STATE_UPDATE, API-compatibile con RuleBasedController."""
        node = event.node
        if node is None:
            return
        state = node.state_machine.current_state
        self.state_entry_counts[state] = self.state_entry_counts.get(state, 0) + 1
