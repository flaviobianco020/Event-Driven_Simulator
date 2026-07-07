"""
simulator.marl — Fase 3: MAPPO (Multi-Agent Proximal Policy Optimization).

Implementazione fedele al documento tecnico "MAPPO — Fase 3 EDS"
(generate_mappo_doc.py): Dec-POMDP, CTDE, Actor 7→64→64→3, Critic
centralizzato (7N+4)→128→128→1, PPO-CLIP con GAE, reward multi-obiettivo
PDR/drop/latenza/Jain.

Import separato da `simulator` (che resta senza dipendenze): questo
sotto-pacchetto richiede numpy.

    from simulator.marl import (
        EDSMarlEnv, Actor, Critic, MAPPOTrainer, RolloutBuffer,
        MARLController, save_checkpoint, load_checkpoint,
    )
"""
from .buffer import GAMMA, LAMBDA, ROLLOUT_T, RolloutBuffer
from .controller import MARLController
from .env import (
    DEESCALATE,
    DT,
    ESCALATE,
    MAINTAIN,
    N_ACTIONS,
    OBS_DIM,
    AgentControlledStateMachine,
    EDSMarlEnv,
)
from .mappo import (
    CLIP_EPS,
    ENTROPY_COEF,
    K_EPOCHS,
    LR_ACTOR,
    LR_CRITIC,
    MINIBATCH,
    MAPPOTrainer,
)
from .networks import Actor, Adam, Critic, load_checkpoint, save_checkpoint

__all__ = [
    "Actor", "Critic", "Adam", "MAPPOTrainer", "RolloutBuffer",
    "EDSMarlEnv", "AgentControlledStateMachine", "MARLController",
    "save_checkpoint", "load_checkpoint",
    "ESCALATE", "MAINTAIN", "DEESCALATE", "N_ACTIONS", "OBS_DIM", "DT",
    "GAMMA", "LAMBDA", "ROLLOUT_T",
    "CLIP_EPS", "K_EPOCHS", "MINIBATCH", "LR_ACTOR", "LR_CRITIC", "ENTROPY_COEF",
]
