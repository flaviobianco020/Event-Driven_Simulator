"""
RuleBasedController — Phase 2 control plane.

Called from Simulator._on_state_update whenever a node transitions to a new
CongestionState. This is the integration point for all higher-level reactions:
  Phase 2 (here): log transitions, foundation for action hooks.
  Phase 3 (MARL): replace react() with learned policy.
  Phase 4 (Agentic): replace react() with LLM tool-use decisions.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from ..network.congestion import CongestionState

if TYPE_CHECKING:
    from ..event import Event
    from ..scheduler import EventScheduler


class RuleBasedController:
    """
    Stateless rule-based reactions to congestion state transitions.

    Current Phase 2 actions:
    - Tracks how many times each state was entered (for metrics/analysis).
    - Provides the hook interface for Phase 3/4 replacement.
    """

    def __init__(self) -> None:
        self.state_entry_counts: dict[CongestionState, int] = {
            s: 0 for s in CongestionState
        }

    def react(self, event: "Event", scheduler: "EventScheduler") -> None:
        """Called on every STATE_UPDATE event after a state transition."""
        node = event.node
        if node is None:
            return
        state = node.state_machine.current_state
        self.state_entry_counts[state] = self.state_entry_counts.get(state, 0) + 1
        # Phase 3 will insert learned throttling / rerouting actions here.
        # Phase 4 will call an LLM agent tool to decide the action.
