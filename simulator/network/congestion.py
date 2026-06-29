from __future__ import annotations
from enum import Enum


class CongestionState(Enum):
    NORMAL = 0
    HEADER_COMPRESSION = 1
    DELTA_COMPRESSION = 2
    INCREMENTAL_COMPRESSION = 3
    DROP_LOW_PRIORITY = 4


DEFAULT_THRESHOLDS: dict[CongestionState, float] = {
    CongestionState.HEADER_COMPRESSION: 0.50,
    CongestionState.DELTA_COMPRESSION: 0.70,
    CongestionState.INCREMENTAL_COMPRESSION: 0.85,
    CongestionState.DROP_LOW_PRIORITY: 0.95,
}

# Phase 2 parameters (eFRAC paper §3.3)
PHASE2_EWMA_ALPHA: float = 0.125          # Jacobson/Karn TCP RTO estimator constant
PHASE2_ESCALATION_DEBOUNCE: float = 1.5   # seconds: fire after sustained threshold exceedance
PHASE2_DEESCALATION_COOLDOWN: float = 4.5 # seconds: 3s cooldown + 1.5s hold per step


class CongestionStateMachine:
    """
    Five-state congestion machine driven by queue occupancy.

    Phase 1 (defaults): instant transitions, no smoothing — backward-compatible.
    Phase 2: EWMA smoothing + asymmetric hysteresis matching eFRAC paper §3.3.

    Parameters
    ----------
    ewma_alpha : float
        EWMA smoothing factor α ∈ (0, 1]. 1.0 = no smoothing (Phase 1 default).
        Paper (Phase 2): 0.125 — converges in ~8 samples, suppresses spikes.
    escalation_debounce : float
        Seconds of sustained threshold exceedance before escalating one level.
        0.0 = instant (Phase 1 default). Paper (Phase 2): 1.5 s.
    deescalation_cooldown : float
        Seconds required below threshold before de-escalating one level.
        0.0 = instant (Phase 1 default). Paper (Phase 2): 4.5 s (3s + 1.5s hold).
        3:1 asymmetry vs escalation: treat escalation as emergency, demand proof
        of genuine recovery before stepping down.
    """

    def __init__(
        self,
        thresholds: dict[CongestionState, float] | None = None,
        ewma_alpha: float = 1.0,
        escalation_debounce: float = 0.0,
        deescalation_cooldown: float = 0.0,
    ) -> None:
        self.current_state = CongestionState.NORMAL
        self.thresholds: dict[CongestionState, float] = thresholds or DEFAULT_THRESHOLDS.copy()
        self._history: list[tuple[float, CongestionState]] = []

        # EWMA state
        self._alpha = ewma_alpha
        self._ewma: float = 0.0

        # Hysteresis timers
        self._escalation_debounce = escalation_debounce
        self._deescalation_cooldown = deescalation_cooldown
        self._escalate_since: float | None = None
        self._deescalate_since: float | None = None

    # ── core methods (unchanged API) ───────────────────────────────────────────

    def evaluate(self, occupancy: float) -> CongestionState:
        """Return target state for the given occupancy (no side-effects)."""
        result = CongestionState.NORMAL
        for state in [
            CongestionState.HEADER_COMPRESSION,
            CongestionState.DELTA_COMPRESSION,
            CongestionState.INCREMENTAL_COMPRESSION,
            CongestionState.DROP_LOW_PRIORITY,
        ]:
            if occupancy >= self.thresholds[state]:
                result = state
        return result

    def transition(self, new_state: CongestionState, sim_time: float = 0.0) -> bool:
        if new_state != self.current_state:
            self._history.append((sim_time, new_state))
            self.current_state = new_state
            return True
        return False

    def update(self, occupancy: float, sim_time: float = 0.0) -> bool:
        """
        Update internal EWMA, apply hysteresis, and transition if appropriate.

        Phase 1 (defaults alpha=1.0, debounce=0.0): behaves identically to the
        original — instant transition to the target state, no smoothing.

        Phase 2 (alpha=0.125, debounce=1.5s, cooldown=4.5s): smooths the input,
        steps one level at a time, requires sustained exceedance to escalate and
        a longer cooldown to de-escalate (paper §3.3, 3:1 asymmetry).
        """
        # EWMA: alpha=1.0 → ewma = raw value (no smoothing, Phase 1 compat)
        self._ewma = (1.0 - self._alpha) * self._ewma + self._alpha * occupancy

        target = self.evaluate(self._ewma)

        # ── Escalation ──────────────────────────────────────────────────────────
        if target.value > self.current_state.value:
            if self._escalation_debounce <= 0.0:
                # Phase 1: instant jump to target
                self._escalate_since = None
                self._deescalate_since = None
                return self.transition(target, sim_time)
            else:
                # Phase 2: one-step-at-a-time with debounce
                if self._escalate_since is None:
                    self._escalate_since = sim_time
                self._deescalate_since = None
                if sim_time - self._escalate_since >= self._escalation_debounce:
                    next_state = CongestionState(self.current_state.value + 1)
                    self._escalate_since = None  # reset: next step needs own debounce
                    return self.transition(next_state, sim_time)
                return False

        # ── De-escalation ────────────────────────────────────────────────────────
        elif target.value < self.current_state.value:
            if self._deescalation_cooldown <= 0.0:
                # Phase 1: instant jump to target
                self._escalate_since = None
                self._deescalate_since = None
                return self.transition(target, sim_time)
            else:
                # Phase 2: one-step-at-a-time with cooldown
                if self._deescalate_since is None:
                    self._deescalate_since = sim_time
                self._escalate_since = None
                if sim_time - self._deescalate_since >= self._deescalation_cooldown:
                    prev_state = CongestionState(self.current_state.value - 1)
                    self._deescalate_since = None  # reset: next step needs own cooldown
                    return self.transition(prev_state, sim_time)
                return False

        # ── Stable in current zone ───────────────────────────────────────────────
        else:
            self._escalate_since = None
            self._deescalate_since = None
            return False

    @property
    def ewma_occupancy(self) -> float:
        """Current EWMA-smoothed occupancy estimate."""
        return self._ewma

    @property
    def history(self) -> list[tuple[float, CongestionState]]:
        return list(self._history)
