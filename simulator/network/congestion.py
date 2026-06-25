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


class CongestionStateMachine:
    def __init__(self, thresholds: dict[CongestionState, float] | None = None) -> None:
        self.current_state = CongestionState.NORMAL
        self.thresholds: dict[CongestionState, float] = thresholds or DEFAULT_THRESHOLDS.copy()
        self._history: list[tuple[float, CongestionState]] = []

    def evaluate(self, occupancy: float) -> CongestionState:
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
        return self.transition(self.evaluate(occupancy), sim_time)

    @property
    def history(self) -> list[tuple[float, CongestionState]]:
        return list(self._history)
