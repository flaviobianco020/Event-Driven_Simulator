from __future__ import annotations
from typing import Any


class ConfigurationManager:
    def __init__(self, random_seed: int = 42, **params: Any) -> None:
        self.random_seed = random_seed
        self._params: dict[str, Any] = dict(params)

    def get(self, key: str, default: Any = None) -> Any:
        return self._params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._params[key] = value
