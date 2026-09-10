"""Content-hash cache. See editor.md §4.4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def content_hash(payload: Any) -> str:
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    total: int = 0


class Cache:
    def __init__(self, maxsize: int = 1024) -> None:
        self._store: dict[str, Any] = {}
        self.stats = CacheStats()
        self.maxsize = maxsize

    def get(self, key: str) -> Any | None:
        self.stats.total += 1
        if key in self._store:
            self.stats.hits += 1
            return self._store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self.maxsize:
            self._store.pop(next(iter(self._store)))
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()
        self.stats = CacheStats()

    def summary(self) -> str:
        return f"{self.stats.hits}/{self.stats.total}"
