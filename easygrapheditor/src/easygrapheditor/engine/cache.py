"""Deterministic content hashes and a bounded memory/disk cache."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> Any:
    """Return a JSON-shaped value without relying on ``repr`` or dict order."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return {
                "__ndarray__": True,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha1": hashlib.sha1(value.tobytes()).hexdigest(),
            }
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if is_dataclass(value) and not isinstance(value, type):
        return {"__dataclass__": type(value).__qualname__, "value": _canonical(asdict(value))}
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted(
            (_canonical(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True)
        )
    if isinstance(value, bytes):
        return {"__bytes__": hashlib.sha1(value).hexdigest(), "length": len(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__object__": type(value).__qualname__, "value": str(value)}


def content_hash(payload: Any) -> str:
    raw = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    total: int = 0


class Cache:
    """True LRU cache, optionally mirrored as pickle values in ``disk_path``.

    Disk entries are an optional persistence tier, not a security boundary. Do
    not point it at files that an untrusted user can modify.
    """

    def __init__(self, maxsize: int = 1024, disk_path: str | Path | None = None) -> None:
        self._store: OrderedDict[str, Any] = OrderedDict()
        self.stats = CacheStats()
        self.maxsize = max(0, int(maxsize))
        self.disk_path = Path(disk_path) if disk_path is not None else None
        if self.disk_path is not None:
            self.disk_path.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        assert self.disk_path is not None
        return self.disk_path / f"{key}.pickle"

    def get(self, key: str) -> Any | None:
        self.stats.total += 1
        if key in self._store:
            self.stats.hits += 1
            self._store.move_to_end(key)
            if self.disk_path is not None:
                self._path(key).touch(exist_ok=True)
            return self._store[key]
        if self.disk_path is None:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            with path.open("rb") as handle:
                value = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError):
            path.unlink(missing_ok=True)
            return None
        self.stats.hits += 1
        path.touch()
        self._remember(key, value, write_disk=False)
        return value

    def _prune_disk(self) -> None:
        if self.disk_path is None or not self.maxsize:
            return
        entries = sorted(self.disk_path.glob("*.pickle"), key=lambda path: path.stat().st_mtime_ns)
        while len(entries) > self.maxsize:
            entries.pop(0).unlink(missing_ok=True)

    def _remember(self, key: str, value: Any, *, write_disk: bool) -> None:
        if self.maxsize:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)
        if write_disk and self.disk_path is not None:
            target = self._path(key)
            fd, name = tempfile.mkstemp(prefix=f".{key}.", dir=self.disk_path)
            try:
                with os.fdopen(fd, "wb") as handle:
                    pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(name, target)
            finally:
                Path(name).unlink(missing_ok=True)
            self._prune_disk()

    def set(self, key: str, value: Any) -> None:
        self._remember(key, value, write_disk=True)

    def clear(self) -> None:
        self._store.clear()
        if self.disk_path is not None:
            for path in self.disk_path.glob("*.pickle"):
                path.unlink(missing_ok=True)
        self.stats = CacheStats()

    def summary(self) -> str:
        return f"{self.stats.hits}/{self.stats.total}"
