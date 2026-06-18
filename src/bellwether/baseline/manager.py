"""Per-(agent, task_class, fingerprint) baselines over feature observations.

A :class:`Baseline` holds, per ``(feature_name, context)``:
- numeric features: a bounded sliding window of recent values (supports online update and
  cheap robust statistics; the window *is* the reference distribution for conformal scoring);
- categorical features: a frequency table (supports novelty / distribution scoring).

A :class:`BaselineManager` routes observations to the right baseline by ``run.baseline_key`` —
so a new config fingerprint opens a fresh lineage instead of polluting the old one (doc 03).

The window size bounds memory and gives the baseline a (slow) adaptivity to gradual benign
change; abrupt change is the detectors' job, not the window's.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bellwether.features import FeatureObservation, extract_observations
from bellwether.schema import AgentRun

DEFAULT_WINDOW = 500


class Baseline:
    """The learned reference for a single baseline lineage."""

    def __init__(self, window_size: int = DEFAULT_WINDOW) -> None:
        self.window_size = window_size
        self._numeric: dict[tuple[str, str], deque[float]] = {}
        self._categorical: dict[tuple[str, str], Counter[str]] = {}

    # --- learning -----------------------------------------------------------------------

    def learn_observation(self, obs: FeatureObservation) -> None:
        for nobs in obs.numerics:
            win = self._numeric.get(nobs.key)
            if win is None:
                win = deque(maxlen=self.window_size)
                self._numeric[nobs.key] = win
            win.append(nobs.value)
        for cobs in obs.categoricals:
            counter = self._categorical.get(cobs.key)
            if counter is None:
                counter = Counter()
                self._categorical[cobs.key] = counter
            counter[cobs.category] += 1

    # --- accessors for detectors --------------------------------------------------------

    def numeric_window(self, name: str, context: str) -> list[float]:
        win = self._numeric.get((name, context))
        return list(win) if win is not None else []

    def numeric_count(self, name: str, context: str) -> int:
        win = self._numeric.get((name, context))
        return len(win) if win is not None else 0

    def categorical_counts(self, name: str, context: str) -> tuple[Counter[str], int]:
        counter = self._categorical.get((name, context))
        if counter is None:
            return (Counter(), 0)
        return (counter, sum(counter.values()))

    @property
    def numeric_keys(self) -> list[tuple[str, str]]:
        return list(self._numeric.keys())

    @property
    def categorical_keys(self) -> list[tuple[str, str]]:
        return list(self._categorical.keys())

    @property
    def total_observations(self) -> int:
        """Rough size signal: the largest per-feature window length."""
        return max((len(w) for w in self._numeric.values()), default=0)

    # --- serialization ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "numeric": {
                f"{name}\x1f{ctx}": list(win) for (name, ctx), win in self._numeric.items()
            },
            "categorical": {
                f"{name}\x1f{ctx}": dict(counter)
                for (name, ctx), counter in self._categorical.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Baseline:
        b = cls(window_size=int(data["window_size"]))
        for k, win in data["numeric"].items():
            name, ctx = k.split("\x1f")
            b._numeric[(name, ctx)] = deque(win, maxlen=b.window_size)
        for k, counter in data["categorical"].items():
            name, ctx = k.split("\x1f")
            b._categorical[(name, ctx)] = Counter(counter)
        return b


class BaselineManager:
    """Owns many baselines, one per ``(agent, task_class, fingerprint)`` lineage."""

    def __init__(self, window_size: int = DEFAULT_WINDOW) -> None:
        self.window_size = window_size
        self._baselines: dict[tuple[str, str, str], Baseline] = {}

    def get_or_create(self, key: tuple[str, str, str]) -> Baseline:
        b = self._baselines.get(key)
        if b is None:
            b = Baseline(window_size=self.window_size)
            self._baselines[key] = b
        return b

    def baseline_for(self, run: AgentRun) -> Baseline | None:
        return self._baselines.get(run.baseline_key)

    def learn(self, run: AgentRun) -> None:
        """Incorporate one run's observations into its lineage baseline."""
        b = self.get_or_create(run.baseline_key)
        for obs in extract_observations(run):
            b.learn_observation(obs)

    def learn_many(self, runs: Iterable[AgentRun]) -> int:
        n = 0
        for run in runs:
            self.learn(run)
            n += 1
        return n

    @property
    def keys(self) -> list[tuple[str, str, str]]:
        return list(self._baselines.keys())

    # --- serialization ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "baselines": {"\x1f".join(key): b.to_dict() for key, b in self._baselines.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineManager:
        mgr = cls(window_size=int(data["window_size"]))
        for joined, bdata in data["baselines"].items():
            name, task, fp = joined.split("\x1f")
            mgr._baselines[(name, task, fp)] = Baseline.from_dict(bdata)
        return mgr

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BaselineManager:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
