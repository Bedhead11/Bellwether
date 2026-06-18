"""Page-Hinkley online change-point detector.

A classic sequential test for a persistent change in the mean of a stream. We use it on the
per-run drift-score stream to tell a *sustained* regime shift (the agent's normal has actually
moved) from a transient spike (one weird run). It is online, O(1) per update, and deterministic —
so it composes with the streaming baseline and is exactly unit-testable.

We detect *increases* (drift going persistently up). ``delta`` is the tolerated drift before
accumulation, ``threshold`` (λ) is how much cumulative evidence is needed to declare a change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageHinkley:
    delta: float = 0.01
    threshold: float = 2.0
    min_samples: int = 8  # don't signal before this many observations (warmup)

    _n: int = 0
    _mean: float = 0.0
    _cum: float = 0.0
    _min_cum: float = 0.0

    def update(self, value: float) -> bool:
        """Feed one observation; return True when a persistent upward change is detected."""
        self._n += 1
        self._mean += (value - self._mean) / self._n
        # Accumulate how far each value sits above its running mean, minus a tolerance.
        self._cum += value - self._mean - self.delta
        self._min_cum = min(self._min_cum, self._cum)
        if self._n < self.min_samples:
            return False
        return (self._cum - self._min_cum) > self.threshold

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._cum = 0.0
        self._min_cum = 0.0

    @property
    def n(self) -> int:
        return self._n
