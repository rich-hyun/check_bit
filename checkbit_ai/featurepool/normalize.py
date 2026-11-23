"""Normalization layer, separate from feature computation.

A Reference defines what "usual" means for a value at index i of a minute-aligned
series. RollingWindow uses the previous `window` minutes (current excluded). A
same-time-of-day reference can be added later by implementing the same protocol.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import params as P


class Reference(Protocol):
    name: str
    lookback_minutes: int

    def baseline(self, series: np.ndarray, i: int) -> tuple[float, float] | None:
        """(median, 1.4826*MAD) of the reference values for index i, or None if too few."""


@dataclass(frozen=True)
class RollingWindow:
    name: str
    window: int
    min_periods: int

    @property
    def lookback_minutes(self) -> int:
        return self.window

    def baseline(self, series: np.ndarray, i: int) -> tuple[float, float] | None:
        w = series[max(0, i - self.window):i]          # current minute excluded
        w = w[np.isfinite(w)]
        if len(w) < self.min_periods:
            return None
        med = float(np.median(w))
        return med, P.MAD_SCALE * float(np.median(np.abs(w - med)))


SHORT = RollingWindow("short", P.SHORT_WINDOW, P.SHORT_MIN_PERIODS)
LONG = RollingWindow("long", P.LONG_WINDOW, P.LONG_MIN_PERIODS)


@dataclass
class Normalizer:
    short: Reference = SHORT
    long: Reference = LONG

    @property
    def lookback_minutes(self) -> int:
        return max(self.short.lookback_minutes, self.long.lookback_minutes)

    @staticmethod
    def _z(x: float, ref: Reference, series: np.ndarray, i: int, floor: float) -> float:
        if not math.isfinite(x):
            return math.nan
        b = ref.baseline(series, i)
        if b is None:
            return math.nan
        med, scale = b
        return (x - med) / max(scale, floor)

    def z(self, series: np.ndarray, i: int, floor: float) -> tuple[float, float, float]:
        """(z_short, z_long, z) for series[i]; z falls back to z_short when z_long is NaN."""
        x = float(series[i])
        zs = self._z(x, self.short, series, i, floor)
        zl = self._z(x, self.long, series, i, floor)
        return zs, zl, (zl if math.isfinite(zl) else zs)
