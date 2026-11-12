"""Shared fixtures. All market data used in tests is SYNTHETIC (see checkbit_ai/synthetic.py)."""
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from checkbit_ai.synthetic import generate  # noqa: E402
from checkbit_ai.train import train  # noqa: E402

# Synthetic data timeline: starts at a fixed epoch; anomalies injected at these minutes.
SYN_START = 1_790_000_000.0
SYN_ANOMALIES = (700, 780)


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("syn") / "synthetic_trades.csv"
    return generate(path, minutes=900, start_ts=SYN_START, anomalies=SYN_ANOMALIES, gap_at=300)


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory, synthetic_csv) -> Path:
    out = tmp_path_factory.mktemp("model") / "lstm_ae"
    train([str(synthetic_csv)], out, epochs=15, data_label="synthetic", log=lambda *a: None)
    return out


def retry_winsock(fn, attempts: int = 5):
    """Retry socket creation on a Windows-only flake (WinError 10014 from
    socket()/socketpair() on this dev machine). Other errors are raised."""
    for attempt in range(attempts):
        try:
            return fn()
        except OSError as exc:
            if getattr(exc, "winerror", None) != 10014 or attempt == attempts - 1:
                raise
            time.sleep(0.2)
