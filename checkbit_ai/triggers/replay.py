"""Recompute detector outputs for a period from the stored feature pool, rebuild
candidates, and print a per-day report comparing the realtime jump rule (Gumbel)
with Benjamini-Hochberg FDR.

    python -m checkbit_ai.triggers.replay                 # everything in features.sqlite3
    python -m checkbit_ai.triggers.replay --from 2026-11-23T00:00Z --to 2026-11-24T00:00Z
    python -m checkbit_ai.triggers.replay --report-only

Run the feature replay first if feature definitions changed.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from .. import config
from ..featurepool import store as fstore
from . import candidates, jump
from . import params as P
from . import store
from .pipeline import process_triggers


def _minute(text: str) -> int:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() // 60)


def report(feat, market: str, m0: int, m1: int) -> list[dict]:
    """Per UTC day: jump tests, Gumbel rejections, BH rejections; z hits; candidates by tier."""
    rows = feat.execute("SELECT minute, detector, is_anomaly, evidence FROM detector_hits"
                        " WHERE market = ? AND minute BETWEEN ? AND ?", (market, m0, m1)).fetchall()
    days: dict[int, dict] = {}
    for minute, det, hit, ev in rows:
        d = days.setdefault(minute // 1440, {"jump_p": [], "jump_gumbel": 0, "z_hits": 0, "lstm_hits": 0})
        if det == "jump":
            d["jump_p"].append(json.loads(ev)[0]["p_value"])
            d["jump_gumbel"] += hit
        elif det == "robust_z":
            d["z_hits"] += hit
        else:
            d["lstm_hits"] += hit
    out = []
    for day in sorted(days):
        d = days[day]
        tiers = dict(feat.execute("SELECT tier, COUNT(*) FROM candidates WHERE market = ? AND status = 'closed'"
                                  " AND start_minute BETWEEN ? AND ? GROUP BY tier",
                                  (market, day * 1440, day * 1440 + 1439)).fetchall())
        out.append({"day": datetime.fromtimestamp(day * 86400, tz=timezone.utc).date().isoformat(),
                    "jump_tests": len(d["jump_p"]), "jump_gumbel": d["jump_gumbel"],
                    "jump_bh": int(jump.bh_reject(np.array(d["jump_p"]), P.BH_Q).sum()),
                    "z_hit_minutes": d["z_hits"], "lstm_hit_minutes": d["lstm_hits"],
                    "candidates_strong": tiers.get("strong", 0), "candidates_ambiguous": tiers.get("ambiguous", 0)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end", help="exclusive")
    ap.add_argument("--market", default=config.MARKET)
    ap.add_argument("--features-db", default=str(config.features_db_path()))
    ap.add_argument("--analyses-db", default=str(config.DB_PATH))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)

    feat = fstore.connect(args.features_db)
    store.ensure(feat)
    try:
        lo, hi = feat.execute("SELECT MIN(minute), MAX(minute) FROM feature_minutes WHERE market = ?",
                              (args.market,)).fetchone()
        if lo is None:
            print("feature pool is empty; run python -m checkbit_ai.featurepool.replay first")
            return 1
        m0 = _minute(args.start) if args.start else lo
        m1 = min((_minute(args.end) - 1) if args.end else hi, hi - P.lag_minutes())
        if not args.report_only:
            prev = store.get_state(feat, args.market)   # read before process_triggers moves it to m1
            n = process_triggers(feat, args.analyses_db, args.market, m0, m1, update_candidates=False)
            with feat:
                last = max(m1, prev if prev is not None else m1)
                c = candidates.rebuild(feat, args.market, last)
                store.set_state(feat, args.market, last)
            print(f"{args.market}: evaluated {n} minutes ({P.TRIGGER_VERSION}); {c} candidates after rebuild")
        print(f"{'day':10} {'tests':>6} {'gumbel':>6} {'BH':>4} {'z_min':>6} {'lstm':>5} {'strong':>6} {'ambig':>6}")
        for r in report(feat, args.market, m0, m1):
            print(f"{r['day']:10} {r['jump_tests']:6d} {r['jump_gumbel']:6d} {r['jump_bh']:4d} "
                  f"{r['z_hit_minutes']:6d} {r['lstm_hit_minutes']:5d} {r['candidates_strong']:6d} "
                  f"{r['candidates_ambiguous']:6d}")
        return 0
    finally:
        feat.close()


if __name__ == "__main__":
    raise SystemExit(main())
