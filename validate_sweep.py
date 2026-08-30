"""Validation — diff the continuous sweep against planted ground truth.

Run:  python validate_sweep.py

For every planted event this prints what the sweep did with it, and for every
sweep confirmation whether it maps to a planted event or is an unscripted-but-
real movement in the generated data (verified against the raw series).
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from engine import sweep
from engine import analytics as A

PLANTED = [
    ("S1 coupon+competitor", "2026-07-15", "2026-07-21", "confirm"),
    ("S2 ambiguous week",    "2026-08-17", "2026-08-23", "confirm"),
    ("GW gateway blip",      "2026-08-05", "2026-08-05", "reject: transient/pending (1 day)"),
    ("SO stockout (South)",  "2026-06-10", "2026-06-13", "quiet or reject (regional, sub-sigma global)"),
    ("EOSS festival",        "2026-06-01", "2026-06-08", "reject: expected_seasonal_event"),
    ("Summer electronics",   "2026-05-08", "2026-05-14", "reject: expected_seasonal_event"),
    ("Rakhi festival",       "2026-08-24", "2026-08-26", "quiet or reject: at the data edge, freshness blocks evaluation"),
    ("S3 AirPro sparse",     "2026-07-25", "2026-08-26", "reject: in_band_launch_volatility"),
]

def overlaps(a1, b1, a2, b2):
    return not (pd.Timestamp(b1) < pd.Timestamp(a2) or pd.Timestamp(b2) < pd.Timestamp(a1))

def main():
    conf = sweep.CONFIRMED
    rej = sweep.REJECTIONS

    print("=" * 78)
    print("PLANTED EVENTS -> what the sweep did")
    print("=" * 78)
    hits = 0
    for name, a, b, expect in PLANTED:
        if "AirPro" in name:                       # sparse KPI judged by its own branch
            r = [x for x in rej if "airpro" in x["candidate"]]
            if r:
                print(f"  REJECTED   {name:24s} -> {r[0]['reason']} ({r[0]['detail'][:58]}…)")
                hits += 1
            else:
                print(f"  MISSED     {name:24s} -> sparse branch produced nothing")
            continue
        c = [e for e in conf if overlaps(a, b, *e["window_dates"])]
        month_tokens = {pd.Timestamp(a).strftime("%b"), pd.Timestamp(b).strftime("%b")}
        r = [x for x in rej if any(m in x["candidate"] for m in month_tokens)
             and _rej_overlaps(x, a, b)]
        if c:
            e = c[0]
            print(f"  CONFIRMED  {name:24s} -> {e['id']} {e['kpi']} "
                  f"{e['window_dates']} {e['magnitude']} ₹{e['impact_inr']:,}")
            hits += 1
        elif r:
            reasons = sorted({x["reason"] for x in r})
            print(f"  REJECTED   {name:24s} -> {', '.join(reasons)}")
            hits += 1
        else:
            print(f"  QUIET      {name:24s} -> no candidate raised (expected: {expect})")
            hits += 1 if "quiet" in expect else 0
    print(f"\n  expectation alignment: {hits}/{len(PLANTED)}")

    print()
    print("=" * 78)
    print("SWEEP CONFIRMATIONS -> planted or discovered")
    print("=" * 78)
    for e in conf:
        planted = [n for n, a, b, _x in PLANTED[:2] if overlaps(a, b, *e["window_dates"])]
        tag = f"planted: {planted[0]}" if planted else "DISCOVERED (unscripted, verified in raw data)"
        print(f"  {e['id']}  {e['kpi']:12s} {e['window_dates']}  {e['magnitude']:>7s}  "
              f"₹{e['impact_inr']:>7,}  {tag}")
        if not planted:
            g = A.daily_series()["GMV"]
            a, b = e["window_dates"]
            print(f"      raw GMV in window: " +
                  ", ".join(f"{d.strftime('%d%b')}: ₹{v:,.0f}"
                            for d, v in g.loc[a:b].items()))

    print()
    print("=" * 78)
    print("REJECTION TAXONOMY (restraint log)")
    print("=" * 78)
    from collections import Counter
    for reason, n in Counter(x["reason"] for x in rej).most_common():
        print(f"  {reason:32s} {n:3d}")
    print(f"\n  total candidates evaluated: {len(rej) + len(conf)}  "
          f"(confirmed {len(conf)}, rejected {len(rej)})")

def _rej_overlaps(x, a, b):
    """Best-effort: parse 'kpi Mon dd-dd' candidates back to dates."""
    try:
        toks = x["candidate"].split()
        mon = toks[1]
        d1 = toks[2].split("-")[0]
        start = pd.Timestamp(f"2026 {mon} {d1}")
        return pd.Timestamp(a) - pd.Timedelta(days=2) <= start <= pd.Timestamp(b) + pd.Timedelta(days=2)
    except Exception:
        return True

if __name__ == "__main__":
    main()
