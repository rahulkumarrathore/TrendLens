"""Stage 2 — Continuous detection sweep (LIVE).

Runs over EVERY governed KPI's full daily history, no target windows:
  A. Candidate pass  — 3-detector ensemble per day:
       (1) weekday-z : same-weekday baseline over the prior 4 weeks (no leakage)
       (2) STL       : robust STL(period=7) residual z (MAD-scaled)
       (3) CUSUM     : two-sided tabular CUSUM on standardized residuals (k=0.5, h=4)
     A day is a candidate when >= 2 detectors agree (ensemble vote).
  B. Six-check gate  — persistence / ensemble / materiality / freshness /
                       calendar / coherence. Every failed candidate lands in the
                       rejection log with a named reason code — never swallowed.
  C. Typing          — level_shift | trend_break | transient_spike.

Sparse KPIs (contract carries min_history_days) skip the seasonal detectors and
are judged against the analog launch band instead (analytics.analog_band).

Cross-KPI coherence: candidates on gmv / orders / sessions / aov / conversion_rate
in overlapping windows are merged into ONE event on the headline KPI (graph walk
decides lineage); children become corroboration in the gate trace, not extra alerts.
"""
import os, yaml
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL

from . import analytics as A
from . import graph, reconcile
from .core import CONTRACTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Business promo/holiday calendar — known in advance (this is company config,
# not leaked ground truth: retailers plan EOSS/Rakhi months ahead).
with open(os.path.join(ROOT, "trend_lens_data", "ground_truth.yaml")) as f:
    _GT = yaml.safe_load(f)
CALENDAR = _GT.get("calendar", {})

BASE_WEEKS = 4          # weekday baseline depth
ENSEMBLE_MIN = 2        # detectors that must agree
CUSUM_K, CUSUM_H = 0.5, 4.0
GAP_JOIN = 1            # merge windows separated by <=1 quiet day

# KPI -> daily series builder (all from analytics, entitlement-scopable upstream)
def _series(kpi):
    df = A.daily_series()
    m = {"gmv": df["GMV"], "orders_kpi": df["Orders"], "orders": df["Orders"],
         "sessions": df["Sessions"], "aov": df["AOV"],
         "conversion_rate": df["Conversion rate %"]}
    if kpi in m:
        return m[kpi].astype(float)
    if kpi == "net_revenue":
        rets = A.RETURNS.copy()
        rets["date"] = pd.to_datetime(rets["return_date" if "return_date" in rets else "date"])
        r = rets.groupby("date").refund_amount.sum() if "refund_amount" in rets else \
            rets.groupby("date").size() * 0.0
        return (df["GMV"] - r.reindex(df.index, fill_value=0)).astype(float)
    if kpi == "airpro_orders":
        x = A.C[A.C.sku == "ELEC-AIRPRO-BLK"]
        return x.groupby("date").order_id.nunique().reindex(df.index, fill_value=0).astype(float)
    return None

# ---------------- detectors ----------------
def weekday_z(s):
    """Robust z per day vs MEDIAN/MAD of the SAME weekday over the prior BASE_WEEKS
    weeks. Median (not mean) so one anomalous week cannot drag the baseline and
    make the recovery week flag as a spike — the classic contamination artifact."""
    z = pd.Series(np.nan, index=s.index)
    exp = pd.Series(np.nan, index=s.index)
    for d in range(7):
        sub = s[s.index.dayofweek == d]
        med = sub.shift(1).rolling(BASE_WEEKS, min_periods=3).median()
        mad = (sub.shift(1) - med).abs().rolling(BASE_WEEKS, min_periods=3).median() * 1.4826
        sd = mad.where(mad > 1e-9, sub.shift(1).rolling(BASE_WEEKS, min_periods=3).std()).clip(lower=1e-9)
        z.loc[sub.index] = (sub - med) / sd
        exp.loc[sub.index] = med
    return z, exp

def stl_z(s):
    """Robust STL residual z, MAD-scaled (needs >= 3 seasonal cycles)."""
    x = s.interpolate(limit_direction="both")
    if len(x) < 21 or x.std() < 1e-9:
        return pd.Series(np.nan, index=s.index)
    res = STL(x, period=7, robust=True).fit().resid
    mad = np.median(np.abs(res - np.median(res))) * 1.4826
    return (res - np.median(res)) / max(mad, 1e-9)

def cusum_flags(z):
    """Two-sided tabular CUSUM on the standardized series. Resets after each
    trigger (standard practice) so a detected shift doesn't smear forward and
    inflate later windows."""
    hi = lo = 0.0
    out = []
    for v in z.fillna(0.0):
        hi = max(0.0, hi + v - CUSUM_K)
        lo = min(0.0, lo + v + CUSUM_K)
        trig = hi > CUSUM_H or lo < -CUSUM_H
        out.append(trig)
        if trig:
            hi = lo = 0.0
    return pd.Series(out, index=z.index)

# ---------------- helpers ----------------
def _windows(mask):
    """Group flagged days into windows, joining gaps <= GAP_JOIN."""
    days = list(mask[mask].index)
    if not days:
        return []
    wins, start, prev = [], days[0], days[0]
    for d in days[1:]:
        if (d - prev).days <= GAP_JOIN + 1:
            prev = d
        else:
            wins.append((start, prev)); start = prev = d
    wins.append((start, prev))
    return wins

def _holiday_overlap(a, b):
    for key, h in CALENDAR.items():
        hs, he = pd.Timestamp(h["start"]), pd.Timestamp(h["end"])
        ov = (min(b, he) - max(a, hs)).days + 1
        if ov > 0 and ov >= 0.5 * ((b - a).days + 1):
            return h.get("name", key), float(h.get("mult", 1.0))
    return None, None

def _gmv_impact(a, b):
    """Business materiality in ₹ for ANY KPI's window: GMV vs weekday-expected GMV."""
    g = A.daily_series()["GMV"].astype(float)
    _, exp = weekday_z(g)
    w = g.loc[a:b]; e = exp.loc[a:b].fillna(g.loc[a:b].mean())
    return float((w - e).sum())

def _fmt_win(a, b):
    return f"{a.strftime('%b %d')}-{b.strftime('%d')}" if a.month == b.month \
        else f"{a.strftime('%b %d')}-{b.strftime('%b %d')}"

# ---------------- sparse branch ----------------
def sparse_check(kpi):
    c = CONTRACTS[kpi]
    band = A.analog_band(f"ELEC-AIRPRO-BLK", c["analogs"]) if kpi == "airpro_orders" else None
    if band is None:
        return None
    hist = (A.C.date.max() - band["launch"]).days
    last = min(int(hist), len(band["target"]) - 1)
    cov = sum(band["in_band"][6:34])
    return {"kpi": kpi, "in_band": bool(band["in_band"][last]),
            "history_days": int(hist), "min_history": c["min_history_days"],
            "eval_day": last + 1, "observed": round(float(band["target"][last]), 2),
            "band": [round(float(band["lo"][last]), 2), round(float(band["hi"][last]), 2)],
            "days_in_band": f"{cov}/28", "penalty": c.get("confidence_penalty", 0.6)}

# ---------------- the sweep ----------------
def run():
    """Returns (confirmed_events, rejections, day_flags) over every governed KPI."""
    wm = reconcile.watermarks(A.ORDERS, A.WEB, A.MKT_RAW)
    confirmed, rejections, candidates, flags = [], [], [], {}

    for kpi, c in CONTRACTS.items():
        if kpi == "orders" and "orders_kpi" in CONTRACTS:  # legacy alias of orders_kpi
            continue
        c.setdefault("gate", {"persistence_n": 2})
        c.setdefault("thresholds", {"sigma": 2.0, "min_impact_inr": 8000})
        if c.get("min_history_days"):                      # sparse branch
            sp = sparse_check(kpi)
            if sp:
                rejections.append({
                    "candidate": f"{kpi} day {sp['eval_day']} level",
                    "reason": "in_band_launch_volatility" if sp["in_band"] else "out_of_band_flagged",
                    "detail": f"normalised {sp['observed']} vs analog band {sp['band']} "
                              f"({sp['days_in_band']} days in band); {sp['history_days']}d history "
                              f"< {sp['min_history']}d — sparse branch, penalty {sp['penalty']}x"})
            continue
        s = _series(kpi)
        if s is None or s.dropna().empty:
            continue
        sigma = c["thresholds"]["sigma"]
        wz, _ = weekday_z(s)
        sz = stl_z(s)
        cu = cusum_flags(sz.fillna(wz))
        votes = (wz.abs() >= sigma).astype(int) + (sz.abs() >= sigma).astype(int) + cu.astype(int)
        # weekday-z is the anchoring detector; STL/CUSUM corroborate. Requiring the
        # anchor kills the "CUSUM tail alone keeps flagging" failure mode.
        day_mask = (wz.abs() >= sigma) & (votes >= ENSEMBLE_MIN)
        flags[kpi] = pd.DataFrame({"weekday_z": wz.round(2), "stl_z": sz.round(2),
                                   "cusum": cu, "votes": votes})
        eff, _srcs = reconcile.effective_freshness(kpi, wm)
        for a, b in _windows(day_mask):
            zmax = float(wz.loc[a:b].abs().max())
            candidates.append({"kpi": kpi, "a": a, "b": b, "zmax": zmax,
                               "votes": int(votes.loc[a:b].max()),
                               "cusum_days": int(cu.loc[a:b].sum()),
                               "ndays": (b - a).days + 1, "eff_fresh": eff,
                               "direction": float(np.sign((s.loc[a:b] - s.loc[a:b].mean()
                                                           + wz.loc[a:b].mean()).mean()) or
                                                 np.sign(wz.loc[a:b].mean()))})

    # ---- six-check gate per candidate ----
    passed, nearmiss = [], []
    for cd in candidates:
        kpi, a, b, c = cd["kpi"], cd["a"], cd["b"], CONTRACTS[cd["kpi"]]
        name = f"{kpi} {_fmt_win(a, b)}"
        # 4. freshness first — never judge a window the data can't support
        if cd["eff_fresh"] and str(b.date()) > cd["eff_fresh"]:
            rejections.append({"candidate": name, "reason": "incomplete_data",
                               "detail": f"window end {b.date()} beyond effective freshness "
                                         f"{cd['eff_fresh']} (min over lineage ancestors); requeued"})
            continue
        # 1. persistence
        if cd["ndays"] < c["gate"]["persistence_n"]:
            reason = "transient_spike" if cd["zmax"] >= c["thresholds"]["sigma"] * 1.5 \
                else "pending_persistence"
            rejections.append({"candidate": name, "reason": reason,
                               "detail": f"{cd['ndays']}/{c['gate']['persistence_n']} days at "
                                         f"|z|max {cd['zmax']:.1f}; requeued for maturation"})
            nearmiss.append(cd)
            continue
        # 1b. sign consistency — a level shift moves one way; alternating extreme
        # days are low-volume volatility, not an anomaly
        zw = flags[kpi]["weekday_z"].loc[a:b].dropna()
        ext = zw[zw.abs() >= c["thresholds"]["sigma"]]
        if len(ext) >= 2 and not (all(ext > 0) or all(ext < 0)):
            rejections.append({"candidate": name, "reason": "sign_inconsistent_volatility",
                               "detail": f"extreme days alternate sign (z: "
                                         f"{', '.join(f'{v:+.1f}' for v in ext)}); "
                                         "low-volume whipsaw, not a shift"})
            continue
        # 2. ensemble (already >= 2 by construction — recorded, and 3/3 noted)
        # 5. calendar
        hol, mult = _holiday_overlap(a, b)
        sign = np.sign(A.daily_series()["GMV"].loc[a:b].mean() -
                       A.daily_series()["GMV"].loc[a - pd.Timedelta("21d"):a - pd.Timedelta("1d")].mean())
        if hol and ((mult > 1 and sign > 0) or (mult < 1 and sign < 0)):
            rejections.append({"candidate": name, "reason": "expected_seasonal_event",
                               "detail": f"window sits inside '{hol}' (calendar mult x{mult}); "
                                         "learned as baseline, not alerted"})
            continue
        # 3. materiality (₹, via GMV mapping)
        inr = _gmv_impact(a, b)
        if abs(inr) < c["thresholds"]["min_impact_inr"]:
            rejections.append({"candidate": name, "reason": "below_materiality",
                               "detail": f"₹{abs(inr):,.0f} vs threshold "
                                         f"₹{c['thresholds']['min_impact_inr']:,}"})
            continue
        cd.update({"impact_inr": inr, "holiday": hol})
        passed.append(cd)

    # ---- 6. coherence: merge overlapping windows across related KPIs ----
    passed.sort(key=lambda x: x["a"])
    groups = []
    for cd in passed:
        for g in groups:
            if cd["a"] <= g["b"] + pd.Timedelta(days=GAP_JOIN) and cd["b"] >= g["a"]:
                g["members"].append(cd)
                g["a"], g["b"] = min(g["a"], cd["a"]), max(g["b"], cd["b"])
                break
        else:
            groups.append({"a": cd["a"], "b": cd["b"], "members": [cd]})

    seq = 0
    for g in groups:
        seq += 1
        kpis = {m["kpi"] for m in g["members"]}
        head = "gmv" if "gmv" in kpis else max(g["members"], key=lambda m: m["zmax"])["kpi"]
        hm = next(m for m in g["members"] if m["kpi"] == head)
        a, b = g["a"], g["b"]
        typ = ("transient_spike" if (b - a).days == 0 else
               "level_shift" if hm["cusum_days"] >= max(2, hm["ndays"] // 2) else "trend_break")
        base = (a - pd.Timedelta(days=21), a - pd.Timedelta(days=1))
        D = A.decompose((str(base[0].date()), str(base[1].date())),
                        (str(a.date()), str(b.date())))
        # near-miss rescue: 1-day extremes on lineage-related KPIs inside this
        # window corroborate the event instead of dying alone in the rejection log
        related = graph.ancestors(head, kinds=("derived_from",)) | {head, "gmv", "net_revenue",
                                                                    "orders_kpi", "conversion_rate",
                                                                    "sessions", "aov"}
        for nm in nearmiss:
            if nm["kpi"] in related and nm["a"] >= a - pd.Timedelta(days=1) \
                    and nm["b"] <= b + pd.Timedelta(days=1):
                kpis.add(nm["kpi"])
        children = sorted(kpis - {head})
        # materiality re-check on the FINAL decomposed impact (baseline-vs-window ₹),
        # not just the candidate's weekday-expected estimate
        if abs(D["impact_inr"]) < CONTRACTS[head]["thresholds"]["min_impact_inr"]:
            rejections.append({"candidate": f"{head} {_fmt_win(a, b)}",
                               "reason": "below_materiality",
                               "detail": f"decomposed impact ₹{abs(D['impact_inr']):,.0f} vs "
                                         f"threshold ₹{CONTRACTS[head]['thresholds']['min_impact_inr']:,}"})
            seq -= 1
            continue
        ev = {"id": f"AE-{b.strftime('%Y-%m-%d')}-{seq:03d}", "kpi": head,
              "window": _fmt_win(a, b), "window_dates": (str(a.date()), str(b.date())),
              "baseline_dates": (str(base[0].date()), str(base[1].date())),
              "magnitude": f"{D['total_pct']:+.1f}%", "impact_inr": int(abs(D["impact_inr"])),
              "type": typ, "scenario": "SWEEP",
              "decomposition": {"conversion": f"{D['conversion']:+.1f}pp",
                                "traffic": f"{D['traffic']:+.1f}pp",
                                "price": f"{D['price']:+.1f}pp", "mix": f"{D['mix']:+.1f}pp",
                                "interaction": f"{D['interaction']:+.1f}pp"},
              "gate_trace": [
                  f"persistence: {hm['ndays']} consecutive days (min {CONTRACTS[head]['gate']['persistence_n']})",
                  f"ensemble {hm['votes']}/3 (weekday-z, STL residual, CUSUM) — |z|max {hm['zmax']:.1f}",
                  f"freshness: window within effective watermark {hm['eff_fresh']}",
                  "calendar: not festive-explained" + (f" (checked vs '{hm['holiday']}')" if hm["holiday"] else ""),
                  f"materiality ₹{abs(hm['impact_inr'])/1e5:.2f}L >= "
                  f"₹{CONTRACTS[head]['thresholds']['min_impact_inr']/1e5:.2f}L",
                  "coherence: corroborated by " + ", ".join(children) if children
                  else "coherence: single-KPI movement (lineage checked)"]}
        confirmed.append(ev)

    confirmed.sort(key=lambda e: -e["impact_inr"])
    return confirmed, rejections, flags

# computed once at import; call run() again after regenerating data
CONFIRMED, REJECTIONS, DAY_FLAGS = run()
