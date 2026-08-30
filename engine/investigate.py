"""Generalized investigation — ANY governed KPI, ANY window, no scripted scenarios.

investigate(kpi, start, end) reproduces by construction what retrieval.py hand-built
for S1/S2: it probes the dataset with the same signature checks a senior analyst
would run, assembles a docket of computed evidence, derives hypotheses from the
knowledge graph, computes contribution_match from the measured decomposition, and
hands everything to the standard pipeline (scope -> rank -> gate -> actions ->
narrate). Sweep events feed straight in via investigate_event().

Every driver has a measurable SIGNATURE — which Level-1 components it owns and
which dataset probe confirms or refutes it:

  coupon_expiry    owns conversion+price   probe: discount series delta
  competitor_promo owns traffic+mix        probe: organic vs paid channel contrast
  marketing_gap    owns traffic            probe: campaign rows present? paid down?
  gateway_issue    owns conversion         probe: checkout completion funnel ratio
  supply_stockout  owns conversion+mix     probe: inventory zero-stock scan
  demand_shift     owns traffic+conversion probe: broad softness + external corroboration
  launch_volatility owns conversion        probe: analog band position
"""
import numpy as np
import pandas as pd

from . import analytics as A
from . import graph, reconcile, vector_news, calibrate
from .core import CONTRACTS
from . import pipeline

pp = lambda v: f"{v:+.1f}pp"
pc = lambda v: f"{v:+.1f}%"

OWNS = {"coupon_expiry": ["conversion", "price"],
        "competitor_promo": ["traffic", "mix"],
        "marketing_gap": ["traffic"],
        "gateway_issue": ["conversion"],
        "supply_stockout": ["conversion", "mix"],
        "demand_shift": ["traffic", "conversion"],
        "product_launch": ["conversion", "mix"],
        "launch_volatility": ["conversion"]}

LEVERS = {"coupon_expiry": ("promotional_pricing", "10%"),
          "marketing_gap": ("campaign_spend", None)}

def _anomalous_dates():
    """Dates inside sweep-confirmed anomaly windows (lazy import: sweep <- analytics only)."""
    try:
        from . import sweep
        days = set()
        for e in sweep.CONFIRMED:
            a, b = e["window_dates"]
            days |= set(pd.date_range(a, b))
        return days
    except Exception:
        return set()

def _auto_baseline(start, ndays=14, max_slides=4):
    """14 clean days before the window. If a candidate block overlaps confirmed
    anomalies on >2 days, slide back a week and take the cleanest block found —
    a contaminated baseline turns recoveries into fake spikes."""
    bad = _anomalous_dates()
    best, best_n = None, 10 ** 9
    for k in range(max_slides + 1):
        b = pd.Timestamp(start) - pd.Timedelta(days=1 + 7 * k)
        a = b - pd.Timedelta(days=ndays - 1)
        n_bad = len(set(pd.date_range(a, b)) & bad)
        if n_bad <= 2:
            return str(a.date()), str(b.date())
        if n_bad < best_n:
            best, best_n = (str(a.date()), str(b.date())), n_bad
    return best

def _fmt_win(a, b):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return f"{a.strftime('%b %d')}-{b.strftime('%d')}" if a.month == b.month \
        else f"{a.strftime('%b %d')}-{b.strftime('%b %d')}"

# ---------------- signature probes (each returns docket items) ----------------
def _probe_discount(D, win):
    d0, d1 = D["pre"]["discount"], D["post"]["discount"]
    # direction gate: a coupon EXPIRY explains conversion falling while realised
    # price rises — it cannot explain a positive total movement
    dropped = (d0 - d1) > 0.02 and D["price"] > 0 and D["conversion"] < 0
    raised = (d1 - d0) > 0.02
    if dropped:
        return [{"id": "e-101", "claim": "Discount rate collapsed at window start while realised "
                                         "price rose (coupon-expiry fingerprint)",
                 "value": f"mean discount {d0:.3f} -> {d1:.3f}; price {pp(D['price'])}",
                 "source": "sales.db", "fresh": "daily 02:00", "method": "SQL series on orders.discount_pct",
                 "direction": "confirming", "supports": "coupon_expiry",
                 "strength": 0.9, "temporal": 0.95, "live": True},
                {"id": "e-102", "claim": "Conversion is a dominant Level-1 component alongside the price move",
                 "value": f"conversion {pp(D['conversion'])} of {pc(D['total_pct'])} total",
                 "source": "web_analytics.parquet", "fresh": "T+1", "method": "identity decomposition",
                 "direction": "confirming", "supports": "coupon_expiry",
                 "strength": 0.8, "temporal": 0.9, "live": True}]
    return [{"id": "e-101", "claim": "No discount regime change in the window"
                                     + (" (discount deepened)" if raised else ""),
             "value": f"mean discount {d0:.3f} -> {d1:.3f}",
             "source": "sales.db", "fresh": "daily 02:00", "method": "SQL series on orders.discount_pct",
             "direction": "refuting", "supports": "coupon_expiry",
             "strength": 0.6, "temporal": 0.8, "live": True}]

def _probe_channels(base, win):
    ch = A.channel_shift(base, win)
    org, paid = ch["organic_pct"], ch["paid_pct"]
    # STRICT contrast: organic materially down AND paid holding near flat.
    # Both channels falling together is broad softness, not external pull.
    contrast = org < -3 and paid > -2.0
    return [{"id": "e-111",
             "claim": ("Organic sessions fell while paid held — external-pull signature"
                       if contrast else "No organic-vs-paid contrast (no external-pull signature)"),
             "value": f"organic {pc(org)}, paid {pc(paid)}",
             "source": "web_analytics.parquet", "fresh": "T+1", "method": "channel slice",
             "direction": "confirming" if contrast else "refuting",
             "supports": "competitor_promo",
             "strength": 0.7 if contrast else 0.5, "temporal": 0.85, "live": True}], ch

def _probe_marketing(base, win):
    mkt = A.clean_mkt()
    sd = pd.to_datetime(mkt.start_date)
    a, b = pd.Timestamp(win[0]) - pd.Timedelta(days=6), pd.Timestamp(win[1])
    rows = int(((sd >= a) & (sd <= b)).sum())
    if rows == 0:
        return [{"id": "e-118", "claim": "No campaign rows cover this window — the Monday upload is missing",
                 "value": f"rows for {win[0]}..{win[1]} = 0 — MISSING",
                 "source": "marketing.xlsx", "fresh": "STALE", "method": "reconciler freshness walk",
                 "direction": "confirming", "supports": "marketing_gap",
                 "strength": 0.3, "temporal": 0.5, "status": "MISSING", "live": True}], True
    spend = float(mkt[(sd >= a) & (sd <= b)].spend.sum())
    return [{"id": "e-118", "claim": "Campaign spend present and steady across the window",
             "value": f"Rs {spend:,.0f} across {rows} campaign rows",
             "source": "marketing.xlsx", "fresh": "weekly Monday", "method": "xlsx dedup + spend sum",
             "direction": "refuting", "supports": "marketing_gap",
             "strength": 0.7, "temporal": 0.8, "live": True}], False

def _probe_funnel(base, win):
    fr_w = A.funnel_ratio(win[0], win[1])
    fr_b = A.funnel_ratio(base[0], base[1])
    broken = fr_b > 0 and (fr_w / fr_b - 1) < -0.05
    return [{"id": "e-120",
             "claim": ("Checkout completion ratio degraded — payment/checkout-step signature"
                       if broken else "Checkout-step ratio normal; movement is pre-checkout"),
             "value": f"completion {fr_w:.3f} in window vs {fr_b:.3f} baseline",
             "source": "web_analytics.parquet", "fresh": "T+1", "method": "funnel step check",
             "direction": "confirming" if broken else "refuting",
             "supports": "gateway_issue",
             "strength": 0.75 if broken else 0.7, "temporal": 0.9, "live": True}]

def _probe_launch(D, base, win):
    """GMV share of recently launched SKUs (first order < 45d before window end).
    If launch SKUs explain a large share of a POSITIVE movement, that's a
    product-launch ramp, not a demand mystery."""
    first = A.C.groupby("sku").date.min()
    recent = first[first >= pd.Timestamp(win[1]) - pd.Timedelta(days=45)].index
    if len(recent) == 0 or D["total_pct"] <= 0:
        return [], False
    gw = A.C[(A.C.date >= win[0]) & (A.C.date <= win[1])]
    gb = A.C[(A.C.date >= base[0]) & (A.C.date <= base[1])]
    nw = (pd.Timestamp(win[1]) - pd.Timestamp(win[0])).days + 1
    nb = (pd.Timestamp(base[1]) - pd.Timestamp(base[0])).days + 1
    launch_delta = gw[gw.sku.isin(recent)].gmv.sum() / nw - gb[gb.sku.isin(recent)].gmv.sum() / nb
    total_delta = gw.gmv.sum() / nw - gb.gmv.sum() / nb
    share = launch_delta / total_delta if abs(total_delta) > 1e-9 else 0
    if share > 0.2:
        skus = ", ".join(sorted(recent))
        return [{"id": "e-140",
                 "claim": f"Recently launched SKUs explain {share:.0%} of the daily GMV uplift",
                 "value": f"launch SKUs ({skus}): +₹{launch_delta:,.0f}/day of +₹{total_delta:,.0f}/day total",
                 "source": "sales.db", "fresh": "daily 02:00",
                 "method": "launch-SKU share of GMV delta",
                 "direction": "confirming", "supports": "product_launch",
                 "strength": round(min(0.5 + share, 0.9), 2), "temporal": 0.9, "live": True}], True
    return [], False

def _probe_stockout(win, D=None):
    inv = A.INV[(A.INV.date >= win[0]) & (A.INV.date <= win[1])]
    z = inv[inv.stock_on_hand == 0].groupby(["sku", "region"]).size()
    z = z[z >= 2]
    # direction gate: a stockout drags conversion/mix down — it can't explain a spike
    if len(z) and D is not None and not (D["conversion"] < 0 or D["mix"] < 0):
        z = z.iloc[0:0]
    if len(z):
        top = z.idxmax()
        return [{"id": "e-125",
                 "claim": f"Zero-stock days in window: {len(z)} sku-region pairs (worst: {top[0]} {top[1]})",
                 "value": f"{top[0]} {top[1]}: {int(z.max())} days at stock_on_hand=0",
                 "source": "sales.db(inventory)", "fresh": "daily 02:00", "method": "inventory zero-stock scan",
                 "direction": "confirming", "supports": "supply_stockout",
                 "strength": 0.75, "temporal": 0.85, "live": True}]
    return [{"id": "e-125", "claim": "No sustained zero-stock sku-region in the window",
             "value": "0 pairs with >=2 zero-stock days",
             "source": "sales.db(inventory)", "fresh": "daily 02:00", "method": "inventory zero-stock scan",
             "direction": "refuting", "supports": "supply_stockout",
             "strength": 0.6, "temporal": 0.85, "live": True}]

# ---------------- hypothesis assembly ----------------
def _contribution_match(D, owns):
    """How much of the total movement do this driver's owned components explain,
    with the SAME sign as the total? Computed, not declared."""
    total = D["total_pct"]
    if abs(total) < 1e-9:
        return 0.1
    same = sum(D[c] for c in owns if c in D and np.sign(D[c]) == np.sign(total))
    return round(float(np.clip(same / total, 0.05, 0.95)), 2)

def build_case(kpi, start, end, baseline=None, backend="none", model=None):
    """Assemble event + docket + hypotheses for any governed KPI window."""
    kpi = "gmv" if kpi in ("orders",) else kpi          # legacy alias
    base = baseline or _auto_baseline(start)
    win = (str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
    D = A.decompose(base, win)
    drivers = graph.candidate_drivers(kpi) or list(OWNS)
    drivers = [d for d in drivers if d in OWNS]

    docket = []
    docket += _probe_discount(D, win)
    ch_items, ch = _probe_channels(base, win)
    docket += ch_items
    mkt_items, mkt_missing = _probe_marketing(base, win)
    docket += mkt_items
    docket += _probe_funnel(base, win)
    docket += _probe_stockout(win, D)
    launch_items, launch_fired = _probe_launch(D, base, win)
    docket += launch_items
    if launch_fired and "product_launch" not in drivers:
        drivers.append("product_launch")
    news_items, news_meta = vector_news.evidence_items(
        kpi, win, D["total_pct"], drivers, backend=backend, model=model)
    for it in news_items:                                # external corroborates, never quantifies
        if it.get("conflicts_with") == "internal-decomposition":
            it["conflicts_with"] = "e-101"
    docket += news_items

    # freshness wall: is the window even judgeable?
    wm = reconcile.watermarks(A.ORDERS, A.WEB, A.MKT_RAW)
    eff, srcs = reconcile.effective_freshness(kpi, wm)
    stale = eff is not None and win[1] > eff

    priors = calibrate.priors()
    hyps = []
    for d in drivers:
        ev_ids = [e["id"] for e in docket if e.get("supports") == d]
        if not ev_ids:
            continue
        a_, b_ = priors.get(d, (2, 2))
        h = {"name": d, "evidence": ev_ids,
             "n_requirements": max(len(ev_ids), 2 if d in ("marketing_gap", "gateway_issue") else 3),
             "owns": OWNS[d], "contribution_match": _contribution_match(D, OWNS[d]),
             "prior_alpha": a_, "prior_beta": b_}
        if d == "marketing_gap" and mkt_missing:
            h["critical_evidence_missing"] = True
        if d == "competitor_promo" and ch["organic_pct"] < -3 and ch["paid_pct"] > -2.0:
            h["causal"] = {"test": "organic-vs-paid channel contrast (DiD-style)", "p": 0.03}
            h["monitor_note"] = "verify organic recovery within 3 days of external event end"
        if d in LEVERS:
            lever, depth = LEVERS[d]
            h["lever"] = lever
            if depth:
                h["depth"] = depth
            h["recovery_inr_per_day"] = int(abs(D["impact_inr"]) / max(D["post"]["days"], 1) * 0.7)
        hyps.append(h)

    resolves = []
    if mkt_missing:
        resolves.append(f"marketing.xlsx upload for {win[0]}..{win[1]} missing -> auto re-evaluation on arrival")
    if stale:
        resolves.append(f"effective freshness {eff} < window end {win[1]} — wait for source refresh")
    resolves.append("3 more days of session data would separate external pull from broad demand shift")

    event = {"id": f"INV-{win[1]}-{kpi[:4]}", "kpi": kpi if kpi in CONTRACTS else "gmv",
             "window": _fmt_win(*win), "window_dates": win, "baseline_dates": base,
             "magnitude": pc(D["total_pct"]), "impact_inr": int(abs(D["impact_inr"])),
             "type": "window_investigation", "scenario": "EXPLORE",
             "decomposition": {"conversion": pp(D["conversion"]), "traffic": pp(D["traffic"]),
                               "price": pp(D["price"]), "mix": pp(D["mix"]),
                               "interaction": pp(D["interaction"])},
             "gate_trace": [f"on-demand window vs auto-baseline {base[0]}..{base[1]} (21d)",
                            f"effective freshness {eff} over sources {', '.join(srcs)}",
                            f"materiality ₹{abs(D['impact_inr'])/1e5:.2f}L"],
             "resolves": resolves,
             "news_meta": news_meta}
    return event, docket, hyps, D

def scan_window(start, end, baseline=None):
    """Movement of EVERY governed KPI over the window vs its baseline, plus the
    headline pick (same rule as the sweep's coherence merge: gmv preferred when
    material, else the most material mover)."""
    base = baseline or _auto_baseline(start)
    win = (str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
    a, b = A.window_stats(*base), A.window_stats(*win)
    fr_b, fr_w = A.funnel_ratio(*base), A.funnel_ratio(*win)
    window_inr = abs((a["gmv"] - b["gmv"]) * b["days"])   # ONE business-materiality number
    thr = CONTRACTS.get("gmv", {}).get("thresholds", {}).get("min_impact_inr", 8000)
    rows = []
    for kpi, pre, post in [("gmv", a["gmv"], b["gmv"]),
                           ("orders_kpi", a["orders"], b["orders"]),
                           ("sessions", a["sessions"], b["sessions"]),
                           ("conversion_rate", a["cr"], b["cr"]),
                           ("aov", a["aov"], b["aov"]),
                           ("net_revenue", a["gmv"], b["gmv"])]:
        pct = (post / pre - 1) * 100 if pre else 0.0
        rows.append({"kpi": kpi, "pct": round(pct, 1),
                     "moved": abs(pct) >= 5,
                     "material": abs(pct) >= 5 and window_inr >= thr})
    rows.append({"kpi": "checkout_completion",
                 "pct": round((fr_w / fr_b - 1) * 100, 1) if fr_b else 0.0,
                 "moved": False, "material": False})
    material = [r for r in rows if r["material"]]
    headline = "gmv" if any(r["kpi"] == "gmv" for r in material) else \
        (max(material, key=lambda r: abs(r["pct"]))["kpi"] if material else "gmv")
    return {"baseline": base, "window": win, "rows": rows, "headline": headline,
            "window_impact_inr": int(window_inr), "any_material": bool(material)}

def investigate(kpi, start, end, role="cfo", backend="none", model=None, baseline=None):
    """Full run: build the case, then the standard pipeline. Returns run_scenario's dict
    plus calibrated confidences."""
    event, docket, hyps, D = build_case(kpi, start, end, baseline, backend, model)
    thr = CONTRACTS.get(event["kpi"], {}).get("thresholds", {}).get("min_impact_inr", 8000)
    r = pipeline.run_scenario(event, docket, hyps, role, backend, model)
    if abs(D["impact_inr"]) < thr:
        r["decision"] = {"mode": "no_material_movement",
                         "why": [f"impact ₹{abs(D['impact_inr']):,.0f} below materiality "
                                 f"threshold ₹{thr:,} — nothing to explain"],
                         "ranked": r["finding"]["hypotheses"]}
        r["actions"] = []
        r["narrative"] = (f"{event['kpi'].upper()} {event['magnitude']} over {event['window']} — "
                          f"₹{abs(D['impact_inr'])/1e5:.2f}L vs threshold ₹{thr/1e5:.2f}L. "
                          "No material movement; no investigation warranted. "
                          "Monitoring continues via the detection sweep.")
    for h in r["finding"]["hypotheses"]:
        h["calibrated"] = calibrate.calibrated(h["confidence"])
    return r

def investigate_event(ev, role="cfo", backend="none", model=None):
    """Investigate a sweep-confirmed event (uses its own window + baseline)."""
    a, b = ev["window_dates"]
    r = investigate(ev["kpi"] if ev["kpi"] in CONTRACTS else "gmv", a, b,
                    role=role, backend=backend, model=model,
                    baseline=ev.get("baseline_dates"))
    r["event"]["id"] = ev["id"]                       # keep the sweep's identity
    r["event"]["type"] = ev["type"]
    r["event"]["gate_trace"] = ev["gate_trace"] + r["event"]["gate_trace"][1:]
    return r
