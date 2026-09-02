"""Computed analytics over trend_lens_data — LIVE.
identity decomposition, window statistics, analog launch bands.
Everything here is derived from the generated dataset at import; nothing hardcoded.
(Full STL/CUSUM detection replaces the window-comparison approach below.)"""
import os, sqlite3
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "trend_lens_data")

con = sqlite3.connect(os.path.join(DATA, "sales.db"))
ORDERS = pd.read_sql("SELECT * FROM orders", con)
RETURNS = pd.read_sql("SELECT * FROM returns", con)
INV = pd.read_sql("SELECT * FROM inventory", con)
con.close()
WEB = pd.read_parquet(os.path.join(DATA, "web_analytics.parquet"))
MKT_RAW = pd.read_excel(os.path.join(DATA, "marketing.xlsx"), sheet_name="campaigns")
NEWS = {}
for f in sorted(os.listdir(os.path.join(DATA, "news"))):
    if f.endswith(".txt"):
        NEWS[f] = open(os.path.join(DATA, "news", f)).read()

C = ORDERS[ORDERS.payment_status == "completed"].copy()
C["date"] = pd.to_datetime(C.date)
C["realized"] = C.list_price * (1 - C.discount_pct)
C["gmv"] = C.qty * C.realized
WEB["date"] = pd.to_datetime(WEB.date)
INV["date"] = pd.to_datetime(INV.date)

def snippet(name, n=400):
    return NEWS.get(name, "")[:n]

def clean_mkt():
    m = MKT_RAW.drop_duplicates()
    s, e = pd.to_datetime(m.start_date, errors="coerce"), pd.to_datetime(m.end_date, errors="coerce")
    return m[(e >= s) & (e < s + pd.Timedelta("180d"))]

def _slice(a, b, region=None):
    n = (pd.Timestamp(b) - pd.Timestamp(a)).days + 1
    o = C[(C.date >= a) & (C.date <= b)]
    w = WEB[(WEB.date >= a) & (WEB.date <= b)]
    if region:                      # row-level entitlement applied before any aggregation
        o = o[o.region == region]; w = w[w.region == region]
    return o, w, n

def window_stats(a, b, region=None):
    o, w, n = _slice(a, b, region)
    gmv = o.gmv.sum() / n
    orders = o.order_id.nunique() / n
    sess = w.sessions.sum() / n
    return {"gmv": gmv, "orders": orders, "sessions": sess,
            "cr": orders / sess if sess else 0, "aov": gmv / orders if orders else 0,
            "organic": w[w.channel == "organic"].sessions.sum() / n,
            "paid": w[w.channel == "paid"].sessions.sum() / n,
            "discount": o.discount_pct.mean(), "days": n,
            "started": w.checkouts_started.sum(), "completed": w.checkouts_completed.sum()}

def decompose(pre, post, region=None):
    """identity split of GMV = Sessions x CR x AOV (first-order contributions,
    additive by construction), plus AOV -> price/mix shift-share rescaled to the AOV term."""
    a, b = window_stats(*pre, region=region), window_stats(*post, region=region)
    total_pct = (b["gmv"] / a["gmv"] - 1) * 100
    traffic = (b["sessions"] / a["sessions"] - 1) * 100
    conversion = (b["cr"] / a["cr"] - 1) * 100
    aov_pct = (b["aov"] / a["aov"] - 1) * 100
    inter = total_pct - (traffic + conversion + aov_pct)

    def cat(win):
        o, _w, n = _slice(*win, region=region)
        g = o.groupby("category").agg(orders=("order_id", "nunique"), rev=("gmv", "sum"), q=("qty", "sum"))
        g["price"] = g.rev / g.q; g["w"] = g.orders / g.orders.sum(); return g
    p0, p1 = cat(pre), cat(post)
    ks = p0.index.intersection(p1.index)
    raw_price = sum(p0.loc[k, "w"] * (p1.loc[k, "price"] - p0.loc[k, "price"]) for k in ks) / a["aov"] * 100
    raw_mix = sum((p1.loc[k, "w"] - p0.loc[k, "w"]) * p0.loc[k, "price"] for k in ks) / a["aov"] * 100
    tot_pm = raw_price + raw_mix
    k_ = aov_pct / tot_pm if abs(tot_pm) > 1e-9 else 1.0     # rescale so price + mix == aov term
    price, mix = raw_price * k_, raw_mix * k_
    return {"total_pct": total_pct, "traffic": traffic, "conversion": conversion,
            "aov": aov_pct, "price": price, "mix": mix, "interaction": inter,
            "impact_inr": (a["gmv"] - b["gmv"]) * b["days"], "pre": a, "post": b}

def channel_shift(pre, post):
    a, b = window_stats(*pre), window_stats(*post)
    return {"organic_pct": (b["organic"] / a["organic"] - 1) * 100,
            "paid_pct": (b["paid"] / a["paid"] - 1) * 100}

def funnel_ratio(a, b):
    w = WEB[(WEB.date >= a) & (WEB.date <= b)]
    return w.checkouts_completed.sum() / max(w.checkouts_started.sum(), 1)

def stockout_days(sku, region, a, b):
    x = INV[(INV.sku == sku) & (INV.region == region) & (INV.date >= a) & (INV.date <= b)]
    return int((x.stock_on_hand == 0).sum())

def launch_curve(sku, days=36, win=5):
    x = C[C.sku == sku]
    if x.empty: return None, None
    l = x.date.min()
    d = x.groupby("date").order_id.nunique().reindex(pd.date_range(l, l + pd.Timedelta(days=days - 1)), fill_value=0)
    sm = d.rolling(win, center=True, min_periods=2).mean()
    return (sm / sm.mean()).values, l

def analog_band(target, analogs, floor=0.45, days=36):
    """Band from analog launches; width floored because only a few analogs exist (sparse history)."""
    tc, tl = launch_curve(target, days)
    cs = [launch_curve(a, days)[0] for a in analogs]
    cs = [c for c in cs if c is not None]
    mean = np.mean(cs, axis=0); spread = (np.max(cs, axis=0) - np.min(cs, axis=0)) / 2
    lo = mean - np.maximum(spread, floor * mean); hi = mean + np.maximum(spread, floor * mean)
    return {"target": tc, "launch": tl, "lo": lo, "hi": hi, "analog_curves": cs,
            "in_band": [bool(lo[k] <= tc[k] <= hi[k]) for k in range(len(tc))]}


def daily_series(region=None):
    """Daily KPI series across the whole dataset window (for history charts)."""
    o = C if region is None else C[C.region == region]
    w = WEB if region is None else WEB[WEB.region == region]
    gmv = o.groupby("date").gmv.sum()
    orders = o.groupby("date").order_id.nunique()
    sess = w.groupby("date").sessions.sum()
    idx = pd.date_range(C.date.min(), C.date.max(), freq="D")
    df = pd.DataFrame({"GMV": gmv, "Orders": orders, "Sessions": sess}).reindex(idx)
    df["AOV"] = df.GMV / df.Orders
    df["Conversion rate %"] = (df.Orders / df.Sessions) * 100
    return df

def coverage():
    """Dataset coverage facts for the UI header."""
    start, end = C.date.min(), C.date.max()
    return {"start": str(start.date()), "end": str(end.date()),
            "days": int((end - start).days) + 1,
            "months": round(((end - start).days + 1) / 30.44, 1),
            "order_rows": int(len(ORDERS)), "orders": int(C.order_id.nunique()),
            "web_rows": int(len(WEB)), "inventory_rows": int(len(INV)),
            "campaign_rows": int(len(MKT_RAW)), "news_files": len(NEWS),
            "skus": int(ORDERS.sku.nunique()), "regions": sorted(ORDERS.region.unique()),
            "categories": sorted(ORDERS.category.unique())}

EVENT_WINDOWS = [
    ("S1a coupon expiry", "2026-07-15", "2026-07-21"),
    ("S1b competitor sale", "2026-07-15", "2026-07-21"),
    ("S2 ambiguous week", "2026-08-17", "2026-08-23"),
    ("S3 AirPro launch", "2026-07-25", "2026-08-26"),
    ("Gateway blip", "2026-08-05", "2026-08-05"),
    ("Stockout", "2026-06-10", "2026-06-13"),
]
