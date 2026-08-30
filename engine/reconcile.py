"""Stage 1.2 — Freshness & grain reconciler (LIVE on sample files).
Watermarks from actual file state; effective freshness via graph walk."""
import sqlite3, os, glob
import pandas as pd
from .core import CONTRACTS
from . import graph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "trend_lens_data")

def load_sources():
    con = sqlite3.connect(os.path.join(DATA, "sales.db"))
    orders = pd.read_sql("SELECT * FROM orders", con)
    returns = pd.read_sql("SELECT * FROM returns", con)
    inv = pd.read_sql("SELECT * FROM inventory", con)
    con.close()
    web = pd.read_parquet(os.path.join(DATA, "web_analytics.parquet"))
    mkt_raw = pd.read_excel(os.path.join(DATA, "marketing.xlsx"), sheet_name="campaigns")
    news = {os.path.basename(f): open(f).read()
            for f in glob.glob(os.path.join(DATA, "news", "*.txt"))}
    return orders, returns, inv, web, mkt_raw, news

def clean_marketing(mkt):
    """Quality repairs: dedup planted duplicate, drop date-typo rows (>180d span)."""
    mkt = mkt.drop_duplicates()
    s, e = pd.to_datetime(mkt.start_date, errors="coerce"), pd.to_datetime(mkt.end_date, errors="coerce")
    return mkt[(e >= s) & (e < s + pd.Timedelta("180d"))]

def watermarks(orders, web, mkt):
    return {
        "sales.orders": str(orders.date.max()),
        "web_analytics.parquet": str(web.date.max())[:10],   # T+1: trails orders
        "marketing.xlsx": str(pd.to_datetime(clean_marketing(mkt).start_date).max())[:10],
    }

def effective_freshness(kpi, wm):
    srcs = ({s for s in graph.ancestors(kpi, kinds=("derived_from", "fed_by"))} |
            set(CONTRACTS.get(kpi, {}).get("sources", []))) & set(wm)
    if not srcs:
        srcs = set(CONTRACTS.get(kpi, {}).get("sources", [])) & set(wm)
    return min((wm[s] for s in srcs), default=None), sorted(srcs)

def freshness_report():
    orders, returns, inv, web, mkt, news = load_sources()
    wm = watermarks(orders, web, mkt)
    rows = []
    for kpi, c in CONTRACTS.items():
        eff, srcs = effective_freshness(kpi, wm)
        note = ""
        if c.get("settling_lag_days"):
            note = f"provisional {c['settling_lag_days']}d (returns settle)"
        if kpi == "conversion_rate" and wm["web_analytics.parquet"] < wm["sales.orders"]:
            note = "parents mismatched: sessions T+1 behind orders -> latest day not computable"
        rows.append({"kpi": kpi, "effective_freshness": eff, "sources": srcs, "note": note})
    return pd.DataFrame(rows), wm
