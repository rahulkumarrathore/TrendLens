"""TrendLens full dataset generator — 6 months (Mar 1 2026 -> Aug 26 2026), ~2000 order rows.
Plants: S1 (coupon expiry + competitor sale), S2 (ambiguous week + missing xlsx),
S3 (AirPro sparse launch + 2 in-window analogs), gateway blip, stockouts, 2% inventory gaps.
Deterministic (--seed). Writes the 4 sources in place + ground_truth.yaml + holidays.yaml.
"""
import argparse, sqlite3, os, shutil
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "trend_lens_data")

REGIONS = ["North", "South"]
CATS = {"Electronics": 14, "Fashion": 14, "Home": 12}   # ~40 skus
PRICE_BAND = {"Electronics": (900, 4500), "Fashion": (300, 3000), "Home": (400, 2500)}
COGS_FRAC = (0.55, 0.75)

ASOF = pd.Timestamp("2026-08-26")        # "this morning"
START = pd.Timestamp("2026-03-01")
ORDERS_WM = ASOF - pd.Timedelta("1d")    # loaded 02:00 -> complete thru Aug 25
WEB_WM = ASOF - pd.Timedelta("2d")       # T+1 export  -> complete thru Aug 24

COUPON_START, COUPON_END = pd.Timestamp("2026-07-25"), pd.Timestamp("2026-08-11")  # dies Aug 12 00:00
COMP_SALE = (pd.Timestamp("2026-08-12"), pd.Timestamp("2026-08-14"))
S2_WEEK = (pd.Timestamp("2026-08-17"), pd.Timestamp("2026-08-21"))
GATEWAY_DAY = pd.Timestamp("2026-08-13")
LAUNCHES = {"ELEC-AIRPRO-BLK": pd.Timestamp("2026-08-03"),
            "ELEC-SOUNDMAX-WHT": pd.Timestamp("2026-03-10"),
            "ELEC-BASSLITE-BLU": pd.Timestamp("2026-05-01")}

HOLIDAYS = {"2026-07-01/2026-07-15": ("EOSS", 1.25, ["Fashion", "Home"]),
            "2026-08-22/2026-08-28": ("Raksha Bandhan", 1.30, ["Fashion", "Home"])}

def launch_curve(day_idx, peak=1.0):
    """Spike -> decay -> plateau shape shared by all launches (analog family)."""
    ramp = 1 - np.exp(-day_idx / 3.0)
    decay = 0.55 + 0.45 * np.exp(-np.maximum(day_idx - 7, 0) / 12.0)
    return peak * ramp * decay

def build_skus(rng):
    skus = []
    for cat, n in CATS.items():
        pre = {"Electronics": "ELEC", "Fashion": "FASH", "Home": "HOME"}[cat]
        for i in range(n):
            lo, hi = PRICE_BAND[cat]
            skus.append({"sku": f"{pre}-SKU{i:02d}", "category": cat,
                         "list_price": float(rng.integers(lo, hi))})
    named = [("ELEC-AIRPRO-BLK", "Electronics", 3499.0),
             ("ELEC-SOUNDMAX-WHT", "Electronics", 2999.0),
             ("ELEC-BASSLITE-BLU", "Electronics", 1799.0)]
    for s, c, p in named:
        skus[len(skus) - len(named) + named.index((s, c, p))] = {"sku": s, "category": c, "list_price": p}
    return pd.DataFrame(skus)

def day_multipliers(d, cat):
    m = 1.0
    m *= [0.92, 0.96, 0.98, 1.0, 1.05, 1.22, 1.18][d.dayofweek]          # weekly
    m *= 1.0 + 0.18 * (d - START).days / 180                              # trend
    for k, (name, up, cats) in HOLIDAYS.items():
        a, b = (pd.Timestamp(x) for x in k.split("/"))
        if a <= d <= b and cat in cats:
            m *= up
    return m

def gen(seed=42, n_orders_target=2000):
    rng = np.random.default_rng(seed)
    skus = build_skus(rng)
    days = pd.date_range(START, ORDERS_WM, freq="D")

    # ---------- sessions first (orders derive from funnel) ----------
    web_rows, daily_sessions = [], {}
    for d in days[days <= WEB_WM]:
        for r in REGIONS:
            for ch in ["paid", "organic"]:
                for dev in ["mobile", "desktop"]:
                    base = 95 if dev == "mobile" else 45
                    base *= 1.12 if r == "North" else 1.0
                    base *= 1.0 if ch == "organic" else 0.85
                    s = base * day_multipliers(d, "Fashion") * rng.lognormal(0, 0.06)
                    if ch == "organic" and COMP_SALE[0] <= d <= COMP_SALE[1]:
                        s *= 1 - 0.08                                     # S1b competitor pull
                    if S2_WEEK[0] <= d <= S2_WEEK[1]:
                        s *= 1 - 0.028                                    # S2 mild broad softness
                    s = int(s)
                    carts = int(s * rng.uniform(0.09, 0.12))
                    started = int(carts * rng.uniform(0.42, 0.5))
                    cr_base = rng.uniform(0.40, 0.48)
                    if COUPON_START <= d <= COUPON_END: cr_base *= 1.13   # coupon lifts completion
                    if d > COUPON_END: cr_base *= 1.13 * (1 - 0.11)       # S1a expiry drop
                    if d == GATEWAY_DAY and dev == "mobile": cr_base *= 0.72  # gateway blip
                    done = int(started * min(cr_base, 0.95))
                    web_rows.append((d, r, ch, dev, s, carts, started, done))
                    daily_sessions[d] = daily_sessions.get(d, 0) + s
    web = pd.DataFrame(web_rows, columns=["date", "region", "channel", "device",
                                          "sessions", "add_to_carts", "checkouts_started", "checkouts_completed"])

    # ---------- orders (~n_orders_target item rows) ----------
    weights = np.array([day_multipliers(d, "Fashion") for d in days]); weights /= weights.sum()
    per_day = rng.multinomial(int(n_orders_target * 0.94), weights)       # rest = multi-item extras
    rows, oid = [], 5000
    for d, n in zip(days, per_day):
        n = max(n, 1)
        cr_mult = 1.13 if COUPON_START <= d <= COUPON_END else (1.13 * (1 - 0.11) if d > COUPON_END else 1.0)
        if COMP_SALE[0] <= d <= COMP_SALE[1]: cr_mult *= 1 - 0.035        # electronics traffic loss bleeds in
        if S2_WEEK[0] <= d <= S2_WEEK[1]: cr_mult *= 1 - 0.022
        n = max(int(round(n * cr_mult)), 1)
        for _ in range(n):
            oid += 1
            n_items = 1 + (rng.random() < 0.18)
            elec_w = 0.38
            if COMP_SALE[0] <= d <= COMP_SALE[1]: elec_w = 0.30           # mix shift
            for it in range(n_items):
                cat = rng.choice(["Electronics", "Fashion", "Home"], p=[elec_w, 0.62 - elec_w + 0.24, 0.38 - 0.24 + 0])
                pool = skus[skus.category == cat]
                pool = pool[[not (s in LAUNCHES and d < LAUNCHES[s]) for s in pool.sku]]
                row = pool.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
                sku = row.sku
                if cat == "Electronics":                                   # launch-curve weighting
                    for ls, ld in LAUNCHES.items():
                        if d >= ld and rng.random() < 0.22 * launch_curve((d - ld).days):
                            sku = ls; row = skus[skus.sku == ls].iloc[0]; break
                disc = float(np.clip(rng.normal(0.15, 0.03), 0.05, 0.25)) if COUPON_START <= d <= COUPON_END \
                    else float(np.clip(rng.normal(0.02, 0.02), 0, 0.08))
                lp = row.list_price
                cogs = round(lp * rng.uniform(*COGS_FRAC), 2)
                realized = lp * (1 - disc)
                rows.append((f"ORD-{oid}", d.strftime("%Y-%m-%d"), sku, row.category,
                             rng.choice(REGIONS, p=[0.55, 0.45]), int(rng.integers(1, 3)),
                             lp, round(disc, 3), round(realized - cogs, 2), cogs,
                             "cancelled" if rng.random() < 0.04 else "completed"))
    orders = pd.DataFrame(rows, columns=["order_id", "date", "sku", "category", "region", "qty",
                                         "list_price", "discount_pct", "margin", "cogs", "payment_status"])

    # ---------- returns ----------
    comp = orders[orders.payment_status == "completed"].drop_duplicates("order_id")
    ret_pick = comp.sample(frac=0.06, random_state=seed)
    returns = pd.DataFrame({
        "return_id": [f"RET-{7000+i}" for i in range(len(ret_pick))],
        "order_ref": ret_pick.order_id.values,
        "date": [(pd.Timestamp(x) + pd.Timedelta(days=int(rng.integers(3, 11)))).strftime("%Y-%m-%d")
                 for x in ret_pick.date],
        "reason": rng.choice(["damaged", "wrong_item", "customer_changed_mind", "size_issue"], len(ret_pick)),
    })
    returns = returns[pd.to_datetime(returns.date) <= ORDERS_WM]

    # ---------- inventory (daily snapshot, 2% missing, planted stockouts) ----------
    inv_rows = []
    stock = {(s, r): int(rng.integers(40, 140)) for s in skus.sku for r in REGIONS}
    sold = orders.groupby(["date", "sku", "region"]).qty.sum()
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        for (s, r), lvl in list(stock.items()):
            if s in LAUNCHES and d < LAUNCHES[s]: continue
            lvl = max(lvl - int(sold.get((ds, s, r), 0)), 0)
            if rng.random() < 0.02: lvl += int(rng.integers(30, 90))       # replenish
            stock[(s, r)] = lvl
            if rng.random() < 0.02: continue                               # 2% missing rows
            inv_rows.append((ds, s, r, lvl))
    inv = pd.DataFrame(inv_rows, columns=["date", "sku", "region", "stock_on_hand"])

    # ---------- marketing.xlsx (weekly Mondays; S2 Monday missing; recurring dirt) ----------
    mondays = pd.date_range(START, ASOF, freq="W-MON")
    camps = []
    for m in mondays:
        if m == pd.Timestamp("2026-08-17"): continue                       # S2: upload missed
        for name, ch, base in [("Always_On_Search", "google_ads", 30000),
                               ("Social_Prospecting", "meta_ads", 22000),
                               ("Category_Push", "google_ads", 15000)]:
            camps.append((f"{name}", ch, int(base * rng.uniform(0.85, 1.2)),
                          m.strftime("%Y-%m-%d"), (m + pd.Timedelta("6d")).strftime("%Y-%m-%d"),
                          int(base * rng.uniform(12, 18))))
    camps.append(camps[10])                                                # planted duplicate
    c = list(camps[20]); c[4] = "2062" + c[4][4:]; camps.append(tuple(c))  # planted 2062 typo
    camps.append(("AirPro_Launch_Awareness", "google_ads", 95000, "2026-08-03", "2026-08-23", 1500000))

    # ---------- write everything ----------
    os.makedirs(f"{DATA}/news", exist_ok=True)
    db = f"{DATA}/sales.db"
    if os.path.exists(db): os.remove(db)
    con = sqlite3.connect(db)
    orders.to_sql("orders", con, index=False)
    returns.to_sql("returns", con, index=False)
    inv.to_sql("inventory", con, index=False)
    con.close()
    web.to_parquet(f"{DATA}/web_analytics.parquet", index=False)

    wb = Workbook(); ws = wb.active; ws.title = "campaigns"
    ws.append(["campaign", "channel", "spend", "start_date", "end_date", "promised_impressions"])
    for cell in ws[1]: cell.font = Font(name="Arial", bold=True)
    for row in camps: ws.append(row)
    wb.save(f"{DATA}/marketing.xlsx")

    import yaml
    with open(f"{ROOT}/holidays.yaml", "w") as f:
        yaml.safe_dump({k: {"name": v[0], "uplift": v[1], "categories": v[2]}
                        for k, v in HOLIDAYS.items()}, f)
    gt = {
        "S1a_coupon_expiry": {"window": "2026-08-12+", "mechanism": "discount 15%->2%, CR x0.89",
                              "expected": {"conversion_pp": [-12, -10], "price_pp": [3, 5]}},
        "S1b_competitor_sale": {"window": "2026-08-12/14", "mechanism": "organic sessions -8%, Electronics-weighted",
                                "expected": {"traffic_pp": [-5.5, -3]}},
        "S2_ambiguous": {"window": "2026-08-17/21", "mechanism": "broad -2 to -3% softness, xlsx Monday missing",
                         "expected": "abstain (top conf < tau)"},
        "S3_airpro": {"launch": "2026-08-03", "expected": "day-18 dip within analog band, no alert"},
        "gateway_blip": {"window": "2026-08-13 mobile", "mechanism": "checkout completion x0.72",
                         "expected": "funnel-step localisation, spike type"},
        "analogs": {"SoundMax": "2026-03-10", "BassLite": "2026-05-01"},
        "seed": seed,
    }
    with open(f"{ROOT}/ground_truth.yaml", "w") as f:
        yaml.safe_dump(gt, f, sort_keys=False)
    return orders, returns, inv, web, len(camps)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--orders", type=int, default=2000)
    a = ap.parse_args()
    o, r, i, w, nc = gen(a.seed, a.orders)
    print(f"orders rows: {len(o)} | returns: {len(r)} | inventory: {len(i)} | web: {len(w)} | campaigns: {nc}")
