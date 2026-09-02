"""TrendLens synthetic data generator — 6-month window (2026-03-01 .. 2026-08-26).

Produces the 4 sources + ground_truth.yaml with every graded scenario planted:
  S1a coupon expiry (conversion + price fingerprint)   S1b competitor sale (organic traffic)
  S2 ambiguous week (abstention)                       S3 AirPro sparse launch (in-band dip)
  gateway blip (funnel-step)                           analog launches (SoundMax, BassLite)
Deterministic: --seed. Cadence-correct watermarks: --asof.
"""
import argparse, os, sqlite3, random, math
import numpy as np, pandas as pd, yaml
from openpyxl import Workbook
from openpyxl.styles import Font

P = argparse.ArgumentParser()
P.add_argument("--seed", type=int, default=42)
P.add_argument("--asof", default="2026-08-26")
P.add_argument("--orders", type=int, default=1250, help="approx total order-item rows")
P.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trend_lens_data"))
A = P.parse_args()
rng = np.random.default_rng(A.seed); random.seed(A.seed)

START, ASOF = pd.Timestamp("2026-03-01"), pd.Timestamp(A.asof)
DAYS = pd.date_range(START, ASOF, freq="D")
OUT = A.out; os.makedirs(os.path.join(OUT, "news"), exist_ok=True)
for _f in os.listdir(os.path.join(OUT, "news")):      # clear stale snippets
    if _f.endswith(".txt"): os.remove(os.path.join(OUT, "news", _f))

REGIONS = ["North", "South"]
CATS = ["Electronics", "Fashion", "Home"]

# ---------------- catalogue (~40 SKUs) ----------------
SKUS = {}
base = {"Electronics": (1900, 2600), "Fashion": (800, 1300), "Home": (700, 1150)}
names = {"Electronics": ["EARBUD","SPKR","PWRBNK","CABLE","MOUSE","KBRD","WATCH","CAM","HDMI","CHRGR"],
         "Fashion": ["TSHIRT","SNEAKER","JEANS","JACKET","KURTA","SAREE","BELT","CAP","SOCKS","SCARF"],
         "Home": ["BLENDER","LAMP","PILLOW","PAN","TOWEL","CLOCK","RUG","MUG","SHELF","VASE"]}
for c, ns in names.items():
    for n in ns:
        lo, hi = base[c]
        price = float(rng.integers(lo // 100, hi // 100) * 100 - 1)
        SKUS[f"{c[:4].upper()}-{n}"] = {"category": c, "list_price": price,
                                        "cogs_ratio": float(rng.uniform(0.55, 0.72))}
# launch SKUs (analogs inside the window + the sparse one)
LAUNCH = {"ELEC-SOUNDMAX-WHT": pd.Timestamp("2026-03-05"),
          "ELEC-BASSLITE-BLU": pd.Timestamp("2026-05-01"),
          "ELEC-AIRPRO-BLK":   pd.Timestamp("2026-07-25")}
for s, (lp, cr) in {"ELEC-SOUNDMAX-WHT": (2450, 0.63), "ELEC-BASSLITE-BLU": (2150, 0.67),
                    "ELEC-AIRPRO-BLK": (2600, 0.69)}.items():
    SKUS[s] = {"category": "Electronics", "list_price": float(lp), "cogs_ratio": cr}

# ---------------- calendar / seasonality ----------------
HOLIDAYS = {"eoss_jun": (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-08"), 1.45,
                          ["Fashion", "Home"], "End of season sale"),
            "rakhi_aug":  (pd.Timestamp("2026-08-24"), pd.Timestamp("2026-08-26"), 1.35,
                          ["Fashion", "Home"], "Raksha Bandhan gifting"),
            "summer_may": (pd.Timestamp("2026-05-08"), pd.Timestamp("2026-05-14"), 1.20,
                          ["Electronics"], "Summer electronics week")}

def festive_mult(d, cat=None):
    m = 1.0
    for _, (s, e, mult, cats, _n) in HOLIDAYS.items():
        if s <= d <= e and (cat is None or cat in cats):
            m *= mult
    return m

def dow_mult(d):                       # weekend uplift, Monday dip
    return {0: 0.88, 1: 0.95, 2: 0.98, 3: 1.02, 4: 1.10, 5: 1.22, 6: 1.15}[d.dayofweek]

def trend(d):                          # +18% across the window
    return 1.0 + 0.18 * ((d - START).days / max((ASOF - START).days, 1))

# ---------------- planted events ----------------
COUPON = (pd.Timestamp("2026-06-20"), pd.Timestamp("2026-07-14"))   # dies Jul 15 00:00
COMPET = (pd.Timestamp("2026-07-15"), pd.Timestamp("2026-07-21"))   # RivalMart electronics (1 week)
S2WEEK = (pd.Timestamp("2026-08-17"), pd.Timestamp("2026-08-23"))   # ambiguous week (full Mon-Sun)
GATEWAY = pd.Timestamp("2026-08-05")
STOCKOUT = (pd.Timestamp("2026-06-10"), pd.Timestamp("2026-06-13"), "FASH-SNEAKER", "South")

def coupon_active(d):  return COUPON[0] <= d <= COUPON[1]
def competitor_on(d):  return COMPET[0] <= d <= COMPET[1]
def s2_week(d):        return S2WEEK[0] <= d <= S2WEEK[1]

def cr_multiplier(d):
    m = 1.0
    if not coupon_active(d) and d >= COUPON[1]:   m *= 0.70   # coupon gone -> conversion drop
    if competitor_on(d):                          m *= 0.97
    if s2_week(d):                                m *= 0.88   # material, but no clean signature
    if d == GATEWAY:                              m *= 0.93   # checkout-step blip
    if STOCKOUT[0] <= d <= STOCKOUT[1]:           m *= 0.97
    return m

def sessions_mult(d, channel):
    m = 1.0
    if s2_week(d): m *= 0.94          # both channels soft -> no dominant driver
    if channel == "organic":
        if competitor_on(d): m *= 0.85
        if s2_week(d):       m *= 0.95
    return m

# ---------------- ORDERS ----------------
per_day = max(A.orders // len(DAYS), 4)
orders, oid = [], 1000
launch_curve = lambda k: 0.45 + 1.35 * math.exp(-k / 11) * (1 - math.exp(-(k + 1) / 3))

for d in DAYS:
    lam = per_day * trend(d) * dow_mult(d) * festive_mult(d) * cr_multiplier(d)
    n_orders = max(1, int(rng.poisson(lam)))
    for _ in range(n_orders):
        oid += 1
        order_id = f"ORD-{oid}"
        region = "North" if rng.random() < 0.54 else "South"
        n_items = 1 + (rng.random() < 0.32) + (rng.random() < 0.08)
        for _i in range(int(n_items)):
            # category weights shift with festive + competitor pull
            w = np.array([0.42, 0.33, 0.25], dtype=float)
            if competitor_on(d): w[0] *= 0.75          # electronics mix falls
            for j, c in enumerate(CATS): w[j] *= festive_mult(d, c)
            w = w / w.sum()
            cat = CATS[int(rng.choice(3, p=w))]
            pool = [s for s, v in SKUS.items() if v["category"] == cat
                    and (s not in LAUNCH or d >= LAUNCH[s])]
            if STOCKOUT[0] <= d <= STOCKOUT[1] and region == STOCKOUT[3]:
                pool = [s for s in pool if s != STOCKOUT[2]]
            sku = pool[int(rng.integers(len(pool)))]
            if sku in LAUNCH:                           # launch-shaped demand
                k = (d - LAUNCH[sku]).days
                if rng.random() > min(launch_curve(k), 1.0): continue
            v = SKUS[sku]
            disc = 0.10 if coupon_active(d) else float(rng.choice([0, 0, 0, 0.02], p=[.6,.2,.1,.1]))
            if festive_mult(d, cat) > 1.0: disc = max(disc, 0.08)
            qty = int(rng.choice([1, 1, 1, 2, 3], p=[.62,.15,.1,.09,.04]))
            realized = round(v["list_price"] * (1 - disc), 2)
            cogs = round(v["list_price"] * v["cogs_ratio"], 2)
            status = "cancelled" if rng.random() < 0.04 else "completed"
            orders.append((order_id, d.strftime("%Y-%m-%d"), sku, cat, region, qty,
                           v["list_price"], round(disc, 2), round(realized - cogs, 2), cogs, status))
# dedicated launch-SKU demand so analog curves are well-formed
for sku, ld in LAUNCH.items():
    v = SKUS[sku]
    for d in DAYS:
        k = (d - ld).days
        if k < 0: continue
        lam = 1.5 * launch_curve(k) * dow_mult(d) * cr_multiplier(d)
        for _ in range(int(rng.poisson(max(lam, 0)))):
            oid += 1
            region = "North" if rng.random() < 0.54 else "South"
            disc = 0.10 if coupon_active(d) else float(rng.choice([0, 0.02], p=[.8, .2]))
            realized = round(v["list_price"] * (1 - disc), 2)
            cogs = round(v["list_price"] * v["cogs_ratio"], 2)
            status = "cancelled" if rng.random() < 0.04 else "completed"
            orders.append((f"ORD-{oid}", d.strftime("%Y-%m-%d"), sku, "Electronics", region, 1,
                           v["list_price"], round(disc, 2), round(realized - cogs, 2), cogs, status))

orders_df = pd.DataFrame(orders, columns=["order_id","date","sku","category","region","qty",
                         "list_price","discount_pct","margin","cogs","payment_status"])

# ---------------- RETURNS ----------------
comp = orders_df[orders_df.payment_status == "completed"]
ret_orders = comp.drop_duplicates("order_id").sample(frac=0.06, random_state=A.seed)
returns = []
for i, r in enumerate(ret_orders.itertuples(), start=2000):
    rd = pd.Timestamp(r.date) + pd.Timedelta(days=int(rng.integers(3, 11)))
    if rd > ASOF: continue
    returns.append((f"RET-{i}", r.order_id, rd.strftime("%Y-%m-%d"),
                    str(rng.choice(["damaged","wrong_item","customer_changed_mind","size_issue"]))))
returns_df = pd.DataFrame(returns, columns=["return_id","order_ref","date","reason"])

# ---------------- INVENTORY ----------------
inv_skus = sorted(set(list(SKUS)[:12] + list(LAUNCH) + [STOCKOUT[2]]))
inv = []
stock = {(s, r): int(rng.integers(40, 160)) for s in inv_skus for r in REGIONS}
for d in DAYS:
    for s in inv_skus:
        for r in REGIONS:
            k = (s, r)
            stock[k] = max(0, stock[k] - int(rng.integers(0, 6)))
            if stock[k] < 12 and rng.random() < 0.4: stock[k] += int(rng.integers(40, 120))
            if s == STOCKOUT[2] and r == STOCKOUT[3] and STOCKOUT[0] <= d <= STOCKOUT[1]:
                stock[k] = 0
            if rng.random() < 0.02: continue          # ~2% missing sku-region-days
            inv.append((d.strftime("%Y-%m-%d"), s, r, stock[k]))
inv_df = pd.DataFrame(inv, columns=["date","sku","region","stock_on_hand"])

# ---------------- WEB ANALYTICS (T+1 watermark) ----------------
web_rows = []
daily_orders = comp.groupby("date").order_id.nunique()
for d in DAYS:
    if d > ASOF - pd.Timedelta(days=1): continue      # T+1: yesterday not yet exported
    key = d.strftime("%Y-%m-%d")
    base_sess = per_day * trend(d) * dow_mult(d) * festive_mult(d) / 0.018
    for region in REGIONS:
        rshare = 0.54 if region == "North" else 0.46
        for channel in ["paid", "organic"]:
            cshare = 0.45 if channel == "paid" else 0.55
            for device in ["mobile", "desktop"]:
                dshare = 0.68 if device == "mobile" else 0.32
                sess = base_sess * rshare * cshare * dshare * sessions_mult(d, channel)
                sess = max(int(rng.normal(sess, sess * 0.06)), 5)
                carts = max(round(sess * rng.uniform(0.10, 0.14)), 2)
                started = max(round(carts * rng.uniform(0.42, 0.52)), 1)
                compl_r = rng.uniform(0.40, 0.48) * (0.72 if d == GATEWAY else 1.0)
                completed = max(round(started * compl_r), 0)
                web_rows.append((d, region, channel, device, sess, carts, started, completed))
web_df = pd.DataFrame(web_rows, columns=["date","region","channel","device","sessions",
                      "add_to_carts","checkouts_started","checkouts_completed"])
# analytics over-reports orders by ~3%
web_df["checkouts_completed"] = (web_df.checkouts_completed * 1.03).round().astype(int)
web_df["checkouts_completed"] = np.minimum(web_df.checkouts_completed, web_df.checkouts_started)

# ---------------- MARKETING (weekly, Monday, S2 week skipped) ----------------
camp_names = {"Electronics": ["Electronics_Always_On","Audio_Push","Gadget_Weekly"],
              "Fashion": ["Fashion_Always_On","Monsoon_Fashion_Push","EOSS_Fashion"],
              "Home": ["Home_Essentials_Always_On","Home_Decor_Weekly"]}
mkt_rows = []
for wk_start in pd.date_range(START, ASOF, freq="W-MON"):
    if S2WEEK[0] <= wk_start <= S2WEEK[1]:            # missed upload -> S2 abstention
        continue
    wk_end = wk_start + pd.Timedelta(days=6)
    for cat, names_ in camp_names.items():
        for nm in names_:
            if rng.random() < 0.35: continue
            spend = int(rng.normal(45000, 12000) * festive_mult(wk_start, cat))
            mkt_rows.append((nm, str(rng.choice(["google_ads","meta_ads"])), max(spend, 8000),
                             wk_start.strftime("%Y-%m-%d"), wk_end.strftime("%Y-%m-%d"),
                             int(max(spend, 8000) * rng.uniform(12, 18))))
    if wk_start == pd.Timestamp("2026-07-27"):
        mkt_rows.append(("AirPro_Launch_Awareness","google_ads",95000,"2026-07-27","2026-08-23",1500000))
mkt_df = pd.DataFrame(mkt_rows, columns=["campaign","channel","spend","start_date","end_date","promised_impressions"])
mkt_df = pd.concat([mkt_df, mkt_df.iloc[[len(mkt_df)//2]]], ignore_index=True)      # planted duplicate
mkt_df.loc[len(mkt_df)-2, "end_date"] = "2062-08-17"                                # planted typo

# ---------------- WRITE FILES ----------------
db = os.path.join(OUT, "sales.db")
if os.path.exists(db): os.remove(db)
con = sqlite3.connect(db)
orders_df.to_sql("orders", con, index=False)
returns_df.to_sql("returns", con, index=False)
inv_df.to_sql("inventory", con, index=False)
con.close()

web_df.to_parquet(os.path.join(OUT, "web_analytics.parquet"), index=False)

wb = Workbook(); ws = wb.active; ws.title = "campaigns"
ws.append(list(mkt_df.columns))
for c in ws[1]: c.font = Font(name="Arial", bold=True)
for row in mkt_df.itertuples(index=False): ws.append(list(row))
ws.column_dimensions["A"].width = 30
for col in "BCDEF": ws.column_dimensions[col].width = 18
wb.save(os.path.join(OUT, "marketing.xlsx"))

NEWS = {
 "2026-07-14_rivalmart_electronics_flash_sale.txt": ("RivalMart announces 3-day Electronics Flash Sale","RetailWire India","2026-07-14",
  "RivalMart has announced a flash sale offering up to 40% off audio devices and small electronics from Jul 15-21.",
  "Electronics category, organic traffic","Reduced organic sessions and conversion in Electronics during Jul 15-21."),
 "2026-08-05_upi_gateway_degradation.txt": ("PayFlow gateway reports elevated UPI timeout rates","TechOps Status Blog","2026-08-05",
  "PayFlow acknowledged elevated 504 timeout rates on its UPI endpoint between 14:00 and 22:00 IST on Aug 5.",
  "Checkout completion, mobile UPI payments","Lower checkout completion on Aug 5; recovery expected once timeouts resolve."),
 "2026-08-20_earbud_demand_boom_claim.txt": ("Analyst report claims wireless earbud demand at all-time high","MarketPulse Research","2026-08-20",
  "A market research note claims consumer demand for true-wireless earbuds is at an all-time high this quarter.",
  "Electronics category demand","Contradicts observed order softness in mid-August; projection unverified."),
 "2026-08-24_rakhi_festive_demand.txt": ("Raksha Bandhan gifting drives demand surge","Economic Times Retail","2026-08-24",
  "Retailers report seasonal uplift in gifting categories ahead of Raksha Bandhan.",
  "Fashion and Home categories","Elevated sessions and orders in gifting categories; expected pattern."),
 "2026-08-18_metro_bus_strike.txt": ("City bus operators strike over fuel subsidy dispute","Local News Network","2026-08-18",
  "Public bus operators in two metro cities went on a one-day strike demanding revised fuel subsidies.",
  "Urban commuting","Minimal direct impact expected on online retail operations."),
 "2026-06-01_eoss_season_start.txt": ("End-of-season sales begin across online retail","RetailWire India","2026-06-01",
  "Major online retailers opened end-of-season sales this week with heavy fashion and home discounting.",
  "Fashion and Home categories","Elevated volumes and lower realised prices during the sale window."),
 "2026-06-10_courier_delays_south.txt": ("Courier partner reports delivery delays in southern hubs","Logistics Daily","2026-06-10",
  "A major courier partner reported 24-48h delivery delays across southern distribution hubs.",
  "South region fulfilment","Possible conversion drag in South region during the delay window."),
 "2026-05-08_summer_electronics_push.txt": ("Summer electronics buying picks up","Economic Times Retail","2026-05-08",
  "Seasonal demand for portable audio and cooling accessories rose through early May.",
  "Electronics category","Elevated Electronics demand in the second week of May."),
}
for fn, (title, src, dt, desc, area, imp) in NEWS.items():
    with open(os.path.join(OUT, "news", fn), "w") as f:
        f.write(f"Date: {dt}\nTitle: {title}\nSource: {src}\nDescription: {desc}\n"
                f"Affected area: {area}\nPotential impact: {imp}\n")

# ---------------- GROUND TRUTH ----------------
gt = {
 "window": {"start": str(START.date()), "asof": str(ASOF.date())}, "seed": A.seed,
 "watermarks": {"sales.orders": str(orders_df.date.max()),
                "web_analytics.parquet": str(web_df.date.max().date()),
                "marketing.xlsx": str(mkt_df.start_date.max())},
 "events": [
  {"id":"S1a","name":"coupon_expiry","date":"2026-07-15","mechanism":"sitewide discount 10% -> ~0 on 2026-07-15",
   "expected":{"conversion_pct":[-14,-6],"price_pct":[5,12],"gmv_pct":[-16,-6]},"scope":"sitewide"},
  {"id":"S1b","name":"competitor_promo","dates":["2026-07-15","2026-07-21"],
   "mechanism":"organic sessions -8%, electronics mix -25%","expected":{"traffic_pp":[-6,-3]},
   "scope":"organic channel, Electronics-concentrated"},
  {"id":"S2","name":"ambiguous_week","dates":["2026-08-17","2026-08-23"],
   "mechanism":"mild broad softness (~-2% CR, -2% organic); marketing.xlsx upload SKIPPED; "
               "contradictory news 2026-08-20","expected":{"gate":"abstain"}},
  {"id":"S3","name":"airpro_sparse_launch","launch":"2026-08-03",
   "analogs":{"ELEC-SOUNDMAX-WHT":"2026-03-05","ELEC-BASSLITE-BLU":"2026-05-01"},
   "expected":{"history_days":(ASOF - LAUNCH['ELEC-AIRPRO-BLK']).days,"alert":"none (in-band)"}},
  {"id":"GW","name":"gateway_blip","date":"2026-08-05",
   "mechanism":"checkout completion x0.72, started untouched","expected":{"funnel_step":"checkout"}},
  {"id":"SO","name":"stockout","dates":["2026-06-10","2026-06-13"],
   "mechanism":f"{STOCKOUT[2]} {STOCKOUT[3]} stock=0","expected":{"acts_on":["conversion","mix"]}},
 ],
 "calendar": {k: {"start": str(v[0].date()), "end": str(v[1].date()), "mult": v[2],
                  "categories": v[3], "name": v[4]} for k, v in HOLIDAYS.items()},
 "quality_flaws": ["marketing.xlsx: 1 duplicate row", "marketing.xlsx: end_date 2062 typo",
                   "inventory: ~2% missing sku-region-days", "web: checkouts over-report orders ~3%",
                   "orders: ~4% cancelled"],
}
with open(os.path.join(OUT, "ground_truth.yaml"), "w") as f:
    yaml.safe_dump(gt, f, sort_keys=False, default_flow_style=False)

print(f"orders rows      {len(orders_df):>6}  ({orders_df.order_id.nunique()} orders)")
print(f"returns rows     {len(returns_df):>6}")
print(f"inventory rows   {len(inv_df):>6}")
print(f"web rows         {len(web_df):>6}   watermark {web_df.date.max().date()}")
print(f"campaign rows    {len(mkt_df):>6}   watermark {mkt_df.start_date.max()}")
print(f"news snippets    {len(NEWS):>6}")
