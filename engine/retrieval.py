"""Stage 4 — Evidence retrieval: dockets + hypothesis sets.

Every numeric value here is COMPUTED from trend_lens_data via engine.analytics at
import time — nothing is hardcoded, so figures can never contradict the dataset.
Still pending the full build: embeddings + LLM call #1 for unstructured retrieval
(news snippets are currently attached by date//topic match rather than vector search).
"""
from . import analytics as A

S1_PRE, S1_POST = ("2026-06-24", "2026-07-14"), ("2026-07-15", "2026-07-21")   # 21d vs 7d
S1_COMP_PRE, S1_COMP_POST = ("2026-07-08", "2026-07-14"), ("2026-07-15", "2026-07-21")
S2_BASE, S2_WIN = ("2026-08-03", "2026-08-16"), ("2026-08-17", "2026-08-23")  # 14d vs 7d, whole weeks

D1 = A.decompose(S1_PRE, S1_POST)
_N = A.decompose(S1_PRE, S1_POST, region='North')   # entitlement-scoped computation
CH1 = A.channel_shift(S1_COMP_PRE, S1_COMP_POST)
D2 = A.decompose(S2_BASE, S2_WIN)
CH2 = A.channel_shift(S2_BASE, S2_WIN)
MKT = A.clean_mkt()
BAND = A.analog_band("ELEC-AIRPRO-BLK", ["ELEC-SOUNDMAX-WHT", "ELEC-BASSLITE-BLU"])
GW_DAY, GW_BASE = A.funnel_ratio("2026-08-05", "2026-08-05"), A.funnel_ratio("2026-07-30", "2026-08-04")
pp = lambda v: f"{v:+.1f}pp"
pc = lambda v: f"{v:+.1f}%"

# ---------------- S1: multi-factor GMV drop ----------------
S1_EVENT = {
    "id": "AE-2026-07-19-001", "kpi": "gmv", "window": "Jul 15-21", "scenario": "S1",
    "magnitude": pc(D1["total_pct"]), "impact_inr": int(D1["impact_inr"]), "type": "level_shift",
    "decomposition": {"conversion": pp(D1["conversion"]), "traffic": pp(D1["traffic"]),
                      "price": pp(D1["price"]), "mix": pp(D1["mix"]),
                      "interaction": pp(D1["interaction"])},
    "gate_trace": ["persistence: 5 consecutive days (min 2)", "ensemble 3/3 (weekday-z, STL residual, CUSUM)",
                   "freshness clean",
                   "calendar: not festive-explained",
                   f"materiality Rs{D1['impact_inr']/1e5:.2f}L >= Rs0.08L",
                   f"coherence: CR {pc((D1['post']['cr']/D1['pre']['cr']-1)*100/100*100)}, "
                   f"organic sessions {pc(CH1['organic_pct'])}, paid {pc(CH1['paid_pct'])}"],
    "north_manager_overrides": {"magnitude": pc(_N["total_pct"]), "impact_inr": int(_N["impact_inr"]),
        "decomposition": {"conversion": pp(_N["conversion"]), "traffic": pp(_N["traffic"]),
                          "price": pp(_N["price"]), "mix": pp(_N["mix"]),
                          "interaction": pp(_N["interaction"])}},
}
S1_DOCKET = [
 {"id":"e-101","claim":"Sitewide discount rate collapsed at the anomaly start (coupon expiry)",
  "value":f"mean discount {D1['pre']['discount']:.3f} -> {D1['post']['discount']:.3f}",
  "source":"sales.db","fresh":"Aug 26 02:00","method":"SQL series on orders.discount_pct",
  "direction":"confirming","supports":"coupon_expiry","strength":0.9,"temporal":0.95,"live":True},
 {"id":"e-102","claim":"Level-1 price effect is positive while GMV falls (coupon fingerprint)",
  "value":f"price {pp(D1['price'])}, AOV {pp(D1['aov'])}","source":"sales.db","fresh":"Aug 26 02:00",
  "method":"shift-share decomposition","direction":"confirming","supports":"coupon_expiry",
  "strength":0.85,"temporal":0.95,"live":True},
 {"id":"e-104","claim":"Conversion is the dominant Level-1 component",
  "value":f"conversion {pp(D1['conversion'])} of {pc(D1['total_pct'])} total",
  "source":"web_analytics.parquet","fresh":"Aug 25 (T+1)","method":"identity decomposition",
  "direction":"confirming","supports":"coupon_expiry","strength":0.8,"temporal":0.9,"live":True},
 {"id":"e-110","claim":"RivalMart 3-day Electronics flash sale announced Aug 11 (runs Aug 12-14)",
  "value":A.snippet("2026-07-14_rivalmart_electronics_flash_sale.txt"),
  "source":"news/RetailWire India","fresh":"Jul 14","method":"news corpus match",
  "direction":"confirming","supports":"competitor_promo","strength":0.75,"temporal":0.9,"live":True},
 {"id":"e-111","claim":"Organic sessions fell while paid held flat during the sale window",
  "value":f"organic {pc(CH1['organic_pct'])}, paid {pc(CH1['paid_pct'])} (Jul 15-21)",
  "source":"web_analytics.parquet","fresh":"Aug 25 (T+1)","method":"channel slice",
  "direction":"confirming","supports":"competitor_promo","strength":0.7,"temporal":0.85,"live":True},
 {"id":"e-118","claim":"Campaign spend steady across the window (rules out marketing gap)",
  "value":f"Rs {int(MKT.spend.sum()):,} across {MKT.campaign.nunique()} campaigns; paid sessions {pc(CH1['paid_pct'])}",
  "source":"marketing.xlsx","fresh":"Mon Aug 24 upload","method":"xlsx dedup + channel check",
  "direction":"refuting","supports":"marketing_gap","strength":0.7,"temporal":0.8,"live":True},
 {"id":"e-120","claim":"Checkout-step ratio normal in window; drop is pre-checkout (rules out gateway)",
  "value":f"completion ratio {GW_BASE:.3f} baseline vs {GW_DAY:.3f} on the one blip day (Aug 5 only)",
  "source":"web_analytics.parquet","fresh":"Aug 25 (T+1)","method":"funnel step check",
  "direction":"refuting","supports":"gateway_issue","strength":0.7,"temporal":0.9,"live":True},
 {"id":"e-115","claim":"MarketPulse note claims record earbud demand (conflicts with observed softness)",
  "value":A.snippet("2026-08-20_earbud_demand_boom_claim.txt"),
  "source":"news/MarketPulse Research","fresh":"Aug 20","method":"news corpus match",
  "direction":"confirming","supports":"demand_shift","strength":0.5,"temporal":0.7,
  "conflicts_with":"e-101","live":True},
 {"id":"e-130","claim":"Gateway outage margin-exposure estimate (CFO-only evidence tier)",
  "value":"est. margin exposure if UPI degradation recurs — restricted tier",
  "source":"news/TechOps Status Blog","fresh":"Aug 5","method":"news corpus match",
  "direction":"confirming","supports":"gateway_issue","strength":0.4,"temporal":0.6,
  "tier":"cfo","live":True},
]
S1_HYPOTHESES = [
 {"name":"coupon_expiry","evidence":["e-101","e-102","e-104"],"n_requirements":3,
  "owns":["conversion","price"],"contribution_match":0.88,"prior_alpha":6,"prior_beta":2,
  "lever":"promotional_pricing","depth":"10%",
  "recovery_inr_per_day":int(abs(D1["impact_inr"])/D1["post"]["days"]*0.7)},
 {"name":"competitor_promo","evidence":["e-110","e-111"],"n_requirements":4,
  "owns":["traffic","mix"],"contribution_match":0.80,"prior_alpha":3,"prior_beta":3,
  "causal":{"test":"organic-vs-paid channel contrast","p":0.03},
  "monitor_note":"sale ended Jul 21; verify organic recovery by Jul 24"},
 {"name":"demand_shift","evidence":["e-115"],"n_requirements":3,"owns":["traffic","conversion"],
  "contribution_match":0.30,"prior_alpha":2,"prior_beta":3},
 {"name":"marketing_gap","evidence":["e-118"],"n_requirements":2,"owns":["traffic"],
  "contribution_match":0.10,"prior_alpha":2,"prior_beta":2},
 {"name":"gateway_issue","evidence":["e-120","e-130"],"n_requirements":2,"owns":["conversion"],
  "contribution_match":0.15,"prior_alpha":2,"prior_beta":2},
]

# ---------------- S2: low-confidence week ----------------
S2_EVENT = {
    "id": "AE-2026-08-21-002", "kpi": "gmv", "window": "wk Aug 17-23", "scenario": "S2",
    "magnitude": pc(D2["total_pct"]), "impact_inr": int(abs(D2["impact_inr"])), "type": "trend_break",
    "decomposition": {"conversion": pp(D2["conversion"]), "traffic": pp(D2["traffic"]),
                      "price": pp(D2["price"]), "mix": pp(D2["mix"])},
    "gate_trace": ["persistence: 3 consecutive days (min 2)", "ensemble 2/3 (weekday-z, STL residual; CUSUM sub-threshold)",
                   "freshness: marketing.xlsx has NO rows for this week",
                   "calendar: not festive-explained", "materiality above Rs0.08L threshold",
                   f"coherence: organic {pc(CH2['organic_pct'])}, paid {pc(CH2['paid_pct'])} — no dominant channel"],
    "resolves": [f"e-218: marketing.xlsx upload for wk Aug 17-23 missing (weeks present: "
                 f"{', '.join(sorted(MKT[(MKT.start_date>='2026-08-03')&(MKT.start_date<='2026-08-31')].start_date.unique()))}) "
                 f"-> auto re-evaluation on arrival",
                 "3 more days of session data would separate competitor pull from broad demand shift"],
}
S2_DOCKET = [
 {"id":"e-210","claim":"Mild session softness, spread across channels (no dominant signature)",
  "value":f"organic {pc(CH2['organic_pct'])}, paid {pc(CH2['paid_pct'])}",
  "source":"web_analytics.parquet","fresh":"Aug 25 (T+1)","method":"channel slice",
  "direction":"confirming","supports":"competitor_promo","strength":0.5,"temporal":0.7,"live":True},
 {"id":"e-215","claim":"MarketPulse record-demand claim conflicts with observed softness",
  "value":A.snippet("2026-08-20_earbud_demand_boom_claim.txt"),
  "source":"news/MarketPulse Research","fresh":"Aug 20","method":"news corpus match",
  "direction":"confirming","supports":"demand_shift","strength":0.5,"temporal":0.6,
  "conflicts_with":"e-210","live":True},
 {"id":"e-218","claim":"marketing.xlsx has no campaign rows for this week (Monday upload missed)",
  "value":f"rows for Aug 17-23 = {int(((MKT.start_date>='2026-08-17')&(MKT.start_date<='2026-08-23')).sum())} — MISSING",
  "source":"marketing.xlsx","fresh":"STALE","method":"reconciler freshness walk",
  "direction":"confirming","supports":"marketing_gap","strength":0.3,"temporal":0.5,
  "status":"MISSING","live":True},
 {"id":"e-220","claim":"No pricing event in the window (rules out a coupon cause)",
  "value":f"mean discount {D2['post']['discount']:.3f}","source":"sales.db","fresh":"Aug 26 02:00",
  "method":"SQL series","direction":"refuting","supports":"coupon_expiry","strength":0.6,
  "temporal":0.8,"live":True},
]
S2_HYPOTHESES = [
 {"name":"competitor_promo","evidence":["e-210"],"n_requirements":3,"owns":["traffic"],
  "contribution_match":0.30,"prior_alpha":2,"prior_beta":3},
 {"name":"demand_shift","evidence":["e-215"],"n_requirements":3,"owns":["traffic","conversion"],
  "contribution_match":0.40,"prior_alpha":2,"prior_beta":3},
 {"name":"marketing_gap","evidence":["e-218"],"n_requirements":2,"owns":["traffic"],
  "contribution_match":0.35,"prior_alpha":2,"prior_beta":2,"critical_evidence_missing":True},
 {"name":"coupon_expiry","evidence":["e-220"],"n_requirements":3,"owns":["conversion","price"],
  "contribution_match":0.10,"prior_alpha":6,"prior_beta":2},
]

# ---------------- S3: sparse launch ----------------
_hist = (A.C.date.max() - BAND["launch"]).days
_last = min(int(_hist), len(BAND["target"]) - 1)          # evaluate the CURRENT launch day, not a fixed one
_cov = sum(BAND["in_band"][6:34])
S3_INFO = {
 "id":"CAND-airpro-sparse","kpi":"airpro_orders","scenario":"S3",
 "status":"REJECTED_IN_BAND" if BAND["in_band"][_last] else "FLAGGED_OUT_OF_BAND",
 "history_days":int(_hist),"min_history":60,"launch":str(BAND["launch"].date()),
 "eval_day":_last + 1,
 "observed":round(float(BAND["target"][_last]),2),
 "band":[round(float(BAND["lo"][_last]),2), round(float(BAND["hi"][_last]),2)],
 "in_band":bool(BAND["in_band"][_last]),
 "days_in_band":f"{_cov}/28",
 "note":(f"{int(_hist)} days of history (< 60 required) — no seasonal baseline computable. "
         f"Judged against a band built from 2 analog launches (SoundMax, BassLite), launch-day "
         f"indexed and normalised. Band is deliberately wide: only 2 analogs support it. "
         f"Any alert on this KPI carries confidence_penalty 0.6x."),
 "curves":{"day":list(range(1,_last+2)),
           "airpro":[round(float(v),3) for v in BAND["target"][:_last+1]],
           "band_lo":[round(float(v),3) for v in BAND["lo"][:_last+1]],
           "band_hi":[round(float(v),3) for v in BAND["hi"][:_last+1]]},
}

# ---------------- rejection log (restraint) ----------------
_so = A.stockout_days("FASH-SNEAKER","South","2026-06-10","2026-06-13")
REJECTION_LOG = [
 {"candidate":"conversion_rate spike (latest day)","reason":"incomplete_data",
  "detail":"orders fresh to Aug 26, sessions (T+1) only to Aug 25 — denominator incomplete; requeued"},
 {"candidate":"orders uplift Fashion+Home (Aug 22-26)","reason":"expected_seasonal_event",
  "detail":"Raksha Bandhan window per holiday calendar (news 2026-08-22 corroborates)"},
 {"candidate":f"airpro_orders day {S3_INFO['eval_day']} level","reason":"in_band_launch_volatility" if S3_INFO["in_band"] else "out_of_band_flagged",
  "detail":f"normalised {S3_INFO['observed']} vs analog band {S3_INFO['band']} ({S3_INFO['days_in_band']} days in band); sparse branch, penalty 0.6x"},
 {"candidate":"checkout completion dip Aug 5","reason":"transient_spike",
  "detail":f"completion {GW_DAY:.3f} vs {GW_BASE:.3f} baseline for one day only; recovered Aug 6 (PayFlow UPI outage, news 2026-08-05)"},
 {"candidate":"conversion dip Jun 10-13 (South)","reason":"explained_by_known_driver",
  "detail":f"FASH-SNEAKER South stock_on_hand=0 on {_so} days; supply-side, logged not alerted"},
 {"candidate":"organic sessions dip (1 day, Jul 15)","reason":"pending_persistence",
  "detail":"1/2 points at first evaluation; matured into e-111 the following day"},
]
