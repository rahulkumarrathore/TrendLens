"""Writes contracts/*.yaml, personas.yaml, source_reliability.yaml — the governed-semantics layer."""
import yaml, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ACCESS_FULL = {"rows": "all", "columns": "all"}
ACCESS = {
    "cfo": ACCESS_FULL,
    "north_manager": {"rows": "region = 'North'", "columns_exclude": ["margin", "cogs"],
                      "domains_exclude": ["marketing_spend"]},
    "marketing_exec": {"rows": "all", "columns_exclude": ["margin", "cogs", "return_reason"]},
    "analyst_intern": {"rows": "all", "columns_exclude": ["sku", "discount_pct"],
                       "min_grain": "category", "domains_only": ["gmv", "orders"]},
}

def c(name, formula, grain, parents, sources, drivers, levers, sigma=2.0, impact=75000, extra=None):
    d = {"kpi": name, "formula": formula, "grain": grain, "parents": parents,
         "sources": sources, "drivers": drivers, "levers": levers,
         "thresholds": {"sigma": sigma, "min_impact_inr": impact},
         "gate": {"persistence_n": 2}, "calendar": "ISO",
         "access": ACCESS}
    if extra: d.update(extra)
    return d

contracts = [
    c("gmv", "sum(qty * list_price * (1 - discount_pct)) [completed]",
      "day x region x category", ["orders_kpi", "aov"], ["sales.orders"],
      ["coupon_expiry", "competitor_promo", "marketing_gap", "supply_stockout", "gateway_issue", "demand_shift"],
      {"promotional_pricing": "marketing_exec"}),
    c("net_revenue", "gmv - returns - cancellations", "day x region x category",
      ["gmv"], ["sales.orders", "sales.returns"], ["return_spike"], {},
      extra={"settling_lag_days": 10, "note": "provisional until returns settle"}),
    c("orders_kpi", "count(distinct order_id) [completed]", "day x region x category",
      [], ["sales.orders"], ["coupon_expiry", "competitor_promo"], {}),
    c("aov", "gmv / orders", "day x region", ["gmv", "orders_kpi"], ["sales.orders"],
      ["coupon_expiry", "mix_shift"], {}),
    c("sessions", "sum(sessions)", "day x region x channel", [], ["web_analytics.parquet"],
      ["competitor_promo", "marketing_gap"], {"campaign_spend": "marketing_exec"}),
    c("conversion_rate", "orders / sessions", "day x region x device",
      ["orders_kpi", "sessions"], ["sales.orders", "web_analytics.parquet"],
      ["coupon_expiry", "supply_stockout", "gateway_issue"], {"promotional_pricing": "marketing_exec"},
      extra={"freshness_rule": "min(parents)"}),
    c("marketing_spend", "sum(spend)", "campaign x week", [], ["marketing.xlsx"],
      ["budget_change"], {"campaign_spend": "marketing_exec"},
      extra={"refresh": "weekly_monday_flaky"}),
    c("airpro_orders", "count(distinct order_id) where sku='ELEC-AIRPRO-BLK'", "day",
      [], ["sales.orders"], ["launch_volatility", "supply_stockout"], {},
      extra={"min_history_days": 60, "history_days": 21,
             "analogs": ["ELEC-SOUNDMAX-WHT", "ELEC-BASSLITE-BLU"],
             "confidence_penalty": 0.6}),
]
for d in contracts:
    with open(f"contracts/{d['kpi']}.yaml", "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)

personas = {
    "cfo": {"depth": "executive", "vocabulary": "financial", "numbers": "aggregated",
            "action_frame": "approve", "channels": ["email_digest", "dashboard"], "length": "120-180 words"},
    "north_manager": {"depth": "operational_regional", "vocabulary": "retail_ops",
                      "numbers": "region_scoped", "action_frame": "execute_regional",
                      "channels": ["alert_push", "dashboard"], "length": "100-150 words"},
    "marketing_exec": {"depth": "operational_task", "vocabulary": "marketing",
                       "numbers": "channel_campaign", "action_frame": "execute",
                       "channels": ["alert_push", "slack"], "length": "100-160 words"},
    "analyst_intern": {"depth": "summary_readonly", "vocabulary": "neutral",
                       "numbers": "category_aggregates", "action_frame": "none",
                       "channels": ["dashboard"], "length": "60-90 words"},
}
with open("personas.yaml", "w") as f:
    yaml.safe_dump(personas, f, sort_keys=False)

reliability = {"sales.db": 0.95, "web_analytics.parquet": 0.85, "marketing.xlsx": 0.60,
               "news/RetailWire India": 0.65, "news/TechOps Status Blog": 0.60,
               "news/Economic Times Retail": 0.65, "news/MarketPulse Research": 0.40,
               "news/Local News Network": 0.45}
with open("source_reliability.yaml", "w") as f:
    yaml.safe_dump(reliability, f)
print("contracts:", len(contracts), "| personas:", len(personas), "| reliability sources:", len(reliability))
