"""Display labels — UI-facing names for engine identifiers.

The engine keeps snake_case keys (contracts, graph, feedback logs depend on
them); the UI translates at render time via nice(). Data-lineage identifiers
(file names like sales.db, evidence ids like e-101) stay literal on purpose —
they ARE the audit trail.
"""

LABELS = {
    # roles / owners
    "cfo": "CFO", "north_manager": "North Region Manager",
    "marketing_exec": "Marketing Executive", "analyst_intern": "Analyst (Intern)",
    "system": "System",
    # KPIs
    "gmv": "GMV", "orders_kpi": "Orders", "orders": "Orders",
    "sessions": "Sessions", "conversion_rate": "Conversion rate", "aov": "AOV",
    "net_revenue": "Net revenue", "airpro_orders": "AirPro orders",
    "marketing_spend": "Marketing spend", "checkout_completion": "Checkout completion",
    # drivers / hypotheses
    "coupon_expiry": "Coupon expiry", "competitor_promo": "Competitor promotion",
    "marketing_gap": "Marketing spend gap", "gateway_issue": "Payment gateway issue",
    "supply_stockout": "Supply stockout", "demand_shift": "Demand shift",
    "product_launch": "Product launch", "launch_volatility": "Launch volatility",
    # decision modes
    "dual_cause": "Dual cause", "single_cause": "Single cause",
    "abstain": "Abstain — clarification requested",
    "no_material_movement": "No material movement",
    # event types
    "level_shift": "Level shift", "trend_break": "Trend break",
    "transient_spike": "Transient spike", "window_investigation": "Window investigation",
    # rejection reasons
    "pending_persistence": "Pending persistence",
    "expected_seasonal_event": "Expected seasonal event",
    "sign_inconsistent_volatility": "Sign-inconsistent volatility",
    "in_band_launch_volatility": "In-band launch volatility",
    "out_of_band_flagged": "Out of band — flagged",
    "below_materiality": "Below materiality threshold",
    "incomplete_data": "Incomplete data",
    "explained_by_known_driver": "Explained by known driver",
    # levers
    "promotional_pricing": "Promotional pricing", "campaign_spend": "Campaign spend",
    # caps / misc
    "refutation_cap": "refutation cap", "freshness_cap:untestable": "freshness cap (untestable)",
    # criteria columns
    "evidence_strength": "Evidence strength", "temporal_alignment": "Temporal alignment",
    "source_reliability": "Source reliability", "contribution_match": "Contribution match",
    "historical_precedent": "Historical precedent",
    "hypothesis": "Hypothesis", "confidence": "Confidence", "calibrated": "Calibrated",
    "caps": "Caps", "kpi": "KPI", "pct": "% move",
}

def nice(x):
    """Pretty label for any identifier; graceful fallback for unknowns."""
    if x is None:
        return "—"
    s = str(x)
    if s in LABELS:
        return LABELS[s]
    if "_" in s and " " not in s:
        return s.replace("_", " ").capitalize()
    return s

def nice_caps(caps):
    return ", ".join(nice(c) for c in caps) if caps else "—"

def nice_cols(df):
    """Rename dataframe columns to display labels."""
    return df.rename(columns={c: nice(c) for c in df.columns})
