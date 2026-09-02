"""TrendLens — decision workspace UI (Streamlit). Renders pipeline outputs; the
persona toggle and feedback re-run invoke the pipeline live."""
import streamlit as st
import pandas as pd
from engine.pipeline import run_all, apply_feedback, reset_reliability, RELIABILITY
from engine.scenarios import S1_DOCKET, S3_INFO
from engine import core
from engine import analytics as A
from engine import retrieval as R
from engine import sweep, investigate, calibrate, nlg
from engine.labels import nice, nice_caps, nice_cols

st.set_page_config(page_title="TrendLens", layout="wide", page_icon="📉")

st.markdown("""<style>
.block-container{padding-top:2.2rem;} h1{letter-spacing:-0.02em;}

.stTabs [data-baseweb="tab-list"]{position:sticky;top:1.0rem;z-index:999;
  background:var(--background-color,white);border-bottom:1px solid #e6e6e6;padding-top:.3rem;}
.stTabs [data-baseweb=tab]{font-size:0.95rem;}
[data-testid=stMetricValue]{font-size:1.5rem;}
</style>""", unsafe_allow_html=True)

# ---------------- sidebar ----------------
st.sidebar.title("TrendLens")

st.sidebar.markdown("""
<div style="
    margin-top: -1.5rem;
    margin-bottom: 0.35rem;
    font-size: 0.72rem;
    font-weight: 800;
    color: #6CB6E8;
    letter-spacing: 0.08em;
    line-height: 1;
">
    <span style="font-size: 0.88rem;">B</span>USINESS<span style="font-size: 0.88rem;">I</span>NTELLIGENCE.AI
</div>
""", unsafe_allow_html=True)

# ---------------- narrative + action rendering ----------------
_BADGE = {
    "abstain":              ("ABSTAINED — no cause committed", "#d29922"),
    "no_material_movement": ("NO MATERIAL MOVEMENT",           "#3fb950"),
    "dual_cause":           ("DUAL CAUSE",                     "#3d8bfd"),
    "single_cause":         ("SINGLE CAUSE",                   "#3d8bfd"),
}


def render_narrative(run, mode=None):
    """Render a narrative as structured markdown inside a bordered container.

    st.info/st.warning render inline markdown only — headers and tables get
    flattened into a single paragraph — so alert boxes cannot carry the
    structured template output. Evidence ids become footnote numbers here.
    Any failure degrades to the raw text rather than taking down the page.
    """
    text = (run or {}).get("narrative") or ""
    mode = mode or (run or {}).get("decision", {}).get("mode") or "single_cause"
    label, colour = _BADGE.get(mode, (str(nice(mode)).upper(), "#3d8bfd"))
    try:
        gloss = (run.get("payload") or {}).get("evidence_glossary") or {}
        body, refs = nlg.render_citations(text, gloss)
    except Exception as exc:
        body, refs = text, []
        st.caption(f"citation rendering skipped: {type(exc).__name__}")
    with st.container(border=True):
        st.markdown(
            f"<div style='color:{colour};font-size:0.72rem;font-weight:700;"
            f"letter-spacing:.09em;margin-bottom:.45rem'>{label}</div>",
            unsafe_allow_html=True)
        st.markdown(body)
        if refs:
            with st.expander(f"Evidence ({len(refs)})"):
                st.dataframe(pd.DataFrame([
                    {"#": r["n"], "Finding": r["claim"], "Source": r["source"],
                     "Fresh": r["fresh"], "Method": r["method"]} for r in refs]),
                    hide_index=True, use_container_width=True)


def render_actions(actions):
    """Action cards. Every text field from assemble_actions is already a whole
    sentence, so nothing is spliced together with dashes here; fields that are
    absent (no lever, no expected impact) are omitted rather than printed."""
    if not actions:
        return
    st.markdown("**Actions**")
    for a in actions:
        with st.container(border=True):
            if a.get("lever"):
                st.markdown(f"**{nice(a['driver'])}** · {nice(a['lever'])} · "
                            f"owner {nice(a['owner'])}")
            else:
                st.markdown(f"**{nice(a['driver'])}** · no lever · monitor only")
            st.markdown(a["action"])
            if a.get("rationale"):
                st.markdown(a["rationale"])
            meta = []
            if a.get("expected_impact"):
                meta.append(str(a["expected_impact"]).rstrip("."))
            meta.append(f"confidence {a['confidence']:.2f}")
            st.caption(" · ".join(meta))
            st.caption(f"Monitoring: {a['monitoring_plan']}")





st.sidebar.caption("KPI intelligence-to-action engine")
role = st.sidebar.selectbox("Role (entitlement)",
                            ["cfo", "north_manager", "marketing_exec", "analyst_intern"],
                            format_func=nice)
backend = st.sidebar.radio("LLM backend",
                           ["Deterministic template (no LLM)", "Ollama — local LLM",
                            "Anthropic API", "OpenAI API"],
                           help="Swappable per run. Template fallback guarantees a correct narrative either way.")
backend_key = {"Deterministic template (no LLM)": "none", "Ollama — local LLM": "ollama",
               "Anthropic API": "api", "OpenAI API": "openai"}[backend]
_default_model = {"ollama": "llama3.2:3b", "api": "claude-haiku-4-5-20251001",
                  "openai": "gpt-5.6-luna"}.get(backend_key)
model = st.sidebar.text_input("Model", value=_default_model) if backend_key != "none" else None
if st.sidebar.button("Reset feedback effects"):
    reset_reliability(); st.sidebar.success("Source reliabilities reset")
_cov = A.coverage()
st.sidebar.markdown("**Dataset loaded**")
st.sidebar.markdown(f"""
| | |
|---|---|
| Window | {_cov['start']} → {_cov['end']} |
| Duration | {_cov['months']} months ({_cov['days']} days) |
| Orders | {_cov['order_rows']:,} rows / {_cov['orders']:,} orders |
| Web | {_cov['web_rows']:,} rows |
| Campaigns | {_cov['campaign_rows']} rows |
| News | {_cov['news_files']} snippets |
| SKUs | {_cov['skus']} |
| Regions | {len(_cov['regions'])} |
| Categories | {len(_cov['categories'])} |
""")

scope = core.audit_scope(role)
st.sidebar.markdown("**Scope applied**")
st.sidebar.code(f"rows: {scope['rows']}\ncols dropped: {scope['columns_dropped'] or '—'}\n"
                f"domains refused: {scope['domains_refused'] or '—'}", language=None)

results = run_all(role, backend_key, model)
S1, S2 = results["S1"], results["S2"]

tab_feed, tab_explore, tab_scenario, tab_scen, tab_fb, tab_tel = st.tabs(
    ["Anomaly feed", "Investigate KPI", "Scenario walkthrough",
     "All scenarios", "Feedback loop", "Telemetry & audit"])



# ---------------- anomaly feed ----------------
with tab_feed:
    st.subheader("KPI History")
    hc1, hc2 = st.columns([3, 1])
    kpi_choice = hc2.selectbox("Series", ["GMV", "Orders", "Sessions", "AOV", "Conversion rate %"])
    smooth = hc2.checkbox("7-day smoothing", value=True)
    region_scope = None if core.CONTRACTS["gmv"]["access"][role].get("rows", "all") == "all" else "North"
    hist = A.daily_series(region_scope)[[kpi_choice]]
    if smooth:
        hist = hist.rolling(7, min_periods=1).mean()
    hc1.line_chart(hist, height=260)
    st.caption(
        f"{_cov['start']} to {_cov['end']} — {_cov['months']} months of daily history")
    st.divider()
    st.subheader("Anomalies Detected")
    st.caption("Significant KPI deviations identified through continuous monitoring. Click any event to investigate.")
    for ev in sweep.CONFIRMED:
        with st.container(border=True):
            a, b, c = st.columns([2, 1, 1])
            a.markdown(f"**{ev['id']}** · {nice(ev['kpi'])} {ev['magnitude']} ({ev['window']})")
            b.metric("Impact", f"₹{ev['impact_inr']/1e5:.2f}L")
            c.markdown(f"`{nice(ev['type'])}`")
            with st.expander("Gate trace (six checks, all computed)"):
                for g in ev["gate_trace"]:
                    st.markdown(f"- {g}")
            if st.button(f"Investigate {ev['id']} →", key=f"inv_{ev['id']}"):
                with st.spinner("Signature probes + vector retrieval + ranking…"):
                    rr = investigate.investigate_event(ev, role=role, backend=backend_key, model=model)
                render_narrative(rr)
                st.dataframe(pd.DataFrame(
                    [{"Hypothesis": nice(h["name"]), "Confidence": h["confidence"],
                      "Calibrated": h.get("calibrated"), "Caps": nice_caps(h["caps"])}
                     for h in rr["finding"]["hypotheses"]]), hide_index=True)
    st.divider()
    st.subheader("Rejected Events")
    st.caption("Every deviation accounted with a clear reason.")
    _total = len(sweep.REJECTIONS) + len(sweep.CONFIRMED)
    _rate = len(sweep.CONFIRMED) / _total if _total else 0

    with st.container(border=True):
        m1, m2, m3 = st.columns(3, border=True)
        m1.metric("🔍 Events swept", _total)
        m2.metric("🚫 Filtered as noise", len(sweep.REJECTIONS),
                  delta=f"{(1 - _rate) * 100:.0f}% of swept", delta_color="off")
        m3.metric("✅ Confirmed anomalies", len(sweep.CONFIRMED),
                  delta=f"{_rate * 100:.0f}% signal", delta_color="normal")
    _rc = pd.Series([r_["reason"] for r_ in sweep.REJECTIONS]).value_counts()
    st.bar_chart(_rc.rename(index=nice))
    pick = st.selectbox("Show rejections with reason", ["ALL DISCARDED EVENTS"] + list(_rc.index),
                        format_func=lambda x: x if x.startswith("(") else nice(x))
    shown = results["rejections"] if pick == "ALL DISCARDED EVENTS" else \
        [r_ for r_ in sweep.REJECTIONS if r_["reason"] == pick][:8]
    for rj in shown:
        with st.container(border=True):
            st.markdown(f"**{rj['candidate']}**")
            st.markdown(f"`{nice(rj['reason'])}` — {rj['detail']}")

# ---------------- explore any window ----------------
with tab_explore:
    st.subheader("Investigate any KPI over any timeline window")
    st.caption("Automatically checks the key drivers behind KPI movement and builds an evidence-based explanation <br>"
    "The baseline is selected from 14 reliable days, excluding confirmed anomalies.",
     unsafe_allow_html=True)
    ec1, ec2, ec3, ec4 = st.columns([1.2, 1, 1, 1])
    e_kpi = ec1.selectbox("KPI", ["all", "gmv", "sessions",
                                  "conversion_rate", "orders_kpi", "aov", "net_revenue"],
                          format_func=lambda x: "All KPIs" if x == "all" else nice(x))
    e_start = ec2.date_input("Window start", value=pd.Timestamp("2026-07-15"),
                             min_value=pd.Timestamp("2026-03-15"), max_value=pd.Timestamp("2026-08-25"))
    e_end = ec3.date_input("Window end", value=pd.Timestamp("2026-07-21"),
                           min_value=pd.Timestamp("2026-03-15"), max_value=pd.Timestamp("2026-08-25"))
    ec4.markdown(
    '<div style="height: 29px;"></div>',
    unsafe_allow_html=True
    )
    go = ec4.button("Investigate →", type="primary", use_container_width=True)
    st.caption("Try: Jul 15–21 (multi-factor drop) / Aug 17–23 (should abstain) / "
               "Jul 26–29 (unscripted spike the sweep discovered) / any quiet week (should say so)")
    if go:
        if pd.Timestamp(e_end) < pd.Timestamp(e_start):
            st.error("Window end precedes start.")
        else:
            target_kpi = e_kpi
            if e_kpi.startswith("all"):
                scan = investigate.scan_window(str(e_start), str(e_end))
                st.markdown("**Every governed KPI over this window** (vs auto-baseline "
                            f"{scan['baseline'][0]} → {scan['baseline'][1]})")
                sc = pd.DataFrame(scan["rows"])
                sc["kpi"] = sc["kpi"].map(nice)
                sc = sc.rename(columns={"kpi": "KPI", "pct": "% move",
                                        "moved": "Moved ≥5%", "material": "Material"})
                st.dataframe(sc, hide_index=True, use_container_width=True)
                st.caption(f"Window business impact: ₹{scan['window_impact_inr']/1e5:.2f}L "
                           f"(one number for the window — materiality is judged in ₹, per contract)")
                target_kpi = scan["headline"]
                if scan["any_material"]:
                    st.caption(f"Headline auto-selected: **{nice(target_kpi)}** "
                               "(gmv preferred when material, else the most material mover — "
                               "the same rule the sweep's coherence merge uses). "
                               "Investigating it now; pick a specific KPI above to override.")
                else:
                    st.caption(f"No KPI clears its ₹ materiality threshold — investigating "
                               f"**{nice(target_kpi)}** anyway to show the honest no-movement verdict.")
            with st.spinner("Probing dataset + retrieving evidence + ranking…"):
                rr = investigate.investigate(target_kpi, str(e_start), str(e_end),
                                             role=role, backend=backend_key, model=model)
            evx, decx = rr["event"], rr["decision"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Movement", evx["magnitude"])
            m2.metric("Impact", f"₹{evx['impact_inr']/1e5:.2f}L")
            m3.metric("Verdict", nice(decx["mode"]))
            m4.metric("Baseline", f"{evx['baseline_dates'][0][5:]} → {evx['baseline_dates'][1][5:]}")
            render_narrative(rr, decx["mode"])
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Decomposition**")
                dd = evx["decomposition"]
                st.bar_chart(pd.DataFrame({"pp": [float(v.replace("pp", "")) for v in dd.values()]},
                                          index=list(dd.keys())))
                st.markdown("**Ranked hypotheses**")
                st.dataframe(nice_cols(pd.DataFrame(
                    [{"hypothesis": nice(h["name"]), "confidence": h["confidence"],
                      "calibrated": h.get("calibrated"),
                      "caps": nice_caps(h["caps"]), **h["criteria"]}
                     for h in rr["finding"]["hypotheses"]])),
                    hide_index=True, use_container_width=True)
            with cc2:
                render_actions(rr["actions"])
                

# ---------------- S1 insight ----------------
with tab_scenario:
    pick = st.radio("Scenario", ["S1: Multi-factor", "S2: Abstention", "S3: Sparse KPI"],
                    horizontal=True, label_visibility="collapsed")

    if pick == "S1: Multi-factor":
        ev, dec = S1["event"], S1["decision"]
        st.subheader(f"{ev['id']} — {nice(dec['mode'])} ")
        render_narrative(S1)
        st.caption(
        f"Insight prepared for {nice(role)}"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Decomposition**")
            d = ev["decomposition"]
            st.bar_chart(pd.DataFrame({"pp": [float(v.replace("pp", "")) for v in d.values()]},
                                      index=list(d.keys())))
            st.markdown("**Ranked hypotheses**")
            rows = [{"hypothesis": nice(h["name"]), "confidence": h["confidence"],
                     "caps": nice_caps(h["caps"]),
                     **h["criteria"]} for h in S1["finding"]["hypotheses"]]
            st.dataframe(nice_cols(pd.DataFrame(rows)), hide_index=True, use_container_width=True)
            
        with c2:
            render_actions(S1["actions"])
           

    elif pick == "S2: Abstention":
        ev = S2["event"]
        st.subheader(f"{ev['id']} — Abstain")
        render_narrative(S2)
        st.markdown("**Ranked Hypothesis**")
        st.dataframe(pd.DataFrame([{"Hypothesis": nice(h["name"]), "Confidence": h["confidence"],
                                    "Caps": nice_caps(h["caps"])}
                                   for h in S2["finding"]["hypotheses"]]), hide_index=True)
        

    else:
        st.subheader(f"AirPro launch ({results['S3']['history_days']} days) — limited history, no anomaly detected")
        s3 = results["S3"]
        curves = pd.DataFrame(s3["curves"]).set_index("day")
        curves.columns = ["AirPro (normalised)", "analog band low", "analog band high"]
        st.line_chart(curves)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("History", f"{s3['history_days']}d / {s3['min_history']}d")
        c2.metric(f"Day {s3['eval_day']} observed", s3["observed"])
        c3.metric("Analog band", f"{s3['band'][0]}–{s3['band'][1]}")
        c4.metric("Days inside band", s3["days_in_band"])
        (st.success if s3["in_band"] else st.warning)(
            ("No alert — level sits inside the analog launch band. " if s3["in_band"]
             else "Flagged — level sits outside the analog band. ") + s3["note"])
        st.caption( f"AirPro launched {s3['launch']}. Analogs SoundMax and BassLite are launch-day-indexed and normalised from the same dataset." "<br>Branch auto-retires at 60 days of history.",
    unsafe_allow_html=True
     )


# ---------------- all scenarios ----------------
with tab_scen:
    st.subheader("Every planted scenario, measured from the dataset")
    st.caption("Each row states what was injected by the generator and what the engine measures now. "
               "ground_truth.yaml holds the injection spec.")
    gw_day, gw_base = R.GW_DAY, R.GW_BASE
    so_days = A.stockout_days("FASH-SNEAKER", "South", "2026-06-10", "2026-06-13")
    d1, d2, ch1, ch2 = R.D1, R.D2, R.CH1, R.CH2
    rows = [
        {"id": "S1a", "scenario": "Coupon expiry (Jul 15)",
         "expectation": "conversion falls while realised price rises",
         "measured": f"CR {d1['conversion']:+.1f}pp, price {d1['price']:+.1f}pp, "
                     f"discount {d1['pre']['discount']:.3f} -> {d1['post']['discount']:.3f}",
         "engine": "Dual cause (leading hypothesis)"},
        {"id": "S1b", "scenario": "RivalMart sale (Jul 15-21)",
         "expectation": "organic traffic falls, paid unaffected",
         "measured": f"organic {ch1['organic_pct']:+.1f}%, paid {ch1['paid_pct']:+.1f}%",
         "engine": "Dual cause (second cause, monitor-only)"},
        {"id": "S2", "scenario": "Ambiguous week (Aug 17-23)",
         "expectation": "material but no dominant signature; a source is missing",
         "measured": f"GMV {d2['total_pct']:+.1f}%, organic {ch2['organic_pct']:+.1f}%, "
                     f"paid {ch2['paid_pct']:+.1f}%, marketing.xlsx rows = 0",
         "engine": "Abstain + clarification request"},
        {"id": "S3", "scenario": f"AirPro launch ({results['S3']['launch']})",
         "expectation": "too little history for a baseline; judge against analogs",
         "measured": f"{results['S3']['history_days']}d history, day {results['S3']['eval_day']} "
                     f"level {results['S3']['observed']} vs band {results['S3']['band']}",
         "engine": "No alert (in band)" if results["S3"]["in_band"] else "Flagged, 0.6x confidence"},
        {"id": "GW", "scenario": "Payment gateway blip (Aug 5)",
         "expectation": "checkout completion drops for one day only",
         "measured": f"completion {gw_day:.3f} vs {gw_base:.3f} baseline",
         "engine": "Rejected: transient spike"},
        {"id": "SO", "scenario": "Stockout (Jun 10-13)",
         "expectation": "supply-side conversion drag, explainable",
         "measured": f"FASH-SNEAKER South stock_on_hand = 0 on {so_days} days",
         "engine": "Rejected: explained by known driver"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown("**Data quality flaws planted (handled by the reconciler before analysis)**")
    mkt_raw = A.MKT_RAW
    st.dataframe(pd.DataFrame([
        {"flaw": "duplicate campaign row", "present": int(mkt_raw.duplicated().sum()),
         "handling": "dropped in clean_mkt() before spend is summed"},
        {"flaw": "end_date year typo (2062)",
         "present": int(mkt_raw.end_date.astype(str).str.contains("2062").sum()),
         "handling": "date-range sanity filter"},
        {"flaw": "missing inventory sku-region-days",
         "present": f"{(1 - len(A.INV)/(len(pd.date_range('2026-03-01','2026-08-26'))*A.INV.sku.nunique()*2))*100:.1f}%",
         "handling": "absent row != zero stock"},
        {"flaw": "cancelled orders",
         "present": f"{(A.ORDERS.payment_status=='cancelled').mean()*100:.1f}%",
         "handling": "excluded from every KPI per contract"},
        {"flaw": "web analytics over-reports orders", "present": "~3%",
         "handling": "sales.db is truth for money, web for behaviour"},
        {"flaw": "web parquet one day behind orders (T+1)", "present": "yes",
         "handling": "freshness walk blocks the latest CR day"},
    ]), hide_index=True, use_container_width=True)

# ---------------- feedback ----------------
with tab_fb:
    st.subheader("Analyst feedback → bounded, evidence-attributed learning")
    hyp = st.selectbox("Hypothesis", [h["name"] for h in S1["finding"]["hypotheses"]],
                       format_func=nice)
    verdict = st.radio("Verdict", ["confirm", "reject"], horizontal=True)
    before = {nice(h["name"]): h["confidence"] for h in S1["finding"]["hypotheses"]}
    if st.button("Submit feedback and re-run"):
        touched = apply_feedback(S1["run_id"], hyp, verdict, S1_DOCKET)
        after_run = run_all(role, "none")["S1"]
        after = {nice(h["name"]): h["confidence"] for h in after_run["finding"]["hypotheses"]}
        st.markdown("**Source reliabilities updated (±0.05, clamped [0.10, 0.99])**")
        st.json({k: f"{v[0]} → {v[1]}" for k, v in touched.items()})
        cmp = pd.DataFrame({"before": before, "after": after})
        cmp["Δ"] = (cmp.after - cmp.before).round(3)
        st.dataframe(cmp)
        st.caption("Event logged to feedback_log.jsonl (replayable; future GBM training data). "
                   "Beta priors update per hypothesis type; corrections enter contracts as status:candidate.")
    st.divider()
    st.subheader("Learning over simulated time — replay analyst verdicts")
    st.caption("Pushes N rounds of ground-truth-consistent verdicts (10% analyst noise) through the "
               "REAL feedback path: source reliabilities, per-driver Beta priors, and the isotonic "
               "calibration map all update. This is the cold-start bridge the GBM slot trains on later.")
    n_rounds = st.slider("Rounds to replay", 2, 12, 6)
    fc1, fc2 = st.columns(2)
    if fc1.button("Replay verdicts →", type="primary"):
        rep = calibrate.replay_feedback(n_rounds=n_rounds)
        st.markdown(f"**Priors drift after {rep['rounds']} rounds** (alpha=confirms, beta=rejects)")
        st.dataframe(pd.DataFrame([
            {"driver": nice(k), "before": str(v["before"]), "after": str(v["after"]),
             "mean before": round(v["before"][0]/sum(v["before"]), 2),
             "mean after": round(v["after"][0]/sum(v["after"]), 2)}
            for k, v in rep["priors"].items()]), hide_index=True)
        if rep["reliability"]:
            st.markdown("**Source reliabilities moved (bounded ±0.05/verdict)**")
            st.json({k: f"{v['before']} → {v['after']}" for k, v in rep["reliability"].items()})
        st.markdown(f"**Calibration:** {rep['calibration']['samples']} samples · "
                    f"{'WARM — isotonic map active' if rep['calibration']['warm'] else 'cold — identity map'}")
        cv = calibrate.curve()
        if cv:
            st.line_chart(pd.DataFrame({"calibrated": cv[1]}, index=cv[0]))
    if fc2.button("Reset learned state"):
        calibrate.reset(); reset_reliability()
        st.success("Priors, calibration samples and reliabilities reset to cold start.")

# ---------------- telemetry ----------------
with tab_tel:
    st.subheader("Per-insight telemetry + audit snapshot")

    with st.expander("What is computed from data (everything)", expanded=False):
        st.markdown(
            f"""
            **Computed live from `trend_lens_data` (6-month generated dataset)**
        
            - **KPI analytics:** KPI series and trends.
            - **Continuous anomaly detection:** Weekday-z, STL residual, and CUSUM ensemble with a six-check gate across the full KPI history.
            - **Decomposition:** GMV = Sessions × CR × AOV, with AOV split into price and mix.
            - **Any-window investigation:** Signature probes → evidence docket → graph-derived hypotheses → decision gate.
            - **News retrieval:** Vector retrieval with LLM call #1 extraction, cached with a deterministic heuristic fallback.
            - **Channel analysis:** Organic vs. paid channel contrasts.
            - **Launch analysis:** Analog launch bands for limited-history products.
            - **Funnel diagnostics:** Funnel ratios and conversion analysis.
            - **Inventory analysis:** Stockout scans.
            - **Data governance:** Freshness watermarks, lineage tracing, and entitlement scoping.
            - **Evidence evaluation:** Fuzzy ranking, conflict detection, and decision gating.
            - **Decision support:** Actions and narrative generation.
            - **Confidence calibration:** Beta priors and isotonic calibration.
            - **Learning & observability:** Analyst feedback and telemetry.
        
            **Deliberately simulated:**
        
            - **Dataset:** Generated with planted ground truth.
            - **Analyst verdicts:** Replayed verdicts in the Feedback tab.
            - **Engine conclusions:** All detections, evidence evaluation, decisions, confidence scores, and actions are computed from the available data.
            """
        )

    for r, label in [(S1, "S1"), (S2, "S2")]:
        t = r["telemetry"]
        with st.container(border=True):
            st.markdown(f"**{label} · run_id `{t['run_id']}` · role {t['role']}**")
            c = st.columns(5)
            c[0].metric("Total", f"{t['timers_s']['total']*1000:.0f} ms")
            c[1].metric("Ranking", f"{t['timers_s']['ranking']*1000:.1f} ms")
            c[2].metric("NLG", f"{t['timers_s']['nlg']*1000:.0f} ms")
            c[3].metric("LLM calls", t["llm"]["calls_this_run"])
            c[4].metric("Marginal cost", t["marginal_cost"].replace("Rs 0 (local/template)", "₹0"))
            
    st.markdown("**LLM vs non-LLM boundary**")
    st.table(pd.DataFrame([
        ["Security, Reconcile, Detection, Decomposition", "SQL/rules, graph walk, statsmodels/CUSUM/ruptures, arithmetic", "no"],
        ["Retrieval: query writing", "LLM call #1", "YES"],
        ["Retrieval: search + structured pulls", "embeddings + SQL", "no"],
        ["Ranking, Gate, Actions", "fuzzy, DiD, Beta priors, Dempster–Shafer, lookups", "no"],
        ["Narration", "LLM call #2 + mechanical checker + template fallback", "YES"],
        ["Feedback, Telemetry", "counters, logs", "no"],
    ], columns=["Stage", "Method", "LLM?"]))
