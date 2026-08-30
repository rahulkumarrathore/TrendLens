"""TrendLens — decision workspace UI (Streamlit). Renders pipeline outputs; the
persona toggle and feedback re-run invoke the pipeline live."""
import streamlit as st
import pandas as pd
from engine.pipeline import run_all, apply_feedback, reset_reliability, RELIABILITY
from engine.scenarios import S1_DOCKET, S3_INFO
from engine import core
from engine import analytics as A
from engine import retrieval as R
from engine import sweep, investigate, calibrate
from engine.labels import nice, nice_caps, nice_cols

st.set_page_config(page_title="TrendLens", layout="wide", page_icon="📉")
st.markdown("""<style>
.block-container{padding-top:2.2rem;} h1{letter-spacing:-0.02em;}
.stTabs [data-baseweb="tab-list"]{position:sticky;top:2.8rem;z-index:999;
  background:var(--background-color,white);border-bottom:1px solid #e6e6e6;padding-top:.3rem;}
.stTabs [data-baseweb=tab]{font-size:0.95rem;}
[data-testid=stMetricValue]{font-size:1.5rem;}
</style>""", unsafe_allow_html=True)

# ---------------- sidebar ----------------
st.sidebar.title("TrendLens")
st.sidebar.caption("KPI intelligence-to-action engine")
role = st.sidebar.selectbox("Role (entitlement + persona)",
                            ["cfo", "north_manager", "marketing_exec", "analyst_intern"],
                            format_func=nice)
backend = st.sidebar.radio("LLM backend",
                           ["Deterministic template (no LLM)", "Ollama — local LLM", "Google Gemini API"],
                           help="Swappable per run. Template fallback guarantees a correct narrative either way.")
backend_key = {"Deterministic template (no LLM)": "none", "Ollama — local LLM": "ollama", "Google Gemini API": "api"}[backend]
model = st.sidebar.text_input("Model", value="llama3.2:3b" if backend_key == "ollama" else "google-gemini-4-5-20251001") \
        if backend_key != "none" else None
if st.sidebar.button("Reset feedback effects"):
    reset_reliability(); st.sidebar.success("Source reliabilities reset")
_cov = A.coverage()
st.sidebar.markdown("**Dataset loaded**")
st.sidebar.code(f"{_cov['start']} -> {_cov['end']}\n{_cov['months']} months ({_cov['days']} days)\n"
                f"{_cov['order_rows']:,} order rows / {_cov['orders']:,} orders\n"
                f"{_cov['web_rows']:,} web / {_cov['campaign_rows']} campaigns / {_cov['news_files']} news\n"
                f"{_cov['skus']} SKUs, {len(_cov['regions'])} regions, {len(_cov['categories'])} categories",
                language=None)

scope = core.audit_scope(role)
st.sidebar.markdown("**Scope applied**")
st.sidebar.code(f"rows: {scope['rows']}\ncols dropped: {scope['columns_dropped'] or '—'}\n"
                f"domains refused: {scope['domains_refused'] or '—'}", language=None)

results = run_all(role, backend_key, model)
S1, S2 = results["S1"], results["S2"]

tab_feed, tab_explore, tab_insight, tab_abstain, tab_sparse, tab_scen, tab_fb, tab_tel = st.tabs(
    ["Anomaly feed", "Investigate KPI", "Scenario 1 multi-factor", "Scenario 2 abstention",
     "Scenario 3 sparse KPI", "All scenarios", "Feedback loop", "Telemetry & audit"])

with st.expander("What is computed from data (everything)", expanded=False):
    st.markdown(
        "**Computed live from `trend_lens_data/` (6-month generated dataset, "
        f"{len(A.ORDERS):,} order rows):** KPI series, the continuous detection sweep "
        "(weekday-z / STL residual / CUSUM ensemble + six-check gate over every governed "
        "KPI's full history — see the feed), Level-1 decomposition (GMV = Sessions x CR x AOV, "
        "AOV split into price/mix), generalized any-window investigation (signature probes -> "
        "docket -> graph-derived hypotheses -> gate), vector retrieval over the news corpus "
        "with LLM call #1 extraction (cached; heuristic fallback keeps it deterministic), "
        "channel contrasts, analog launch bands, funnel ratios, stockout scans, freshness "
        "watermarks and the lineage walk, entitlement scoping, fuzzy ranking, conflict, gate, "
        "actions, narrative, Beta-prior + isotonic calibration, feedback and telemetry.")
    st.markdown(
        "**Deliberately simulated:** the dataset itself (datagen with planted ground truth) "
        "and the replayed analyst verdicts in the feedback tab. Everything the engine "
        "*concludes* is computed.")

# ---------------- anomaly feed ----------------
with tab_feed:
    st.subheader("KPI history — full dataset window")
    hc1, hc2 = st.columns([3, 1])
    kpi_choice = hc2.selectbox("Series", ["GMV", "Orders", "Sessions", "AOV", "Conversion rate %"])
    smooth = hc2.checkbox("7-day smoothing", value=True)
    region_scope = None if core.CONTRACTS["gmv"]["access"][role].get("rows", "all") == "all" else "North"
    hist = A.daily_series(region_scope)[[kpi_choice]]
    if smooth:
        hist = hist.rolling(7, min_periods=1).mean()
    hc1.line_chart(hist, height=260)
    st.caption(
        f"{_cov['start']} to {_cov['end']} — {_cov['months']} months of daily history"
        + (f", scoped to region North for this role" if region_scope else "")
        + ". Planted event windows: "
        + " · ".join(f"{n} ({s[5:]}–{e[5:]})" for n, s, e in A.EVENT_WINDOWS[:3])
        + ". Seasonal peaks (EOSS early Jun, Raksha Bandhan late Aug) are learned as baseline, not flagged."
    )
    st.divider()
    st.subheader("Confirmed by the continuous sweep (no target windows given)")
    st.caption("3-detector ensemble (weekday-z · STL residual · CUSUM) + six-check gate over every "
               "governed KPI's full 6-month history. Click any event to investigate it live.")
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
                (st.info if rr["decision"]["mode"] != "abstain" else st.warning)(rr["narrative"])
                st.dataframe(pd.DataFrame(
                    [{"Hypothesis": nice(h["name"]), "Confidence": h["confidence"],
                      "Calibrated": h.get("calibrated"), "Caps": nice_caps(h["caps"])}
                     for h in rr["finding"]["hypotheses"]]), hide_index=True)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Curated walkthrough events (deep instrumentation)")
        for r, label in [(S1, "S1"), (S2, "S2")]:
            ev = r["event"]
            with st.container(border=True):
                a, b, c = st.columns([2, 1, 1])
                a.markdown(f"**{ev['id']}** · {nice(ev['kpi'])} {ev['magnitude']} ({ev['window']})")
                b.metric("Impact", f"₹{ev['impact_inr']/1e5:.2f}L")
                c.markdown(f"`{nice(ev['type'])}` · **{nice(r['decision']['mode'])}**")
                with st.expander("Gate trace (6-check confirmation)"):
                    for g in ev["gate_trace"]:
                        st.markdown(f"- {g}")
    with c2:
        st.subheader("Restraint log — sweep rejections by reason")
        st.caption("Every non-alert has a named reason — non-anomalies are understood, not swallowed. "
                   f"{len(sweep.REJECTIONS)} candidates rejected, {len(sweep.CONFIRMED)} confirmed.")
        _rc = pd.Series([r_["reason"] for r_ in sweep.REJECTIONS]).value_counts()
        st.bar_chart(_rc.rename(index=nice))
        pick = st.selectbox("Show rejections with reason", ["(all curated)"] + list(_rc.index),
                            format_func=lambda x: x if x.startswith("(") else nice(x))
        shown = results["rejections"] if pick == "(all curated)" else \
            [r_ for r_ in sweep.REJECTIONS if r_["reason"] == pick][:8]
        for rj in shown:
            with st.container(border=True):
                st.markdown(f"**{rj['candidate']}**")
                st.markdown(f"`{nice(rj['reason'])}` — {rj['detail']}")

# ---------------- explore any window ----------------
with tab_explore:
    st.subheader("Investigate ANY KPI over ANY timeline window")
    st.caption("The system automatically runs the key checks an analyst would normally perform: it looks at discount trends, compares organic and paid traffic, checks campaign coverage and funnel performance, detects products that went out of stock, measures the share of newly launched products, and retrieves relevant news. These findings are then organized, prioritized, filtered for reliability, and presented as a clear narrative. For comparison, the system automatically selects 14 reliable days as the baseline, skipping days where confirmed anomalies could distort the analysis.")
    ec1, ec2, ec3, ec4 = st.columns([1.2, 1, 1, 1])
    e_kpi = ec1.selectbox("KPI", ["all", "gmv", "sessions",
                                  "conversion_rate", "orders_kpi", "aov", "net_revenue"],
                          format_func=lambda x: "All KPIs" if x == "all" else nice(x))
    e_start = ec2.date_input("Window start", value=pd.Timestamp("2026-07-15"),
                             min_value=pd.Timestamp("2026-03-15"), max_value=pd.Timestamp("2026-08-25"))
    e_end = ec3.date_input("Window end", value=pd.Timestamp("2026-07-21"),
                           min_value=pd.Timestamp("2026-03-15"), max_value=pd.Timestamp("2026-08-25"))
    ec4.markdown("&nbsp;")
    go = ec4.button("Investigate →", type="primary", use_container_width=True)
    st.caption("Try: Jul 15–21 (multi-factor drop) · Aug 17–23 (should abstain) · "
               "Jul 26–29 (unscripted spike the sweep discovered) · any quiet week (should say so)")
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
            (st.warning if decx["mode"] in ("abstain",) else
             st.success if decx["mode"] == "no_material_movement" else st.info)(rr["narrative"])
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Level-1 decomposition (computed for this window)**")
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
                st.caption(f"κ = {rr['finding']['kappa']} · gap(top2) = {rr['finding']['gap_top2']} · "
                           f"calibration: {calibrate.status()}")
            with cc2:
                if rr["actions"]:
                    st.markdown("**Actions**")
                    for a_ in rr["actions"]:
                        with st.container(border=True):
                            st.markdown(f"**{nice(a_['driver'])}** · lever `{nice(a_['lever'])}` · "
                                        f"owner {nice(a_['owner'])}")
                            st.markdown(f"{a_['action']} — {a_['expected_impact']} "
                                        f"(conf {a_['confidence']:.2f})")
                            st.caption(f"Monitoring: {a_['monitoring_plan']}")
                st.markdown("**Evidence docket (all probes, confirming and refuting)**")
                for e_ in rr["docket"]:
                    with st.expander(f"[{e_['id']}] {e_['claim'][:70]}"):
                        st.markdown(f"**Value:** {e_['value']}")
                        st.markdown(f"source `{e_['source']}` · fresh **{e_['fresh']}** · "
                                    f"*{e_['method']}* · direction **{e_['direction']}** · "
                                    f"reliability {RELIABILITY.get(e_['source'], 0.5)}")
                        if e_.get("conflicts_with"):
                            st.warning(f"Conflicts with {e_['conflicts_with']} — feeds κ")

# ---------------- S1 insight ----------------
with tab_insight:
    ev, dec = S1["event"], S1["decision"]
    st.subheader(f"{ev['id']} — {nice(dec['mode'])} · persona: {nice(role)}")
    st.info(S1["narrative"])
    st.caption(f"narrative_source: {S1['telemetry']['llm'].get('narrative_source')} · "
               f"channels: {', '.join(core.PERSONAS[role]['channels'])}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Level-1 decomposition (exact, additive)**")
        d = ev["decomposition"]
        st.bar_chart(pd.DataFrame({"pp": [float(v.replace("pp", "")) for v in d.values()]},
                                  index=list(d.keys())))
        st.markdown("**Ranked hypotheses (fuzzy + causal + caps)**")
        rows = [{"hypothesis": nice(h["name"]), "confidence": h["confidence"],
                 "caps": nice_caps(h["caps"]),
                 **h["criteria"]} for h in S1["finding"]["hypotheses"]]
        st.dataframe(nice_cols(pd.DataFrame(rows)), hide_index=True, use_container_width=True)
        st.caption(f"Dempster–Shafer conflict κ = {S1['finding']['kappa']} · gap(top2) = {S1['finding']['gap_top2']}")
    with c2:
        st.markdown("**Actions (driver → lever → action → impact → owner → confidence → monitoring)**")
        for a in S1["actions"]:
            with st.container(border=True):
                st.markdown(f"**{nice(a['driver'])}** · lever: `{nice(a['lever'])}` · owner: {nice(a['owner'])}")
                st.markdown(f"{a['action']} — {a['expected_impact']} (conf {a['confidence']:.2f})")
                st.caption(f"Monitoring: {a['monitoring_plan']}")
        st.markdown("**Evidence panel (click to trace)**")
        for e in S1["docket"]:
            tag = "live" if e.get("live") else "simulated (full datagen pending)"
            with st.expander(f"[{e['id']}] {e['claim'][:70]}…" if len(e['claim']) > 70 else f"[{e['id']}] {e['claim']}"):
                st.markdown(f"**Value:** {e['value']}")
                st.markdown(f"source `{e['source']}` · fresh **{e['fresh']}** · method *{e['method']}* · "
                            f"reliability {RELIABILITY.get(e['source'], 0.5)} · direction {e['direction']} · _{tag}_")
                if e.get("conflicts_with"):
                    st.warning(f"Conflicts with {e['conflicts_with']} — feeds κ")

# ---------------- S2 abstention ----------------
with tab_abstain:
    ev = S2["event"]
    st.subheader(f"{ev['id']} — engine abstains and requests clarification")
    st.warning(S2["narrative"])
    st.markdown("**Why the gate refused to commit**")
    for w in S2["decision"]["why"]:
        st.markdown(f"- {w}")
    st.markdown("**What resolves it (clarification request, auto re-evaluation armed)**")
    for r_ in ev["resolves"]:
        st.markdown(f"- {r_}")
    st.markdown("**Hypothesis scores at abstention**")
    st.dataframe(pd.DataFrame([{"Hypothesis": nice(h["name"]), "Confidence": h["confidence"],
                                "Caps": nice_caps(h["caps"])}
                               for h in S2["finding"]["hypotheses"]]), hide_index=True)
    st.caption("No action slot exists in abstain mode — over-commitment is structurally unrepresentable.")

# ---------------- S3 sparse ----------------
with tab_sparse:
    st.subheader(f"AirPro launch ({results['S3']['history_days']} days history) — analog-band fallback, correctly quiet")
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
    st.caption(f"AirPro launched {s3['launch']}. Analogs SoundMax and BassLite are launch-day-indexed "
               "and normalised from the same dataset. Branch auto-retires at 60 days of history.")


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
    hyp = st.selectbox("Hypothesis (S1)", [h["name"] for h in S1["finding"]["hypotheses"]],
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
            st.caption(f"backend {t['llm'].get('backend')} · tokens in/out "
                       f"{t['llm'].get('tokens_in', 0)}/{t['llm'].get('tokens_out', 0)} · "
                       f"narrative_source {t['llm'].get('narrative_source')} · "
                       f"counterfactual hosted-API cost {t['counterfactual_api_cost']} · "
                       f"audit snapshot appended to audit_log.jsonl")
    st.markdown("**LLM vs non-LLM boundary**")
    st.table(pd.DataFrame([
        ["Security, Reconcile, Detection, Decomposition", "SQL/rules, graph walk, statsmodels/CUSUM/ruptures, arithmetic", "no"],
        ["Retrieval: query writing", "LLM call #1", "YES"],
        ["Retrieval: search + structured pulls", "embeddings + SQL", "no"],
        ["Ranking, Gate, Actions", "fuzzy, DiD, Beta priors, Dempster–Shafer, lookups", "no"],
        ["Narration", "LLM call #2 + mechanical checker + template fallback", "YES"],
        ["Feedback, Telemetry", "counters, logs", "no"],
    ], columns=["Stage", "Method", "LLM?"]))
