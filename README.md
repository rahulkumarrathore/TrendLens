# TrendLens — KPI Intelligence-to-Action Engine (Prototype)

Decision workspace that detects material KPI movements, decomposes them into
interacting drivers, ranks root causes with soft computing, abstains when evidence
is insufficient or contradictory, and narrates persona-specific insights with
traceable evidence. LLM is used for narration only (swappable: offline Ollama /
Anthropic API / deterministic template).

## Run
```
pip install streamlit pandas pyyaml openpyxl pyarrow
streamlit run app.py
```
Optional LLM backends (select in sidebar):
- **Ollama (offline)**: install ollama, `ollama pull llama3.3:8b`, keep it running on :11434
- **API**: `export ANTHROPIC_API_KEY=...`
- **None**: deterministic template renderer (always available; also the fallback
  when the LLM narrative fails the citation/number checker)

## Platform choice
Fully custom Python (Streamlit + SQLite/parquet/xlsx + statsmodels-class logic) —
fastest to implement and run, zero platform lock-in. Slide mapping if asked:
sources emulate the raw layer of a lakehouse (Databricks/Snowflake); contracts =
semantic layer (dbt/Fabric); the engine sits on top either way.
Component labels per the brief: native (none) · configured (statsmodels, vector
store) · custom-built (contracts, graph, decomposition, fuzzy ranking, gate,
checker, telemetry) · externally integrated (LLM backends).

## Layout
```
app.py                  Streamlit decision workspace (6 tabs)
engine/core.py          entitlements, fuzzy ranking, DS kappa, gate, actions
engine/nlg.py           payload, swappable LLM, mechanical checker, template fallback
engine/pipeline.py      orchestrator, feedback (bounded, event-logged), telemetry, audit
engine/graph.py         Stage 0.4 knowledge graph, compiled from contracts (LIVE)
engine/reconcile.py     Stage 1.2 watermarks + freshness walk (LIVE on sample files)
engine/detection.py     Stage 2 outputs (fixtures; STL/CUSUM/gate pending datagen)
engine/decompose.py     Stage 3 outputs (fixtures; identity math pending datagen)
engine/retrieval.py     Stage 4 dockets + hypotheses (live-valued from sample files)
engine/scenarios.py     back-compat shim re-exporting the above
contracts/*.yaml        8 KPI semantic contracts (formula, drivers, thresholds,
                        levers, access, analogs, min_history)
personas.yaml           4 role rendering profiles (depth/vocab/action-frame/channels)
source_reliability.yaml base trust per source (feedback adjusts, bounded)
trend_lens_data/        the 4 sample sources (sales.db, parquet, xlsx, news/)
make_config.py          regenerates the yaml config layer
```

## What is live vs simulated
Evidence marked **live** is computed from the sample files at runtime (news
snippets, campaign spend, funnel totals, stockout rows). Evidence marked
**simulated** carries the planted full-dataset values (decomposition pp,
DiD p-value) — the 18-month generator replaces these; every interface is final.

## Demo script (6 beats, ~5 min)
1. Anomaly feed: AE-001 confirmed via 6-check gate; rejection log shows 4 named
   non-alerts (morning-CR `incomplete_data`, festive, in-band AirPro, held persistence)
2. S1 insight as CFO: dual-cause narrative (coupon 0.92 + competitor 0.88),
   decomposition chart, clickable evidence with freshness/reliability, κ=0.33
   with the contradictory boom-snippet visibly discounted
3. Toggle role → north_manager: numbers change (−12.1%, ₹1.31L), CFO-tier
   evidence disappears; analyst_intern: actions section gone
4. S2 tab: abstention with named reasons, clarification request, no action slot
5. S3 tab: analog launch band, day-18 dip inside it → correctly quiet
6. Feedback: reject competitor → source reliabilities −0.05 → re-run shows
   score drop; telemetry tab: per-stage ms, LLM calls, ₹0 marginal cost, boundary table

## Data generation (6-month synthetic dataset)
```
python datagen/generate.py                      # seed 42, asof 2026-08-26, ~2000 order rows
python datagen/generate.py --orders 4000 --seed 7
```
Window 2026-03-01 .. 2026-08-26 | 2 regions | 3 categories | 33 SKUs.
Overwrites the four sources in trend_lens_data/ (schemas frozen) and writes
ground_truth.yaml recording every injected event.

### Planted scenarios (measured from the generated data)
| id | scenario | signal |
|----|----------|--------|
| S1a | coupon expiry Jul 15 | conversion down, realised price UP, discount 0.100 -> 0.002 |
| S1b | RivalMart sale Jul 15-21 | organic sessions down ~15%, paid flat (~-1%) |
| S2 | ambiguous week Aug 17-23 | material drop, both channels soft, marketing.xlsx upload SKIPPED, contradictory news Aug 20 |
| S3 | AirPro launch Jul 25 | ~32d history (< 60 min_history), 2 analogs with full curves |
| GW | gateway blip Aug 5 | checkout completion drops one day, recovers |
| SO | stockout Jun 10-13 | FASH-SNEAKER South stock_on_hand = 0 |

Quality flaws planted: duplicate campaign row, end_date 2062 typo, ~2% missing
inventory sku-region-days, ~3.5% cancelled orders, web over-reports orders ~3%,
returns lag 3-10 days, web parquet one day behind orders (T+1).

### Note on dataset size
~2000 order rows over 6 months is ~11 orders/day, so weekly aggregates carry
+/-15% sampling noise and mix terms are jumpy. Scenario *directions* and
fingerprints are stable; *magnitudes* are not. Run with --orders 4000-6000 for
tighter magnitudes if a demo needs them.

## What is computed vs still simulated
Computed live over the dataset: KPI series, Level-1 decomposition (identity split
+ price/mix shift-share), channel contrasts, analog launch bands, funnel ratios,
stockout days, freshness watermarks and lineage walk, entitlement scoping
(north_manager numbers are computed on North rows only), fuzzy ranking, DS
conflict, ambiguity gate, actions, narrative, feedback, telemetry.

Not yet implemented: the continuous STL/CUSUM candidate pass and six-check gate
sweeping every KPI (windows are evaluated on demand instead); embeddings + LLM
call #1 for unstructured retrieval (news matched by date/topic); isotonic
calibration and the GBM precedent model (cold start).

## Gap-closure build (final)

All previously "not yet implemented" stages are now computed live:

| Module | Stage | What it does |
|---|---|---|
| `engine/sweep.py` | 2 | Continuous detection over EVERY governed KPI's full history: weekday-z (median/MAD, contamination-proof) + robust STL residual + resetting two-sided CUSUM, ensemble vote, then the six-check gate (persistence, ensemble, materiality in ₹, freshness walk, business calendar, cross-KPI coherence merge). Confirmed events carry honest gate traces; every rejection carries a reason code. Sparse KPIs route to the analog-band branch. |
| `engine/investigate.py` | 3–6 | Generalized investigation of ANY KPI over ANY window: signature probes (discount series, strict organic-vs-paid contrast, campaign coverage, funnel step, zero-stock scan, launch-SKU share) assemble the docket; hypotheses derive from the knowledge graph; contribution_match is computed from the measured decomposition; the standard pipeline ranks, gates, acts, narrates. Auto-baseline slides past sweep-confirmed anomalies. Immaterial windows short-circuit to `no_material_movement`. |
| `engine/vector_news.py` | 4 | TF-IDF + cosine vector retrieval over the news corpus with a temporal prior; LLM call #1 extracts structured evidence (claims stripped of numbers — external evidence corroborates, never quantifies), cached to `news_extract_cache.json`; deterministic heuristic fallback. |
| `engine/calibrate.py` | 5 | Per-driver Beta priors persisted to `priors.json`; isotonic calibration (PAV, implemented directly) mapping raw confidence → empirical precision, identity until 12 samples; `replay_feedback()` drives ground-truth-consistent verdicts through the real feedback path. |
| `validate_sweep.py` | — | Diffs the sweep against planted ground truth: S1 + S2 confirmed, festivals/gateway/stockout/sparse correctly rejected with reasons, plus one unscripted-but-real discovery (the Jul 28–29 launch ramp) verified against raw data. |

Run the validation: `python validate_sweep.py`
