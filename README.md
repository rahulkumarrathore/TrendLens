# TrendLens

TrendLens is a KPI intelligence-to-action decision workspace that detects material business-metric movements, explains the movement through measurable drivers, ranks competing hypotheses, abstains when evidence is insufficient or contradictory, and produces persona-specific actions with traceable evidence.

The prototype is intentionally built as a modular Python application so that the analytical engine can run locally today and map cleanly to a production lakehouse architecture later.

## Table of contents

- Overview
- Key capabilities
- Architecture
- End-to-end flow
- Repository structure
- Requirements
- Installation
- Configuration
- Running the application
- Data model and semantic contracts
- Detection and investigation
- Evidence and explainability
- Decisioning, abstention and feedback
- Roles and entitlements
- LLM architecture
- Validation
- Troubleshooting
- Production evolution
- References

## Overview

TrendLens addresses a common analytics failure mode: dashboards can show that a KPI moved, but the analyst still has to manually determine why it moved, whether the evidence is reliable, whether multiple causes interact, and what action should follow.

TrendLens turns that workflow into a governed decision pipeline:

1. Detect material KPI movements continuously.
2. Reconcile source freshness and data quality before reasoning.
3. Decompose the movement into measurable components.
4. Retrieve structured and unstructured evidence.
5. Generate candidate causes from a KPI knowledge graph.
6. Rank causes using soft evidence rather than a single brittle rule.
7. Abstain when confidence or evidence quality is insufficient.
8. Recommend actions appropriate to the user's role and access scope.
9. Narrate the result with citations, number consistency and a deterministic fallback.
10. Record feedback, reliability changes, telemetry and audit information.

## Key capabilities

- Continuous KPI anomaly sweep using weekday-robust z-scores, robust STL residuals and two-sided CUSUM.
- Six-check anomaly gate covering persistence, ensemble agreement, materiality, freshness, business calendar and cross-KPI coherence.
- Level-1 GMV decomposition: `GMV = Sessions × Conversion Rate × AOV`, followed by price/mix shift-share within AOV.
- Generalized investigation for any governed KPI and any selected time window.
- Signature probes for coupon expiry, competitor promotion, marketing gaps, gateway issues, stockouts, demand shifts and product-launch volatility.
- Knowledge graph compiled directly from semantic KPI contracts.
- Vector-style news retrieval using TF-IDF + cosine similarity with a temporal prior.
- Structured external-evidence extraction with a cached LLM step and deterministic heuristic fallback.
- Soft-computing hypothesis ranking with reliability weighting and conflict handling.
- Abstention/ambiguity gate instead of forced answers.
- Persona-aware actions and narratives for CFO, regional manager, marketing executive and analyst.
- Row-level entitlement scoping before aggregation.
- Bounded source reliability updates from analyst feedback.
- Beta priors and isotonic calibration as a cold-start learning bridge.
- Run IDs, stage telemetry and audit snapshots.
- Fully deterministic template narration when no LLM is available.

## Architecture

TrendLens follows a modular, event-driven analytical architecture combining time-series analytics, KPI semantic contracts, knowledge graphs, evidence retrieval, reliability-aware hypothesis ranking, LLM-assisted narration, and human feedback.

### High-level architecture

![TrendLens exact implementation architecture](docs/trendlens_architecture_clean.png)
> **Architecture accuracy:** This diagram intentionally shows only the components implemented in the repository. The four data sources are `sales.db`, `web_analytics.parquet`, `marketing.xlsx`, and `news/*.txt`. The final user-facing output is the **LLM narrative**, with the implemented deterministic template fallback when an LLM is unavailable. The architecture does not claim Airflow, Databricks, Neo4j, Slack/email alerts, PDF exports, or other unimplemented services.


**Architecture flow:** Data Sources → Ingestion & Quality → Core Analytics → Knowledge & Context → Insight Generation → Delivery & Action, with Feedback, Learning & Governance operating across the complete lifecycle.

### Core architectural principles

- **Evidence before narration:** quantitative claims are produced from structured business data before any LLM is invoked.
- **Detect continuously:** anomaly detection operates across governed KPI history rather than relying on manually supplied windows.
- **Abstain when uncertain:** contradictory, stale, weak or non-explanatory evidence can stop the system from forcing a root cause.
- **Semantic contracts as the control plane:** KPI formulas, grain, drivers, thresholds, lineage and access rules are configuration-driven.
- **Security before aggregation:** entitlement scoping is applied before users can derive restricted information from aggregates.
- **Human-in-the-loop learning:** analyst feedback improves reliability and calibration while preserving bounded updates and auditability.
- **LLM as narrator, not decision-maker:** deterministic analytical logic remains the source of truth, with a deterministic fallback when an LLM is unavailable or fails validation.


![TrendLens exact implementation architecture](docs/trendlens_architecture_clean.png)

> **Implementation accuracy:** The architecture uses the four data sources that actually exist in this repository — `sales.db`, `web_analytics.parquet`, `marketing.xlsx`, and `news/*.txt`. The final user-facing result is the **LLM narrative** (or the implemented deterministic template fallback when an LLM is unavailable). No additional ingestion platforms, dashboards, APIs, databases, or delivery channels are implied by this diagram.


### Logical layers

**1. Data sources**

The prototype reads four source families:

- `sales.db` — orders, returns and inventory.
- `web_analytics.parquet` — sessions and funnel activity.
- `marketing.xlsx` — campaign metadata and spend.
- `news/` — external narrative evidence.

**2. Reconciliation and quality**

`engine/reconcile.py` calculates source watermarks, effective KPI freshness and lineage-aware freshness. It also repairs known sample-data quality issues such as duplicate campaign rows and invalid campaign date spans.

**3. Semantic layer**

The YAML contracts under `contracts/` define KPI formulas, grain, parents, sources, drivers, levers, thresholds, minimum history and access restrictions. This makes business semantics configuration-driven rather than buried in application code.

**4. Knowledge graph**

`engine/graph.py` compiles a lightweight graph from the contracts. It represents relationships such as KPI → derived_from → parent KPI, KPI → fed_by → source, KPI → influenced_by → driver, KPI → analog_of → comparable KPI, and Lever → owned_by → persona.

**5. Detection**

`engine/sweep.py` scans the full governed KPI history without being given target anomaly windows. Three detectors vote on candidate days: weekday median/MAD z-score, robust STL residual, and resetting two-sided CUSUM. At least two detectors must agree before the six-check gate evaluates the candidate.

**6. Investigation and evidence**

`engine/investigate.py` automatically selects a clean baseline, runs driver-specific signature probes, performs decomposition and builds an evidence docket. `engine/vector_news.py` retrieves relevant news and converts external snippets into structured evidence.

**7. Decision layer**

`engine/core.py` scopes evidence, ranks hypotheses, evaluates conflict and uncertainty, applies the ambiguity gate and assembles actions.

**8. Narrative layer**

`engine/nlg.py` constructs a constrained payload and optionally calls an LLM. The LLM is a renderer, not the decision-maker. A mechanical checker validates factual sentences/numbers and a deterministic template renderer provides the safe fallback.

**9. Feedback and governance**

`engine/pipeline.py` threads run IDs through the execution, records audit snapshots, measures stage latency and applies bounded source-reliability updates. `engine/calibrate.py` maintains Beta priors and isotonic calibration samples.

## End-to-end flow

```text
Raw sources
   ↓
Reconcile freshness + repair known quality issues
   ↓
Semantic KPI contracts + access rules + knowledge graph
   ↓
Daily KPI series
   ↓
3-detector anomaly ensemble
   ↓
Six-check gate
   ├── reject → named rejection reason
   └── confirm
         ↓
   Clean baseline + signature probes
         ↓
   Level-1 decomposition + channel/funnel/inventory checks
         ↓
   Vector news retrieval + structured evidence extraction
         ↓
   Graph-derived hypotheses
         ↓
   Reliability-weighted ranking + conflict handling
         ↓
   Ambiguity gate
      ├── abstain + clarification request
      └── explain + action recommendations
         ↓
   Persona-scoped narrative + citations
         ↓
   Telemetry + audit + analyst feedback
```

## Repository structure

```text
trendlens/
├── app.py
├── make_config.py
├── datagen.py
├── validate_sweep.py
├── personas.yaml
├── source_reliability.yaml
├── contracts/
│   ├── gmv.yaml
│   ├── orders.yaml
│   ├── orders_kpi.yaml
│   ├── sessions.yaml
│   ├── aov.yaml
│   ├── conversion_rate.yaml
│   ├── net_revenue.yaml
│   ├── marketing_spend.yaml
│   └── airpro_orders.yaml
├── engine/
│   ├── core.py
│   ├── pipeline.py
│   ├── graph.py
│   ├── reconcile.py
│   ├── sweep.py
│   ├── investigate.py
│   ├── analytics.py
│   ├── decompose.py
│   ├── retrieval.py
│   ├── vector_news.py
│   ├── calibrate.py
│   ├── nlg.py
│   ├── detection.py
│   ├── scenarios.py
│   └── labels.py
└── trend_lens_data/
    ├── sales.db
    ├── web_analytics.parquet
    ├── marketing.xlsx
    ├── ground_truth.yaml
    └── news/
```

## Requirements

Python 3.10+ is recommended.

```bash
pip install streamlit pandas pyyaml openpyxl pyarrow statsmodels numpy matplotlib
```

Optional LLM backends are Ollama for local/offline generation and a hosted API when an API key is available. The application remains usable without an LLM because deterministic narration is always available.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit pandas pyyaml openpyxl pyarrow statsmodels numpy matplotlib
streamlit run app.py
```

## Configuration

KPI contracts define formulas, parents, sources, drivers, thresholds, access restrictions, analogs and levers. `personas.yaml` controls narrative depth, vocabulary, number aggregation, action frame and channel. `source_reliability.yaml` supplies initial reliability priors.

## Running the application

Start with:

```bash
streamlit run app.py
```

The workspace contains eight tabs: Anomaly feed, Investigate KPI, Scenario 1 multi-factor, Scenario 2 abstention, Scenario 3 sparse KPI, All scenarios, Feedback loop, and Telemetry & audit.

## Detection and investigation

TrendLens continuously scans every governed KPI using weekday-robust z-score, robust STL and CUSUM. A candidate needs two detector votes and then passes the six-check gate: persistence, ensemble agreement, materiality, freshness, business calendar and cross-KPI coherence.

For investigation, the system selects a clean baseline, measures the KPI movement, decomposes it, runs driver-specific signature probes, retrieves relevant evidence, builds graph-derived hypotheses, ranks them and either explains the movement or abstains.

## Evidence and explainability

Each evidence item carries provenance such as source, freshness, retrieval/measurement method, direction, supporting hypothesis, strength and temporal relevance. Structured sources are authoritative for quantitative claims. External news supports or contests hypotheses but does not create KPI numbers.

## Decisioning, abstention and feedback

The ambiguity gate can abstain when confidence is low, evidence is stale, hypotheses are too close, or the measured decomposition is not explained by the leading cause. An abstention includes reasons and evidence that would resolve the uncertainty.

Analyst feedback updates source reliability within fixed bounds and is recorded with a run ID. Beta priors and isotonic calibration provide a transparent cold-start learning mechanism.

## Roles and entitlements

- **CFO:** enterprise scope, financial vocabulary, aggregated numbers, approval-oriented actions.
- **North manager:** North-region scope, operational vocabulary and regional execution.
- **Marketing executive:** channel/campaign focus with restricted financial fields.
- **Analyst intern:** restricted fields and grain, read-only summary.

Entitlements are applied before aggregation wherever possible.

## LLM architecture

```text
Measured evidence → constrained payload → optional LLM → checker
                                      ↘ deterministic fallback
```

The LLM renders; it does not decide. Numeric conclusions originate from measured data.

## Validation

Run:

```bash
python validate_sweep.py
```

The supplied validation compares the continuous sweep against planted ground truth and checks expected confirmations, named rejections and an unscripted discovery. The synthetic dataset intentionally includes data-quality flaws to exercise reconciliation.

## Troubleshooting

- Run Streamlit from the repository root.
- Use deterministic narration if an optional LLM is unavailable.
- Expect conversion-rate freshness to be limited by the lagging web source in the sample data.
- Use `python datagen/generate.py --orders 4000 --seed 7` for a smoother demo.

## Production evolution

| Prototype | Production mapping |
|---|---|
| SQLite / parquet / XLSX | governed Delta/warehouse tables |
| YAML KPI contracts | governed semantic/metric layer |
| Local graph | metadata/lineage graph |
| TF-IDF retrieval | managed vector search |
| Streamlit | governed application |
| JSONL audit logs | governed audit tables |
| Optional local/API LLM | managed model endpoint / AI gateway |

A production implementation should add centralized identity, secrets management, monitoring, CI/CD, data-quality SLAs, scalable distributed processing and model governance.

## Maintainers

TrendLens prototype — hackathon / innovation submission.
