"""Stage 2 — Detection (LIVE).

The continuous sweep in engine.sweep runs the 3-detector ensemble
(weekday-z / STL residual / CUSUM) plus the six-check gate over every governed
KPI's full daily history. Confirmed events and the full rejection log are
computed from the dataset at import.

The curated S1/S2 events from retrieval.py remain available as the deeply
instrumented walkthrough scenarios; SWEEP_EVENTS is what the engine itself
found with no target windows given.
"""
from .sweep import CONFIRMED as SWEEP_EVENTS, REJECTIONS as SWEEP_REJECTIONS, DAY_FLAGS  # noqa: F401
from .retrieval import S1_EVENT, S2_EVENT, S3_INFO, REJECTION_LOG  # noqa: F401

EVENTS = [S1_EVENT, S2_EVENT]
