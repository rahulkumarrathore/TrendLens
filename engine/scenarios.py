"""Back-compat shim — stages now live in per-stage modules:
analytics.py (computed KPI math, LIVE) | graph.py (0.4, LIVE) | reconcile.py (1.2, LIVE)
detection.py (2) | decompose.py (3) | retrieval.py (4, values computed from the dataset)."""
from .analytics import ORDERS, INV, WEB, MKT_RAW as MKT, NEWS, snippet  # noqa: F401
from .retrieval import (S1_DOCKET, S1_HYPOTHESES, S1_EVENT,
    S2_DOCKET, S2_HYPOTHESES, S2_EVENT, S3_INFO, REJECTION_LOG)  # noqa: F401
