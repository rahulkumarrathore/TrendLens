"""Stage 3 — Decomposition.

Level-1 identity split (GMV = Sessions x CR x AOV, AOV -> price/mix shift-share)
is implemented in engine.analytics.decompose and computed live over the dataset.
"""
from .analytics import decompose, channel_shift, window_stats  # noqa: F401
from .retrieval import S1_EVENT, S2_EVENT, D1, D2  # noqa: F401

L1 = {S1_EVENT["id"]: S1_EVENT["decomposition"], S2_EVENT["id"]: S2_EVENT["decomposition"]}
