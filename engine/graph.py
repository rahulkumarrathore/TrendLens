"""Stage 0.4 — Knowledge graph, compiled from contracts at import (LIVE).
Lightweight adjacency (no networkx dependency). Edges: derived_from, fed_by,
influenced_by, analog_of, owned_by."""
from .core import CONTRACTS

EDGES = []          # (src, edge_type, dst)
for kpi, c in CONTRACTS.items():
    for p in c.get("parents", []):
        EDGES.append((kpi, "derived_from", p))
    for s in c.get("sources", []):
        EDGES.append((kpi, "fed_by", s))
    for d in c.get("drivers", []):
        EDGES.append((kpi, "influenced_by", d))
    for a in c.get("analogs", []):
        EDGES.append((kpi, "analog_of", a))
    for lever, owner in (c.get("levers") or {}).items():
        EDGES.append((lever, "owned_by", owner))

def ancestors(kpi, kinds=("derived_from", "fed_by")):
    """All upstream nodes reachable via the given edge types (lineage walk)."""
    out, stack = set(), [kpi]
    while stack:
        n = stack.pop()
        for s, t, d in EDGES:
            if s == n and t in kinds and d not in out:
                out.add(d); stack.append(d)
    return out

def candidate_drivers(kpi):
    """Hypothesis candidates = influenced_by edges of the KPI and its parents."""
    nodes = {kpi} | ancestors(kpi, kinds=("derived_from",))
    return sorted({d for s, t, d in EDGES if t == "influenced_by" and s in nodes})

def analogs(kpi):
    return [d for s, t, d in EDGES if s == kpi and t == "analog_of"]

def owner_of(lever):
    for s, t, d in EDGES:
        if s == lever and t == "owned_by":
            return d
    return "system"
