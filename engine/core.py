"""TrendLens engine core: entitlements, fuzzy ranking, DS conflict, decision layer."""
import yaml, os, json, itertools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_yaml(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return yaml.safe_load(f)

PERSONAS = load_yaml("personas.yaml")
RELIABILITY_BASE = load_yaml("source_reliability.yaml")
CONTRACTS = {f[:-5]: load_yaml(f"contracts/{f}") for f in os.listdir(os.path.join(ROOT, "contracts"))}

TAU, DELTA, KAPPA_MAX = 0.55, 0.15, 0.50
WEIGHTS = {"evidence_strength": 0.30, "temporal_alignment": 0.15,
           "source_reliability": 0.20, "contribution_match": 0.20,
           "historical_precedent": 0.15}

# ---------------- security ----------------
def scope_docket(docket, role):
    """Entitlement filter on evidence: tier + region scope. Runs BEFORE ranking."""
    access = CONTRACTS["gmv"]["access"][role]
    out = []
    for e in docket:
        if e.get("tier", "all") != "all" and e["tier"] != role:
            continue
        if access.get("rows", "all") != "all" and e.get("region_scope") == "cross_region":
            continue
        out.append(e)
    return out

def audit_scope(role):
    a = CONTRACTS["gmv"]["access"][role]
    return {"role": role, "rows": a.get("rows", "all"),
            "columns_dropped": a.get("columns_exclude", []),
            "domains_refused": a.get("domains_exclude", [])}

# ---------------- fuzzy ranking ----------------
def score_hypothesis(h, docket, reliability):
    ev = [e for e in docket if e["id"] in h["evidence"]]
    if not ev:
        crit = {k: 0.0 for k in WEIGHTS}
    else:
        confirming = [e for e in ev if e["direction"] == "confirming"]
        refuting = [e for e in ev if e["direction"] == "refuting"]
        n_req = h.get("n_requirements", max(len(h["evidence"]), 1))
        crit = {
            "evidence_strength": round(min(len(confirming) / n_req, 1.0) * (0.5 if refuting else 1.0), 2),
            "temporal_alignment": round(sum(e.get("temporal", 0.8) for e in ev) / len(ev), 2),
            "source_reliability": round(max((reliability.get(e["source"], 0.5) for e in confirming), default=0.2), 2),
            "contribution_match": h.get("contribution_match", 0.5),
            "historical_precedent": round(h["prior_alpha"] / (h["prior_alpha"] + h["prior_beta"]), 2),
        }
    raw = round(sum(WEIGHTS[k] * v for k, v in crit.items()), 3)
    caps = []
    if any(e["direction"] == "refuting" for e in ev):
        raw, caps = min(raw, 0.25), caps + ["refutation_cap"]
    if h.get("critical_evidence_missing"):
        raw, caps = min(raw, 0.30), caps + ["freshness_cap:untestable"]
    if h.get("causal"):
        mult = 1.15 if h["causal"]["p"] < 0.05 else 0.85
        raw = round(min(raw * mult, 0.99), 3)
        h["causal"]["mult"] = mult
    if h.get("sparse_penalty"):
        raw = round(raw * h["sparse_penalty"], 3); caps.append(f"sparse x{h['sparse_penalty']}")
    return raw, crit, caps

def ds_kappa(docket, reliability):
    """Dempster–Shafer conflict: pairwise mass conflict between evidence items
    supporting different hypotheses (theta mass = 1 - strength*reliability)."""
    masses = []
    for e in docket:
        if e["direction"] != "confirming" or not e.get("supports"):
            continue
        m = min(e.get("strength", 0.7) * reliability.get(e["source"], 0.5), 0.95)
        masses.append((frozenset([e["supports"]]), m))
    if len(masses) < 2:
        return 0.0
    kappa_total, pairs = 0.0, 0
    for (s1, m1), (s2, m2) in itertools.combinations(masses, 2):
        if s1 != s2:  # different singletons -> full conflict on m1*m2
            kappa_total += m1 * m2
        pairs += 1
    explicit = sum(1 for e in docket if e.get("conflicts_with"))
    return round(min(kappa_total / max(pairs, 1) + 0.12 * explicit, 1.0), 2)

def rank(hypotheses, docket, reliability):
    ranked = []
    for h in hypotheses:
        conf, crit, caps = score_hypothesis(h, docket, reliability)
        ranked.append({**h, "confidence": conf, "criteria": crit, "caps": caps})
    ranked.sort(key=lambda x: -x["confidence"])
    kappa = ds_kappa(docket, reliability)
    gap = round(ranked[0]["confidence"] - ranked[1]["confidence"], 3) if len(ranked) > 1 else 1.0
    return {"hypotheses": ranked, "kappa": kappa, "gap_top2": gap}

# ---------------- decision layer ----------------
def ambiguity_gate(finding):
    hs = finding["hypotheses"]
    top, second = hs[0], (hs[1] if len(hs) > 1 else None)
    reasons = []
    if any(c.startswith("freshness_cap") for c in top["caps"]):
        return {"mode": "abstain", "why": ["top hypothesis untestable (stale evidence)"]}
    if top["confidence"] < TAU:
        reasons.append(f"top confidence {top['confidence']} < tau {TAU}")
    dual = False
    if second and finding["gap_top2"] < DELTA:
        overlap = set(top.get("owns", [])) & set(second.get("owns", []))
        if not overlap and second["confidence"] >= TAU:
            dual = True  # disjoint components -> co-causes, waive gap
        else:
            reasons.append(f"gap {finding['gap_top2']} < delta {DELTA}")
    if finding["kappa"] > KAPPA_MAX:
        reasons.append(f"conflict kappa {finding['kappa']} > {KAPPA_MAX}")
    if reasons:
        return {"mode": "abstain", "why": reasons}
    proceed = [top] + ([second] if (dual or (second and second["confidence"] >= TAU
               and not (set(top.get("owns", [])) & set(second.get("owns", []))))) else [])
    return {"mode": "dual_cause" if len(proceed) == 2 else "single_cause", "causes": proceed}

# ---------------- actions ----------------
# Each template is a complete imperative sentence so it reads in flow rather
# than as fragments joined by dashes at render time.
ACTION_TEMPLATES = {
    "promotional_pricing": "Launch a replacement coupon at comparable depth (~{depth}).",
    "campaign_spend": "Rebalance campaign spend across the affected channels.",
}
NO_LEVER_ACTION = ("No corrective action is available: this cause sits outside "
                   "the levers you control.")


def _join(items):
    """Join a list into readable prose: 'a', 'a and b', 'a, b and c'."""
    items = [str(i) for i in items if i]
    if not items:
        return "the affected metric"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _monitor_sentence(note):
    """Normalise a monitor note into one capitalised sentence."""
    note = (note or "Recheck in three days").strip().rstrip(".")
    return note[0].upper() + note[1:] + "."


def assemble_actions(causes, event):
    """Build action records. Every text field is a standalone sentence; the
    renderer never has to splice fragments together, and no sentence is
    duplicated across fields."""
    acts = []
    levers = CONTRACTS[event["kpi"]].get("levers", {})
    for c in causes:
        lever = c.get("lever")
        owns = _join(c.get("owns", []))
        if lever and lever in levers:
            impact = c.get("recovery_inr_per_day", 0)
            acts.append({
                "driver": c["name"],
                "lever": lever,
                "action": ACTION_TEMPLATES[lever].format(depth=c.get("depth", "15%")),
                "rationale": f"This cause acts on {owns}, which the lever directly moves.",
                "expected_impact": f"Recovers about Rs {impact/1e5:.2f}L per day.",
                "owner": levers[lever],
                "confidence": c["confidence"],
                "monitoring_plan": _monitor_sentence(
                    f"Watch {owns} daily and escalate if less than 70% recovers within five days"),
            })
        else:
            acts.append({
                "driver": c["name"],
                "lever": None,
                "action": NO_LEVER_ACTION,
                "rationale": f"It acts on {owns}, but no lever in this contract addresses it.",
                "expected_impact": None,
                "owner": "system",
                "confidence": c["confidence"],
                "monitoring_plan": _monitor_sentence(c.get("monitor_note")),
            })
    return acts
