"""Pipeline orchestrator: scope -> rank -> gate -> actions -> narrate, with telemetry,
feedback (bounded, event-logged), and run_id threading."""
import time, uuid, json, os, copy
from . import core, nlg
from .scenarios import (S1_EVENT, S1_DOCKET, S1_HYPOTHESES,
                        S2_EVENT, S2_DOCKET, S2_HYPOTHESES, S3_INFO, REJECTION_LOG)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_LOG = os.path.join(ROOT, "feedback_log.jsonl")
AUDIT_LOG = os.path.join(ROOT, "audit_log.jsonl")

# mutable reliability store (feedback adjusts, bounded)
RELIABILITY = dict(core.RELIABILITY_BASE)

def reset_reliability():
    RELIABILITY.clear(); RELIABILITY.update(core.RELIABILITY_BASE)

def apply_feedback(run_id, hypothesis, verdict, docket):
    """confirm|reject -> bounded source-reliability update via evidence IDs + Beta prior note."""
    event = {"run_id": run_id, "hypothesis": hypothesis, "verdict": verdict, "ts": time.time()}
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")
    delta = -0.05 if verdict == "reject" else 0.05
    touched = {}
    seen = set()
    for e in docket:
        if e.get("supports") == hypothesis and e["direction"] == "confirming":
            if e["source"] in seen:
                continue
            seen.add(e["source"])
            old = RELIABILITY.get(e["source"], 0.5)
            new = round(min(max(old + delta, 0.10), 0.99), 2)
            RELIABILITY[e["source"]] = new
            touched[e["source"]] = (old, new)
    return touched

def run_scenario(event, docket, hypotheses, role, backend="none", model=None):
    run_id = str(uuid.uuid4())[:8]
    timers, t0 = {}, time.time()

    t = time.time()
    ev = copy.deepcopy(event)
    if role == "north_manager" and "north_manager_overrides" in ev:
        ev.update(ev.pop("north_manager_overrides"))
    scoped = core.scope_docket(docket, role)
    timers["security_reconcile"] = round(time.time() - t, 4)

    t = time.time()
    hyps = copy.deepcopy(hypotheses)
    scoped_ids = {e["id"] for e in scoped}
    for h in hyps:
        h["evidence"] = [i for i in h["evidence"] if i in scoped_ids]
    finding = core.rank(hyps, scoped, RELIABILITY)
    timers["ranking"] = round(time.time() - t, 4)

    t = time.time()
    decision = core.ambiguity_gate(finding)
    decision["ranked"] = finding["hypotheses"]
    actions = core.assemble_actions(decision.get("causes", []), ev) if decision["mode"] != "abstain" else []
    timers["decision"] = round(time.time() - t, 4)

    t = time.time()
    persona = core.PERSONAS[role]
    payload = nlg.build_payload(ev, decision, actions, scoped, persona, role)
    narrative, meta = nlg.narrate(payload, backend, model)
    timers["nlg"] = round(time.time() - t, 4)
    timers["total"] = round(time.time() - t0, 4)

    llm_calls = 0 if meta.get("backend") == "none" else (2 if meta.get("narrative_source") == "llm_retry" else 1)
    telemetry = {"run_id": run_id, "role": role, "timers_s": timers,
                 "llm": {**meta, "calls_this_run": llm_calls,
                         "note": "call #1 (query-writing) not exercised in sample mode"},
                 "marginal_cost": "Rs 0 (local/template)" if meta.get("backend") != "api" else "API-billed",
                 "counterfactual_api_cost": "~Rs 0.4/insight at hosted-API rates"}
    snapshot = {"run_id": run_id, "user_scope": core.audit_scope(role), "event": ev["id"],
                "mode": decision["mode"], "kappa": finding["kappa"],
                "scores": {h["name"]: h["confidence"] for h in finding["hypotheses"]},
                "narrative_source": meta.get("narrative_source"), "ts": time.time()}
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    return {"run_id": run_id, "event": ev, "docket": scoped, "finding": finding,
            "decision": decision, "actions": actions, "payload": payload,
            "narrative": narrative, "telemetry": telemetry}

def run_all(role, backend="none", model=None):
    return {"S1": run_scenario(S1_EVENT, S1_DOCKET, S1_HYPOTHESES, role, backend, model),
            "S2": run_scenario(S2_EVENT, S2_DOCKET, S2_HYPOTHESES, role, backend, model),
            "S3": S3_INFO, "rejections": REJECTION_LOG}
