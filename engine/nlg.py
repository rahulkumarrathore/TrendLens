"""NLG: payload assembly, swappable LLM (ollama / api / none), mechanical checker,
deterministic template fallback. The LLM renders; it never decides."""
import os, re, json, time, urllib.request
from .labels import nice

CONF_BAND = [(0.85, "driven by"), (0.70, "very likely"), (0.55, "likely")]

def band_phrase(conf):
    for lo, p in CONF_BAND:
        if conf >= lo:
            return p
    return "possibly"

# ---------------- payload ----------------
def build_payload(event, decision, actions, docket, persona, role):
    glossary = {e["id"]: {"claim": e["claim"], "value": str(e["value"])[:200],
                          "source": e["source"], "fresh": e["fresh"], "method": e["method"]}
                for e in docket}
    p = {"mode": decision["mode"], "persona": role, "persona_profile": persona,
         "event": {k: event[k] for k in ("id", "kpi", "window", "magnitude", "impact_inr", "type")},
         "decomposition": event.get("decomposition", {}),
         "evidence_glossary": glossary,
         "constraints": {"cite_every_factual_sentence": True, "numbers_verbatim_only": True}}
    if decision["mode"] == "abstain":
        p["why"] = decision["why"]
        p["hypotheses"] = [{"name": h["name"], "confidence": h["confidence"],
                            "caps": h["caps"], "evidence": h["evidence"]}
                           for h in decision["ranked"][:3]]
        p["resolves"] = event.get("resolves", [])
    else:
        p["causes"] = [{"name": c["name"], "confidence": c["confidence"], "owns": c["owns"],
                        "evidence": c["evidence"], "phrase": band_phrase(c["confidence"]),
                        "causal": c.get("causal")} for c in decision["causes"]]
        p["discounted_evidence"] = [{"id": e["id"], "note": f"conflicts with {e['conflicts_with']}, reliability discounted"}
                                    for e in docket if e.get("conflicts_with")]
        p["actions"] = actions
    return p

# ---------------- llm backends ----------------
def llm_generate(prompt, backend="none", model=None):
    """Returns (text, meta). backend: none | ollama | api"""
    t0 = time.time()
    if backend == "ollama":
        try:
            req = urllib.request.Request("http://localhost:11434/api/generate",
                data=json.dumps({"model": model or "llama3.3:8b", "prompt": prompt,
                                 "stream": False}).encode(), headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return r["response"], {"backend": "ollama", "latency_s": round(time.time()-t0, 2),
                                   "tokens_in": r.get("prompt_eval_count", 0),
                                   "tokens_out": r.get("eval_count", 0)}
        except Exception as ex:
            return None, {"backend": "ollama", "error": str(ex)[:200]}
    if backend == "api":
        try:
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            req = urllib.request.Request("https://api.anthropic.com/v1/messages",
                data=json.dumps({"model": model or "claude-haiku-4-5-20251001", "max_tokens": 600,
                                 "messages": [{"role": "user", "content": prompt}]}).encode(),
                headers={"Content-Type": "application/json", "x-api-key": key,
                         "anthropic-version": "2023-06-01"})
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            txt = "".join(b.get("text", "") for b in r.get("content", []))
            u = r.get("usage", {})
            return txt, {"backend": "api", "latency_s": round(time.time()-t0, 2),
                         "tokens_in": u.get("input_tokens", 0), "tokens_out": u.get("output_tokens", 0)}
        except Exception as ex:
            return None, {"backend": "api", "error": str(ex)[:200]}
    return None, {"backend": "none", "latency_s": 0, "tokens_in": 0, "tokens_out": 0}

def make_prompt(payload):
    rules = ("You render a business insight. RULES: use ONLY facts from the JSON payload; "
             "cite every factual sentence with its evidence id in [brackets]; copy numbers verbatim; "
             "use the given confidence phrase for each cause; "
             f"length {payload['persona_profile']['length']}; "
             f"vocabulary: {payload['persona_profile']['vocabulary']}; "
             f"action frame: {payload['persona_profile']['action_frame']}. "
             "If mode is abstain: state cause is not yet determinable, list hypotheses with confidence, "
             "list what resolves them, recommend NO action.\n\nPAYLOAD:\n")
    return rules + json.dumps(payload, indent=1, default=str)

# ---------------- mechanical checker ----------------
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:pp|%|L)?")

def check_narrative(text, payload):
    errors = []
    cited = set(re.findall(r"\[(e-\d+)\]", text))
    valid = set(payload["evidence_glossary"].keys())
    for c in cited - valid:
        errors.append(f"citation {c} not in glossary")
    if payload["mode"] != "abstain":
        for c in payload.get("causes", []):
            if not (set(c["evidence"]) & cited):
                errors.append(f"cause {c['name']} narrated without citing its evidence")
        allowed_nums = " ".join([json.dumps(payload["decomposition"]),
                                 payload["event"]["magnitude"],
                                 json.dumps([str(c["confidence"]) for c in payload["causes"]]),
                                 json.dumps(payload.get("actions", []), default=str),
                                 json.dumps(payload["evidence_glossary"], default=str)])
        for n in NUM_RE.findall(text):
            if len(n) > 2 and n not in allowed_nums:
                errors.append(f"number {n} not found in payload")
    else:
        if re.search(r"the cause (is|was)\b", text, re.I):
            errors.append("abstain mode but committed phrasing found")
        if "action" in payload and payload.get("actions"):
            errors.append("abstain mode must not carry actions")
    return errors

# ---------------- deterministic fallback renderer ----------------
def template_render(payload):
    ev = payload["event"]
    head = f"**{nice(ev['kpi'])} {ev['magnitude']} ({ev['window']}) — Rs {ev['impact_inr']/1e5:.2f}L impact; {nice(ev['type']).lower()}.**"
    if payload["mode"] == "abstain":
        lines = [head, "Cause not yet determinable (below evidence threshold). Competing hypotheses:"]
        for h in payload["hypotheses"]:
            caps = f" [{', '.join(h['caps'])}]" if h["caps"] else ""
            lines.append(f"- {nice(h['name'])} ({h['confidence']:.2f}){caps} " +
                         " ".join(f"[{e}]" for e in h["evidence"]))
        lines.append("To resolve: " + "; ".join(payload.get("resolves", [])) + ".")
        lines.append("No action recommended under this uncertainty. Auto re-evaluation on data arrival.")
        return "\n".join(lines)
    lines = [head]
    d = payload["decomposition"]
    lines.append("Decomposition: " + ", ".join(f"{k} {v}" for k, v in d.items()) + ".")
    for c in payload["causes"]:
        causal = f"; causal check {c['causal']['test']} p={c['causal']['p']}" if c.get("causal") else ""
        lines.append(f"{c['phrase'].capitalize()} **{nice(c['name'])}** (confidence {c['confidence']:.2f}) "
                     f"acting on {'+'.join(c['owns'])}{causal} " +
                     " ".join(f"[{e}]" for e in c["evidence"]))
    for de in payload.get("discounted_evidence", []):
        lines.append(f"Discounted: [{de['id']}] — {de['note']}.")
    frame = payload["persona_profile"]["action_frame"]
    if frame != "none":
        verb = "Decision requested" if frame == "approve" else "Your tasks"
        for a in payload.get("actions", []):
            if a["lever"] is None:
                lines.append(f"No counter-action for **{nice(a['driver'])}**: {a['action']}. "
                             f"Monitoring: {a['monitoring_plan']}.")
            else:
                lines.append(f"{verb}: {a['action']} — {a['expected_impact']}, owner {nice(a['owner'])} "
                             f"(conf {a['confidence']:.2f}). Monitoring: {a['monitoring_plan']}.")
    return "\n".join(lines)

def narrate(payload, backend="none", model=None):
    prompt = make_prompt(payload)
    text, meta = llm_generate(prompt, backend, model)
    source = "template"
    if text:
        errs = check_narrative(text, payload)
        if not errs:
            source = "llm"
        else:
            text2, meta2 = llm_generate(prompt + "\n\nPrevious draft failed checks: " +
                                        "; ".join(errs) + ". Regenerate correctly.", backend, model)
            if text2 and not check_narrative(text2, payload):
                text, meta, source = text2, meta2, "llm_retry"
            else:
                text = None
    if not text:
        text = template_render(payload)
    meta["narrative_source"] = source
    return text, meta
