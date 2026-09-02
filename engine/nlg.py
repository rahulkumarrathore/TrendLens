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
    # In abstain mode the persona's action_frame must not reach the LLM:
    # "action frame: approve" alongside "recommend NO action" is a contradictory
    # instruction, and the model tends to follow the action frame.
    prof = {k: v for k, v in persona.items()
            if not (decision["mode"] == "abstain" and k == "action_frame")}
    p = {"mode": decision["mode"], "persona": role, "persona_profile": prof,
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
        p["discounted_evidence"] = [{"id": e["id"], "note": f"conflicts with [{e['conflicts_with']}], reliability discounted"}
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
    if backend == "gemini":
        try:
            key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
            mdl = model or "gemini-2.5-pro"
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent?key={key}",
                data=json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                                 "generationConfig": {"maxOutputTokens": 800}}).encode(),
                headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            txt = "".join(p.get("text", "") for c in r.get("candidates", [])[:1]
                          for p in c.get("content", {}).get("parts", []))
            u = r.get("usageMetadata", {})
            return txt or None, {"backend": "gemini", "latency_s": round(time.time()-t0, 2),
                                 "tokens_in": u.get("promptTokenCount", 0),
                                 "tokens_out": u.get("candidatesTokenCount", 0)}
        except Exception as ex:
            return None, {"backend": "gemini", "error": str(ex)[:200]}


    if backend == "openai":
        try:
            key = os.environ.get("OPENAI_API_KEY", "")
            req = urllib.request.Request("https://api.openai.com/v1/responses",
                data=json.dumps({"model": model or "gpt-5.6-luna", "input": prompt,
                                 "max_output_tokens": 800,
                                 "reasoning": {"effort": "low"}}).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            txt = "".join(p.get("text", "") for item in r.get("output", [])
                          for p in item.get("content", []) if p.get("type") == "output_text")
            u = r.get("usage", {})
            return txt or None, {"backend": "openai", "latency_s": round(time.time()-t0, 2),
                                 "tokens_in": u.get("input_tokens", 0),
                                 "tokens_out": u.get("output_tokens", 0)}
        except Exception as ex:
            return None, {"backend": "openai", "error": str(ex)[:200]}
            
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
    pp = payload["persona_profile"]
    common = ("You write a short business insight for a reader who has never seen the "
              "underlying data structure. Never mention the payload, JSON, fields, persona, "
              "profile, or these instructions. Use ONLY facts given below; copy numbers "
              "verbatim. Cite a fact by placing the bare evidence id in square brackets at "
              "the end of the sentence, like this: [e-104]. Cite each id at most once per "
              "sentence. Never write field names, labels, or key-value pairs into the prose "
              "(no 'confidence: 0.9', no 'evidence id: e-104', no 'phrase: very likely') — "
              "state the value in plain words instead. "
              f"Length {pp['length']}. Vocabulary: {pp['vocabulary']}. ")
    if payload["mode"] == "abstain":
        rules = common + (
            "The engine ABSTAINED: the evidence does not yet determine the cause. "
            "You must NOT recommend, approve, endorse, or suggest any action, and you must "
            "not name a winning cause. Structure: (1) state the cause is not yet "
            "determinable and why; (2) list the competing hypotheses with their exact "
            "confidences; (3) list what would resolve them; (4) end by stating that no "
            "action is recommended until the ambiguity resolves.\n\nDATA:\n")
    else:
        rules = common + (
            f"Action frame: {pp['action_frame']}. "
            "The causes below are CONFIRMED findings of the engine. State them as findings; "
            "do not say the cause is unknown, unclear, or not determinable. Use the given "
            "confidence phrase for each cause and present the listed actions.\n\nDATA:\n")
    return rules + json.dumps(payload, indent=1, default=str)

# ---------------- mechanical checker ----------------
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:pp|%|L)?")
# a citation bracket: [e-101] or the grouped form [e-101, e-102, e-104]
CITE_RE = re.compile(r"\[((?:e-\d+)(?:\s*,\s*e-\d+)*)\]")
# a model-written reference section, which collides with render_citations
REFSEC_RE = re.compile(r"^\s*(references|sources|citations|notes)\s*:", re.I | re.M)


def cited_ids(text):
    """All evidence ids cited in text, in order, expanding grouped brackets."""
    out = []
    for grp in CITE_RE.findall(text or ""):
        out += [i.strip() for i in grp.split(",")]
    return out

# committed / action phrasing that must never appear in an abstain narrative
COMMIT_RE = re.compile(
    r"\b(the cause (is|was)|caused by|driven by|primarily due to|dominant cause"
    r"|should (approve|launch|increase|reduce|adjust|proceed|act|implement)"
    r"|we recommend|recommend(s|ed)? (approving|launching|increasing|reducing|adjusting)"
    r"|approve the|is recommended to)\b", re.I)
# hedging / abstain phrasing that must never appear in a commit narrative
HEDGE_RE = re.compile(
    r"\b(not (yet )?determinable|cannot (yet )?be determined|cause (is|remains) (unknown|unclear)"
    r"|insufficient evidence|no action is recommended)\b", re.I)
# prompt / schema scaffolding that must never leak into user-facing text
LEAK_RE = re.compile(r"\b(json|payload|persona|profile|schema|glossary|provided data)\b", re.I)
# payload field names echoed literally, e.g. "[confidence: 0.918]" or "[evidence id: e-104]"
FIELD_ECHO_RE = re.compile(
    r"\[\s*(evidence[ _]?id|confidence|phrase|mode|owns|caps|name|source|method|fresh|"
    r"impact[ _]?inr|magnitude|window|kpi)\s*[:=]", re.I)


def _check_numbers(text, allowed_nums, errors):
    for n in NUM_RE.findall(text):
        if len(n) > 2 and n not in allowed_nums:
            errors.append(f"number {n} not found in payload")


def check_narrative(text, payload):
    errors = []
    cited = set(cited_ids(text))
    valid = set(payload["evidence_glossary"].keys())
    for c in cited - valid:
        errors.append(f"citation {c} not in glossary")
    if LEAK_RE.search(text):
        errors.append("scaffolding term leaked into narrative (json/payload/persona/...)")
    if FIELD_ECHO_RE.search(text):
        errors.append("payload field name echoed literally into narrative")
    if REFSEC_RE.search(text):
        errors.append("narrative wrote its own reference section")
    for sent in re.split(r"(?<=[.!?])\s+", text):
        ids = cited_ids(sent)
        dupes = {i for i in ids if ids.count(i) > 1}
        for d in dupes:
            errors.append(f"citation {d} repeated within one sentence")
    if payload["mode"] != "abstain":
        if HEDGE_RE.search(text):
            errors.append("commit mode but abstain/hedging phrasing found")
        for c in payload.get("causes", []):
            if not (set(c["evidence"]) & cited):
                errors.append(f"cause {c['name']} narrated without citing its evidence")
        allowed_nums = " ".join([json.dumps(payload["decomposition"]),
                                 payload["event"]["magnitude"],
                                 json.dumps([str(c["confidence"]) for c in payload["causes"]]),
                                 json.dumps(payload.get("actions", []), default=str),
                                 json.dumps(payload["evidence_glossary"], default=str)])
        _check_numbers(text, allowed_nums, errors)
    else:
        if COMMIT_RE.search(text):
            errors.append("abstain mode but committed/action phrasing found")
        if not re.search(r"no action", text, re.I):
            errors.append("abstain narrative must state that no action is recommended")
        for h in payload.get("hypotheses", []):
            if not (set(h["evidence"]) & cited):
                errors.append(f"hypothesis {h['name']} narrated without citing its evidence")
        allowed_nums = " ".join([json.dumps(payload.get("hypotheses", []), default=str),
                                 json.dumps(payload.get("why", []), default=str),
                                 json.dumps(payload.get("resolves", []), default=str),
                                 json.dumps(payload["event"], default=str),
                                 payload["event"]["magnitude"],
                                 json.dumps(payload["evidence_glossary"], default=str)])
        _check_numbers(text, allowed_nums, errors)
    return errors

# ---------------- display layer: id -> footnote ----------------
def citation_map(text):
    """{evidence_id: footnote_number} for ids cited in text, in reading order.

    Shared by render_citations and by the evidence docket UI, so the numbers in
    the narrative and the ids listed in the docket always agree.
    """
    order = {}
    for eid in cited_ids(text):
        if eid not in order:
            order[eid] = len(order) + 1
    return order


def render_citations(text, glossary=None):
    """Map raw evidence ids to sequential footnote numbers for display.

    Returns (display_text, references). Raw e-ids are the internal contract used
    by check_narrative; this runs AFTER validation, purely for presentation.
    Handles [e-101] and the grouped form [e-101, e-102]. Never raises.
    """
    if not text:
        return "", []
    glossary = glossary or {}
    order, refs = citation_map(text), []
    for eid, n in order.items():
        g = glossary.get(eid) or {}
        refs.append({"n": n, "id": eid,
                     "claim": g.get("claim", f"({eid} — not in this run's docket)"),
                     "value": g.get("value", ""),
                     "source": g.get("source", "—"),
                     "fresh": g.get("fresh", "—"),
                     "method": g.get("method", "—")})

    def _sub(m):
        ids = [i.strip() for i in m.group(1).split(",")]
        return "".join(f"[{order[i]}]" for i in ids if i in order) or m.group(0)

    return CITE_RE.sub(_sub, text), refs


# ---------------- deterministic fallback renderer ----------------
RANK_WORD = ["Primary", "Secondary", "Tertiary"]


def _headline(ev):
    return (f"#### {nice(ev['kpi'])} {ev['magnitude']} · {ev['window']} · "
            f"Rs {ev['impact_inr']/1e5:.2f}L impact\n\n"
            f"*{nice(ev['type'])}*")


def _decomp_table(d):
    if not d:
        return None
    rows = "\n".join(f"| {(nice(k) or k).capitalize()} | {v} |" for k, v in d.items())
    return "**Decomposition**\n\n| Component | Contribution |\n|---|---:|\n" + rows


def _cites(ids):
    return " ".join(f"[{e}]" for e in ids)


def template_render(payload):
    """Deterministic renderer. Emits structured markdown with blank-line separated
    blocks so it renders as sections rather than one collapsed paragraph.
    Evidence ids stay raw here; render_citations() converts them for display."""
    ev = payload["event"]
    blocks = [_headline(ev)]

    if payload["mode"] == "abstain":
        why = payload.get("why", [])
        blocks.append("**No cause committed — the evidence does not separate the candidates.**"
                      + ("\n\n" + "\n".join(f"- {w}" for w in why) if why else ""))

        rows = []
        for i, h in enumerate(payload.get("hypotheses", []), 1):
            caps = f" · {', '.join(h['caps'])}" if h["caps"] else ""
            rows.append(f"| {i} | {nice(h['name'])} | {h['confidence']:.2f}{caps} | "
                        f"{_cites(h['evidence'])} |")
        if rows:
            blocks.append("**Competing hypotheses**\n\n"
                          "| # | Hypothesis | Confidence | Evidence |\n|---|---|---:|---|\n"
                          + "\n".join(rows))

        res = payload.get("resolves", [])
        if res:
            blocks.append("**What would resolve this**\n\n"
                          + "\n".join(f"- {r}" for r in res))

        blocks.append("**No action recommended** under this uncertainty. "
                      "Re-evaluation runs automatically when the missing data arrives.")
        return "\n\n".join(blocks)

    dt = _decomp_table(payload["decomposition"])
    if dt:
        blocks.append(dt)

    cause_lines = []
    for i, c in enumerate(payload["causes"]):
        rank = RANK_WORD[i] if i < len(RANK_WORD) else f"Cause {i+1}"
        head = (f"**{rank} — {nice(c['name'])}** · {c['phrase']} · "
                f"confidence {c['confidence']:.2f}")
        detail = [f"Acting on {' + '.join(nice(o) for o in c['owns'])}."]
        if c.get("causal"):
            detail.append(f"Causal check: {c['causal']['test']}, p={c['causal']['p']}.")
        cause_lines.append(f"{head}  \n{' '.join(detail)} {_cites(c['evidence'])}")
    blocks.append("**What caused it**\n\n" + "\n\n".join(cause_lines))

    disc = payload.get("discounted_evidence", [])
    if disc:
        blocks.append("**Discounted evidence**\n\n"
                      + "\n".join(f"- [{de['id']}] — {de['note']}." for de in disc))

    frame = payload["persona_profile"].get("action_frame", "none")
    acts = payload.get("actions", [])
    if frame != "none" and acts:
        heading = "Decision requested" if frame == "approve" else "Your tasks"
        items = []
        for a in acts:
            lead = f"**{nice(a['driver'])}.** {a['action']}"
            if a.get("rationale"):
                lead += f" {a['rationale']}"
            meta = []
            if a.get("expected_impact"):
                meta.append(a["expected_impact"].rstrip("."))
            if a["lever"]:
                meta.append(f"owner {nice(a['owner'])}")
            meta.append(f"confidence {a['confidence']:.2f}")
            items.append(f"{lead}  \n*{' · '.join(meta)}*  \n"
                         f"Monitoring: {a['monitoring_plan']}")
        blocks.append(f"**{heading}**\n\n" + "\n\n".join(items))

    return "\n\n".join(blocks)


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
