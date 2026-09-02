"""Stage 4 — Unstructured retrieval (LIVE): vector search over the news corpus
+ LLM call #1 (structured evidence extraction) with disk cache and a
deterministic heuristic fallback.

Design rules preserved:
  * External evidence may SUPPORT or CONTEST hypotheses — it never originates a
    quantitative claim. Extraction strips numbers from claims before scoring.
  * Extraction results are cached to news_extract_cache.json so demo runs need
    zero live LLM calls after the first pass.

Vector method: TF-IDF + cosine, implemented directly (a corpus of ~8 documents
does not justify an embedding service or a vector DB — the pitch states the
production swap is one function). A temporal prior boosts snippets dated inside
or just before the anomaly window.
"""
import os, re, json, math
import pandas as pd
from . import analytics as A
from . import nlg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(ROOT, "news_extract_cache.json")

_TOK = re.compile(r"[a-z0-9]+")

def _tokens(text):
    return _TOK.findall(text.lower())

# ---------------- corpus + index (built once at import) ----------------
def _parse(fname, text):
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip().lower().replace(" ", "_")] = v.strip()
    date = fields.get("date") or fname[:10]
    return {"file": fname, "date": date, "title": fields.get("title", fname),
            "source": fields.get("source", "news"), "text": text}

CORPUS = [_parse(f, t) for f, t in A.NEWS.items()]

def _build_index(corpus):
    docs = [_tokens(d["title"] + " " + d["text"]) for d in corpus]
    df = {}
    for toks in docs:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n = max(len(docs), 1)
    idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
    vecs = []
    for toks in docs:
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        v = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})
    return vecs, idf

VECS, IDF = _build_index(CORPUS)

def _qvec(query):
    tf = {}
    for t in _tokens(query):
        tf[t] = tf.get(t, 0) + 1
    v = {t: (1 + math.log(c)) * IDF.get(t, 1.0) for t, c in tf.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / norm for t, x in v.items()}

def search(query, window=None, k=3, floor=0.08):
    """Cosine top-k with a temporal prior: snippets dated within
    [window_start - 10d, window_end + 3d] get x1.3."""
    qv = _qvec(query)
    scored = []
    for d, dv in zip(CORPUS, VECS):
        sim = sum(qv[t] * dv.get(t, 0.0) for t in qv)
        if window:
            try:
                nd = pd.Timestamp(d["date"])
                a, b = pd.Timestamp(window[0]), pd.Timestamp(window[1])
                if a - pd.Timedelta(days=10) <= nd <= b + pd.Timedelta(days=3):
                    sim *= 1.3
            except Exception:
                pass
        scored.append((round(sim, 4), d))
    scored.sort(key=lambda x: -x[0])
    return [(s, d) for s, d in scored[:k] if s >= floor]

# ---------------- LLM call #1: structured extraction ----------------
_HEUR = [  # keyword -> (driver, boom_flag) fallback when no LLM backend
    (("flash sale", "rival", "competitor", "discount war"), "competitor_promo", False),
    (("upi", "gateway", "payment", "checkout", "outage", "degradation"), "gateway_issue", False),
    (("courier", "logistics", "delay", "strike", "supply", "stock"), "supply_stockout", False),
    (("festive", "rakhi", "raksha", "eoss", "gifting", "season sale", "seasonal demand"), "seasonal", True),
    (("boom", "record demand", "surge", "soar", "all-time high", "record high",
      "strong demand", "demand spike"), "demand_shift", True),
]

def _heuristic_extract(doc, drivers, total_pct):
    text = (doc["title"] + " " + doc["text"]).lower()
    for keys, drv, boom in _HEUR:
        if any(kw in text for kw in keys):
            supports = drv if drv in drivers or drv == "seasonal" else None
            conflicts = bool(boom and total_pct < 0)  # "record demand" vs measured softness
            return {"claim": doc["title"], "direction": "confirming",
                    "supports": supports, "temporal": 0.8,
                    "conflicts_with_internal": conflicts, "method": "heuristic keyword map"}
    return {"claim": doc["title"], "direction": "neutral", "supports": None,
            "temporal": 0.5, "conflicts_with_internal": False, "method": "heuristic keyword map"}

_EXTRACT_PROMPT = """You extract structured evidence from ONE news snippet for a KPI
anomaly investigation. Candidate drivers: {drivers}. The KPI ({kpi}) moved {mag} in
{window}. Respond with ONLY a JSON object, no prose:
{{"claim": "<one-sentence claim, NO numbers>",
 "direction": "confirming|refuting|neutral",
 "supports": "<one candidate driver or null>",
 "temporal": <0..1 how well the snippet's dates align with the window>,
 "conflicts_with_internal": <true if the snippet's story contradicts the measured direction>}}

SNIPPET:
{snippet}"""

def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}

def _save_cache(c):
    with open(CACHE_PATH, "w") as f:
        json.dump(c, f, indent=1)

def extract(doc, kpi, window, magnitude_pct, drivers, backend="none", model=None):
    """LLM call #1 with caching; heuristic fallback keeps the pipeline deterministic."""
    key = f"{doc['file']}|{kpi}|{window[0]}|{window[1]}"
    cache = _load_cache()
    if key in cache:
        return {**cache[key], "cached": True}
    out, meta = None, {"backend": backend}
    if backend != "none":
        prompt = _EXTRACT_PROMPT.format(drivers=", ".join(drivers), kpi=kpi,
                                        mag=f"{magnitude_pct:+.1f}%",
                                        window=f"{window[0]}..{window[1]}",
                                        snippet=doc["text"][:800])
        text, meta = nlg.llm_generate(prompt, backend, model)
        if text:
            try:
                j = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                j["claim"] = re.sub(r"[-+]?\d[\d,.]*%?", "", str(j.get("claim", "")))[:160].strip()
                if j.get("supports") not in drivers:
                    j["supports"] = None
                j["temporal"] = float(min(max(j.get("temporal", 0.7), 0), 1))
                j["method"] = f"LLM call #1 ({meta.get('backend')})"
                out = j
            except Exception:
                out = None
    if out is None:
        out = _heuristic_extract(doc, drivers, magnitude_pct)
    out["llm_meta"] = {k: meta.get(k) for k in ("backend", "latency_s", "tokens_in", "tokens_out")}
    cache[key] = out
    _save_cache(cache)
    return {**out, "cached": False}

def evidence_items(kpi, window, magnitude_pct, drivers, base_id=300,
                   backend="none", model=None, reliability_key=lambda d: f"news/{d['source']}"):
    """Full Stage-4 unstructured pass: search -> extract -> docket-shaped items."""
    q = f"{kpi} {' '.join(drivers)} retail India {window[0]} {window[1]}"
    hits = search(q, window=window, k=3)
    items, meta = [], []
    for i, (sim, doc) in enumerate(hits):
        ex = extract(doc, kpi, window, magnitude_pct, drivers, backend, model)
        if not ex.get("supports"):
            continue
        if ex["supports"] == "seasonal":       # calendar handled it upstream
            continue
        item = {"id": f"e-{base_id + i}",
                "claim": ex["claim"] or doc["title"],
                "value": doc["text"][:280],
                "source": reliability_key(doc), "fresh": doc["date"],
                "method": f"vector retrieval (cos {sim:.2f}) + {ex['method']}",
                "direction": ex["direction"] if ex["direction"] in ("confirming", "refuting") else "confirming",
                "supports": ex["supports"], "strength": round(min(0.4 + sim, 0.8), 2),
                "temporal": ex["temporal"], "live": True, "external": True}
        if ex.get("conflicts_with_internal"):
            item["conflicts_with"] = "internal-decomposition"
        items.append(item)
        meta.append({"file": doc["file"], "cos": sim, "cached": ex.get("cached"),
                     "llm": ex.get("llm_meta")})
    return items, meta
