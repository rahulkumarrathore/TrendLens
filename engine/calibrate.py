"""Stage 5 — Calibration & precedent (cold-start bridge, now LIVE).

Three pieces:
  1. Beta priors per driver — the historical_precedent criterion's source of truth,
     persisted to priors.json and updated by every feedback verdict. This IS the
     cold-start precedent model; the GBM slot activates when feedback history is
     deep enough to train on (the replayable feedback_log.jsonl is its dataset).
  2. Isotonic calibration (pool-adjacent-violators, implemented directly) mapping
     raw fuzzy confidence -> empirical precision, fit on (confidence, verdict)
     pairs from calib_samples.jsonl. Identity until MIN_SAMPLES — calibration
     must never pretend to knowledge it doesn't have.
  3. replay_feedback() — simulates N analyst verdicts consistent with ground
     truth through the REAL feedback path (pipeline.apply_feedback + prior
     updates + calibration samples), demonstrating the learning loop over
     simulated time without training any model.
"""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIORS_PATH = os.path.join(ROOT, "priors.json")
SAMPLES_PATH = os.path.join(ROOT, "calib_samples.jsonl")
MIN_SAMPLES = 12

DEFAULT_PRIORS = {"coupon_expiry": [6, 2], "competitor_promo": [3, 3],
                  "demand_shift": [2, 3], "marketing_gap": [2, 2],
                  "gateway_issue": [2, 2], "supply_stockout": [3, 2],
                  "launch_volatility": [2, 2]}

# ---------------- priors ----------------
def priors():
    if os.path.exists(PRIORS_PATH):
        with open(PRIORS_PATH) as f:
            return {k: tuple(v) for k, v in json.load(f).items()}
    return {k: tuple(v) for k, v in DEFAULT_PRIORS.items()}

def _save_priors(p):
    with open(PRIORS_PATH, "w") as f:
        json.dump({k: list(v) for k, v in p.items()}, f, indent=1)

def update_prior(driver, verdict):
    """confirm -> alpha+1, reject -> beta+1 (bounded so no prior saturates)."""
    p = dict(priors())
    a, b = p.get(driver, (2, 2))
    if verdict == "confirm":
        a = min(a + 1, 40)
    else:
        b = min(b + 1, 40)
    p[driver] = (a, b)
    _save_priors(p)
    return p[driver]

def reset():
    for path in (PRIORS_PATH, SAMPLES_PATH):
        if os.path.exists(path):
            os.remove(path)

# ---------------- isotonic calibration (PAV) ----------------
def add_sample(confidence, verdict):
    with open(SAMPLES_PATH, "a") as f:
        f.write(json.dumps({"c": round(float(confidence), 3),
                            "y": 1 if verdict == "confirm" else 0}) + "\n")

def _samples():
    if not os.path.exists(SAMPLES_PATH):
        return []
    with open(SAMPLES_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]

def _pav(xs, ys):
    """Pool-adjacent-violators: nondecreasing fit of y on x."""
    order = np.argsort(xs)
    x, y = np.asarray(xs)[order], np.asarray(ys)[order].astype(float)
    w = np.ones_like(y)
    blocks = [[y[i], w[i], x[i], x[i]] for i in range(len(y))]   # mean, weight, xlo, xhi
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) \
                / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [m, blocks[i][1] + blocks[i + 1][1], blocks[i][2], blocks[i + 1][3]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return blocks

def curve():
    """Returns (xs, ys) of the current calibration map, or None pre-warmup."""
    s = _samples()
    if len(s) < MIN_SAMPLES:
        return None
    blocks = _pav([r["c"] for r in s], [r["y"] for r in s])
    xs = [b[2] for b in blocks] + [blocks[-1][3]]
    ys = [b[0] for b in blocks] + [blocks[-1][0]]
    return xs, ys

def calibrated(confidence):
    """Map raw confidence through the isotonic fit; identity while cold."""
    cv = curve()
    if cv is None:
        return round(float(confidence), 3)
    xs, ys = cv
    return round(float(np.interp(confidence, xs, ys)), 3)

def status():
    s = _samples()
    return {"samples": len(s), "warm": len(s) >= MIN_SAMPLES, "min_samples": MIN_SAMPLES}

# ---------------- feedback replay ----------------
def replay_feedback(n_rounds=8, seed=7):
    """Simulate analyst verdicts consistent with planted ground truth, pushed
    through the REAL feedback machinery. Verdict noise ~10% models imperfect
    analysts. Returns a drift report."""
    from . import pipeline, investigate            # late import: avoid cycles
    rng = np.random.default_rng(seed)
    truth = {"coupon_expiry": "confirm", "competitor_promo": "confirm",
             "demand_shift": "reject", "marketing_gap": "reject",
             "gateway_issue": "reject", "supply_stockout": "reject"}
    before_p, before_r = dict(priors()), dict(pipeline.RELIABILITY)
    r = investigate.investigate("gmv", "2026-07-15", "2026-07-21", role="cfo")
    for _ in range(n_rounds):
        for h in r["finding"]["hypotheses"]:
            v = truth.get(h["name"])
            if v is None:
                continue
            if rng.random() < 0.10:                        # analyst noise
                v = "reject" if v == "confirm" else "confirm"
            pipeline.apply_feedback(r["run_id"], h["name"], v, r["docket"])
            update_prior(h["name"], v)
            add_sample(h["confidence"], v)
    after_p, after_r = dict(priors()), dict(pipeline.RELIABILITY)
    return {"rounds": n_rounds,
            "priors": {k: {"before": list(before_p.get(k, (2, 2))),
                           "after": list(after_p.get(k, (2, 2)))}
                       for k in truth},
            "reliability": {k: {"before": before_r.get(k), "after": after_r.get(k)}
                            for k in after_r if after_r.get(k) != before_r.get(k)},
            "calibration": status()}
