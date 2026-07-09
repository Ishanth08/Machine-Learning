"""
features.py -- THE CORRECT, shared feature builder.

This is the "single source of truth". train.py uses it to build features, and the
CORRECT serving path uses the very same function. Because both sides call this one
function, there is NO training-serving skew when it is used everywhere.

The Skew Lab works by deliberately calling a *different*, slightly-wrong builder at
"deployment time" (see skews.py) so you can measure the damage.
"""
import os
import json
import math

STATS_PATH = os.path.join(os.path.dirname(__file__), "train_stats.json")

# Fixed feature order. Training and serving MUST agree on this exactly.
STREAMS = ["CSE", "ECE", "MECH"]
FEATURE_NAMES = ["cgpa_scaled", "aptitude_scaled", "experience", "soft_skills",
                 "stream_CSE", "stream_ECE", "stream_MECH"]

def load_stats():
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            return json.load(f)
    return {"cgpa_mean": 7.1, "aptitude_mean": 65.0}

def is_missing(v):
    """True if a value is absent: None, empty string, or NaN."""
    if v is None or v == "":
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False

def _num(v, default):
    return float(default) if is_missing(v) else float(v)

def build_features(record, stats=None):
    """Correct feature vector for one student record (a dict)."""
    if stats is None:
        stats = load_stats()

    cgpa        = _num(record.get("cgpa"),     stats["cgpa_mean"])
    aptitude    = _num(record.get("aptitude"), stats["aptitude_mean"])
    internships = _num(record.get("internships"), 0)
    projects    = _num(record.get("projects"),    0)
    soft_skills = _num(record.get("soft_skills"), 0)
    stream      = record.get("stream", "CSE")

    experience  = internships + 0.5 * projects
    onehot      = [1.0 if stream == s else 0.0 for s in STREAMS]

    return [cgpa / 10.0,        # scale CGPA 0..10 -> 0..1
            aptitude / 100.0,   # scale aptitude 0..100 -> 0..1
            experience,
            soft_skills] + onehot
