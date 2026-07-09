"""
skews.py -- the deployment-time BUGS.

Each function here is a *broken* re-implementation of build_features, of the kind
that creeps in when the deployment/serving environment computes features slightly
differently from training. Every one looks harmless. Each returns a feature vector
the SAME length as the correct one, so nothing crashes -- the errors are silent.

A student can read each function and predict what will go wrong, then run demo_skew.py
to measure it.
"""
from features import STREAMS, _num, is_missing

def _base(record, stats):
    """Pull the raw fields (correctly) so each skew changes ONE thing only."""
    cgpa        = _num(record.get("cgpa"),     stats["cgpa_mean"])
    aptitude    = _num(record.get("aptitude"), stats["aptitude_mean"])
    internships = _num(record.get("internships"), 0)
    projects    = _num(record.get("projects"),    0)
    soft_skills = _num(record.get("soft_skills"), 0)
    stream      = record.get("stream", "CSE")
    return cgpa, aptitude, internships, projects, soft_skills, stream

# 1) Forgot to scale: training divided cgpa/10 and aptitude/100; serving forgot.
def no_scaling(record, stats):
    cgpa, apt, ins, prj, soft, stream = _base(record, stats)
    onehot = [1.0 if stream == s else 0.0 for s in STREAMS]
    return [cgpa, apt, ins + 0.5*prj, soft] + onehot     # <-- not divided!

# 2) Wrong missing-value fill: training used the train mean; serving uses 0.
def zero_impute(record, stats):
    # Simulate a missing cgpa arriving at serving, filled with 0 instead of mean.
    cgpa = 0.0 if is_missing(record.get("cgpa")) else float(record["cgpa"])
    _, apt, ins, prj, soft, stream = _base(record, stats)
    onehot = [1.0 if stream == s else 0.0 for s in STREAMS]
    return [cgpa/10.0, apt/100.0, ins + 0.5*prj, soft] + onehot

# 3) Category mismatch: serving sends stream in lower-case, so no one-hot matches
#    -> all-zero encoding, an "unknown stream" the model never saw that way.
def category_case(record, stats):
    cgpa, apt, ins, prj, soft, stream = _base(record, stats)
    stream = str(stream).lower()                         # "CSE" -> "cse"
    onehot = [1.0 if stream == s else 0.0 for s in STREAMS]   # never matches
    return [cgpa/10.0, apt/100.0, ins + 0.5*prj, soft] + onehot

# 4) Feature order swapped: serving puts soft_skills and experience in wrong slots.
def wrong_order(record, stats):
    cgpa, apt, ins, prj, soft, stream = _base(record, stats)
    onehot = [1.0 if stream == s else 0.0 for s in STREAMS]
    return [cgpa/10.0, apt/100.0, soft, ins + 0.5*prj] + onehot   # <-- swapped

# 5) Unit change upstream: a new form sends CGPA as a percentage (0..100),
#    but serving still divides by 10, so values are 10x too big.
def unit_shift(record, stats):
    cgpa, apt, ins, prj, soft, stream = _base(record, stats)
    cgpa = cgpa * 10.0                                   # 8.4 -> 84 (percentage)
    onehot = [1.0 if stream == s else 0.0 for s in STREAMS]
    return [cgpa/10.0, apt/100.0, ins + 0.5*prj, soft] + onehot

SKEWS = {
    "no_scaling":   ("Forgot to scale cgpa/aptitude", no_scaling),
    "zero_impute":  ("Missing value filled with 0, not the training mean", zero_impute),
    "category_case":("Stream sent lower-case -> unknown category", category_case),
    "wrong_order":  ("Two features placed in the wrong slots", wrong_order),
    "unit_shift":   ("CGPA arrives as a percentage (10x too big)", unit_shift),
}
