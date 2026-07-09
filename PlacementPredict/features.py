"""
features.py -- THE SINGLE SOURCE OF TRUTH FOR FEATURES.

Both training (train.py) and serving (app.py, batch_predict.py, predict_offline.py)
import build_features() from THIS file. Because the feature logic lives in exactly
one place, the numbers fed to the model at serving time can never disagree with the
numbers used at training time. That is the cure for "training-serving skew".

A student's raw record is a plain dict, e.g.:
    {"cgpa": 8.4, "internships": 2, "projects": 5, "aptitude": 78, "soft_skills": 4}
build_features() turns it into the list of numbers the model actually expects.
"""

import os
import json

# Where the learned parameters (means used to fill missing values) are stored.
# train.py writes this file; serving reads it back.
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "feature_params.json")

# Order of the numbers we hand to the model. Handy for debugging.
FEATURE_NAMES = ["cgpa_scaled", "aptitude_scaled", "experience", "soft_skills"]


def load_params():
    """Load the feature parameters saved by train.py.

    If the file does not exist yet (e.g. before the first training run) we fall
    back to sensible defaults so nothing crashes.
    """
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            return json.load(f)
    return {"cgpa_mean": 7.1, "aptitude_mean": 65.0}


# Loaded once when this module is imported, and reused for every prediction.
PARAMS = load_params()


def _num(value, default):
    """Return `value` as a float, or `default` if it is missing/blank/None."""
    if value is None or value == "":
        return float(default)
    return float(value)


def build_features(record, params=None):
    """Turn one student's raw fields into the model's input vector (a list of floats).

    `record` : a dict like {"cgpa": 8.4, "internships": 2, ...}
    `params` : optional dict of learned parameters. If None, use the ones loaded
               from feature_params.json. train.py passes freshly-computed params.
    """
    if params is None:
        params = PARAMS

    # Fill missing CGPA / aptitude with the TRAINING-SET mean (same rule everywhere).
    cgpa = _num(record.get("cgpa"), params["cgpa_mean"])
    aptitude = _num(record.get("aptitude"), params["aptitude_mean"])
    internships = _num(record.get("internships"), 0)
    projects = _num(record.get("projects"), 0)
    soft_skills = _num(record.get("soft_skills"), 0)

    # A derived feature: rough "hands-on experience" score.
    experience = internships + 0.5 * projects

    # Scale big numbers into a small range so no single feature dominates.
    return [
        cgpa / 10.0,        # CGPA is 0..10  -> 0..1
        aptitude / 100.0,   # aptitude is 0..100 -> 0..1
        experience,
        soft_skills,
    ]
