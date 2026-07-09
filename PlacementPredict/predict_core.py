"""
predict_core.py -- the shared prediction logic used by EVERY entry point.

Both the online API (app.py) and the offline scripts (predict_offline.py,
batch_predict.py) call predict_one() here. One function, reused everywhere, so
online and offline answers are always identical.
"""

import joblib
from features import build_features

MODEL_PATH = "models/model.joblib"
THRESHOLD = 0.5

# Load the trained model ONCE when this module is first imported.
_model = joblib.load(MODEL_PATH)


def predict_one(record):
    """Score a single student record (a dict) and return a tidy result dict."""
    x = build_features(record)                       # STEP: build feature vector
    prob = float(_model.predict_proba([x])[0][1])    # STEP: model inference
    return {                                          # STEP: format the answer
        "placed": bool(prob >= THRESHOLD),
        "probability": round(prob, 3),
    }


def predict_many(records):
    """Score a list of student records. Returns a list of result dicts."""
    return [predict_one(r) for r in records]
