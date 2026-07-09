"""
train.py -- make the model. RUN THIS ONCE before serving.

    python train.py

Steps:
  1. read the historical data (ingest.py)
  2. compute the feature parameters (means) and SAVE them to feature_params.json
  3. build features with the SAME build_features() the server will use
  4. fit a Logistic Regression model
  5. save the trained model to models/model.joblib
"""

import os
import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from ingest import load_students
from features import build_features, PARAMS_PATH


def main():
    df = load_students("data/students.csv")
    records = df.to_dict("records")

    # ---- 2. Learn the feature parameters from the TRAINING data, then save them.
    params = {
        "cgpa_mean": float(df["cgpa"].mean()),
        "aptitude_mean": float(df["aptitude"].mean()),
    }
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved feature parameters -> {PARAMS_PATH}: {params}")

    # ---- 3. Build features (same code as serving), passing the fresh params.
    X = [build_features(r, params) for r in records]
    y = df["placed"].tolist()

    # Hold out 20% to honestly measure quality.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- 4. Fit the model ("learning" happens here).
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)

    # Quick honesty check on the held-out set.
    preds = model.predict(X_te)
    print(f"Test accuracy: {accuracy_score(y_te, preds):.3f}")
    print(f"Test F1-score: {f1_score(y_te, preds):.3f}")

    # ---- 5. Save the trained model.
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    print("Saved trained model -> models/model.joblib")


if __name__ == "__main__":
    main()
