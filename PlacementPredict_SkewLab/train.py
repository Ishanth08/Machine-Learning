"""
train.py -- train the model correctly, using features.py.

    python train.py

Saves:
  model.joblib       the trained model
  train_stats.json   the means learned from training (so serving can reuse them)
"""
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import build_features, STATS_PATH

def main():
    df = pd.read_csv("data/train.csv")

    stats = {"cgpa_mean": float(df["cgpa"].mean()),
             "aptitude_mean": float(df["aptitude"].mean())}
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    X = [build_features(r, stats) for r in df.to_dict("records")]
    y = df["placed"].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    print(f"Held-out accuracy (clean): {accuracy_score(y_te, model.predict(X_te)):.3f}")

    joblib.dump(model, "model.joblib")
    print("Saved model.joblib and train_stats.json")

if __name__ == "__main__":
    main()
