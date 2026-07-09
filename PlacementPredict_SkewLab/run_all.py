"""
run_all.py -- run EVERY skew and print one summary table.

    python run_all.py

Great for a single side-by-side view of which inconsistencies hurt the most.
"""
import joblib
import pandas as pd
from features import build_features, load_stats
from skews import SKEWS

def main():
    stats = load_stats()
    model = joblib.load("model.joblib")
    df = pd.read_csv("data/live.csv")
    records = df.to_dict("records")

    X_correct = [build_features(r, stats) for r in records]
    p_correct = model.predict(X_correct)
    acc_correct = (p_correct == df["placed"]).mean()

    rows = [["(none) correct serving", f"{acc_correct:.3f}", "0.000", "0"]]
    for name, (desc, fn) in SKEWS.items():
        p = model.predict([fn(r, stats) for r in records])
        acc = (p == df["placed"]).mean()
        flipped = int((p != p_correct).sum())
        rows.append([name, f"{acc:.3f}", f"{acc_correct-acc:+.3f}", str(flipped)])

    out = pd.DataFrame(rows, columns=["skew", "accuracy", "acc_lost", "flipped"])
    print(f"Live students: {len(df)}   Clean accuracy: {acc_correct:.3f}\n")
    print(out.to_string(index=False))
    print("\nLesson: every one of these is a SILENT bug -- nothing crashes, the "
          "model just gets quietly worse. The cure is to use ONE shared feature "
          "function (features.py) in training AND serving.")

if __name__ == "__main__":
    main()
