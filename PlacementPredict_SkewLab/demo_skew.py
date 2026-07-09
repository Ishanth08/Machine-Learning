"""
demo_skew.py -- see how ONE deployment inconsistency changes predictions.

    python demo_skew.py                 # list available skews
    python demo_skew.py zero_impute     # run a specific skew

It loads the trained model, then scores the 'live' students TWICE:
  * CORRECT : using the shared features.py (no skew)   -> the right pipeline
  * SKEWED  : using the broken builder from skews.py   -> the buggy pipeline
and reports the accuracy drop, how many predictions flipped, and -- for the
flipped students -- the predicted PROBABILITIES under each pipeline plus the
true outcome, so you can see the probability slide across the 0.5 line.
"""
import sys
import joblib
import pandas as pd
from features import build_features, load_stats
from skews import SKEWS

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SKEWS:
        print("Usage: python demo_skew.py <skew_name>\n\nAvailable skews:")
        for name, (desc, _) in SKEWS.items():
            print(f"  {name:14s} - {desc}")
        return

    name = sys.argv[1]
    desc, skew_fn = SKEWS[name]
    stats = load_stats()
    model = joblib.load("model.joblib")
    df = pd.read_csv("data/live.csv")
    records = df.to_dict("records")

    X_correct = [build_features(r, stats) for r in records]
    X_skewed  = [skew_fn(r, stats)        for r in records]

    # class labels (0/1)
    p_correct = model.predict(X_correct)
    p_skewed  = model.predict(X_skewed)
    # probability of placement (the '1' class)
    prob_correct = model.predict_proba(X_correct)[:, 1]
    prob_skewed  = model.predict_proba(X_skewed)[:, 1]

    acc_correct = (p_correct == df["placed"]).mean()
    acc_skewed  = (p_skewed  == df["placed"]).mean()
    flipped     = (p_correct != p_skewed).sum()

    print(f"SKEW: {name}  --  {desc}\n")
    print(f"  Accuracy WITHOUT skew (correct serving): {acc_correct:.3f}")
    print(f"  Accuracy WITH skew    (buggy serving)  : {acc_skewed:.3f}")
    print(f"  Accuracy lost to the bug               : {acc_correct-acc_skewed:+.3f}")
    print(f"  Predictions that FLIPPED               : {flipped} out of {len(df)}"
          f"  ({100*flipped/len(df):.1f}%)\n")

    # show a few students whose prediction flipped, WITH probabilities + truth
    changed = [i for i in range(len(df)) if p_correct[i] != p_skewed[i]]
    if changed:
        print("  Examples where the bug changed the answer")
        print("  (P = predicted probability of placement; 0.5 is the decision line):\n")
        print(f"  {'cgpa':>5} {'apt':>4} {'intern':>6} {'proj':>4} {'soft':>4} "
              f"{'stream':>6} | {'P(correct)':>10} {'P(skewed)':>9} | "
              f"{'pred_correct':>12} {'pred_skewed':>11} {'true':>4}")
        print("  " + "-"*94)
        for i in changed[:8]:
            r = records[i]
            cgpa = "NaN" if pd.isna(r["cgpa"]) else f"{r['cgpa']:.2f}"
            print(f"  {cgpa:>5} {r['aptitude']:4.0f} {r['internships']:6.0f} "
                  f"{r['projects']:4.0f} {r['soft_skills']:4.0f} {str(r['stream']):>6} | "
                  f"{prob_correct[i]:10.3f} {prob_skewed[i]:9.3f} | "
                  f"{int(p_correct[i]):12d} {int(p_skewed[i]):11d} {int(r['placed']):4d}")
        print(f"\n  Read a row like: P(correct)=0.63 (>=0.5 -> placed) but "
              f"P(skewed)=0.03 (<0.5 -> not placed).")
    else:
        print("  (No predictions flipped for this skew on this data.)")

if __name__ == "__main__":
    main()
