"""
zero_impute_study.py -- what happens when a MISSING cgpa is filled differently
in training vs serving.

The practically important facts:
  * During TRAINING we fill a missing cgpa with the TRAINING MEAN (a typical student).
  * At SERVING we must do the SAME. If serving instead fills with 0, we get the
    'zero_impute' skew -- and predictions for missing-cgpa students go wrong.

This script trains ONE model (mean-imputed, as we actually do) and then serves the
live students two ways -- fill=mean (correct) and fill=0 (bug) -- and prints the
difference.

Run:  python make_data.py   (once, to create the data)
      python zero_impute_study.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

STREAMS = ["CSE", "ECE", "MECH"]

def is_missing(v):
    if v is None or v == "":
        return True
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return False

def build_features(record, cgpa_fill):
    """Only the missing-cgpa fill value changes; everything else is identical."""
    cgpa = cgpa_fill if is_missing(record.get("cgpa")) else float(record["cgpa"])
    aptitude    = float(record.get("aptitude", 65))
    internships = float(record.get("internships", 0))
    projects    = float(record.get("projects", 0))
    soft_skills = float(record.get("soft_skills", 0))
    stream      = record.get("stream", "CSE")
    experience  = internships + 0.5 * projects
    onehot      = [1.0 if stream == s else 0.0 for s in STREAMS]
    return [cgpa/10.0, aptitude/100.0, experience, soft_skills] + onehot

def make_X(records, fill):
    return [build_features(r, fill) for r in records]

def acc(model, records, y, fill):
    return (np.array(model.predict(make_X(records, fill))) == np.array(y)).mean()

def main():
    train = pd.read_csv("data/train.csv")
    live  = pd.read_csv("data/live.csv")
    train_recs, live_recs = train.to_dict("records"), live.to_dict("records")
    y_train, y_live = train["placed"].tolist(), live["placed"].tolist()

    TRAIN_MEAN = float(train["cgpa"].mean())    # mean of the KNOWN cgpa values
    ZERO = 0.0

    print("="*64)
    print("WHAT WE FILL MISSING CGPA WITH DURING TRAINING")
    print("="*64)
    print(f"  Training rows          : {len(train)}  "
          f"(missing cgpa: {int(train['cgpa'].isna().sum())})")
    print(f"  Fill value in TRAINING : the training mean = {TRAIN_MEAN:.2f}")
    print(f"  (A missing student is treated as a TYPICAL student, not a zero one.)\n")

    # The model we actually ship: trained with mean-imputation.
    model = LogisticRegression(max_iter=1000).fit(make_X(train_recs, TRAIN_MEAN), y_train)

    print("="*64)
    print("SERVING THE SAME MODEL TWO WAYS  (all 300 live students)")
    print("="*64)
    a_mean = acc(model, live_recs, y_live, TRAIN_MEAN)
    a_zero = acc(model, live_recs, y_live, ZERO)
    print(f"  serve fill = MEAN  (correct, matches training): accuracy {a_mean:.3f}")
    print(f"  serve fill = ZERO  (the bug)                  : accuracy {a_zero:.3f}")
    print(f"  accuracy lost to the zero-impute bug          : {a_mean-a_zero:+.3f}\n")

    # Zoom in: only the students whose cgpa is actually missing.
    miss = [r for r in live_recs if is_missing(r.get("cgpa"))]
    y_miss = [r["placed"] for r in miss]
    am = acc(model, miss, y_miss, TRAIN_MEAN)
    az = acc(model, miss, y_miss, ZERO)
    print("="*64)
    print(f"ONLY THE {len(miss)} MISSING-CGPA STUDENTS (where the fill matters)")
    print("="*64)
    print(f"  serve fill = MEAN : accuracy {am:.3f}")
    print(f"  serve fill = ZERO : accuracy {az:.3f}")
    print(f"  accuracy lost     : {am-az:+.3f}\n")

    pm = model.predict_proba(make_X(miss, TRAIN_MEAN))[:, 1]
    pz = model.predict_proba(make_X(miss, ZERO))[:, 1]
    flipped = int(((pm >= 0.5) != (pz >= 0.5)).sum())

    print("  Predicted probability of placement, same student, two fills:")
    print(f"  {'apt':>4} {'intern':>6} {'proj':>4} {'soft':>4} {'stream':>6} "
          f"{'P(mean)':>8} {'P(zero)':>8} {'true':>4}  result")
    print("  " + "-"*60)
    for j, r in enumerate(miss[:8]):
        flip = (pm[j] >= 0.5) != (pz[j] >= 0.5)
        tag = "FLIPPED to 'not placed'" if flip else "same"
        print(f"  {r['aptitude']:4.0f} {r['internships']:6.0f} {r['projects']:4.0f} "
              f"{r['soft_skills']:4.0f} {str(r['stream']):>6} "
              f"{pm[j]:8.3f} {pz[j]:8.3f} {int(r['placed']):4d}  {tag}")

    print(f"\n  {flipped} of {len(miss)} missing-cgpa students FLIPPED to 'not placed'")
    print("  purely because 0 (scaled 0.0) looks like the weakest possible student,")
    print("  while the mean looks like a typical one -- which is what the model")
    print("  was trained to expect for a missing value.")

if __name__ == "__main__":
    main()
