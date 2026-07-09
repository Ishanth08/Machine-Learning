"""
make_data.py -- create dummy student data for the Skew Lab.

Run once:  python make_data.py
Creates data/train.csv (to train the model) and data/live.csv (new students
that arrive at 'deployment time', WITH their true outcome so we can score errors).
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)          # fixed seed => same data for everyone
STREAMS = ["CSE", "ECE", "MECH"]

def make(n):
    cgpa        = np.round(RNG.normal(7.1, 1.1, n).clip(4, 10), 2)
    aptitude    = RNG.normal(65, 15, n).clip(20, 100).round().astype(int)
    internships = RNG.integers(0, 4, n)
    projects    = RNG.integers(0, 8, n)
    soft_skills = RNG.integers(1, 6, n)
    stream      = RNG.choice(STREAMS, n)
    # hidden 'true' rule (plus noise) the model will learn
    score = (0.9*(cgpa-7) + 0.6*internships + 0.25*projects
             + 0.03*(aptitude-65) + 0.4*(soft_skills-3) + RNG.normal(0, 0.7, n))
    placed = (score > 0).astype(int)
    return pd.DataFrame({"cgpa": cgpa, "aptitude": aptitude,
        "internships": internships, "projects": projects,
        "soft_skills": soft_skills, "stream": stream, "placed": placed})

def blank_some_cgpa(df, frac=0.15):
    """Erase CGPA for a fraction of rows to mimic real missing values."""
    idx = RNG.choice(df.index, size=int(frac*len(df)), replace=False)
    df.loc[idx, "cgpa"] = np.nan
    return df

def main():
    os.makedirs("data", exist_ok=True)
    blank_some_cgpa(make(800)).to_csv("data/train.csv", index=False)
    blank_some_cgpa(make(300)).to_csv("data/live.csv", index=False)
    print("Wrote data/train.csv (800 rows) and data/live.csv (300 rows).")
    print("~15% of rows have a MISSING cgpa on purpose (for the zero_impute skew).")

if __name__ == "__main__":
    main()
