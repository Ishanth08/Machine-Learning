"""
make_data.py -- generate DUMMY student data so the project runs out of the box.

Run once:  python make_data.py

It creates:
  data/students.csv     -> historical records WITH the known outcome (for training)
  data/batch_input.csv  -> new students WITHOUT outcome (for batch prediction)

The data is completely synthetic (made up). Nothing here is a real person.
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)   # fixed seed => everyone gets the same data


def make_students(n):
    cgpa = np.round(RNG.normal(7.1, 1.1, n).clip(4.0, 10.0), 2)
    internships = RNG.integers(0, 4, n)
    projects = RNG.integers(0, 8, n)
    aptitude = RNG.normal(65, 15, n).clip(20, 100).round().astype(int)
    soft_skills = RNG.integers(1, 6, n)   # 1..5

    # A hidden "true" rule the model will try to learn (plus some randomness).
    score = (0.9 * (cgpa - 7)
             + 0.6 * internships
             + 0.25 * projects
             + 0.03 * (aptitude - 65)
             + 0.4 * (soft_skills - 3)
             + RNG.normal(0, 0.8, n))
    placed = (score > 0).astype(int)

    return pd.DataFrame({
        "student_id": [f"S{1000 + i}" for i in range(n)],
        "cgpa": cgpa,
        "internships": internships,
        "projects": projects,
        "aptitude": aptitude,
        "soft_skills": soft_skills,
        "placed": placed,
    })


def main():
    os.makedirs("data", exist_ok=True)

    train = make_students(600)
    train.to_csv("data/students.csv", index=False)
    print(f"Wrote data/students.csv  ({len(train)} rows, with 'placed' label)")

    # New students to score later. Drop the label to mimic "unknown outcome".
    batch = make_students(20).drop(columns=["placed"])
    batch.to_csv("data/batch_input.csv", index=False)
    print(f"Wrote data/batch_input.csv ({len(batch)} rows, no label)")


if __name__ == "__main__":
    main()
