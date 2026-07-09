"""
batch_predict.py -- the OFFLINE BATCH job (the "overnight" style run).

    python batch_predict.py

Reads all students from data/batch_input.csv, scores them with the SAME model and
SAME feature code as the online app, and writes the results to
data/batch_output.csv. In real life a scheduler (cron / Task Scheduler) would run
this automatically, e.g. every night.
"""

import pandas as pd
from predict_core import predict_many

INPUT = "data/batch_input.csv"
OUTPUT = "data/batch_output.csv"


def main():
    df = pd.read_csv(INPUT)
    results = predict_many(df.to_dict("records"))

    df["placed"] = [r["placed"] for r in results]
    df["probability"] = [r["probability"] for r in results]
    df.to_csv(OUTPUT, index=False)

    print(f"Scored {len(df)} students from {INPUT}")
    print(f"Wrote results -> {OUTPUT}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
