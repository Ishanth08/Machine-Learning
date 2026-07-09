"""
predict_offline.py -- make predictions WITHOUT starting a server (offline).

Single student (pass JSON on the command line):
    python predict_offline.py --json "{\"cgpa\": 8.4, \"internships\": 2, \"projects\": 5, \"aptitude\": 78, \"soft_skills\": 4}"

Batch from a CSV file:
    python predict_offline.py --csv data/batch_input.csv

If you give neither flag, a built-in example student is scored so you can see it work.
"""

import argparse
import json
import pandas as pd

from predict_core import predict_one, predict_many
from monitor import log_prediction

EXAMPLE = {"cgpa": 8.4, "internships": 2, "projects": 5, "aptitude": 78, "soft_skills": 4}


def run_single(record):
    result = predict_one(record)
    log_prediction(record, result)
    print("Input :", record)
    print("Result:", result)


def run_batch(csv_path):
    df = pd.read_csv(csv_path)
    records = df.to_dict("records")
    results = predict_many(records)
    for rec, res in zip(records, results):
        log_prediction(rec, res)

    out = df.copy()
    out["placed"] = [r["placed"] for r in results]
    out["probability"] = [r["probability"] for r in results]
    out_path = csv_path.replace(".csv", "_predictions.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved predictions -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline PlacementPredict scorer")
    parser.add_argument("--json", help="one student as a JSON string")
    parser.add_argument("--csv", help="path to a CSV of students (batch)")
    args = parser.parse_args()

    if args.csv:
        run_batch(args.csv)
    elif args.json:
        run_single(json.loads(args.json))
    else:
        print("No input given -- scoring a built-in example student.\n")
        run_single(EXAMPLE)


if __name__ == "__main__":
    main()
