"""
query_online.py -- a small CLIENT that calls the running server over HTTP.

First, in one terminal, start the server:
    uvicorn app:app --reload

Then, in a SECOND terminal:
    python query_online.py                 # single example student
    python query_online.py --batch         # a small batch of students
    python query_online.py --csv data/batch_input.csv   # batch from a CSV

This shows the ONLINE path: an HTTP request goes out, a JSON answer comes back.
"""

import argparse
import pandas as pd
import requests

URL = "http://127.0.0.1:8000"

EXAMPLE = {"cgpa": 8.4, "internships": 2, "projects": 5, "aptitude": 78, "soft_skills": 4}
EXAMPLE_BATCH = [
    {"cgpa": 8.4, "internships": 2, "projects": 5, "aptitude": 78, "soft_skills": 4},
    {"cgpa": 6.1, "internships": 0, "projects": 1, "aptitude": 52, "soft_skills": 2},
    {"cgpa": 9.2, "internships": 3, "projects": 7, "aptitude": 88, "soft_skills": 5},
]


def single(record):
    r = requests.post(f"{URL}/predict", json=record)
    r.raise_for_status()
    print("Sent   :", record)
    print("Got    :", r.json())


def batch(records):
    r = requests.post(f"{URL}/predict_batch", json=records)
    r.raise_for_status()
    for rec, res in zip(records, r.json()):
        print(rec, "->", res)


def main():
    parser = argparse.ArgumentParser(description="Online client for PlacementPredict")
    parser.add_argument("--batch", action="store_true", help="send the example batch")
    parser.add_argument("--csv", help="send a batch read from a CSV file")
    args = parser.parse_args()

    try:
        if args.csv:
            batch(pd.read_csv(args.csv).to_dict("records"))
        elif args.batch:
            batch(EXAMPLE_BATCH)
        else:
            single(EXAMPLE)
    except requests.exceptions.ConnectionError:
        print("Could not reach the server. Is it running?")
        print("Start it first with:  uvicorn app:app --reload")


if __name__ == "__main__":
    main()
