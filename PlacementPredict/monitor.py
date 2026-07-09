"""
monitor.py -- monitoring hooks.

A "hook" is a small piece of code that runs on the side to record what happened,
without changing the answer. Here we append every prediction to a log file so we
can later inspect traffic and check for drift.
"""

import os
import json
import datetime


def log_prediction(inputs, output):
    """Append one JSON line describing this prediction to logs/predictions.jsonl."""
    os.makedirs("logs", exist_ok=True)
    record = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "inputs": inputs,   # what the user sent
        "output": output,   # what we answered
    }
    with open(os.path.join("logs", "predictions.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
