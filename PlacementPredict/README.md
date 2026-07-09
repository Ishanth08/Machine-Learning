# PlacementPredict

A tiny, complete machine-learning system that predicts whether a student will be
**placed**, based on their profile (CGPA, internships, projects, aptitude score,
soft-skills rating).

This repo is built for **students who are new to Python**. Every file is short and
heavily commented. You can clone it and run it on your own PC in a few minutes, and
try predictions in **four** ways:

| | **Single** (one student) | **Batch** (many students) |
|--------------|--------------------------|---------------------------|
| **Offline** (no server) | `predict_offline.py --json ...` | `predict_offline.py --csv ...` / `batch_predict.py` |
| **Online** (HTTP server) | `POST /predict` | `POST /predict_batch` |

---

## 1. What's in the folder

```
PlacementPredict/
├── make_data.py         # creates the dummy dataset (run once)
├── ingest.py            # reads the CSV into a table
├── features.py          # turns raw fields into model inputs  <-- shared by all
├── train.py             # trains the model and saves it (run once)
├── predict_core.py      # shared "score a student" logic
├── app.py               # ONLINE server (FastAPI): /predict, /predict_batch
├── query_online.py      # client that calls the running server over HTTP
├── predict_offline.py   # OFFLINE scorer: single (--json) or batch (--csv)
├── batch_predict.py     # OFFLINE batch job: scores data/batch_input.csv
├── monitor.py           # logs every prediction to logs/predictions.jsonl
├── requirements.txt     # the Python libraries you need
└── data/
    ├── students.csv       # dummy historical data WITH the known outcome
    └── batch_input.csv    # dummy new students WITHOUT outcome (to score)
```

The single most important idea: **`features.py` is imported by both training and
serving.** The feature logic lives in exactly one place, so the numbers the model
sees during prediction can never disagree with the numbers used during training.
(This avoids the classic bug called *training–serving skew*.)

---

## 2. Setup (do this once)

You need **Python 3.9 or newer**. Check with `python --version`.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/PlacementPredict.git
cd PlacementPredict



# 2. Install the required libraries
pip install -r requirements.txt
```

> On Windows, if `python` doesn't work, try `py` instead (e.g. `py -m venv .venv`).

---

## 3. Prepare data and train the model (do this once)

```bash
python make_data.py   # creates data/students.csv and data/batch_input.csv
python train.py       # trains the model -> models/model.joblib
```

`train.py` prints a test accuracy / F1-score and saves two files the predictors
need: `models/model.joblib` (the trained model) and `feature_params.json` (the
averages used to fill missing values).

> The dummy `data/*.csv` files are already included in the repo, so you can even
> skip `make_data.py` if you like — but you **must** run `train.py` once, because
> the trained model is generated locally (not committed).

---

## 4. Make predictions — OFFLINE (no server needed)

**Single student** (pass one student as JSON):

Run with no arguments to score a built-in example:

```bash
python predict_offline.py
```

**Batch** (score a whole CSV of students):

```bash
python predict_offline.py --csv data/batch_input.csv
# or the dedicated batch job (writes data/batch_output.csv):
python batch_predict.py
```

---

## 5. Make predictions — ONLINE (HTTP server)

**Terminal 1 — start the server** and leave it running:

```bash
uvicorn app:app --reload
```

You should see `Uvicorn running on http://127.0.0.1:8000`.
Open **http://127.0.0.1:8000/docs** in a browser for a clickable test page.

**Terminal 2 — send queries** with the included client:

```bash
python query_online.py                 # single example student
python query_online.py --batch         # a small example batch
python query_online.py --csv data/batch_input.csv   # batch from a CSV
```

A reply looks like:

```json
{ "placed": true, "probability": 0.991 }
```

---

## 6. How one prediction flows (the big picture)

```
your request  ->  app.py (validate)  ->  features.py (build features)
              ->  model (predict)     ->  JSON answer back to you
```

Every prediction is also written to `logs/predictions.jsonl` by `monitor.py`, so you
can later inspect the traffic and check whether incoming data is drifting.

---

## 7. Input fields

| Field         | Meaning                         | Example |
|---------------|---------------------------------|---------|
| `cgpa`        | GPA on a 0–10 scale             | 8.4     |
| `internships` | number of internships done      | 2       |
| `projects`    | number of projects done         | 5       |
| `aptitude`    | aptitude test score (0–100)     | 78      |
| `soft_skills` | soft-skills rating (1–5)        | 4       |

---

## 8. Troubleshooting

- **`ModuleNotFoundError`** — you skipped `pip install -r requirements.txt`, or your
  virtual environment isn't activated.
- **`FileNotFoundError: models/model.joblib`** — run `python train.py` first.
- **`Could not reach the server`** — start it in another terminal with
  `uvicorn app:app --reload`.
- **`python` not found on Windows** — use `py` instead.

---

*This is a teaching project. The data is completely synthetic and the model is
intentionally simple. Do not use it for real placement decisions.*
