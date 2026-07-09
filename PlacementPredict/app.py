"""
app.py -- the ONLINE (real-time) serving app, built with FastAPI.

Start the server:
    uvicorn app:app --reload

Then it answers HTTP requests at:
    POST /predict        -> score ONE student
    POST /predict_batch  -> score a LIST of students in one call
    GET  /health         -> simple "is it alive?" check

Open http://127.0.0.1:8000/docs in a browser for an auto-generated test page.
"""

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from predict_core import predict_one, predict_many
from monitor import log_prediction

app = FastAPI(title="PlacementPredict")


class Student(BaseModel):
    """Describes the JSON body. FastAPI validates the types automatically:
    a bad request gets a 422 error and the model is never called."""
    cgpa: float
    internships: int
    projects: int
    aptitude: float
    soft_skills: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(student: Student):
    """Score ONE student (online / single query)."""
    record = student.dict()
    result = predict_one(record)
    log_prediction(record, result)
    return result


@app.post("/predict_batch")
def predict_batch(students: List[Student]):
    """Score MANY students in one request (online / batch query)."""
    records = [s.dict() for s in students]
    results = predict_many(records)
    for rec, res in zip(records, results):
        log_prediction(rec, res)
    return results
