"""
ingest.py -- read the raw student records from disk into a table.

A "DataFrame" (from the pandas library) is just a spreadsheet held in memory:
each row is one student, each column is one field.
"""

import pandas as pd


def load_students(path):
    """Read the CSV file at `path` and return it as a pandas DataFrame.

    Rows with no known placement outcome are dropped, because we can only learn
    from students whose result we actually know.
    """
    df = pd.read_csv(path)
    if "placed" in df.columns:
        df = df.dropna(subset=["placed"])
    return df


if __name__ == "__main__":
    # Quick self-test: python ingest.py
    df = load_students("data/students.csv")
    print(f"Loaded {len(df)} students. First few rows:")
    print(df.head())
