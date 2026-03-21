"""Create a standalone SQLite database file for deployment.

Run this script to generate a fresh SQLite DB with the schema used by the app.
Optionally specify a target path via the PREDICTION_DB environment variable.

Example:
  python create_deploy_db.py
  PREDICTION_DB=../prod_predictions.db python create_deploy_db.py
"""

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get(
    "PREDICTION_DB",
    os.path.join(BASE_DIR, "data", "predictions_prod.db"),
)

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

schema = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hospital TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    file_name TEXT,
    record_id INTEGER,
    wbc REAL,
    rbc REAL,
    hb REAL,
    platelets REAL,
    probability REAL,
    risk TEXT,
    
);
"""

with sqlite3.connect(DB_PATH) as conn:
    conn.execute(schema)
    conn.commit()

print(f"Initialized deployment DB at: {DB_PATH}")
