"""Local prediction history (SQLite file inside the project; nothing leaves the machine)."""
from __future__ import annotations

import json
import sqlite3
import time

import pandas as pd

from .config import ROOT

DB_PATH = ROOT / 'results' / 'prediction_history.sqlite'


def _conn(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, mode TEXT NOT NULL, synthetic INTEGER NOT NULL,
        summary TEXT NOT NULL, inputs_json TEXT NOT NULL, ctr REAL NOT NULL, result TEXT NOT NULL)''')
    return c


def add(mode, synthetic, summary, inputs, ctr, result, path=DB_PATH):
    with _conn(path) as c:
        c.execute('INSERT INTO predictions (ts, mode, synthetic, summary, inputs_json, ctr, result) VALUES (?,?,?,?,?,?,?)',
                  (time.strftime('%Y-%m-%d %H:%M:%S'), mode, int(synthetic), summary, json.dumps(inputs, default=str),
                   float(ctr), result))


def load(path=DB_PATH) -> pd.DataFrame:
    c = _conn(path)
    try:
        return pd.read_sql_query('SELECT id, ts, mode, synthetic, summary, ctr, result FROM predictions ORDER BY id DESC', c)
    finally:
        c.close()


def clear(path=DB_PATH):
    with _conn(path) as c:
        c.execute('DELETE FROM predictions')
