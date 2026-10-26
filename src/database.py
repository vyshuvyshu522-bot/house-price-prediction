"""SQLite storage for prediction history (parameterised queries only)."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.config import FEATURES

# Column types for the input features, used to build the table.
_COLUMN_TYPES = {name: "TEXT" if name == "zipcode" else "REAL" for name in FEATURES}


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    columns = ", ".join(f"{name} {ctype} NOT NULL" for name, ctype in _COLUMN_TYPES.items())
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, "
            f"{columns}, predicted_price INTEGER NOT NULL, model_name TEXT NOT NULL)"
        )


def save_prediction(db_path, values: dict, price: int, model_name: str) -> int:
    cols = ["created_at", *FEATURES, "predicted_price", "model_name"]
    params = [datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
              *(values[f] for f in FEATURES), price, model_name]
    placeholders = ", ".join("?" for _ in cols)
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(f"INSERT INTO predictions ({', '.join(cols)}) VALUES ({placeholders})", params)
        return cur.lastrowid


def get_history(db_path, limit=200):
    with closing(_connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def clear_history(db_path) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM predictions")
