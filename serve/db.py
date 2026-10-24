"""SQLite prediction log: schema, inserts, and the named analytics queries."""

import json
import os
import re
import sqlite3

SQL_DIR = os.path.join(os.path.dirname(__file__), "sql")


def _load_queries():
    text = open(os.path.join(SQL_DIR, "analytics.sql"), encoding="utf-8").read()
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    # parts = [preamble, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


QUERIES = _load_queries()


class Store:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(open(os.path.join(SQL_DIR, "schema.sql"), encoding="utf-8").read())
        self.conn.commit()

    def ensure_slide(self, slide_id, source="upload", note=None):
        self.conn.execute("INSERT OR IGNORE INTO slides(slide_id, source, note) VALUES (?, ?, ?)",
                          (slide_id, source, note))
        self.conn.commit()

    def log_cell(self, slide_id, model_version, pred_class, confidence, probs, accepted, true_class=None):
        cur = self.conn.execute(
            "INSERT INTO cells(slide_id, model_version, pred_class, confidence, probs, accepted, true_class)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (slide_id, model_version, pred_class, float(confidence),
             json.dumps([round(float(p), 5) for p in probs]), int(accepted), true_class))
        self.conn.commit()
        return cur.lastrowid

    def save_report(self, slide_id, author, status, text, notes):
        cur = self.conn.execute(
            "INSERT INTO reports(slide_id, author, status, text, verifier_notes) VALUES (?, ?, ?, ?, ?)",
            (slide_id, author, status, text, json.dumps(notes)))
        self.conn.commit()
        return cur.lastrowid

    def query(self, name, **params):
        return [dict(r) for r in self.conn.execute(QUERIES[name], params).fetchall()]
