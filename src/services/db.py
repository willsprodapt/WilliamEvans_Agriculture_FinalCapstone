"""SQLite persistence (stdlib sqlite3): advisory queries, feedback, evaluation runs/jobs."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from ..config import settings

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS advisory_queries(
  id TEXT PRIMARY KEY, created_at TEXT, query TEXT, status TEXT, duration_ms INTEGER, response_json TEXT);
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, query_id TEXT, recommendation_id TEXT, crop TEXT,
  helpful INTEGER NOT NULL, rating INTEGER, category TEXT, comment TEXT, case_ids_json TEXT);
CREATE TABLE IF NOT EXISTS eval_runs(
  run_id TEXT PRIMARY KEY, created_at TEXT, status TEXT, label TEXT, result_json TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@contextmanager
def conn():
    path = settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        c = sqlite3.connect(path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            c.executescript(SCHEMA)
            yield c
            c.commit()
        finally:
            c.close()


# ---- advisory queries
def save_query(qid: str, query: str, status: str, duration_ms: int, response: dict):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO advisory_queries VALUES(?,?,?,?,?,?)",
                  (qid, now(), query, status, duration_ms, json.dumps(response, default=str)))


def get_query(qid: str):
    with conn() as c:
        r = c.execute("SELECT * FROM advisory_queries WHERE id=?", (qid,)).fetchone()
    return None if r is None else {**dict(r), "response": json.loads(r["response_json"])}


def recent_queries(limit: int = 10) -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT id, created_at, query, status, duration_ms FROM advisory_queries ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def count_queries() -> int:
    with conn() as c:
        return c.execute("SELECT COUNT(*) FROM advisory_queries").fetchone()[0]


# ---- feedback
def add_feedback(d: dict) -> dict:
    with conn() as c:
        cur = c.execute("INSERT INTO feedback(created_at, query_id, recommendation_id, crop, helpful, rating, category, comment, case_ids_json) "
                        "VALUES(?,?,?,?,?,?,?,?,?)",
                        (now(), d.get("query_id"), d.get("recommendation_id"), d.get("crop"), int(bool(d["helpful"])), d.get("rating"),
                         d.get("category"), d.get("comment"), json.dumps(d.get("case_ids") or [])))
        fid = cur.lastrowid
        r = c.execute("SELECT * FROM feedback WHERE id=?", (fid,)).fetchone()
    return _fb(r)


def _fb(r) -> dict:
    d = dict(r)
    d["helpful"] = bool(d["helpful"])
    d["case_ids"] = json.loads(d.pop("case_ids_json") or "[]")
    return d


def list_feedback(page: int = 1, page_size: int = 25) -> dict:
    with conn() as c:
        total = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        rows = c.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ? OFFSET ?", (page_size, (page - 1) * page_size)).fetchall()
    return {"items": [_fb(r) for r in rows], "total": total, "page": page, "page_size": page_size}


def all_feedback() -> list[dict]:
    with conn() as c:
        return [_fb(r) for r in c.execute("SELECT * FROM feedback ORDER BY id").fetchall()]


# ---- evaluation runs
def save_eval_run(run_id: str, status: str, label: str | None, result: dict):
    with conn() as c:
        old = c.execute("SELECT created_at FROM eval_runs WHERE run_id=?", (run_id,)).fetchone()
        c.execute("INSERT OR REPLACE INTO eval_runs VALUES(?,?,?,?,?)",
                  (run_id, old["created_at"] if old else now(), status, label, json.dumps(result, default=str)))


def list_eval_runs() -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT run_id, created_at, status, label, result_json FROM eval_runs ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        res = json.loads(r["result_json"])
        out.append({"run_id": r["run_id"], "created_at": r["created_at"], "status": r["status"], "label": r["label"],
                    "summary": res.get("summary"), "versions": res.get("versions")})
    return out


def get_eval_run(run_id: str):
    with conn() as c:
        r = c.execute("SELECT * FROM eval_runs WHERE run_id=?", (run_id,)).fetchone()
    return None if r is None else {"run_id": r["run_id"], "created_at": r["created_at"], "status": r["status"],
                                   "label": r["label"], **json.loads(r["result_json"])}
