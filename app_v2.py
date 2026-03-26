"""
Hard-fix patch for edswrrer/general app_v2.py indexing inconsistency.

Top-level knobs (program başından değiştirilebilir):
- RAG_VIDEO_LIMIT: RAG/indexing tarafında işlenecek maksimum video sayısı.
  Varsayılan 20 (isteğe göre değiştir).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock
from typing import Dict, Any

# ──────────────────────────────────────────────────────────────────────────────
# Program başından değiştirilebilir sabitler
# ──────────────────────────────────────────────────────────────────────────────
RAG_VIDEO_LIMIT = 20


class RuntimeHardFix:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance


def resolve_data_root(base_file: str) -> Path:
    """Force all runtime artifacts to the app's directory (not caller CWD)."""
    return Path(base_file).resolve().parent


def absolute_data_paths(base_file: str) -> Dict[str, str]:
    root = resolve_data_root(base_file)
    return {
        "db_path": str((root / "yt_rag.db").resolve()),
        "chroma_path": str((root / "chroma_store").resolve()),
    }


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def safe_count(c: sqlite3.Cursor, sql: str, params: tuple = ()) -> int:
    row = c.execute(sql, params).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def reconcile_stats(db: sqlite3.Connection, chroma_count: int) -> Dict[str, int]:
    c = db.cursor()
    stats = {
        "videos": safe_count(c, "SELECT COUNT(*) FROM videos"),
        "processed": safe_count(c, "SELECT COUNT(*) FROM videos WHERE processed=1"),
        "chunks": safe_count(c, "SELECT COUNT(*) FROM chunks"),
        "relations": safe_count(c, "SELECT COUNT(*) FROM relations"),
        "chroma": int(chroma_count or 0),
        "rag_video_limit": int(RAG_VIDEO_LIMIT),
    }

    # Hard architectural fallback: if processed is stale, derive from chunked videos.
    if stats["processed"] == 0 and stats["chunks"] > 0:
        stats["processed"] = safe_count(
            c,
            """
            SELECT COUNT(DISTINCT v.id)
            FROM videos v
            JOIN chunks c2 ON c2.video_id = v.id
            """,
        )
    return stats


def pick_process_existing_candidates(
    db: sqlite3.Connection,
    rag_video_limit: int = RAG_VIDEO_LIMIT,
) -> Dict[str, Any]:
    """
    Process-existing adaylarını döndürür ve RAG tarafında işlenecek video sayısını
    hard-limit ile sınırlar (varsayılan 20).
    """
    c = db.cursor()
    total_videos = safe_count(c, "SELECT COUNT(*) FROM videos")
    rows = c.execute(
        """
        SELECT v.id
        FROM videos v
        LEFT JOIN (
            SELECT video_id, COUNT(*) AS chunk_count
            FROM chunks
            GROUP BY video_id
        ) cc ON cc.video_id = v.id
        WHERE COALESCE(TRIM(v.transcript), '') <> ''
          AND (v.processed = 0 OR COALESCE(cc.chunk_count, 0) = 0)
        ORDER BY v.id DESC
        LIMIT ?
        """,
        (int(rag_video_limit),),
    ).fetchall()

    candidates = [r[0] for r in rows]
    state = "ready" if candidates else ("empty_source" if total_videos > 0 else "no_videos")
    return {
        "state": state,
        "total_videos": total_videos,
        "rag_video_limit": int(rag_video_limit),
        "candidates": candidates,
    }


def ask_guard(stats: Dict[str, int]) -> Dict[str, Any]:
    if stats.get("chunks", 0) <= 0 and stats.get("chroma", 0) <= 0:
        return {
            "ok": False,
            "error": "No data indexed yet. Index transcript-bearing videos first.",
            "hint": (
                "Use process-existing after ensuring videos.transcript is populated "
                f"(current RAG_VIDEO_LIMIT={RAG_VIDEO_LIMIT})."
            ),
            "stats": stats,
        }
    return {"ok": True, "stats": stats}
