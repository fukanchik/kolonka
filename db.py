"""
db.py — SQLite persistence for СИЭ headword extraction results.

One shared database (sie_titles.db) across however many volumes you
work on; rows are distinguished by the `volume` column (the djvu
file's stem, e.g. "СИЭ-01").

Columns match the CLI script's original CSV output (page, column,
column_page, guessed, title, bold_ratio, first_line), plus a few
fields the interactive editor needs to actually redraw/edit boxes
(id, bbox, volume).

Saving a page is a full replace: every Save from the editor deletes
that page's existing rows for that volume and re-inserts the current
box list. This keeps the semantics simple (no diffing/upsert-by-id
logic) and matches how the old per-page JSON corrections file worked.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "sie_titles.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS titles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    volume TEXT NOT NULL,
    page INTEGER NOT NULL,
    "column" TEXT,
    column_page INTEGER,
    guessed INTEGER,
    title TEXT NOT NULL,
    bold_ratio REAL,
    first_line TEXT,
    bbox TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_titles_volume_page ON titles(volume, page);

-- Tracks which (volume, page) have been explicitly saved at least once,
-- independent of how many boxes they currently have. Needed because
-- "saved with 0 boxes" (e.g. you deleted the only entry and saved) must
-- be distinguishable from "never saved" -- an empty list is falsy in
-- Python/JS, so we can't tell those two cases apart just by looking at
-- whether `titles` has any rows for that page.
CREATE TABLE IF NOT EXISTS saved_pages (
    volume TEXT NOT NULL,
    page INTEGER NOT NULL,
    PRIMARY KEY (volume, page)
);

-- Rejected candidates ("suggestions") that the user has already acted
-- on -- promoted into a real box, whether or not that box was later
-- deleted -- so they don't keep reappearing as a red suggestion on
-- every reload. Keyed by the suggestion's stable per-page id (e.g.
-- "r3"), which is just its index into that page's cached analysis
-- rejected-list, stable for as long as the analysis cache is.
CREATE TABLE IF NOT EXISTS dismissed_suggestions (
    volume TEXT NOT NULL,
    page INTEGER NOT NULL,
    suggestion_id TEXT NOT NULL,
    PRIMARY KEY (volume, page, suggestion_id)
);

-- Per-page clockwise rotation in degrees (any angle, not just 90-degree
-- turns -- useful for deskewing a slightly tilted scan), applied at
-- render time before OCR so everything downstream (analysis, display,
-- crops) is consistently in the rotated orientation.
CREATE TABLE IF NOT EXISTS page_rotations (
    volume TEXT NOT NULL,
    page INTEGER NOT NULL,
    angle REAL NOT NULL,
    PRIMARY KEY (volume, page)
);

-- Manual correction of a page's printed column numbers (the ones
-- normally read from its running header, e.g. "45 ... — ... 46").
-- Takes priority over both the OCR'd header and backward extrapolation
-- for that page -- and, since extrapolation for LATER pages searches
-- backward for a known anchor, a correction here also fixes any later
-- pages that would otherwise have extrapolated from a bad value.
CREATE TABLE IF NOT EXISTS page_header_overrides (
    volume TEXT NOT NULL,
    page INTEGER NOT NULL,
    left_num INTEGER NOT NULL,
    right_num INTEGER NOT NULL,
    PRIMARY KEY (volume, page)
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate_add_column_page_end(conn)
    return conn


def _migrate_add_column_page_end(conn):
    """
    Adds the column_page_end column (end of a printed-page range, for
    entries spanning more than one column) to pre-existing databases.
    Safe to call on every connection: checks first, so it's a no-op
    once the column exists.
    """
    cols = [row["name"] for row in conn.execute("PRAGMA table_info(titles)")]
    if "column_page_end" not in cols:
        with conn:
            conn.execute("ALTER TABLE titles ADD COLUMN column_page_end INTEGER")


def backfill_saved_pages():
    """
    One-time migration for databases created before saved_pages existed:
    any (volume, page) that already has rows in `titles` clearly was
    saved at some point, so mark it as such. Safe to call every startup
    (INSERT OR IGNORE is a no-op once caught up).
    """
    conn = get_conn()
    try:
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO saved_pages (volume, page) "
                "SELECT DISTINCT volume, page FROM titles"
            )
    finally:
        conn.close()


def get_header_override(volume, page):
    """(left_num, right_num) manually set for this page, or None if not overridden."""
    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT left_num, right_num FROM page_header_overrides WHERE volume=? AND page=?",
            (volume, page),
        )
        row = cur.fetchone()
        return (row["left_num"], row["right_num"]) if row else None
    finally:
        conn.close()


def set_header_override(volume, page, left_num, right_num):
    conn = get_conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO page_header_overrides (volume, page, left_num, right_num) VALUES (?,?,?,?) "
                "ON CONFLICT(volume, page) DO UPDATE SET left_num=excluded.left_num, right_num=excluded.right_num",
                (volume, page, left_num, right_num),
            )
    finally:
        conn.close()


def clear_header_override(volume, page):
    conn = get_conn()
    try:
        with conn:
            conn.execute(
                "DELETE FROM page_header_overrides WHERE volume=? AND page=?", (volume, page)
            )
    finally:
        conn.close()


def get_rotation(volume, page):
    """Current rotation in degrees (float, e.g. 90 or 1.5) for this page, 0 if never set."""
    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT angle FROM page_rotations WHERE volume=? AND page=?", (volume, page)
        )
        row = cur.fetchone()
        return row["angle"] if row else 0.0
    finally:
        conn.close()


def set_rotation(volume, page, angle):
    angle = round(float(angle) % 360, 2)
    conn = get_conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO page_rotations (volume, page, angle) VALUES (?,?,?) "
                "ON CONFLICT(volume, page) DO UPDATE SET angle=excluded.angle",
                (volume, page, angle),
            )
    finally:
        conn.close()
    return angle


ALLOWED_DIRECT_UPDATE_FIELDS = {"column_page", "column_page_end"}


def update_title_field(row_id, field, value):
    """
    Directly updates one field on one row by its id -- used by the
    /titles browse-all page for quick inline edits without going back
    to the per-page editor. Restricted to a small allowlist of fields
    since this takes a raw column name. Editing column_page directly
    also clears the `guessed` flag, since a manually-entered value is
    no longer a guess.
    """
    if field not in ALLOWED_DIRECT_UPDATE_FIELDS:
        raise ValueError(f"Field not editable this way: {field}")
    conn = get_conn()
    try:
        with conn:
            if field == "column_page":
                conn.execute(
                    "UPDATE titles SET column_page=?, guessed=0 WHERE id=?", (value, row_id)
                )
            else:
                conn.execute(f"UPDATE titles SET {field}=? WHERE id=?", (value, row_id))
    finally:
        conn.close()


def is_page_saved(volume, page):
    """True if this page was ever explicitly saved (even with 0 boxes)."""
    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT 1 FROM saved_pages WHERE volume=? AND page=? LIMIT 1", (volume, page)
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def get_page_boxes(volume, page):
    """Returns list of dicts (with 'id' and 'bbox' as [x0,y0,x1,y1]) or [] if none saved."""
    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT * FROM titles WHERE volume=? AND page=? ORDER BY id", (volume, page)
        )
        rows = cur.fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "bbox": json.loads(row["bbox"]),
                "column": row["column"],
                "column_page": row["column_page"],
                "column_page_end": row["column_page_end"],
                "guessed": bool(row["guessed"]),
                "bold_ratio": row["bold_ratio"],
                "first_line": row["first_line"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def replace_page_boxes(volume, page, boxes):
    """
    boxes: list of dicts with title, bbox, column, column_page,
    column_page_end, guessed, bold_ratio, first_line. Deletes existing
    rows for (volume, page) and inserts these instead, in one
    transaction. Also marks the page as saved (see saved_pages) even if
    boxes is empty -- an intentional "I deleted everything on this
    page" is a valid saved state, not the same as "never touched this
    page".
    """
    conn = get_conn()
    try:
        with conn:
            conn.execute("DELETE FROM titles WHERE volume=? AND page=?", (volume, page))
            conn.executemany(
                """INSERT INTO titles
                   (volume, page, "column", column_page, column_page_end, guessed, title, bold_ratio, first_line, bbox)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        volume, page,
                        b.get("column"), b.get("column_page"), b.get("column_page_end"),
                        int(bool(b.get("guessed"))),
                        b["title"], b.get("bold_ratio"), b.get("first_line"),
                        json.dumps(b["bbox"]),
                    )
                    for b in boxes
                ],
            )
            conn.execute(
                "INSERT OR IGNORE INTO saved_pages (volume, page) VALUES (?, ?)",
                (volume, page),
            )
    finally:
        conn.close()


def get_dismissed_suggestions(volume, page):
    """Set of suggestion_id strings already handled (promoted at some point) for this page."""
    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT suggestion_id FROM dismissed_suggestions WHERE volume=? AND page=?",
            (volume, page),
        )
        return {row["suggestion_id"] for row in cur.fetchall()}
    finally:
        conn.close()


def add_dismissed_suggestions(volume, page, suggestion_ids):
    """Adds to (doesn't replace) the dismissed set -- once dismissed, always dismissed."""
    if not suggestion_ids:
        return
    conn = get_conn()
    try:
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO dismissed_suggestions (volume, page, suggestion_id) VALUES (?,?,?)",
                [(volume, page, sid) for sid in suggestion_ids],
            )
    finally:
        conn.close()


def delete_page_boxes(volume, page):
    conn = get_conn()
    try:
        with conn:
            conn.execute("DELETE FROM titles WHERE volume=? AND page=?", (volume, page))
            conn.execute("DELETE FROM saved_pages WHERE volume=? AND page=?", (volume, page))
            conn.execute("DELETE FROM dismissed_suggestions WHERE volume=? AND page=?", (volume, page))
    finally:
        conn.close()


def export_all(volume=None):
    """All saved rows, optionally filtered to one volume. For building a final export."""
    conn = get_conn()
    try:
        if volume:
            cur = conn.execute(
                "SELECT * FROM titles WHERE volume=? ORDER BY page, id", (volume,)
            )
        else:
            cur = conn.execute("SELECT * FROM titles ORDER BY volume, page, id")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _title_sort_key(title):
    """
    Case-insensitive (full Unicode, not just ASCII -- SQLite's built-in
    COLLATE NOCASE only folds ASCII, which isn't enough for Cyrillic),
    ignoring all spaces, hyphens ('-'), and commas (','), and ignoring
    a leading « (so e.g. «Известия» sorts under 'и' next to Индия
    rather than clustering all «-prefixed titles ahead of everything
    else).
    """
    t = title.lower()
    for ch in (" ", "-", ","):
        t = t.replace(ch, "")
    if t.startswith("«"):
        t = t[1:]
    return t


def search_titles(volume, query=None):
    """Like export_all, but optionally filtered to titles containing `query`
    (case-insensitive substring match), for the browse-all-titles page.
    Sorted per _title_sort_key (case/space/leading-« insensitive)."""
    conn = get_conn()
    try:
        if query:
            cur = conn.execute(
                "SELECT * FROM titles WHERE volume=? AND title LIKE ?", (volume, f"%{query}%")
            )
        else:
            cur = conn.execute("SELECT * FROM titles WHERE volume=?", (volume,))
        rows = [dict(row) for row in cur.fetchall()]
        rows.sort(key=lambda r: (_title_sort_key(r["title"]), r["page"], r["id"]))
        return rows
    finally:
        conn.close()


def count_pages_with_saved_titles(volume):
    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM saved_pages WHERE volume=?", (volume,)
        )
        return cur.fetchone()[0]
    finally:
        conn.close()
