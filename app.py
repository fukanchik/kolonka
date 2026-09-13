#!/usr/bin/env python3
"""
Flask viewer + EDITOR for СИЭ headword detection.

Renders each DjVu page, runs the detection pipeline from sie_ocr.py,
and shows the page image with interactive overlay boxes you can:
  - drag to move
  - resize via corner handles
  - rename (double-click)
  - delete (x button)
  - create new ones (Add box, then drag on the page)

Auto-detected results seed the editable box list the first time you
open a page:
  - accepted headwords -> editable green boxes
  - rejected candidates -> read-only dashed red "suggestions"; click
    one to promote it into an editable green box (pre-filled with its
    OCR guess as the title, which you can then rename/reposition).

Anything with NO box at all around real headword text on the page is a
straight-up miss (regex never even flagged it) -- draw a new box for
those yourself.

Click "Save page" to write your corrected list to disk. It's saved per
page, so you can leave and come back without losing work. "Reset to
auto-detected" throws away your edits for that page and reloads the
original algorithm output.

MULTIPLE VOLUMES
----------------
Pass more than one .djvu file on the command line to work across
several volumes in one running server. Use the volume selector in the
nav bar (or GET /switch_volume/<name>) to switch which one is
"current" for your browser session -- the page editor, /titles, and
all API endpoints only ever operate on the currently-selected volume.
Each volume's cache and saved data (keyed by its filename stem) are
kept completely separate in the shared sie_titles.db.

USAGE
-----
    python3 app.py /path/to/СИЭ-01.djvu
    # then open http://localhost:5000 in a browser

    python3 app.py /path/to/СИЭ-01.djvu /path/to/СИЭ-02.djvu --port 8080

    python3 app.py /path/to/СИЭ-01.djvu --start-page 13
"""

import argparse
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import pytesseract
from flask import Flask, jsonify, redirect, request, send_file, session, url_for
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import sie_ocr  # noqa: E402
import db  # noqa: E402

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # local single-machine tool; just needs to sign the session cookie

VOLUMES = {}  # {stem: djvu_path}, populated in main() from command-line args
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
MAX_DISPLAY_WIDTH = 1300  # downscale for browser display
REGION_OCR_LANG = "rus"  # rus-only for box-level re-reads: these crops are almost
                          # always a single Cyrillic headword, and combining with
                          # "eng" (needed at full-page-analysis time for embedded
                          # transliterations) made Tesseract mis-read plain Cyrillic
                          # words as English/Latin far too often for these short crops.
REGION_OCR_PADDING = 6  # px, in original resolution, added around a box before OCR/bold scoring


def volume_selector_html(next_target):
    """
    A <select> for switching the current volume, or '' if there's only
    one volume loaded (no point showing a picker with a single option).
    next_target is 'page' (goes through the editor's unsaved-changes
    guard via JS) or 'titles' (plain navigation, no dirty state there).
    """
    if len(VOLUMES) <= 1:
        return ""
    current = volume_name()
    opts = "".join(
        f'<option value="{v}"{" selected" if v == current else ""}>{v}</option>'
        for v in sorted(VOLUMES)
    )
    if next_target == "page":
        onchange = 'onchange="switchVolume(this)"'
    else:
        onchange = f"onchange=\"location.href='/switch_volume/'+this.value+'?next={next_target}'\""
    return f'<label style="margin-right:14px;">📚 <select {onchange}>{opts}</select></label>'


def volume_name():
    """The currently-selected volume's stem (e.g. 'СИЭ-01'), tracked per browser
    session. Defaults to (and self-heals to) the first volume alphabetically if
    unset or no longer valid -- e.g. server restarted with a different file list."""
    vol = session.get("volume")
    if vol not in VOLUMES:
        vol = sorted(VOLUMES.keys())[0]
        session["volume"] = vol
    return vol


def current_djvu_path():
    return VOLUMES[volume_name()]


def cache_paths(page_num):
    stem = volume_name()
    return {
        "analysis": CACHE_DIR / f"{stem}_p{page_num}_analysis.json",
        "raw_png": CACHE_DIR / f"{stem}_p{page_num}_raw.png",
        "full_png": CACHE_DIR / f"{stem}_p{page_num}_full.png",
    }


def get_analysis(page_num):
    """Auto-detected results (unedited). Cached forever once computed (until invalidated, e.g. by rotation)."""
    paths = cache_paths(page_num)
    if paths["analysis"].exists():
        result = _load_json(paths["analysis"])
        return _heal_cached_header(result, paths["analysis"])

    rotation = db.get_rotation(volume_name(), page_num)
    result = sie_ocr.analyze_page(current_djvu_path(), page_num, rotation=rotation)
    tiff_path = result.pop("tiff_path")
    with open(paths["analysis"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    # Cache full-res + downscaled-for-display PNGs now, while we still
    # have the just-rendered TIFF handy (avoids re-rendering via ddjvu
    # again later for on-demand region OCR / bold scoring).
    _cache_full_and_raw(tiff_path, paths)
    return result


def _heal_cached_header(result, analysis_path):
    """
    Re-validates an already-cached page's header against the
    left-odd/right=left+1 invariant (see sie_ocr.reconcile_header_numbers)
    and silently fixes + persists it if it was cached before that check
    existed, or before a fix to it. A no-op for already-correct headers.
    """
    header = result.get("header")
    if not header:
        return result
    left, right, corrected = sie_ocr.reconcile_header_numbers(header["left_num"], header["right_num"])
    if corrected:
        print(
            f"[app] healed cached header for {analysis_path.name}: "
            f"{header['left_num']}/{header['right_num']} -> {left}/{right}",
            file=sys.stderr,
        )
        header["left_num"], header["right_num"] = left, right
        result["header"] = header
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    return result


def invalidate_page_cache(page_num):
    """Force a fresh render+OCR next time this page is requested (e.g. after rotating)."""
    for p in cache_paths(page_num).values():
        if p.exists():
            p.unlink()


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cache_full_and_raw(tiff_path, paths):
    im = Image.open(tiff_path).convert("RGB")
    im.save(paths["full_png"], "PNG")
    scale = min(1.0, MAX_DISPLAY_WIDTH / im.width)
    disp = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS) if scale < 1.0 else im
    disp.save(paths["raw_png"], "PNG")


def ensure_full_png(page_num):
    """Full-resolution rendered page PNG, cached permanently. Returns (path, width, height)."""
    paths = cache_paths(page_num)
    if not paths["full_png"].exists():
        import tempfile
        tmpdir = tempfile.mkdtemp()
        tiff_path = str(Path(tmpdir) / f"p{page_num}.tiff")
        rotation = db.get_rotation(volume_name(), page_num)
        sie_ocr.render_page(current_djvu_path(), page_num, tiff_path, rotation=rotation)
        _cache_full_and_raw(tiff_path, paths)
    with Image.open(paths["full_png"]) as im:
        return paths["full_png"], im.width, im.height


def ensure_raw_png(page_num):
    paths = cache_paths(page_num)
    if not paths["raw_png"].exists():
        ensure_full_png(page_num)  # regenerates both full_png and raw_png
    with Image.open(paths["raw_png"]) as im:
        return paths["raw_png"], im.width, im.height


def crop_region(page_num, bbox, padding=REGION_OCR_PADDING):
    """Crop [x0,y0,x1,y1] (original-res coords) out of the full-res page image, padded/clamped."""
    full_path, width, height = ensure_full_png(page_num)
    im = Image.open(full_path).convert("RGB")
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0) - padding)
    y0 = max(0, int(y0) - padding)
    x1 = min(width, int(x1) + padding)
    y1 = min(height, int(y1) + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return im.crop((x0, y0, x1, y1))


def ocr_crop_text(crop_im):
    if crop_im is None:
        return ""
    text = pytesseract.image_to_string(crop_im, lang=REGION_OCR_LANG, config="--psm 6")
    text = " ".join(text.split())  # collapse whitespace/newlines
    return text.translate(sie_ocr.HOMOGLYPH_MAP)


def bold_ratio_for_crop(crop_im):
    if crop_im is None:
        return None
    gray = np.array(crop_im.convert("L"))
    binary = gray < 128
    h, w = binary.shape
    r = sie_ocr.retention_ratio(binary, 0, 0, w, h)
    return round(r, 3) if r is not None else None


def known_header_for_page(volume, page_num, trigger_ocr=False):
    """
    (left_num, right_num) for this exact page, checking in priority
    order:
      1. a manual override (highest priority -- always trusted)
      2. its own header, from an already-cached analysis, or freshly
         computed if trigger_ocr=True (only the *current* page being
         resolved should do this; anchor search for other pages never
         should, to avoid silently OCRing dozens of untouched pages)
      3. already-saved (non-guessed) DB rows for that page
    Returns (left_num, right_num) or None if nothing usable was found.
    """
    override = db.get_header_override(volume, page_num)
    if override:
        return override

    analysis_path = cache_paths(page_num)["analysis"]
    if analysis_path.exists() or trigger_ocr:
        # get_analysis (not a raw file read) so an already-cached-but-
        # since-corrected header gets healed here too.
        header = get_analysis(page_num).get("header")
        if header:
            return header["left_num"], header["right_num"]

    left_num = right_num = None
    for row in db.get_page_boxes(volume, page_num):
        if row["guessed"] or row["column_page"] is None:
            continue
        if row["column"] == "left":
            left_num = row["column_page"]
        elif row["column"] == "right":
            right_num = row["column_page"]
    if left_num is not None or right_num is not None:
        if left_num is None:
            left_num = right_num - 1
        if right_num is None:
            right_num = left_num + 1
        # Validate a DB-derived pair: a page saved before the header-
        # invariant fix existed could have an inconsistent pair baked
        # in, which would otherwise silently become a bad anchor.
        left_num, right_num, _corrected = sie_ocr.reconcile_header_numbers(left_num, right_num)
        return left_num, right_num

    return None


def find_backward_anchor(volume, current_page):
    """
    Search pages current_page-1 down to 1 for a known left/right
    printed-column-number pair (see known_header_for_page) to
    extrapolate forward from. Never triggers a fresh render+OCR on a
    page just to search it, since that could mean OCRing dozens of
    untouched pages on every save -- only already-available
    information (manual overrides, already-cached analysis, or
    already-saved DB rows) is used.
    Returns (anchor_page, left_num, right_num) or None if nothing
    usable was found anywhere earlier in the volume.
    """
    for k in range(current_page - 1, 0, -1):
        found = known_header_for_page(volume, k, trigger_ocr=False)
        if found:
            return k, found[0], found[1]
    return None


def resolve_column_numbers(page_num):
    """
    (left_num, right_num, guessed) for this page: a manual override or
    its own parsed header if either is available (guessed=False),
    otherwise extrapolated forward two columns per page from the
    nearest earlier page with a known anchor (guessed=True), otherwise
    (None, None, True) if nothing usable was found at all.
    """
    volume = volume_name()
    found = known_header_for_page(volume, page_num, trigger_ocr=True)
    if found:
        return found[0], found[1], False

    anchor = find_backward_anchor(volume, page_num)
    if anchor:
        anchor_page, _anchor_left, anchor_right = anchor
        pages_between = page_num - anchor_page
        left_num = anchor_right + 2 * pages_between - 1
        right_num = anchor_right + 2 * pages_between
        return left_num, right_num, True

    return None, None, True


def get_column_page_info(page_num, x0, page_width):
    """(column, column_page, guessed) for a box at x0 on this page -- see resolve_column_numbers."""
    column = "left" if x0 < page_width / 2 else "right"
    left_num, right_num, guessed = resolve_column_numbers(page_num)
    column_page = (left_num if column == "left" else right_num) if not (left_num is None and right_num is None) else None
    return column, column_page, guessed


@app.route("/titles")
def titles_view():
    q = request.args.get("q", "").strip()
    rows = db.search_titles(volume_name(), q or None)
    total_pages = sie_ocr.get_page_count(current_djvu_path())
    reviewed_pages = db.count_pages_with_saved_titles(volume_name())

    def esc(s):
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def editable_cell(row_id, field, value, suffix_html=""):
        display = value if value is not None else ""
        raw = "" if value is None else str(value)
        return (
            f'<td class="colcell" data-id="{row_id}" data-field="{field}" data-value="{raw}" '
            f'onclick="editCell(this)" title="Click to edit">{display}{suffix_html}</td>'
        )

    body_rows = "".join(
        f"""<tr>
              <td><a href="{url_for('page_view', n=r['page'])}">p.{r['page']}</a></td>
              <td>{esc(r['column'])}</td>
              {editable_cell(r['id'], 'column_page', r['column_page'],
                              ' <span class="guess">(guessed)</span>' if r['guessed'] else '')}
              {editable_cell(r['id'], 'column_page_end', r.get('column_page_end'))}
              <td>{esc(r['title'])}</td>
              <td>{r['bold_ratio'] if r['bold_ratio'] is not None else ''}</td>
              <td class="fl">{esc(r['first_line'])}</td>
            </tr>"""
        for r in rows
    ) or '<tr><td colspan="7"><em>No saved titles yet' + (f' matching "{esc(q)}"' if q else "") + '.</em></td></tr>'

    return f"""
    <html>
    <head>
      <title>СИЭ — all saved titles ({volume_name()})</title>
      <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
        td, th {{ border: 1px solid #ccc; padding: 5px 9px; text-align: left; }}
        th {{ background: #eee; position: sticky; top: 0; }}
        tr:hover {{ background: #f7f7f7; }}
        .fl {{ color: #555; font-size: 12px; }}
        .guess {{ color: #a60; font-size: 11px; }}
        #top {{ margin-bottom: 16px; }}
        input[type=text] {{ font-size: 15px; padding: 5px 8px; width: 260px; }}
        button {{ font-size: 15px; padding: 5px 12px; }}
        .stats {{ color: #555; margin-bottom: 10px; }}
        .colcell {{ cursor: pointer; min-width: 40px; }}
        .colcell:hover {{ background: #eef; }}
      </style>
    </head>
    <body>
      <div id="top">
        <h2>Saved titles — {esc(volume_name())}</h2>
        {volume_selector_html('titles')}
        <div class="stats">
          {len(rows)} title(s) shown &middot;
          {reviewed_pages} of {total_pages} pages have saved edits
        </div>
        <form method="get">
          <input type="text" name="q" placeholder="Search titles…" value="{esc(q)}">
          <button type="submit">Search</button>
          {'<a href="' + url_for('titles_view') + '" style="margin-left:10px;">Clear</a>' if q else ''}
        </form>
      </div>
      <table>
        <tr><th>Page</th><th>Col</th><th>Printed # start</th><th>Printed # end</th><th>Title</th><th>Bold</th><th>First line</th></tr>
        {body_rows}
      </table>
      <script>
        async function editCell(td) {{
          const id = td.dataset.id, field = td.dataset.field;
          const current = td.dataset.value || '';
          const label = field === 'column_page' ? 'Printed # start' : 'Printed # end';
          const input = prompt(label + ' (leave blank to clear):', current);
          if (input === null) return;
          const value = input.trim();
          const res = await fetch('/api/titles/update', {{
            method: 'POST', headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{id, field, value: value === '' ? null : value}})
          }});
          const data = await res.json();
          if (data.ok) {{
            const shown = data.value !== null && data.value !== undefined ? data.value : '';
            td.textContent = shown;  // also drops any stale "(guessed)" suffix
            td.dataset.value = shown;
          }} else {{
            alert('Update failed: ' + (data.error || 'unknown error'));
          }}
        }}
      </script>
    </body>
    </html>
    """


@app.route("/")
def root():
    return redirect(url_for("page_view", n=app.config.get("START_PAGE", 1)))


@app.route("/switch_volume/<vol>")
def switch_volume(vol):
    if vol not in VOLUMES:
        return f"Unknown volume: {vol}", 404
    session["volume"] = vol
    if request.args.get("next") == "titles":
        return redirect(url_for("titles_view"))
    return redirect(url_for("page_view", n=1))


@app.route("/page/<int:n>")
def page_view(n):
    total = sie_ocr.get_page_count(current_djvu_path())
    n = max(1, min(n, total))
    html = PAGE_HTML.replace("__TOTAL__", str(total))
    html = html.replace("__PREV__", str(max(1, n - 1)))
    html = html.replace("__NEXT__", str(min(total, n + 1)))
    html = html.replace("__VOLUME_SELECTOR__", volume_selector_html("page"))
    html = html.replace("__VOLUME__", volume_name())
    html = html.replace("__N__", str(n))
    return html


@app.route("/image_raw/<int:n>")
def page_image(n):
    get_analysis(n)  # ensures raw PNG is cached
    png_path, _w, _h = ensure_raw_png(n)
    return send_file(png_path, mimetype="image/png")


@app.route("/api/page_data/<int:n>")
def api_page_data(n):
    analysis = get_analysis(n)
    _png_path, disp_w, disp_h = ensure_raw_png(n)
    scale = disp_w / analysis["width"]

    if db.is_page_saved(volume_name(), n):
        saved = db.get_page_boxes(volume_name(), n)
        boxes = [
            {
                "id": b["id"], "title": b["title"], "bbox": b["bbox"],
                "column_override": b["column_page"],
                "column_end_override": b["column_page_end"],
            }
            for b in saved
        ]
        source = "corrections"
    else:
        boxes = [
            {"id": f"a{i}", "title": e["title"], "bbox": e["bbox"],
             "column_override": None, "column_end_override": None}
            for i, e in enumerate(analysis["accepted"])
        ]
        source = "auto"

    dismissed = db.get_dismissed_suggestions(volume_name(), n)
    suggestions = [
        {"id": f"r{i}", "text": e["text"], "bbox": e["bbox"], "reason": e["reason"]}
        for i, e in enumerate(analysis["rejected"])
        if f"r{i}" not in dismissed
    ]

    header = analysis.get("header")
    override = db.get_header_override(volume_name(), n)
    resolved_left, resolved_right, resolved_guessed = resolve_column_numbers(n)
    if override:
        header_str = f"{override[0]} — {override[1]} (manually set)"
    elif header:
        header_str = f"{header['left_num']} {header['first_word']} — {header['last_word']} {header['right_num']}"
    elif resolved_left is not None:
        header_str = f"{resolved_left} — {resolved_right} (extrapolated, page header not parsed)"
    else:
        header_str = "not parsed for this page, and no earlier known column number to extrapolate from"

    return jsonify({
        "page_num": n,
        "orig_width": analysis["width"],
        "orig_height": analysis["height"],
        "display_width": disp_w,
        "display_height": disp_h,
        "scale": scale,
        "boxes": boxes,
        "suggestions": suggestions,
        "source": source,
        "header": header_str,
        "header_is_override": override is not None,
        "header_left_num": resolved_left,
        "header_right_num": resolved_right,
        "header_numbers_guessed": resolved_guessed,
        "rotation": db.get_rotation(volume_name(), n),
    })


@app.route("/api/ocr_region/<int:n>", methods=["POST"])
def api_ocr_region(n):
    """Re-OCR just the given box (after a resize or a brand-new box) so
    the title can be auto-filled/suggested from the actual pixels."""
    payload = request.get_json(force=True)
    bbox = payload["bbox"]
    crop_im = crop_region(n, bbox)
    text = ocr_crop_text(crop_im)
    return jsonify({"text": text})


@app.route("/api/save/<int:n>", methods=["POST"])
def api_save(n):
    payload = request.get_json(force=True)
    in_boxes = payload.get("boxes", [])
    dismissed_ids = payload.get("dismissed_suggestions", [])
    analysis = get_analysis(n)
    page_width = analysis["width"]

    out_boxes = []
    for b in in_boxes:
        bbox = b["bbox"]
        x0 = bbox[0]
        override = b.get("column_override")
        if override is not None and str(override).strip() != "":
            column = "left" if x0 < page_width / 2 else "right"  # still a geometric fact
            column_page = int(override)
            guessed = False
        else:
            column, column_page, guessed = get_column_page_info(n, x0, page_width)

        end_override = b.get("column_end_override")
        column_page_end = (
            int(end_override)
            if end_override is not None and str(end_override).strip() != ""
            else None
        )

        crop_im = crop_region(n, bbox)
        out_boxes.append({
            "title": b["title"],
            "bbox": bbox,
            "column": column,
            "column_page": column_page,
            "column_page_end": column_page_end,
            "guessed": guessed,
            "bold_ratio": bold_ratio_for_crop(crop_im),
            "first_line": ocr_crop_text(crop_im),
        })

    db.replace_page_boxes(volume_name(), n, out_boxes)
    db.add_dismissed_suggestions(volume_name(), n, dismissed_ids)
    return jsonify({"ok": True, "count": len(out_boxes)})


@app.route("/api/reset/<int:n>", methods=["POST"])
def api_reset(n):
    db.delete_page_boxes(volume_name(), n)
    return jsonify({"ok": True})


@app.route("/api/titles/update", methods=["POST"])
def api_titles_update():
    """Direct single-cell edit from the /titles browse-all page (Printed # start/end)."""
    payload = request.get_json(force=True)
    row_id = int(payload["id"])
    field = payload["field"]
    raw = payload.get("value")
    value = int(raw) if raw is not None and str(raw).strip() != "" else None
    try:
        db.update_title_field(row_id, field, value)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "value": value})


@app.route("/api/header_override/<int:n>", methods=["POST"])
def api_set_header_override(n):
    """Manually correct this page's printed column numbers (left/right)."""
    payload = request.get_json(force=True)
    left = int(payload["left"])
    right = int(payload["right"])
    db.set_header_override(volume_name(), n, left, right)
    return jsonify({"ok": True, "left": left, "right": right})


@app.route("/api/header_override/<int:n>/clear", methods=["POST"])
def api_clear_header_override(n):
    db.clear_header_override(volume_name(), n)
    left, right, guessed = resolve_column_numbers(n)
    return jsonify({"ok": True, "left": left, "right": right, "guessed": guessed})


@app.route("/api/rotate/<int:n>", methods=["POST"])
def api_rotate(n):
    """
    Rotate a page, clockwise positive, by any angle (not just 90-degree
    turns -- e.g. 1.5 degrees to deskew a slightly tilted scan). Accepts
    either {"delta": X} (rotate further by X degrees from the current
    angle) or {"angle": X} (set the absolute angle directly).

    Since this changes the pixel orientation everything else is
    measured against, it invalidates that page's cached analysis
    (forcing a fresh render+OCR under the new orientation next time
    it's requested) and clears any saved boxes/dismissed-suggestions
    for that page, since their coordinates no longer correspond to
    anything sensible in the new orientation.
    """
    payload = request.get_json(force=True)
    volume = volume_name()
    if "angle" in payload:
        target = float(payload["angle"])
    else:
        target = db.get_rotation(volume, n) + float(payload.get("delta", 90))
    new_angle = db.set_rotation(volume, n, target)
    invalidate_page_cache(n)
    db.delete_page_boxes(volume, n)
    return jsonify({"ok": True, "angle": new_angle})


PAGE_HTML = r"""
<html>
<head>
  <title>СИЭ page __N__/__TOTAL__</title>
  <style>
    body { font-family: sans-serif; margin: 0; display: flex; height: 100vh; }
    #imgpane { flex: 0 0 auto; padding: 10px; background: #ddd; overflow: auto; }
    #sidepane { flex: 1 1 auto; padding: 10px 20px; max-width: 420px; overflow: auto; }
    #stage { position: relative; display: inline-block; }
    #stage img { display: block; border: 1px solid #999; }
    .col-num-label {
      position: absolute; top: 6px; padding: 2px 8px; font-size: 14px; font-weight: bold;
      border-radius: 3px; background: rgba(255,255,255,0.9); cursor: pointer;
    }
    .col-num-label:hover { background: rgba(255,255,180,0.95); }
    .col-num-label.left { left: 6px; }
    .col-num-label.right { right: 6px; }
    .col-num-label.parsed { color: #060; border: 1px solid #060; }
    .col-num-label.guessed { color: #a60; border: 1px dashed #a60; }
    .col-num-label.override { color: #06c; border: 1px solid #06c; }
    .nav { margin: 10px 0; }
    .nav a, .nav button, #controls button { font-size: 15px; padding: 6px 12px; margin-right: 6px; cursor: pointer; }
    .legend span { display:inline-block; width:14px; height:14px; margin-right:6px; vertical-align:middle; }
    .g { background: rgb(0,150,0); }
    .r { border: 2px dashed rgb(200,0,0); }
    table { border-collapse: collapse; width: 100%; margin-bottom: 16px; font-size: 13px; }
    td, th { border: 1px solid #ccc; padding: 3px 6px; text-align: left; }
    .box {
      position: absolute; border: 3px solid rgb(0,150,0); background: rgba(0,150,0,0.08);
      cursor: move; box-sizing: border-box;
    }
    .box.labels-hidden {
      border-color: #ccc; background: rgba(0,0,0,0.02);
    }
    .box.labels-hidden .label, .box.labels-hidden .del {
      display: none;
    }
    .box .label {
      position: absolute; top: -22px; left: -3px; background: rgb(0,150,0); color: #fff;
      font-size: 12px; padding: 1px 5px; white-space: nowrap; user-select: none;
    }
    .box .del {
      position: absolute; top: -22px; right: -3px; background: #900; color: #fff;
      font-size: 12px; width: 18px; height: 18px; text-align: center; line-height: 18px;
      cursor: pointer; user-select: none;
    }
    .handle {
      position: absolute; width: 12px; height: 12px; background: #063;
      border: 1px solid #fff;
    }
    .h-nw { top: -6px; left: -6px; cursor: nwse-resize; }
    .h-ne { top: -6px; right: -6px; cursor: nesw-resize; }
    .h-sw { bottom: -6px; left: -6px; cursor: nesw-resize; }
    .h-se { bottom: -6px; right: -6px; cursor: nwse-resize; }
    .suggestion {
      position: absolute; border: 3px dashed rgb(200,0,0); box-sizing: border-box;
      cursor: pointer; background: rgba(200,0,0,0.05);
    }
    .suggestion .label {
      position: absolute; top: -20px; left: -3px; background: rgb(200,0,0); color: #fff;
      font-size: 11px; padding: 1px 5px; white-space: nowrap; user-select: none;
    }
    #status { font-size: 13px; color: #060; height: 18px; }
    #dirtyIndicator { font-size: 13px; color: #b30; font-weight: bold; margin-left: 14px; }
    #saveBtn.dirty { background: #ffe0b3; border-color: #c60; font-weight: bold; }
    .colcell { cursor: pointer; }
    .colcell:hover { background: #eef; }
    .override-mark { color: #06c; font-size: 11px; }
  </style>
</head>
<body>
  <div id="imgpane">
    <div class="nav">
      __VOLUME_SELECTOR__
      <a href="/page/__PREV__" onclick="return guardNav()">&laquo; Prev</a>
      <input id="jumpbox" type="number" value="__N__" min="1" max="__TOTAL__" style="width:70px">
      / __TOTAL__
      <button onclick="jumpTo()">Go</button>
      <a href="/page/__NEXT__" onclick="return guardNav()">Next &raquo;</a>
      <a href="/titles" onclick="return guardNav()" style="margin-left:16px;">📋 All saved titles</a>
      <span id="dirtyIndicator"></span>
    </div>
    <div id="controls">
      <button id="addBtn" onclick="toggleAddMode()">+ Add box</button>
      <button id="labelsBtn" onclick="toggleLabels()">👁 Hide labels</button>
      <button onclick="rotatePage({delta:-90})">⟲ 90°</button>
      <button onclick="rotatePage({delta:90})">⟳ 90°</button>
      <input id="customAngle" type="number" step="0.5" style="width:65px" title="Absolute angle in degrees">
      <button onclick="applyCustomAngle()">Set angle</button>
      <button onclick="rotatePage({angle:0})">↩ Reset rotation</button>
      <span id="rotationInfo" style="margin-left:6px; color:#555; font-size:13px;"></span>
      <button id="saveBtn" onclick="saveBoxes()" style="margin-left:16px;">💾 Save page</button>
      <button onclick="resetAuto()">↺ Reset to auto-detected</button>
      <span id="status"></span>
    </div>
    <div class="legend" style="margin:8px 0;">
      <span class="g"></span> editable headword &nbsp;&nbsp;
      <span style="border:2px dashed rgb(200,0,0); display:inline-block; width:14px; height:14px; vertical-align:middle;"></span>
      rejected suggestion (click to promote)
    </div>
    <div id="stage">
      <img id="pageimg" src="/image_raw/__N__">
    </div>
  </div>
  <div id="sidepane">
    <h2>Page __N__ / __TOTAL__</h2>
    <p><b>Running header:</b> <span id="headerinfo">loading…</span></p>
    <p><b>Source:</b> <span id="sourceinfo"></span></p>
    <h3>Boxes on this page</h3>
    <table id="boxtable"><tr><th>Title</th><th>Start #</th><th>End #</th></tr></table>
    <p style="color:#666; font-size:12px;">
      Drag a box to move it, drag a corner handle to resize, double-click to rename,
      click the × to delete. Click "+ Add box" then drag on the page to draw a new one
      around anything the algorithm missed entirely. Don't forget "Save page".
    </p>
  </div>

<script>
const PAGE_NUM = __N__;
const CURRENT_VOLUME = "__VOLUME__";
let scale = 1, origW = 0, origH = 0;
let headerLeftNum = null, headerRightNum = null;
let boxes = [];       // {id, title, bbox:[x0,y0,x1,y1]} in ORIGINAL pixel coords
let nextIdCounter = 1;
let addMode = false;
let dirty = false;
let labelsHidden = false;
let handledSuggestionIds = new Set();  // promoted this session, whether or not later deleted

function setDirty(v) {
  dirty = v;
  document.getElementById('dirtyIndicator').textContent = v ? '● Unsaved changes' : '';
  document.getElementById('saveBtn').classList.toggle('dirty', v);
}

function guardNav() {
  if (!dirty) return true;
  return confirm('You have unsaved changes on this page. Leave without saving them?');
}

function switchVolume(sel) {
  if (!guardNav()) { sel.value = CURRENT_VOLUME; return; }
  location.href = '/switch_volume/' + encodeURIComponent(sel.value);
}

window.addEventListener('beforeunload', (e) => {
  if (dirty) { e.preventDefault(); e.returnValue = ''; }
});

function jumpTo() {
  if (!guardNav()) return;
  const v = document.getElementById('jumpbox').value;
  window.location.href = '/page/' + v;
}

function setStatus(msg, ms) {
  const el = document.getElementById('status');
  el.textContent = msg;
  if (ms) setTimeout(() => { if (el.textContent === msg) el.textContent = ''; }, ms);
}

function toOrig(px) { return px / scale; }
function toDisp(px) { return px * scale; }

async function loadPage() {
  const res = await fetch('/api/page_data/' + PAGE_NUM);
  const data = await res.json();
  scale = data.scale; origW = data.orig_width; origH = data.orig_height;
  headerLeftNum = data.header_left_num; headerRightNum = data.header_right_num;
  renderColumnLabels(headerLeftNum, headerRightNum, data.header_numbers_guessed, data.header_is_override);
  document.getElementById('headerinfo').textContent = data.header;
  document.getElementById('sourceinfo').textContent =
    data.source === 'corrections' ? 'your saved edits' : 'auto-detected (unsaved)';
  document.getElementById('rotationInfo').textContent =
    data.rotation ? ('rotated ' + data.rotation + '°') : '';
  document.getElementById('customAngle').value = data.rotation;
  boxes = data.boxes.map(b => ({
    id: b.id, title: b.title, bbox: b.bbox.slice(),
    column_override: b.column_override ?? null,
    column_end_override: b.column_end_override ?? null,
  }));
  handledSuggestionIds = new Set();
  renderSuggestions(data.suggestions);
  renderBoxes();
  setDirty(false);
}

function renderSuggestions(suggestions) {
  const stage = document.getElementById('stage');
  stage.querySelectorAll('.suggestion').forEach(e => e.remove());
  suggestions.forEach(s => {
    const div = document.createElement('div');
    div.className = 'suggestion';
    positionEl(div, s.bbox);
    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = s.text + ' (' + s.reason + ')';
    div.appendChild(label);
    div.onclick = () => {
      boxes.push({id: 'm' + (nextIdCounter++), title: s.text, bbox: s.bbox.slice(), column_override: null, column_end_override: null});
      handledSuggestionIds.add(s.id);
      div.remove();
      setDirty(true);
      renderBoxes();
    };
    stage.appendChild(div);
  });
}

function positionEl(el, bbox) {
  const [x0, y0, x1, y1] = bbox;
  el.style.left = toDisp(x0) + 'px';
  el.style.top = toDisp(y0) + 'px';
  el.style.width = (toDisp(x1) - toDisp(x0)) + 'px';
  el.style.height = (toDisp(y1) - toDisp(y0)) + 'px';
}

function renderBoxes() {
  const stage = document.getElementById('stage');
  stage.querySelectorAll('.box').forEach(e => e.remove());
  boxes.forEach(b => stage.appendChild(makeBoxEl(b)));
  renderTable();
}

function renderColumnLabels(leftNum, rightNum, guessed, isOverride) {
  const stage = document.getElementById('stage');
  stage.querySelectorAll('.col-num-label').forEach(e => e.remove());
  const cls = isOverride ? 'override' : (guessed ? 'guessed' : 'parsed');
  const tip = isOverride
    ? 'Manually set. Click to change, or clear to go back to auto-detection.'
    : (guessed
        ? 'Extrapolated from an earlier page, not read directly. Click to correct it.'
        : "Read from this page's own header. Click to override it.");
  if (leftNum !== null && leftNum !== undefined) {
    const l = document.createElement('div');
    l.className = 'col-num-label left ' + cls;
    l.textContent = leftNum;
    l.title = tip;
    l.onclick = () => editHeaderNumbers(leftNum, rightNum, isOverride);
    stage.appendChild(l);
  }
  if (rightNum !== null && rightNum !== undefined) {
    const r = document.createElement('div');
    r.className = 'col-num-label right ' + cls;
    r.textContent = rightNum;
    r.title = tip;
    r.onclick = () => editHeaderNumbers(leftNum, rightNum, isOverride);
    stage.appendChild(r);
  }
}

async function editHeaderNumbers(currentLeft, currentRight, isOverride) {
  const input = prompt(
    "This page's printed column numbers, as \"left/right\" (e.g. \"45/46\").\n" +
    'Leave blank to clear a manual override and go back to auto-detection.',
    (currentLeft ?? '') + '/' + (currentRight ?? '')
  );
  if (input === null) return; // cancelled

  if (input.trim() === '') {
    if (!isOverride) return; // nothing to clear
    const res = await fetch('/api/header_override/' + PAGE_NUM + '/clear', {method: 'POST'});
    const data = await res.json();
    setStatus('Cleared manual column numbers (now ' +
      (data.left !== null ? data.left + '/' + data.right : 'unknown') + ')', 3000);
    await loadPage();
    return;
  }

  const parts = input.split('/').map(s => s.trim());
  if (parts.length !== 2 || isNaN(parseInt(parts[0], 10)) || isNaN(parseInt(parts[1], 10))) {
    alert('Enter two numbers separated by a slash, e.g. "45/46".');
    return;
  }
  const left = parseInt(parts[0], 10), right = parseInt(parts[1], 10);
  await fetch('/api/header_override/' + PAGE_NUM, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({left, right})
  });
  setStatus('Set page columns to ' + left + '/' + right, 3000);
  await loadPage();
}

function columnNumFor(bbox) {
  const isLeft = bbox[0] < origW / 2;
  const num = isLeft ? headerLeftNum : headerRightNum;
  return num !== null && num !== undefined ? num : '?';
}

function renderTable() {
  const table = document.getElementById('boxtable');
  let rows = '<tr><th>Title</th><th>Start #</th><th>End #</th></tr>';
  boxes.forEach(b => {
    const overridden = b.column_override !== null && b.column_override !== undefined;
    const startVal = overridden ? b.column_override : columnNumFor(b.bbox);
    const hasEnd = b.column_end_override !== null && b.column_end_override !== undefined;
    const endVal = hasEnd ? b.column_end_override : '';
    rows += '<tr><td>' + escapeHtml(b.title) + '</td>' +
            '<td class="colcell" onclick="editColumnOverride(\'' + b.id + '\')" title="Click to set/clear manually">' +
            startVal + (overridden ? ' <span class="override-mark">✎</span>' : '') + '</td>' +
            '<td class="colcell" onclick="editColumnEndOverride(\'' + b.id + '\')" title="Click to set/clear manually">' +
            endVal + '</td></tr>';
  });
  table.innerHTML = rows;
}

function editColumnEndOverride(boxId) {
  const b = boxes.find(x => String(x.id) === String(boxId));
  if (!b) return;
  const current = (b.column_end_override !== null && b.column_end_override !== undefined) ? b.column_end_override : '';
  const input = prompt(
    'Printed column/page number where "' + b.title + '" ENDS (leave blank if it fits in one column):',
    current
  );
  if (input === null) return; // cancelled
  if (input.trim() === '') {
    b.column_end_override = null;
  } else {
    const num = parseInt(input.trim(), 10);
    if (isNaN(num)) { alert('Enter a whole number, or leave blank to clear it.'); return; }
    b.column_end_override = num;
  }
  setDirty(true);
  renderTable();
}

function editColumnOverride(boxId) {
  const b = boxes.find(x => String(x.id) === String(boxId));
  if (!b) return;
  const current = (b.column_override !== null && b.column_override !== undefined)
    ? b.column_override : columnNumFor(b.bbox);
  const input = prompt(
    'Printed column/page number for "' + b.title + '".\n' +
    'Leave blank to go back to auto-detecting it from the page header.',
    current
  );
  if (input === null) return; // cancelled
  if (input.trim() === '') {
    b.column_override = null;
  } else {
    const num = parseInt(input.trim(), 10);
    if (isNaN(num)) { alert('Enter a whole number, or leave blank to clear the override.'); return; }
    b.column_override = num;
  }
  setDirty(true);
  renderTable();
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function makeBoxEl(b) {
  const div = document.createElement('div');
  div.className = 'box' + (labelsHidden ? ' labels-hidden' : '');
  positionEl(div, b.bbox);

  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = b.title;
  div.appendChild(label);

  const del = document.createElement('div');
  del.className = 'del';
  del.textContent = '×';
  del.onclick = (ev) => { ev.stopPropagation(); boxes = boxes.filter(x => x.id !== b.id); setDirty(true); renderBoxes(); };
  div.appendChild(del);

  ['nw','ne','sw','se'].forEach(corner => {
    const h = document.createElement('div');
    h.className = 'handle h-' + corner;
    h.onmousedown = (ev) => startResize(ev, b, corner);
    div.appendChild(h);
  });

  div.ondblclick = (ev) => {
    ev.stopPropagation();
    const t = prompt('Title:', b.title);
    if (t !== null && t.trim() !== '') { b.title = t.trim(); label.textContent = b.title; setDirty(true); renderTable(); }
  };

  div.onmousedown = (ev) => {
    if (ev.target !== div) return; // let handles/label/del handle their own
    startMove(ev, b);
  };

  return div;
}

function startMove(ev, b) {
  ev.preventDefault();
  const startX = ev.clientX, startY = ev.clientY;
  const [ox0, oy0, ox1, oy1] = b.bbox;
  let moved = false;
  function onMove(e2) {
    const dx = toOrig(e2.clientX - startX), dy = toOrig(e2.clientY - startY);
    b.bbox = [ox0 + dx, oy0 + dy, ox1 + dx, oy1 + dy];
    moved = true;
    renderBoxes();
  }
  function onUp() {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    if (moved) { setDirty(true); reparseBox(b); }
  }
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

function startResize(ev, b, corner) {
  ev.preventDefault(); ev.stopPropagation();
  const startX = ev.clientX, startY = ev.clientY;
  const [ox0, oy0, ox1, oy1] = b.bbox;
  function onMove(e2) {
    const dx = toOrig(e2.clientX - startX), dy = toOrig(e2.clientY - startY);
    let [x0,y0,x1,y1] = [ox0,oy0,ox1,oy1];
    if (corner.includes('n')) y0 = oy0 + dy; else y1 = oy1 + dy;
    if (corner.includes('w')) x0 = ox0 + dx; else x1 = ox1 + dx;
    if (x1 - x0 > 15 && y1 - y0 > 15) b.bbox = [x0,y0,x1,y1];
    renderBoxes();
  }
  function onUp() {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    setDirty(true);
    reparseBox(b);
  }
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

async function reparseBox(b) {
  setStatus('Re-reading text…');
  try {
    const res = await fetch('/api/ocr_region/' + PAGE_NUM, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({bbox: b.bbox})
    });
    const data = await res.json();
    if (data.text && data.text.trim() !== '') {
      b.title = data.text.trim();
      renderBoxes();
    }
    setStatus('', 0);
  } catch (e) {
    setStatus('Re-read failed (see console)', 3000);
    console.error(e);
  }
}

function toggleAddMode() {
  addMode = !addMode;
  document.getElementById('addBtn').style.background = addMode ? '#cfc' : '';
  document.getElementById('stage').style.cursor = addMode ? 'crosshair' : '';
}

function toggleLabels() {
  labelsHidden = !labelsHidden;
  document.getElementById('labelsBtn').textContent = labelsHidden ? '👁 Show labels' : '👁 Hide labels';
  document.getElementById('labelsBtn').style.background = labelsHidden ? '#eee' : '';
  renderBoxes();
}

let drawing = null;
document.addEventListener('DOMContentLoaded', () => {
  const stage = document.getElementById('stage');
  stage.addEventListener('mousedown', (ev) => {
    if (!addMode || ev.target.closest('.box')) return;
    const rect = stage.getBoundingClientRect();
    const startX = ev.clientX - rect.left, startY = ev.clientY - rect.top;
    const div = document.createElement('div');
    div.className = 'box';
    div.style.left = startX + 'px'; div.style.top = startY + 'px';
    div.style.width = '0px'; div.style.height = '0px';
    stage.appendChild(div);
    drawing = {div, startX, startY};

    function onMove(e2) {
      const x = e2.clientX - rect.left, y = e2.clientY - rect.top;
      const l = Math.min(x, startX), t = Math.min(y, startY);
      const w = Math.abs(x - startX), h = Math.abs(y - startY);
      div.style.left = l + 'px'; div.style.top = t + 'px';
      div.style.width = w + 'px'; div.style.height = h + 'px';
    }
    function onUp(e2) {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      const x = e2.clientX - rect.left, y = e2.clientY - rect.top;
      const l = Math.min(x, startX), t = Math.min(y, startY);
      const w = Math.abs(x - startX), h = Math.abs(y - startY);
      div.remove();
      drawing = null;
      if (w < 8 || h < 8) return; // too small, ignore accidental click
      const bbox = [toOrig(l), toOrig(t), toOrig(l + w), toOrig(t + h)];
      finishNewBox(bbox);
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });

  loadPage();
});

async function finishNewBox(bbox) {
  setStatus('Reading text in new box…');
  let suggested = '';
  try {
    const res = await fetch('/api/ocr_region/' + PAGE_NUM, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({bbox})
    });
    const data = await res.json();
    suggested = data.text || '';
  } catch (e) {
    console.error(e);
  }
  setStatus('', 0);
  const title = prompt('Title for new entry (auto-read from image, edit if wrong):', suggested);
  if (title === null || title.trim() === '') { toggleAddMode(); return; }
  boxes.push({id: 'm' + (nextIdCounter++), title: title.trim(), bbox, column_override: null, column_end_override: null});
  setDirty(true);
  renderBoxes();
  toggleAddMode();
}

async function saveBoxes() {
  const res = await fetch('/api/save/' + PAGE_NUM, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({boxes, dismissed_suggestions: Array.from(handledSuggestionIds)})
  });
  const data = await res.json();
  setDirty(false);
  setStatus('Saved ' + data.count + ' boxes ✓', 2500);
}

async function resetAuto() {
  if (!confirm('Discard your edits for this page and reload auto-detected results?')) return;
  await fetch('/api/reset/' + PAGE_NUM, {method: 'POST'});
  loadPage();
  setStatus('Reset to auto-detected', 2000);
}

async function rotatePage(payload) {
  if (!confirm('Rotating this page will re-run OCR under the new orientation and clear any ' +
               'saved/auto-detected boxes for it (their positions would no longer line up). Continue?')) {
    return;
  }
  setStatus('Rotating and re-reading page…');
  const res = await fetch('/api/rotate/' + PAGE_NUM, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  // cache-bust the image since the URL itself doesn't change
  document.getElementById('pageimg').src = '/image_raw/' + PAGE_NUM + '?t=' + Date.now();
  await loadPage();
  setStatus('Rotated to ' + data.angle + '°', 3000);
}

function applyCustomAngle() {
  const v = parseFloat(document.getElementById('customAngle').value);
  if (isNaN(v)) { alert('Enter a valid angle in degrees.'); return; }
  rotatePage({angle: v});
}
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("djvu", nargs="+", help="Path(s) to one or more .djvu volumes")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--start-page", type=int, default=1)
    args = ap.parse_args()

    for p in args.djvu:
        path = Path(p)
        if not path.exists():
            sys.exit(f"File not found: {p}")
        stem = path.stem
        if stem in VOLUMES:
            sys.exit(
                f"Two volumes both resolve to the same name '{stem}' "
                f"({VOLUMES[stem]} and {p}) -- rename one of the files, since "
                f"the database and cache key everything by filename stem."
            )
        VOLUMES[stem] = str(path)

    print(f"Loaded {len(VOLUMES)} volume(s): {', '.join(sorted(VOLUMES))}", file=sys.stderr)

    app.config["START_PAGE"] = args.start_page
    db.backfill_saved_pages()  # migrate any pre-existing database
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
