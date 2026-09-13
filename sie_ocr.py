"""
sie_ocr.py — core page analysis for СИЭ (Soviet Historical Encyclopedia)
DjVu volumes: renders a page, OCRs it, and detects bold headwords.

This is the same heuristic pipeline developed iteratively against real
sample pages:
  - ddjvu renders the page (scan-only DjVu, no text layer).
  - Tesseract (rus+eng) OCRs it; rus+eng reads embedded Latin
    transliterations correctly ("(Aureli-anus)") at the cost of
    occasionally swapping in Latin homoglyphs for plain Cyrillic
    letters, which HOMOGLYPH_MAP corrects back.
  - Tesseract's own paragraph grouping (block_num/par_num) lines up
    with dictionary-entry boundaries; hyphen-wrapped words that got
    split into two "paragraphs" are re-merged.
  - A candidate headword is the leading run of uppercase Cyrillic
    word(s) on a paragraph's first line.
  - Candidates are accepted only if BOTH:
      (a) bold: 1px erosion retention ratio clears the page's own
          body-text baseline by a margin (bilevel scans have no font
          metadata, so this stroke-weight proxy stands in for "is this
          actually typeset bold like a real headword").
      (b) the paragraph has enough total text to be a real entry, not
          an isolated bold map/illustration caption.

Unlike the CLI version, this module also returns REJECTED candidates
(matched the headword shape but failed a check) with their failure
reason, and bounding boxes for everything -- so a viewer can render
them for visual QA instead of trusting the heuristic blind.
"""

import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image
from scipy import ndimage

_HOMOGLYPH_LATIN = "ABEKMHOPCTYX"
_HOMOGLYPH_CYRILLIC = "АВЕКМНОРСТУХ"
HOMOGLYPH_MAP = str.maketrans(_HOMOGLYPH_LATIN, _HOMOGLYPH_CYRILLIC)
_WORD_CHARS = "А-ЯЁ" + _HOMOGLYPH_LATIN

HEADWORD_RE = re.compile(
    r'^[«"]?'
    rf'([{_WORD_CHARS}]{{2,}}(?:-[{_WORD_CHARS}]{{2,}})*'
    rf'(?:\s+[{_WORD_CHARS}]{{2,}}(?:-[{_WORD_CHARS}]{{2,}})*){{0,4}})'
    r'(?=[\s,.\u2014\u2013(\u00ab]|$)'
)

HEADER_RE = re.compile(
    r'^\s*(\d{1,4})\s+(.+?)\s*[—\-\u2014\u2013]\s*(.+?)\s+(\d{1,4})\s*$'
)

BOLD_MARGIN = 0.045
BOLD_ABS_FLOOR = 0.79
MIN_PARAGRAPH_CHARS = 30


def get_page_count(djvu_path):
    out = subprocess.run(["djvudump", str(djvu_path)], capture_output=True, text=True)
    m = re.search(r"(\d+)\s+pages\)", out.stdout)
    return int(m.group(1)) if m else out.stdout.count("FORM:DJVU")


def render_page(djvu_path, page_num, out_path, rotation=0):
    """
    Renders the page via ddjvu, then bakes in a clockwise rotation (any
    angle, not just 90-degree turns) if requested, overwriting out_path
    so every caller downstream (OCR, display image, crop regions) sees
    the same already-rotated pixels -- no risk of one part of the
    pipeline using rotated coordinates while another still assumes the
    original scan orientation.

    Exact 90-degree-multiple rotations use nearest-neighbor resampling
    (no blur, since pixels map 1:1). Arbitrary angles (e.g. deskewing a
    slightly tilted scan) use bicubic resampling and expand the canvas,
    filling the newly-exposed corners white to match the scan's paper
    background rather than PIL's default black.
    """
    cmd = ["ddjvu", "-format=tiff", f"-page={page_num}", str(djvu_path), str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    rotation = float(rotation) % 360
    if rotation:
        im = Image.open(out_path).convert("L")
        axis_aligned = rotation % 90 == 0
        resample = Image.NEAREST if axis_aligned else Image.BICUBIC
        im = im.rotate(-rotation, expand=True, resample=resample, fillcolor=255)
        # Explicit compression: the original ddjvu-produced TIFF uses
        # CCITT Group4 (fax) compression, which only supports 1-bit
        # images. Now that we've converted to 8-bit grayscale, saving
        # without overriding this inherited setting fails with
        # "Bits/sample must be 1 for Group 3/4 encoding". tiff_lzw is
        # lossless and works fine for grayscale.
        im.save(out_path, format="TIFF", compression="tiff_lzw")


def retention_ratio(binary_arr, x, y, w, h):
    if w <= 0 or h <= 0:
        return None
    crop = binary_arr[y:y + h, x:x + w]
    ink = crop.sum()
    if ink == 0:
        return None
    eroded = ndimage.binary_erosion(crop, structure=np.ones((2, 2)))
    return eroded.sum() / ink


def page_body_baseline(data, binary, sample_cap=60):
    n = len(data["text"])
    vals = []
    for i in range(n):
        t = data["text"][i].strip()
        if len(t) < 4 or not t.isalpha():
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        r = retention_ratio(binary, x, y, w, h)
        if r is not None:
            vals.append(r)
        if len(vals) >= sample_cap:
            break
    if not vals:
        return 0.75
    vals.sort()
    return vals[len(vals) // 2]


def group_paragraphs(data):
    lines = defaultdict(list)
    order = []
    n = len(data["text"])
    for i in range(n):
        t = data["text"][i]
        if not t.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            order.append(key)
        lines[key].append((data["left"][i], i, t))

    line_info = []
    for key in order:
        words = sorted(lines[key])
        text = " ".join(w[2] for w in words)
        word_indices = [w[1] for w in words]
        line_info.append((key, text, word_indices))

    paragraphs = defaultdict(list)
    para_order = []
    for (block, par, _line), text, word_idx in line_info:
        pkey = (block, par)
        if pkey not in paragraphs:
            para_order.append(pkey)
        paragraphs[pkey].append((text, word_idx))

    result = [(pkey, paragraphs[pkey]) for pkey in para_order]
    return merge_wrapped_paragraphs(result)


def merge_wrapped_paragraphs(paragraphs):
    merged = []
    i = 0
    while i < len(paragraphs):
        pkey, lines = paragraphs[i]
        while lines and lines[-1][0].rstrip().endswith("-") and i + 1 < len(paragraphs):
            i += 1
            _next_pkey, next_lines = paragraphs[i]
            lines = lines + next_lines
        merged.append((pkey, lines))
        i += 1
    return merged


def word_bbox(data, idx):
    x, y, w, h = data["left"][idx], data["top"][idx], data["width"][idx], data["height"][idx]
    return x, y, x + w, y + h


def union_bbox(boxes):
    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[2] for b in boxes)
    ys1 = max(b[3] for b in boxes)
    return xs0, ys0, xs1, ys1


def reconcile_header_numbers(left_num, right_num):
    """
    This encyclopedia's running headers are strictly sequential: the
    left column is always an odd printed-page number and the right
    column is always exactly left+1. Tesseract occasionally misreads a
    single digit in one of the two numbers (e.g. '42' -> '49'), which
    breaks that invariant in an otherwise-correctly-parsed header line.

    Since the invariant is so reliable, use it to self-correct: if the
    two numbers as read don't satisfy right == left+1, trust whichever
    one has the expected parity (left odd / right even) and derive the
    other from it. If parity doesn't clearly point to one side (both or
    neither match expectations -- a rarer, messier misread), default to
    trusting the left number, since it comes first on the line and
    empirically seems to survive OCR a bit more reliably than the
    right-hand number does.

    Returns (left_num, right_num, was_corrected).
    """
    if right_num == left_num + 1:
        return left_num, right_num, False

    left_parity_ok = (left_num % 2 == 1)
    right_parity_ok = (right_num % 2 == 0)

    if left_parity_ok and not right_parity_ok:
        return left_num, left_num + 1, True
    if right_parity_ok and not left_parity_ok:
        return right_num - 1, right_num, True
    # Ambiguous (both or neither look right by parity alone) -- trust left.
    return left_num, left_num + 1, True


def analyze_page(djvu_path, page_num, lang="rus+eng", tmpdir=None, rotation=0):
    """
    Render + OCR + detect for one page. Returns a dict:
      width, height        : page pixel dimensions (post-rotation)
      tiff_path             : path to the rendered TIFF (caller reads pixels)
      header                : parsed running-header dict or None
      accepted              : list of {title, bbox, bold_ratio, column}
      rejected              : list of {text, bbox, bold_ratio, reason}
    bbox is (x0, y0, x1, y1) in the *rendered* (post-rotation) pixel space.
    """
    own_tmpdir = tmpdir is None
    if own_tmpdir:
        tmpdir = tempfile.mkdtemp()
    tiff_path = Path(tmpdir) / f"p{page_num}.tiff"
    render_page(djvu_path, page_num, tiff_path, rotation=rotation)

    im = Image.open(tiff_path)
    gray = np.array(im.convert("L"))
    binary = gray < 128
    height, width = binary.shape

    data = pytesseract.image_to_data(im, lang=lang, output_type=pytesseract.Output.DICT)
    baseline = page_body_baseline(data, binary)
    paragraphs = group_paragraphs(data)

    header_info = None
    accepted = []
    rejected = []

    for idx, (pkey, lines) in enumerate(paragraphs):
        if not lines:
            continue
        first_text, first_word_idx = lines[0]
        first_line = first_text.strip()

        if idx == 0:
            hm = HEADER_RE.match(first_line)
            if hm:
                raw_left, raw_right = int(hm.group(1)), int(hm.group(4))
                left_num, right_num, corrected = reconcile_header_numbers(raw_left, raw_right)
                if corrected:
                    import sys as _sys
                    print(
                        f"[sie_ocr] page {page_num}: header numbers {raw_left}/{raw_right} "
                        f"violate the left-odd/right=left+1 rule; corrected to "
                        f"{left_num}/{right_num}",
                        file=_sys.stderr,
                    )
                header_info = {
                    "left_num": left_num,
                    "first_word": hm.group(2).strip(" «»"),
                    "last_word": hm.group(3).strip(" «»"),
                    "right_num": right_num,
                }
                continue

        m = HEADWORD_RE.match(first_line)
        if not m:
            continue
        raw_headword = m.group(1).strip()
        headword = raw_headword.translate(HOMOGLYPH_MAP)

        n_words_in_match = len(raw_headword.split())
        candidate_word_idx = first_word_idx[:n_words_in_match]
        if not candidate_word_idx:
            continue

        boxes = [word_bbox(data, i) for i in candidate_word_idx]
        bbox = union_bbox(boxes)

        ratios = []
        for i in candidate_word_idx:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            r = retention_ratio(binary, x, y, w, h)
            if r is not None:
                ratios.append(r)
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        threshold = max(baseline + BOLD_MARGIN, BOLD_ABS_FLOOR)
        is_bold = avg_ratio >= threshold

        paragraph_chars = sum(len(t) for t, _ in lines)
        long_enough = paragraph_chars >= MIN_PARAGRAPH_CHARS

        x0 = data["left"][first_word_idx[0]]
        column = "left" if x0 < width / 2 else "right"

        if is_bold and long_enough:
            accepted.append({
                "title": headword,
                "bbox": bbox,
                "bold_ratio": round(avg_ratio, 3),
                "column": column,
            })
        else:
            reason = []
            if not is_bold:
                reason.append(f"not bold ({avg_ratio:.3f} < {threshold:.3f})")
            if not long_enough:
                reason.append(f"paragraph too short ({paragraph_chars} chars)")
            rejected.append({
                "text": headword,
                "bbox": bbox,
                "bold_ratio": round(avg_ratio, 3),
                "reason": "; ".join(reason),
            })

    return {
        "page_num": page_num,
        "width": width,
        "height": height,
        "tiff_path": str(tiff_path),
        "header": header_info,
        "baseline": round(baseline, 3),
        "accepted": accepted,
        "rejected": rejected,
    }
