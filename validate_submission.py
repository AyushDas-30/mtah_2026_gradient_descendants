#!/usr/bin/env python3
"""Standalone submission validator for More than a Hack 2027, Phase 1 (ball possession).

Checks:
  1. Exact column names and order (frame, label)
  2. Row count (199,319 data rows)
  3. No NaN / null / empty values
  4. Valid labels: H, A, D, X only
  5. Frame numbering: sequential 0..199318
  6. Graded frames (halves) contain only H/A/D — no X
  7. Non-graded frames contain only X
  8. File size under 10 MB
  9. File uses unix-style line endings (optional warning)
 10. SHA-256 hash reported for reproducibility

Usage:
    python validate_submission.py <path_to_csv>
"""

import hashlib
import os
import sys


# -- Constants from the competition spec -------------------------------------------------------

N_VIDEO_FRAMES = 199319
GRADED_HALVES = ((1, 11185, 90651), (2, 110808, 190197))
VALID_LABELS = {"H", "A", "D", "X"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def validate(path):
    """Run every check. Returns (pass_count, fail_count, warn_count)."""
    passes = 0
    fails = 0
    warns = 0

    def ok(msg):
        nonlocal passes
        passes += 1
        print(f"  [PASS] {msg}")

    def fail(msg):
        nonlocal fails
        fails += 1
        print(f"  [FAIL] {msg}")

    def warn(msg):
        nonlocal warns
        warns += 1
        print(f"  [WARN] {msg}")

    def info(msg):
        print(f"  [INFO] {msg}")

    # --- File existence and size ---
    if not os.path.isfile(path):
        fail(f"file not found: {path}")
        return passes, fails, warns

    file_size = os.path.getsize(path)
    info(f"file size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    if file_size <= MAX_FILE_SIZE_BYTES:
        ok(f"file size under 10 MB ({file_size / (1024*1024):.2f} MB)")
    else:
        fail(f"file size exceeds 10 MB ({file_size / (1024*1024):.2f} MB)")

    # --- Read raw bytes for line-ending check, then decode ---
    raw = open(path, "rb").read()
    has_crlf = b"\r\n" in raw
    has_bare_lf = b"\n" in raw.replace(b"\r\n", b"")
    if has_crlf and not has_bare_lf:
        warn("file uses CRLF line endings (Windows); LF is safer for cross-platform")
    elif has_crlf and has_bare_lf:
        warn("file has mixed line endings")
    else:
        ok("file uses LF line endings")

    text = raw.decode("utf-8")
    lines = text.splitlines()

    # --- SHA-256 ---
    sha = hashlib.sha256(raw).hexdigest()
    info(f"SHA-256: {sha}")
    info(f"SHA-256 (first 16 hex): {sha[:16]}")

    # --- Header ---
    if not lines:
        fail("file is empty")
        return passes, fails, warns

    header_raw = lines[0]
    header_parts = [c.strip().lower() for c in header_raw.split(",")]
    if header_parts == ["frame", "label"]:
        ok(f"header columns correct: {header_parts}")
    else:
        fail(f"header columns wrong: {header_parts} — expected ['frame', 'label']")

    # --- Row count ---
    n_data = len(lines) - 1
    # Handle possible trailing newline creating an empty last element
    if lines and lines[-1] == "":
        n_data -= 1
    info(f"data rows (excluding header): {n_data:,}")
    if n_data == N_VIDEO_FRAMES:
        ok(f"row count is exactly {N_VIDEO_FRAMES:,}")
    else:
        fail(f"row count is {n_data:,}, expected {N_VIDEO_FRAMES:,}")
        # Still continue checks on whatever we have

    # --- Parse rows, check frame numbering, labels, NaN/null ---
    graded_set = set()
    for half_num, start, end in GRADED_HALVES:
        for f in range(start, end + 1):
            graded_set.add(f)

    frames_ok = True
    labels_ok = True
    nan_null_ok = True
    graded_ok = True       # graded frames must have H/A/D
    non_graded_ok = True   # non-graded frames must have X
    label_counts = {"H": 0, "A": 0, "D": 0, "X": 0}
    bad_labels = set()
    n_graded_x = 0
    n_non_graded_had = 0
    first_frame_error = None
    first_label_error = None
    first_nan_error = None
    first_graded_error = None
    first_non_graded_error = None

    for i, line in enumerate(lines[1:]):
        if not line.strip():
            continue  # skip trailing empty line
        parts = line.split(",")
        if len(parts) != 2:
            if nan_null_ok:
                nan_null_ok = False
                first_nan_error = f"row {i}: expected 2 columns, got {len(parts)}"
            continue

        frame_str, label_str = parts[0].strip(), parts[1].strip().upper()

        # Check for empty / null-like values
        if not frame_str or not label_str or label_str.lower() in ("nan", "null", "none", ""):
            if nan_null_ok:
                nan_null_ok = False
                first_nan_error = f"row {i}: empty or null-like value (frame={frame_str!r}, label={label_str!r})"

        # Frame numbering
        try:
            frame_int = int(frame_str)
            if frame_int != i:
                if frames_ok:
                    frames_ok = False
                    first_frame_error = f"row {i}: frame={frame_int}, expected {i}"
        except ValueError:
            if frames_ok:
                frames_ok = False
                first_frame_error = f"row {i}: non-integer frame={frame_str!r}"

        # Label validity
        if label_str in VALID_LABELS:
            label_counts[label_str] += 1
        else:
            bad_labels.add(label_str)
            if labels_ok:
                labels_ok = False
                first_label_error = f"row {i}: invalid label={label_str!r}"

        # Graded / non-graded checks
        if label_str in VALID_LABELS:
            if i in graded_set:
                if label_str == "X":
                    n_graded_x += 1
                    if graded_ok:
                        graded_ok = False
                        first_graded_error = f"row {i}: graded frame has label X"
            else:
                if label_str != "X":
                    n_non_graded_had += 1
                    if non_graded_ok:
                        non_graded_ok = False
                        first_non_graded_error = f"row {i}: non-graded frame has label {label_str}"

    # --- Report results ---
    if frames_ok:
        ok("frame numbering is sequential 0..{:,}".format(n_data - 1))
    else:
        fail(f"frame numbering error — first: {first_frame_error}")

    if labels_ok:
        ok("all labels are valid (H, A, D, X)")
    else:
        fail(f"invalid labels found: {bad_labels} — first: {first_label_error}")

    if nan_null_ok:
        ok("no NaN, null, or empty values")
    else:
        fail(f"NaN/null/empty values found — first: {first_nan_error}")

    if graded_ok:
        ok("all graded frames have H/A/D labels (no X)")
    else:
        fail(f"graded frames with X: {n_graded_x:,} — first: {first_graded_error}")

    if non_graded_ok:
        ok("all non-graded frames have X labels")
    else:
        fail(f"non-graded frames with H/A/D: {n_non_graded_had:,} — first: {first_non_graded_error}")

    # --- Label distribution (info) ---
    info("label distribution:")
    total = sum(label_counts.values())
    for lbl in ("H", "A", "D", "X"):
        c = label_counts[lbl]
        pct = 100 * c / max(1, total)
        info(f"  {lbl}: {c:>7,}  ({pct:5.1f}%)")

    n_graded = sum(1 for f in range(min(n_data, N_VIDEO_FRAMES)) if f in graded_set)
    n_had = label_counts["H"] + label_counts["A"] + label_counts["D"]
    info(f"graded frame slots: {n_graded:,}  |  H+A+D labels: {n_had:,}")

    # --- Dtype check: frame column should be integer-only ---
    info("frame column dtype: integer (checked during parse)")
    info("label column dtype: single uppercase character (checked during parse)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  RESULT: {passes} PASS, {fails} FAIL, {warns} WARN")
    if fails == 0:
        print("  VALID: Submission file passes all checks")
    else:
        print("  INVALID: Submission file has ERRORS -- fix before uploading")
    print(f"{'='*60}")

    return passes, fails, warns


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_csv>")
        sys.exit(2)

    path = sys.argv[1]
    print(f"Validating: {path}\n")
    p, f, w = validate(path)
    sys.exit(1 if f > 0 else 0)
