#!/usr/bin/env python3
"""
Extract expert Tursiops truncatus detections from Harrapatu annotation Excel files
into a single unified CSV.

Two Excel formats are supported:
  FORMAT A (segmented, files L1-L3): each hydrophone block has SegStart_s / SegEnd_s
           columns giving the contour-detected area within the 5-min recording.
  FORMAT B (5-min interval, e.g. L8): no segment columns; the annotation applies to
           the whole 5-minute recording.

Each file holds TWO side-by-side hydrophone blocks (recorded simultaneously).
A valid detection requires "SI" in the T. truncatus column (case/space tolerant).
Any other label (NO, OJO RUIDO, ...) is NOT a detection.

Merge logic per (paired) row:
  - detection on only one hydrophone  -> take that one
  - detection on both hydrophones:
        * Format A (has timestamps)    -> take BOTH (segments may differ in time)
        * Format B (no timestamps)     -> take ONE (same 5-min interval)
"""

import argparse
import re
from pathlib import Path

import pandas as pd

VALID_LABEL = "SI"


def norm_label(v):
    if pd.isna(v):
        return ""
    return str(v).strip().upper()


def find_col(cols, *needles):
    """Find first column whose lowercased name contains all needles."""
    for c in cols:
        lc = str(c).lower()
        if all(n in lc for n in needles):
            return c
    return None


def split_blocks(df):
    """Split columns into hydrophone blocks.

    Each block starts at a 'File' column. Most files have two side-by-side blocks;
    some may have only one. We locate every column whose name starts with 'File'
    and slice between them, so this works regardless of pandas' '.1' suffixing.
    """
    cols = list(df.columns)
    starts = [i for i, c in enumerate(cols) if str(c).lower().startswith("file")]
    if not starts:
        return [cols]
    blocks = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(cols)
        blocks.append(cols[s:e])
    return blocks


def parse_block(df, cols, hydrophone_tag):
    """Return a normalized dataframe with one row per recording for a single block."""
    file_c = find_col(cols, "file")
    date_c = find_col(cols, "date")
    trunc_c = find_col(cols, "truncatus")
    seg_start_c = find_col(cols, "segstart")
    seg_end_c = find_col(cols, "segend")

    empty = pd.DataFrame(columns=["wav_name", "date", "label",
                                  "seg_start_s", "seg_end_s",
                                  "hydrophone", "has_segment"])
    if file_c is None or trunc_c is None:
        return empty

    out = pd.DataFrame()
    out["wav_name"] = df[file_c]
    out["date"] = df[date_c] if date_c else pd.NaT
    out["label"] = df[trunc_c].map(norm_label)
    out["seg_start_s"] = df[seg_start_c] if seg_start_c else pd.NA
    out["seg_end_s"] = df[seg_end_c] if seg_end_c else pd.NA
    out["hydrophone"] = hydrophone_tag
    out["has_segment"] = seg_start_c is not None

    # drop fully empty rows (no file name)
    out = out[out["wav_name"].notna()].copy()
    return out


def recorder_from_wav(name):
    """Recorder id is the part before the first dot: '9488.251006163000.wav' -> '9488'."""
    if pd.isna(name):
        return None
    m = re.match(r"^([^.]+)\.", str(name))
    return m.group(1) if m else None


def interval_key_from_wav(name):
    """Timestamp token that identifies the simultaneous 5-min interval.

    '9488.251006163000.wav' -> '251006163000' (YYMMDDHHMMSS).
    Two hydrophones recording the same interval share this token.
    """
    if pd.isna(name):
        return None
    parts = str(name).split(".")
    if len(parts) >= 2:
        return parts[1]
    return None


def process_file(path):
    raw = pd.read_excel(path, header=0)
    blocks = split_blocks(raw)

    parsed = [parse_block(raw, b, f"H{i+1}") for i, b in enumerate(blocks)]
    # pad to at least two so downstream logic is uniform
    while len(parsed) < 2:
        parsed.append(parse_block(raw, [], f"H{len(parsed)+1}"))

    left, right = parsed[0], parsed[1]

    has_seg = bool(left["has_segment"].any()) if len(left) else \
              bool(right["has_segment"].any()) if len(right) else False

    # keep only valid "SI" detections
    left_det = left[left["label"] == VALID_LABEL].copy()
    right_det = right[right["label"] == VALID_LABEL].copy()

    left_det["interval"] = left_det["wav_name"].map(interval_key_from_wav)
    right_det["interval"] = right_det["wav_name"].map(interval_key_from_wav)

    rows = []

    if has_seg:
        # Format A: timestamps present -> take both hydrophones' detections as-is.
        rows.append(left_det)
        rows.append(right_det)
    else:
        # Format B: no timestamps. For intervals detected on both hydrophones,
        # keep only one (the left/H1) since it's the same 5-min window.
        left_intervals = set(left_det["interval"])
        # all left detections
        rows.append(left_det)
        # right detections only where that interval was NOT already on the left
        right_only = right_det[~right_det["interval"].isin(left_intervals)]
        rows.append(right_only)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result["source_file"] = Path(path).name
    result["format"] = "segmented" if has_seg else "interval_5min"
    return result


def collect_excel_files(inputs):
    """Expand inputs (files and/or folders) into a sorted list of .xlsx paths."""
    files = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.xlsx")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"Warning: '{inp}' not found, skipping.")
    # drop temporary Excel lock files (~$...) and dedupe
    files = [f for f in files if not f.name.startswith("~$")]
    seen, unique = set(), []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(f)
    return unique


def main():
    ap = argparse.ArgumentParser(description="Extract expert T. truncatus detections to CSV.")
    ap.add_argument("inputs", nargs="+",
                    help="A folder of annotation .xlsx files, and/or individual .xlsx files.")
    ap.add_argument("-o", "--output", default="expert_detections.csv", help="Output CSV path.")
    args = ap.parse_args()

    excel_files = collect_excel_files(args.inputs)
    if not excel_files:
        print("No .xlsx files found.")
        return

    all_rows = []
    for f in excel_files:
        df = process_file(f)
        n = len(df)
        fmt = df["format"].iloc[0] if n else "?"
        print(f"{Path(f).name}: {n} expert detections ({fmt})")
        all_rows.append(df)

    final = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # tidy columns
    final["recorder"] = final["wav_name"].map(recorder_from_wav)
    final["interval"] = final["wav_name"].map(interval_key_from_wav)
    cols = ["source_file", "format", "wav_name", "recorder", "interval", "date",
            "hydrophone", "label", "seg_start_s", "seg_end_s"]
    final = final[[c for c in cols if c in final.columns]]

    final.to_csv(args.output, index=False)
    print(f"\nWrote {len(final)} rows -> {args.output}")


if __name__ == "__main__":
    main()