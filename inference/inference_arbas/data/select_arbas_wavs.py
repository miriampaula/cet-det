"""
select_arbas_wavs.py
--------------------
Scans a hard drive for SoundTrap WAV files belonging to the two ARBAS
deployments and produces:

  1) A CSV list of files matching deployment date ranges (with full timestamp
     info so you can discard by hour later if needed).
  2) A plain text file (one WAV path per line) that can be fed into the
     Perch inference pipeline.

Expected filename format:  <serial>.<YYMMDDHHMMSS>.wav   e.g. 6338.240528160459.wav

Expected folder layout on the drive:
    <root>/
      2024.05.28_ARBAS/
        Soundtrap 6338/
          6338.240528160459.wav
          6338.240528160459.log.xml
          6338.240528160459.sud
          ...
      2024.08.06_ARBAS/
        Soundtrap 6312/
          6312.240806141500.wav
          ...

Usage
-----
    python select_arbas_wavs.py ^
        --drive_root  "D:/IM-23-ARBAS" ^
        --out_dir     "C:/Users/surra/inference-arbas/selected_files"
"""

# -*- coding: utf-8 -*-

import argparse
import csv
import re
from datetime import datetime, date
from pathlib import Path


# Deployment definitions — only ARBAS, only the two relevant Soundtraps
DEPLOYMENTS = [
    {
        "deployment_id":   "ARBAS_2024-05-28",
        "folder_name":     "2024.05.28_ARBAS",
        "serial":          "6338",
        "start_date":      date(2024, 5, 28),
        "end_date":        date(2024, 5, 31),
    },
    {
        "deployment_id":   "ARBAS_2024-08-06",
        "folder_name":     "2024.08.06_ARBAS",
        "serial":          "6312",
        "start_date":      date(2024, 8, 6),
        "end_date":        date(2024, 8, 9),
    },
]


FILENAME_RE = re.compile(r"^(\d+)\.(\d{12})\.wav$", re.IGNORECASE)


def parse_filename(fname: str):
    """Return (serial:str, datetime) or None if filename doesn't match the pattern."""
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    serial, ts = m.group(1), m.group(2)
    try:
        dt = datetime.strptime(ts, "%y%m%d%H%M%S")
    except ValueError:
        return None
    return serial, dt


def parse_args():
    p = argparse.ArgumentParser(description="Select ARBAS WAVs by deployment date range")
    p.add_argument("--drive_root", required=True,
                   help="Path on the drive that contains the 2024.05.28_ARBAS and "
                        "2024.08.06_ARBAS folders.")
    p.add_argument("--out_dir",    required=True,
                   help="Where to write the CSV and the WAV path list.")
    return p.parse_args()


def main():
    args       = parse_args()
    drive_root = Path(args.drive_root)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not drive_root.exists():
        raise FileNotFoundError(f"drive_root does not exist: {drive_root}")

    selected_rows = []
    skipped_rows  = []   # files that were inside the deployment folder but didn't match
    summary       = []

    for dep in DEPLOYMENTS:
        dep_folder = drive_root / dep["folder_name"]
        n_found_total = 0
        n_kept        = 0
        n_skipped     = 0

        if not dep_folder.exists():
            print(f"[WARN] Missing deployment folder: {dep_folder}")
            summary.append((dep["deployment_id"], "MISSING_FOLDER", 0, 0, 0))
            continue

        # Find all WAVs anywhere under the deployment folder
        all_wavs = sorted(dep_folder.rglob("*.wav"))
        n_found_total = len(all_wavs)

        for wav_path in all_wavs:
            parsed = parse_filename(wav_path.name)
            if parsed is None:
                skipped_rows.append({
                    "deployment_id": dep["deployment_id"],
                    "wav_path":      str(wav_path),
                    "reason":        "filename_pattern_no_match",
                })
                n_skipped += 1
                continue

            serial, dt = parsed
            file_date  = dt.date()

            # Check serial
            if serial != dep["serial"]:
                skipped_rows.append({
                    "deployment_id": dep["deployment_id"],
                    "wav_path":      str(wav_path),
                    "reason":        f"serial_mismatch_{serial}_expected_{dep['serial']}",
                })
                n_skipped += 1
                continue

            # Check date range (inclusive)
            if not (dep["start_date"] <= file_date <= dep["end_date"]):
                skipped_rows.append({
                    "deployment_id": dep["deployment_id"],
                    "wav_path":      str(wav_path),
                    "reason":        f"date_out_of_range_{file_date.isoformat()}",
                })
                n_skipped += 1
                continue

            # Keep it
            selected_rows.append({
                "deployment_id": dep["deployment_id"],
                "serial":        serial,
                "wav_path":      str(wav_path),
                "wav_filename":  wav_path.name,
                "date":          file_date.isoformat(),
                "time":          dt.strftime("%H:%M:%S"),
                "datetime":      dt.isoformat(),
            })
            n_kept += 1

        summary.append((dep["deployment_id"], "OK", n_found_total, n_kept, n_skipped))

    # ---- write outputs ---------------------------------------------------
    selected_csv = out_dir / "arbas_selected_wavs.csv"
    skipped_csv  = out_dir / "arbas_skipped_wavs.csv"
    list_txt     = out_dir / "arbas_wav_paths.txt"

    with open(selected_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "deployment_id", "serial", "wav_path", "wav_filename",
            "date", "time", "datetime",
        ])
        writer.writeheader()
        writer.writerows(selected_rows)

    with open(skipped_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["deployment_id", "wav_path", "reason"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    with open(list_txt, "w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(row["wav_path"] + "\n")

    # ---- print summary ---------------------------------------------------
    print()
    print("=" * 70)
    print("SELECTION SUMMARY")
    print("=" * 70)
    for dep_id, status, n_total, n_kept, n_skipped in summary:
        print(f"  {dep_id:<22s} {status:<15s} found={n_total:<4d}  "
              f"kept={n_kept:<4d}  skipped={n_skipped}")
    print("-" * 70)
    print(f"  TOTAL KEPT      : {len(selected_rows)}")
    print(f"  TOTAL SKIPPED   : {len(skipped_rows)}")
    print()
    print(f"  → {selected_csv}")
    print(f"  → {skipped_csv}")
    print(f"  → {list_txt}")
    print("=" * 70)


if __name__ == "__main__":
    main()
