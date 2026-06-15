"""
check_source_file.py — Inspect the source_file column per row for HARRAPATU.

We need to know WHY s.get('source_file') is empty for 9488/9489 rows.
"""

import csv
from pathlib import Path
from collections import Counter

HARRAPATU_CSV = "/data2/mromaniuc/cet-det/inference/inference_harrapatu/comparison/harrapatu_comparison_5sec_segments.csv"

with open(HARRAPATU_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

print("=" * 60)
print("FIELD NAMES (exact, with repr to expose hidden chars):")
print("=" * 60)
for i, fn in enumerate(fieldnames):
    print(f"  col {i:>2}: {fn!r}")

# Find the source_file key exactly as DictReader sees it
src_key = None
for fn in fieldnames:
    if 'source' in fn.lower():
        src_key = fn
        break
print(f"\nDetected source_file key: {src_key!r}")

print("\n" + "=" * 60)
print("source_file value distribution (per ROW, not unique wav):")
print("=" * 60)
src_counter = Counter(r.get(src_key, '<<KEY MISSING>>') for r in rows)
for val, cnt in src_counter.most_common(20):
    print(f"  {cnt:>8}  {val!r}")

print("\n" + "=" * 60)
print("Cross-tab: recorder ID  ×  source_file presence")
print("=" * 60)
by_rec = {}
for r in rows:
    rec = Path(r['wav_name']).stem.split('.')[0]
    src = r.get(src_key, '')
    has_src = bool(src and src.strip())
    by_rec.setdefault(rec, {'with_src': 0, 'no_src': 0})
    if has_src:
        by_rec[rec]['with_src'] += 1
    else:
        by_rec[rec]['no_src'] += 1

for rec, d in sorted(by_rec.items()):
    print(f"  recorder {rec:>6}: with source_file={d['with_src']:>8}  |  empty={d['no_src']:>8}")

print("\n" + "=" * 60)
print("Sample 9488 rows — full source_file + nearby columns:")
print("=" * 60)
shown = 0
for r in rows:
    rec = Path(r['wav_name']).stem.split('.')[0]
    if rec == '9488':
        print(f"  wav={r['wav_name']}")
        print(f"      date={r.get('date','')!r}  format={r.get('format','')!r}  "
              f"hydrophone={r.get('hydrophone','')!r}  source_file={r.get(src_key,'')!r}")
        shown += 1
        if shown >= 5:
            break
