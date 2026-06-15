"""
export_detections.py — Generate data/detections.json for CET·DET from your real CSVs.

Usage:
python export_detections.py \
  --arbas              "/data2/mromaniuc/cet-det/inference/inference_arbas/comparison/arbas_comparison_5sec_segments.csv" \
  --harrapatu          "/data2/mromaniuc/cet-det/inference/inference_harrapatu/comparison/harrapatu_comparison_5sec_segments.csv" \
  --arbas-spec-dir     "/data2/mromaniuc/cet-det/inference/inference_arbas/spectrograms/spectrograms" \
  --harrapatu-spec-dir "/data2/mromaniuc/cet-det/inference/inference_harrapatu/spectrograms/spectrograms" \
  --arbas-spec-url     "spectrograms_arbas" \
  --harrapatu-spec-url "spectrograms_harrapatu" \
  --spec-ext           .png \
  --out                data/detections.json

For opening the html:
  python -m http.server 8080
  open http://localhost:8080/index.html

Spectrogram naming conventions (per 5-second segment):

  ARBAS:
      ARBAS_2024-05-28_6338.240528160459_00.10-00.15.png
      {DATASET}_{DATE}_{WAV_STEM}_{MM.SS_start}-{MM.SS_end}

  HARRAPATU:
      L1_9488.250925084001_00.00-00.05.png   (L* = hydrophone location label)
      6338_6338.251010055958_00.00-00.05.png (6338 rows use recorder id as prefix)
      {LABEL}_{WAV_STEM}_{MM.SS_start}-{MM.SS_end}

  The HARRAPATU L* prefix is NOT reliably available in the CSV (source_file is
  empty for ~93% of rows), so we reverse-look-up the real filename on disk by
  the prefix-independent key '{wav_stem}_{time_range}', which is unique.

  Where MM.SS uses zero-padded minutes and seconds, e.g.:
      offset  10 s  →  00.10-00.15
      offset  65 s  →  01.05-01.10
      offset 295 s  →  04.55-05.00
"""

import argparse, json, os, csv, re
from pathlib import Path
from datetime import datetime, timezone


CODE_MAP = {
    # long forms (raw model output)
    'background':               'bg',
    'Delphinidae_unknown':      'Ambig',
    'Tursiops_truncatus':       'Tt',
    'Balaenoptera_acutorostrata': 'Ba',
    'Balaenoptera_physalus':    'Bp',
    'Delphinus_delphis':        'Dd',
    'Globicephala_melas':       'Gm',
    'Grampus_griseus':          'Gg',
    'Orcinus_orca':             'Oo',
    'Physeter_macrocephalus':   'Pm',
    'Stenella_coeruleoalba':    'Sc',
    'ba':                       'Ba',
    # short / already-normalised forms
    'bg': 'bg', 'Ambig': 'Ambig', 'Tt': 'Tt', 'Ba': 'Ba',
    'Bp': 'Bp', 'Dd': 'Dd', 'Gm': 'Gm', 'Gg': 'Gg',
    'Oo': 'Oo', 'Pm': 'Pm', 'Sc': 'Sc',
    'uncertain': 'bg',
}

SP_COLS = ['Ba', 'Bp', 'Ambig', 'Dd', 'Gm', 'Gg', 'Oo', 'Pm', 'Sc', 'Tt']
EXP_COLS_ARBAS = ['Ambig', 'Bb', 'Dc', 'Dd', 'Gg', 'Gm', 'Lo', 'Oo', 'Pm', 'Sc', 'Tt', 'Zc']

STRATEGIES = ['argmax', 'vec', 'pr', 'consensus3', 'consensus2']

SPECIES_NAMES = {
    'Ba':    'Balaenoptera acutorostrata',
    'Bp':    'Balaenoptera physalus',
    'Ambig': 'Delphinidae (unknown)',
    'Dd':    'Delphinus delphis',
    'Gm':    'Globicephala melas',
    'Gg':    'Grampus griseus',
    'Oo':    'Orcinus orca',
    'Pm':    'Physeter macrocephalus',
    'Sc':    'Stenella coeruleoalba',
    'Tt':    'Tursiops truncatus',
    'bg':    'Background',
}


# ── spectrogram filename logic ─────────────────────────────────────────────────

def _fmt_time(seconds):
    """10 → '00.10',  65 → '01.05',  295 → '04.55'"""
    s = int(seconds)
    return f'{s // 60:02d}.{s % 60:02d}'


def seg_spec_stem(wav_name, offset_s, dataset, source_file=''):
    """
    Return the filename stem (no extension, no directory) for the 5-second
    spectrogram starting at offset_s within wav_name.

    Used for ARBAS (and as a fallback). HARRAPATU now uses disk reverse-lookup
    via seg_spec_key() + build_harrapatu_lookup() instead, because the L* prefix
    is not reliably present in the CSV.
    """
    stem  = Path(wav_name).stem
    start = int(float(offset_s))
    end   = start + 5
    time_range = f'{_fmt_time(start)}-{_fmt_time(end)}'

    if dataset == 'ARBAS':
        m = re.search(r'\d+\.(\d{2})(\d{2})(\d{2})', stem)
        date_str = f'20{m.group(1)}-{m.group(2)}-{m.group(3)}' if m else 'unknown-date'
        return f'ARBAS_{date_str}_{stem}_{time_range}'

    elif dataset == 'HARRAPATU':
        m_label = re.search(r'L(\d+)', source_file)
        prefix  = f'L{m_label.group(1)}' if m_label else stem.split('.')[0]
        return f'{prefix}_{stem}_{time_range}'

    else:
        return f'{stem}_{time_range}'


def seg_spec_key(wav_name, offset_s):
    """
    Prefix-independent key for matching HARRAPATU spectrograms on disk.
    '9488.250925084001' + offset 0  ->  '9488.250925084001_00.00-00.05'
    """
    stem  = Path(wav_name).stem
    start = int(float(offset_s))
    end   = start + 5
    return f'{stem}_{_fmt_time(start)}-{_fmt_time(end)}'


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_ts(wav_name):
    m = re.search(r'(\d{12})', wav_name)
    if not m:
        return None
    s = m.group(1)
    try:
        dt = datetime(2000 + int(s[0:2]), int(s[2:4]), int(s[4:6]),
                      int(s[6:8]), int(s[8:10]), int(s[10:12]), tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def safe_float(value, default=0.0):
    """Convert to float safely; returns default for None/empty/unparseable."""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def top4(row):
    """
    Return sorted list of [species_code, probability] for the top-4 cetacean
    species, plus background as an 11th entry so the UI can show it if needed.

    Format: {'cet': [[code, prob], ...], 'bg': prob}
    """
    probs = [(c, safe_float(row.get('prob_' + c))) for c in SP_COLS]
    probs.sort(key=lambda x: -x[1])
    return {
        'cet': [[c, round(p, 4)] for c, p in probs[:4]],
        'bg':  round(safe_float(row.get('prob_bg')), 4),
    }


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def group_by_wav(rows):
    wavs, wav_map = [], {}
    for r in rows:
        w = r['wav_name']
        if w not in wav_map:
            wav_map[w] = {'wav': w, 'segs': []}
            wavs.append(wav_map[w])
        wav_map[w]['segs'].append(r)
    return wavs


def seg_spec_url(spec_base_url, stem, spec_ext):
    return spec_base_url.rstrip('/') + '/' + stem + spec_ext


def read_strategies(row, suffix):
    """
    Read all 5 strategy values for a given column suffix.
    suffix is one of: 'pred', 'fired', 'outcome'

    Returns dict keyed by strategy short name, e.g.:
      {'argmax': 'bg', 'vec': 'Tt', 'pr': 'bg', 'consensus3': 'bg', 'consensus2': 'Tt'}
    For 'pred' the values are code-mapped species strings.
    For 'fired' the values are booleans.
    For 'outcome' the values are raw strings (TN/FP/TP/FN/…).
    """
    out = {}
    for strat in STRATEGIES:
        col = f'{suffix}_{strat}'
        raw = row.get(col, '')
        if suffix == 'pred':
            out[strat] = CODE_MAP.get(raw, 'bg')
        elif suffix == 'fired':
            out[strat] = raw == 'True'
        else:
            out[strat] = raw
    return out


def build_spec_index(spec_dir, spec_ext):
    """Return a set of stems (no extension) present in spec_dir."""
    if not spec_dir:
        return set()
    try:
        return {
            Path(f).stem
            for f in os.listdir(spec_dir)
            if f.endswith(spec_ext)
        }
    except OSError as e:
        print(f'  Warning: could not list spec dir {spec_dir}: {e}')
        return set()


def build_harrapatu_lookup(spec_dir, spec_ext):
    """
    Build a reverse-lookup index for HARRAPATU spectrograms.

    Maps the prefix-independent key '{wav_stem}_{time_range}' to the full
    filename stem (which carries the correct L* / recorder prefix):

        '9488.250925084001_00.00-00.05'  ->  'L1_9488.250925084001_00.00-00.05'
        '6338.251010055958_00.00-00.05'  ->  '6338_6338.251010055958_00.00-00.05'

    Verified unique: every key maps to exactly one prefix on disk.
    """
    if not spec_dir:
        return {}
    pat = re.compile(r'^[A-Za-z0-9]+_(.+\.\d{12}_\d{2}\.\d{2}-\d{2}\.\d{2})$')
    lookup = {}
    try:
        for f in os.listdir(spec_dir):
            if not f.endswith(spec_ext):
                continue
            full_stem = Path(f).stem
            m = pat.match(full_stem)
            if m:
                lookup[m.group(1)] = full_stem
    except OSError as e:
        print(f'  Warning: could not list spec dir {spec_dir}: {e}')
    return lookup


def build_arbas_lookup(spec_dir, spec_ext):
    """
    Build a reverse-lookup index for ARBAS spectrograms.

    Maps the prefix/date-independent key '{wav_stem}_{time_range}' to the full
    filename stem (which carries the correct ARBAS_{date}_ prefix):

        '6338.240528160459_00.10-00.15'
            -> 'ARBAS_2024-05-28_6338.240528160459_00.10-00.15'

    This avoids re-deriving the date prefix (which can disagree with the wav
    stem for recordings near a day boundary). Verified unique on disk.
    """
    if not spec_dir:
        return {}
    # ARBAS_2024-05-28_6338.240528160459_00.10-00.15
    pat = re.compile(r'^ARBAS_\d{4}-\d{2}-\d{2}_(.+\.\d{12}_\d{2}\.\d{2}-\d{2}\.\d{2})$')
    lookup = {}
    try:
        for f in os.listdir(spec_dir):
            if not f.endswith(spec_ext):
                continue
            full_stem = Path(f).stem
            m = pat.match(full_stem)
            if m:
                lookup[m.group(1)] = full_stem
    except OSError as e:
        print(f'  Warning: could not list spec dir {spec_dir}: {e}')
    return lookup


# ── ARBAS ──────────────────────────────────────────────────────────────────────

def process_arbas(csv_path, spec_dir, spec_ext, spec_base_url):
    rows = read_csv(csv_path)
    wavs = group_by_wav(rows)
    out  = []

    missing_stems  = []   # for debug output
    total_segs     = 0
    found_spec     = 0

    arbas_lookup = build_arbas_lookup(spec_dir, spec_ext)

    for w in wavs:
        r0 = w['segs'][0]
        exp_det  = r0.get('exp_cetacean_detected', 'False') == 'True'
        exp_sp   = r0.get('exp_top_species', 'background')
        exp_flags = {}
        for c in EXP_COLS_ARBAS:
            v = safe_float(r0.get('exp_' + c))
            if v > 0:
                exp_flags[c] = round(v, 3)

        segs_out = []
        mine_pos = False   # True if any strategy fires on any segment
        any_spec = False

        for s in w['segs']:
            total_segs += 1
            pred    = read_strategies(s, 'pred')
            fired   = read_strategies(s, 'fired')
            outcome = read_strategies(s, 'outcome')
            mp      = round(safe_float(s.get('max_cetacean_prob')), 4)
            t4      = top4(s)
            offset  = safe_float(s.get('offset_s', 0))

            key       = seg_spec_key(w['wav'], offset)
            full_stem = arbas_lookup.get(key)
            has_spec  = full_stem is not None
            sp_path   = seg_spec_url(spec_base_url, full_stem, spec_ext) if has_spec else None
            if has_spec:
                any_spec = True
                found_spec += 1
            elif spec_dir and len(missing_stems) < 5:
                missing_stems.append(key)

            segs_out.append({
                'o':       offset,
                'pred':    pred,      # {'argmax': 'bg', 'vec': 'Tt', ...}
                'fired':   fired,     # {'argmax': False, 'vec': True, ...}
                'outcome': outcome,   # {'argmax': 'TN', 'vec': 'TP', ...}
                'mp':      mp,
                't4':      t4,        # {'cet': [[code, prob], ...], 'bg': prob}
                'specPath': sp_path,
            })
            if any(fired.values()):
                mine_pos = True

        out.append({
            'wav':        w['wav'],
            't':          parse_ts(w['wav']),
            'dur':        60,
            'expDet':     exp_det,
            'expSp':      exp_sp,
            'expFlags':   exp_flags,
            'expAnnotated': r0.get('exp_annotated', ''),
            'source':     r0.get('source', ''),
            'minePos':    mine_pos,
            'spec':       any_spec,
            'specPath':   None,   # per-segment paths stored on each seg
            'segs':       segs_out,
        })

    # debug report
    print(f'  ARBAS spec check: {found_spec}/{total_segs} segments have spectrograms')
    if missing_stems:
        print(f'  First {len(missing_stems)} missing stems (check your --arbas-spec-dir):')
        for stem in missing_stems:
            print(f'    {stem}{spec_ext}')

    return out


# ── HARRAPATU ──────────────────────────────────────────────────────────────────

def process_harrapatu(csv_path, spec_dir, spec_ext, spec_base_url):
    rows = read_csv(csv_path)

    # Group by interval, not wav_name — multiple recorders per 5-min bucket
    intervals, ivl_map = [], {}
    for r in rows:
        key = r.get('interval') or r['wav_name']
        if key not in ivl_map:
            ivl_map[key] = {'interval': key, 'segs': []}
            intervals.append(ivl_map[key])
        ivl_map[key]['segs'].append(r)

    out          = []
    missing_stems = []
    total_segs   = 0
    found_spec   = 0

    # Reverse-lookup index: '{wav_stem}_{time_range}' -> full stem with L* prefix
    harrapatu_lookup = build_harrapatu_lookup(spec_dir, spec_ext)

    for ivl in intervals:
        segs = ivl['segs']
        r0   = segs[0]

        exp_det     = any(s.get('exp_cetacean_detected', 'False') == 'True' for s in segs)
        ivl_outcome = r0.get('outcome', '')    # interval-level outcome if present

        segs_out = []
        mine_pos = False
        any_spec = False

        for s in segs:
            total_segs += 1
            pred    = read_strategies(s, 'pred')
            fired   = read_strategies(s, 'fired')
            outcome = read_strategies(s, 'outcome')
            mp      = round(safe_float(s.get('max_cetacean_prob')), 4)
            t4      = top4(s)
            offset  = safe_float(s.get('offset_s', 0))
            wav     = s.get('wav_name', '')

            # Reverse-lookup the real filename (with correct L* prefix) from disk
            key       = seg_spec_key(wav, offset)
            full_stem = harrapatu_lookup.get(key)
            has_spec  = full_stem is not None
            sp_path   = seg_spec_url(spec_base_url, full_stem, spec_ext) if has_spec else None
            if has_spec:
                any_spec = True
                found_spec += 1
            elif spec_dir and len(missing_stems) < 5:
                missing_stems.append(key)

            segs_out.append({
                'o':       offset,
                'wav':     wav,        # which recorder this segment came from
                'pred':    pred,
                'fired':   fired,
                'outcome': outcome,
                'mp':      mp,
                't4':      t4,
                'specPath': sp_path,
            })
            if any(fired.values()):
                mine_pos = True

        out.append({
            'wav':      r0.get('wav_name', ivl['interval']),
            'interval': ivl['interval'],
            't':        parse_ts(ivl['interval']),
            'dur':      300,
            'expDet':   exp_det,
            'expSp':    'Tt' if exp_det else 'no_label',
            'outcome':  ivl_outcome,
            'minePos':  mine_pos,
            'spec':     any_spec,
            'specPath': None,
            'box':      None,
            'segs':     segs_out,
        })

    # debug report
    print(f'  HARRAPATU spec check: {found_spec}/{total_segs} segments have spectrograms')
    if missing_stems:
        print(f'  First {len(missing_stems)} missing keys (check your --harrapatu-spec-dir):')
        for key in missing_stems:
            print(f'    {key}{spec_ext}')

    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Build detections.json for CET·DET')
    ap.add_argument('--arbas',              required=True,  help='ARBAS CSV path')
    ap.add_argument('--harrapatu',          required=True,  help='HARRAPATU CSV path')
    ap.add_argument('--arbas-spec-dir',     default=None,   help='Folder of ARBAS spectrograms')
    ap.add_argument('--harrapatu-spec-dir', default=None,   help='Folder of HARRAPATU spectrograms')
    ap.add_argument('--arbas-spec-url',     default=None,   help='Base URL for ARBAS spectrograms')
    ap.add_argument('--harrapatu-spec-url', default=None,   help='Base URL for HARRAPATU spectrograms')
    ap.add_argument('--spec-ext',           default='.png')
    ap.add_argument('--out',                default='data/detections.json')
    args = ap.parse_args()

    print('Processing ARBAS…')
    arbas_wavs = process_arbas(
        args.arbas, args.arbas_spec_dir, args.spec_ext, args.arbas_spec_url)
    a_spec = sum(1 for w in arbas_wavs if w['spec'])
    print(f'  → {len(arbas_wavs)} minutes  |  {a_spec} WAVs with ≥1 spectrogram')

    print('Processing HARRAPATU…')
    harra_wavs = process_harrapatu(
        args.harrapatu, args.harrapatu_spec_dir, args.spec_ext, args.harrapatu_spec_url)
    h_spec = sum(1 for w in harra_wavs if w['spec'])
    print(f'  → {len(harra_wavs)} intervals  |  {h_spec} intervals with ≥1 spectrogram')

    result = {
        'generated': datetime.now(timezone.utc).isoformat(),
        'species':   SPECIES_NAMES,
        'strategies': STRATEGIES,   # tells the UI which strategy keys exist
        'datasets': {
            'ARBAS': {
                'id':                  'ARBAS',
                'expertGranularity':   60,
                'myGranularity':       5,
                'expertSpeciesNote':   'multi-species',
                'wavs':                arbas_wavs,
            },
            'HARRAPATU': {
                'id':                  'HARRAPATU',
                'expertGranularity':   300,
                'myGranularity':       5,
                'expertSpeciesNote':   'Tursiops truncatus only',
                'wavs':                harra_wavs,
            },
        },
    }

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f)
    size_mb = Path(args.out).stat().st_size / 1_048_576
    print(f'\n✓  Wrote {args.out}  ({size_mb:.1f} MB)')
    print('\nTest locally:')
    print('  python -m http.server 8080')
    print('  open http://localhost:8080/cetdet.html')


if __name__ == '__main__':
    main()