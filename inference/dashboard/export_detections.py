"""
export_detections.py — Generate data/detections.json for CET·DET from your real CSVs.

Usage:
python export_detections.py \
  --arbas              "/data2/mromaniuc/cet-det/inference/inference_arbas/comparison/arbas_comparison_5s_v3.csv" \
  --harrapatu          "/data2/mromaniuc/cet-det/inference/inference_harrapatu/comparison/harrapatu_comparison_5min_v1_segments.csv" \
  --arbas-spec-dir     "/data2/mromaniuc/cet-det/inference/inference_arbas/spectrograms/spectrograms" \
  --harrapatu-spec-dir "/data2/mromaniuc/cet-det/inference/inference_harrapatu/spectrograms/spectrograms" \
  --arbas-spec-url     "spectrograms_arbas" \
  --harrapatu-spec-url "spectrograms_harrapatu" \
  --spec-ext           .png \
  --out                data/detections.json
  
  
  For opening the html:
  
  python -m http.server 8080
  open http://localhost:8080/cetdet.html 



Spectrogram naming conventions (per 5-second segment):

  ARBAS:
      ARBAS_2024-05-28_6338.240528160459_00.10-00.15.png
      {DATASET}_{DATE}_{WAV_STEM}_{MM.SS_start}-{MM.SS_end}

  HARRAPATU:
      6338_6338.251010055958_00.00-00.05.png
      {STATION}_{WAV_STEM}_{MM.SS_start}-{MM.SS_end}

  Where MM.SS uses zero-padded minutes and seconds, e.g.:
      offset  10 s  →  00.10-00.15
      offset  65 s  →  01.05-01.10
      offset 295 s  →  04.55-05.00

Edit seg_spec_stem() below if your naming differs.
"""

import argparse, json, os, csv, re
from pathlib import Path
from datetime import datetime, timezone



CODE_MAP = {
    # long forms (raw model output)
    'background': 'bg', 'Delphinidae_unknown': 'Ambig',
    'Tursiops_truncatus': 'Tt', 'Balaenoptera_acutorostrata': 'Ba',
    'Balaenoptera_physalus': 'Bp', 'Delphinus_delphis': 'Dd',
    'Globicephala_melas': 'Gm', 'Grampus_griseus': 'Gg',
    'Orcinus_orca': 'Oo', 'Physeter_macrocephalus': 'Pm',
    'Stenella_coeruleoalba': 'Sc', 'ba': 'Ba',
    # short forms (already normalised by notebook)
    'bg': 'bg', 'Ambig': 'Ambig', 'Tt': 'Tt', 'Ba': 'Ba',
    'Bp': 'Bp', 'Dd': 'Dd', 'Gm': 'Gm', 'Gg': 'Gg',
    'Oo': 'Oo', 'Pm': 'Pm', 'Sc': 'Sc',
    'uncertain': 'bg',
}

SP_COLS = ['Ba', 'Bp', 'Ambig', 'Dd', 'Gm', 'Gg', 'Oo', 'Pm', 'Sc', 'Tt']
EXP_COLS_ARBAS = ['Ambig', 'Bb', 'Dc', 'Dd', 'Gg', 'Gm', 'Lo', 'Oo', 'Pm', 'Sc', 'Tt', 'Zc']
SPECIES_NAMES = {
    'Ba': 'Balaenoptera acutorostrata', 'Bp': 'Balaenoptera physalus',
    'Ambig': 'Delphinidae (unknown)',   'Dd': 'Delphinus delphis',
    'Gm': 'Globicephala melas',         'Gg': 'Grampus griseus',
    'Oo': 'Orcinus orca',               'Pm': 'Physeter macrocephalus',
    'Sc': 'Stenella coeruleoalba',      'Tt': 'Tursiops truncatus',
    'bg': 'Background',
}


# ── spectrogram filename logic ─────────────────────────────────────────────────

def _fmt_time(seconds):
    """10 → '00.10',  65 → '01.05',  295 → '04.55'"""
    s = int(seconds)
    return f'{s // 60:02d}.{s % 60:02d}'

def seg_spec_stem(wav_name, offset_s, dataset):
    """
    Return the filename stem (no extension, no directory) for the 5-second
    spectrogram starting at offset_s within wav_name.

    Edit this function if your naming convention differs.
    """
    stem = Path(wav_name).stem          # e.g. '6338.240528160459'
    start = int(float(offset_s))
    end   = start + 5
    time_range = f'{_fmt_time(start)}-{_fmt_time(end)}'   # e.g. '00.10-00.15'

    if dataset == 'ARBAS':
        # Date from wav stem: pattern YYMMDD embedded after station+dot
        # e.g. 6338.240528160459  →  2024-05-28
        m = re.search(r'\d+\.(\d{2})(\d{2})(\d{2})', stem)
        if m:
            date_str = f'20{m.group(1)}-{m.group(2)}-{m.group(3)}'
        else:
            date_str = 'unknown-date'
        return f'ARBAS_{date_str}_{stem}_{time_range}'
        # → ARBAS_2024-05-28_6338.240528160459_00.10-00.15

    elif dataset == 'HARRAPATU':
        # Station = digits before first dot  e.g. '6338'
        station = stem.split('.')[0]
        return f'{station}_{stem}_{time_range}'
        # → 6338_6338.251010055958_00.00-00.05

    else:
        # Fallback for unknown dataset
        return f'{stem}_{time_range}'


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

def top4(row):
    probs = [(c, float(row.get('prob_' + c) or 0)) for c in SP_COLS]
    probs.sort(key=lambda x: -x[1])
    return [[c, round(p, 4)] for c, p in probs[:4]]

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

def check_spec(spec_dir, stem, spec_ext):
    """Return True if the spectrogram file exists on disk."""
    if not spec_dir:
        return False
    return Path(spec_dir, stem + spec_ext).exists()

def seg_spec_url(spec_base_url, stem, spec_ext):
    return spec_base_url.rstrip('/') + '/' + stem + spec_ext


# ── ARBAS ──────────────────────────────────────────────────────────────────────

def process_arbas(csv_path, spec_dir, spec_ext, spec_base_url):
    rows  = read_csv(csv_path)
    wavs  = group_by_wav(rows)
    out   = []

    for w in wavs:
        r0 = w['segs'][0]
        exp_det = r0.get('exp_cetacean_detected', 'False') == 'True'
        exp_sp  = r0.get('exp_top_species', 'background')
        exp_flags = {}
        for c in EXP_COLS_ARBAS:
            v = float(r0.get('exp_' + c) or 0)
            if v > 0:
                exp_flags[c] = round(v, 3)

        segs_out   = []
        mine_pos   = False
        any_spec   = False

        for s in w['segs']:
            p_code = CODE_MAP.get(s.get('pred_argmax', 'background'), 'bg')
            mp     = round(float(s.get('max_cetacean_prob') or 0), 4)
            ag     = s.get('agreement_pred_pr', '')
            t4     = top4(s)
            offset = float(s.get('offset_s', 0))

            # per-segment spectrogram
            stem     = seg_spec_stem(w['wav'], offset, 'ARBAS')
            has_spec = check_spec(spec_dir, stem, spec_ext)
            sp_path  = seg_spec_url(spec_base_url, stem, spec_ext) if has_spec else None
            if has_spec:
                any_spec = True

            segs_out.append({
                'o': offset, 'p': p_code, 'mp': mp,
                't': t4, 'ag': ag, 'specPath': sp_path,
            })
            if p_code != 'bg':
                mine_pos = True

        out.append({
            'wav': w['wav'], 't': parse_ts(w['wav']), 'dur': 60,
            'expDet': exp_det, 'expSp': exp_sp, 'expFlags': exp_flags,
            'minePos': mine_pos,
            'spec': any_spec,     # True if ≥1 segment has a spectrogram
            'specPath': None,     # per-segment paths stored on each seg
            'segs': segs_out,
        })

    return out


# ── HARRAPATU ──────────────────────────────────────────────────────────────────

def process_harrapatu(csv_path, spec_dir, spec_ext, spec_base_url):
    rows = read_csv(csv_path)

    # group by interval, not wav_name — multiple recorders per 5-min bucket
    intervals, ivl_map = [], {}
    for r in rows:
        key = r.get('interval') or r['wav_name']
        if key not in ivl_map:
            ivl_map[key] = {'interval': key, 'segs': []}
            intervals.append(ivl_map[key])
        ivl_map[key]['segs'].append(r)

    out = []
    for ivl in intervals:
        segs = ivl['segs']
        r0   = segs[0]

        exp_det = any(s.get('exp_cetacean_detected', 'False') == 'True' for s in segs)
        ivl_outcome = r0.get('outcome', '')

        segs_out = []
        mine_pos = False
        any_spec = False

        for s in segs:
            p_code = CODE_MAP.get(s.get('pred_argmax', 'background'), 'bg')
            mp     = round(float(s.get('max_cetacean_prob') or 0), 4)
            oc     = s.get('seg_outcome', '')
            t4     = top4(s)
            offset = float(s.get('offset_s', 0))
            wav    = s.get('wav_name', '')

            stem     = seg_spec_stem(wav, offset, 'HARRAPATU')
            has_spec = check_spec(spec_dir, stem, spec_ext)
            sp_path  = seg_spec_url(spec_base_url, stem, spec_ext) if has_spec else None
            if has_spec:
                any_spec = True

            segs_out.append({
                'o': offset, 'p': p_code, 'mp': mp,
                't': t4, 'oc': oc, 'specPath': sp_path,
            })
            if p_code != 'bg':
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

    return out

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Build detections.json for CET·DET')
    ap.add_argument('--arbas',              required=True,  help='ARBAS CSV path')
    ap.add_argument('--harrapatu',          required=True,  help='HARRAPATU CSV path')
    ap.add_argument('--arbas-spec-dir',     default=None,   help='Folder of ARBAS spectrograms')
    ap.add_argument('--harrapatu-spec-dir', default=None,   help='Folder of HARRAPATU spectrograms')
    ap.add_argument('--arbas-spec-url',     default=None,   help='Base URL for ARBAS spectrograms')
    ap.add_argument('--harrapatu-spec-url', default=None,  help='Base URL for HARRAPATU spectrograms')
    ap.add_argument('--spec-ext',           default='.png')
    ap.add_argument('--out',                default='data/detections.json')
    args = ap.parse_args()

    print('Processing ARBAS…')
    arbas_wavs = process_arbas(
        args.arbas, args.arbas_spec_dir, args.spec_ext, args.arbas_spec_url)
    a_spec = sum(1 for w in arbas_wavs if w['spec'])
    print(f'  {len(arbas_wavs)} minutes  |  {a_spec} with spectrograms')

    print('Processing HARRAPATU…')
    harra_wavs = process_harrapatu(
        args.harrapatu, args.harrapatu_spec_dir, args.spec_ext, args.harrapatu_spec_url)
    h_spec = sum(1 for w in harra_wavs if w['spec'])
    print(f'  {len(harra_wavs)} intervals  |  {h_spec} with spectrograms')

    result = {
        'generated': datetime.now(timezone.utc).isoformat(),
        'species':   SPECIES_NAMES,
        'datasets': {
            'ARBAS': {
                'id': 'ARBAS', 'expertGranularity': 60, 'myGranularity': 5,
                'expertSpeciesNote': 'multi-species', 'wavs': arbas_wavs,
            },
            'HARRAPATU': {
                'id': 'HARRAPATU', 'expertGranularity': 300, 'myGranularity': 5,
                'expertSpeciesNote': 'Tursiops truncatus only', 'wavs': harra_wavs,
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
