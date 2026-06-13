#!/usr/bin/env python3
"""
Cetacean Inspector — aligned spectrogram / model / expert browser (dark)
========================================================================

Qualitative inspection of the ARBAS model-vs-expert comparison. For a window of
3 consecutive 5-second segments it shows, column-aligned:

    [ kHz axis ]  spectrogram   spectrogram   spectrogram
                  argmax chip   argmax chip   ...
                  vec    chip   ...
                  pr     chip
                  consensus chip
                  EXPERT chip

Read one segment top-to-bottom: what the audio looks like, what each decoding
strategy said, and what the expert said. Click a column for the per-species
probability-vs-vote detail.

Theme: cet_l20 (deep blue -> teal -> green -> chartreuse -> yellow). 
No warm/orange colors. The frequency axis reflects the log-mel scale.

Run:
    python cetacean_inspector_app.py \
        --csv  arbas_comparison_5s_v3.csv \
        --spectrograms /path/to/spectrograms \
        --port 8050
"""
from __future__ import annotations

import argparse
import base64
import math
import re
from pathlib import Path

import pandas as pd
import dash
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

# ── design tokens (cet_l20) ──────────────────────────────────────────────────
BG        = '#08111f'   # near-black blue (page)
PANEL     = '#0d1b2e'   # raised panel
PANEL_2   = '#11233a'   # secondary panel / detail
LINE      = '#1b3a5b'   # hairline / borders
TEAL      = '#1f6f6a'
GREEN     = '#2e8b6f'
CHART     = '#9bbf3a'   # chartreuse
YELLOW    = '#f2d65c'   # highlight
INK       = '#dce6f2'   # primary text
INK_DIM   = '#8aa0bd'   # secondary text
INK_FAINT = '#52688a'   # tertiary / captions

PAGE_SIZE = 3           # fixed: three spectrograms at a time
SPECTRO_W = 380         # px width per spectrogram column
SPECTRO_H = 650         # px height of the spectrogram image area (full vertical space)
FREQ_MAX_KHZ = 16.0     # top of the mel axis (override via --freq-max-khz)

# species -> strictly blue/green/yellow accents for contrast
SPECIES_COLOR = {
    'Dd': '#3b6ea5', 'Gg': '#2e8b6f', 'Gm': '#4ea36b', 'Oo': '#a2c423',
    'Pm': '#1f6f6a', 'Sc': '#c9b037', 'Tt': '#5c9ba3', 'Ambig': '#4d7560',
    'Ba': '#3f8e8a', 'Bp': '#418b9c', 'Dc': '#5b7d79', 'Lo': '#89b83e',
    'Zc': '#457a66', 'Bb': '#41708f',
}
NEUTRAL = {
    'background': '#16273d', 'uncertain': '#385775',
    'no_label': '#101d2e', 'nan': '#101d2e', '': '#101d2e', '—': '#101d2e',
}
SHARED_SPECIES = ['Dd', 'Gg', 'Gm', 'Oo', 'Pm', 'Sc', 'Tt', 'Ambig']
STRATEGIES = [('pred_argmax', 'argmax'), ('pred_vec', 'vec'),
              ('pred_pr', 'pr'), ('pred_consensus', 'consensus')]


def chip_color(label) -> str:
    s = str(label)
    if s in SPECIES_COLOR:
        return SPECIES_COLOR[s]
    return NEUTRAL.get(s, '#3a5863')  # unknown species -> muted dark teal


def is_neutral(label) -> bool:
    return str(label) in ('background', 'uncertain', 'no_label', 'nan', '', '—')


# ── spectrogram path reconstruction ──────────────────────────────────────────
def wav_to_date(wav_name: str) -> str:
    m = re.search(r'\.(\d{2})(\d{2})(\d{2})', wav_name)
    if m:
        yy, mm, dd = m.groups()
        return f"20{yy}-{mm}-{dd}"
    return 'unknown-date'


def offset_to_timerange(offset_s) -> str:
    s = int(float(offset_s)); e = s + 5
    fmt = lambda x: f"{x // 60:02d}.{x % 60:02d}"
    return f"{fmt(s)}-{fmt(e)}"


def find_spectrogram(spectro_dir: Path, wav_name: str, offset_s):
    stem = Path(wav_name).stem
    trange = offset_to_timerange(offset_s)
    for s in (stem, stem.upper()):
        p = spectro_dir / f"ARBAS_{wav_to_date(wav_name)}_{s}_{trange}.png"
        if p.exists():
            return p
    needle = f"_{stem.lower()}_{trange}.png"
    for p in spectro_dir.glob(f"*_{trange}.png"):
        if p.name.lower().endswith(needle):
            return p
    return None


def encode_png(path):
    if path is None or isinstance(path, float):
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        return f"data:image/png;base64,{data}"
    except Exception:
        return None


# ── data loading ─────────────────────────────────────────────────────────────
def load_comparison(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    sort_cols = [c for c in ['wav_name', 'segment_index', 'offset_s'] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    for col, fill in [('exp_top_species', 'no_label'),
                      ('exp_cetacean_detected', False),
                      ('expert_annotated', False),
                      ('max_cetacean_prob', 0.0)]:
        if col not in df.columns:
            df[col] = fill
    for strat, _ in STRATEGIES:
        if strat not in df.columns:
            df[strat] = '—'
    return df


# ── filtering ────────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame, mode: str, species) -> pd.DataFrame:
    out = df
    if mode == 'disagree_strat':
        out = out[out['pred_consensus'] == 'uncertain']
    elif mode == 'disagree_expert':
        if 'cetacean_consensus' in out.columns:
            me = out['cetacean_consensus'].astype(bool)
        else:
            me = out['pred_consensus'].ne('background') & out['pred_consensus'].ne('uncertain')
        out = out[out['expert_annotated'].astype(bool) &
                  (me != out['exp_cetacean_detected'].astype(bool))]
    elif mode == 'expert_positive':
        out = out[out['exp_cetacean_detected'].astype(bool)]
    elif mode == 'has_png':
        if '_has_png' in out.columns:
            out = out[out['_has_png']]
    if species and species != 'ALL':
        in_any = (
            df.get('pred_pr', pd.Series('', index=df.index)).eq(species) |
            df.get('pred_vec', pd.Series('', index=df.index)).eq(species) |
            df.get('pred_argmax', pd.Series('', index=df.index)).eq(species) |
            df.get('exp_top_species', pd.Series('', index=df.index)).eq(species)
        )
        out = out[in_any.reindex(out.index, fill_value=False)]
    return out.reset_index(drop=True)


# ── frequency axis (log-mel) ─────────────────────────────────────────────────
def mel_axis(height_px: int = SPECTRO_H, freq_max_khz: float = 16.0):
    def hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)
    mel_max = hz_to_mel(FREQ_MAX_KHZ * 1000.0)
    ticks = [t for t in [0, 1, 2, 4, 8, 12, 16, 20, 24] if t <= FREQ_MAX_KHZ]
    children = [html.Div('kHz', style={
        'position': 'absolute', 'top': '-18px', 'right': '0',
        'fontSize': '9px', 'letterSpacing': '0.1em', 'color': INK_DIM,
        'textTransform': 'uppercase'})]
    for khz in ticks:
        frac = hz_to_mel(khz * 1000.0) / mel_max
        top = (1.0 - frac) * height_px
        children.append(html.Div(f"{khz}", style={
            'position': 'absolute', 'top': f'{top:.1f}px', 'right': '8px',
            'transform': 'translateY(-50%)', 'fontSize': '10px',
            'fontFamily': 'ui-monospace, monospace', 'color': INK_FAINT}))
        children.append(html.Div(style={
            'position': 'absolute', 'top': f'{top:.1f}px', 'right': '0',
            'width': '5px', 'height': '1px', 'background': INK_FAINT}))
    return html.Div(children, style={
        'position': 'relative', 'width': '36px', 'height': f'{height_px}px',
        'flex': '0 0 auto', 'borderRight': f'1px solid {LINE}',
        'marginRight': '12px', 'marginTop': '4px'})


# ── chips ────────────────────────────────────────────────────────────────────
def chip(label, role='strat'):
    c = chip_color(label)
    neutral = is_neutral(label)
    style = {
        'borderRadius': '6px', 'padding': '5px 6px', 'fontSize': '11px',
        'textAlign': 'center', 'overflow': 'hidden', 'textOverflow': 'ellipsis',
        'whiteSpace': 'nowrap', 'fontFamily': 'ui-monospace, monospace',
        'letterSpacing': '0.02em', 'height': '16px', 'lineHeight': '16px',
        'background': (c + '22') if neutral else c,
        'color': INK_DIM if neutral else '#06101c',
        'border': (f'1px solid {c}55' if neutral else f'1px solid {c}'),
        'fontWeight': '400' if neutral else '700',
    }
    if role == 'expert':
        style['borderColor'] = (f'{YELLOW}88' if neutral else YELLOW)
        style['boxShadow'] = f'0 0 0 1px {YELLOW}33'
    return html.Div(str(label), style=style)


# ── one segment column ───────────────────────────────────────────────────────
def segment_column(pos: int, row, spectro_dir: Path):
    png = row.get('_png') if '_png' in row.index else None
    if png is None or isinstance(png, float):
        png = find_spectrogram(spectro_dir, row['wav_name'], row['offset_s'])
    src = encode_png(png)

    if src:
        img = html.Img(src=src, style={
            'width': '100%', 'height': f'{SPECTRO_H}px', 'objectFit': 'contain',
            'display': 'block', 'borderRadius': '8px',
            'background': '#040a14', 'border': f'1px solid {LINE}'})
    else:
        img = html.Div('no spectrogram rendered', style={
            'height': f'{SPECTRO_H}px', 'display': 'flex', 'alignItems': 'center',
            'justifyContent': 'center', 'borderRadius': '8px',
            'border': f'1px dashed {LINE}', 'background': '#0a1422',
            'color': INK_FAINT, 'fontSize': '11px',
            'fontFamily': 'ui-monospace, monospace', 'textAlign': 'center'})

    exp_detected = bool(row.get('exp_cetacean_detected', False))
    exp_label = (row.get('exp_top_species', 'no_label') if exp_detected
                 else ('background' if bool(row.get('expert_annotated', False)) else '—'))

    return html.Div([
        img,
        html.Div(offset_to_timerange(row['offset_s']), style={
            'fontSize': '10px', 'color': INK_DIM, 'textAlign': 'center',
            'fontFamily': 'ui-monospace, monospace', 'margin': '7px 0 9px'}),
        *[chip(row.get(s, '—')) for s, _ in STRATEGIES],
        html.Div(style={'height': '1px', 'background': LINE, 'margin': '7px 2px'}),
        chip(exp_label, role='expert'),
    ],
        id={'type': 'segcol', 'index': int(pos)},
        n_clicks=0,
        style={'width': f'{SPECTRO_W}px', 'flex': '0 0 auto',
               'display': 'flex', 'flexDirection': 'column', 'gap': '7px',
               'cursor': 'pointer', 'padding': '4px',
               'borderRadius': '10px', 'transition': 'background .12s'})


def rail_labels():
    rows = [('', f'{SPECTRO_H}px'), ('', '24px')]
    rows += [(name, '26px') for _, name in STRATEGIES]
    rows += [('', '8px'), ('EXPERT', '26px')]
    children = []
    for txt, h in rows:
        children.append(html.Div(txt, style={
            'height': h, 'display': 'flex', 'alignItems': 'center',
            'justifyContent': 'flex-end', 'paddingRight': '12px',
            'fontSize': '11px', 'letterSpacing': '0.06em',
            'fontFamily': 'ui-monospace, monospace',
            'color': YELLOW if txt == 'EXPERT' else INK_DIM,
            'fontWeight': '700' if txt == 'EXPERT' else '500',
            'textTransform': 'uppercase' if txt else 'none'}))
    return html.Div(children, style={
        'width': '92px', 'flex': '0 0 auto', 'display': 'flex',
        'flexDirection': 'column', 'gap': '7px', 'paddingTop': '4px',
        'marginRight': '4px'})


def detail_panel(row=None):
    if row is None:
        return html.Div('Select a segment above to inspect its probabilities.',
                        style={'color': INK_FAINT, 'padding': '24px',
                               'fontSize': '13px',
                               'fontFamily': 'ui-monospace, monospace'})
    body = []
    for sp in SHARED_SPECIES:
        # NOTE: Adjust these column names depending on how you store the 3 scores in your CSV.
        # Currently using `prob_argmax_{sp}`, `prob_vec_{sp}`, `prob_pr_{sp}` as placeholders 
        # and falling back to a single `prob_{sp}` if missing to avoid breaking.
        s1 = row.get(f'prob_argmax_{sp}', row.get(f'prob_{sp}', float('nan')))
        s2 = row.get(f'prob_vec_{sp}', row.get(f'prob_{sp}', float('nan')))
        s3 = row.get(f'prob_pr_{sp}', row.get(f'prob_{sp}', float('nan')))
        
        v1 = float(s1) if pd.notna(s1) else 0.0
        v2 = float(s2) if pd.notna(s2) else 0.0
        v3 = float(s3) if pd.notna(s3) else 0.0

        ev = row.get(f'exp_{sp}', float('nan'))
        ev_n = float(ev) if pd.notna(ev) else 0.0

        def make_bar(val, c1, c2):
            return html.Div(style={
                'width': f'{min(1.0, max(0.0, val)) * 100}px', 'height': '6px',
                'borderRadius': '3px', 'marginBottom': '3px',
                'background': f'linear-gradient(90deg, {c1}, {c2})'})

        body.append(html.Tr([
            html.Td(sp, style={'padding': '6px 14px 6px 0', 'color': INK,
                               'fontWeight': '700', 'verticalAlign': 'top'}),
            html.Td([
                make_bar(v1, TEAL, GREEN),
                make_bar(v2, GREEN, CHART),
                make_bar(v3, TEAL, CHART)
            ], style={'padding': '6px 8px', 'verticalAlign': 'top'}),
            html.Td(html.Div([
                html.Div(f"{v1:.4f}" if pd.notna(s1) else '—'),
                html.Div(f"{v2:.4f}" if pd.notna(s2) else '—'),
                html.Div(f"{v3:.4f}" if pd.notna(s3) else '—'),
            ]), style={'padding': '3px 14px', 'color': INK_DIM, 'fontSize': '10px', 
                       'lineHeight': '1.3', 'verticalAlign': 'top'}),
            html.Td(make_bar(ev_n, LINE, YELLOW), style={'padding': '11px 8px', 'verticalAlign': 'top'}),
            html.Td(f"{ev_n * 100:.1f}%" if pd.notna(ev) else '—',
                    style={'padding': '6px 14px', 'color': YELLOW, 'fontWeight': '700', 'verticalAlign': 'top'}),
        ]))

    return html.Div([
        html.Div([
            html.Span(str(row['wav_name']), style={'color': INK, 'fontWeight': '700'}),
            html.Span(f"   seg {row.get('segment_index', '?')}   ·   "
                      f"{offset_to_timerange(row['offset_s'])}",
                      style={'color': INK_DIM}),
        ], style={'fontFamily': 'ui-monospace, monospace', 'fontSize': '13px',
                  'marginBottom': '14px'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th('species', style={'textAlign': 'left', 'color': INK_FAINT,
                                          'fontWeight': '500', 'padding': '0 14px 8px 0'}),
                html.Th('model scores (x3)', style={'textAlign': 'left', 'color': INK_FAINT,
                                                    'fontWeight': '500', 'padding': '0 8px 8px'}),
                html.Th('probs', style={'textAlign': 'left', 'color': INK_FAINT,
                                        'fontWeight': '500', 'padding': '0 14px 8px'}),
                html.Th('expert votes', style={'textAlign': 'left', 'color': INK_FAINT,
                                               'fontWeight': '500', 'padding': '0 8px 8px'}),
                html.Th('%', style={'textAlign': 'left', 'color': INK_FAINT,
                                    'fontWeight': '500', 'padding': '0 14px 8px'}),
            ])),
            html.Tbody(body),
        ], style={'borderCollapse': 'collapse', 'fontSize': '12px',
                  'fontFamily': 'ui-monospace, monospace'}),
    ], style={'padding': '20px 22px'})


# ── app factory ──────────────────────────────────────────────────────────────
def make_app(df, spectro_dir, freq_max_khz):
    app = Dash(__name__)

    png_index = {p.name.lower(): p for p in spectro_dir.glob('*.png')} \
        if spectro_dir.exists() else {}

    def _expected_name(row):
        stem = Path(str(row['wav_name'])).stem
        return (f"ARBAS_{wav_to_date(str(row['wav_name']))}_{stem}_"
                f"{offset_to_timerange(row['offset_s'])}.png").lower()

    df = df.copy()
    df['_png'] = [png_index.get(_expected_name(r)) for _, r in df.iterrows()]
    df['_has_png'] = df['_png'].notna()
    print(f"Indexed {len(png_index):,} PNG files; "
          f"{df['_has_png'].sum():,} of {len(df):,} segments have a rendered spectrogram.")

    # Deep dark mode overrides for dropdown components included directly in index_string
    app.index_string = (
        '<!DOCTYPE html><html><head>{%metas%}<title>Cetacean Inspector</title>'
        '{%favicon%}{%css%}<style>'
        f'body{{margin:0;background:{BG};}}'
        f'::selection{{background:{GREEN}55;}}'
        f'.Select-control, .Select-menu-outer, .Select-multi-value-wrapper, .select-up, .is-open>.Select-control '
        f'{{background-color:{PANEL}!important;border-color:{LINE}!important;color:{INK}!important;border-radius:8px!important;}}'
        f'.has-value.Select--single>.Select-control .Select-value .Select-value-label, .Select-value-label '
        f'{{color:{INK}!important;}}'
        f'.VirtualizedSelectOption {{background-color:{PANEL}!important;color:{INK}!important;}}'
        f'.VirtualizedSelectFocusedOption {{background-color:{PANEL_2}!important;color:{YELLOW}!important;}}'
        f'.Select-placeholder {{color:{INK_DIM}!important;}}'
        f'.Select-arrow {{border-color:{INK_DIM} transparent transparent!important;}}'
        f'.is-open .Select-arrow {{border-color:transparent transparent {INK_DIM}!important;}}'
        f'input{{background:{PANEL}!important;color:{INK}!important;border:1px solid {LINE}!important;'
        f'border-radius:8px!important;padding:6px 8px!important;font-family:ui-monospace,monospace!important;}}'
        f'.ci-btn{{background:{PANEL_2};color:{INK};border:1px solid {LINE};border-radius:8px;'
        f'padding:7px 14px;font-family:ui-monospace,monospace;font-size:13px;cursor:pointer;'
        f'transition:background .15s,border-color .15s;}}'
        f'.ci-btn:hover{{background:{TEAL};border-color:{GREEN};}}'
        f'#filmstrip>div:hover{{background:{PANEL_2};}}'
        f'::-webkit-scrollbar{{height:10px;width:10px;}}'
        f'::-webkit-scrollbar-track{{background:{BG};}}'
        f'::-webkit-scrollbar-thumb{{background:{LINE};border-radius:5px;}}'
        f'::-webkit-scrollbar-thumb:hover{{background:{TEAL};}}'
        '</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'
    )

    species_opts = [{'label': 'all species', 'value': 'ALL'}] + \
                   [{'label': s, 'value': s} for s in SHARED_SPECIES]
    label_css = {'color': INK_DIM, 'fontSize': '12px',
                 'fontFamily': 'ui-monospace, monospace',
                 'letterSpacing': '0.04em', 'marginRight': '8px'}

    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Span('CET', style={'color': YELLOW}),
                html.Span('·', style={'color': INK_FAINT, 'margin': '0 6px'}),
                html.Span('inspector', style={'color': INK}),
            ], style={'fontSize': '20px', 'fontWeight': '800',
                      'letterSpacing': '0.04em',
                      'fontFamily': 'ui-monospace, monospace'}),
            html.Div('spectrogram · model · expert', style={
                'color': INK_DIM, 'fontSize': '12px', 'letterSpacing': '0.12em',
                'textTransform': 'uppercase', 'marginTop': '2px'}),
        ], style={'marginBottom': '4px'}),
        html.Div(id='status', style={
            'color': INK_FAINT, 'fontSize': '12px', 'marginBottom': '16px',
            'fontFamily': 'ui-monospace, monospace'}),

        html.Div([
            html.Label('view', style=label_css),
            dcc.Dropdown(id='mode', clearable=False, value='has_png',
                         style={'width': '300px'},
                         options=[
                             {'label': 'only segments with a spectrogram', 'value': 'has_png'},
                             {'label': 'sequential — all segments', 'value': 'all'},
                             {'label': 'strategies disagree (uncertain)', 'value': 'disagree_strat'},
                             {'label': 'I disagree with expert', 'value': 'disagree_expert'},
                             {'label': 'expert heard a cetacean', 'value': 'expert_positive'},
                         ]),
            html.Label('species', style={**label_css, 'marginLeft': '14px'}),
            dcc.Dropdown(id='species', clearable=False, value='ALL',
                         options=species_opts, style={'width': '150px'}),
            html.Div(style={'flex': '1 1 auto'}),
            html.Button('◀', id='prev', n_clicks=0, className='ci-btn'),
            dcc.Input(id='jump', type='number', value=1, min=1, step=1,
                      style={'width': '72px', 'margin': '0 8px', 'textAlign': 'center'}),
            html.Button('go', id='gobtn', n_clicks=0, className='ci-btn'),
            html.Button('▶', id='next', n_clicks=0, className='ci-btn',
                        style={'marginLeft': '8px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap',
                  'gap': '6px', 'marginBottom': '16px'}),

        html.Div([
            rail_labels(),
            mel_axis(SPECTRO_H, freq_max_khz),
            html.Div(id='filmstrip', style={
                'display': 'flex', 'flexDirection': 'row', 'gap': '16px',
                'overflowX': 'auto', 'flex': '1 1 auto', 'paddingTop': '4px'}),
        ], style={'display': 'flex', 'flexDirection': 'row',
                  'background': PANEL, 'border': f'1px solid {LINE}',
                  'borderRadius': '16px', 'padding': '20px'}),

        html.Div(id='detail', style={
            'marginTop': '16px', 'background': PANEL_2,
            'border': f'1px solid {LINE}', 'borderRadius': '16px',
            'minHeight': '90px'}),

        dcc.Store(id='page', data=0),
    ], style={'fontFamily': 'system-ui, -apple-system, sans-serif',
              'padding': '28px 32px', 'maxWidth': '100%', 'boxSizing': 'border-box',
              'background': BG, 'minHeight': '100vh', 'color': INK})

    @app.callback(
        Output('filmstrip', 'children'),
        Output('status', 'children'),
        Output('page', 'data'),
        Output('jump', 'value'),
        Input('prev', 'n_clicks'), Input('next', 'n_clicks'),
        Input('gobtn', 'n_clicks'), Input('mode', 'value'),
        Input('species', 'value'),
        State('page', 'data'), State('jump', 'value'),
    )
    def render(prev_c, next_c, go_c, mode, species, page, jump):
        sub = apply_filters(df, mode, species)
        n = len(sub)
        n_pages = max(1, (n + PAGE_SIZE - 1) // PAGE_SIZE)
        trig = ctx.triggered_id
        if trig in ('mode', 'species'):
            page = 0
        elif trig == 'prev':
            page = max(0, (page or 0) - 1)
        elif trig == 'next':
            page = min(n_pages - 1, (page or 0) + 1)
        elif trig == 'gobtn':
            page = min(n_pages - 1, max(0, int(jump or 1) - 1))
        page = min(max(0, page or 0), n_pages - 1)

        if n == 0:
            return ([html.Div('No segments match this view.', style={
                        'padding': '40px', 'color': INK_FAINT,
                        'fontFamily': 'ui-monospace, monospace'})],
                    'No segments match the current view.', page, 1)

        lo, hi = page * PAGE_SIZE, min(page * PAGE_SIZE + PAGE_SIZE, n)
        window = sub.iloc[lo:hi]
        cols = [segment_column(p, r, spectro_dir)
                for p, (_, r) in enumerate(window.iterrows())]
        status = (f"segments {lo + 1:,}–{hi:,} of {n:,}   ·   "
                  f"page {page + 1} / {n_pages}   ·   view: {mode}   ·   species: {species}")
        return cols, status, page, page + 1

    @app.callback(
        Output('detail', 'children'),
        Input({'type': 'segcol', 'index': dash.ALL}, 'n_clicks'),
        State('mode', 'value'), State('species', 'value'), State('page', 'data'),
        prevent_initial_call=True,
    )
    def show_detail(_clicks, mode, species, page):
        trig = ctx.triggered_id
        if not isinstance(trig, dict):
            return no_update
        sub = apply_filters(df, mode, species)
        idx = int(page or 0) * PAGE_SIZE + int(trig['index'])
        if idx >= len(sub):
            return no_update
        return detail_panel(sub.iloc[idx])

    return app


def main():
    ap = argparse.ArgumentParser(description='Cetacean inspector Dash app (dark)')
    ap.add_argument('--csv', default='arbas_comparison_5s_v3.csv')
    ap.add_argument('--spectrograms', required=True)
    ap.add_argument('--port', type=int, default=8050)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--freq-max-khz', type=float, default=16.0)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    global FREQ_MAX_KHZ
    FREQ_MAX_KHZ = args.freq_max_khz

    df = load_comparison(args.csv)
    spectro_dir = Path(args.spectrograms)
    print(f"Loaded {len(df):,} segments from {args.csv}")
    print(f"Spectrograms: {spectro_dir}  (exists: {spectro_dir.exists()})")

    app = make_app(df, spectro_dir, args.freq_max_khz)
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except AttributeError:
        app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()