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
strategy said, and what the expert said. The detail panel below shows the
per-species probability-vs-vote detail for ALL three segments in the window.

Spectrograms render at their native 500x128 aspect ratio (wide & short) and are
butted together so the window reads as one continuous recording, with a thin
separator between adjacent segments.

Theme: cet_l20 (deep blue -> teal -> green -> chartreuse -> yellow), warmed
toward green/yellow. The frequency axis reflects the log-mel scale.

PROBABILITY COLUMNS (detail panel)
----------------------------------
The panel auto-detects which probability sets are present in the CSV and shows
one bar per set, per species:
  * raw / argmax softmax  -> prob_{sp}            (always present)
  * vector-scaled         -> prob_vec_{sp}        (if the decoder saved them)
  * pr-input prob         -> prob_pr_{sp}         (if the decoder saved them)
If only the single legacy prob_{sp} exists, the panel shows one bar (current
behaviour). PR thresholds, if provided as pr_threshold_{sp} columns, are drawn
as a marker on the relevant bar rather than a separate bar.

NEW FILTERS (pred_consensus vs expert):
  * true_positive  — consensus=cetacean AND expert=cetacean
  * false_positive — consensus=cetacean AND expert=background/no_label
  * false_negative — consensus=background AND expert=cetacean
  * true_negative  — consensus=background AND expert=background/no_label

Run:
    python cetacean_inspector_app.py \
        --csv  arbas_comparison_5s_v3.csv \
        --spectrograms /path/to/spectrograms \
        --port 8050
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
import dash
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import flask

# ── design tokens (cet_l20, warmed toward green/yellow) ──────────────────────
BG        = '#0a1422'   # near-black blue (page) — a touch warmer/lighter
PANEL     = '#0f2033'   # raised panel
PANEL_2   = '#16304a'   # secondary panel / detail (lifted, less navy)
LINE      = '#244a6b'   # hairline / borders (slightly brighter)
TEAL      = '#2a857d'
GREEN     = '#3da57f'   # primary green (lighter than before)
GREEN_LT  = '#6fcf97'   # light green accent (new)
CHART     = '#b6d44a'   # chartreuse
YELLOW    = '#ffd95c'   # highlight (brighter, more saturated)
YELLOW_LT = '#ffe88a'   # soft yellow wash
INK       = '#e4edf7'   # primary text
INK_DIM   = '#9fb4cf'   # secondary text
INK_FAINT = '#5e779a'   # tertiary / captions
VIOLET    = '#9a8cff'   # third prob set (pr) — cool accent, distinct from green/yellow

# Confusion-matrix badge colours
TP_COL = '#3da57f'   # green  — true positive
FP_COL = '#ffd95c'   # yellow — false positive
FN_COL = '#ff8c5a'   # orange — false negative
TN_COL = '#5e779a'   # muted  — true negative

PAGE_SIZE = 3           # three spectrograms at a time
# Native PNG is 500x128 (≈3.9:1). Keep that ratio; size to fit 3-up on desktop.
SPECTRO_W = 460
SPECTRO_H = int(round(SPECTRO_W * 128 / 500))   # ≈118px — true aspect ratio
FREQ_MAX_KHZ = 16.0     # top of the mel axis (override via --freq-max-khz)

# Which decoding the single stored prob_{sp} column corresponds to.
# Used as the label for the legacy single-bar fallback.
PROB_SOURCE = 'pred_vec'

# species -> blue/green/yellow accents, biased warm (green/chartreuse/teal)
SPECIES_COLOR = {
    'Dd': '#4a8fb0', 'Gg': '#3da57f', 'Gm': '#6fcf97', 'Oo': '#b6d44a',
    'Pm': '#2a857d', 'Sc': '#d8c84a', 'Tt': '#5fb0a0', 'Ambig': '#5c8a6e',
    'Ba': '#49a39a', 'Bp': '#4f9fb0', 'Dc': '#6f9a86', 'Lo': '#a4cc4e',
    'Zc': '#52a37a', 'Bb': '#4d86a8',
}
NEUTRAL = {
    'background': '#1a2f47', 'uncertain': '#3f6488',
    'no_label': '#13243a', 'nan': '#13243a', '': '#13243a', '—': '#13243a',
}
SHARED_SPECIES = ['Dd', 'Gg', 'Gm', 'Oo', 'Pm', 'Sc', 'Tt', 'Ambig']
STRATEGIES = [('pred_argmax', 'argmax'), ('pred_vec', 'vec'),
              ('pred_pr', 'pr'), ('pred_consensus', 'consensus')]

# Probability sets the detail panel can render, in display order.
PROB_SETS = [
    ('prob_',     'raw / argmax', GREEN,  GREEN_LT),
    ('prob_vec_', 'vector',       CHART,  YELLOW),
    ('prob_pr_',  'pr',           VIOLET, '#c4bcff'),
]


def chip_color(label) -> str:
    s = str(label)
    if s in SPECIES_COLOR:
        return SPECIES_COLOR[s]
    return NEUTRAL.get(s, '#4a7560')


def is_neutral(label) -> bool:
    return str(label) in ('background', 'uncertain', 'no_label', 'nan', '', '—')


def consensus_is_cetacean(val) -> bool:
    """True if pred_consensus is a real species (not background/uncertain/—)."""
    return str(val) not in ('background', 'uncertain', 'no_label', 'nan', '', '—')


# ── spectrogram path reconstruction ──────────────────────────────────────────
def wav_to_date(wav_name: str) -> str:
    m = re.search(r'\.(\d{2})(\d{2})(\d{2})', wav_name)
    if m:
        yy, mm, dd = m.groups()
        return f"20{yy}-{mm}-{dd}"
    return 'unknown-date'


def wav_to_clock(wav_name: str) -> str:
    m = re.search(r'\.\d{6}(\d{2})(\d{2})(\d{2})', wav_name)
    if m:
        hh, mm, ss = m.groups()
        return f"{hh}:{mm}:{ss}"
    return ''


def offset_to_timerange(offset_s) -> str:
    s = int(float(offset_s)); e = s + 5
    fmt = lambda x: f"{x // 60:02d}.{x % 60:02d}"
    return f"{fmt(s)}-{fmt(e)}"


def expected_png_name(wav_name, offset_s) -> str:
    stem = Path(str(wav_name)).stem
    return (f"ARBAS_{wav_to_date(str(wav_name))}_{stem}_"
            f"{offset_to_timerange(offset_s)}.png")


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


def detect_prob_sets(df: pd.DataFrame):
    present = []
    for prefix, label, c0, c1 in PROB_SETS:
        if any(f'{prefix}{sp}' in df.columns for sp in SHARED_SPECIES):
            present.append((prefix, label, c0, c1))
    return present


# ── confusion helpers ─────────────────────────────────────────────────────────
def confusion_label(row) -> str:
    """Return 'TP', 'FP', 'FN', 'TN', or '' if expert not annotated."""
    if not bool(row.get('expert_annotated', False)):
        return ''
    model_pos = consensus_is_cetacean(row.get('pred_consensus', ''))
    expert_pos = bool(row.get('exp_cetacean_detected', False))
    if model_pos and expert_pos:
        return 'TP'
    if model_pos and not expert_pos:
        return 'FP'
    if not model_pos and expert_pos:
        return 'FN'
    return 'TN'


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
    elif mode in ('true_positive', 'false_positive', 'false_negative', 'true_negative'):
        # only consider rows that have a rendered spectrogram
        if '_has_png' in out.columns:
            out = out[out['_has_png']]
        target = {'true_positive': 'TP', 'false_positive': 'FP',
                  'false_negative': 'FN', 'true_negative': 'TN'}[mode]
        mask = out.apply(lambda r: confusion_label(r) == target, axis=1)
        out = out[mask]
    if species and species != 'ALL':
        in_any = (
            df.get('pred_pr', pd.Series('', index=df.index)).eq(species) |
            df.get('pred_vec', pd.Series('', index=df.index)).eq(species) |
            df.get('pred_argmax', pd.Series('', index=df.index)).eq(species) |
            df.get('exp_top_species', pd.Series('', index=df.index)).eq(species)
        )
        out = out[in_any.reindex(out.index, fill_value=False)]
    return out.reset_index(drop=True)


# ── species stats panel ───────────────────────────────────────────────────────
def species_stats_panel(sub: pd.DataFrame) -> html.Div:
    """Always-visible panel above the overview. Shows per-species segment counts
    for (a) expert labels and (b) pred_consensus, in the currently filtered data.
    Also shows TP/FP/FN/TN totals for annotated segments."""
    n = len(sub)
    if n == 0:
        return html.Div()

    # ── per-species counts ────────────────────────────────────────────────────
    expert_counts   = {}
    consensus_counts = {}
    for sp in SHARED_SPECIES:
        expert_counts[sp]    = int((sub.get('exp_top_species', pd.Series()) == sp).sum())
        consensus_counts[sp] = int((sub.get('pred_consensus', pd.Series()) == sp).sum())

    max_exp  = max(expert_counts.values()) or 1
    max_cons = max(consensus_counts.values()) or 1

    def bar_row(sp):
        ec = expert_counts[sp]
        cc = consensus_counts[sp]
        col = SPECIES_COLOR.get(sp, GREEN)
        bar_w_exp  = f"{ec  / max_exp  * 100:.1f}%"
        bar_w_cons = f"{cc / max_cons * 100:.1f}%"
        return html.Div([
            # species label
            html.Div(sp, style={
                'width': '46px', 'color': col, 'fontWeight': '700',
                'fontSize': '12px', 'flexShrink': '0',
                'fontFamily': 'ui-monospace, monospace'}),
            # expert bar + count
            html.Div([
                html.Div(style={
                    'width': bar_w_exp, 'height': '8px', 'borderRadius': '3px',
                    'background': YELLOW, 'minWidth': '2px' if ec > 0 else '0'}),
            ], style={'flex': '1 1 0', 'display': 'flex', 'alignItems': 'center',
                      'background': f'{YELLOW}18', 'borderRadius': '3px',
                      'padding': '2px 4px', 'marginRight': '6px'}),
            html.Div(str(ec), style={
                'width': '36px', 'textAlign': 'right', 'color': YELLOW,
                'fontSize': '11px', 'fontFamily': 'ui-monospace, monospace',
                'flexShrink': '0', 'marginRight': '16px'}),
            # consensus bar + count
            html.Div([
                html.Div(style={
                    'width': bar_w_cons, 'height': '8px', 'borderRadius': '3px',
                    'background': GREEN, 'minWidth': '2px' if cc > 0 else '0'}),
            ], style={'flex': '1 1 0', 'display': 'flex', 'alignItems': 'center',
                      'background': f'{GREEN}18', 'borderRadius': '3px',
                      'padding': '2px 4px', 'marginRight': '6px'}),
            html.Div(str(cc), style={
                'width': '36px', 'textAlign': 'right', 'color': GREEN,
                'fontSize': '11px', 'fontFamily': 'ui-monospace, monospace',
                'flexShrink': '0'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '0',
                  'marginBottom': '5px'})

    legend = html.Div([
        html.Div(style={'width': '46px', 'flexShrink': '0'}),
        html.Div([
            html.Div(style={'width': '10px', 'height': '10px', 'borderRadius': '2px',
                            'background': YELLOW, 'marginRight': '5px',
                            'flexShrink': '0'}),
            html.Span('expert', style={'color': YELLOW, 'fontSize': '11px',
                                       'fontFamily': 'ui-monospace, monospace'}),
        ], style={'flex': '1 1 0', 'display': 'flex', 'alignItems': 'center',
                  'marginRight': '42px'}),
        html.Div([
            html.Div(style={'width': '10px', 'height': '10px', 'borderRadius': '2px',
                            'background': GREEN, 'marginRight': '5px',
                            'flexShrink': '0'}),
            html.Span('consensus', style={'color': GREEN, 'fontSize': '11px',
                                          'fontFamily': 'ui-monospace, monospace'}),
        ], style={'flex': '1 1 0', 'display': 'flex', 'alignItems': 'center'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'})

    # ── confusion matrix totals (PNG-rendered AND expert-annotated rows only) ─
    ann = sub[sub['_has_png']].copy() if '_has_png' in sub.columns else sub.copy()
    if 'expert_annotated' in ann.columns:
        ann = ann[ann['expert_annotated'].astype(bool)]
    else:
        ann = pd.DataFrame()
    if len(ann) > 0:
        tp = int(ann.apply(lambda r: confusion_label(r) == 'TP', axis=1).sum())
        fp = int(ann.apply(lambda r: confusion_label(r) == 'FP', axis=1).sum())
        fn = int(ann.apply(lambda r: confusion_label(r) == 'FN', axis=1).sum())
        tn = int(ann.apply(lambda r: confusion_label(r) == 'TN', axis=1).sum())
        total_ann = tp + fp + fn + tn

        def cm_badge(label, count, col):
            pct = f" ({count/total_ann*100:.0f}%)" if total_ann > 0 else ""
            return html.Div([
                html.Div(label, style={
                    'fontSize': '10px', 'fontWeight': '700',
                    'fontFamily': 'ui-monospace, monospace',
                    'color': col, 'letterSpacing': '0.06em'}),
                html.Div(f"{count:,}{pct}", style={
                    'fontSize': '18px', 'fontWeight': '800',
                    'fontFamily': 'ui-monospace, monospace', 'color': col,
                    'lineHeight': '1.1'}),
            ], style={
                'background': f'{col}1a', 'border': f'1px solid {col}55',
                'borderRadius': '10px', 'padding': '10px 18px',
                'textAlign': 'center', 'minWidth': '80px'})

        cm_row = html.Div([
            cm_badge('TP', tp, TP_COL),
            cm_badge('FP', fp, FP_COL),
            cm_badge('FN', fn, FN_COL),
            cm_badge('TN', tn, TN_COL),
            html.Div([
                html.Div('annotated', style={
                    'fontSize': '10px', 'color': INK_FAINT,
                    'fontFamily': 'ui-monospace, monospace', 'letterSpacing': '0.06em'}),
                html.Div(f"{total_ann:,}", style={
                    'fontSize': '18px', 'fontWeight': '800',
                    'fontFamily': 'ui-monospace, monospace', 'color': INK_DIM}),
            ], style={'padding': '10px 18px', 'textAlign': 'center',
                      'borderLeft': f'1px solid {LINE}', 'marginLeft': '8px'}),
        ], style={'display': 'flex', 'gap': '10px', 'alignItems': 'center',
                  'marginTop': '16px', 'paddingTop': '14px',
                  'borderTop': f'1px solid {LINE}'})
    else:
        cm_row = html.Div()

    header = html.Div([
        html.Span('DETECTIONS', style={
            'color': INK_DIM, 'fontSize': '11px', 'letterSpacing': '0.1em',
            'fontFamily': 'ui-monospace, monospace'}),
        html.Span(f'  ·  {n:,} segments in current view',
                  style={'color': INK_FAINT, 'fontSize': '11px',
                         'fontFamily': 'ui-monospace, monospace'}),
    ], style={'marginBottom': '12px'})

    return html.Div([
        header,
        legend,
        *[bar_row(sp) for sp in SHARED_SPECIES],
        cm_row,
    ], style={'background': PANEL, 'border': f'1px solid {LINE}',
              'borderRadius': '16px', 'padding': '16px 20px',
              'marginBottom': '14px'})


# ── frequency axis (log-mel) ─────────────────────────────────────────────────
def mel_axis(height_px: int = SPECTRO_H, freq_max_khz: float = FREQ_MAX_KHZ):
    def hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)
    mel_max = hz_to_mel(freq_max_khz * 1000.0)
    ticks = [t for t in [0, 1, 2, 4, 8, 12, 16, 20, 24] if t <= freq_max_khz]
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
        'borderRadius': '6px', 'padding': '6px 6px', 'fontSize': '12px',
        'textAlign': 'center', 'overflow': 'hidden', 'textOverflow': 'ellipsis',
        'whiteSpace': 'nowrap', 'fontFamily': 'ui-monospace, monospace',
        'letterSpacing': '0.02em', 'height': '18px', 'lineHeight': '18px',
        'background': (c + '22') if neutral else c,
        'color': INK_DIM if neutral else '#06101c',
        'border': (f'1px solid {c}55' if neutral else f'1px solid {c}'),
        'fontWeight': '400' if neutral else '700',
    }
    if role == 'expert':
        style['borderColor'] = (f'{YELLOW}99' if neutral else YELLOW)
        style['boxShadow'] = f'0 0 0 1px {YELLOW}44'
        if not neutral:
            style['background'] = YELLOW
            style['color'] = '#06101c'
    return html.Div(str(label), style=style)


# ── one segment column ───────────────────────────────────────────────────────
def segment_column(pos: int, row, spectro_url_base: str, glued: bool = False,
                   last: bool = False):
    has_png = bool(row.get('_has_png', False))

    if glued:
        radius = ('6px 0 0 6px' if pos == 0
                  else ('0 6px 6px 0' if last else '0'))
    else:
        radius = '6px'

    if has_png:
        fname = row.get('_png_name') or expected_png_name(row['wav_name'], row['offset_s'])
        img_style = {
            'width': '100%', 'height': f'{SPECTRO_H}px', 'objectFit': 'fill',
            'display': 'block', 'borderRadius': radius, 'background': '#040a14'}
        if glued:
            img_style['borderTop'] = f'1px solid {LINE}'
            img_style['borderBottom'] = f'1px solid {LINE}'
            if pos == 0:
                img_style['borderLeft'] = f'1px solid {LINE}'
            if last:
                img_style['borderRight'] = f'1px solid {LINE}'
        else:
            img_style['border'] = f'1px solid {LINE}'
        img = html.Img(src=f"{spectro_url_base}/{fname}", style=img_style)
    else:
        img = html.Div('no spectrogram rendered', style={
            'height': f'{SPECTRO_H}px', 'display': 'flex', 'alignItems': 'center',
            'justifyContent': 'center', 'borderRadius': radius,
            'border': f'1px dashed {LINE}', 'background': '#0a1422',
            'color': INK_FAINT, 'fontSize': '11px',
            'fontFamily': 'ui-monospace, monospace', 'textAlign': 'center'})

    if glued and not last and has_png:
        img_wrap = html.Div([
            img,
            html.Div(style={
                'position': 'absolute', 'top': '0', 'right': '0',
                'width': '2px', 'height': f'{SPECTRO_H}px',
                'background': BG, 'opacity': '0.9'}),
        ], style={'position': 'relative'})
    else:
        img_wrap = html.Div(img, style={'position': 'relative'})

    exp_detected = bool(row.get('exp_cetacean_detected', False))
    exp_label = (row.get('exp_top_species', 'no_label') if exp_detected
                 else ('background' if bool(row.get('expert_annotated', False)) else '—'))

    clock = wav_to_clock(str(row['wav_name']))
    sub = f"{offset_to_timerange(row['offset_s'])}" + (f"  ·  {clock}" if clock else '')

    # confusion badge for this segment
    cm = confusion_label(row)
    cm_colors = {'TP': TP_COL, 'FP': FP_COL, 'FN': FN_COL, 'TN': TN_COL}
    cm_badge = html.Div(cm, style={
        'fontSize': '10px', 'fontWeight': '700', 'letterSpacing': '0.06em',
        'fontFamily': 'ui-monospace, monospace',
        'color': cm_colors.get(cm, INK_FAINT),
        'background': f"{cm_colors.get(cm, LINE)}22",
        'border': f"1px solid {cm_colors.get(cm, LINE)}55",
        'borderRadius': '4px', 'padding': '2px 6px',
        'display': 'inline-block', 'marginLeft': '6px',
    }) if cm else html.Span()

    return html.Div([
        img_wrap,
        html.Div([
            html.Span(sub, style={
                'fontSize': '11px', 'color': INK_DIM,
                'fontFamily': 'ui-monospace, monospace'}),
            cm_badge,
        ], style={'display': 'flex', 'alignItems': 'center',
                  'justifyContent': 'center', 'margin': '8px 0 10px'}),
        *[chip(row.get(s, '—')) for s, _ in STRATEGIES],
        html.Div(style={'height': '1px', 'background': LINE, 'margin': '8px 2px'}),
        chip(exp_label, role='expert'),
    ],
        id={'type': 'segcol', 'index': int(pos)},
        n_clicks=0,
        style={'width': f'{SPECTRO_W}px', 'flex': '0 0 auto',
               'display': 'flex', 'flexDirection': 'column', 'gap': '8px',
               'cursor': 'pointer',
               'padding': '6px 0' if glued else '6px',
               'borderRadius': '12px', 'transition': 'background .12s'})


def rail_labels():
    rows = [('', f'{SPECTRO_H}px'), ('', '26px')]
    rows += [(name, '30px') for _, name in STRATEGIES]
    rows += [('', '9px'), ('EXPERT', '30px')]
    children = []
    for txt, h in rows:
        children.append(html.Div(txt, style={
            'height': h, 'display': 'flex', 'alignItems': 'center',
            'justifyContent': 'flex-end', 'paddingRight': '12px',
            'fontSize': '12px', 'letterSpacing': '0.06em',
            'fontFamily': 'ui-monospace, monospace',
            'color': YELLOW if txt == 'EXPERT' else INK_DIM,
            'fontWeight': '700' if txt == 'EXPERT' else '500',
            'textTransform': 'uppercase' if txt else 'none'}))
    return html.Div(children, style={
        'width': '92px', 'flex': '0 0 auto', 'display': 'flex',
        'flexDirection': 'column', 'gap': '8px', 'paddingTop': '4px',
        'marginRight': '4px'})


def overview_bars(sub: pd.DataFrame, page: int):
    n = len(sub)
    if n == 0:
        return html.Div()
    if 'cetacean_consensus' in sub.columns:
        det = sub['cetacean_consensus'].astype(bool).to_numpy()
    else:
        det = (sub['pred_consensus'].ne('background')
               & sub['pred_consensus'].ne('uncertain')).to_numpy()

    n_pages = max(1, (n + PAGE_SIZE - 1) // PAGE_SIZE)

    dates = [wav_to_date(str(w)) for w in sub['wav_name']]
    clocks = [wav_to_clock(str(w)) for w in sub['wav_name']]

    bars = []
    page_first_date = []
    for p in range(n_pages):
        a = p * PAGE_SIZE
        z = min(a + PAGE_SIZE, n)
        frac = float(det[a:z].mean()) if z > a else 0.0
        count = int(det[a:z].sum())
        is_cur = (p == page)
        h = 6 + frac * 46
        col = GREEN_LT if frac < 0.34 else (CHART if frac < 0.67 else YELLOW)
        d0 = dates[a]
        page_first_date.append(d0)
        c0 = clocks[a] or ''
        bars.append(html.Div(
            title=(f"page {p + 1} · segments {a + 1}–{z} · {d0}"
                   + (f" {c0}" if c0 else "")
                   + f" · {count}/{z - a} cetacean ({frac:.0%})"),
            id={'type': 'ovbar', 'seg': int(a)}, n_clicks=0,
            style={
                'flex': '1 1 0', 'height': f'{h:.0f}px', 'alignSelf': 'flex-end',
                'background': col if frac > 0 else LINE,
                'opacity': '1' if frac > 0 else '0.35',
                'borderRadius': '2px 2px 0 0', 'cursor': 'pointer',
                'outline': f'2px solid {INK}' if is_cur else 'none',
                'outlineOffset': '1px', 'minWidth': '1px',
                'transition': 'height .15s'}))

    bar_row = html.Div(bars, style={
        'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-end',
        'gap': '1px', 'height': '56px', 'width': '100%'})

    ticks = []
    for p in range(n_pages):
        if p == 0 or page_first_date[p] != page_first_date[p - 1]:
            left_frac = p / n_pages
            ticks.append(html.Div([
                html.Div(style={'width': '1px', 'height': '6px',
                                'background': INK_FAINT}),
                html.Div(page_first_date[p][5:] if len(page_first_date[p]) >= 5
                         else page_first_date[p],
                         style={'fontSize': '9px', 'color': INK_DIM,
                                'fontFamily': 'ui-monospace, monospace',
                                'whiteSpace': 'nowrap', 'marginTop': '1px'}),
            ], style={'position': 'absolute', 'left': f'{left_frac * 100:.3f}%',
                      'top': '0', 'textAlign': 'left'}))
    ruler = html.Div(ticks, style={
        'position': 'relative', 'width': '100%', 'height': '24px',
        'marginTop': '3px', 'borderTop': f'1px solid {LINE}'})

    uniq_dates = sorted(set(dates))
    n_days = len(uniq_dates)
    span_date = (uniq_dates[0] if n_days == 1
                 else f"{uniq_dates[0]} … {uniq_dates[-1]}")
    cl0, cl1 = clocks[0], clocks[-1]
    time_bit = f"   ·   {cl0}–{cl1}" if (cl0 and cl1) else ""
    span = (f"{n_pages:,} pages × {PAGE_SIZE} = {n:,} segments   ·   "
            f"{n_days} day{'s' if n_days != 1 else ''} ({span_date}){time_bit}")
    span_div = html.Div(span, style={
        'fontSize': '11px', 'color': INK_FAINT, 'marginTop': '6px',
        'fontFamily': 'ui-monospace, monospace'})

    return html.Div([bar_row, ruler, span_div])


def detail_panel(rows=None, prob_sets=None):
    if not rows:
        return html.Div('No segments in this view.',
                        style={'color': INK_FAINT, 'padding': '24px',
                               'fontSize': '13px',
                               'fontFamily': 'ui-monospace, monospace'})
    prob_sets = prob_sets or [('prob_', dict(STRATEGIES).get(PROB_SOURCE, PROB_SOURCE),
                               GREEN, GREEN_LT)]

    def prob_bar(val, c0, c1, thresh=None):
        v = float(val) if pd.notna(val) else 0.0
        pct = f'{min(1.0, v) * 100:.1f}%'
        cell = [html.Div(style={
            'width': pct, 'height': '9px', 'borderRadius': '4px',
            'background': f'linear-gradient(90deg,{c0},{c1})'})]
        if thresh is not None and pd.notna(thresh):
            t = min(1.0, float(thresh))
            cell.append(html.Div(title=f'pr threshold {t:.3f}', style={
                'position': 'absolute', 'left': f'{t * 100:.1f}%', 'top': '-2px',
                'width': '2px', 'height': '13px', 'background': INK}))
        return html.Td(html.Div(cell, style={
            'position': 'relative', 'width': '100%', 'height': '9px'}),
            style={'padding': '5px 6px', 'width': '18%'})

    def num_td(val, color):
        return html.Td(f"{val:.3f}" if pd.notna(val) else '—',
                       style={'padding': '5px 8px', 'color': color,
                              'fontSize': '11px', 'width': '44px',
                              'whiteSpace': 'nowrap'})

    cols_per_seg = len(prob_sets) + 1
    span_per_seg = 2 * cols_per_seg

    def sticky_name_td(text, color, weight):
        return html.Td(text, style={'padding': '5px 12px 5px 0', 'color': color,
                                    'fontWeight': weight, 'position': 'sticky',
                                    'left': '0', 'background': PANEL_2,
                                    'zIndex': '1', 'width': '60px',
                                    'whiteSpace': 'nowrap'})

    def species_row(sp):
        cells = [sticky_name_td(sp, INK, '700')]
        for i, (_pos, row) in enumerate(rows):
            for prefix, _label, c0, c1 in prob_sets:
                val = row.get(f'{prefix}{sp}', float('nan'))
                thr = row.get(f'pr_threshold_{sp}') if prefix == 'prob_pr_' else None
                cells.append(prob_bar(val, c0, c1, thr))
                cells.append(num_td(val, INK_DIM))
            ev = row.get(f'exp_{sp}', float('nan'))
            cells.append(prob_bar(ev, CHART, YELLOW))
            cells.append(num_td(ev, YELLOW))
            if i < len(rows) - 1:
                cells.append(html.Td(style={'borderRight': f'1px solid {LINE}',
                                            'width': '0', 'padding': '0'}))
        return html.Tr(cells)

    body = [species_row(sp) for sp in SHARED_SPECIES]

    have_bg = any(any(c in r.index for c in ('prob_bg', 'prob_background'))
                  for _p, r in rows)
    if have_bg:
        total_span = 1 + len(rows) * (span_per_seg + 1)
        body.append(html.Tr([html.Td(html.Div(style={
            'height': '1px', 'background': LINE, 'margin': '4px 0'}),
            colSpan=total_span)]))
        bg_cells = [sticky_name_td('background', INK_DIM, '700')]
        for i, (_pos, row) in enumerate(rows):
            bg_col = next((c for c in ('prob_bg', 'prob_background')
                           if c in row.index), None)
            bgv = row.get(bg_col, float('nan')) if bg_col else float('nan')
            for j, (_p, _l, c0, c1) in enumerate(prob_sets):
                if j == 0:
                    bg_cells.append(prob_bar(bgv, c0, c1))
                    bg_cells.append(num_td(bgv, INK_DIM))
                else:
                    bg_cells.append(html.Td())
                    bg_cells.append(num_td(float('nan'), INK_DIM))
            bg_cells.append(html.Td())
            bg_cells.append(num_td(float('nan'), YELLOW))
            if i < len(rows) - 1:
                bg_cells.append(html.Td(style={'borderRight': f'1px solid {LINE}',
                                               'width': '0', 'padding': '0'}))
        body.append(html.Tr(bg_cells))

    seg_header_cells = [html.Th('', style={'position': 'sticky', 'left': '0',
                                           'background': PANEL_2, 'zIndex': '1',
                                           'width': '60px'})]
    sub_header_cells = [html.Th('species', style={
        'textAlign': 'left', 'color': INK_FAINT, 'fontWeight': '500',
        'padding': '0 12px 8px 0', 'position': 'sticky', 'left': '0',
        'background': PANEL_2, 'zIndex': '1', 'width': '60px'})]
    for i, (_pos, row) in enumerate(rows):
        clock = wav_to_clock(str(row['wav_name']))
        seg_title = (f"{offset_to_timerange(row['offset_s'])}"
                     + (f"  ·  {clock}" if clock else ''))
        seg_header_cells.append(html.Th(seg_title, colSpan=span_per_seg, style={
            'textAlign': 'left', 'color': INK, 'fontWeight': '700',
            'padding': '0 6px 6px', 'fontSize': '12px',
            'borderBottom': f'1px solid {LINE}'}))
        for _prefix, label, c0, _c1 in prob_sets:
            sub_header_cells.append(html.Th(label, colSpan=2, style={
                'textAlign': 'left', 'color': c0, 'fontWeight': '600',
                'padding': '4px 6px 8px', 'fontSize': '10px'}))
        sub_header_cells.append(html.Th('expert', colSpan=2, style={
            'textAlign': 'left', 'color': YELLOW, 'fontWeight': '600',
            'padding': '4px 6px 8px', 'fontSize': '10px'}))
        if i < len(rows) - 1:
            seg_header_cells.append(html.Th(style={'borderRight': f'1px solid {LINE}',
                                                   'width': '0', 'padding': '0'}))
            sub_header_cells.append(html.Th(style={'borderRight': f'1px solid {LINE}',
                                                   'width': '0', 'padding': '0'}))

    first_row = rows[0][1]
    fdate = wav_to_date(str(first_row['wav_name']))
    caption = html.Div([
        html.Span(str(first_row['wav_name']), style={'color': INK, 'fontWeight': '700'}),
        html.Span(f"   ·   {fdate}   ·   "
                  f"{len(rows)} segment{'s' if len(rows) != 1 else ''} in window",
                  style={'color': INK_DIM}),
    ], style={'fontFamily': 'ui-monospace, monospace', 'fontSize': '13px',
              'marginBottom': '14px'})

    return html.Div([
        caption,
        html.Div(html.Table([
            html.Thead([html.Tr(seg_header_cells), html.Tr(sub_header_cells)]),
            html.Tbody(body),
        ], style={'borderCollapse': 'collapse', 'fontSize': '12px',
                  'fontFamily': 'ui-monospace, monospace',
                  'width': '100%', 'tableLayout': 'fixed'}),
            style={'overflowX': 'auto', 'width': '100%'}),
    ], style={'padding': '20px 22px'})


# ── app factory ──────────────────────────────────────────────────────────────
def make_app(df, spectro_dir, freq_max_khz):
    app = Dash(__name__)
    server = app.server

    png_index = {p.name.lower(): p for p in spectro_dir.glob('*.png')} \
        if spectro_dir.exists() else {}

    df = df.copy()
    df['_png_name'] = [expected_png_name(r['wav_name'], r['offset_s'])
                       for _, r in df.iterrows()]
    df['_has_png'] = [nm.lower() in png_index for nm in df['_png_name']]
    print(f"Indexed {len(png_index):,} PNG files; "
          f"{df['_has_png'].sum():,} of {len(df):,} segments have a rendered spectrogram.")

    prob_sets = detect_prob_sets(df)
    print("Probability sets detected for detail panel: "
          + ", ".join(lbl for _p, lbl, _a, _b in prob_sets))

    SPECTRO_URL = '/spectro'
    _path_cache: dict[str, Path] = {}

    @server.route(f'{SPECTRO_URL}/<path:fname>')
    def _serve_spectro(fname):
        key = fname.lower()
        p = _path_cache.get(key)
        if p is None:
            p = png_index.get(key)
            if p is None:
                flask.abort(404)
            _path_cache[key] = p
        resp = flask.send_file(p, mimetype='image/png')
        resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
        return resp

    app.index_string = (
        '<!DOCTYPE html><html><head>{%metas%}<title>Cetacean Inspector</title>'
        '{%favicon%}{%css%}<style>'
        f'body{{margin:0;background:{BG};}}'
        f'::selection{{background:{GREEN}55;}}'
        f'div[class*="-control"]{{background-color:#ffffff!important;'
        f'border-color:#c3cdd9!important;color:#10212f!important;'
        f'box-shadow:none!important;border-radius:8px!important;}}'
        f'div[class*="-control"]:hover{{border-color:{GREEN}!important;}}'
        f'div[class*="-menu"]{{background-color:#ffffff!important;color:#10212f!important;'
        f'border:1px solid #c3cdd9!important;z-index:30!important;}}'
        f'div[class*="-MenuList"]{{background-color:#ffffff!important;}}'
        f'div[class*="-option"]{{background-color:#ffffff!important;color:#10212f!important;}}'
        f'div[class*="-option"]:hover{{background-color:#eef3f8!important;color:#0a1422!important;}}'
        f'div[class*="-singleValue"]{{color:#10212f!important;}}'
        f'div[class*="-placeholder"]{{color:#5e779a!important;}}'
        f'div[class*="-input"]{{color:#10212f!important;}}'
        f'div[class*="-ValueContainer"],div[class*="-valueContainer"]'
        f'{{background-color:#ffffff!important;}}'
        f'div[class*="-control"] input{{color:#10212f!important;background:transparent!important;'
        f'border:none!important;box-shadow:none!important;}}'
        f'div[class*="-indicatorContainer"] svg{{fill:#5e779a!important;}}'
        f'div[class*="-indicatorSeparator"]{{background-color:#c3cdd9!important;}}'
        f'.Select-control,.Select-menu-outer,.Select-menu,.Select-multi-value-wrapper,'
        f'.is-open>.Select-control,.is-focused:not(.is-open)>.Select-control'
        f'{{background-color:#ffffff!important;border-color:#c3cdd9!important;'
        f'color:#10212f!important;border-radius:8px!important;}}'
        f'.Select-value-label,.has-value.Select--single>.Select-control .Select-value .Select-value-label,'
        f'.Select-input>input{{color:#10212f!important;}}'
        f'.Select-placeholder{{color:#5e779a!important;}}'
        f'.Select-option,.VirtualizedSelectOption{{background-color:#ffffff!important;color:#10212f!important;}}'
        f'.Select-option.is-focused,.VirtualizedSelectFocusedOption'
        f'{{background-color:#eef3f8!important;color:#0a1422!important;}}'
        f'.Select-arrow{{border-color:#5e779a transparent transparent!important;}}'
        f'.is-open .Select-arrow{{border-color:transparent transparent #5e779a!important;}}'
        f'input[type="number"]{{background:{PANEL}!important;color:{INK}!important;'
        f'border:1px solid {LINE}!important;border-radius:8px!important;'
        f'padding:6px 8px!important;font-family:ui-monospace,monospace!important;}}'
        f'.ci-btn{{background:{PANEL_2};color:{INK};border:1px solid {LINE};border-radius:8px;'
        f'padding:7px 14px;font-family:ui-monospace,monospace;font-size:13px;cursor:pointer;'
        f'transition:background .15s,border-color .15s;}}'
        f'.ci-btn:hover{{background:{GREEN};border-color:{GREEN_LT};color:#06101c;}}'
        # confusion filter buttons
        f'.cf-btn{{border-radius:8px;padding:6px 14px;font-family:ui-monospace,monospace;'
        f'font-size:12px;font-weight:700;letter-spacing:0.04em;cursor:pointer;'
        f'border:1px solid;transition:background .15s,opacity .15s;}}'
        f'.cf-btn-tp{{background:{TP_COL}22;color:{TP_COL};border-color:{TP_COL}66;}}'
        f'.cf-btn-tp:hover,.cf-btn-tp.active{{background:{TP_COL};color:#06101c;border-color:{TP_COL};}}'
        f'.cf-btn-fp{{background:{FP_COL}22;color:{FP_COL};border-color:{FP_COL}66;}}'
        f'.cf-btn-fp:hover,.cf-btn-fp.active{{background:{FP_COL};color:#06101c;border-color:{FP_COL};}}'
        f'.cf-btn-fn{{background:{FN_COL}22;color:{FN_COL};border-color:{FN_COL}66;}}'
        f'.cf-btn-fn:hover,.cf-btn-fn.active{{background:{FN_COL};color:#06101c;border-color:{FN_COL};}}'
        f'.cf-btn-tn{{background:{TN_COL}22;color:{TN_COL};border-color:{TN_COL}66;}}'
        f'.cf-btn-tn:hover,.cf-btn-tn.active{{background:{TN_COL};color:{INK};border-color:{TN_COL};}}'
        f'#filmstrip>div:hover{{background:{PANEL_2};}}'
        f'::-webkit-scrollbar{{height:10px;width:10px;}}'
        f'::-webkit-scrollbar-track{{background:{BG};}}'
        f'::-webkit-scrollbar-thumb{{background:{LINE};border-radius:5px;}}'
        f'::-webkit-scrollbar-thumb:hover{{background:{GREEN};}}'
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
        ], style={'marginBottom': '12px'}),

        html.Div(id='status', style={
            'color': INK_FAINT, 'fontSize': '12px', 'marginBottom': '16px',
            'fontFamily': 'ui-monospace, monospace'}),

        # ── filter row ────────────────────────────────────────────────────────
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
                             {'label': 'true positives  (TP)', 'value': 'true_positive'},
                             {'label': 'false positives (FP)', 'value': 'false_positive'},
                             {'label': 'false negatives (FN)', 'value': 'false_negative'},
                             {'label': 'true negatives  (TN)', 'value': 'true_negative'},
                         ]),
            html.Label('species', style={**label_css, 'marginLeft': '14px'}),
            dcc.Dropdown(id='species', clearable=False, value='ALL',
                         options=species_opts, style={'width': '150px'}),
            # confusion quick-buttons
            html.Div([
                html.Span('jump to:', style={**label_css, 'marginLeft': '14px',
                                             'marginRight': '8px'}),
                html.Button('TP', id='btn-tp', n_clicks=0,
                             className='cf-btn cf-btn-tp'),
                html.Button('FP', id='btn-fp', n_clicks=0,
                             className='cf-btn cf-btn-fp'),
                html.Button('FN', id='btn-fn', n_clicks=0,
                             className='cf-btn cf-btn-fn'),
                html.Button('TN', id='btn-tn', n_clicks=0,
                             className='cf-btn cf-btn-tn'),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px'}),
            html.Div(style={'flex': '1 1 auto'}),
            html.Button('◀', id='prev', n_clicks=0, className='ci-btn'),
            dcc.Input(id='jump', type='number', value=1, min=1, step=1,
                      style={'width': '72px', 'margin': '0 8px', 'textAlign': 'center'}),
            html.Button('go', id='gobtn', n_clicks=0, className='ci-btn'),
            html.Button('▶', id='next', n_clicks=0, className='ci-btn',
                        style={'marginLeft': '8px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap',
                  'gap': '6px', 'marginBottom': '16px'}),

        # ── species stats panel (always visible, above overview) ──────────────
        html.Div(id='species-stats'),

        # ── overview bar ──────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span('OVERVIEW', style={
                    'color': INK_DIM, 'fontSize': '11px', 'letterSpacing': '0.1em',
                    'fontFamily': 'ui-monospace, monospace'}),
                html.Span('one bar = one page (3 segments) · cetacean density · '
                          'dates marked below · click to jump',
                          style={'color': INK_FAINT, 'fontSize': '11px',
                                 'marginLeft': '10px',
                                 'fontFamily': 'ui-monospace, monospace'}),
            ], style={'marginBottom': '8px'}),
            html.Div(id='overview'),
        ], style={'background': PANEL, 'border': f'1px solid {LINE}',
                  'borderRadius': '16px', 'padding': '16px 20px',
                  'marginBottom': '14px'}),

        html.Div([
            html.Div([
                rail_labels(),
                mel_axis(SPECTRO_H, freq_max_khz),
                html.Div(id='filmstrip', style={
                    'display': 'flex', 'flexDirection': 'row', 'gap': '0',
                    'overflowX': 'auto', 'flex': '1 1 auto', 'paddingTop': '4px'}),
            ], style={'display': 'flex', 'flexDirection': 'row',
                      'background': PANEL, 'border': f'1px solid {LINE}',
                      'borderRadius': '16px', 'padding': '20px'}),

            html.Div(id='detail', style={
                'marginTop': '8px', 'background': PANEL_2,
                'border': f'1px solid {LINE}', 'borderRadius': '16px',
                'minHeight': '90px'}),
        ], style={'display': 'flex', 'flexDirection': 'column',
                  'alignItems': 'stretch', 'width': 'fit-content',
                  'maxWidth': '100%', 'marginBottom': '16px'}),

        dcc.Store(id='page', data=0),
    ], style={'fontFamily': 'system-ui, -apple-system, sans-serif',
              'padding': '28px 32px', 'maxWidth': '100%', 'boxSizing': 'border-box',
              'background': BG, 'minHeight': '100vh', 'color': INK})

    # ── confusion quick-buttons set the mode dropdown ─────────────────────────
    @app.callback(
        Output('mode', 'value'),
        Input('btn-tp', 'n_clicks'),
        Input('btn-fp', 'n_clicks'),
        Input('btn-fn', 'n_clicks'),
        Input('btn-tn', 'n_clicks'),
        State('mode', 'value'),
        prevent_initial_call=True,
    )
    def confusion_buttons(tp_c, fp_c, fn_c, tn_c, current_mode):
        mapping = {
            'btn-tp': 'true_positive',
            'btn-fp': 'false_positive',
            'btn-fn': 'false_negative',
            'btn-tn': 'true_negative',
        }
        trig = ctx.triggered_id
        return mapping.get(trig, current_mode)

    # ── main render callback ──────────────────────────────────────────────────
    @app.callback(
        Output('filmstrip', 'children'),
        Output('status', 'children'),
        Output('page', 'data'),
        Output('jump', 'value'),
        Output('overview', 'children'),
        Output('detail', 'children'),
        Output('species-stats', 'children'),
        Input('prev', 'n_clicks'), Input('next', 'n_clicks'),
        Input('gobtn', 'n_clicks'), Input('mode', 'value'),
        Input('species', 'value'),
        Input({'type': 'ovbar', 'seg': dash.ALL}, 'n_clicks'),
        State('page', 'data'), State('jump', 'value'),
    )
    def render(prev_c, next_c, go_c, mode, species, _ovclicks, page, jump):
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
        elif isinstance(trig, dict) and trig.get('type') == 'ovbar':
            page = min(n_pages - 1, int(trig['seg']) // PAGE_SIZE)
        page = min(max(0, page or 0), n_pages - 1)

        stats = species_stats_panel(sub)

        if n == 0:
            empty = [html.Div('No segments match this view.', style={
                        'padding': '40px', 'color': INK_FAINT,
                        'fontFamily': 'ui-monospace, monospace'})]
            return (empty, 'No segments match the current view.', page, 1,
                    html.Div(), detail_panel(None, prob_sets), stats)

        lo, hi = page * PAGE_SIZE, min(page * PAGE_SIZE + PAGE_SIZE, n)
        window = sub.iloc[lo:hi]
        n_in_window = len(window)
        cols = [segment_column(p, r, SPECTRO_URL, glued=True,
                               last=(p == n_in_window - 1))
                for p, (_, r) in enumerate(window.iterrows())]
        status = (f"segments {lo + 1:,}–{hi:,} of {n:,}   ·   "
                  f"page {page + 1} / {n_pages}   ·   view: {mode}   ·   species: {species}")
        detail_rows = [(p, r) for p, (_, r) in enumerate(window.iterrows())]
        detail = detail_panel(detail_rows, prob_sets)
        return cols, status, page, page + 1, overview_bars(sub, page), detail, stats

    return app


def main():
    global FREQ_MAX_KHZ, PROB_SOURCE
    ap = argparse.ArgumentParser(description='Cetacean inspector Dash app (dark)')
    ap.add_argument('--csv', default='arbas_comparison_5s_v3.csv')
    ap.add_argument('--spectrograms', required=True)
    ap.add_argument('--port', type=int, default=8050)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--freq-max-khz', type=float, default=16.0)
    ap.add_argument('--prob-source', default=PROB_SOURCE,
                    help='label for the legacy single-bar fallback only')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    FREQ_MAX_KHZ = args.freq_max_khz
    PROB_SOURCE = args.prob_source

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