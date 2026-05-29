"""
app.py
======
Dash app for interactive cetacean detection prediction review.
Reads the L4 prediction CSV/parquet produced by notebook 7.

Run:
    python app.py --predictions /data2/mromaniuc/cet-det/inference_arbas/predictions/arbas_predictions_l4.parquet
    python app.py --predictions /data2/mromaniuc/cet-det/inference_harrapatu/predictions/harrapatu_predictions_l4.parquet --port 8051

    # VSCode SSH will auto-forward the port. Or manually:
    #   ssh -L 8050:localhost:8050 user@server
    # then open http://localhost:8050
"""

from __future__ import annotations
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx, ALL
from dash.exceptions import PreventUpdate
from flask import abort, send_file

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--predictions', required=True,
                    help='Path to predictions parquet or CSV (arbas_predictions_l4.*)')
parser.add_argument('--port',  type=int, default=8050)
parser.add_argument('--host',  default='0.0.0.0')
parser.add_argument('--max-scatter', type=int, default=5_000,
                    help='Max dots in the timeline scatter (sampled per species)')
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = [
    'Balaenoptera_acutorostrata', 'Balaenoptera_physalus', 'Delphinidae_unknown',
    'Delphinus_delphis', 'Globicephala_melas', 'Grampus_griseus', 'Orcinus_orca',
    'Physeter_macrocephalus', 'Stenella_coeruleoalba', 'Tursiops_truncatus', 'background',
]
SPECIES = [c for c in CLASSES if c != 'background']
RELIABLE = {'Tursiops_truncatus', 'Delphinus_delphis', 'Orcinus_orca', 'Delphinidae_unknown'}
SP_SHORT = {
    'Balaenoptera_acutorostrata': 'B. acutorostrata',
    'Balaenoptera_physalus':      'B. physalus',
    'Delphinidae_unknown':        'Delphinidae sp.',
    'Delphinus_delphis':          'D. delphis',
    'Globicephala_melas':         'G. melas',
    'Grampus_griseus':            'G. griseus',
    'Orcinus_orca':               'O. orca',
    'Physeter_macrocephalus':     'P. macrocephalus',
    'Stenella_coeruleoalba':      'S. coeruleoalba',
    'Tursiops_truncatus':         'T. truncatus',
    'background':                 'background',
}
SP_COLORS = {
    'Tursiops_truncatus':         '#378ADD',
    'Delphinus_delphis':          '#1D9E75',
    'Orcinus_orca':               '#D4537E',
    'Delphinidae_unknown':        '#7F77DD',
    'Globicephala_melas':         '#BA7517',
    'Grampus_griseus':            '#639922',
    'Balaenoptera_physalus':      '#D85A30',
    'Balaenoptera_acutorostrata': '#993556',
    'Physeter_macrocephalus':     '#888780',
    'Stenella_coeruleoalba':      '#E24B4A',
    'background':                 '#B4B2A9',
}
PR_THRESHOLDS = {
    'background': 0.000, 'Orcinus_orca': 0.812, 'Delphinidae_unknown': 0.834,
    'Stenella_coeruleoalba': 0.863, 'Grampus_griseus': 0.872,
    'Tursiops_truncatus': 0.967, 'Delphinus_delphis': 0.996,
    'Balaenoptera_physalus': 0.999, 'Globicephala_melas': 1.0,
    'Balaenoptera_acutorostrata': 1.0, 'Physeter_macrocephalus': 1.0,
}
STRAT_COLS = ['pred_argmax', 'pred_temp', 'pred_vec', 'pred_pr']
STRAT_LABELS = {
    'pred_argmax': 'Argmax',
    'pred_temp':   'Temperature',
    'pred_vec':    'Vector scaling',
    'pred_pr':     'PR-threshold',
}

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
p = Path(args.predictions)
print(f"Loading {p} ...")
if p.suffix == '.parquet':
    df = pd.read_parquet(p)
else:
    df = pd.read_csv(p)
print(f"  {len(df):,} rows, {len(df.columns)} cols")

# Derive wav_name if missing
if 'wav_name' not in df.columns:
    df['wav_name'] = df['wav_path'].apply(lambda x: Path(str(x)).name if pd.notna(x) else None)

# Parse timestamps from SoundTrap filenames: recorderID.YYMMDDHHMMSS.wav
_TS_RE = re.compile(r'\.(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.')

def _parse_file_ts(name: str | None) -> datetime | None:
    if not name:
        return None
    m = _TS_RE.search(str(name))
    if not m:
        return None
    yy, mo, dy, hh, mm, ss = m.groups()
    try:
        return datetime(2000 + int(yy), int(mo), int(dy),
                        int(hh), int(mm), int(ss), tzinfo=timezone.utc)
    except ValueError:
        return None

print("  Parsing timestamps ...")
df['_file_ts'] = df['wav_name'].map(_parse_file_ts)
df['_seg_ts']  = df.apply(
    lambda r: r['_file_ts'] + pd.Timedelta(seconds=float(r['offset_s']))
    if pd.notna(r['_file_ts']) else pd.NaT, axis=1
)
df['_seg_ts'] = pd.to_datetime(df['_seg_ts'], utc=True, errors='coerce')
df['_hour']   = df['_seg_ts'].dt.hour
df['_date']   = df['_seg_ts'].dt.date

n_ts = df['_seg_ts'].notna().sum()
print(f"  Timestamps parsed: {n_ts:,} / {len(df):,}")

# Ensure prob columns exist (fill 0 if missing)
for sp in CLASSES:
    col = f'prob_{sp}'
    if col not in df.columns:
        df[col] = 0.0

if 'max_species_prob' not in df.columns:
    prob_cols = [f'prob_{sp}' for sp in SPECIES]
    df['max_species_prob'] = df[prob_cols].max(axis=1)
if 'max_species_class' not in df.columns:
    prob_cols = [f'prob_{sp}' for sp in SPECIES]
    df['max_species_class'] = df[prob_cols].idxmax(axis=1).str.replace('prob_', '', regex=False)

# Infer deployment name
deploy_name = p.stem.replace('_predictions_l4', '').upper()
N_TOTAL = len(df)
recorders = sorted(df['wav_name'].dropna().apply(
    lambda x: x.split('.')[0] if '.' in str(x) else '?'
).unique().tolist())
print(f"  Recorders: {recorders}")
print(f"  Date range: {df['_date'].min()} → {df['_date'].max()}")
print(f"  Ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Plotly themes
# ─────────────────────────────────────────────────────────────────────────────
THEMES = {
    'light': dict(paper='#ffffff', plot='#ffffff', font='#111827',
                  grid='#f3f4f6', axis='#e5e7eb', tick='#9ca3af',
                  leg_bg='rgba(255,255,255,0.9)', leg_border='#e5e7eb'),
    'dark':  dict(paper='#0e1014', plot='#0e1014', font='#e5e7eb',
                  grid='#1c2027', axis='#262b33', tick='#6b7280',
                  leg_bg='rgba(22,25,31,0.9)', leg_border='#262b33'),
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _axis_style(thm: dict) -> dict:
    return dict(gridcolor=thm['grid'], zerolinecolor=thm['axis'],
                color=thm['tick'], linecolor=thm['axis'])

def _filter_df(strat: str, threshold: float, species: list[str]) -> pd.DataFrame:
    """Return detection rows matching current filter state."""
    sp_set = set(species)
    mask = (
        (df[strat].notna()) &
        (df[strat] != 'background') &
        (df[strat].isin(sp_set)) &
        (df['max_species_prob'] >= threshold)
    )
    return df[mask]

def _status_pills(items: list[tuple[str, str, bool]]) -> list:
    pills = []
    for label, value, accent in items:
        pills.append(html.Div(
            className=f'status-pill{"  accent" if accent else ""}',
            children=[html.Span(label, className='pill-label'),
                      html.Span(value,  className='pill-value')],
        ))
    return pills

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = Dash(__name__, title=f'Cetacean Predictor · {deploy_name}',
           assets_folder='assets', update_title=None,
           suppress_callback_exceptions=False)
server = app.server

app.index_string = '''<!DOCTYPE html>
<html data-theme="light">
<head>
    {%metas%}
    <title>{%title%}</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    {%favicon%}
    {%css%}
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
def _species_chips():
    chips = []
    for sp in SPECIES:
        reliable = sp in RELIABLE
        chips.append(
            html.Button(
                [('⚠ ' if not reliable else '') + SP_SHORT[sp]],
                id={'type': 'sp-chip', 'sp': sp},
                className='chip on' + ('' if reliable else ' unreliable'),
                title=('Reliable' if reliable else 'Low training samples — treat with caution'),
                n_clicks=0,
            )
        )
    return chips


app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div(className='app-header', children=[

        html.Div(className='app-topbar', children=[
            html.Span(f'🐋  Cetacean Predictor · {deploy_name}', className='app-title'),
            html.Div(className='divider'),

            html.Div(className='control-group', children=[
                html.Span('Strategy', className='label'),
                dcc.Dropdown(
                    id='strat',
                    value='pred_pr',
                    options=[{'label': v, 'value': k} for k, v in STRAT_LABELS.items()],
                    clearable=False, style={'width': '180px'},
                ),
            ]),

            html.Div(className='control-group', children=[
                html.Span('View', className='label'),
                html.Div(className='segmented', children=[
                    dcc.RadioItems(
                        id='view',
                        value='timeline',
                        inline=True,
                        options=[
                            {'label': 'Timeline',  'value': 'timeline'},
                            {'label': 'Diel',      'value': 'diel'},
                            {'label': 'Histogram', 'value': 'hist'},
                            {'label': 'Summary',   'value': 'summary'},
                        ],
                    ),
                ]),
            ]),

            html.Button('☀  Light', id='theme-toggle', className='theme-toggle', n_clicks=0),
        ]),

        html.Div(className='app-slider-row', children=[
            html.Span('Prob threshold', className='label'),
            html.Div(className='slider-wrap', children=[
                dcc.Slider(
                    id='threshold', min=0.0, max=1.0, step=0.01, value=0.50,
                    marks={0: '0', 0.5: '0.5', 0.8: '0.8', 0.9: '0.9', 0.96: '0.96', 1.0: '1.0'},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    updatemode='drag',
                ),
            ]),
        ]),
    ]),

    # ── Status bar ──────────────────────────────────────────────────────────
    html.Div(id='status-bar', className='status-bar'),

    # ── Main split ──────────────────────────────────────────────────────────
    html.Div(className='main-split', children=[

        # Left: plot
        html.Div(className='plot-pane', children=[
            dcc.Loading(
                dcc.Graph(id='main-plot', style={'height': '100%'},
                          config={'displayModeBar': True, 'scrollZoom': True,
                                  'displaylogo': False}),
                type='circle', color='#3b82f6',
            ),
        ]),

        # Right: sidebar
        html.Div(className='detail-pane', children=[

            html.Div(className='sb-section', children=[
                html.Div('Species filter', className='sb-label'),
                html.Div(id='species-chips-wrap',
                         className='chip-grid',
                         children=_species_chips()),
                html.Div(style={'marginTop': '8px', 'display': 'flex', 'gap': '6px'}, children=[
                    html.Button('all',           id='sp-all',      className='mini-btn', n_clicks=0),
                    html.Button('none',          id='sp-none',     className='mini-btn', n_clicks=0),
                    html.Button('reliable only', id='sp-reliable', className='mini-btn', n_clicks=0),
                ]),
                html.Div('⚠ = low training samples', className='hint'),
            ]),

            html.Div(className='sb-section', children=[
                html.H4('Detection detail'),
                html.Div(id='detail-panel',
                         children=[html.Div('Click a point in the timeline to inspect it.',
                                            className='detail-empty')]),
            ]),
        ]),
    ]),

    # Hidden stores
    dcc.Store(id='theme-store', data='light', storage_type='local'),
    dcc.Store(id='active-species', data=SPECIES),
])


# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('theme-store',  'data'),
    Output('theme-toggle', 'children'),
    Input('theme-toggle',  'n_clicks'),
    State('theme-store',   'data'),
    prevent_initial_call=False,
)
def toggle_theme(n, current):
    current = current or 'light'
    if not n:
        return current, ('☾  Dark' if current == 'dark' else '☀  Light')
    nxt = 'dark' if current == 'light' else 'light'
    return nxt, ('☾  Dark' if nxt == 'dark' else '☀  Light')

# Apply theme to <html data-theme=...> via inline JS (no clientside_callback needed)
app.clientside_callback(
    """
    function(theme) {
        document.documentElement.setAttribute('data-theme', theme || 'light');
        return theme;
    }
    """,
    Output('theme-store', 'data', allow_duplicate=True),
    Input('theme-store', 'data'),
    prevent_initial_call='initial_duplicate',
)


# ─────────────────────────────────────────────────────────────────────────────
# Species chip state → active-species store
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('active-species', 'data'),
    Output({'type': 'sp-chip', 'sp': ALL}, 'className'),
    Input({'type': 'sp-chip', 'sp': ALL}, 'n_clicks'),
    Input('sp-all',      'n_clicks'),
    Input('sp-none',     'n_clicks'),
    Input('sp-reliable', 'n_clicks'),
    State('active-species', 'data'),
    prevent_initial_call=True,
)
def update_species(chip_clicks, _all, _none, _reliable, current):
    current   = set(current or SPECIES)
    triggered = ctx.triggered_id

    if triggered == 'sp-all':
        current = set(SPECIES)
    elif triggered == 'sp-none':
        current = set()
    elif triggered == 'sp-reliable':
        current = set(RELIABLE)
    elif isinstance(triggered, dict) and triggered.get('type') == 'sp-chip':
        sp = triggered['sp']
        if sp in current: current.discard(sp)
        else:             current.add(sp)

    classes = []
    for sp in SPECIES:
        reliable = sp in RELIABLE
        on = sp in current
        cls = 'chip' + (' on' if on else '') + ('' if reliable else ' unreliable')
        classes.append(cls)

    return list(current), classes


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('main-plot',  'figure'),
    Output('status-bar', 'children'),
    Input('strat',         'value'),
    Input('threshold',     'value'),
    Input('active-species','data'),
    Input('view',          'value'),
    Input('theme-store',   'data'),
)
def update_plot(strat, threshold, species, view, theme):
    thm = THEMES.get(theme or 'light', THEMES['light'])
    species = species or []

    dets = _filter_df(strat, threshold or 0, species)

    pills = [
        ('corpus',     f'{N_TOTAL:,}',          False),
        ('strategy',   STRAT_LABELS.get(strat, strat), True),
        ('threshold',  f'≥{threshold:.2f}',     False),
        ('detections', f'{len(dets):,}',         True),
    ]
    status = _status_pills(pills)

    base_layout = dict(
        margin=dict(l=0, r=0, t=12, b=0),
        paper_bgcolor=thm['paper'],
        plot_bgcolor=thm['plot'],
        font=dict(color=thm['font'],
                  family='-apple-system, BlinkMacSystemFont, sans-serif',
                  size=11),
        legend=dict(itemsizing='constant', font=dict(size=10),
                    bgcolor=thm['leg_bg'], bordercolor=thm['leg_border'],
                    borderwidth=1, x=1, y=1, xanchor='right', yanchor='top'),
        hoverlabel=dict(bgcolor=thm['paper'], bordercolor=thm['leg_border'],
                        font=dict(size=11)),
    )

    if view == 'timeline':
        fig = _render_timeline(dets, strat, thm, base_layout)
    elif view == 'diel':
        fig = _render_diel(dets, strat, thm, base_layout)
    elif view == 'hist':
        fig = _render_hist(threshold, species, thm, base_layout)
    else:
        fig = _render_summary(dets, strat, thm, base_layout)

    return fig, status


def _render_timeline(dets: pd.DataFrame, strat: str, thm: dict, base: dict) -> go.Figure:
    sub = dets.dropna(subset=['_seg_ts'])
    if sub.empty:
        return _empty_fig('No detections — adjust strategy, threshold, or species filter', thm, base)

    active_sp = [sp for sp in SPECIES if (sub[strat] == sp).any()]
    sp_y = {sp: i for i, sp in enumerate(active_sp)}

    # Sample per species to keep rendering fast
    per_sp = max(1, args.max_scatter // max(len(active_sp), 1))
    traces = []
    for sp in active_sp:
        s = sub[sub[strat] == sp]
        if len(s) > per_sp:
            s = s.sample(n=per_sp, random_state=42)
        col = SP_COLORS[sp]
        reliable = sp in RELIABLE
        hover = (
            f'<b>{SP_SHORT[sp]}</b><br>'
            'time: %{x|%Y-%m-%d %H:%M:%S}<br>'
            'prob: %{marker.size:.2f}<br>'
            'file: %{customdata[0]}<br>'
            'offset: %{customdata[1]}s<br>'
            'Perch: %{customdata[2]}<extra></extra>'
        )
        traces.append(go.Scattergl(
            x=s['_seg_ts'],
            y=[sp_y[sp]] * len(s),
            mode='markers',
            name=f'{SP_SHORT[sp]}  ({(sub[strat]==sp).sum():,})',
            marker=dict(
                size=s['max_species_prob'].clip(0.01, 1.0) * 14,
                color=col,
                opacity=0.65 if reliable else 0.45,
                line=dict(width=0),
                sizemode='diameter',
            ),
            customdata=s[['wav_name', 'offset_s', 'top_predicted_class']].values,
            hovertemplate=hover,
            yaxis='y',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        **base,
        yaxis=dict(
            tickvals=list(sp_y.values()),
            ticktext=[SP_SHORT[sp] + ('' if sp in RELIABLE else ' ⚠') for sp in active_sp],
            **_axis_style(thm),
        ),
        xaxis=dict(title='Time (UTC)', **_axis_style(thm)),
        clickmode='event+select',
    )
    n_shown = sum(min(len(sub[sub[strat]==sp]), max(1, args.max_scatter//max(len(active_sp),1)))
                  for sp in active_sp)
    if n_shown < len(sub):
        fig.add_annotation(
            text=f'Showing {n_shown:,} / {len(sub):,} detections (sampled per species)',
            xref='paper', yref='paper', x=0, y=1.02, showarrow=False,
            font=dict(size=10, color=thm['tick']), align='left',
        )
    return fig


def _render_diel(dets: pd.DataFrame, strat: str, thm: dict, base: dict) -> go.Figure:
    sub = dets.dropna(subset=['_hour'])
    if sub.empty:
        return _empty_fig('No detections with parseable timestamps', thm, base)

    active_sp = [sp for sp in SPECIES if (sub[strat] == sp).any()]
    # Build a count matrix: species × hour
    data_mat = []
    for sp in active_sp:
        s = sub[sub[strat] == sp]
        counts = s['_hour'].value_counts().reindex(range(24), fill_value=0).values
        data_mat.append(counts)

    z = np.array(data_mat, dtype=float)
    labels_y = [SP_SHORT[sp] + ('' if sp in RELIABLE else ' ⚠') for sp in active_sp]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(range(24)),
        y=labels_y,
        colorscale='Blues',
        hovertemplate='Species: %{y}<br>Hour: %{x}:00 UTC<br>Detections: %{z}<extra></extra>',
        colorbar=dict(title='Count', thickness=12),
    ))
    fig.update_layout(
        **base,
        xaxis=dict(title='Hour of day (UTC)', dtick=1, **_axis_style(thm)),
        yaxis=dict(**_axis_style(thm)),
    )
    return fig


def _render_hist(threshold: float, species: list[str], thm: dict, base: dict) -> go.Figure:
    """Overlaid probability histograms per active species (all rows, not just detections)."""
    if not species:
        return _empty_fig('No species selected', thm, base)

    traces = []
    for sp in [s for s in SPECIES if s in species]:
        col = f'prob_{sp}'
        vals = df[col].dropna().values
        vals = vals[vals > 0.01]
        if not len(vals):
            continue
        traces.append(go.Histogram(
            x=vals,
            name=SP_SHORT[sp],
            nbinsx=40,
            marker_color=SP_COLORS[sp],
            opacity=0.65,
            hovertemplate=f'<b>{SP_SHORT[sp]}</b><br>prob: %{{x:.2f}}<br>count: %{{y:,}}<extra></extra>',
        ))

    if not traces:
        return _empty_fig('No probability data found', thm, base)

    fig = go.Figure(data=traces)

    # Threshold line
    fig.add_vline(x=threshold, line_dash='dash', line_color='#E24B4A',
                  annotation_text=f'threshold {threshold:.2f}',
                  annotation_font_color='#E24B4A',
                  annotation_position='top right')

    # PR threshold lines per species (if single species selected, show it)
    if len(species) == 1:
        sp = species[0]
        pr_thr = PR_THRESHOLDS.get(sp)
        if pr_thr and pr_thr < 1.0:
            fig.add_vline(x=pr_thr, line_dash='dot', line_color='#639922',
                          annotation_text=f'PR-thr {pr_thr:.3f}',
                          annotation_font_color='#639922',
                          annotation_position='top left')

    fig.update_layout(
        **base,
        barmode='overlay',
        xaxis=dict(title='Softmax probability', range=[0, 1], **_axis_style(thm)),
        yaxis=dict(title='Segment count', **_axis_style(thm)),
        bargap=0.02,
    )
    return fig


def _render_summary(dets: pd.DataFrame, strat: str, thm: dict, base: dict) -> go.Figure:
    """Bar chart of detection counts per species, coloured by species."""
    if dets.empty:
        return _empty_fig('No detections match current filters', thm, base)

    counts = dets[strat].value_counts()
    sp_order = [sp for sp in SPECIES if sp in counts.index]
    vals  = [counts[sp] for sp in sp_order]
    cols  = [SP_COLORS[sp] for sp in sp_order]
    names = [SP_SHORT[sp] + ('' if sp in RELIABLE else ' ⚠') for sp in sp_order]

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=cols,
        hovertemplate='<b>%{x}</b><br>detections: %{y:,}<extra></extra>',
    ))
    fig.update_layout(
        **base,
        xaxis=dict(**_axis_style(thm)),
        yaxis=dict(title='Detections', **_axis_style(thm)),
        showlegend=False,
    )
    return fig


def _empty_fig(msg: str, thm: dict, base: dict) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **base,
        annotations=[dict(text=msg, showarrow=False,
                          font=dict(size=14, color=thm['tick']))],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Detection detail panel (click on timeline)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('detail-panel', 'children'),
    Input('main-plot', 'clickData'),
    State('strat', 'value'),
    State('active-species', 'data'),
    State('threshold', 'value'),
    prevent_initial_call=True,
)
def show_detail(click_data, strat, species, threshold):
    if not click_data:
        raise PreventUpdate

    pt = click_data['points'][0]
    wav_name = None
    offset_s = None
    cd = pt.get('customdata')
    if cd and len(cd) >= 2:
        wav_name = cd[0]
        offset_s = cd[1]

    if not wav_name:
        return html.Div('Could not resolve point.', className='detail-empty')

    # Find the row
    match = df[(df['wav_name'] == wav_name) & (df['offset_s'].astype(str) == str(offset_s))]
    if match.empty:
        # Try float comparison
        try:
            match = df[(df['wav_name'] == wav_name) &
                       (np.isclose(df['offset_s'].astype(float), float(offset_s)))]
        except Exception:
            pass

    if match.empty:
        return html.Div(f'Row not found for {wav_name} @ {offset_s}s', className='detail-empty')

    row = match.iloc[0]
    sp  = row[strat]
    prob = float(row['max_species_prob'])
    ts   = row['_seg_ts']
    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S UTC') if pd.notna(ts) else '—'
    reliable = sp in RELIABLE

    prob_bars = []
    probs_sorted = sorted(
        [(c, float(row.get(f'prob_{c}', 0))) for c in CLASSES],
        key=lambda x: -x[1]
    )
    for c, pv in probs_sorted:
        pct = pv * 100
        prob_bars.append(html.Div(className='prob-bar-row', children=[
            html.Span(SP_SHORT[c], className='prob-bar-label'),
            html.Div(className='prob-bar-track', children=[
                html.Div(className='prob-bar-fill',
                         style={'width': f'{pct:.1f}%', 'background': SP_COLORS.get(c, '#888')}),
            ]),
            html.Span(f'{pct:.1f}%', className='prob-bar-num'),
        ]))

    strat_compare = []
    for s, slabel in STRAT_LABELS.items():
        sv = row.get(s, '—')
        strat_compare.append(html.Div(className='sel-det-kv', children=[
            html.Span(slabel, className='sel-det-k'),
            html.Span(SP_SHORT.get(str(sv), str(sv)), className='sel-det-v',
                      style={'color': SP_COLORS.get(str(sv), 'inherit'),
                             'fontSize': '11px'}),
        ]))

    return html.Div([
        html.Div(className='sel-det', children=[
            html.Div(className='sel-det-title', children=[
                html.Span(SP_SHORT.get(sp, sp)),
                html.Span(f'  {prob*100:.1f}%', style={'fontWeight': '400', 'fontSize': '12px', 'opacity': '.8'}),
                html.Span('  reliable' if reliable else '  ⚠ low-n',
                          className='badge badge-reliable' if reliable else 'badge badge-unreliable'),
            ]),
            html.Div(className='sel-det-grid', children=[
                html.Div(className='sel-det-kv', children=[
                    html.Span('file',         className='sel-det-k'),
                    html.Span(wav_name or '—', className='sel-det-v', style={'fontSize': '10px'}),
                ]),
                html.Div(className='sel-det-kv', children=[
                    html.Span('offset',        className='sel-det-k'),
                    html.Span(f'{offset_s}s',  className='sel-det-v'),
                ]),
                html.Div(className='sel-det-kv', children=[
                    html.Span('timestamp',  className='sel-det-k'),
                    html.Span(ts_str,       className='sel-det-v', style={'fontSize': '10px'}),
                ]),
                html.Div(className='sel-det-kv', children=[
                    html.Span('Perch top',           className='sel-det-k'),
                    html.Span(
                        f"{row.get('top_predicted_class', '—')}  ({float(row.get('top_logit_score', 0)):.2f})",
                        className='sel-det-v', style={'fontSize': '10px'},
                    ),
                ]),
            ]),
            html.Div('Strategy comparison', className='sb-label', style={'marginBottom': '4px', 'marginTop': '8px'}),
            html.Div(className='sel-det-grid', children=strat_compare),
        ]),
        html.Div([
            html.Div('Probability vector', className='sb-label', style={'marginBottom': '6px'}),
            *prob_bars,
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Starting on http://{args.host}:{args.port}")
    print(f"  (VSCode SSH: port {args.port} should auto-forward)")
    print(f"  Or: ssh -L {args.port}:localhost:{args.port} user@server\n")
    app.run(host=args.host, port=args.port, debug=False)
