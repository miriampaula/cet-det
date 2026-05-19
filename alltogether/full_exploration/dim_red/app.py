"""
app.py
======
Dash app for interactive PCA/UMAP/t-SNE viewer over the Perch-v2 corpus.

Styling lives in ./assets/styles.css. Theme bootstrap in ./assets/theme.js.
Both files are auto-loaded by Dash from the assets/ directory.

Run:
    python app.py --parquet ./projector_data.parquet --port 8050
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ClientsideFunction
from dash import dash_table
from flask import abort, send_file

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--parquet', default='./projector_data.parquet')
parser.add_argument('--port', type=int, default=8050)
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--default-render', type=int, default=30_000)
parser.add_argument('--max-classes', type=int, default=20,
                    help='Max distinct categories to color separately (rest -> "_other" in grey)')
args = parser.parse_args()

# -------------------------------------------------------------------------
# Load data + parquet metadata
# -------------------------------------------------------------------------
print(f"Loading {args.parquet} ...")
import pyarrow.parquet as pq
pq_file = pq.ParquetFile(args.parquet)
df = pq_file.read().to_pandas()
print(f"  {len(df):,} rows, {len(df.columns)} cols")

PCA_VARIANCE = None
raw_meta = pq_file.schema_arrow.metadata or {}
key = b'pca_variance_ratio_50d'
if key in raw_meta:
    PCA_VARIANCE = np.array(json.loads(raw_meta[key].decode('utf-8')))
    print(f"  PCA variance loaded: PC1={PCA_VARIANCE[0]:.3f} "
          f"PC2={PCA_VARIANCE[1]:.3f} PC3={PCA_VARIANCE[2]:.3f}")
else:
    print(f"  (no PCA variance in parquet metadata)")

COORD_COLS = [c for c in df.columns if c.endswith(('_x', '_y', '_z')) and
              any(c.startswith(p) for p in ('pca_', 'umap_', 'tsne_'))]
for c in COORD_COLS:
    df[c] = df[c].astype(np.float32)

SKIP_COLS = set(COORD_COLS) | {
    'png_path', 'wav_path', 'wav_name',
    'row', 'global_idx', 'embedding_idx',
    'signal_set',
}

def _is_color_candidate(col: str) -> bool:
    if col in SKIP_COLS:
        return False
    s = df[col]
    if pd.api.types.is_float_dtype(s):
        return False
    if s.notna().sum() < 10:
        return False
    n_unique = s.nunique(dropna=True)
    if n_unique < 2 or n_unique > 500:
        return False
    return True

COLOR_OPTIONS = ['dataset'] + [c for c in [
    'environment', 'region',
    'label_t1', 'label_t2', 'label_t3', 'label_t4',
] if c in df.columns]
extra = [c for c in df.columns
         if c not in COLOR_OPTIONS and _is_color_candidate(c)]
COLOR_OPTIONS = COLOR_OPTIONS + sorted(extra)
print(f"  color-by options: {COLOR_OPTIONS}")

DATASETS_AVAILABLE = sorted(df['dataset'].dropna().unique().tolist())
N_TOTAL = len(df)
print(f"  datasets: {len(DATASETS_AVAILABLE)}")
print(f"  png_path matches: {df['png_path'].notna().sum():,} / {N_TOTAL:,}")

METHOD_VALID = {
    'pca':  np.ones(N_TOTAL, dtype=bool),
    'umap': np.ones(N_TOTAL, dtype=bool),
    'tsne': df['tsne_2d_x'].notna().to_numpy(),
}
print(f"  t-SNE coverage: {METHOD_VALID['tsne'].sum():,} rows")

# -------------------------------------------------------------------------
# Palette
# -------------------------------------------------------------------------
try:
    import colorcet as cc
    _raw = None
    for attr in ('b_glasbey_bw_minc_20', 'glasbey_bw_minc_20'):
        if hasattr(cc, attr):
            _raw = getattr(cc, attr)
            break
    if _raw is None and hasattr(cc, 'palette'):
        _raw = cc.palette.get('glasbey_bw_minc_20') or cc.palette.get('b_glasbey_bw_minc_20')
    if _raw is None:
        raise ImportError('glasbey_bw_minc_20 not found')

    def _to_hex(c):
        if isinstance(c, str):
            return c if c.startswith('#') else f'#{c}'
        r, g, b = c[:3]
        return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    _GLASBEY = [_to_hex(c) for c in _raw]
    print(f"  using colorcet glasbey palette ({len(_GLASBEY)} colors)")
except ImportError:
    import plotly.express as px
    _GLASBEY = (
        px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Pastel
    )
    print(f"  colorcet not installed, using plotly qualitative palette ({len(_GLASBEY)})")

GREY = '#bfbfbf'

def palette(n: int) -> list[str]:
    if n <= len(_GLASBEY):
        return _GLASBEY[:n]
    return [_GLASBEY[i % len(_GLASBEY)] for i in range(n)]


# -------------------------------------------------------------------------
# Plotly themes — match the app's design system
# -------------------------------------------------------------------------
PLOTLY_THEMES = {
    'light': {
        'paper_bgcolor': '#ffffff',
        'plot_bgcolor':  '#ffffff',
        'font_color':    '#111827',
        'grid_color':    '#f3f4f6',
        'axis_color':    '#e5e7eb',
        'tick_color':    '#9ca3af',
        'legend_bg':     'rgba(255,255,255,0.85)',
        'legend_border': '#e5e7eb',
    },
    'dark': {
        'paper_bgcolor': '#0e1014',
        'plot_bgcolor':  '#0e1014',
        'font_color':    '#e5e7eb',
        'grid_color':    '#1c2027',
        'axis_color':    '#262b33',
        'tick_color':    '#6b7280',
        'legend_bg':     'rgba(22,25,31,0.85)',
        'legend_border': '#262b33',
    },
}

# -------------------------------------------------------------------------
# App
# -------------------------------------------------------------------------
app = Dash(__name__, title='Perch-v2 Embedding Viewer',
           assets_folder='assets',
           update_title=None,
           suppress_callback_exceptions=False)
server = app.server


# Custom <head> with viewport + sets initial data-theme attr to avoid flash
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


@server.route('/spectrogram/<int:row_id>')
def serve_spectrogram(row_id):
    if row_id < 0 or row_id >= N_TOTAL:
        abort(404)
    p = df.iloc[row_id]['png_path']
    if pd.isna(p) or not Path(p).exists():
        abort(404)
    return send_file(str(p), mimetype='image/png')


# -------------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------------
def _segmented(id_, options, value):
    """Custom segmented-radio control — looks like iOS-style segmented control."""
    return html.Div(
        className='segmented',
        children=[
            html.Label([
                dcc.RadioItems(
                    id=id_, value=value, inline=True,
                    options=options,
                ),
            ])
        ],
    )


app.layout = html.Div([
    # ---------- Header (topbar + slider row) ----------
    html.Div(className='app-header', children=[
        # Row 1: title + main controls
        html.Div(className='app-topbar', children=[
            html.Span('Perch v2 · Embedding Viewer', className='app-title'),
            html.Div(className='divider'),

            html.Div(className='control-group', children=[
                html.Span('Method', className='label'),
                html.Div(className='segmented', children=[
                    dcc.RadioItems(
                        id='method', value='umap', inline=True,
                        options=[{'label': 'PCA', 'value': 'pca'},
                                 {'label': 'UMAP', 'value': 'umap'},
                                 {'label': 't-SNE', 'value': 'tsne'}],
                    ),
                ]),
            ]),

            html.Div(className='control-group', children=[
                html.Span('Dim', className='label'),
                html.Div(className='segmented', children=[
                    dcc.RadioItems(
                        id='dim', value='3d', inline=True,
                        options=[{'label': '2D', 'value': '2d'},
                                 {'label': '3D', 'value': '3d'}],
                    ),
                ]),
            ]),

            html.Div(className='divider'),

            html.Div(className='control-group', children=[
                html.Span('Color', className='label'),
                dcc.Dropdown(
                    id='color-by',
                    value='label_t2' if 'label_t2' in COLOR_OPTIONS else 'dataset',
                    options=[{'label': c, 'value': c} for c in COLOR_OPTIONS],
                    style={'width': '170px'}, clearable=False,
                ),
            ]),

            html.Div(className='control-group', children=[
                html.Span('Datasets', className='label'),
                dcc.Dropdown(
                    id='dataset-filter', value=[], multi=True,
                    options=[{'label': d, 'value': d} for d in DATASETS_AVAILABLE],
                    placeholder='all', style={'width': '260px'},
                ),
            ]),

            html.Div(className='control-group', children=[
                html.Span('Labels', className='label'),
                dcc.Dropdown(
                    id='label-filter', value=[], multi=True,
                    options=[], placeholder='all', style={'width': '260px'},
                ),
            ]),

            html.Button('☀  Light', id='theme-toggle', className='theme-toggle', n_clicks=0),
        ]),

        # Row 2: render slider on its own
        html.Div(className='app-slider-row', children=[
            html.Span('Render cap', className='label'),
            html.Div(className='slider-wrap', children=[
                dcc.Slider(
                    id='render-cap', min=1000, max=N_TOTAL, step=1000,
                    value=min(args.default_render, N_TOTAL),
                    marks={1000: '1k', 10_000: '10k', 50_000: '50k',
                           100_000: '100k', N_TOTAL: f'{N_TOTAL/1000:.0f}k (all)'},
                    tooltip={'placement': 'bottom', 'always_visible': False},
                    updatemode='mouseup',
                ),
            ]),
        ]),
    ]),

    # ---------- Status bar ----------
    html.Div(id='status-bar', className='status-bar'),

    # ---------- Main split ----------
    html.Div(className='main-split', children=[
        html.Div(className='plot-pane', children=[
            dcc.Loading(
                dcc.Graph(id='scatter', style={'height': '100%'},
                          config={'displayModeBar': True, 'scrollZoom': True,
                                  'displaylogo': False}),
                type='circle', color='#3b82f6',
            ),
        ]),
        html.Div(className='detail-pane', children=[
            html.H4('Point detail'),
            html.Div(id='detail-spectrogram', className='detail-spectrogram'),
            html.Div(id='detail-meta', className='metadata-table'),
        ]),
    ]),

    # ---------- Hidden state ----------
    dcc.Store(id='trace-to-rows'),
    dcc.Store(id='theme-store', data='light',
              storage_type='local'),  # persists across reloads
])


# -------------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------------
@app.callback(
    Output('label-filter', 'options'),
    Output('label-filter', 'value'),
    Input('color-by', 'value'),
)
def update_label_options(color_by):
    if color_by == 'dataset':
        return [], []
    vals = sorted(df[color_by].dropna().astype(str).unique().tolist())
    opts = [{'label': v, 'value': v} for v in vals]
    return opts, []


# Theme toggle: button click flips theme-store, clientside applies it to <html>
@app.callback(
    Output('theme-store', 'data'),
    Output('theme-toggle', 'children'),
    Input('theme-toggle', 'n_clicks'),
    State('theme-store', 'data'),
    prevent_initial_call=False,
)
def toggle_theme(n_clicks, current):
    if not current:
        current = 'light'
    if not n_clicks:
        # Initial render: just reflect current state in button label
        return current, ('☾  Dark' if current == 'dark' else '☀  Light')
    nxt = 'dark' if current == 'light' else 'light'
    return nxt, ('☾  Dark' if nxt == 'dark' else '☀  Light')


# Clientside callback: apply theme to <html data-theme=...>
app.clientside_callback(
    ClientsideFunction(namespace='theme', function_name='apply'),
    Output('theme-store', 'data', allow_duplicate=True),
    Input('theme-store', 'data'),
    prevent_initial_call='initial_duplicate',
)


def _status_pills(items: list[tuple[str, str, bool]]) -> list:
    """Build a list of pill divs. items = [(label, value, accent), ...]"""
    pills = []
    for label, value, accent in items:
        pills.append(html.Div(
            className=f'status-pill{" accent" if accent else ""}',
            children=[
                html.Span(label, className='pill-label'),
                html.Span(value, className='pill-value'),
            ],
        ))
    return pills


def _variance_tuple(method: str, dim: str):
    if method != 'pca' or PCA_VARIANCE is None:
        return None
    n = 2 if dim == '2d' else 3
    parts = [f'{PCA_VARIANCE[i]:.3f}' for i in range(n)]
    return ', '.join(parts), f'{PCA_VARIANCE[:n].sum():.3f}'


@app.callback(
    Output('scatter', 'figure'),
    Output('status-bar', 'children'),
    Output('trace-to-rows', 'data'),
    Input('method', 'value'),
    Input('dim', 'value'),
    Input('color-by', 'value'),
    Input('dataset-filter', 'value'),
    Input('label-filter', 'value'),
    Input('render-cap', 'value'),
    Input('theme-store', 'data'),
)
def update_figure(method, dim, color_by, dataset_filter, label_filter, render_cap, theme):
    theme = theme or 'light'
    thm = PLOTLY_THEMES[theme]

    mask = METHOD_VALID[method].copy()
    if dataset_filter:
        mask &= df['dataset'].isin(dataset_filter).to_numpy()
    if label_filter and color_by != 'dataset':
        mask &= df[color_by].astype(str).isin(label_filter).to_numpy()

    candidate_idx = np.where(mask)[0]
    n_match = len(candidate_idx)
    if n_match == 0:
        empty = go.Figure()
        empty.update_layout(
            paper_bgcolor=thm['paper_bgcolor'], plot_bgcolor=thm['plot_bgcolor'],
            font=dict(color=thm['font_color']),
            annotations=[dict(text='No points match filters', showarrow=False,
                              font=dict(size=14, color=thm['tick_color']))],
        )
        return empty, _status_pills([('status', 'no points', False)]), {}

    if n_match > render_cap:
        sub = df.iloc[candidate_idx][[color_by]].copy()
        sub['__i__'] = candidate_idx
        sub[color_by] = sub[color_by].fillna('__nan__').astype(str)
        n_classes = max(1, sub[color_by].nunique())
        per_class = max(1, render_cap // n_classes)
        picked = (
            sub.groupby(color_by, dropna=False, group_keys=False)
               .apply(lambda g: g.sample(n=min(per_class, len(g)), random_state=42),
                      include_groups=False)
        )
        if len(picked) > render_cap:
            picked = picked.sample(n=render_cap, random_state=42)
        idx = np.sort(picked['__i__'].to_numpy())
    else:
        idx = candidate_idx

    sub = df.iloc[idx]

    xc = f'{method}_{dim}_x'
    yc = f'{method}_{dim}_y'
    zc = f'{method}_{dim}_z' if dim == '3d' else None

    cats_raw = sub[color_by].fillna('__nan__').astype(str)
    top_counts = cats_raw.value_counts()
    top_keep = top_counts.head(args.max_classes).index
    n_truncated = max(0, len(top_counts) - args.max_classes)
    cats = cats_raw.where(cats_raw.isin(top_keep), other='_other')

    real_cats = [c for c in top_counts.index if c in top_keep
                 and c not in ('_other', '__nan__')]
    tail_cats = [c for c in ('_other', '__nan__') if c in cats.unique()]
    ordered_cats = real_cats + tail_cats

    colors = palette(len(real_cats))
    color_map = {c: colors[i] for i, c in enumerate(real_cats)}
    color_map['_other'] = GREY
    color_map['__nan__'] = GREY

    traces = []
    trace_to_rows = []
    for cat in ordered_cats:
        m = (cats == cat).to_numpy()
        if not m.any():
            continue
        s = sub.iloc[m]
        cd = s.index.to_numpy().reshape(-1, 1)
        trace_to_rows.append(s.index.to_numpy().tolist())
        common = dict(
            mode='markers',
            name=f'{cat}  ({m.sum():,})',
            customdata=cd,
            hovertemplate=(
                f'<b>{color_by}</b>: {cat}<br>'
                'dataset: %{text}<br>'
                'row: %{customdata[0]}<extra></extra>'
            ),
            text=s['dataset'].astype(str).tolist(),
            marker=dict(size=3 if dim == '3d' else 4,
                        opacity=0.5 if cat in ('_other', '__nan__') else 0.78,
                        color=color_map[cat],
                        line=dict(width=0)),
        )
        if dim == '3d':
            traces.append(go.Scatter3d(
                x=s[xc].to_numpy(), y=s[yc].to_numpy(), z=s[zc].to_numpy(),
                **common,
            ))
        else:
            traces.append(go.Scattergl(
                x=s[xc].to_numpy(), y=s[yc].to_numpy(),
                **common,
            ))

    fig = go.Figure(data=traces)

    # Theme-aware layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor=thm['paper_bgcolor'],
        plot_bgcolor=thm['plot_bgcolor'],
        font=dict(color=thm['font_color'], family='-apple-system, BlinkMacSystemFont, sans-serif',
                  size=11),
        legend=dict(
            itemsizing='constant',
            font=dict(size=10, color=thm['font_color']),
            bgcolor=thm['legend_bg'],
            bordercolor=thm['legend_border'],
            borderwidth=1,
            x=1.0, y=1.0, xanchor='right', yanchor='top',
        ),
        hoverlabel=dict(
            bgcolor=thm['paper_bgcolor'],
            bordercolor=thm['legend_border'],
            font=dict(color=thm['font_color'], size=11,
                      family='-apple-system, BlinkMacSystemFont, sans-serif'),
        ),
    )
    if dim == '3d':
        axis_style = dict(
            backgroundcolor=thm['plot_bgcolor'],
            gridcolor=thm['grid_color'],
            zerolinecolor=thm['axis_color'],
            color=thm['tick_color'],
            showbackground=False,
            tickfont=dict(size=9),
        )
        fig.update_layout(scene=dict(
            xaxis=dict(title=xc, **axis_style),
            yaxis=dict(title=yc, **axis_style),
            zaxis=dict(title=zc, **axis_style),
            aspectmode='cube',
            bgcolor=thm['plot_bgcolor'],
        ))
    else:
        fig.update_xaxes(title=xc, gridcolor=thm['grid_color'],
                         zerolinecolor=thm['axis_color'], color=thm['tick_color'])
        fig.update_yaxes(title=yc, scaleanchor='x', scaleratio=1,
                         gridcolor=thm['grid_color'], zerolinecolor=thm['axis_color'],
                         color=thm['tick_color'])

    # Status pills
    var_tuple = _variance_tuple(method, dim)
    pills_data = [
        ('method', f'{method.upper()}·{dim.upper()}', True),
        ('rendered', f'{len(idx):,} / {n_match:,}', False),
        ('corpus', f'{N_TOTAL:,}', False),
        ('color', color_by, False),
    ]
    if var_tuple is not None:
        components, cum = var_tuple
        pills_data.append(('var', f'{components}', False))
        pills_data.append(('cum var', cum, False))
    if n_truncated > 0:
        pills_data.append(('collapsed', f'{n_truncated} cats → _other', False))

    return fig, _status_pills(pills_data), trace_to_rows


@app.callback(
    Output('detail-spectrogram', 'children'),
    Output('detail-meta', 'children'),
    Input('scatter', 'clickData'),
    State('trace-to-rows', 'data'),
)
def show_point_detail(click_data, trace_to_rows):
    if not click_data:
        return html.Div('Click a point to see its spectrogram and metadata.',
                        className='detail-empty'), None

    pt = click_data['points'][0]
    row_id = None
    cd = pt.get('customdata')
    if cd is not None:
        if isinstance(cd, (list, tuple, np.ndarray)) and len(cd) > 0:
            try: row_id = int(cd[0])
            except (TypeError, ValueError): pass
        else:
            try: row_id = int(cd)
            except (TypeError, ValueError): pass

    if row_id is None and trace_to_rows is not None:
        curve = pt.get('curveNumber')
        pidx = pt.get('pointNumber', pt.get('pointIndex'))
        if curve is not None and pidx is not None:
            try:
                row_id = int(trace_to_rows[curve][pidx])
            except (IndexError, TypeError, ValueError):
                pass

    if row_id is None:
        return (
            html.Div([
                html.Div('Could not resolve row from click.', className='detail-empty'),
                html.Pre(str(pt), style={'fontSize': '10px', 'color': 'var(--text-faint)',
                                         'maxHeight': '200px', 'overflow': 'auto',
                                         'marginTop': '8px'}),
            ]),
            None,
        )

    row = df.iloc[row_id]
    png = row.get('png_path')

    if pd.notna(png) and png and Path(str(png)).exists():
        img = html.Div([
            html.Img(src=f'/spectrogram/{row_id}?r={row_id}'),
            html.Div(f'row {row_id:,}', className='detail-row-id'),
        ])
    else:
        reason = ('no spectrogram available' if pd.isna(png)
                  else f'file missing: {Path(str(png)).name}')
        img = html.Div(reason, className='detail-empty')

    skip = set(COORD_COLS)
    items = [(k, v) for k, v in row.items() if k not in skip]
    meta = dash_table.DataTable(
        columns=[{'name': 'field', 'id': 'field'}, {'name': 'value', 'id': 'value'}],
        data=[{'field': k, 'value': str(v)} for k, v in items],
        style_cell={'textAlign': 'left', 'padding': '5px 8px',
                    'whiteSpace': 'normal', 'height': 'auto',
                    'maxWidth': '0', 'overflow': 'hidden', 'textOverflow': 'ellipsis',
                    'border': 'none'},
        style_header={'border': 'none'},
        style_cell_conditional=[
            {'if': {'column_id': 'field'}, 'width': '38%'},
        ],
        page_size=100,
    )
    return img, meta


if __name__ == '__main__':
    print(f"\nStarting Dash on http://{args.host}:{args.port}")
    print(f"  (VSCode SSH should auto-forward port {args.port})\n")
    app.run(host=args.host, port=args.port, debug=False)
