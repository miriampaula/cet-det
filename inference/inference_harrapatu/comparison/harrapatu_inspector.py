#!/usr/bin/env python3
"""
Harrapatu Cetacean Inspector
============================
Navigate by 5-min interval. All heavy data structures are pre-computed
at startup — callbacks are O(1) lookups, not O(N) scans.

Run:
    python harrapatu_inspector.py \
        --csv  harrapatu_comparison_5min_v1_segments.csv \
        --spectrograms /path/to/spectrograms \
        --port 8077
"""
from __future__ import annotations
import argparse, math
from pathlib import Path

import pandas as pd
import flask
import dash
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL

# ── palette ───────────────────────────────────────────────────────────────────
BG       = '#080e1a'
PANEL    = '#0d1b2e'
PANEL2   = '#112540'
LINE     = '#1e3a5a'
CYAN     = '#38d9d0'
AMBER    = '#f5a623'
ROSE     = '#f05675'
LIME     = '#7ed957'
INK      = '#dceeff'
INK_DIM  = '#7aa3cc'
INK_FAINT= '#3d6080'

TP_COL = LIME; FP_COL = AMBER; FN_COL = ROSE; TN_COL = '#3d6080'

SPECIES_COLOR = {
    'Tt':'#38d9d0','Ambig':'#6c8fff','Gg':'#7ed957','Dd':'#f5a623',
    'Gm':'#c97bff','Oo':'#f05675','Sc':'#40c8e0','Pm':'#ffd080',
    'Ba':'#80d4a0','Bp':'#4db8ff',
}
NEUTRAL = {'background','uncertain','no_label','nan','','—'}

STRATEGIES = [('pred_argmax','argmax'),('pred_vec','vec'),
              ('pred_pr','pr'),('pred_consensus','consensus')]
PROB_COLS_ORDERED = ['prob_Tt','prob_Ambig','prob_Gg','prob_Dd','prob_Gm',
                     'prob_Oo','prob_Sc','prob_Pm','prob_Ba','prob_Bp','prob_bg']

OUTCOME_FILTER = {'true_positive':'TP','false_positive':'FP',
                  'false_negative':'FN','true_negative':'TN'}
MODES = [('all','all intervals'),
         ('true_positive','TP — model caught it'),
         ('false_positive','FP — false alarm'),
         ('false_negative','FN — model missed it'),
         ('true_negative','TN — both silent'),
         ('expert_positive','expert heard Tt')]

SPECTRO_W = 108
SPECTRO_H = int(round(SPECTRO_W * 128 / 500))   # ≈28 px
FREQ_MAX_KHZ = 16.0

# ── helpers ───────────────────────────────────────────────────────────────────
def wav_parts(wav_name):
    stem = Path(str(wav_name)).stem.split('.')
    return (stem[0], stem[1]) if len(stem) >= 2 else (stem[0], '')

def token_to_date(t):
    return f"20{t[0:2]}-{t[2:4]}-{t[4:6]}" if len(t) >= 6 else t

def token_to_clock(t):
    return f"{t[6:8]}:{t[8:10]}:{t[10:12]}" if len(t) >= 12 else ''

def offset_to_mmss(s):
    s = int(float(s)); e = s + 5
    return f"{s//60:02d}.{s%60:02d}-{e//60:02d}.{e%60:02d}"

def expected_png(wav_name, offset_s):
    rec, _ = wav_parts(wav_name)
    stem = Path(str(wav_name)).stem
    return f"{rec}_{stem}_{offset_to_mmss(offset_s)}.png"

def chip_color(label):
    return SPECIES_COLOR.get(str(label), INK_FAINT if str(label) in NEUTRAL else '#2a4a6a')

def chip(label, role='strat'):
    s = str(label); neutral = s in NEUTRAL
    c = chip_color(s)
    style = {
        'borderRadius':'5px','padding':'3px 4px','fontSize':'10px',
        'textAlign':'center','overflow':'hidden','textOverflow':'ellipsis',
        'whiteSpace':'nowrap','fontFamily':'ui-monospace,monospace',
        'height':'16px','lineHeight':'16px','width':'100%',
        'background':(c+'22') if neutral else c,
        'color':INK_FAINT if neutral else '#04090f',
        'border':(f'1px solid {c}40') if neutral else f'1px solid {c}',
        'fontWeight':'400' if neutral else '700',
    }
    if role == 'expert':
        style.update({
            'borderColor': AMBER if not neutral else f'{AMBER}60',
            'boxShadow': f'0 0 0 1px {AMBER}30',
            'background': AMBER if not neutral else f'{AMBER}18',
            'color': '#04090f' if not neutral else INK_DIM,
        })
    return html.Div(s, style=style)

def outcome_badge(cm):
    c = {' TP':TP_COL,'FP':FP_COL,'FN':FN_COL,'TN':TN_COL}.get(cm, INK_FAINT)
    c = {'TP':TP_COL,'FP':FP_COL,'FN':FN_COL,'TN':TN_COL}.get(cm, INK_FAINT)
    return html.Span(cm, style={
        'fontSize':'10px','fontWeight':'800','letterSpacing':'0.08em',
        'fontFamily':'ui-monospace,monospace','color':c,
        'background':f'{c}22','border':f'1px solid {c}55',
        'borderRadius':'4px','padding':'1px 5px',
    }) if cm else html.Span()

def mel_axis(h=SPECTRO_H, fmax=FREQ_MAX_KHZ):
    def m(f): return 2595*math.log10(1+f/700)
    mm = m(fmax*1000)
    ticks = [t for t in [0,1,2,4,8,12,16] if t<=fmax]
    kids = [html.Div('kHz',style={'position':'absolute','top':'-16px','right':'0',
                                   'fontSize':'8px','color':INK_FAINT})]
    for k in ticks:
        top = (1-m(k*1000)/mm)*h
        kids += [
            html.Div(str(k),style={'position':'absolute','top':f'{top:.1f}px','right':'6px',
                                    'transform':'translateY(-50%)','fontSize':'8px',
                                    'fontFamily':'ui-monospace,monospace','color':INK_FAINT}),
            html.Div(style={'position':'absolute','top':f'{top:.1f}px','right':'0',
                             'width':'4px','height':'1px','background':INK_FAINT}),
        ]
    return html.Div(kids, style={'position':'relative','width':'28px','flex':'0 0 28px',
                                  'height':f'{h}px','borderRight':f'1px solid {LINE}',
                                  'marginRight':'4px','marginTop':'2px'})

def strategy_rail():
    rows = [('', f'{SPECTRO_H}px'),('','14px'),
            ('argmax','22px'),('vec','22px'),('pr','22px'),('consensus','22px'),
            ('','7px'),('EXPERT','22px')]
    kids = [html.Div(t, style={
        'height':h,'display':'flex','alignItems':'center','justifyContent':'flex-end',
        'paddingRight':'8px','fontSize':'9px','letterSpacing':'0.07em',
        'fontFamily':'ui-monospace,monospace','textTransform':'uppercase',
        'color':AMBER if t=='EXPERT' else INK_FAINT,
        'fontWeight':'700' if t=='EXPERT' else '500',
    }) for t,h in rows]
    return html.Div(kids, style={'width':'64px','flex':'0 0 64px','display':'flex',
                                  'flexDirection':'column','gap':'3px','paddingTop':'2px'})

# ── segment column ────────────────────────────────────────────────────────────
def segment_col(wav_name, offset_s, row, spectro_url, is_last=False):
    png = expected_png(wav_name, offset_s)
    exp_det = bool(row.get('exp_cetacean_detected', False))
    ann     = bool(row.get('expert_annotated', False))
    exp_lbl = row.get('exp_top_species','no_label') if exp_det else ('background' if ann else 'no_label')
    return html.Div([
        html.Img(src=f'{spectro_url}/{png}',
                 style={'width':f'{SPECTRO_W}px','height':f'{SPECTRO_H}px',
                        'objectFit':'fill','display':'block',
                        'borderRight':f'1px solid {BG}' if not is_last else 'none'}),
        html.Div(offset_to_mmss(offset_s).replace('-','–'),
                 style={'fontSize':'8px','color':INK_FAINT,'fontFamily':'ui-monospace,monospace',
                        'textAlign':'center','margin':'2px 0 4px'}),
        chip(str(row.get('pred_argmax','—'))),
        chip(str(row.get('pred_vec','—'))),
        chip(str(row.get('pred_pr','—'))),
        chip(str(row.get('pred_consensus','—'))),
        html.Div(style={'height':'1px','background':LINE,'margin':'3px 0'}),
        chip(exp_lbl, role='expert'),
    ], style={'width':f'{SPECTRO_W}px','flex':f'0 0 {SPECTRO_W}px',
              'display':'flex','flexDirection':'column','gap':'3px'})

def recorder_strip(wav_name, segs_df, spectro_url, fmax):
    rec, token = wav_parts(wav_name)
    n = len(segs_df)
    cols = [segment_col(wav_name, row['offset_s'], row, spectro_url, i==n-1)
            for i,(_,row) in enumerate(segs_df.iterrows())]
    label = html.Div([
        html.Span(f'REC {rec}', style={'color':CYAN,'fontWeight':'700',
                                        'fontFamily':'ui-monospace,monospace','fontSize':'11px'}),
        html.Span(f'  {token_to_clock(token)}',
                  style={'color':INK_DIM,'fontSize':'10px','fontFamily':'ui-monospace,monospace'}),
        html.Span(f'  ·  {n} segs',
                  style={'color':INK_FAINT,'fontSize':'10px','fontFamily':'ui-monospace,monospace'}),
    ], style={'marginBottom':'4px','display':'flex','alignItems':'baseline'})
    strip = html.Div([mel_axis(SPECTRO_H,fmax), *cols],
                     style={'display':'flex','flexDirection':'row','overflowX':'auto',
                            'alignItems':'flex-start'})
    return html.Div([label, strip],
                    style={'marginBottom':'16px','background':PANEL,
                           'border':f'1px solid {LINE}','borderRadius':'12px',
                           'padding':'12px 14px'})

# ── detail panel ──────────────────────────────────────────────────────────────
def detail_panel(ivl_df):
    if ivl_df is None or len(ivl_df)==0:
        return html.Div('No segments.', style={'color':INK_FAINT,'padding':'20px',
                                                'fontFamily':'ui-monospace,monospace'})
    prob_cols = [c for c in PROB_COLS_ORDERED if c in ivl_df.columns]

    def bar(val, color):
        v = min(1.0, float(val)) if pd.notna(val) else 0.0
        return html.Div(html.Div(style={'width':f'{v*100:.1f}%','height':'8px',
                                         'borderRadius':'3px','background':color,
                                         'minWidth':'2px' if v>0 else '0'}),
                        style={'width':'100%','height':'8px','background':f'{color}1a',
                               'borderRadius':'3px','overflow':'hidden'})

    def numtd(val, c):
        return html.Td(f"{float(val):.3f}" if pd.notna(val) else '—',
                       style={'padding':'3px 8px','color':c,'fontSize':'10px',
                              'whiteSpace':'nowrap','fontFamily':'ui-monospace,monospace',
                              'width':'44px'})

    seg_list = list(ivl_df.iterrows())
    sp_rows = []
    for pc in prob_cols:
        sp = pc.replace('prob_','')
        c = SPECIES_COLOR.get(sp, INK_FAINT)
        cells = [html.Td(sp, style={'padding':'3px 10px 3px 0','color':c,'fontWeight':'700',
                                     'position':'sticky','left':'0','background':PANEL2,
                                     'zIndex':'1','fontSize':'10px',
                                     'fontFamily':'ui-monospace,monospace',
                                     'width':'52px','whiteSpace':'nowrap'})]
        for _,row in seg_list:
            v = row.get(pc, float('nan'))
            cells += [html.Td(bar(v,c), style={'padding':'3px 4px','width':'80px'}),
                      numtd(v,c)]
        sp_rows.append(html.Tr(cells))

    hdr = [html.Th('',style={'position':'sticky','left':'0','background':PANEL2,
                               'zIndex':'1','width':'52px'})]
    for _,row in seg_list:
        t = offset_to_mmss(row['offset_s']).replace('-','–')
        hdr.append(html.Th(t, colSpan=2,
                            style={'textAlign':'left','color':INK_DIM,'fontWeight':'600',
                                   'padding':'0 4px 6px','fontSize':'9px',
                                   'fontFamily':'ui-monospace,monospace',
                                   'borderBottom':f'1px solid {LINE}'}))

    table = html.Table([html.Thead([html.Tr(hdr)]),html.Tbody(sp_rows)],
                        style={'borderCollapse':'collapse','width':'100%','tableLayout':'fixed'})
    return html.Div(html.Div(table, style={'overflowX':'auto','width':'100%'}),
                    style={'padding':'16px 18px'})

# ── overview as inline SVG (fast — no Dash elements per bar) ─────────────────
OC_COL_SVG = {'TP':LIME,'FP':FP_COL,'FN':FN_COL,'TN':TN_COL,'':LINE}

def overview_svg(intervals, ivl_meta, current_idx):
    """Return an overview bar chart as a tiny plotly figure via dcc.Graph."""
    import plotly.graph_objects as go
    n = len(intervals)
    if n == 0:
        return html.Div()

    OC_H  = {'TP': 32, 'FN': 32, 'FP': 20, 'TN': 12, '': 8}
    x, y_vals, colors, cur_outlines = [], [], [], []
    for i, ivl in enumerate(intervals):
        oc  = ivl_meta.get(ivl, {}).get('outcome', '')
        col = OC_COL_SVG.get(oc, LINE)
        x.append(i)
        y_vals.append(OC_H.get(oc, 8))
        colors.append(col)
        cur_outlines.append(INK if i == current_idx else col)

    fig = go.Figure(go.Bar(
        x=x, y=y_vals,
        marker_color=colors,
        marker_line_color=cur_outlines,
        marker_line_width=[3 if i == current_idx else 0 for i in range(n)],
        width=1.0,
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=40,
        bargap=0.02,
        xaxis=dict(visible=False, range=[-0.5, n-0.5]),
        yaxis=dict(visible=False, range=[0, 36]),
        showlegend=False,
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False},
                     style={'height': '44px', 'width': '100%'})

# ── summary badges ─────────────────────────────────────────────────────────────
def summary_badges(ivl_meta):
    counts = {}
    for v in ivl_meta.values():
        oc = v.get('outcome','')
        counts[oc] = counts.get(oc,0)+1
    total = len(ivl_meta)
    def badge(label, count, col):
        pct = f"{count/total*100:.0f}%" if total else '—'
        return html.Div([
            html.Div(label, style={'fontSize':'9px','fontWeight':'700',
                                    'letterSpacing':'0.1em','color':col,
                                    'fontFamily':'ui-monospace,monospace'}),
            html.Div(f"{count}", style={'fontSize':'22px','fontWeight':'800',
                                        'color':col,'lineHeight':'1.1',
                                        'fontFamily':'ui-monospace,monospace'}),
            html.Div(pct, style={'fontSize':'10px','color':INK_FAINT,
                                  'fontFamily':'ui-monospace,monospace'}),
        ], style={'background':f'{col}18','border':f'1px solid {col}44',
                  'borderRadius':'10px','padding':'10px 18px','textAlign':'center'})
    return html.Div([
        badge('TP', counts.get('TP',0), TP_COL),
        badge('FP', counts.get('FP',0), FP_COL),
        badge('FN', counts.get('FN',0), FN_COL),
        badge('TN', counts.get('TN',0), TN_COL),
        html.Div([
            html.Div('INTERVALS',style={'fontSize':'9px','fontWeight':'700',
                                        'letterSpacing':'0.1em','color':INK_FAINT,
                                        'fontFamily':'ui-monospace,monospace'}),
            html.Div(f"{total}",style={'fontSize':'22px','fontWeight':'800',
                                       'color':INK_DIM,'fontFamily':'ui-monospace,monospace'}),
        ], style={'padding':'10px 18px','textAlign':'center',
                  'borderLeft':f'1px solid {LINE}','marginLeft':'4px'}),
    ], style={'display':'flex','gap':'10px','alignItems':'center'})

# ── app factory ───────────────────────────────────────────────────────────────
def make_app(df: pd.DataFrame, spectro_dir: Path, fmax: float = 16.0) -> Dash:

    # ── pre-compute everything once ───────────────────────────────────────────
    print("Pre-computing interval index…")
    df['interval'] = df['interval'].astype(str)

    # ordered list of all intervals (preserves recording chronology)
    all_intervals_ordered = list(dict.fromkeys(df['interval']))

    # per-interval metadata dict  {interval: {outcome, exp_positive, wav_name, …}}
    meta_cols = [c for c in ('outcome','exp_cetacean_detected','wav_name') if c in df.columns]
    ivl_meta = (df.drop_duplicates('interval')
                  .set_index('interval')[meta_cols]
                  .to_dict('index'))

    # fast interval-list per mode (pre-filtered lists)
    def build_lists():
        out = {'all': all_intervals_ordered}
        for mode, oc in OUTCOME_FILTER.items():
            out[mode] = [i for i in all_intervals_ordered
                         if ivl_meta.get(i,{}).get('outcome')==oc]
        out['expert_positive'] = [i for i in all_intervals_ordered
                                   if ivl_meta.get(i,{}).get('outcome') in ('TP','FN')]
        return out

    mode_lists = build_lists()

    # group segments by interval once
    print("Grouping segments by interval…")
    grouped = {ivl: grp.reset_index(drop=True)
               for ivl, grp in df.groupby('interval', sort=False)}
    print("Ready.")

    # PNG index (recursive)
    png_index = {p.name.lower(): p for p in spectro_dir.rglob('*.png')} \
        if spectro_dir.exists() else {}
    print(f"Indexed {len(png_index):,} PNG files.")

    SPECTRO_URL = '/spectro'
    _cache: dict[str, Path] = {}

    app = Dash(__name__)
    server = app.server

    @server.route(f'{SPECTRO_URL}/<path:fname>')
    def serve_spectro(fname):
        key = fname.lower()
        p = _cache.get(key) or png_index.get(key)
        if p is None: flask.abort(404)
        _cache[key] = p
        resp = flask.send_file(p, mimetype='image/png')
        resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
        return resp

    # ── static parts (computed once, not in callbacks) ────────────────────────
    BADGES   = summary_badges(ivl_meta)
    RAIL     = strategy_rail()
    lbl = {'color':INK_DIM,'fontSize':'11px','fontFamily':'ui-monospace,monospace',
           'letterSpacing':'0.04em','marginRight':'6px'}

    app.index_string = (
        '<!DOCTYPE html><html><head>{%metas%}<title>Harrapatu Inspector</title>'
        '{%favicon%}{%css%}<style>'
        f'*{{box-sizing:border-box;}}body{{margin:0;background:{BG};}}'
        f'::-webkit-scrollbar{{height:8px;width:8px;}}'
        f'::-webkit-scrollbar-track{{background:{BG};}}'
        f'::-webkit-scrollbar-thumb{{background:{LINE};border-radius:4px;}}'
        f'::-webkit-scrollbar-thumb:hover{{background:{CYAN};}}'
        f'div[class*="-control"]{{background:#fff!important;border-color:#b0bec5!important;'
        f'border-radius:8px!important;color:#0a1422!important;}}'
        f'div[class*="-menu"]{{background:#fff!important;color:#0a1422!important;'
        f'border:1px solid #b0bec5!important;z-index:50!important;}}'
        f'div[class*="-option"]{{background:#fff!important;color:#0a1422!important;}}'
        f'div[class*="-option"]:hover{{background:#e8f4fd!important;}}'
        f'div[class*="-singleValue"]{{color:#0a1422!important;}}'
        f'div[class*="-placeholder"]{{color:#607d8b!important;}}'
        f'div[class*="-indicatorContainer"] svg{{fill:#607d8b!important;}}'
        f'.hbtn{{background:{PANEL2};color:{INK};border:1px solid {LINE};'
        f'border-radius:8px;padding:6px 14px;font-family:ui-monospace,monospace;'
        f'font-size:12px;cursor:pointer;transition:background .15s;}}'
        f'.hbtn:hover{{background:{CYAN};border-color:{CYAN};color:#04090f;}}'
        f'.ofbtn{{border-radius:8px;padding:5px 12px;font-family:ui-monospace,monospace;'
        f'font-size:11px;font-weight:700;letter-spacing:0.05em;cursor:pointer;border:1px solid;}}'
        f'.tp-btn{{background:{TP_COL}22;color:{TP_COL};border-color:{TP_COL}66;}}'
        f'.tp-btn:hover{{background:{TP_COL};color:#04090f;border-color:{TP_COL};}}'
        f'.fp-btn{{background:{FP_COL}22;color:{FP_COL};border-color:{FP_COL}66;}}'
        f'.fp-btn:hover{{background:{FP_COL};color:#04090f;border-color:{FP_COL};}}'
        f'.fn-btn{{background:{FN_COL}22;color:{FN_COL};border-color:{FN_COL}66;}}'
        f'.fn-btn:hover{{background:{FN_COL};color:#04090f;border-color:{FN_COL};}}'
        f'.tn-btn{{background:{TN_COL}22;color:{TN_COL};border-color:{TN_COL}66;}}'
        f'.tn-btn:hover{{background:{TN_COL};color:{INK};border-color:{TN_COL};}}'
        '</style></head><body>{%app_entry%}'
        '<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'
    )

    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Span('HARRAPATU', style={'color':CYAN,'fontWeight':'800','letterSpacing':'0.08em'}),
                html.Span(' · ', style={'color':INK_FAINT,'margin':'0 4px'}),
                html.Span('inspector', style={'color':INK,'fontWeight':'300'}),
            ], style={'fontSize':'18px','fontFamily':'ui-monospace,monospace'}),
            html.Div('cetacean passive acoustic monitoring  ·  model vs expert  ·  5-min interval view',
                     style={'color':INK_FAINT,'fontSize':'10px','letterSpacing':'0.1em',
                            'textTransform':'uppercase','marginTop':'3px',
                            'fontFamily':'ui-monospace,monospace'}),
        ], style={'marginBottom':'16px'}),

        # badges — static, rendered once
        BADGES,

        html.Div([
            html.Label('view', style=lbl),
            dcc.Dropdown(id='mode', clearable=False, value='all',
                         options=[{'label':v,'value':k} for k,v in MODES],
                         style={'width':'260px'}),
            html.Div([
                html.Span('jump to:', style={**lbl,'marginLeft':'16px'}),
                html.Button('TP',id='btn-tp',n_clicks=0,className='ofbtn tp-btn'),
                html.Button('FP',id='btn-fp',n_clicks=0,className='ofbtn fp-btn'),
                html.Button('FN',id='btn-fn',n_clicks=0,className='ofbtn fn-btn'),
                html.Button('TN',id='btn-tn',n_clicks=0,className='ofbtn tn-btn'),
            ], style={'display':'flex','alignItems':'center','gap':'6px'}),
            html.Div(style={'flex':'1'}),
            html.Button('◀ prev',id='prev',n_clicks=0,className='hbtn'),
            dcc.Input(id='jump',type='number',value=1,min=1,step=1,
                      style={'width':'68px','textAlign':'center','margin':'0 6px',
                             'background':PANEL2,'color':INK,'border':f'1px solid {LINE}',
                             'borderRadius':'8px','padding':'6px 8px',
                             'fontFamily':'ui-monospace,monospace','fontSize':'12px'}),
            html.Button('go',id='gobtn',n_clicks=0,className='hbtn'),
            html.Button('next ▶',id='next',n_clicks=0,className='hbtn',
                        style={'marginLeft':'6px'}),
        ], style={'display':'flex','alignItems':'center','flexWrap':'wrap',
                  'gap':'6px','margin':'16px 0 10px'}),

        html.Div(id='status', style={'color':INK_FAINT,'fontSize':'11px',
                                      'marginBottom':'10px',
                                      'fontFamily':'ui-monospace,monospace'}),

        html.Div([
            html.Div([
                html.Span('OVERVIEW', style={'color':INK_FAINT,'fontSize':'10px',
                                              'letterSpacing':'0.1em',
                                              'fontFamily':'ui-monospace,monospace'}),
                html.Span('  TP=lime  FP=amber  FN=rose  TN=dim  ·  taller = TP/FN  ·  current outlined',
                           style={'color':INK_FAINT,'fontSize':'10px',
                                  'fontFamily':'ui-monospace,monospace','marginLeft':'8px'}),
            ], style={'marginBottom':'6px'}),
            html.Div(id='overview'),
        ], style={'background':PANEL,'border':f'1px solid {LINE}','borderRadius':'12px',
                  'padding':'14px 16px','marginBottom':'12px'}),

        html.Div(id='interval-header', style={'marginBottom':'8px'}),

        html.Div([
            RAIL,
            html.Div(id='filmstrip', style={'flex':'1 1 auto','minWidth':'0'}),
        ], style={'display':'flex','flexDirection':'row','alignItems':'flex-start'}),

        html.Div(id='detail', style={'background':PANEL2,'border':f'1px solid {LINE}',
                                      'borderRadius':'12px','marginTop':'10px',
                                      'minHeight':'80px'}),
        dcc.Store(id='page', data=0),
    ], style={'fontFamily':'system-ui,-apple-system,sans-serif','padding':'24px 28px',
              'background':BG,'minHeight':'100vh','color':INK})

    # ── callbacks ─────────────────────────────────────────────────────────────
    @app.callback(Output('mode','value'),
                  Input('btn-tp','n_clicks'),Input('btn-fp','n_clicks'),
                  Input('btn-fn','n_clicks'),Input('btn-tn','n_clicks'),
                  State('mode','value'), prevent_initial_call=True)
    def outcome_btns(tp,fp,fn,tn,cur):
        m={'btn-tp':'true_positive','btn-fp':'false_positive',
           'btn-fn':'false_negative','btn-tn':'true_negative'}
        return m.get(ctx.triggered_id, cur)

    @app.callback(
        Output('filmstrip','children'), Output('detail','children'),
        Output('interval-header','children'), Output('status','children'),
        Output('page','data'), Output('jump','value'), Output('overview','children'),
        Input('prev','n_clicks'), Input('next','n_clicks'),
        Input('gobtn','n_clicks'), Input('mode','value'),
        State('page','data'), State('jump','value'),
    )
    def render(prev_c, next_c, go_c, mode, page, jump):
        intervals = mode_lists.get(mode, all_intervals_ordered)
        n = len(intervals)
        trig = ctx.triggered_id

        if trig == 'mode':   page = 0
        elif trig == 'prev': page = max(0, (page or 0) - 1)
        elif trig == 'next': page = min(n-1, (page or 0) + 1)
        elif trig == 'gobtn': page = min(n-1, max(0, int(jump or 1)-1))
        page = min(max(0, page or 0), max(0, n-1))

        ov = overview_svg(intervals, ivl_meta, page)

        if n == 0:
            empty = html.Div('No intervals match.',
                             style={'color':INK_FAINT,'padding':'40px',
                                    'fontFamily':'ui-monospace,monospace'})
            return empty, html.Div(), html.Div(), 'No intervals.', page, 1, ov

        ivl_token = intervals[page]
        ivl_df = grouped.get(ivl_token, pd.DataFrame())
        if len(ivl_df) == 0:
            return (html.Div('Interval not found.'), html.Div(), html.Div(),
                    f'interval {ivl_token} not found', page, page+1, ov)

        ivl_df = ivl_df.sort_values(['wav_name','offset_s'])
        sample  = ivl_df.iloc[0]
        oc      = str(sample.get('outcome',''))
        date_s  = token_to_date(str(ivl_token))
        clock_s = token_to_clock(str(ivl_token))
        src     = str(sample.get('source_file',''))
        hydro   = str(sample.get('hydrophone',''))

        header = html.Div([
            outcome_badge(oc),
            html.Span(f'  {date_s}  {clock_s}',
                      style={'color':INK,'fontWeight':'700',
                             'fontFamily':'ui-monospace,monospace',
                             'fontSize':'14px','marginLeft':'8px'}),
            html.Span(f'  ·  {ivl_token}',
                      style={'color':INK_DIM,'fontSize':'11px',
                             'fontFamily':'ui-monospace,monospace'}),
            html.Span(f'  ·  {ivl_df["wav_name"].nunique()} recorder(s)',
                      style={'color':INK_FAINT,'fontSize':'11px',
                             'fontFamily':'ui-monospace,monospace'}),
            *([] if not src or src=='nan' else [
                html.Span(f'  ·  {src}  ({hydro})',
                          style={'color':AMBER,'fontSize':'11px',
                                 'fontFamily':'ui-monospace,monospace'})]),
        ], style={'display':'flex','alignItems':'center','flexWrap':'wrap'})

        strips = [recorder_strip(wn, wdf, SPECTRO_URL, fmax)
                  for wn, wdf in ivl_df.groupby('wav_name', sort=True)]

        status = (f"interval {page+1:,} / {n:,}   ·   {ivl_token}   ·   "
                  f"{date_s}  {clock_s}   ·   outcome: {oc}   ·   mode: {mode}")

        return strips, detail_panel(ivl_df), header, status, page, page+1, ov

    return app


def main():
    global FREQ_MAX_KHZ
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',          default='harrapatu_comparison_5min_v1_segments.csv')
    ap.add_argument('--spectrograms', required=True)
    ap.add_argument('--port',         type=int, default=8077)
    ap.add_argument('--host',         default='127.0.0.1')
    ap.add_argument('--freq-max-khz', type=float, default=16.0)
    ap.add_argument('--debug',        action='store_true')
    args = ap.parse_args()
    FREQ_MAX_KHZ = args.freq_max_khz

    df = pd.read_csv(args.csv, low_memory=False)
    spectro_dir = Path(args.spectrograms)
    print(f"Loaded {len(df):,} segments, {df['interval'].nunique():,} intervals")

    app = make_app(df, spectro_dir, args.freq_max_khz)
    try:    app.run(host=args.host, port=args.port, debug=args.debug)
    except AttributeError: app.run_server(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()