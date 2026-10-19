from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='AML Sentinel', page_icon='🛡️', layout='wide')
st.markdown("""
<style>
.block-container{padding-top:1.15rem;padding-bottom:2.2rem;max-width:1600px}
[data-testid="stMetric"]{background:linear-gradient(135deg,rgba(59,130,246,.09),rgba(14,165,233,.04));border:1px solid rgba(100,116,139,.22);padding:14px 16px;border-radius:14px}
.hero{padding:1.35rem 1.6rem;border-radius:18px;background:linear-gradient(115deg,rgba(30,64,175,.20),rgba(14,165,233,.08));border:1px solid rgba(100,116,139,.24);margin-bottom:1.1rem}
.hero h1{margin:0;font-size:2.2rem}.hero p{margin:.4rem 0 0 0;opacity:.82}
[data-testid="stTabs"] button{font-weight:650}
</style>
""", unsafe_allow_html=True)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

def read_json(name):
    p = OUT / name
    if not p.exists(): return None
    try: return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError): return None

def read_csv(name):
    p = OUT / name
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

def flatten(obj, prefix=''):
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v, dict): result.update(flatten(v, key))
            elif not isinstance(v, list): result[key] = v
    return result

def metric(data, *names):
    if data is None: return None
    names = {n.lower().replace('-', '_').replace(' ', '_') for n in names}
    for key, value in flatten(data).items():
        leaf = key.split('.')[-1].lower().replace('-', '_').replace(' ', '_')
        if leaf in names or any(leaf.endswith('_' + n) for n in names):
            try: return float(value)
            except (TypeError, ValueError): pass
    return None

def fmt(value, percent=False):
    if value is None: return 'Not available'
    try:
        x = float(value)
        if percent:
            if 0 <= x <= 1: x *= 100
            return f'{x:.2f}%'
        return f'{x:,.0f}' if abs(x) >= 1000 else f'{x:.4f}'.rstrip('0').rstrip('.')
    except (TypeError, ValueError): return str(value)

def show_table(frame, height=320):
    if frame is None or frame.empty:
        st.info('No data file found for this section yet.')
        return
    frame = frame.copy()
    def safe_display(value):
        if value is None:
            return ''
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, ensure_ascii=False, default=str)
        if hasattr(value, 'tolist') and not isinstance(value, str):
            try:
                return json.dumps(value.tolist(), ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                pass
        try:
            missing = pd.isna(value)
            if getattr(missing, 'ndim', 0) == 0 and bool(missing):
                return ''
        except (TypeError, ValueError):
            pass
        return str(value)
    for col in frame.columns:
        frame[col] = frame[col].map(safe_display)
    st.dataframe(frame, width='stretch', hide_index=True, height=height)

metrics = read_json('evaluation_metrics.json')
rules = read_json('rule_validation_comparison.json')
ablation = read_json('rule_ablation_validation.json')
thresholds = read_json('fanin_threshold_comparison_test.json')
network = read_json('network_pattern_summary.json')
coverage = read_csv('test_fanin_pattern_coverage.csv')
accounts = read_csv('network_pattern_accounts.csv')
edges = read_csv('network_pattern_edges.csv')
precision = metric(metrics, 'precision', 'test_precision')
recall = metric(metrics, 'recall', 'test_recall', 'sensitivity')
f1 = metric(metrics, 'f1', 'f1_score', 'test_f1')
f2 = metric(metrics, 'f2', 'f2_score', 'test_f2')
total = metric(metrics, 'total_rows', 'rows_processed', 'transactions_processed')
tp = metric(metrics, 'true_positives', 'tp')
fp = metric(metrics, 'false_positives', 'fp')
fn = metric(metrics, 'false_negatives', 'fn')
alerts = metric(metrics, 'alerts', 'alert_count', 'total_alerts')

st.markdown('''<style>.block-container{padding-top:1.4rem}.hero{padding:1.25rem 1.5rem;border-radius:14px;background:linear-gradient(120deg,rgba(30,64,175,.16),rgba(14,165,233,.08));border:1px solid rgba(100,116,139,.25);margin-bottom:1rem}.hero h1{margin:0}.hero p{margin:.4rem 0 0 0;opacity:.8}</style>''', unsafe_allow_html=True)
with st.sidebar:
    st.title('🛡️ AML Sentinel')
    st.caption('Big Data Analytics for Anti-Money Laundering')
    st.markdown('**Sections**\n\n- Overview\n- Network Analytics\n- Detection Evaluation')
    st.divider()
    st.caption('Metrics are loaded from local outputs. Missing values are not invented.')
st.markdown('''<div class="hero"><h1>🛡️ AML Sentinel</h1><p>Interactive anti-money-laundering analytics · Big-data detection · Transaction-network intelligence</p></div>''', unsafe_allow_html=True)
overview, network_tab, evaluation, stream_tab = st.tabs(['📊 Overview', '🕸️ Network Analytics', '🧪 Detection Evaluation', '📡 Kafka Stream'])

with overview:
    st.subheader('Project at a glance')
    st.write('AML Sentinel analyzes transaction data to identify suspicious activity using risk signals and transaction-network patterns. Evaluation results are shown transparently.')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Transactions processed', fmt(total) if total is not None else 'See run output')
    c2.metric('Precision', fmt(precision, True)); c3.metric('Recall', fmt(recall, True)); c4.metric('F1-score', fmt(f1, True))
    st.markdown('### Analysis workflow')
    cols = st.columns(4)
    for col,title,desc in zip(cols,['1 · Data','2 · Analytics','3 · Detection','4 · Evaluation'],['Load and prepare transaction records.','Engineer behavioural and transaction-risk signals.','Score transactions and flag suspicious activity.','Measure alert quality and inspect network patterns.']):
        with col: st.markdown(f'**{title}**'); st.write(desc)
    st.markdown('### Available output artifacts')
    show_table(pd.DataFrame([
        {'Artifact':'Evaluation metrics','Status':'Available' if metrics is not None else 'Not found'},
        {'Artifact':'Rule comparison','Status':'Available' if rules is not None else 'Not found'},
        {'Artifact':'Rule ablation','Status':'Available' if ablation is not None else 'Not found'},
        {'Artifact':'Fan-in threshold comparison','Status':'Available' if thresholds is not None else 'Not found'},
        {'Artifact':'Network summary','Status':'Available' if network is not None else 'Not found'},
        {'Artifact':'Pattern coverage','Status':'Available' if not coverage.empty else 'Not found'}]), 260)
    st.markdown('### Why can precision and recall be low?')
    st.warning('Low precision means many flagged transactions are false positives. Low recall means many known suspicious transactions are missed. A likely contributing factor is that fixed rules and thresholds capture only a limited set of suspicious behaviours; legitimate high-value or frequent transactions can look similar to suspicious ones.')
    st.markdown('**Other factors to investigate**')
    st.markdown('- **Class imbalance:** suspicious transactions are rare compared with legitimate activity.\n- **Limited behaviour signals:** rules may miss new counterparties, account-level changes, timing patterns, or multi-hop relationships.\n- **Threshold trade-off:** stricter rules can reduce false alerts but may also miss more suspicious cases.\n- **Graph scope:** graph metrics are based on selected suspicious edges, not necessarily the complete transaction network.\n- **Labels and time split:** label quality and temporal differences can affect measured performance.')
    st.caption('These are plausible explanations to investigate, not individually proven causes. Metrics shown here are loaded from saved evaluation results and are not altered.')


    st.markdown('### Performance snapshot')
    perf_values = []
    for label, value in [('Precision', precision), ('Recall', recall), ('F1-score', f1), ('F2-score', f2)]:
        if value is not None:
            value = float(value) * 100 if 0 <= float(value) <= 1 else float(value)
            perf_values.append({'Metric': label, 'Score (%)': value})
    if perf_values:
        perf_df = pd.DataFrame(perf_values)
        left, right = st.columns(2)
        with left:
            fig = px.bar(perf_df, x='Metric', y='Score (%)', color='Metric',
                         title='Detection metric comparison', text_auto='.2f',
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(showlegend=False, yaxis_title='Percent (%)')
            st.plotly_chart(fig, width='stretch')
        with right:
            fig = go.Figure(go.Pie(labels=perf_df['Metric'], values=perf_df['Score (%)'], hole=.58,
                                   textinfo='label+percent'))
            fig.update_layout(title='Metric profile (relative share)', height=400)
            st.plotly_chart(fig, width='stretch')
    artifact_df = pd.DataFrame([
        {'Artifact': 'Evaluation metrics', 'Ready': int(metrics is not None)},
        {'Artifact': 'Network summary', 'Ready': int(network is not None)},
        {'Artifact': 'Network accounts', 'Ready': int(not accounts.empty)},
        {'Artifact': 'Network edges', 'Ready': int(not edges.empty)},
        {'Artifact': 'Rule comparison', 'Ready': int(rules is not None)},
        {'Artifact': 'Pattern coverage', 'Ready': int(not coverage.empty)},
    ])
    left, right = st.columns(2)
    with left:
        fig = px.bar(artifact_df, x='Artifact', y='Ready', color='Ready',
                     title='Data artifact readiness', range_y=[0, 1.15],
                     color_continuous_scale=['#CBD5E1', '#0D9488'])
        fig.update_layout(xaxis_tickangle=-25, yaxis=dict(tickvals=[0, 1], ticktext=['Missing', 'Available']))
        st.plotly_chart(fig, width='stretch')
    with right:
        st.markdown('#### How to read this dashboard')
        st.write('Use Overview for the project snapshot, Network Analytics for account relationships, Detection Evaluation for measured quality, and Kafka Stream for the optional streaming demonstration.')
        st.caption('Charts are generated only from available saved data. No evaluation values are fabricated.')

with network_tab:
    st.subheader('Transaction network analytics')
    if isinstance(network, dict):
        rows = [{'Measure':k.split('.')[-1].replace('_',' ').title(),'Value':v} for k,v in flatten(network).items() if isinstance(v,(int,float,str))]
        if rows: show_table(pd.DataFrame(rows), 230)
        else: st.info('Network summary exists but has no simple summary fields.')
    else: st.info('Network summary not found. Run src/analytics/pattern_graph_analysis.py.')
    st.markdown('### Network structure visualizations')
    if not edges.empty:
        src = next((c for c in edges.columns if c.lower() in ('source','sender_account','sender','from_account','src')), None)
        dst = next((c for c in edges.columns if c.lower() in ('target','receiver_account','receiver','to_account','dst')), None)
        if src and dst:
            try:
                import networkx as nx
                graph_edges = edges[[src, dst]].dropna().astype(str).head(150)
                graph = nx.DiGraph()
                graph.add_edges_from(graph_edges.itertuples(index=False, name=None))
                if graph.number_of_nodes():
                    pos = nx.spring_layout(graph, seed=42)
                    ex, ey = [], []
                    for u, v in graph.edges():
                        ex.extend([pos[u][0], pos[v][0], None]); ey.extend([pos[u][1], pos[v][1], None])
                    edge_trace = go.Scatter(x=ex, y=ey, mode='lines', line=dict(width=1, color='#9CA3AF'), hoverinfo='skip')
                    nodes = list(graph.nodes())
                    node_trace = go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode='markers+text',
                        text=[n for n in nodes], textposition='top center',
                        marker=dict(size=[10 + min(graph.degree(n)*2, 18) for n in nodes], color=[graph.degree(n) for n in nodes], colorscale='Blues', showscale=True),
                        hovertext=[f'Account: {n}<br>Degree: {graph.degree(n)}' for n in nodes], hoverinfo='text')
                    fig = go.Figure([edge_trace, node_trace])
                    fig.update_layout(title='Interactive account relationship graph (up to 150 edges)', height=550,
                        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10,r=10,t=50,b=10))
                    st.plotly_chart(fig, width='stretch')
                    st.caption('This is a sampled graph of the available edge file, not the complete transaction network.')
            except Exception as exc:
                st.caption(f'Network graph could not be drawn from the current edge schema: {exc}')
        else:
            st.caption('A node-link graph needs source/sender and target/receiver columns in the edge CSV.')

    if not accounts.empty:
        account_col = next((c for c in accounts.columns if c.lower() in ('account','account_id','node')), accounts.columns[0])
        in_col = next((c for c in accounts.columns if c.lower() == 'in_degree'), None)
        out_col = next((c for c in accounts.columns if c.lower() == 'out_degree'), None)
        rank_col = next((c for c in accounts.columns if c.lower() in ('pagerank','page_rank')), None)
        degree_col = next((c for c in accounts.columns if c.lower() in ('total_degree','degree_centrality')), None)
        if degree_col is None and in_col and out_col:
            accounts['_total_degree_chart'] = pd.to_numeric(accounts[in_col], errors='coerce').fillna(0) + pd.to_numeric(accounts[out_col], errors='coerce').fillna(0)
            degree_col = '_total_degree_chart'
        left_chart, right_chart = st.columns(2)
        if degree_col:
            chart_df = accounts.copy(); chart_df[degree_col] = pd.to_numeric(chart_df[degree_col], errors='coerce')
            chart_df = chart_df.dropna(subset=[degree_col]).nlargest(20, degree_col)
            with left_chart:
                fig = px.bar(chart_df, x=degree_col, y=account_col, orientation='h', color=degree_col, title='Top 20 accounts by total degree', color_continuous_scale='Blues')
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=480)
                st.plotly_chart(fig, width='stretch')
        if in_col and out_col:
            chart_df = accounts.copy()
            chart_df[in_col] = pd.to_numeric(chart_df[in_col], errors='coerce'); chart_df[out_col] = pd.to_numeric(chart_df[out_col], errors='coerce')
            with right_chart:
                fig = px.scatter(chart_df, x=in_col, y=out_col, hover_name=account_col, title='Incoming vs outgoing activity', size_max=18)
                fig.update_layout(height=480)
                st.plotly_chart(fig, width='stretch')
        if rank_col:
            chart_df = accounts.copy(); chart_df[rank_col] = pd.to_numeric(chart_df[rank_col], errors='coerce')
            chart_df = chart_df.dropna(subset=[rank_col]).nlargest(20, rank_col)
            fig = px.bar(chart_df, x=account_col, y=rank_col, title='Top accounts by PageRank', color=rank_col, color_continuous_scale='Teal')
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, width='stretch')

    if not accounts.empty:
        st.markdown('### Explore account behaviour')
        numeric_options = [c for c in accounts.columns if c.lower() not in ('account','account_id','node') and pd.to_numeric(accounts[c], errors='coerce').notna().sum() > 0]
        label_option = next((c for c in accounts.columns if c.lower() in ('account','account_id','node')), accounts.columns[0])
        if numeric_options:
            control1, control2, control3 = st.columns([1.2,1.2,1])
            with control1:
                chart_kind = st.selectbox('Visualization', ['Bubble chart','Degree distribution','Box plot','Correlation heatmap','Treemap'], key='network_chart_kind')
            with control2:
                metric_x = st.selectbox('Primary metric', numeric_options, index=0, key='network_metric_x')
            with control3:
                metric_y = st.selectbox('Compare with', numeric_options, index=min(1, len(numeric_options)-1), key='network_metric_y')
            chart_data = accounts.copy()
            for col in numeric_options:
                chart_data[col] = pd.to_numeric(chart_data[col], errors='coerce')
            chart_data = chart_data.dropna(subset=[metric_x])
            if chart_kind == 'Bubble chart':
                if metric_y == metric_x and len(numeric_options) > 1:
                    metric_y = numeric_options[1] if numeric_options[0] == metric_x else numeric_options[0]
                fig = px.scatter(chart_data, x=metric_x, y=metric_y, hover_name=label_option,
                                 size=metric_x, color=metric_y, size_max=36,
                                 title=f'Account map: {metric_x} vs {metric_y}',
                                 color_continuous_scale='Viridis')
                fig.update_layout(height=520)
                st.plotly_chart(fig, width='stretch')
            elif chart_kind == 'Degree distribution':
                fig = px.histogram(chart_data, x=metric_x, nbins=35, marginal='box',
                                   title=f'Distribution of {metric_x}', color_discrete_sequence=['#2563EB'])
                fig.update_layout(height=520)
                st.plotly_chart(fig, width='stretch')
            elif chart_kind == 'Box plot':
                fig = px.box(chart_data, y=metric_x, points='all', hover_name=label_option,
                             title=f'Spread and outliers: {metric_x}', color_discrete_sequence=['#0D9488'])
                fig.update_layout(height=520)
                st.plotly_chart(fig, width='stretch')
            elif chart_kind == 'Correlation heatmap':
                corr = chart_data[numeric_options].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1, title='Correlation between account metrics')
                fig.update_layout(height=520)
                st.plotly_chart(fig, width='stretch')
            else:
                tree = chart_data.nlargest(80, metric_x).copy()
                tree[label_option] = tree[label_option].astype(str)
                fig = px.treemap(tree, path=[label_option], values=metric_x, color=metric_y,
                                 title=f'Account hierarchy view: size={metric_x}, colour={metric_y}',
                                 color_continuous_scale='Blues')
                fig.update_layout(height=520)
                st.plotly_chart(fig, width='stretch')
            with st.expander('Show chart data and filters'):
                min_value = float(chart_data[metric_x].min()) if not chart_data.empty else 0.0
                max_value = float(chart_data[metric_x].max()) if not chart_data.empty else 1.0
                if min_value < max_value:
                    chosen_range = st.slider('Filter accounts by selected metric', min_value, max_value, (min_value, max_value), key='network_metric_range')
                    filtered = chart_data[chart_data[metric_x].between(chosen_range[0], chosen_range[1])]
                else:
                    filtered = chart_data
                show_table(filtered.head(250), 300)

    left,right = st.columns(2)
    with left:
        st.markdown('### Network accounts')
        if not accounts.empty: st.caption(f'{len(accounts):,} rows available; showing up to 100.'); show_table(accounts.head(100),330)
        else: st.info('network_pattern_accounts.csv not found.')
    with right:
        st.markdown('### Network edges')
        if not edges.empty: st.caption(f'{len(edges):,} rows available; showing up to 100.'); show_table(edges.head(100),330)
        else: st.info('network_pattern_edges.csv not found.')
    if not coverage.empty:
        st.markdown('### Pattern coverage'); show_table(coverage,360)
        name_col = next((c for c in coverage.columns if c.lower() in ('typology','pattern','name','typology_name')), coverage.columns[0])
        recall_col = next((c for c in coverage.columns if 'current' in c.lower() and 'recall' in c.lower()), None)
        if recall_col:
            chart = coverage[[name_col,recall_col]].copy(); chart[recall_col] = pd.to_numeric(chart[recall_col],errors='coerce'); chart = chart.dropna()
            if not chart.empty: st.markdown('#### Current coverage by typology'); st.bar_chart(chart.set_index(name_col)[recall_col])

with evaluation:
    st.subheader('Detection performance')
    st.write('Precision measures the proportion of alerts that are correct; recall measures the proportion of known suspicious transactions detected.')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Precision',fmt(precision,True)); c2.metric('Recall',fmt(recall,True)); c3.metric('F1-score',fmt(f1,True)); c4.metric('F2-score',fmt(f2,True))
    if any(v is not None for v in [tp,fp,fn,alerts]):
        a,b,c,d = st.columns(4); a.metric('True positives',fmt(tp)); b.metric('False positives',fmt(fp)); c.metric('False negatives',fmt(fn)); d.metric('Alerts',fmt(alerts))
    st.info('Precision = TP / (TP + FP): the share of alerts that are correct. Recall = TP / (TP + FN): the share of known suspicious transactions detected. The current low values indicate false alarms and missed positives; rule coverage, class imbalance, feature limits, and threshold choices should be investigated.')
    tn = metric(metrics, 'true_negatives', 'tn')
    if all(v is not None for v in [tp, fp, fn, tn]):
        matrix = pd.DataFrame([[tn, fp], [fn, tp]], index=['Actual legitimate', 'Actual suspicious'], columns=['Predicted legitimate', 'Predicted alert'])
        fig = px.imshow(matrix, text_auto='.0f', aspect='auto', color_continuous_scale='Blues', title='Confusion matrix')
        st.plotly_chart(fig, width='stretch')
    if metrics is not None:
        st.markdown('### Saved evaluation metrics'); show_table(pd.DataFrame([{'Metric':k,'Value':v} for k,v in flatten(metrics).items()]),280)
    st.markdown('### Rule comparison')
    if isinstance(rules,list): show_table(pd.DataFrame(rules))
    elif isinstance(rules,dict): show_table(pd.DataFrame([{'Rule / strategy':k,**v} if isinstance(v,dict) else {'Rule / strategy':k,'Value':v} for k,v in rules.items()]))
    else: st.info('rule_validation_comparison.json not found.')
    if isinstance(rules, dict):
        rows = [{'Strategy': k, **v} for k, v in rules.items() if isinstance(v, dict)]
        if rows:
            df = pd.DataFrame(rows)
            pcol = next((c for c in df.columns if c.lower() == 'precision'), None)
            rcol = next((c for c in df.columns if c.lower() == 'recall'), None)
            if pcol and rcol:
                for c in [pcol, rcol]: df[c] = pd.to_numeric(df[c], errors='coerce')
                plot = df.melt(id_vars=['Strategy'], value_vars=[pcol, rcol], var_name='Metric', value_name='Score')
                fig = px.bar(plot, x='Strategy', y='Score', color='Metric', barmode='group', title='Precision and recall by strategy')
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, width='stretch')

    st.markdown('### Individual rule contribution (validation)')
    if isinstance(ablation,list): show_table(pd.DataFrame(ablation))
    elif isinstance(ablation,dict):
        candidate = next((ablation[k] for k in ('results','rules','comparisons','ablation_results') if isinstance(ablation.get(k),list)),None)
        if candidate is not None: show_table(pd.DataFrame(candidate))
        else: show_table(pd.DataFrame([{'Rule':k,**v} if isinstance(v,dict) else {'Rule':k,'Value':v} for k,v in ablation.items()]))
    else: st.info('rule_ablation_validation.json not found.')
    st.markdown('### Fan-in threshold comparison')
    if isinstance(thresholds,list): show_table(pd.DataFrame(thresholds))
    elif isinstance(thresholds,dict):
        candidate = next((thresholds[k] for k in ('results','thresholds','comparisons') if isinstance(thresholds.get(k),list)),None)
        if candidate is not None: show_table(pd.DataFrame(candidate))
        else: show_table(pd.DataFrame([{'Threshold / strategy':k,**v} if isinstance(v,dict) else {'Threshold / strategy':k,'Value':v} for k,v in thresholds.items()]))
    else: st.info('fanin_threshold_comparison_test.json not found.')
    st.markdown('### Interactive metric explorer')
    candidate_frames = []
    if isinstance(rules, dict):
        for name, values in rules.items():
            if isinstance(values, dict): candidate_frames.append({'Strategy': name, **values})
    elif isinstance(rules, list):
        candidate_frames = rules
    if candidate_frames:
        metric_df = pd.DataFrame(candidate_frames)
        metric_cols = [c for c in metric_df.columns if c.lower() in ('precision','recall','f1','f1_score','f2','f2_score','alerts','alert_count')]
        if metric_cols:
            selected_metrics = st.multiselect('Metrics to compare', metric_cols, default=metric_cols[:min(3,len(metric_cols))], key='evaluation_metric_select')
            if selected_metrics:
                strategy_col = next((c for c in metric_df.columns if c.lower() in ('strategy','rule','name')), metric_df.columns[0])
                long_df = metric_df.melt(id_vars=[strategy_col], value_vars=selected_metrics, var_name='Metric', value_name='Value')
                long_df['Value'] = pd.to_numeric(long_df['Value'], errors='coerce')
                long_df = long_df.dropna(subset=['Value'])
                if not long_df.empty:
                    graph_style = st.radio('Comparison style', ['Grouped bars','Dot plot'], horizontal=True, key='evaluation_graph_style')
                    if graph_style == 'Grouped bars':
                        fig = px.bar(long_df, x=strategy_col, y='Value', color='Metric', barmode='group',
                                     title='Compare detection strategies', hover_data=['Metric','Value'])
                    else:
                        fig = px.scatter(long_df, x='Value', y=strategy_col, color='Metric', symbol='Metric',
                                         title='Detection strategy metric comparison', hover_data=['Metric','Value'])
                    fig.update_layout(xaxis_tickangle=-25, height=480)
                    st.plotly_chart(fig, width='stretch')

    st.caption('Previously explored TEST-period threshold results are exploratory, not an untouched final holdout. Use validation results for rule selection.')


with stream_tab:
    st.subheader('Kafka streaming monitor')
    st.caption('This section is optional. Historical Spark analytics and Kafka demo events are separate pipelines and their counts should not be combined.')
    stream_events = pd.DataFrame()
    stream_alerts = pd.DataFrame()
    stream_summary = {}
    for name, target in [('streaming_events.jsonl', 'events'), ('streaming_alerts.jsonl', 'alerts')]:
        path = OUT / name
        rows = []
        if path.exists():
            try:
                with path.open('r', encoding='utf-8') as handle:
                    lines = handle.readlines()[-5000:]
                for line in lines:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict): rows.append(obj)
                    except json.JSONDecodeError:
                        continue
            except OSError:
                rows = []
        if target == 'events': stream_events = pd.DataFrame(rows)
        else: stream_alerts = pd.DataFrame(rows)
    stream_summary = read_json('streaming_summary.json')
    a, b, c, d = st.columns(4)
    a.metric('Saved events', f'{len(stream_events):,}')
    b.metric('Saved alerts', f'{len(stream_alerts):,}')
    risk_col = next((c for c in stream_events.columns if c.lower() in ('stream_risk_level', 'risk_level')), None) if not stream_events.empty else None
    if risk_col:
        high = int((stream_events[risk_col].astype(str).str.upper() == 'HIGH').sum())
        medium = int((stream_events[risk_col].astype(str).str.upper() == 'MEDIUM').sum())
        low = int((stream_events[risk_col].astype(str).str.upper() == 'LOW').sum())
    else:
        high = medium = low = 0
    c.metric('High-risk events', f'{high:,}')
    d.metric('Medium-risk events', f'{medium:,}')
    if not stream_events.empty:
        if risk_col:
            counts = stream_events[risk_col].astype(str).value_counts().rename_axis('Risk level').reset_index(name='Events')
            left, right = st.columns(2)
            with left:
                fig = px.bar(counts, x='Risk level', y='Events', color='Risk level', title='Events by risk level', text_auto=True)
                st.plotly_chart(fig, width='stretch')
            with right:
                fig = px.pie(counts, names='Risk level', values='Events', hole=.5, title='Risk mix')
                st.plotly_chart(fig, width='stretch')
        amount_col = next((c for c in stream_events.columns if c.lower() in ('amount', 'transaction_amount')), None)
        time_col = next((c for c in stream_events.columns if c.lower() in ('event_time', 'timestamp', 'created_at', 'stream_processed_at')), None)
        if amount_col:
            amounts = pd.to_numeric(stream_events[amount_col], errors='coerce').dropna()
            if not amounts.empty:
                left, right = st.columns(2)
                with left:
                    fig = px.histogram(pd.DataFrame({'Amount': amounts}), x='Amount', nbins=30, marginal='box', title='Streaming transaction amount distribution')
                    st.plotly_chart(fig, width='stretch')
                with right:
                    fig = px.box(pd.DataFrame({'Amount': amounts}), y='Amount', points='outliers', title='Amount outlier view')
                    st.plotly_chart(fig, width='stretch')
        if time_col:
            timed = stream_events.copy()
            timed[time_col] = pd.to_datetime(timed[time_col], errors='coerce')
            timed = timed.dropna(subset=[time_col]).sort_values(time_col)
            if not timed.empty:
                timeline = timed.set_index(time_col).resample('1min').size().rename('Events').reset_index()
                fig = px.line(timeline, x=time_col, y='Events', markers=True, title='Streaming event timeline')
                st.plotly_chart(fig, width='stretch')
        with st.expander('Recent stream events'):
            show_table(stream_events.tail(150), 320)
    else:
        st.info('No streaming event log found. You can still demonstrate the other three sections without Kafka.')
    if not stream_alerts.empty:
        st.markdown('### Saved streaming alerts')
        show_table(stream_alerts.tail(200), 320)
        st.download_button('Download streaming alerts CSV', stream_alerts.to_csv(index=False).encode('utf-8'),
                           file_name='aml_sentinel_streaming_alerts.csv', mime='text/csv')
    if stream_summary:
        with st.expander('Consumer run summary'):
            st.json(stream_summary)

st.divider()
st.caption('AML Sentinel · Academic demonstration dashboard · Results are sourced from local output artifacts.')
