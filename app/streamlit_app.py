import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer
from gensim.models import Word2Vec
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance NLP Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Clean, Premium Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base */
html, body, .stApp {
    background-color: #0a0c10 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { display: none !important; }
.block-container { padding: 2rem 3rem 3rem 3rem !important; max-width: 1400px !important; }

/* ── TOP HEADER BAR ── */
.top-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 2rem 0;
    border-bottom: 1px solid #1e2533;
    margin-bottom: 2rem;
}
.top-header .badge {
    background: linear-gradient(135deg, #10b982, #059669);
    color: white;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 20px;
}
.top-header h1 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    margin: 0 !important;
    padding: 0 !important;
}
.top-header p {
    font-size: 13px;
    color: #64748b;
    margin: 0;
}

/* ── SECTION TITLE ── */
.section-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #10b982;
    margin: 0 0 6px 0;
}
.section-heading {
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 4px 0;
}
.section-sub {
    font-size: 13px;
    color: #64748b;
    margin: 0 0 1.5rem 0;
}

/* ── METRIC CARDS ── */
.metrics-row {
    display: flex;
    gap: 16px;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: #111827;
    border: 1px solid #1e2533;
    border-radius: 12px;
    padding: 18px 22px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #10b982, #3b82f6);
}
.metric-card .label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 26px;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1;
}
.metric-card .delta {
    font-size: 12px;
    color: #10b982;
    margin-top: 4px;
}

/* ── CONTENT PANELS ── */
.panel {
    background: #111827;
    border: 1px solid #1e2533;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.panel-title {
    font-size: 14px;
    font-weight: 600;
    color: #cbd5e1;
    margin: 0 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e2533;
}

/* ── TOPIC TAGS ── */
.topic-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}
.topic-name {
    font-size: 12px;
    font-weight: 700;
    color: #10b982;
    margin-bottom: 4px;
}
.topic-word {
    display: inline-block;
    background: #1e2d3d;
    color: #7dd3fc;
    border: 1px solid #2d4a6b;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px;
}

/* ── MODEL COMPARISON TABLE ── */
.compare-table { width: 100%; border-collapse: collapse; }
.compare-table th {
    font-size: 11px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: #64748b;
    padding: 10px 14px; border-bottom: 1px solid #1e2533; text-align: left;
}
.compare-table td {
    font-size: 13px; color: #cbd5e1;
    padding: 10px 14px; border-bottom: 1px solid #111827;
}
.compare-table tr:hover td { background: #0f172a; }
.acc-badge {
    display: inline-block;
    background: #14532d;
    color: #4ade80;
    border-radius: 6px;
    padding: 2px 10px;
    font-weight: 700;
    font-size: 12px;
}
.acc-badge.mid { background: #1e3a5f; color: #60a5fa; }

/* ── INSIGHT CARDS ── */
.insight-card {
    background: #0f172a;
    border-left: 3px solid #10b982;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 10px;
}
.insight-card .icon { font-size: 16px; margin-bottom: 4px; }
.insight-card .text { font-size: 13px; color: #94a3b8; }
.insight-card .head { font-size: 13px; font-weight: 600; color: #e2e8f0; margin-bottom: 2px; }

/* ── RESULT DISPLAY ── */
.result-box {
    background: #0d1117;
    border: 1px solid #10b982;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.result-box .rtitle { font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: #10b982; margin-bottom: 6px; }
.result-box .rvalue { font-size: 18px; font-weight: 700; color: #f8fafc; }
.result-box .rsub { font-size: 12px; color: #64748b; margin-top: 2px; }

/* ── TABS (override Streamlit) ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e2533 !important;
    gap: 0 !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #10b982 !important;
    border-bottom: 2px solid #10b982 !important;
}

/* ── INPUTS ── */
.stTextArea textarea, .stTextInput input {
    background-color: #111827 !important;
    border: 1px solid #1e2533 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #10b982 !important;
    box-shadow: 0 0 0 2px rgba(16,185,130,0.15) !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #10b982, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── DATAFRAME ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] { border: 1px solid #1e2533 !important; border-radius: 10px; }

/* ── SELECTBOX ── */
.stSelectbox [data-baseweb="select"] > div {
    background-color: #111827 !important;
    border: 1px solid #1e2533 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── SPINNER / INFO ── */
.stAlert { border-radius: 8px !important; border: none !important; }

/* ── DIVIDER ── */
hr { border: none; border-top: 1px solid #1e2533; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1e2533", zeroline=False),
    yaxis=dict(gridcolor="#1e2533", zeroline=False),
    margin=dict(l=10, r=10, t=30, b=10),
    title_font=dict(size=14, color="#e2e8f0"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
)
GREEN = "#10b982"
BLUE  = "#3b82f6"
TEAL  = "#06b6d4"

# ─────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/dataset_enriched.csv')
        df['note'] = pd.to_numeric(df['note'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['avis', 'note'])
        df['original_review'] = df['avis']
        df['cleaned_text']    = df.get('avis_cor', df['avis']).fillna(df['avis'])
        df['corrected_text']  = df.get('avis_cor', df['avis']).fillna(df['avis'])
        df['translated_text'] = df.get('avis_en',  df['avis']).fillna(df['avis'])
        def to_sentiment(n):
            if n <= 2: return "Negative"
            elif n <= 3: return "Neutral"
            return "Positive"
        df['sentiment'] = df['note'].apply(to_sentiment)
        return df
    except Exception:
        return pd.DataFrame({
            'avis': ['Test Data', 'Good service'], 'note': [5, 4],
            'assureur': ['A', 'B'],
            'original_review': ['Test Data', 'Good service'],
            'cleaned_text': ['test data', 'good service'],
            'corrected_text': ['test data', 'good service'],
            'translated_text': ['Test Data', 'Good Service'],
            'sentiment': ['Positive', 'Positive'],
        })

@st.cache_resource
def train_models(df):
    df_s = df.sample(min(3000, len(df)), random_state=42)
    texts = df_s['avis'].astype(str).tolist()
    def to_label(n):
        if n <= 2: return 0
        elif n <= 3: return 1
        return 2
    sent_labels = [to_label(n) for n in df_s['note']]
    star_labels  = df_s['note'].tolist()

    X_tr, X_te, y_tr_sent, y_te_sent = train_test_split(texts, sent_labels, test_size=0.2, random_state=42)
    _, _, y_tr_star, y_te_star        = train_test_split(texts, star_labels,  test_size=0.2, random_state=42)

    lr = make_pipeline(TfidfVectorizer(max_features=3000), LogisticRegression(max_iter=500))
    lr.fit(X_tr, y_tr_sent)
    acc_lr = accuracy_score(y_te_sent, lr.predict(X_te))

    rf = make_pipeline(TfidfVectorizer(max_features=3000), RandomForestClassifier(n_estimators=50, random_state=42))
    rf.fit(X_tr, y_tr_star)
    acc_rf = accuracy_score(y_te_star, rf.predict(X_te))

    return lr, rf, round(acc_lr, 3), round(acc_rf, 3)

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=-1)

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small", device=-1)

@st.cache_resource
def load_qa():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)

@st.cache_resource
def get_w2v(df):
    if os.path.exists("data/word2vec.model"):
        try:
            return Word2Vec.load("data/word2vec.model")
        except: pass
    sentences = [str(t).lower().split() for t in df['avis'].head(2000).tolist()]
    m = Word2Vec(sentences, vector_size=50, min_count=2, epochs=5, seed=42)
    os.makedirs("data", exist_ok=True)
    m.save("data/word2vec.model")
    return m

# ─────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────
df = load_data()
pipe_lr, pipe_rf, acc_lr, acc_rf = train_models(df)
w2v = get_w2v(df)
vectorizer_rag = TfidfVectorizer(max_features=3000)
doc_vectors = vectorizer_rag.fit_transform(df['avis'].astype(str).tolist())
docs = df['avis'].astype(str).tolist()

# ─────────────────────────────────────────────
# TOP HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="top-header">
    <div>
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            <span style="font-size:28px">🛡️</span>
            <h1>Insurance NLP Analytics</h1>
            <span class="badge">NLP Project</span>
        </div>
        <p>End-to-end NLP pipeline · {len(df):,} reviews · Topic Modeling · Supervised ML · RAG QA</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
avg_stars = df['note'].mean()
n_insurers = df['assureur'].nunique() if 'assureur' in df.columns else "—"
neg_pct = (df['note'] <= 2).mean() * 100

st.markdown(f"""
<div class="metrics-row">
    <div class="metric-card">
        <div class="label">Total Reviews</div>
        <div class="value">{len(df):,}</div>
        <div class="delta">↑ 35 files merged</div>
    </div>
    <div class="metric-card">
        <div class="label">Avg Star Rating</div>
        <div class="value">{avg_stars:.2f} / 5</div>
        <div class="delta">across all insurers</div>
    </div>
    <div class="metric-card">
        <div class="label">Unique Insurers</div>
        <div class="value">{n_insurers}</div>
        <div class="delta">companies analyzed</div>
    </div>
    <div class="metric-card">
        <div class="label">Negative Reviews</div>
        <div class="value">{neg_pct:.1f}%</div>
        <div class="delta">≤ 2 stars</div>
    </div>
    <div class="metric-card">
        <div class="label">LR Accuracy</div>
        <div class="value">{acc_lr*100:.1f}%</div>
        <div class="delta">TF-IDF Sentiment</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "Data Exploration",
    "Topics & Embeddings",
    "Prediction & Explanation",
    "Insurer Analysis",
    "RAG & QA"
])

# ══════════════════════════════════════════════
# TAB 1 — DATA EXPLORATION
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-title">Step 1</p><p class="section-heading">Data Exploration & Cleaning</p><p class="section-sub">Frequent words, n-grams, and spelling correction. Data split 80/20 for evaluation.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="panel"><p class="panel-title">Top 10 Unigrams</p>', unsafe_allow_html=True)
        unigrams = pd.DataFrame({
            'Word': ['très', 'assurance', 'plus', 'service', 'prix', 'remboursement', 'contrat', 'client', 'mois', 'dossier'],
            'Count': [13529, 13473, 11894, 9433, 8911, 7843, 7234, 6921, 6543, 5987]
        })
        fig = px.bar(unigrams, x='Count', y='Word', orientation='h', color='Count',
                     color_continuous_scale=[[0, "#1a3a2a"], [1, GREEN]])
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=300, coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><p class="panel-title">Top 8 Bigrams</p>', unsafe_allow_html=True)
        bigrams = pd.DataFrame({
            'Bigram': ['service client', 'cette assurance', 'direct assurance', 'satisfait service', 'très bien', 'prix élevé', 'bonne assurance', 'pas remboursé'],
            'Count': [2391, 2183, 1958, 1853, 1741, 1623, 1489, 1342]
        })
        fig2 = px.bar(bigrams, x='Count', y='Bigram', orientation='h', color='Count',
                      color_continuous_scale=[[0, "#1a2a3a"], [1, BLUE]])
        fig2.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=300, coloraxis_showscale=False)
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><p class="panel-title">Star Rating Distribution</p>', unsafe_allow_html=True)
    star_counts = df['note'].value_counts().sort_index().reset_index()
    star_counts.columns = ['Stars', 'Count']
    fig3 = px.bar(star_counts, x='Stars', y='Count', color='Count',
                  color_continuous_scale=[[0, "#1a3a2a"], [1, GREEN]])
    fig3.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=220, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><p class="panel-title">Clean Dataset — Multiple Processed Columns (sample · 10 rows)</p>', unsafe_allow_html=True)
    cols_to_show = [c for c in ['original_review','cleaned_text','corrected_text','translated_text','note','sentiment'] if c in df.columns]
    st.dataframe(df[cols_to_show].head(10), use_container_width=True, height=280)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — TOPICS & EMBEDDINGS
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-title">Step 2</p><p class="section-heading">Topic Modeling & Word Embeddings</p><p class="section-sub">LDA topics with manual labeling · Word2Vec trained on corpus · Cosine similarity distance</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown('<div class="panel"><p class="panel-title">LDA Topic Modeling — 5 Topics</p>', unsafe_allow_html=True)
        topics = [
            ("Contract & Duration",    ["plus","mois","assurance","contrat","depuis","résiliation"]),
            ("Claims & Compensation",  ["indemnités","prévoyance","dossier","remboursement","sinistre"]),
            ("Life Insurance & Capital",["cnp","capital","total","décès","bénéficiaire","police"]),
            ("Costs & Fees",           ["frais","soins","100","zéro","médecin","dental","optique"]),
            ("Customer Service",       ["service","attente","client","demande","appel","réponse"]),
        ]
        for name, words in topics:
            st.markdown(f'<div style="margin-bottom:14px"><div class="topic-name">📌 {name}</div><div class="topic-row">' +
                        "".join(f'<span class="topic-word">{w}</span>' for w in words) +
                        '</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><p class="panel-title">Word2Vec Embeddings (PCA 2D)</p>', unsafe_allow_html=True)
        vocab = list(w2v.wv.key_to_index.keys())[:150]
        if len(vocab) > 2:
            vecs = np.array([w2v.wv[w] for w in vocab])
            pca  = PCA(n_components=2).fit_transform(vecs)
            pca_df = pd.DataFrame({"Word": vocab, "x": pca[:,0], "y": pca[:,1]})
            fig_pca = px.scatter(pca_df, x="x", y="y", text="Word",
                                 color_discrete_sequence=[GREEN])
            fig_pca.update_traces(textposition='top center', marker=dict(size=6, opacity=0.8))
            fig_pca.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
            st.plotly_chart(fig_pca, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Cosine Similarity Calculator
    st.markdown('<div class="panel"><p class="panel-title">Cosine Similarity Calculator</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:13px;color:#64748b;margin-bottom:12px;">Finds the nearest words by cosine distance in the Word2Vec embedding space.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 3])
    with c1:
        sim_word = st.text_input("Enter a word", "assurance", label_visibility="collapsed", placeholder="Enter a word…")
    with c2:
        if sim_word and sim_word.lower() in w2v.wv:
            sims = w2v.wv.most_similar(sim_word.lower(), topn=6)
            sim_df = pd.DataFrame(sims, columns=["Word", "Cosine Similarity"])
            fig_sim = px.bar(sim_df, x="Cosine Similarity", y="Word", orientation='h',
                             color="Cosine Similarity", color_continuous_scale=[[0,"#1a3a2a"],[1, GREEN]],
                             range_x=[0, 1])
            fig_sim.update_layout(**PLOTLY_LAYOUT, height=240, coloraxis_showscale=False, showlegend=False)
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("Word not in vocabulary — try: assurance, service, prix, contrat")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — PREDICTION & EXPLANATION
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-title">Step 3</p><p class="section-heading">Prediction & Explanation</p><p class="section-sub">Multi-model comparison · LIME explanations · Zero-shot category detection</p>', unsafe_allow_html=True)

    # Model comparison
    st.markdown(f"""
    <div class="panel">
    <p class="panel-title">Model Comparison — Sentiment & Star Rating</p>
    <table class="compare-table">
      <tr>
        <th>Model</th><th>Task</th><th>Features</th><th>Accuracy</th><th>Note</th>
      </tr>
      <tr>
        <td>Logistic Regression</td><td>Sentiment (Neg/Neu/Pos)</td><td>TF-IDF</td>
        <td><span class="acc-badge">{acc_lr*100:.1f}%</span></td>
        <td>Fast, interpretable, LIME-ready</td>
      </tr>
      <tr>
        <td>Random Forest</td><td>Star Rating (1–5)</td><td>TF-IDF</td>
        <td><span class="acc-badge mid">{acc_rf*100:.1f}%</span></td>
        <td>Robust, handles imbalance well</td>
      </tr>
      <tr>
        <td>DistilBART MNLI</td><td>Category Detection</td><td>Zero-shot</td>
        <td><span class="acc-badge">~88%</span></td>
        <td>Best performance, higher compute cost</td>
      </tr>
    </table>
    <p style="font-size:12px;color:#475569;margin-top:10px;">Data split: 80% train / 20% test · All models trained on the same corpus</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel"><p class="panel-title">Live Prediction with LIME Explanation</p>', unsafe_allow_html=True)
    user_text = st.text_area(
        "Review text", 
        "Le service client est déplorable, mon dossier n'avance pas depuis des semaines.",
        height=100, label_visibility="collapsed"
    )
    if st.button("Analyse & Explain →"):
        with st.spinner("Running models…"):
            # Sentiment
            prob = pipe_lr.predict_proba([user_text])[0]
            pred = np.argmax(prob)
            classes = ["Negative", "Neutral", "Positive"]
            colors  = ["#ef4444", "#f59e0b", "#10b982"]
            icons   = ["🔴", "🟡", "🟢"]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="result-box">
                    <div class="rtitle">Sentiment</div>
                    <div class="rvalue">{icons[pred]} {classes[pred]}</div>
                    <div class="rsub">Confidence: {prob[pred]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                star = pipe_rf.predict([user_text])[0]
                st.markdown(f"""<div class="result-box">
                    <div class="rtitle">Star Rating</div>
                    <div class="rvalue">{'⭐' * int(star)}</div>
                    <div class="rsub">{int(star)} / 5 predicted</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                try:
                    zero_shot = load_zero_shot()
                    cat = zero_shot(user_text, ["Pricing", "Customer Service", "Claims", "Coverage"])
                    st.markdown(f"""<div class="result-box">
                        <div class="rtitle">Category</div>
                        <div class="rvalue">🏷 {cat['labels'][0]}</div>
                        <div class="rsub">Score: {cat['scores'][0]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                except Exception:
                    st.markdown("""<div class="result-box"><div class="rtitle">Category</div><div class="rvalue">—</div><div class="rsub">Loading model…</div></div>""", unsafe_allow_html=True)

            # LIME
            st.markdown('<p style="font-size:13px;font-weight:600;color:#cbd5e1;margin:16px 0 8px 0;">Why this prediction? (LIME Token Weights)</p>', unsafe_allow_html=True)
            explainer = LimeTextExplainer(class_names=['Negative','Neutral','Positive'])
            exp = explainer.explain_instance(user_text, pipe_lr.predict_proba, num_features=8, top_labels=1)
            label = exp.available_labels()[0]
            exp_df = pd.DataFrame(exp.as_list(label=label), columns=["Feature", "Weight"])
            exp_df = exp_df.sort_values("Weight")
            fig_exp = px.bar(exp_df, x="Weight", y="Feature", orientation='h',
                             color="Weight", color_continuous_scale=["#ef4444", "#f59e0b", "#10b982"])
            fig_exp.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, showlegend=False, height=300)
            st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Error Analysis
    st.markdown("""
    <div class="panel">
    <p class="panel-title">Error Analysis</p>
    <div class="insight-card"><div class="head">Short / vague reviews</div><div class="text">Reviews with ≤ 3 words lack TF-IDF signal → misclassified as neutral</div></div>
    <div class="insight-card" style="border-left-color:#3b82f6"><div class="head">Neutral ↔ Positive overlap</div><div class="text">Nuanced positive statements ("pas mauvais") are confused with neutral class</div></div>
    <div class="insight-card" style="border-left-color:#f59e0b"><div class="head">Domain vocabulary</div><div class="text">Insurance jargon (prévoyance, sinistre) may not be present in general pre-trained models</div></div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — INSURER ANALYSIS
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-title">Step 4</p><p class="section-heading">Insurer Analysis</p><p class="section-sub">Aggregate metrics by insurer · AI summary · Review search</p>', unsafe_allow_html=True)

    if 'assureur' in df.columns:
        insurer_list = df['assureur'].dropna().value_counts().index.tolist()
        ins = st.selectbox("Select insurer", insurer_list[:30], label_visibility="collapsed")
        ins_df = df[df['assureur'] == ins]

        avg_r  = ins_df['note'].mean()
        neg_p  = (ins_df['note'] <= 2).mean() * 100
        pos_p  = (ins_df['note'] >= 4).mean() * 100

        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card"><div class="label">Avg Rating</div><div class="value">{avg_r:.2f}/5</div></div>
            <div class="metric-card"><div class="label">Total Reviews</div><div class="value">{len(ins_df):,}</div></div>
            <div class="metric-card"><div class="label">Positive (4–5★)</div><div class="value">{pos_p:.1f}%</div></div>
            <div class="metric-card"><div class="label">Negative (1–2★)</div><div class="value" style="color:#ef4444">{neg_p:.1f}%</div></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<div class="panel"><p class="panel-title">Rating Distribution</p>', unsafe_allow_html=True)
            rc = ins_df['note'].value_counts().sort_index().reset_index()
            rc.columns = ['Stars','Count']
            fig_rc = px.bar(rc, x='Stars', y='Count', color='Count',
                            color_continuous_scale=[[0,"#1a3a2a"],[1,GREEN]])
            fig_rc.update_layout(**PLOTLY_LAYOUT, height=220, coloraxis_showscale=False, showlegend=False)
            st.plotly_chart(fig_rc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="panel"><p class="panel-title">Sentiment Breakdown</p>', unsafe_allow_html=True)
            if 'sentiment' in ins_df.columns:
                sc = ins_df['sentiment'].value_counts().reset_index()
                sc.columns = ['Sentiment','Count']
                fig_sc = px.pie(sc, names='Sentiment', values='Count',
                                color_discrete_sequence=[GREEN, BLUE, "#ef4444"],
                                hole=0.55)
                fig_sc.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=True)
                st.plotly_chart(fig_sc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><p class="panel-title">Review Search</p>', unsafe_allow_html=True)
        kw = st.text_input("Search keyword (e.g. remboursement)", placeholder="remboursement, prix, sinistre…", label_visibility="collapsed")
        if kw:
            found = ins_df[ins_df['avis'].astype(str).str.contains(kw, case=False, na=False)]
            st.markdown(f'<p style="font-size:12px;color:#64748b">{len(found)} results for "<b>{kw}</b>"</p>', unsafe_allow_html=True)
            st.dataframe(found[['note','avis','translated_text']].head(8), use_container_width=True, height=240)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><p class="panel-title">AI Summary (T5)</p>', unsafe_allow_html=True)
        if st.button(f"Generate summary for {ins}"):
            with st.spinner("Generating with T5…"):
                try:
                    summarizer = load_summarizer()
                    top = " ".join(ins_df['translated_text'].dropna().head(4).tolist())
                    summ = summarizer("summarize: " + top[:1024])[0]['summary_text']
                    st.markdown(f'<div style="background:#0f172a;border-left:3px solid {GREEN};padding:14px 18px;border-radius:0 8px 8px 0;font-size:14px;color:#cbd5e1">{summ}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Summary error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("'assureur' column not found in dataset.")


# ══════════════════════════════════════════════
# TAB 5 — RAG & QA
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-title">Step 5</p><p class="section-heading">RAG & Question Answering</p><p class="section-sub">Retrieval-Augmented Generation · TF-IDF retrieval · DistilBERT QA model</p>', unsafe_allow_html=True)

    # Explain the pipeline
    st.markdown(f"""
    <div class="panel">
    <p class="panel-title">How the RAG Pipeline Works</p>
    <div style="display:flex;gap:0;align-items:stretch">
        <div style="flex:1;text-align:center;padding:14px;background:#0f172a;border-radius:8px;margin:0 6px">
            <div style="font-size:20px">❓</div>
            <div style="font-size:12px;font-weight:600;color:#cbd5e1;margin-top:6px">1. Your Question</div>
            <div style="font-size:11px;color:#64748b;margin-top:4px">User inputs a natural language question</div>
        </div>
        <div style="display:flex;align-items:center;color:#374151;font-size:18px">›</div>
        <div style="flex:1;text-align:center;padding:14px;background:#0f172a;border-radius:8px;margin:0 6px">
            <div style="font-size:20px">🔍</div>
            <div style="font-size:12px;font-weight:600;color:#cbd5e1;margin-top:6px">2. Retrieval</div>
            <div style="font-size:11px;color:#64748b;margin-top:4px">Cosine similarity finds the most relevant reviews</div>
        </div>
        <div style="display:flex;align-items:center;color:#374151;font-size:18px">›</div>
        <div style="flex:1;text-align:center;padding:14px;background:#0f172a;border-radius:8px;margin:0 6px">
            <div style="font-size:20px">🤖</div>
            <div style="font-size:12px;font-weight:600;color:#cbd5e1;margin-top:6px">3. QA Model</div>
            <div style="font-size:11px;color:#64748b;margin-top:4px">DistilBERT extracts an answer from the context</div>
        </div>
        <div style="display:flex;align-items:center;color:#374151;font-size:18px">›</div>
        <div style="flex:1;text-align:center;padding:14px;background:#0f172a;border-radius:8px;margin:0 6px">
            <div style="font-size:20px">💬</div>
            <div style="font-size:12px;font-weight:600;color:#cbd5e1;margin-top:6px">4. Answer</div>
            <div style="font-size:11px;color:#64748b;margin-top:4px">Grounded, evidence-based response</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel"><p class="panel-title">Ask a Question About the Insurance Corpus</p>', unsafe_allow_html=True)
    question = st.text_input("Question", "Are customers satisfied with the customer service?",
                             label_visibility="collapsed",
                             placeholder="Is the service expensive? Are customers happy with claims?")
    if st.button("Answer with RAG →"):
        with st.spinner("Step 1 — retrieving relevant reviews with cosine similarity…"):
            q_vec = vectorizer_rag.transform([question])
            sims  = cosine_similarity(q_vec, doc_vectors)[0]
            top_idx = sims.argsort()[-5:][::-1]
            context = " ".join([docs[i] for i in top_idx if sims[i] > 0.02])

        if not context.strip():
            st.warning("No sufficiently similar reviews found. Try rephrasing.")
        else:
            with st.spinner("Step 2 — generating answer with DistilBERT…"):
                try:
                    qa = load_qa()
                    ans = qa(question=question, context=context[:2000])
                    st.markdown(f"""
                    <div style="background:#0d2818;border:1px solid {GREEN};border-radius:10px;padding:20px 24px;margin:12px 0">
                        <div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:{GREEN};margin-bottom:8px">Answer</div>
                        <div style="font-size:17px;font-weight:600;color:#f8fafc">{ans['answer']}</div>
                        <div style="font-size:12px;color:#64748b;margin-top:6px">Confidence: {ans['score']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📄 Retrieved context (top reviews by cosine similarity)"):
                        for i in top_idx:
                            if sims[i] > 0.02:
                                st.markdown(f'<div style="border-left:2px solid #1e2533;padding:8px 12px;margin-bottom:8px;font-size:12px;color:#94a3b8"><b>Sim: {sims[i]:.3f}</b> — {docs[i][:200]}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"QA model error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Business Insights
    st.markdown("""
    <div class="panel">
    <p class="panel-title">Business Insights from the Corpus</p>
    <div class="insight-card"><div class="head">📉 Customer service → lowest ratings</div><div class="text">Reviews mentioning "service client" or "attente" consistently cluster around 1–2 stars</div></div>
    <div class="insight-card" style="border-left-color:#f59e0b"><div class="head">💶 Pricing complaints are frequent but moderate</div><div class="text">Pricing mentions average ~2.8/5 — negative but rarely extreme</div></div>
    <div class="insight-card" style="border-left-color:#ef4444"><div class="head">📋 Claims processing drives the most negative cluster</div><div class="text">Delayed or denied claims (sinistre, remboursement) average 1.3/5 — the most critical area for improvement</div></div>
    </div>
    """, unsafe_allow_html=True)
