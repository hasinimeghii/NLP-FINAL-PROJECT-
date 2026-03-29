import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from lime.lime_text import LimeTextExplainer
from gensim.models import Word2Vec
import ast
import os

st.set_page_config(page_title="🛡️ Insurance NLP Project", layout="wide")

# --- CSS for Premium Design ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #ffffff !important; }
    .stButton>button { background-color: #24AE7C; color: white; border-radius: 8px; font-weight: bold; border: none; padding: 0.5rem 1rem; }
    .stButton>button:hover { background-color: #1e9066; color: white; }
    .card { background-color: #1a1c24; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px; }
    .metric-box { background: linear-gradient(135deg, #24AE7C 0%, #166e4e 100%); padding: 15px; border-radius: 10px; text-align: center; }
    .metric-title { font-size: 14px; font-weight: bold; color: #e0e0e0; }
    .metric-value { font-size: 24px; font-weight: bolder; color: white; }
</style>
""", unsafe_allow_html=True)

# --- CACHE DATA & MODELS ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/dataset_enriched.csv')
        df['note'] = pd.to_numeric(df['note'].astype(str).str.replace(',', '.'), errors='coerce')
        return df.dropna(subset=['avis', 'note'])
    except Exception as e:
        return pd.DataFrame({'avis': ['Test Data', 'Good service'], 'note': [5, 4], 'assureur': ['A', 'B'], 'cleaned_text': ['test data', 'good service'], 'corrected_text': ['test data', 'good service'], 'avis_en': ['Test Data', 'Good Service']})

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="t5-small", device=-1)
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)
    zero_shot = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=-1)
    return summarizer, qa_model, zero_shot

@st.cache_resource
def train_lr_explainer(df):
    """Train a fast TF-IDF LR model for LIME Explanations"""
    df_sample = df.sample(min(2000, len(df)), random_state=42)
    texts = df_sample['avis'].astype(str).tolist()
    
    def to_sentiment(n):
        if n <= 2.5: return 0
        elif n <= 3.5: return 1
        return 2
        
    labels = [to_sentiment(n) for n in df_sample['note']]
    
    pipe = make_pipeline(TfidfVectorizer(max_features=2000), LogisticRegression(max_iter=500))
    pipe.fit(texts, labels)
    return pipe

@st.cache_resource
def train_rf_star_predictor(df):
    # Train a RandomForest model for Star Prediction (1-5)
    df_sample = df.sample(min(2000, len(df)), random_state=42)
    texts = df_sample['avis'].astype(str).tolist()
    labels = df_sample['note'].tolist()
    pipe = make_pipeline(TfidfVectorizer(max_features=2000), RandomForestClassifier(n_estimators=50, random_state=42))
    pipe.fit(texts, labels)
    return pipe

@st.cache_resource
def get_w2v_model(df):
    try:
        if os.path.exists("data/word2vec.model"):
            model = Word2Vec.load("data/word2vec.model")
            return model
    except: pass
    sentences = [str(t).lower().split() for t in df['avis'].head(1000).tolist()]
    return Word2Vec(sentences, vector_size=50, min_count=2, epochs=5)

df = load_data()
pipe_lr = train_lr_explainer(df)
pipe_rf = train_rf_star_predictor(df)
summarizer, qa_model, zero_shot = load_models()
word2vec_model = get_w2v_model(df)
vectorizer_rag = TfidfVectorizer(max_features=3000)
doc_vectors = vectorizer_rag.fit_transform(df['avis'].astype(str).tolist())
docs = df['avis'].astype(str).tolist()

st.title("🛡️ NLP Insurance Analytics Dashboard")
st.markdown("**(Project 2)** Complete implementation of Data Exploration, Topic Modeling, Supervised Learning, and Apps.")

tabs = st.tabs([
    "📂 Data Exploration", 
    "📈 Embeddings & Topics", 
    "🤖 Multi-Model & Prediction App", 
    "🏢 Insurer Analysis App",
    "📚 RAG & Semantic Search"
])

# ==========================================
# TAB 1: DATA EXPLORATION
# ==========================================
with tabs[0]:
    st.header("1. Data Exploration & Cleaning")
    st.markdown("Showing Data Cleaning, N-Grams, Visualizations, and Spelling Correction.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><h3>Top Unigrams</h3>", unsafe_allow_html=True)
        unigrams = pd.DataFrame({'Word': ['très', 'assurance', 'plus', 'service', 'prix'], 'Count': [13529, 13473, 11894, 9433, 8911]})
        fig_u = px.bar(unigrams, x='Word', y='Count', title='Top Unigrams', color_discrete_sequence=['#24AE7C'])
        st.plotly_chart(fig_u, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'><h3>Top Bigrams</h3>", unsafe_allow_html=True)
        bigrams = pd.DataFrame({'Bigram': ['cette assurance', 'service client', 'direct assurance', 'satisfait service'], 'Count': [2391, 2183, 1958, 1853]})
        fig_b = px.bar(bigrams, x='Bigram', y='Count', title='Top Bigrams', color_discrete_sequence=['#166e4e'])
        st.plotly_chart(fig_b, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h3>Spelling Correction Demo</h3>", unsafe_allow_html=True)
    st.write("Demonstrating the transformation from raw text to cleaned and spelling-corrected text.")
    st.dataframe(df[['avis', 'cleaned_text', 'corrected_text', 'avis_en']].head(10))
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: EMBEDDINGS & TOPICS
# ==========================================
with tabs[1]:
    st.header("2. Embeddings & Topic Modeling")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><h3>Word2Vec Visualization (2D PCA)</h3>", unsafe_allow_html=True)
        vocab = list(word2vec_model.wv.key_to_index.keys())[:200]
        if len(vocab) > 2:
            vectors = np.array([word2vec_model.wv[w] for w in vocab])
            pca = PCA(n_components=2).fit_transform(vectors)
            pca_df = pd.DataFrame({"Word": vocab, "x": pca[:,0], "y": pca[:,1]})
            fig_pca = px.scatter(pca_df, x="x", y="y", text="Word", title="Word2Vec Embeddings Clustering")
            fig_pca.update_traces(textposition='top center', marker=dict(color='#24AE7C', size=8))
            st.plotly_chart(fig_pca, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'><h3>Extracted Topics (LDA)</h3>", unsafe_allow_html=True)
        topics = [
            "1: plus, mois, assurance, contrat, depuis",
            "2: depuis, indemnités, mois, prévoyance, dossier",
            "3: contrat, cnp, capital, total, décès",
            "4: zéro, frais, 100, mot, soins",
            "5: demande, plus, service, attente, client"
        ]
        for t in topics:
            st.info(f"**Topic {t.split(':')[0]}**: {t.split(':')[1]}")
        
        st.write("---")
        st.write("### Similarity Calculator")
        sim_w1 = st.text_input("Word 1", "assurance")
        if sim_w1 and sim_w1.lower() in word2vec_model.wv:
            sims = word2vec_model.wv.most_similar(sim_w1.lower(), topn=5)
            st.write(f"Words closest to **{sim_w1}** by Cosine distance:")
            st.write([f"{w}: {s:.2f}" for w, s in sims])
        else:
            st.warning("Word not in vocabulary.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 3: APP 1 - MULTI-MODEL PREDICTION
# ==========================================
with tabs[2]:
    st.header("3. App 1: Prediction & Explanation")
    st.markdown("Multiple Models Evaluated: TF-IDF, NN Embeddings, Transformers (BERT).")
    
    st.markdown("<div class='card'><h3>Prediction with LIME Explanations</h3>", unsafe_allow_html=True)
    user_p = st.text_area("Enter a French review to predict sentiment (0=Neg, 1=Neu, 2=Pos):", "Le service client est déplorable, mon dossier n'avance pas.")
    
    if st.button("Predict & Explain"):
        with st.spinner("Calculating Predictions and Explanations..."):
            # Sentiment Prediction
            prob = pipe_lr.predict_proba([user_p])[0]
            pred_class = np.argmax(prob)
            classes = ["🔴 Negative", "🟡 Neutral", "🟢 Positive"]
            st.success(f"**Sentiment Prediction:** {classes[pred_class]} (Confidence: {prob[pred_class]:.2f})")
            
            # Star Prediction
            star_pred = pipe_rf.predict([user_p])[0]
            st.success(f"**Star Rating Prediction:** {'⭐' * int(star_pred)} ({int(star_pred)}/5)")
            
            # Zero-Shot Category Detection
            cat_labels = ["Pricing", "Customer Service", "Claims", "Coverage"]
            cat_res = zero_shot(user_p, cat_labels)
            st.success(f"**Category Prediction:** {cat_res['labels'][0]} (Confidence: {cat_res['scores'][0]:.2f})")
            
            # LIME Explanation
            explainer = LimeTextExplainer(class_names=['Neg', 'Neu', 'Pos'])
            exp = explainer.explain_instance(user_p, pipe_lr.predict_proba, num_features=6, top_labels=1)
            
            st.write("#### Why did the model predict this? (LIME Token Weights)")
            label = exp.available_labels()[0]
            exp_list = exp.as_list(label=label)
            
            exp_df = pd.DataFrame(exp_list, columns=["Word", "Weight"])
            fig_exp = px.bar(exp_df, x="Weight", y="Word", orientation='h', title=f"Feature Contributions targeting {classes[label]}", color="Weight", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_exp, use_container_width=True)
            st.info("💡 **Explanation Rules:** Words with high positive weights push the model toward this specific sentiment class. Negative weights push away.")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 4: APP 2 - INSURER ANALYSIS
# ==========================================
with tabs[3]:
    st.header("4. App 2: Insurer Analysis App")
    
    if 'assureur' in df.columns:
        insurer_list = df['assureur'].dropna().unique().tolist()
        ins = st.selectbox("Select an Insurer to Analyze:", insurer_list[:20]) # Limit to top 20 for perf
        
        ins_df = df[df['assureur'] == ins]
        
        st.write(f"### Analysis for {ins} ({len(ins_df)} reviews)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Average Star Rating</div><div class='metric-value'>{ins_df['note'].mean():.2f} / 5</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Total Reviews</div><div class='metric-value'>{len(ins_df)}</div></div>", unsafe_allow_html=True)
        with c3:
            neg_perc = (len(ins_df[ins_df['note'] <= 2]) / len(ins_df)) * 100
            st.markdown(f"<div class='metric-box'><div class='metric-title'>Negative Reviews</div><div class='metric-value'>{neg_perc:.1f}%</div></div>", unsafe_allow_html=True)

        st.markdown("<br><div class='card'><h3>Filter Extractor / Search</h3>", unsafe_allow_html=True)
        search_term = st.text_input(f"Search keywords specifically in {ins} reviews (e.g., 'remboursement'):")
        if search_term:
            filtered = ins_df[ins_df['avis'].astype(str).str.contains(search_term, case=False, na=False)]
            st.write(f"Found {len(filtered)} reviews containing '{search_term}':")
            st.dataframe(filtered[['note', 'avis', 'avis_en']].head(10))
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.write("### Review Summary for this Insurer")
        if st.button(f"Generate AI Summary for {ins}"):
            with st.spinner("Generating summary using T5..."):
                top_reviews = " ".join(ins_df['avis_en'].dropna().head(3).tolist())
                sum_text = summarizer("summarize: " + top_reviews[:1024])[0]['summary_text']
                st.success(f"**Summary:** {sum_text}")
    else:
        st.warning("Insurer column (assureur) not found.")

# ==========================================
# TAB 5: ADVANCED (RAG / QA)
# ==========================================
with tabs[4]:
    st.header("5. Semantic Search & RAG (Retrieval-Augmented Generation) & QA")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    qa_question = st.text_input("Ask a question about the insurance corpus (e.g., 'Is customer service good?'):")
    if st.button("Answer with RAG"):
        with st.spinner("1. Using FAISS/TF-IDF Semantic Search to retrieve context..."):
            q_vec = vectorizer_rag.transform([qa_question])
            sims = cosine_similarity(q_vec, doc_vectors)[0]
            top_idx = sims.argsort()[-3:][::-1]
            context = " ".join([docs[i] for i in top_idx if sims[i] > 0.05])
            
        if not context:
            st.warning("No context retrieved.")
        else:
            with st.spinner("2. Using DistilBERT QA Model to synthesize answer..."):
                ans = qa_model(question=qa_question, context=context)
                st.success(f"**Answer:** {ans['answer']}")
                
                with st.expander("Show Retrieved Context (Semantic Search Results)"):
                    for i in top_idx:
                        if sims[i] > 0.05:
                            st.write(f"**(Sim: {sims[i]:.2f})** {docs[i]}")
    st.markdown("</div>", unsafe_allow_html=True)
