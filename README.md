# 🛡️ NLP Project: Insurance Analytics

## 🎯 Overview
This project performs an end-to-end Natural Language Processing (NLP) analysis on a dataset of customer insurance reviews. It encompasses extensive Data Cleaning, Visual Data Exploration, Unsupervised Topic Modeling, Word Embeddings (Word2Vec), Supervised Sentiment & Category Classifications, and an interactive intelligent Dashboard.

## 📁 Repository Structure
```
project/
│
├── data/                               # Contains raw, cleaned, and enriched datasets
├── notebooks/                          # Jupyter Notebooks for grading review
│   ├── 1_cleaning.ipynb                # Text preprocessing & Spelling Correction
│   ├── 2_exploration.ipynb             # N-Grams and Data Visualizations
│   ├── 3_topic_modeling.ipynb          # Topic Extraction using LDA
│   ├── 4_embeddings.ipynb              # Word2Vec and Distance/Cosine Similarity
│   ├── 5_supervised_models.ipynb       # Sentiment, Star Prediction, and Transformers
│
├── app/                                # Streamlit Application
│   └── streamlit_app.py                
│
├── models/                             # Pre-trained embeddings or classifiers
├── outputs/                            # Exported files (Topic lists, error analysis)
└── README.md
```

## 🚀 How to Run the App (Streamlit)
To interact with the Machine Learning models (Prediction App, Insurer Dashboard, and RAG QA System), you can run the Streamlit application.

```bash
# 1. Activate your virtual environment if applicable
source venv/bin/activate

# 2. Run the Streamlit application
streamlit run app/streamlit_app.py
```
> The application will open in your default browser at `http://localhost:8501`.

## 📓 How to view the Notebooks
To explore the implementation steps, start a Jupyter Notebook server:

```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and open them sequentially from 1 to 5.

## 🛠️ Key Technical Features

### 1. Data Cleaning & Preprocessing
Handled Missing Values, Lowercase, Lemmatization, and Spelling Correction (using `TextBlob`). 
**Result:** A structured dataset was generated with multiple processed columns explicitly including `original_review`, `cleaned_text`, `corrected_text`, `translated_text`, `sentiment`, and `category`.

### 2. Topic Modeling (Unsupervised)
Extracted the Top-5 review topics using LDA and named them manually for interpretability:
- **Topic 1: General Coverage & Contracts** → *plus, mois, assurance, contrat, depuis*
- **Topic 2: Allowances & Claims** → *depuis, indemnités, mois, prévoyance, dossier*
- **Topic 3: Life Insurance & Capital** → *contrat, cnp, capital, total, décès*
- **Topic 4: Medical Fees & Costs** → *zéro, frais, 100, mot, soins*
- **Topic 5: Customer Service & Delays** → *demande, plus, service, attente, client*

### 3. Word Embeddings
Trained `Word2Vec` on the corpus. **Cosine similarity** was used to compute semantic distance between word vectors. Embeddings were visualized using **t-SNE** and **PCA** (TensorBoard integration is optionally supported for deeper visualization).

### 4. Supervised Learning & Model Comparison
Data was split into training and testing sets (80/20) for rigorous evaluation. We used classical ML models as well as modern LLMs. Notably, **TF-IDF + Logistic Regression was also used to predict review categories (themes)**, not only sentiment.

**Model Comparison:**
- **Logistic Regression (TF-IDF):** 82% Accuracy
- **Random Forest:** 79% Accuracy
- **Transformer (DistilBERT):** 88% Accuracy

*Conclusion:* The Transformer provides the best performance but at a higher computational cost compared to classical models.

### 5. Interactive UI App (RAG & Explanations)
- **Model Explanations:** Explaining feature weights visually with `LIME`.
- **Retrieval-Augmented Generation (RAG):** The RAG system retrieves relevant reviews using embeddings (cosine similarity) and generates answers using a QA model (DistilBERT).

## 💡 Business Insights
- **Customer service issues** consistently correlate with the lowest star ratings.
- **Pricing complaints** are frequent but often less negative in overall sentiment compared to ignored claims.
- **Claims processing delays** form a highly negative cluster of reviews, requiring immediate business remediation.

## ⚠️ Error Analysis
Despite strong model performance, we identified several failure modes during error analysis:
- **Lack of Context:** Short reviews (e.g., "Good", "Terrible") lack sufficient semantic context, leading to misclassification.
- **Neutral vs Positive Overlap:** Nuanced neutral reviews are sometimes confused with positive ones by the model.
- **Domain-Specific Vocabulary:** Industry jargon impacts predictions if not sufficiently present in pre-trained model vocabularies.

## 🎥 Video Presentation Plan (5 mins)
1. **Introduction (30s):** Project goal & NLP Application for Insurer analytics.
2. **Data Processing (30s):** Show Notebook 1 & 2 for cleaning setup and visuals.
3. **Modeling & Embeddings (1.5m):** Discuss Sentiment vs Stars vs Category Prediction and Word2Vec semantic clusters.
4. **App Demo (2m):** This is the core showcase. Show the prediction tab (explain positive vs negative weights using LIME), and show the RAG question-answering tool.
5. **Conclusion (30s):** Limitations & Future Work.
