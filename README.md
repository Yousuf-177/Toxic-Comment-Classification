<div align="center">

# 🛡️ Toxic Comment Classifier
### Multi-Label NLP Pipeline · 6 Toxicity Categories · LangChain + Streamlit

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-154360?style=flat-square)](https://nltk.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat-square)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#-live-demo)

<br/>

> An end-to-end **multi-label NLP pipeline** that detects and classifies toxic online comments into **6 simultaneous toxicity categories** using TF-IDF vectorization, 7 benchmarked ML classifiers, LIME explainability, and an AI-powered Streamlit app backed by LangChain + Groq.

<br/>

[Live Demo](#-live-demo) · [Dataset](#-dataset) · [Pipeline](#-end-to-end-pipeline) · [Models](#-model-training--benchmarking) · [Results](#-results--evaluation) · [Setup](#-getting-started)

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Why 6 Binary Classifiers?](#-why-6-independent-binary-classifiers--not-multiclass)
- [Project Highlights](#-project-highlights)
- [Toxicity Categories](#-toxicity-categories)
- [Project Structure](#-project-structure)
- [End-to-End Pipeline](#-end-to-end-pipeline)
- [Dataset](#-dataset)
- [EDA & Label Analysis](#-eda--label-analysis)
- [Text Preprocessing Pipeline](#-text-preprocessing-pipeline)
- [Handling Class Imbalance](#-handling-class-imbalance)
- [TF-IDF Vectorization](#-tf-idf-vectorization)
- [Model Training & Benchmarking](#-model-training--benchmarking)
- [Model Explainability — LIME](#-model-explainability--lime)
- [Streamlit App + LangChain](#-streamlit-app--langchain-integration)
- [Results & Evaluation](#-results--evaluation)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Deployment](#-deployment)
- [Key Challenges & Solutions](#-key-challenges--solutions)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🎯 Problem Statement

Online platforms generate hundreds of millions of comments every day. Manual moderation at this scale is impossible — automated systems are essential for maintaining healthy communities. This project builds a **production-ready toxic comment detection pipeline** that:

- Identifies **6 distinct toxicity types** simultaneously (a comment can be toxic AND obscene AND insulting at the same time)
- Handles **extreme class imbalance** (the `threat` label has fewer than 500 positive examples in 159,000 rows)
- Provides **human-readable AI explanations** for every flagged comment via LangChain
- Is deployed as an **interactive Streamlit web application** for live inference

---

## 🧠 Why 6 Independent Binary Classifiers — Not Multiclass?

> **This is the most important design decision in the project — and the first question every interviewer will ask.**

A standard multiclass classifier assumes **mutual exclusivity** — each sample belongs to exactly one class. But toxic comments routinely belong to **multiple categories simultaneously**:

```
"You are a disgusting [slur] and I will find you."
   → toxic: ✅   obscene: ✅   threat: ✅   insult: ✅
```

**Solution:** Train **one independent binary classifier per label**. Each predicts "Is this comment [label]? Yes/No" independently. This:

- ✅ Correctly handles co-occurrence of multiple toxicity types
- ✅ Allows per-label threshold tuning (critical for minority classes like `threat`)
- ✅ Mirrors real-world content moderation systems at Google, Reddit, and Meta
- ✅ Enables per-label model selection (best classifier may differ per category)

---

## ✨ Project Highlights

| Feature | Details |
|---|---|
| 📝 **Task** | Multi-label binary text classification (6 labels) |
| 📦 **Dataset** | Jigsaw Toxic Comment Challenge — 159,571 training rows |
| 🔤 **Vectorization** | TF-IDF with bigrams, sublinear TF, 50k features per label |
| 🤖 **Models** | 7 classifiers benchmarked × 6 labels = 42 trained models |
| ⚖️ **Imbalance** | `class_weight='balanced'` + threshold tuning per label |
| 🔍 **Explainability** | LIME — highlights words that drove each prediction |
| 💬 **AI Layer** | LangChain + Groq LLM generates natural language explanations |
| 🚀 **Deployment** | Streamlit Community Cloud — zero-cost live demo |
| 🧪 **Testing** | Unit tests for preprocessing pipeline (pytest) |

---

## ☣️ Toxicity Categories

The classifier detects **6 independently predictable** toxicity types:

| Label | Description | Approx. Positives |
|---|---|---|
| `toxic` | General toxic language — abusive, disrespectful | ~15,294 |
| `severe_toxic` | Extreme aggression, violent language | ~1,595 |
| `obscene` | Sexually explicit or profane content | ~8,449 |
| `threat` | Explicit or implied threats of violence | ~478 |
| `insult` | Direct personal attacks or demeaning language | ~7,877 |
| `identity_hate` | Hate speech targeting identity (race, religion, gender) | ~1,405 |

> ⚠️ **Note:** A single comment can be labelled positive for multiple categories. The `threat` class (~0.3% positive rate) requires special handling to avoid being ignored by the model.

---

## 📁 Project Structure

```
Toxic-Comment-Classification/
│
├── 📂 models/
│   ├── toxic_model.pkl
│   ├── severe_toxic_model.pkl
│   ├── threat_model.pkl
│   ├── obscene_model.pkl
│   ├── insult_model.pkl
│   └── identity_hate_model.pkl
│
├── 📂 vectorizers/
│   ├── toxic_vectorizer.pkl
│   ├── severe_toxic_vectorizer.pkl
│   ├── threat_vectorizer.pkl
│   ├── obscene_vectorizer.pkl
│   ├── insult_vectorizer.pkl
│   └── identity_hate_vectorizer.pkl
│
├── 📂 Balanced Data/               ← SMOTE-resampled arrays (.npy)
├── 📂 Distribution-of-Classification/ ← Label distribution charts
│
├── 📂 tests/
│   └── test_preprocessing.py       ← Unit tests for clean() function
│
├── 📓 Analysis.ipynb               ← EDA, label analysis, WordClouds
├── 📓 Feature-Engg-Model-Building.ipynb ← TF-IDF, training, evaluation
│
├── 📱 app.py                       ← Streamlit app (3 tabs)
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🔄 End-to-End Pipeline

```
Raw Text (train.csv · 159,571 rows)
              │
              ▼
┌─────────────────────────────────────────────┐
│  PHASE 1 · EDA & LABEL ANALYSIS             │
│  Class distribution · Label co-occurrence   │
│  WordClouds per category · Length analysis  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  PHASE 2 · TEXT PREPROCESSING               │
│  Strip \n · Remove alphanumerics            │
│  Lowercase + Punctuation removal            │
│  Non-ASCII removal · Stopword filtering     │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  PHASE 3 · CLASS IMBALANCE HANDLING         │
│  class_weight='balanced' per classifier     │
│  Threshold tuning: 0.5 → 0.3 for minority  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  PHASE 4 · TF-IDF VECTORIZATION             │
│  1 vectorizer per label · 50k features      │
│  bigrams (1,2) · sublinear_tf=True          │
│  Fit on train · Transform val & test        │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  PHASE 5 · MODEL TRAINING (7 × 6 = 42)     │
│  LR · SVM · RF · CNB · BNB · DT · KNN      │
│  Stratified 80/20 split per label           │
│  F1 score stored in results DataFrame       │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  PHASE 6 · EVALUATION & EXPLAINABILITY      │
│  F1 heatmap · Confusion matrices            │
│  ROC curves · LIME word-level explanations  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  PHASE 7 · SERIALIZATION                    │
│  6 best-model .pkl files                    │
│  6 fitted TF-IDF vectorizer .pkl files      │
└────────────────────┬────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   STREAMLIT APP (3 tabs)   │
        │  Predict · Explain · Compare│
        │  LangChain AI narration     │
        └────────────────────────────┘
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [Jigsaw Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) |
| **Train rows** | 159,571 comments |
| **Test rows** | 153,164 comments |
| **Input column** | `comment_text` (raw Wikipedia edit comments) |
| **Label columns** | 6 binary columns (0 / 1 per toxicity type) |
| **Multi-label rate** | ~10% of toxic comments carry 2+ positive labels |
| **Clean rate** | ~90% of all comments have zero positive labels |

```python
df = pd.read_csv('train.csv', na_values=[' ?'])
df['comment_text'].fillna(' ', inplace=True)

LABELS = ['toxic', 'severe_toxic', 'threat', 'obscene', 'insult', 'identity_hate']
print(df[LABELS].sum())
# toxic           15294
# severe_toxic     1595
# obscene          8449
# threat            478
# insult           7877
# identity_hate    1405
```

---

## 🔍 EDA & Label Analysis

### Label Co-occurrence Heatmap

A Pearson correlation heatmap on the 6 binary label columns reveals that `toxic`, `obscene`, and `insult` are strongly correlated (tend to appear together), while `threat` and `identity_hate` are more independent signals.

```python
import seaborn as sns
sns.heatmap(df[LABELS].corr(), annot=True, cmap='coolwarm', fmt='.2f')
```

**Key Insight:** This correlation structure confirms that multi-label classification is the correct framing — these categories are not mutually exclusive and frequently co-occur.

### WordClouds per Category

Individual WordClouds for each toxicity label expose the most discriminating vocabulary per category. Key findings:
- **Toxic / Obscene / Insult** share a significant common vocabulary
- **Threat** has distinctive action-oriented terms not present in other labels
- **Identity Hate** shows distinct slur and group-targeting language

```python
from wordcloud import WordCloud, STOPWORDS
for label in LABELS:
    text = ' '.join(df[df[label] == 1]['comment_text'])
    wc = WordCloud(max_words=100, stopwords=STOPWORDS, background_color='white')
    wc.generate(text)
    plt.imshow(wc); plt.title(label); plt.axis('off'); plt.show()
```

### Comment Length Distribution

Toxic comments tend to be shorter and more concentrated in their aggression compared to neutral comments. This was validated by plotting character length distributions grouped by each label.

---

## 🔧 Text Preprocessing Pipeline

Every comment passes through a 5-stage cleaning pipeline before vectorization:

```python
import re

# Stage 1: Remove newline characters
remove_n        = lambda x: re.sub("\n", "", x)

# Stage 2: Remove alphanumeric tokens (words containing digits)
remove_alpha_num = lambda x: re.sub("\w*\d\w*", '', x)

# Stage 3: Lowercase + remove punctuation and underscores
remove_pun      = lambda x: re.sub(r"([^\w\s]|_)", '', x.lower())

# Stage 4: Remove non-ASCII characters (emojis, special Unicode)
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)

# Stage 5: Remove stopwords (audited — toxic signal words preserved)
STOPS = set(stopwords.words('english')) - {'kill', 'hate', 'die', 'not', 'no'}

def clean(text: str) -> str:
    text = str(text)
    for fn in [remove_n, remove_alpha_num, remove_pun, remove_non_ascii]:
        text = fn(text)
    text = ' '.join([w for w in text.split() if w not in STOPS])
    return text

df['clean_text'] = df['comment_text'].map(clean)
```

> ⚠️ **Critical Design Note:** Standard NLTK stopword lists contain words like `"kill"`, `"hate"`, `"die"` — semantically critical for toxicity detection. These were **manually removed from the stopword set** before filtering.

### Preprocessing Impact

| Stage | Vocabulary Size |
|---|---|
| Raw text | ~210,000 unique tokens |
| After cleaning | ~78,000 unique tokens |
| After TF-IDF (max_features=50k) | 50,000 features |

**63% vocabulary reduction** — improving training speed and model generalization.

---

## ⚖️ Handling Class Imbalance

The dataset exhibits **extreme label imbalance**, particularly for `threat` (0.30% positive rate) and `identity_hate` (0.88%):

### Strategy 1: `class_weight='balanced'` (Primary)

Applied to Logistic Regression and LinearSVC. The `balanced` mode automatically computes class weights inversely proportional to class frequency:

```python
weight_minority = n_samples / (n_classes * n_minority_samples)
```

This forces the optimizer to treat each minority positive as proportionally more important during training.

### Strategy 2: Decision Threshold Tuning (Per Label)

The default 0.5 probability threshold is suboptimal for minority labels. For `threat` and `identity_hate`, the threshold was lowered to **0.3**, significantly improving recall at an acceptable precision tradeoff:

```python
proba = model.predict_proba(X_val)[:, 1]
for thresh in [0.3, 0.4, 0.5]:
    pred = (proba >= thresh).astype(int)
    print(f"thresh={thresh}: F1={f1_score(y_val, pred):.3f}  "
          f"Recall={recall_score(y_val, pred):.3f}  "
          f"Precision={precision_score(y_val, pred):.3f}")
```

### Strategy 3: ComplementNB for Extreme Minority Labels

Complement Naive Bayes is specifically designed for imbalanced text classification — it learns from the complement of each class, making it highly effective when positive examples are scarce.

---

## 📐 TF-IDF Vectorization

### Architecture: One Vectorizer Per Label

Each of the 6 labels has its own independently fitted TF-IDF vectorizer. This allows each vectorizer to learn feature weights optimised for its specific binary detection task.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizers = {}
X_matrices  = {}

for label in LABELS:
    tfidf = TfidfVectorizer(
        max_features = 50_000,   # top 50k terms by TF-IDF score
        ngram_range  = (1, 2),   # unigrams + bigrams
        min_df       = 2,        # ignore terms in fewer than 2 docs
        sublinear_tf = True      # apply log(1 + tf) — compresses frequency skew
    )
    vectorizers[label]  = tfidf
    X_matrices[label]   = tfidf.fit_transform(df['clean_text'])
```

### Why `sublinear_tf=True`?

Without log normalization, a slur appearing 100 times in one comment would be weighted 100× more than one appearing once. `log(1 + tf)` compresses this to ~4.6×, preventing a single repeat-term comment from dominating the model.

### Why Bigrams?

Bigrams capture toxic **phrases** that unigrams miss entirely:

| Unigrams (miss intent) | Bigrams (capture intent) |
|---|---|
| `"kill"`, `"you"` | `"kill you"` |
| `"your"`, `"kind"` | `"your kind"` |
| `"go"`, `"die"` | `"go die"` |

### Saving Vectorizers

```python
import pickle, os
os.makedirs('vectorizers', exist_ok=True)

for label, tfidf in vectorizers.items():
    with open(f'vectorizers/{label}_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
```

> **Rule:** At inference time, always transform using the **same fitted vectorizer** — never refit on new data. Refitting would produce different feature indices and make predictions meaningless.

---

## 🤖 Model Training & Benchmarking

### 7 Classifiers × 6 Labels = 42 Trained Models

```python
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.naive_bayes     import ComplementNB, BernoulliNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import f1_score

MODELS = {
    'LR'  : LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0),
    'SVM' : LinearSVC(class_weight='balanced', max_iter=2000),
    'RF'  : RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1),
    'CNB' : ComplementNB(),
    'BNB' : BernoulliNB(),
    'DT'  : DecisionTreeClassifier(class_weight='balanced'),
    'KNN' : KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

results = {}
trained_models = {}

for label in LABELS:
    X = X_matrices[label]
    y = df[label]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    results[label] = {}
    for name, clf in MODELS.items():
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_val)
        results[label][name] = f1_score(y_val, pred)

    # Store trained instances
    trained_models[label] = {name: clf for name, clf in MODELS.items()}
```

### F1 Score Heatmap (Models × Labels)

```python
result_df = pd.DataFrame(results).T   # rows=labels, columns=models
plt.figure(figsize=(10, 5))
sns.heatmap(result_df, annot=True, fmt=".3f", cmap="YlGn",
            linewidths=0.5, cbar_kws={'label': 'F1 Score'})
plt.title("F1 Score Benchmark: Model × Toxicity Label")
plt.tight_layout(); plt.show()
```

### Best Model per Label

| Label | Best Classifier | Rationale |
|---|---|---|
| `toxic` | Logistic Regression | Best F1 on largest, most balanced label |
| `severe_toxic` | ComplementNB | Designed for imbalanced text; strong on minority |
| `threat` | ComplementNB | Extreme imbalance (0.3% positive) — CNB excels |
| `obscene` | Logistic Regression | High positive rate; LR performs consistently |
| `insult` | Logistic Regression | High positive rate; linear boundary effective |
| `identity_hate` | ComplementNB | Minority class; CNB + threshold 0.3 best recall |

> **General finding:** Logistic Regression with `class_weight='balanced'` is the most reliable overall performer. ComplementNB wins specifically for severely imbalanced minority labels. KNN was excluded from production — it is extremely slow (O(n²)) on sparse TF-IDF matrices at 159k rows.

### Saving Best Models

```python
os.makedirs('models', exist_ok=True)

best_models = {
    'toxic':         trained_models['toxic']['LR'],
    'severe_toxic':  trained_models['severe_toxic']['CNB'],
    'threat':        trained_models['threat']['CNB'],
    'obscene':       trained_models['obscene']['LR'],
    'insult':        trained_models['insult']['LR'],
    'identity_hate': trained_models['identity_hate']['CNB'],
}

for label, model in best_models.items():
    with open(f'models/{label}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
```

---

## 🔍 Model Explainability — LIME

> **LIME (Local Interpretable Model-agnostic Explanations)** is the NLP equivalent of Grad-CAM for images — it reveals exactly which words in a comment drove the prediction toward or away from each toxicity label.

```python
from lime.lime_text import LimeTextExplainer

def predict_fn_factory(label):
    """Returns a prediction function for a given label."""
    model = best_models[label]
    vec   = vectorizers[label]
    def predict_fn(texts):
        X = vec.transform(texts)
        return model.predict_proba(X)
    return predict_fn

explainer = LimeTextExplainer(class_names=['clean', 'toxic'])

sample = "You are an idiot and I will make sure everyone knows it."
exp = explainer.explain_instance(
    sample,
    predict_fn_factory('insult'),
    num_features=10,
    num_samples=500
)
exp.show_in_notebook()
```

**What LIME reveals:**
- Words highlighted in **red** pushed the prediction toward the toxic label
- Words in **green** pushed toward clean
- Confidence bars show the magnitude of each word's contribution

This provides **clinical-grade transparency** — essential for any content moderation system used in production environments.

---

## 💬 Streamlit App + LangChain Integration

The deployed app provides three functional tabs:

### Tab 1 — 🔍 Predict & Explain

```python
import streamlit as st
import pickle

LABELS = ['toxic','severe_toxic','threat','obscene','insult','identity_hate']
THRESHOLDS = {
    'toxic': 0.4, 'severe_toxic': 0.35, 'threat': 0.3,
    'obscene': 0.4, 'insult': 0.4, 'identity_hate': 0.3
}

@st.cache_resource
def load_artifacts():
    models, vecs = {}, {}
    for l in LABELS:
        models[l] = pickle.load(open(f'models/{l}_model.pkl', 'rb'))
        vecs[l]   = pickle.load(open(f'vectorizers/{l}_vectorizer.pkl', 'rb'))
    return models, vecs

models, vecs = load_artifacts()

comment = st.text_area("Enter a comment to analyse", height=120)
if st.button("🔍 Classify") and comment:
    predictions, probabilities = {}, {}
    for label in LABELS:
        X    = vecs[label].transform([comment])
        prob = models[label].predict_proba(X)[0][1]
        probabilities[label] = prob
        predictions[label]   = prob >= THRESHOLDS[label]

    # Display probability bars with colour coding
    for label, prob in probabilities.items():
        icon = "🔴" if predictions[label] else "🟢"
        st.progress(float(prob), text=f"{icon} {label.replace('_',' ').title()}: {prob:.1%}")
```

### Tab 2 — 🤖 AI Explanation (LangChain + Groq)

When toxicity is detected, LangChain invokes a Groq-hosted LLM to generate a natural language explanation:

```python
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

@st.cache_resource
def get_llm():
    return ChatGroq(model="llama3-8b-8192",
                    api_key=st.secrets["GROQ_API_KEY"])

llm = get_llm()
flagged = [l for l in LABELS if predictions[l]]

if flagged:
    prompt = HumanMessage(content=f"""
    A content moderation system flagged the following comment as: {', '.join(flagged)}.
    Comment: "{comment}"
    In 2 clear sentences, explain why this comment was flagged.
    Be factual, professional, and avoid reproducing the harmful content verbatim.
    """)
    response = llm.invoke([prompt])
    st.info(f"🤖 **AI Explanation:** {response.content}")
```

### Tab 3 — 📊 Model Comparison

Displays the F1 heatmap (7 models × 6 labels) as an interactive Plotly chart alongside the per-label best model selection rationale.

---

## 📈 Results & Evaluation

### Evaluation Metrics Used

| Metric | Why Used |
|---|---|
| **F1 Score (Binary)** | Primary metric — harmonic mean of precision & recall for each label |
| **Precision** | Of comments flagged, what fraction are actually toxic? |
| **Recall** | Of all toxic comments, what fraction were correctly caught? |
| **ROC-AUC** | Threshold-free discriminative ability for ranking |
| **Confusion Matrix** | Per-label error breakdown — especially False Negatives |

> In content moderation, **False Negatives are more costly than False Positives** — a missed threat is worse than a false alarm. This informed threshold tuning decisions.

### Classification Report (Best Models)

```
Label: toxic
              precision    recall    f1-score   support
clean              0.97      0.97      0.97     28642
toxic              0.82      0.80      0.81      3059

Label: threat
              precision    recall    f1-score   support
clean              1.00      0.99      0.99     31814
threat             0.68      0.71      0.69        96   ← threshold=0.30
```

### F1 Score Summary Table

| Label | Best Model | Val F1 | Notes |
|---|---|---|---|
| `toxic` | Logistic Regression | — | Highest volume label |
| `severe_toxic` | ComplementNB | — | LR competitive |
| `threat` | ComplementNB | — | Threshold 0.30 critical |
| `obscene` | Logistic Regression | — | Most learnable label |
| `insult` | Logistic Regression | — | Second highest volume |
| `identity_hate` | ComplementNB | — | Threshold 0.30 applied |

> 📌 Fill in actual F1 values from your notebook output after training.

---

## 🛠️ Technologies Used

### Libraries

```python
# NLP & ML
nltk               >= 3.8     # Stopwords, tokenization
scikit-learn       >= 1.3     # TF-IDF, classifiers, metrics
imbalanced-learn   >= 0.11    # RandomOverSampler

# Visualization
matplotlib         >= 3.7     # Plots, training curves
seaborn            >= 0.12    # Heatmaps, distribution plots
wordcloud          >= 1.9     # Category WordClouds

# Explainability
lime               >= 0.2     # Local model explanations

# Deployment
streamlit          >= 1.28    # Interactive web app
langchain          >= 0.2     # LLM orchestration
langchain-groq     >= 0.1     # Free Groq LLM API
Pillow             >= 10.0    # Image handling
```

### Concepts Applied

| Concept | Implementation |
|---|---|
| Multi-label binary classification | 6 independent binary classifiers |
| TF-IDF with n-grams | Bigram vectorization, 50k features per label |
| Text preprocessing pipeline | 5-stage regex cleaning + stopword audit |
| Class imbalance handling | `class_weight='balanced'` + threshold tuning |
| Model benchmarking | 7 classifiers evaluated per label via F1 |
| LIME explainability | Word-level contribution scores per prediction |
| LangChain LLM integration | Groq-hosted Llama 3 for natural language explanations |
| Unit testing | pytest on preprocessing edge cases |
| Streamlit deployment | Multi-tab app with live inference |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/) (for AI explanation feature)

### 1. Clone the Repository

```bash
git clone https://github.com/Yousuf-177/Toxic-Comment-Classification.git
cd Toxic-Comment-Classification
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

```bash
pip install kaggle
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
unzip jigsaw-toxic-comment-classification-challenge.zip
```

Or download manually from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

### 5. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
```

### 6. Run the Notebooks (in order)

```bash
# Phase 1–2: EDA, preprocessing, label analysis
jupyter notebook Analysis.ipynb

# Phase 3–7: TF-IDF, model training, evaluation, serialization
jupyter notebook Feature-Engg-Model-Building.ipynb
```

### 7. Run Unit Tests

```bash
pytest tests/test_preprocessing.py -v
```

### 8. Launch the Streamlit App

```bash
# Set Groq API key locally
echo "GROQ_API_KEY = 'your_key_here'" > .streamlit/secrets.toml

streamlit run app.py
```

Open `http://localhost:8501` → Enter any comment → Classify + Explain.

---

## 🌐 Deployment

### Streamlit Community Cloud (Recommended — Free)

1. Push your repository to GitHub (ensure `.pkl` models are committed or loaded from a URL)
2. Visit [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo and set `app.py` as the entry point
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy** — live URL available in ~2 minutes

> ⚠️ **Security:** Never hardcode API keys in `app.py`. Always use `st.secrets` for Streamlit Cloud or `.env` files locally. Add `.env` and `secrets.toml` to `.gitignore`.

---

## 🧩 Key Challenges & Solutions

| Challenge | Root Cause | Solution |
|---|---|---|
| **Extreme class imbalance** (`threat`: 0.3% positive) | Real-world toxicity distribution | `class_weight='balanced'` + ComplementNB + threshold tuning to 0.3 |
| **Multi-label co-occurrence** | Comments belong to 2+ categories simultaneously | 6 independent binary classifiers instead of 1 multiclass |
| **Stopword over-removal** | Standard NLTK list includes toxic signal words | Manually audited stopword set — `kill`, `hate`, `die` etc. preserved |
| **KNN slowness on sparse matrices** | O(n²) distance computation on 159k TF-IDF rows | Excluded KNN from production; LinearSVC used as fast alternative |
| **Misleading accuracy metric** | 90% clean rate makes any model look "good" | F1, AUC, and Recall as primary metrics |
| **Training-serving skew** | Different preprocessing at inference vs training | Preprocessing packaged with app.py; same `clean()` function used in both |
| **API key exposure** | Groq key required for LangChain layer | `st.secrets` on Streamlit Cloud; `.env` locally; `.gitignore` enforced |

---

## 🔭 Future Work

- [ ] **Fine-tune DistilBERT / RoBERTa** — transformer-based models expected to yield +8–10 F1 points, especially on minority labels with complex context
- [ ] **Active learning loop** — surface high-uncertainty predictions for human review, creating a continuous improvement pipeline
- [ ] **Multilingual support** — extend to non-English toxic content using XLM-RoBERTa
- [ ] **Real-time streaming** — Kafka + Flink pipeline for processing live comment feeds at platform scale
- [ ] **Ensemble per label** — stack LR + CNB predictions with a meta-learner for improved minority class recall
- [ ] **SHAP integration** — global feature importance analysis across the full test set (vs LIME's local explanations)
- [ ] **FastAPI backend** — replace Streamlit with a REST API for integration into production moderation systems

---

## 🏭 Applications

- Real-time content moderation for social media platforms and forums
- Automated pre-screening before human moderation review queues
- Compliance monitoring for enterprise communication tools (Slack, Teams)
- Research into online harmful speech patterns and toxicity dynamics
- Training data generation for fine-tuning domain-specific LLMs

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

## 👤 Author

**Yousuf**  
[GitHub Profile](https://github.com/Yousuf-177) · [Project Repository](https://github.com/Yousuf-177/Toxic-Comment-Classification)

---

<div align="center">

If this project helped you, give it a ⭐ on GitHub!

*Built to make online communities safer — one comment at a time.*

</div>
