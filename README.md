# Toxic Comment Classifier using NLP & Machine Learning

An end-to-end **Natural Language Processing (NLP)** and **Machine Learning** project that detects and classifies toxic online comments into multiple toxicity categories using traditional ML algorithms, TF-IDF vectorization, and an interactive web interface powered by **Streamlit** and **LangChain**.

---

## Project Overview

Online platforms generate millions of comments daily, making moderation difficult. This project focuses on automatically identifying toxic comments and classifying them into six categories:

- Toxic
- Severe Toxic
- Threat
- Obscene
- Insult
- Identity Hate

### Important Note
Instead of using a **single multiclass classifier**, this project performs:

> **Independent Binary Classification for each toxicity category**

This improves flexibility and allows a single comment to belong to multiple toxicity labels simultaneously.

---

# 🛠 Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| NLP | NLTK, Regular Expressions |
| ML Models | Logistic Regression, SVM, Random Forest, Decision Tree, KNN, CNB, BNB |
| Vectorization | TF-IDF (Unigram) |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Deployment | Streamlit |
| AI Integration | LangChain |

---

# Dataset

Dataset used from Kaggle:

👉 [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

---

# Required Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from PIL import Image

from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
```

---

# Data Loading

```python
df_train = pd.read_csv('train.csv', na_values=[' ?'])
df_test = pd.read_csv('test.csv')

df_train['comment_text'].fillna(' ')
df_test['comment_text'].fillna(' ')
```

---

# Dataset Preview

![Dataset Preview](https://github.com/user-attachments/assets/c4e85c56-7f33-4318-b9e4-0b3455d395e4)

---

# Text Preprocessing Pipeline

Before training the models, the comment text undergoes multiple preprocessing steps to clean noisy data.

### Preprocessing Steps

- Removing newline characters (`\n`)
- Removing alphanumeric words
- Removing punctuations
- Removing non-ASCII characters
- Converting text to lowercase

---

## Preprocessing Code

```python
import re
import string

remove_n = lambda x: re.sub("\n", "", x)

remove_alpha_num = lambda x: re.sub("\w*\d\w*", '', x)

remove_pun = lambda x: re.sub(r"([^\w\s]|_)", '', x.lower())

remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)

df_train['comment_text'] = (
    df_train['comment_text']
    .map(remove_n)
    .map(remove_alpha_num)
    .map(remove_pun)
    .map(remove_non_ascii)
)

df_test['comment_text'] = (
    df_test['comment_text']
    .map(remove_n)
    .map(remove_alpha_num)
    .map(remove_pun)
    .map(remove_non_ascii)
)
```

---

# Model Building

## TF-IDF Vectorization

The project uses:

- TF-IDF Vectorizer
- Unigram Approach

### Why TF-IDF?

TF-IDF converts textual data into numerical vectors by evaluating:

- **Term Frequency (TF)** → Importance of a word inside a document
- **Inverse Document Frequency (IDF)** → Rarity of a word across the dataset

This helps the model focus on meaningful words while reducing the influence of extremely common words.

---

# Machine Learning Models Used

Each toxicity category is trained independently using multiple ML algorithms:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Complement Naive Bayes (CNB)
- Bernoulli Naive Bayes (BNB)
- Decision Tree (DT)
- Random Forest (RF)

---

# Performance Evaluation

## F1-Score of Toxic Category

![F1 Score Toxic](https://github.com/user-attachments/assets/519c180f-4b28-4c01-871a-bcd785488510)

---

## Visualization of F1-Scores for All Categories

```python
result = pd.concat([
    result_toxic,
    result_severe_toxic,
    result_threat,
    result_obscene,
    result_insult,
    result_identity_hate
], axis=1)

result = result.transpose()

plt.figure(figsize=(15,15))
sns.lineplot(data=result, markers=True)
plt.legend(loc='best')
```

![F1 Score Visualization](https://github.com/user-attachments/assets/ac6d26bf-185e-4e28-b624-76815c11a3de)

---

# Final Model Selection

### Selected Model: Logistic Regression

### Exported Components

- 6 Trained ML Models (`.pkl`)
- 6 TF-IDF Vectorizers (`.pkl`)

Each category has:
- One trained classifier
- One corresponding vectorizer

---

# Interactive Web Application

## Streamlit Integration

The project uses **Streamlit** to create an interactive web application where users can:

- Enter comments
- Predict toxicity categories instantly
- Visualize outputs interactively

### Why Streamlit?

Streamlit enables rapid development of ML applications with minimal frontend code.

---

# LangChain Integration

LangChain is integrated to enhance interaction between users and AI systems.

### Components Used

- `HumanMessage`
- `AIMessage`

These components help structure communication between the application and the AI pipeline.

---

# Project Structure

```bash
📦 Toxic-Comment-Classifier
 ┣ 📂 models
 ┃ ┣ 📜 toxic_model.pkl
 ┃ ┣ 📜 insult_model.pkl
 ┃ ┗ 📜 ...
 ┣ 📂 vectorizers
 ┃ ┣ 📜 toxic_vectorizer.pkl
 ┃ ┗ 📜 ...
 ┣ 📂 Balanced Data
 ┣ 📂 Distribution-of-Classification
 ┣ 📜 app.py
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┣ 📜 Analysis.ipynb
 ┗ 📜 Feature-Engg-Model-Building.ipynb
```

---

# Run the Project

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run Streamlit App

```bash
streamlit run app.py
```

---


# Author

Developed as an NLP + Machine Learning project focused on toxic comment detection and moderation systems.