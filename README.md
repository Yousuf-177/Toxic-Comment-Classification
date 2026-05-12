# Toxic Comment Classifier Using NLP and ML

Using Natural Language Processing, `nltk` and Machine Learning to classify comments into Six different Categories including their Neutral Cases. Using Streamlit and Langchain to create interactive web experience. 

- Toxic 
- Severe Toxic
- Threat 
- Obscene
- Insult
- Identity Hate

`Instead of Multiclass classification, Binary Classification of Each Category is performed`

<hr>

## Libraries and Data Collection 

**From Kaggle:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

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
```python
df_train = pd.read_csv('train.csv', na_values=[' ?'])
df_test = pd.read_csv('test.csv')
df_train['comment_text'].fillna(' ')
df_test['comment_text'].fillna(' ')
```
![image](https://github.com/user-attachments/assets/c4e85c56-7f33-4318-b9e4-0b3455d395e4)

<hr>

## Data Pre-Procesing - Text Pre-Processing Using Regular Expressions

- Removing \n characters 
- Removing Aplha-Numeric Characters
- Removing Punctuations
- Removing Non Ascii Characters

```python
import re
import string

remove_n = lambda x: re.sub("\n", "", x)

remove_alpha_num = lambda x: re.sub("\w*\d\w*", '', x)

remove_pun = lambda x: re.sub(r"([^\w\s]|_)", '', x.lower())

remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)

df_train['comment_text'] = df_train['comment_text'].map(remove_n).map(remove_alpha_num).map(remove_pun).map(remove_non_ascii)
df_test['comment_text'] = df_test['comment_text'].map(remove_n).map(remove_alpha_num).map(remove_pun).map(remove_non_ascii)
```
<hr>

