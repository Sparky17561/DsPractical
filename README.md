
````markdown
# 🧠 Data Science + NLP Cheat Sheet

> **Author:** Saiprasad Jamdar  
> **Purpose:** Quick-recall guide for ML, NLP & Visualization — compact and memory-friendly.

---

## 📦 1️⃣ Import Essentials

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
````

| Category | Example         | Meaning            |
| -------- | --------------- | ------------------ |
| Data     | `pd.read_csv()` | Load CSV           |
| Info     | `df.info()`     | Data types & nulls |
| Stats    | `df.describe()` | Summary stats      |

---

## 🧹 2️⃣ Cleaning

```python
df[col].replace(0, np.nan)
df[col].fillna(df[col].mean(), inplace=True)
df.isnull().sum()
```

---

## 📊 3️⃣ EDA (Exploratory Data Analysis)

```python
sns.histplot(df['Glucose'])
sns.scatterplot(x='Glucose', y='Insulin', data=df)
sns.boxplot(x='Outcome', y='BMI', data=df)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

---

## 🧮 4️⃣ Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## ⚙️ 5️⃣ Scale Data

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 🤖 6️⃣ Train Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = {
 'LogReg': LogisticRegression(),
 'RF': RandomForestClassifier(),
 'DT': DecisionTreeClassifier(),
 'GB': GradientBoostingClassifier(),
 'SVM': SVC()
}

for n, m in models.items():
    m.fit(X_train, y_train)
```

---

## 📈 7️⃣ Evaluate

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 🧾 8️⃣ Compare Models

```python
scores = {n: accuracy_score(y_test, m.predict(X_test)) for n, m in models.items()}
pd.DataFrame(scores.items(), columns=['Model','Accuracy'])
```

---

## 🧠 Quick Recall Mnemonics

| Step | Action           | Keyword |
| ---- | ---------------- | ------- |
| 1️⃣  | Load Data        | Load    |
| 2️⃣  | Clean Data       | Clean   |
| 3️⃣  | Visualize        | See     |
| 4️⃣  | Train-Test Split | Split   |
| 5️⃣  | Scale Data       | Scale   |
| 6️⃣  | Train Model      | Train   |
| 7️⃣  | Predict          | Test    |
| 8️⃣  | Evaluate         | Score   |
| 9️⃣  | Compare Models   | Compare |

---

# 💉 Biomedical NLP (spaCy + Sentiment)

```python
import spacy, matplotlib.pyplot as plt
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# Extract Entities
def extract_entities(text):
    doc = nlp(text)
    ents = [(e.text, e.label_) for e in doc.ents]
    print("\nEntities:")
    for e in ents: print(e[0], "→", e[1])
    return ents

# Rule-based Sentiment
def analyze_sentiment(notes):
    POS, NEG = {"improve","better","stable","recover","normal"}, {"severe","worse","pain","difficult","fever"}
    scores = []
    for n in notes:
        s = sum(w in n.lower() for w in POS) - sum(w in n.lower() for w in NEG)
        sentiment = "Positive" if s>0 else "Negative" if s<0 else "Neutral"
        print(f"{n} → {sentiment}")
        scores.append(s)
    return scores

# Plot Results
def plot_results(ents, scores):
    if ents:
        lbls = [e[1] for e in ents]
        plt.bar(Counter(lbls).keys(), Counter(lbls).values())
        plt.title("Entity Count"); plt.show()
    plt.bar(range(len(scores)), scores, color=['g' if s>0 else 'r' if s<0 else 'b' for s in scores])
    plt.title("Sentiment per Note"); plt.show()

# Example
text = "Patient diagnosed with diabetes and hypertension. Prescribed Metformin."
notes = ["Patient improving after medication.","Severe pain persists.","Condition stable."]
ents = extract_entities(text)
scores = analyze_sentiment(notes)
plot_results(ents, scores)
```

---

# 💊 Drug Review NLP + Word Cloud + Predictive Model

```python
import pandas as pd, re, matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Simulate Data ---
df = pd.DataFrame({
 'review': [
  "This medicine worked great for my headache, quick relief!",
  "Terrible side effects, I felt dizzy and nauseous.",
  "It helped reduce my fever in just one day.",
  "Did not work at all, complete waste of money.",
  "Mild improvement, but caused stomach pain.",
  "Highly effective for pain relief, totally recommend it!",
  "Too expensive for such little effect.",
  "Very satisfied with the results, no side effects.",
  "I had allergic reactions after taking this pill.",
  "Works okay, but takes too long to show effect."
 ],
 'drug_name': [
  "PainAway","PainAway","FeverGo","FeverGo","CurePlus",
  "PainAway","CurePlus","FeverGo","CurePlus","PainAway"
 ]
})

# --- Clean & Sentiment ---
df['clean'] = df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).lower())
df['sentiment'] = df['clean'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['label'] = df['sentiment'].apply(lambda x: 'Positive' if x>0 else 'Negative')

# --- Sentiment Plot ---
df['label'].value_counts().plot(kind='bar', color=['green','red'])
plt.title('Sentiment Distribution'); plt.show()

# --- Word Clouds ---
pos = " ".join(df[df['label']=='Positive']['clean'])
neg = " ".join(df[df['label']=='Negative']['clean'])
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(WordCloud(bg_color='white').generate(pos)); plt.title('Positive Reviews')
plt.subplot(1,2,2); plt.imshow(WordCloud(bg_color='white').generate(neg)); plt.title('Negative Reviews')
plt.show()

# --- Predictive Model ---
df['keyword_score'] = df['clean'].apply(lambda x:
 sum(w in x for w in ['good','great','effective','recommend','relief'])
 - sum(w in x for w in ['bad','terrible','waste','pain','dizzy','allergic'])
)
X = df[['sentiment','keyword_score']]
y = df['label'].map({'Positive':1,'Negative':0})
Xtr,Xte,Ytr,Yte = train_test_split(X,y,test_size=0.3,random_state=42)
m = LogisticRegression().fit(Xtr,Ytr)
print("Accuracy:", accuracy_score(Yte,m.predict(Xte)))
print(classification_report(Yte,m.predict(Xte), target_names=['Negative','Positive']))

# --- Drug-wise Sentiment Plot ---
drug_avg = df.groupby('drug_name')['sentiment'].mean()
drug_avg.plot(kind='bar', color='skyblue')
plt.title('Average Sentiment per Drug'); plt.ylabel('Polarity'); plt.show()
```

---

## ☁️ Visualization Summary

| Plot | Function                 | Purpose                  |
| ---- | ------------------------ | ------------------------ |
| 📊   | `value_counts().plot()`  | Sentiment distribution   |
| ☁️   | `WordCloud().generate()` | Positive/Negative clouds |
| 💊   | `groupby('drug_name')`   | Drug-level sentiment     |

---

## ⚡ TL;DR Quick Recall Table

| Section  | Function / Command        | Purpose              |
| -------- | ------------------------- | -------------------- |
| Data     | `pd.read_csv()`           | Load data            |
| Cleaning | `replace()`, `fillna()`   | Handle nulls         |
| Split    | `train_test_split()`      | Train/Test           |
| Model    | `LogisticRegression()`    | Train model          |
| Evaluate | `accuracy_score()`        | Check accuracy       |
| NLP      | `TextBlob`, `spaCy`       | Sentiment & entities |
| Viz      | `matplotlib`, `WordCloud` | Plots                |

---

```

