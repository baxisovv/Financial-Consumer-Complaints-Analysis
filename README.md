# 💬 Financial Consumer Complaints Analysis

> Hybrid NLP pipeline that clusters and classifies financial consumer complaints using K-Means, traditional ML (Naive Bayes, Logistic Regression, Random Forest), and a deep learning LSTM — all in a single end-to-end script.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=flat)

---

## 📌 Overview

This project analyzes the [CFPB Consumer Complaints dataset](https://www.consumerfinance.gov/data-research/consumer-complaints/) — a public collection of complaints submitted to the Consumer Financial Protection Bureau. Each complaint includes a free-text narrative and a product category label.

The pipeline has two goals:
1. **Unsupervised** — discover natural complaint groupings via K-Means clustering
2. **Supervised** — classify complaints to their financial product category using both ML and deep learning models

---

## 🔄 Pipeline

```
Consumer_Complaints.csv (up to 140,000 rows)
        │
        ▼
  NLP Preprocessing
  (lowercase → remove anonymized XXX → remove non-alpha →
   tokenize → remove stopwords → lemmatize)
        │
        ├──── TF-IDF (1,000 features) ──── K-Means (6 clusters) ──── PCA plot
        │
        ├──── TF-IDF (same matrix) ──────  Naive Bayes
        │                                  Logistic Regression (balanced)
        │                                  Random Forest (balanced)
        │
        └──── Tokenizer (20k vocab) ──────  LSTM
             + Padding (150 tokens)         (Embedding → SpatialDropout → LSTM → Dense)
                                            with class_weight='balanced'
        │
        ▼
  Results table + 7 saved plots → görsel13/
```

---

## 🧠 Models

### K-Means Clustering (Unsupervised)
- TF-IDF on complaint text (max 1,000 features)
- 6 clusters, `n_init=10`
- PCA (2D) scatter plot saved for visual inspection

### Classical ML (Supervised)
All trained on the same TF-IDF matrix, 70/30 train-test split:

| Model | Notes |
|---|---|
| Naive Bayes | MultinomialNB, baseline |
| Logistic Regression | `class_weight='balanced'`, `max_iter=1000` |
| Random Forest | 50 trees, `class_weight='balanced'`, `n_jobs=-1` |

Metrics reported: Accuracy, Precision, Recall, F1 (weighted and macro)

### LSTM (Deep Learning)
```
Tokenizer (vocab=20,000) → Padding (maxlen=150)
  → Embedding(20000, 100)
  → SpatialDropout1D(0.2)
  → LSTM(100, dropout=0.2, recurrent_dropout=0.2)
  → Dense(n_classes, softmax)
```
- Optimizer: Adam, Loss: categorical crossentropy
- 10 epochs, batch size 64
- Class weights computed via `sklearn.utils.class_weight.compute_class_weight` to handle label imbalance
- Stratified train/test split

---

## 📂 Output Files

All outputs are saved to the `görsel13/` directory:

| File | Description |
|---|---|
| `1_KMeans_Clusters_PCA.png` | PCA scatter plot of 6 K-Means clusters |
| `2_Cluster_Distribution.png` | Complaint count per cluster |
| `3_CM_Naive_Bayes.png` | Confusion matrix — Naive Bayes |
| `3_CM_Logistic_Regression.png` | Confusion matrix — Logistic Regression |
| `3_CM_Random_Forest.png` | Confusion matrix — Random Forest |
| `4_LSTM_Training_History.png` | Train/val accuracy and loss curves |
| `4_CM_LSTM.png` | Confusion matrix — LSTM |
| `5_Final_Comparison_Accuracy.png` | Bar chart comparing all model accuracies |
| `Final_Results_Table.csv` | All metrics for all models in one table |

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow nltk wordcloud
```

### Download NLTK resources

The script handles this automatically on first run, but you can also run manually:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### Download the dataset

Download **Consumer_Complaints.csv** from the CFPB:

👉 [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)

Place it in the same directory as `main.py`.

### Run

```bash
python main.py
```

The script processes up to 140,000 rows and runs all steps sequentially. Expect a few minutes for NLP preprocessing and LSTM training.

---

## ⚙️ Configuration

Key settings at the top of `main.py`:

```python
FILE_PATH    = 'Consumer_Complaints.csv'
OUTPUT_DIR   = 'görsel13'
MAX_ROWS     = 140000   # rows to process
TEST_SIZE    = 0.3      # 70/30 train-test split
RANDOM_STATE = 42
```

---

## 📦 Dependencies

```
pandas · numpy · matplotlib · seaborn
scikit-learn · nltk · wordcloud
tensorflow (keras)
```

---

## 📄 License

MIT
