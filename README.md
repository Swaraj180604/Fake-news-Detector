# 🔍 Fake News Detector AI

An AI-powered misinformation detection tool built with Python, scikit-learn, and Streamlit. It uses a **TF-IDF + Ensemble ML** pipeline to classify news headlines and article excerpts as **Fake** or **Real** — with credibility scoring, risk levels, and linguistic red-flag detection.

---

## 🚀 Live Demo

> Deploy your own on Streamlit Cloud — see [Deployment](#-deployment) below.

---

## 📸 Features

- **Ensemble ML model** — Logistic Regression + Random Forest + Gradient Boosting (soft voting)
- **TF-IDF feature extraction** — unigrams, bigrams, and trigrams (10,000 features)
- **Credibility score** — 0–100 score indicating trustworthiness
- **Risk level** — Very Low → Very High based on fake probability
- **Linguistic red-flag detection** — ALL-CAPS words, excessive punctuation, trigger words (BREAKING, BOMBSHELL, EXPOSED, etc.)
- **Credibility signal detection** — peer-reviewed, WHO, FDA, university, journal, etc.
- **Interactive UI** — built with Streamlit, includes example presets and probability progress bars
- **Auto-trains on startup** — no pre-saved model file needed; trains fresh and caches via `@st.cache_resource`

---

## 🧠 How It Works

### Pipeline

```
Raw Text
   ↓
Text Cleaning (lowercase, remove URLs, normalize punctuation)
   ↓
TF-IDF Vectorizer (1–3 ngrams, 10k features, sublinear_tf)
   ↓
Soft Voting Ensemble
   ├── Logistic Regression  (weight: 3)
   ├── Random Forest        (weight: 2)
   └── Gradient Boosting    (weight: 2)
   ↓
Fake Probability | Real Probability | Credibility Score | Risk Level | Red Flags
```

### Red Flag Signals (Fake indicators)
| Signal | Example |
|--------|---------|
| ALL-CAPS words | `BREAKING`, `PROOF`, `SHOCKING` |
| Excessive punctuation | `!!!`, `???` |
| Trigger phrases | `BOMBSHELL`, `EXPOSED`, `THEY DON'T WANT YOU TO KNOW` |
| Emotional manipulation patterns | `You WON'T BELIEVE`, `MUST SHARE` |

### Credibility Signals (Real indicators)
| Signal | Example |
|--------|---------|
| Academic language | `peer-reviewed`, `published`, `journal` |
| Authority sources | `WHO`, `FDA`, `CDC`, `university` |
| Precise data | percentages, statistics, figures |
| Neutral phrasing | `announced`, `report`, `study` |

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py              # Main Streamlit app (self-contained, includes model)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

> **Note:** `app.py` is fully self-contained. The ML model code, training data, and UI are all in one file — no separate modules needed for Streamlit Cloud.

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ☁️ Deployment

### Streamlit Cloud (Recommended — Free)

1. Push your code to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repository and set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy"** ✓

The model trains automatically on first load and is cached for all subsequent users.

> ⚠️ **Common mistake:** Do NOT use Flask's `app.run()` on Streamlit Cloud — it will crash with a `SIGTERM` signal error. This app uses pure Streamlit.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.35.0 | Web UI framework |
| `scikit-learn` | ≥1.3.0 | ML models + TF-IDF |
| `numpy` | ≥1.24.0 | Numerical operations |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🤖 Model Details

### Training Data
- **76 labeled samples** — 38 fake, 38 real
- Fake samples: conspiracy theories, health hoaxes, political misinformation, sensationalist headlines
- Real samples: factual news from finance, science, health, government, research

### Model Architecture

```python
Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,3), max_features=10000, sublinear_tf=True)),
    ('clf',   VotingClassifier(
                estimators=[('lr', LogisticRegression(C=5.0)),
                            ('rf', RandomForestClassifier(n_estimators=100)),
                            ('gb', GradientBoostingClassifier(n_estimators=100))],
                voting='soft',
                weights=[3, 2, 2]
              ))
])
```

### Why This Ensemble?
- **Logistic Regression** (highest weight) — excels on sparse high-dimensional TF-IDF features; interpretable and fast
- **Random Forest** — captures non-linear feature interactions; robust to noise
- **Gradient Boosting** — sequential error correction; catches patterns the other two miss
- **Soft voting** — averages class probabilities for a smoother, more calibrated final score

---

## 📊 Output Explained

| Output | Description |
|--------|-------------|
| **Verdict** | `LIKELY FAKE` or `LIKELY REAL` |
| **Confidence** | How certain the model is (max of fake/real probability) |
| **Fake Probability** | % chance the text is misinformation |
| **Real Probability** | % chance the text is genuine news |
| **Credibility Score** | 0–100, higher = more trustworthy (= Real Probability × 100) |
| **Risk Level** | Very Low / Low / Medium / High / Very High |
| **Red Flags** | Specific linguistic patterns that triggered the classifier |

### Risk Level Thresholds

| Fake Probability | Risk Level |
|-----------------|------------|
| ≥ 80% | 🔴 Very High |
| 60–79% | 🟠 High |
| 40–59% | 🟡 Medium |
| 20–39% | 🟢 Low |
| < 20% | ✅ Very Low |

---

## 🛠️ Extending the Model

### Add more training data
Edit the `FAKE_SAMPLES` and `REAL_SAMPLES` lists in `app.py` to improve accuracy with more examples.

### Use a real dataset
For production use, replace the built-in samples with a dataset like:
- [LIAR dataset](https://huggingface.co/datasets/liar) — 12,800+ labeled statements
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) — news articles with social context
- [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) — 44,000+ articles

```python
import pandas as pd
df = pd.read_csv("your_dataset.csv")
FAKE_SAMPLES = df[df['label'] == 'FAKE']['text'].tolist()
REAL_SAMPLES = df[df['label'] == 'REAL']['text'].tolist()
```

### Swap in a transformer model
For higher accuracy, replace the TF-IDF + sklearn pipeline with a HuggingFace model:
```bash
pip install transformers torch
```
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="hamzab/roberta-fake-news-classification")
```

---

## ⚠️ Disclaimer

This tool is a **demonstration project** using a small synthetic training dataset. It is not production-grade and should not be used as the sole arbiter of whether a news story is true or false. Always verify information from multiple reputable sources.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Built With

- [Streamlit](https://streamlit.io) — UI framework
- [scikit-learn](https://scikit-learn.org) — machine learning
- [Python](https://python.org) — language
