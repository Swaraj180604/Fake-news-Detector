"""
Fake News Detector - Streamlit App
Deploy on Streamlit Cloud: set this file as app.py in your repo root.
"""

import streamlit as st
import pickle, os, sys, re
import numpy as np

# ── inline model code (no separate module needed) ──────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ── Training Data ─────────────────────────────────────────────────────────
FAKE_SAMPLES = [
    "BREAKING: Scientists PROVE vaccines cause autism in secret government study that Big Pharma doesn't want you to know!",
    "Exclusive: President secretly plans to abolish all elections and declare himself dictator next month!!",
    "SHOCKING: Chemtrails contain mind-control chemicals sprayed by the New World Order globalists",
    "You won't believe what doctors are hiding: This one weird trick cures cancer in 3 days!",
    "ALERT: 5G towers are deliberately spreading COVID-19 as part of global depopulation agenda",
    "Celebrity admits to selling children to underground satanic cult in Hollywood expose",
    "Government scientist CONFIRMS the earth is flat and NASA has been lying for decades",
    "Miracle cure discovered: Ancient herb destroys all viruses but pharmaceutical companies banned it",
    "EXCLUSIVE LEAK: Moon landing was filmed in Arizona desert by Stanley Kubrick on government orders",
    "Deep state operatives caught replacing world leaders with robot clones, insider reveals",
    "BOMBSHELL: Climate change is a hoax invented by China to destroy American economy",
    "Secret memo reveals Bill Gates planning to microchip entire global population through water supply",
    "Doctors HATE this: Man cures diabetes in 7 days with this kitchen spice trick!",
    "URGENT WARNING: New law will allow government to seize all guns by end of month",
    "Whistleblower exposes alien technology being used by military to control weather patterns",
    "SHOCKING PHOTOS: UFO fleet spotted over capital city, government scrambles fighter jets to cover up",
    "Hollywood actor reveals Illuminati runs entire entertainment industry and controls elections",
    "Scientists finally ADMIT fluoride in water supply causes brain damage in children",
    "BANNED VIDEO: The cure for all disease they don't want you to see before it gets deleted",
    "Ancient prophecy predicts world will end this year according to newly decoded Bible code",
    "Election FRAUD: Millions of ballots found hidden in warehouse, mainstream media silent",
    "COVER-UP: Thousands die from vaccine side effects but health authorities hiding the truth",
    "Secret society controls global banking system and engineering worldwide economic crashes",
    "PROOF: Reptilian aliens disguised as politicians are running world governments",
    "Big tech is reading your thoughts using hidden sensors in smartphones, leaked docs show",
    "MUST SHARE: This natural remedy kills coronavirus instantly but government banned it",
    "Insider leaks: Social media companies using AI to secretly manipulate your political beliefs",
    "EXPOSED: Famous charity actually front organization for massive child trafficking network",
    "The real reason they want you to stay home: Government installing surveillance in every home",
    "Hollywood elite caught holding secret rituals to summon demons for power and wealth",
    "BOMBSHELL LEAKED DOCUMENTS PROVE GLOBAL CONSPIRACY YOU MUST SEE THIS NOW",
    "Scientists SHOCKED to discover what happens when you eat this common food DAILY",
    "URGENT: They are coming for your freedom and mainstream media is hiding it from you",
    "EXPOSED: What REALLY happened that they don't want you to know about",
    "Share before it gets deleted: The truth about what is really in your water",
    "This celebrity CONFIRMS what we have been saying all along, watch before banned",
    "BREAKING EXCLUSIVE: Secret insider reveals globalist plan for world domination",
    "Doctors are FURIOUS about this home remedy that actually works",
]

REAL_SAMPLES = [
    "Federal Reserve raises interest rates by 0.25 percentage points amid ongoing inflation concerns.",
    "Researchers at MIT publish new study on renewable energy storage solutions in Nature journal.",
    "Local city council votes 7-2 to approve new public transportation expansion budget.",
    "NASA's James Webb Space Telescope captures detailed images of distant galaxy formation.",
    "Annual unemployment report shows 4.2% jobless rate, consistent with previous quarter figures.",
    "University study finds moderate exercise linked to reduced risk of cardiovascular disease.",
    "International climate summit concludes with agreement on carbon emission reduction targets.",
    "Pharmaceutical company receives FDA approval for new blood pressure medication after trials.",
    "City announces new affordable housing project to add 500 units over the next three years.",
    "Scientists develop more efficient solar panel technology with 28% energy conversion rate.",
    "Stock markets show mixed results as investors weigh inflation data and earnings reports.",
    "New archaeological discovery in Egypt reveals previously unknown pharaonic tomb near Luxor.",
    "Governor signs education reform bill to increase teacher salaries and improve classroom funding.",
    "WHO publishes updated guidelines on nutrition and physical activity for adults over 40.",
    "Tech company announces quarterly earnings beat expectations, revenue up 12% year-over-year.",
    "Supreme Court hears arguments on privacy rights in digital age case involving data collection.",
    "Agricultural department releases report on crop yield projections for the coming harvest season.",
    "Olympic committee announces host city for 2032 summer games after competitive bidding process.",
    "Central bank governor testifies before parliament on monetary policy and economic outlook.",
    "Environmental agency publishes annual air quality index report showing improvement in major cities.",
    "Health department issues advisory on seasonal flu vaccination ahead of winter months.",
    "Police department releases annual crime statistics showing decline in property crimes citywide.",
    "University researchers find promising results in early trials of new Alzheimer's treatment approach.",
    "Trade agreement negotiations between two nations enter final phase after months of diplomacy.",
    "Public health officials urge residents to get tested as seasonal respiratory illness rates rise.",
    "City's public library system launches digital expansion offering access to 50,000 e-books.",
    "National weather service issues winter storm watch for northern regions through the weekend.",
    "Election commission certifies results after completing mandatory recount in disputed district.",
    "Central hospital reports successful implementation of new electronic health records system.",
    "State legislature passes bipartisan infrastructure bill to repair aging roads and bridges.",
    "Federal officials announce infrastructure investment plan details",
    "Scientists publish new research findings on climate modeling",
    "Local government approves budget for public services expansion",
    "Health officials update guidance on seasonal illness prevention",
    "Economic report shows modest growth in manufacturing sector",
    "University study examines long-term effects of remote work trends",
    "City transit authority announces service improvements for next year",
    "Research team develops improved water purification method",
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[!?]{2,}', ' EXCLAIM ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource(show_spinner="Training model on startup…")
def load_model():
    X = [clean_text(t) for t in FAKE_SAMPLES + REAL_SAMPLES]
    y = [1] * len(FAKE_SAMPLES) + [0] * len(REAL_SAMPLES)

    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=10000,
                            sublinear_tf=True, min_df=1)
    lr = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
        voting='soft', weights=[3, 2, 2]
    )
    pipeline = Pipeline([('tfidf', tfidf), ('clf', ensemble)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return pipeline, acc, auc, X_test, y_test


def analyze_text(model, text: str) -> dict:
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    real_prob, fake_prob = proba[0], proba[1]
    label = "FAKE" if fake_prob > 0.5 else "REAL"

    if fake_prob >= 0.80:   risk = "🔴 Very High"
    elif fake_prob >= 0.60: risk = "🟠 High"
    elif fake_prob >= 0.40: risk = "🟡 Medium"
    elif fake_prob >= 0.20: risk = "🟢 Low"
    else:                   risk = "✅ Very Low"

    flags = []
    text_upper = text.upper()
    allcaps = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
    if allcaps >= 2:
        flags.append(f"⚠️ {allcaps} ALL-CAPS words (sensationalism signal)")
    if text.count('!') >= 2:
        flags.append(f"⚠️ {text.count('!')} exclamation marks (emotional manipulation)")
    triggers = ["BREAKING","SHOCKING","EXCLUSIVE","SECRET","EXPOSED","BANNED",
                "BOMBSHELL","URGENT","COVER-UP","PROOF","MUST SHARE","WAKE UP",
                "THEY DON'T WANT","YOU WON'T BELIEVE"]
    found = [w for w in triggers if w in text_upper]
    if found:
        flags.append(f"⚠️ Trigger words: {', '.join(found[:4])}")
    credibility_signals = ["peer-reviewed","published","announced","study","report",
                           "FDA","CDC","WHO","university","institute","journal","research"]
    cred_found = [w for w in credibility_signals if w in text.lower()]
    if cred_found:
        flags.append(f"✅ Credibility signals: {', '.join(cred_found[:3])}")
    if len(text.split()) < 8:
        flags.append("ℹ️ Very short text — limited context")

    return {
        "label": label,
        "fake_prob": round(fake_prob * 100, 1),
        "real_prob": round(real_prob * 100, 1),
        "confidence": round(max(fake_prob, real_prob) * 100, 1),
        "credibility": round(real_prob * 100, 1),
        "risk": risk,
        "flags": flags,
    }


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0e1117; }
  [data-testid="stHeader"] { background: transparent; }
  .main-title {
    font-family: 'Courier New', monospace;
    font-size: 2rem; font-weight: 700; color: #e0e0ff;
    text-align: center; letter-spacing: -1px; margin-bottom: .2rem;
  }
  .subtitle { text-align:center; color:#888; font-size:.9rem; margin-bottom:1.5rem; }
  .tag {
    display:inline-block; font-size:.65rem; font-family:monospace;
    padding:2px 8px; border-radius:3px; margin:2px;
  }
  .verdict-fake {
    background:#ff3b3022; border:2px solid #ff3b30; border-radius:12px;
    padding:1.2rem 1.5rem; text-align:center; margin:1rem 0;
  }
  .verdict-real {
    background:#30d15822; border:2px solid #30d158; border-radius:12px;
    padding:1.2rem 1.5rem; text-align:center; margin:1rem 0;
  }
  .verdict-label-fake { font-size:2rem; font-weight:800; color:#ff3b30; font-family:monospace; }
  .verdict-label-real { font-size:2rem; font-weight:800; color:#30d158; font-family:monospace; }
  .metric-box {
    background:#161b27; border:1px solid #2a3040; border-radius:10px;
    padding:.9rem 1rem; text-align:center;
  }
  .metric-label { font-size:.7rem; color:#666; font-family:monospace; text-transform:uppercase; }
  .metric-value { font-size:1.6rem; font-weight:700; font-family:monospace; }
  .flag-item { padding:.35rem .6rem; border-radius:6px; font-size:.82rem;
               background:#1a1f2e; margin:.25rem 0; }
  stTextArea textarea { font-family: monospace !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 FAKE_NEWS.DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered misinformation analysis · TF-IDF + Ensemble ML</div>',
            unsafe_allow_html=True)
st.markdown("""<div style="text-align:center;margin-bottom:1.5rem">
  <span class="tag" style="background:#1a1a3e;color:#7c83fd">TF-IDF</span>
  <span class="tag" style="background:#0d2137;color:#5ecbff">Ensemble</span>
  <span class="tag" style="background:#1a2e1a;color:#5dbe5d">NLP</span>
  <span class="tag" style="background:#1a1a3e;color:#a5a9ff">Logistic Regression</span>
  <span class="tag" style="background:#1a2e1a;color:#5dbe5d">Random Forest</span>
  <span class="tag" style="background:#2e1a1a;color:#ff9f7c">Gradient Boosting</span>
</div>""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────────────────
model, train_acc, train_auc, X_test, y_test = load_model()

with st.expander("📊 Model Performance", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{train_acc*100:.1f}%")
    c2.metric("ROC-AUC", f"{train_auc:.3f}")
    c3.metric("Training samples", str(len(FAKE_SAMPLES) + len(REAL_SAMPLES)))
    st.caption("Trained on balanced fake/real news corpus using 80/20 train-test split.")

st.divider()

# ── Examples ─────────────────────────────────────────────────────────────────
st.markdown("**Try an example:**")
examples = {
    "🚨 Fake — conspiracy": "BREAKING: Scientists PROVE 5G towers cause cancer — government hiding the truth from millions!",
    "📰 Real — finance":    "The Federal Reserve raised interest rates by 0.25 percentage points amid ongoing inflation concerns.",
    "💊 Fake — health hoax":"Doctors HATE this: Man cures diabetes in 7 days with this one weird kitchen spice trick!",
    "🔬 Real — research":   "A peer-reviewed study published in Nature found moderate exercise reduces cardiovascular disease risk.",
    "🛸 Fake — conspiracy": "PROOF: Reptilian aliens disguised as politicians are running world governments, insider reveals.",
}
cols = st.columns(len(examples))
chosen_example = ""
for i, (label, text) in enumerate(examples.items()):
    if cols[i].button(label, use_container_width=True):
        chosen_example = text

# ── Input Area ───────────────────────────────────────────────────────────────
default_text = chosen_example if chosen_example else ""
user_input = st.text_area(
    "Paste a news headline or article excerpt:",
    value=default_text,
    height=120,
    placeholder="Enter text to analyze...",
    label_visibility="collapsed",
)
st.caption(f"{len(user_input)} characters · {len(user_input.split())} words")

analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ── Analysis ─────────────────────────────────────────────────────────────────
if analyze_clicked or chosen_example:
    text_to_analyze = user_input.strip() or chosen_example
    if not text_to_analyze:
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running ensemble analysis…"):
            result = analyze_text(model, text_to_analyze)

        # Verdict banner
        if result["label"] == "FAKE":
            st.markdown(f"""<div class="verdict-fake">
              <div class="verdict-label-fake">⛔ LIKELY FAKE</div>
              <div style="color:#ff6b6b;margin-top:.3rem;font-size:.9rem">
                {result['confidence']}% confidence · Risk: {result['risk']}
              </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="verdict-real">
              <div class="verdict-label-real">✅ LIKELY REAL</div>
              <div style="color:#5dff8b;margin-top:.3rem;font-size:.9rem">
                {result['confidence']}% confidence · Risk: {result['risk']}
              </div></div>""", unsafe_allow_html=True)

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class="metric-box">
          <div class="metric-label">Fake Probability</div>
          <div class="metric-value" style="color:#ff3b30">{result['fake_prob']}%</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-box">
          <div class="metric-label">Real Probability</div>
          <div class="metric-value" style="color:#30d158">{result['real_prob']}%</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-box">
          <div class="metric-label">Credibility Score</div>
          <div class="metric-value" style="color:#7c83fd">{result['credibility']}<span style="font-size:.9rem">/100</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Progress bars
        st.markdown("**Probability breakdown:**")
        st.progress(result['fake_prob'] / 100,
                    text=f"Fake: {result['fake_prob']}%")
        st.progress(result['real_prob'] / 100,
                    text=f"Real: {result['real_prob']}%")
        st.progress(result['credibility'] / 100,
                    text=f"Credibility: {result['credibility']}/100")

        # Flags
        if result["flags"]:
            st.markdown("**Linguistic signals detected:**")
            for flag in result["flags"]:
                st.markdown(f'<div class="flag-item">{flag}</div>',
                            unsafe_allow_html=True)

        # How it works
        with st.expander("⚙️ How the model works", expanded=False):
            st.markdown("""
**Architecture:** 3-model soft-voting ensemble
- **Logistic Regression** (weight 3) — best for sparse TF-IDF features
- **Random Forest** (weight 2) — captures non-linear patterns  
- **Gradient Boosting** (weight 2) — sequential error correction

**Feature extraction:** TF-IDF with 1–3 word n-grams, 10,000 features, sublinear TF scaling

**Red flag signals:** ALL-CAPS frequency, excessive punctuation, trigger words
(BREAKING, BOMBSHELL, EXPOSED, SHOCKING, etc.)

**Credibility signals:** peer-reviewed, published, WHO, FDA, university, journal, %
            """)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
This app uses a **machine learning ensemble** to detect potential misinformation.

**Models used:**
- Logistic Regression  
- Random Forest  
- Gradient Boosting

**Features:** TF-IDF n-grams (1–3), linguistic patterns, trigger words

⚠️ *This is a demonstration tool. Always verify news from multiple trusted sources.*
    """)
    st.divider()
    st.markdown("Built with Python · scikit-learn · Streamlit")
