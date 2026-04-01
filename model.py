"""
Fake News Detector - ML Model
Uses TF-IDF + Ensemble of Logistic Regression, Random Forest, and Gradient Boosting
"""

import numpy as np
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder


# ── Synthetic Training Data ──────────────────────────────────────────────────
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
]

FAKE_HEADLINES = [
    "BOMBSHELL LEAKED DOCUMENTS PROVE GLOBAL CONSPIRACY YOU MUST SEE THIS NOW",
    "Scientists SHOCKED to discover what happens when you eat this common food DAILY",
    "URGENT: They are coming for your freedom and mainstream media is hiding it from you",
    "EXPOSED: What REALLY happened that they don't want you to know about",
    "Share before it gets deleted: The truth about what is really in your water",
    "This celebrity CONFIRMS what we have been saying all along, watch before banned",
    "BREAKING EXCLUSIVE: Secret insider reveals globalist plan for world domination",
    "Doctors are FURIOUS about this home remedy that actually works",
]

REAL_HEADLINES = [
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
    """Preprocess text for ML model."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'\d+', ' NUM ', text)              # replace numbers
    text = re.sub(r'[!?]{2,}', ' EXCLAIM ', text)    # multiple punctuation
    text = re.sub(r'[A-Z]{3,}', lambda m: m.group().lower() + ' ALLCAPS ', text)  # all-caps words
    text = re.sub(r'[^\w\s]', ' ', text)              # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_dataset():
    """Combine samples into X (texts) and y (labels)."""
    all_fake = FAKE_SAMPLES + FAKE_HEADLINES
    all_real = REAL_SAMPLES + REAL_HEADLINES

    X = [clean_text(t) for t in all_fake + all_real]
    y = [1] * len(all_fake) + [0] * len(all_real)   # 1 = fake, 0 = real

    return np.array(X), np.array(y)


def build_model():
    """Build an ensemble voting classifier pipeline."""
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        sublinear_tf=True,
        min_df=1,
        analyzer='word',
    )

    lr = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[3, 2, 2]        # LR weighted higher for text tasks
    )

    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', ensemble),
    ])
    return pipeline


def train_and_evaluate(model, X, y):
    """Train model and print evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training ensemble model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  Model Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Real', 'Fake'])}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"              Predicted Real  Predicted Fake")
    print(f"  Actual Real      {cm[0,0]:4d}            {cm[0,1]:4d}")
    print(f"  Actual Fake      {cm[1,0]:4d}            {cm[1,1]:4d}")
    print(f"{'='*50}\n")

    return acc, auc


def save_model(model, path="fake_news_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path="fake_news_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_text(model, text: str) -> dict:
    """Return detailed prediction results for input text."""
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    real_prob, fake_prob = proba[0], proba[1]
    label = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(fake_prob, real_prob)

    # Credibility score (0–100, higher = more credible)
    credibility = round(real_prob * 100, 1)

    # Risk level
    if fake_prob >= 0.80:
        risk = "Very High"
    elif fake_prob >= 0.60:
        risk = "High"
    elif fake_prob >= 0.40:
        risk = "Medium"
    elif fake_prob >= 0.20:
        risk = "Low"
    else:
        risk = "Very Low"

    # Red-flag linguistic signals
    flags = []
    text_upper = text.upper()
    if sum(1 for w in text.split() if w.isupper() and len(w) > 2) >= 2:
        flags.append("Multiple ALL-CAPS words detected")
    if text.count('!') >= 2 or text.count('?') >= 2:
        flags.append("Excessive punctuation (!! or ??)")
    trigger_words = ["BREAKING", "SHOCKING", "EXCLUSIVE", "SECRET", "EXPOSED",
                     "THEY DON'T WANT", "BANNED", "MUST SHARE", "BOMBSHELL",
                     "URGENT", "COVER-UP", "PROOF", "WAKE UP"]
    found = [w for w in trigger_words if w in text_upper]
    if found:
        flags.append(f"Sensationalist trigger words: {', '.join(found[:3])}")
    if len(text.split()) < 8:
        flags.append("Very short text – limited context for analysis")

    return {
        "label": label,
        "fake_probability": round(fake_prob * 100, 1),
        "real_probability": round(real_prob * 100, 1),
        "confidence": round(confidence * 100, 1),
        "credibility_score": credibility,
        "risk_level": risk,
        "red_flags": flags,
    }


if __name__ == "__main__":
    X, y = build_dataset()
    model = build_model()
    train_and_evaluate(model, X, y)
    save_model(model)

    # Quick demo predictions
    test_cases = [
        "BREAKING: Scientists PROVE 5G causes cancer, government hiding the truth from millions!",
        "The Federal Reserve announced a 0.25% interest rate hike in response to inflation data.",
        "You won't BELIEVE what this ancient remedy does to diabetes – doctors hate this secret!",
        "Researchers published a peer-reviewed study on vaccine efficacy in the New England Journal.",
    ]

    print("\nDemo Predictions:")
    print("=" * 60)
    for text in test_cases:
        result = analyze_text(model, text)
        print(f"\nText: {text[:70]}...")
        print(f"  → {result['label']} | Fake: {result['fake_probability']}% | "
              f"Credibility: {result['credibility_score']}/100 | Risk: {result['risk_level']}")
        if result['red_flags']:
            print(f"  ⚠ Flags: {'; '.join(result['red_flags'])}")
