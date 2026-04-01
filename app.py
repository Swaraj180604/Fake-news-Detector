"""
Fake News Detector - Flask API Server
Run: python3 app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from model import analyze_text, build_dataset, build_model, save_model

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fake_news_model.pkl")

def get_model():
    if not os.path.exists(MODEL_PATH):
        print("Training model...")
        X, y = build_dataset()
        m = build_model()
        m.fit(X, y)
        save_model(m, MODEL_PATH)
        return m
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

MODEL = get_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = analyze_text(MODEL, text)
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "FakeNewsDetector v1.0"})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
