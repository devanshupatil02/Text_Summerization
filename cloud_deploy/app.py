"""
cloud_deploy/app.py
====================
Lightweight cloud version — NO BART (too large for free tiers).

Features:
  - Extractive summarization  (instant, no model needed)
  - ROUGE-1, ROUGE-2, ROUGE-L evaluation
  - Beautiful web UI (same design as local version)
  - Works on Render / Railway free tier (< 100 MB total)

Deploy to Render:
  1. Push this cloud_deploy/ folder to a GitHub repo
  2. Create Web Service on render.com
  3. Build command:  pip install -r requirements.txt
  4. Start command:  gunicorn app:app
  5. Done — free public URL

Deploy to Railway:
  1. Push to GitHub
  2. New Project → Deploy from GitHub Repo
  3. Add variable: PORT=5000
  4. Done — free public URL
"""

import os, re, sys
from collections import defaultdict
from flask import Flask, render_template, request, jsonify

# ── NLTK setup ───────────────────────────────────────────────
import nltk
_nltk_data = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(_nltk_data, exist_ok=True)
nltk.data.path.insert(0, _nltk_data)
for _pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        try:
            nltk.data.find(f"corpora/{_pkg}")
        except LookupError:
            nltk.download(_pkg, download_dir=_nltk_data, quiet=True)

# ── ROUGE ────────────────────────────────────────────────────
from rouge_score import rouge_scorer as _rs

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Samples ──────────────────────────────────────────────────
SAMPLES = [
    ("Deep-Sea Fish Discovery",
     "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean "
     "that can survive at extreme depths of more than 8,000 meters. The fish, named "
     "Pseudoliparis swirei, was found in the Mariana Trench, the deepest part of the "
     "ocean. Researchers from the University of Washington conducted the study using a "
     "specially designed underwater vehicle. The fish has several unique adaptations "
     "including a flexible skeleton and large amounts of a special chemical called TMAO. "
     "The discovery sheds new light on the limits of vertebrate life and suggests that "
     "even the deepest ocean environments are inhabited by complex organisms. The team "
     "published their findings in the journal Nature Ecology and Evolution."),
    ("Electric Vehicles Market",
     "The global electric vehicle market has grown significantly over the past five years. "
     "In 2023, more than 10 million electric vehicles were sold worldwide, representing a "
     "35 percent increase compared to the previous year. China remains the dominant market, "
     "accounting for approximately 60 percent of all EV sales globally. Major automakers "
     "including Tesla, BYD, Volkswagen, and General Motors have all committed to "
     "transitioning their fleets to electric power by 2035. Battery technology continues "
     "to improve with new solid-state batteries promising greater energy density. "
     "Analysts predict that electric vehicles will account for more than 50 percent of "
     "all new car sales worldwide by 2030."),
    ("AI in Healthcare",
     "Artificial intelligence is transforming the healthcare industry in ways that were "
     "unimaginable just a decade ago. Machine learning algorithms can now analyze medical "
     "images such as X-rays, MRI scans, and CT scans with accuracy comparable to "
     "experienced radiologists. AI-powered diagnostic tools are being deployed in hospitals "
     "across the United States, Europe, and Asia to help doctors detect diseases such as "
     "cancer, diabetic retinopathy, and cardiovascular conditions at earlier stages. "
     "Natural language processing models are being used to extract information from "
     "electronic health records. Drug discovery is another area where AI is making a "
     "significant impact. Despite these advances, concerns remain about data privacy, "
     "algorithmic bias, and the need to maintain the human element in medical care."),
]

# ── Extractive summarizer ────────────────────────────────────
def extractive_summarize(text: str, num_sentences: int = 3) -> str:
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 3]
    if not sentences:
        return text[:400]
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = {"the","a","an","is","are","was","were","in","on","at","to","of","and"}
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    freq  = defaultdict(int)
    for w in words:
        if w not in stop_words and len(w) > 2:
            freq[w] += 1
    if not freq:
        return " ".join(sentences[:num_sentences])
    max_f = max(freq.values())
    freq  = {w: c / max_f for w, c in freq.items()}
    scores = {}
    for i, sent in enumerate(sentences):
        ws = re.findall(r'\b[a-zA-Z]+\b', sent.lower())
        s  = sum(freq.get(w, 0) for w in ws)
        if ws:
            scores[i] = s / len(ws)
    top = sorted(scores, key=lambda i: scores[i], reverse=True)[:num_sentences]
    top = sorted(top)
    return " ".join(sentences[i] for i in top)

# ── ROUGE ────────────────────────────────────────────────────
def calc_rouge(hypothesis: str, reference: str) -> dict:
    if not hypothesis.strip() or not reference.strip():
        return {}
    sc = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s  = sc.score(reference, hypothesis)
    return {
        m: {"precision": round(s[m].precision, 4),
            "recall":    round(s[m].recall,    4),
            "f1":        round(s[m].fmeasure,  4)}
        for m in ["rouge1", "rouge2", "rougeL"]
    }

# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", samples=SAMPLES)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    data      = request.get_json(force=True)
    text      = data.get("text", "").strip()
    reference = data.get("reference", "").strip()
    num_ext   = max(1, min(int(data.get("num_ext", 3)), 7))

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text.split()) < 20:
        return jsonify({"error": "Text too short. Need at least 20 words."}), 400

    ext = extractive_summarize(text, num_sentences=num_ext)
    result = {
        "extractive":  ext,
        "word_counts": {
            "original":   len(text.split()),
            "extractive": len(ext.split()),
        }
    }
    if reference:
        result["rouge"] = calc_rouge(ext, reference)
    return jsonify(result)

@app.route("/api/rouge_only", methods=["POST"])
def api_rouge_only():
    data = request.get_json(force=True)
    gen  = data.get("generated", "").strip()
    ref  = data.get("reference", "").strip()
    if not gen or not ref:
        return jsonify({"error": "Both fields required."}), 400
    return jsonify(calc_rouge(gen, ref))

@app.route("/api/samples")
def api_samples():
    return jsonify([{"title": s[0], "text": s[1]} for s in SAMPLES])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
