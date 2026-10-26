"""
web/app.py  —  Flask Web Application
Text Summarization Using Transformer Models
BTech AI & DS Mini Project

Run from project root:
    python web/app.py
Then open browser: http://localhost:5000
"""

import sys
import os
import json
import traceback

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd

from utils import clean_text, validate_text, load_sample_articles
from extractive import extractive_summarize, get_summary_sentence_count
from evaluate import calculate_rouge

# ─────────────────────────────────────────────────────────────
# FLASK APP SETUP
# ─────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# ─────────────────────────────────────────────────────────────
# LAZY MODEL LOADER  (load BART once, reuse for every request)
# ─────────────────────────────────────────────────────────────

_bart_loaded = False
_bart_error  = None

def get_bart():
    """
    Loads BART the first time it is called, then returns
    the cached tokenizer + model for every subsequent call.
    Thread-safe enough for a local demo server.
    """
    global _bart_loaded, _bart_error
    if _bart_loaded:
        return True, None
    try:
        from summarizer import load_model
        load_model(verbose=True)
        _bart_loaded = True
        return True, None
    except Exception as e:
        _bart_error = str(e)
        return False, _bart_error


# ─────────────────────────────────────────────────────────────
# ROUTES — PAGES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main summarizer page."""
    samples = load_sample_articles(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_articles.txt")
    )
    return render_template("index.html",
                           samples=samples,
                           bart_loaded=_bart_loaded)


@app.route("/compare")
def compare():
    """Compare Extractive vs Abstractive page."""
    samples = load_sample_articles(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_articles.txt")
    )
    return render_template("compare.html",
                           samples=samples,
                           bart_loaded=_bart_loaded)


@app.route("/dataset")
def dataset():
    """CNN/DailyMail dataset evaluation page."""
    return render_template("dataset.html", bart_loaded=_bart_loaded)


@app.route("/about")
def about():
    """How it works / about page."""
    return render_template("about.html", bart_loaded=_bart_loaded)


# ─────────────────────────────────────────────────────────────
# ROUTES — API ENDPOINTS  (called by JavaScript fetch())
# ─────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """Returns whether BART model is loaded."""
    return jsonify({"bart_loaded": _bart_loaded, "error": _bart_error})


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    """Triggers BART model loading (called from browser)."""
    ok, err = get_bart()
    return jsonify({"success": ok, "error": err})


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """
    Main summarization endpoint.

    Expects JSON body:
    {
        "text":       "...",
        "reference":  "...",   (optional)
        "min_len":    56,
        "max_len":    142,
        "num_ext":    3
    }

    Returns JSON:
    {
        "extractive":       "...",
        "abstractive":      "...",
        "rouge":            {...},   (only if reference provided)
        "word_counts":      {...},
        "elapsed":          1.23
    }
    """
    import time

    data = request.get_json(force=True)
    text      = data.get("text", "").strip()
    reference = data.get("reference", "").strip()
    min_len   = int(data.get("min_len", 56))
    max_len   = int(data.get("max_len", 142))
    num_ext   = int(data.get("num_ext", 3))

    # Validate
    cleaned = clean_text(text)
    ok, msg = validate_text(cleaned, min_words=30)
    if not ok:
        return jsonify({"error": msg}), 400

    try:
        # Extractive (always fast)
        ext_sum = extractive_summarize(cleaned, num_sentences=num_ext)

        # Abstractive — load BART if needed
        ok, err = get_bart()
        if not ok:
            return jsonify({"error": f"BART model not loaded: {err}"}), 503

        from summarizer import summarize_long_text
        t0 = time.time()
        abs_sum = summarize_long_text(
            cleaned,
            min_length=min_len,
            max_length=max_len,
            verbose=False
        )
        elapsed = round(time.time() - t0, 1)

        result = {
            "extractive":  ext_sum,
            "abstractive": abs_sum,
            "elapsed":     elapsed,
            "word_counts": {
                "original":    len(cleaned.split()),
                "extractive":  len(ext_sum.split()),
                "abstractive": len(abs_sum.split()),
            }
        }

        # ROUGE if reference provided
        if reference:
            ext_rouge = calculate_rouge(ext_sum, reference)
            abs_rouge = calculate_rouge(abs_sum,  reference)
            result["rouge"] = {
                "extractive":  ext_rouge,
                "abstractive": abs_rouge
            }

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/extractive_only", methods=["POST"])
def api_extractive_only():
    """
    Extractive summarization only — no BART needed.
    Used for instant feedback while BART loads.
    """
    data    = request.get_json(force=True)
    text    = data.get("text", "").strip()
    num_ext = int(data.get("num_ext", 3))

    cleaned = clean_text(text)
    ok, msg = validate_text(cleaned, min_words=20)
    if not ok:
        return jsonify({"error": msg}), 400

    try:
        ext_sum = extractive_summarize(cleaned, num_sentences=num_ext)
        return jsonify({
            "extractive":  ext_sum,
            "word_counts": {
                "original":   len(cleaned.split()),
                "extractive": len(ext_sum.split()),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rouge_only", methods=["POST"])
def api_rouge_only():
    """
    Calculate ROUGE between two texts.
    Used on the Compare page's quick ROUGE demo.
    """
    data      = request.get_json(force=True)
    generated = data.get("generated", "").strip()
    reference = data.get("reference", "").strip()

    if not generated or not reference:
        return jsonify({"error": "Both generated and reference are required."}), 400

    try:
        scores = calculate_rouge(generated, reference)
        return jsonify(scores)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/samples")
def api_samples():
    """Returns the list of sample articles."""
    samples = load_sample_articles(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_articles.txt")
    )
    return jsonify({"samples": samples})


@app.route("/api/dataset_sample", methods=["POST"])
def api_dataset_sample():
    """
    Loads N articles from CNN/DailyMail, runs both summarizers,
    calculates ROUGE, and streams results back as JSON.
    """
    import time
    data       = request.get_json(force=True)
    num        = max(1, min(int(data.get("num_samples", 3)), 20))
    min_len    = int(data.get("min_len", 56))
    max_len    = int(data.get("max_len", 142))

    ok, err = get_bart()
    if not ok:
        return jsonify({"error": f"BART not loaded: {err}"}), 503

    try:
        from datasets import load_dataset
        from summarizer import summarize_long_text

        dataset = load_dataset("cnn_dailymail", "3.0.0",
                               split="test", trust_remote_code=True)
        subset  = dataset.select(range(num))

        results = []
        for i, item in enumerate(subset):
            article   = item.get("article", "")
            reference = item.get("highlights", "")
            if len(article.split()) < 50:
                continue

            ext_sum = extractive_summarize(
                article, num_sentences=get_summary_sentence_count(article)
            )
            abs_sum = summarize_long_text(
                article, min_length=min_len, max_length=max_len, verbose=False
            )

            ext_rouge = calculate_rouge(ext_sum, reference)
            abs_rouge = calculate_rouge(abs_sum, reference)

            results.append({
                "id":          i + 1,
                "article":     article[:300] + "…",
                "reference":   reference[:200] + "…",
                "extractive":  ext_sum,
                "abstractive": abs_sum,
                "ext_rouge1":  ext_rouge["rouge1"]["f1"],
                "ext_rouge2":  ext_rouge["rouge2"]["f1"],
                "ext_rougeL":  ext_rouge["rougeL"]["f1"],
                "abs_rouge1":  abs_rouge["rouge1"]["f1"],
                "abs_rouge2":  abs_rouge["rouge2"]["f1"],
                "abs_rougeL":  abs_rouge["rougeL"]["f1"],
            })

        if not results:
            return jsonify({"error": "No valid articles found."}), 500

        # Averages
        avg = lambda key: round(sum(r[key] for r in results) / len(results), 4)
        averages = {
            "ext_rouge1": avg("ext_rouge1"), "ext_rouge2": avg("ext_rouge2"),
            "ext_rougeL": avg("ext_rougeL"), "abs_rouge1": avg("abs_rouge1"),
            "abs_rouge2": avg("abs_rouge2"), "abs_rougeL": avg("abs_rougeL"),
        }

        return jsonify({"results": results, "averages": averages})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  TEXT SUMMARIZATION WEB APP")
    print("  BTech AI & DS Mini Project")
    print("="*55)
    print("  Open browser: http://localhost:5000")
    print("  Press Ctrl+C to stop the server")
    print("="*55 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,   # Keep False — debug=True reloads and breaks model cache
        threaded=True
    )
