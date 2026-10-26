# ============================================================
# FILE: src/dataset_test.py
# PURPOSE: Load CNN/DailyMail dataset and run evaluation.
#
# CNN/DAILYMAIL DATASET OVERVIEW:
#   - One of the most widely used news summarization datasets.
#   - Contains ~300,000 news articles from CNN and DailyMail.
#   - Each article has a human-written "highlights" summary.
#   - These highlights serve as our REFERENCE summaries for
#     ROUGE evaluation.
#
# DATASET SPLITS:
#   - train:      ~287,000 articles (used to train models)
#   - validation: ~13,000 articles  (used during training)
#   - test:       ~11,000 articles  (used for final evaluation)
#
#   WE USE THE TEST SPLIT because we are evaluating, not training.
#
# FIELDS IN EACH EXAMPLE:
#   - article:    The full news article text.
#   - highlights: The human-written summary (bullet points joined).
#   - id:         Unique identifier.
#
# CONFIGURABLE SUBSET:
#   We use only a small subset (default: 20 articles) because:
#   1. Running BART on all 11,000 test articles would take hours.
#   2. Even 20 articles gives a representative result.
#   3. Your laptop/desktop likely can't run all articles in one day.
#
#   SIMPLE HINDI:
#   CNN/DailyMail dataset mein ~3 lakh news articles hain.
#   Hum sirf 20-50 articles use karenge kyunki BART slow hai.
#   Har article summarize karne mein 30-120 second lagte hain.
# ============================================================

import os
import sys
import pandas as pd

# Add parent directory to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from summarizer import summarize_long_text, load_model
from extractive import extractive_summarize, get_summary_sentence_count
from evaluate import calculate_rouge, evaluate_batch, print_rouge_comparison_table
from utils import (
    print_header, print_section,
    save_results_to_csv, save_summaries_to_txt,
    display_rouge_scores
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# How many articles to test? Change this number.
# Start with 5 for a quick test, use 20-50 for a proper experiment.
NUM_SAMPLES = 10

# Dataset version (CNN/DailyMail version 3.0.0)
DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"

# Split to use for evaluation
DATASET_SPLIT = "test"


# ─────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────

def load_cnn_dailymail(num_samples: int = NUM_SAMPLES, verbose: bool = True):
    """
    Loads a subset of the CNN/DailyMail dataset.

    WHAT HAPPENS ON FIRST RUN:
    1. Hugging Face downloads the dataset (~1 GB compressed).
    2. It is cached locally (usually in C:/Users/YOU/.cache/huggingface).
    3. Subsequent runs load from cache instantly.

    TROUBLESHOOTING:
    If download fails:
    - Check internet connection.
    - Try: pip install datasets --upgrade
    - Try disabling VPN if any.

    Returns: List of dicts with 'article', 'highlights', 'id'.

    SIMPLE HINDI:
    CNN/DailyMail dataset download karta hai aur num_samples
    articles load karta hai. Pehli baar ~1GB download hoga.
    Baad mein local cache se load hoga.
    """
    if verbose:
        print(f"  Loading CNN/DailyMail dataset ({num_samples} samples)...")
        print("  First run may download ~1GB. Please wait...")

    try:
        from datasets import load_dataset
        dataset = load_dataset(
            DATASET_NAME,
            DATASET_VERSION,
            split=DATASET_SPLIT,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"\n  ERROR loading dataset: {e}")
        print("\n  Troubleshooting:")
        print("  pip install datasets --upgrade")
        raise

    # Select a subset
    subset = dataset.select(range(min(num_samples, len(dataset))))

    articles = []
    for item in subset:
        articles.append({
            "id":         item.get("id", ""),
            "article":    item.get("article", ""),
            "highlights": item.get("highlights", "")
        })

    if verbose:
        print(f"  Loaded {len(articles)} articles from CNN/DailyMail.")

    return articles


# ─────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────

def run_dataset_experiment(num_samples: int = NUM_SAMPLES,
                            verbose: bool = True) -> list:
    """
    Complete experiment pipeline:
    For each article:
      1. Load article + reference summary.
      2. Generate BART abstractive summary.
      3. Generate extractive summary.
      4. Calculate ROUGE-1, ROUGE-2, ROUGE-L for both.
      5. Store results.
    Then:
      6. Calculate average ROUGE scores.
      7. Save results to CSV.
      8. Print comparison table.

    Returns: List of result dictionaries.
    """
    print_header("CNN/DAILYMAIL DATASET EXPERIMENT")

    # Step 1: Load model (ensures download happens before the loop)
    print("  Step 1: Loading BART model...")
    load_model(verbose=verbose)

    # Step 2: Load dataset
    print("\n  Step 2: Loading dataset...")
    articles = load_cnn_dailymail(num_samples, verbose)

    results = []
    extractive_hypotheses = []
    abstractive_hypotheses = []
    references = []

    print(f"\n  Step 3: Running evaluation on {len(articles)} articles...")
    print("  (This may take several minutes on CPU)\n")

    for i, item in enumerate(articles):
        article   = item["article"]
        reference = item["highlights"]
        art_id    = item["id"] or str(i)

        print(f"  [{i+1}/{len(articles)}] Processing article...")

        # Skip empty articles
        if not article or len(article.split()) < 50:
            print("    Skipping: article too short.")
            continue

        # Generate extractive summary
        n_sentences = get_summary_sentence_count(article)
        extractive_sum = extractive_summarize(article, num_sentences=n_sentences)

        # Generate abstractive summary (BART)
        print("    Generating BART summary (please wait)...")
        abstractive_sum = summarize_long_text(article, verbose=False)

        # Calculate ROUGE
        ext_rouge = calculate_rouge(extractive_sum, reference)
        abs_rouge = calculate_rouge(abstractive_sum, reference)

        # Store for batch averaging
        extractive_hypotheses.append(extractive_sum)
        abstractive_hypotheses.append(abstractive_sum)
        references.append(reference)

        # Build result row
        row = {
            "article_id":             art_id[:50],   # Truncate long IDs
            "reference_summary":      reference[:300],
            "abstractive_summary":    abstractive_sum[:300],
            "extractive_summary":     extractive_sum[:300],
            "abstractive_rouge1_f1":  abs_rouge["rouge1"]["f1"],
            "abstractive_rouge2_f1":  abs_rouge["rouge2"]["f1"],
            "abstractive_rougeL_f1":  abs_rouge["rougeL"]["f1"],
            "extractive_rouge1_f1":   ext_rouge["rouge1"]["f1"],
            "extractive_rouge2_f1":   ext_rouge["rouge2"]["f1"],
            "extractive_rougeL_f1":   ext_rouge["rougeL"]["f1"],
        }
        results.append(row)

        # Print progress
        print(
            f"    Abstractive ROUGE-1: {abs_rouge['rouge1']['f1']:.4f}  "
            f"ROUGE-2: {abs_rouge['rouge2']['f1']:.4f}  "
            f"ROUGE-L: {abs_rouge['rougeL']['f1']:.4f}"
        )
        print(
            f"    Extractive  ROUGE-1: {ext_rouge['rouge1']['f1']:.4f}  "
            f"ROUGE-2: {ext_rouge['rouge2']['f1']:.4f}  "
            f"ROUGE-L: {ext_rouge['rougeL']['f1']:.4f}"
        )
        print()

    if not results:
        print("  No valid results generated.")
        return []

    # ─────────────────────────────────────────────────────────
    # AVERAGE SCORES
    # ─────────────────────────────────────────────────────────

    print_section("AVERAGE ROUGE SCORES")

    avg_extractive  = evaluate_batch(extractive_hypotheses, references)
    avg_abstractive = evaluate_batch(abstractive_hypotheses, references)

    display_rouge_scores(avg_extractive,  "Extractive (Average)")
    display_rouge_scores(avg_abstractive, "BART Abstractive (Average)")

    print_rouge_comparison_table(avg_extractive, avg_abstractive)

    # ─────────────────────────────────────────────────────────
    # SAVE TO CSV
    # ─────────────────────────────────────────────────────────

    # Add average row at the bottom of the CSV
    avg_row = {
        "article_id":             "AVERAGE",
        "reference_summary":      "",
        "abstractive_summary":    "",
        "extractive_summary":     "",
        "abstractive_rouge1_f1":  avg_abstractive["rouge1"]["f1"],
        "abstractive_rouge2_f1":  avg_abstractive["rouge2"]["f1"],
        "abstractive_rougeL_f1":  avg_abstractive["rougeL"]["f1"],
        "extractive_rouge1_f1":   avg_extractive["rouge1"]["f1"],
        "extractive_rouge2_f1":   avg_extractive["rouge2"]["f1"],
        "extractive_rougeL_f1":   avg_extractive["rougeL"]["f1"],
    }
    results.append(avg_row)

    save_results_to_csv(results)

    # Save summaries to text file
    summaries_data = [
        {
            "article":     a["article"][:500],
            "reference":   a["highlights"],
            "extractive":  r["extractive_summary"],
            "abstractive": r["abstractive_summary"]
        }
        for a, r in zip(articles[:len(results)-1], results[:-1])
    ]
    save_summaries_to_txt(summaries_data)

    return results


# ─────────────────────────────────────────────────────────────
# QUICK PREVIEW (show one example without running full experiment)
# ─────────────────────────────────────────────────────────────

def preview_dataset(num_samples: int = 3):
    """
    Quickly preview a few articles from the dataset.
    Use this to check if the dataset loaded correctly.
    Does NOT run BART (fast).
    """
    print_header("CNN/DAILYMAIL DATASET PREVIEW")

    articles = load_cnn_dailymail(num_samples=num_samples)

    for i, item in enumerate(articles):
        print(f"\n--- Article {i+1} ---")
        print(f"ID: {item['id'][:40]}...")
        print(f"Article (first 200 chars): {item['article'][:200]}...")
        print(f"Reference summary: {item['highlights'][:200]}...")
        print(f"Article word count: {len(item['article'].split())}")


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run from the project root:
    python src/dataset_test.py

    Options:
    1. Preview dataset (fast, no BART)
    2. Full experiment (slow, runs BART)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="CNN/DailyMail Dataset Test"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Only preview dataset, do not run BART"
    )
    parser.add_argument(
        "--samples", type=int, default=NUM_SAMPLES,
        help=f"Number of samples to evaluate (default: {NUM_SAMPLES})"
    )
    args = parser.parse_args()

    if args.preview:
        preview_dataset(num_samples=3)
    else:
        run_dataset_experiment(num_samples=args.samples)
