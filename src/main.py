# ============================================================
# FILE: src/main.py
# PURPOSE: Main entry point for the project.
#          Provides a text menu to access all features.
#
# HOW TO RUN:
#   From project root folder:
#     python src/main.py
#
# MENU OPTIONS:
#   1. Enter your own article → generates summaries + ROUGE
#   2. Test on CNN/DailyMail dataset → runs full experiment
#   3. Preview dataset → quick look at dataset (no BART)
#   4. Run ROUGE evaluation demo → tests ROUGE module
#   5. Exit
# ============================================================

import os
import sys

# Add the src directory to Python's path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from summarizer import summarize_long_text, load_model
from extractive import extractive_summarize, get_summary_sentence_count
from evaluate import calculate_rouge, print_rouge_comparison_table
from utils import (
    print_header, print_section,
    clean_text, validate_text,
    load_sample_articles,
    display_summary_comparison,
    display_rouge_scores,
    save_results_to_csv,
    save_summaries_to_txt
)


# ─────────────────────────────────────────────────────────────
# OPTION 1: USER'S OWN ARTICLE
# ─────────────────────────────────────────────────────────────

def run_custom_article():
    """
    Allows the user to paste their own article and get summaries.

    Workflow:
    1. Ask user to paste article.
    2. Validate it.
    3. Generate extractive summary.
    4. Generate BART abstractive summary.
    5. Optionally calculate ROUGE (if user provides reference).
    6. Display results.
    """
    print_header("CUSTOM ARTICLE SUMMARIZATION")

    print("Paste your article below.")
    print("When done, press ENTER twice (blank line to finish):\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)

    raw_text = "\n".join(lines).strip()

    if not raw_text:
        # Load a sample article if user didn't enter anything
        print("No input provided. Loading sample article...")
        sample_articles = load_sample_articles()
        if sample_articles:
            raw_text = sample_articles[0]
            print("Using sample article 1.")
        else:
            print("Error: No sample articles found either.")
            return

    # Clean and validate
    text = clean_text(raw_text)
    is_valid, msg = validate_text(text, min_words=30)
    if not is_valid:
        print(f"\n{msg}")
        return

    print(f"\nArticle word count: {len(text.split())} words")

    # ─── EXTRACTIVE SUMMARY ───
    print_section("Generating Extractive Summary...")
    n = get_summary_sentence_count(text)
    extractive_summary = extractive_summarize(text, num_sentences=n)
    print("Done.")

    # ─── BART ABSTRACTIVE SUMMARY ───
    print_section("Loading BART model and generating abstractive summary...")
    print("This may take 30-120 seconds on CPU. Please wait...\n")
    abstractive_summary = summarize_long_text(text, verbose=True)
    print("Done.")

    # ─── DISPLAY RESULTS ───
    display_summary_comparison(text, extractive_summary, abstractive_summary)

    # ─── OPTIONAL ROUGE ───
    print_section("ROUGE Evaluation (Optional)")
    print("Do you have a reference/gold summary to compare against? (y/n): ", end="")
    try:
        choice = input().strip().lower()
    except EOFError:
        choice = "n"

    if choice == "y":
        print("Paste reference summary (press ENTER when done):")
        try:
            reference = input().strip()
        except EOFError:
            reference = ""

        if reference:
            ext_scores = calculate_rouge(extractive_summary, reference)
            abs_scores = calculate_rouge(abstractive_summary, reference)

            display_rouge_scores(ext_scores, "Extractive vs Reference")
            display_rouge_scores(abs_scores, "BART Abstractive vs Reference")
            print_rouge_comparison_table(ext_scores, abs_scores)

    # ─── SAVE ───
    print_section("Saving Results")
    summaries = [{
        "article":     text[:500],
        "extractive":  extractive_summary,
        "abstractive": abstractive_summary
    }]
    save_summaries_to_txt(summaries)
    print("Results saved to outputs/summaries.txt")


# ─────────────────────────────────────────────────────────────
# OPTION 2: CNN/DAILYMAIL DATASET EXPERIMENT
# ─────────────────────────────────────────────────────────────

def run_dataset_evaluation():
    """
    Runs the full experiment on CNN/DailyMail dataset.
    Prompts user for number of samples.
    """
    print_header("CNN/DAILYMAIL DATASET EVALUATION")

    print("How many articles to evaluate?")
    print("  (Recommended: 5-10 for quick test, 20-50 for full experiment)")
    print("  Note: Each article takes ~60 seconds on CPU")
    print(f"  Enter number (press ENTER for default=10): ", end="")

    try:
        user_input = input().strip()
        num = int(user_input) if user_input else 10
        num = max(1, min(num, 100))  # Clamp between 1 and 100
    except (ValueError, EOFError):
        num = 10

    print(f"\nRunning experiment with {num} articles...")

    # Import here to avoid slow import at startup
    from dataset_test import run_dataset_experiment
    results = run_dataset_experiment(num_samples=num)

    if results:
        print("\nExperiment complete!")
        print("Results saved to: outputs/results.csv")
        print("Summaries saved to: outputs/summaries.txt")


# ─────────────────────────────────────────────────────────────
# OPTION 3: SAMPLE ARTICLES (NO INTERNET NEEDED)
# ─────────────────────────────────────────────────────────────

def run_sample_articles():
    """
    Uses the pre-loaded sample articles in data/sample_articles.txt.
    Good for a quick demo when internet is unavailable.
    Does not require dataset download.
    """
    print_header("SAMPLE ARTICLES DEMONSTRATION")

    articles = load_sample_articles()
    if not articles:
        print("Error: No sample articles found in data/sample_articles.txt")
        return

    print(f"Found {len(articles)} sample articles.\n")

    print("Loading BART model (please wait)...")
    load_model(verbose=True)

    results = []
    ext_scores_list = []
    abs_scores_list = []

    for i, article in enumerate(articles):
        print(f"\n{'='*60}")
        print(f"  ARTICLE {i+1}")
        print(f"{'='*60}")
        print(f"Article (first 200 chars):\n{article[:200]}...")

        # Extractive
        n = get_summary_sentence_count(article)
        ext_sum = extractive_summarize(article, num_sentences=n)

        # Abstractive
        print("\nGenerating BART summary (please wait)...")
        abs_sum = summarize_long_text(article, verbose=False)

        print_section("EXTRACTIVE SUMMARY")
        print(ext_sum)

        print_section("BART ABSTRACTIVE SUMMARY")
        print(abs_sum)

        results.append({
            "article_id":          f"sample_{i+1}",
            "reference_summary":   "",
            "abstractive_summary": abs_sum[:300],
            "extractive_summary":  ext_sum[:300],
        })

    save_summaries_to_txt([
        {"article": a, "extractive": r["extractive_summary"],
         "abstractive": r["abstractive_summary"]}
        for a, r in zip(articles, results)
    ])

    print("\nDone! Summaries saved to outputs/summaries.txt")


# ─────────────────────────────────────────────────────────────
# OPTION 4: ROUGE DEMO
# ─────────────────────────────────────────────────────────────

def run_rouge_demo():
    """
    Demonstrates ROUGE evaluation with fixed example text.
    No model loading required — runs instantly.
    Good for explaining ROUGE in a viva.
    """
    print_header("ROUGE EVALUATION DEMONSTRATION")

    reference = (
        "Scientists discovered a new species of deep-sea fish "
        "in the Mariana Trench that can survive at extreme depths."
    )

    extractive_summary = (
        "Scientists have discovered a new species of deep-sea fish "
        "in the Pacific Ocean that can survive at extreme depths of "
        "more than 8,000 meters."
    )

    abstractive_summary = (
        "Researchers have found a new fish species capable of "
        "surviving extreme ocean depths in the Mariana Trench, "
        "shedding light on deep-sea life."
    )

    print("Reference Summary:")
    print(f"  {reference}")

    print("\nExtractive Summary:")
    print(f"  {extractive_summary}")

    print("\nAbstractive Summary (BART-style):")
    print(f"  {abstractive_summary}")

    ext_scores = calculate_rouge(extractive_summary, reference)
    abs_scores = calculate_rouge(abstractive_summary, reference)

    display_rouge_scores(ext_scores, "Extractive")
    display_rouge_scores(abs_scores, "Abstractive")

    print_rouge_comparison_table(ext_scores, abs_scores)

    print("\n  EXPLANATION:")
    print("  ROUGE-1: Counts matching single words (unigrams)")
    print("  ROUGE-2: Counts matching word pairs (bigrams)")
    print("  ROUGE-L: Matches the longest common word sequence")
    print("\n  Extractive copies sentences directly, so it often")
    print("  has higher ROUGE scores. But abstractive summaries")
    print("  can be more concise and readable.")


# ─────────────────────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────────────────────

def print_menu():
    print_header("TEXT SUMMARIZATION USING TRANSFORMER MODELS")
    print("  BTech AI & DS Mini Project\n")
    print("  Choose an option:\n")
    print("  1. Summarize your own article")
    print("     (Enter any text — get extractive + BART summary)")
    print()
    print("  2. Run CNN/DailyMail dataset experiment")
    print("     (Downloads dataset, runs BART, calculates ROUGE)")
    print()
    print("  3. Demo with sample articles (no internet needed)")
    print("     (Uses pre-loaded articles from data/ folder)")
    print()
    print("  4. ROUGE evaluation demo (no model needed, fast)")
    print("     (Explains ROUGE with a quick example)")
    print()
    print("  5. Exit")
    print()


def main():
    """
    Main loop: shows menu, reads user choice, runs function.
    """
    while True:
        print_menu()

        try:
            print("  Enter choice (1-5): ", end="")
            choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if choice == "1":
            run_custom_article()

        elif choice == "2":
            run_dataset_evaluation()

        elif choice == "3":
            run_sample_articles()

        elif choice == "4":
            run_rouge_demo()

        elif choice == "5":
            print("\nThank you! Goodbye.")
            break

        else:
            print(f"\n  Invalid option '{choice}'. Please enter 1-5.\n")

        print("\n  Press ENTER to return to menu...", end="")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            break


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
