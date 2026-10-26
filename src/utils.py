# ============================================================
# FILE: src/utils.py
# PURPOSE: Helper / utility functions used by all other modules.
#          Like a toolbox - common tools kept in one place.
#
# WHAT IT DOES:
#   - Cleans / preprocesses text
#   - Reads sample articles from the data/ folder
#   - Prints nicely formatted section headers
#   - Saves results to outputs/
# ============================================================

import os
import re
import pandas as pd


# ─────────────────────────────────────────────────────────────
# SECTION HEADER PRINTER
# ─────────────────────────────────────────────────────────────

def print_header(title: str):
    """
    Prints a formatted header to make terminal output readable.
    Example output:
    ============================================================
    TEXT SUMMARIZATION USING TRANSFORMERS
    ============================================================
    """
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def print_section(title: str):
    """
    Prints a smaller section divider.
    """
    print(f"\n--- {title} ---")


# ─────────────────────────────────────────────────────────────
# TEXT CLEANING / PREPROCESSING
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lightly cleans the input text.

    WHY LIGHT CLEANING?
    BART is a pretrained Transformer that was trained on natural
    English text with normal punctuation, capitalization, and
    sentence structure. If we aggressively remove punctuation
    or stop-words, we damage the input and BART performs worse.

    So we only:
      1. Remove extra whitespace (multiple spaces → one space)
      2. Remove leading/trailing spaces
      3. Handle Windows-style line endings (\r\n → \n)

    SIMPLE HINDI EXPLANATION:
    Hum text ko halka saaf karte hain - extra spaces aur line
    endings hatate hain. Punctuation nahi hatate kyunki BART
    ko natural English chahiye.
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert Windows line endings to Unix
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Replace multiple whitespace/newlines with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text


def validate_text(text: str, min_words: int = 20) -> tuple:
    """
    Checks if the input text is valid for summarization.

    Returns: (is_valid: bool, message: str)

    SIMPLE HINDI EXPLANATION:
    Ye function check karta hai ki text summarize karne layak
    hai ya nahi. Agar text bahut chota hai, toh summarization
    ka matlab nahi banta.
    """
    if not text or not text.strip():
        return False, "Error: Input text is empty. Please provide some text."

    words = text.split()
    if len(words) < min_words:
        return False, (
            f"Error: Text is too short ({len(words)} words). "
            f"Please provide at least {min_words} words for meaningful summarization."
        )

    return True, "OK"


# ─────────────────────────────────────────────────────────────
# FILE READING
# ─────────────────────────────────────────────────────────────

def load_sample_articles(filepath: str = None) -> list:
    """
    Reads the sample_articles.txt file from the data/ folder.
    Each article is separated by ===ARTICLE_N=== markers.

    Returns a list of article strings.
    """
    if filepath is None:
        # Build path relative to this file's location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "sample_articles.txt")

    if not os.path.exists(filepath):
        print(f"Warning: Sample articles file not found at {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by the article markers
    # Pattern: ===ARTICLE_N===
    articles = re.split(r'===ARTICLE_\d+===', content)

    # Remove empty strings and clean each article
    articles = [clean_text(a) for a in articles if a.strip()]

    return articles


# ─────────────────────────────────────────────────────────────
# RESULTS SAVING
# ─────────────────────────────────────────────────────────────

def save_results_to_csv(results: list, filepath: str = None):
    """
    Saves experiment results to a CSV file.

    'results' should be a list of dictionaries, where each
    dictionary represents one article's results.

    SIMPLE HINDI EXPLANATION:
    Ye function experiment ke results ek CSV file mein save
    karta hai. CSV file Excel mein open ki ja sakti hai.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "outputs", "results.csv")

    if not results:
        print("Warning: No results to save.")
        return

    df = pd.DataFrame(results)

    # Make sure outputs directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"\nResults saved to: {filepath}")
    return filepath


def save_summaries_to_txt(summaries: list, filepath: str = None):
    """
    Saves generated summaries to a text file for easy review.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "outputs", "summaries.txt")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for i, item in enumerate(summaries, 1):
            f.write(f"{'='*60}\n")
            f.write(f"ARTICLE {i}\n")
            f.write(f"{'='*60}\n\n")

            if "article" in item:
                f.write(f"ORIGINAL ARTICLE:\n{item['article']}\n\n")
            if "extractive" in item:
                f.write(f"EXTRACTIVE SUMMARY:\n{item['extractive']}\n\n")
            if "abstractive" in item:
                f.write(f"ABSTRACTIVE SUMMARY (BART):\n{item['abstractive']}\n\n")
            if "reference" in item:
                f.write(f"REFERENCE SUMMARY:\n{item['reference']}\n\n")
            f.write("\n")

    print(f"Summaries saved to: {filepath}")


# ─────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────

def display_summary_comparison(article: str,
                                 extractive_summary: str,
                                 abstractive_summary: str,
                                 reference_summary: str = None):
    """
    Nicely prints a side-by-side view of all summaries.
    """
    print_section("ORIGINAL ARTICLE (first 300 chars)")
    print(article[:300] + "..." if len(article) > 300 else article)

    print_section("EXTRACTIVE SUMMARY")
    print(extractive_summary)

    print_section("ABSTRACTIVE SUMMARY (BART)")
    print(abstractive_summary)

    if reference_summary:
        print_section("REFERENCE SUMMARY (Human Written)")
        print(reference_summary)


def display_rouge_scores(scores: dict, method_name: str = ""):
    """
    Prints ROUGE scores in a readable table format.
    """
    label = f"ROUGE Scores — {method_name}" if method_name else "ROUGE Scores"
    print_section(label)
    print(f"  {'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*45}")

    for metric in ["rouge1", "rouge2", "rougeL"]:
        if metric in scores:
            s = scores[metric]
            print(
                f"  {metric.upper():<12} "
                f"{s['precision']:>10.4f} "
                f"{s['recall']:>10.4f} "
                f"{s['f1']:>10.4f}"
            )
