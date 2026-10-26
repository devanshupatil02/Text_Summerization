# ============================================================
# FILE: src/evaluate.py
# PURPOSE: ROUGE Evaluation Module
#
# WHAT IS ROUGE?
#   ROUGE = Recall-Oriented Understudy for Gisting Evaluation.
#   It is a standard metric for evaluating text summarization.
#   It compares the GENERATED summary with a REFERENCE summary
#   (written by a human expert) and measures overlap.
#
#   SIMPLE HINDI:
#   ROUGE ek measurement hai jo batata hai ki hamare
#   generated summary mein aur human-written reference
#   summary mein kitna overlap hai. Jyada overlap =
#   better summary.
#
# THE THREE ROUGE METRICS:
#
#   ROUGE-1: Measures overlap of UNIGRAMS (single words).
#   Example:
#     Reference: "Scientists discovered a new fish species"
#     Generated: "Researchers found a new deep-sea species"
#     Overlap:   "a", "new", "species" → ROUGE-1 captures this
#
#   ROUGE-2: Measures overlap of BIGRAMS (2-word sequences).
#   Example:
#     Reference: "new fish species"
#     Bigrams:   ("new","fish"), ("fish","species")
#     Generated: "new deep-sea species"
#     Bigrams:   ("new","deep-sea"), ("deep-sea","species")
#     Overlap:   0 here — ROUGE-2 is stricter
#
#   ROUGE-L: Based on the Longest Common Subsequence (LCS).
#   It finds the longest sequence of words that appear in
#   BOTH summaries (not necessarily consecutive).
#   Captures sentence-level structure similarity.
#
# PRECISION vs RECALL vs F1 (SIMPLE EXPLANATION):
#
#   Imagine we're looking for apples in a basket.
#   Generated summary = apples we picked
#   Reference summary = all actual apples in the basket
#
#   PRECISION: Of all the words we picked, how many were
#              actually in the reference?
#              = Correct / Total Generated
#              "Did we pick only apples, or also some oranges?"
#
#   RECALL:    Of all the words in the reference, how many
#              did we include in our summary?
#              = Correct / Total Reference
#              "Did we get all the apples, or miss some?"
#
#   F1:        Harmonic mean of Precision and Recall.
#              F1 = 2 * (P * R) / (P + R)
#              "Overall balance of precision and recall."
#
#   IN SUMMARIZATION: We usually report F1 score.
#   High Precision + High Recall = Good Summary.
#
#   SIMPLE HINDI:
#   Precision: Hamne jo kuch generate kiya, usmein se kitna
#              sahi tha? (Kitni galat cheezein nahi generate ki)
#   Recall:    Reference mein jo tha, usmein se kitna
#              hamare summary mein hai?
#   F1:        Dono ka balanced score.
# ============================================================

from rouge_score import rouge_scorer


# ─────────────────────────────────────────────────────────────
# ROUGE SCORER INITIALIZATION
# ─────────────────────────────────────────────────────────────

def get_rouge_scorer():
    """
    Creates and returns a ROUGE scorer object.
    We calculate ROUGE-1, ROUGE-2, and ROUGE-L.

    use_stemmer=True means words are reduced to their root
    before comparison.
    Example: "running" and "run" are treated as same.

    SIMPLE HINDI:
    use_stemmer=True karne se "running" aur "run" ko
    same maana jata hai, jo zyada fair comparison deta hai.
    """
    return rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )


# ─────────────────────────────────────────────────────────────
# SINGLE DOCUMENT ROUGE EVALUATION
# ─────────────────────────────────────────────────────────────

def calculate_rouge(hypothesis: str, reference: str) -> dict:
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores between
    a generated summary (hypothesis) and a reference summary.

    Args:
        hypothesis: The generated summary (from BART or extractive).
        reference:  The human-written reference summary.

    Returns:
        A dictionary of the form:
        {
          "rouge1": {"precision": 0.xx, "recall": 0.xx, "f1": 0.xx},
          "rouge2": {"precision": 0.xx, "recall": 0.xx, "f1": 0.xx},
          "rougeL": {"precision": 0.xx, "recall": 0.xx, "f1": 0.xx}
        }

    SIMPLE HINDI:
    Generated summary aur reference summary ke beech ROUGE
    scores calculate karta hai. Output ek dictionary hai
    jismein har metric ke liye precision, recall aur f1 hai.
    """
    if not hypothesis or not hypothesis.strip():
        return _empty_rouge_scores()
    if not reference or not reference.strip():
        return _empty_rouge_scores()

    scorer = get_rouge_scorer()
    scores = scorer.score(reference, hypothesis)

    # Convert to plain dictionary (from named tuple)
    result = {}
    for metric, score in scores.items():
        result[metric] = {
            "precision": round(score.precision, 4),
            "recall":    round(score.recall,    4),
            "f1":        round(score.fmeasure,  4)
        }

    return result


def _empty_rouge_scores() -> dict:
    """Returns zero scores when input is invalid."""
    return {
        "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    }


# ─────────────────────────────────────────────────────────────
# BATCH EVALUATION (multiple documents)
# ─────────────────────────────────────────────────────────────

def evaluate_batch(hypotheses: list, references: list) -> dict:
    """
    Evaluates a batch of generated summaries against references.
    Returns the AVERAGE ROUGE scores across all documents.

    This is the main function for dataset-level evaluation.
    You run it on N articles from CNN/DailyMail and get
    average ROUGE scores for the full experiment.

    SIMPLE HINDI:
    Multiple articles ke liye ROUGE calculate karta hai
    aur average nikaalta hai. Yahi hum report mein likhte hain.

    Args:
        hypotheses: List of generated summaries.
        references:  List of reference summaries (same order).

    Returns:
        Average ROUGE scores dictionary (same format as
        calculate_rouge, but averaged over all documents).
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Mismatch: {len(hypotheses)} summaries vs "
            f"{len(references)} references."
        )

    if not hypotheses:
        return _empty_rouge_scores()

    # Accumulate scores
    totals = {
        "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    }

    valid_count = 0

    for hyp, ref in zip(hypotheses, references):
        scores = calculate_rouge(hyp, ref)
        for metric in ["rouge1", "rouge2", "rougeL"]:
            totals[metric]["precision"] += scores[metric]["precision"]
            totals[metric]["recall"]    += scores[metric]["recall"]
            totals[metric]["f1"]        += scores[metric]["f1"]
        valid_count += 1

    if valid_count == 0:
        return _empty_rouge_scores()

    # Calculate averages
    averages = {}
    for metric in ["rouge1", "rouge2", "rougeL"]:
        averages[metric] = {
            "precision": round(totals[metric]["precision"] / valid_count, 4),
            "recall":    round(totals[metric]["recall"]    / valid_count, 4),
            "f1":        round(totals[metric]["f1"]        / valid_count, 4)
        }

    return averages


# ─────────────────────────────────────────────────────────────
# COMPARISON TABLE PRINTER
# ─────────────────────────────────────────────────────────────

def print_rouge_comparison_table(extractive_scores: dict,
                                  abstractive_scores: dict):
    """
    Prints a formatted comparison table:

    Method               | ROUGE-1  | ROUGE-2  | ROUGE-L
    ---------------------|----------|----------|--------
    Extractive           | 0.xxxx   | 0.xxxx   | 0.xxxx
    BART Abstractive     | 0.xxxx   | 0.xxxx   | 0.xxxx

    This is what you'll show in your report and viva.
    """
    print("\n")
    print("=" * 60)
    print("  ROUGE COMPARISON TABLE (F1 Scores)")
    print("=" * 60)
    print(
        f"  {'Method':<22} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}"
    )
    print(f"  {'-'*55}")

    ext_r1 = extractive_scores.get("rouge1", {}).get("f1", 0.0)
    ext_r2 = extractive_scores.get("rouge2", {}).get("f1", 0.0)
    ext_rl = extractive_scores.get("rougeL", {}).get("f1", 0.0)

    abs_r1 = abstractive_scores.get("rouge1", {}).get("f1", 0.0)
    abs_r2 = abstractive_scores.get("rouge2", {}).get("f1", 0.0)
    abs_rl = abstractive_scores.get("rougeL", {}).get("f1", 0.0)

    print(f"  {'Extractive':<22} {ext_r1:>10.4f} {ext_r2:>10.4f} {ext_rl:>10.4f}")
    print(f"  {'BART Abstractive':<22} {abs_r1:>10.4f} {abs_r2:>10.4f} {abs_rl:>10.4f}")
    print("=" * 60)

    # Interpretation
    print("\n  INTERPRETATION:")
    print("  Higher = Better overlap with human reference summary.")
    print("  ROUGE-1: Unigram overlap | ROUGE-2: Bigram overlap")
    print("  ROUGE-L: Longest Common Subsequence similarity")
    print()

    if ext_r1 > abs_r1:
        print("  Note: Extractive ROUGE-1 is typically higher because")
        print("  it copies sentences directly from the original, which")
        print("  may also appear in the reference summary.")
        print("  BART generates NEW sentences — it may score lower on")
        print("  ROUGE but produce more readable, concise summaries.")
    else:
        print("  BART achieved competitive or better ROUGE scores,")
        print("  showing strong abstractive ability.")


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run directly to test ROUGE evaluation:
    cd src
    python evaluate.py
    """
    print("=" * 60)
    print("  ROUGE EVALUATION TEST")
    print("=" * 60)

    # Example: reference vs hypothesis
    reference = (
        "Scientists discovered a new species of deep-sea fish "
        "in the Mariana Trench that can survive at extreme depths."
    )

    hypothesis_abstractive = (
        "Researchers found a new fish species in the Pacific Ocean's "
        "Mariana Trench that can survive at great depths."
    )

    hypothesis_extractive = (
        "Scientists have discovered a new species of deep-sea fish "
        "in the Pacific Ocean that can survive at extreme depths of "
        "more than 8,000 meters."
    )

    print("\nReference Summary:")
    print(reference)

    print("\nAbstractive Hypothesis:")
    print(hypothesis_abstractive)

    print("\nExtractive Hypothesis:")
    print(hypothesis_extractive)

    scores_abs = calculate_rouge(hypothesis_abstractive, reference)
    scores_ext = calculate_rouge(hypothesis_extractive, reference)

    print("\n--- Abstractive ROUGE Scores ---")
    print(f"ROUGE-1 F1: {scores_abs['rouge1']['f1']:.4f}")
    print(f"ROUGE-2 F1: {scores_abs['rouge2']['f1']:.4f}")
    print(f"ROUGE-L F1: {scores_abs['rougeL']['f1']:.4f}")

    print("\n--- Extractive ROUGE Scores ---")
    print(f"ROUGE-1 F1: {scores_ext['rouge1']['f1']:.4f}")
    print(f"ROUGE-2 F1: {scores_ext['rouge2']['f1']:.4f}")
    print(f"ROUGE-L F1: {scores_ext['rougeL']['f1']:.4f}")

    print_rouge_comparison_table(scores_ext, scores_abs)
