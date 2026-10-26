# ============================================================
# FILE: src/extractive.py
# PURPOSE: Extractive Summarization using Word Frequency Method
#
# WHAT IS EXTRACTIVE SUMMARIZATION?
#   Extractive summarization selects the most important
#   sentences from the ORIGINAL document and returns them
#   as the summary. No new sentences are created.
#   It is like highlighting important lines in a textbook.
#
#   SIMPLE HINDI:
#   Extractive summarization mein hum original document ke
#   important sentences ko uthate hain aur unhe hi summary
#   ke roop mein present karte hain. Koi naya sentence nahi
#   banaya jata.
#
# HOW IT WORKS (ALGORITHM):
#   Step 1: Split the document into individual sentences.
#   Step 2: Count the frequency of each word (ignore stopwords).
#   Step 3: Score each sentence by adding up the frequencies
#           of the words it contains.
#   Step 4: Pick the top-N highest scoring sentences.
#   Step 5: Return those sentences in their ORIGINAL ORDER.
#
# WHY THIS ALGORITHM?
#   - Very easy to understand and explain in a viva.
#   - No model download required (works offline instantly).
#   - Good enough to compare against BART.
#   - Industry-standard baseline for summarization.
# ============================================================

import re
import nltk
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# NLTK DATA DOWNLOAD (runs once, then cached)
# ─────────────────────────────────────────────────────────────

def download_nltk_data():
    """
    Downloads required NLTK data files if not already present.
    This only runs once; NLTK caches them locally.
    """
    required = ['punkt', 'stopwords', 'punkt_tab']
    for item in required:
        try:
            nltk.data.find(f'tokenizers/{item}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{item}')
            except LookupError:
                print(f"  Downloading NLTK data: {item} ...")
                nltk.download(item, quiet=True)


# ─────────────────────────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> list:
    """
    Splits a paragraph/document into individual sentences.

    Uses NLTK's sentence tokenizer which handles:
      - Abbreviations (Dr., Mr., U.S.A.)
      - Decimal numbers (3.14)
      - Multiple punctuation

    SIMPLE HINDI:
    Document ko individual sentences mein todta hai.
    Ye sirf period (.) pe split nahi karta kyunki Dr. jaise
    abbreviations mein bhi period hota hai.

    Example:
    Input:  "The cat sat. Dr. Smith arrived."
    Output: ["The cat sat.", "Dr. Smith arrived."]
    """
    download_nltk_data()
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        # Fallback: simple split by period if NLTK fails
        sentences = re.split(r'(?<=[.!?])\s+', text)

    # Remove very short sentences (likely noise)
    sentences = [s.strip() for s in sentences if len(s.split()) > 3]
    return sentences


# ─────────────────────────────────────────────────────────────
# WORD FREQUENCY CALCULATION
# ─────────────────────────────────────────────────────────────

def compute_word_frequencies(text: str) -> dict:
    """
    Counts how often each meaningful word appears in the text.

    WHAT ARE STOPWORDS?
    Words like 'the', 'is', 'at', 'which', 'on' appear very
    frequently but carry no real information. We ignore them.

    SIMPLE HINDI:
    Har word ki frequency count karte hain. 'the', 'is', 'a'
    jaise common words ko ignore karte hain kyunki ye
    information nahi dete. Baaki words ki frequency count
    karke ek dictionary banate hain.

    Returns: dict {word: normalized_frequency}
    """
    download_nltk_data()

    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    except Exception:
        # Fallback basic stopwords
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
            'by', 'from', 'up', 'about', 'into', 'through', 'and',
            'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
            'neither', 'not', 'only', 'own', 'same', 'than',
            'too', 'very', 'just', 'this', 'that', 'these', 'those',
            'it', 'its', 'as', 'if'
        }

    # Convert text to lowercase and extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    # Count frequencies, skipping stopwords
    freq = defaultdict(int)
    for word in words:
        if word not in stop_words and len(word) > 2:
            freq[word] += 1

    if not freq:
        return {}

    # Normalize by dividing by the maximum frequency
    # This scales all values between 0 and 1
    max_freq = max(freq.values())
    normalized = {word: count / max_freq for word, count in freq.items()}

    return normalized


# ─────────────────────────────────────────────────────────────
# SENTENCE SCORING
# ─────────────────────────────────────────────────────────────

def score_sentences(sentences: list, word_frequencies: dict) -> dict:
    """
    Scores each sentence based on the sum of word frequencies.

    LOGIC:
    A sentence that contains many high-frequency important
    words is considered more important.

    SIMPLE HINDI:
    Har sentence ka score calculate karte hain. Jo words
    zyada important hain (high frequency), unhe contain karne
    wale sentences ko zyada score milta hai.

    Example:
    Sentence: "Scientists discovered new deep-sea fish."
    Words:    scientists(0.9) + discovered(0.7) + new(0.3) +
              deep(0.6) + sea(0.8) + fish(1.0) = 4.3
    Score:    4.3 (divided by sentence length for fairness)

    Returns: dict {sentence_index: score}
    """
    sentence_scores = {}

    for i, sentence in enumerate(sentences):
        words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        score = 0
        word_count = 0

        for word in words:
            if word in word_frequencies:
                score += word_frequencies[word]
                word_count += 1

        # Normalize by sentence length to avoid bias toward
        # very long sentences
        if word_count > 0:
            sentence_scores[i] = score / word_count

    return sentence_scores


# ─────────────────────────────────────────────────────────────
# MAIN EXTRACTIVE SUMMARIZATION FUNCTION
# ─────────────────────────────────────────────────────────────

def extractive_summarize(text: str, num_sentences: int = 3) -> str:
    """
    Main function: performs extractive summarization.

    ALGORITHM STEPS:
    1. Split text into sentences
    2. Calculate word frequencies
    3. Score each sentence
    4. Select top-N sentences
    5. Return them in original order (preserves flow)

    Args:
        text: The original article/document as a string.
        num_sentences: How many sentences to include in summary.
                       Default is 3.

    Returns:
        A string containing the extractive summary.

    SIMPLE HINDI:
    1. Text ko sentences mein todenge
    2. Word frequencies calculate karenge
    3. Har sentence ko score karenge
    4. Top-N sentences select karenge
    5. Unhe original order mein return karenge
    """
    # Validate input
    if not text or not text.strip():
        return "Error: Cannot summarize empty text."

    words = text.split()
    if len(words) < 20:
        return "Error: Text too short to summarize."

    # Step 1: Split into sentences
    sentences = split_into_sentences(text)

    if len(sentences) == 0:
        return "Error: Could not split text into sentences."

    # If text has very few sentences, return all of them
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Step 2: Compute word frequencies
    word_frequencies = compute_word_frequencies(text)

    if not word_frequencies:
        # Fallback: return first N sentences
        return " ".join(sentences[:num_sentences])

    # Step 3: Score sentences
    sentence_scores = score_sentences(sentences, word_frequencies)

    if not sentence_scores:
        return " ".join(sentences[:num_sentences])

    # Step 4: Select top-N sentences by score
    # sorted() returns [(index, score), ...] sorted by score descending
    top_indices = sorted(
        sentence_scores,
        key=lambda i: sentence_scores[i],
        reverse=True
    )[:num_sentences]

    # Step 5: Sort selected indices to preserve original order
    top_indices = sorted(top_indices)

    # Build the summary
    summary = " ".join(sentences[i] for i in top_indices)
    return summary


# ─────────────────────────────────────────────────────────────
# UTILITY: AUTO-DETERMINE NUMBER OF SENTENCES
# ─────────────────────────────────────────────────────────────

def get_summary_sentence_count(text: str, ratio: float = 0.3) -> int:
    """
    Automatically determines how many sentences to include.
    Uses a ratio of the total sentence count.

    Default ratio = 0.3 means keep 30% of sentences.
    Minimum is 2, Maximum is 5 for typical articles.

    SIMPLE HINDI:
    Article kitna lamba hai uske hisaab se automatically
    decide karta hai kitne sentences summary mein rakhne hain.
    30% sentences rakhta hai, minimum 2, maximum 5.
    """
    sentences = split_into_sentences(text)
    n = max(2, min(5, int(len(sentences) * ratio)))
    return n


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test extractive summarization:
    cd src
    python extractive.py
    """
    print("=" * 60)
    print("  EXTRACTIVE SUMMARIZATION TEST")
    print("=" * 60)

    test_article = """
    Artificial intelligence is transforming the healthcare industry
    in ways that were unimaginable just a decade ago. Machine
    learning algorithms can now analyze medical images such as
    X-rays, MRI scans, and CT scans with accuracy comparable to
    or exceeding that of experienced radiologists. AI-powered
    diagnostic tools are being deployed in hospitals across the
    United States, Europe, and Asia to help doctors detect diseases
    such as cancer, diabetic retinopathy, and cardiovascular
    conditions at earlier stages. Natural language processing models
    are being used to extract information from electronic health
    records, enabling researchers to identify patterns in patient
    data that can inform treatment decisions. Drug discovery is
    another area where AI is making a significant impact, with
    companies using machine learning to identify potential drug
    candidates and predict how they will interact with biological
    targets. Despite these advances, concerns remain about data
    privacy, algorithmic bias, regulatory oversight, and the need
    to maintain the human element in medical care.
    """

    n = get_summary_sentence_count(test_article)
    print(f"\nAuto-detected sentence count: {n}")

    summary = extractive_summarize(test_article, num_sentences=n)

    print("\nORIGINAL ARTICLE:")
    print(test_article.strip())

    print("\nEXTRACTIVE SUMMARY:")
    print(summary)

    print(
        f"\nOriginal word count: {len(test_article.split())}"
    )
    print(f"Summary word count:  {len(summary.split())}")
    compression = (1 - len(summary.split()) / len(test_article.split())) * 100
    print(f"Compression ratio:   {compression:.1f}%")
