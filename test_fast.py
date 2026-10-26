"""
Quick test of all fast modules (no BART download needed).
Run this to verify the installation worked.

Usage:
    python test_fast.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("=" * 50)
print("  FAST MODULE TEST (no BART needed)")
print("=" * 50)

# Test 1: utils
print("\n[1] Testing utils.py ...")
from utils import load_sample_articles, clean_text, validate_text
articles = load_sample_articles(
    os.path.join(os.path.dirname(__file__), "data", "sample_articles.txt")
)
print(f"    Loaded {len(articles)} sample articles: OK")

# Test 2: extractive
print("\n[2] Testing extractive.py ...")
from extractive import extractive_summarize, get_summary_sentence_count
n = get_summary_sentence_count(articles[0])
summary = extractive_summarize(articles[0], num_sentences=n)
print(f"    Extractive summary ({len(summary.split())} words): OK")
print(f"    Preview: {summary[:100]}...")

# Test 3: evaluate
print("\n[3] Testing evaluate.py ...")
from evaluate import calculate_rouge, print_rouge_comparison_table
ref = "Scientists discovered a new deep-sea fish species in the Mariana Trench."
ext_scores = calculate_rouge(summary, ref)
fake_abs = "A new fish species was found in the Pacific Ocean at extreme depths."
abs_scores = calculate_rouge(fake_abs, ref)
r1 = ext_scores["rouge1"]["f1"]
r2 = ext_scores["rouge2"]["f1"]
rl = ext_scores["rougeL"]["f1"]
print(f"    ROUGE-1:{r1:.4f}  ROUGE-2:{r2:.4f}  ROUGE-L:{rl:.4f}: OK")

print_rouge_comparison_table(ext_scores, abs_scores)

print("\n" + "=" * 50)
print("  ALL FAST TESTS PASSED")
print("  Next step: python src/summarizer.py")
print("  (downloads ~1.6GB BART model, takes time)")
print("=" * 50)
