"""
Hugging Face Spaces entry point.
Gradio web UI — runs BART fully for free on HF infrastructure.

Deploy:
  1. Create a new Space at https://huggingface.co/spaces
  2. SDK: Gradio
  3. Upload this entire hf_space/ folder as the Space repo
  4. HF will install requirements and run app.py automatically
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import re
from collections import defaultdict
import nltk

# ── Download NLTK data once ──────────────────────────────────
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# ── Constants ────────────────────────────────────────────────
MODEL_NAME  = "facebook/bart-large-cnn"
MAX_TOKENS  = 1024
CHUNK_SIZE  = 900

# ── Lazy model cache ─────────────────────────────────────────
_tokenizer = None
_model     = None

def load_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    print("Loading BART model…")
    _tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    _model     = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    _model.eval()
    print("BART ready.")
    return _tokenizer, _model

# ── Extractive summarizer ────────────────────────────────────
def extractive_summarize(text: str, num_sentences: int = 3) -> str:
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 3]
    if not sentences:
        return text[:300]
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = {"the","a","an","is","are","was","were","in","on","at","to","of","and","or","it","its"}

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    freq  = defaultdict(int)
    for w in words:
        if w not in stop_words and len(w) > 2:
            freq[w] += 1
    if not freq:
        return " ".join(sentences[:num_sentences])

    max_f = max(freq.values())
    freq  = {w: c/max_f for w, c in freq.items()}

    scores = {}
    for i, sent in enumerate(sentences):
        ws = re.findall(r'\b[a-zA-Z]+\b', sent.lower())
        s  = sum(freq.get(w, 0) for w in ws)
        if ws:
            scores[i] = s / len(ws)

    top = sorted(scores, key=lambda i: scores[i], reverse=True)[:num_sentences]
    top = sorted(top)
    return " ".join(sentences[i] for i in top)

# ── Chunked BART summarizer ──────────────────────────────────
def bart_summarize(text: str, min_len: int = 56, max_len: int = 142) -> str:
    tokenizer, model = load_model()

    tokens = tokenizer.encode(text)
    if len(tokens) <= MAX_TOKENS:
        inputs = tokenizer(text, return_tensors="pt",
                           max_length=MAX_TOKENS, truncation=True)
        with torch.no_grad():
            ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4, min_length=min_len, max_length=max_len,
                length_penalty=2.0, early_stopping=True
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    # Chunking
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur, cur_tok = [], [], 0
    for s in sentences:
        st = len(tokenizer.encode(s))
        if cur_tok + st > CHUNK_SIZE and cur:
            chunks.append(" ".join(cur))
            cur, cur_tok = [s], st
        else:
            cur.append(s)
            cur_tok += st
    if cur:
        chunks.append(" ".join(cur))

    summaries = []
    for chunk in chunks:
        inp = tokenizer(chunk, return_tensors="pt",
                        max_length=MAX_TOKENS, truncation=True)
        with torch.no_grad():
            ids = model.generate(
                inp["input_ids"], inp["attention_mask"],
                num_beams=4, min_length=20, max_length=max_len // max(1, len(chunks)),
                length_penalty=2.0, early_stopping=True
            )
        summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    combined = " ".join(summaries)
    if len(tokenizer.encode(combined)) <= MAX_TOKENS:
        inp = tokenizer(combined, return_tensors="pt",
                        max_length=MAX_TOKENS, truncation=True)
        with torch.no_grad():
            ids = model.generate(
                inp["input_ids"], inp["attention_mask"],
                num_beams=4, min_length=min_len, max_length=max_len,
                length_penalty=2.0, early_stopping=True
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True)
    return combined

# ── ROUGE calculation ────────────────────────────────────────
def calc_rouge(hypothesis: str, reference: str) -> str:
    if not hypothesis.strip() or not reference.strip():
        return ""
    sc = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    s  = sc.score(reference, hypothesis)
    lines = []
    for m in ["rouge1","rouge2","rougeL"]:
        label = m.upper().replace("ROUGE","ROUGE-")
        lines.append(
            f"{label}  →  P: {s[m].precision:.4f}  "
            f"R: {s[m].recall:.4f}  F1: {s[m].fmeasure:.4f}"
        )
    return "\n".join(lines)

# ── Main inference function (called by Gradio) ───────────────
def run_summarization(article, reference, min_len, max_len, num_ext):
    article = article.strip()
    if not article:
        return "⚠️ Please enter an article.", "", "", ""
    if len(article.split()) < 30:
        return "⚠️ Article too short. Please enter at least 30 words.", "", "", ""

    # Extractive
    ext = extractive_summarize(article, num_sentences=int(num_ext))

    # BART abstractive
    try:
        abs_ = bart_summarize(article, min_len=int(min_len), max_len=int(max_len))
    except Exception as e:
        abs_ = f"Error generating summary: {e}"

    # Stats
    stats = (
        f"Original: {len(article.split())} words  |  "
        f"Extractive: {len(ext.split())} words  |  "
        f"Abstractive: {len(abs_.split())} words"
    )

    # ROUGE
    rouge_out = ""
    if reference.strip():
        ext_r  = calc_rouge(ext,  reference)
        abs_r  = calc_rouge(abs_, reference)
        rouge_out = (
            "── Extractive vs Reference ──\n" + ext_r +
            "\n\n── BART Abstractive vs Reference ──\n" + abs_r
        )

    return ext, abs_, stats, rouge_out

# ── Sample articles ──────────────────────────────────────────
SAMPLES = [
    ("Deep-Sea Fish",
     "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean "
     "that can survive at extreme depths of more than 8,000 meters. The fish, named "
     "Pseudoliparis swirei, was found in the Mariana Trench, the deepest part of the "
     "ocean. Researchers from the University of Washington conducted the study using a "
     "specially designed underwater vehicle equipped with cameras and collection systems. "
     "The fish has several unique adaptations that allow it to survive under tremendous "
     "water pressure, including a flexible skeleton, large amounts of a special chemical "
     "called TMAO, and a highly efficient metabolism. The discovery sheds new light on "
     "the limits of vertebrate life. The team published their findings in the journal "
     "Nature Ecology and Evolution."),
    ("Electric Vehicles",
     "The global electric vehicle market has grown significantly over the past five years. "
     "In 2023, more than 10 million electric vehicles were sold worldwide, representing a "
     "35 percent increase compared to the previous year. China remains the dominant market, "
     "accounting for approximately 60 percent of all EV sales globally. Major automakers "
     "including Tesla, BYD, Volkswagen, and General Motors have all committed to "
     "transitioning their fleets to electric power by 2035. Battery technology continues "
     "to improve, with new solid-state batteries promising greater energy density and "
     "faster charging times. Analysts predict that electric vehicles will account for "
     "more than 50 percent of all new car sales worldwide by 2030."),
    ("AI in Healthcare",
     "Artificial intelligence is transforming the healthcare industry in ways that were "
     "unimaginable just a decade ago. Machine learning algorithms can now analyze medical "
     "images such as X-rays, MRI scans, and CT scans with accuracy comparable to "
     "experienced radiologists. AI-powered diagnostic tools are being deployed in hospitals "
     "across the United States, Europe, and Asia to help doctors detect diseases such as "
     "cancer, diabetic retinopathy, and cardiovascular conditions at earlier stages. "
     "Natural language processing models are being used to extract information from "
     "electronic health records. Drug discovery is another area where AI is making a "
     "significant impact, with companies using machine learning to identify potential "
     "drug candidates. Despite these advances, concerns remain about data privacy, "
     "algorithmic bias, and the need to maintain the human element in medical care."),
]

# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(
    title="AI Text Summarization",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css="""
    .header { text-align:center; padding:20px 0 10px; }
    .header h1 { font-size:2rem; margin-bottom:4px; }
    .header p  { color:#6b7280; font-size:0.95rem; }
    .result-box textarea { font-size:0.92rem !important; line-height:1.7 !important; }
    """
) as demo:

    gr.HTML("""
    <div class="header">
      <h1>🤖 AI Text Summarization System</h1>
      <p>BTech AI &amp; DS Mini Project &nbsp;·&nbsp;
         Model: <strong>facebook/bart-large-cnn</strong> &nbsp;·&nbsp;
         Evaluation: ROUGE-1 · ROUGE-2 · ROUGE-L</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")

            example_dd = gr.Dropdown(
                choices=["— paste your own —"] + [s[0] for s in SAMPLES],
                value="— paste your own —",
                label="Load sample article"
            )
            article_box = gr.Textbox(
                lines=12, label="Article text",
                placeholder="Paste a long news article here…"
            )
            reference_box = gr.Textbox(
                lines=3, label="Reference summary (optional — enables ROUGE scores)",
                placeholder="Paste a human-written reference summary to calculate ROUGE…"
            )

            with gr.Row():
                min_slider = gr.Slider(20, 100, value=56, step=1, label="Min summary tokens")
                max_slider = gr.Slider(80, 300, value=142, step=1, label="Max summary tokens")
                ext_slider = gr.Slider(1, 7,   value=3,   step=1, label="Extractive sentences")

            run_btn = gr.Button("🚀 Generate Summary", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")
            stats_box  = gr.Textbox(label="Word counts", interactive=False, lines=1)
            ext_out    = gr.Textbox(label="🔍 Extractive Summary",        lines=5,
                                    elem_classes=["result-box"])
            abs_out    = gr.Textbox(label="🤖 BART Abstractive Summary",  lines=5,
                                    elem_classes=["result-box"])
            rouge_out  = gr.Textbox(label="📈 ROUGE Scores",              lines=7,
                                    interactive=False)

    # Load sample
    def load_sample(name):
        for s in SAMPLES:
            if s[0] == name:
                return s[1]
        return ""

    example_dd.change(fn=load_sample, inputs=example_dd, outputs=article_box)

    run_btn.click(
        fn=run_summarization,
        inputs=[article_box, reference_box, min_slider, max_slider, ext_slider],
        outputs=[ext_out, abs_out, stats_box, rouge_out]
    )

    gr.Markdown("""
---
**How it works:**
- **Extractive** — picks the top-N most important sentences using word-frequency scoring
- **BART** — reads the article with its Encoder, then generates new summary sentences with its Decoder
- **ROUGE** — measures word overlap between generated summary and human reference (needs a reference to calculate)

*Note: First run downloads the BART model (~1.6 GB). Subsequent runs use the HF cache.*
    """)

if __name__ == "__main__":
    demo.launch()
