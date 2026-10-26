"""
streamlit_app.py
=================
Streamlit Community Cloud deployment.
Full project: Extractive + BART Abstractive + ROUGE evaluation.

Deploy (FREE):
  1. Push this repo to GitHub
  2. Go to https://share.streamlit.io
  3. Sign in with GitHub
  4. New app → repo: devanshupatil02/Text_Summerization
               branch: main
               main file: streamlit_app.py
  5. Deploy → get free public URL

Run locally:
  pip install streamlit
  streamlit run streamlit_app.py
"""

import os, sys, re, time
from collections import defaultdict

# ── path setup ────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "src"))

import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be FIRST streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text Summarization",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark background override */
  .stApp { background-color: #0d1117; color: #e6edf3; }

  /* Result cards */
  .res-card {
    background: #161b22;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    line-height: 1.75;
    font-size: 0.93rem;
    border-left: 4px solid #58a6ff;
  }
  .res-card.ext  { border-left-color: #3fb950; }
  .res-card.abs  { border-left-color: #bc8cff; }
  .res-card.ref  { border-left-color: #d29922; }

  /* Section label */
  .sec-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #8b949e;
    margin: 14px 0 5px;
  }

  /* Chip row */
  .chip {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.78rem;
    color: #8b949e;
    margin: 2px 3px;
  }
  .chip.g { border-color: #238636; color: #3fb950; background: #12261a; }
  .chip.p { border-color: #6e40c9; color: #bc8cff; background: #1e1433; }
  .chip.b { border-color: #1f6feb; color: #58a6ff; background: #0d1f38; }

  /* ROUGE table */
  .rouge-tbl { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
  .rouge-tbl th {
    background: #21262d; color: #8b949e;
    padding: 8px 12px; text-align: left;
    font-size: 0.75rem; text-transform: uppercase;
    border-bottom: 1px solid #30363d;
  }
  .rouge-tbl td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
  .rouge-tbl tr:last-child td { border-bottom: none; }
  .rouge-tbl .big { font-size: 1.1rem; font-weight: 700; }
  .rouge-tbl .gv  { color: #3fb950; }
  .rouge-tbl .pv  { color: #bc8cff; }

  /* Hide default streamlit elements that clash */
  #MainMenu { visibility: hidden; }
  .stDeployButton { display: none; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SAMPLE ARTICLES
# ─────────────────────────────────────────────────────────────
SAMPLES = {
    "— paste your own —": "",
    "Sample 1: Deep-Sea Fish Discovery": (
        "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean "
        "that can survive at extreme depths of more than 8,000 meters. The fish, named "
        "Pseudoliparis swirei, was found in the Mariana Trench, the deepest part of the "
        "ocean. Researchers from the University of Washington conducted the study using a "
        "specially designed underwater vehicle equipped with cameras and collection systems. "
        "The fish has several unique adaptations that allow it to survive under tremendous "
        "water pressure, including a flexible skeleton, large amounts of a special chemical "
        "called TMAO, and a highly efficient metabolism. The discovery sheds new light on "
        "the limits of vertebrate life and suggests that even the deepest ocean environments "
        "are inhabited by complex organisms. The team published their findings in the journal "
        "Nature Ecology and Evolution. This is considered one of the most significant marine "
        "biology discoveries of the decade."
    ),
    "Sample 2: Electric Vehicles Market": (
        "The global electric vehicle market has grown significantly over the past five years. "
        "In 2023, more than 10 million electric vehicles were sold worldwide, representing a "
        "35 percent increase compared to the previous year. China remains the dominant market, "
        "accounting for approximately 60 percent of all EV sales globally. Europe and the "
        "United States are also showing strong growth. Major automakers including Tesla, BYD, "
        "Volkswagen, and General Motors have all committed to transitioning their fleets to "
        "electric power by 2035. Battery technology continues to improve, with new solid-state "
        "batteries promising greater energy density, faster charging times, and longer "
        "lifespans. Governments around the world are offering tax incentives and subsidies to "
        "encourage consumers to adopt electric vehicles. Analysts predict that electric "
        "vehicles will account for more than 50 percent of all new car sales worldwide by 2030."
    ),
    "Sample 3: AI in Healthcare": (
        "Artificial intelligence is transforming the healthcare industry in ways that were "
        "unimaginable just a decade ago. Machine learning algorithms can now analyze medical "
        "images such as X-rays, MRI scans, and CT scans with accuracy comparable to "
        "experienced radiologists. AI-powered diagnostic tools are being deployed in hospitals "
        "across the United States, Europe, and Asia to help doctors detect diseases such as "
        "cancer, diabetic retinopathy, and cardiovascular conditions at earlier stages. "
        "Natural language processing models are being used to extract information from "
        "electronic health records, enabling researchers to identify patterns in patient data. "
        "Drug discovery is another area where AI is making a significant impact, with "
        "companies using machine learning to identify potential drug candidates and predict "
        "how they will interact with biological targets. Despite these advances, concerns "
        "remain about data privacy, algorithmic bias, regulatory oversight, and the need to "
        "maintain the human element in medical care."
    ),
}

# ─────────────────────────────────────────────────────────────
# CACHED IMPORTS (heavy libraries loaded once per session)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading BART model… (first run ~2 min, then instant)")
def load_bart():
    """Loads BART tokenizer + model once and caches for the session."""
    import torch
    from transformers import BartForConditionalGeneration, BartTokenizer
    tok = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    mdl = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    mdl.eval()
    return tok, mdl


@st.cache_resource(show_spinner=False)
def get_nltk():
    """Downloads NLTK data once."""
    import nltk
    for item in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{item}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{item}")
            except LookupError:
                nltk.download(item, quiet=True)
    return True


# ─────────────────────────────────────────────────────────────
# EXTRACTIVE SUMMARIZER
# ─────────────────────────────────────────────────────────────

def extractive_summarize(text: str, num_sentences: int = 3) -> str:
    get_nltk()
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
        stop_words = {"the","a","an","is","are","was","were","in","on",
                      "at","to","of","and","or","it","its","this","that"}

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


# ─────────────────────────────────────────────────────────────
# BART ABSTRACTIVE SUMMARIZER
# ─────────────────────────────────────────────────────────────

def bart_summarize(text: str, min_len: int = 56, max_len: int = 142) -> str:
    import torch
    tokenizer, model = load_bart()

    token_count = len(tokenizer.encode(text))

    if token_count <= 1024:
        inputs = tokenizer(
            text, return_tensors="pt",
            max_length=1024, truncation=True
        )
        with torch.no_grad():
            ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4, min_length=min_len, max_length=max_len,
                length_penalty=2.0, early_stopping=True
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    # Chunking for long documents
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur, cur_tok = [], [], 0
    for s in sentences:
        st = len(tokenizer.encode(s))
        if cur_tok + st > 900 and cur:
            chunks.append(" ".join(cur))
            cur, cur_tok = [s], st
        else:
            cur.append(s); cur_tok += st
    if cur:
        chunks.append(" ".join(cur))

    summaries = []
    for chunk in chunks:
        inp = tokenizer(chunk, return_tensors="pt",
                        max_length=1024, truncation=True)
        with torch.no_grad():
            ids = model.generate(
                inp["input_ids"], inp["attention_mask"],
                num_beams=4, min_length=20,
                max_length=max(40, max_len // max(1, len(chunks))),
                length_penalty=2.0, early_stopping=True
            )
        summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    combined = " ".join(summaries)
    if len(tokenizer.encode(combined)) <= 1024:
        inp = tokenizer(combined, return_tensors="pt",
                        max_length=1024, truncation=True)
        with torch.no_grad():
            ids = model.generate(
                inp["input_ids"], inp["attention_mask"],
                num_beams=4, min_length=min_len, max_length=max_len,
                length_penalty=2.0, early_stopping=True
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True)
    return combined


# ─────────────────────────────────────────────────────────────
# ROUGE EVALUATION
# ─────────────────────────────────────────────────────────────

def calc_rouge(hypothesis: str, reference: str) -> dict:
    from rouge_score import rouge_scorer
    if not hypothesis.strip() or not reference.strip():
        return {}
    sc = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    s = sc.score(reference, hypothesis)
    return {
        m: {
            "precision": round(s[m].precision, 4),
            "recall":    round(s[m].recall,    4),
            "f1":        round(s[m].fmeasure,  4),
        }
        for m in ["rouge1", "rouge2", "rougeL"]
    }


# ─────────────────────────────────────────────────────────────
# ROUGE CHART (matplotlib, no Chart.js needed)
# ─────────────────────────────────────────────────────────────

def rouge_chart(ext_scores: dict, abs_scores: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    keys    = ["rouge1",  "rouge2",  "rougeL"]
    ext_v   = [ext_scores[k]["f1"] for k in keys]
    abs_v   = [abs_scores[k]["f1"] for k in keys]

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    b1 = ax.bar(x - w/2, ext_v, w, label="Extractive",     color="#238636", alpha=0.88)
    b2 = ax.bar(x + w/2, abs_v, w, label="BART Abstractive", color="#6e40c9", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="white", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#161b22")
    ax.legend(facecolor="#21262d", labelcolor="white", fontsize=9, framealpha=1)
    ax.set_title("ROUGE Score Comparison (F1)", color="white", fontsize=12, pad=10)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", color="#3fb950", fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", color="#bc8cff", fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 AI Text Summarizer")
    st.markdown("**BTech AI & DS Mini Project**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Summarize", "⚖️ Compare", "📚 How It Works"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    min_len = st.slider("Min summary tokens", 20, 100, 56)
    max_len = st.slider("Max summary tokens", 80, 300, 142)
    num_ext = st.slider("Extractive sentences", 1, 7, 3)

    st.markdown("---")
    st.markdown("### 📡 Model Info")
    st.markdown("`facebook/bart-large-cnn`")
    st.markdown("406M parameters · 1024 token limit")

    st.markdown("---")
    st.caption(
        "GitHub: [devanshupatil02/Text_Summerization]"
        "(https://github.com/devanshupatil02/Text_Summerization)"
    )


# ─────────────────────────────────────────────────────────────
# PAGE 1 — SUMMARIZE
# ─────────────────────────────────────────────────────────────

def page_summarize():
    st.markdown("# 📄 Text Summarization System")
    st.markdown(
        "Paste any article → get **Extractive** + **BART Abstractive** summary "
        "with optional **ROUGE** evaluation."
    )
    st.markdown("---")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("### ✍️ Input")

        sample_choice = st.selectbox("Load a sample article", list(SAMPLES.keys()))
        default_text  = SAMPLES[sample_choice]

        article = st.text_area(
            "Article text",
            value=default_text,
            height=260,
            placeholder="Paste a news article or any long text here…"
        )

        word_count = len(article.split()) if article.strip() else 0
        if word_count > 0:
            color = "green" if word_count >= 30 else "orange"
            st.markdown(
                f'<span style="font-size:.8rem;color:var(--{color});">'
                f'{word_count} words</span>',
                unsafe_allow_html=True
            )

        reference = st.text_area(
            "📌 Reference summary (optional — enables ROUGE scores)",
            height=90,
            placeholder="Paste a human-written reference summary to calculate ROUGE…"
        )

        generate = st.button(
            "🚀 Generate Summary",
            type="primary",
            use_container_width=True
        )

    with col_out:
        st.markdown("### 📊 Results")

        if not generate:
            st.info("Paste an article and click **Generate Summary** to see results here.")
            return

        if not article.strip():
            st.error("Please enter an article.")
            return
        if word_count < 30:
            st.error(f"Article too short ({word_count} words). Need at least 30 words.")
            return

        # Extractive (fast, instant)
        with st.spinner("Generating extractive summary…"):
            ext_sum = extractive_summarize(article, num_sentences=num_ext)

        # BART (slow, but cached after first run)
        with st.spinner("Running BART model… (30-120s on CPU, instant if cached)"):
            t0 = time.time()
            try:
                abs_sum = bart_summarize(article, min_len=min_len, max_len=max_len)
                elapsed = round(time.time() - t0, 1)
            except Exception as e:
                st.error(f"BART error: {e}")
                return

        # Stats
        st.markdown(
            f'<span class="chip b">📄 Original: {word_count} words</span>'
            f'<span class="chip g">🔍 Extractive: {len(ext_sum.split())} words</span>'
            f'<span class="chip p">🤖 Abstractive: {len(abs_sum.split())} words</span>'
            f'<span class="chip">⏱️ {elapsed}s</span>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # Extractive result
        st.markdown('<div class="sec-label">🔍 Extractive Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="res-card ext">{ext_sum}</div>', unsafe_allow_html=True)

        # Abstractive result
        st.markdown('<div class="sec-label">🤖 BART Abstractive Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="res-card abs">{abs_sum}</div>', unsafe_allow_html=True)

        # ROUGE scores
        if reference.strip():
            st.markdown("---")
            st.markdown("### 📈 ROUGE Evaluation")

            ext_rouge = calc_rouge(ext_sum, reference.strip())
            abs_rouge = calc_rouge(abs_sum, reference.strip())

            c1, c2, c3 = st.columns(3)
            for col, key, label in zip(
                [c1, c2, c3],
                ["rouge1", "rouge2", "rougeL"],
                ["ROUGE-1 (Unigram)", "ROUGE-2 (Bigram)", "ROUGE-L (LCS)"]
            ):
                with col:
                    st.metric(
                        label=label,
                        value=f"Ext  {ext_rouge[key]['f1']:.4f}",
                        delta=f"Abs {abs_rouge[key]['f1']:.4f}"
                    )

            fig = rouge_chart(ext_rouge, abs_rouge)
            st.pyplot(fig, use_container_width=True)
            import matplotlib.pyplot as plt
            plt.close(fig)

        # Downloads
        st.markdown("---")
        dl_text = (
            f"ORIGINAL ARTICLE\n{'='*60}\n{article}\n\n"
            f"EXTRACTIVE SUMMARY\n{'='*60}\n{ext_sum}\n\n"
            f"BART ABSTRACTIVE SUMMARY\n{'='*60}\n{abs_sum}\n"
        )
        st.download_button(
            "⬇️ Download summaries (.txt)",
            data=dl_text,
            file_name="summaries.txt",
            mime="text/plain",
            use_container_width=True
        )


# ─────────────────────────────────────────────────────────────
# PAGE 2 — COMPARE
# ─────────────────────────────────────────────────────────────

def page_compare():
    st.markdown("# ⚖️ Compare Extractive vs Abstractive")
    st.markdown("Run both methods and see a full ROUGE breakdown side by side.")
    st.markdown("---")

    sample_choice = st.selectbox("Load a sample article", list(SAMPLES.keys()))
    default_text  = SAMPLES[sample_choice]

    col1, col2 = st.columns(2, gap="large")
    with col1:
        article = st.text_area("Article", value=default_text, height=220)
    with col2:
        # Auto-fill reference from first 2 sentences of sample
        auto_ref = ""
        if default_text:
            sents = re.split(r'(?<=[.!?])\s+', default_text)
            auto_ref = " ".join(sents[:2])
        reference = st.text_area("Reference summary (required for ROUGE)", value=auto_ref, height=220)

    run = st.button("⚡ Run Comparison", type="primary", use_container_width=True)
    if not run:
        return

    if not article.strip():
        st.error("Please enter an article.")
        return
    if len(article.split()) < 30:
        st.error("Article too short.")
        return
    if not reference.strip():
        st.warning("Please add a reference summary to enable ROUGE.")
        return

    with st.spinner("Generating extractive summary…"):
        ext_sum = extractive_summarize(article, num_sentences=num_ext)

    with st.spinner("Running BART… (30-120s on CPU)"):
        t0 = time.time()
        abs_sum = bart_summarize(article, min_len=min_len, max_len=max_len)
        elapsed = round(time.time() - t0, 1)

    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 🔍 Extractive Summary")
        st.markdown(f'<div class="res-card ext">{ext_sum}</div>', unsafe_allow_html=True)
        st.caption(f"{len(ext_sum.split())} words")
    with c2:
        st.markdown("#### 🤖 BART Abstractive Summary")
        st.markdown(f'<div class="res-card abs">{abs_sum}</div>', unsafe_allow_html=True)
        st.caption(f"{len(abs_sum.split())} words · ⏱️ {elapsed}s")

    st.markdown("---")
    st.markdown("### 📊 ROUGE Scores")

    ext_rouge = calc_rouge(ext_sum, reference.strip())
    abs_rouge = calc_rouge(abs_sum, reference.strip())

    # Metric cards
    c1, c2, c3 = st.columns(3)
    for col, key, label in zip(
        [c1, c2, c3],
        ["rouge1", "rouge2", "rougeL"],
        ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    ):
        with col:
            ef = ext_rouge[key]["f1"]
            af = abs_rouge[key]["f1"]
            winner = "🟢 Ext" if ef >= af else "🟣 Abs"
            st.markdown(
                f"<div style='background:#161b22;border-radius:10px;padding:14px;text-align:center;"
                f"border:1px solid #30363d;'>"
                f"<div style='font-size:.75rem;color:#8b949e;text-transform:uppercase;"
                f"font-weight:700;margin-bottom:8px'>{label}</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:#3fb950'>{ef:.4f}</div>"
                f"<div style='font-size:.7rem;color:#8b949e'>Extractive F1</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:#bc8cff;margin-top:6px'>{af:.4f}</div>"
                f"<div style='font-size:.7rem;color:#8b949e'>Abstractive F1</div>"
                f"<div style='margin-top:8px;font-size:.75rem;font-weight:600'>{winner}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    # Chart
    fig = rouge_chart(ext_rouge, abs_rouge)
    st.pyplot(fig, use_container_width=True)
    import matplotlib.pyplot as plt
    plt.close(fig)

    # Full table
    st.markdown("### 📋 Full P / R / F1 Table")
    import pandas as pd
    rows = []
    for key, label in [("rouge1","ROUGE-1"),("rouge2","ROUGE-2"),("rougeL","ROUGE-L")]:
        rows.append({
            "Metric":       label,
            "Ext P":        ext_rouge[key]["precision"],
            "Ext R":        ext_rouge[key]["recall"],
            "Ext F1":       ext_rouge[key]["f1"],
            "Abs P":        abs_rouge[key]["precision"],
            "Abs R":        abs_rouge[key]["recall"],
            "Abs F1":       abs_rouge[key]["f1"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Interpretation
    ext_avg = sum(ext_rouge[k]["f1"] for k in ["rouge1","rouge2","rougeL"]) / 3
    abs_avg = sum(abs_rouge[k]["f1"] for k in ["rouge1","rouge2","rougeL"]) / 3
    st.info(
        f"**Average ROUGE F1** — Extractive: **{ext_avg:.4f}** vs "
        f"Abstractive: **{abs_avg:.4f}**\n\n"
        "Extractive typically scores higher on ROUGE because it copies sentences "
        "verbatim that may also appear in the reference. BART generates *new* sentences "
        "— lower ROUGE but more concise and readable. ROUGE measures word overlap, "
        "not meaning or quality."
    )

    # Download
    import io
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download ROUGE table (.csv)",
        data=csv_buf.getvalue(),
        file_name="rouge_comparison.csv",
        mime="text/csv",
        use_container_width=True
    )


# ─────────────────────────────────────────────────────────────
# PAGE 3 — HOW IT WORKS
# ─────────────────────────────────────────────────────────────

def page_how():
    st.markdown("# 📚 How It Works")
    st.markdown("Plain-English explanation of every component — useful for viva prep.")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Extractive", "🤖 BART", "📊 ROUGE", "⚙️ Architecture", "📖 Viva Q&A"
    ])

    with tab1:
        st.markdown("## Extractive Summarization")
        st.markdown(
            "Selects the most important sentences from the original document. "
            "No new sentences are created — like highlighting key lines in a textbook."
        )
        st.code("""Step 1 → Split article into sentences (NLTK sent_tokenize)
Step 2 → Count word frequencies (skip stopwords: "the", "is", "at"…)
Step 3 → Normalize: divide each count by max count (range 0–1)
Step 4 → Score each sentence = sum(word_freq) ÷ sentence_length
Step 5 → Pick top-N sentences by score
Step 6 → Return them in ORIGINAL ORDER (preserves flow)""", language="text")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**✅ Advantages**")
            st.markdown("- Instant — no model needed\n- Always factually correct\n- Works offline\n- Easy to explain")
        with col_b:
            st.markdown("**❌ Disadvantages**")
            st.markdown("- No semantic understanding\n- Output can be choppy\n- Cannot rephrase\n- Just keyword matching")

    with tab2:
        st.markdown("## BART — Bidirectional & Auto-Regressive Transformer")
        st.markdown("Facebook AI Research, 2020. Fine-tuned on CNN/DailyMail. **406M parameters. 1024 token limit.**")
        st.code("""INPUT ARTICLE
    │
    ▼  BartTokenizer → token IDs
┌─────────────────────────────────┐
│  BART ENCODER  (12 layers)      │
│  Reads ALL tokens at once       │
│  Self-Attention: every word     │
│  attends to every other word    │
│  → Contextual Representations   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  BART DECODER  (12 layers)      │
│  Generates ONE token at a time  │
│  Cross-Attention → encoder      │
│  Self-Attention  → own tokens   │
│  Beam Search (4 beams)          │
│  Stops at <EOS> or max_length   │
└──────────────┬──────────────────┘
               │
               ▼  Decode → text
ABSTRACTIVE SUMMARY""", language="text")

        st.markdown("**Key concepts:**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tokenization** — text → numbers. `don't` = 2 tokens. BART limit = 1024 tokens, not 1024 words.")
            st.markdown("**Attention** — lets each word decide which other words to focus on when generating.")
        with c2:
            st.markdown("**Chunking** — articles > 1024 tokens are split into chunks, each summarized, then combined.")
            st.markdown("**Hallucination** — model may generate fluent but factually wrong text. Known LLM limitation.")

    with tab3:
        st.markdown("## ROUGE Metrics")
        st.markdown("Measures **word overlap** between generated summary and a human reference. Does NOT measure meaning.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**ROUGE-1**\nCounts matching **single words** (unigrams)")
        with col2:
            st.markdown("**ROUGE-2**\nCounts matching **word pairs** (bigrams) — stricter")
        with col3:
            st.markdown("**ROUGE-L**\n**Longest Common Subsequence** similarity")

        st.code("""Precision = Matching words ÷ Total words GENERATED
          "Of what I generated, how much was correct?"

Recall    = Matching words ÷ Total words in REFERENCE
          "Of the reference, how much did I cover?"

F1        = 2 × (Precision × Recall) ÷ (Precision + Recall)
          Balanced score — what we REPORT""", language="text")

        st.markdown("### 🎮 Live ROUGE Calculator")
        rc1, rc2 = st.columns(2)
        with rc1:
            gen_text = st.text_area(
                "Generated summary",
                value="Researchers found a new species in the Pacific Ocean at extreme depths.",
                height=80, key="rouge_gen"
            )
        with rc2:
            ref_text = st.text_area(
                "Reference summary",
                value="Scientists discovered a new fish species in the Mariana Trench at extreme depths.",
                height=80, key="rouge_ref"
            )
        if st.button("Calculate ROUGE", key="calc_rouge_btn"):
            scores = calc_rouge(gen_text, ref_text)
            if scores:
                r1, r2, r3 = st.columns(3)
                for col, key, label in zip([r1,r2,r3],["rouge1","rouge2","rougeL"],["ROUGE-1","ROUGE-2","ROUGE-L"]):
                    with col:
                        st.metric(label, f"F1: {scores[key]['f1']:.4f}",
                                  f"P:{scores[key]['precision']:.3f} R:{scores[key]['recall']:.3f}")

    with tab4:
        st.markdown("## System Architecture")
        st.code("""User Input (long article)
        │
        ▼  Preprocessing (clean whitespace · validate)
        │
   ┌────┴────┐
   ▼         ▼
EXTRACTIVE   BART ABSTRACTIVE
   │             │
   │  1.Sentences│  1. Tokenize → token IDs
   │  2.Word freq│  2. >1024? → chunk
   │  3.Score    │  3. Encoder (bidirectional)
   │  4.Top-N    │  4. Decoder (autoregressive)
   ▼             ▼
Extractive    Abstractive
Summary       Summary
   │             │
   └──────┬──────┘
          ▼
   ROUGE Evaluation (vs human reference)
   ROUGE-1 · ROUGE-2 · ROUGE-L
          ▼
   Comparison Table + Bar Chart""", language="text")

        st.markdown("### Module Map")
        import pandas as pd
        st.table(pd.DataFrame([
            ("src/summarizer.py",    "BART loading, tokenization, abstractive summarization, chunking"),
            ("src/extractive.py",    "Word-frequency extractive algorithm, sentence scoring"),
            ("src/evaluate.py",      "ROUGE-1, ROUGE-2, ROUGE-L (Precision / Recall / F1)"),
            ("src/utils.py",         "Text cleaning, validation, file I/O, display helpers"),
            ("src/dataset_test.py",  "CNN/DailyMail dataset loading and batch experiment"),
            ("web/app.py",           "Flask local server (full version with BART)"),
            ("streamlit_app.py",     "This file — Streamlit cloud deployment"),
        ], columns=["File", "Purpose"]))

    with tab5:
        st.markdown("## Viva Q&A")
        import pandas as pd
        qa = [
            ("What is BART?", "Bidirectional & Auto-Regressive Transformer by Facebook AI (2020). Encoder reads input, Decoder generates summary."),
            ("What does the encoder do?", "Reads ALL tokens simultaneously using bidirectional self-attention. Produces contextual representations."),
            ("What does the decoder do?", "Generates summary ONE token at a time using cross-attention (encoder) and self-attention (own tokens)."),
            ("What is tokenization?", "Converts text → numbers the model understands. 'don't' = 2 tokens. BART limit = 1024 tokens."),
            ("What is attention?", "Mechanism that lets each word decide which other words to focus on. Core of all Transformers."),
            ("Why BART not T5?", "facebook/bart-large-cnn is fine-tuned specifically on CNN/DailyMail — perfect for our task."),
            ("What is ROUGE-1?", "Counts matching single words (unigrams) between generated and reference summary."),
            ("What is ROUGE-2?", "Counts matching word pairs (bigrams). Stricter than ROUGE-1."),
            ("What is ROUGE-L?", "Longest Common Subsequence similarity — non-consecutive word matching."),
            ("What is F1?", "Harmonic mean of Precision and Recall. F1 = 2×P×R/(P+R). We report F1."),
            ("What is hallucination?", "Model generates fluent but factually wrong text. Known LLM limitation."),
            ("Why chunking?", "BART max = 1024 tokens. Long articles must be split, each chunk summarized, then combined."),
            ("Limitation of ROUGE?", "Measures word overlap only — not meaning, readability, or factual accuracy."),
            ("Why pretrained?", "Training BART from scratch needs millions of articles + weeks on GPUs. Pretrained = free + instant."),
            ("What is beam search?", "Explores top-4 token sequences at each step, picks globally best summary. Avoids greedy errors."),
            ("What is CNN/DailyMail?", "~300k news articles with human-written highlights. Industry-standard summarization benchmark."),
            ("Extractive vs Abstractive?", "Extractive copies sentences. Abstractive generates new ones. Abstractive is more advanced."),
            ("What is fine-tuning?", "Training a pretrained model a bit more on a specific dataset. We only do inference here."),
            ("What is positional encoding?", "Transformers process tokens in parallel — positional encoding tells model each token's position."),
            ("What is the research gap?", "Comparing BART vs extractive methods under same evaluation framework with limited compute."),
        ]
        st.dataframe(
            pd.DataFrame(qa, columns=["Question", "Answer"]),
            use_container_width=True,
            hide_index=True,
            height=600
        )


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────

if page == "🏠 Summarize":
    page_summarize()
elif page == "⚖️ Compare":
    page_compare()
elif page == "📚 How It Works":
    page_how()
