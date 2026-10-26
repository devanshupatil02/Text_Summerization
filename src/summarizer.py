# ============================================================
# FILE: src/summarizer.py
# PURPOSE: Abstractive Summarization using BART Transformer
#
# MODEL USED: facebook/bart-large-cnn
#
# WHAT IS ABSTRACTIVE SUMMARIZATION?
#   Unlike extractive summarization (which picks existing
#   sentences), abstractive summarization GENERATES NEW
#   sentences that convey the meaning of the original text.
#   It reads and understands the article, then writes a
#   summary in its own words — just like a human would.
#
#   SIMPLE HINDI:
#   Abstractive summarization mein model original article
#   ko samajhta hai aur phir apne words mein ek naya summary
#   likhta hai. Ye human ki tarah kaam karta hai.
#
# WHAT IS BART?
#   BART = Bidirectional and Auto-Regressive Transformer.
#   It was created by Facebook AI Research.
#   It has two parts:
#     - ENCODER: Reads and understands the input text.
#     - DECODER: Generates the output summary word by word.
#
#   BART was pre-trained on two tasks:
#     1. Denoising: Given a corrupted sentence, reconstruct it.
#     2. It was then fine-tuned specifically on CNN/DailyMail
#        for news article summarization.
#   This is why facebook/bart-large-cnn is perfect for our task.
#
# TOKEN LIMIT:
#   BART can process at most 1024 tokens at once.
#   For longer documents, we use a chunking strategy.
#   Chunking means: split the document into pieces, summarize
#   each piece, then combine the chunk summaries.
# ============================================================

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

MODEL_NAME = "facebook/bart-large-cnn"

# BART's maximum input token limit
BART_MAX_INPUT_TOKENS = 1024

# Default summary length controls
DEFAULT_MIN_LENGTH = 50   # Minimum tokens in generated summary
DEFAULT_MAX_LENGTH = 200  # Maximum tokens in generated summary

# Chunk size for long document splitting (in tokens)
CHUNK_SIZE = 900  # Slightly less than 1024 to be safe


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

# Global variables to hold loaded model and tokenizer.
# We load them ONCE and reuse them for all summarizations.
# Loading a large model is expensive — takes 30-60 seconds.
_model = None
_tokenizer = None


def load_model(model_name: str = MODEL_NAME, verbose: bool = True):
    """
    Loads the BART tokenizer and model from Hugging Face.

    FIRST CALL: Downloads model from the internet (~1.6 GB).
    Subsequent calls: Loads from local cache (fast).

    HOW THE MODEL IS LOADED:
    1. BartTokenizer: converts words → numbers (token IDs).
    2. BartForConditionalGeneration: the actual neural network.

    WHAT IS A TOKENIZER? (SIMPLE EXPLANATION)
    The model does not understand English words directly.
    It works with numbers called "token IDs".
    The tokenizer converts: "Hello world" → [50118, 232, 232]
    After generation, it converts IDs back to words.

    SIMPLE HINDI:
    Model English words directly nahi samajhta. Tokenizer
    words ko numbers mein convert karta hai. Ye numbers
    model ke through pass hote hain. Output numbers ko
    phir se words mein convert kiya jata hai.

    Returns: (tokenizer, model) tuple
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        if verbose:
            print("  Model already loaded. Using cached version.")
        return _tokenizer, _model

    if verbose:
        print(f"\n  Loading BART model: {model_name}")
        print("  First time: downloads ~1.6GB. Subsequent runs use cache.")
        print("  Please wait...")

    try:
        _tokenizer = BartTokenizer.from_pretrained(model_name)
        _model = BartForConditionalGeneration.from_pretrained(model_name)

        # Move model to GPU if available, otherwise use CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()  # Set to evaluation mode (disables dropout)

        if verbose:
            print(f"  Model loaded successfully. Using device: {device.upper()}")
            if device == "cpu":
                print(
                    "  Note: CPU inference is slow for BART-large. "
                    "Each summary may take 30-120 seconds."
                )

        return _tokenizer, _model

    except Exception as e:
        print(f"\n  ERROR loading model: {e}")
        print("\n  Troubleshooting:")
        print("  1. Check internet connection (first-time download)")
        print("  2. Ensure transformers is installed: pip install transformers")
        print("  3. Ensure torch is installed: pip install torch")
        raise


def get_device():
    """Returns the current device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# TOKENIZATION HELPER
# ─────────────────────────────────────────────────────────────

def count_tokens(text: str, tokenizer) -> int:
    """
    Counts how many tokens a piece of text has.

    WHY DO WE COUNT TOKENS (not words)?
    Tokens are not the same as words.
    "don't" → ["don", "'t"]  (2 tokens, 1 word)
    "unhappiness" → ["un", "happiness"]  (2 tokens, 1 word)

    BART's limit is 1024 TOKENS, not 1024 WORDS.
    So we must tokenize first, then count.

    SIMPLE HINDI:
    Words aur tokens same nahi hote. "don't" ek word hai
    lekin 2 tokens hain. BART ki limit 1024 tokens hai
    isliye tokens count karna zaroori hai.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)


# ─────────────────────────────────────────────────────────────
# TEXT CHUNKING FOR LONG DOCUMENTS
# ─────────────────────────────────────────────────────────────

def split_text_into_chunks(text: str, tokenizer, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Splits a long document into chunks that each fit within
    BART's token limit.

    STRATEGY:
    1. Split text into individual sentences.
    2. Keep adding sentences to the current chunk.
    3. When adding the next sentence would exceed the limit,
       save the current chunk and start a new one.

    WHY SENTENCE-LEVEL SPLITTING?
    Splitting mid-sentence creates incoherent chunks. So we
    always split at sentence boundaries.

    LIMITATION:
    Context can be lost between chunks. A pronoun in chunk 2
    might refer to a noun in chunk 1, but the model doesn't
    see chunk 1 when processing chunk 2.
    This is an inherent limitation of chunking.

    SIMPLE HINDI:
    Lamba document ko chhote pieces mein todna padta hai.
    Hum sentences pe split karte hain taaki beech mein
    sentence na kate. Lekin ek limitation hai: chunk 2 mein
    aaya "he" ya "it" chunk 1 ki kisi cheez ko refer kar
    sakta hai, lekin model ko pata nahi chalega.

    Returns: list of text strings (each chunk)
    """
    import re
    # Simple sentence splitting (we keep it simple here
    # to avoid circular import with extractive.py)
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, tokenizer)

        # If a single sentence exceeds chunk size, truncate it
        if sentence_tokens > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            # Hard-truncate the oversized sentence
            tokens = tokenizer.encode(sentence)[:chunk_size]
            truncated = tokenizer.decode(tokens, skip_special_tokens=True)
            chunks.append(truncated)
            continue

        # If adding this sentence would exceed limit, save chunk
        if current_token_count + sentence_tokens > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_token_count = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_tokens

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ─────────────────────────────────────────────────────────────
# CORE SUMMARIZATION FUNCTION
# ─────────────────────────────────────────────────────────────

def summarize_text(text: str,
                   min_length: int = DEFAULT_MIN_LENGTH,
                   max_length: int = DEFAULT_MAX_LENGTH,
                   verbose: bool = False) -> str:
    """
    Generates an abstractive summary for a SINGLE CHUNK of text
    that fits within BART's token limit.

    HOW BART GENERATES THE SUMMARY (STEP BY STEP):

    Step 1 — TOKENIZATION:
    The tokenizer converts the input text into token IDs.
    Example: "AI helps doctors" → [0, 4132, 2116, 13073, 2]

    Step 2 — ENCODING:
    The BART Encoder reads all the tokens simultaneously
    (bidirectional — it sees the whole input at once).
    It produces a rich "contextual representation" — a set of
    numbers that captures the meaning of the entire input.

    Step 3 — DECODING (Autoregressive Generation):
    The BART Decoder generates the summary one token at a time.
    At each step, it:
      a. Looks at the encoder's contextual representation
         using CROSS-ATTENTION.
      b. Looks at its own previously generated tokens
         using SELF-ATTENTION.
      c. Predicts the most likely next token.
    This continues until a special <EOS> (end of sequence)
    token is generated or max_length is reached.

    Step 4 — DECODING BACK TO TEXT:
    The generated token IDs are converted back to readable text.

    WHAT IS ATTENTION? (SIMPLE EXPLANATION)
    Attention is a mechanism that lets the model decide
    which parts of the input are most relevant when
    generating each word of the summary.
    Example: When generating "researchers", the model
    "attends to" (focuses on) the word "scientists" in
    the input.

    SIMPLE HINDI:
    Step 1: Text ko numbers mein convert karo (tokenization).
    Step 2: BART Encoder poora text ek saath padhta hai aur
            meaning samajhta hai (contextual representation).
    Step 3: BART Decoder ek ek word generate karta hai.
            Har word generate karte time wo dekhhta hai ki
            input mein kya likha tha (cross-attention) aur
            usne abhi tak kya generate kiya (self-attention).
    Step 4: Generated numbers ko wapas text mein convert karo.

    Args:
        text: Input text (must fit within BART's token limit).
        min_length: Minimum number of tokens in summary.
        max_length: Maximum number of tokens in summary.
        verbose: Print progress information.

    Returns: Generated summary as a string.
    """
    tokenizer, model = load_model(verbose=False)
    device = get_device()

    if not text or not text.strip():
        return ""

    # Tokenize the input
    inputs = tokenizer(
        text,
        return_tensors="pt",        # Return PyTorch tensors
        max_length=BART_MAX_INPUT_TOKENS,
        truncation=True,            # Truncate if over limit
        padding=False
    )

    # Move input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if verbose:
        token_count = inputs["input_ids"].shape[1]
        print(f"  Input tokens: {token_count}")

    # Generate summary using BART
    # no_grad() tells PyTorch: don't calculate gradients
    # (we are not training, just using the model)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,            # Beam search: explore 4 paths
            min_length=min_length,
            max_length=max_length,
            length_penalty=2.0,     # Favor longer summaries
            early_stopping=True     # Stop when EOS token generated
        )

    # Decode the generated token IDs back to text
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True   # Remove <s>, </s> tokens
    )

    return summary.strip()


# ─────────────────────────────────────────────────────────────
# LONG DOCUMENT SUMMARIZATION
# ─────────────────────────────────────────────────────────────

def summarize_long_text(text: str,
                         min_length: int = DEFAULT_MIN_LENGTH,
                         max_length: int = DEFAULT_MAX_LENGTH,
                         verbose: bool = True) -> str:
    """
    Handles summarization of documents that exceed BART's
    1024-token input limit.

    CHUNKING STRATEGY:
    1. Split the document into sentence-level chunks.
    2. Each chunk is within the token limit.
    3. Summarize each chunk individually.
    4. Combine the chunk summaries into one final summary.

    LIMITATION (IMPORTANT TO KNOW FOR VIVA):
    Chunking is a practical workaround, NOT a perfect solution.
    The model does not see the full document at once.
    Context that spans across chunk boundaries is lost.
    This can sometimes produce summaries that miss
    global themes or misinterpret coreferences.

    For TRULY long documents, models like Longformer or
    LED (Longformer Encoder-Decoder) are better suited.
    They support sequences up to 16,384 tokens.

    SIMPLE HINDI:
    Agar document bahut lamba hai toh hum use pieces mein
    todh te hain. Har piece ka alag summary banate hain.
    Phir saare summaries ko jodh te hain. Limitation ye hai
    ki ek piece dusre piece ka context nahi jaanta.

    Args:
        text: Full document text (any length).
        min_length, max_length: Summary length controls.
        verbose: Print progress.

    Returns: Final summary string.
    """
    tokenizer, model = load_model(verbose=verbose)

    if not text or not text.strip():
        return "Error: Empty input text."

    token_count = count_tokens(text, tokenizer)

    if verbose:
        print(f"  Input token count: {token_count}")

    # If text fits within limit, summarize directly
    if token_count <= BART_MAX_INPUT_TOKENS:
        if verbose:
            print("  Text fits within BART's limit. Summarizing directly.")
        return summarize_text(text, min_length, max_length, verbose)

    # Text is too long — use chunking
    if verbose:
        print(
            f"  Text exceeds {BART_MAX_INPUT_TOKENS} tokens. "
            "Using chunking strategy..."
        )

    chunks = split_text_into_chunks(text, tokenizer)

    if verbose:
        print(f"  Split into {len(chunks)} chunks.")

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"  Summarizing chunk {i+1}/{len(chunks)}...")
        chunk_min = max(20, min_length // len(chunks))
        chunk_max = max_length // max(1, len(chunks) - 1)
        summary = summarize_text(chunk, chunk_min, chunk_max, verbose=False)
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return "Error: Could not generate summary."

    # Combine all chunk summaries
    combined = " ".join(chunk_summaries)

    # If we have multiple chunks, do a final summarization pass
    # on the combined chunk summaries
    if len(chunk_summaries) > 1:
        if verbose:
            print("  Combining chunk summaries...")
        final_token_count = count_tokens(combined, tokenizer)
        if final_token_count <= BART_MAX_INPUT_TOKENS:
            combined = summarize_text(
                combined, min_length, max_length, verbose=False
            )

    return combined


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test BART summarization:
    cd src
    python summarizer.py

    EXPECTED OUTPUT:
    - Model loads (first time downloads ~1.6GB)
    - Summary is generated (may take 30-120 seconds on CPU)
    - Summary is printed
    """

    print("=" * 60)
    print("  BART ABSTRACTIVE SUMMARIZATION TEST")
    print("=" * 60)

    test_article = (
        "Scientists have discovered a new species of deep-sea fish "
        "in the Pacific Ocean that can survive at extreme depths of "
        "more than 8,000 meters. The fish, named Pseudoliparis "
        "swirei, was found in the Mariana Trench, the deepest part "
        "of the ocean. Researchers from the University of Washington "
        "conducted the study using a specially designed underwater "
        "vehicle equipped with cameras and collection systems. The "
        "fish has several unique adaptations that allow it to survive "
        "under tremendous water pressure, including a flexible "
        "skeleton, large amounts of a special chemical called TMAO, "
        "and a highly efficient metabolism. The discovery sheds new "
        "light on the limits of vertebrate life and suggests that "
        "even the deepest ocean environments are inhabited by complex "
        "organisms. The team published their findings in the journal "
        "Nature Ecology and Evolution."
    )

    print("\nOriginal Article:")
    print(test_article)

    print("\nGenerating BART summary (please wait)...")
    summary = summarize_long_text(test_article, verbose=True)

    print("\nBART Summary:")
    print(summary)

    print(f"\nOriginal word count: {len(test_article.split())}")
    print(f"Summary word count:  {len(summary.split())}")
