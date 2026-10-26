# Text Summarization Using Transformer Models
### BTech Artificial Intelligence and Data Science — Mini Project

---

## Project Overview

This project builds a complete text summarization system that:
- Generates **extractive summaries** using a word-frequency scoring algorithm.
- Generates **abstractive summaries** using the pretrained `facebook/bart-large-cnn` Transformer model.
- Evaluates summaries using **ROUGE-1, ROUGE-2, and ROUGE-L** metrics.
- Compares both methods on the **CNN/DailyMail** dataset.

---

## Project Structure

```
Text-Summarization-Transformer/
│
├── data/
│   └── sample_articles.txt        # 3 sample articles for quick demo
│
├── outputs/
│   ├── summaries.txt              # Generated summaries
│   └── results.csv                # ROUGE results table
│
├── src/
│   ├── main.py                    # Main menu program (run this!)
│   ├── summarizer.py              # BART abstractive summarizer
│   ├── extractive.py              # Word-frequency extractive method
│   ├── evaluate.py                # ROUGE evaluation module
│   ├── dataset_test.py            # CNN/DailyMail experiment
│   └── utils.py                   # Helper utilities
│
├── app.py                         # Optional Streamlit web UI
├── requirements.txt               # All Python dependencies
└── README.md                      # This file
```

---

## Quick Start (Windows 11 / VS Code)

### Step 1: Open the project folder in VS Code
```
File → Open Folder → Select "Text-Summarization-Transformer"
```

### Step 2: Open the terminal
```
Terminal → New Terminal  (or press Ctrl + `)
```

### Step 3: Create a virtual environment
```powershell
python -m venv venv
```

### Step 4: Activate the virtual environment
```powershell
venv\Scripts\activate
```
You should see `(venv)` in your terminal prompt.

### Step 5: Install all dependencies
```powershell
pip install -r requirements.txt
```

### Step 6: Run the main program
```powershell
python src/main.py
```

---

## Running Individual Modules

### Test extractive summarization only (fast, no download):
```powershell
python src/extractive.py
```

### Test ROUGE evaluation only (fast, no download):
```powershell
python src/evaluate.py
```

### Test BART summarizer (downloads ~1.6GB on first run):
```powershell
python src/summarizer.py
```

### Preview CNN/DailyMail dataset (no BART):
```powershell
python src/dataset_test.py --preview
```

### Run full dataset experiment (10 samples):
```powershell
python src/dataset_test.py --samples 10
```

### Run the web interface (optional):
```powershell
pip install streamlit
streamlit run app.py
```

### Deploy to Vercel
This project can be prepared for deployment on Vercel as a Python serverless app. To deploy it:

1. Install the Vercel CLI:
```powershell
npm install -g vercel
```
2. Log in:
```powershell
vercel login
```
3. From the project root, run:
```powershell
vercel
```
4. Follow the prompts to link the project and deploy it.

Note: Because this app loads the `facebook/bart-large-cnn` model and needs a large runtime footprint, cold starts and free-tier limits may be restrictive. For best results, use a paid plan or a lighter model.

---

## Common Errors and Solutions

### "python is not recognized"
Python is not in PATH. Download from python.org and check "Add to PATH" during install.

### PowerShell execution policy error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "ModuleNotFoundError: No module named 'transformers'"
Virtual environment not activated or packages not installed:
```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

### Model download is slow / fails
- Check internet connection.
- BART-large-cnn is ~1.6GB. Download takes 10-30 minutes on slow connections.
- Re-run the script; it resumes from cache.

### Out of memory error
- Close other applications.
- Reduce NUM_SAMPLES in dataset_test.py.
- Use smaller model: change MODEL_NAME to "sshleifer/distilbart-cnn-6-6" in summarizer.py.

### Slow inference (30-120 seconds per article)
- Normal on CPU. BART-large is a large model.
- Consider using distilbart for faster results.

---

## Technology Stack

| Component | Library |
|-----------|---------|
| Deep Learning | PyTorch |
| BART Model | Hugging Face Transformers |
| Dataset | Hugging Face Datasets |
| ROUGE Metrics | rouge-score |
| Sentence Tokenization | NLTK |
| Data Handling | Pandas |
| Web UI (optional) | Streamlit |

---

## Model Information

**Model**: `facebook/bart-large-cnn`  
**Parameters**: ~406 million  
**Max Input**: 1024 tokens  
**Training**: Pre-trained on BookCorpus + Wikipedia, fine-tuned on CNN/DailyMail  
**Size**: ~1.6 GB  

---

## Expected ROUGE Scores

ROUGE scores depend on the actual experiment you run. Do not use hardcoded values. After running `dataset_test.py`, read your actual scores from `outputs/results.csv`.

Typical ranges for BART on CNN/DailyMail (literature values):
- ROUGE-1 F1: ~0.39 – 0.44
- ROUGE-2 F1: ~0.17 – 0.21
- ROUGE-L F1: ~0.36 – 0.40

Your actual values may vary. Always report your own experimental results.
