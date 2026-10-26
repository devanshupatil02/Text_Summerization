# TEAM SETUP GUIDE
## Text Summarization Using Transformer Models
### BTech AI & DS — Mini Project

---

## SHARE THIS FILE WITH YOUR TEAM

Send your team members the **entire project folder** (zip it or share via Google Drive / GitHub).
They must follow this guide exactly to run the project on their machine.

---

## WHAT THIS PROJECT IS

A complete AI text summarization web application that:
- Takes a news article as input
- Generates an **Extractive Summary** (picks key sentences)
- Generates an **Abstractive Summary** using Facebook's **BART** Transformer model
- Evaluates both using **ROUGE-1, ROUGE-2, ROUGE-L** scores
- Shows results on a **local website** (Flask web app)

**Tech stack:** Python · Flask · PyTorch · Hugging Face Transformers · NLTK · ROUGE

---

## PROJECT FOLDER STRUCTURE

```
Text-Summarization-Transformer/
│
├── data/
│   └── sample_articles.txt        ← 3 built-in sample articles
│
├── outputs/
│   ├── summaries.txt              ← saved after running experiment
│   └── results.csv                ← ROUGE results table
│
├── src/                           ← Python backend modules
│   ├── summarizer.py              ← BART model (abstractive)
│   ├── extractive.py              ← Word-frequency algorithm
│   ├── evaluate.py                ← ROUGE calculation
│   ├── dataset_test.py            ← CNN/DailyMail experiment
│   ├── utils.py                   ← Helper functions
│   └── main.py                    ← CLI version (optional)
│
├── web/                           ← Flask web application
│   ├── app.py                     ← Flask server (MAIN FILE TO RUN)
│   ├── templates/
│   │   ├── base.html              ← Navbar + footer layout
│   │   ├── index.html             ← Summarize page
│   │   ├── compare.html           ← Compare page
│   │   ├── dataset.html           ← Dataset experiment page
│   │   └── about.html             ← How it works + Viva Q&A
│   └── static/
│       ├── css/style.css          ← All styles
│       └── js/main.js             ← Charts, AJAX, downloads
│
├── requirements.txt               ← All Python packages
├── install.bat                    ← One-click install (Windows)
├── TEAM_SETUP.md                  ← THIS FILE
└── README.md                      ← Project overview
```

---

## SYSTEM REQUIREMENTS

| Requirement | Details |
|-------------|---------|
| OS | Windows 10 / 11 (this guide) or macOS / Linux |
| Python | **3.10, 3.11, or 3.12** (NOT 3.13) |
| RAM | Minimum 8 GB (16 GB recommended for BART) |
| Disk space | ~4 GB free (models + dataset cache) |
| Internet | Required on first run to download BART (~1.6 GB) and dataset (~1 GB) |
| GPU | NOT required — works on CPU |

---

## STEP 1 — CHECK IF PYTHON IS INSTALLED

Open **Command Prompt** or **PowerShell** and run:

```powershell
python --version
```

**Expected output:** `Python 3.12.7` (or 3.10 / 3.11 / 3.12 — any of these is fine)

**If you see an error:**
1. Download Python from: https://www.python.org/downloads/
2. During install, check ✅ **"Add Python to PATH"**
3. Restart your terminal after installing

---

## STEP 2 — NAVIGATE TO THE PROJECT FOLDER

```powershell
cd "path\to\Text-Summarization-Transformer"
```

Example (adjust to where you saved the folder):
```powershell
cd "C:\Users\YourName\Desktop\Text-Summarization-Transformer"
```

**Tip:** In VS Code, open the folder then press **Ctrl + `** to open the terminal — it starts in the right folder automatically.

---

## STEP 3 — CREATE A VIRTUAL ENVIRONMENT

A virtual environment keeps this project's packages separate from other Python projects.

```powershell
python -m venv venv
```

You will see a new `venv/` folder appear inside the project.

---

## STEP 4 — ACTIVATE THE VIRTUAL ENVIRONMENT

```powershell
venv\Scripts\activate
```

**You will see `(venv)` appear at the start of your terminal prompt:**
```
(venv) PS C:\...\Text-Summarization-Transformer>
```

> ⚠️ **IMPORTANT:** You must activate the venv every time you open a new terminal.
> If you close VS Code and reopen it, run `venv\Scripts\activate` again before anything else.

**If you get a PowerShell execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then run `venv\Scripts\activate` again.

---

## STEP 5 — INSTALL ALL PACKAGES

### Option A — One command (may work on most systems)

```powershell
pip install -r requirements.txt --no-cache-dir
```

### Option B — Install group by group (if Option A gives errors)

Run these commands one by one:

```powershell
# Step 5a: Install Flask and basic tools
pip install flask==3.1.3 rouge-score==0.1.2 nltk==3.9.1 pandas==2.2.3 matplotlib==3.9.4 --no-cache-dir

# Step 5b: Install PyTorch (CPU version - smaller download)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Step 5c: Install Hugging Face packages
pip install transformers==4.44.2 datasets==2.21.0 sentencepiece==0.2.0 accelerate==0.34.2 --no-cache-dir
```

### Verify installation

```powershell
python -c "import flask, torch, transformers, datasets, rouge_score, nltk; print('All packages OK')"
```

Expected output: `All packages OK`

---

## STEP 6 — START THE WEB APPLICATION (BACKEND + FRONTEND)

> This project is a **Flask web application**.
> Flask is the **backend** (Python server).
> The HTML/CSS/JS in `web/templates/` and `web/static/` is the **frontend**.
> **Both run together with a single command** — you do NOT start them separately.

### Run this command (from the project root folder):

```powershell
python web\app.py
```

**You will see:**
```
=======================================================
  TEXT SUMMARIZATION WEB APP
  BTech AI & DS Mini Project
=======================================================
  Open browser: http://localhost:5000
  Press Ctrl+C to stop the server
=======================================================
 * Running on http://0.0.0.0:5000
```

### Open the website

Open your browser and go to:

```
http://localhost:5000
```

The website has 4 pages:
| Page | URL | What it does |
|------|-----|-------------|
| Summarize | http://localhost:5000/ | Paste article → get both summaries |
| Compare | http://localhost:5000/compare | Side-by-side ROUGE comparison |
| Dataset | http://localhost:5000/dataset | CNN/DailyMail experiment |
| How It Works | http://localhost:5000/about | Explanations + Viva Q&A |

---

## STEP 7 — LOAD THE BART MODEL

When you open the website for the first time, the BART model is **not loaded yet**.

You will see a blue banner at the top of the page:
> ⚡ BART model not loaded — Click to load

Click the **"Load BART Model"** button.

**What happens:**
- First time: Downloads ~1.6 GB from Hugging Face (takes 10-30 minutes depending on internet speed)
- After download: Loaded in ~30 seconds from local cache
- The navbar badge changes from `○ BART Not Loaded` → `● BART Ready`

> ✅ The download only happens ONCE. After that it is cached at:
> `C:\Users\YourName\.cache\huggingface\hub\`

---

## HOW BACKEND AND FRONTEND WORK TOGETHER

This is important to understand for viva/demo:

```
Browser (Frontend)          Flask Server (Backend)
──────────────────          ──────────────────────
User opens                  
http://localhost:5000   →   Flask reads index.html template
                        ←   Sends HTML page to browser
Browser displays page

User clicks                 
"Generate Summary"      →   Browser sends POST request to
                             /api/summarize
                             with the article text as JSON

                             Flask receives the request
                             Calls summarizer.py (BART)
                             Calls extractive.py
                             Calls evaluate.py (ROUGE)
                        ←   Returns result as JSON

Browser receives JSON        
Displays summaries and       
ROUGE chart on page
(no page reload needed)
```

**In simple words:**
- Flask serves the HTML pages (frontend)
- JavaScript in the browser calls Flask API endpoints (backend)
- Flask runs Python code (BART, ROUGE) and returns results as JSON
- JavaScript updates the page with the results

---

## STOPPING THE SERVER

Press **Ctrl + C** in the terminal where the server is running.

---

## DAILY WORKFLOW (once everything is set up)

Every day when you want to work on the project:

```powershell
# 1. Open terminal in the project folder
cd "path\to\Text-Summarization-Transformer"

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Start the server
python web\app.py

# 4. Open browser
# http://localhost:5000
```

That's it — 3 commands.

---

## COMMON ERRORS AND FIXES

### Error: `python is not recognized`
Python is not installed or not in PATH.
→ Download from python.org, check "Add to PATH" during install.

### Error: `venv\Scripts\activate` — execution policy
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Error: `ModuleNotFoundError: No module named 'flask'`
Virtual environment is not activated.
```powershell
venv\Scripts\activate
```

### Error: `ModuleNotFoundError: No module named 'transformers'`
Packages not installed in this venv.
```powershell
pip install transformers==4.44.2 --no-cache-dir
```

### Error: `Address already in use` (port 5000)
Another program is using port 5000. Change the port in `web/app.py`:
Find the last line `port=5000` and change to `port=5001`.
Then open `http://localhost:5001`.

### Error: BART model loading fails
- Check your internet connection
- Try running again — large downloads sometimes time out and resume
- Model cache is at `C:\Users\YourName\.cache\huggingface\`

### Error: Out of Memory during BART inference
- Close Chrome tabs and other apps
- Reduce article length
- Or use the smaller model: open `src/summarizer.py` and change:
  ```python
  MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
  ```
  This is 3x faster and uses less RAM (slightly lower quality).

### BART is very slow (60-120 seconds per article)
This is normal on CPU. BART-large has 406 million parameters.
Options:
1. Use `sshleifer/distilbart-cnn-6-6` for 3x speed
2. If you have an NVIDIA GPU, PyTorch will use it automatically

---

## QUICK TEST (without BART)

To verify the installation works before downloading BART:

```powershell
python test_fast.py
```

Expected output:
```
All fast modules: WORKING
  - utils:        OK
  - extractive:   OK (38 words)
  - evaluate:     ROUGE-1: 0.3846  ROUGE-2: 0.2000  ROUGE-L: 0.3462
```

---

## FOR macOS / Linux USERS

The commands are slightly different:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install packages
pip install -r requirements.txt --no-cache-dir

# Install PyTorch (CPU)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Run the server
python web/app.py
```

---

## WHAT EACH PERSON ON THE TEAM CAN DO

| Role | What to do |
|------|-----------|
| **Demo person** | Runs the server, shows the website, loads BART, pastes articles |
| **Explanation person** | Explains BART architecture, ROUGE scores from the "How It Works" page |
| **Results person** | Shows outputs/results.csv, explains the ROUGE comparison table |
| **Viva person** | Uses the Viva Q&A tab on the How It Works page |

---

## SUMMARY — 6 COMMANDS TO REMEMBER

```powershell
# 1. Go to project folder
cd "path\to\Text-Summarization-Transformer"

# 2. Activate venv
venv\Scripts\activate

# 3. (First time only) Install packages
pip install -r requirements.txt --no-cache-dir

# 4. (First time only) Install PyTorch
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# 5. Run the server
python web\app.py

# 6. Open browser
# http://localhost:5000
```

---

*BTech AI & DS Mini Project — Text Summarization Using Transformer Models*
