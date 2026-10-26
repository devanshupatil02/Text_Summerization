@echo off
echo ============================================================
echo  TEXT SUMMARIZATION PROJECT - SETUP SCRIPT
echo ============================================================
echo.

echo [1/6] Creating virtual environment...
python -m venv venv
echo Done.

echo.
echo [2/6] Installing rouge-score...
venv\Scripts\pip.exe install rouge-score==0.1.2 --no-cache-dir

echo.
echo [3/6] Installing nltk and pandas...
venv\Scripts\pip.exe install nltk==3.9.1 pandas==2.2.3 matplotlib==3.9.4 --no-cache-dir

echo.
echo [4/6] Installing PyTorch (CPU version - no GPU needed)...
venv\Scripts\pip.exe install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

echo.
echo [5/6] Installing Hugging Face Transformers...
venv\Scripts\pip.exe install transformers==4.44.2 --no-cache-dir

echo.
echo [6/6] Installing Datasets + extras...
venv\Scripts\pip.exe install datasets==2.21.0 sentencepiece==0.2.0 accelerate==0.34.2 --no-cache-dir

echo.
echo [7/7] Installing Streamlit (web interface)...
venv\Scripts\pip.exe install streamlit==1.38.0 --no-cache-dir

echo.
echo ============================================================
echo  SETUP COMPLETE! Now run:
echo    venv\Scripts\activate
echo    python src\extractive.py      (test fast module)
echo    python src\evaluate.py        (test ROUGE)
echo    python src\main.py            (run CLI project)
echo    streamlit run app.py          (run web interface)
echo ============================================================
pause
