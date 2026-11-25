LEGAL CONTRACT RISK-CLAUSE DETECTION USING NLP
================================================

WHAT THIS PROJECT DOES
- Uploads a text-based PDF, DOCX, or TXT contract.
- Extracts text and separates it into clauses.
- Uses a TF-IDF + Logistic Regression model to classify clauses.
- Shows potential risk categories and short screening explanations.
- Lets you download results as CSV and extracted clauses as TXT.
- Includes a built-in demo contract.

IMPORTANT LIMITATION
This is a student project prototype. It uses a small illustrative sample dataset, so results are not legally validated and may be inaccurate. The model learns patterns from its training examples and is not a substitute for a lawyer. "No Apparent Risk" does not mean a clause is safe. Do not upload confidential contracts to a demo application unless you have permission and understand how the files are handled. This app processes uploaded documents locally on your computer.

HOW TO RUN ON WINDOWS
1. Install Python 3.10 or newer from https://www.python.org/downloads/
   During installation, tick "Add Python to PATH".
2. Extract this ZIP folder.
3. Double-click run_windows.bat.
4. Wait while required packages install. Internet is needed for the first setup.
5. Your browser should open the app. If not, copy the local address shown in the terminal (usually http://localhost:8501) into your browser.
6. Select "Use built-in demo contract instead" and click "Analyze Contract", or upload your own PDF/DOCX/TXT file.
7. Use the download buttons to save the results.

ALTERNATIVE: RUN FROM VS CODE TERMINAL
Open this folder in VS Code, then run:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

If PowerShell blocks activation, use Command Prompt terminal or run:
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m streamlit run app.py

FILES
- app.py: Streamlit web app and NLP workflow
- sample_clauses.csv: small illustrative training dataset
- requirements.txt: Python dependencies
- run_windows.bat: Windows setup and launcher
- README.txt: this guide

MODEL DETAILS
- Text features: TF-IDF word unigrams and bigrams
- Classifier: Logistic Regression
- Labels: payment risk, termination risk, liability risk, confidentiality risk, indemnity risk, no apparent risk

DATA PRIVACY
The app is designed to run locally. Uploaded files are processed in memory by the app and are not intentionally sent to an external service. Package installation requires internet access. Avoid using real sensitive contracts until you have reviewed the code and your environment.
