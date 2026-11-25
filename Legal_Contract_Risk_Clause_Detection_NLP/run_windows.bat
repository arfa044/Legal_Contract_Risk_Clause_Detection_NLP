@echo off
title Legal Contract Risk-Clause Detection NLP
cd /d "%~dp0"
echo ==============================================
echo Legal Contract Risk-Clause Detection Using NLP
echo ==============================================
where py >nul 2>nul
if %errorlevel%==0 (
    set PYTHON=py
) else (
    set PYTHON=python
)
%PYTHON% --version
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
    echo During installation, select "Add Python to PATH".
    pause
    exit /b 1
)
if not exist .venv (
    echo Creating virtual environment...
    %PYTHON% -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing required packages. This may take a few minutes on first run...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Package installation failed. Check your internet connection and Python installation.
    pause
    exit /b 1
)
echo Starting the web app...
python -m streamlit run app.py
pause
