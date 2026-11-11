@echo off
rem Web arayuzunu baslatir (tarayicida acilir).
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app\app.py
pause
