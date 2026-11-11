@echo off
rem Tum deneyleri calistirir (--quick: hizli deneme). Sonuc: results\EnerjiRadar_Sonuclar.xlsx
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\run_experiments.py %*
pause
