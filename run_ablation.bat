@echo off
rem Ozellik grubu ablasyonu (yeni ozelliklerin dogrulugu artirip artirmadigi). Sonuc: results\EnerjiRadar_Ablasyon.xlsx
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\run_ablation.py %*
pause
