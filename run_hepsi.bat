@echo off
rem Sirayla: mevsimlik (LSTM dahil) + aylik + ablasyon. Uzun surer; cikti dosyalari results\ altinda.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo [1/3] Mevsimlik koşu (LSTM + ensemble + kosul anlamliligi)...
python scripts\run_experiments.py
echo [2/3] Aylik koşu (34 blok, baseline + GB)...
python scripts\run_experiments.py --folds monthly
echo [3/3] Ablasyon...
python scripts\run_ablation.py
echo.
echo BITTI. Dosyalar: results\EnerjiRadar_Sonuclar.xlsx, results\monthly\..., results\EnerjiRadar_Ablasyon.xlsx
pause
