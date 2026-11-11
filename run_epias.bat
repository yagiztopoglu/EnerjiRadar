@echo off
rem EPIAS (Turkiye) dis dogrulama. Dosyayi data\external\epias.csv veya epias.xlsx olarak koyun.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
set F=
if exist data\external\epias.csv set F=data\external\epias.csv
if exist data\external\epias.xlsx set F=data\external\epias.xlsx
if "%F%"=="" (
  echo HATA: data\external\epias.csv veya epias.xlsx bulunamadi.
  pause
  exit /b 1
)
python scripts\run_external.py --dataset epias --file %F% %*
pause
