@echo off
rem Dis dogrulama: baska verilerde ayni yontemi dener.
rem   run_external.bat --dataset epias  --file data\external\epias.csv
rem   run_external.bat --dataset ld2011 --file data\external\electricityloaddiagrams20112014.zip
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\run_external.py %*
pause
