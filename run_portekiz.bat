@echo off
rem Portekiz (UCI 370 musteri) dis dogrulama. Zip'i data\external\ icine koyun (acmayin).
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
if not exist data\external\electricityloaddiagrams20112014.zip (
  echo HATA: data\external\electricityloaddiagrams20112014.zip bulunamadi.
  echo Indirme: https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\run_external.py --dataset ld2011 --file data\external\electricityloaddiagrams20112014.zip %*
pause
