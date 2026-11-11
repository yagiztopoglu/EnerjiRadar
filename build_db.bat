@echo off
rem UCI veriyi SQLite veritabanina yukler. Once data\uci klasorune zip dosyasini koyun.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\build_db.py %*
pause
