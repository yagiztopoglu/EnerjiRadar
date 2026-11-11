@echo off
rem EnerjiRadar kurulum: sanal ortam + paketler + testler. Cift tiklayin.
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 goto nopython
echo [1/4] Sanal ortam (.venv)...
if not exist .venv python -m venv .venv
if errorlevel 1 goto fail
call .venv\Scripts\activate.bat
echo [2/4] Paketler kuruluyor. PyTorch buyuk, birkac dakika surebilir...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto fail
echo [3/4] LightGBM (istege bagli; kurulamazsa scikit-learn alternatifi kullanilir)...
pip install lightgbm
if errorlevel 1 echo UYARI: LightGBM kurulamadi. Proje scikit-learn HistGradientBoosting ile calisir; sonuclarda hangisinin kullanildigi yazar.
echo [4/4] Testler...
python -m unittest discover -s tests
if errorlevel 1 echo UYARI: Bazi testler basarisiz oldu. Ciktiyi bana gonderin.
echo.
echo KURULUM TAMAM. Sirada: data\uci klasorune UCI zip dosyasini koyun, sonra build_db.bat.
pause
exit /b 0
:nopython
echo HATA: Python bulunamadi. python.org adresinden Python 3.11 veya 3.12 kurun ("Add Python to PATH" isaretli).
pause
exit /b 1
:fail
echo HATA olustu. Son satirlari kopyalayip bana gonderin.
pause
exit /b 1
