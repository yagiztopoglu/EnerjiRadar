# Kullanılan teknolojiler

| Araç / paket | Görevi |
|---|---|
| Python 3.11 / 3.12, venv, pip | Ana dil, izole ortam, paket kurulumu (`setup.bat`) |
| **SQLite** (Python `sqlite3`, ek kurulum yok) | Ham dakikalık veri, saatlik/günlük/haftalık VIEW'lar, window function'lar (AVG OVER, LAG, RANK, ROW_NUMBER), JOIN, gaps-and-islands, israf tablosu |
| pandas, numpy | Tablo işleme, zaman serisi, bootstrap |
| scikit-learn | HistGradientBoosting (kuantil) yedek model |
| LightGBM (`pip install lightgbm`, isteğe bağlı) | Kuantil regresyonlu gradyan artırma |
| PyTorch | LSTM (CPU yeter) |
| plotly, streamlit | Grafikler, web arayüzü |
| holidays (`pip install holidays`) | Türkiye ve Portekiz resmi tatilleri, dinî bayramlar (yalnızca dış doğrulama) |
| xlsxwriter | Biçimli Excel raporu |
| unittest | Birim testleri (ek kurulum gerektirmez) |
| .bat dosyaları | Windows'ta çift tıkla çalıştırma |

npm / Node.js kullanılmıyor. Bu ortamda test edilen sürümler: Python 3.13, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, xlsxwriter 3.2.9.
