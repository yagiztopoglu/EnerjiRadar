"""Ayarlar. Tüm deney parametreleri tek yerde; her sonuç dosyasına bu ayarlar yazılır (tekrarlanabilirlik)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "uci"
DB_PATH = ROOT / "data" / "energyradar.db"
RESULTS_DIR = ROOT / "results"

# Yürüyen pencere (walk-forward) test blokları: (ad, başlangıç, bitiş). Her blokta model, bloktan ÖNCEKİ
# verilerle eğitilir; bloğun hemen öncesindeki `calib_days` gün aralık kalibrasyonu için ayrılır.
FOLDS = [
    ("F1 sonbahar 2009", "2009-09-01", "2009-11-30"),
    ("F2 kış 2009-10", "2009-12-01", "2010-02-28"),
    ("F3 ilkbahar 2010", "2010-03-01", "2010-05-31"),
    ("F4 yaz 2010", "2010-06-01", "2010-08-31"),
    ("F5 sonbahar 2010", "2010-09-01", "2010-11-25"),
]


def monthly_folds(first="2008-01", last="2010-10"):
    """Aylık yürüyen test blokları (her ay bir blok, model o aydan önceki verilerle eğitilir)."""
    import pandas as pd
    out = []
    for m in pd.period_range(first, last, freq="M"):
        end = min(m.end_time.normalize(), pd.Timestamp("2010-11-25"))
        out.append((f"M{m}", m.start_time.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return out


@dataclass
class Config:
    min_minutes_per_hour: int = 45      # saatte en az bu kadar geçerli dakika yoksa saat "eksik" sayılır
    calib_days: int = 60                # kalibrasyon penceresi (test bloğundan hemen önce)
    quantiles: tuple = (0.05, 0.5, 0.95)  # %90'lık tahmin aralığı
    nominal_coverage: float = 0.90
    seeds: int = 5
    # LSTM
    lstm_hidden: int = 64
    lstm_epochs: int = 40
    lstm_batch: int = 32
    lstm_lr: float = 1e-3
    # GBM
    gbm_trees: int = 300
    # israf analizi
    waste_model: str = "Gradient Boosting"
    inject_sizes: tuple = (0.25, 0.5, 1.0, 2.0)   # kWh (4 ardışık saate eklenir)
    inject_events_per_fold: int = 40
    inject_hours: int = 4
    persistent_sizes: tuple = (0.25, 0.5, 1.0)    # kalıcı artış testi (saat başına eklenen miktar, veri birimiyle)
    country: str = "FR"                 # resmi tatil takvimi (FR, TR, PT ...)
    unit: str = "kWh"                   # hedef değişkenin birimi (EPİAŞ verisi MWh)
    bootstrap: int = 2000
    folds: list = field(default_factory=lambda: list(FOLDS))
