"""Tüm deneyleri çalıştırır. Kullanım: python scripts/run_experiments.py [--quick] [--seeds N] [--models "..."]"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from energyradar.config import DB_PATH, RESULTS_DIR, Config, monthly_folds  # noqa: E402
from energyradar.models import MODEL_NAMES  # noqa: E402
from energyradar.pipeline import run_pipeline  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_NAMES))
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="2 tohum, 10 LSTM epoch")
    ap.add_argument("--folds", choices=["seasonal", "monthly"], default="seasonal",
                    help="seasonal: 5 mevsimlik blok (varsayılan, LSTM dahil); monthly: ~34 aylık blok (yalnız baseline + GB, güven aralıklarını daraltır)")
    ap.add_argument("--with-lstm", action="store_true", help="aylık modda da LSTM çalıştır (çok uzun sürer)")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--out", default=str(RESULTS_DIR))
    a = ap.parse_args()
    cfg = Config()
    if a.quick:
        cfg = replace(cfg, seeds=2, lstm_epochs=10, bootstrap=500, inject_events_per_fold=20)
    if a.seeds:
        cfg = replace(cfg, seeds=a.seeds)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    out_dir = a.out
    monthly = a.folds == "monthly"
    if monthly:
        cfg = replace(cfg, folds=monthly_folds())
        if not a.with_lstm:
            models = [m for m in models if m != "LSTM"]
        if out_dir == str(RESULTS_DIR):
            out_dir = str(RESULTS_DIR / "monthly")
    t0 = time.time()
    r = run_pipeline(a.db, out_dir, cfg, models, write_db=not monthly, persistent=not monthly)
    a.out = out_dir
    print("\n=== Model özeti ===")
    print(r["summary"].round(3).to_string(index=False))
    print("\n=== MAE güven aralığı ===")
    print(r["boot_ci"].round(3).to_string(index=False))
    print("\n=== Model farkları ===")
    print(r["boot_diff"].round(3).to_string(index=False))
    print(f"\nBitti ({time.time() - t0:.0f} sn). Okumak için açın: {Path(a.out) / 'EnerjiRadar_Sonuclar.xlsx'}")


if __name__ == "__main__":
    main()
