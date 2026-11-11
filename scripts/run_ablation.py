"""Özellik grubu ablasyonu (kararlar geliştirme bloklarında, kanıt son bloklarda). Kullanım: python scripts/run_ablation.py [--seeds N]"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from energyradar.ablation import run_ablation  # noqa: E402
from energyradar.config import DB_PATH, RESULTS_DIR, Config  # noqa: E402
from energyradar.models import GBM_BACKEND  # noqa: E402
from energyradar.pipeline import load_context  # noqa: E402
from energyradar.report import build_xlsx  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--out", default=str(RESULTS_DIR))
    a = ap.parse_args()
    cfg = replace(Config(), bootstrap=2000)
    _, _, ctx = load_context(a.db, cfg)
    t0 = time.time()
    r = run_ablation(ctx, cfg, seeds=range(a.seeds), progress=lambda f, l, s: print(f"[{f}] {l} tohum {s}"))
    for k, title in (("dev", "Geliştirme blokları (F1-F3): temel modele karşı"), ("decision", "Karar"),
                     ("final", "Son bloklar (F4-F5): hiç bakılmamış test")):
        print(f"\n=== {title} ===")
        print(r[k].round(4).to_string(index=False))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sheets = {"Geliştirme F1-F3": r["dev"], "Karar": r["decision"], "Son bloklar F4-F5": r["final"],
              "Blok bazında MAE": r["per_fold"], "Grup açıklamaları": r["groups"]}
    info = {"GBM arka ucu": GBM_BACKEND, "Tohum sayısı": a.seeds,
            "Karar kuralı": "Ek grup: geliştirme bloklarında MAE farkı %95 aralığı tamamen negatifse benimsenir. "
                            "Temel grup: çıkarınca fark aralığı 0'ı içeriyorsa çıkarılabilir. Asıl kanıt F4-F5."}
    (out / "EnerjiRadar_Ablasyon.xlsx").write_bytes(build_xlsx(sheets, info))
    print(f"\nBitti ({time.time() - t0:.0f} sn). Okumak için: {out / 'EnerjiRadar_Ablasyon.xlsx'}")


if __name__ == "__main__":
    main()
