"""Dış doğrulama: aynı yöntemi BAŞKA verilerde sınar.
  A) EPİAŞ (Türkiye, saatlik tüketim):  python scripts/run_external.py --dataset epias --file data/external/epias.csv
  B) Portekiz, 370 müşteri (UCI):       python scripts/run_external.py --dataset ld2011 --file data/external/electricityloaddiagrams20112014.zip
Her seri için ayrı Excel + hepsini özetleyen 'Dis_Dogrulama_*.xlsx' üretilir. Veri indirilmez; yalnızca verdiğiniz dosya okunur."""

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from energyradar.config import RESULTS_DIR, Config  # noqa: E402
from energyradar.datasets import auto_folds, load_epias, load_ld2011, scaled_config  # noqa: E402
from energyradar.models import MODEL_NAMES  # noqa: E402
from energyradar.pipeline import run_pipeline  # noqa: E402
from energyradar.report import build_xlsx  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def one_series(name, hourly, cfg, models, out_root, persistent, meta):
    cfg = scaled_config(cfg, hourly, unit=meta.get("birim", "kWh"))
    note = ""
    try:
        folds = auto_folds(hourly, block=meta["blocks"])
    except ValueError:
        # Veri ~1.5 yıldan kısa: en az 150 gün eğitim, aylık bloklar. Tek yılda mevsimlerin hepsi eğitimde görülmez, bu
        # yüzden sonuç daha kötümser/belirsiz olur ve ayrıca not edilir.
        folds = auto_folds(hourly, block="month", min_train_days=150, test_years=1)
        note = ("UYARI: veri 1.5 yıldan kısa. En az 150 gün eğitim + aylık bloklar kullanıldı; model her mevsimi eğitimde görmedi, "
                "bayram/tatil örnekleri az. Daha uzun veri (2+ yıl) daha güvenilirdir.")
        print(note)
    cfg = replace(cfg, folds=folds)
    r = run_pipeline(out_dir=out_root / name, cfg=cfg, models=models, hourly=hourly, persistent=persistent,
                     info_extra={"Seri": name, "Kısa veri notu": note or "yok", "Veri kaynağı bilgisi": json.dumps(meta, ensure_ascii=False, default=str),
                                 "Enjeksiyon büyüklükleri (ortalama düzeyin 0.2/0.5/1/2 katı)": str(cfg.inject_sizes)})
    return name, hourly, cfg, r


def summarize_all(results):
    rows, diffs = [], []
    for name, hourly, cfg, r in results:
        lvl = float(hourly["kwh"].mean())
        s = r["summary"].set_index("model")
        naive = s.loc["Seasonal Naive", "mae"]
        for m, row in s.iterrows():
            rows.append({"seri": name, "model": m, "mae": row["mae"], "wape": row["wape"], "ortalama_düzey": lvl,
                         "baseline'a göre MAE azalması %": 100 * (1 - row["mae"] / naive) if m != "Seasonal Naive" else 0.0,
                         "coverage_cal": row.get("coverage_cal"), "n_hours": row["n_hours"]})
        d = r["boot_diff"].copy()
        d.insert(0, "seri", name)
        diffs.append(d)
    summ = pd.DataFrame(rows)
    diff = pd.concat(diffs, ignore_index=True)
    ov = []
    for (a, b), g in diff.groupby(["model_a", "model_b"]):
        ov.append({"model_a": a, "model_b": b, "seri_sayısı": len(g), "a anlamlı iyi (CI<0)": int((g["ci_high"] < 0).sum()),
                   "a anlamlı kötü (CI>0)": int((g["ci_low"] > 0).sum()), "ayırt edilemez": int(((g["ci_low"] <= 0) & (g["ci_high"] >= 0)).sum())})
    return summ, diff, pd.DataFrame(ov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["epias", "ld2011"], required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--clients", type=int, default=5, help="ld2011: rastgele seçilecek müşteri sayısı")
    ap.add_argument("--client-names", default="", help="ld2011: elle müşteri adları, ör. MT_002,MT_050")
    ap.add_argument("--blocks", choices=["quarter", "month"], default="quarter")
    ap.add_argument("--models", default=",".join(MODEL_NAMES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--persistent", action="store_true", help="kalıcı artış testi (yavaş)")
    ap.add_argument("--out", default=str(RESULTS_DIR / "external"))
    a = ap.parse_args()

    cfg = Config(seeds=a.seeds)
    if a.quick:
        cfg = replace(cfg, seeds=2, lstm_epochs=10, bootstrap=500, inject_events_per_fold=20)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    out_root = Path(a.out)
    t0 = time.time()
    series, meta = {}, {}
    if a.dataset == "epias":
        h, meta = load_epias(a.file)
        print("EPİAŞ okundu:", json.dumps(meta, ensure_ascii=False))
        series["EPIAS_Turkiye"] = h
        cfg = replace(cfg, country="TR")
    else:
        names = [c.strip() for c in a.client_names.split(",") if c.strip()] or None
        series, meta = load_ld2011(a.file, a.clients, names)
        print("Portekiz verisi okundu:", json.dumps(meta, ensure_ascii=False, default=str))
        cfg = replace(cfg, country="PT")
    meta["blocks"] = a.blocks
    results = []
    for i, (name, h) in enumerate(series.items(), 1):
        print(f"\n=== [{i}/{len(series)}] {name} ===  ({len(h)} saat, ortalama düzey {h['kwh'].mean():.3g})")
        results.append(one_series(name, h, cfg, models, out_root, a.persistent, {**meta, "birim": meta.get("birim", "kWh")}))
    summ, diff, ov = summarize_all(results)
    info = {"Çalıştırma zamanı": datetime.now().isoformat(timespec="seconds"), "Veri": a.dataset, "Kaynak bilgisi": json.dumps(meta, ensure_ascii=False, default=str),
            "Tatil takvimi": cfg.country, "Test blokları": a.blocks, "Tohum": cfg.seeds, "Modeller": ", ".join(models),
            "Not": "Dış doğrulama: ana proje tek Fransız haneydi. Burada yöntem başka veride sınanır; sonuçlar ayrı raporlanır, ana sonuçla karıştırılmaz."}
    xlsx = out_root / f"Dis_Dogrulama_{a.dataset}.xlsx"
    xlsx.write_bytes(build_xlsx({"Özet": summ, "Model farkları": diff, "Genel tablo": ov}, info))
    print("\n=== Seri bazında özet ===")
    print(summ.round(3).to_string(index=False))
    print("\n=== Model farkları (hafta bloğu bootstrap) ===")
    print(diff.round(3).to_string(index=False))
    print(f"\nBitti ({time.time() - t0:.0f} sn). Özet dosyası: {xlsx}")


if __name__ == "__main__":
    main()
