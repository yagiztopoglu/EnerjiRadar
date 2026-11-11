"""Tüm deneyi çalıştırır: tahminler, metrikler, bootstrap, israf analizi (SQL), enjeksiyon testi, Excel raporu."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import DB_PATH, RESULTS_DIR, Config
from .data import connect, data_report, read_hourly
from .datasets import describe
from .evaluation import (add_ensemble, bootstrap_mae, by_condition, condition_bootstrap, excess_hours, injection_rules,
                         injection_test, persistent_injection, run_all, summarize)
from .features import build_frame
from .models import GBM_BACKEND, MODEL_NAMES, torch_available
from .queries import QUERIES, WASTE_QUERIES
from .report import build_xlsx


def load_context(db_path=DB_PATH, cfg: Config | None = None):
    cfg = cfg or Config()
    con = connect(db_path)
    hourly = read_hourly(con)
    return con, hourly, build_frame(hourly, cfg.country)


def run_pipeline(db_path=DB_PATH, out_dir=RESULTS_DIR, cfg: Config | None = None, models=None, progress=print,
                 write_db: bool = True, persistent: bool = True, hourly: pd.DataFrame | None = None,
                 info_extra: dict | None = None) -> dict:
    """`hourly` verilirse (dış doğrulama: EPİAŞ, Portekiz müşterileri) SQLite kullanılmaz; veri doğrudan bu tablodan gelir."""
    cfg = cfg or Config()
    models = models or MODEL_NAMES
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hourly is None:
        con, hourly, ctx = load_context(db_path, cfg)
    else:
        con, ctx, write_db = None, build_frame(hourly, cfg.country), False
    skipped = [m for m in models if m == "LSTM" and not torch_available()]
    if skipped:
        progress("UYARI: PyTorch yok, LSTM atlandı (setup.bat ile kurulur).")

    preds = run_all(ctx, cfg, models, progress=lambda f, m, s: progress(f"[{f}] {m} tohum {s}"))
    preds = add_ensemble(preds)
    s = summarize(preds)
    cond = by_condition(preds)
    boot_ci, boot_diff = bootstrap_mae(preds, cfg.bootstrap)

    names = set(preds["model"])
    pairs = [(a, b) for a, b in (("Gradient Boosting", "LSTM"), ("Gradient Boosting", "Seasonal Naive"),
                                 ("Ensemble GB+LSTM", "Gradient Boosting")) if a in names and b in names]
    cond_sig = pd.concat([condition_bootstrap(preds, a, b, cfg.bootstrap) for a, b in pairs], ignore_index=True) if pairs else pd.DataFrame()

    # israf: SQL'e yaz, SQL ile özetle
    waste_model = cfg.waste_model if cfg.waste_model in set(preds["model"]) else preds["model"].iloc[0]
    exc = excess_hours(preds, waste_model)
    exc_db = exc.rename_axis("hour_ts").reset_index()
    exc_db["hour_ts"] = pd.to_datetime(exc_db["hour_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    fh = pd.DataFrame({"hour_ts": preds[preds["model"] == waste_model].index.unique().strftime("%Y-%m-%d %H:%M:%S")})
    fc = preds.rename_axis("hour_ts").reset_index()
    fc["hour_ts"] = fc["hour_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    waste_sql = {}
    if write_db:
        with connect(db_path) as wcon:
            exc_db.to_sql("excess", wcon, if_exists="replace", index=False)
            fh.to_sql("forecast_hours", wcon, if_exists="replace", index=False)
            fc.to_sql("forecasts", wcon, if_exists="replace", index=False)
        waste_sql = {n: pd.read_sql(q, con) for n, q in WASTE_QUERIES.items()}
    inj = injection_test(preds, waste_model, cfg)
    inj_rules = pd.concat([injection_rules(preds, waste_model, cfg, interval=iv) for iv in ("hi_cal", "hi_roll")], ignore_index=True)
    pers = pd.DataFrame()
    if persistent and waste_model == "Gradient Boosting":
        progress("Kalıcı artış testi (her blokta yeniden tahmin)...")
        pers = pd.concat([persistent_injection(hourly, ctx, f, cfg) for f in cfg.folds], ignore_index=True)
    exp_rate = float((preds[(preds["model"] == waste_model) & (preds["seed"] == 0)].eval("y > hi_cal")).mean())

    info = {"Çalıştırma zamanı": datetime.now().isoformat(timespec="seconds"),
            "Ayarlar (config)": json.dumps({k: v for k, v in asdict(cfg).items() if k != "folds"}),
            "Test blokları": "; ".join(f"{n}: {a}..{b}" for n, a, b in cfg.folds),
            "GBM arka ucu": GBM_BACKEND, "Modeller": ", ".join(sorted(set(preds["model"]))),
            "LSTM çalıştı mı": ("evet" if "LSTM" in set(preds["model"]) else
                                ("hayır (bu koşuda istenmedi)" if "LSTM" not in models else "hayır (PyTorch yok)")),
            "Birim (hedef)": cfg.unit, "Resmi tatil takvimi": cfg.country,
            "Veri raporu": json.dumps(data_report(con) if con is not None else describe(hourly)),
            "İsraf modeli": waste_model, "Aralık aşım oranı (israf modeli, beklenen ~0.05)": round(exp_rate, 4),
            "Not": "Aşım = beklenenden fazla tüketim; israfın kanıtı değildir. "
                   + ("Veri: tek hane, Fransa, 2006-2010." if con is not None else "Dış doğrulama verisi (ana proje verisi değil).")}
    info.update(info_extra or {})
    sheets = {"Model özeti": s["summary"], "Blok bazında": s["detail"], "Koşullara göre": cond,
              "Güven aralığı (MAE)": boot_ci, "Model farkları (MAE)": boot_diff,
              "Enjeksiyon testi": inj, "Enjeksiyon kuralları": inj_rules}
    if len(cond_sig):
        sheets["Koşul anlamlılık"] = cond_sig
    if len(pers):
        sheets["Kalıcı artış"] = pers
    for i, (n, d) in enumerate(waste_sql.items(), 1):
        sheets[f"İsraf SQL {i}"] = d
    if waste_sql:
        sheets["SQL sorguları"] = pd.DataFrame({"Sayfa": [f"İsraf SQL {i}" for i in range(1, len(waste_sql) + 1)],
                                                "Ne soruyor?": list(WASTE_QUERIES), "SQL": [q.strip() for q in WASTE_QUERIES.values()]})
    for name, df in {"summary": s["summary"], "detail": s["detail"], "conditions": cond, "bootstrap_mae": boot_ci,
                     "bootstrap_diff": boot_diff, "injection": inj, "injection_rules": inj_rules, "condition_significance": cond_sig,
                     "persistent": pers}.items():
        if not len(df):
            continue
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "EnerjiRadar_Sonuclar.xlsx").write_bytes(build_xlsx(sheets, info))
    return {"preds": preds, "summary": s["summary"], "detail": s["detail"], "conditions": cond,
            "boot_ci": boot_ci, "boot_diff": boot_diff, "injection": inj, "injection_rules": inj_rules,
            "cond_sig": cond_sig, "persistent": pers, "waste_sql": waste_sql, "info": info}
