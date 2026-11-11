"""Değerlendirme: yürüyen pencere (walk-forward), aralık kalibrasyonu (CQR), koşul kırılımı, bootstrap,
israf (beklenenden fazla tüketim) tespiti ve enjeksiyon testi."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Config
from .features import soft_forecast_days
from .models import make_model, MODEL_NAMES


# ---------------------------------------------------------------- aralık kalibrasyonu

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal: ceil((n+1)(1-alpha))/n yüzdeliği. n küçükse en büyük skor."""
    s = np.sort(np.asarray(scores, dtype=float)[~np.isnan(scores)])
    n = len(s)
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * (1 - alpha))
    return float(s[min(k, n) - 1])


def calibrate(calib: pd.DataFrame, test: pd.DataFrame, nominal: float) -> pd.DataFrame:
    """Kalibrasyon penceresi tahminlerinden (model eğitimine girmemiş) aralığı ayarlar.
    Aralığı olan model: CQR, skor = max(lo - y, y - hi). Aralığı olmayan (baseline): p50 ± |hata| yüzdeliği.
    NOT: zaman serisinde gözlemler değiştirilebilir (exchangeable) değildir; garanti yaklaşıktır, sapma ÖLÇÜLÜR."""
    alpha = 1 - nominal
    c = calib.dropna(subset=["y", "p50"])
    out = test.copy()
    if test["p05"].notna().all() and c["p05"].notna().all():
        qhat = conformal_quantile(np.maximum(c["p05"] - c["y"], c["y"] - c["p95"]).values, alpha)
        out["lo_cal"], out["hi_cal"] = out["p05"] - qhat, out["p95"] + qhat
    else:
        qhat = conformal_quantile((c["y"] - c["p50"]).abs().values, alpha)
        out["lo_cal"], out["hi_cal"] = out["p50"] - qhat, out["p50"] + qhat
    out["qhat"] = qhat
    return out


def calibrate_rolling(calib: pd.DataFrame, test: pd.DataFrame, nominal: float, window_days: int = 60) -> pd.DataFrame:
    """Kayan conformal: test gününün aralığı, yalnızca o günden ÖNCEKİ son `window_days` günün (kalibrasyon + önceki test
    günleri) skorlarından ayarlanır. Geçmiş gözlemler gün sonunda bilindiği için gün öncesi tahminde kullanılabilir.
    Mevsim değişimine uyum sağlar; statik kalibrasyonun zayıf kaldığı yer budur."""
    alpha = 1 - nominal
    both = pd.concat([calib, test]).sort_index()
    has_int = both["p05"].notna().all() and both["p95"].notna().all()
    score = (np.maximum(both["p05"] - both["y"], both["y"] - both["p95"]) if has_int else (both["y"] - both["p50"]).abs()).values
    day = both.index.normalize()
    out = test.copy()
    lo_hi = {}
    for d in pd.DatetimeIndex(test.index.normalize().unique()):
        m = (day >= d - pd.Timedelta(days=window_days)) & (day < d)
        q = conformal_quantile(score[m], alpha)
        if np.isnan(q):
            q = 0.0
        lo_hi[d] = q
    q = out.index.normalize().map(lo_hi).values.astype(float)
    if has_int:
        out["lo_roll"], out["hi_roll"] = out["p05"] - q, out["p95"] + q
    else:
        out["lo_roll"], out["hi_roll"] = out["p50"] - q, out["p50"] + q
    return out


# ---------------------------------------------------------------- metrikler

def point_metrics(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["y", "p50"])
    if d.empty:
        return {k: float("nan") for k in ("mae", "rmse", "mape", "wape", "n_hours")}
    e = d["p50"] - d["y"]
    return {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "mape": float((e.abs() / d["y"].clip(lower=1e-6)).mean() * 100),
            "wape": float(e.abs().sum() / d["y"].sum() * 100), "n_hours": int(len(d))}


def interval_metrics(df: pd.DataFrame) -> dict:
    out = {}
    for tag, lo, hi in (("raw", "p05", "p95"), ("cal", "lo_cal", "hi_cal"), ("roll", "lo_roll", "hi_roll")):
        if lo not in df.columns:
            continue
        d = df.dropna(subset=["y", lo, hi])
        out[f"coverage_{tag}"] = float(((d["y"] >= d[lo]) & (d["y"] <= d[hi])).mean()) if len(d) else float("nan")
        out[f"width_{tag}"] = float((d[hi] - d[lo]).mean()) if len(d) else float("nan")
    return out


def _season(m):
    return {12: "kış", 1: "kış", 2: "kış", 3: "ilkbahar", 4: "ilkbahar", 5: "ilkbahar", 6: "yaz", 7: "yaz", 8: "yaz"}.get(m, "sonbahar")


def condition_labels(idx: pd.DatetimeIndex, holiday: pd.Series) -> pd.DataFrame:
    hr = idx.hour
    return pd.DataFrame({
        "gün tipi": np.where(holiday.values == 1, "resmi tatil", np.where(idx.dayofweek >= 5, "hafta sonu", "hafta içi")),
        "mevsim": [_season(m) for m in idx.month],
        "günün saati": np.select([hr < 6, hr < 12, hr < 18], ["gece (0-5)", "sabah (6-11)", "öğleden sonra (12-17)"], "akşam (18-23)"),
    }, index=idx)


# ---------------------------------------------------------------- yürüyen pencere

def fold_days(ctx: pd.DataFrame, fold, cfg: Config):
    name, start, end = fold
    test_days = soft_forecast_days(ctx, start, end)
    calib_start = pd.Timestamp(start) - pd.Timedelta(days=cfg.calib_days)
    calib_days = soft_forecast_days(ctx, calib_start.strftime("%Y-%m-%d"), (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    train_days = [d for d in ctx["day"].unique() if d < calib_start]
    return train_days, calib_days, test_days


def run_fold(ctx: pd.DataFrame, fold, model_name: str, seed: int, cfg: Config, keep_model: bool = False):
    """Bir blok için: eğit (blok öncesi), kalibre et (blok öncesi son `calib_days`), test tahminleri döndür."""
    train_days, calib_days, test_days = fold_days(ctx, fold, cfg)
    model = make_model(model_name, cfg, seed).fit(ctx, train_days)
    cal = model.predict(ctx, calib_days).join(ctx["y"])
    te = model.predict(ctx, test_days).join(ctx[["y", "is_holiday"]])
    te_static = calibrate(cal, te, cfg.nominal_coverage)
    te = calibrate_rolling(cal, te_static, cfg.nominal_coverage)
    te.index.name = "ts"
    te.insert(0, "seed", seed)
    te.insert(0, "model", model_name)
    te.insert(0, "fold", fold[0])
    return (te, model) if keep_model else te


def run_all(ctx, cfg: Config, models=None, seeds=None, progress=None) -> pd.DataFrame:
    from .models import torch_available
    models = models or MODEL_NAMES
    seeds = seeds if seeds is not None else range(cfg.seeds)
    parts = []
    for fold in cfg.folds:
        for m in models:
            if m == "LSTM" and not torch_available():
                continue
            for sd in ([0] if m == "Seasonal Naive" else seeds):
                if progress:
                    progress(fold[0], m, sd)
                parts.append(run_fold(ctx, fold, m, sd, cfg))
    return pd.concat(parts)


def summarize(preds: pd.DataFrame) -> dict:
    """Her (fold, model, seed) için metrik tablosu ve model özeti."""
    rows = []
    for (f, m, s), g in preds.groupby(["fold", "model", "seed"], sort=False):
        rows.append({"fold": f, "model": m, "seed": s, **point_metrics(g), **interval_metrics(g)})
    detail = pd.DataFrame(rows)
    num = [c for c in detail.columns if c not in ("fold", "model", "seed")]
    per_seed = detail.groupby(["model", "seed"], sort=False)[num].mean().reset_index()
    summary = per_seed.groupby("model", sort=False)[num].mean().reset_index()
    spread = per_seed.groupby("model", sort=False)[["mae", "rmse", "wape"]].std(ddof=1).add_suffix("_seed_std").reset_index()
    return {"detail": detail, "summary": summary.merge(spread, on="model", how="left")}


def by_condition(preds: pd.DataFrame) -> pd.DataFrame:
    p = preds.rename_axis("ts").reset_index()
    ts = pd.DatetimeIndex(p["ts"])
    lab = condition_labels(ts, p["is_holiday"])
    p = pd.concat([p.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
    rows = []
    for col in ("gün tipi", "mevsim", "günün saati"):
        for (m, s, v), g in p.groupby(["model", "seed", col], sort=False):
            rows.append({"model": m, "seed": s, "koşul": col, "değer": v, **point_metrics(g), **interval_metrics(g)})
    d = pd.DataFrame(rows)
    num = [c for c in d.columns if c not in ("model", "seed", "koşul", "değer")]
    return d.groupby(["koşul", "değer", "model"], sort=False)[num].mean().reset_index()


# ---------------------------------------------------------------- bootstrap (hafta blokları)

def weekly_mae(preds: pd.DataFrame) -> pd.DataFrame:
    p = preds.dropna(subset=["y", "p50"]).copy()
    p["week"] = p.index.to_period("W").astype(str)
    p["ae"] = (p["p50"] - p["y"]).abs()
    w = p.groupby(["model", "seed", "week"])["ae"].mean().groupby(["model", "week"]).mean().unstack(0)
    return w.dropna()


def bootstrap_mae(preds: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    """Haftaları yerine koyarak yeniden örnekle (saatler birbirine bağlı; blok bootstrap). Model MAE'si ve çiftli
    fark için %95 aralık. Fark aralığı 0'ı içeriyorsa ayırt edilemez."""
    w = weekly_mae(preds)
    rng = np.random.default_rng(seed)
    models = list(w.columns)
    arr = w.values
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = arr[idx].mean(axis=1)          # (n_boot, n_models)
    rows = [{"model": m, "mae": float(arr[:, i].mean()), "ci_low": float(np.percentile(boot[:, i], 2.5)),
             "ci_high": float(np.percentile(boot[:, i], 97.5)), "n_weeks": len(arr)} for i, m in enumerate(models)]
    drows = []
    for i, a in enumerate(models):
        for j in range(i + 1, len(models)):
            d = boot[:, i] - boot[:, j]
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            drows.append({"model_a": a, "model_b": models[j], "mae_fark_a_eksi_b": float(arr[:, i].mean() - arr[:, j].mean()),
                          "ci_low": lo, "ci_high": hi, "ayırt edilebilir mi?": "evet" if (lo > 0 or hi < 0) else "hayır"})
    return pd.DataFrame(rows), pd.DataFrame(drows)


# ---------------------------------------------------------------- israf (beklenenden fazla tüketim)

def excess_hours(preds: pd.DataFrame, model: str, seed: int = 0) -> pd.DataFrame:
    """Gerçek tüketimin kalibre %90 aralığın ÜST sınırını aştığı saatler. Not: kalibrasyon yeterliyse aşım oranı
    yaklaşık %5'tir (tek yönlü); bu oranın üstü 'beklenenden fazla' sinyalidir, israfın kanıtı değildir."""
    d = preds[(preds["model"] == model) & (preds["seed"] == (seed if model != "Seasonal Naive" else 0))]
    d = d.dropna(subset=["y", "hi_cal"])
    e = d[d["y"] > d["hi_cal"]].copy()
    e["excess_kwh"] = e["y"] - e["hi_cal"]
    return e[["fold", "model", "seed", "y", "hi_cal", "excess_kwh"]]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / den), min(1.0, (c + r) / den))


def injection_test(preds: pd.DataFrame, model: str, cfg: Config, seed: int = 0, rng_seed: int = 0) -> pd.DataFrame:
    """Bilinen büyüklükte yapay fazla tüketim (cfg.inject_hours ardışık saate +X kWh) ekleyip kaç tanesinin yakalandığını
    ölçer. Tahminler temiz veriyle yapılmıştır; ekleme yalnızca GÖZLENEN değere uygulanır (tek seferlik aşım).
    Taban yanlış alarm: eklenmemiş saatlerin aşım oranı."""
    d = preds[(preds["model"] == model) & (preds["seed"] == (seed if model != "Seasonal Naive" else 0))]
    rng = np.random.default_rng(rng_seed)
    rows = []
    for fold, g in d.groupby("fold", sort=False):
        g = g.dropna(subset=["y", "hi_cal"])
        y, hi = g["y"].values, g["hi_cal"].values
        base = float((y > hi).mean())
        H = cfg.inject_hours
        starts = rng.choice(np.arange(0, len(g) - H), size=cfg.inject_events_per_fold, replace=False)
        for size in cfg.inject_sizes:
            hit = sum(bool(((y[s:s + H] + size) > hi[s:s + H]).any()) for s in starts)
            lo, up = wilson(hit, len(starts))
            rows.append({"fold": fold, "model": model, "bump_kwh": size, "events": len(starts), "detected": hit,
                         "detection_rate": hit / len(starts), "ci_low": lo, "ci_high": up, "base_exceedance_rate": base})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ensemble

def add_ensemble(preds: pd.DataFrame, members=("Gradient Boosting", "LSTM"), name="Ensemble GB+LSTM") -> pd.DataFrame:
    """Üyelerin tahmin ve aralık sütunlarının ortalaması (aynı fold, tohum ve saat için). Üyelerin hepsi yoksa eklemez."""
    if not set(members) <= set(preds["model"]):
        return preds
    cols = [c for c in ("p05", "p50", "p95", "lo_cal", "hi_cal", "lo_roll", "hi_roll", "qhat") if c in preds.columns]
    a = preds[preds["model"].isin(members)].rename_axis("ts").reset_index()
    g = a.groupby(["fold", "seed", "ts"])
    out = g[cols].mean()
    out["y"] = g["y"].first()
    out["is_holiday"] = g["is_holiday"].first()
    out["n"] = g.size()
    out = out[out["n"] == len(members)].drop(columns="n").reset_index()
    out["model"] = name
    out = out.set_index("ts")
    for c in preds.columns:          # üyelerde olmayan ek sütunlar (ör. ileride eklenenler) çökertmesin
        if c not in out.columns:
            out[c] = np.nan
    return pd.concat([preds, out[preds.columns]])


# ---------------------------------------------------------------- koşul bazlı anlamlılık

def condition_bootstrap(preds: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Model a ile b'nin MAE farkı, her koşul değeri (hafta içi, kış, gece...) için hafta bloğu bootstrap'iyle.
    Koşul bazında bu soruyu cevaplar: 'LSTM kışın gerçekten daha mı iyi, yoksa şans mı?'"""
    p = preds[preds["model"].isin([a, b])].copy()
    p = p.dropna(subset=["y", "p50"]).rename_axis("ts").reset_index()
    lab = condition_labels(pd.DatetimeIndex(p["ts"]), p["is_holiday"]).reset_index(drop=True)
    p = pd.concat([p.reset_index(drop=True), lab], axis=1)
    p["week"] = pd.DatetimeIndex(p["ts"]).to_period("W").astype(str)
    p["ae"] = (p["p50"] - p["y"]).abs()
    rng = np.random.default_rng(seed)
    rows = []
    for col in ("gün tipi", "mevsim", "günün saati"):
        for val, g in p.groupby(col):
            w = g.groupby(["model", "seed", "week"])["ae"].mean().groupby(["model", "week"]).mean().unstack(0).dropna()
            if len(w) < 8 or a not in w or b not in w:
                continue
            d = (w[a] - w[b]).values
            idx = rng.integers(0, len(d), size=(n_boot, len(d)))
            bs = d[idx].mean(axis=1)
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            rows.append({"koşul": col, "değer": val, "model_a": a, "model_b": b, "n_hafta": len(d),
                         "MAE farkı (a - b)": float(d.mean()), "ci_low": lo, "ci_high": hi,
                         "ayırt edilebilir mi?": "evet" if (lo > 0 or hi < 0) else "hayır"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- israf: alarm kuralları ve kalıcı artış

def _has_run(flags: np.ndarray, k: int) -> bool:
    run = 0
    for f in flags:
        run = run + 1 if f else 0
        if run >= k:
            return True
    return False


def injection_rules(preds: pd.DataFrame, model: str, cfg: Config, seed: int = 0, rng_seed: int = 0, interval: str = "hi_cal") -> pd.DataFrame:
    """Tek seferlik 4 saatlik fazla tüketimi iki alarm kuralıyla ölçer: (A) 4 saat içinde herhangi bir saat aralığı aşarsa,
    (B) en az 2 ARDIŞIK saat aşarsa. Taban: hiçbir şey eklenmemiş 4 saatlik pencerelerin alarm oranı (yanlış alarm)."""
    d = preds[(preds["model"] == model) & (preds["seed"] == (seed if model != "Seasonal Naive" else 0))]
    rng = np.random.default_rng(rng_seed)
    H, rows = cfg.inject_hours, []
    for fold, g in d.groupby("fold", sort=False):
        g = g.dropna(subset=["y", interval])
        y, hi = g["y"].values, g[interval].values
        n = len(g) - H
        clean = [(y[i:i + H] > hi[i:i + H]) for i in range(0, n, 1)]
        base_a = float(np.mean([c.any() for c in clean]))
        base_b = float(np.mean([_has_run(c, 2) for c in clean]))
        starts = rng.choice(np.arange(0, n), size=cfg.inject_events_per_fold, replace=False)
        for size in cfg.inject_sizes:
            fl = [(y[s:s + H] + size) > hi[s:s + H] for s in starts]
            ka, kb = sum(f.any() for f in fl), sum(_has_run(f, 2) for f in fl)
            for rule, k, base in (("herhangi 1 saat aşım", ka, base_a), ("en az 2 ardışık saat aşım", kb, base_b)):
                lo, up = wilson(int(k), len(starts))
                rows.append({"fold": fold, "model": model, "aralık": interval, "kural": rule, "bump_kwh": size, "events": len(starts),
                             "detection_rate": k / len(starts), "ci_low": lo, "ci_high": up,
                             "yanlis_alarm_tabani_4saatlik_pencere": base})
    return pd.DataFrame(rows)


def persistent_injection(hourly: pd.DataFrame, ctx: pd.DataFrame, fold, cfg: Config, seed: int = 0, n_events: int = 12,
                         days: int = 7, hours=(18, 19, 20, 21, 22), sizes=None) -> pd.DataFrame:
    """KALICI artış: `days` gün boyunca her akşam `hours` saatlerine +X kWh eklenir (ör. unutulan ısıtıcı). Tahminler gerçek
    şekilde YENİDEN hesaplanır (artış gecikmeli özelliklere de girer, model uyum sağlamaya çalışır). Aralık: statik kalibrasyon.
    Alarm günü: o günün aralık üstü saatlerinde en az 2 ardışık aşım. Ölçülen: 'days' gün içinde alarm verilme oranı ve ilk
    alarmın gecikmesi (gün). Taban: temiz günlerde alarm verilen gün oranı."""
    from .features import build_frame
    from .models import make_model
    sizes = sizes or cfg.persistent_sizes
    train_days, calib_days, test_days = fold_days(ctx, fold, cfg)
    model = make_model("Gradient Boosting", cfg, seed).fit(ctx, train_days)
    cal = model.predict(ctx, calib_days).join(ctx["y"])
    base = calibrate(cal, model.predict(ctx, test_days).join(ctx[["y", "is_holiday"]]), cfg.nominal_coverage)
    qhat = float(base["qhat"].iloc[0])

    def alarm_days(ctx2, ds):
        p = model.predict(ctx2, ds).join(ctx2["y"])
        p["hi"] = p["p95"] + qhat
        out = {}
        for d, g in p.groupby(p.index.normalize()):
            g = g.dropna(subset=["y", "hi"])
            out[d] = _has_run((g["y"] > g["hi"]).values, 2) if len(g) else False
        return out

    clean = alarm_days(ctx, test_days)
    base_rate = float(np.mean(list(clean.values())))
    rng = np.random.default_rng(seed)
    cand = [d for d in test_days if d + pd.Timedelta(days=days) <= test_days[-1]]
    starts = list(rng.choice(len(cand), size=min(n_events, len(cand)), replace=False))
    rows = []
    for size in sizes:
        det, delays = 0, []
        for si in starts:
            s0 = cand[si]
            h2 = hourly.copy()
            m = (h2.index >= s0) & (h2.index < s0 + pd.Timedelta(days=days)) & np.isin(h2.index.hour, hours)
            h2.loc[m, "kwh"] = h2.loc[m, "kwh"] + size
            ctx2 = build_frame(h2, cfg.country)
            win = [s0 + pd.Timedelta(days=i) for i in range(days)]
            al = alarm_days(ctx2, win)
            first = next((i for i, d in enumerate(win) if al.get(d)), None)
            if first is not None:
                det += 1
                delays.append(first)
        lo, up = wilson(det, len(starts))
        rows.append({"fold": fold[0], "bump_kwh_per_hour": size, "days": days, "events": len(starts), "detected": det,
                     "detection_rate": det / len(starts), "ci_low": lo, "ci_high": up,
                     "mean_delay_days": float(np.mean(delays)) if delays else float("nan"),
                     "clean_alarm_day_rate": base_rate})
    return pd.DataFrame(rows)
