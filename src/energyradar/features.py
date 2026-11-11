"""Özellikler. Görev: GÜN ÖNCESİ tahmin. Gün D'nin 24 saati, D günü 00:00'da (yalnızca D-1 23:00'a kadar bilinen
veriyle) tahmin edilir. Bu yüzden yalnızca şunlar kullanılır: takvim bilgisi (önceden bilinir), aynı saatin 24/48/168
saat öncesi değeri (hepsi D-1 içinde veya öncesinde) ve D-1'e kadar hesaplanmış günlük ortalamalar. Gelecek sızıntısı
testle kontrol edilir (tests/test_core.py)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

# Özellik grupları (ablasyon deneyi bu gruplar üzerinden çalışır).
#  calendar, lags, daily_means : temel (ilk sürümde kullanılan 10 özellik)
#  sub_lags   : alt sayaçların (mutfak, çamaşırhane, su ısıtıcı+klima) dünkü ve geçen haftaki aynı saat değeri
#  sub_daily  : alt sayaçların önceki gün ortalaması
#  rhythm     : kullanım ritmi. Alt sayacın 'kullanıldı' günleri (saatlik >= SESSION_KWH) arasından: son kullanımdan
#               bu yana gün, son 28 günde kaç gün kullanıldı, son 8 kullanım arasındaki medyan aralık (kaç günde bir),
#               gecikme oranı (bugüne dek geçen gün / alışılmış aralık). Yalnızca mutfak (sub1) ve çamaşırhane (sub2)
#               için; su ısıtıcı+klima (sub3) günlerin %97'sinde aktif olduğu için ritim taşımaz.
SESSION_KWH = 0.3      # saatlik 300 Wh üstü = gerçek bir cihaz çevrimi. Eşik, kullanım sıklığına bakılarak seçildi
                       # (mutfak: günlerin %76'sı, çamaşırhane: %43'ü), tahmin doğruluğuna bakılarak DEĞİL.
RHYTHM_SUBS = (1, 2)
GROUPS = {
    "calendar": ["hour", "dow", "month", "is_weekend", "is_holiday"],
    "lags": ["lag24", "lag48", "lag168"],
    "daily_means": ["prev_day_mean", "week_mean"],
    "sub_lags": [f"sub{k}_lag{l}" for k in (1, 2, 3) for l in (24, 168)],
    "sub_daily": [f"prev_day_sub{k}" for k in (1, 2, 3)],
    "rhythm": [f"{n}_{k}" for k in RHYTHM_SUBS for n in ("days_since_use", "use_days_28", "gap_med_days", "overdue_ratio")],
}
BASE_GROUPS = ["calendar", "lags", "daily_means"]


def columns_for(groups) -> list:
    return [c for g in groups for c in GROUPS[g]]


FEATURES = columns_for(BASE_GROUPS)


def easter(year: int) -> date:
    """Gregoryen Paskalya (anonim algoritma)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def french_holidays(years) -> set:
    """Fransa resmi tatilleri (veri Fransa'dan: Sceaux). Okul tatilleri dahil değildir (sınırlılık)."""
    out = set()
    for y in years:
        e = easter(y)
        out |= {date(y, 1, 1), date(y, 5, 1), date(y, 5, 8), date(y, 7, 14), date(y, 8, 15), date(y, 11, 1),
                date(y, 11, 11), date(y, 12, 25), e + timedelta(days=1), e + timedelta(days=39), e + timedelta(days=50)}
    return out


def holiday_dates(country: str, years) -> set:
    """Resmi tatil günleri. FR: kendi hesabımız (Paskalya algoritması). Diğer ülkeler (TR, PT, ...): `holidays` paketi
    (pip install holidays); dinî bayramlar gibi değişken tarihleri o paket yönetir. Bulunan tatil sayısı sonuç dosyasına yazılır."""
    years = list(years)
    if country.upper() == "FR":
        return french_holidays(years)
    try:
        import holidays as _h
    except ImportError as e:  # pragma: no cover - ortama bağlı
        raise ImportError(f"{country} resmi tatilleri için 'holidays' paketi gerekir: pip install holidays") from e
    return set(_h.country_holidays(country.upper(), years=years).keys())


def build_frame(hourly: pd.DataFrame, country: str = "FR") -> pd.DataFrame:
    """Saatlik tablodan (`kwh` sütunu, tam saatlik indeks) model çerçevesi. Hedef: y. `country`: resmi tatil takvimi."""
    idx = hourly.index
    y = hourly["kwh"].astype(float)
    f = pd.DataFrame(index=idx)
    f["y"] = y
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    hol = holiday_dates(country, range(idx.year.min(), idx.year.max() + 1))
    f["is_holiday"] = np.array([d in hol for d in idx.date]).astype(int)
    f["lag24"], f["lag48"], f["lag168"] = y.shift(24), y.shift(48), y.shift(168)
    daily = y.resample("D").mean()
    prev_day = daily.shift(1)
    week = daily.rolling(7, min_periods=4).mean().shift(1)
    day_key = idx.normalize()
    f["prev_day_mean"] = prev_day.reindex(day_key).values
    f["week_mean"] = week.reindex(day_key).values
    f["day"] = day_key
    # ---- alt sayaç özellikleri (hepsi gün öncesi bilinen veriden)
    for k in (1, 2, 3):
        col = f"sub{k}_kwh"
        if col not in hourly.columns:
            continue
        sub = hourly[col].astype(float)
        f[f"sub{k}_lag24"], f[f"sub{k}_lag168"] = sub.shift(24), sub.shift(168)
        f[f"prev_day_sub{k}"] = sub.resample("D").mean().shift(1).reindex(day_key).values
    for k in RHYTHM_SUBS:
        col = f"sub{k}_kwh"
        if col in hourly.columns:
            feats = rhythm_features(hourly[col].astype(float))
            for n in ("days_since_use", "use_days_28", "gap_med_days", "overdue_ratio"):
                f[f"{n}_{k}"] = feats[n].reindex(day_key).values
    return f


def rhythm_features(sub: pd.Series, thr: float = SESSION_KWH) -> pd.DataFrame:
    """Günlük 'kullanım ritmi'. Gün D için yalnızca D-1'e kadar olan günler kullanılır (sızıntı yok).
    Kullanım günü: o gün en az bir saatte alt sayaç >= thr kWh. NaN saatler kullanım sayılmaz."""
    used = (sub >= thr).groupby(sub.index.normalize()).any()
    days = used.index
    u = used.values
    n = len(u)
    out = {k: np.full(n, np.nan) for k in ("days_since_use", "use_days_28", "gap_med_days", "overdue_ratio")}
    use_idx = []
    for i in range(n):
        if use_idx:
            out["days_since_use"][i] = i - use_idx[-1]
            if len(use_idx) >= 4:
                gaps = np.diff(use_idx[-9:])
                med = float(np.median(gaps))
                out["gap_med_days"][i] = med
                out["overdue_ratio"][i] = (i - use_idx[-1]) / max(med, 1.0)
        out["use_days_28"][i] = u[max(0, i - 28):i].sum()
        if u[i]:
            use_idx.append(i)
    return pd.DataFrame(out, index=days)


def soft_forecast_days(f: pd.DataFrame, start: str, end: str) -> list:
    """[start, end] arasındaki günler (tam 24 saati çerçevede olanlar)."""
    days = pd.date_range(start, end, freq="D")
    have = set(f["day"].unique())
    return [d for d in days if d in have]
