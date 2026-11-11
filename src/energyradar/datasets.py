"""Dış doğrulama verileri. Ana proje tek bir Fransız haneyle (UCI) kuruldu; burada aynı yöntem BAŞKA verilerde sınanır.

  A) EPİAŞ Şeffaflık Platformu: Türkiye saatlik gerçek zamanlı elektrik tüketimi (kullanıcının indirdiği CSV/Excel).
  B) UCI 'ElectricityLoadDiagrams20112014' (CC BY 4.0): Portekiz'de 370 müşteri, 15 dakikalık kW (LD2011_2014.txt).
     Not: veri setinde müşterilerin hane mi işyeri mi olduğu belirtilmez; bu yüzden 'hane' diye adlandırılmaz.

Her okuyucu aynı çıktıyı üretir: tam saatlik indeksli DataFrame, `kwh` sütunu (hedef; birimi `unit`), eksik saatler NaN
(doldurulmaz). Hiçbir veri üretilmez veya indirilmez; yalnızca kullanıcının verdiği dosya okunur.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- ortak yardımcılar

def _fold(s: str) -> str:
    tr = str.maketrans("İıŞşĞğÜüÖöÇç", "iissgguuoocc")
    return str(s).translate(tr).lower().strip()


def to_number(col: pd.Series) -> pd.Series:
    """Sayı sütunu. '35.123,45' (Türkçe), '35123,45', '35123.45' ve gerçek sayılar desteklenir."""
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float)
    s = col.astype(str).str.strip().str.replace(r"\s", "", regex=True)
    has_dot, has_comma = s.str.contains(r"\.", regex=True), s.str.contains(",", regex=False)
    both = has_dot & has_comma
    s = s.where(~both, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    only_comma = has_comma & ~has_dot
    s = s.where(~only_comma, s.str.replace(",", ".", regex=False))
    thousands = s.str.fullmatch(r"\d{1,3}(\.\d{3})+") & ~both     # '35.123' -> otuz beş bin yüz yirmi üç
    s = s.where(~thousands, s.str.replace(".", "", regex=False))
    return pd.to_numeric(s, errors="coerce")


def to_hourly(s: pd.Series, name: str = "kwh", how: str = "mean") -> pd.DataFrame:
    """Düzensiz/alt-saatlik seriyi tam saatlik indekse oturtur. Eksik saatler NaN kalır."""
    s = s[~s.index.isna()].sort_index()
    g = s.resample("h")
    h = g.mean() if how == "mean" else g.sum(min_count=1)
    full = pd.date_range(h.index.min(), h.index.max(), freq="h")
    out = h.reindex(full).to_frame(name)
    out.index.name = "ts"
    return out


def describe(hourly: pd.DataFrame) -> dict:
    v = hourly["kwh"]
    return {"rows_hours": int(len(v)), "hours_valid": int(v.notna().sum()), "missing_pct": round(float(v.isna().mean() * 100), 3),
            "first_hour": str(v.index.min()), "last_hour": str(v.index.max()), "mean_level": round(float(v.mean()), 4)}


# ---------------------------------------------------------------- A) EPİAŞ

def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1254", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def _dayfirst(d: pd.Series) -> bool:
    """'2023-01-02' (ISO) ay-gün sırasıyla okunmalı; '02.01.2023' (Türkçe) gün-ay. dayfirst=True ISO'yu bozar."""
    return not d.astype(str).str.match(r"^\s*\d{4}-\d{2}-\d{2}").mean() > 0.5


def load_epias(path) -> tuple[pd.DataFrame, dict]:
    """EPİAŞ Şeffaflık Platformu'ndan indirilen saatlik tüketim dosyası (.csv/.xlsx). Sütunlar adlarından bulunur:
    tarih (+ ayrı 'saat' sütunu varsa o), tüketim (MWh). Bulamazsa bulduğu sütunları yazıp durur (tahmin yürütmez)."""
    path = Path(path)
    df = _read_table(path)
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    cols = {c: _fold(c) for c in df.columns}
    date_c = next((c for c, f in cols.items() if "tarih" in f or "date" in f), None)
    hour_c = next((c for c, f in cols.items() if f in ("saat", "hour", "time") or f.startswith("saat")), None)
    val_c = next((c for c, f in cols.items() if any(k in f for k in ("tuketim", "consumption", "load", "yuk"))), None)
    if date_c is None or val_c is None:
        raise ValueError(f"EPİAŞ dosyasında tarih/tüketim sütunu bulunamadı. Bulunan sütunlar: {list(df.columns)}. "
                         f"Bu listeyi bana gönderin, okuyucuyu o sütunlara göre ayarlarım.")
    d = df[date_c]
    if hour_c is not None and hour_c != date_c:
        ts = pd.to_datetime(d.astype(str).str.slice(0, 10) + " " + df[hour_c].astype(str).str.slice(0, 5),
                            dayfirst=_dayfirst(d), errors="coerce")
    else:
        ts = pd.to_datetime(d, dayfirst=_dayfirst(d), errors="coerce", utc=False)
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    ok = ts.notna()
    if ok.mean() < 0.9:
        raise ValueError(f"Tarih sütunu ('{date_c}') çözülemedi (geçerli oran {ok.mean():.0%}). İlk değerler: {list(d.head(3))}")
    val = to_number(df[val_c])
    s = pd.Series(val.values, index=pd.DatetimeIndex(ts))[ok.values]
    s = s.groupby(level=0).mean()                      # çift kayıt (ör. saat dilimi geçişi) -> ortalama
    hourly = to_hourly(s)
    meta = {"kaynak": path.name, "tarih_sütunu": date_c, "saat_sütunu": hour_c, "tüketim_sütunu": val_c, "birim": "MWh (EPİAŞ)",
            **describe(hourly)}
    return hourly, meta


# ---------------------------------------------------------------- B) LD2011_2014 (Portekiz, 370 müşteri)

def _open_ld(path: Path):
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        name = next(n for n in zf.namelist() if n.endswith("LD2011_2014.txt") and "__MACOSX" not in n)
        return zf.open(name)
    return open(path, "rb")


def _dst_hours(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Veri setinin belgelediği saat değişimi günleri: mart (son pazar) 01:00-02:00 hepsi 0; ekim 01:00-02:00 iki saat birleşik."""
    out = []
    for y in range(idx.year.min(), idx.year.max() + 1):
        for m in (3, 10):
            last = pd.Timestamp(y, m, 31)
            while last.dayofweek != 6:
                last -= pd.Timedelta(days=1)
            out.append(last + pd.Timedelta(hours=1))
    return pd.DatetimeIndex(out)


def read_ld2011_all(path, progress=print) -> tuple[pd.DataFrame, str]:
    """Tüm müşterilerin SAATLİK ortalama kW'ı (= saatlik kWh). 15 dk okumalar: zaman damgası aralığın sonuysa (ilk satır
    dakika=15) bir önceki saate yazılır. Dönüş: (saatlik tablo, zaman damgası yorumu)."""
    path = Path(path)
    parts, mode = [], None
    with _open_ld(path) as fh:
        for i, ch in enumerate(pd.read_csv(fh, sep=";", decimal=",", index_col=0, parse_dates=True, chunksize=40000)):
            if mode is None:
                mode = "aralık sonu (ilk okuma :15)" if ch.index[0].minute == 15 else "aralık başı (ilk okuma :00)"
            t = ch.index - pd.Timedelta(minutes=15) if mode.startswith("aralık sonu") else ch.index
            key = t.floor("h")
            g = ch.groupby(key)
            parts.append((g.sum().astype("float64"), g.size().astype("int64")))
            progress(f"  okunan satır ≈ {(i + 1) * 40000:,}")
    sums = pd.concat([p[0] for p in parts]).groupby(level=0).sum()
    cnt = pd.concat([p[1] for p in parts]).groupby(level=0).sum()
    hourly = sums.div(cnt, axis=0)                      # saat içi ortalama kW = o saatin kWh'si
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full)
    hourly.index.name = "ts"
    return hourly, mode


def clean_ld_client(s: pd.Series) -> pd.Series:
    """Belgelenmiş veri sorunları: müşteri kaydı açılmadan önceki sıfırlar ve saat değişimi saatleri eksik (NaN) sayılır."""
    s = s.copy()
    nz = s[s > 0]
    if len(nz):
        s[s.index < nz.index[0]] = np.nan
    bad = _dst_hours(s.index)
    s[s.index.isin(bad)] = np.nan
    return s


def eligible_clients(hourly: pd.DataFrame, max_start_days: int = 14, max_zero_frac: float = 0.01) -> list:
    """Tam geçmişi olan müşteriler: ilk sıfırdan büyük okuma ilk `max_start_days` içinde ve sıfır saat oranı < %1.
    Seçim KRİTERİ doğruluğa bakmaz (yalnızca veri kalitesi)."""
    out = []
    for c in hourly.columns:
        s = hourly[c]
        nz = s[s > 0]
        if not len(nz) or (nz.index[0] - s.index[0]).days > max_start_days:
            continue
        if float((clean_ld_client(s).dropna() == 0).mean()) < max_zero_frac:
            out.append(c)
    return sorted(out)


def pick_clients(hourly: pd.DataFrame, n: int = 5, seed: int = 0, names: list | None = None) -> list:
    """Müşteri seçimi: `names` verildiyse onlar; yoksa uygun müşteriler arasından tohumlu RASTGELE n tane (elle seçim yok)."""
    if names:
        miss = [c for c in names if c not in hourly.columns]
        if miss:
            raise ValueError(f"Veride olmayan müşteri(ler): {miss}")
        return list(names)
    el = eligible_clients(hourly)
    if len(el) < n:
        raise ValueError(f"Uygun müşteri sayısı ({len(el)}) istenenden az ({n}).")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(el, size=n, replace=False).tolist())


def load_ld2011(path, n_clients: int = 5, clients: list | None = None, seed: int = 0, progress=print):
    """Dönüş: ({müşteri: saatlik DataFrame}, meta)."""
    allh, mode = read_ld2011_all(path, progress)
    chosen = pick_clients(allh, n_clients, seed, clients)
    out = {c: to_hourly_from_series(clean_ld_client(allh[c])) for c in chosen}
    meta = {"kaynak": Path(path).name, "zaman damgası": mode, "müşteri_sayısı_veride": int(allh.shape[1]),
            "uygun_müşteri": len(eligible_clients(allh)), "seçilen": chosen, "seçim": "adlar verildi" if clients else f"tohum={seed}, rastgele",
            "birim": "kWh (saatlik ortalama kW)", "lisans": "CC BY 4.0 (UCI)"}
    return out, meta


def to_hourly_from_series(s: pd.Series) -> pd.DataFrame:
    out = s.to_frame("kwh")
    out.index.name = "ts"
    return out


# ---------------------------------------------------------------- dış doğrulama ayarları

def auto_folds(hourly: pd.DataFrame, test_years: int = 2, block: str = "quarter", min_train_days: int = 365):
    """Verinin son `test_years` yılını (en az `min_train_days` eğitim kalacak şekilde) çeyrek/ay bloklarına böler.
    Bloklar Config.folds biçimindedir: (ad, başlangıç, bitiş)."""
    v = hourly["kwh"].dropna()
    first, last = v.index.min().normalize(), v.index.max().normalize()
    start = max(first + pd.Timedelta(days=min_train_days), last - pd.Timedelta(days=365 * test_years))
    freq = "Q" if block == "quarter" else "M"
    p0 = pd.Period(start, freq=freq)
    if p0.start_time < start:
        p0 += 1
    folds = []
    for p in pd.period_range(p0, pd.Period(last, freq=freq), freq=freq):
        e = min(p.end_time.normalize(), last)
        if (e - p.start_time.normalize()).days >= 20:      # çok kısa son blok alınmaz
            folds.append((f"{'Ç' if block == 'quarter' else 'M'}{p}", p.start_time.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    if not folds:
        raise ValueError("Test bloğu çıkarılamadı: veri en az ~1.5 yıl olmalı.")
    return folds


def scaled_config(cfg, hourly: pd.DataFrame, factors=(0.2, 0.5, 1.0, 2.0), unit="kWh", country=None):
    """Enjeksiyon büyüklüklerini serinin ortalama düzeyine göre ölçekler (UCI'de ortalama ≈1.1 kWh idi). Yoksa MWh gibi
    farklı ölçekli verilerde 0.25 'birim' anlamsız olurdu."""
    from dataclasses import replace
    lvl = float(hourly["kwh"].mean())
    sizes = tuple(float(f"{f * lvl:.4g}") for f in factors)
    return replace(cfg, inject_sizes=sizes, persistent_sizes=sizes[:3], unit=unit, country=country or cfg.country)
