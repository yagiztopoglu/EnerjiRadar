"""Veri katmanı: UCI 'Individual household electric power consumption' -> SQLite.

SQL burada gerçek bir iş yapar: dakikalık ham ölçümler `raw_minute` tablosunda durur, saatlik/günlük/haftalık
toplulaştırmalar VIEW olarak SQL ile hesaplanır, Python yalnızca view'ı okur.

Birim: Global_active_power kW (dakika ortalaması). Saatlik enerji (kWh) = saat içindeki geçerli dakikaların
ortalama kW'ı x 1 saat. Saatte 45'ten az geçerli dakika varsa o saat EKSİK sayılır (doldurulmaz).
Veri: Hebrail & Berard (2012), UCI ML Repository (lisans: CC BY 4.0, UCI sayfasından doğrulayın).
"""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DB_PATH, Config

TXT = "household_power_consumption.txt"

SCHEMA = """
DROP VIEW IF EXISTS v_hourly; DROP VIEW IF EXISTS v_daily; DROP VIEW IF EXISTS v_weekly;
CREATE TABLE IF NOT EXISTS raw_minute (
    ts TEXT PRIMARY KEY, active_kw REAL, reactive_kvar REAL, voltage REAL, intensity REAL,
    sub1_wh REAL, sub2_wh REAL, sub3_wh REAL);
CREATE TABLE IF NOT EXISTS load_report (key TEXT PRIMARY KEY, value TEXT);
"""


def views_sql(min_minutes: int) -> str:
    return f"""
DROP VIEW IF EXISTS v_hourly; DROP VIEW IF EXISTS v_daily; DROP VIEW IF EXISTS v_weekly;
CREATE VIEW v_hourly AS
SELECT strftime('%Y-%m-%d %H:00:00', ts)              AS hour_ts,
       COUNT(*)                                       AS n_min,
       AVG(active_kw)                                 AS kwh,
       AVG(voltage)                                   AS voltage_avg,
       SUM(sub1_wh) * 60.0 / COUNT(*) / 1000.0        AS sub1_kwh,
       SUM(sub2_wh) * 60.0 / COUNT(*) / 1000.0        AS sub2_kwh,
       SUM(sub3_wh) * 60.0 / COUNT(*) / 1000.0        AS sub3_kwh
FROM raw_minute
GROUP BY hour_ts
HAVING COUNT(*) >= {int(min_minutes)};

CREATE VIEW v_daily AS
SELECT date(hour_ts) AS day, COUNT(*) AS hours, SUM(kwh) AS kwh, MAX(kwh) AS peak_kwh
FROM v_hourly GROUP BY day;

CREATE VIEW v_weekly AS
SELECT strftime('%Y-%W', hour_ts) AS week, MIN(date(hour_ts)) AS week_start, COUNT(*) AS hours, SUM(kwh) AS kwh
FROM v_hourly GROUP BY week;
"""


def _open_source(src: Path):
    src = Path(src)
    if src.suffix.lower() == ".zip":
        z = zipfile.ZipFile(src)
        names = [n for n in z.namelist() if n.endswith(TXT)]
        if not names:
            raise FileNotFoundError(f"{src} içinde {TXT} yok.")
        return z.open(names[0])
    return open(src, "rb")


def build_db(src, db_path=DB_PATH, cfg: Config | None = None, max_rows: int | None = None, progress=print) -> dict:
    """Ham dosyadan (zip veya txt) SQLite veritabanını sıfırdan kurar. Eksik ('?') dakikalar atlanır ve sayılır."""
    cfg = cfg or Config()
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    total = missing = 0
    with _open_source(src) as fh:
        reader = pd.read_csv(fh, sep=";", na_values=["?", ""], chunksize=250_000, low_memory=False,
                             nrows=max_rows)
        for chunk in reader:
            total += len(chunk)
            ts = pd.to_datetime(chunk["Date"] + " " + chunk["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            num = chunk.drop(columns=["Date", "Time"]).apply(pd.to_numeric, errors="coerce")
            ok = ts.notna() & num.notna().all(axis=1)
            missing += int((~ok).sum())
            out = pd.DataFrame({"ts": ts[ok].dt.strftime("%Y-%m-%d %H:%M:%S")})
            out["active_kw"] = num.loc[ok, "Global_active_power"].values
            out["reactive_kvar"] = num.loc[ok, "Global_reactive_power"].values
            out["voltage"] = num.loc[ok, "Voltage"].values
            out["intensity"] = num.loc[ok, "Global_intensity"].values
            out["sub1_wh"] = num.loc[ok, "Sub_metering_1"].values
            out["sub2_wh"] = num.loc[ok, "Sub_metering_2"].values
            out["sub3_wh"] = num.loc[ok, "Sub_metering_3"].values
            out.drop_duplicates("ts").to_sql("raw_minute", con, if_exists="append", index=False)
            progress(f"  {total:,} satır okundu")
    con.executescript(views_sql(cfg.min_minutes_per_hour))
    rep = {"rows_read": total, "rows_missing_or_invalid": missing,
           "missing_pct": round(100 * missing / max(total, 1), 3)}
    h = pd.read_sql("SELECT MIN(hour_ts) a, MAX(hour_ts) b, COUNT(*) n FROM v_hourly", con)
    rep.update(hours_valid=int(h.n[0]), first_hour=h.a[0], last_hour=h.b[0],
               min_minutes_per_hour=cfg.min_minutes_per_hour)
    con.executemany("INSERT OR REPLACE INTO load_report VALUES (?,?)", [(k, str(v)) for k, v in rep.items()])
    con.commit()
    con.close()
    return rep


def connect(db_path=DB_PATH, read_only: bool = False) -> sqlite3.Connection:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Veritabanı yok: {db_path}. Önce build_db.bat çalıştırın.")
    if read_only:
        return sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    return sqlite3.connect(db_path)


def read_hourly(con) -> pd.DataFrame:
    """v_hourly'yi tam saatlik zaman ekseniyle okur; eksik saatler NaN kalır (doldurulmaz)."""
    df = pd.read_sql("SELECT * FROM v_hourly ORDER BY hour_ts", con, parse_dates=["hour_ts"])
    df = df.set_index("hour_ts")
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full)
    df.index.name = "ts"
    return df


def data_report(con) -> dict:
    return dict(pd.read_sql("SELECT key, value FROM load_report", con).values)
