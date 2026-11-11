"""Hazır SQL sorguları (arayüzde gösterilir) ve salt-okunur sorgu çalıştırıcı."""

from __future__ import annotations

import re

import pandas as pd

QUERIES = {
    "Günlük tüketim ve 7 günlük hareketli ortalama": """
SELECT day, ROUND(kwh, 2) AS kwh,
       ROUND(AVG(kwh) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS hareketli_ort_7g
FROM v_daily WHERE hours = 24 ORDER BY day LIMIT 200;""",
    "Bir önceki haftaya göre değişim (LAG)": """
SELECT week_start, ROUND(kwh, 1) AS kwh, ROUND(LAG(kwh) OVER (ORDER BY week_start), 1) AS onceki_hafta,
       ROUND(100.0 * (kwh - LAG(kwh) OVER (ORDER BY week_start)) / LAG(kwh) OVER (ORDER BY week_start), 1) AS degisim_yuzde
FROM v_weekly WHERE hours >= 150 ORDER BY week_start LIMIT 200;""",
    "Günün tepe saatleri (saat ortalaması ve sıra)": """
SELECT CAST(strftime('%H', hour_ts) AS INTEGER) AS saat, ROUND(AVG(kwh), 3) AS ort_kwh,
       RANK() OVER (ORDER BY AVG(kwh) DESC) AS sira
FROM v_hourly GROUP BY saat ORDER BY sira;""",
    "Aylık toplam tüketim": """
SELECT strftime('%Y-%m', day) AS ay, ROUND(SUM(kwh), 0) AS toplam_kwh, COUNT(*) AS gun
FROM v_daily WHERE hours = 24 GROUP BY ay ORDER BY ay;""",
    "Hafta içi / hafta sonu saatlik profil": """
SELECT CASE WHEN strftime('%w', hour_ts) IN ('0','6') THEN 'hafta sonu' ELSE 'hafta içi' END AS gun_tipi,
       CAST(strftime('%H', hour_ts) AS INTEGER) AS saat, ROUND(AVG(kwh), 3) AS ort_kwh
FROM v_hourly GROUP BY gun_tipi, saat ORDER BY gun_tipi, saat;""",
    "Alt sayaçların payı (mutfak / çamaşırhane / su ısıtıcı+klima / diğer)": """
SELECT ROUND(100*SUM(sub1_kwh)/SUM(kwh),1) AS mutfak_yuzde, ROUND(100*SUM(sub2_kwh)/SUM(kwh),1) AS camasirhane_yuzde,
       ROUND(100*SUM(sub3_kwh)/SUM(kwh),1) AS isitici_klima_yuzde,
       ROUND(100*(SUM(kwh)-SUM(sub1_kwh)-SUM(sub2_kwh)-SUM(sub3_kwh))/SUM(kwh),1) AS diger_yuzde
FROM v_hourly;""",
}

# israf tablosu (excess) kaydedildikten sonra çalışan sorgular
WASTE_QUERIES = {
    "Aşım saatleri: toplam fazla kWh ve saat sayısı": """
SELECT model, COUNT(*) AS asim_saati, ROUND(SUM(excess_kwh), 1) AS toplam_fazla_kwh, ROUND(AVG(excess_kwh), 3) AS ort_fazla_kwh
FROM excess GROUP BY model;""",
    "Aşımın günün saatine dağılımı": """
SELECT CAST(strftime('%H', hour_ts) AS INTEGER) AS saat, COUNT(*) AS asim_saati, ROUND(SUM(excess_kwh), 1) AS fazla_kwh
FROM excess GROUP BY saat ORDER BY saat;""",
    "Aşımın aya dağılımı": """
SELECT strftime('%Y-%m', hour_ts) AS ay, COUNT(*) AS asim_saati, ROUND(SUM(excess_kwh), 1) AS fazla_kwh
FROM excess GROUP BY ay ORDER BY ay;""",
    "Ardışık aşım saatleri (gaps-and-islands: aynı olay)": """
WITH t AS (SELECT hour_ts, excess_kwh,
                  CAST(strftime('%s', hour_ts) AS INTEGER)/3600 - ROW_NUMBER() OVER (ORDER BY hour_ts) AS grp FROM excess)
SELECT MIN(hour_ts) AS baslangic, COUNT(*) AS sure_saat, ROUND(SUM(excess_kwh), 2) AS fazla_kwh
FROM t GROUP BY grp HAVING COUNT(*) >= 2 ORDER BY fazla_kwh DESC LIMIT 20;""",
    "Aşım saatlerinde hangi alt sayaç yüksek? (aşım vs diğer saatler)": """
SELECT CASE WHEN e.hour_ts IS NULL THEN 'diğer saatler' ELSE 'aşım saatleri' END AS grup,
       ROUND(AVG(h.sub1_kwh), 3) AS mutfak, ROUND(AVG(h.sub2_kwh), 3) AS camasirhane, ROUND(AVG(h.sub3_kwh), 3) AS isitici_klima,
       ROUND(AVG(h.kwh - h.sub1_kwh - h.sub2_kwh - h.sub3_kwh), 3) AS diger
FROM v_hourly h LEFT JOIN (SELECT DISTINCT hour_ts FROM excess) e ON e.hour_ts = h.hour_ts
WHERE h.hour_ts IN (SELECT hour_ts FROM forecast_hours) GROUP BY grup;""",
}

_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b", re.I)


def run_select(con, sql: str, limit: int = 1000) -> pd.DataFrame:
    """Kullanıcı sorgusu için: tek ifade, yalnızca SELECT/WITH. Ek güvenlik: bağlantı salt-okunur açılmalıdır."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise ValueError("Tek bir SQL ifadesi girin.")
    if not re.match(r"^(select|with)\b", s, re.I) or _FORBIDDEN.search(s):
        raise ValueError("Yalnızca SELECT sorgularına izin verilir.")
    return pd.read_sql(f"SELECT * FROM ({s}) LIMIT {int(limit)}", con)
