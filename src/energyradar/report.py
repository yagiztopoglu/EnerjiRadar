"""Excel raporu: her sayfa biçimli, ilk sayfa sütun açıklamaları (Türkçe Excel ham CSV'yi tek sütunda açar)."""

from __future__ import annotations

import io

import pandas as pd

GLOSSARY = [
    ("model", "Tahmin yöntemi: Seasonal Naive (geçen haftanın aynı saati), Gradient Boosting, LSTM."),
    ("mae", "Ortalama mutlak hata (kWh). Tahminle gerçek arasındaki ortalama fark. Küçük iyidir. En kolay yorumlanan ölçüt."),
    ("rmse", "Hataların karesinin ortalamasının karekökü (kWh). Büyük hatalara daha çok ceza verir."),
    ("wape", "Toplam mutlak hata / toplam tüketim (%). Tüketim sıfıra yakın saatlerde bozulmaz, MAPE'den daha güvenilirdir."),
    ("mape", "Ortalama yüzde hata (%). Gerçek değer çok küçük olan saatlerde çok büyüyebilir; WAPE ile birlikte okuyun."),
    ("n_hours", "Değerlendirmeye giren saat sayısı (eksik veya tahmin edilemeyen saatler dışarıda)."),
    ("coverage_raw", "Modelin ham %90 aralığının gerçek değeri kapsama oranı. Hedef 0.90. Altında ise model gereğinden emin."),
    ("coverage_cal", "Kalibrasyondan sonra kapsama oranı. Hedef 0.90; sapma zaman serisinin kalibrasyon varsayımını bozduğunu gösterir."),
    ("width_raw / width_cal", "Aralığın ortalama genişliği (kWh). Dar ve doğru kapsayan aralık iyidir; yalnız genişlikle yorum yapılmaz."),
    ("seed_std", "Farklı rastgele tohumlar arasındaki standart sapma (tohuma duyarlılık)."),
    ("ci_low / ci_high", "Hafta blokları üzerinden bootstrap ile %95 güven aralığı."),
    ("ayırt edilebilir mi?", "Fark aralığı 0'ı içermiyorsa 'evet'. 'Hayır' ise bu veriyle iki modelin farkı şansla açıklanabilir."),
    ("asim_saati / fazla_kwh", "Gerçek tüketimin kalibre %90 aralığın üst sınırını aştığı saatler ve aşan miktar. Beklenen aşım oranı ~%5'tir; "
                               "bu, israfın kanıtı değil 'beklenenden fazla' sinyalidir."),
    ("detection_rate", "Enjeksiyon testi: yapay eklenen fazla tüketimin kaçta kaçının yakalandığı (Wilson %95 aralığıyla)."),
    ("base_exceedance_rate", "Hiçbir şey eklenmeden, normal saatlerin aralık üst sınırını aşma oranı (yanlış alarm tabanı)."),
]


def glossary_frame() -> pd.DataFrame:
    return pd.DataFrame(GLOSSARY, columns=["Sütun", "Ne demek?"])


def build_xlsx(sheets: dict, info: dict) -> bytes:
    buf = io.BytesIO()
    first = [("Okuma rehberi", glossary_frame()),
             ("Çalıştırma bilgisi", pd.DataFrame({"Bilgi": list(info), "Değer": [str(v) for v in info.values()]}))]
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        wb = xw.book
        head = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1, "text_wrap": True, "valign": "top"})
        num = wb.add_format({"num_format": "0.000"})
        for title, df in first + [(k[:31], v) for k, v in sheets.items()]:
            df = df.copy()
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = df[c].astype(str)
            df.to_excel(xw, sheet_name=title[:31], index=False)
            ws = xw.sheets[title[:31]]
            for j, col in enumerate(df.columns):
                ws.write(0, j, col, head)
                longest = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)])
                is_text = not pd.api.types.is_numeric_dtype(df[col])
                ws.set_column(j, j, min(max(12, longest + 2), 90 if is_text else 22),
                              num if pd.api.types.is_float_dtype(df[col]) else None)
            ws.set_row(0, 32)
            ws.freeze_panes(1, 0)
    return buf.getvalue()
