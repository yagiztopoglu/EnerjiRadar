"""EnerjiRadar arayüzü (Streamlit). Çalıştırma: run_app.bat"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from energyradar.config import DB_PATH, RESULTS_DIR  # noqa: E402
from energyradar.data import connect, data_report, read_hourly  # noqa: E402
from energyradar.queries import QUERIES, WASTE_QUERIES, run_select  # noqa: E402
from energyradar.report import GLOSSARY  # noqa: E402

st.set_page_config(page_title="EnerjiRadar", layout="wide")

PAGES = ["Genel", "Rehber", "Veri ve SQL", "Tahmin laboratuvarı", "Sonuçlar", "İsraf analizi"]
HELP = {
    "Veri ve SQL": "Ham dakikalık ölçümler SQLite veritabanında durur; saatlik/günlük/haftalık tablolar SQL ile hesaplanır. "
                   "Hazır sorguları seçip SQL'ini ve sonucunu görebilir, kendi SELECT sorgunuzu yazabilirsiniz (yalnızca okuma).",
    "Tahmin laboratuvarı": "Bir test bloğu ve model seçin. Mavi çizgi gerçek tüketim, turuncu çizgi tahmin, gri bant %90 tahmin aralığıdır. "
                           "Kırmızı noktalar gerçek tüketimin aralığın üstüne çıktığı saatlerdir.",
    "Sonuçlar": "Modellerin hata ölçütleri, farkların güven aralıkları ve hangi koşulda (hafta sonu, mevsim, saat) kimin iyi olduğu.",
    "İsraf analizi": "Beklenenden fazla tüketilen saatler. Bu israfın kanıtı değil, 'tahminin üstünde' sinyalidir; aşağıdaki test, "
                     "yapay eklenen fazla tüketimin ne kadarının yakalandığını gösterir.",
}


@st.cache_data(show_spinner=False)
def _hourly():
    con = connect(DB_PATH, read_only=True)
    try:
        return read_hourly(con)
    finally:
        con.close()


def _con():
    return connect(DB_PATH, read_only=True)


def _need_db() -> bool:
    if not DB_PATH.exists():
        st.error("Veritabanı yok. Önce `build_db.bat` çalıştırın (data/uci klasörüne UCI zip dosyasını koyduktan sonra).")
        return False
    return True


def _csv(name):
    p = RESULTS_DIR / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def help_box(page):
    if page in HELP:
        st.info(HELP[page])


def page_overview():
    st.title("EnerjiRadar")
    st.write("Saatlik elektrik tüketimini tahmin eder, tahminin %90 aralığını üretir ve gerçek tüketimin aralığın "
             "üstüne çıktığı saatleri 'beklenenden fazla tüketim' olarak işaretler. Üç yöntem aynı veri bölmesi ve aynı "
             "ölçütlerle karşılaştırılır: geçen haftanın aynı saati (baseline), gradyan artırma ve LSTM.")
    if _need_db():
        con = _con()
        rep = data_report(con)
        con.close()
        a, b, c = st.columns(3)
        a.metric("Geçerli saat", f"{int(rep.get('hours_valid', 0)):,}")
        b.metric("Eksik dakika (%)", rep.get("missing_pct", "-"))
        c.metric("Dönem", f"{str(rep.get('first_hour', ''))[:10]} - {str(rep.get('last_hour', ''))[:10]}")
    st.caption("Veri: Hebrail & Berard (2012), UCI Machine Learning Repository, tek bir Fransız hanenin ölçümleri (2006-2010). "
               "Sonuçlar tek hane için geçerlidir; Türkiye'ye veya sanayiye genellenemez.")


def page_guide():
    st.header("Rehber")
    st.write("**Nasıl okunur?** MAE ne kadar küçükse tahmin o kadar iyidir (birim kWh). Aralık kapsama oranı %90'a ne kadar yakınsa "
             "belirsizlik o kadar dürüst raporlanıyordur. Bir farkın gerçek olup olmadığını 'ayırt edilebilir mi?' sütunu söyler.")
    st.table(pd.DataFrame(GLOSSARY, columns=["Sütun", "Ne demek?"]))


def page_data():
    st.header("Veri ve SQL")
    help_box("Veri ve SQL")
    if not _need_db():
        return
    h = _hourly()
    s, e = h.index.min().date(), h.index.max().date()
    start = st.date_input("Başlangıç", value=pd.Timestamp("2009-01-05").date(), min_value=s, max_value=e)
    days = st.slider("Kaç gün", 3, 60, 14)
    w = h.loc[pd.Timestamp(start): pd.Timestamp(start) + pd.Timedelta(days=days)]
    fig = go.Figure(go.Scatter(x=w.index, y=w["kwh"], mode="lines", name="kWh"))
    fig.update_layout(height=320, yaxis_title="kWh (saatlik)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)
    st.subheader("Hazır SQL sorguları")
    name = st.selectbox("Sorgu", list(QUERIES))
    st.code(QUERIES[name].strip(), language="sql")
    con = _con()
    try:
        st.dataframe(run_select(con, QUERIES[name]))
        st.subheader("Kendi sorgunuz (yalnızca SELECT)")
        sql = st.text_area("SQL", "SELECT * FROM v_daily ORDER BY kwh DESC LIMIT 10")
        if st.button("Çalıştır"):
            try:
                st.dataframe(run_select(con, sql))
            except Exception as exc:  # kullanıcı hatası: mesajı göster
                st.error(str(exc))
    finally:
        con.close()


def page_forecast():
    st.header("Tahmin laboratuvarı")
    help_box("Tahmin laboratuvarı")
    if not _need_db():
        return
    con = _con()
    try:
        fc = pd.read_sql("SELECT * FROM forecasts", con, parse_dates=["hour_ts"])
    except Exception:
        st.warning("Henüz tahmin sonucu yok. Önce `run_experiments.bat` çalıştırın.")
        con.close()
        return
    con.close()
    fold = st.selectbox("Test bloğu", list(dict.fromkeys(fc["fold"])))
    model = st.selectbox("Model", list(dict.fromkeys(fc["model"])))
    g = fc[(fc["fold"] == fold) & (fc["model"] == model) & (fc["seed"] == fc[(fc["model"] == model)]["seed"].min())].set_index("hour_ts")
    start = st.date_input("Başlangıç günü", value=g.index.min().date(), min_value=g.index.min().date(), max_value=g.index.max().date())
    days = st.slider("Kaç gün", 2, 21, 7)
    w = g.loc[pd.Timestamp(start): pd.Timestamp(start) + pd.Timedelta(days=days)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.index, y=w["hi_cal"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=w.index, y=w["lo_cal"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(150,150,150,0.3)", name="%90 aralık (kalibre)"))
    fig.add_trace(go.Scatter(x=w.index, y=w["y"], mode="lines", name="gerçek"))
    fig.add_trace(go.Scatter(x=w.index, y=w["p50"], mode="lines", name="tahmin", line=dict(dash="dot")))
    ex = w[w["y"] > w["hi_cal"]]
    fig.add_trace(go.Scatter(x=ex.index, y=ex["y"], mode="markers", name="aralığın üstünde", marker=dict(color="red", size=7)))
    fig.update_layout(height=420, yaxis_title="kWh (saatlik)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)
    d = w.dropna(subset=["y", "p50"])
    if len(d):
        st.write(f"Gösterilen {len(d)} saatte ortalama mutlak hata: **{(d['y'] - d['p50']).abs().mean():.3f} kWh**; "
                 f"aralığın üstüne çıkan saat: **{len(ex)}** (beklenen yaklaşık %5 = {0.05 * len(d):.0f}).")


def page_results():
    st.header("Sonuçlar")
    help_box("Sonuçlar")
    summary = _csv("summary")
    if summary is None:
        st.warning("Henüz sonuç yok. Önce `run_experiments.bat` çalıştırın. Okunabilir Excel: results/EnerjiRadar_Sonuclar.xlsx")
        return
    st.subheader("Model özeti (blok ve tohum ortalaması)")
    st.dataframe(summary)
    ci, diff = _csv("bootstrap_mae"), _csv("bootstrap_diff")
    if ci is not None:
        st.subheader("MAE ve %95 güven aralığı (hafta blokları üzerinden bootstrap)")
        st.dataframe(ci)
    if diff is not None:
        st.subheader("Model farkları")
        st.dataframe(diff)
        for _, r in diff.iterrows():
            if r["ayırt edilebilir mi?"] == "evet":
                better = r["model_a"] if r["mae_fark_a_eksi_b"] < 0 else r["model_b"]
                st.write(f"- {r['model_a']} ile {r['model_b']} arasında fark ayırt edilebilir: {better} daha düşük hata veriyor.")
            else:
                st.write(f"- {r['model_a']} ile {r['model_b']} arasındaki fark bu veriyle ayırt edilemiyor.")
    cond = _csv("conditions")
    if cond is not None:
        st.subheader("Koşullara göre MAE")
        k = st.selectbox("Koşul", list(dict.fromkeys(cond["koşul"])))
        piv = cond[cond["koşul"] == k].pivot_table(index="değer", columns="model", values="mae")
        st.dataframe(piv)
        fig = go.Figure([go.Bar(x=piv.index, y=piv[c], name=c) for c in piv.columns])
        fig.update_layout(barmode="group", height=340, yaxis_title="MAE (kWh)")
        st.plotly_chart(fig)


def page_waste():
    st.header("İsraf analizi")
    help_box("İsraf analizi")
    if not _need_db():
        return
    con = _con()
    try:
        names = list(WASTE_QUERIES)
        q = st.selectbox("Sorgu", names)
        st.code(WASTE_QUERIES[q].strip(), language="sql")
        st.dataframe(pd.read_sql(WASTE_QUERIES[q], con))
    except Exception:
        st.warning("İsraf tablosu yok. Önce `run_experiments.bat` çalıştırın.")
    finally:
        con.close()
    inj = _csv("injection")
    if inj is not None:
        st.subheader("Enjeksiyon testi: yapay fazla tüketim ne kadar yakalanıyor?")
        agg = inj.groupby("bump_kwh")[["detection_rate", "base_exceedance_rate"]].mean().reset_index()
        fig = go.Figure(go.Scatter(x=agg["bump_kwh"], y=agg["detection_rate"], mode="lines+markers", name="yakalama oranı"))
        fig.update_layout(height=320, xaxis_title="4 saate eklenen fazla tüketim (kWh/saat)", yaxis_title="yakalama oranı")
        st.plotly_chart(fig)
        st.caption("Normal saatlerin aralığı aşma oranı (yanlış alarm tabanı): %.1f%%" % (100 * agg["base_exceedance_rate"].mean()))


page = st.sidebar.radio("Sayfa", PAGES)
{"Genel": page_overview, "Rehber": page_guide, "Veri ve SQL": page_data, "Tahmin laboratuvarı": page_forecast,
 "Sonuçlar": page_results, "İsraf analizi": page_waste}[page]()
