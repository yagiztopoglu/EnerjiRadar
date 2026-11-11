"""Özellik grubu ablasyonu. Karar kuralı SONUÇLARDAN ÖNCE sabitlenmiştir:

  - Geliştirme blokları: ilk 3 test bloğu (F1-F3). Tüm kararlar yalnızca bunlara bakılarak verilir.
  - Bir ek grup (sub_lags, sub_daily, rhythm) ancak temel modele göre MAE farkının hafta-bloğu bootstrap %95 aralığı
    TAMAMEN negatifse benimsenir.
  - Bir temel grup, çıkarılınca MAE farkının aralığı 0'ı içeriyorsa (veya fark negatifse) 'çıkarılabilir' sayılır.
  - Seçilen yapılandırma, hiç bakılmamış son 2 blokta (F4-F5) temel modelle karşılaştırılır; asıl kanıt budur.
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from .evaluation import bootstrap_mae, point_metrics, run_fold
from .features import BASE_GROUPS, GROUPS

EXTRA = ["sub_lags", "sub_daily", "rhythm"]
GROUP_INFO = {
    "calendar": "Saat, haftanın günü, ay, hafta sonu, resmi tatil",
    "lags": "Aynı saatin 24/48/168 saat öncesi tüketimi",
    "daily_means": "Önceki günün ve önceki 7 günün ortalama tüketimi",
    "sub_lags": "Mutfak/çamaşırhane/su ısıtıcı+klima alt sayaçlarının dünkü ve geçen haftaki aynı saat değeri",
    "sub_daily": "Alt sayaçların önceki gün ortalaması",
    "rhythm": "Kullanım ritmi: mutfak ve çamaşırhane için son kullanımdan beri gün, son 28 günde kaç gün, kaç günde bir, gecikme oranı",
}


def gname(groups) -> str:
    return "GB:" + "+".join(groups)


def configs() -> dict:
    base = list(BASE_GROUPS)
    c = {"temel": base}
    for g in EXTRA:
        c[f"+{g}"] = base + [g]
    c["+hepsi"] = base + EXTRA
    for g in BASE_GROUPS:
        c[f"-{g}"] = [x for x in base if x != g]
    return c


def run_ablation(ctx: pd.DataFrame, cfg: Config, seeds=range(3), progress=None) -> dict:
    cf = configs()
    folds = list(cfg.folds)
    dev, final = folds[:3], folds[3:]
    parts = {}
    for fold in folds:
        for label, groups in cf.items():
            for sd in seeds:
                if progress:
                    progress(fold[0], label, sd)
                p = run_fold(ctx, fold, gname(groups), sd, cfg)
                p["model"] = label
                parts.setdefault(fold[0], []).append(p)
    allp = pd.concat([x for v in parts.values() for x in v])
    dev_names = {f[0] for f in dev}
    fin_names = {f[0] for f in final}
    pdev = allp[allp["fold"].isin(dev_names)]
    pfin = allp[allp["fold"].isin(fin_names)]

    def compare(preds, label, base="temel"):
        sub = preds[preds["model"].isin([label, base])]
        ci, diff = bootstrap_mae(sub, cfg.bootstrap)
        r = diff[(diff["model_a"] == label) & (diff["model_b"] == base)]
        if r.empty:
            r = diff[(diff["model_a"] == base) & (diff["model_b"] == label)]
            sign = -1.0
            row = r.iloc[0]
            return sign * row["mae_fark_a_eksi_b"], -row["ci_high"], -row["ci_low"]
        row = r.iloc[0]
        return row["mae_fark_a_eksi_b"], row["ci_low"], row["ci_high"]

    def table(preds):
        rows = []
        for label in cf:
            g = preds[preds["model"] == label]
            m = pd.DataFrame([point_metrics(x) for _, x in g.groupby(["fold", "seed"])]).mean()
            row = {"yapılandırma": label, "özellik grupları": "+".join(cf[label]), "MAE": m["mae"], "WAPE %": m["wape"]}
            if label != "temel":
                d, lo, hi = compare(preds, label)
                row.update({"MAE farkı (bu - temel)": d, "ci_low": lo, "ci_high": hi,
                            "fark ayırt edilebilir mi?": "evet" if (lo > 0 or hi < 0) else "hayır"})
            rows.append(row)
        return pd.DataFrame(rows)

    dev_tab = table(pdev)
    adopted = [g for g in EXTRA if dev_tab.set_index("yapılandırma").loc[f"+{g}", "ci_high"] < 0]
    removable = []
    for g in BASE_GROUPS:
        r = dev_tab.set_index("yapılandırma").loc[f"-{g}"]
        if r["ci_low"] <= 0:
            removable.append(g)
    selected = [g for g in BASE_GROUPS if g not in removable] + adopted
    joint_note = "uygulanmadı (çıkarılabilir grup yok)"
    if removable:
        # Tek tek 'çıkarılabilir' çıkan gruplar birbirinin yerini tutabilir (ör. gecikmeler ve günlük ortalamalar). Bu yüzden
        # birlikte çıkarma, yine YALNIZCA geliştirme bloklarında sınanır; anlamlı biçimde kötüyse çıkarma geri alınır.
        # (Bu adım, ilk koşuda F4-F5 bakıldıktan sonra kurala eklendi; F4-F5 artık bu karar için tam bağımsız değil.)
        sel_dev = [run_fold(ctx, f, gname(selected), sd, cfg) for f in dev for sd in seeds]
        for p in sel_dev:
            p["model"] = "seçilen"
        d, lo, hi = compare(pd.concat([pdev] + sel_dev), "seçilen")
        joint_note = f"MAE farkı {d:+.4f}, %95 aralık [{lo:+.4f}, {hi:+.4f}]"
        if lo > 0:
            removable = []
            selected = list(BASE_GROUPS) + adopted
            joint_note += " -> birlikte çıkarma anlamlı biçimde kötü, çıkarma geri alındı"
    decision = pd.DataFrame({
        "karar": ["benimsenen ek gruplar (geliştirme blokları)", "tek tek çıkarılabilir çıkan temel gruplar",
                  "birlikte çıkarma testi (geliştirme blokları)", "seçilen yapılandırma"],
        "değer": [", ".join(adopted) or "yok", ", ".join(g for g in BASE_GROUPS if dev_tab.set_index("yapılandırma").loc[f"-{g}", "ci_low"] <= 0) or "yok",
                  joint_note, "+".join(selected)]})

    final_labels = ["temel"] + (["seçilen"] if selected != list(BASE_GROUPS) else []) + ["+hepsi"]
    pf = pfin.copy()
    if selected != list(BASE_GROUPS):
        sel_parts = [run_fold(ctx, f, gname(selected), sd, cfg) for f in final for sd in seeds]
        for p in sel_parts:
            p["model"] = "seçilen"
        pf = pd.concat([pf] + sel_parts)
    rows = []
    for label in final_labels:
        g = pf[pf["model"] == label]
        m = pd.DataFrame([point_metrics(x) for _, x in g.groupby(["fold", "seed"])]).mean()
        row = {"yapılandırma": label, "MAE": m["mae"], "WAPE %": m["wape"]}
        if label != "temel":
            d, lo, hi = compare(pf, label)
            row.update({"MAE farkı (bu - temel)": d, "ci_low": lo, "ci_high": hi,
                        "fark ayırt edilebilir mi?": "evet" if (lo > 0 or hi < 0) else "hayır"})
        rows.append(row)
    fin_tab = pd.DataFrame(rows)
    perfold = pd.DataFrame([{"fold": f, "yapılandırma": lab, "MAE": point_metrics(g)["mae"]}
                            for (f, lab), g in allp.groupby(["fold", "model"], sort=False)])
    info_tab = pd.DataFrame({"grup": list(GROUP_INFO), "içerik": list(GROUP_INFO.values()),
                             "özellikler": [", ".join(GROUPS[g]) for g in GROUP_INFO]})
    return {"dev": dev_tab, "decision": decision, "final": fin_tab, "per_fold": perfold, "groups": info_tab,
            "selected": selected}
