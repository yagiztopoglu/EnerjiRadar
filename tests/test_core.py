"""Birim testleri. Not: sentetik veriler yalnızca KODUN doğruluğunu sınamak içindir; sonuç üretmek için kullanılmaz."""

import io
import sys
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energyradar.config import Config  # noqa: E402
from energyradar.data import build_db, connect, read_hourly  # noqa: E402
from energyradar.evaluation import (add_ensemble, bootstrap_mae, calibrate, calibrate_rolling, condition_bootstrap,  # noqa: E402
                                    conformal_quantile, excess_hours, injection_rules, injection_test, point_metrics, wilson)
from energyradar.features import FEATURES, build_frame, easter, french_holidays  # noqa: E402
from energyradar.models import GradientBoosting, LSTMForecaster, SeasonalNaive, torch_available  # noqa: E402
from energyradar.queries import run_select  # noqa: E402
from energyradar.report import build_xlsx  # noqa: E402

HEADER = "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n"


def tiny_txt(path):
    """3 saat: 00:00 tam (60 dk, 2 kW), 01:00 yalnızca 40 dk (eksik sayılmalı), 02:00 tam (4 kW, '?' satırı yok)."""
    lines = [HEADER]
    for m in range(60):
        lines.append(f"01/01/2010;00:{m:02d}:00;2.000;0.1;240;8;1;2;3\n")
    for m in range(40):
        lines.append(f"01/01/2010;01:{m:02d}:00;9.000;0.1;240;8;0;0;0\n")
    for m in range(40, 60):
        lines.append(f"01/01/2010;01:{m:02d}:00;?;?;?;?;?;?;?\n")
    for m in range(60):
        lines.append(f"01/01/2010;02:{m:02d}:00;4.000;0.1;240;8;0;0;0\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


def synth_hourly(days=60, seed=0):
    idx = pd.date_range("2010-01-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(seed)
    y = 1.0 + 0.5 * np.sin(2 * np.pi * idx.hour / 24) + 0.3 * (idx.dayofweek >= 5) + rng.normal(0, 0.1, len(idx))
    return pd.DataFrame({"kwh": y}, index=idx)


class CalendarTest(unittest.TestCase):
    def test_easter(self):
        self.assertEqual(easter(2009), date(2009, 4, 12))
        self.assertEqual(easter(2010), date(2010, 4, 4))

    def test_holidays(self):
        h = french_holidays([2009, 2010])
        self.assertIn(date(2009, 4, 13), h)      # Paskalya Pazartesi
        self.assertIn(date(2010, 7, 14), h)
        self.assertNotIn(date(2010, 7, 15), h)


class DataTest(unittest.TestCase):
    def test_build_db_hourly_view(self):
        with tempfile.TemporaryDirectory() as t:
            txt = Path(t) / "household_power_consumption.txt"
            tiny_txt(txt)
            rep = build_db(txt, Path(t) / "x.db", progress=lambda s: None)
            self.assertEqual(rep["rows_missing_or_invalid"], 20)
            con = connect(Path(t) / "x.db")
            h = pd.read_sql("SELECT * FROM v_hourly ORDER BY hour_ts", con)
            self.assertEqual(len(h), 2)                      # 40 dakikalık saat (45'ten az) dışarıda
            self.assertAlmostEqual(h.kwh[0], 2.0)
            self.assertAlmostEqual(h.sub3_kwh[0], 3 * 60 / 1000)   # 3 Wh/dk x 60 dk = 0.18 kWh
            full = read_hourly(con)
            self.assertEqual(len(full), 3)
            self.assertTrue(np.isnan(full["kwh"].iloc[1]))   # eksik saat doldurulmadı
            d = pd.read_sql("SELECT * FROM v_daily", con)
            self.assertEqual(int(d.hours[0]), 2)
            con.close()

    def test_zip_source(self):
        with tempfile.TemporaryDirectory() as t:
            txt = Path(t) / "household_power_consumption.txt"
            tiny_txt(txt)
            z = Path(t) / "d.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.write(txt, "household_power_consumption.txt")
            rep = build_db(z, Path(t) / "y.db", progress=lambda s: None)
            self.assertEqual(rep["hours_valid"], 2)


class SqlGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        txt = Path(self.tmp.name) / "household_power_consumption.txt"
        tiny_txt(txt)
        build_db(txt, Path(self.tmp.name) / "g.db", progress=lambda s: None)
        self.con = connect(Path(self.tmp.name) / "g.db", read_only=True)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_select_ok(self):
        self.assertEqual(len(run_select(self.con, "SELECT * FROM v_hourly")), 2)

    def test_rejects_write_and_multi(self):
        for bad in ("DROP TABLE raw_minute", "SELECT 1; DROP TABLE raw_minute", "DELETE FROM raw_minute",
                    "SELECT * FROM v_hourly WHERE 1=1; SELECT 2"):
            with self.assertRaises(ValueError):
                run_select(self.con, bad)

    def test_read_only_connection(self):
        with self.assertRaises(Exception):
            self.con.execute("DELETE FROM raw_minute")


class FeatureTest(unittest.TestCase):
    def test_no_future_leakage(self):
        """Gün D'nin özellikleri, D günü 00:00 ve sonrasındaki gözlemlere bağlı OLMAMALI."""
        h = synth_hourly(40)
        f1 = build_frame(h)
        D = pd.Timestamp("2010-01-25")
        h2 = h.copy()
        h2.loc[h2.index >= D, "kwh"] += 100.0      # gelecek değerlerini boz
        f2 = build_frame(h2)
        rows = f1.index[(f1.index >= D) & (f1.index < D + pd.Timedelta(days=1))]
        pd.testing.assert_frame_equal(f1.loc[rows, FEATURES], f2.loc[rows, FEATURES])

    def test_lags(self):
        h = synth_hourly(20)
        f = build_frame(h)
        t = pd.Timestamp("2010-01-12 05:00")
        self.assertAlmostEqual(f.loc[t, "lag24"], h.loc[t - pd.Timedelta(hours=24), "kwh"])
        self.assertAlmostEqual(f.loc[t, "lag168"], h.loc[t - pd.Timedelta(hours=168), "kwh"])


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.h = synth_hourly(80)
        self.ctx = build_frame(self.h)
        days = pd.date_range("2010-01-01", periods=80, freq="D")
        self.train, self.test = list(days[:60]), list(days[60:70])

    def test_seasonal_naive(self):
        p = SeasonalNaive(Config()).predict(self.ctx, self.test)
        t = p.index[10]
        self.assertAlmostEqual(p.loc[t, "p50"], self.h.loc[t - pd.Timedelta(hours=168), "kwh"])
        self.assertTrue(p["p05"].isna().all())

    def test_gbm_quantiles_ordered_and_learns(self):
        cfg = Config(gbm_trees=60)
        m = GradientBoosting(cfg).fit(self.ctx, self.train)
        p = m.predict(self.ctx, self.test).join(self.ctx["y"])
        self.assertTrue((p["p05"] <= p["p50"]).all() and (p["p50"] <= p["p95"]).all())
        naive = SeasonalNaive(cfg).predict(self.ctx, self.test).join(self.ctx["y"])
        self.assertLess(point_metrics(p)["mae"], point_metrics(naive)["mae"] * 1.5)

    def test_lstm_data_prep_shapes(self):
        m = LSTMForecaster(Config())
        m.mu_, m.sd_, m.how_mean_ = 1.0, 0.5, np.zeros(168)
        X, C, Y, M, keep = m._prepare(self.ctx, self.test)
        self.assertEqual(X.shape, (10, 168, 4))
        self.assertEqual(C.shape, (10, 10))
        self.assertEqual(Y.shape, (10, 24))
        self.assertTrue(np.all(M == 1))

    @unittest.skipUnless(torch_available(), "PyTorch yok")
    def test_lstm_learns_and_orders_quantiles(self):
        cfg = Config(lstm_epochs=25, lstm_hidden=16)
        m = LSTMForecaster(cfg, seed=0).fit(self.ctx, self.train)
        self.assertLess(m.loss_history_[-1], m.loss_history_[0])
        p = m.predict(self.ctx, self.test)
        self.assertEqual(len(p), 240)
        self.assertTrue((p["p05"] <= p["p95"]).all())


class EvalTest(unittest.TestCase):
    def test_conformal_coverage_iid(self):
        rng = np.random.default_rng(0)
        n = 4000
        y = rng.normal(0, 1, n)
        calib = pd.DataFrame({"y": y[:2000], "p05": -0.5, "p50": 0.0, "p95": 0.5})      # aralık fazla dar
        test = pd.DataFrame({"y": y[2000:], "p05": -0.5, "p50": 0.0, "p95": 0.5})
        out = calibrate(calib, test, 0.9)
        cov = ((out["y"] >= out["lo_cal"]) & (out["y"] <= out["hi_cal"])).mean()
        self.assertGreater(cov, 0.87)
        self.assertLess(cov, 0.93)

    def test_conformal_quantile_small_n(self):
        self.assertEqual(conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.1), 3.0)

    def test_metrics_hand_example(self):
        d = pd.DataFrame({"y": [1.0, 2.0, 4.0], "p50": [1.0, 3.0, 2.0]})
        m = point_metrics(d)
        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertAlmostEqual(m["rmse"], np.sqrt(5 / 3))
        self.assertAlmostEqual(m["wape"], 3 / 7 * 100)

    def _preds(self):
        idx = pd.date_range("2010-01-04", periods=24 * 7 * 12, freq="h")
        rng = np.random.default_rng(1)
        y = rng.normal(1, 0.1, len(idx))
        rows = []
        for model, noise in (("A", 0.05), ("B", 0.5)):
            rows.append(pd.DataFrame({"model": model, "seed": 0, "fold": "F", "y": y, "p50": y + rng.normal(0, noise, len(y)),
                                      "hi_cal": y + 0.3, "is_holiday": 0}, index=idx))
        return pd.concat(rows)

    def test_bootstrap_detects_clear_difference(self):
        ci, diff = bootstrap_mae(self._preds(), n_boot=300)
        self.assertEqual(diff["ayırt edilebilir mi?"].iloc[0], "evet")
        self.assertLess(ci.set_index("model").loc["A", "mae"], ci.set_index("model").loc["B", "mae"])

    def test_wilson_and_injection(self):
        lo, hi = wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.2)
        cfg = Config(inject_events_per_fold=20, inject_sizes=(0.1, 5.0))
        inj = injection_test(self._preds(), "A", cfg)
        small, big = inj[inj.bump_kwh == 0.1], inj[inj.bump_kwh == 5.0]
        self.assertLess(small.detection_rate.iloc[0], big.detection_rate.iloc[0])
        self.assertEqual(big.detection_rate.iloc[0], 1.0)

    def test_excess_hours(self):
        p = self._preds()
        p.loc[p.index[5], ["y", "model"]] = [99.0, "A"]
        p.loc[p.index[5], "hi_cal"] = 1.0
        e = excess_hours(p[p.model == "A"], "A")
        self.assertIn(99.0, e["y"].values)
        self.assertTrue((e["excess_kwh"] > 0).all())


class RhythmTest(unittest.TestCase):
    def _sub(self, use_days, n_days=40):
        idx = pd.date_range("2010-01-01", periods=24 * n_days, freq="h")
        v = np.zeros(len(idx))
        for d in use_days:
            v[d * 24 + 19] = 1.0           # o günün 19:00'unda 1 kWh çevrim
        return pd.Series(v, index=idx)

    def test_rhythm_hand_example(self):
        from energyradar.features import rhythm_features
        r = rhythm_features(self._sub([0, 7, 14, 21, 28, 35]))     # haftada bir kullanım
        self.assertEqual(r["days_since_use"].iloc[10], 3.0)          # gün 10: son kullanım gün 7
        self.assertEqual(r["gap_med_days"].iloc[30], 7.0)
        self.assertAlmostEqual(r["overdue_ratio"].iloc[30], (30 - 28) / 7.0)
        self.assertEqual(r["use_days_28"].iloc[29], 4.0)             # gün 1..28 arası: 7,14,21,28
        self.assertTrue(np.isnan(r["days_since_use"].iloc[0]))       # geçmiş yokken bilinmiyor

    def test_rhythm_no_future_leakage(self):
        from energyradar.features import rhythm_features
        a = self._sub([0, 3, 6, 9, 12, 15, 18, 21])
        b = a.copy()
        b[b.index >= "2010-01-15"] = 5.0
        ra, rb = rhythm_features(a), rhythm_features(b)
        pd.testing.assert_frame_equal(ra.loc[:"2010-01-15"], rb.loc[:"2010-01-15"])   # D günü D'nin kendi verisini görmez

    def test_group_model_name(self):
        from energyradar.models import make_model
        m = make_model("GB:calendar+lags+rhythm", Config())
        self.assertIn("days_since_use_1", m.features)
        self.assertNotIn("prev_day_mean", m.features)


class ExtraEvalTest(unittest.TestCase):
    def _preds(self):
        idx = pd.date_range("2010-01-04", periods=24 * 7 * 12, freq="h").rename("ts")
        rng = np.random.default_rng(2)
        y = rng.normal(1, 0.1, len(idx))
        rows = []
        for model, noise in (("Gradient Boosting", 0.05), ("LSTM", 0.3)):
            rows.append(pd.DataFrame({"model": model, "seed": 0, "fold": "F", "y": y, "p05": y - 0.2, "p50": y + rng.normal(0, noise, len(y)),
                                      "p95": y + 0.2, "lo_cal": y - 0.2, "hi_cal": y + 0.2, "lo_roll": y - 0.2, "hi_roll": y + 0.2,
                                      "is_holiday": 0}, index=idx))
        return pd.concat(rows)

    def test_ensemble_is_mean(self):
        p = self._preds()
        e = add_ensemble(p)
        a = p[p.model == "Gradient Boosting"]["p50"].values
        b = p[p.model == "LSTM"]["p50"].values
        np.testing.assert_allclose(e[e.model == "Ensemble GB+LSTM"]["p50"].values, (a + b) / 2)

    def test_ensemble_skipped_when_member_missing(self):
        p = self._preds()
        self.assertEqual(len(add_ensemble(p[p.model == "LSTM"])), len(p[p.model == "LSTM"]))

    def test_rolling_calibration_uses_only_past(self):
        """Test gününün aralığı, o günün kendi gözlemlerine bağlı olmamalı."""
        idx = pd.date_range("2010-01-01", periods=24 * 20, freq="h")
        rng = np.random.default_rng(3)
        d = pd.DataFrame({"y": rng.normal(0, 1, len(idx)), "p05": -1.0, "p50": 0.0, "p95": 1.0}, index=idx)
        calib, test = d.iloc[:24 * 10], d.iloc[24 * 10:]
        a = calibrate_rolling(calib, test, 0.9, 7)
        test2 = test.copy()
        test2.loc[test2.index.normalize() == pd.Timestamp("2010-01-15"), "y"] += 50.0   # 15 Ocak'ı boz
        b = calibrate_rolling(calib, test2, 0.9, 7)
        day = a.index.normalize() <= pd.Timestamp("2010-01-15")
        np.testing.assert_allclose(a.loc[day, "hi_roll"].values, b.loc[day, "hi_roll"].values)   # 15'i ve öncesi etkilenmedi
        self.assertFalse(np.allclose(a["hi_roll"].values, b["hi_roll"].values))                  # sonraki günler etkilendi

    def test_condition_bootstrap_flags_clear_difference(self):
        r = condition_bootstrap(self._preds(), "Gradient Boosting", "LSTM", n_boot=200)
        self.assertGreater(len(r), 3)
        self.assertTrue((r["ayırt edilebilir mi?"] == "evet").all())
        self.assertTrue((r["MAE farkı (a - b)"] < 0).all())

    def test_injection_rules_consecutive_stricter(self):
        cfg = Config(inject_events_per_fold=20, inject_sizes=(0.3, 5.0))
        p = self._preds()
        r = injection_rules(p, "Gradient Boosting", cfg)
        for size in (0.3, 5.0):
            a = r[(r.bump_kwh == size) & (r.kural == "herhangi 1 saat aşım")].detection_rate.iloc[0]
            b = r[(r.bump_kwh == size) & (r.kural == "en az 2 ardışık saat aşım")].detection_rate.iloc[0]
            self.assertLessEqual(b, a)


class ReportTest(unittest.TestCase):
    def test_xlsx_sheets(self):
        b = build_xlsx({"Model özeti": pd.DataFrame({"model": ["A"], "mae": [0.5]})}, {"Bilgi": "x"})
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            wb = z.read("xl/workbook.xml").decode("utf-8")
        for name in ("Okuma rehberi", "Çalıştırma bilgisi", "Model özeti"):
            self.assertIn(name, wb)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- dış doğrulama okuyucuları (sentetik dosya yalnızca KOD testi içindir)
from energyradar.datasets import (auto_folds, clean_ld_client, eligible_clients, load_epias, load_ld2011,  # noqa: E402
                                  pick_clients, scaled_config, to_number)
from energyradar.pipeline import run_pipeline  # noqa: E402


def write_ld_fixture(path, days=40, decimal_comma=True):
    """LD2011_2014.txt biçimi: ilk sütun adsız, ';' ayırıcı, ondalık virgül, 15 dk, zaman damgası aralık SONU (ilk satır 00:15)."""
    idx = pd.date_range("2011-01-01 00:15", periods=days * 96, freq="15min")
    rng = np.random.default_rng(0)
    base = 5 + 3 * np.sin(2 * np.pi * (idx.hour + idx.minute / 60) / 24)
    cols = {"MT_001": base + rng.normal(0, .1, len(idx)), "MT_002": 2 * base + rng.normal(0, .1, len(idx)),
            "MT_003": np.where(idx < pd.Timestamp("2011-02-10"), 0.0, base)}     # geç açılan müşteri
    df = pd.DataFrame(cols, index=idx)
    text = df.to_csv(sep=";", index_label="", float_format="%.7f")
    if decimal_comma:
        head, *rows = text.split("\n")
        text = "\n".join([head] + [r.replace(".", ",") for r in rows])
    Path(path).write_text(text, encoding="utf-8")


class ExternalDataTest(unittest.TestCase):
    def test_to_number_formats(self):
        s = pd.Series(["35.123,45", "35123,45", "35123.45", "35.123", "  1.234.567,5 "])
        v = to_number(s).values
        np.testing.assert_allclose(v, [35123.45, 35123.45, 35123.45, 35123.0, 1234567.5])

    def test_epias_date_hour_columns(self):
        with tempfile.TemporaryDirectory() as d:
            idx = pd.date_range("2023-01-01", periods=48, freq="h")
            df = pd.DataFrame({"Tarih": idx.strftime("%d.%m.%Y"), "Saat": idx.strftime("%H:%M"),
                               "Tüketim Miktarı (MWh)": [f"{30000 + i * 10:,}".replace(",", ".") + ",5" for i in range(48)]})
            p = Path(d) / "e.csv"
            df.to_csv(p, sep=";", index=False, encoding="utf-8-sig")
            h, meta = load_epias(p)
            self.assertEqual(len(h), 48)
            self.assertAlmostEqual(h["kwh"].iloc[0], 30000.5)
            self.assertAlmostEqual(h["kwh"].iloc[-1], 30470.5)
            self.assertEqual(h.index[1], pd.Timestamp("2023-01-01 01:00"))
            self.assertIn("MWh", meta["birim"])

    def test_epias_iso_datetime_with_offset_and_gap(self):
        with tempfile.TemporaryDirectory() as d:
            idx = pd.date_range("2023-01-01", periods=30, freq="h", tz="Europe/Istanbul").delete(10)   # 1 saat eksik
            df = pd.DataFrame({"date": idx.strftime("%Y-%m-%dT%H:%M:%S%z"), "consumption": np.arange(len(idx), dtype=float)})
            p = Path(d) / "e.csv"
            df.to_csv(p, index=False)
            h, _ = load_epias(p)
            self.assertEqual(len(h), 30)
            self.assertTrue(np.isnan(h["kwh"].iloc[10]))           # eksik saat doldurulmaz

    def test_epias_unknown_columns_stops_with_message(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.csv"
            pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)
            with self.assertRaises(ValueError) as cm:
                load_epias(p)
            self.assertIn("Bulunan sütunlar", str(cm.exception))

    def test_ld2011_hourly_alignment_cleaning_and_selection(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "LD2011_2014.txt"
            write_ld_fixture(p)
            out, meta = load_ld2011(p, clients=["MT_001", "MT_003"], progress=lambda *_: None)
            self.assertIn("aralık sonu", meta["zaman damgası"])
            h1 = out["MT_001"]
            self.assertEqual(h1.index[0], pd.Timestamp("2011-01-01 00:00"))        # 00:15..01:00 okumaları -> 00:00 saati
            self.assertEqual(len(h1), 40 * 24)
            self.assertAlmostEqual(h1["kwh"].iloc[0], 5 + 3 * np.sin(2 * np.pi * np.mean([0.25, 0.5, 0.75, 1.0]) / 24), delta=0.3)
            h3 = out["MT_003"]["kwh"]
            self.assertTrue(h3[:"2011-02-09 22:00"].isna().all())                  # açılıştan önceki sıfırlar eksik sayılır
            self.assertTrue(h3["2011-02-10 06:00":].notna().all())

    def test_ld2011_zip_and_random_pick_is_seeded(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "LD2011_2014.txt"
            write_ld_fixture(p)
            z = Path(d) / "x.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.write(p, "LD2011_2014.txt")
            out, meta = load_ld2011(z, n_clients=2, seed=1, progress=lambda *_: None)
            out2, _ = load_ld2011(z, n_clients=2, seed=1, progress=lambda *_: None)
            self.assertEqual(list(out), list(out2))
            self.assertNotIn("MT_003", out)                                         # geç açılan müşteri uygun değil
            self.assertEqual(meta["uygun_müşteri"], 2)

    def test_auto_folds_and_scaled_config(self):
        h = pd.DataFrame({"kwh": 100.0}, index=pd.date_range("2011-01-01", "2014-12-31 23:00", freq="h"))
        f = auto_folds(h, test_years=2)
        self.assertEqual(f[0][1], "2013-01-01")
        self.assertEqual(len(f), 8)
        self.assertTrue(all(pd.Timestamp(a) - pd.Timestamp("2011-01-01") >= pd.Timedelta(days=365) for _, a, _ in f))
        cfg = scaled_config(Config(), h, unit="MWh", country="TR")
        self.assertEqual(cfg.inject_sizes, (20.0, 50.0, 100.0, 200.0))
        self.assertEqual((cfg.unit, cfg.country), ("MWh", "TR"))

    def test_pipeline_on_external_hourly(self):
        h = synth_hourly(150)
        h = h[["kwh"]]
        cfg = Config(seeds=1, gbm_trees=30, bootstrap=50, inject_events_per_fold=5,
                     folds=[("A", "2010-04-01", "2010-04-20"), ("B", "2010-04-21", "2010-05-10")])
        with tempfile.TemporaryDirectory() as d:
            r = run_pipeline(out_dir=d, cfg=cfg, models=["Seasonal Naive", "Gradient Boosting"], hourly=h,
                             persistent=False, progress=lambda *_: None)
            self.assertEqual(set(r["summary"]["model"]), {"Seasonal Naive", "Gradient Boosting"})
            self.assertTrue((Path(d) / "EnerjiRadar_Sonuclar.xlsx").exists())
            self.assertEqual(r["info"]["LSTM çalıştı mı"], "hayır (bu koşuda istenmedi)")


class EnsembleRealColumnsTest(unittest.TestCase):
    def test_ensemble_with_run_fold_output(self):
        """run_fold'un GERÇEK çıktı sütunlarıyla (qhat dahil) ensemble. Eski sürüm 'qhat not in index' ile çöküyordu."""
        from energyradar.evaluation import run_fold
        h = synth_hourly(120)
        ctx = build_frame(h)
        cfg = Config(gbm_trees=20, calib_days=20)
        fold = ("A", "2010-04-10", "2010-04-25")
        a = run_fold(ctx, fold, "Gradient Boosting", 0, cfg)
        b = run_fold(ctx, fold, "Gradient Boosting", 1, cfg).assign(model="LSTM")
        b["seed"] = 0
        a["seed"] = 0
        e = add_ensemble(pd.concat([a, b]))
        self.assertIn("Ensemble GB+LSTM", set(e["model"]))
        self.assertEqual(list(e.columns), list(pd.concat([a, b]).columns))
