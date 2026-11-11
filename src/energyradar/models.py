"""Tahmin modelleri. Ortak arayüz (gün öncesi tahmin, saatlik, %90 aralık için 5/50/95. yüzdelik):

    model.fit(ctx, train_days)           ctx: build_frame çıktısı (tüm tarih), train_days: eğitim günleri
    model.predict(ctx, days) -> DataFrame (indeks: saat; sütunlar p05, p50, p95; aralık yoksa p05/p95 NaN)

Modeller yalnızca `ctx` içindeki gün öncesi bilinen özellikleri kullanır (features.py). Gerçek hedef `y` yalnızca
eğitimde ve değerlendirmede kullanılır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .features import FEATURES, columns_for

MODEL_NAMES = ["Seasonal Naive", "Gradient Boosting", "LSTM"]

try:  # LightGBM varsa o, yoksa scikit-learn HistGradientBoosting (aynı arayüz, sonuçlara hangisinin kullanıldığı yazılır)
    import lightgbm as _lgb  # noqa: F401
    GBM_BACKEND = "lightgbm"
except Exception:  # pragma: no cover - ortama bağlı
    GBM_BACKEND = "sklearn-hgb"


def _rows(ctx: pd.DataFrame, days) -> pd.DataFrame:
    return ctx[ctx["day"].isin(set(pd.DatetimeIndex(days)))]


class SeasonalNaive:
    """Baseline: geçen haftanın aynı saati (lag168); o eksikse dünün aynı saati (lag24). Aralık üretmez."""
    name = "Seasonal Naive"

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg, self.seed = cfg, seed

    def fit(self, ctx, train_days):
        return self

    def predict(self, ctx, days):
        r = _rows(ctx, days)
        p = r["lag168"].fillna(r["lag24"])
        return pd.DataFrame({"p05": np.nan, "p50": p, "p95": np.nan}, index=r.index)


class GradientBoosting:
    """Kuantil regresyonlu gradyan artırma (5/50/95). Takvim + gecikme özellikleri."""
    name = "Gradient Boosting"

    def __init__(self, cfg: Config, seed: int = 0, features=None):
        self.cfg, self.seed = cfg, seed
        self.features = list(features) if features else list(FEATURES)
        self.models = {}

    def _make(self, q):
        if GBM_BACKEND == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=self.cfg.gbm_trees, learning_rate=0.05,
                                     num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     random_state=self.seed, verbose=-1)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=self.cfg.gbm_trees,
                                             learning_rate=0.05, random_state=self.seed)

    def fit(self, ctx, train_days):
        r = _rows(ctx, train_days)
        r = r[r["y"].notna()]
        for q in self.cfg.quantiles:
            self.models[q] = self._make(q).fit(r[self.features], r["y"])
        return self

    def predict(self, ctx, days):
        r = _rows(ctx, days)
        p = np.column_stack([self.models[q].predict(r[self.features]) for q in self.cfg.quantiles])
        p = np.sort(p, axis=1)   # kuantillerin birbirini kesmesini önle
        return pd.DataFrame(p, index=r.index, columns=["p05", "p50", "p95"])


class LSTMForecaster:
    """PyTorch LSTM: girdi son 168 saat (değer + eksiklik maskesi + saat sin/cos), hedef günün takvimi koşul olarak
    eklenir; çıktı hedef günün 24 saati x 3 kuantil (pinball kaybı, eksik hedefler maskelenir). Eksik girdi saatleri,
    eğitim verisinin hafta-saat ortalamasıyla doldurulur ve maske kanalıyla işaretlenir."""
    name = "LSTM"

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg, self.seed = cfg, seed
        self.net = None

    # ---- veri hazırlama (torch gerektirmez, test edilebilir)
    def _prepare(self, ctx, days):
        y = ctx["y"].values
        idx = ctx.index
        pos = {t: i for i, t in enumerate(idx)}
        X, C, Y, M, keep = [], [], [], [], []
        how = self.how_mean_  # (168,) hafta-saat ortalaması (z-skorlu)
        for d in pd.DatetimeIndex(days):
            i0 = pos.get(d)
            if i0 is None or i0 < 168 or i0 + 24 > len(y):
                continue
            hist = y[i0 - 168:i0]
            mask = ~np.isnan(hist)
            z = (hist - self.mu_) / self.sd_
            win = idx[i0 - 168:i0]
            hrs = win.hour.values
            z = np.where(mask, z, how[win.dayofweek.values * 24 + hrs])   # eksik saat -> hafta-saat ortalaması
            X.append(np.stack([z, mask.astype(float), np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24)], 1))
            tgt = y[i0:i0 + 24]
            r = ctx.iloc[i0]
            cond = np.zeros(7 + 1 + 2)
            cond[int(r["dow"])] = 1
            cond[7] = r["is_holiday"]
            cond[8] = np.sin(2 * np.pi * r["month"] / 12)
            cond[9] = np.cos(2 * np.pi * r["month"] / 12)
            C.append(cond)
            Y.append((tgt - self.mu_) / self.sd_)
            M.append(~np.isnan(tgt))
            keep.append(d)
        return (np.array(X, dtype=np.float32), np.array(C, dtype=np.float32),
                np.nan_to_num(np.array(Y, dtype=np.float32)), np.array(M, dtype=np.float32), keep)

    def fit(self, ctx, train_days):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        r = _rows(ctx, train_days)
        yt = r["y"].dropna()
        self.mu_, self.sd_ = float(yt.mean()), float(yt.std() + 1e-9)
        prof = ((r["y"] - self.mu_) / self.sd_).groupby(r.index.dayofweek * 24 + r.index.hour).mean()
        self.how_mean_ = prof.reindex(range(168)).fillna(0.0).values
        X, C, Y, M, _ = self._prepare(ctx, train_days)
        if len(X) < 30:
            raise ValueError("LSTM için yeterli eğitim günü yok.")
        qs = torch.tensor(self.cfg.quantiles, dtype=torch.float32)

        class Net(nn.Module):
            def __init__(s, hid):
                super().__init__()
                s.lstm = nn.LSTM(4, hid, batch_first=True)
                s.head = nn.Sequential(nn.Linear(hid + 10, 128), nn.ReLU(), nn.Linear(128, 24 * len(qs)))

            def forward(s, x, c):
                _, (h, _) = s.lstm(x)
                return s.head(torch.cat([h[-1], c], 1)).view(-1, 24, len(qs))

        self.net = Net(self.cfg.lstm_hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lstm_lr, weight_decay=1e-5)
        Xt, Ct, Yt, Mt = map(torch.from_numpy, (X, C, Y, M))
        g = torch.Generator().manual_seed(self.seed)
        self.loss_history_ = []
        for _ in range(self.cfg.lstm_epochs):
            perm = torch.randperm(len(Xt), generator=g)
            tot = 0.0
            for b in range(0, len(perm), self.cfg.lstm_batch):
                ix = perm[b:b + self.cfg.lstm_batch]
                out = self.net(Xt[ix], Ct[ix])
                err = Yt[ix].unsqueeze(-1) - out
                loss = (torch.maximum(qs * err, (qs - 1) * err) * Mt[ix].unsqueeze(-1)).sum() / (Mt[ix].sum() * len(qs) + 1e-9)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(ix)
            self.loss_history_.append(tot / len(Xt))
        return self

    def predict(self, ctx, days):
        import torch
        X, C, _, _, keep = self._prepare(ctx, days)
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(X), torch.from_numpy(C)).numpy()
        out = np.sort(out * self.sd_ + self.mu_, axis=2)
        rows = []
        for k, d in enumerate(keep):
            idx = pd.date_range(d, periods=24, freq="h")
            rows.append(pd.DataFrame(out[k], index=idx, columns=["p05", "p50", "p95"]))
        return pd.concat(rows) if rows else pd.DataFrame(columns=["p05", "p50", "p95"])


MODELS = {"Seasonal Naive": SeasonalNaive, "Gradient Boosting": GradientBoosting, "LSTM": LSTMForecaster}


def make_model(name: str, cfg: Config, seed: int = 0):
    """'Gradient Boosting' (temel özellikler) veya 'GB:calendar+lags+rhythm' (özellik grupları, ablasyon için)."""
    if name.startswith("GB:"):
        groups = [g for g in name[3:].split("+") if g]
        return GradientBoosting(cfg, seed, features=columns_for(groups))
    if name not in MODELS:
        raise ValueError(f"Bilinmeyen model: {name}. Seçenekler: {list(MODELS)}")
    return MODELS[name](cfg, seed)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False
