# models/forecaster.py
# Çok-ufuklu (multi-horizon) tahmin: Hepsi + suç türü bazlı
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# -----------------------------
# Yardımcılar
# -----------------------------
def _need(df: pd.DataFrame, cols: List[str]):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Eksik kolon(lar): {miss}")

def _time_feats(dt: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(dt)
    out = pd.DataFrame(index=dt.index)
    out["hour"] = dt.dt.hour
    out["dow"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["hour_of_week"] = (out["dow"] * 24 + out["hour"]).astype(int)
    return out

# -----------------------------
# Konfig
# -----------------------------
@dataclass
class ForecastConfig:
    horizons: List[int] = (24, 72, 24*7, 24*30, 24*90, 24*365)  # saat
    datetime_col: str = "datetime"
    geoid_col: str = "geoid"
    label_col: str = "Y_label"      # 0/1 (herhangi bir suç)
    crime_col: str = "crime_type"   # opsiyonel

# -----------------------------
# Model
# -----------------------------
class CrimeForecaster:
    """
    - 'Hepsi' (any crime) için ayrı bir model (y = Y_label).
    - Her suç türü için ayrı model (y_type = 1{Y_label=1 & crime_type==type}).
    - Conformal benzeri öngörü bandı için kalibrasyon artıkları tutulur.
    """
    def __init__(self, cfg: ForecastConfig):
        self.cfg = cfg
        # key: ("ALL" | crime_type, "pipe")
        self.models: Dict[Tuple[str, str], Pipeline] = {}
        # conformal residuals: key -> horizon -> np.ndarray
        self.residuals_: Dict[Tuple[str, str], Dict[int, np.ndarray]] = {}

    def _pipe(self) -> Pipeline:
        cat = ["hour", "dow", "month", "is_weekend", "hour_of_week", "horizon"]
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
        base = LogisticRegression(max_iter=1000)
        clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
        return Pipeline([("prep", pre), ("clf", clf)])

    def _prep_common(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        _need(df, [c.geoid_col, c.datetime_col, c.label_col])
        out = df.copy().reset_index(drop=True)
        out[c.geoid_col] = (out[c.geoid_col]
                            .astype(str).str.replace(r"\D", "", regex=True)
                            .str.zfill(11).str[:11])
        t = _time_feats(out[c.datetime_col])
        out = pd.concat([out, t], axis=1)
        out["horizon"] = 0
        return out

    def _make_train_table(self, base: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        target = "ALL" -> y = Y_label
        target = crime_type -> y = 1{Y_label=1 & crime_type==target}
        """
        c = self.cfg
        D = base.copy()
        if target == "ALL":
            y = D[c.label_col].astype(int)
        else:
            if c.crime_col not in D.columns:
                raise ValueError(f"{c.crime_col} kolonu yok, suç türü bazlı eğitim yapılamaz.")
            y = ((D[c.label_col] == 1) & (D[c.crime_col] == target)).astype(int)

        # ufukları tek tabloda istifleyelim (horizon feature ile)
        rows = []
        for h in self.cfg.horizons:
            tmp = D.copy()
            tmp["horizon"] = h
            # geleceğin etiketini t anına kaydır: y(t+h) -> y_h(t)
            # Basit yaklaşım: aynı (geoid, datetime-h) ile eşleşecek "gerçek" y'yi bul
            left = D[[self.cfg.geoid_col, self.cfg.datetime_col]].copy()
            left["__dt_future"] = pd.to_datetime(left[self.cfg.datetime_col]) + pd.to_timedelta(h, unit="h")
            right = D[[self.cfg.geoid_col, self.cfg.datetime_col]].copy()
            right["y"] = y.values
            m = left.merge(
                right.rename(columns={self.cfg.datetime_col: "__dt_future"}),
                on=[self.cfg.geoid_col, "__dt_future"], how="left"
            )
            tmp["y"] = m["y"].fillna(0).astype(int)
            rows.append(tmp)
        return pd.concat(rows, ignore_index=True)

    def fit(self, df: pd.DataFrame):
        """
        df: en az [geoid, datetime, Y_label] ve varsa [crime_type].
        """
        c = self.cfg
        base = self._prep_common(df)

        # Hedef listesi: Hepsi + (varsa) suç türleri
        targets = ["ALL"]
        if c.crime_col in base.columns:
            targets += sorted(base[c.crime_col].dropna().unique().tolist())

        for tgt in targets:
            T = self._make_train_table(base, tgt).sort_values(c.datetime_col)
            cut = int(len(T) * 0.8)
            tr, cal = T.iloc[:cut].copy(), T.iloc[cut:].copy()

            Xcols = ["hour", "dow", "month", "is_weekend", "hour_of_week", "horizon"]
            pipe = self._pipe()
            pipe.fit(tr[Xcols], tr["y"])
            self.models[(tgt, "pipe")] = pipe

            # conformal rezidüeller (horizon'a göre)
            self.residuals_[(tgt, "pipe")] = {}
            for h in self.cfg.horizons:
                C = cal[cal["horizon"] == h]
                if C.empty:
                    continue
                p = pipe.predict_proba(C[Xcols])[:, 1]
                e_true = C["y"].astype(float).values
                res = e_true - p  # gerçekleşen - tahmin
                self.residuals_[(tgt, "pipe")][h] = res

    def _predict_one(self, grid: pd.DataFrame, target: str, horizons: List[int]) -> pd.DataFrame:
        c = self.cfg
        base = self._prep_common(grid)
        Xcols = ["hour", "dow", "month", "is_weekend", "hour_of_week", "horizon"]

        if (target, "pipe") not in self.models:
            raise RuntimeError(f"Model yok: {target}. Önce fit() çalıştırın.")

        rows = []
        for h in horizons:
            D = base.copy()
            D["horizon"] = h
            p = self.models[(target, "pipe")].predict_proba(D[Xcols])[:, 1]
            # E[y] ~ p (ileride exposure eklenirse p × exposure yapılır)
            Ey = p
            # conformal bandı
            res = self.residuals_.get((target, "pipe"), {}).get(h, None)
            if res is not None and len(res) >= 30:
                lo_q, hi_q = np.quantile(res, [0.05, 0.95])
            else:
                lo_q, hi_q = -0.15, 0.15
            lo = np.clip(Ey + lo_q, 0, 1)
            hi = np.clip(Ey + hi_q, 0, 1)

            out = D[[c.geoid_col, c.datetime_col]].copy()
            out["horizon"] = h
            out["risk_score"] = p
            out["pred_expected"] = Ey
            out["pi_low"] = lo
            out["pi_high"] = hi
            out["target"] = target
            rows.append(out)
        return pd.concat(rows, ignore_index=True)

    def predict(self, grid: pd.DataFrame, horizons: List[int],
                crime: Optional[str] = None, return_all_crimes_if_none: bool = True) -> pd.DataFrame:
        """
        crime=None:
          - return_all_crimes_if_none=True ise TÜM suç türleri + 'ALL' birlikte döner.
          - False ise sadece 'ALL' (Hepsi) döner.
        crime='<type>': yalnızca o suç türü döner.
        """
        c = self.cfg
        if crime is not None and crime != "ALL":
            return self._predict_one(grid, crime, horizons)

        # crime None veya "ALL"
        outs = []
        if crime is None and return_all_crimes_if_none and (c.crime_col in grid.columns or True):
            # Tüm crime modellerini bul
            crime_targets = [t for (t, k) in self.models.keys() if k == "pipe" and t != "ALL"]
            for t in crime_targets:
                outs.append(self._predict_one(grid, t, horizons))
        # Hepsi
        outs.append(self._predict_one(grid, "ALL", horizons))
        return pd.concat(outs, ignore_index=True)
