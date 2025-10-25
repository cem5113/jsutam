# components/utils/forecast.py
from __future__ import annotations
from typing import Optional, Dict, List
import math

import numpy as np
import pandas as pd

# Paket içi sabitler (göreli import)
try:
    from .constants import KEY_COL as DEFAULT_KEY_COL
except Exception:
    DEFAULT_KEY_COL = "geoid"

__all__ = [
    "_normalize_events",
    "prob_ge_k",
    "poisson_q",
    "pois_pi90",
    "precompute_base_intensity",
    "aggregate_fast",
]


# ---------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------
def _normalize_events(
    events: Optional[pd.DataFrame],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    ts_col: str = "timestamp",
    type_col: str = "type",
) -> pd.DataFrame:
    """events DF'ini ts/lat/lon isimlerine normalize eder, UTC ts üretir."""
    if events is None or not isinstance(events, pd.DataFrame) or events.empty:
        return pd.DataFrame(columns=[ts_col, lat_col, lon_col, type_col])

    ev = events.copy()
    lower = {str(c).lower(): c for c in ev.columns}

    # ts
    if ts_col not in ev.columns:
        for cand in ["ts", "timestamp", "datetime", "occurred_at", "reported_at", "date_time", "time", "date"]:
            if cand in lower:
                ev[ts_col] = pd.to_datetime(ev[lower[cand]], utc=True, errors="coerce")
                break
        else:
            ev[ts_col] = pd.NaT
    else:
        ev[ts_col] = pd.to_datetime(ev[ts_col], utc=True, errors="coerce")

    # lat/lon
    if lat_col not in ev.columns:
        if "lat" in lower:
            ev = ev.rename(columns={lower["lat"]: lat_col})
        elif "latitude" in lower:
            ev = ev.rename(columns={lower["latitude"]: lat_col})
    if lon_col not in ev.columns:
        if "lon" in lower:
            ev = ev.rename(columns={lower["lon"]: lon_col})
        elif "longitude" in lower:
            ev = ev.rename(columns={lower["longitude"]: lon_col})

    # type opsiyonel
    if type_col not in ev.columns:
        ev[type_col] = None

    ev = ev.dropna(subset=[ts_col, lat_col, lon_col])
    return ev


def prob_ge_k(lmbd: float, k: int) -> float:
    """
    Poisson P(X >= k).
    k=1 için kapalı form kullanır; k>=2 için e^-λ λ^i / i! serisini güvenli toplar.
    """
    l = float(max(lmbd, 0.0))
    if k <= 1:
        return 1.0 - math.exp(-l)
    term = math.exp(-l)  # i=0
    cdf = term
    for i in range(1, k):
        term *= l / i
        cdf += term
    return float(max(0.0, 1.0 - cdf))


# --------- Poisson kantil yardımcıları (SciPy gerektirmez) ---------
def poisson_q(lmbd: float, q: float = 0.9, k_max: int | None = None) -> int:
    """
    Poisson(lmbd) için q-kantili (en küçük k: P(X<=k) >= q).
    SciPy olmadan stabil hesap (yinelemeli terim güncelleme).
    """
    l = float(max(lmbd, 0.0))
    if l == 0.0:
        return 0
    if k_max is None:
        # güvenli üst sınır
        k_max = int(l + 10.0 * math.sqrt(l) + 10)

    term = math.exp(-l)  # k=0
    cdf = term
    k = 0
    while cdf < q and k < k_max:
        k += 1
        term *= l / k
        cdf += term
    return k


def pois_pi90(lmbd: float) -> int:
    """Poisson için %90 üst eşik (90. yüzdelik)."""
    return poisson_q(lmbd, q=0.90)


# ---------------------------------------------------------------------
# 1) Taban yoğunluğu (baseline λ) — geoid başına sabit ağırlık
# ---------------------------------------------------------------------
def precompute_base_intensity(geo_df: pd.DataFrame, key_col: Optional[str] = None) -> pd.DataFrame:
    """
    GEO hücreleri için uniform bir temel yoğunluk üretir.
    Gerçek hayatta geçmiş oranlardan gelir; burada güvenli bir başlangıç sunuyor.
    Dönüş: [key_col, base_lambda]
    """
    kcol = key_col or DEFAULT_KEY_COL
    if geo_df is None or geo_df.empty or kcol not in geo_df.columns:
        return pd.DataFrame(columns=[kcol, "base_lambda"])

    n = len(geo_df)
    # Çok küçük de olsa >0 tut (Poisson hesapları için)
    base = np.full(n, 0.05, dtype=float)
    out = geo_df[[kcol]].copy()
    out["base_lambda"] = base
    return out


# ---------------------------------------------------------------------
# 2) Hızlı toplulaştırma ve tahmin (NR-lite + kategori filtresi)
# ---------------------------------------------------------------------
def aggregate_fast(
    start_iso: str,
    horizon_h: int,
    geo_df: pd.DataFrame,
    base_int: pd.DataFrame,
    *,
    events: Optional[pd.DataFrame] = None,
    near_repeat_alpha: float = 0.35,
    nr_lookback_h: int = 24,
    nr_radius_m: float = 400.0,  # şu an kullanılmıyor; ileride mekânsal NR için
    nr_decay_h: float = 12.0,
    filters: Optional[Dict[str, List[str]]] = None,
    key_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Uygulamanın beklediği ana tabloyu üretir.
    Çıktı kolonları:
      - [key_col], expected (λ), tier ("Yüksek"/"Orta"/"Hafif"), nr_boost (ops.), hour/dow
    """
    kcol = key_col or DEFAULT_KEY_COL
    assert kcol in geo_df.columns, f"{kcol} sütunu geo_df'te yok."

    # Başlangıç zamanı/ufuk
    try:
        start_ts = pd.to_datetime(start_iso, utc=True)
    except Exception:
        start_ts = pd.Timestamp.utcnow().floor("h")
    horizon_h = int(max(1, horizon_h))

    # 2.1 Tabana katıl
    df = geo_df[[kcol]].merge(base_int[[kcol, "base_lambda"]], on=kcol, how="left")
    df["lambda_base"] = df["base_lambda"].fillna(0.05).astype(float)

    # 2.2 Near-repeat (çok hafif, sadece sayısal boost) — olaylardan ısı üret
    if events is not None and isinstance(events, pd.DataFrame) and not events.empty:
        ev = _normalize_events(events)
        # lookback penceresi
        ev = ev[(ev["timestamp"] >= (start_ts - pd.Timedelta(hours=nr_lookback_h))) & (ev["timestamp"] < start_ts)]
        # kategori filtresi (filters={"cats":[...]})
        if filters and filters.get("cats"):
            cats = set([str(c).lower() for c in filters["cats"]])
            if "type" in ev.columns:
                ev = ev[ev["type"].astype(str).str.lower().isin(cats)]

        # GEOID eşlemesi varsa (ör: events içinde geoid/KEY_COL kolonu)
        if kcol in ev.columns:
            counts = ev.groupby(kcol).size().reindex(df[kcol], fill_value=0).to_numpy(dtype=float)
        else:
            # GEOID yoksa çok kaba fallback: tüm hücrelere aynı küçük artış
            counts = np.full(len(df), float(len(ev)) / max(len(df), 1), dtype=float)

        # zaman sönümüyle normalize edilmiş minik bir güçlendirme
        decay = math.exp(-nr_lookback_h / max(nr_decay_h, 1e-6))
        boost = near_repeat_alpha * counts * (1.0 - decay)
        df["nr_boost"] = boost
    else:
        df["nr_boost"] = 0.0

    # 2.3 Beklenen olay sayısı (λ)
    df["expected"] = (df["lambda_base"] + df["nr_boost"]).clip(lower=0.0)

    # 2.4 Tier (Yüksek/Orta/Hafif) — basit quantile eşiği
    if df["expected"].max() > 0:
        q66 = df["expected"].quantile(0.66)
        q33 = df["expected"].quantile(0.33)
    else:
        q66 = q33 = 0.0

    def _tier(x: float) -> str:
        if x >= q66 and x > 0:
            return "Yüksek"
        if x >= q33 and x > 0:
            return "Orta"
        return "Hafif"

    df["tier"] = [_tier(v) for v in df["expected"].to_numpy()]

    # 2.5 Heatmap için hour/dow sahte alanları (başlangıç saatinden)
    df["hour"] = start_ts.hour
    df["dow"] = start_ts.day_name()[:3]

    # Görsel/tooltip dostu sıralama
    df = df.sort_values("expected", ascending=False).reset_index(drop=True)

    # Temiz kolon seti
    keep = [kcol, "expected", "tier", "nr_boost", "hour", "dow"]
    for c in ["neighborhood", "centroid_lat", "centroid_lon"]:
        if c in geo_df.columns and c not in keep:
            keep.append(c)
    return df[keep]
