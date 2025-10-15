# -*- coding: utf-8 -*-
# pages/Forecast.py — yalnızca GitHub Actions artifact’tan çalışır
# Gereksinimler: streamlit, pandas, requests, folium, streamlit-folium, scikit-learn, pyarrow

import io
import os
import json
import zipfile
from datetime import datetime, date, time, timedelta
from typing import Dict, Tuple, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# ====== dorecast API ======
try:
    from dorecast import run_forecasts, predict_at_datetime
except Exception:
    run_forecasts = None
    predict_at_datetime = None

# =========================
# Sabitler
# =========================
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT_NAME = "sf-crime-parquet"

GEOJSON_PATH_LOCAL_DEFAULT = "data/sf_cells.geojson"
GEOJSON_IN_ZIP_PATH_DEFAULT = "data/sf_cells.geojson"
RAW_GEOJSON_OWNER = "cem5113"
RAW_GEOJSON_REPO  = "crimepredict"

GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

st.set_page_config(page_title="🧭 Suç Tahmini (Forecast)", layout="wide")
st.title("🧭 Suç Tahmini (Forecast)")
st.caption("t0 seç, 24h/72h/7g tahmin üret; haritada E[olay] ve öncelik gör. (Veri kaynağı: GitHub Actions artifact)")

# =========================
# Yardımcılar — IO
# =========================
def _gh_headers():
    hdrs = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        hdrs["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return hdrs

def _digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())

@st.cache_data(show_spinner=True, ttl=15*60)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> bytes:
    if not GITHUB_TOKEN:
        raise RuntimeError("GitHub token yok. `st.secrets['github_token']` veya GITHUB_TOKEN env ayarlayın.")
    base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    r = requests.get(base, headers=_gh_headers(), timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    cand = [a for a in items if a.get("name") == artifact_name and not a.get("expired", False)]
    if not cand:
        raise FileNotFoundError(f"Artifact bulunamadı: {artifact_name}")
    cand.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    url = cand[0].get("archive_download_url")
    if not url:
        raise RuntimeError("archive_download_url bulunamadı")
    r2 = requests.get(url, headers=_gh_headers(), timeout=60)
    r2.raise_for_status()
    return r2.content

@st.cache_data(show_spinner=True, ttl=60*60)
def fetch_geojson_smart(path_local: str, path_in_zip: str, raw_owner: str, raw_repo: str) -> dict:
    # 1) Local
    try:
        if os.path.exists(path_local):
            with open(path_local, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    # 2) Artifact ZIP
    try:
        zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            memlist = zf.namelist()
            candidates = [n for n in memlist if n.endswith("/" + path_in_zip) or n.endswith(path_in_zip)]
            if candidates:
                with zf.open(candidates[0]) as f:
                    return json.load(io.TextIOWrapper(f, encoding="utf-8"))
    except Exception:
        pass
    # 3) Raw GitHub
    try:
        raw = f"https://raw.githubusercontent.com/{raw_owner}/{raw_repo}/main/{path_local}"
        import requests
        r = requests.get(raw, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

# =========================
# Harita yardımcıları
# =========================
def sanitize_props(geojson_dict: dict) -> dict:
    def _fix(v):
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        if isinstance(v, (np.generic, np.bool_)):
            return v.item()
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        return v
    feats = geojson_dict.get("features", [])
    for i in range(len(feats)):
        props = feats[i].get("properties") or {}
        feats[i]["properties"] = {k: _fix(v) for k, v in props.items()}
    geojson_dict["features"] = feats
    return geojson_dict

def geojson_centroids(geojson: dict) -> Dict[str, Tuple[float,float]]:
    out = {}
    for feat in geojson.get("features", []):
        props = feat.get("properties") or {}
        raw = props.get("geoid") or props.get("GEOID") or props.get("id") or ""
        key = _digits(raw)[:11]
        geom = feat.get("geometry") or {}
        coords = []
        if geom.get("type") == "Polygon":
            for ring in geom.get("coordinates", []):
                coords.extend(ring)
        elif geom.get("type") == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                for ring in poly:
                    coords.extend(ring)
        if coords:
            xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
            cx = sum(xs)/len(xs); cy = sum(ys)/len(ys)
            out[key] = (cy, cx)
    return out

def make_priority_color(priority: str) -> str:
    return {
        "zero": "#C8C8C8",
        "low": "#38A800",
        "medium": "#FFDD00",
        "high": "#FF8C00",
        "critical": "#CC0000",
    }.get(priority, "#CCCCCC")

def add_priority(df: pd.DataFrame, value_col: str = "pred_expected") -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["priority"] = []
        df["q25"] = df["q50"] = df["q75"] = 0.0
        return df
    qs = df[value_col].quantile([0, .25, .5, .75]).tolist()
    _, q25, q50, q75 = qs
    def lab(x: float) -> str:
        if x <= 0: return "zero"
        if x <= q25: return "low"
        if x <= q50: return "medium"
        if x <= q75: return "high"
        return "critical"
    df["priority"] = df[value_col].apply(lab)
    df["q25"], df["q50"], df["q75"] = float(q25), float(q50), float(q75)
    return df

def enrich_geojson_with_slice(geojson: dict, slice_df: pd.DataFrame) -> dict:
    d = slice_df.set_index("geoid")[["pred_expected","risk_score","priority"]].to_dict(orient="index")
    out_feats = []
    for feat in geojson.get("features", []):
        props = (feat.get("properties") or {}).copy()
        raw = props.get("geoid") or props.get("GEOID") or props.get("id") or ""
        key = _digits(raw)[:11]
        info = d.get(key)
        if info:
            props["geoid"] = key
            props["pred_expected"] = float(info["pred_expected"])
            props["risk_score"] = float(info["risk_score"])
            props["priority"] = str(info["priority"])
        else:
            props.setdefault("geoid", key)
            props.setdefault("pred_expected", 0.0)
            props.setdefault("risk_score", 0.0)
            props.setdefault("priority", "zero")
        out_feats.append({**feat, "properties": props})
    return {**geojson, "features": out_feats}

def render_map(geojson: dict, slice_df: pd.DataFrame, value_col: str = "pred_expected") -> Tuple[folium.Map, Dict[str, Tuple[float,float]]]:
    geojson = sanitize_props(geojson)
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
    centroids = geojson_centroids(geojson)
    heat_pts = []
    for _, row in slice_df.iterrows():
        g = str(row["geoid"]); val = float(row.get(value_col, 0.0))
        if g in centroids and val > 0:
            lat, lon = centroids[g]; heat_pts.append([lat, lon, val])
    if heat_pts:
        HeatMap(heat_pts, name="E[olay] ısı", radius=25, blur=20, max_zoom=13).add_to(m)
    def style_func(feat):
        p = (feat.get("properties") or {})
        col = make_priority_color(str(p.get("priority", "zero")))
        return {"fillColor": col, "color": "#555555", "weight": 0.5, "fillOpacity": 0.55}
    tooltip = folium.features.GeoJsonTooltip(
        fields=["geoid", "pred_expected", "priority"],
        aliases=["GEOID", "E[olay] (toplam)", "Öncelik"],
        localize=True, sticky=False, labels=True,
    )
    popup = folium.features.GeoJsonPopup(
        fields=["geoid", "pred_expected", "priority"],
        aliases=["GEOID", "E[olay] (toplam)", "Öncelik"],
        localize=True, labels=True, parse_html=False, sticky=False, max_width=350,
    )
    folium.GeoJson(geojson, name="Öncelik (choropleth)", style_function=style_func, tooltip=tooltip, popup=popup).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m, centroids

# =========================
# Sidebar — Artifact & Zaman
# =========================
with st.sidebar:
    st.header("Veri Kaynağı (History) — GitHub Artifact")
    st.caption(f"Repo: {OWNER}/{REPO} • Artifact: {ARTIFACT_NAME}")
    fetch_btn = st.button("Artifact’ı listele / yenile")
    if fetch_btn or "_art_zip" not in st.session_state:
        try:
            st.session_state["_art_zip"] = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
            st.success("Artifact indirildi.")
        except Exception as e:
            st.error(f"Artifact indirilemedi: {e}")
            st.stop()

    parquet_names = []
    try:
        with zipfile.ZipFile(io.BytesIO(st.session_state["_art_zip"])) as zf:
            parquet_names = [n for n in zf.namelist() if n.lower().endswith((".parquet",".pq",".parq"))]
    except Exception as e:
        st.error(f"ZIP açılamadı: {e}")
        st.stop()

    artifact_file = st.selectbox("ZIP içinden Parquet seç", parquet_names, index=parquet_names.index("risk_hourly.parquet") if "risk_hourly.parquet" in parquet_names else 0)

    st.divider()
    st.header("Zaman & Ufuk")
    t0_date = st.date_input("Uygulama açılış tarihi (t0)", value=date.today())
    t0_time = st.time_input("Saat", value=time(9,0))
    tau = st.slider("Temporal decay (τ, saat)", min_value=12, max_value=168, value=72, step=6)
    horizon = st.radio("Ufuk", ["24 saat (1s)", "72 saat (3s blok)", "1 hafta (24s)"])
    if "24 saat" in horizon:
        hour_sel = st.slider("Saat slotu", 0, 23, 18)
        modes = ["24h"]
    elif "72 saat" in horizon:
        start_hour = st.slider("Başlangıç saati", 0, 23, 0)
        bin_index = st.selectbox("Gösterilecek 3s blok", list(range(8)), index=6, format_func=lambda i: f"{i*3:02d}-{i*3+2:02d}")
        modes = ["72h"]
    else:
        day_index = st.selectbox("Gösterilecek gün (0..6)", list(range(7)), index=0)
        modes = ["7d"]

    st.divider()
    st.header("Harita (GeoJSON)")
    geojson_local = st.text_input("Local yol", value=GEOJSON_PATH_LOCAL_DEFAULT)
    geojson_zip   = st.text_input("Artifact ZIP içi yol", value=GEOJSON_IN_ZIP_PATH_DEFAULT)

    run_btn = st.button("🚀 Tahmin Üret")

# =========================
# Artifact’tan history oku
# =========================
def read_history_from_artifact(parquet_in_zip: str) -> pd.DataFrame:
    zip_bytes = st.session_state.get("_art_zip")
    if not zip_bytes:
        raise RuntimeError("Artifact bellekte yok.")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open(parquet_in_zip) as f:
            return pd.read_parquet(f)

def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # datetime
    dtcol = None
    for c in df.columns:
        if c.lower() in ["datetime","timestamp","time","date_time","dt"]:
            dtcol = c; break
    if dtcol is None:
        raise ValueError("datetime sütunu bulunamadı.")
    df.rename(columns={dtcol:"datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # GEOID
    gcol = None
    for c in df.columns:
        if c.lower() in ["geoid","cell_id","geoid10","geoid11","geoid_10","geoid_11","id"]:
            gcol = c; break
    if gcol is None:
        raise ValueError("GEOID sütunu bulunamadı.")
    df.rename(columns={gcol:"GEOID"}, inplace=True)
    df["GEOID"] = df["GEOID"].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)

    # Y_label
    ycol = None
    for c in df.columns:
        if c in ["Y_label","y_label","label","y","target","crime"]:
            ycol = c; break
    if ycol is None:
        raise ValueError("Y_label (hedef) sütunu bulunamadı.")
    if ycol != "Y_label":
        df.rename(columns={ycol:"Y_label"}, inplace=True)
    df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype(int).clip(0,1)
    return df[["datetime","GEOID","Y_label"] + [c for c in df.columns if c not in ["datetime","GEOID","Y_label"]]]

def suggest_top_geoids(hist: pd.DataFrame, t0: pd.Timestamp, k: int = 200) -> List[str]:
    past = hist[hist["datetime"] < t0]
    last30 = past[past["datetime"] >= (t0 - pd.Timedelta(days=30))]
    if last30.empty: last30 = past.tail(24*30)
    agg = (last30.groupby("GEOID")["Y_label"].sum().sort_values(ascending=False).head(k))
    return agg.index.astype(str).tolist()

def compute_base_rate(hist: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    past = hist[(hist["datetime"] < t0) & (hist["datetime"] >= (t0 - pd.Timedelta(days=30)))]
    if past.empty: past = hist[hist["datetime"] < t0].tail(24*30)
    if past.empty: return pd.DataFrame(columns=["GEOID","base_rate_daily"])
    daily = (past.assign(date=past["datetime"].dt.date).groupby(["GEOID","date"], as_index=False)["Y_label"].sum())
    base = daily.groupby("GEOID", as_index=False)["Y_label"].mean().rename(columns={"Y_label":"base_rate_daily"})
    return base

def attach_expected(out_df: pd.DataFrame, base_rate_df: pd.DataFrame) -> pd.DataFrame:
    df = out_df.copy()
    base = base_rate_df.copy()
    base["GEOID"] = base["GEOID"].astype(str)
    df["GEOID"] = df["GEOID"].astype(str)
    df = df.merge(base, on="GEOID", how="left")
    df["base_rate_daily"] = df["base_rate_daily"].fillna(df["base_rate_daily"].median() if not base.empty else 0.1)
    df["risk_score"] = df["prob_decayed"].astype(float)
    df["pred_expected"] = df["risk_score"] * df["base_rate_daily"] * (df["window_hours"].astype(float)/24.0)
    df["geoid"] = df["GEOID"].astype(str)
    df["date"] = pd.to_datetime(df["t_start"]).dt.date
    df["hour"] = pd.to_datetime(df["t_start"]).dt.hour
    return df

def slice_by_ui(df: pd.DataFrame, t0_date: date, horizon: str, hour_sel: int = 0, start_hour: int = 0, bin_index: int = 0, day_index: int = 0):
    if "24 saat" in horizon:
        d = t0_date
        m = (df["date"] == d) & (df["hour"] == int(hour_sel)) & (df["window_hours"] == 1)
        sl = df.loc[m].copy(); label = f"{d} — {hour_sel:02d}:00"
    elif "72 saat" in horizon:
        base_dt = datetime.combine(t0_date, time(hour=start_hour))
        target_start = base_dt + timedelta(hours=bin_index*3)
        m = (df["t_start"] == pd.Timestamp(target_start)) & (df["window_hours"] == 3)
        sl = df.loc[m].copy(); label = f"{t0_date} +72h — blok {bin_index} ({bin_index*3:02d}-{bin_index*3+2:02d})"
    else:
        start_day = t0_date + timedelta(days=day_index)
        m = (df["date"] == start_day) & (df["window_hours"] == 24)
        sl = df.loc[m].copy(); label = f"{t0_date} +7g — gün {day_index}"
    return sl, label

# =========================
# Çalıştır
# =========================
if run_btn:
    # 1) History (artifact içi Parquet)
    try:
        hist = read_history_from_artifact(artifact_file)
        hist = normalize_history(hist)
    except Exception as e:
        st.error(f"History okunamadı: {e}")
        st.stop()

    # 2) t0 & GEOID seçimi
    t0 = datetime.combine(t0_date, t0_time)
    geoids = suggest_top_geoids(hist, t0, k=200)
    if not geoids:
        st.error("GEOID listesi bulunamadı.")
        st.stop()

    # 3) Base rate
    base_rate = compute_base_rate(hist, t0)

    # 4) dorecast tahminleri
    if run_forecasts is None:
        st.error("dorecast.py import edilemedi. Proje köküne 'dorecast.py' ekleyin.")
        st.stop()
    # Geçici parquet (dorecast için)
    tmp_path = "history_tmp.parquet"
    hist.to_parquet(tmp_path, index=False)
    try:
        out = run_forecasts(tmp_path, t0.isoformat(sep=" "), geoids, modes, out_path=None, tau_hours=float(tau))
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

    if out is None or out.empty:
        st.warning("Tahmin üretilemedi veya boş çıktı.")
        st.stop()

    # 5) E[olay]
    out = attach_expected(out, base_rate)

    # 6) GeoJSON
    geojson = fetch_geojson_smart(GEOJSON_PATH_LOCAL_DEFAULT, GEOJSON_IN_ZIP_PATH_DEFAULT, RAW_GEOJSON_OWNER, RAW_GEOJSON_REPO)
    if not geojson:
        st.error("GeoJSON bulunamadı (local/artifact/raw).")
        st.stop()

    # 7) Slice + görselleştir
    sl, time_label = slice_by_ui(out, t0_date, horizon,
                                 hour_sel=locals().get("hour_sel", 0),
                                 start_hour=locals().get("start_hour", 0),
                                 bin_index=locals().get("bin_index", 0),
                                 day_index=locals().get("day_index", 0))
    if sl.empty:
        st.warning("Seçili dilimde veri bulunamadı.")
        st.stop()

    sl = add_priority(sl, value_col="pred_expected")
    geojson_enriched = enrich_geojson_with_slice(geojson, sl)
    folium_map, _ = render_map(geojson_enriched, sl, value_col="pred_expected")
    mres = st_folium(folium_map, width=None, height=600, use_container_width=True)

    # KPI + tablo
    st.subheader(f"Harita — {time_label}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kapsam (GEOID)", f"{sl['geoid'].nunique()}")
    c2.metric("Medyan E[olay]", f"{sl['pred_expected'].median():.2f}")
    c3.metric("Q75", f"{sl['pred_expected'].quantile(0.75):.2f}")
    c4.metric("Toplam E[olay]", f"{sl['pred_expected'].sum():.2f}")

    st.subheader("Top Hotspots")
    topk = sl.sort_values("pred_expected", ascending=False).head(50)
    st.dataframe(topk.reset_index(drop=True), use_container_width=True)
    st.download_button("Top-50 Hotspots CSV", data=topk.to_csv(index=False).encode("utf-8"), file_name="hotspots.csv", mime="text/csv")

    # Çıktının tamamı
    st.download_button("Tüm Tahminler (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="forecasts_all.csv", mime="text/csv")
