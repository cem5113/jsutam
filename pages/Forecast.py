# -*- coding: utf-8 -*-
# pages/Forecast.py
# Gereksinimler: streamlit, pandas, requests, folium, streamlit-folium

import io
import os
import json
import zipfile
from datetime import datetime, date, timedelta
from typing import Dict, Tuple, List, Optional

import pandas as pd
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

import numpy as np

def sanitize_props(geojson_dict: dict) -> dict:
    """GeoJSON properties içindeki NaN / numpy tipleri / Timestamp'leri JSON uyumlu hale getirir."""
    def _fix(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (np.generic, np.bool_)):   # np.int64, np.float32, np.bool_ -> native
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

# =========================
# Sabitler
# =========================
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT_NAME = "sf-crime-parquet"
EXPECTED_PARQUET = "risk_hourly.parquet"

GEOJSON_PATH_LOCAL_DEFAULT = "data/sf_cells.geojson"
GEOJSON_IN_ZIP_PATH_DEFAULT = "data/sf_cells.geojson"

RAW_GEOJSON_OWNER = "cem5113"
RAW_GEOJSON_REPO  = "crimepredict"

GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

st.set_page_config(page_title="🧭 Suç Tahmini (Forecast)", layout="wide")


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
        r = requests.get(raw, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

# =========================
# Veri — risk_hourly + pred_expected
# =========================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # geoid
    if "geoid" not in df.columns:
        for alt in ["cell_id","geoid10","geoid11","geoid_10","geoid_11","id"]:
            if alt in df.columns:
                df["geoid"] = df[alt]
                break
    df["geoid"] = df["geoid"].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)
    # date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    # hour / hour_range
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int).clip(0,23)
    if "hour_range" in df.columns:
        # "18-20" -> start hour
        hr0 = df["hour_range"].astype(str).str.extract(r"(\d{1,2})")[0].astype(float)
        df["hour_from"] = hr0.fillna(0).astype(int).clip(0,23)
    # risk_score alias
    if "risk_score" not in df.columns:
        if "proba" in df.columns: df = df.rename(columns={"proba":"risk_score"})
        elif "risk" in df.columns: df = df.rename(columns={"risk":"risk_score"})
    return df

@st.cache_data(show_spinner=True, ttl=15*60)
def read_risk_from_artifact() -> pd.DataFrame:
    zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        matches = [n for n in memlist if n.endswith("/" + EXPECTED_PARQUET) or n.endswith(EXPECTED_PARQUET)]
        if not matches:
            raise FileNotFoundError(f"Zip içinde {EXPECTED_PARQUET} yok.")
        with zf.open(matches[0]) as f:
            df = pd.read_parquet(f)
    return _normalize_cols(df)

@st.cache_data(ttl=30*60)
def load_exposure_fallback() -> pd.DataFrame:
    """sf_crime_09.csv'dan exposure tahmini (varsa)"""
    try:
        RAW = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/main/crime_prediction_data/sf_crime_09.csv"
        c9 = pd.read_csv(RAW)
        if "GEOID" not in c9.columns:
            c9["GEOID"] = c9["GEOID"].astype(str).str.extract(r"(\d+)").fillna("").str[:11]
        c9["GEOID"] = c9["GEOID"].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)
        # saatlik normalize opsiyonel; burada günlük ortalama alıyoruz
        c9["exposure_guess"] = (c9.get("crime_last_7d", 0) / 7.0).clip(lower=0.1)
        keep = ["GEOID","hour_range","exposure_guess"]
        return c9[[c for c in keep if c in c9.columns]]
    except Exception:
        return pd.DataFrame(columns=["GEOID","hour_range","exposure_guess"])

def ensure_pred_expected(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pred_expected" in df.columns:
        df["pred_expected"] = pd.to_numeric(df["pred_expected"], errors="coerce")
        return df
    # fallback ile exposure
    exp = load_exposure_fallback()
    if not exp.empty:
        # hour_range bazlı eşleşme varsa kullan
        if "hour_range" in df.columns and "hour_range" in exp.columns:
            df = df.merge(exp.rename(columns={"GEOID":"geoid"}), on=["geoid","hour_range"], how="left")
        else:
            df = df.merge(exp.rename(columns={"GEOID":"geoid"})[["geoid","exposure_guess"]], on="geoid", how="left")
    if "exposure_guess" not in df.columns or df["exposure_guess"].isna().all():
        # tamamen yoksa risk ortalamasını proxy al
        df["exposure_guess"] = (
            df.groupby("geoid")["risk_score"].transform("mean").clip(lower=0.1)
        )
    df["pred_expected"] = (df["risk_score"].astype(float) * df["exposure_guess"].astype(float)).fillna(0.0)
    return df

# =========================
# Zaman pencereleri
# =========================
def slice_24h(df: pd.DataFrame, d: date, hour: int) -> pd.DataFrame:
    m = (df["date"] == d)
    if "hour" in df.columns:
        m = m & (df["hour"] == int(hour))
    elif "hour_from" in df.columns:
        m = m & (df["hour_from"] == int(hour))
    return df.loc[m].copy()

def slice_72h_bins(df: pd.DataFrame, start_dt: datetime, bin_index: int) -> Tuple[pd.DataFrame, List[str]]:
    """72 saatlik pencerede 3'er saatlik 24 blok (0..23). bin_index: gösterilecek blok."""
    dt_end = start_dt + timedelta(hours=72)
    m_date = (pd.to_datetime(df["date"]).between(start_dt.date(), dt_end.date()))
    sdf = df.loc[m_date].copy()
    # hour zorunlu; yoksa hour_from
    if "hour" not in sdf.columns and "hour_from" in sdf.columns:
        sdf["hour"] = sdf["hour_from"]
    sdf["bin"] = (sdf["hour"] // 3).astype(int)  # 0..7 günde 24*? (gün sınırı önemli değil)
    # seçili 72h aralığa indir
    # saat mutlak değil; pratikte kullanıcı seçimi + tarih ile eşleriz
    # burada basitçe sadece bin'e göre gösterimi yapacağız
    bins_labels = [f"{i*3:02d}-{i*3+2:02d}" for i in range(8)]
    show = sdf[sdf["bin"] == bin_index].copy()
    # agregasyon
    show = show.groupby(["geoid"], as_index=False).agg(
        risk_score=("risk_score","mean"),
        pred_expected=("pred_expected","sum"),
    )
    return show, bins_labels

def slice_weekly(df: pd.DataFrame, start_date: date, day_index: int) -> Tuple[pd.DataFrame, List[str]]:
    """7 gün; day_index=0..6 -> gösterilecek gün."""
    end_date = start_date + timedelta(days=7)
    m = (df["date"] >= start_date) & (df["date"] < end_date)
    sdf = df.loc[m].copy()
    show = sdf.groupby(["geoid","date"], as_index=False).agg(
        risk_score=("risk_score","mean"),
        pred_expected=("pred_expected","sum"),
    )
    days = sorted(show["date"].unique())
    if not days:
        return show.iloc[0:0], []
    day_index = max(0, min(day_index, len(days)-1))
    dsel = days[day_index]
    return show.loc[show["date"] == dsel].copy(), [str(d) for d in days]

def add_priority(df: pd.DataFrame, value_col: str = "pred_expected") -> pd.DataFrame:
    df = df.copy()
    qs = df[value_col].quantile([0, .25, .5, .75]).tolist()
    q0, q25, q50, q75 = qs
    def lab(x: float) -> str:
        if x <= 0: return "zero"
        if x <= q25: return "low"
        if x <= q50: return "medium"
        if x <= q75: return "high"
        return "critical"
    df["priority"] = df[value_col].apply(lab)
    df["q25"], df["q50"], df["q75"] = q25, q50, q75
    return df

# =========================
# Top-3 suç tipi (opsiyonel)
# =========================
def compute_top3_offense(slice_df: pd.DataFrame) -> pd.DataFrame:
    """slice_df içinde offense/crime_type varsa GEOID bazında ilk 3'ü döndür."""
    df = slice_df.copy()
    cand_cols = [c for c in ["offense","offense_category","crime_type","primary_type"] if c in df.columns]
    if not cand_cols:
        return pd.DataFrame(columns=["geoid","top3"])
    col = cand_cols[0]
    # pred_expected varsa onun toplamına göre; yoksa sayım
    if "pred_expected" in df.columns:
        grp = df.groupby(["geoid", col], as_index=False)["pred_expected"].sum()
        grp = grp.sort_values(["geoid","pred_expected"], ascending=[True, False])
    else:
        grp = df.groupby(["geoid", col], as_index=False).size().rename(columns={"size":"cnt"})
        grp = grp.sort_values(["geoid","cnt"], ascending=[True, False])
    top3 = (
        grp.groupby("geoid")
           .head(3)
           .groupby("geoid")[col]
           .apply(list)
           .reset_index(name="top3")
    )
    return top3

# =========================
# GeoJSON zenginleştirme + ısı verisi
# =========================
def geojson_centroids(geojson: dict) -> Dict[str, Tuple[float,float]]:
    """Basit centroid (koordinat ortalaması). Shapely kullanmadan hızlıca."""
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
            out[key] = (cy, cx)  # folium lat, lon
    return out

def enrich_geojson_with_slice(geojson: dict, slice_df: pd.DataFrame, top3_df: pd.DataFrame) -> dict:
    d = slice_df.set_index("geoid")[["pred_expected","risk_score","priority"]].to_dict(orient="index")
    tmap = top3_df.set_index("geoid")["top3"].to_dict() if not top3_df.empty else {}
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
            t3 = tmap.get(key)
            if t3:
                props["top3"] = ", ".join(map(str, t3))
            else:
                # <<< kritik: her feature'da top3 anahtarı olsun
                props.setdefault("top3", "")
        else:
            props.setdefault("geoid", key)
            props.setdefault("pred_expected", 0.0)
            props.setdefault("risk_score", 0.0)
            props.setdefault("priority", "zero")
            # <<< kritik: her feature'da top3 anahtarı olsun
            props.setdefault("top3", "")
        out_feats.append({**feat, "properties": props})
    return {**geojson, "features": out_feats}

def make_priority_color(priority: str) -> str:
    cmap = {
        "zero":     "#C8C8C8",
        "low":      "#38A800",
        "medium":   "#FFDD00",
        "high":     "#FF8C00",
        "critical": "#CC0000",
    }
    return cmap.get(priority, "#CCCCCC")

def render_map(geojson: dict, slice_df: pd.DataFrame, value_col: str = "pred_expected") -> Tuple[folium.Map, Dict[str, Tuple[float,float]]]:
    # JSON serileştirme için temizle (NaN, numpy tipleri, Timestamp -> safe)
    geojson = sanitize_props(geojson)

    # merkez SF
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")

    # Heatmap: centroids + intensity
    centroids = geojson_centroids(geojson)
    heat_pts = []
    for _, row in slice_df.iterrows():
        g = str(row["geoid"])
        val = float(row.get(value_col, 0.0))
        if g in centroids and val > 0:
            lat, lon = centroids[g]
            heat_pts.append([lat, lon, val])
    if heat_pts:
        HeatMap(heat_pts, name="E[olay] ısı", radius=25, blur=20, max_zoom=13).add_to(m)

    # Choropleth + tooltip/popup
    def style_func(feat):
        p = (feat.get("properties") or {})
        col = make_priority_color(str(p.get("priority", "zero")))
        return {"fillColor": col, "color": "#555555", "weight": 0.5, "fillOpacity": 0.55}

    # Tüm feature'larda bu alanlar olduğundan emin olduğumuz için fields sabit
    tooltip = folium.features.GeoJsonTooltip(
        fields=["geoid", "pred_expected", "priority"],
        aliases=["GEOID", "E[olay] (toplam)", "Öncelik"],
        localize=True,
        sticky=False,
        labels=True,
    )
    popup = folium.features.GeoJsonPopup(
        fields=["geoid", "pred_expected", "priority", "top3"],
        aliases=["GEOID", "E[olay] (toplam)", "Öncelik", "En olası 3 suç"],
        localize=True,
        labels=True,
        parse_html=False,
        sticky=False,
        max_width=350,
    )

    folium.GeoJson(
        geojson,
        name="Öncelik (choropleth)",
        style_function=style_func,
        tooltip=tooltip,
        popup=popup,
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m, centroids

# =========================
# UI
# =========================
st.title("🧭 Suç Tahmini (Forecast)")
st.caption("Zaman ufku seç, ısı haritasını gör, bir bölgeye tıkla → pop-up ve alttaki ‘Sonuç Kartı’ güncellensin.")

with st.sidebar:
    st.header("Zaman & Filtreler")
    horizon = st.radio("Ufuk", ["24 saat (saatlik)","72 saat (3s blok)","1 hafta (günlük)"])
    today = date.today()
    base_date = st.date_input("Tarih", value=today)
    if "24 saat" in horizon:
        hour_sel = st.slider("Saat", 0, 23, 18)
    elif "72 saat" in horizon:
        start_hour = st.slider("Başlangıç saati (72h)", 0, 23, 0)
        bin_index = st.selectbox("Gösterilecek 3s blok", list(range(8)), index=6, format_func=lambda i: f"{i*3:02d}-{i*3+2:02d}")
    else:
        day_index = st.selectbox("Gösterilecek gün (0..6)", list(range(7)), index=0)
    refresh = st.button("Veriyi Yenile")

    st.divider()
    st.subheader("Harita sınırları")
    geojson_local = st.text_input("Local yol", value=GEOJSON_PATH_LOCAL_DEFAULT)
    geojson_zip   = st.text_input("Artifact ZIP içi yol", value=GEOJSON_IN_ZIP_PATH_DEFAULT)

# veri oku
try:
    if refresh:
        fetch_latest_artifact_zip.clear()
        read_risk_from_artifact.clear()
        fetch_geojson_smart.clear()
        load_exposure_fallback.clear()
    risk = read_risk_from_artifact()
except Exception as e:
    st.error(f"Artifact indirilemedi/okunamadı: {e}")
    st.stop()

risk = ensure_pred_expected(risk)

# kategori filtresi (varsa)
cand_cols = [c for c in ["offense","offense_category","crime_type","primary_type"] if c in risk.columns]
if cand_cols:
    cat_col = cand_cols[0]
    with st.sidebar:
        cats = sorted([c for c in risk[cat_col].dropna().astype(str).unique() if c != ""])
        chosen = st.multiselect("Suç kategorisi (opsiyonel)", cats, default=[])
    if chosen:
        risk = risk[risk[cat_col].astype(str).isin(chosen)].copy()

# slice
if "24 saat" in horizon:
    sl = slice_24h(risk, base_date, hour_sel)
    time_label = f"{base_date} — {hour_sel:02d}:00"
elif "72 saat" in horizon:
    start_dt = datetime.combine(base_date, datetime.min.time()).replace(hour=start_hour)
    sl, bins_labels = slice_72h_bins(risk, start_dt, bin_index)
    time_label = f"{base_date} +72h — blok {bins_labels[bin_index]}"
else:
    sl, days_labels = slice_weekly(risk, base_date, day_index)
    time_label = f"{base_date} +7g — gün {day_index} ({days_labels[day_index] if days_labels else ''})"

if sl.empty:
    st.warning("Seçili aralıkta veri bulunamadı.")
    st.stop()

sl = add_priority(sl, value_col="pred_expected")

# top-3 offense
top3 = compute_top3_offense(sl)

# geojson
geojson = fetch_geojson_smart(
    path_local=geojson_local,
    path_in_zip=geojson_zip,
    raw_owner=RAW_GEOJSON_OWNER,
    raw_repo=RAW_GEOJSON_REPO,
)
if not geojson:
    st.error("GeoJSON bulunamadı (local/artifact/raw).")
    st.stop()

# enrich + harita
geojson_enriched = enrich_geojson_with_slice(geojson, sl, top3)
folium_map, centroids = render_map(geojson_enriched, sl, value_col="pred_expected")
mres = st_folium(folium_map, width=None, height=600, use_container_width=True)

st.subheader(f"Harita — {time_label}")
mres = st_folium(folium_map, width=None, height=600, use_container_width=True)

# KPI kutuları
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kapsam (GEOID)", f"{sl['geoid'].nunique()}")
c2.metric("Medyan E[olay]", f"{sl['pred_expected'].median():.2f}")
c3.metric("Q75", f"{sl['pred_expected'].quantile(0.75):.2f}")
c4.metric("Toplam E[olay]", f"{sl['pred_expected'].sum():.2f}")

# Top-K tablo
st.subheader("Top Hotspots")
topk = sl.sort_values("pred_expected", ascending=False).head(50)
st.dataframe(topk.reset_index(drop=True), use_container_width=True)
csv = topk.to_csv(index=False).encode("utf-8")
st.download_button("Top-50 Hotspots CSV", data=csv, file_name="hotspots.csv", mime="text/csv")

# =========================
# Seçili GEOID → Sonuç Kartı
# =========================
clicked_geoid: Optional[str] = None
if mres and mres.get("last_object_clicked"):
    props = mres["last_object_clicked"].get("properties") or {}
    clicked_geoid = _digits(props.get("geoid") or props.get("GEOID") or "")

if clicked_geoid:
    sel = sl[sl["geoid"] == clicked_geoid]
    if not sel.empty:
        total_e = sel["pred_expected"].sum()
        pr = sel["priority"].iloc[0]
        mean_p = sel["risk_score"].mean()

        # top-3 verisini çek
        t3 = top3[top3["geoid"] == clicked_geoid]["top3"]
        raw_t3 = t3.iloc[0] if len(t3) > 0 else None

        # top-3'u güvenli biçimde çöz
        def _parse_top3(x):
            import json
            if x is None:
                return []
            # zaten liste/tuple ise
            if isinstance(x, (list, tuple)):
                out = []
                for it in x:
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        out.append((str(it[0]), float(it[1])))
                    else:
                        # yalnızca isim ise skoru None kabul et
                        out.append((str(it), None))
                return out[:3]
            # metin ise: JSON dene
            s = str(x).strip()
            if not s:
                return []
            try:
                obj = json.loads(s)
                return _parse_top3(obj)
            except Exception:
                pass
            # "a:0.6,b:0.4" biçimi
            if ":" in s and "," in s:
                pairs = []
                for tok in s.split(","):
                    if ":" in tok:
                        k, v = tok.split(":", 1)
                        try:
                            pairs.append((k.strip(), float(v)))
                        except Exception:
                            pairs.append((k.strip(), None))
                    else:
                        pairs.append((tok.strip(), None))
                return pairs[:3]
            # yalnızca virgüllü isim listesi
            return [(tok.strip(), None) for tok in s.split(",")[:3]]

        t3_list = _parse_top3(raw_t3)

        # Ufuk etiketi
        horizon_label = locals().get("UFUK_LABEL") or locals().get("horizon_label") or "Seçili Dilim"

        # ---- Sonuç Kartı ----
        st.markdown("### 🟠 Sonuç Kartı")
        c1, c2, c3 = st.columns([1.1, 1.2, 1.1])

        with c1:
            st.markdown("**Bölge (GEOID)**")
            st.code(clicked_geoid, language="text")
            st.markdown("**Öncelik**")
            st.write(str(pr).title())
            st.markdown("**Ufuk**")
            st.write(horizon_label)

        with c2:
            st.markdown("**En olası suç türleri (Top-3)**")
            if t3_list:
                for name, score in t3_list:
                    if score is None:
                        st.write(f"• {name}")
                    else:
                        st.write(f"• {name}: {score:.2f}")
            else:
                st.caption("Top-3 tür bilgisi bulunamadı.")

        with c3:
            st.markdown("**Beklenen olay (ΣE[olay])**")
            st.metric(label="Toplam", value=f"{total_e:.2f}")
            st.markdown("**Olasılık ort. (p)**")
            st.metric(label="Mean p", value=f"{mean_p:.3f}")

        st.divider()
