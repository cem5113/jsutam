# -*- coding: utf-8 -*-
# app.py — Suç Tahmini Uygulaması (dorecast & Forecast page uyumlu)

import io, os, json, zipfile
from datetime import date
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

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

st.set_page_config(page_title="Suç Tahmini Uygulaması", layout="wide")

# =========================
# Yardımcılar (GENEL)
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

@st.cache_data(show_spinner=True, ttl=15*60)
def read_risk_from_artifact() -> pd.DataFrame:
    """risk_hourly.parquet -> kolonları normalize et."""
    zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        matches = [n for n in memlist if n.endswith("/" + EXPECTED_PARQUET) or n.endswith(EXPECTED_PARQUET)]
        if not matches:
            raise FileNotFoundError(f"Zip içinde {EXPECTED_PARQUET} yok. Örnek içerik: {memlist[:15]}")
        with zf.open(matches[0]) as f:
            df = pd.read_parquet(f)

    df.columns = [c.strip().lower() for c in df.columns]

    # GEOID
    if "geoid" not in df.columns:
        for alt in ["cell_id", "geoid10", "geoid11", "geoid_10", "geoid_11", "id"]:
            if alt in df.columns:
                df["geoid"] = df[alt]
                break
    df["geoid"] = df["geoid"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)

    # tarih
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # risk alias
    if "risk_score" not in df.columns:
        if "proba" in df.columns:
            df = df.rename(columns={"proba": "risk_score"})
        elif "risk" in df.columns:
            df = df.rename(columns={"risk": "risk_score"})

    return df

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
# Home sekmesi yardımcıları
# =========================
def daily_average(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    needed = {"geoid", "date", "risk_score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolon(lar): {', '.join(sorted(missing))}")
    return (
        df.groupby(["geoid", "date"], as_index=False)["risk_score"]
        .mean()
        .rename(columns={"risk_score": "risk_score_daily"})
    )

def classify_quantiles(daily_df: pd.DataFrame, day: date) -> pd.DataFrame:
    one = daily_df[daily_df["date"] == day].copy()
    if one.empty:
        return one
    q25, q50, q75 = one["risk_score_daily"].quantile([0.25, 0.5, 0.75]).tolist()
    def lab(x: float) -> str:
        if x <= q25:   return "low"
        elif x <= q50:return "medium"
        elif x <= q75:return "high"
        return "critical"
    one["risk_level"] = one["risk_score_daily"].apply(lab)
    one["q25"], one["q50"], one["q75"] = q25, q50, q75
    return one

def inject_properties(geojson_dict: dict, day_df: pd.DataFrame) -> dict:
    """GeoJSON'a risk ve renkleri enjekte et (Home için günlük ortalama)."""
    if not geojson_dict or day_df.empty:
        return geojson_dict
    df = day_df.copy()
    df["geoid_digits"] = df["geoid"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)
    df_key = df.groupby("geoid_digits", as_index=True)["risk_score_daily"].mean().to_frame().reset_index()
    df_key = df_key.rename(columns={"geoid_digits":"match_key"})
    dmap = df_key.set_index("match_key")
    feats = geojson_dict.get("features", [])
    enriched = 0
    q25 = float(df["risk_score_daily"].quantile(0.25))
    q50 = float(df["risk_score_daily"].quantile(0.50))
    q75 = float(df["risk_score_daily"].quantile(0.75))
    EPS = 1e-12
    COLOR_MAP = {
        "zero":[200,200,200],
        "low":[56,168,0],
        "medium":[255,221,0],
        "high":[255,140,0],
        "critical":[204,0,0],
    }
    out=[]
    for feat in feats:
        props = (feat.get("properties") or {}).copy()
        raw=None
        for k in ("geoid","GEOID","cell_id","id"):
            if k in props: raw = props[k]; break
        props.setdefault("display_id", str(raw if raw is not None else ""))
        key = _digits(raw)[:11] if raw is not None else ""
        lvl=None
        if key and key in dmap.index:
            val = float(dmap.loc[key, "risk_score_daily"])
            props["risk_score_daily"] = val
            props["risk_score_txt"] = f"{val:.4f}"
            if abs(val) <= EPS: lvl="zero"
            elif val <= q25:   lvl="low"
            elif val <= q50:   lvl="medium"
            elif val <= q75:   lvl="high"
            else:              lvl="critical"
            enriched += 1
        if lvl is None: lvl = props.get("risk_level", "zero")
        props["risk_level"] = lvl
        props["fill_color"] = COLOR_MAP.get(lvl, [220,220,220])
        out.append({**feat, "properties": props})
    st.caption(f"Eşleşme özeti → DF(tract,11h): {len(dmap)} anahtar, enjekte: {enriched}/{len(feats)}")
    return {**geojson_dict, "features": out}

def make_map_home(geojson_enriched: dict):
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_enriched,
        stroked=True,
        get_line_color=[80,80,80],
        line_width_min_pixels=0.5,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=0.65,
    )
    tooltip = {
        "html":"<b>GEOID:</b> {display_id}<br/><b>Risk:</b> {risk_level}<br/><b>Skor:</b> {risk_score_txt}",
        "style":{"backgroundColor":"#262730","color":"white"},
    }
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

# =========================
# UI — Sekmeler
# =========================
st.title("🗺️ Suç Tahmini Uygulaması")

tabs = st.tabs(["Home", "Forecast", "Planning", "Stats", "Reports"])

# ---------- Home ----------
with tabs[0]:
    st.subheader("Suç Risk Haritası — Günlük Ortalama")
    st.sidebar.header("GitHub Artifact")
    refresh = st.sidebar.button("Veriyi Yenile (artifact'i tazele)")
    try:
        if refresh:
            fetch_latest_artifact_zip.clear()
            read_risk_from_artifact.clear()
            fetch_geojson_smart.clear()
        risk_df = read_risk_from_artifact()
    except Exception as e:
        st.error(f"Artifact indirilemedi/okunamadı: {e}")
        st.stop()

    risk_daily = daily_average(risk_df)
    dates = sorted(risk_daily["date"].unique()) if "date" in risk_daily.columns else []
    sel_date = st.sidebar.selectbox("Gün seçin", dates, index=len(dates)-1, format_func=str) if dates else None

    st.sidebar.header("Harita Sınırları (GeoJSON)")
    geojson_local = st.sidebar.text_input("Local yol", value=GEOJSON_PATH_LOCAL_DEFAULT)
    geojson_zip   = st.sidebar.text_input("Artifact ZIP içi yol", value=GEOJSON_IN_ZIP_PATH_DEFAULT)

    geojson = fetch_geojson_smart(
        path_local=geojson_local,
        path_in_zip=geojson_zip,
        raw_owner=RAW_GEOJSON_OWNER,
        raw_repo=RAW_GEOJSON_REPO,
    )
    
    if sel_date:
        one_day = classify_quantiles(risk_daily, sel_date)
        if not one_day.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Q25", f"{one_day['q25'].iloc[0]:.4f}")
            c2.metric("Q50", f"{one_day['q50'].iloc[0]:.4f}")
            c3.metric("Q75", f"{one_day['q75'].iloc[0]:.4f}")

        st.subheader(f"Harita — {sel_date}")
        enriched = inject_properties(geojson, one_day) if (geojson and not one_day.empty) else geojson
        if not geojson:
            st.warning("GeoJSON bulunamadı (local/artifact/raw). Yolları kontrol edin.")
        else:
            make_map_home(enriched)
        st.subheader("Seçilen Gün Tablosu")
        table_df = one_day.drop(columns=["q25","q50","q75"], errors="ignore") if sel_date and not one_day.empty else pd.DataFrame()
        st.dataframe(
            table_df.sort_values("risk_score_daily", ascending=False) if not table_df.empty else table_df,
            use_container_width=True,
        )
        if not table_df.empty:
            csv = table_df.to_csv(index=False).encode("utf-8")
            st.download_button("Günlük tabloyu CSV indir", csv, file_name=f"risk_daily_{sel_date}.csv", mime="text/csv")
    else:
        st.info("Artifact içinde tarih bulunamadı. Forecast sekmesinden saatlik görünüme bakabilirsiniz.")

# ---------- Forecast (sayfaya link) ----------
with tabs[1]:
    st.subheader("🧭 Forecast")
    st.write("Saatlik/3s/günlük ufuk tahminleri için ayrı sayfayı açın:")
    try:
        st.page_link("pages/Forecast.py", label="Forecast sayfasını aç", icon="🧭")
    except Exception:
        st.info("Streamlit sürümünüz 'page_link' desteklemiyor olabilir. Lütfen sol menüden 'Forecast' sayfasını açın.")

# ---------- Planning ----------
with tabs[2]:
    st.subheader("🚓 Devriye Planlama (yakında)")
    st.info("Ekip sayısı / rota uzunluğu / dwell / çeşitlilik ayarı buraya taşınacak.")

# ---------- Stats ----------
with tabs[3]:
    st.subheader("📊 Suç İstatistikleri (yakında)")
    st.info("Saat-gün-ay dağılımları, ısı haritaları, tür dağılımı vs.")

# ---------- Reports ----------
with tabs[4]:
    st.subheader("🧾 Raporlar & Operasyonel Öneriler (yakında)")
    st.info("Günlük/Haftalık/Aylık rapor üretimi + PDF/CSV indirme burada.")
