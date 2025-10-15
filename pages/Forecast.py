# pages/Forecast.py
import io
import os
import json
import zipfile
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Suç Tahmini (Forecast)", layout="wide")

# ---------------------------
# Kaynaklar
# ---------------------------
RAW_BASE = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/crime_prediction_data"
URL_RISK_CSV     = f"{RAW_BASE}/risk_hourly.csv"
URL_RISK_PARQUET = f"{RAW_BASE}/risk_hourly.parquet"
URL_C09          = f"{RAW_BASE}/sf_crime_09.csv"
URL_GEO          = f"{RAW_BASE}/sf_census_blocks_with_population.geojson"

OWNER = "cem5113"
REPO  = "crime_prediction_data"
ARTIFACT_NAME = "sf-crime-parquet"  # Actions artifact adı
GHTOKEN = st.secrets.get("github_token", "") or os.environ.get("GITHUB_TOKEN", "")

# ---------------------------
# Helpers
# ---------------------------
def _norm_geoid(series: pd.Series) -> pd.Series:
    """Rakam dışını at, 11 haneye sıfır doldur."""
    return (
        series.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(11)
        .str[:11]
    )

def _gh_headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if GHTOKEN:
        h["Authorization"] = f"Bearer {GHTOKEN}"
    return h

@st.cache_data(ttl=15 * 60)
def _fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> bytes:
    """En güncel aynı isimli artifact'ı indirir (ZIP bytes)."""
    if not GHTOKEN:
        raise RuntimeError("GitHub token yok. st.secrets['github_token'] ya da GITHUB_TOKEN ayarlayın.")
    api = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    r = requests.get(api, headers=_gh_headers(), timeout=30); r.raise_for_status()
    arts = [a for a in r.json().get("artifacts", []) if a.get("name")==artifact_name and not a.get("expired", False)]
    if not arts:
        raise FileNotFoundError(f"Artifact bulunamadı: {artifact_name}")
    arts.sort(key=lambda a: a.get("updated_at",""), reverse=True)
    url = arts[0]["archive_download_url"]
    r2 = requests.get(url, headers=_gh_headers(), timeout=60); r2.raise_for_status()
    return r2.content

def _pick_prob_col(df: pd.DataFrame) -> pd.Series:
    """Olasılık kolonunu seç (0/1 etiket kolonlarını ele). Bulamazsa 0.0 döner."""
    candidates = [c for c in ["proba","risk_score","score","p","prob"] if c in df.columns]
    risk_col = None
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        uniq = set(s.dropna().unique())
        if not uniq.issubset({0.0, 1.0}):  # ikili değilse olasılık olarak kabul
            risk_col = c
            break
    if risk_col is None and "proba" in df.columns:
        risk_col = "proba"
    return pd.to_numeric(df.get(risk_col, 0.0), errors="coerce").fillna(0.0)

# ---------------------------
# Yükleyiciler (CSV → Parquet → Artifact)
# ---------------------------
@st.cache_data(ttl=15 * 60)
def load_risk() -> pd.DataFrame:
    """risk_hourly verisini CSV→Parquet→Artifact sırasıyla dener, normalize eder."""
    err = []
    df, source = None, None

    # 1) RAW CSV
    try:
        r = requests.get(URL_RISK_CSV, timeout=20)
        if r.ok and r.content:
            df = pd.read_csv(io.BytesIO(r.content)); source = "raw_csv"
        else:
            raise requests.HTTPError(f"GET {URL_RISK_CSV} → {r.status_code}")
    except Exception as e:
        err.append(f"csv: {e}")

    # 2) RAW Parquet
    if df is None:
        try:
            r = requests.get(URL_RISK_PARQUET, timeout=20); r.raise_for_status()
            df = pd.read_parquet(io.BytesIO(r.content)); source = "raw_parquet"
        except Exception as e:
            err.append(f"parquet: {e}")

    # 3) Artifact ZIP
    if df is None:
        try:
            zbytes = _fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
            with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
                names = zf.namelist()
                # önce parquet
                cand = [n for n in names if n.endswith("/risk_hourly.parquet") or n.endswith("risk_hourly.parquet")]
                if cand:
                    with zf.open(cand[0]) as f:
                        df = pd.read_parquet(f); source = "artifact_parquet"
                else:
                    cand = [n for n in names if n.endswith("/risk_hourly.csv") or n.endswith("risk_hourly.csv")]
                    if cand:
                        with zf.open(cand[0]) as f:
                            df = pd.read_csv(f); source = "artifact_csv"
                    else:
                        raise FileNotFoundError("risk_hourly.* artifact içinde yok.")
        except Exception as e:
            err.append(f"artifact: {e}")

    if df is None:
        st.error("risk verisi okunamadı:\n" + "\n".join(err))
        raise RuntimeError("risk data not found in any source")

    # --- normalize ---
    df.columns = [c.strip().lower() for c in df.columns]
    if "geoid" not in df.columns:
        for alt in ["cell_id","geoid10","geoid11","geoid_10","geoid_11","id"]:
            if alt in df.columns:
                df["geoid"] = df[alt]; break
    df["geoid"] = _norm_geoid(df["geoid"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # risk_score (olasılık)
    df["risk_score"] = _pick_prob_col(df)

    st.caption(f"risk_hourly kaynağı: {source}")
    return df

@st.cache_data(ttl=24 * 60 * 60)
def load_geojson() -> dict:
    """GeoJSON: RAW, başarısızsa artifact'tan (opsiyonel), normalize GEOID."""
    # RAW
    try:
        r = requests.get(URL_GEO, timeout=30); r.raise_for_status()
        gj = r.json()
    except Exception as e:
        st.warning(f"GeoJSON RAW okunamadı: {e}")
        gj = {"type":"FeatureCollection","features":[]}

    for ft in gj.get("features", []):
        props = ft.get("properties", {}) or {}
        geo_val = props.get("GEOID", props.get("geoid", ""))
        props["GEOID"] = _norm_geoid(pd.Series([geo_val])).iloc[0]
        ft["properties"] = props
    return gj

@st.cache_data(ttl=30 * 60)
def load_exposure_fallback() -> pd.DataFrame:
    """
    risk_hourly içinde pred_expected yoksa,
    saatlik exposure için sf_crime_09.csv'den tahmin üret:
      exposure ≈ crime_last_7d / 7  (alt sınır 0.1)
    RAW yoksa artifact'tan dener.
    """
    # RAW
    try:
        r = requests.get(URL_C09, timeout=20); r.raise_for_status()
        c9 = pd.read_csv(io.BytesIO(r.content))
        source = "raw_csv"
    except Exception:
        # Artifact
        try:
            zbytes = _fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
            with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
                names = zf.namelist()
                cand = [n for n in names if n.endswith("/sf_crime_09.csv") or n.endswith("sf_crime_09.csv")]
                if not cand:
                    return pd.DataFrame(columns=["geoid","hour_range","exposure_guess"])
                with zf.open(cand[0]) as f:
                    c9 = pd.read_csv(f); source = "artifact_csv"
        except Exception:
            return pd.DataFrame(columns=["geoid","hour_range","exposure_guess"])

    # GEOID normalize
    geocol = "GEOID" if "GEOID" in c9.columns else ("geoid" if "geoid" in c9.columns else None)
    if geocol is None:
        return pd.DataFrame(columns=["geoid","hour_range","exposure_guess"])
    c9["geoid"] = _norm_geoid(c9[geocol])

    # exposure
    base = pd.to_numeric(c9.get("crime_last_7d", 0), errors="coerce").fillna(0.0)
    c9["exposure_guess"] = (base / 7.0).clip(lower=0.1)

    keep = ["geoid","hour_range","exposure_guess"]
    cols = [c for c in keep if c in c9.columns]
    out = c9[cols].copy()
    if "hour_range" not in out.columns:
        # saat yoksa sadece GEOID düzeyinde ortalama
        out = out.groupby("geoid", as_index=False)["exposure_guess"].mean()
    st.caption("exposure kaynağı: " + ("RAW" if source=="raw_csv" else "Artifact"))
    return out

def color_layer(geojson: dict, df_layer: pd.DataFrame):
    dmap = df_layer.set_index("geoid")["risk_score"].to_dict()
    qs = df_layer["risk_score"].quantile([0, .25, .5, .75, 1]).tolist()
    return pdk.Layer(
        "GeoJsonLayer",
        geojson,
        stroked=False,
        opacity=0.7,
        pickable=True,
        get_fill_color={
            "function": """
            const M = Object.fromEntries(py_dmap), Q = py_qs;
            return (f) => {
              const g = String(f.properties.GEOID || "")
                .replace(/\\D/g,"").padStart(11,"0").slice(0,11);
              const p = (M[g]===undefined) ? 0 : M[g];
              if (p <= Q[1]) return [178,223,138,220];  // low
              if (p <= Q[2]) return [255,255,178,220];  // medium
              if (p <= Q[3]) return [254,204,92,230];   // high
              return [227,26,28,235];                  // critical
            };
            """
        },
        parameters={"py_dmap": list(dmap.items()), "py_qs": qs},
    )

# ---------------------------
# UI
# ---------------------------
st.title("🧭 Suç Tahmini (Forecast)")
colL, colR = st.columns([3, 1], gap="large")

with colR:
    df_risk = load_risk()
    hours   = sorted(df_risk.get("event_hour", pd.Series(range(24))).dropna().unique().tolist())
    dows    = sorted(df_risk.get("day_of_week", pd.Series([0,1,2,3,4,5,6])).dropna().unique().tolist())
    seasons = sorted(df_risk.get("season", pd.Series(["Winter","Spring","Summer","Fall"])).dropna().unique().tolist())

    sel_hour   = st.select_slider("Saat", options=hours, value=hours[0] if hours else 0)
    sel_dow    = st.selectbox("Haftanın Günü (0=Mon ... 6=Sun)", options=dows, index=0)
    sel_season = st.selectbox("Sezon", options=seasons, index=0)
    topn       = st.slider("Top-K (kritik liste)", 10, 200, 50)

# Filtre
mask = pd.Series(True, index=df_risk.index)
if "event_hour" in df_risk.columns: mask &= (df_risk["event_hour"] == sel_hour)
if "day_of_week" in df_risk.columns: mask &= (df_risk["day_of_week"] == sel_dow)
if "season" in df_risk.columns:      mask &= (df_risk["season"] == sel_season)
layer_df = df_risk.loc[mask].copy()

# E[olay] = pred_expected varsa kullan; yoksa risk_score × exposure_guess
if "pred_expected" not in layer_df.columns:
    exp = load_exposure_fallback()
    if exp.empty:
        st.warning("Exposure kaynağı yüklenemedi → geçici 0.3 tabanı kullanılıyor.")
        layer_df["exposure_guess"] = 0.3
    else:
        if ("hour_range" in layer_df.columns) and ("hour_range" in exp.columns):
            layer_df = layer_df.merge(
                exp[["geoid","hour_range","exposure_guess"]],
                on=["geoid","hour_range"], how="left"
            )
        else:
            exp_geo = exp.groupby("geoid", as_index=False)["exposure_guess"].mean()
            layer_df = layer_df.merge(exp_geo, on="geoid", how="left")
        layer_df["exposure_guess"] = layer_df["exposure_guess"].fillna(0.3)
    layer_df["pred_expected"] = (layer_df["risk_score"] * layer_df["exposure_guess"]).round(3)
else:
    layer_df["pred_expected"] = pd.to_numeric(layer_df["pred_expected"], errors="coerce").fillna(0.0).round(3)

# Harita
geojson = load_geojson()
layer   = color_layer(geojson, layer_df)
view    = pdk.ViewState(latitude=37.76, longitude=-122.44, zoom=11)

with colL:
    st.subheader("Risk Haritası (seçili saat)")
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={
            "html": (
                "<b>GEOID:</b> {GEOID}<br/>"
                "<b>Risk (p):</b> {risk_score}<br/>"
                "<b>E[olay] (beklenen):</b> {pred_expected}"
            )
        }
    ))

with colR:
    st.markdown("**Kritik Top-K (E[olay] yüksek)**")
    top = (
        layer_df[["geoid","risk_score","pred_expected"]]
        .sort_values(["pred_expected","risk_score"], ascending=False)
        .head(topn)
        .reset_index(drop=True)
    )
    st.dataframe(top, use_container_width=True)
    if not layer_df.empty:
        q25, q50, q75 = layer_df["risk_score"].quantile([.25,.5,.75]).round(4)
        st.caption(f"Q25={q25:.4f} • Q50={q50:.4f} • Q75={q75:.4f}")
