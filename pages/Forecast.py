# pages/Forecast.py
import io, os, json, zipfile
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

st.set_page_config(page_title="Suç Tahmini (Forecast)", layout="wide")

# ====== GitHub ayarları (Home ile aynı kalıbı kullan) ======
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT_NAME = "sf-crime-parquet"
EXPECTED_PARQUET = "risk_hourly.parquet"

# GeoJSON için aynı fallback zinciri
GEOJSON_PATH_LOCAL_DEFAULT = "data/sf_cells.geojson"
GEOJSON_IN_ZIP_PATH_DEFAULT = "data/sf_cells.geojson"
RAW_GEOJSON_OWNER = "cem5113"
RAW_GEOJSON_REPO  = "crimepredict"

GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

def _gh_headers():
    hdrs = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        hdrs["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return hdrs

def _digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())

@st.cache_data(ttl=15*60, show_spinner=True)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> bytes:
    if not GITHUB_TOKEN:
        raise RuntimeError("GitHub token yok. st.secrets['github_token'] veya GITHUB_TOKEN ayarlayın.")
    r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts",
                     headers=_gh_headers(), timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    cand = [a for a in items if a.get("name")==artifact_name and not a.get("expired", False)]
    if not cand:
        raise FileNotFoundError(f"Artifact bulunamadı: {artifact_name}")
    cand.sort(key=lambda x: x.get("updated_at",""), reverse=True)
    url = cand[0]["archive_download_url"]
    r2 = requests.get(url, headers=_gh_headers(), timeout=60)
    r2.raise_for_status()
    return r2.content

@st.cache_data(ttl=15*60, show_spinner=True)
def read_risk_from_artifact() -> pd.DataFrame:
    """risk_hourly.parquet -> kolonları normalize et."""
    zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        matches = [n for n in memlist if n.endswith("/"+EXPECTED_PARQUET) or n.endswith(EXPECTED_PARQUET)]
        if not matches:
            raise FileNotFoundError(f"Zip içinde {EXPECTED_PARQUET} yok. Örnek içerik: {memlist[:15]}")
        with zf.open(matches[0]) as f:
            df = pd.read_parquet(f)

    # kolon adlarını normalize et
    df.columns = [c.strip().lower() for c in df.columns]

    # GEOID çıkar / normalize (11 hane)
    if "geoid" not in df.columns:
        for alt in ("cell_id","geoid10","geoid11","geoid_10","geoid_11","id"):
            if alt in df.columns:
                df["geoid"] = df[alt]; break
    df["geoid"] = df["geoid"].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)

    # saat/dow/season alanları değişken olabiliyor → alias
    if "event_hour" not in df.columns and "hour" in df.columns:
        df = df.rename(columns={"hour":"event_hour"})

    # risk kolonu alias
    if "risk_score" not in df.columns:
        if "proba" in df.columns: df = df.rename(columns={"proba":"risk_score"})
        elif "risk" in df.columns: df = df.rename(columns={"risk":"risk_score"})

    # hour_range üret (00-01 format) — exposure eşleşmesi için
    if "event_hour" in df.columns and "hour_range" not in df.columns:
        hr = df["event_hour"].astype(int) % 24
        df["hour_range"] = hr.map(lambda h: f"{h:02d}-{(h+1)%24:02d}")

    return df

@st.cache_data(ttl=60*60, show_spinner=True)
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
            candidates = [n for n in memlist if n.endswith("/"+path_in_zip) or n.endswith(path_in_zip)]
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

# ====== Exposure fallback (sf_crime_09.csv) ======
@st.cache_data(ttl=30*60)
def load_exposure_fallback() -> pd.DataFrame:
    try:
        RAW = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/main/crime_prediction_data/sf_crime_09.csv"
        c9 = pd.read_csv(RAW)
        # GEOID normalize (11 hane)
        gcol = "GEOID" if "GEOID" in c9.columns else ("geoid" if "geoid" in c9.columns else None)
        if gcol is None:
            return pd.DataFrame(columns=["GEOID","hour_range","event_hour","exposure_guess"])
        c9["GEOID"] = c9[gcol].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)
        # hour_range varsa harika, yoksa hour/event_hour'tan üret
        if "hour_range" not in c9.columns:
            hsrc = None
            for k in ("event_hour","hour"):
                if k in c9.columns: hsrc = k; break
            if hsrc is not None:
                h = c9[hsrc].astype(int) % 24
                c9["hour_range"] = h.map(lambda x: f"{x:02d}-{(x+1)%24:02d}")
        # saatlik exposure tahmini
        c9["exposure_guess"] = (c9.get("crime_last_7d", 0) / 7.0).clip(lower=0.1)
        keep = [c for c in ["GEOID","hour_range","event_hour","exposure_guess"] if c in c9.columns]
        return c9[keep]
    except Exception:
        return pd.DataFrame(columns=["GEOID","hour_range","event_hour","exposure_guess"])

# ====== UI ======
st.title("🧭 Suç Tahmini (Forecast)")
colL, colR = st.columns([3,1], gap="large")

with colR:
    src = st.radio("risk_hourly kaynağı:", ["artifact_parquet"], horizontal=True)
    # (Gerektiğinde başka kaynaklar eklenebilir)

# Veriyi çek
try:
    df_risk = read_risk_from_artifact()
except Exception as e:
    st.error(f"Artifact okunamadı: {e}")
    st.stop()

# Kontroller
with colR:
    hours   = sorted(df_risk.get("event_hour", pd.Series(range(24))).dropna().unique().tolist())
    dows    = sorted(df_risk.get("day_of_week", pd.Series([0,1,2,3,4,5,6])).unique().tolist())
    seasons = sorted(df_risk.get("season", pd.Series(["Winter","Spring","Summer","Fall"])).unique().tolist())

    sel_hour   = st.select_slider("Saat", options=hours, value=hours[0])
    sel_dow    = st.selectbox("Haftanın Günü (0=Mon ... 6=Sun)", options=dows, index=0)
    sel_season = st.selectbox("Sezon", options=seasons, index=0)
    topn       = st.slider("Top-K (kritik liste)", 10, 200, 50)

# Filtre
mask = (df_risk["event_hour"]==sel_hour) if "event_hour" in df_risk.columns else pd.Series(True, index=df_risk.index)
if "day_of_week" in df_risk.columns: mask &= (df_risk["day_of_week"]==sel_dow)
if "season" in df_risk.columns:      mask &= (df_risk["season"]==sel_season)
layer_df = df_risk.loc[mask].copy()

# Risk kolon adı garantile
layer_df["risk_score"] = layer_df["risk_score"].astype(float)

# ---- E[olay] (pred_expected) ----
if "pred_expected" in layer_df.columns:
    layer_df["pred_expected"] = layer_df["pred_expected"].astype(float)
    exp_note = "artifact: pred_expected var"
else:
    exp = load_exposure_fallback()
    merged = layer_df.copy()
    merged["GEOID"] = merged["geoid"].astype(str)  # risk DF 11 hane
    # 1) GEOID + hour_range
    if not exp.empty and "hour_range" in exp.columns and "hour_range" in merged.columns:
        merged = merged.merge(exp[["GEOID","hour_range","exposure_guess"]],
                              on=["GEOID","hour_range"], how="left")
        reason = "GEOID+hour_range"
    # 2) GEOID + event_hour
    elif not exp.empty and "event_hour" in exp.columns and "event_hour" in merged.columns:
        merged = merged.merge(exp[["GEOID","event_hour","exposure_guess"]],
                              on=["GEOID","event_hour"], how="left")
        reason = "GEOID+event_hour"
    # 3) sadece GEOID
    else:
        exp_geo = exp.groupby("GEOID", as_index=False)["exposure_guess"].mean() if not exp.empty else pd.DataFrame(columns=["GEOID","exposure_guess"])
        merged = merged.merge(exp_geo, on="GEOID", how="left")
        reason = "GEOID only"

    merged["exposure_guess"] = merged["exposure_guess"].fillna(0.3)
    merged["pred_expected"]  = (merged["risk_score"] * merged["exposure_guess"]).round(3)
    layer_df = merged
    exp_note = f"exposure eşleşmesi: {reason}"

# ---- Harita ----
geojson = fetch_geojson_smart(
    path_local=GEOJSON_PATH_LOCAL_DEFAULT,
    path_in_zip=GEOJSON_IN_ZIP_PATH_DEFAULT,
    raw_owner=RAW_GEOJSON_OWNER,
    raw_repo=RAW_GEOJSON_REPO,
)
if not geojson:
    st.warning("GeoJSON bulunamadı (local/artifact/raw). Yolları kontrol edin.")

def make_map_forecast(geojson_dict: dict, df_layer: pd.DataFrame):
    dmap = df_layer.set_index("geoid")["risk_score"].to_dict()
    qs = df_layer["risk_score"].quantile([0,.25,.5,.75,1]).tolist()
    layer = pdk.Layer(
        "GeoJsonLayer", geojson_dict, stroked=False, opacity=.7, pickable=True,
        get_fill_color={
            "function": """
            const M=Object.fromEntries(py_dmap), Q=py_qs;
            return (f)=>{
              const raw=f.properties.GEOID ?? f.properties.geoid ?? f.properties.cell_id ?? f.properties.id ?? "";
              const g=String(raw).replace(/\\D/g,"").padStart(11,"0").slice(0,11);
              const p=(M[g]===undefined)?0:M[g];
              if (p<=Q[1]) return [178,223,138,220];
              if (p<=Q[2]) return [255,255,178,220];
              if (p<=Q[3]) return [254,204,92,230];
              return [227,26,28,235];
            }
            """
        },
        parameters={"py_dmap": list(dmap.items()), "py_qs": qs},
    )
    tooltip = {"html":"<b>GEOID:</b> {GEOID}<br/><b>Risk (p):</b> {risk_score}<br/><b>E[olay]:</b> {pred_expected}"}
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

with colL:
    st.subheader("Risk Haritası (seçili saat)")
    if geojson:
        make_map_forecast(geojson, layer_df)
    else:
        st.info("GeoJSON yüklenemediği için harita gösterilemiyor.")

with colR:
    st.caption(exp_note)
    st.markdown("**Kritik Top-K (E[olay] yüksek)**")
    top = layer_df[["geoid","risk_score","pred_expected"]].sort_values(
        ["pred_expected","risk_score"], ascending=False
    ).head(topn)
    st.dataframe(top.reset_index(drop=True), use_container_width=True)
    if not layer_df.empty:
        q25,q50,q75 = layer_df["risk_score"].quantile([.25,.5,.75]).round(4)
        st.caption(f"Q25={q25:.4f} • Q50={q50:.4f} • Q75={q75:.4f}")
