# pages/Forecast.py
import io, os, json, zipfile
from datetime import datetime, date
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
EXPECTED_C09 = "sf_crime_09.parquet"
EXPECTED_METRICS = "metrics_all.parquet"

GEOJSON_PATH_LOCAL_DEFAULT = "data/sf_cells.geojson"
GEOJSON_IN_ZIP_PATH_DEFAULT = "data/sf_cells.geojson"

RAW_GEOJSON_OWNER = "cem5113"
RAW_GEOJSON_REPO  = "crimepredict"

GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

st.set_page_config(page_title="🧭 Forecast — Saatlik Suç Riski", layout="wide")

# =========================
# Yardımcılar
# =========================
def _gh_headers():
    hdrs = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        hdrs["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return hdrs

def _digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())

def _season_from_month(m: int) -> str:
    return {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
            6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}.get(int(m), "All")

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

def _ensure_temporal_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hour" not in df.columns and "hour_range" in df.columns:
        df["hour"] = pd.to_numeric(df["hour_range"], errors="coerce").fillna(0).astype(int).clip(0, 23)
    elif "hour" not in df.columns:
        df["hour"] = 0
    if "dow" not in df.columns:
        if "date" in df.columns:
            t = pd.to_datetime(df["date"], errors="coerce")
            df["dow"] = t.dt.dayofweek.fillna(0).astype(int)
        else:
            df["dow"] = 0
    if "season" not in df.columns:
        if "date" in df.columns:
            month = pd.to_datetime(df["date"], errors="coerce").dt.month
            df["season"] = month.map(_season_from_month).fillna("All").astype(str)
        else:
            df["season"] = "All"
    df["hour"] = df["hour"].astype(int).clip(0, 23)
    df["dow"]  = df["dow"].astype(int).clip(0, 6)
    df["season"] = df["season"].astype(str)
    return df

# =========================
# Artifact okumalar
# =========================
@st.cache_data(show_spinner=True, ttl=15*60)
def read_risk_from_artifact() -> pd.DataFrame:
    zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        matches = [n for n in memlist if n.endswith("/" + EXPECTED_PARQUET) or n.endswith(EXPECTED_PARQUET)]
        if not matches:
            raise FileNotFoundError(f"Zip içinde {EXPECTED_PARQUET} yok. Örnek içerik: {memlist[:15]}")
        with zf.open(matches[0]) as f:
            df = pd.read_parquet(f)

    df.columns = [c.strip().lower() for c in df.columns]
    if "geoid" not in df.columns:
        for alt in ["cell_id", "geoid10", "geoid11", "geoid_10", "geoid_11", "id"]:
            if alt in df.columns:
                df["geoid"] = df[alt]
                break
    df["geoid"] = df["geoid"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "risk_score" not in df.columns:
        if "proba" in df.columns: df = df.rename(columns={"proba": "risk_score"})
        elif "risk" in df.columns: df = df.rename(columns={"risk": "risk_score"})
    df = _ensure_temporal_cols(df)
    need = {"geoid", "hour", "dow", "season", "risk_score"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Eksik kolon(lar): {', '.join(sorted(miss))}")
    return df

@st.cache_data(show_spinner=True, ttl=15*60)
def read_c09_from_artifact() -> pd.DataFrame:
    zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        matches = [n for n in memlist if n.endswith("/" + EXPECTED_C09) or n.endswith(EXPECTED_C09)]
        if not matches:
            raise FileNotFoundError(f"Zip içinde {EXPECTED_C09} yok. Örnek içerik: {memlist[:15]}")
        with zf.open(matches[0]) as f:
            c9 = pd.read_parquet(f)

    c9.columns = [c.strip() for c in c9.columns]
    if "geoid" not in c9.columns and "GEOID" in c9.columns:
        c9["geoid"] = c9["GEOID"]
    if "geoid" in c9.columns:
        c9["geoid"] = c9["geoid"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)
    if "event_hour" not in c9.columns and "hour_range" in c9.columns:
        c9["event_hour"] = pd.to_numeric(c9["hour_range"], errors="coerce").fillna(0).astype(int)
    if "day_of_week" not in c9.columns and "dow" in c9.columns:
        c9 = c9.rename(columns={"dow": "day_of_week"})
    if "season" not in c9.columns:
        c9["season"] = "All"
    if "exposure_guess" not in c9.columns:
        base = pd.to_numeric(c9.get("crime_last_7d", 0), errors="coerce")/7.0
        c9["exposure_guess"] = base.clip(lower=0.1).fillna(0.1)

    keep = ["geoid", "season", "day_of_week", "event_hour", "exposure_guess"]
    return c9[[k for k in keep if k in c9.columns]].copy()

@st.cache_data(show_spinner=True, ttl=15*60)
def read_metrics_best() -> pd.DataFrame:
    try:
        zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            memlist = zf.namelist()
            matches = [n for n in memlist if n.endswith("/" + EXPECTED_METRICS) or n.endswith(EXPECTED_METRICS)]
            if not matches:
                return pd.DataFrame()
            with zf.open(matches[0]) as f:
                m = pd.read_parquet(f)
        return m
    except Exception:
        return pd.DataFrame()

# =========================
# GeoJSON
# =========================
@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_geojson_smart(path_local: str, path_in_zip: str, raw_owner: str, raw_repo: str) -> dict:
    try:
        if os.path.exists(path_local):
            with open(path_local, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
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
    try:
        raw = f"https://raw.githubusercontent.com/{raw_owner}/{raw_repo}/main/{path_local}"
        r = requests.get(raw, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

# =========================
# E[olay] ekleme
# =========================
def attach_pred_expected(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy(); out["pred_expected"] = []; return out
    expo = read_c09_from_artifact()
    out = df.copy()
    if expo.empty:
        out["pred_expected"] = (out["risk_score"] * 0.3).round(3)
        return out

    out["geoid"] = out["geoid"].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)
    out["hour"]  = pd.to_numeric(out["hour"], errors="coerce").fillna(0).astype(int)
    out["dow"]   = pd.to_numeric(out["dow"],  errors="coerce").fillna(0).astype(int)
    out["season"]= out.get("season","All").astype(str)

    expo = expo.copy()
    expo["geoid"] = expo["geoid"].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)
    expo["event_hour"]  = pd.to_numeric(expo["event_hour"],  errors="coerce").fillna(0).astype(int)
    expo["day_of_week"] = pd.to_numeric(expo["day_of_week"], errors="coerce").fillna(0).astype(int)
    expo["season"] = expo.get("season","All").astype(str)

    m = out.merge(
        expo[["geoid","season","day_of_week","event_hour","exposure_guess"]],
        left_on=["geoid","season","dow","hour"],
        right_on=["geoid","season","day_of_week","event_hour"],
        how="left"
    )
    miss = m["exposure_guess"].isna()
    if miss.any():
        j = out.merge(
            expo[["geoid","event_hour","exposure_guess"]],
            left_on=["geoid","hour"],
            right_on=["geoid","event_hour"],
            how="left"
        )["exposure_guess"]
        m.loc[miss, "exposure_guess"] = j[miss].values
    miss = m["exposure_guess"].isna()
    if miss.any():
        ge_mean = expo.groupby("geoid", as_index=False)["exposure_guess"].mean()
        j = out.merge(ge_mean, on="geoid", how="left")["exposure_guess"]
        m.loc[miss, "exposure_guess"] = j[miss].values

    m["exposure_guess"] = m["exposure_guess"].fillna(0.3)
    m["pred_expected"]  = (pd.to_numeric(m["risk_score"], errors="coerce").fillna(0.0) *
                           pd.to_numeric(m["exposure_guess"], errors="coerce").fillna(0.3)).round(3)
    return m

def enrich_geojson(geojson: dict, df_layer: pd.DataFrame) -> dict:
    if not geojson or df_layer.empty: return geojson
    g = df_layer.copy()
    g["key"] = g["geoid"].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)
    q25 = float(g["risk_score"].quantile(0.25))
    q50 = float(g["risk_score"].quantile(0.50))
    q75 = float(g["risk_score"].quantile(0.75))
    COLOR = {
        "low":[178,223,138,220],
        "med":[255,255,178,220],
        "high":[254,204,92,230],
        "crit":[227,26,28,235]
    }
    kmap = g.set_index("key")[["risk_score","pred_expected"]].to_dict(orient="index")
    out_feats=[]
    for feat in geojson.get("features", []):
        props = (feat.get("properties") or {}).copy()
        raw = props.get("GEOID") or props.get("geoid") or props.get("cell_id") or props.get("id") or ""
        key = _digits(raw).zfill(11)[:11]
        rs = pe = 0.0
        if key in kmap:
            rs = float(kmap[key]["risk_score"])
            pe = float(kmap[key]["pred_expected"])
        if rs <= q25: col = COLOR["low"]
        elif rs <= q50: col = COLOR["med"]
        elif rs <= q75: col = COLOR["high"]
        else: col = COLOR["crit"]
        props.update({
            "GEOID": key,
            "risk_score": rs,
            "pred_expected": pe,
            "fill_color": col,
        })
        out_feats.append({**feat, "properties": props})
    return {"type":"FeatureCollection","features":out_feats}

# =========================
# Harita
# =========================
def make_map_forecast(geojson_enriched: dict):
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_enriched,
        stroked=False,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=.75,
    )
    tooltip = {
        "html": "<b>GEOID:</b> {GEOID}<br/><b>Risk (p):</b> {risk_score}<br/><b>E[olay]:</b> {pred_expected}",
        "style": {"backgroundColor": "#262730", "color": "white"}
    }
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=11),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

# =========================
# UI
# =========================
st.title("🧭 Forecast — Saatlik Suç Riski ve Beklenen Olay")

with st.sidebar:
    refresh = st.button("Veriyi Yenile (artifact'i tazele)")
    target_date = st.date_input("Tarih", value=date.today())
    target_hour = st.slider("Saat", 0, 23, datetime.now().hour)
    dow_label = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    # sadece gösterim için
    st.caption(f"Haftanın günü: {dow_label[pd.Timestamp(target_date).dayofweek]}")

try:
    if refresh:
        fetch_latest_artifact_zip.clear()
        read_risk_from_artifact.clear()
        read_c09_from_artifact.clear()
        fetch_geojson_smart.clear()
        read_metrics_best.clear()
    base_df = read_risk_from_artifact()
except Exception as e:
    st.error(f"Artifact indirilemedi/okunamadı: {e}")
    st.stop()

df = attach_pred_expected(base_df)

sel_dow = pd.Timestamp(target_date).dayofweek
sel_season = _season_from_month(pd.Timestamp(target_date).month)

hours_avail   = sorted(pd.to_numeric(df["hour"], errors="coerce").dropna().astype(int).unique().tolist())
dows_avail    = sorted(pd.to_numeric(df["dow"],  errors="coerce").dropna().astype(int).unique().tolist())
seasons_avail = sorted(df["season"].astype(str).dropna().unique().tolist())
if sel_dow not in dows_avail: sel_dow = dows_avail[0] if dows_avail else 0
if sel_season not in seasons_avail: sel_season = seasons_avail[0] if seasons_avail else "All"

sel = df[(df["hour"]==int(target_hour)) & (df["dow"]==int(sel_dow)) & (df["season"].astype(str)==str(sel_season))].copy()
sel = sel.sort_values("risk_score", ascending=False)

c1, c2, c3 = st.columns(3)
if not sel.empty:
    q = sel["risk_score"].quantile([0.25,0.5,0.75]).tolist()
    c1.metric("Q25", f"{q[0]:.3f}")
    c2.metric("Q50", f"{q[1]:.3f}")
    c3.metric("Q75", f"{q[2]:.3f}")
else:
    c1.metric("Q25", "-"); c2.metric("Q50", "-"); c3.metric("Q75", "-")

st.sidebar.header("Harita Sınırları (GeoJSON)")
geojson_local = st.sidebar.text_input("Local yol", value=GEOJSON_PATH_LOCAL_DEFAULT)
geojson_zip   = st.sidebar.text_input("Artifact ZIP içi yol", value=GEOJSON_IN_ZIP_PATH_DEFAULT)
geojson = fetch_geojson_smart(
    path_local=geojson_local,
    path_in_zip=geojson_zip,
    raw_owner=RAW_GEOJSON_OWNER,
    raw_repo=RAW_GEOJSON_REPO,
)

st.subheader(f"Harita — Saatlik Risk (Tarih: {target_date}, Saat: {target_hour}, Sezon: {sel_season})")
if not geojson:
    st.warning("GeoJSON bulunamadı (local/artifact/raw).")
else:
    if sel.empty:
        st.info("Seçime göre satır bulunamadı.")
    else:
        enriched = enrich_geojson(geojson, sel[["geoid","risk_score","pred_expected"]])
        make_map_forecast(enriched)

st.subheader("Kritik Top-K (E[olay] yüksek)")
if sel.empty:
    st.info("Seçilen filtreler için veri bulunamadı.")
else:
    topk = st.slider("Top-K", 10, 200, 50)
    top_df = sel.nlargest(topk, columns=["pred_expected"])[["geoid","risk_score","pred_expected"]]
    st.dataframe(top_df.reset_index(drop=True), use_container_width=True)
    parquet_bytes = top_df.to_parquet(index=False)
    st.download_button(
        "Top-K PARQUET indir",
        parquet_bytes,
        file_name=f"forecast_topk_{target_date}_h{int(target_hour)}_d{int(sel_dow)}_{sel_season}.parquet",
        mime="application/octet-stream"
    )
