# app.py — Giriş + Anlık (Nowcast) Risk Haritası
import io, os, json, zipfile
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

st.set_page_config(page_title="Suç Tahmini Uygulaması", layout="wide")
st.title("🗺️ Suç Tahmini Uygulaması")

# -----------------------------
# Ayarlar
# -----------------------------
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT = "sf-crime-parquet"
PARQUET = "risk_hourly.parquet"
# GeoJSON için basit RAW fallback (gerekirse değiştir)
RAW_GEO = "https://raw.githubusercontent.com/cem5113/crimepredict/main/data/sf_cells.geojson"

GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

def _hdr():
    h = {"Accept":"application/vnd.github+json"}
    if GITHUB_TOKEN: h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

@st.cache_data(ttl=15*60, show_spinner=True)
def fetch_artifact_zip():
    if not GITHUB_TOKEN:
        raise RuntimeError("GitHub token gerekli (st.secrets['github_token'] ya da env GITHUB_TOKEN).")
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/artifacts"
    r = requests.get(url, headers=_hdr(), timeout=30); r.raise_for_status()
    arts = [a for a in r.json().get("artifacts",[]) if a.get("name")==ARTIFACT and not a.get("expired",False)]
    if not arts: raise FileNotFoundError("Artifact bulunamadı.")
    arts.sort(key=lambda x:x.get("updated_at",""), reverse=True)
    dl = arts[0]["archive_download_url"]
    r2 = requests.get(dl, headers=_hdr(), timeout=60); r2.raise_for_status()
    return r2.content

@st.cache_data(ttl=15*60, show_spinner=True)
def load_latest_hour():
    """Artifact içinden risk_hourly.parquet → en güncel saatlik dilim."""
    z = fetch_artifact_zip()
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        names = zf.namelist()
        path = next((n for n in names if n.endswith("/"+PARQUET) or n.endswith(PARQUET)), None)
        if not path: raise FileNotFoundError(f"ZIP içinde {PARQUET} yok.")
        with zf.open(path) as f:
            df = pd.read_parquet(f)

    # alias & normalize
    cols = {c.lower(): c for c in df.columns}
    # risk/proba
    if "risk_score" in cols: df = df.rename(columns={cols["risk_score"]:"proba"})
    elif "risk" in cols:     df = df.rename(columns={cols["risk"]:"proba"})
    elif "proba" not in cols: raise ValueError("risk/proba kolonu yok.")
    # geoid
    gcol = None
    for k in ("geoid","geoid10","geoid11","cell_id","id"):
        if k in cols: gcol = cols[k]; break
    if gcol is None: raise ValueError("GEOID kolonu yok.")
    df["geoid"] = (df[gcol].astype(str).str.replace(r"\D","",regex=True).str.zfill(11).str[:11])
    # datetime
    if "datetime" in cols:
        df["datetime"] = pd.to_datetime(df[cols["datetime"]])
    else:
        if "date" not in cols: raise ValueError("date/datetime yok.")
        base = pd.to_datetime(df[cols["date"]])
        hr = (pd.to_numeric(df.get(cols.get("hour","hour"), 0), errors="coerce").fillna(0).astype(int)
              if "hour" in cols else 0)
        df["datetime"] = base.dt.floor("D") + pd.to_timedelta(hr, unit="h")

    # EN GÜNCEL SAAT → filtre
    last_dt = df["datetime"].max()
    now_df = df[df["datetime"] == last_dt].copy()

    # harita için dict ve çeyreklikler
    dmap = now_df.groupby("geoid", as_index=False)["proba"].mean()
    qs = dmap["proba"].quantile([0.25, 0.5, 0.75]).tolist() if not dmap.empty else [0,0,0]
    return dmap, last_dt, qs

@st.cache_data(ttl=60*60, show_spinner=False)
def load_geojson():
    try:
        r = requests.get(RAW_GEO, timeout=30); r.raise_for_status()
        gj = r.json()
        # GEOID normalize
        for ft in gj.get("features", []):
            props = ft.get("properties") or {}
            raw = str(props.get("geoid") or props.get("GEOID") or props.get("cell_id") or props.get("id") or "")
            digits = "".join(ch for ch in raw if ch.isdigit())
            props["GEOID"] = digits.zfill(11)[:11]
            ft["properties"] = props
        return gj
    except Exception as e:
        st.warning(f"GeoJSON yüklenemedi (RAW): {e}")
        return {}

# -----------------------------
# Menüler (Pages linkleri)
# -----------------------------
st.write("Soldaki **Pages** menüsünden detay sayfalarına gidebilirsiniz.")
try:
    st.page_link("pages/Home.py", label="→ Günlük Risk Haritası", icon="🗺️")
    st.page_link("pages/Forecast.py", label="→ Forecast (Model+)", icon="🔮")
except Exception:
    pass

# -----------------------------
# ANLIK (NOWCAST) RİSK HARİTASI
# -----------------------------
st.divider()
st.subheader("⚡ Anlık Risk Haritası (son saat)")

refresh = st.button("Veriyi Yenile (artifact)")
if refresh:
    fetch_artifact_zip.clear(); load_latest_hour.clear()

try:
    daily_now, last_dt, qs = load_latest_hour()
except Exception as e:
    st.error(f"Anlık veriyi okuyamadım: {e}")
else:
    st.caption(f"Gösterilen saat: **{pd.to_datetime(last_dt)}**")
    gj = load_geojson()

    # GeoJSON'a renk enjekte
    dmap = daily_now.set_index("geoid")["proba"].to_dict()
    q25,q50,q75 = (qs + [0,0,0])[:3]
    for ft in gj.get("features", []):
        props = ft.get("properties") or {}
        g = props.get("GEOID","")
        p = float(dmap.get(g, 0.0))
        if p <= 1e-12: lvl="zero"
        elif p <= q25: lvl="low"
        elif p <= q50: lvl="medium"
        elif p <= q75: lvl="high"
        else: lvl="critical"
        props["nowcast"] = round(p,4)
        props["level"] = lvl
        props["fill_color"] = {
            "zero":[200,200,200],
            "low":[56,168,0],
            "medium":[255,221,0],
            "high":[255,140,0],
            "critical":[204,0,0],
        }[lvl]
        ft["properties"] = props

    layer = pdk.Layer(
        "GeoJsonLayer",
        gj,
        stroked=True,
        get_line_color=[80,80,80],
        line_width_min_pixels=0.5,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=0.65,
    )
    tooltip = {
        "html": "<b>GEOID:</b> {GEOID}<br/><b>Şu an risk:</b> {nowcast}<br/><b>Seviye:</b> {level}",
        "style": {"backgroundColor":"#262730","color":"white"},
    }
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

    # mini KPI
    if not daily_now.empty:
        c1,c2,c3 = st.columns(3)
        c1.metric("Q25", f"{daily_now['proba'].quantile(.25):.4f}")
        c2.metric("Q50", f"{daily_now['proba'].quantile(.50):.4f}")
        c3.metric("Q75", f"{daily_now['proba'].quantile(.75):.4f}")
    else:
        st.info("Bu saat için kayıt bulunamadı.")
