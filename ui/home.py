# ui/home.py 
import pandas as pd, pydeck as pdk, json
from urllib.request import urlopen
import streamlit as st

RAW = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/crime_prediction_data"
URL_RISK = f"{RAW}/risk_hourly.parquet"
URL_GEO  = f"{RAW}/sf_census_blocks_with_population.geojson"

@st.cache_data(ttl=15*60)
def _load_risk_daily(date_str: str):
    df = pd.read_parquet(URL_RISK)

    # --- kolon alias ---
    cols = {c.lower(): c for c in df.columns}
    if "proba" not in cols:
        if "risk_score" in cols:
            df = df.rename(columns={cols["risk_score"]: "proba"})
        elif "risk" in cols:
            df = df.rename(columns={cols["risk"]: "proba"})
        else:
            raise ValueError("risk/proba kolonu bulunamadı.")

    # --- tarih filtrasyonu güvenli ---
    if "date" in cols:
        dcol = pd.to_datetime(df[cols["date"]]).dt.date
        target = pd.to_datetime(date_str).date()
        d = df[dcol == target].copy()
    else:
        d = df.copy()  # (gerekirse dışarıdan saat bazında filtreleyeceğiz)

    # --- GEOID normalize: 11 hane ---
    g = None
    for k in ("GEOID","geoid","geoid10","geoid11","cell_id","id"):
        if k in df.columns:
            g = k; break
    if g is None:
        raise ValueError("GEOID benzeri kolon yok.")
    d["GEOID"] = (d[g].astype(str).str.replace(r"\D","",regex=True).str.zfill(11).str[:11])

    daily = d.groupby("GEOID", as_index=False)["proba"].mean()

    # günlük çeyreklikler (legend/KPI için)
    q = daily["proba"].quantile([.25,.5,.75]).to_list() if not daily.empty else [0,0,0]
    daily["q25"], daily["q50"], daily["q75"] = q[0], q[1], q[2]
    return daily

@st.cache_data(ttl=24*60*60)
def _load_geo():
    with urlopen(URL_GEO) as f:
        gj = json.load(f)
    for ft in gj["features"]:
        raw = str(ft["properties"].get("GEOID",""))
        digits = "".join(ch for ch in raw if ch.isdigit())
        ft["properties"]["GEOID"] = digits.zfill(11)[:11]
    return gj

def _enrich_geo(gj: dict, daily_df: pd.DataFrame) -> dict:
    """GeoJSON feature.properties içine günlük ortalama risk ('daily') ve seviye ('level') enjekte eder."""
    if not gj or daily_df.empty:
        return gj
    dm = daily_df.set_index("GEOID")["proba"].to_dict()
    q25 = float(daily_df["q25"].iloc[0]); q50 = float(daily_df["q50"].iloc[0]); q75 = float(daily_df["q75"].iloc[0])
    out = {"type": gj["type"], "features": []}
    for ft in gj["features"]:
        props = dict(ft.get("properties") or {})
        g = props.get("GEOID", "")
        val = float(dm.get(g, 0.0))
        if val <= 1e-12:
            lvl = "zero"
        elif val <= q25:
            lvl = "low"
        elif val <= q50:
            lvl = "medium"
        elif val <= q75:
            lvl = "high"
        else:
            lvl = "critical"
        props["daily"] = round(val, 4)
        props["level"] = lvl
        props["fill_color"] = {
            "zero":[200,200,200],
            "low":[56,168,0],
            "medium":[255,221,0],
            "high":[255,140,0],
            "critical":[204,0,0],
        }[lvl]
        out["features"].append({**ft, "properties": props})
    return out

def _deck_layer(gj):
    return pdk.Layer(
        "GeoJsonLayer",
        gj,
        stroked=False, opacity=.7, pickable=True,
        get_fill_color="properties.fill_color",
    )

st.divider(); st.subheader("Günün Risk Haritası (ortalama)")
date_str = st.session_state.get("selected_date", None) or pd.Timestamp.utcnow().date().isoformat()
daily = _load_risk_daily(date_str)
gj = _load_geo()
gj_enriched = _enrich_geo(gj, daily)

layer = _deck_layer(gj_enriched)
view = pdk.ViewState(latitude=37.76, longitude=-122.44, zoom=11)
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view,
    tooltip={"html": "<b>GEOID:</b> {GEOID}<br/><b>Günlük risk:</b> {daily}<br/><b>Seviye:</b> {level}"}
))
# mini KPI
if not daily.empty:
    q25,q50,q75 = daily["proba"].quantile([.25,.5,.75]).round(4)
    c1,c2,c3=st.columns(3)
    c1.metric("Q25", f"{q25:.4f}"); c2.metric("Q50", f"{q50:.4f}"); c3.metric("Q75", f"{q75:.4f}")
else:
    st.info("Seçilen gün için veri bulunamadı.")
