# pages/02_Forecast.py
import streamlit as st
import pandas as pd
import pydeck as pdk
import json
from urllib.request import urlopen

st.set_page_config(page_title="Suç Tahmini (Forecast)", layout="wide")

RAW = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/crime_prediction_data"
URL_RISK = f"{RAW}/risk_hourly.parquet"  # model çıktısı (ols. proba/pred_expected olabilir)
URL_C09  = f"{RAW}/sf_crime_09.csv"      # exposure için fallback kaynak
URL_GEO  = f"{RAW}/sf_census_blocks_with_population.geojson"

@st.cache_data(ttl=15*60)
def load_risk():
    df = pd.read_parquet(URL_RISK)
    # GEOID normalize
    df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)").fillna("").str[:10]
    # Bazı artifact sürümlerinde 'date' olmayabilir; sorun değil
    return df

@st.cache_data(ttl=24*60*60)
def load_geojson():
    with urlopen(URL_GEO) as f:
        gj = json.load(f)
    for ft in gj["features"]:
        ft["properties"]["GEOID"] = str(ft["properties"].get("GEOID", ""))[:10]
    return gj

@st.cache_data(ttl=30*60)
def load_exposure_fallback():
    """
    risk_hourly.parquet içinde pred_expected yoksa
    saat başına 'exposure' tahmini için sf_crime_09.csv'den türet.
    Mantık: crime_last_7d'i saatlik ölçeğe indir (≈ / 7) ve taban minimum ver.
    """
    try:
        c9 = pd.read_csv(URL_C09)
        # normalize
        if "GEOID" not in c9.columns:
            # bazı sürümlerde geoid int olabilir
            c9["GEOID"] = c9["GEOID"].astype(str).str.extract(r"(\d+)").fillna("").str[:10]
        c9["exposure_guess"] = (c9.get("crime_last_7d", 0) / 7.0).clip(lower=0.1)
        return c9[["GEOID", "hour_range", "exposure_guess"]]
    except Exception:
        # tamamen yoksa şehir geneli sabit min
        return pd.DataFrame(columns=["GEOID","hour_range","exposure_guess"])

def color_layer(geojson, df_layer):
    dmap = df_layer.set_index("GEOID")["proba"].to_dict()
    qs = df_layer["proba"].quantile([0,.25,.5,.75,1]).tolist()
    return pdk.Layer(
        "GeoJsonLayer",
        geojson,
        stroked=False,
        opacity=0.7,
        pickable=True,
        get_fill_color={
            "function": """
            const M=Object.fromEntries(py_dmap);
            const Q=py_qs;
            return (f)=>{
              const g=String(f.properties.GEOID||"").slice(0,10);
              const p=(M[g]===undefined)?0:M[g];
              if (p<=Q[1]) return [178,223,138,220];   // low
              if (p<=Q[2]) return [255,255,178,220];   // medium
              if (p<=Q[3]) return [254,204,92,230];    // high
              return [227,26,28,235];                  // critical
            }
            """
        },
        parameters={"py_dmap": list(dmap.items()), "py_qs": qs},
    )

st.title("🧭 Suç Tahmini (Forecast)")
colL, colR = st.columns([3,1], gap="large")

# --- Kontroller ---
with colR:
    df_risk = load_risk()
    hours = sorted(df_risk["event_hour"].dropna().unique().tolist())
    dows  = sorted(df_risk.get("day_of_week", pd.Series([0,1,2,3,4,5,6])).unique().tolist())
    seasons = sorted(df_risk.get("season", pd.Series(["Winter","Spring","Summer","Fall"])).unique().tolist())

    sel_hour   = st.select_slider("Saat", options=hours, value=hours[0])
    sel_dow    = st.selectbox("Haftanın Günü (0=Mon ... 6=Sun)", options=dows, index=0)
    sel_season = st.selectbox("Sezon", options=seasons, index=0)
    topn = st.slider("Top-K (kritik liste)", 10, 200, 50)

# --- Filtre & E[olay] hesap ---
mask = (df_risk["event_hour"]==sel_hour)
if "day_of_week" in df_risk.columns: mask &= (df_risk["day_of_week"]==sel_dow)
if "season" in df_risk.columns:      mask &= (df_risk["season"]==sel_season)
layer_df = df_risk.loc[mask].copy()

# 1) proba zaten modelden geliyor
layer_df["proba"] = layer_df["proba"].astype(float)

# 2) E[olay] (pred_expected)
if "pred_expected" in layer_df.columns:
    # artifact zaten hesaplamışsa direkt kullan
    layer_df["pred_expected"] = layer_df["pred_expected"].astype(float)
else:
    # Yoksa exposure tahmin et (fallback)
    exp = load_exposure_fallback()
    # hour_range eşleşmesi yoksa sadece GEOID ile eşle ve genel exposure kullan
    if not exp.empty and "hour_range" in layer_df.columns and "hour_range" in exp.columns:
        layer_df = layer_df.merge(exp[["GEOID","hour_range","exposure_guess"]],
                                  on=["GEOID","hour_range"], how="left")
    else:
        # sadece GEOID ile (saatten bağımsız) eşle
        exp_geo = exp.groupby("GEOID", as_index=False)["exposure_guess"].mean()
        layer_df = layer_df.merge(exp_geo, on="GEOID", how="left")
    layer_df["exposure_guess"] = layer_df["exposure_guess"].fillna(0.3)  # güvenli taban
    layer_df["pred_expected"] = (layer_df["proba"] * layer_df["exposure_guess"]).round(2)

# --- Harita ---
geojson = load_geojson()
layer = color_layer(geojson, layer_df)
view = pdk.ViewState(latitude=37.76, longitude=-122.44, zoom=11)

with colL:
    st.subheader("Risk Haritası (seçili saat)")
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={
            "html": (
                "<b>GEOID:</b> {GEOID}<br/>"
                "<b>Risk (p):</b> {proba}<br/>"
                "<b>E[olay] (beklenen):</b> {pred_expected}"
            )
        }
    ))

with colR:
    st.markdown("**Kritik Top-K (E[olay] yüksek)**")
    top = layer_df[["GEOID","proba","pred_expected"]].sort_values(
        ["pred_expected","proba"], ascending=False
    ).head(topn)
    st.dataframe(top.reset_index(drop=True), use_container_width=True)
    q25,q50,q75 = layer_df["proba"].quantile([.25,.5,.75]).round(4)
    st.caption(f"Q25={q25:.4f} • Q50={q50:.4f} • Q75={q75:.4f}")
