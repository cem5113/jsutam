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
    # varsa 'date' kullan; yoksa day_of_week/season ile ağırlıklı ortalama
    if "date" in df.columns:
        d = df[df["date"]==date_str].copy()
    else:
        d = df.copy()  # dışarıdan seçici uygularsan burada filtrele
    d["GEOID"] = d["GEOID"].astype(str).str.extract(r"(\d+)").fillna("").str[:10]
    daily = d.groupby("GEOID", as_index=False)["proba"].mean()
    return daily

@st.cache_data(ttl=24*60*60)
def _load_geo():
    with urlopen(URL_GEO) as f:
        gj = json.load(f)
    for ft in gj["features"]:
        ft["properties"]["GEOID"] = str(ft["properties"].get("GEOID",""))[:10]
    return gj

def _deck_layer(gj, daily_df):
    dmap = daily_df.set_index("GEOID")["proba"].to_dict()
    qs = daily_df["proba"].quantile([0,.25,.5,.75,1]).tolist()
    return pdk.Layer(
        "GeoJsonLayer",
        gj,
        stroked=False, opacity=.7, pickable=True,
        get_fill_color={
            "function": """
            const v=(d)=>d===undefined?0:d;
            const m=Object.fromEntries(py_dmap); // dict -> entries
            const q=py_qs;
            return (f)=>{
              const g=String(f.properties.GEOID||"").slice(0,10);
              const p=v(m[g]);
              if (p<=q[1]) return [178,223,138,220];   // low
              if (p<=q[2]) return [255,255,178,220];   // med
              if (p<=q[3]) return [254,204,92,230];    // high
              return [227,26,28,235];                 // critical
            }
            """
        },
        parameters={"py_dmap": list(dmap.items()), "py_qs": qs},
    )

st.divider(); st.subheader("Günün Risk Haritası (ortalama)")
date_str = st.session_state.get("selected_date", None) or pd.Timestamp.utcnow().date().isoformat()
daily = _load_risk_daily(date_str)
gj = _load_geo()
layer = _deck_layer(gj, daily)
view = pdk.ViewState(latitude=37.76, longitude=-122.44, zoom=11)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={
    "html": "<b>GEOID:</b> {GEOID}<br/><b>Günlük risk:</b> {{daily}}",
}))
# mini KPI
c1,c2,c3=st.columns(3)
q25,q50,q75 = daily.proba.quantile([.25,.5,.75]).round(4)
c1.metric("Q25", f"{q25:.4f}"); c2.metric("Q50", f"{q50:.4f}"); c3.metric("Q75", f"{q75:.4f}")
