# -*- coding: utf-8 -*-
# pages/Forecast.py
# Gereksinimler: streamlit, pandas, requests, folium, streamlit-folium, scikit-learn, pyarrow (Parquet için)
#
# Bu sayfa, dorecast.py ile uyumludur:
# - Kullanıcı "uygulamayı açtığı anı" t0 olarak seçer.
# - 24 saat (1s), 72 saat (3s) ve 7 gün (24s) pencereleri için tahmin üretir.
# - dorecast.run_forecasts / dorecast.predict_at_datetime fonksiyonlarını çağırır.
# - Harita üzerinde beklenen olay sayısını (E[olay]) göstermek için basit bir exposure proxy uygular:
#     E[olay] ≈ prob_decayed × base_rate × (window_hours/24)
#   Burada base_rate, t0'dan önceki 30 günün günlük ortalama olay sayısıdır (GEOID bazında).
#
# Not: Eğitim verisi (history) CSV/Parquet olarak "datetime, GEOID, Y_label" sütunlarını içermelidir.

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

# dorecast fonksiyonları
try:
    from dorecast import run_forecasts, predict_at_datetime
except Exception as e:
    run_forecasts = None
    predict_at_datetime = None

# -------------------------
# Yardımcılar
# -------------------------

def _digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())

def sanitize_props(geojson_dict: dict) -> dict:
    """GeoJSON properties içindeki NaN / numpy tipleri / Timestamp'leri JSON uyumlu hale getirir."""
    def _fix(v):
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
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

def geojson_centroids(geojson: dict) -> Dict[str, Tuple[float,float]]:
    """Basit centroid (koordinat ortalaması). (lat, lon) döner."""
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
            out[key] = (cy, cx)  # (lat, lon)
    return out

def make_priority_color(priority: str) -> str:
    cmap = {
        "zero":     "#C8C8C8",
        "low":      "#38A800",
        "medium":   "#FFDD00",
        "high":     "#FF8C00",
        "critical": "#CC0000",
    }
    return cmap.get(priority, "#CCCCCC")

def add_priority(df: pd.DataFrame, value_col: str = "pred_expected") -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["priority"] = []
        df["q25"] = df["q50"] = df["q75"] = 0.0
        return df
    qs = df[value_col].quantile([0, .25, .5, .75]).tolist()
    q0, q25, q50, q75 = qs
    def lab(x: float) -> str:
        if x <= 0: return "zero"
        if x <= q25: return "low"
        if x <= q50: return "medium"
        if x <= q75: return "high"
        return "critical"
    df["priority"] = df[value_col].apply(lab)
    df["q25"], df["q50"], df["q75"] = float(q25), float(q50), float(q75)
    return df

def compute_top3_offense(slice_df: pd.DataFrame) -> pd.DataFrame:
    """slice_df içinde offense/crime_type varsa GEOID bazında ilk 3'ü döndürür."""
    df = slice_df.copy()
    cand_cols = [c for c in ["offense","offense_category","crime_type","primary_type"] if c in df.columns]
    if not cand_cols:
        return pd.DataFrame(columns=["geoid","top3"])
    col = cand_cols[0]
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
            t3 = tmap.get(key, [])
            if isinstance(t3, (list, tuple)):
                props["top3"] = ", ".join(map(str, t3))
            else:
                props["top3"] = str(t3)
        else:
            props.setdefault("geoid", key)
            props.setdefault("pred_expected", 0.0)
            props.setdefault("risk_score", 0.0)
            props.setdefault("priority", "zero")
            props.setdefault("top3", "")
        out_feats.append({**feat, "properties": props})
    return {**geojson, "features": out_feats}

def render_map(geojson: dict, slice_df: pd.DataFrame, value_col: str = "pred_expected") -> Tuple[folium.Map, Dict[str, Tuple[float,float]]]:
    # JSON serileştirme için temizle
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

    tooltip = folium.features.GeoJsonTooltip(
        fields=["geoid", "pred_expected", "priority"],
        aliases=["GEOID", "E[olay] (toplam)", "Öncelik"],
        localize=True,
        sticky=False,
        labels=True,
    )
    popup = folium.features.GeoJsonPopup(
        fields=["geoid", "pred_expected", "priority"],
        aliases=["GEOID", "E[olay] (toplam)", "Öncelik"],
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

# -------------------------
# Veri Girişi & Forecast
# -------------------------

st.set_page_config(page_title="🧭 Suç Tahmini (Forecast)", layout="wide")
st.title("🧭 Suç Tahmini (Forecast) — dorecast uyumlu")
st.caption("t0 seç, 24h/72h/7g tahmin üret; haritada E[olay] ve öncelik gör.")

with st.sidebar:
    st.header("Veri Kaynağı (History)")
    data_src = st.radio("Kaynak", ["Dosya yükle", "Yerel yol/URL"], index=0)
    uploaded = st.file_uploader("CSV veya Parquet yükle", type=["csv","parquet","pq","parq"]) if data_src=="Dosya yükle" else None
    data_path = st.text_input("Yerel yol veya URL", value="", help="History verisi en az 'datetime, GEOID, Y_label' içermeli.")

    st.divider()
    st.header("Zaman & Ufuk")
    t0_date = st.date_input("Uygulama açılış tarihi (t0)", value=date.today())
    t0_time = st.time_input("Saat", value=time(9,0))
    tau = st.slider("Temporal decay (τ, saat)", min_value=12, max_value=168, value=72, step=6, help="Olasılık uzak gelecekte üssel olarak düşer.")

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
    st.header("GEOJSON (Sınırlar)")
    geojson_path = st.text_input("GeoJSON yolu", value="data/sf_cells.geojson")

    st.divider()
    st.header("GEOID Seçimi")
    geoid_text = st.text_area("GEOID listesi (virgüllü)", value="", help="Boş bırakılırsa veri içinden en aktif 200 GEOID otomatik seçilir.")
    run_btn = st.button("🚀 Tahmin Üret")

# History verisini oku
def read_history_from_any(uploaded, data_path: str) -> pd.DataFrame:
    if uploaded is not None:
        ext = os.path.splitext(uploaded.name)[1].lower()
        if ext in [".parquet",".pq",".parq"]:
            return pd.read_parquet(uploaded)
        else:
            return pd.read_csv(uploaded)
    if data_path.strip():
        if data_path.startswith("http"):
            if data_path.endswith((".parquet",".pq",".parq")):
                import requests
                r = requests.get(data_path, timeout=60)
                r.raise_for_status()
                return pd.read_parquet(io.BytesIO(r.content))
            else:
                return pd.read_csv(data_path)
        else:
            ext = os.path.splitext(data_path)[1].lower()
            if ext in [".parquet",".pq",".parq"]:
                return pd.read_parquet(data_path)
            else:
                return pd.read_csv(data_path)
    raise RuntimeError("History verisi sağlanmadı.")

def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # sütun adları
    cols = {c.lower(): c for c in df.columns}
    # datetime
    dcol = "datetime" if "datetime" in cols else None
    for alt in ["timestamp","time","date_time","dt"]:
        if dcol is None and alt in cols:
            dcol = cols[alt]
    if dcol is None:
        raise ValueError("datetime sütunu bulunamadı.")
    df.rename(columns={dcol: "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # GEOID
    gcol = "GEOID" if "geoid" in [c.lower() for c in df.columns] else None
    if gcol is None:
        for alt in ["cell_id","geoid10","geoid11","geoid_10","geoid_11","id"]:
            if alt in df.columns:
                gcol = alt
                break
    if gcol is None:
        raise ValueError("GEOID sütunu bulunamadı.")
    df.rename(columns={gcol: "GEOID"}, inplace=True)
    df["GEOID"] = df["GEOID"].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)

    # Y_label
    ycol = None
    for cand in ["Y_label","y_label","label","y","target","crime"]:
        if cand in df.columns:
            ycol = cand
            break
    if ycol is None:
        raise ValueError("Y_label (hedef) sütunu bulunamadı.")
    if ycol != "Y_label":
        df.rename(columns={ycol: "Y_label"}, inplace=True)
    df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype(int).clip(0,1)
    return df[["datetime","GEOID","Y_label"] + [c for c in df.columns if c not in ["datetime","GEOID","Y_label"]]]

def suggest_top_geoids(hist: pd.DataFrame, t0: pd.Timestamp, k: int = 200) -> List[str]:
    past = hist[hist["datetime"] < t0]
    last30 = past[past["datetime"] >= (t0 - pd.Timedelta(days=30))]
    if last30.empty:
        last30 = past.tail(24*30)
    agg = (last30.groupby("GEOID")["Y_label"].sum()
                  .sort_values(ascending=False).head(k))
    return agg.index.astype(str).tolist()

def compute_base_rate(hist: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """GEOID bazında günlük olay ortalaması (t0 öncesi 30 günden)."""
    past = hist[(hist["datetime"] < t0) & (hist["datetime"] >= (t0 - pd.Timedelta(days=30)))]
    if past.empty:
        past = hist[hist["datetime"] < t0].tail(24*30)
    if past.empty:
        return pd.DataFrame(columns=["GEOID","base_rate_daily"])
    daily = (past.assign(date=past["datetime"].dt.date)
                  .groupby(["GEOID","date"], as_index=False)["Y_label"].sum())
    base = daily.groupby("GEOID", as_index=False)["Y_label"].mean().rename(columns={"Y_label":"base_rate_daily"})
    return base

def forecast_with_dorecast(hist: pd.DataFrame,
                           t0: pd.Timestamp,
                           geoids: List[str],
                           modes: List[str],
                           tau_hours: float) -> pd.DataFrame:
    """dorecast.run_forecasts'i çağırıp çıktı DataFrame döndürür."""
    # Geçici dosyaya kaydedip CLI fonksiyonunu doğrudan import edilen API üzerinden çağıracağız
    tmp_path = "history_tmp.parquet"
    hist.to_parquet(tmp_path, index=False)
    out = run_forecasts(tmp_path, t0.isoformat(sep=" "), geoids, modes, out_path=None, tau_hours=tau_hours)
    os.remove(tmp_path)
    return out

def attach_expected(out_df: pd.DataFrame, base_rate_df: pd.DataFrame) -> pd.DataFrame:
    """E[olay] ≈ prob_decayed × base_rate_daily × (window_hours/24)"""
    df = out_df.copy()
    base = base_rate_df.copy()
    base["GEOID"] = base["GEOID"].astype(str)
    df["GEOID"] = df["GEOID"].astype(str)
    df = df.merge(base, on="GEOID", how="left")
    df["base_rate_daily"] = df["base_rate_daily"].fillna(df["base_rate_daily"].median() if not base.empty else 0.1)
    df["risk_score"] = df["prob_decayed"].astype(float)
    df["pred_expected"] = df["risk_score"] * df["base_rate_daily"] * (df["window_hours"].astype(float)/24.0)
    # Görsel uyum için "geoid" alt-sütunu
    df["geoid"] = df["GEOID"].astype(str)
    # Gün/saat alanları
    df["date"] = pd.to_datetime(df["t_start"]).dt.date
    df["hour"] = pd.to_datetime(df["t_start"]).dt.hour
    return df

def slice_by_ui(df: pd.DataFrame, t0_date: date, horizon: str, hour_sel: int = 0, start_hour: int = 0, bin_index: int = 0, day_index: int = 0):
    if "24 saat" in horizon:
        d = t0_date
        m = (df["date"] == d) & (df["hour"] == int(hour_sel)) & (df["window_hours"] == 1)
        sl = df.loc[m].copy()
        label = f"{d} — {hour_sel:02d}:00"
    elif "72 saat" in horizon:
        # 3 saatlik blok seçimi: t0_date + start_hour başlangıcından itibaren bin_index
        # window_hours = 3
        base_dt = datetime.combine(t0_date, time(hour=start_hour))
        # seçilen blok başlangıcı
        target_start = base_dt + timedelta(hours=bin_index*3)
        m = (df["t_start"] == pd.Timestamp(target_start)) & (df["window_hours"] == 3)
        sl = df.loc[m].copy()
        label = f"{t0_date} +72h — blok {bin_index} ({bin_index*3:02d}-{bin_index*3+2:02d})"
    else:
        # 7 gün: window_hours = 24 ve day_index'e göre gün
        start_day = t0_date + timedelta(days=day_index)
        m = (df["date"] == start_day) & (df["window_hours"] == 24)
        sl = df.loc[m].copy()
        label = f"{t0_date} +7g — gün {day_index}"
    return sl, label

# -------------------------
# Çalıştır
# -------------------------

if run_btn:
    # 1) history oku
    try:
        hist = read_history_from_any(uploaded, data_path)
        hist = normalize_history(hist)
    except Exception as e:
        st.error(f"History okunamadı: {e}")
        st.stop()

    # 2) t0 ve GEOID listesi
    t0 = datetime.combine(t0_date, t0_time)
    if geoid_text.strip():
        geoids = [g.strip() for g in geoid_text.split(",") if g.strip()]
    else:
        geoids = suggest_top_geoids(hist, t0, k=200)
        if not geoids:
            st.error("GEOID listesi bulunamadı. Lütfen manuel girin.")
            st.stop()

    # 3) base rate
    base_rate = compute_base_rate(hist, t0)

    # 4) dorecast tahminleri
    if run_forecasts is None:
        st.error("dorecast.py bulunamadı veya import edilemedi. Proje köküne 'dorecast.py' ekleyin.")
        st.stop()

    try:
        out = forecast_with_dorecast(hist, t0, geoids, modes, tau_hours=float(tau))
    except Exception as e:
        st.error(f"Tahmin üretilemedi: {e}")
        st.stop()

    if out is None or out.empty:
        st.warning("Tahmin üretilemedi veya boş çıktı.")
        st.stop()

    # 5) E[olay] bağlama
    out = attach_expected(out, base_rate)

    # 6) GeoJSON oku
    try:
        if not os.path.exists(geojson_path):
            st.warning("GeoJSON yolu yerel dosyada bulunamadı. Lütfen doğru yolu girin.")
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    except Exception as e:
        st.error(f"GeoJSON okunamadı: {e}")
        st.stop()

    # 7) UI dilimine göre slice
    sl, time_label = slice_by_ui(out, t0_date, horizon,
                                 hour_sel=locals().get("hour_sel", 0),
                                 start_hour=locals().get("start_hour", 0),
                                 bin_index=locals().get("bin_index", 0),
                                 day_index=locals().get("day_index", 0))
    if sl.empty:
        st.warning("Seçili dilimde veri bulunamadı.")
        st.stop()

    # 8) Öncelik & top-3
    sl = add_priority(sl, value_col="pred_expected")
    top3 = compute_top3_offense(sl)

    # 9) Harita
    geojson_enriched = enrich_geojson_with_slice(geojson, sl, top3)
    folium_map, centroids = render_map(geojson_enriched, sl, value_col="pred_expected")
    mres = st_folium(folium_map, width=None, height=600, use_container_width=True)

    # 10) KPI ve tablo
    st.subheader(f"Harita — {time_label}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kapsam (GEOID)", f"{sl['geoid'].nunique()}")
    c2.metric("Medyan E[olay]", f"{sl['pred_expected'].median():.2f}")
    c3.metric("Q75", f"{sl['pred_expected'].quantile(0.75):.2f}")
    c4.metric("Toplam E[olay]", f"{sl['pred_expected'].sum():.2f}")

    st.subheader("Top Hotspots")
    topk = sl.sort_values("pred_expected", ascending=False).head(50)
    st.dataframe(topk.reset_index(drop=True), use_container_width=True)
    st.download_button("Top-50 Hotspots CSV",
                       data=topk.to_csv(index=False).encode("utf-8"),
                       file_name="hotspots.csv", mime="text/csv")

    # 11) Tıklanan GEOID kartı
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

            st.markdown("### 🟠 Sonuç Kartı")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Bölge (GEOID)**")
                st.code(clicked_geoid, language="text")
                st.markdown("**Öncelik**")
                st.write(str(pr).title())
                st.markdown("**Ufuk**")
                st.write(horizon)
            with c2:
                st.markdown("**Beklenen olay (ΣE[olay])**")
                st.metric(label="Toplam", value=f"{total_e:.2f}")
                st.markdown("**Olasılık ort. (p)**")
                st.metric(label="Mean p", value=f"{mean_p:.3f}")

    # 12) Çıktıyı indir
    st.download_button("Tüm Tahminler (CSV)",
                       data=out.to_csv(index=False).encode("utf-8"),
                       file_name="forecasts_all.csv", mime="text/csv")
