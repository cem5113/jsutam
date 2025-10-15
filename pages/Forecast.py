# pages/Forecast.py
# Çok-ufuklu tahmin + (Hepsi veya suç türü) + Top-3 suç özeti
try:
    import sklearn  # noqa
except Exception as e:
    import streamlit as st
    st.error("Gerekli paketler yüklü değil: scikit-learn (ve bağımlılıkları). "
             "Lütfen requirements.txt dosyasını repo köküne ekleyin/güncelleyin ve uygulamayı yeniden başlatın.")
    st.code("pip install -r requirements.txt", language="bash")
    st.stop()

import io, os, zipfile
from datetime import datetime
import pandas as pd
import streamlit as st
import requests

from models.forecaster import CrimeForecaster, ForecastConfig

# -------- Artifact ayarları --------
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT = "sf-crime-parquet"
LABEL_PARQUETS = [
    "sf_crime_grid_full_labeled.parquet",  # tercih edilen
    "sf_crime_52.parquet",
    "sf_crime_52.csv",
    "sf_crime_y.parquet",
]
GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

st.set_page_config(page_title="Forecast (Model+)", layout="wide")
st.title("🔮 Suç Tahmini (Model+)")

# -----------------------------
# Artifact yardımcıları
# -----------------------------
def _hdr():
    h={"Accept":"application/vnd.github+json"}
    if GITHUB_TOKEN: h["Authorization"]=f"Bearer {GITHUB_TOKEN}"
    return h

@st.cache_data(ttl=15*60, show_spinner=True)
def fetch_zip() -> bytes:
    if not GITHUB_TOKEN:
        raise RuntimeError("GitHub token gerekli: st.secrets['github_token'] veya env GITHUB_TOKEN.")
    url=f"https://api.github.com/repos/{OWNER}/{REPO}/actions/artifacts"
    r=requests.get(url,headers=_hdr(),timeout=30); r.raise_for_status()
    arts=[a for a in r.json().get("artifacts",[]) if a.get("name")==ARTIFACT and not a.get("expired",False)]
    if not arts: raise FileNotFoundError("Artifact bulunamadı.")
    arts.sort(key=lambda x:x.get("updated_at",""), reverse=True)
    dl=arts[0]["archive_download_url"]
    r2=requests.get(dl,headers=_hdr(),timeout=60); r2.raise_for_status()
    return r2.content

def _normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "datetime" in cols:
        df["datetime"] = pd.to_datetime(df[cols["datetime"]])
        return df
    if "date" not in cols:
        raise ValueError("date/datetime alanı bulunamadı.")
    base = pd.to_datetime(df[cols["date"]])
    if "hour" in cols:
        hr = pd.to_numeric(df[cols["hour"]], errors="coerce").fillna(0).astype(int)
    elif "hour_range" in cols:
        hr = pd.to_numeric(df[cols["hour_range"]].astype(str).str.extract(r"(\d+)")[0],
                           errors="coerce").fillna(0).astype(int)
    else:
        hr = 0
    df["datetime"] = base.dt.floor("D") + pd.to_timedelta(hr, unit="h")
    return df

def _normalize_geoid(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    df["geoid"] = (df[colname].astype(str)
                   .str.replace(r"\D","",regex=True)
                   .str.zfill(11).str[:11])
    return df

@st.cache_data(ttl=15*60, show_spinner=True)
def load_labeled() -> pd.DataFrame:
    z = fetch_zip()
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        names = zf.namelist()
        path = None
        for cand in LABEL_PARQUETS:
            hit = [n for n in names if n.endswith("/"+cand) or n.endswith(cand)]
            if hit:
                path = hit[0]; break
        if path is None:
            raise FileNotFoundError(f"Etiketli grid dosyası yok: {LABEL_PARQUETS}")
        with zf.open(path) as f:
            if path.endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_parquet(f)

    # normalize kolonlar
    cols = {c.lower(): c for c in df.columns}
    gcol = None
    for k in ("geoid","geoid10","geoid11","cell_id","id"):
        if k in cols: gcol = cols[k]; break
    if gcol is None:
        raise ValueError("GEOID kolonu bulunamadı.")
    df = _normalize_geoid(df, gcol)
    df = _normalize_datetime(df)

    # Y_label
    ycol = None
    for k in ("Y_label","y_label","label","y"):
        if k in df.columns: ycol = k; break
        if k.lower() in cols: ycol = cols[k.lower()]; break
    if ycol is None:
        raise ValueError("Y_label bulunamadı (0/1).")
    df["Y_label"] = pd.to_numeric(df[ycol], errors="coerce").fillna(0).astype(int)

    # crime_type opsiyonel
    if "crime_type" not in df.columns:
        if "crime" in cols: df = df.rename(columns={cols["crime"]: "crime_type"})
    # sadece gerekli kolonlar
    keep = ["geoid","datetime","Y_label"] + (["crime_type"] if "crime_type" in df.columns else [])
    return df[keep].sort_values("datetime")

# -----------------------------
# UI
# -----------------------------
df = load_labeled()

crime_opts = ["Hepsi"]
if "crime_type" in df.columns:
    crime_opts += sorted(df["crime_type"].dropna().unique().tolist())

c1, c2, c3 = st.columns([2,2,1])
with c1:
    crime_sel = st.selectbox("Suç türü", crime_opts, index=0)
    # crime_arg:
    #  - None  -> tüm suç türleri + Hepsi birlikte (Top-3 için gerekli)
    #  - "ALL" -> sadece Hepsi
    #  - "<type>" -> sadece o tür
    crime_arg = None if crime_sel == "Hepsi" else crime_sel

with c2:
    horizon_labels = ["24 saat","72 saat","7 gün","1 ay","3 ay","12 ay"]
    hz_label = st.selectbox("Ufuk", horizon_labels, index=0)
    to_hours = {"24 saat":24, "72 saat":72, "7 gün":24*7, "1 ay":24*30, "3 ay":24*90, "12 ay":24*365}
    horizons = [to_hours[hz_label]]

with c3:
    start_dt = st.datetime_input("Başlangıç", value=pd.to_datetime(df["datetime"].max()).floor("H"))
    refresh = st.button("Veriyi Yenile")
    if refresh:
        fetch_zip.clear(); load_labeled.clear()
        st.rerun()

st.markdown("---")

if st.button("Eğit & Tahmin"):
    # 1) Modeli eğit
    cfg = ForecastConfig(horizons=horizons,
                         crime_col=("crime_type" if "crime_type" in df.columns else "crime_type"))
    clf = CrimeForecaster(cfg)
    clf.fit(df)

    # 2) Gelecek gridini oluştur (tüm GEOID × seçili saat penceresi)
    geos = df["geoid"].unique()
    future_hours = pd.date_range(start=pd.to_datetime(start_dt), periods=horizons[0], freq="H", inclusive="right")
    grid = pd.MultiIndex.from_product([geos, future_hours], names=["geoid","datetime"]).to_frame(index=False)

    # 3) Tahmin
    if crime_arg is None:
        # Hepsi + TÜM suç türleri birlikte (Top-3 için gerekli)
        preds = clf.predict(grid, horizons=horizons, crime=None, return_all_crimes_if_none=True)
    elif crime_arg == "ALL":
        preds = clf.predict(grid, horizons=horizons, crime="ALL")
    else:
        preds = clf.predict(grid, horizons=horizons, crime=crime_arg)

    # 4) Çıktıları göster
    st.subheader("Tahminler — satır bazında")
    show_cols = ["geoid","datetime","horizon","risk_score","pred_expected","pi_low","pi_high","target"]
    if "target" not in preds.columns: preds["target"] = "ALL"
    st.dataframe(preds[show_cols].sort_values(["datetime","risk_score"], ascending=[True, False]),
                 use_container_width=True, height=520)

    st.download_button(
        "CSV indir (ham tahminler)",
        preds[show_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{crime_sel}_{hz_label.replace(' ','_')}.csv",
        mime="text/csv"
    )

    # 5) Özetler
    st.markdown("### Özet (GEOID bazında pencere özeti)")
    agg_geo = (preds.groupby(["geoid","target"], as_index=False)
                    .agg(risk_score_mean=("risk_score","mean"),
                         Ey_sum=("pred_expected","sum"),
                         Ey_low_sum=("pi_low","sum"),
                         Ey_high_sum=("pi_high","sum"))
                    .sort_values(["target","risk_score_mean"], ascending=[True, False]))
    st.dataframe(agg_geo, use_container_width=True, height=420)

    if crime_arg is None:
        # Tüm suç türleri mevcutken TOP-3 suç (ortalama risk ve toplam beklenen olay)
        st.markdown("### 🔝 En olası 3 suç türü (pencere boyunca)")
        top3 = (preds[preds["target"]!="ALL"]
                .groupby("target", as_index=False)
                .agg(mean_risk=("risk_score","mean"),
                     expected_count=("pred_expected","sum"))
                .sort_values("mean_risk", ascending=False)
                .head(3))
        st.dataframe(top3, use_container_width=True)

        st.download_button(
            "CSV indir (Top-3 suç özeti)",
            top3.to_csv(index=False).encode("utf-8"),
            file_name=f"top3_crimes_{hz_label.replace(' ','_')}.csv",
            mime="text/csv"
        )

    st.success("Tahminler üretildi.")
