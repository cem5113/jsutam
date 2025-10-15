# pages/Forecast.py
import io, os, json, zipfile
from datetime import datetime
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

RAW_C09 = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/main/crime_prediction_data/sf_crime_09.csv"

GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

st.set_page_config(page_title="🧭 Forecast — Saatlik Suç Riski", layout="wide")


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
    """
    risk_hourly.parquet'i oku ve kolonları normalize et.
    Beklenen kolonlar: geoid, hour (0–23), dow (0=Mon..6=Sun), season, risk_score, (opsiyonel) pred_expected
    """
    zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        matches = [n for n in memlist if n.endswith("/" + EXPECTED_PARQUET) or n.endswith(EXPECTED_PARQUET)]
        if not matches:
            raise FileNotFoundError(f"Zip içinde {EXPECTED_PARQUET} yok. Örnek içerik: {memlist[:12]}")
        with zf.open(matches[0]) as f:
            df = pd.read_parquet(f)

    # kolon adlarını normalize et
    df.columns = [c.strip().lower() for c in df.columns]

    # GEOID çıkar / normalize
    if "geoid" not in df.columns:
        for alt in ["cell_id", "geoid10", "geoid11", "geoid_10", "geoid_11", "id"]:
            if alt in df.columns:
                df["geoid"] = df[alt]
                break
    df["geoid"] = df["geoid"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)

    # hour/dow/season türetmeleri
    # varsa timestamp/datetime'ten çek
    ts_col = None
    for cand in ["ts", "timestamp", "datetime"]:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is not None:
        t = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_convert("US/Pacific")
        df["hour"] = df.get("hour", t.dt.hour)
        df["dow"] = df.get("dow", t.dt.dayofweek)
        # season kaba etiket
        month = t.dt.month
        season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                      6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
        df["season"] = df.get("season", month.map(season_map))
    else:
        # alternatif kolon isimleri
        for src, dst in [("hour_of_day","hour"), ("weekday","dow")]:
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

    # risk kolonunu aliasla (artifact'a göre değişebiliyor: proba / risk)
    if "risk_score" not in df.columns:
        if "proba" in df.columns:
            df = df.rename(columns={"proba": "risk_score"})
        elif "risk" in df.columns:
            df = df.rename(columns={"risk": "risk_score"})

    # güvenlik: gerekli kolonlar
    need = {"geoid","hour","dow","season","risk_score"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Eksik kolon(lar): {', '.join(sorted(miss))}")

    # tipler
    df["hour"] = df["hour"].astype(int).clip(0,23)
    df["dow"]  = df["dow"].astype(int).clip(0,6)
    df["season"] = df["season"].astype(str)

    return df

@st.cache_data(ttl=60*60, show_spinner=False)
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

# -------------------------
# Exposure fallback (sf_crime_09.csv)
# -------------------------
@st.cache_data(ttl=30*60, show_spinner=False)
def load_exposure_fallback() -> pd.DataFrame:
    """
    risk_hourly.parquet içinde 'pred_expected' yoksa
    saatlik beklenen olay için sf_crime_09.csv'den yaklaşık 'exposure_guess' üret:
    crime_last_7d / 7 (min 0.1). hour_range yoksa tüm saatlere uygula.
    """
    try:
        c9 = pd.read_csv(RAW_C09)
        if "GEOID" not in c9.columns:
            return pd.DataFrame(columns=["geoid","hour","exposure_guess"])
        c9["geoid"] = c9["GEOID"].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)
        if "hour_range" in c9.columns:
            # hour_range 0-23 kabul edilerek
            c9["hour"] = pd.to_numeric(c9["hour_range"], errors="coerce").fillna(0).astype(int).clip(0,23)
        else:
            c9 = c9.assign(hour=-1)  # -1 -> tüm saatler
        base = (c9.get("crime_last_7d", 0) / 7.0).clip(lower=0.1)
        c9["exposure_guess"] = base
        return c9[["geoid","hour","exposure_guess"]]
    except Exception:
        return pd.DataFrame(columns=["geoid","hour","exposure_guess"])

def attach_pred_expected(df: pd.DataFrame) -> pd.DataFrame:
    """
    'pred_expected' yoksa risk_score * exposure_guess ile tahmin et.
    """
    if "pred_expected" in df.columns:
        return df
    expo = load_exposure_fallback()
    if expo.empty:
        st.warning("Exposure kaynağı yüklenemedi → geçici 0.3 tabanı kullanılıyor.")
        df = df.copy()
        df["pred_expected"] = (df["risk_score"] * 0.3).round(3)
        return df
    # saat eşleşmesi (hour==-1 tüm saatlere uygula)
    df = df.copy()
    left = df.merge(expo[expo["hour"]!=-1], on=["geoid","hour"], how="left")
    if left["exposure_guess"].isna().all():
        # sadece hour==-1 varsa
        left = df.merge(expo[expo["hour"]==-1][["geoid","exposure_guess"]], on="geoid", how="left")
    left["exposure_guess"] = left["exposure_guess"].fillna(0.3)
    left["pred_expected"] = (left["risk_score"] * left["exposure_guess"]).round(3)
    return left

# -------------------------
# Harita çizimi
# -------------------------
def make_map_forecast(geojson: dict, df_layer: pd.DataFrame):
    if not geojson or df_layer.empty:
        st.info("Harita için GeoJSON veya veri yok.")
        return
    dmap = df_layer.set_index("geoid")["risk_score"].to_dict()
    qs = df_layer["risk_score"].quantile([0,.25,.5,.75,1]).tolist()
    layer = pdk.Layer(
        "GeoJsonLayer", geojson, stroked=False, opacity=.70, pickable=True,
        get_fill_color={
            "function": """
            const M=Object.fromEntries(py_dmap), Q=py_qs;
            return (f)=>{
              const raw=f.properties.GEOID ?? f.properties.geoid ?? f.properties.cell_id ?? "";
              const g=String(raw).replace(/\\D/g,"").padStart(11,"0").slice(0,11);
              const p=(M[g]===undefined)?0:M[g];
              if (p<=Q[1]) return [178,223,138,220];
              if (p<=Q[2]) return [255,255,178,220];
              if (p<=Q[3]) return [254,204,92,230];
              return [227,26,28,235];
            }"""
        },
        parameters={"py_dmap": list(dmap.items()), "py_qs": qs},
    )
    tooltip = {
        "html": (
            "<b>GEOID:</b> {GEOID}<br/>"
            "<b>Risk (p):</b> {risk_score}<br/>"
            "<b>E[olay]:</b> {pred_expected}"
        ),
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

# veri
with st.sidebar:
    st.caption("risk_hourly kaynağı: artifact_parquet")
    refresh = st.button("Veriyi Yenile (artifact'i tazele)")
try:
    if refresh:
        fetch_latest_artifact_zip.clear()
        read_risk_from_artifact.clear()
        fetch_geojson_smart.clear()
        load_exposure_fallback.clear()
    base_df = read_risk_from_artifact()
except Exception as e:
    st.error(f"Artifact indirilemedi/okunamadı: {e}")
    st.stop()

# pred_expected ekle
df = attach_pred_expected(base_df)

# filtre kontrolleri
with st.sidebar:
    hour = st.slider("Saat", 0, 23, 0)
    dow  = st.selectbox("Haftanın Günü (0=Mon … 6=Sun)", list(range(7)), index=0)
    # sezondaki seçenekleri veriden türet
    seasons = sorted(df["season"].astype(str).unique().tolist())
    def _idx_default():
        # içinde "Fall" vb. varsa uygun index, yoksa 0
        try:
            mo = datetime.now().month
            m2s = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                   6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
            return max(0, seasons.index(m2s[mo]))
        except Exception:
            return 0
    season = st.selectbox("Sezon", seasons, index=_idx_default())
    topk = st.slider("Top-K (kritik liste)", 10, 200, 50)

# seçime göre veri
sel = df[(df["hour"]==hour) & (df["dow"]==dow) & (df["season"].astype(str)==str(season))].copy()
sel = sel.sort_values("risk_score", ascending=False)

# üst metrikler
c1, c2, c3 = st.columns(3)
if not sel.empty:
    q = sel["risk_score"].quantile([0.25,0.5,0.75]).tolist()
    c1.metric("Q25", f"{q[0]:.3f}")
    c2.metric("Q50", f"{q[1]:.3f}")
    c3.metric("Q75", f"{q[2]:.3f}")
else:
    c1.metric("Q25", "-"); c2.metric("Q50", "-"); c3.metric("Q75", "-")

# GeoJSON
st.sidebar.header("Harita Sınırları (GeoJSON)")
geojson_local = st.sidebar.text_input("Local yol", value=GEOJSON_PATH_LOCAL_DEFAULT)
geojson_zip   = st.sidebar.text_input("Artifact ZIP içi yol", value=GEOJSON_IN_ZIP_PATH_DEFAULT)
geojson = fetch_geojson_smart(
    path_local=geojson_local,
    path_in_zip=geojson_zip,
    raw_owner=RAW_GEOJSON_OWNER,
    raw_repo=RAW_GEOJSON_REPO,
)

# Harita
st.subheader("Harita — Saatlik Risk (seçime göre)")
if not geojson:
    st.warning("GeoJSON bulunamadı (local/artifact/raw). Yolları kontrol edin.")
else:
    # pydeck tooltip verilerini properties'e enjekte etmeye gerek yok; layer parametreleriyle okuyoruz
    # Ancak pydeck, tooltipte {risk_score}/{pred_expected} için feature.properties'te arar.
    # Bunun için küçük bir binding tablosu oluşturup pydeck'in 'data' parametresini kullanabiliriz.
    # Basit yol: GeoJSONLayer kullanırken tooltipteki alanları feature.properties'ten alamıyorsak
    # JS tarafında gösterilenleri sınırlı tutarız. Burada mapping yöntemi: py_dmap ile renk, tablo ile bilgi.
    # Pratikte popover için tabloyu alta veriyoruz.
    make_map_forecast(geojson, sel[["geoid","risk_score","pred_expected"]])

# Top-K tablo
st.subheader("Kritik Top-K (E[olay] yüksek)")
if sel.empty:
    st.info("Seçilen filtreler için veri bulunamadı.")
else:
    top_df = sel.nlargest(topk, columns=["pred_expected"])[["geoid","risk_score","pred_expected"]]
    st.dataframe(top_df.reset_index(drop=True), use_container_width=True)
    csv = top_df.to_csv(index=False).encode("utf-8")
    st.download_button("Top-K CSV indir", csv, file_name=f"forecast_topk_h{hour}_d{dow}_{season}.csv", mime="text/csv")

# Alt notlar
with st.expander("Notlar"):
    st.markdown(
        "- `risk_score`: modelin saatlik olasılık/tahmin skorudur.\n"
        "- `E[olay] (pred_expected)`: **risk_score × exposure_guess** (artifact'ta yoksa) ile yaklaşık beklenen olay sayısıdır.\n"
        "- Exposure kaynağı `sf_crime_09.csv`; bulunamazsa 0.3 tabanı kullanılır."
    )
