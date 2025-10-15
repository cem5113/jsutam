# dorecast.py
# A self-contained forecasting script to produce short-, medium-, and weekly-ahead crime risk forecasts.
# - Treats a user-provided "app open" time as t0 (time zero).
# - Produces:
#     1) 24-hour forecast at 1-hour resolution
#     2) 72-hour forecast at 3-hour resolution
#     3) 7-day forecast at 24-hour resolution
# - Also supports ad-hoc prediction for any future datetime.
# - Trains a light baseline model (GradientBoostingClassifier) on history, using simple seasonal and lag features.
# - Applies an exponential temporal confidence decay so probabilities decrease as the horizon increases.
#
# Usage examples:
#   python dorecast.py --data /path/to/sf_crime_01.csv --start "2025-10-15 09:00" --geoid 60750117001 --modes 24h 72h 7d --out forecasts.csv
#   python dorecast.py --data /path/to/sf-crime.parquet --start "2025-10-15 09:00" --geoid 60750117001 60750117002 --predict_at "2025-10-20 18:00" --out single_pred.csv
#
# Notes:
# - Input data must include at minimum: ['datetime','GEOID','Y_label'].
# - The script will auto-detect CSV or Parquet based on extension.
# - Optional extra columns (e.g., 911/311/weather/POI) will be used if present.
# - No internet access required.
#
# Author: ChatGPT (GPT-5 Thinking)

import argparse
import sys
import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Utility functions
# ---------------------------

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq", ".parq"]:
        return pd.read_parquet(path)
    elif ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} (use .csv or .parquet)")

def ensure_datetime(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in data.")
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df

def add_time_features(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    dt = df[dt_col]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek      # 0=Mon
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df

def add_simple_lags(df: pd.DataFrame, group_col: str, target_col: str, dt_col: str, lags: List[int], roll_windows: List[int]) -> pd.DataFrame:
    """Create lagged and rolling sum features of the binary target for each group (e.g., GEOID).
       Assumes (group, dt) sorted ascending and (approx) hourly cadence. Missing hours are allowed; use resampling upstream if needed.
    """
    df = df.copy()
    df = df.sort_values([group_col, dt_col])
    for lag in lags:
        df[f"lag_{lag}h"] = df.groupby(group_col)[target_col].shift(lag)
    for w in roll_windows:
        # rolling sum over past w hours (exclusive of current hour)
        df[f"rollsum_{w}h"] = (
            df.groupby(group_col)[target_col]
              .shift(1)
              .rolling(window=w, min_periods=1)
              .sum()
        )
    return df

def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    """Selects model features from available columns.
       Priority: engineered lags/rolls + time features + any known contextual features if present.
    """
    base = [
        "hour", "dayofweek", "weekofyear", "month", "is_weekend",
        "lag_1h","lag_3h","lag_6h","lag_24h","lag_48h","lag_168h",
        "rollsum_6h","rollsum_24h","rollsum_72h","rollsum_168h"
    ]
    contextual_candidates = [
        # typical contextual columns; will be used if present
        "911_request_count_hour_range","911_request_count_daily",
        "311_request_count",
        "past_7d_crimes","crime_count_past_48h","neighbor_crime_7d",
        "poi_total_count","train_stop_count","bus_stop_count",
        "distance_to_police",
        "precipitation_mm","temp_max","temp_min"
    ]
    feats = [c for c in base if c in df.columns]
    feats += [c for c in contextual_candidates if c in df.columns]
    # avoid duplicates
    feats = list(dict.fromkeys(feats))
    return feats

def temporal_decay(prob: np.ndarray, horizon_hours: np.ndarray, tau_hours: float = 72.0) -> np.ndarray:
    """Exponential decay to downweight probabilities further out in time.
       p_decayed = p * exp(-h / tau)
    """
    decay = np.exp(-np.asarray(horizon_hours) / float(tau_hours))
    return np.clip(prob * decay, 0.0, 1.0)

def build_training_frame(df: pd.DataFrame,
                         geoid_list: Optional[List[str]],
                         t0: pd.Timestamp,
                         dt_col: str = "datetime",
                         geoid_col: str = "GEOID",
                         target_col: str = "Y_label") -> pd.DataFrame:
    """Filter to history up to t0 (exclusive) and selected GEOIDs (if provided)."""
    df = df.copy()
    if geoid_list:
        df = df[df[geoid_col].astype(str).isin([str(g) for g in geoid_list])]
    df = df[df[dt_col] < t0]
    return df

def fit_model(train_df: pd.DataFrame, features: List[str], target_col: str = "Y_label"):
    # Simple baseline classifier; robust and self-contained
    X = train_df[features].fillna(0.0).values
    y = train_df[target_col].astype(int).values
    if len(np.unique(y)) < 2:
        raise ValueError("Training target has a single class. Provide more data or different GEOIDs.")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    # optional quick CV (lightweight)
    try:
        tscv = TimeSeriesSplit(n_splits=3)
        aucs = []
        for tr, va in tscv.split(X):
            model_cv = GradientBoostingClassifier(random_state=42)
            model_cv.fit(X[tr], y[tr])
            p = model_cv.predict_proba(X[va])[:,1]
            aucs.append(roc_auc_score(y[va], p))
        if len(aucs) > 0:
            print(f"[Info] TimeSeriesSplit AUC (mean±std): {np.mean(aucs):.3f} ± {np.std(aucs):.3f}", file=sys.stderr)
    except Exception as e:
        print(f"[Warn] CV skipped: {e}", file=sys.stderr)
    return model

def simulate_forecast(last_history: pd.DataFrame,
                      model,
                      features: List[str],
                      start_time: pd.Timestamp,
                      horizon_hours: int,
                      step_hours: int,
                      geoid: str,
                      geoid_col: str = "GEOID",
                      dt_col: str = "datetime",
                      tau_hours: float = 72.0) -> pd.DataFrame:
    """Recursive multi-step forecast per GEOID at a fixed step size.
       last_history must contain the most recent rows for that GEOID up to start_time.
    """
    # Work on a copy
    hist = last_history.copy().sort_values(dt_col)
    # Prepare container for outputs
    out_rows = []

    # Helper to build a single row of features for a candidate timestamp
    def make_feat_row(ts: pd.Timestamp) -> pd.Series:
        tmp = pd.DataFrame({dt_col: [ts]})
        tmp = add_time_features(tmp, dt_col)
        # Carry forward the latest known/estimated lags/rolls from hist
        # We re-compute engineered features by appending a dummy target (predicted expectation)
        return tmp.iloc[0]

    # For engineered lags/rolls, we will rebuild on the fly by appending a synthetic row.
    # To do so efficiently, maintain only necessary columns.
    needed_cols = [dt_col, "Y_label"] + [c for c in hist.columns if c not in [dt_col, "Y_label"]]
    hist = hist[needed_cols]

    steps = int(np.ceil(horizon_hours / step_hours))
    for k in range(1, steps + 1):
        ts = start_time + pd.Timedelta(hours=step_hours * (k-1))
        ts_end = ts + pd.Timedelta(hours=step_hours)

        # Build a candidate row using current hist to compute lags/rolls
        # Create a placeholder row with Y_label = np.nan (unknown), then compute lags/rolls using helper
        cand = pd.DataFrame({dt_col: [ts], "Y_label": [np.nan]})
        # Add seasonal features
        cand = add_time_features(cand, dt_col)

        # Merge with latest contextual columns if they exist and are static (we don't know future true values).
        # For simplicity, we set them to the last known values in history (per-geo).
        static_cols = [
            "911_request_count_hour_range","911_request_count_daily","311_request_count",
            "past_7d_crimes","crime_count_past_48h","neighbor_crime_7d",
            "poi_total_count","train_stop_count","bus_stop_count",
            "distance_to_police","precipitation_mm","temp_max","temp_min"
        ]
        last_vals = hist.dropna().tail(1)
        for sc in static_cols:
            if sc in hist.columns:
                val = last_vals[sc].iloc[0] if sc in last_vals.columns and len(last_vals) else np.nan
                cand[sc] = val

        # To compute lag/rolling features, temporarily append cand with a synthetic Y estimate (use previous prob or mean)
        # We'll compute lags from hist (which contains true past), so first compute engineered lags on a combined frame
        tmp = pd.concat([hist, cand], ignore_index=True)
        tmp = add_simple_lags(tmp, group_col=None if geoid_col not in tmp.columns else geoid_col,
                              target_col="Y_label", dt_col=dt_col,
                              lags=[1,3,6,24,48,168], roll_windows=[6,24,72,168])

        # The new engineered values will be on the last row:
        engineered_cols = [c for c in tmp.columns if c.startswith("lag_") or c.startswith("rollsum_")]
        for c in engineered_cols:
            cand[c] = tmp[c].iloc[-1]

        # Select feature columns
        feat_cols = pick_feature_columns(pd.concat([cand], axis=0))
        X = cand[feat_cols].fillna(0.0).values
        raw_prob = float(model.predict_proba(X)[:,1][0])

        # Apply temporal decay based on this step's start time relative to t0
        h = (ts - start_time).total_seconds() / 3600.0
        prob_decayed = float(temporal_decay(np.array([raw_prob]), np.array([h]), tau_hours=tau_hours)[0])

        out_rows.append({
            geoid_col: geoid,
            "window_hours": step_hours,
            "t_start": ts,
            "t_end": ts_end,
            "prob_raw": raw_prob,
            "prob_decayed": prob_decayed
        })

        # Update history with a synthetic outcome expectation to advance lags:
        # we use expected value of event occurrence ~ prob_decayed (between 0 and 1)
        synth = cand.copy()
        synth["Y_label"] = prob_decayed
        hist = pd.concat([hist, synth[hist.columns]], ignore_index=True)

    return pd.DataFrame(out_rows)

def run_forecasts(data_path: str,
                  start_time: str,
                  geoids: List[str],
                  modes: List[str],
                  out_path: Optional[str],
                  tau_hours: float):
    # Load
    df = read_any(data_path)
    df = ensure_datetime(df, "datetime")
    df["GEOID"] = df["GEOID"].astype(str)

    # Sort and engineer base features
    df = df.sort_values(["GEOID", "datetime"])
    df = add_time_features(df, "datetime")

    # Build simple lag/rolling features on history (used for training)
    df = add_simple_lags(df, group_col="GEOID", target_col="Y_label", dt_col="datetime",
                         lags=[1,3,6,24,48,168], roll_windows=[6,24,72,168])

    t0 = pd.to_datetime(start_time)

    all_outputs = []

    for geoid in geoids:
        # Prepare train frame up to t0
        tr = build_training_frame(df, [geoid], t0, dt_col="datetime", geoid_col="GEOID", target_col="Y_label")

        if tr.empty or tr["Y_label"].sum() == 0:
            print(f"[Warn] Not enough positive samples for GEOID={geoid} before {t0}. Skipping.", file=sys.stderr)
            continue

        # Fit model
        feats = pick_feature_columns(tr)
        model = fit_model(tr, feats, target_col="Y_label")

        # Last history slice for recursive forecasting (recent 14 days to keep memory small)
        last_hist = df[(df["GEOID"] == geoid) & (df["datetime"] < t0)]
        last_hist = last_hist[last_hist["datetime"] >= (t0 - pd.Timedelta(days=14))]

        # Generate forecasts per requested modes
        for m in modes:
            m = m.lower()
            if m == "24h":
                horizon, step = 24, 1
            elif m == "72h":
                horizon, step = 72, 3
            elif m == "7d":
                horizon, step = 24*7, 24
            else:
                print(f"[Warn] Unknown mode '{m}', skipping. Use one of: 24h, 72h, 7d", file=sys.stderr)
                continue

            fc = simulate_forecast(last_hist, model, feats, t0, horizon, step, geoid,
                                   geoid_col="GEOID", dt_col="datetime", tau_hours=tau_hours)
            fc["mode"] = m
            all_outputs.append(fc)

    if not all_outputs:
        print("[Error] No forecasts produced. Check data, start time, and GEOIDs.", file=sys.stderr)
        return None

    out = pd.concat(all_outputs, ignore_index=True).sort_values(["GEOID", "t_start"])
    if out_path:
        # Save as CSV (universal); also save as Parquet if requested by extension
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".parquet":
            out.to_parquet(out_path, index=False)
        else:
            out.to_csv(out_path, index=False)
        print(f"[Info] Saved forecasts to: {out_path}", file=sys.stderr)
    return out

def predict_at_datetime(data_path: str,
                        start_time: str,
                        target_time: str,
                        geoids: List[str],
                        tau_hours: float) -> pd.DataFrame:
    """Ad-hoc prediction for an arbitrary future datetime per GEOID, using the same trained baseline per GEOID."""
    df = read_any(data_path)
    df = ensure_datetime(df, "datetime")
    df["GEOID"] = df["GEOID"].astype(str)
    df = df.sort_values(["GEOID", "datetime"])
    df = add_time_features(df, "datetime")
    df = add_simple_lags(df, group_col="GEOID", target_col="Y_label", dt_col="datetime",
                         lags=[1,3,6,24,48,168], roll_windows=[6,24,72,168])
    t0 = pd.to_datetime(start_time)
    tt = pd.to_datetime(target_time)

    rows = []
    for geoid in geoids:
        tr = build_training_frame(df, [geoid], t0, dt_col="datetime", geoid_col="GEOID", target_col="Y_label")
        if tr.empty or tr["Y_label"].sum() == 0:
            print(f"[Warn] Not enough positives for GEOID={geoid} before {t0}. Skipping.", file=sys.stderr)
            continue

        feats = pick_feature_columns(tr)
        model = fit_model(tr, feats, target_col="Y_label")

        last_hist = df[(df["GEOID"] == geoid) & (df["datetime"] < t0)]
        last_hist = last_hist[last_hist["datetime"] >= (t0 - pd.Timedelta(days=14))]

        # Single step: reuse simulate_forecast with horizon/step so that ts == target bin start
        horizon_hours = max(1, int(np.ceil((tt - t0).total_seconds() / 3600.0)))
        step = 1
        fc = simulate_forecast(last_hist, model, feats, t0, horizon_hours, step, geoid,
                               geoid_col="GEOID", dt_col="datetime", tau_hours=tau_hours)
        # pick closest row whose [t_start, t_end) covers target time
        sel = fc[(fc["t_start"] <= tt) & (fc["t_end"] > tt)]
        if sel.empty:
            # fallback to nearest t_start
            sel = fc.iloc[[np.argmin(np.abs((fc["t_start"] - tt).dt.total_seconds()))]]
        r = sel.iloc[0].to_dict()
        r["mode"] = "adhoc"
        rows.append(r)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Short/medium/weekly-ahead crime risk forecasting with temporal decay.")
    p.add_argument("--data", required=True, help="Path to CSV or Parquet data with columns [datetime, GEOID, Y_label, ...]")
    p.add_argument("--start", required=True, help="Start timestamp (t0, 'app open' time). e.g., '2025-10-15 09:00'")
    p.add_argument("--geoid", nargs="+", required=True, help="One or more GEOID values to forecast")
    p.add_argument("--modes", nargs="*", default=["24h","72h","7d"], help="Any of: 24h 72h 7d")
    p.add_argument("--out", default=None, help="Output file path (.csv or .parquet). If omitted, prints nothing but saves via API.")
    p.add_argument("--tau", type=float, default=72.0, help="Decay constant (hours) for probability downweighting (default: 72h)")
    p.add_argument("--predict_at", default=None, help="Optional single datetime to get ad-hoc prediction (e.g., '2025-10-20 18:00')")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if args.predict_at:
        out = predict_at_datetime(args.data, args.start, args.predict_at, args.geoid, tau_hours=args.tau)
    else:
        out = run_forecasts(args.data, args.start, args.geoid, args.modes, args.out, tau_hours=args.tau)
    if out is None or out.empty:
        print("[Error] No output.", file=sys.stderr)
        return 1
    # If no out path was provided, write to a default file next to input
    if args.out is None:
        default_out = os.path.join(os.path.dirname(args.data) or ".", "forecasts_out.csv")
        out.to_csv(default_out, index=False)
        print(f"[Info] Saved forecasts to: {default_out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
