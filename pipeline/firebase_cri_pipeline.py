"""
firebase_cri_pipeline.py
========================
Scheduled batch job: Firebase Realtime Database -> cleaned readings -> CRI -> results feed.
Designed to run as a GitHub Actions scheduled workflow (see the .yml alongside
this file) — which means it MUST NOT depend on local disk surviving between
runs. All state (cursor position, rolling CRI history) lives in Firebase
itself, not in local files.

============================================================================
 DEPLOYMENT: GITHUB ACTIONS (recommended for your setup)
============================================================================
Why this over the alternatives, given you're not managing existing cloud
infra: zero servers to maintain, free for a job this light and infrequent,
version-controlled alongside your notebooks, and easy to see run history/
logs/failures in one place. The one thing it requires — since runners are
stateless — is that this script keep NO state on local disk. That's why
cursor + rolling store below read/write Firebase, not files.

Setup:
  1. Put this script in a repo, e.g. `pipeline/firebase_cri_pipeline.py`.
  2. Also commit the frozen, precomputed files it reads (these don't change
     every run, only when you recompute them offline in Colab):
       pipeline/SEASONAL_CAUSAL_WEIGHTS_FAST/{season}/Global_Ensemble_Causal_Weights_{season}.csv
       pipeline/Normalisation_Bounds_AllSites.csv
  3. Firebase console -> Project settings -> Service accounts -> Generate new
     private key. Copy the ENTIRE JSON contents.
  4. GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository
     secret. Name: FIREBASE_SERVICE_ACCOUNT_KEY. Paste the JSON as the value.
     Never commit this key to the repo itself.
  5. Firebase console -> Realtime Database -> Rules -> add:
       {
         "rules": {
           "sensor_readings": { ".indexOn": ["timestamp"] },
           "waterLogs":       { ".indexOn": ["time"] },
           "pipeline_state":  { ".read": "auth != null", ".write": "auth != null" },
           "processed_cri":   { ".read": "auth != null", ".write": "auth != null" },
           "results":         { ".read": true, ".write": "auth != null" }
         }
       }
     (results is public-read so the dashboard can fetch it with no credentials;
     everything else stays private, only this script's service account can touch it.)
  6. Add the workflow file (deploy/cri_pipeline.yml in this bundle) to
     `.github/workflows/` in the repo. It runs on a schedule and installs
     `firebase-admin`, `pandas`, `numpy` before invoking this script.
  7. Push. Check the Actions tab for the first run's logs.

Alternatives, if you outgrow this: Firebase Cloud Functions (scheduled) keeps
everything inside one project but needs the Blaze plan; a cron job on a VM
gives you more control but is one more thing to keep patched and running.
GitHub Actions is the right starting point.

============================================================================
 WHAT THIS DOES
============================================================================
1. Pulls new records from `sensor_readings` and `waterLogs` since the cursor
   stored at Firebase `/pipeline_state/cursor`.
2. Matches each `waterLogs` row to one of your 11 sites by GPS (SITE_COORDS).
3. Applies the site-specific anomaly handling ported from
   Data_Processing_AllSites.ipynb (SITE_ANOMALIES catalogue: S10 dual-stream
   removal, ORP scale correction, BGA/chlorophyll sign handling, turbidity
   sensor-cap detection, temp error-code interpolation, etc.), plus one new
   rule (pH==0 exact) specific to this logger's fault pattern.
4. Computes CRI per window using your EXISTING frozen per-site causal weights
   + normalisation bounds (does NOT re-run PCMCI/TE/NGC — see below).
5. Appends results to `/processed_cri` in Firebase (not local disk) and
   rebuilds `/results/dashboard_feed`, which the dashboard fetches directly.

============================================================================
 WHAT THIS DELIBERATELY DOES NOT DO
============================================================================
- Does NOT recompute the ensemble causal weights (PCMCI + TE + NGC + GC) —
  that took ~80 min PER SEASON on Colab. Recompute manually/offline as you
  already do, then update the CSVs committed in step 2 above.
- Does NOT recompute CWM-CSD (PE / DFA / RQA) every cycle — those need much
  larger accumulated windows (DFA wants 1200+, RQA a 120-step window). Use
  --recompute-cwm-csd on a separate, slower cron (daily/weekly), not every run.
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
FIREBASE_DB_URL = "https://mywaterproject-e6489-default-rtdb.asia-southeast1.firebasedatabase.app/"
SERVICE_ACCOUNT_KEY_PATH = os.environ.get("FIREBASE_KEY_PATH", "serviceAccountKey.json")

CAUSAL_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "SEASONAL_CAUSAL_WEIGHTS_FAST")
NORMALISATION_BOUNDS_PATH = os.path.join(os.path.dirname(__file__), "Normalisation_Bounds_AllSites.csv")

FEATURES = ["dissolvedo2", "orp", "ph", "tds", "temp", "turbidity", "bga", "chlorophyll"]
WINDOW_MINUTES = 5

# Firebase paths used for STATE (must survive between ephemeral runner instances)
FB_CURSOR_PATH = "pipeline_state/cursor"
FB_PROCESSED_CRI_PATH = "processed_cri"     # {site}/{season}/{window_key} -> {cri, t}
FB_RESULTS_PATH = "results/dashboard_feed"  # public-read, what the dashboard fetches


# ── SITE MATCHING (GPS-based) ──────────────────────────────────────────────────
# True centroids from raw Premonsoon_Final.xlsx, GPS-fault (0,0) rows excluded.
# 9 of 11 sites are near-fixed logger deployments (radius ~0-0.1km); S5, S9, S11
# were boat-surveyed across wider areas, so each site gets its own match radius.
SITE_COORDS = {
    "S1-Ulhas River":     (19.2542,    73.221363, 0.3),
    "S2-Kalu River":      (19.304704,  73.241197, 0.3),
    "S3-Ambernath Lake":  (19.182509,  73.192435, 0.3),
    "S4-Bhandup Lake":    (19.14652,   72.930468, 0.3),
    "S5-Dhamapur Lake":   (16.035893,  73.590129, 1.8),
    "S6-Shirpur Lake":    (21.068528,  74.859325, 0.3),
    "S7-Varawade River":  (16.152213,  73.403326, 0.3),
    "S8-Gotli River":     (16.14326,   73.37373,  0.3),
    "S9-Karli River":     (16.020966,  73.630686, 8.0),
    "S10-Chivla River":   (16.04156,   73.28391,  0.3),
    "S11-Powai Lake":     (19.122641,  72.911688, 3.0),
}


def haversine_km(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def match_site(lat, lon):
    if lat is None or lon is None or (lat == 0 and lon == 0):
        return "UNMATCHED"
    best_site, best_dist, best_radius = None, float("inf"), None
    for site, (slat, slon, radius) in SITE_COORDS.items():
        d = haversine_km(lat, lon, slat, slon)
        if d < best_dist:
            best_site, best_dist, best_radius = site, d, radius
    return best_site if best_dist <= best_radius else "UNMATCHED"


def infer_season(dt):
    month = dt.month
    if month in (3, 4, 5):
        return "premonsoon"
    elif month in (6, 7, 8, 9):
        return "monsoon"
    else:
        return "postmonsoon"


def firebase_safe_key(dt):
    """Firebase keys can't contain . $ # [ ] / — encode the window timestamp safely."""
    return dt.strftime("%Y%m%dT%H%M")


# ── 1. FIREBASE I/O ────────────────────────────────────────────────────────────
def get_db():
    import firebase_admin
    from firebase_admin import credentials, db
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    return db


def fetch_new_records(node, since_value, timestamp_field):
    """Requires `.indexOn": ["<timestamp_field>"]` on this node — see rules in the module docstring."""
    db = get_db()
    ref = db.reference(node)
    snapshot = ref.order_by_child(timestamp_field).start_at(since_value).get() or {}
    return list(snapshot.values())


def load_cursor():
    db = get_db()
    cursor = db.reference(FB_CURSOR_PATH).get()
    if cursor:
        return cursor
    default_since = datetime.utcnow() - timedelta(hours=24)
    return {
        "sensor_readings_since": default_since.strftime("%Y-%m-%d %H:%M:%S"),
        "waterLogs_since": default_since.strftime("%d-%m-%Y %H:%M:%S"),
    }


def save_cursor(cursor):
    db = get_db()
    db.reference(FB_CURSOR_PATH).set(cursor)


def append_processed_cri_to_firebase(df):
    """
    Writes new CRI windows as a single multi-location update (one network
    round-trip) rather than one write per row. Path shape:
      processed_cri/{site}/{season}/{YYYYMMDDTHHMM} -> {"t": iso, "cri": float}
    """
    if df.empty:
        return
    db = get_db()
    updates = {}
    for _, row in df.iterrows():
        site_key = row["site"].replace(" ", "_").replace(".", "")
        key = firebase_safe_key(row["window"])
        updates[f"{FB_PROCESSED_CRI_PATH}/{site_key}/{row['season']}/{key}"] = {
            "t": row["window"].strftime("%Y-%m-%d %H:%M"),
            "cri": None if pd.isna(row["cri"]) else round(float(row["cri"]), 4),
            "site_display": row["site"],
        }
    db.reference().update(updates)


def read_processed_cri_from_firebase():
    """Pulls the full accumulated CRI history back out for feed-building."""
    db = get_db()
    raw = db.reference(FB_PROCESSED_CRI_PATH).get() or {}
    rows = []
    for site_key, seasons in raw.items():
        for season, windows in (seasons or {}).items():
            for _, rec in (windows or {}).items():
                rows.append({
                    "site": rec.get("site_display", site_key),
                    "season": season,
                    "t": rec.get("t"),
                    "cri": rec.get("cri"),
                })
    return pd.DataFrame(rows)


def write_results_to_firebase(feed):
    """
    Public-readable results the browser dashboard fetches with a plain,
    unauthenticated GET at: {FIREBASE_DB_URL}/results/dashboard_feed.json
    """
    db = get_db()
    db.reference(FB_RESULTS_PATH).set(feed)


# ── 2. SITE-SPECIFIC PREPROCESSING (ported from Data_Processing_AllSites.ipynb) ─
def interpolate_flagged(series, mask, limit=3):
    s = series.copy().astype(float)
    s[mask] = np.nan
    return s.interpolate(method="linear", limit=limit, limit_direction="both")


def remove_s10_zero_stream(df):
    if df.empty or "temp" not in df.columns:
        return df, 0
    temp_rounded = df["temp"].round(3)
    if temp_rounded.dropna().empty:
        return df, 0
    temp_mode = temp_rounded.value_counts().idxmax()
    zero_stream = (df["dissolvedo2"] == 0) & (df["tds"] == 0) & (temp_rounded == temp_mode)
    n_removed = int(zero_stream.sum())
    return df[~zero_stream].reset_index(drop=True), n_removed


def estimate_orp_scale(site_name, season, orp_median):
    if pd.isna(orp_median):
        return 1.0, "no_data"
    if "S3-Ambernath" in site_name:
        return 1.0, "valid_no_correction"
    if "S4-Bhandup" in site_name and season == "postmonsoon":
        scale = max(1.0, round(abs(orp_median) / 600))
        return scale, f"partial_scale_{scale}x"
    scale = round(abs(orp_median) / 600)
    scale = max(2, min(scale, 20))
    return scale, f"scale_{scale}x"


def process_site_batch(df, site_name, season):
    df = df.copy()
    log = []

    if "S10" in site_name or "Chivla" in site_name:
        if season in ("monsoon", "postmonsoon"):
            df, n_removed = remove_s10_zero_stream(df)
            if n_removed:
                log.append({"param": "ALL", "n": n_removed,
                            "note": f"S10 dual-stream artefact: removed {n_removed} all-zero logger rows"})

    if "temp" in df.columns:
        temp_hi = df["temp"] > 45
        if temp_hi.sum():
            df["temp"] = interpolate_flagged(df["temp"], temp_hi)
            log.append({"param": "temp", "n": int(temp_hi.sum()), "note": "interpolated YSI error-code spikes >45C"})
        temp_neg = df["temp"] < 0
        if temp_neg.sum():
            df["temp"] = interpolate_flagged(df["temp"], temp_neg)
            log.append({"param": "temp", "n": int(temp_neg.sum()), "note": "interpolated negative temp (sensor noise)"})

    if "ph" in df.columns:
        ph_neg = df["ph"] < 0
        if ph_neg.sum():
            df["ph"] = interpolate_flagged(df["ph"], ph_neg)
            log.append({"param": "ph", "n": int(ph_neg.sum()),
                        "note": "interpolated pH<0 (physically impossible); low positive pH retained"})
        # NEW RULE (not in the original Excel-based catalogue): this Firebase
        # logger reports pH==0 exactly when the probe is disconnected/unstable.
        # The 11-site Excel data never hit this pattern.
        ph_zero = df["ph"] == 0
        if ph_zero.sum():
            df["ph"] = interpolate_flagged(df["ph"], ph_zero)
            log.append({"param": "ph", "n": int(ph_zero.sum()),
                        "note": "NEW RULE: interpolated pH==0 exact (probe disconnected on this logger)"})

    if "dissolvedo2" in df.columns:
        do_neg = df["dissolvedo2"] < 0
        if do_neg.sum():
            df["dissolvedo2"] = interpolate_flagged(df["dissolvedo2"], do_neg)
            log.append({"param": "dissolvedo2", "n": int(do_neg.sum()),
                        "note": "interpolated negative DO (optical zero-point noise); supersaturation >150% retained"})

    if "bga" in df.columns:
        df["bga_raw"] = df["bga"].copy()
        df["bga_sign"] = np.sign(df["bga"])
        n_neg = int((df["bga_raw"] < 0).sum())
        df["bga"] = df["bga"].abs()
        if n_neg:
            log.append({"param": "bga", "n": n_neg, "note": "abs(BGA) transform; probe oscillation in bloom, magnitude retained"})

    if "chlorophyll" in df.columns:
        df["chlorophyll_raw"] = df["chlorophyll"].copy()
        df["chlorophyll_sign"] = np.sign(df["chlorophyll"])
        n_neg = int((df["chlorophyll_raw"] < 0).sum())
        df["chlorophyll"] = df["chlorophyll"].abs()
        if n_neg:
            log.append({"param": "chlorophyll", "n": n_neg,
                        "note": "abs(chlorophyll) transform; phycocyanin quenching in cyanobacterial bloom"})

    if "turbidity" in df.columns:
        turb = pd.to_numeric(df["turbidity"], errors="coerce")
        if len(turb) and (turb == 0).all():
            df["turbidity_flag"] = "missing_sensor_cap"
            df["turbidity"] = np.nan
            log.append({"param": "turbidity", "n": len(df),
                        "note": "ALL turbidity=0 -> sensor cap not removed; excluded from causal model"})
        else:
            n_neg = int((turb < 0).sum())
            if n_neg:
                df["turbidity"] = turb.clip(lower=0)
                log.append({"param": "turbidity", "n": n_neg, "note": "clipped near-zero negative turbidity to 0"})

    if "tds" in df.columns:
        tds = pd.to_numeric(df["tds"], errors="coerce")
        deep_neg = tds < -500
        mild_neg = (tds >= -500) & (tds < 0)
        if deep_neg.sum():
            df["tds"] = interpolate_flagged(df["tds"], deep_neg)
            log.append({"param": "tds", "n": int(deep_neg.sum()), "note": "interpolated TDS<-500 (conductivity drift)"})
        if mild_neg.sum():
            df.loc[mild_neg, "tds"] = 0
            log.append({"param": "tds", "n": int(mild_neg.sum()), "note": "set TDS in [-500,0) to 0 (calibration noise)"})

    if "orp" in df.columns:
        df["orp_raw"] = df["orp"].copy()
        orp_median = df["orp"].median()
        scale, label = estimate_orp_scale(site_name, season, orp_median)
        if scale == 1.0 and label == "valid_no_correction":
            n_extreme = int((df["orp"].abs() > 3000).sum())
            df["orp"] = df["orp"].clip(-3000, 3000)
            df["orp_correction"] = "valid_clipped_3000mV"
            log.append({"param": "orp", "n": n_extreme, "note": "S3-Ambernath: valid, clipped extremes beyond +-3000mV"})
        elif label != "no_data":
            df["orp"] = df["orp"] / scale
            df["orp_correction"] = label
            log.append({"param": "orp", "n": len(df), "note": f"scale correction /{scale} (median was {orp_median:.0f} mV)"})

    for col in ["dissolvedo2", "ph", "tds", "temp", "bga", "chlorophyll"]:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if 0 < n_nan < len(df) * 0.3:
                df[col] = df[col].interpolate(limit=5, limit_direction="both").bfill().ffill()
                log.append({"param": col, "n": int(n_nan), "note": "residual NaN fill after processing"})

    for row in log:
        row["site"] = site_name
        row["season"] = season
    return df, log


# ── 3. LOAD + GPS-MATCH + JOIN THE TWO RAW STREAMS ────────────────────────────
def load_and_match(wl_raw, sr_raw):
    wl_df = pd.DataFrame(wl_raw)
    sr_df = pd.DataFrame(sr_raw)

    if not wl_df.empty:
        wl_df = wl_df.rename(columns={"dissolvedO2": "dissolvedo2", "pH": "ph", "tempC": "temp"})
        wl_df["site"] = wl_df.apply(lambda r: match_site(r.get("lat"), r.get("lon")), axis=1)
        wl_df["time_parsed"] = pd.to_datetime(wl_df["time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        wl_df["window"] = wl_df["time_parsed"].dt.floor(f"{WINDOW_MINUTES}min")

    if not sr_df.empty:
        sr_df = sr_df.rename(columns={
            "blue_green_algae_cells_per_mL": "bga", "chlorophyll_ug_per_L": "chlorophyll",
        })
        sr_df["time_parsed"] = pd.to_datetime(sr_df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        sr_df["window"] = sr_df["time_parsed"].dt.floor(f"{WINDOW_MINUTES}min")

    return wl_df, sr_df


def join_streams(wl_df, sr_df):
    if wl_df.empty:
        return pd.DataFrame()

    n_unmatched = (wl_df["site"] == "UNMATCHED").sum()
    if n_unmatched:
        print(f"NOTE: {n_unmatched} waterLogs rows didn't match any known site "
              f"(or had a (0,0) GPS fault) — excluded. Check hdop/sat if unexpected.")

    wl_g = (
        wl_df[wl_df["site"] != "UNMATCHED"]
        .groupby(["site", "window"])[["dissolvedo2", "orp", "ph", "tds", "temp", "turbidity"]]
        .mean()
        .reset_index()
    )

    if not sr_df.empty:
        sr_g = sr_df.groupby("window")[["bga", "chlorophyll"]].mean().reset_index()
        merged = wl_g.merge(sr_g, on="window", how="left")
    else:
        merged = wl_g
        merged["bga"] = np.nan
        merged["chlorophyll"] = np.nan

    return merged


# ── 4. CRI COMPUTATION (frozen weights, same formula as CRI_Computation_AllSites.ipynb) ──
def load_causal_weights(season, site):
    path = os.path.join(CAUSAL_WEIGHTS_DIR, season, f"Global_Ensemble_Causal_Weights_{season}.csv")
    if not os.path.exists(path):
        print(f"WARNING: no frozen weights for {season} at {path} — using equal weights")
        return {f: 1 / len(FEATURES) for f in FEATURES}
    wdf = pd.read_csv(path)
    wdf.columns = wdf.columns.str.lower()
    site_rows = wdf[wdf["site"].str.lower() == site.lower()]
    if site_rows.empty:
        print(f"WARNING: site '{site}' not in frozen weights — using equal weights")
        return {f: 1 / len(FEATURES) for f in FEATURES}
    return dict(zip(site_rows["driver"], site_rows["weight"]))


def load_normalisation_bounds(site):
    if not os.path.exists(NORMALISATION_BOUNDS_PATH):
        return None
    bdf = pd.read_csv(NORMALISATION_BOUNDS_PATH)
    bdf.columns = bdf.columns.str.lower()
    site_rows = bdf[bdf["site"].str.lower() == site.lower()]
    if site_rows.empty:
        return None
    return {row["feature"]: (row["min"], row["max"]) for _, row in site_rows.iterrows()}


def compute_cri(df, season, site):
    if df.empty:
        return df
    weights = load_causal_weights(season, site)
    bounds = load_normalisation_bounds(site)

    df = df.copy()
    cri = np.zeros(len(df))
    total_weight = np.zeros(len(df))

    for feat in FEATURES:
        if feat not in df.columns:
            continue
        col = pd.to_numeric(df[feat], errors="coerce")
        if bounds and feat in bounds:
            fmin, fmax = bounds[feat]
        else:
            fmin, fmax = col.min(), col.max()  # fallback only; noisy on small batches
        rng = (fmax - fmin) if (fmax - fmin) > 1e-9 else 1.0
        norm = ((col - fmin) / rng).clip(0, 1).fillna(0)
        w = weights.get(feat, 0.0)
        cri += norm.values * w
        total_weight += np.where(col.notna(), w, 0)

    df["cri"] = np.where(total_weight > 0, cri / np.where(total_weight == 0, 1, total_weight), np.nan)
    df["season"] = season
    df["site"] = site
    return df


# ── 5. RESULTS FEED ────────────────────────────────────────────────────────────
def build_dashboard_feed(all_cri_history):
    if all_cri_history.empty:
        print("No CRI history yet — nothing to write.")
        return None

    feed = {"generated_at": datetime.utcnow().isoformat(), "criTimeseries": [], "cri_summary_live": []}

    for (site, season), g in all_cri_history.groupby(["site", "season"]):
        for _, row in g.iterrows():
            feed["criTimeseries"].append({"season": season, "site": site, "t": row["t"], "cri": row["cri"]})
        cri_vals = g["cri"].dropna()
        if len(cri_vals):
            feed["cri_summary_live"].append({
                "site": site, "season": season,
                "cri_mean": round(float(cri_vals.mean()), 4),
                "cri_std": round(float(cri_vals.std()), 4),
                "n_windows": int(len(cri_vals)),
                "last_updated": str(g["t"].max()),
            })
    return feed


# ── 6. MAIN ────────────────────────────────────────────────────────────────────
def run_batch():
    cursor = load_cursor()

    wl_raw = fetch_new_records("waterLogs", cursor["waterLogs_since"], "time")
    sr_raw = fetch_new_records("sensor_readings", cursor["sensor_readings_since"], "timestamp")
    print(f"Pulled {len(wl_raw)} waterLogs, {len(sr_raw)} sensor_readings since last run.")

    if wl_raw or sr_raw:
        wl_df, sr_df = load_and_match(wl_raw, sr_raw)
        joined = join_streams(wl_df, sr_df)

        if not joined.empty:
            joined["season"] = joined["window"].apply(infer_season)

            all_audit_rows, cri_frames = [], []
            for (site, season), g in joined.groupby(["site", "season"]):
                cleaned, audit_rows = process_site_batch(g, site, season)
                all_audit_rows.extend(audit_rows)
                cri_frames.append(compute_cri(cleaned, season, site))

            new_results = pd.concat(cri_frames, ignore_index=True) if cri_frames else pd.DataFrame()
            if not new_results.empty:
                append_processed_cri_to_firebase(new_results)
                for row in all_audit_rows:
                    print(f"  [{row['site']} / {row['season']}] {row['param']}: {row['note']} (n={row['n']})")

        if wl_raw:
            cursor["waterLogs_since"] = max(r.get("time", cursor["waterLogs_since"]) for r in wl_raw)
        if sr_raw:
            cursor["sensor_readings_since"] = max(r.get("timestamp", cursor["sensor_readings_since"]) for r in sr_raw)
        save_cursor(cursor)
    else:
        print("No new data this cycle.")

    # Always rebuild + push the feed, even on a no-new-data cycle, so
    # `generated_at` reflects that the job is alive and running.
    history = read_processed_cri_from_firebase()
    feed = build_dashboard_feed(history)
    if feed is not None:
        write_results_to_firebase(feed)
        print(f"Pushed results feed: {len(feed['criTimeseries'])} points, "
              f"{len(feed['cri_summary_live'])} site-season summaries.")
    print("Done.")


def recompute_cwm_csd():
    """
    Once /processed_cri has enough windows per site (RQA needs >=120, DFA
    wants >=500-1000+), lift permutation_entropy / dfa / build_recurrence_matrix
    / rqa_measures / kendall_trend from CWM_CSD_v2.ipynb unchanged, point them
    at read_processed_cri_from_firebase() instead of the static CRI_Windowwise
    CSVs, and write the updated 3-tier verdict alongside the CRI feed.
    Run this daily/weekly (a separate, less frequent GitHub Actions schedule),
    not every batch tick.
    """
    print("recompute_cwm_csd(): plug in CWM_CSD_v2.ipynb's PE/DFA/RQA functions here, "
          "reading from read_processed_cri_from_firebase().")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-cwm-csd", action="store_true",
                         help="Run the heavier CWM-CSD step instead of the fast CRI batch step.")
    args = parser.parse_args()

    if args.recompute_cwm_csd:
        recompute_cwm_csd()
    else:
        run_batch()
