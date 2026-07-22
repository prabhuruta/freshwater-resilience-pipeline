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
       pipeline/site_registry.csv   <- known sites (site,lat,lon,match_radius_km,notes).
                                        TO ADD A FUTURE SITE: add one row to this file and
                                        push. No code changes needed. See the "Unregistered
                                        GPS Clusters" panel on the dashboard (or
                                        /unmatched_gps_log in Firebase) to find candidate new
                                        sites before naming them - a recurring cluster of
                                        unmatched readings is a real deployment worth adding.
  3. Firebase console -> Project settings -> Service accounts -> Generate new
     private key. Copy the ENTIRE JSON contents.
  4. GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository
     secret. Name: FIREBASE_SERVICE_ACCOUNT_KEY. Paste the JSON as the value.
     Never commit this key to the repo itself.
  5. Firebase console -> Realtime Database -> Rules -> add:
       {
         "rules": {
           "pipeline_state":   { ".read": "auth != null", ".write": "auth != null" },
           "processed_cri":    { ".read": "auth != null", ".write": "auth != null" },
           "unmatched_gps_log":{ ".read": "auth != null", ".write": "auth != null" },
           "results":          { ".read": true, ".write": "auth != null" }
         }
       }
     (results is public-read so the dashboard can fetch it with no credentials;
     everything else stays private, only this script's service account can touch it.
     Note: no .indexOn needed on sensor_readings/waterLogs — pagination now uses
     Firebase's built-in key ordering, not the record's own timestamp field, so
     no query index is required on those nodes.)
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
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# ── CONFIG ────────────────────────────────────────────────────────────────────
FIREBASE_DB_URL = "https://mywaterproject-e6489-default-rtdb.asia-southeast1.firebasedatabase.app/"
SERVICE_ACCOUNT_KEY_PATH = os.environ.get("FIREBASE_KEY_PATH", "serviceAccountKey.json")

CAUSAL_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "SEASONAL_CAUSAL_WEIGHTS_FAST")
NORMALISATION_BOUNDS_PATH = os.path.join(os.path.dirname(__file__), "Normalisation_Bounds_AllSites.csv")
# Condensed export of the validated CWM_CSD_Results_v2.csv (season, site, cri_mean,
# cri_std, verdict, n_windows) — the reference this script checks against to answer
# "has this site/season been through the full dissertation analysis before?"
VALIDATED_SITE_SEASONS_PATH = os.path.join(os.path.dirname(__file__), "validated_site_seasons.csv")

FEATURES = ["dissolvedo2", "orp", "ph", "tds", "temp", "turbidity", "bga", "chlorophyll"]
WINDOW_MINUTES = 5

# Minimum accumulated windows before a site-season is considered ready for a
# REAL CWM-CSD recompute (PE + DFA + RQA, not a shortcut). This mirrors what
# the dissertation's own methodology required: RQA needs >=120 windows for its
# rolling recurrence calculation, and the per-season DFA alpha in the original
# study was computed on 204-452 windows per site-season. 250 is a conservative
# floor sitting inside that established range - not a new number invented for
# the live pipeline.
MIN_WINDOWS_FOR_FULL_RECOMPUTE = 250

# Exact CWM-CSD parameters, ported unchanged from CWM_CSD_v2.ipynb (Cell 4 config),
# so a full recompute here matches the original dissertation methodology precisely.
PE_M = 4          # permutation entropy embedding dimension
PE_TAU = 1        # permutation entropy lag
DFA_SCALES = [8, 12, 16, 24, 32, 48, 64, 96, 128]
RQA_M = 3
RQA_TAU = 1
RQA_EPS = 0.20    # fraction of std
ROLLING_W = 60    # PE rolling window (5 hrs at 5-min resolution)
RQA_WIN = 120     # RQA window (10 hrs) - FIX 1 from the original notebook
EARLY_WARNING_THRESH = 1   # >=1 method triggers EARLY WARNING
DESTABILISING_THRESH = 2   # >=2 methods triggers DESTABILISING

# Firebase paths used for STATE (must survive between ephemeral runner instances)
FB_CURSOR_PATH = "pipeline_state/cursor"
FB_PROCESSED_CRI_PATH = "processed_cri"     # {site}/{season}/{window_key} -> {cri, t}
FB_RESULTS_PATH = "results/dashboard_feed"  # public-read, what the dashboard fetches


# ── SITE MATCHING (GPS-based, loaded from an editable registry file) ──────────
# Site coordinates now live in site_registry.csv, NOT hardcoded here — this is
# the mechanism for adding a future/unknown site: add one row to that CSV
# (site, lat, lon, match_radius_km, notes) and push it, no code changes needed.
#
# 9 of the original 11 sites are near-fixed logger deployments (radius ~0.3km);
# a few were boat-surveyed across wider areas and carry a larger radius — set
# whatever radius fits how that site was/will be surveyed.
SITE_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "site_registry.csv")

# How close together repeated UNMATCHED readings need to be (km) to count as
# the same candidate new site, when clustering for the discovery report below.
UNMATCHED_CLUSTER_RADIUS_KM = 1.0
# Minimum number of readings at a cluster before it's worth surfacing as a
# candidate new site (filters out one-off GPS glitches / passing-through pings).
UNMATCHED_MIN_CLUSTER_SIZE = 5


def load_site_registry():
    """
    Returns {site_name: (lat, lon, radius_km)}. Falls back to an empty registry
    (everything becomes UNMATCHED) with a clear warning if the file is missing,
    rather than crashing — a missing registry shouldn't take down the batch job.
    """
    if not os.path.exists(SITE_REGISTRY_PATH):
        print(f"WARNING: {SITE_REGISTRY_PATH} not found — no known sites loaded, "
              f"everything will be UNMATCHED. Add the file (site,lat,lon,match_radius_km,notes) to fix.")
        return {}
    reg = pd.read_csv(SITE_REGISTRY_PATH)
    return {row["site"]: (row["lat"], row["lon"], row["match_radius_km"]) for _, row in reg.iterrows()}


SITE_COORDS = load_site_registry()


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
    if not SITE_COORDS:
        return "UNMATCHED"
    best_site, best_dist, best_radius = None, float("inf"), None
    for site, (slat, slon, radius) in SITE_COORDS.items():
        d = haversine_km(lat, lon, slat, slon)
        if d < best_dist:
            best_site, best_dist, best_radius = site, d, radius
    return best_site if best_dist <= best_radius else "UNMATCHED"


def cluster_unmatched_points(points, radius_km=UNMATCHED_CLUSTER_RADIUS_KM):
    """
    Simple greedy clustering of (lat, lon) points that didn't match any known
    site — groups points within radius_km of each other so that a real,
    recurring new deployment location shows up as ONE candidate with a count,
    not dozens of individual unmatched pings. Not a rigorous clustering
    algorithm (no need for one at this scale) - just nearest-existing-cluster-
    centroid assignment, which is sufficient for a discovery/review report.
    """
    clusters = []  # each: {'lat_sum', 'lon_sum', 'n', 'points': [...]}
    for lat, lon in points:
        placed = False
        for c in clusters:
            c_lat, c_lon = c["lat_sum"] / c["n"], c["lon_sum"] / c["n"]
            if haversine_km(lat, lon, c_lat, c_lon) <= radius_km:
                c["lat_sum"] += lat
                c["lon_sum"] += lon
                c["n"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"lat_sum": lat, "lon_sum": lon, "n": 1})
    return [
        {"centroid_lat": round(c["lat_sum"] / c["n"], 6),
         "centroid_lon": round(c["lon_sum"] / c["n"], 6),
         "n_readings": c["n"]}
        for c in clusters if c["n"] >= UNMATCHED_MIN_CLUSTER_SIZE
    ]


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


def fetch_new_records(node, since_key):
    """
    Pulls records newer than `since_key` using Firebase's PUSH KEY ordering,
    NOT the record's own timestamp field. This matters because waterLogs.time
    is stored as "DD-MM-YYYY HH:MM:SS" (day-first) — that format does NOT
    sort chronologically as a string (e.g. "10-04-2026" < "18-05-2025"
    alphabetically, despite being a year later), so a query filtered by that
    field was returning essentially arbitrary results with respect to real
    time. Push keys are chronologically ordered by Firebase itself regardless
    of what's inside the record, so ordering/paginating by key sidesteps the
    problem entirely.

    Returns a dict of {push_key: record}, in chronological order.
    """
    db = get_db()
    ref = db.reference(node)
    if since_key:
        # start_at is inclusive, so the previously-seen key comes back too —
        # the caller drops it before processing.
        snapshot = ref.order_by_key().start_at(since_key).get() or {}
    else:
        # First run ever for this node: no key to start from. Pulling the
        # entire node once is the correct (if occasionally large) one-time
        # cost — there is no reliable way to seed "last 24h" from push keys
        # alone, and seeding from the malformed time field would reintroduce
        # the exact bug this function exists to avoid.
        snapshot = ref.order_by_key().get() or {}
    return snapshot


def load_cursor():
    db = get_db()
    cursor = db.reference(FB_CURSOR_PATH).get()
    if cursor:
        return cursor
    return {"waterLogs_last_key": None, "sensor_readings_last_key": None}


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


def extract_unmatched_points(wl_df):
    """Pulls (lat, lon) for every row that didn't match a known site, for logging/clustering."""
    if wl_df.empty or "site" not in wl_df.columns:
        return []
    unmatched = wl_df[(wl_df["site"] == "UNMATCHED") & (wl_df["lat"] != 0) & (wl_df["lon"] != 0)]
    return list(zip(unmatched["lat"], unmatched["lon"]))


def log_unmatched_points_to_firebase(points):
    """
    Appends unmatched GPS points to a Firebase log so they can be reviewed later —
    without this, an unmatched reading is just silently dropped and a real new
    deployment location could go unnoticed for a long time.
    """
    if not points:
        return
    db = get_db()
    updates = {}
    now_key = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    for i, (lat, lon) in enumerate(points):
        updates[f"unmatched_gps_log/{now_key}_{i}"] = {"lat": lat, "lon": lon}
    db.reference().update(updates)


def read_unmatched_log_and_cluster():
    """
    Pulls the full unmatched-GPS log back out and clusters it into candidate new
    sites. This is what powers the dashboard's "Unregistered GPS Clusters" panel —
    a recurring cluster of unmatched readings is a strong signal a new site
    should be added to site_registry.csv.
    """
    db = get_db()
    raw = db.reference("unmatched_gps_log").get() or {}
    points = [(rec["lat"], rec["lon"]) for rec in raw.values() if "lat" in rec and "lon" in rec]
    return cluster_unmatched_points(points)


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
    """
    Expects columns: site, driver, ensemble_weight, season — this is the
    exact schema Ensemble_Causal_Weights_Fast.ipynb saves to
    Global_Ensemble_Causal_Weights_{season}.csv (site_feat dataframe,
    groupby(['site','driver'])['ensemble_weight']).
    """
    path = os.path.join(CAUSAL_WEIGHTS_DIR, season, f"Global_Ensemble_Causal_Weights_{season}.csv")
    if not os.path.exists(path):
        print(f"WARNING: no frozen weights for {season} at {path} — using equal weights")
        return {f: 1 / len(FEATURES) for f in FEATURES}
    try:
        wdf = pd.read_csv(path)
        wdf.columns = wdf.columns.str.lower()
        if not {"site", "driver", "ensemble_weight"}.issubset(wdf.columns):
            print(f"WARNING: {path} missing expected columns "
                  f"{{'site','driver','ensemble_weight'}} - {set(wdf.columns)} — using equal weights")
            return {f: 1 / len(FEATURES) for f in FEATURES}
        site_rows = wdf[wdf["site"].str.lower() == site.lower()]
        if site_rows.empty:
            print(f"WARNING: site '{site}' not in frozen weights — using equal weights")
            return {f: 1 / len(FEATURES) for f in FEATURES}
        return dict(zip(site_rows["driver"], site_rows["ensemble_weight"]))
    except Exception as e:
        print(f"WARNING: failed to load causal weights ({e}) — using equal weights")
        return {f: 1 / len(FEATURES) for f in FEATURES}


def load_validated_baseline(season, site):
    """
    Checks whether this (site, season) went through the full dissertation
    analysis (CWM_CSD_Results_v2.csv). Returns a dict with the historical
    cri_mean/cri_std/verdict/n_windows if found, or None if this is genuinely
    new territory — a site or season combination never analysed before.
    """
    if not os.path.exists(VALIDATED_SITE_SEASONS_PATH):
        return None
    try:
        vdf = pd.read_csv(VALIDATED_SITE_SEASONS_PATH)
        vdf.columns = vdf.columns.str.lower()
        match = vdf[(vdf["site"].str.lower() == site.lower()) & (vdf["season"].str.lower() == season.lower())]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            "cri_mean": float(row["cri_mean"]), "cri_std": float(row["cri_std"]),
            "verdict": row["verdict"], "n_windows": int(row["n_windows"]),
        }
    except Exception as e:
        print(f"WARNING: failed to load validated baseline ({e}) — treating as no past record.")
        return None


def load_normalisation_bounds(site):
    """
    Expects columns: season, site, feature, site_min, site_max, site_range
    (this is the exact schema CRI_Computation_AllSites.ipynb saves, and what
    Normalisation_Bounds_AllSites.csv should have). Returns None on any
    mismatch rather than raising — a missing/malformed bounds file should
    degrade to the noisier fallback in compute_cri(), not crash the run.
    """
    if not os.path.exists(NORMALISATION_BOUNDS_PATH):
        return None
    try:
        bdf = pd.read_csv(NORMALISATION_BOUNDS_PATH)
        bdf.columns = bdf.columns.str.lower()
        required = {"site", "feature", "site_min", "site_max"}
        if not required.issubset(bdf.columns):
            print(f"WARNING: {NORMALISATION_BOUNDS_PATH} is missing expected columns "
                  f"{required - set(bdf.columns)} — ignoring bounds file for this run.")
            return None
        site_rows = bdf[bdf["site"].str.lower() == site.lower()]
        if site_rows.empty:
            return None
        return {row["feature"]: (row["site_min"], row["site_max"]) for _, row in site_rows.iterrows()}
    except Exception as e:
        print(f"WARNING: failed to load normalisation bounds ({e}) — ignoring bounds file for this run.")
        return None


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
def build_dashboard_feed(all_cri_history, unmatched_clusters=None):
    if all_cri_history.empty:
        print("No CRI history yet — nothing to write.")
        return None

    feed = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "criTimeseries": [], "cri_summary_live": [],
        "unmatched_clusters": unmatched_clusters or [],
        "known_sites": list(SITE_COORDS.keys()),
    }

    for (site, season), g in all_cri_history.groupby(["site", "season"]):
        for _, row in g.iterrows():
            feed["criTimeseries"].append({"season": season, "site": site, "t": row["t"], "cri": row["cri"]})
        cri_vals = g["cri"].dropna()
        if len(cri_vals):
            live_mean = round(float(cri_vals.mean()), 4)
            live_std_raw = cri_vals.std()
            live_std = round(float(live_std_raw), 4) if pd.notna(live_std_raw) else None
            n_windows = int(len(cri_vals))

            baseline = load_validated_baseline(season, site)
            entry = {
                "site": site, "season": season,
                "cri_mean": live_mean, "cri_std": live_std,
                "n_windows": n_windows,
                "last_updated": str(g["t"].max()),
                # Data-sufficiency check against the SAME thresholds the real
                # methodology already requires (RQA_WIN=120, DFA per-season used
                # 204-452 windows in the dissertation) - not a new metric, just
                # reporting readiness against established requirements.
                "ready_for_full_recompute": n_windows >= MIN_WINDOWS_FOR_FULL_RECOMPUTE,
                "windows_needed": MIN_WINDOWS_FOR_FULL_RECOMPUTE,
            }

            if baseline is not None:
                # A real prior record exists from the dissertation analysis -
                # shown as-is as reference context. This is the actual, fully
                # computed CWM-CSD verdict (PE + DFA + RQA), not a shortcut.
                entry["has_historical_baseline"] = True
                entry["baseline_cri_mean"] = baseline["cri_mean"]
                entry["baseline_verdict"] = baseline["verdict"]
                entry["baseline_n_windows"] = baseline["n_windows"]
            else:
                # No past record for this exact (site, season) combination —
                # could be a brand-new site, or a known site reporting in a
                # season it's never been analysed for before. No verdict is
                # fabricated here; a real one only appears once
                # recompute_cwm_csd() has run PE/DFA/RQA on enough accumulated
                # data (see that function below).
                entry["has_historical_baseline"] = False
                entry["baseline_note"] = "No past records found for this site/season."

            feed["cri_summary_live"].append(entry)
    return feed


# ── 6. MAIN ────────────────────────────────────────────────────────────────────
def run_batch():
    if not SITE_COORDS:
        # The site registry failed to load - every reading would be marked
        # UNMATCHED and silently discarded if we proceeded. Stop here WITHOUT
        # touching the cursor, so once site_registry.csv is actually present,
        # the next run re-fetches this exact same data instead of having
        # silently lost it (which is what happened before this guard existed:
        # the cursor advanced past a whole batch that got discarded).
        print(f"FATAL: site_registry.csv not found or empty at {SITE_REGISTRY_PATH} — "
              f"refusing to process or advance the cursor this run, to avoid silently "
              f"discarding real data. Commit site_registry.csv to the repo and re-run.")
        return

    cursor = load_cursor()

    wl_snapshot = fetch_new_records("waterLogs", cursor.get("waterLogs_last_key"))
    sr_snapshot = fetch_new_records("sensor_readings", cursor.get("sensor_readings_last_key"))

    # start_at() is inclusive, so the previously-processed boundary key comes
    # back again — drop it here so it isn't processed twice.
    if cursor.get("waterLogs_last_key") in wl_snapshot:
        wl_snapshot.pop(cursor["waterLogs_last_key"])
    if cursor.get("sensor_readings_last_key") in sr_snapshot:
        sr_snapshot.pop(cursor["sensor_readings_last_key"])

    wl_raw = list(wl_snapshot.values())
    sr_raw = list(sr_snapshot.values())
    print(f"Pulled {len(wl_raw)} waterLogs, {len(sr_raw)} sensor_readings since last run.")

    if wl_raw or sr_raw:
        wl_df, sr_df = load_and_match(wl_raw, sr_raw)

        unmatched_points = extract_unmatched_points(wl_df)
        if unmatched_points:
            log_unmatched_points_to_firebase(unmatched_points)
            print(f"Logged {len(unmatched_points)} unmatched GPS points to /unmatched_gps_log "
                  f"for future site-discovery review.")

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

        # Firebase push keys sort correctly as strings by construction
        # (chronologically monotonic), so max(keys) is a safe, correct cursor —
        # unlike max() over the record's own malformed date-string field.
        if wl_snapshot:
            cursor["waterLogs_last_key"] = max(wl_snapshot.keys())
        if sr_snapshot:
            cursor["sensor_readings_last_key"] = max(sr_snapshot.keys())
        save_cursor(cursor)
    else:
        print("No new data this cycle.")

    # Always rebuild + push the feed, even on a no-new-data cycle, so
    # `generated_at` reflects that the job is alive and running.
    history = read_processed_cri_from_firebase()
    unmatched_clusters = read_unmatched_log_and_cluster()
    if unmatched_clusters:
        print(f"NOTE: {len(unmatched_clusters)} candidate new site(s) detected from unmatched GPS "
              f"readings (each with >= {UNMATCHED_MIN_CLUSTER_SIZE} readings). See the dashboard's "
              f"'Unregistered GPS Clusters' panel, or /unmatched_gps_log in Firebase, to review and "
              f"add them to site_registry.csv if they're a real new deployment.")
    feed = build_dashboard_feed(history, unmatched_clusters)
    if feed is not None:
        write_results_to_firebase(feed)
        print(f"Pushed results feed: {len(feed['criTimeseries'])} points, "
              f"{len(feed['cri_summary_live'])} site-season summaries, "
              f"{len(unmatched_clusters)} unmatched clusters.")
    print("Done.")


def permutation_entropy(x, m=PE_M, tau=PE_TAU, normalise=True):
    """Ported unchanged from CWM_CSD_v2.ipynb."""
    x = np.array(x, dtype=float)
    if len(x) < m * tau + 1:
        return np.nan
    N = len(x) - (m - 1) * tau
    counts = {}
    for i in range(N):
        sub = x[i: i + m * tau: tau]
        perm = tuple(np.argsort(sub))
        counts[perm] = counts.get(perm, 0) + 1
    probs = np.array(list(counts.values())) / N
    pe = -np.sum(probs * np.log(probs + 1e-12))
    if normalise:
        pe = pe / np.log(math.factorial(m))
    return round(pe, 6)


def rolling_pe(x, m=PE_M, tau=PE_TAU, window=ROLLING_W):
    x, n = np.array(x, dtype=float), len(x)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = permutation_entropy(x[i - window:i], m, tau)
    return out


def cwm_pe(pe_series, cdr):
    """CWM-PE = (1-PE) x CDR — rising = increasing regularity = CSD."""
    return (1 - pe_series) * cdr


def dfa(x, scales=DFA_SCALES):
    """DFA scaling exponent alpha. alpha>0.5 = persistent = CSD building. Ported unchanged."""
    x = np.array(x, dtype=float)
    x = x - x.mean()
    y = np.cumsum(x)
    n = len(y)
    F_vals, valid_scales = [], []
    for s in scales:
        if s < 4 or s > n // 2:
            continue
        n_boxes = n // s
        if n_boxes < 2:
            continue
        rms = []
        for b in range(n_boxes):
            seg = y[b * s:(b + 1) * s]
            t = np.arange(len(seg))
            coef = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(coef, t)) ** 2)))
        F_vals.append(np.mean(rms))
        valid_scales.append(s)
    if len(valid_scales) < 4:
        return np.nan
    alpha, _ = np.polyfit(np.log(valid_scales), np.log(F_vals), 1)
    return round(alpha, 6)


def build_recurrence_matrix(x, m=RQA_M, tau=RQA_TAU, eps=None):
    x = np.array(x, dtype=float)
    N = len(x) - (m - 1) * tau
    emb = np.array([x[i:i + (m - 1) * tau + 1:tau] for i in range(N)])
    D = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(axis=2))
    if eps is None:
        eps = RQA_EPS * x.std()
    return (D <= eps).astype(np.uint8)


def rqa_measures(R):
    N = R.shape[0]
    np.fill_diagonal(R, 0)
    total_rp = R.sum()
    if total_rp == 0:
        return 0.0, 0.0, 0.0
    det_rp = 0
    for d in range(1, N):
        diag = np.diag(R, d)
        L, inL = 0, False
        for v in diag:
            if v: inL, L = True, L + 1
            elif inL:
                if L >= 2: det_rp += L
                inL, L = False, 0
        if inL and L >= 2: det_rp += L
    lam_rp, tt_lens = 0, []
    for col in range(N):
        L, inL = 0, False
        for v in R[:, col]:
            if v: inL, L = True, L + 1
            elif inL:
                if L >= 2: lam_rp += L; tt_lens.append(L)
                inL, L = False, 0
        if inL and L >= 2: lam_rp += L; tt_lens.append(L)
    det = det_rp / total_rp if total_rp > 0 else 0.0
    lam = lam_rp / total_rp if total_rp > 0 else 0.0
    tt = np.mean(tt_lens) if tt_lens else 0.0
    return round(det, 6), round(lam, 6), round(tt, 6)


def rolling_rqa(x, m=RQA_M, tau=RQA_TAU, eps_frac=RQA_EPS, window=RQA_WIN, cdr=1.5):
    x = np.array(x, dtype=float)
    n = len(x)
    eps = eps_frac * x.std() * (1 / cdr)
    det_out = np.full(n, np.nan)
    lam_out = np.full(n, np.nan)
    for i in range(window, n):
        seg = x[i - window:i]
        try:
            R = build_recurrence_matrix(seg, m, tau, eps)
            d, l, t = rqa_measures(R)
            det_out[i] = d
            lam_out[i] = l
        except Exception:
            pass
    return det_out, lam_out


def kendall_trend(series):
    s = np.array(series, dtype=float)
    idx = np.where(~np.isnan(s))[0]
    if len(idx) < 10:
        return np.nan, np.nan
    tau, p = kendalltau(idx, s[idx])
    return round(tau, 4), round(p, 4)


def compute_cdr(weights_dict):
    """Same definition used everywhere else in this pipeline: top weight / mean of the rest."""
    vals = sorted(weights_dict.values(), reverse=True)
    if len(vals) < 2:
        return 1.0
    top = vals[0]
    rest_mean = np.mean(vals[1:]) if np.mean(vals[1:]) > 1e-9 else 1e-9
    return round(top / rest_mean, 4)


def run_full_cwm_csd(x, cdr):
    """
    Runs the actual, unmodified CWM-CSD verdict logic (PE + DFA + RQA + majority
    vote) on a CRI series, given that site-season's CDR. This is the same
    computation as CWM_CSD_v2.ipynb, not an approximation - only called once a
    site-season has both enough windows (MIN_WINDOWS_FOR_FULL_RECOMPUTE) and a
    real CDR from actual causal weights (not equal-weight fallback).
    """
    pe_series = rolling_pe(x)
    cwmpe_series = cwm_pe(pe_series, cdr)
    tau_cwmpe, p_cwmpe = kendall_trend(cwmpe_series)
    csd_pe = bool(not np.isnan(tau_cwmpe) and tau_cwmpe > 0 and p_cwmpe < 0.05)

    alpha = dfa(x)
    csd_dfa = bool(not np.isnan(alpha) and alpha > 0.65)

    det_series, lam_series = rolling_rqa(x, cdr=cdr)
    tau_det, p_det = kendall_trend(det_series)
    tau_lam, p_lam = kendall_trend(lam_series)
    csd_rqa = bool((not np.isnan(tau_det) and tau_det > 0 and p_det < 0.05) or
                    (not np.isnan(tau_lam) and tau_lam > 0 and p_lam < 0.05))

    n_agree = int(csd_pe) + int(csd_dfa) + int(csd_rqa)
    if n_agree >= DESTABILISING_THRESH:
        verdict = "DESTABILISING"
    elif n_agree >= EARLY_WARNING_THRESH:
        verdict = "EARLY WARNING"
    else:
        verdict = "STABLE"

    return {
        "tau_cwmpe": tau_cwmpe, "p_cwmpe": p_cwmpe, "csd_pe": csd_pe,
        "dfa_alpha": alpha, "csd_dfa": csd_dfa,
        "tau_det": tau_det, "p_det": p_det, "tau_lam": tau_lam, "p_lam": p_lam, "csd_rqa": csd_rqa,
        "n_methods": n_agree, "verdict": verdict,
    }


def recompute_cwm_csd():
    """
    Runs the REAL CWM-CSD recompute (not a shortcut) for every site-season in
    /processed_cri that is actually ready for it. "Ready" means both:
      1. Enough accumulated windows (>= MIN_WINDOWS_FOR_FULL_RECOMPUTE, itself
         set from the same RQA/DFA data requirements the dissertation used).
      2. Real causal weights available for that site (from Ensemble Causal
         Weights) - NOT the equal-weight fallback, since CDR is meaningless
         without genuine causal discovery behind it.
    A site-season missing either precondition is reported as not-yet-ready,
    with a clear reason, rather than silently skipped or approximated.
    Run this on a separate, slower schedule (daily/weekly) - not every batch
    tick, since DFA/RQA are computationally heavier than the CRI step.
    """
    history = read_processed_cri_from_firebase()
    if history.empty:
        print("No processed CRI history yet — nothing to recompute.")
        return

    results = {}
    for (site, season), g in history.groupby(["site", "season"]):
        g = g.sort_values("t")
        x = g["cri"].dropna().values
        n_windows = len(x)

        if n_windows < MIN_WINDOWS_FOR_FULL_RECOMPUTE:
            print(f"  [{site} / {season}] NOT READY — {n_windows}/{MIN_WINDOWS_FOR_FULL_RECOMPUTE} "
                  f"windows collected. Waiting for more data.")
            continue

        weights = load_causal_weights(season, site)
        is_equal_weight_fallback = len(set(round(w, 6) for w in weights.values())) <= 1
        if is_equal_weight_fallback:
            print(f"  [{site} / {season}] NOT READY — {n_windows} windows collected (sufficient), "
                  f"but no real causal weights found for this site (Ensemble Causal Weights "
                  f"hasn't been run for it yet). Run that offline first, then this will proceed.")
            continue

        cdr = compute_cdr(weights)
        verdict_data = run_full_cwm_csd(x, cdr)
        verdict_data.update({"site": site, "season": season, "cdr": cdr, "n_windows": n_windows,
                              "computed_at": datetime.utcnow().isoformat() + "Z"})
        results[f"{site}|{season}"] = verdict_data
        print(f"  [{site} / {season}] RECOMPUTED — verdict={verdict_data['verdict']} "
              f"(n_methods={verdict_data['n_methods']}/3, n_windows={n_windows})")

    if results:
        db = get_db()
        updates = {f"results/live_verdicts/{k.replace('|', '/')}": v for k, v in results.items()}
        db.reference().update(updates)
        print(f"Wrote {len(results)} real recomputed verdict(s) to /results/live_verdicts")
    else:
        print("No site-seasons were ready for a full recompute this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-cwm-csd", action="store_true",
                         help="Run the heavier CWM-CSD step instead of the fast CRI batch step.")
    args = parser.parse_args()

    if args.recompute_cwm_csd:
        recompute_cwm_csd()
    else:
        run_batch()
