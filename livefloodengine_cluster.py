"""
FloodLink – Live Flood Risk Evaluator (RAW + Linear)
Evaluates high-risk features from Citiesglobal.csv using NOAA GFS forecasts.

Now includes:
- NOAA GFS 0.25° grids (bulk download, single pass for all points)
- Configurable forecast horizon (3h, 6h, 12h, etc.)
- Linear, unit-aware multipliers (rain unbounded; soil clipped)
- RAW score only (no compression)
- Level-transition alerts only (Medium↔High, High↔Extreme; downgrades toggle)
- Single-file comparison (alerts_comparison_cluster.json)
- Rich Tweet Tracker (tweeted_alerts_cluster.json)
- Region-level grouping: 1 tweet per (country, region) per run
- Repost behaviour restored:
    * Upgrades & downgrades only if there was a previous tweet
    * Upgrades & downgrades quote the previous tweet (threading)
"""

import os
import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

import pandas as pd
import requests
import tweepy
import unicodedata
from requests.exceptions import RequestException, ReadTimeout, ConnectionError

import numpy as np
import pygrib

# -------------------------------
# CONFIGURATION
# -------------------------------
CSV_PATH = "cities15000.csv"
COMPARISON_PATH = "alerts_comparison_cluster.json"   # single source of truth
TWEET_LOG_PATH = "tweeted_alerts_cluster.json"       # map-ready tweet history

SLEEP_BETWEEN_CALLS = 0.1         # kept for compatibility (not used for bulk NOAA)
COMPARISON_HISTORY = 5            # or 10
TIMEZONE = "Europe/Madrid"
MAX_RETRIES = 1                   # per NOAA file download
TIMEOUT = 30                      # request timeout (s) per NOAA download
FORECAST_HOURS = 6  # e.g. 3, 6, 12 ...; with 0.25° this is hourly steps

# --- Twitter config ---
TWITTER_ENABLED = os.getenv("TWITTER_ENABLED", "false").lower() == "true"
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_SECRET = os.getenv("TWITTER_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
MIN_SECONDS_BETWEEN_TWEETS = 30

# -------------------------------
# TUNABLE CONSTANTS (units!)
# -------------------------------
RISK_THRESHOLD = 8.4         # baseline FRisk cutoff from GIS layer (tweakable)

RAIN_UNIT_MM   = 100.0       # 100 mm → 1.0× rain multiplier
SOIL_MIN_MULT  = 0.95        # soil=0 -> 0.95×
SOIL_MAX_MULT  = 1.8         # soil=1 -> 1.8×
RAIN_CUTOFF_MM = 0.0         # set 0.5 to ignore drizzle; 0.0 keeps strict linearity

# RAW alert bands (tune later or learn from rolling percentiles)
RAW_LOW_MAX   = 7.0          # 0..7   -> Low
RAW_MED_MAX   = 12.0         # 7..12  -> Medium
RAW_HIGH_MAX  = 24.0         # 12..24 -> High
# >24 -> Extreme

# -------------------------------
# ALERT TRANSITION POLICY
# -------------------------------
COOLDOWN_HOURS = 6  # downgrade tweets blocked within this window after last tweet

TWEET_LEVELS = ["Medium", "High", "Extreme"]   # which levels are tweet-worthy at all
ALERT_ON_UPGRADES   = True                     # Medium→High, High→Extreme
ALERT_ON_DOWNGRADES = True                     # High→Medium, Extreme→High (and lower)

LEVELS = ["None", "Low", "Medium", "High", "Extreme"]

# -------------------------------
# NOAA GFS CONFIG
# -------------------------------
# Use 0.25° GFS with hourly forecast steps
GFS_RES = "0p25"   # was "0p50"

# Variables to extract from GFS
VARIABLES = ["APCP", "SOILW"]

# Level spec for the GRIB filter
LEVELS_DICT = {
    "APCP": "surface",
    "SOILW": "0-0.1 m below ground",  # depth layer for top soil
}

# -------------------------------
# NOAA GFS HELPERS
# -------------------------------
def get_latest_cycle():
    """
    Determine latest available 6-hourly GFS cycle and the previous one for fallback.
    """
    now = datetime.utcnow()
    date = now.strftime("%Y%m%d")
    cycle_hour = (now.hour // 6) * 6
    cycle = f"{cycle_hour:02d}"

    if cycle_hour == 0:
        prev_date = (now - timedelta(days=1)).strftime("%Y%m%d")
        prev_cycle = "18"
        return date, cycle, prev_date, prev_cycle

    prev_cycle = f"{cycle_hour - 6:02d}"
    return date, cycle, date, prev_cycle


def get_forecast_steps_for_now(forecast_hours: int, cycle_dt_utc, now_utc=None):
    """
    Return forecast steps that cover the NEXT `forecast_hours` from *now*,
    relative to the chosen cycle datetime.

    For 0p25: hourly steps (f001, f002, ...)
    For 0p50/1p00: 3-hourly steps (f003, f006, ...)
    """
    if now_utc is None:
        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))

    # How many hours since the cycle started?
    lead_hours = (now_utc - cycle_dt_utc).total_seconds() / 3600.0

    if GFS_RES in ("0p50", "1p00"):
        step = 3
        # align start to next available 3-hour step
        start_fhr = max(step, int(np.ceil(lead_hours / step)) * step)
        # cover roughly `forecast_hours` ahead from that start
        end_hour = start_fhr + (forecast_hours - 1)
        end_fhr = int(np.ceil(end_hour / step)) * step
        return list(range(start_fhr, end_fhr + 1, step))
    else:
        # hourly
        start_fhr = max(1, int(np.ceil(lead_hours)))
        end_fhr = start_fhr + forecast_hours - 1
        return list(range(start_fhr, end_fhr + 1))



def download_gfs_file(date, cycle, fhr):
    base_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{GFS_RES}.pl"

    # File naming differs between 0p50 "full" grid and 0.25° standard grid
    if GFS_RES == "0p50":
        file_name = f"gfs.t{cycle}z.pgrb2full.0p50.f{fhr:03d}"
    else:
        file_name = f"gfs.t{cycle}z.pgrb2.{GFS_RES}.f{fhr:03d}"

    params = {
        "dir": f"/gfs.{date}/{cycle}/atmos",
        "file": file_name,
    }

    # Global domain
    params.update({
        "leftlon": 0,
        "rightlon": 360,
        "toplat": 90,
        "bottomlat": -90,
    })

    for var in VARIABLES:
        params[f"var_{var}"] = "on"
        lev = LEVELS_DICT[var].replace(" ", "_")
        params[f"lev_{lev}"] = "on"

    full_url = base_url + "?" + "&".join([f"{k}={v}" for k, v in params.items()])
    print(f"Attempting download with URL: {full_url}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(base_url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            file_path = f"gfs_{cycle}_f{fhr:03d}.grb2"
            with open(file_path, "wb") as f:
                f.write(r.content)
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"Downloaded {file_path} (size: {size_mb:.2f} MB)")
            return file_path
        except Exception as e:
            print(f"Download failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(2 ** attempt)

    return None

def load_gfs_grids(forecast_hours):
    """
    Download and load GFS grib2 files for APCP and SOILW into memory once.

    Returns:
        grids: dict[var] -> np.array [time, lat, lon]
        lats, lons: 2D arrays
        times: list of datetime (naive UTC)
    """
    date, cycle, prev_date, prev_cycle = get_latest_cycle()

    now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))

    cycle_dt = datetime.strptime(date + cycle, "%Y%m%d%H").replace(tzinfo=ZoneInfo("UTC"))
    prev_cycle_dt = datetime.strptime(prev_date + prev_cycle, "%Y%m%d%H").replace(tzinfo=ZoneInfo("UTC"))

    # Compute window for latest cycle
    window_steps = get_forecast_steps_for_now(forecast_hours, cycle_dt, now_utc)

    if not window_steps:
        print("❌ window_steps is empty (unexpected).")
        return None, None, None, []

    # Baseline step to compute the first increment correctly
    baseline_step = None
    if GFS_RES in ("0p50", "1p00"):
        step = 3
        if window_steps[0] > step:
            baseline_step = window_steps[0] - step
    else:
        if window_steps[0] > 1:
            baseline_step = window_steps[0] - 1

    # We will download baseline first (if exists), then the window steps
    steps_to_download = ([baseline_step] if baseline_step is not None else []) + window_steps

    # Probe the first needed *download* to choose cycle (baseline if present, else first window step)
    probe_fhr = steps_to_download[0]
    probe_file = download_gfs_file(date, cycle, probe_fhr)

    use_date, use_cycle, use_cycle_dt = date, cycle, cycle_dt
    predownloaded = {}

    if probe_file is not None:
        predownloaded[probe_fhr] = probe_file
    else:
        probe_file_prev = download_gfs_file(prev_date, prev_cycle, probe_fhr)
        if probe_file_prev is None:
            print(f"❌ Unable to download probe step f{probe_fhr:03d} from latest or previous cycle.")
            return None, None, None, []

        use_date, use_cycle, use_cycle_dt = prev_date, prev_cycle, prev_cycle_dt

        # Recompute window/baseline for fallback cycle (IMPORTANT)
        window_steps = get_forecast_steps_for_now(forecast_hours, use_cycle_dt, now_utc)

        if not window_steps:
            print("❌ window_steps is empty (unexpected) after fallback recompute.")
            return None, None, None, []

        baseline_step = None
        if GFS_RES in ("0p50", "1p00"):
            step = 3
            if window_steps[0] > step:
                baseline_step = window_steps[0] - step
        else:
            if window_steps[0] > 1:
                baseline_step = window_steps[0] - 1

        steps_to_download = ([baseline_step] if baseline_step is not None else []) + window_steps

        # If the probe hour changed after recompute, discard the old probe and download the correct one
        new_probe_fhr = steps_to_download[0]
        if new_probe_fhr != probe_fhr:
            try:
                os.remove(probe_file_prev)
            except Exception:
                pass
            probe_file_prev = download_gfs_file(use_date, use_cycle, new_probe_fhr)
            if probe_file_prev is None:
                print(f"❌ Unable to download recomputed probe step f{new_probe_fhr:03d} for fallback cycle.")
                return None, None, None, []
            probe_fhr = new_probe_fhr

        predownloaded[probe_fhr] = probe_file_prev

    print(f"🛰️ Using GFS cycle: {use_date} {use_cycle}Z | window: f{window_steps[0]:03d}..f{window_steps[-1]:03d}"
          + (f" | baseline: f{baseline_step:03d}" if baseline_step is not None else ""))

    grids = {var: [] for var in VARIABLES}
    grids["_meta"] = {
        "window_steps": window_steps,
        "baseline_step": baseline_step,
        "has_baseline": baseline_step is not None
    }

    times = []
    lats, lons = None, None
    soil_ok = True

    for fhr in steps_to_download:
        file = predownloaded.get(fhr)
        if file is None:
            file = download_gfs_file(use_date, use_cycle, fhr)
            if file is None:
                print(f"⚠️ Skipping f{fhr:03d} – unable to download from chosen cycle.")
                continue

        grb = pygrib.open(file)
         
        # APCP / Total precipitation (prefer shortName="tp")
        try:
            # 1) Prefer tp directly (most reliable across filtered outputs)
            tp_msgs = grb.select(shortName="tp")

            # Prefer the message whose endStep matches this fhr
            apcp_msg = next(
                (m for m in tp_msgs if int(getattr(m, "endStep", -1)) == fhr),
                tp_msgs[0]
            )

        except Exception:
            # 2) Fallback: try by name (some files expose it this way)
            try:
                tp_msgs = grb.select(name="Total Precipitation")
                apcp_msg = next(
                    (m for m in tp_msgs if int(getattr(m, "endStep", -1)) == fhr),
                    tp_msgs[0]
                )
            except Exception as e:
                print(f"⚠️ No APCP/tp in {file}: {e}")
                grb.close()
                os.remove(file)
                continue


        grids["APCP"].append(apcp_msg.values)
        if lats is None:
            lats, lons = apcp_msg.latlons()

        # SOILW: keep aligned with APCP/times (include baseline if we included it)
        if soil_ok:
            try:
                soil_msg = grb.select(shortName="soilw")[0]
                grids["SOILW"].append(soil_msg.values)
            except Exception:
                soil_ok = False
                grids["SOILW"] = []
                print(f"⚠️ SOILW missing at f{fhr:03d}; disabling soil for this run.")

        # Build valid times ourselves (naive UTC)
        times.append((use_cycle_dt + timedelta(hours=fhr)).replace(tzinfo=None))

        grb.close()
        os.remove(file)

    grids["APCP"] = np.stack(grids["APCP"]) if grids["APCP"] else None
    if soil_ok and grids["SOILW"]:
        grids["SOILW"] = np.stack(grids["SOILW"])
    else:
        grids["SOILW"] = None

    return grids, lats, lons, times


def precompute_city_indices(lats, lons, df):
    """
    Precompute nearest grid indices (ilat, ilon) for each city.
    """
    lat_axis = lats[:, 0]
    lon_axis = lons[0, :]

    idx_map = {}
    for _, row in df.iterrows():
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])

        # Handle 0–360 vs -180–180 if needed
        if np.any(lon_axis > 180):
            if lon < 0:
                lon = lon + 360.0

        ilat = int(np.argmin(np.abs(lat_axis - lat)))
        ilon = int(np.argmin(np.abs(lon_axis - lon)))

        idx_map[row["JOIN_ID"]] = (ilat, ilon)

    return idx_map

#FOR COOLDOWN DOWNGRADES
def parse_utc_z(ts: str):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ZoneInfo("UTC"))
    except Exception:
        return None

def within_cooldown(last_entry: dict, now_utc: datetime) -> bool:
    # Cooldown is based ONLY on the last time we actually tweeted.
    last_ts = parse_utc_z(last_entry.get("last_tweeted_at"))
    if not last_ts:
        return False
    return (now_utc - last_ts) < timedelta(hours=COOLDOWN_HOURS)

def level_index(lvl: str) -> int:
    try:
        return LEVELS.index(lvl)
    except Exception:
        return 0

def get_tweeted_level(entry: dict) -> str:
    # tweeted_level = last level that was ACTUALLY tweeted (cooldown anchor for severity)
    return entry.get("tweeted_level") or entry.get("risk_level", "None")

def mark_pending_downgrade(entry: dict, target_level: str, now_utc: datetime, is_region: bool = False):
    entry["pending_downgrade"] = True
    entry["pending_target_level"] = target_level
    entry["pending_set_at"] = now_utc.isoformat().replace("+00:00", "Z")
    entry["pending_is_region"] = bool(is_region)

def clear_pending(entry: dict):
    entry.pop("pending_downgrade", None)
    entry.pop("pending_target_level", None)
    entry.pop("pending_set_at", None)
    entry.pop("pending_is_region", None)


#CLEANER TEXT
def clean_text(val) -> str:
    """Fix common mojibake like 'ParanÃ¡' -> 'Paraná' and normalize accents."""
    if val is None:
        return ""

    # Handles np.nan, pd.NA, NaT, etc.
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass

    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return ""

    if any(x in s for x in ("Ã", "Â", "�")):
        try:
            repaired = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if repaired:
                s = repaired
        except Exception:
            pass

    return unicodedata.normalize("NFC", s)


def normalize_country(country_val) -> str:
    """
    Fix country names that come as 'Congo, The Democratic Republic of the'
    into 'The Democratic Republic of the Congo'
    Also handles 'Bahamas, The' -> 'The Bahamas'
    """
    c = clean_text(country_val)
    if not c:
        return ""

    if "," not in c:
        return c

    left, right = [p.strip() for p in c.split(",", 1)]
    if not right:
        return left

    rlow = right.lower()

    # "Bahamas, The" / "Gambia, The"
    if rlow == "the":
        return f"The {left}".strip()

    # "Congo, The Democratic Republic of the"
    if rlow.startswith("the "):
        return f"The {right[4:].strip()} {left}".replace("  ", " ").strip()

    # "Korea, Republic of" -> "Republic of Korea"
    return f"{right} {left}".replace("  ", " ").strip()


# -------------------------------
# WEATHER INDICATORS
# -------------------------------
def compute_indicators_at_index(grids, times, ilat, ilon):
    """
    Returns:
        rain_sum (mm),
        soil_avg (0–1),
        peak_dt_local (datetime or None)
    """
    if grids is None or grids.get("APCP") is None or not times:
        return 0.0, 0.0, None

    meta = grids.get("_meta", {})
    has_baseline = bool(meta.get("has_baseline", False))
    start_i = 1 if has_baseline else 0  # index where the real window begins

    rain_vals = []
    soil_vals = []

    n_steps = min(len(times), grids["APCP"].shape[0])

    # Need at least 2 points if we have baseline (baseline + 1 window step)
    if has_baseline and n_steps < 2:
        return 0.0, 0.0, None

    for t in range(start_i, n_steps):
        apcp_current = float(grids["APCP"][t, ilat, ilon])
        apcp_prev = float(grids["APCP"][t - 1, ilat, ilon]) if t > 0 else 0.0
        rain_inc = apcp_current - apcp_prev
        rain_vals.append(max(0.0, rain_inc))

        if grids.get("SOILW") is not None:
            soil_val = float(grids["SOILW"][t, ilat, ilon])
            soil_vals.append(soil_val)

    rain_sum = sum(rain_vals)

    if soil_vals:
        soil_norm = [min(max(x / 0.6, 0.0), 1.0) for x in soil_vals]
        soil_avg = sum(soil_norm) / len(soil_norm)
    else:
        soil_avg = 0.0

    if rain_vals and any(rain_vals):
        max_idx_in_rain = int(np.argmax(rain_vals))  # index within rain_vals
        # map back to times index (offset by start_i)
        t_idx = start_i + max_idx_in_rain

        peak_dt_utc = times[t_idx].replace(tzinfo=ZoneInfo("UTC"))
        tz = ZoneInfo(TIMEZONE)
        peak_dt_local = peak_dt_utc.astimezone(tz)
    else:
        peak_dt_local = None

    return rain_sum, soil_avg, peak_dt_local


# -------------------------------
# LINEAR MULTIPLIERS
# -------------------------------
def rainfall_multiplier(rain_mm: float) -> float:
    return max(0.0, rain_mm / RAIN_UNIT_MM)


def soil_multiplier(soil_frac: float) -> float:
    s = max(0.0, min(1.0, soil_frac))
    return SOIL_MIN_MULT + s * (SOIL_MAX_MULT - SOIL_MIN_MULT)


# -------------------------------
# RISK MODEL (RAW ONLY)
# -------------------------------
def calculate_dynamic_risk_raw(base_risk: float, rain_mm: float, soil_frac: float):
    """
    Returns: (raw_score, level, r_mult, s_mult)
    """
    if rain_mm < RAIN_CUTOFF_MM:
        return 0.0, "None", 0.0, soil_multiplier(0.0)

    r_mult = rainfall_multiplier(rain_mm)
    s_mult = soil_multiplier(soil_frac)

    raw_score = max(0.0, base_risk) * r_mult * s_mult

    if raw_score == 0:
        level = "None"
    elif raw_score < RAW_LOW_MAX:
        level = "Low"
    elif raw_score < RAW_MED_MAX:
        level = "Medium"
    elif raw_score < RAW_HIGH_MAX:
        level = "High"
    else:
        level = "Extreme"

    return round(raw_score, 3), level, r_mult, s_mult


# -------------------------------
# ALERT COMPARISON (level transitions only)
# -------------------------------
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"alerts": []}


def rotate_comparison_snapshots(max_history=COMPARISON_HISTORY):
    """
    Rotate alerts_comparison snapshots.
    """
    base = COMPARISON_PATH
    root, ext = os.path.splitext(base)  # e.g. "alerts_comparison_cluster", ".json"

    # Shift older snapshots up one index
    for i in range(max_history - 1, 0, -1):
        older = f"{root}_{i}{ext}"
        newer = f"{root}_{i + 1}{ext}"
        if os.path.exists(older):
            if os.path.exists(newer):
                os.remove(newer)
            os.replace(older, newer)

    # Move current base file to _1
    if os.path.exists(base):
        first_snapshot = f"{root}_1{ext}"
        if os.path.exists(first_snapshot):
            os.remove(first_snapshot)
        os.replace(base, first_snapshot)


def build_alert_dict(alerts):
    return {(round(a["latitude"], 4), round(a["longitude"], 4)): a for a in alerts}


def compare_alerts(prev, curr):
    """
    Tweet when:
      • First time we see a site at a tweet-worthy level (Medium/High/Extreme)
      • Any UPGRADE into a tweet-worthy level
      • Downgrades from tweet-worthy levels (optional)

    NOTE: final gating (upgrade requires prior tweet; downgrade requires prior tweet and cooldown)
          is handled later in the tweeting loop using tweeted_alerts log.
    """
    changes = []
    for key, c in curr.items():
        cur_lvl = c["dynamic_level"]

        # New site this run
        if key not in prev:
            if cur_lvl in TWEET_LEVELS:
                changes.append(("New", c))
            continue

        prev_lvl = prev[key]["dynamic_level"]
        if prev_lvl == cur_lvl:
            continue

        prev_i, cur_i = LEVELS.index(prev_lvl), LEVELS.index(cur_lvl)

        # Any upgrade into a tweet-worthy level
        if ALERT_ON_UPGRADES and cur_i > prev_i and cur_lvl in TWEET_LEVELS:
            changes.append(("Upgrade", c))
            continue

        # Downgrades from tweet-worthy levels (prev in tweet-levels)
        if ALERT_ON_DOWNGRADES and cur_i < prev_i and prev_lvl in TWEET_LEVELS:
            changes.append(("Downgrade", c))

    return changes


# -------------------------------
# TWEET MANAGEMENT
# -------------------------------
def load_tweeted_alerts():
    if os.path.exists(TWEET_LOG_PATH):
        with open(TWEET_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_tweeted_alerts(tweeted):
    with open(TWEET_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(tweeted, f, indent=2, ensure_ascii=False)


def cleanup_tweeted_alerts(tweeted, valid_coords, now_utc):
    """
    Keep only coordinates that still exist in the CSV.
    If resolved, keep for at least COOLDOWN_HOURS since resolved_at.
    """
    cleaned = {}
    for k, v in tweeted.items():
        if k not in valid_coords:
            continue

        if v.get("resolved", False):
            resolved_at = parse_utc_z(v.get("resolved_at"))
            # If we don't have a timestamp, keep it (fail-safe)
            if resolved_at is None:
                cleaned[k] = v
                continue

            if (now_utc - resolved_at) < timedelta(hours=COOLDOWN_HOURS):
                cleaned[k] = v
            # else: drop it after cooldown window
            continue

        cleaned[k] = v

    if len(cleaned) < len(tweeted):
        print(f"🧹 Cleaned {len(tweeted) - len(cleaned)} resolved entries after cooldown expiry.")
    return cleaned


def create_client():
    """Create Tweepy client."""
    return tweepy.Client(
        consumer_key=TWITTER_API_KEY,
        consumer_secret=TWITTER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_SECRET,
        wait_on_rate_limit=True,
    )


def tweet_alert(change_type, alert, quote_tweet_id=None):
    """Post a tweet for a new or transitioned flood alert (single point style)."""
    lat, lon = alert["latitude"], alert["longitude"]
    level = alert["dynamic_level"]

    level_colors = {
        "None": "⚪",
        "Low": "⚪",
        "Medium": "🟢",
        "High": "🟠",
        "Extreme": "🔴",
    }

    color_emoji = level_colors.get(level, "⚪")

    name    = alert.get("name", "Location")
    country = alert.get("country", "")
    flag    = alert.get("country_flag", "")

    if country:
        if flag:
            place = f"{flag} {name}, {country}"
        else:
            place = f"{name}, {country}"
    else:
        place = name

    level_upper = level.upper()
    peak_time_str = alert.get("peak_time_local_str", "unknown")

    tweet_text = (
        f"{color_emoji} {level_upper} FLOOD RISK – {place}\n\n"
        f"Type: {change_type}\n"
        f"Local Time: {peak_time_str}\n"
        f"Location: ({lat:.2f}, {lon:.2f})\n"
        f"Rain: {alert[f'rain_{FORECAST_HOURS}h_mm']:.1f} mm\n"
        f"Soil moisture: {alert['soil_moisture_avg']:.2f}\n"
    )

    print(
        f"🚨 Tweet → {tweet_text}\n"
        + (f"(Quoting ID: {quote_tweet_id})\n" if quote_tweet_id else "")
    )

    if not TWITTER_ENABLED:
        print("🧪 DRY RUN (tweet suppressed). Set TWITTER_ENABLED=true to send.")
        return None

    try:
        client = create_client()
        response = client.create_tweet(
            text=tweet_text,
            quote_tweet_id=quote_tweet_id
        )
        new_tweet_id = response.data["id"]
        print(f"✅ Tweet posted with ID: {new_tweet_id}")
        return str(new_tweet_id)
    except Exception as e:
        print(f"❌ Tweet failed: {e}")
        return None


# -------------------------------
# REGION CLUSTER HELPERS
# -------------------------------
def cluster_level(alerts_in_region):
    """Cluster level = highest city-level in this region."""
    levels_here = [a["dynamic_level"] for _, a in alerts_in_region]
    max_idx = max(LEVELS.index(lvl) for lvl in levels_here)
    return LEVELS[max_idx]


def cluster_change_type(alerts_in_region):
    """Upgrade beats New, New beats Downgrade."""
    types = {ct for ct, _ in alerts_in_region}
    if "Upgrade" in types:
        return "Upgrade"
    if "New" in types:
        return "New"
    if "Downgrade" in types:
        return "Downgrade"
    return "New"


def cluster_stats(alerts_in_region):
    """Peak rain + soil range + representative local time."""
    rains = [a[f"rain_{FORECAST_HOURS}h_mm"] for _, a in alerts_in_region]
    soils = [a["soil_moisture_avg"] for _, a in alerts_in_region]

    peak_rain = max(rains)
    soil_min, soil_max = min(soils), max(soils)

    max_i = max(
        range(len(alerts_in_region)),
        key=lambda i: alerts_in_region[i][1][f"rain_{FORECAST_HOURS}h_mm"],
    )
    peak_time = alerts_in_region[max_i][1].get("peak_time_local_str", "unknown")

    return peak_rain, soil_min, soil_max, peak_time


def cluster_city_list(alerts_in_region):
    """
    Return a comma-separated list of ALL city names in the region,
    sorted by severity (Extreme→High→Medium→Low→None) then score.
    """
    sorted_alerts = sorted(
        (a for _, a in alerts_in_region),
        key=lambda a: (LEVELS.index(a["dynamic_level"]), a["raw_dynamic_score"]),
        reverse=True,
    )
    names = [a["name"] for a in sorted_alerts]
    return ", ".join(names)


def tweet_region_cluster(region_key, alerts_in_region, client=None,
                         change_type=None, quote_tweet_id=None):
    """
    Compose and send a tweet for a region (multi-city only).

    Region style:
       🟠 HIGH FLOOD RISK – 🇧🇷 Mato Grosso do Sul, Brazil
       Type: Upgrade
       Key locations: city1, city2, ...
       Local time (approx.): ...
       Peak rain ...
       Soil moisture range ...
    """
    country, region = region_key
    cluster_lvl = cluster_level(alerts_in_region)
    if change_type is None:
        change_type = cluster_change_type(alerts_in_region)
    peak_rain, soil_min, soil_max, peak_time = cluster_stats(alerts_in_region)
    key_locs = cluster_city_list(alerts_in_region)

    level_colors = {
        "None": "⚪",
        "Low": "⚪",
        "Medium": "🟢",
        "High": "🟠",
        "Extreme": "🔴",
    }
    color_emoji = level_colors.get(cluster_lvl, "⚪")

    sample_alert = alerts_in_region[0][1]
    flag = sample_alert.get("country_flag", "")

    if region and country:
        header_place = f"{flag} {region}, {country}" if flag else f"{region}, {country}"
    elif country:
        header_place = f"{flag} {country}" if flag else country
    else:
        header_place = region or "Region"

    tweet_text = (
        f"{color_emoji} {cluster_lvl.upper()} FLOOD RISK – {header_place}\n\n"
        f"Type: {change_type}\n"
        f"Key locations: {key_locs}\n"
        f"Local time (approx.): {peak_time}\n"
        f"Peak rain (next {FORECAST_HOURS}h): {peak_rain:.1f} mm\n"
        f"Soil moisture range: {soil_min:.2f}–{soil_max:.2f}"
    )

    print(f"🚨 Region tweet →\n{tweet_text}\n"
          + (f"(Quoting ID: {quote_tweet_id})\n" if quote_tweet_id else ""))

    if not TWITTER_ENABLED:
        print("🧪 DRY RUN (tweet suppressed). Set TWITTER_ENABLED=true to send.")
        return None

    try:
        if client is None:
            client = create_client()
        response = client.create_tweet(text=tweet_text, quote_tweet_id=quote_tweet_id)
        new_tweet_id = response.data["id"]
        print(f"✅ Region tweet posted with ID: {new_tweet_id}")
        return str(new_tweet_id)
    except Exception as e:
        print(f"❌ Region tweet failed: {e}")
        return None


# -------------------------------
# PENDING DOWNGRADE PROCESSOR
# -------------------------------
def process_pending_downgrades(tweeted_alerts, alerts, now_utc, client, last_tweet_ts_holder):
    """
    If a downgrade was skipped due to cooldown, we store it as pending.
    After cooldown, if it's STILL downgraded vs tweeted_level, we tweet it.
    """
    coord_to_alert = {
        f"{a['latitude']:.4f},{a['longitude']:.4f}": a
        for a in alerts
    }

    # Build region index from current alerts (only for coords we actually track)
    region_to_alerts = defaultdict(list)
    for a in alerts:
        ck = f"{a['latitude']:.4f},{a['longitude']:.4f}"
        if ck not in tweeted_alerts:
            continue
        rk = (a.get("country", "") or "", a.get("region", "") or "")
        region_to_alerts[rk].append(("Active", a))

    # --- 1) point-level pending downgrades ---
    for coord_key, entry in list(tweeted_alerts.items()):
        if not entry.get("pending_downgrade"):
            continue
        if entry.get("pending_is_region"):
            continue

        if within_cooldown(entry, now_utc):
            continue

        a = coord_to_alert.get(coord_key)
        if not a:
            continue

        cur_lvl = a["dynamic_level"]
        tweeted_lvl = get_tweeted_level(entry)

        # If recovered back to tweeted level or higher, cancel pending
        if level_index(cur_lvl) >= level_index(tweeted_lvl):
            clear_pending(entry)
            continue

        quote_id = entry.get("tweet_id")
        if not quote_id:
            # Shouldn't happen (pending only set if there was a tweet), but fail-safe
            clear_pending(entry)
            continue

        now_ts = time.time()
        if now_ts - last_tweet_ts_holder[0] < MIN_SECONDS_BETWEEN_TWEETS:
            time.sleep(MIN_SECONDS_BETWEEN_TWEETS - (now_ts - last_tweet_ts_holder[0]))

        new_id = tweet_alert("Downgrade", a, quote_tweet_id=quote_id)
        last_tweet_ts_holder[0] = time.time()

        if new_id:
            entry["tweet_id"] = new_id
            entry["tweeted_level"] = cur_lvl
            entry["risk_level"] = cur_lvl
            entry["last_tweeted_at"] = now_utc.isoformat().replace("+00:00", "Z")
            entry["last_updated"] = now_utc.isoformat().replace("+00:00", "Z")

            clear_pending(entry)

            if cur_lvl not in TWEET_LEVELS:
                entry["resolved"] = True
                entry.setdefault("resolved_at", now_utc.isoformat().replace("+00:00", "Z"))
            else:
                entry["resolved"] = False
                entry.pop("resolved_at", None)

    # --- 2) region-level pending downgrades ---
    processed_regions = set()

    for coord_key, entry in list(tweeted_alerts.items()):
        if not entry.get("pending_downgrade"):
            continue
        if not entry.get("pending_is_region"):
            continue

        if within_cooldown(entry, now_utc):
            continue

        a = coord_to_alert.get(coord_key)
        if not a:
            continue

        region_key = (a.get("country", "") or "", a.get("region", "") or "")
        if region_key in processed_regions:
            continue

        current_pairs = region_to_alerts.get(region_key, [])
        if not current_pairs:
            current_pairs = [("Active", a)]

        cur_cluster_lvl = cluster_level(current_pairs)
        tweeted_lvl = get_tweeted_level(entry)

        # If recovered back to tweeted level or higher, cancel pending
        if level_index(cur_cluster_lvl) >= level_index(tweeted_lvl):
            clear_pending(entry)
            processed_regions.add(region_key)
            continue

        quote_id = entry.get("tweet_id")
        if not quote_id:
            clear_pending(entry)
            processed_regions.add(region_key)
            continue

        now_ts = time.time()
        if now_ts - last_tweet_ts_holder[0] < MIN_SECONDS_BETWEEN_TWEETS:
            time.sleep(MIN_SECONDS_BETWEEN_TWEETS - (now_ts - last_tweet_ts_holder[0]))

        new_id = tweet_region_cluster(
            region_key,
            current_pairs,
            client=client,
            change_type="Downgrade",
            quote_tweet_id=quote_id
        )
        last_tweet_ts_holder[0] = time.time()

        if new_id:
            # Update all coords in this region that exist in the log
            for _, aa in current_pairs:
                ck = f"{aa['latitude']:.4f},{aa['longitude']:.4f}"
                if ck not in tweeted_alerts:
                    continue
                e = tweeted_alerts[ck]
                e["tweet_id"] = new_id
                e["tweeted_level"] = cur_cluster_lvl
                e["risk_level"] = aa["dynamic_level"]
                e["last_tweeted_at"] = now_utc.isoformat().replace("+00:00", "Z")
                e["last_updated"] = now_utc.isoformat().replace("+00:00", "Z")
                clear_pending(e)

                if aa["dynamic_level"] not in TWEET_LEVELS:
                    e["resolved"] = True
                    e.setdefault("resolved_at", now_utc.isoformat().replace("+00:00", "Z"))
                else:
                    e["resolved"] = False
                    e.pop("resolved_at", None)

        processed_regions.add(region_key)


# -------------------------------
# MAIN WORKFLOW
# -------------------------------
def main():
    print(f"🌧️ FloodLink Live Risk Evaluation started ({FORECAST_HOURS}-hour window)…")
    print(f"Current working directory: {os.getcwd()}")

    previous = load_json(COMPARISON_PATH)
    prev_alerts_dict = build_alert_dict(previous.get("alerts", []))
    tweeted_alerts = load_tweeted_alerts()

    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file not found: {CSV_PATH} – skipping evaluation.")
        return

    df = pd.read_csv(CSV_PATH)

    # Clean common text fields (fixes 'ParanÃ¡' -> 'Paraná', etc.)
    for col in ["region", "Country", "Name", "ETIQUETA", "CountryFlag"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)


    # --- Basic CSV + FRisk summary ---
    print("📊 CSV summary:")
    print(f"  Total rows: {len(df)}")
    if "FRisk" in df.columns:
        print(f"  FRisk min/max: {df['FRisk'].min()} / {df['FRisk'].max()}")
    else:
        print("  ⚠️ Column 'FRisk' not found in CSV!")

    high_risk = df[df["FRisk"] > RISK_THRESHOLD].copy()
    print(f"  High-risk rows (FRisk > {RISK_THRESHOLD}): {len(high_risk)}")

    FAST_MODE = False
    FAST_SAMPLE = 200
    if FAST_MODE and not high_risk.empty:
        high_risk = high_risk.head(FAST_SAMPLE)
        print(f"⚡ FAST MODE: only evaluating first {len(high_risk)} cities")

    valid_coords = {f"{row['Latitude']:.4f},{row['Longitude']:.4f}" for _, row in df.iterrows()}
    now_utc = datetime.now(ZoneInfo("UTC"))
    tweeted_alerts = cleanup_tweeted_alerts(tweeted_alerts, valid_coords, now_utc)


    alerts = []
    start_time = time.time()

    # Load NOAA GFS grids once for all locations
    grids, lats, lons, times = load_gfs_grids(FORECAST_HOURS)

    # DEBUG: confirm the forecast window we are evaluating (UTC + local)
    meta = grids.get("_meta", {}) if grids else {}
    has_baseline = bool(meta.get("has_baseline", False))
    start_i = 1 if has_baseline else 0
   
    if times and len(times) > start_i:
        tz = ZoneInfo(TIMEZONE)
        start_utc = times[start_i].replace(tzinfo=ZoneInfo("UTC"))
        end_utc   = times[-1].replace(tzinfo=ZoneInfo("UTC"))
        print(f"🕒 Window UTC:   {start_utc.strftime('%Y-%m-%d %H:%M')} → {end_utc.strftime('%Y-%m-%d %H:%M')}")
        print(f"🕒 Window local: {start_utc.astimezone(tz).strftime('%Y-%m-%d %H:%M')} → {end_utc.astimezone(tz).strftime('%Y-%m-%d %H:%M')}")
    else:
        print("🕒 Window debug: no usable times[] returned (download/parse failed)")

    if grids is None or grids.get("APCP") is None or lats is None or lons is None:
        print("❌ Failed to load GFS data – using previous alerts where available.")
        for _, row in high_risk.iterrows():
            key = (round(row["Latitude"], 4), round(row["Longitude"], 4))
            prev_alert = prev_alerts_dict.get(key)
            if prev_alert:
                alerts.append(prev_alert)
    else:
        idx_map = precompute_city_indices(lats, lons, high_risk)
        total = len(high_risk)

        for idx, (_, row) in enumerate(high_risk.iterrows(), start=1):
            if idx % 100 == 0 or idx == total:
                print(f"… processed {idx}/{total} high-risk cities")

            lat = float(row["Latitude"])
            lon = float(row["Longitude"])
            base_risk = float(row["FRisk"])

            if "Name" in row and pd.notna(row["Name"]):
                name = clean_text(row["Name"])
            elif "ETIQUETA" in row and pd.notna(row["ETIQUETA"]):
                name = clean_text(row["ETIQUETA"])
            else:
                name = f"id_{row['JOIN_ID']}"
               
            country = normalize_country(row.get("Country", ""))
            country_flag = clean_text(row.get("CountryFlag", ""))
            region = clean_text(row.get("region", ""))

            ilat, ilon = idx_map[row["JOIN_ID"]]

            rain_sum, soil_avg, peak_dt_local = compute_indicators_at_index(
                grids, times, ilat, ilon
            )

            raw_score, dyn_level, r_mult, s_mult = calculate_dynamic_risk_raw(
                base_risk, rain_sum, soil_avg
            )

            if peak_dt_local is not None:
                peak_time_local_str = peak_dt_local.strftime("%H:%M")
            else:
                peak_time_local_str = "unknown"

            alerts.append({
                "id": str(row["JOIN_ID"]),
                "country": country,
                "country_flag": country_flag,
                "region": region,
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "base_risk": round(base_risk, 2),

                f"rain_{FORECAST_HOURS}h_mm": round(rain_sum, 2),
                "soil_moisture_avg": round(soil_avg, 3),

                "rain_mult": round(r_mult, 3),
                "soil_mult": round(s_mult, 3),

                "raw_dynamic_score": raw_score,
                "dynamic_level": dyn_level,
                "peak_time_local_str": peak_time_local_str,
            })

    # Persist current results
    result = {
        "timestamp": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
        "source": "NOAA GFS",
        "forecast_window_hours": FORECAST_HOURS,
        "features_evaluated": len(alerts),
        "alerts": alerts,
    }

    # Detect level-change events
    curr_alerts_dict = build_alert_dict(alerts)
    changes = compare_alerts(prev_alerts_dict, curr_alerts_dict)
    print(f"🔍 Detected {len(changes)} level-change events.")

    if changes:
        for change_type, a in changes:
            key = (round(a["latitude"], 4), round(a["longitude"], 4))
            prev_lvl = prev_alerts_dict.get(key, {}).get("dynamic_level", "None")
            print(
                "🛰️ "
                f"{a['name']} [{a['latitude']:.4f},{a['longitude']:.4f}]: "
                f"{prev_lvl} → {a['dynamic_level']} ({change_type}); "
                f"rain={a[f'rain_{FORECAST_HOURS}h_mm']} mm, "
                f"soil={a['soil_moisture_avg']:.3f}"
            )
    else:
        print("ℹ️ No tweetable transitions this run.")

    # -----------------------
    # REGION-LEVEL TWEETS
    # -----------------------
    region_clusters = defaultdict(list)
    for change_type, alert in changes:
        region = alert.get("region", "") or ""
        country = alert.get("country", "") or ""
        key = (country, region)
        region_clusters[key].append((change_type, alert))
    print(f"📡 Region groups to tweet: {len(region_clusters)}")

    last_tweet_ts = 0.0
    client = create_client() if TWITTER_ENABLED else None

    for region_key, alerts_in_region in region_clusters.items():
        # Single-location region → classic per-point behaviour
        if len(alerts_in_region) == 1:
            change_type, alert = alerts_in_region[0]
            coord_key = f"{alert['latitude']:.4f},{alert['longitude']:.4f}"
            last_entry = tweeted_alerts.get(coord_key)

            if change_type == "Upgrade" and last_entry is None:
                change_type = "New"

            if change_type == "Downgrade" and last_entry is None:
                print(f"↘️ Skipping downgrade for {alert['name']} – no prior tweet.")
                continue

            if change_type == "Downgrade" and last_entry is not None and within_cooldown(last_entry, now_utc):
                print(f"⏳ Skipping downgrade tweet for {alert['name']} – within {COOLDOWN_HOURS}h cooldown.")
                last_entry["risk_level"] = alert["dynamic_level"]
                last_entry["rain_mm"] = alert[f"rain_{FORECAST_HOURS}h_mm"]
                last_entry["soil_moisture"] = alert["soil_moisture_avg"]
                last_entry["raw_dynamic_score"] = alert["raw_dynamic_score"]
                last_entry["last_updated"] = now_utc.isoformat().replace("+00:00", "Z")

                if alert["dynamic_level"] not in TWEET_LEVELS:
                    last_entry["resolved"] = True
                    last_entry.setdefault("resolved_at", now_utc.isoformat().replace("+00:00", "Z"))
                else:
                    last_entry["resolved"] = False
                    last_entry.pop("resolved_at", None)

                mark_pending_downgrade(last_entry, alert["dynamic_level"], now_utc, is_region=False)
                continue

            quote_tweet_id = None
            if change_type in ["Upgrade", "Downgrade"] and last_entry and "tweet_id" in last_entry:
                quote_tweet_id = last_entry["tweet_id"]

            now_ts = time.time()
            if now_ts - last_tweet_ts < MIN_SECONDS_BETWEEN_TWEETS:
                time.sleep(MIN_SECONDS_BETWEEN_TWEETS - (now_ts - last_tweet_ts))

            new_tweet_id = tweet_alert(change_type, alert, quote_tweet_id=quote_tweet_id)
            last_tweet_ts = time.time()

            if new_tweet_id:
                level = alert["dynamic_level"]
                resolved = level not in TWEET_LEVELS

                tweeted_alerts[coord_key] = {
                    "country": alert.get("country", ""),
                    "name": alert["name"],
                    "risk_level": level,
                    "tweeted_level": level,
                    "latitude": alert["latitude"],
                    "longitude": alert["longitude"],
                    "rain_mm": alert[f"rain_{FORECAST_HOURS}h_mm"],
                    "soil_moisture": alert["soil_moisture_avg"],
                    "raw_dynamic_score": alert["raw_dynamic_score"],
                    "last_updated": now_utc.isoformat().replace("+00:00", "Z"),
                    "last_tweeted_at": now_utc.isoformat().replace("+00:00", "Z"),
                    "tweet_id": new_tweet_id,
                    "resolved": resolved,
                }

                clear_pending(tweeted_alerts[coord_key])
                if resolved:
                    tweeted_alerts[coord_key].setdefault("resolved_at", now_utc.isoformat().replace("+00:00", "Z"))
                else:
                    tweeted_alerts[coord_key].pop("resolved_at", None)

            continue

        # ---------------- Multi-city region → cluster style ----------------
        cluster_type = cluster_change_type(alerts_in_region)

        if cluster_type == "Downgrade" and not ALERT_ON_DOWNGRADES:
            continue

        rep_alert = alerts_in_region[0][1]
        rep_key = f"{rep_alert['latitude']:.4f},{rep_alert['longitude']:.4f}"
        rep_last = tweeted_alerts.get(rep_key)

        if cluster_type == "Upgrade" and rep_last is None:
            cluster_type = "New"

        if cluster_type == "Downgrade" and rep_last is None:
            print(f"↘️ Skipping region downgrade for {region_key} – no prior tweet.")
            continue

        if cluster_type == "Downgrade" and rep_last is not None and within_cooldown(rep_last, now_utc):
            print(f"⏳ Skipping region downgrade tweet for {region_key} – within {COOLDOWN_HOURS}h cooldown.")

            for _, alert in alerts_in_region:
                coord_key = f"{alert['latitude']:.4f},{alert['longitude']:.4f}"
                entry = tweeted_alerts.get(coord_key)
                if not entry:
                    continue
                entry["risk_level"] = alert["dynamic_level"]
                entry["rain_mm"] = alert[f"rain_{FORECAST_HOURS}h_mm"]
                entry["soil_moisture"] = alert["soil_moisture_avg"]
                entry["raw_dynamic_score"] = alert["raw_dynamic_score"]
                entry["last_updated"] = now_utc.isoformat().replace("+00:00", "Z")

                if alert["dynamic_level"] not in TWEET_LEVELS:
                    entry["resolved"] = True
                    entry.setdefault("resolved_at", now_utc.isoformat().replace("+00:00", "Z"))
                else:
                    entry["resolved"] = False
                    entry.pop("resolved_at", None)

            mark_pending_downgrade(rep_last, cluster_level(alerts_in_region), now_utc, is_region=True)
            continue

        quote_tweet_id = None
        if cluster_type in ["Upgrade", "Downgrade"] and rep_last and "tweet_id" in rep_last:
            quote_tweet_id = rep_last["tweet_id"]

        now_ts = time.time()
        if now_ts - last_tweet_ts < MIN_SECONDS_BETWEEN_TWEETS:
            time.sleep(MIN_SECONDS_BETWEEN_TWEETS - (now_ts - last_tweet_ts))

        new_tweet_id = tweet_region_cluster(
            region_key,
            alerts_in_region,
            client=client,
            change_type=cluster_type,
            quote_tweet_id=quote_tweet_id,
        )
        last_tweet_ts = time.time()

        if new_tweet_id:
            cluster_lvl = cluster_level(alerts_in_region)

            for _, alert in alerts_in_region:
                coord_key = f"{alert['latitude']:.4f},{alert['longitude']:.4f}"
                level = alert["dynamic_level"]
                resolved = level not in TWEET_LEVELS

                tweeted_alerts[coord_key] = {
                    "country": alert.get("country", ""),
                    "name": alert["name"],
                    "risk_level": level,
                    "tweeted_level": cluster_lvl,
                    "latitude": alert["latitude"],
                    "longitude": alert["longitude"],
                    "rain_mm": alert[f"rain_{FORECAST_HOURS}h_mm"],
                    "soil_moisture": alert["soil_moisture_avg"],
                    "raw_dynamic_score": alert["raw_dynamic_score"],
                    "last_updated": now_utc.isoformat().replace("+00:00", "Z"),
                    "last_tweeted_at": now_utc.isoformat().replace("+00:00", "Z"),
                    "tweet_id": new_tweet_id,
                    "resolved": resolved,
                }

                clear_pending(tweeted_alerts[coord_key])
                if resolved:
                    tweeted_alerts[coord_key].setdefault("resolved_at", now_utc.isoformat().replace("+00:00", "Z"))
                else:
                    tweeted_alerts[coord_key].pop("resolved_at", None)

    # ✅ After region tweets: process pending downgrades that have waited out the cooldown
    last_tweet_ts_holder = [last_tweet_ts]
    process_pending_downgrades(
        tweeted_alerts=tweeted_alerts,
        alerts=alerts,
        now_utc=now_utc,
        client=client,
        last_tweet_ts_holder=last_tweet_ts_holder
    )
    last_tweet_ts = last_tweet_ts_holder[0]

    save_tweeted_alerts(tweeted_alerts)

    rotate_comparison_snapshots(COMPARISON_HISTORY)

    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"✅ Completed in {round((time.time() - start_time)/60, 1)} min. "
        f"Features evaluated: {len(alerts)} | "
        f"Updated {COMPARISON_PATH} and {TWEET_LOG_PATH}."
    )


if __name__ == "__main__":
    main()
