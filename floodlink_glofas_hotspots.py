"""
FloodLink – GloFAS Hotspot Evaluator

Goal:
- Pull river flood "hotspots" from GloFAS (reporting points / thresholds)
- Attach nearest cities from GeoNames cities1000.csv
- Detect level transitions (Medium/High/Extreme) over time
- Tweet upgrades/downgrades, logging to JSON files.

This script is intentionally separate from the Open-Meteo city evaluator.
"""

import os
import json
import time
from math import radians, sin, cos, asin, sqrt
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import tweepy
from requests.exceptions import RequestException, ReadTimeout, ConnectionError

# ---------------------------------
# CONFIGURATION
# ---------------------------------
CITIES_PATH = "cities1000.csv"  # GeoNames cities (pop >= 1000)

GLOFAS_COMPARISON_PATH = "glofas_alerts_comparison.json"
GLOFAS_TWEET_LOG_PATH   = "glofas_tweeted_alerts.json"

COMPARISON_HISTORY = 5       # how many past comparison snapshots to keep
TIMEZONE = "UTC"             # timestamps for logs

# Nearest-city search
MAX_CITY_DISTANCE_KM = 50.0  # radius for "nearby towns" in tweets
TOP_NEAREST_CITIES   = 3     # how many to keep

# --- Twitter config (same env vars as your other script) ---
TWITTER_ENABLED       = os.getenv("TWITTER_ENABLED", "false").lower() == "true"
TWITTER_API_KEY       = os.getenv("TWITTER_API_KEY")
TWITTER_SECRET        = os.getenv("TWITTER_SECRET")
TWITTER_ACCESS_TOKEN  = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
MIN_SECONDS_BETWEEN_TWEETS = 30

# GloFAS-related config (we'll refine once the data access is wired)
GLOFAS_BASE_URL = "https://example-glofas-endpoint"  # TODO: replace with real service / file path

# Map GloFAS return periods → FloodLink levels
RETURN_PERIOD_LEVEL_MAP = {
    2:  "Medium",
    5:  "High",
    20: "Extreme",
}

LEVELS = ["None", "Low", "Medium", "High", "Extreme"]
TWEET_LEVELS = ["Medium", "High", "Extreme"]
ALERT_ON_UPGRADES   = True
ALERT_ON_DOWNGRADES = True


# ---------------------------------
# BASIC UTILS
# ---------------------------------
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"alerts": []}


def rotate_comparison_snapshots(base_path, max_history=COMPARISON_HISTORY):
    """
    Rotate comparison JSON snapshots for this script only.

    base_path: e.g. "glofas_alerts_comparison.json"
    Rotates:
      glofas_alerts_comparison_{i}.json
    """
    prefix, ext = os.path.splitext(base_path)  # ("glofas_alerts_comparison", ".json")

    # Shift numbered snapshots up: N-1 -> N, ..., 1 -> 2
    for i in range(max_history - 1, 0, -1):
        older = f"{prefix}_{i}{ext}"
        newer = f"{prefix}_{i + 1}{ext}"
        if os.path.exists(older):
            if os.path.exists(newer):
                os.remove(newer)
            os.replace(older, newer)

    # Move current base file to _1
    if os.path.exists(base_path):
        first_snapshot = f"{prefix}_1{ext}"
        if os.path.exists(first_snapshot):
            os.remove(first_snapshot)
        os.replace(base_path, first_snapshot)


def build_alert_dict(alerts):
    """Key alerts by (lat, lon) rounded to 4 decimals."""
    return {(round(a["latitude"], 4), round(a["longitude"], 4)): a for a in alerts}


def compare_alerts(prev, curr):
    """
    Compare previous vs current alerts, focusing on dynamic_level changes.

    Returns a list of (change_type, alert) tuples, where change_type is
    "New", "Upgrade", or "Downgrade".
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

        # Downgrades from tweet-worthy levels (optional)
        if ALERT_ON_DOWNGRADES and cur_i < prev_i and prev_lvl in TWEET_LEVELS:
            changes.append(("Downgrade", c))

    return changes


# ---------------------------------
# TWEET MANAGEMENT
# ---------------------------------
def load_tweeted_alerts(path=GLOFAS_TWEET_LOG_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_tweeted_alerts(tweeted, path=GLOFAS_TWEET_LOG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tweeted, f, indent=2, ensure_ascii=False)


def tweet_alert(change_type, alert):
    """Post a tweet for a new or transitioned GloFAS river flood alert."""
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

    # Build a human-readable description of nearest cities
    nearest = alert.get("nearest_cities", [])
    if nearest:
        city_bits = [
            f"{c['name']} ({c['distance_km']:.0f} km)"
            for c in nearest
        ]
        nearest_str = "; ".join(city_bits)
    else:
        nearest_str = "No major towns within range"

    river_name = alert.get("river_name", "river")
    rp = alert.get("return_period", None)
    lead = alert.get("lead_time_days", None)

    rp_text = f"≥{rp}-year event" if rp else "flood event"
    lead_text = f"in ~{lead} days" if lead is not None else "in coming days"

    tweet_text = (
        f"{color_emoji} River flood risk near {alert.get('headline_city','Location')}.\n\n"
        f"{level} risk ({change_type})\n"
        f"River: {river_name} ({rp_text} {lead_text})\n"
        f"Nearest towns: {nearest_str}\n"
        f"Location ({lat:.2f}, {lon:.2f})\n"
    )

    print(f"🚨 GloFAS Tweet → {tweet_text}\n")

    if not TWITTER_ENABLED:
        print("🧪 DRY RUN (tweet suppressed). Set TWITTER_ENABLED=true to send.")
        return

    try:
        client = tweepy.Client(
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_SECRET,
            wait_on_rate_limit=True,
        )
        client.create_tweet(text=tweet_text)
    except Exception as e:
        print(f"❌ Tweet failed: {e}")


# ---------------------------------
# GEO / CITIES UTILITIES
# ---------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def load_cities(path=CITIES_PATH):
    """
    Load cities1000.csv as you have it:

    Columns (from your screenshot):
      - Name
      - ASCII Name
      - Country Code
      - Country name EN
      - Population
      - Timezone
      - Latitude
      - Longitude
    """
    df = pd.read_csv(path)  # header row already present

    # Keep only what we need and normalize column names
    df = df[[
        "Name",
        "Country Code",
        "Country name EN",
        "Latitude",
        "Longitude",
        "Population",
    ]].copy()

    df.rename(
        columns={
            "Name": "name",
            "Country Code": "country_code",
            "Country name EN": "country_name",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Population": "population",
        },
        inplace=True,
    )

    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["population"] = df["population"].fillna(0).astype(int)

    return df



def find_nearest_cities(lat, lon, cities_df,
                        max_distance_km=MAX_CITY_DISTANCE_KM,
                        top_n=TOP_NEAREST_CITIES):
    """
    Return up to top_n nearest cities within max_distance_km of (lat, lon).
    """
    dists = haversine_km(
        lat,
        lon,
        cities_df["latitude"].to_numpy(),
        cities_df["longitude"].to_numpy(),
    )

    temp = cities_df.copy()
    temp["distance_km"] = dists

    temp = temp[temp["distance_km"] <= max_distance_km]
    temp = temp.sort_values("distance_km").head(top_n)

    nearest = []
    for _, row in temp.iterrows():
        nearest.append({
            "name": row["name"],
            "country": row["country_name"],   # <-- uses "Country name EN"
            "population": int(row["population"]),
            "distance_km": float(row["distance_km"]),
        })
    return nearest



# ---------------------------------
# GLOFAS HOTSPOT FETCHING (SKELETON)
# ---------------------------------
def fetch_glofas_hotspots():
    """
    Fetch / load GloFAS flood reporting points that are above a chosen
    return-period threshold.

    This is a placeholder: implement with whichever access route you prefer:
      - Download daily reporting-points file before running the script
      - Query a WMS / API that returns GeoJSON
      - Read a NetCDF / CSV preprocessed elsewhere

    Must return a list of dicts like:

        {
            "glofas_id": "string",
            "latitude": 12.34,
            "longitude": 56.78,
            "river_name": "Rio Example",
            "country": "BR",
            "return_period": 20,   # 2, 5, 20, ...
            "lead_time_days": 3    # when peak is expected
        }
    """
    # TODO: implement real GloFAS integration
    hotspots = []

    # Example dummy hotspot for testing wiring:
    # hotspots.append({
    #     "glofas_id": "dummy-1",
    #     "latitude": 40.0,
    #     "longitude": -3.7,
    #     "river_name": "Rio Dummy",
    #     "country": "ES",
    #     "return_period": 5,
    #     "lead_time_days": 2,
    # })

    return hotspots


def return_period_to_level(rp):
    """Map a GloFAS return period (years) to FloodLink discrete level."""
    if rp is None:
        return "Medium"  # default if unknown
    if rp >= 20:
        return "Extreme"
    if rp >= 5:
        return "High"
    if rp >= 2:
        return "Medium"
    return "Low"


def build_alert_from_hotspot(h, cities_df):
    """
    Turn a single GloFAS hotspot dict into an alert dict compatible with
    the comparison + tweet framework.
    """
    lat = float(h["latitude"])
    lon = float(h["longitude"])

    nearest = find_nearest_cities(lat, lon, cities_df)
    headline_city = nearest[0]["name"] if nearest else h.get("country", "Location")

    level = return_period_to_level(h.get("return_period"))

    alert = {
        "id": str(h.get("glofas_id")),
        "country": h.get("country", ""),
        "name": headline_city,          # for backwards compatibility with tweet_alert
        "headline_city": headline_city, # explicit field
        "latitude": lat,
        "longitude": lon,

        "river_name": h.get("river_name", "river"),
        "return_period": h.get("return_period"),
        "lead_time_days": h.get("lead_time_days"),

        "nearest_cities": nearest,

        # Use a simple scalar score from return period for now
        "raw_dynamic_score": float(h.get("return_period", 2) * 10.0),
        "dynamic_level": level,
    }
    return alert


# ---------------------------------
# MAIN WORKFLOW
# ---------------------------------
def main():
    print("🌊 FloodLink GloFAS Hotspot Evaluation started…")

    previous = load_json(GLOFAS_COMPARISON_PATH)
    prev_alerts_dict = build_alert_dict(previous.get("alerts", []))
    tweeted_alerts = load_tweeted_alerts()

    # Load cities once
    cities_df = load_cities()
    print(f"🏙 Loaded {len(cities_df)} cities from {CITIES_PATH}")

    # Fetch hotspot points from GloFAS
    hotspots = fetch_glofas_hotspots()
    print(f"📡 Retrieved {len(hotspots)} GloFAS hotspots.")

    alerts = []
    start_time = time.time()

    for h in hotspots:
        alert = build_alert_from_hotspot(h, cities_df)
        alerts.append(alert)

    # Persist current results
    result = {
        "timestamp": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "source": "GloFAS",
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
                f"{a['headline_city']} [{a['latitude']:.4f},{a['longitude']:.4f}]: "
                f"{prev_lvl} → {a['dynamic_level']} ({change_type}); "
                f"RP={a.get('return_period')}, lead={a.get('lead_time_days')} days"
            )
    else:
        print("ℹ️ No tweetable transitions this run.")

    last_tweet_ts = 0.0

    # Tweet + update tracker
    for change_type, alert in changes:
        key = f"{alert['latitude']:.4f},{alert['longitude']:.4f}"
        current_level = alert["dynamic_level"]

        # For downgrades, only tweet if we tweeted before
        if change_type == "Downgrade" and key not in tweeted_alerts:
            print(f"↘️ Skipping downgrade tweet for {key} "
                  f"({alert['headline_city']}) – no prior tweet recorded.")
            continue

        # Global rate limiting
        now_ts = time.time()
        if now_ts - last_tweet_ts < MIN_SECONDS_BETWEEN_TWEETS:
            time.sleep(MIN_SECONDS_BETWEEN_TWEETS - (now_ts - last_tweet_ts))

        tweet_alert(change_type, alert)
        last_tweet_ts = time.time()

        # Update tweet log
        if current_level in TWEET_LEVELS:
            tweeted_alerts[key] = {
                "country": alert.get("country", ""),
                "name": alert["headline_city"],
                "risk_level": current_level,
                "latitude": alert["latitude"],
                "longitude": alert["longitude"],
                "river_name": alert.get("river_name"),
                "return_period": alert.get("return_period"),
                "lead_time_days": alert.get("lead_time_days"),
                "raw_dynamic_score": alert["raw_dynamic_score"],
                "last_updated": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
            }
        else:
            # Downgraded below Medium: mark and optionally clean later if you want
            tweeted_alerts[key] = {
                "country": alert.get("country", ""),
                "name": alert["headline_city"],
                "risk_level": current_level,
                "latitude": alert["latitude"],
                "longitude": alert["longitude"],
                "river_name": alert.get("river_name"),
                "return_period": alert.get("return_period"),
                "lead_time_days": alert.get("lead_time_days"),
                "raw_dynamic_score": alert["raw_dynamic_score"],
                "last_updated": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
                "resolved": True,
            }

    save_tweeted_alerts(tweeted_alerts)

    # Rotate old comparison snapshots, then write the new one
    rotate_comparison_snapshots(GLOFAS_COMPARISON_PATH)

    with open(GLOFAS_COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"✅ GloFAS run completed in {round((time.time() - start_time) / 60, 1)} min. "
        f"Updated {GLOFAS_COMPARISON_PATH} and {GLOFAS_TWEET_LOG_PATH}."
    )


# ---------------------------------
if __name__ == "__main__":
    main()
