"""
FloodLink – NOAA GFS → Global 1h Rainfall Time Series (JSON)
+ Precomputed Isolines (GeoJSON)

Downloads GFS 0.25° GRIB2, extracts hourly total precipitation
for the next FORECAST_HOURS hours, and writes:

  1) gfs_rain_6h.json      (time-series grid, 1h steps)
  2) rain_isolines.geojson (LineString isobars from one hour slice)

gfs_rain_6h.json structure:

{
  "header": {
    "parameter": "rain_1h_mm",
    "parameterUnit": "mm",
    "nx": ...,
    "ny": ...,
    "tCount": T,
    "tStepHours": 1,
    "lo1": ...,
    "la1": ...,
    "dx": 0.25,
    "dy": 0.25,
    "forecastHours": FORECAST_HOURS,
    "refTime": "YYYY-MM-DDTHH:00:00Z",
    "times": ["...", "...", ...],   # one per hour slice
    "source": "NOAA GFS 0p25",
    "layout": "time-major"
  },
  "data": [
    # Hour 1 (t=0): ny*nx values, row-major (lat 0..ny-1, lon 0..nx-1),
    # Hour 2 (t=1): next ny*nx values,
    # ...
    # Hour T: ...
  ]
}
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
import pygrib

# For contour/isoline generation
import matplotlib
matplotlib.use("Agg")  # headless backend for GitHub Actions
import matplotlib.pyplot as plt

# --------------------------------
# CONFIG
# --------------------------------
GFS_RES = "0p25"             # 0.25° grid
FORECAST_HOURS = 6           # next N hours (max we try to load)
MAX_RETRIES = 2
TIMEOUT = 60                 # seconds

RAINFIELD_PATH = "gfs_rain_6h.json"      # time series, 1h increments
ISOLINES_PATH = "rain_isolines.geojson"  # precomputed isobars

# Which time slice to use for isobars (0..tCount-1)
ISO_HOUR_INDEX = 0

# Rain thresholds (mm) for contour lines
ISO_LEVELS_MM = [1, 5, 10, 20, 40, 80]


# --------------------------------
# GFS helpers (same logic as engine)
# --------------------------------
def get_latest_cycle():
    """
    Determine latest available 6-hourly GFS cycle and a fallback.
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


def get_forecast_steps(max_hours: int):
    """
    For 0.25° GFS, we request hourly forecasts: f001..f{max_hours}.
    """
    return list(range(1, max_hours + 1))


def download_gfs_file(date, cycle, fhr):
    base_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{GFS_RES}.pl"

    if GFS_RES == "0p50":
        file_name = f"gfs.t{cycle}z.pgrb2full.0p50.f{fhr:03d}"
    else:
        file_name = f"gfs.t{cycle}z.pgrb2.{GFS_RES}.f{fhr:03d}"

    params = {
        "dir": f"/gfs.{date}/{cycle}/atmos",
        "file": file_name,
        # global domain
        "leftlon": 0,
        "rightlon": 360,
        "toplat": 90,
        "bottomlat": -90,
        # variables: total precipitation
        "var_APCP": "on",
        "lev_surface": "on",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"⬇️ Downloading f{fhr:03d} from {date} {cycle}z (try {attempt})")
            r = requests.get(base_url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            file_path = f"gfs_{cycle}_f{fhr:03d}.grb2"
            with open(file_path, "wb") as f:
                f.write(r.content)
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"   ✔ Saved {file_path} ({size_mb:.1f} MB)")
            return file_path
        except Exception as e:
            print(f"   ⚠ Download failed: {e}")
            time.sleep(2 ** attempt)

    print(f"   ❌ Giving up on f{fhr:03d}")
    return None


def load_gfs_apcp_grid(forecast_hours):
    """
    Download and load APCP (total precipitation) for all steps up to forecast_hours.

    Returns:
        apcp: np.array [T, Y, X] in mm (kg/m^2), cumulative
        lats, lons: 2D arrays
        times: list of datetime (UTC) for each step
        ref_time: datetime (UTC) of the cycle start
    """
    date, cycle, prev_date, prev_cycle = get_latest_cycle()
    ref_time = datetime.strptime(date + cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)

    steps = get_forecast_steps(forecast_hours)

    apcp_list = []
    times = []
    lats, lons = None, None

    for fhr in steps:
        file_path = download_gfs_file(date, cycle, fhr)
        if file_path is None:
            file_path = download_gfs_file(prev_date, prev_cycle, fhr)
            if file_path is None:
                print(f"⚠ Skipping forecast hour {fhr}")
                continue

        grb = pygrib.open(file_path)
        try:
            msg = grb.select(name="Total Precipitation")[0]
        except ValueError:
            print(f"⚠ No APCP in {file_path}, skipping")
            grb.close()
            os.remove(file_path)
            continue

        vals = msg.values.astype("float32")  # cumulative kg/m^2 ≈ mm
        apcp_list.append(vals)

        if lats is None:
            lats, lons = msg.latlons()

        times.append(msg.validDate.replace(tzinfo=timezone.utc))
        grb.close()
        os.remove(file_path)

    if not apcp_list:
        return None, None, None, None

    apcp = np.stack(apcp_list)  # [T, Y, X]
    return apcp, lats, lons, times, ref_time


# --------------------------------
# Hourly rain time series & JSON export
# --------------------------------
def compute_hourly_rain_series(apcp):
    """
    Given APCP cumulative field [T,Y,X] (mm), compute per-hour
    increments for each time step.

    Returns:
        inc: np.array [T,Y,X] where inc[t] = rain in the hour (t-1, t]
    """
    T, NY, NX = apcp.shape
    inc = np.zeros_like(apcp, dtype="float32")

    # first hour: just the first cumulative field (>=0)
    inc[0] = np.maximum(apcp[0], 0.0)
    for t in range(1, T):
        diff = apcp[t] - apcp[t - 1]
        inc[t] = np.maximum(diff, 0.0)

    return inc


def timeseries_to_json(hourly_rain, lats, lons, ref_time, times):
    """
    Build an Earth-style JSON grid: header + flattened time-series data.

    hourly_rain: [T,Y,X] mm for each hour.
    times: list of datetime (UTC) for each T.
    """
    T, ny, nx = hourly_rain.shape
    lat_axis = lats[:, 0]
    lon_axis = lons[0, :]

    # iso strings for each time step
    time_strs = [t.isoformat().replace("+00:00", "Z") for t in times]

    header = {
        "parameter": "rain_1h_mm",
        "parameterUnit": "mm",
        "nx": int(nx),
        "ny": int(ny),
        "tCount": int(T),
        "tStepHours": 1,
        "lo1": float(lon_axis[0]),
        "la1": float(lat_axis[0]),          # usually 90
        "lo2": float(lon_axis[-1]),
        "la2": float(lat_axis[-1]),         # usually -90
        "dx": float(lon_axis[1] - lon_axis[0]),
        "dy": float(lat_axis[0] - lat_axis[1]),  # positive step in degrees
        "forecastHours": FORECAST_HOURS,
        "refTime": ref_time.isoformat().replace("+00:00", "Z"),
        "times": time_strs,
        "source": "NOAA GFS " + GFS_RES,
        "layout": "time-major",   # T blocks, each of size ny*nx
    }

    # Flatten time-major: for t in 0..T-1, concatenate hourly_rain[t].flatten()
    flat = []
    for t in range(T):
        flat.extend(hourly_rain[t].flatten().round(2).tolist())

    return {
        "header": header,
        "data": flat,
    }


# --------------------------------
# Isoline generation (contours) from hourly_rain
# --------------------------------
def generate_isolines_geojson(hourly_rain, lats, lons, levels_mm, hour_index, out_path):
    """
    Create contour lines (isobars) from a single hour slice and write GeoJSON.

    hourly_rain: [T, Y, X] array (mm)
    lats, lons : 2D arrays (Y, X)
    levels_mm  : list of rainfall thresholds in mm
    hour_index : which time slice to use (0..T-1)
    out_path   : output GeoJSON path
    """
    T, NY, NX = hourly_rain.shape

    if T == 0:
        print("⚠ No time steps in hourly_rain; skipping isolines.")
        return

    h = max(0, min(int(hour_index), T - 1))

    field = hourly_rain[h]  # [NY, NX]
    # Convert to numpy array of float32 just in case
    field = np.array(field, dtype="float32")

    # lats/lons already shape [NY, NX]
    lon_grid = np.array(lons, dtype="float32")
    lat_grid = np.array(lats, dtype="float32")

    print(f"   Generating isolines for hour index {h}, levels={levels_mm}")

    # Create contours with matplotlib
    fig, ax = plt.subplots(figsize=(6, 3))
    cs = ax.contour(lon_grid, lat_grid, field, levels=levels_mm)

    features = []
    for level, collection in zip(cs.levels, cs.collections):
        for path in collection.get_paths():
            vertices = path.vertices  # Nx2 array (lon, lat)
            if len(vertices) < 2:
                continue
            coords = vertices.tolist()
            coords = [[float(x), float(y)] for x, y in coords]

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {
                    "level": float(level),
                    "hourIndex": int(h)
                }
            })

    plt.close(fig)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    print(f"✅ Wrote {out_path} with {len(features)} contour lines")


# --------------------------------
# Main
# --------------------------------
def main():
    print(f"🌧  FloodLink GFS rain time series export – next {FORECAST_HOURS}h")

    apcp, lats, lons, times, ref_time = load_gfs_apcp_grid(FORECAST_HOURS)
    if apcp is None:
        print("❌ Failed to load APCP grid.")
        return

    print(f"   Cumulative APCP grid shape: {apcp.shape} (T, Y, X)")
    print(f"   First valid time: {times[0].isoformat()}")
    print(f"   Last  valid time: {times[-1].isoformat()}")

    hourly_rain = compute_hourly_rain_series(apcp)
    print(
        f"   Hourly rain range: "
        f"{float(hourly_rain.min()):.2f} – {float(hourly_rain.max()):.2f} mm"
    )

    # --- JSON time series export ---
    grid_json = timeseries_to_json(hourly_rain, lats, lons, ref_time, times)

    with open(RAINFIELD_PATH, "w", encoding="utf-8") as f:
        json.dump(grid_json, f, ensure_ascii=False)

    size_mb = os.path.getsize(RAINFIELD_PATH) / 1024 / 1024
    print(f"✅ Wrote {RAINFIELD_PATH} ({size_mb:.1f} MB)")

    # --- Isoline (isobar-style) export ---
    generate_isolines_geojson(
        hourly_rain=hourly_rain,
        lats=lats,
        lons=lons,
        levels_mm=ISO_LEVELS_MM,
        hour_index=ISO_HOUR_INDEX,
        out_path=ISOLINES_PATH,
    )


if __name__ == "__main__":
    main()
