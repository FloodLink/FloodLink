"""
FloodLink – NOAA GFS → Global 6h Rain Grid (JSON)

Downloads GFS 0.25° GRIB2, computes 6-hour accumulated rain
for the whole globe, and writes a compact JSON grid:

  gfs_rain_6h.json

This file will later be used by the FloodLink Earth-style
visualization (canvas/WebGL).
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
import pygrib

# --------------------------------
# CONFIG
# --------------------------------
GFS_RES = "0p25"             # 0.25° grid
FORECAST_HOURS = 6           # next 6 hours
MAX_RETRIES = 2
TIMEOUT = 60                 # seconds
RAINFIELD_PATH = "gfs_rain_6h.json"


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
    For 0.25° GFS, forecasts are hourly → 1..max_hours inclusive.
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
        apcp: np.array [time, lat, lon] in mm (kg/m^2)
        lats, lons: 2D arrays
        times: list of datetime (UTC)
        ref_time: datetime (UTC) of the cycle
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
# Rain accumulation & JSON export
# --------------------------------
def compute_6h_rain(apcp):
    """
    Given APCP cumulative field [T,Y,X] (mm), compute sum of
    1-hour increments over all T steps (up to 6h).

    Returns:
        rain_6h: [Y,X] mm
    """
    T, NY, NX = apcp.shape
    inc = np.zeros_like(apcp)

    # first hour: just the first cumulative field (>=0)
    inc[0] = np.maximum(apcp[0], 0.0)
    for t in range(1, T):
        diff = apcp[t] - apcp[t - 1]
        inc[t] = np.maximum(diff, 0.0)

    rain_6h = np.sum(inc, axis=0)  # [Y,X]
    return rain_6h


def grid_to_json(rain_6h, lats, lons, ref_time, valid_time):
    """
    Build an Earth-style JSON grid: header + flattened data.
    """
    ny, nx = rain_6h.shape
    lat_axis = lats[:, 0]
    lon_axis = lons[0, :]

    header = {
        "parameter": "rain_6h_mm",
        "parameterUnit": "mm",
        "nx": int(nx),
        "ny": int(ny),
        "lo1": float(lon_axis[0]),
        "la1": float(lat_axis[0]),       # usually 90
        "lo2": float(lon_axis[-1]),
        "la2": float(lat_axis[-1]),      # usually -90
        "dx": float(lon_axis[1] - lon_axis[0]),
        "dy": float(lat_axis[0] - lat_axis[1]),  # positive step in degrees
        "forecastHours": FORECAST_HOURS,
        "refTime": ref_time.isoformat().replace("+00:00", "Z"),
        "validTime": valid_time.isoformat().replace("+00:00", "Z"),
        "source": "NOAA GFS " + GFS_RES,
    }

    # Flatten row-major (lat index 0..ny-1, lon index 0..nx-1)
    data = rain_6h.flatten().round(2).tolist()

    return {
        "header": header,
        "data": data,
    }


def main():
    print(f"🌧  FloodLink GFS rain field export – next {FORECAST_HOURS}h")

    apcp, lats, lons, times, ref_time = load_gfs_apcp_grid(FORECAST_HOURS)
    if apcp is None:
        print("❌ Failed to load APCP grid.")
        return

    print(f"   Grid shape: {apcp.shape} (T, Y, X)")
    print(f"   First valid time: {times[0].isoformat()}")
    print(f"   Last  valid time: {times[-1].isoformat()}")

    rain_6h = compute_6h_rain(apcp)
    print(f"   Rain range: {float(rain_6h.min()):.2f} – {float(rain_6h.max()):.2f} mm")

    grid_json = grid_to_json(rain_6h, lats, lons, ref_time, times[-1])

    with open(RAINFIELD_PATH, "w", encoding="utf-8") as f:
        json.dump(grid_json, f, ensure_ascii=False)

    size_mb = os.path.getsize(RAINFIELD_PATH) / 1024 / 1024
    print(f"✅ Wrote {RAINFIELD_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
