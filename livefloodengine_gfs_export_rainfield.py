"""
FloodLink – NOAA GFS → Global 1h Rainfall Time Series (JSON)
+ Precomputed Isocurves (GeoJSON, all hours)
+ Per-hour PNG data textures
+ 6h Cumulative grid JSON (compact)

Outputs:
  1) gfs_rain_6h.json                 (time-series grid, 1h steps, up to FORECAST_HOURS)
  2) gfs_rain_6h_cum.json             (cumulative over window, compact grid, tCount=1)
  3) gfs_rain_isolines_6h.geojson     (LineString contours for each hour)
  4) gfs_rain_6h_textures/
       ├─ rain_t000.png
       ├─ rain_t001.png
       ├─ ...
       └─ gfs_rain_6h_textures_meta.json
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
import pygrib

# For contour/isoline generation
import matplotlib
matplotlib.use("Agg")  # headless backend for GitHub Actions
import matplotlib.pyplot as plt

# For PNG texture export
from PIL import Image

# --------------------------------
# CONFIG
# --------------------------------
GFS_RES = "0p25"                 # 0.25° grid
FORECAST_HOURS = 6               # next N hours (window from *now*)
MAX_RETRIES = 2
TIMEOUT = 60                     # seconds

RAINFIELD_PATH = "gfs_rain_6h.json"              # time series, 1h increments
CUMFIELD_PATH  = "gfs_rain_6h_cum.json"          # cumulative over window, compact grid (tCount=1)
ISOLINES_PATH  = "gfs_rain_isolines_6h.geojson"  # precomputed isocurves for all hours

# Directory for PNG data textures
TEXTURE_DIR = Path("gfs_rain_6h_textures")

# Max rain value encoded into 0..255 in PNGs (mm/hour)
MAX_RAIN_MM_FOR_TEXTURE = 80.0

# Rain thresholds (mm) for contour lines
ISO_LEVELS_MM = [0.25, 1, 5, 10, 20, 40, 80]

# Isoline smoothing / compression knobs
SMOOTH_ITERATIONS = 1   # 0 = no smoothing, 1 = light, 2 = stronger
DECIMATE_STEP     = 2   # keep every Nth vertex after smoothing
COORD_DECIMALS    = 3   # round lon/lat to this many decimals


# --------------------------------
# Smoothing helper (Chaikin)
# --------------------------------
def chaikin_smooth(coords, iterations=2):
    """
    Simple Chaikin corner-cutting algorithm to smooth a polyline.

    coords: list of (x, y) pairs.
    iterations: how many smoothing passes (1–3 is usually enough).

    Returns a new list of (x, y) pairs.
    """
    if len(coords) < 3:
        return coords

    new_coords = coords
    for _ in range(iterations):
        if len(new_coords) < 3:
            break
        smoothed = [new_coords[0]]  # keep first point
        for i in range(len(new_coords) - 1):
            x0, y0 = new_coords[i]
            x1, y1 = new_coords[i + 1]

            # Q is closer to P0, R is closer to P1
            qx = 0.75 * x0 + 0.25 * x1
            qy = 0.75 * y0 + 0.25 * y1
            rx = 0.25 * x0 + 0.75 * x1
            ry = 0.25 * y0 + 0.75 * y1

            smoothed.append((qx, qy))
            smoothed.append((rx, ry))

        smoothed.append(new_coords[-1])  # keep last point
        new_coords = smoothed

    return new_coords


# --------------------------------
# GFS helpers
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


def get_forecast_steps_for_now(
    forecast_hours: int,
    cycle_dt_utc: datetime,
    now_utc: datetime | None = None,
    gfs_res: str = "0p25"
):
    """
    Return forecast steps that cover the NEXT `forecast_hours` from *now*,
    relative to the chosen cycle datetime.

    For 0p25: hourly steps (f001, f002, ...)
    For 0p50/1p00: 3-hourly steps (f003, f006, ...)
    """
    if now_utc is None:
        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)

    lead_hours = (now_utc - cycle_dt_utc).total_seconds() / 3600.0

    if gfs_res in ("0p50", "1p00"):
        step = 3
        start_fhr = max(step, int(np.ceil(lead_hours / step)) * step)
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
            if size_mb < 0.05:
                print(f"   ⚠ Saved {file_path} but it is tiny ({size_mb:.2f} MB)")

            print(f"   ✔ Saved {file_path} ({size_mb:.2f} MB)")
            return file_path

        except Exception as e:
            print(f"   ⚠ Download failed: {e}")
            time.sleep(2 ** attempt)

    print(f"   ❌ Giving up on f{fhr:03d}")
    return None


def load_gfs_apcp_grid(forecast_hours: int):
    """
    Load cumulative APCP (tp) for the NEXT `forecast_hours` from *now*.
    Includes a baseline step (previous hour) so the first increment is correct.

    Returns:
        apcp: np.array [T, Y, X] cumulative (includes baseline at index 0 if present)
        lats, lons: 2D arrays
        times: list of datetime (UTC, tz-aware) aligned to apcp
        ref_time: cycle start datetime (UTC, tz-aware)
        meta: dict with window_steps / baseline_step / has_baseline / steps_used
    """
    date, cycle, prev_date, prev_cycle = get_latest_cycle()
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)

    cycle_dt = datetime.strptime(date + cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    prev_cycle_dt = datetime.strptime(prev_date + prev_cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)

    window_steps = get_forecast_steps_for_now(forecast_hours, cycle_dt, now_utc, gfs_res=GFS_RES)
    if not window_steps:
        return None, None, None, None, None, None

    baseline_step = None
    if GFS_RES in ("0p50", "1p00"):
        if window_steps[0] > 3:
            baseline_step = window_steps[0] - 3
    else:
        if window_steps[0] > 1:
            baseline_step = window_steps[0] - 1

    probe_fhr = window_steps[0]
    probe_file = download_gfs_file(date, cycle, probe_fhr)

    use_date, use_cycle, use_cycle_dt = date, cycle, cycle_dt

    if probe_file is None:
        probe_file_prev = download_gfs_file(prev_date, prev_cycle, probe_fhr)
        if probe_file_prev is None:
            print(f"❌ Could not download probe f{probe_fhr:03d} from latest or previous cycle.")
            return None, None, None, None, None, None

        use_date, use_cycle, use_cycle_dt = prev_date, prev_cycle, prev_cycle_dt
        try:
            os.remove(probe_file_prev)
        except Exception:
            pass

        window_steps = get_forecast_steps_for_now(forecast_hours, use_cycle_dt, now_utc, gfs_res=GFS_RES)
        baseline_step = None
        if GFS_RES in ("0p50", "1p00"):
            if window_steps and window_steps[0] > 3:
                baseline_step = window_steps[0] - 3
        else:
            if window_steps and window_steps[0] > 1:
                baseline_step = window_steps[0] - 1
    else:
        try:
            os.remove(probe_file)
        except Exception:
            pass

    steps_to_download = ([baseline_step] if baseline_step is not None else []) + window_steps

    print(
        f"🛰️ Using GFS cycle: {use_date} {use_cycle}Z | "
        f"window: f{window_steps[0]:03d}..f{window_steps[-1]:03d}"
        + (f" | baseline: f{baseline_step:03d}" if baseline_step is not None else "")
    )

    apcp_list = []
    times = []
    steps_used = []
    lats, lons = None, None

    for fhr in steps_to_download:
        file_path = download_gfs_file(use_date, use_cycle, fhr)
        if file_path is None:
            print(f"⚠ Skipping f{fhr:03d} (download failed)")
            continue
        grb = pygrib.open(file_path)
        try:
            tp_msgs = grb.select(shortName="tp")
            msg = next((m for m in tp_msgs if int(getattr(m, "endStep", -1)) == fhr), tp_msgs[0])
        except Exception:
            try:
                tp_msgs = grb.select(name="Total Precipitation")
                msg = next((m for m in tp_msgs if int(getattr(m, "endStep", -1)) == fhr), tp_msgs[0])
            except Exception as e:
                print(f"⚠ No tp/APCP in {file_path}: {e}")
                grb.close()
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                continue

        # ---- DEBUG: print GRIB metadata for tp/APCP ----
        name      = getattr(msg, "name", None)
        shortName = getattr(msg, "shortName", None)
        units     = getattr(msg, "units", None)

        startStep = getattr(msg, "startStep", None)
        endStep   = getattr(msg, "endStep", None)
        stepRange = getattr(msg, "stepRange", None)
        stepType  = getattr(msg, "stepType", None)

        print(f"   📌 tp id  f{fhr:03d}: name={name} / shortName={shortName}")
        print(
            f"   🌧 tp meta f{fhr:03d}: units={units} "
            f"startStep={startStep} endStep={endStep} "
            f"stepRange={stepRange} stepType={stepType}"
        )

        # ---- Read values (and OPTIONAL unit fix if units are meters) ----
        vals = msg.values.astype("float32")

        raw_units = getattr(msg, "units", None)
        u = (str(raw_units).strip().lower() if raw_units is not None else "")

        # normalize common formatting differences
        u_norm = (
            u.replace("**", "^")
             .replace("−", "-")
             .replace(" ", "")
        )

        # Helpful debug (so you can see exactly what string you're matching)
        print(f"   🔎 units raw='{u}' | norm='{u_norm}'")

        # If NOAA gives meters of water equivalent, convert to mm
        is_meters = (u_norm in ("m", "meter", "meters")) or ("mofwaterequivalent" in u_norm)
        if is_meters:
            print(f"   ⚠ Converting tp from meters to mm for f{fhr:03d}")
            vals *= 1000.0

        # If units are kg/m^2 (or kg m-2), that's effectively mm of water (NO conversion needed)
        is_kg_m2 = ("kg" in u_norm) and (
            ("m-2" in u_norm) or ("m^-2" in u_norm) or ("m^(-2)" in u_norm) or
            ("/m^2" in u_norm) or ("/m2" in u_norm)
        )
        if is_kg_m2:
            print(f"   ℹ️ units look like kg/m² (treat as mm) for f{fhr:03d}")

        # If it's something unexpected, flag it loudly
        if (not is_meters) and (not is_kg_m2) and u_norm not in ("", "mm"):
            print(f"   ⚠️ Unrecognized tp units '{raw_units}' for f{fhr:03d} — check conversion logic.")

        # quick sanity
        print(
            f"   ✅ tp values f{fhr:03d}: "
            f"min={float(vals.min()):.4f} "
            f"mean={float(vals.mean()):.4f} "
            f"max={float(vals.max()):.4f}"
        )

        apcp_list.append(vals)

        if lats is None:
            lats, lons = msg.latlons()

        times.append(use_cycle_dt + timedelta(hours=fhr))
        steps_used.append(fhr)

        grb.close()
        try:
            os.remove(file_path)
        except Exception:
            pass


    if not apcp_list:
        return None, None, None, None, None, None

    apcp = np.stack(apcp_list)  # [T,Y,X] cumulative
    ref_time = use_cycle_dt

    has_baseline = (
        baseline_step is not None
        and len(steps_used) >= 2
        and steps_used[0] == baseline_step
        and steps_used[1] == window_steps[0]
    )

    meta = {
        "window_steps": window_steps,
        "baseline_step": baseline_step,
        "has_baseline": has_baseline,
        "steps_used": steps_used,
    }

    return apcp, lats, lons, times, ref_time, meta


# --------------------------------
# Hourly rain time series & JSON export
# --------------------------------
def compute_hourly_rain_series(apcp: np.ndarray, has_baseline: bool):
    """
    apcp: cumulative [T,Y,X] (includes baseline at index 0 if has_baseline)
    Returns:
      hourly_inc_window: [W,Y,X] mm/hour for the window only
      start_index: int (where the real window begins in the apcp array)
    """
    T, NY, NX = apcp.shape
    inc = np.zeros_like(apcp, dtype="float32")

    if not has_baseline:
        inc[0] = np.maximum(apcp[0], 0.0)

    for t in range(1, T):
        inc[t] = np.maximum(apcp[t] - apcp[t - 1], 0.0)

    start_i = 1 if has_baseline else 0
    return inc[start_i:], start_i


def timeseries_to_json(hourly_rain, lats, lons, ref_time, times):
    """
    Build an Earth-style JSON grid: header + flattened time-series data.
    hourly_rain: [T,Y,X] mm for each hour.
    times: list of datetime (UTC) for each T.
    """
    T, ny, nx = hourly_rain.shape
    lat_axis = lats[:, 0]
    lon_axis = lons[0, :]

    time_strs = [t.isoformat().replace("+00:00", "Z") for t in times]

    header = {
        "parameter": "rain_1h_mm",
        "parameterUnit": "mm",
        "nx": int(nx),
        "ny": int(ny),
        "tCount": int(T),
        "tStepHours": 1,
        "lo1": float(lon_axis[0]),
        "la1": float(lat_axis[0]),
        "lo2": float(lon_axis[-1]),
        "la2": float(lat_axis[-1]),
        "dx": float(lon_axis[1] - lon_axis[0]),
        "dy": float(lat_axis[0] - lat_axis[1]),
        "forecastHours": FORECAST_HOURS,
        "refTime": ref_time.isoformat().replace("+00:00", "Z"),
        "times": time_strs,
        "source": "NOAA GFS " + GFS_RES,
        "layout": "time-major",
    }

    flat = []
    for t in range(T):
        flat.extend(hourly_rain[t].flatten().round(2).tolist())

    return {"header": header, "data": flat}


def cumulative_to_grid_json(hourly_rain, lats, lons, ref_time, times_window, include_max_1h: bool = True):
    """
    Export 6h cumulative rain in the same compact grid format as gfs_rain_6h.json,
    but with tCount=1.
    """
    T, ny, nx = hourly_rain.shape
    lat_axis = lats[:, 0]
    lon_axis = lons[0, :]

    cum = np.maximum(hourly_rain, 0.0).sum(axis=0).astype("float32")  # [ny, nx]

    max1h = None
    if include_max_1h:
        max1h = np.maximum(hourly_rain, 0.0).max(axis=0).astype("float32")

    valid_start = times_window[0].isoformat().replace("+00:00", "Z") if times_window else None
    valid_end   = times_window[-1].isoformat().replace("+00:00", "Z") if times_window else None

    header = {
        "parameter": "rain_6h_mm",
        "parameterUnit": "mm",
        "nx": int(nx),
        "ny": int(ny),
        "tCount": 1,
        "tStepHours": int(T),
        "lo1": float(lon_axis[0]),
        "la1": float(lat_axis[0]),
        "lo2": float(lon_axis[-1]),
        "la2": float(lat_axis[-1]),
        "dx": float(lon_axis[1] - lon_axis[0]),
        "dy": float(lat_axis[0] - lat_axis[1]),
        "forecastHours": int(T),
        "refTime": ref_time.isoformat().replace("+00:00", "Z"),
        "times": [valid_end] if valid_end else [],
        "source": "NOAA GFS " + GFS_RES,
        "layout": "time-major",
        "validStart": valid_start,
        "validEnd": valid_end,
    }

    # data: [1, ny, nx] flattened (time-major with only one block)
    data = cum.round(2).flatten().tolist()

    out = {"header": header, "data": data}

    if include_max_1h and max1h is not None:
        out["max1h_header"] = {
            "parameter": "rain_1h_max_mm",
            "parameterUnit": "mm",
            "nx": int(nx),
            "ny": int(ny),
            "tCount": 1,
            "layout": "time-major",
            "note": "Peak 1-hour rainfall within the same window."
        }
        out["max1h_data"] = max1h.round(2).flatten().tolist()

    return out


# --------------------------------
# Isoline generation (contours) – all hours
# --------------------------------
def generate_isolines_all_hours(hourly_rain, lats, lons, times, levels_mm, out_path):
    """
    Create contour lines (isocurves) for each hour slice and write one GeoJSON.
    """
    T, NY, NX = hourly_rain.shape
    if T == 0:
        print("⚠ No time steps in hourly_rain; skipping isolines.")
        return

    lon_grid = np.array(lons, dtype="float32")
    lat_grid = np.array(lats, dtype="float32")

    features = []

    print(f"   Generating isolines for all {T} hours, levels={levels_mm}")
    print(f"   Smoothing={SMOOTH_ITERATIONS} iter, decimate_step={DECIMATE_STEP}, coord_decimals={COORD_DECIMALS}")

    for h in range(T):
        field = np.array(hourly_rain[h], dtype="float32")
        valid_time = times[h].isoformat().replace("+00:00", "Z")

        fig, ax = plt.subplots(figsize=(6, 3))
        cs = ax.contour(lon_grid, lat_grid, field, levels=levels_mm)

        for level, seglist in zip(cs.levels, cs.allsegs):
            for seg in seglist:
                if len(seg) < 2:
                    continue

                seg_coords = list(seg)

                if SMOOTH_ITERATIONS > 0:
                    seg_coords = chaikin_smooth(seg_coords, iterations=SMOOTH_ITERATIONS)

                if DECIMATE_STEP > 1:
                    seg_coords = seg_coords[::DECIMATE_STEP]

                if len(seg_coords) < 2:
                    continue

                coords = [
                    [round(float(x), COORD_DECIMALS), round(float(y), COORD_DECIMALS)]
                    for x, y in seg_coords
                ]

                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"level": float(level), "hourIndex": int(h), "validTime": valid_time}
                })

        plt.close(fig)

    geojson = {"type": "FeatureCollection", "features": features}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"✅ Wrote {out_path} with {len(features)} contour lines ({size_mb:.2f} MB)")


# --------------------------------
# PNG data textures export
# --------------------------------
def export_png_textures(hourly_rain, times, out_dir: Path,
                        max_rain_mm: float = MAX_RAIN_MM_FOR_TEXTURE,
                        flip_y: bool = True):
    """
    Export each hour slice as a grayscale PNG data texture.
    """
    T, ny, nx = hourly_rain.shape
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Exporting PNG textures to {out_dir} ...")
    print(f"   Texture shape per frame: {nx}x{ny}, frames={T}, max_rain_mm={max_rain_mm}")

    time_strs = [t.isoformat().replace("+00:00", "Z") for t in times]

    meta = {
        "nx": int(nx),
        "ny": int(ny),
        "tCount": int(T),
        "maxRainMm": float(max_rain_mm),
        "times": time_strs,
        "flipY": bool(flip_y),
        "note": (
            "Each rain_t###.png is grayscale: value / 255 * maxRainMm = mm/hour "
            "(values above maxRainMm were clamped)."
        ),
    }

    for t in range(T):
        slice_mm = np.array(hourly_rain[t], dtype=np.float32)
        slice_clamped = np.clip(slice_mm, 0.0, max_rain_mm) / max_rain_mm
        img_8 = (slice_clamped * 255.0).round().astype(np.uint8)

        if flip_y:
            img_8 = np.flipud(img_8)

        img = Image.fromarray(img_8, mode="L")

        out_name = f"rain_t{t:03d}.png"
        out_path = out_dir / out_name
        img.save(out_path, format="PNG")

        if t < 3 or t == T - 1:
            t_str = time_strs[t] if t < len(time_strs) else f"t={t}"
            print(f"      🖼  Saved {out_name}  (time={t_str})")

    meta_path = out_dir / "gfs_rain_6h_textures_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"   📄 Wrote texture metadata → {meta_path}")


# --------------------------------
# Main
# --------------------------------
def main():
    print(f"🌧  FloodLink GFS rain time series export – next {FORECAST_HOURS}h (from now)")

    apcp, lats, lons, times, ref_time, meta = load_gfs_apcp_grid(FORECAST_HOURS)
    if apcp is None:
        print("❌ Failed to load APCP grid.")
        return

  
    hourly_rain, start_i = compute_hourly_rain_series(apcp, has_baseline=meta["has_baseline"])
    times_window = times[start_i:]

    print(f"   Cumulative APCP grid shape: {apcp.shape} (T, Y, X)")
    print(f"   Steps used: {meta.get('steps_used')}")
    print(f"   Has baseline: {meta.get('has_baseline')} (baseline={meta.get('baseline_step')})")
    print(f"   First window time: {times_window[0].isoformat()}")
    print(f"   Last  window time: {times_window[-1].isoformat()}")

    # ✅ extra sanity check for the produced hourly increments (this is what your map is visualizing)
    print(
        f"   ✅ hourly_inc stats: "
        f"min={float(hourly_rain.min()):.4f} "
        f"mean={float(hourly_rain.mean()):.4f} "
        f"max={float(hourly_rain.max()):.4f}"
    )

    print(
        f"   Hourly rain range: "
        f"{float(hourly_rain.min()):.2f} – {float(hourly_rain.max()):.2f} mm"
    )

    # --- JSON time series export (WINDOW ONLY) ---
    grid_json = timeseries_to_json(hourly_rain, lats, lons, ref_time, times_window)
    with open(RAINFIELD_PATH, "w", encoding="utf-8") as f:
        json.dump(grid_json, f, ensure_ascii=False)
    size_mb = os.path.getsize(RAINFIELD_PATH) / 1024 / 1024
    print(f"✅ Wrote {RAINFIELD_PATH} ({size_mb:.2f} MB)")

    # --- 6H cumulative compact grid (WINDOW ONLY) ---
    cum_json = cumulative_to_grid_json(
        hourly_rain=hourly_rain,
        lats=lats,
        lons=lons,
        ref_time=ref_time,
        times_window=times_window,
        include_max_1h=True
    )
    with open(CUMFIELD_PATH, "w", encoding="utf-8") as f:
        json.dump(cum_json, f, ensure_ascii=False)
    cum_mb = os.path.getsize(CUMFIELD_PATH) / 1024 / 1024
    print(f"✅ Wrote {CUMFIELD_PATH} ({cum_mb:.2f} MB)")

    # --- Isolines GeoJSON for all hours (WINDOW ONLY) ---
    generate_isolines_all_hours(
        hourly_rain=hourly_rain,
        lats=lats,
        lons=lons,
        times=times_window,
        levels_mm=ISO_LEVELS_MM,
        out_path=ISOLINES_PATH,
    )

    # --- PNG data textures (WINDOW ONLY) ---
    export_png_textures(
        hourly_rain=hourly_rain,
        times=times_window,
        out_dir=TEXTURE_DIR,
        max_rain_mm=MAX_RAIN_MM_FOR_TEXTURE,
        flip_y=True,
    )


if __name__ == "__main__":
    main()
