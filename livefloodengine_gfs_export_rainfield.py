"""
FloodLink – NOAA GFS → Global 1h Rainfall Time Series (JSON)
+ 6h Cumulative grid JSON (compact)

Outputs:
  1) gfs_rain_6h.json      (time-series grid, 1h steps, up to FORECAST_HOURS)
  2) gfs_rain_6h_cum.json  (cumulative over window, compact grid, tCount=1)

NOTE:
- Isolines + PNG textures have been removed.
- Optional cleanup deletes old isoline/texture artifacts so they can be removed from the repo on commit.
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
import pygrib


# --------------------------------
# CONFIG
# --------------------------------
GFS_RES = "0p25"                 # 0.25° grid
FORECAST_HOURS = 6               # next N hours (window from *now*)
MAX_RETRIES = 2
TIMEOUT = 60                     # seconds

RAINFIELD_PATH = "gfs_rain_6h.json"       # time series, 1h increments
CUMFIELD_PATH  = "gfs_rain_6h_cum.json"   # cumulative over window, compact grid (tCount=1)

# Put downloads in a dedicated temp folder to reduce collisions
TMP_DIR = Path("_tmp_gfs_gribs")


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
        "leftlon": 0, "rightlon": 360,
        "toplat": 90, "bottomlat": -90,
        "var_APCP": "on",
        "lev_surface": "on",
    }

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    file_path = TMP_DIR / f"gfs_{date}_{cycle}_f{fhr:03d}.grb2"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"⬇️ Downloading f{fhr:03d} from {date} {cycle}z (try {attempt})")
            r = requests.get(base_url, params=params, timeout=TIMEOUT)
            r.raise_for_status()

            with file_path.open("wb") as f:
                f.write(r.content)

            size_mb = file_path.stat().st_size / 1024 / 1024
            if size_mb < 0.05:
                print(f"   ⚠ Saved {file_path} but it is tiny ({size_mb:.2f} MB)")

            print(f"   ✔ Saved {file_path} ({size_mb:.2f} MB)")
            return str(file_path)

        except Exception as e:
            print(f"   ⚠ Download failed: {e}")
            time.sleep(2 ** attempt)

    print(f"   ❌ Giving up on f{fhr:03d}")
    return None


def load_gfs_apcp_grid(forecast_hours: int):
    """
    Load cumulative APCP (tp) for the NEXT `forecast_hours` from *now*.
    Includes a baseline step (previous hour) so the first increment is correct.

    IMPORTANT:
    GFS can change accumulation origin (e.g., 0-6 then 6-7 etc).
    We return startStep/endStep per frame to compute true 1h increments robustly.

    Returns:
        apcp: np.array [T, Y, X] cumulative
        lats, lons: 2D arrays
        times: list of datetime (UTC) aligned to each frame's endStep
        ref_time: cycle start datetime (UTC)
        meta: dict including window_steps, baseline_step, has_baseline, steps_used, start_steps, end_steps
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

    frames = []  # (startStep, endStep, vals, frame_time_utc)
    lats, lons = None, None

    def _parse_step_range(msg, fallback_end: int):
        """Return (startStep, endStep) as ints, best-effort."""
        s = getattr(msg, "startStep", None)
        e = getattr(msg, "endStep", None)

        try:
            s_int = int(s) if s is not None else None
        except Exception:
            s_int = None
        try:
            e_int = int(e) if e is not None else None
        except Exception:
            e_int = None

        if s_int is not None and e_int is not None:
            return s_int, e_int

        sr = getattr(msg, "stepRange", None)
        if isinstance(sr, str) and "-" in sr:
            try:
                a, b = sr.split("-", 1)
                return int(a), int(b)
            except Exception:
                pass

        return 0, int(fallback_end)

    for fhr in steps_to_download:
        file_path = download_gfs_file(use_date, use_cycle, fhr)
        if file_path is None:
            print(f"⚠ Skipping f{fhr:03d} (download failed)")
            continue

        grb = pygrib.open(file_path)
        try:
            try:
                tp_msgs = grb.select(shortName="tp")
                msg = next((m for m in tp_msgs if int(getattr(m, "endStep", -1)) == fhr), tp_msgs[0])
            except Exception:
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

        vals = msg.values.astype("float32")

        raw_units = getattr(msg, "units", None)
        u = (str(raw_units).strip().lower() if raw_units is not None else "")
        u_norm = (
            u.replace("**", "^")
             .replace("−", "-")
             .replace(" ", "")
        )
        print(f"   🔎 units raw='{u}' | norm='{u_norm}'")

        is_meters = (u_norm in ("m", "meter", "meters")) or ("mofwaterequivalent" in u_norm)
        if is_meters:
            print(f"   ⚠ Converting tp from meters to mm for f{fhr:03d}")
            vals *= 1000.0

        is_kg_m2 = ("kg" in u_norm) and (
            ("m-2" in u_norm) or ("m^-2" in u_norm) or ("m^(-2)" in u_norm) or
            ("/m^2" in u_norm) or ("/m2" in u_norm)
        )
        if is_kg_m2:
            print(f"   ℹ️ units look like kg/m² (treat as mm) for f{fhr:03d}")

        if (not is_meters) and (not is_kg_m2) and u_norm not in ("", "mm"):
            print(f"   ⚠️ Unrecognized tp units '{raw_units}' for f{fhr:03d} — check conversion logic.")

        s_step, e_step = _parse_step_range(msg, fallback_end=fhr)

        print(
            f"   ✅ tp values f{fhr:03d}: "
            f"min={float(vals.min()):.4f} "
            f"mean={float(vals.mean()):.4f} "
            f"max={float(vals.max()):.4f}"
        )

        if lats is None:
            lats, lons = msg.latlons()

        frame_time = use_cycle_dt + timedelta(hours=int(e_step))
        frames.append((int(s_step), int(e_step), vals.astype("float32"), frame_time))

        grb.close()
        try:
            os.remove(file_path)
        except Exception:
            pass

    if not frames:
        return None, None, None, None, None, None

    # ✅ FIX: sort frames by endStep so apcp/start/end/times stay aligned even if some downloads were skipped
    frames.sort(key=lambda x: x[1])

    start_steps = [f[0] for f in frames]
    end_steps   = [f[1] for f in frames]
    times       = [f[3] for f in frames]
    apcp        = np.stack([f[2] for f in frames])

    ref_time = use_cycle_dt
    steps_used = end_steps[:]

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
        "start_steps": start_steps,
        "end_steps": end_steps,
    }

    return apcp, lats, lons, times, ref_time, meta

# --------------------------------
# Hourly rain time series & JSON export
# --------------------------------
def compute_hourly_rain_window(
    apcp: np.ndarray,
    start_steps: list[int],
    end_steps: list[int],
    window_end_steps: list[int],
):
    """
    Convert GFS 'tp' accumulations into true 1-hour increments for the desired window endSteps.

    Returns:
        hourly_rain: np.array [W, Y, X] mm/hour for each requested endStep in window_end_steps
        report: list of dicts per requested hour {endStep, quality, method, origin}
                quality: 2=exact 1h, 0=distributed (approx), -1=missing
    """
    if apcp is None or apcp.size == 0:
        return None, []

    # origin -> { endStep -> vals }
    origin_map: dict[int, dict[int, np.ndarray]] = {}
    for i in range(apcp.shape[0]):
        s = int(start_steps[i])
        e = int(end_steps[i])
        origin_map.setdefault(s, {})[e] = apcp[i]

    inc_map: dict[int, np.ndarray] = {}
    q_map: dict[int, int] = {}           # 0=distributed, 2=exact 1h
    origin_used: dict[int, int] = {}     # chosen origin for that hour_end
    method_used: dict[int, str] = {}     # "direct_1h", "delta_1h", "distributed"

    def set_inc(hour_end: int, vals: np.ndarray, quality: int, origin: int, method: str):
        prev_q = q_map.get(hour_end, -999)
        if quality > prev_q:
            inc_map[hour_end] = vals.astype("float32")
            q_map[hour_end] = quality
            origin_used[hour_end] = int(origin)
            method_used[hour_end] = method

    for origin, end_dict in origin_map.items():
        ends = sorted(end_dict.keys())
        if not ends:
            continue

        prev_end = None
        prev_vals = None

        for end in ends:
            vals = end_dict[end].astype("float32")

            if prev_end is None:
                duration = end - origin
                if duration <= 0:
                    prev_end, prev_vals = end, vals
                    continue

                if duration == 1:
                    # already a true 1h accumulation (origin -> end)
                    set_inc(end, np.maximum(vals, 0.0), quality=2, origin=origin, method="direct_1h")
                else:
                    # no previous frame for this origin -> distribute evenly across the span
                    per_h = np.maximum(vals, 0.0) / float(duration)
                    for h in range(origin + 1, end + 1):
                        set_inc(h, per_h, quality=0, origin=origin, method="distributed")

            else:
                duration = end - prev_end
                if duration <= 0:
                    prev_end, prev_vals = end, vals
                    continue

                delta = np.maximum(vals - prev_vals, 0.0)

                if duration == 1:
                    # true 1h increment via differencing
                    set_inc(end, delta, quality=2, origin=origin, method="delta_1h")
                else:
                    # distribute across missing hours (approx)
                    per_h = delta / float(duration)
                    for h in range(prev_end + 1, end + 1):
                        set_inc(h, per_h, quality=0, origin=origin, method="distributed")

            prev_end, prev_vals = end, vals

    # Build output aligned to requested window endSteps
    W = len(window_end_steps)
    ny, nx = apcp.shape[1], apcp.shape[2]
    out = np.zeros((W, ny, nx), dtype="float32")

    report = []
    missing = []
    approx = []
    exact = []

    for i, h in enumerate(window_end_steps):
        q = q_map.get(h, -1)
        if h in inc_map:
            out[i] = inc_map[h]
        else:
            missing.append(h)

        entry = {
            "endStep": int(h),
            "quality": int(q),
            "method": method_used.get(h, "missing"),
            "origin": origin_used.get(h, None),
        }
        report.append(entry)

        if q == 2:
            exact.append(h)
        elif q == 0:
            approx.append(h)

    if missing:
        print(f"⚠ Missing hourly increments for endSteps: {missing} (filled with zeros)")

    # Print a clean quality summary
    print(f"🧪 Hourly quality: exact={len(exact)}/{W}, approx={len(approx)}/{W}, missing={len(missing)}/{W}")
    if approx:
        print(f"   ⚠ Approximated (distributed) endSteps: {approx}")
    if missing:
        print(f"   ❌ Missing endSteps: {missing}")

    # Optional: per-hour detail
    for r in report:
        qtxt = "EXACT" if r["quality"] == 2 else ("APPROX" if r["quality"] == 0 else "MISSING")
        print(f"   • endStep={r['endStep']:>3} -> {qtxt:7} | method={r['method']:<11} | origin={r['origin']}")

    return out, report



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
# Main
# --------------------------------
def main():
    print(f"🌧  FloodLink GFS rain time series export – next {FORECAST_HOURS}h (from now)")

    apcp, lats, lons, times, ref_time, meta = load_gfs_apcp_grid(FORECAST_HOURS)
    if apcp is None:
        print("❌ Failed to load APCP grid.")
        return
      
    hourly_rain, quality_report = compute_hourly_rain_window(
        apcp=apcp,
        start_steps=meta["start_steps"],
        end_steps=meta["end_steps"],
        window_end_steps=meta["window_steps"],
    )
    
    if hourly_rain is None:
        print("❌ Failed to compute hourly rain increments.")
        return
    
    # Hard warning if not all 6 are exact
    if any(r["quality"] != 2 for r in quality_report):
        print("⚠ WARNING: Not all requested hours were exact 1h increments (some were approximated or missing).")


    # ✅ always build window times from ref_time + requested window endSteps
    times_window = [ref_time + timedelta(hours=int(h)) for h in meta["window_steps"]]

    print(f"   Window endSteps requested: {meta['window_steps']}")
    print(f"   Frames downloaded endSteps: {meta['end_steps']}")
    print(f"   Frames downloaded startSteps: {meta['start_steps']}")
    print(f"   Cumulative APCP grid shape: {apcp.shape} (T, Y, X)")
    print(f"   Steps used: {meta.get('steps_used')}")
    print(f"   Has baseline: {meta.get('has_baseline')} (baseline={meta.get('baseline_step')})")
    print(f"   First window time: {times_window[0].isoformat()}")
    print(f"   Last  window time: {times_window[-1].isoformat()}")

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


if __name__ == "__main__":
    main()
