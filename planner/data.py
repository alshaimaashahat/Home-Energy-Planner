"""Data acquisition and preprocessing: download, hourly aggregation, gap filling, features, chronological split."""
from __future__ import annotations
import math, urllib.request, zipfile
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path("data")

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"

RAW_FILENAME = "household_power_consumption.txt"

MIN_VALID_MINUTES = 30      # minimum valid minutes for an hour to count as observed

MAX_FILL_GAP_H = 3          # fill only gaps up to this many hours

VAL_DAYS, TEST_DAYS = 90, 365

def download_dataset(data_dir: Path = DATA_DIR, url: str = UCI_ZIP_URL) -> Path:
    """Download and extract the UCI dataset (skipped if the text file already exists)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    txt = data_dir / RAW_FILENAME
    if txt.exists():
        return txt
    zip_path = data_dir / "household_power_consumption.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as out:
            out.write(resp.read())
    except Exception as exc:  # network problems must be reported, not hidden
        raise RuntimeError(
            f"Could not download the dataset from {url}: {exc}. "
            f"Download it manually and place '{RAW_FILENAME}' in '{data_dir}/'.") from exc
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    if not txt.exists():
        found = list(data_dir.rglob(RAW_FILENAME))
        if not found:
            raise FileNotFoundError(f"{RAW_FILENAME} not found after extraction in {data_dir}")
        return found[0]
    return txt

def load_raw_minute_data(path: Path) -> pd.Series:
    """Load Global_active_power (kW, minute resolution) indexed by timestamp. '?' -> NaN."""
    df = pd.read_csv(path, sep=";", usecols=["Date", "Time", "Global_active_power"],
                     na_values=["?", ""], low_memory=False)
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    series = pd.Series(pd.to_numeric(df["Global_active_power"], errors="coerce").values,
                       index=ts, name="global_active_power_kw").sort_index()
    return series[~series.index.duplicated(keep="first")]

def aggregate_hourly(minute: pd.Series, min_valid_minutes: int = MIN_VALID_MINUTES) -> pd.DataFrame:
    """Minute -> hourly mean (kW). Returns load_kw (NaN if too few valid minutes) and valid_minutes."""
    res = minute.resample("h")
    out = pd.DataFrame({"load_kw": res.mean(), "valid_minutes": res.count()})
    out.loc[out["valid_minutes"] < min_valid_minutes, "load_kw"] = np.nan
    full = pd.date_range(out.index.min().floor("D"),
                         out.index.max().floor("D") + pd.Timedelta(hours=23), freq="h")
    return out.reindex(full).rename_axis("timestamp").fillna({"valid_minutes": 0})

def fill_short_gaps(s: pd.Series, max_gap: int = MAX_FILL_GAP_H) -> tuple[pd.Series, int]:
    """Causal forward-fill of NaN runs whose length is <= max_gap. Longer runs stay NaN."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")        # length of the NaN run (0 for valid runs)
    fill_mask = isna & (run_len <= max_gap)
    ffilled = s.ffill()
    out = s.copy()
    out[fill_mask] = ffilled[fill_mask]
    return out, int(fill_mask.sum() - out[fill_mask].isna().sum())

def to_daily_matrix(hourly: pd.Series) -> pd.DataFrame:
    """Hourly series (whole days) -> DataFrame (rows = days, columns = hours 0..23).
    Days containing any missing hour are set to NaN completely (= invalid day)."""
    if len(hourly) % 24 != 0:
        raise ValueError("hourly series must contain whole days")
    start = hourly.index[0].normalize()
    mat = pd.DataFrame(hourly.values.reshape(-1, 24), index=pd.date_range(start, periods=len(hourly) // 24, freq="D"))
    mat.loc[mat.isna().any(axis=1)] = np.nan
    return mat

def chronological_split_masks(dates: pd.DatetimeIndex, last_date: pd.Timestamp,
                              val_days: int = VAL_DAYS, test_days: int = TEST_DAYS) -> dict:
    """Boolean masks (train / val / test) by target date. Test = final `test_days` days,
    validation = the `val_days` days before it, train = everything earlier."""
    test_start = last_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    return {"train": np.asarray(dates < val_start),
            "val": np.asarray((dates >= val_start) & (dates < test_start)),
            "test": np.asarray(dates >= test_start)}

FEATURE_NAMES = ([f"prev_day_h{h:02d}" for h in range(24)] + [f"last_week_h{h:02d}" for h in range(24)]
                 + ["dow_sin", "dow_cos", "is_weekend", "month_sin", "month_cos"])

N_LOAD_FEATURES = 48        # first 48 inputs are loads in kW (scaled together)

def time_features(date: pd.Timestamp) -> np.ndarray:
    """Calendar features of the TARGET day (known in advance, so no leakage)."""
    dow, month = date.dayofweek, date.month
    return np.array([math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
                     float(dow >= 5),
                     math.sin(2 * math.pi * (month - 1) / 12), math.cos(2 * math.pi * (month - 1) / 12)],
                    dtype=np.float32)

def make_feature_vector(prev_day: np.ndarray, last_week_day: np.ndarray, date: pd.Timestamp) -> np.ndarray:
    return np.concatenate([prev_day, last_week_day, time_features(date)]).astype(np.float32)

def build_supervised(daily_mat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build (X, Y, target_dates). A sample needs valid days D, D-1 and D-7."""
    vals, idx = daily_mat.values, daily_mat.index
    X, Y, dates = [], [], []
    for i in range(7, len(daily_mat)):
        prev, week, target = vals[i - 1], vals[i - 7], vals[i]
        if np.isnan(prev).any() or np.isnan(week).any() or np.isnan(target).any():
            continue
        X.append(make_feature_vector(prev, week, idx[i])); Y.append(target); dates.append(idx[i])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), pd.DatetimeIndex(dates)

DATA_DICTIONARY_ROWS = [
    # (column / item, meaning, unit, real or generated, how it was made)
    ("Date", "Calendar date dd/mm/YYYY of the measurement", "date", "REAL (UCI raw)", "measured by the household logger"),
    ("Time", "Time hh:mm:ss of the measurement", "time", "REAL (UCI raw)", "measured"),
    ("Global_active_power", "Household minute-averaged active power (source of the forecasting target)", "kW", "REAL (UCI raw), used", "measured; '?' = missing"),
    ("Global_reactive_power", "Household minute-averaged reactive power", "kW", "REAL (UCI raw), not used", "measured"),
    ("Voltage", "Minute-averaged voltage", "V", "REAL (UCI raw), not used", "measured"),
    ("Global_intensity", "Minute-averaged current intensity", "A", "REAL (UCI raw), not used", "measured"),
    ("Sub_metering_1/2/3", "Energy sub-meters (kitchen / laundry / water heater + AC)", "Wh", "REAL (UCI raw), not used", "measured"),
    ("load_kw", "Hourly mean of Global_active_power over valid minutes; numerically = kWh in that hour", "kW", "DERIVED from real data", "resample('h').mean(); NaN if < 30 valid minutes; gaps <= 3 h forward-filled"),
    ("valid_minutes", "Number of valid minutes inside the hour (hour kept if >= 30)", "count", "DERIVED from real data", "resample('h').count()"),
    ("prev_day_h00..h23", "Hourly load of target day - 1", "kW", "DERIVED from real data", "daily matrix row D-1"),
    ("last_week_h00..h23", "Hourly load of target day - 7 (same hour, previous week)", "kW", "DERIVED from real data", "daily matrix row D-7"),
    ("dow_sin, dow_cos, is_weekend, month_sin, month_cos", "Calendar features of the target day", "-", "DERIVED (calendar)", "sin/cos of 2*pi*dow/7 and 2*pi*(month-1)/12; weekend flag"),
    ("target y_h00..y_h23", "Real hourly load of the target day", "kW", "DERIVED from real data", "daily matrix row D"),
    ("price_*, peak/off-peak hours", "Time-of-use electricity tariff", "EUR/kWh", "SIMULATED (SimConfig)", "hand-chosen 3-level tariff: off-peak 0.12 (22-06), peak 0.32 (17-21), mid 0.20"),
    ("household_limit_kw, grid_limit_kw, circuit_limit_kw", "Consumption / grid-import / circuit power limits", "kW", "SIMULATED (SimConfig)", "hand-chosen constants (5.0 / 4.0 / 3.6 kW)"),
    ("battery_* parameters", "Capacity, step, max charge/discharge power, initial/final energy", "kWh, kW", "SIMULATED (SimConfig)", "hand-chosen constants (5 kWh, 0.5 kWh step, 2.5 kW, start/end 2.5 kWh)"),
    ("Appliance profiles (duration, power, window, circuit)", "Eight flexible appliances (washer, dryer, dishwasher, EV, oven, water heater, pool pump, iron)", "h, kW", "SIMULATED (APPLIANCE_POOL)", "hand-chosen constants; Exp. 2 narrows each window by a seeded random 0-2 h"),
    ("Experiment scenarios", "Extra 2-hour appliance block added to real days in Exp. 1; random day / window choices in Exp. 2-3", "kW", "GENERATED (seeded)", "numpy default_rng(seed), seeds 42-46"),
    ("Battery plan, schedule, simulated cost", "Planner outputs", "kWh, EUR", "GENERATED by the algorithms", "CSP solver + A*/UCS; costs are SIMULATED"),
]


# ======================================================================================
# High-level pipeline (used by make_data.py, run_demo.py, run_experiments.py, tests)
# ======================================================================================
import json
from dataclasses import dataclass
from .config import SEED, set_seed


def data_dictionary() -> pd.DataFrame:
    """Data dictionary: every column / item, meaning, unit, real-or-generated, and how it was made."""
    return pd.DataFrame(DATA_DICTIONARY_ROWS, columns=["column / item", "meaning", "unit", "real or generated", "how generated / derived"])


@dataclass
class DataBundle:
    """Everything downstream code needs: cleaned daily matrix + chronological train/val/test samples."""
    daily: pd.DataFrame
    X_train: np.ndarray; Y_train: np.ndarray; d_train: pd.DatetimeIndex
    X_val: np.ndarray;   Y_val: np.ndarray;   d_val: pd.DatetimeIndex
    X_test: np.ndarray;  Y_test: np.ndarray;  d_test: pd.DatetimeIndex
    summary: dict


def bundle_from_daily(daily: pd.DataFrame, summary: Optional[dict] = None) -> DataBundle:
    X, Y, dates = build_supervised(daily)
    m = chronological_split_masks(dates, daily.index[-1])
    assert dates[m["train"]].max() < dates[m["val"]].min() and dates[m["val"]].max() < dates[m["test"]].min(), "split must be chronological"
    return DataBundle(daily, X[m["train"]], Y[m["train"]], dates[m["train"]], X[m["val"]], Y[m["val"]], dates[m["val"]],
                      X[m["test"]], Y[m["test"]], dates[m["test"]], summary or {})


def prepare_data(data_dir: Path = DATA_DIR, seed: int = SEED, force: bool = False, verbose: bool = True) -> DataBundle:
    """Reproducible pipeline: download -> parse -> hourly -> gap fill -> valid days -> split.
    The result is cached in <data_dir>/processed/. The pipeline itself is deterministic (no randomness);
    `seed` is applied anyway so that anything downstream that uses random numbers starts from the same state."""
    set_seed(seed)
    data_dir = Path(data_dir); out = data_dir / "processed"; cache = out / "daily_matrix.csv"
    if cache.exists() and not force:
        daily = pd.read_csv(cache, index_col=0, parse_dates=True); daily.columns = range(24)
        summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        if verbose: print(f"[data] loaded cached daily matrix {cache} ({len(daily)} days)")
        return bundle_from_daily(daily, summary)
    raw = download_dataset(data_dir)
    minute = load_raw_minute_data(raw)
    hourly = aggregate_hourly(minute)
    filled, n_filled = fill_short_gaps(hourly["load_kw"])
    daily = to_daily_matrix(filled)
    valid = ~daily.isna().any(axis=1)
    summary = {"minute_rows": int(len(minute)), "missing_minute_share": float(minute.isna().mean()),
               "hours_total": int(len(hourly)), "hours_missing_before_fill": int(hourly["load_kw"].isna().sum()),
               "hours_filled": int(n_filled), "hours_still_missing": int(filled.isna().sum()),
               "days_total": int(len(daily)), "days_valid": int(valid.sum()), "days_invalid": int((~valid).sum()),
               "first_timestamp": str(minute.index.min()), "last_timestamp": str(minute.index.max()), "seed": seed}
    bundle = bundle_from_daily(daily, summary)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache); (out / "summary.json").write_text(json.dumps(summary, indent=2))
    data_dictionary().to_csv(out / "data_dictionary.csv", index=False)
    pd.DataFrame({"split": ["train", "validation", "test"], "n_days": [len(bundle.d_train), len(bundle.d_val), len(bundle.d_test)],
                  "first_date": [d.min().date() for d in (bundle.d_train, bundle.d_val, bundle.d_test)],
                  "last_date": [d.max().date() for d in (bundle.d_train, bundle.d_val, bundle.d_test)]}).to_csv(out / "split_info.csv", index=False)
    if verbose: print(f"[data] processed data written to {out}/")
    return bundle
