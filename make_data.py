#!/usr/bin/env python
"""make_data.py - download and preprocess the UCI Individual Household Electric Power Consumption dataset.

Usage:  python make_data.py [--data-dir data] [--seed 42] [--force]

Steps (see README): download zip -> parse timestamps -> hourly mean (kW) -> forward-fill gaps <= 3 h -> drop invalid days
-> daily matrix -> chronological train/validation/test split. Outputs go to <data-dir>/processed/ :
daily_matrix.csv, summary.json, data_dictionary.csv, split_info.csv.
The pipeline is deterministic; the seed is set for reproducibility of anything downstream.
"""
import argparse
from pathlib import Path
from planner.data import prepare_data
from planner.config import SEED

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="re-run the full pipeline even if processed files exist")
    a = ap.parse_args()
    b = prepare_data(Path(a.data_dir), seed=a.seed, force=a.force)
    print("Summary:"); [print(f"  {k}: {v}") for k, v in b.summary.items()]
    print(f"Samples: train={len(b.d_train)} ({b.d_train.min().date()}..{b.d_train.max().date()}), "
          f"val={len(b.d_val)} ({b.d_val.min().date()}..{b.d_val.max().date()}), test={len(b.d_test)} ({b.d_test.min().date()}..{b.d_test.max().date()})")
