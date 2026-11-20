"""Data preprocessing tests (synthetic inputs except the leakage test)."""
import numpy as np, pandas as pd, pytest
from planner.data import (aggregate_hourly, build_supervised, chronological_split_masks, fill_short_gaps, to_daily_matrix)


@pytest.mark.normal
def test_hourly_aggregation_and_gap_filling():
    idx = pd.date_range("2008-01-01", periods=3 * 24 * 60, freq="min"); s = pd.Series(1.0, index=idx)
    s.iloc[60:120] = 3.0; s.iloc[600:660] = np.nan
    h = aggregate_hourly(s)["load_kw"]
    assert abs(h.iloc[1] - 3.0) < 1e-9 and abs(h.iloc[0] - 1.0) < 1e-9 and np.isnan(h.iloc[10])
    filled, n = fill_short_gaps(h); assert n == 1 and filled.iloc[10] == h.iloc[9], "1-hour gap must be forward filled"


@pytest.mark.normal
def test_only_short_gaps_are_filled_and_bad_days_dropped():
    s = pd.Series(np.ones(72), index=pd.date_range("2008-01-01", periods=72, freq="h")); s.iloc[5:9] = np.nan; s.iloc[30:32] = np.nan
    f, _ = fill_short_gaps(s); assert f.iloc[5:9].isna().all() and f.iloc[30:32].notna().all()
    mat = to_daily_matrix(f); assert mat.iloc[0].isna().all() and mat.iloc[1].notna().all(), "day with a 4h gap must be invalid"


@pytest.mark.normal
def test_chronological_split_is_ordered_and_disjoint():
    dates = pd.date_range("2008-01-01", periods=800, freq="D"); m = chronological_split_masks(dates, dates[-1])
    assert (m["train"] ^ m["val"] ^ m["test"]).all() and not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any()
    assert dates[m["train"]].max() < dates[m["val"]].min() and dates[m["val"]].max() < dates[m["test"]].min() and m["test"].sum() == 365


@pytest.mark.normal
def test_features_use_no_future_information(data):
    daily = data.daily; _, _, all_dates = build_supervised(daily); target_date = all_dates[len(all_dates) // 2]
    i = daily.index.get_loc(target_date)
    Xa, _, da = build_supervised(daily.iloc[:i + 1])
    changed = daily.copy(); changed.iloc[i + 1:] += 100.0
    Xb, _, db = build_supervised(changed)
    assert target_date in da and target_date in db
    assert np.allclose(Xa[list(da).index(target_date)], Xb[list(db).index(target_date)]), "features changed when only future data changed"


@pytest.mark.normal
def test_real_data_split_is_chronological(data):
    assert data.d_train.max() < data.d_val.min() and data.d_val.max() < data.d_test.min()
