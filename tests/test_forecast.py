import math, numpy as np, pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from planner.integration import forecast_for_date
from planner.nn import mae, mape, rmse


@pytest.mark.normal
def test_forecast_shape_and_validity(data, forecaster):
    P = forecaster.predict_batch(data.X_test); assert P.shape == (len(data.X_test), 24) and np.isfinite(P).all() and (P >= 0).all()
    assert forecast_for_date(data.d_test[0], data.daily, forecaster).shape == (24,)


@pytest.mark.normal
def test_metrics_match_sklearn():
    y, p = np.random.default_rng(0).uniform(0.5, 3, 50), np.random.default_rng(1).uniform(0.5, 3, 50)
    assert abs(mae(y, p) - mean_absolute_error(y, p)) < 1e-9 and abs(rmse(y, p) - math.sqrt(mean_squared_error(y, p))) < 1e-9


@pytest.mark.edge
def test_edge_mape_ignores_near_zero_actuals():
    assert abs(mape([0.0, 0.1, 2.0], [1.0, 1.0, 1.0]) - 50.0) < 1e-9 and math.isnan(mape([0.0], [1.0]))


@pytest.mark.edge
def test_edge_wrong_feature_shape_is_rejected(forecaster):
    with pytest.raises(ValueError): forecaster.predict_batch(np.zeros((2, 10)))
