from dataclasses import replace
import numpy as np, pytest
from planner.config import CFG, HORIZON
from planner.csp import Appliance
from planner.integration import forecast_for_date, plan_day


@pytest.mark.normal
def test_integrated_planner_returns_valid_plan_on_real_test_day(data, forecaster):
    plan = plan_day(forecast_for_date(data.d_test[len(data.d_test) // 2], data.daily, forecaster))
    assert plan.success and all(plan.constraints_ok.values()), plan.messages


@pytest.mark.normal
def test_feedback_loop_triggers_and_resolves_conflict():
    cfg = replace(CFG, household_limit_kw=6.0, grid_limit_kw=4.2, battery_capacity_kwh=6.0, battery_max_charge_kw=1.0,
                  battery_max_discharge_kw=1.0, battery_initial_kwh=6.0, battery_final_min_kwh=0.0)
    plan = plan_day(np.full(HORIZON, 1.5), cfg, [Appliance("ev_charger", 4, 3.3, 0, 24, "garage"), Appliance("iron", 1, 1.2, 0, 24, "laundry")])
    assert plan.success and plan.iterations >= 2 and len(plan.feedback_log) >= 1 and all(plan.constraints_ok.values())


@pytest.mark.impossible
def test_impossible_plan_gives_clear_message_and_relaxation_offer():
    plan = plan_day(np.full(HORIZON, 3.0), CFG, [Appliance("ev_charger", 4, 3.3, 0, 24, "garage")])
    assert not plan.success and any("NO SOLUTION" in m for m in plan.messages) and any("relaxed" in m for m in plan.messages)


@pytest.mark.edge
def test_edge_forecast_for_date_without_history_raises(data):
    with pytest.raises(ValueError): forecast_for_date(data.daily.index[0], data.daily, None)
