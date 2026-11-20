import numpy as np, pytest
from planner.search import (BatteryProblem, battery_search, check_heuristic_admissible, dp_optimal_cost, random_small_problem)


def tiny_problem(load, prices, **kw) -> BatteryProblem:
    base = dict(capacity_kwh=2.0, step_kwh=1.0, max_charge_kw=1.0, max_discharge_kw=1.0, grid_limit_kw=10.0, initial_kwh=0.0, final_min_kwh=0.0)
    base.update(kw); return BatteryProblem(np.array(load, float), np.array(prices, float), **base)


@pytest.mark.normal
def test_search_known_answer_ucs_astar_dp():
    pr = tiny_problem([1, 1, 1], [1, 3, 1])      # hand solution: charge@0 (cost 2), discharge@1 (0), idle@2 (1) = 3
    u, a = battery_search(pr, False), battery_search(pr, True)
    assert abs(u.total_cost - 3.0) < 1e-9 and abs(a.total_cost - 3.0) < 1e-9 and abs(dp_optimal_cost(pr) - 3.0) < 1e-9
    assert [s.kind for s in u.steps] == ["charge", "discharge", "idle"] and u.nodes_expanded > 0 and a.nodes_generated > 0


@pytest.mark.normal
def test_battery_transitions_respect_bounds_and_limits():
    pr = tiny_problem([1, 1], [1, 1], grid_limit_kw=1.5)
    assert {k for k, *_ in pr.successors(0, 1)} == {"idle", "discharge"}, "charging (grid 2.0) exceeds the 1.5 kW limit"
    for h in range(2):
        for l in range(pr.n_levels):
            for k, q, nl, grid, c in pr.successors(h, l): assert 0 <= nl < pr.n_levels and grid <= 1.5 + 1e-9 and c >= 0


@pytest.mark.normal
def test_astar_heuristic_admissible_on_small_instances():
    rng = np.random.default_rng(7); assert all(check_heuristic_admissible(random_small_problem(rng)) for _ in range(40))


@pytest.mark.normal
def test_search_counts_nodes_and_trace():
    pr = tiny_problem([1, 1, 1], [1, 3, 1]); tr = []; r = battery_search(pr, True, trace=tr)
    assert len(tr) == r.nodes_expanded and r.nodes_generated >= r.nodes_expanded - 1


@pytest.mark.edge
def test_edge_battery_full_cannot_charge_and_empty_cannot_discharge():
    pr = tiny_problem([1, 1], [1, 1], initial_kwh=2.0)
    assert "charge" not in {k for k, *_ in pr.successors(0, pr.n_levels - 1)} and "discharge" not in {k for k, *_ in pr.successors(0, 0)}
    assert all(0 <= s.soc_after <= 2.0 for s in battery_search(pr, True).steps)


@pytest.mark.edge
def test_edge_flat_prices_battery_cannot_save_money():
    assert abs(battery_search(tiny_problem([1, 2, 1], [0.2, 0.2, 0.2]), True).total_cost - 0.8) < 1e-9


@pytest.mark.edge
def test_edge_invalid_inputs_are_rejected():
    for bad in (lambda: tiny_problem([1], [1], capacity_kwh=-1), lambda: tiny_problem([1, 2], [1]), lambda: tiny_problem([-1], [1])):
        with pytest.raises(ValueError): bad()


@pytest.mark.impossible
def test_impossible_battery_plan_when_load_exceeds_grid_limit():
    pr = tiny_problem([5.0, 1.0], [1, 1], grid_limit_kw=2.0, max_discharge_kw=1.0, initial_kwh=2.0)
    for use_h in (False, True):
        r = battery_search(pr, use_h); assert not r.found and "NO SOLUTION" in r.message
    assert not np.isfinite(dp_optimal_cost(pr))
