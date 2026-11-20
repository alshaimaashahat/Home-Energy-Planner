import numpy as np, pytest
from planner.csp import (Appliance, ApplianceCSP, CSPSolver, CSPStats, SOLVER_VARIANTS, explain_no_solution, make_csp_instance)


def tiny_apps():
    return [Appliance("washer", 2, 1.0, 0, 2, "c1"), Appliance("dryer", 1, 1.0, 0, 3, "c2", after="washer")]


@pytest.mark.normal
def test_csp_known_unique_solution():
    csp = ApplianceCSP(tiny_apps(), np.zeros(24), 5, 3.6)
    brute = [(w, d) for w in csp.domains["washer"] for d in csp.domains["dryer"] if not csp.check_solution({"washer": w, "dryer": d})]
    assert brute == [(0, 2)], f"independent enumeration should give exactly (0, 2), got {brute}"
    for fc, ac in SOLVER_VARIANTS.values(): assert CSPSolver(fc, ac).solve(csp)[0] == {"washer": 0, "dryer": 2}


@pytest.mark.normal
def test_all_csp_variants_return_valid_schedules(data):
    csp = make_csp_instance(5, 42, data.Y_test[3])
    for name, (fc, ac) in SOLVER_VARIANTS.items():
        sol, _ = CSPSolver(fc, ac).solve(csp); assert sol is not None and not csp.check_solution(sol), name


@pytest.mark.normal
def test_forward_checking_and_ac3_prune_domains():
    csp = ApplianceCSP(tiny_apps(), np.zeros(24), 5, 3.6)
    _, s_bt = CSPSolver(False, False).solve(csp); _, s_fc = CSPSolver(True, False).solve(csp); _, s_ac = CSPSolver(False, True).solve(csp)
    assert s_bt.domain_reductions == 0 and s_fc.domain_reductions > 0 and s_ac.domain_reductions > 0 and s_ac.ac3_revisions > 0
    doms = {v: list(d) for v, d in csp.domains.items()}; CSPSolver(False, True)._ac3(csp, doms, CSPStats())
    assert doms["washer"] == [0] and doms["dryer"] == [2], "AC-3 must remove dryer starts before the washer can finish"


@pytest.mark.normal
def test_shared_circuit_appliances_do_not_overlap():
    apps = [Appliance("a", 2, 2.0, 0, 6, "same"), Appliance("b", 2, 2.0, 0, 6, "same")]
    sol, _ = CSPSolver(True, True).solve(ApplianceCSP(apps, np.zeros(24), 9.0, 3.6))
    assert sol is not None and abs(sol["a"] - sol["b"]) >= 2


@pytest.mark.edge
def test_edge_no_appliances_gives_empty_schedule():
    sol, _ = CSPSolver(True, True).solve(ApplianceCSP([], np.ones(24), 5, 3.6)); assert sol == {}


@pytest.mark.edge
def test_edge_invalid_inputs_are_rejected():
    for bad in (lambda: ApplianceCSP(tiny_apps(), np.zeros(23), 5, 3.6), lambda: Appliance("x", 0, 1.0, 0, 5, "c"),
                lambda: ApplianceCSP([Appliance("a", 1, 1, 0, 5, "c", after="zzz")], np.zeros(24), 5, 3.6)):
        with pytest.raises(ValueError): bad()


@pytest.mark.impossible
def test_impossible_dryer_cannot_follow_washer():
    apps = [Appliance("washer", 2, 1.0, 0, 2, "c1"), Appliance("dryer", 1, 1.0, 0, 2, "c2", after="washer")]
    csp = ApplianceCSP(apps, np.zeros(24), 5, 3.6)
    for fc, ac in SOLVER_VARIANTS.values():
        sol, st = CSPSolver(fc, ac).solve(csp); assert sol is None and "NO SOLUTION" in explain_no_solution(csp, st)


@pytest.mark.impossible
def test_impossible_household_limit_below_forecast_everywhere():
    csp = ApplianceCSP([Appliance("oven", 1, 2.2, 0, 24, "k")], np.full(24, 4.0), 5.0, 3.6)
    sol, st = CSPSolver(True, True).solve(csp); msg = explain_no_solution(csp, st); assert sol is None and "oven" in msg and "NO SOLUTION" in msg
