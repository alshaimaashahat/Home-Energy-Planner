#!/usr/bin/env python
"""make_report.py - builds the project report (.docx, target <= 12 pages) from the REAL result files.
Every number comes from results/*.csv produced by train_nn.py and run_experiments.py; nothing is typed in by hand.

Usage:  python train_nn.py && python run_experiments.py && python run_demo.py --no-show && python make_report.py --authors "A, B, C"
Output: report/Home_Energy_Planner_Report.docx   (open in Word, check the page count, export to PDF if needed)"""
import argparse, sys
from pathlib import Path
import pandas as pd

try:
    from docx import Document
    from docx.shared import Pt, Cm, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    sys.exit("python-docx is required: pip install python-docx")

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--outputs", default="outputs")
ap.add_argument("--authors", default="[Team members]"); ap.add_argument("--out", default="report/Home_Energy_Planner_Report.docx"); a = ap.parse_args()
R, O = Path(a.results), Path(a.outputs)
need = ["nn_vs_baselines.csv", "exp1_runs.csv", "exp1_summary.csv", "exp2_runs.csv", "exp2_summary.csv", "exp3_runs.csv", "exp3_summary.csv"]
missing = [f for f in need if not (R / f).exists()]
if missing: sys.exit(f"Missing result files in {R}/: {missing}. Run train_nn.py and run_experiments.py first.")
nn_t = pd.read_csv(R / "nn_vs_baselines.csv", index_col=0); e1 = pd.read_csv(R / "exp1_runs.csv"); e2 = pd.read_csv(R / "exp2_runs.csv"); e3 = pd.read_csv(R / "exp3_runs.csv")
s1 = pd.read_csv(R / "exp1_summary.csv"); s2 = pd.read_csv(R / "exp2_summary.csv"); s3 = pd.read_csv(R / "exp3_summary.csv", index_col=0)
summ = {}
for p in (Path("data/processed/summary.json"),):
    if p.exists():
        import json; summ = json.loads(p.read_text())

doc = Document()
sec = doc.sections[0]; sec.left_margin = sec.right_margin = Cm(2.0); sec.top_margin = sec.bottom_margin = Cm(1.9)
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
def H(t, l=1): doc.add_heading(t, level=l)
def Pp(t): p = doc.add_paragraph(t); p.paragraph_format.space_after = Pt(4); return p
def B(items):
    for t in items: doc.add_paragraph(t, style="List Bullet").paragraph_format.space_after = Pt(1)
def T(df, caption, index=True):
    cap = doc.add_paragraph(caption); cap.runs[0].italic = True; cap.runs[0].font.size = Pt(9)
    df = df.reset_index() if index else df
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row): cells[i].text = f"{v:.4g}" if isinstance(v, float) else str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs: r.font.size = Pt(8)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
def F(path, caption, w=15.5):
    path = Path(path)
    if path.exists():
        doc.add_picture(str(path), width=Cm(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = doc.add_paragraph(caption); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.runs[0].italic = True; c.runs[0].font.size = Pt(9)
    else: Pp(f"[Figure missing: {path}]")

# ---- numbers used in the text (all computed)
m = nn_t["MAE (kW)"]; nn_mae, y_mae, w_mae = m["Neural Network"], m["Yesterday (same hour)"], m["Last week (same hour)"]
best_base = "yesterday" if y_mae < w_mae else "same-day-last-week"; best_base_mae = min(y_mae, w_mae)
g1 = e1.groupby("algorithm")[["nodes expanded", "nodes generated", "time ms", "cost EUR (simulated)"]].mean()
red_exp = (1 - g1.loc["A*", "nodes expanded"] / g1.loc["UCS", "nodes expanded"]) * 100
e2["effort"] = e2["assignments"] + e2["rejected"]
g2 = e2.groupby(["scenario", "appliances", "algorithm"])[["effort", "backtracks", "time ms"]].mean()
g3 = e3.groupby("method")[["cost EUR/day (simulated)", "grid-limit violations (h / 30 days)", "household-limit violations (h / 30 days)", "MAE kW (all test days)"]].mean()

# ---- document
doc.add_heading("Home Energy Planner: Neural Network, CSP and Search for Household Energy Management", 0)
Pp(f"Authors: {a.authors}\nCourse project P2 (Search + CSP + Neural Network). All costs and savings in this report are SIMULATED results, not measured household savings.")

H("1. Problem and objectives")
Pp("A household has a base electricity load, flexible appliances (washer, dryer, dishwasher, EV charger, oven), a home battery, a time-of-use tariff and power limits. "
   "The planner decides, for the next 24 hours, when each appliance runs and when the battery charges or discharges so that the simulated bill is low and no limit is broken. "
   "Three AI techniques cooperate: a neural network forecasts the hourly load, a constraint satisfaction problem (CSP) schedules the appliances, and a search algorithm (UCS and A*) plans the battery.")
Pp("Users are households and energy advisers; the system is a decision-support simulation only.")

H("2. Data: real versus simulated")
Pp("Electricity consumption comes from the UCI Individual Household Electric Power Consumption dataset (one house near Paris, minute resolution, December 2006 to November 2010, licence CC BY 4.0). "
   "Prices, appliance profiles, battery specification and power limits are simulated configuration parameters (planner/config.py, planner/csp.py); the full data dictionary is in data/processed/data_dictionary.csv.")
if summ:
    Pp(f"Preprocessing: {summ['minute_rows']:,} minute rows ({summ['missing_minute_share']:.2%} missing) were averaged to {summ['hours_total']:,} hourly values; an hour needs at least 30 valid minutes. "
       f"Gaps of at most 3 hours were forward-filled ({summ['hours_filled']} hours, causal, no future information); days that still had a missing hour were dropped ({summ['days_invalid']} of {summ['days_total']} days, {summ['days_valid']} valid). "
       "The split is chronological by target day: the final 365 days are the test set, the 90 days before are validation and everything earlier is training; scalers use training data only.")
else: Pp("Preprocessing: minute data are averaged to hourly values (at least 30 valid minutes), gaps of at most 3 hours are forward-filled, days with remaining gaps are dropped, and the split is chronological (final 365 days test, 90 days validation, rest training).")

H("3. System design")
B(["Architecture: NN forecast -> CSP schedule -> battery search; if the battery plan fails, a clashing appliance pair is forbidden and the CSP re-runs (at most 5 iterations).",
   "Two limits (design decision): household_limit_kw (breaker; enforced by the CSP on forecast + appliances) and grid_limit_kw (grid-import contract; enforced by the battery search). With one limit the battery could never fail and the feedback loop would have nothing to do.",
   "Battery must end the day with at least its initial energy; otherwise emptying it for free is trivially optimal. Battery is lossless, cannot export, and energy is discretised in 0.5 kWh steps (this can make the optimum slightly worse than continuous control).",
   "Modular code: planner/{config,data,nn,csp,search,integration,evaluation,plotting}.py; scripts make_data.py, train_nn.py, run_experiments.py, run_demo.py; tests in tests/ (pytest)."])

H("4. Neural network: load forecasting")
Pp("A multilayer perceptron (53 -> 128 -> 64 -> 24, ReLU, dropout 0.2) predicts the 24 hourly loads of the next day at once. Inputs: the 24 loads of the previous day, the 24 loads of the same day one week earlier (same hour, previous week), "
   "day-of-week and month as sine/cosine, and a weekend flag; the hour of day is encoded by the position in the 24-value output. Training uses MSE, AdamW (weight decay 1e-4), mini-batches of 64, dropout and early stopping (patience 20) on the validation loss.")
T(nn_t.round(4), "Table 1. Test-set forecast error (identical test days for all models). MAPE only counts actuals >= 0.2 kW.")
rel = (1 - nn_mae / best_base_mae) * 100
Pp(f"On the test set the network reached MAE {nn_mae:.4f} kW against {best_base_mae:.4f} kW for the best baseline ({best_base}); "
   + (f"that is {rel:.1f}% lower." if rel > 0 else f"the network is {abs(rel):.1f}% worse than that baseline, which we report as is."))
F(R / "nn_training_loss.png", "Figure 1. Training and validation loss with the early-stopping epoch.", 10.5)

H("5. CSP: appliance scheduling")
Pp("Variables are appliance start hours. Domains contain the start hours that keep the appliance inside its allowed window and, as a unary constraint, do not push forecast plus appliance above the household limit. "
   "Constraints: the dryer starts after the washer has finished; appliances on a shared circuit do not overlap if their combined power exceeds the circuit limit; "
   "in every hour forecast plus running appliances stays below the household limit (global constraint, checked on the partial assignment); durations are respected. "
   "Backtracking, Forward Checking and AC-3 are implemented by hand and are independently switchable; counters: assignments, rejected values, backtracks, domain reductions. "
   "Forward Checking removes values that conflict with the current partial assignment (including the load limit) from unassigned variables; AC-3 removes values without support in a neighbour's domain before the search. "
   "If no schedule exists, the planner names the appliances without a feasible start hour and shows the household limit relaxed in 10% steps.")

H("6. Search: battery planning")
Pp("State = (hour, battery level in 0.5 kWh steps); actions = charge, discharge or idle (in multiples of the step, within power limits, capacity, no export, grid import below the grid limit); "
   "path cost = price x energy bought from the grid; goal = hour 24 with at least the required final energy. UCS expands by path cost g, A* by g + h with h = (cheapest remaining price) x max(0, remaining load - stored energy + required final energy). "
   "The heuristic never overestimates because the energy bought over the rest of the day is at least remaining load minus stored energy plus final energy, and every bought kWh costs at least the cheapest remaining price; it is also consistent. "
   "An independent dynamic-programming solver verifies the optimal cost, and the heuristic is checked against the exact cost-to-go of every state.")

H("7. Integration and feedback loop")
Pp("run_home_energy_planner(date) loads the history, forecasts the day, solves the CSP (BT + FC + AC-3), adds the scheduled appliances to the forecast, plans the battery with A*, verifies every constraint independently and reports the simulated cost. "
   "When the battery plan fails, the hour that cannot be covered is identified, the two most power-hungry appliances running then are forbidden to overlap (or the hour is blocked for a single appliance) and the CSP runs again. "
   "The mechanism is demonstrated on a synthetic stress scenario (tests/test_integration.py and run_demo.py) because the default configuration may never make the battery fail on real days.")
F(O / "demo_plan.png", "Figure 2. Demo day: forecast vs actual, schedule, battery plan, grid import and simulated cost comparison.", 15.5)

H("8. Experiments (5 runs, seeds 42-46, mean +- std)")
H("Experiment 1: UCS vs A*", 2)
T(s1.rename(columns={"algorithm": "algorithm"}), "Table 2. 30 scenarios per run (real daily loads plus a seeded extra appliance block).", index=False)
ok_dp = bool(e1["cost == DP"].all()); ok_h = bool(e1["heuristic admissible (all states)"].all())
Pp(f"Both algorithms returned the cost of the dynamic-programming optimum in all runs: {ok_dp}; the heuristic never exceeded the exact cost-to-go: {ok_h}. "
   f"A* expanded {g1.loc['A*', 'nodes expanded']:.1f} nodes on average against {g1.loc['UCS', 'nodes expanded']:.1f} for UCS ({red_exp:.1f}% fewer).")
F(R / "exp1_ucs_vs_astar.png", "Figure 3. UCS vs A*.", 15.5)
H("Experiment 2: BT vs BT+FC vs BT+FC+AC3", 2)
T(s2, "Table 3. 3, 5 and 8 appliances; 'rejected' = values tried but inconsistent; instances with the 'tight' limit may be unsolvable (see solved counts in run files).", index=False)
if ("tight limit (4.2 kW)", 8, "BT") in g2.index:
    bt, fc_ = g2.loc[("tight limit (4.2 kW)", 8, "BT"), "effort"], g2.loc[("tight limit (4.2 kW)", 8, "BT+FC"), "effort"]
    Pp(f"With 8 appliances and the tight limit, plain backtracking needed {bt:.1f} value trials on average (assignments + rejected) against {fc_:.1f} with Forward Checking. "
       "Because the extra work of pruning costs time per node, wall-clock time can still favour plain backtracking on easy instances.")
F(R / "exp2_csp_comparison.png", "Figure 4. CSP algorithm comparison.", 15.5)
H("Experiment 3: NN vs baselines inside the planner", 2)
T(s3.T, "Table 4. Mean +- std over 5 runs; 30 identical test days per run; the Oracle row uses the real load as forecast (reference only).")
nnc, yc, wc = (g3.loc[k, "cost EUR/day (simulated)"] for k in ("Neural Network", "Yesterday", "Last week"))
nnv, yv, wv = (g3.loc[k, "grid-limit violations (h / 30 days)"] for k in ("Neural Network", "Yesterday", "Last week"))
Pp(f"Simulated cost per day: NN {nnc:.3f}, yesterday {yc:.3f}, last week {wc:.3f} EUR. Grid-limit violations per 30 days: NN {nnv:.1f}, yesterday {yv:.1f}, last week {wv:.1f}. "
   "A more accurate forecast does not automatically give a cheaper plan: cost depends on the sign and timing of the errors, so differences should be judged against the standard deviation across runs.")
F(R / "exp3_nn_vs_baselines.png", "Figure 5. NN vs baselines.", 15.5)

H("9. Testing")
Pp("The pytest suite (tests/) contains normal, edge and impossible cases (run 'pytest -m normal', '-m edge', '-m impossible'). Covered: preprocessing, leakage, forecast shape and validity, metrics, search transitions, "
   "a hand-solved search example checked against UCS, A* and DP, a CSP with a unique solution verified by brute force, Forward Checking and AC-3 pruning, battery capacity boundaries, invalid inputs, and no-solution messages for the CSP, the battery search and the integrated planner.")

H("10. Limitations")
B(["One household, hourly resolution, no holiday or price data; results may not generalise.",
   "All prices, limits, appliance and battery parameters are invented; costs are simulated and only relative comparisons are informative.",
   "Battery is lossless and discretised; the heuristic assumes efficiency 1.0. Appliances are always flexible; user comfort is not modelled.",
   "AC-3 is used as pre-processing only (not maintained during search); the CSP variable order is static; the benchmark has a node limit.",
   "The forecast is made once at midnight; no intraday updates. Strong baselines (same day last week) are hard to beat on household data."])

H("11. Conclusions")
B([f"Forecasting: NN MAE {nn_mae:.4f} kW vs {best_base_mae:.4f} kW for the best baseline ({best_base}).",
   f"Search: A* returned the DP-optimal cost in all runs ({ok_dp}) with {red_exp:.1f}% fewer expansions than UCS.",
   "CSP: Forward Checking and AC-3 reduce the number of value trials on constrained instances; see Table 3 for the measured effect.",
   "Integration: the planner produces verified schedules and battery plans and reports clear messages when no plan exists; the feedback loop resolves battery failures in the stress scenario."])
Path(a.out).parent.mkdir(parents=True, exist_ok=True); doc.save(a.out); print("saved", a.out)
