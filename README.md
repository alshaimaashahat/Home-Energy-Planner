# Home Energy Planner - Neural Network + CSP + Search

Plans a household's next 24 hours of electricity use: a **neural network** forecasts the hourly load, a **CSP** (Backtracking, Forward Checking, AC-3 - all hand-written) schedules flexible appliances, and **Uniform Cost Search / A\*** (hand-written, verified against dynamic programming) plans a home battery. A feedback loop (max. 5 iterations) forbids clashing appliance pairs when the battery plan fails.

> **Real vs simulated.** Consumption data are real (UCI Individual Household Electric Power Consumption, CC BY 4.0). Prices, appliance profiles, battery specification and power limits are **simulated parameters** (`planner/config.py`, `planner/csp.py`). All costs and savings are **simulated**, not measured.

## Project structure

```
planner/config.py        simulated parameters (SimConfig), seeding
planner/data.py          download, hourly aggregation, gap filling, features, chronological split, data dictionary
planner/nn.py            MLP forecaster, baselines, metrics, save/load
planner/csp.py           appliance CSP + BT / Forward Checking / AC-3 solver with counters
planner/search.py        battery problem, UCS / A*, search trace, DP checker, heuristic validation
planner/integration.py   plan_day (CSP -> search -> feedback loop), run_home_energy_planner
planner/evaluation.py    experiments 1-3, NN on/off comparison, experiment plots
planner/plotting.py      plot helpers
make_data.py             data script: download + preprocess (deterministic, seed 42)
train_nn.py              trains the network, compares with baselines, saves models/forecaster.pt
run_demo.py              integrated demo (input, solution, plots, trace, pruning, feedback, NN on/off)
run_experiments.py       the three experiments, 5 seeds, mean +- std -> results/
make_report.py           builds the project report (.docx) from the real results -> reports/
tests/                   pytest suite (normal / edge / impossible markers)
reports/                 generated report (Home_Energy_Planner_Report.docx)
Home_Energy_Planner2.ipynb   Google Colab notebook that runs the whole project end to end
                             (setup, data, training, experiments, demo, tests, report) and keeps the outputs
```

## Environment setup (tested with Python 3.12; also ran on Python 3.13 in Google Colab)

```sh
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # pinned versions
```

CPU is enough; no GPU needed. Internet access is needed once to download the dataset (about 20 MB zip). If the download is blocked, download `household_power_consumption.txt` manually and place it in `data/`.

## Exact run commands

```sh
python make_data.py                  # 1. download + preprocess -> data/processed/ (daily_matrix.csv, summary.json, data_dictionary.csv, split_info.csv)
python train_nn.py                   # 2. train NN, test results vs baselines -> results/nn_vs_baselines.csv, models/forecaster.pt
python run_experiments.py            # 3. experiments 1-3 -> results/*.csv and *.png   (a few minutes)
python run_demo.py                   # 4. integrated demo (plots saved to outputs/; add --no-show on a server)
pytest -q                            # 5. tests (also: pytest -m normal | -m edge | -m impossible)
python make_report.py --authors "Name1, Name2, Name3" --out reports/Home_Energy_Planner_Report.docx   # 6. report from the real results
```

Options: `run_demo.py --date 2010-02-17 --skip-nn-compare --retrain`; `run_experiments.py --only 1 3`. Tests need the dataset (they call `prepare_data()`, which downloads it if missing); the first run trains one network (~10-20 s).

## What each script produces

| Script | Output |
| :---- | :---- |
| `make_data.py` | cleaned daily matrix, summary of missing data / invalid days, data dictionary, split info |
| `train_nn.py` | trained network, MAE/RMSE/MAPE table vs "yesterday" and "same day last week", loss and forecast plots |
| `run_experiments.py` | Exp. 1 UCS vs A\* (cost, expanded, generated, time, DP check); Exp. 2 BT vs BT+FC vs BT+FC+AC3 for 3/5/8 appliances (normal and tight limit); Exp. 3 NN vs baselines (MAE, RMSE, cost, limit violations) - each with 5 seeds, mean +- std |
| `run_demo.py` | input, forecast, schedule, battery plan, simulated cost, constraint check, plots; AC-3 pruning; search trace; feedback loop; no-solution message; NN ON vs OFF |
| `make_report.py` | project report (.docx) built from the real files in `results/` and `outputs/` |

## Reproducibility

Seed 42 (runs use seeds 42-46). Data preparation is deterministic. Neural-network training is seeded; tiny differences can occur across PyTorch versions/hardware.

## Limitations

One household, hourly resolution, no holiday/price data; lossless discretised battery; invented prices/limits/appliances; AC-3 used as pre-processing only; strong baselines can match or beat the NN. See the report for details.

## Dataset licence

UCI Individual Household Electric Power Consumption - CC BY 4.0. <https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>
