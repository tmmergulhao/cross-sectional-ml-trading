# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Uses Python 3.9 with a `.venv` virtual environment. Activate before running anything:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `pandas`, `numpy`, `yfinance`, `scikit-learn`, `xgboost`, `matplotlib`, `pyarrow`, `scipy`, `tqdm`.

## Running notebooks

```bash
jupyter notebook notebooks/
```

## Architecture

This is a research pipeline for cross-sectional equity return prediction. The intended data flow is:

```
yfinance (raw) → data/raw/
                → src/data_loader.py      → data/processed/
                → src/feature_engineer.py → (features built inline in notebooks)
                → src/evaluation.py       → results/<model>/charts/ + results/<model>/tables/
src/utils.py   — diagnostics (IC, ICIR, plots, saving)
src/models/    — model wrappers (one file per model)
```

**Cross-sectional** means models rank or score stocks relative to each other at each point in time (not time-series forecasting). Walk-forward (expanding window) methodology is used throughout to avoid look-ahead bias.

### Source modules

- `src/data/` — data infrastructure: abstract loader, Yahoo Finance + Ken French implementations, transforms, validators
- `src/data_loader.py` — orchestrates download and caching to `data/raw/` and `data/processed/`
- `src/feature_engineer.py` — momentum, volatility, liquidity, and extreme-value features (all use `shift(1)` to prevent look-ahead)
- `src/evaluation.py` — `WalkForwardBacktester` class; plug in any `AbstractCrossSectionalModel`
- `src/models/base.py` — `AbstractCrossSectionalModel` interface (implement to add a new model)
- `src/models/xgboost_model.py` — XGBoost implementation
- `src/utils.py` — `diagnose()` (IC, ICIR, NW t-stat, plots), `plot_feature_importance()`, save helpers

### Notebooks

```
notebooks/
├── research/          — educational deep-dives (read-only reference)
│   ├── quant_finance_primer.ipynb         — CAPM → FF3 → PCA → Markowitz
│   ├── jegadeesh_titman_1993_replication.ipynb — momentum paper replication
│   ├── french_data_library.ipynb          — Ken French factor data
│   └── checking_data.ipynb                — basic EDA
├── models/            — one notebook per model (runs backtest, saves results)
│   ├── xgboost.ipynb
│   └── linear_regression.ipynb
└── comparison/
    └── model_comparison.ipynb             — loads saved results, side-by-side comparison
```

### Results

```
results/
├── xgboost/
│   ├── charts/   — PNG plots (diagnostic, IC, L/S spread, feature importance, per-fold IC)
│   └── tables/   — CSV tables (summary metrics, monthly IC series, L/S series, per-fold IC)
├── linear_regression/
│   ├── charts/
│   └── tables/
└── comparison/
    ├── charts/   — side-by-side IC and L/S spread plots
    └── tables/   — metrics comparison, win-rate table
```

## Adding a new model

1. Create `src/models/<model_name>.py` implementing `AbstractCrossSectionalModel` (name, fit, predict)
2. Copy `notebooks/models/linear_regression.ipynb` → `notebooks/models/<model_name>.ipynb`
3. Update the model class and `RESULTS_DIR` in the notebook
4. Add the model to `MODELS` dict in `notebooks/comparison/model_comparison.ipynb`
