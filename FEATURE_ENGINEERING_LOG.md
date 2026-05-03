# Feature Engineering Log
## Cross-Sectional Return Prediction Project

---

## 1. Core Concepts

### Information Coefficient (IC)
The IC is the **Spearman rank correlation** between a feature and the forward return target, computed **cross-sectionally** (per date, across all stocks). It answers: does ranking stocks by this feature correspond to ranking them by their future return?

```
IC_t = Spearman(feature_rank_t, return_rank_t)   [computed per date]
Mean IC = average of IC_t across all dates
```

Benchmarks:
- Mean IC > 0.05 → strong signal
- Mean IC 0.02–0.05 → acceptable, typical in practice
- Mean IC < 0.02 → weak, review features/target

### ICIR (Information Coefficient Information Ratio)
Measures **consistency** of the signal over time, not just its average level.

```
ICIR = Mean_IC / Std_IC
```

Benchmark: ICIR > 0.5 is considered consistent.

### Newey-West t-statistic
IC observations are autocorrelated (because forward returns overlap across dates), so standard t-tests overstate significance. The Newey-West HAC estimator corrects for this.

```
NW t-stat = Mean_IC / NW_SE
```

Rule of thumb: |t| > 2 → statistically significant.

### Overlapping Returns Bias
Computing IC **daily** on a 21-day forward return inflates the t-stat by ~√21 because consecutive observations share most of the same return window. Fix: **subsample IC every 21 days** so observations are non-overlapping.

```python
sampled_dates = pred_df.index.unique().sort_values()[::ic_on_every]
pred_df_sampled = pred_df[pred_df.index.isin(sampled_dates)]
# BOTH ic_series AND ls_spread must be computed from pred_df_sampled
```

### Cross-Sectional Ranking
All features are ranked cross-sectionally (per date, across all stocks) before entering the model. This removes time-series level effects and makes features comparable across different market regimes.

```python
df[FEATURE_COLS] = df.groupby("Date")[FEATURE_COLS].rank(pct=True)
```

This must happen **after** all features are computed and **before** train/test split.

### IC Decay Curve
IC computed at multiple forward horizons (1, 5, 10, 21, 42, 63, 126, 252, 504 days) to identify the natural prediction horizon of each feature. Features with IC that peaks at 252+ days belong in a long-horizon model; features peaking at 21 days belong in a medium-horizon model.

---

## 2. Features — Full List with Equations

### 2.1 Momentum
```
mom_{l}_{r} = Close[t-l] / Close[t-r] - 1
```
Both endpoints are in the past — no look-ahead by construction.

| Feature | Window | Notes |
|---|---|---|
| mom_1_5 | 1-day to 5-day | Very short-term (weekly reversal) |
| mom_1_21 | 1-day to 21-day | Short-term (monthly reversal) — **negative IC** |
| mom_21_126 | 21-day to 126-day | Medium-term — **negative IC** |
| mom_21_252 | 21-day to 252-day | Medium-term — **negative IC** |
| mom_126_252 | 126-day to 252-day | Medium-term clean momentum |
| mom_252_756 | 252-day to 756-day | Long-term momentum |

⚠️ `mom_21_126` and `mom_21_252` have negative IC — they are the **reversal** effect, not momentum. Do not confuse with `mom_126_252`.

### 2.2 Volatility
```
vol_{h} = rolling std of daily returns over h days, shifted by 1
downside_dev_{h} = sqrt(rolling mean of (min(r, 0))^2 over h days), shifted by 1
```

| Feature | Window |
|---|---|
| vol_21, vol_63, vol_252 | 21, 63, 252 days |
| downside_dev_21, downside_dev_63, downside_dev_252 | 21, 63, 252 days |

All positively correlated with future returns (illiquidity/risk premium). High correlation with each other and with Parkinson vol.

### 2.3 Liquidity (Dollar Volume)
```
dollar_volume_{h} = rolling mean of (Volume × AdjClose) over h days, shifted by 1
```

**Negative IC** — large liquid stocks earn lower future returns (consistent with illiquidity premium). Apply `log(1 + x)` transform before use in linear models.

| Feature | Window |
|---|---|
| dollar_volume_21, dollar_volume_63 | 21, 63 days |

### 2.4 Extreme / 52-Week High
```
dist_{h}_high = prev_close / rolling_max(close, h) - 1   [always ≤ 0]
max_{h}       = rolling max of daily returns over h days, shifted by 1
```

| Feature | Window |
|---|---|
| dist_252_high | 252-day rolling max |
| max_21 | 21-day rolling max return |

`dist_252_high` is bounded at 0 on the right. Spike at 0 = stock at its 52-week high.

### 2.5 Skewness
```
skew_{h} = rolling skewness of daily returns over h days, shifted by 1
```

| Feature | IC Result |
|---|---|
| skew_21 | Near zero IC, not significant |
| skew_63 | Slightly negative IC |
| skew_252 | Marginally positive ICIR=0.063, not significant |

Weak feature overall. Keep `skew_252` only if including at all.

### 2.6 Beta and Idiosyncratic Return
```
rolling_beta_{t} = Cov(r_i - RF, Mkt-RF) / Var(Mkt-RF)   [252-day window, shifted by 1]
idiosyncratic_{t} = (r_i - RF) - rolling_beta * (Mkt-RF)
```

⚠️ `Mkt-RF` from Fama-French is the **excess** market return — do NOT add RF back.

```python
Mkt_minus_RF = factors_6f.loc[common_dates]['Mkt-RF']   # correct
# NOT: factors_6f['Mkt-RF'] + factors_6f['RF']           # wrong — this is raw market return
```

Idiosyncratic momentum:
```
idio_mom_{h} = sum of idiosyncratic returns from t-21 to t-21-h
             = groupby Ticker: idiosyncratic.shift(21).rolling(h).sum()
```

| Feature | IC Result |
|---|---|
| rolling_beta | ICIR=0.096, marginally significant (p=0.08) |
| idiosyncratic | Near-zero IC, not useful as standalone feature |

### 2.7 Amihud Illiquidity
```
illiq_{t} = |return_{t}| / dollar_volume_{t}
amihud_{h} = rolling mean of illiq over h days, shifted by 1
```

⚠️ Replace `dollar_volume = 0` with NaN before dividing to avoid infinities.

```python
illiq = returns.abs() / dollar_volume.replace(0, np.nan)
```

**Dominant signal** in the dataset. Captures the illiquidity premium: illiquid stocks earn higher future returns as compensation for bearing liquidity risk.

| Feature | ICIR | NW t-stat | Significant |
|---|---|---|---|
| amihud_21 | 0.414 | 8.11 | ✓✓ |
| amihud_63 | 0.406 | 7.95 | ✓✓ |
| amihud_252 | 0.397 | 7.73 | ✓✓ |

### 2.8 Parkinson Volatility
```
log_hl_{t}       = (ln(High_t / Low_t))^2
parkinson_{t}    = log_hl_t / (4 * ln(2))
parkinson_vol_{h} = sqrt(rolling mean of parkinson_t over h days, shifted by 1)
```

Use `** 0.5` not `.apply(np.sqrt)` for vectorised performance.

Highly correlated with realised volatility (vol_h). Positive IC, significant.

### 2.9 Volume Surprise
```
avg_vol_{h}       = groupby Ticker: Volume.shift(1).rolling(h).mean()
prev_vol          = groupby Ticker: Volume.shift(1)   [NOT global shift]
vol_surprise_{h}  = prev_vol / avg_vol_{h} - 1
```

⚠️ Always use `groupby Ticker transform shift(1)` — a global `df["Volume"].shift(1)` contaminates observations across ticker boundaries.

**Negative IC** — high volume surprise predicts lower future returns (consistent with attention-driven demand).

### 2.10 Return Z-Score
```
rolling_mean_{h} = groupby Ticker: return.shift(2).rolling(h).mean()   [excludes yesterday]
rolling_std_{h}  = groupby Ticker: return.shift(2).rolling(h).std()
return_zscore_{h} = (return.shift(1) - rolling_mean_{h}) / rolling_std_{h}
```

`shift(2)` is intentional — excludes yesterday's return from the historical mean to avoid contamination.

**Negative IC** — high z-score (yesterday's return unusually high) predicts lower future returns (short-term reversal).

### 2.11 Price-Volume Interaction
```
pv_interaction_{h} = return_zscore_{h} × vol_surprise_{h}
```

Near-zero IC across all windows. Not contributing meaningfully.

---

## 3. FEATURE_CONFIG (Full Generation Set)

```python
FEATURE_CONFIG = {
    "mom_windows":         [(1,5), (1,21), (21,126), (21,252), (126,252), (252,756)],
    "vol_windows":         [21, 63, 252],
    "liq_windows":         [21, 63],
    "high_windows":        [252],
    "max_windows":         [21],
    "skew_windows":        [21, 63, 252],
    "beta_window":         252,
    "amihud_windows":      [21, 63, 252],
    "parkinson_windows":   [21, 63, 252],
    "vol_surprise_windows":[21, 63, 252],
    "zscore_windows":      [21, 63, 252],
    "pv_interaction_windows": [21, 63, 252],
}
```

---

## 4. IC Summary — Key Results

Sorted by ICIR. Features sorted into tiers:

**Tier 1 — Strong, statistically significant:**
- amihud_21, amihud_63, amihud_252 (ICIR ≈ 0.40, p < 0.001)
- parkinson_vol_63, vol_252, downside_dev_21, parkinson_vol_252 (ICIR ≈ 0.14–0.16)
- vol_63, vol_21, downside_dev_63, downside_dev_252 (ICIR ≈ 0.13–0.15)

**Tier 2 — Marginal signal:**
- max_21 (ICIR=0.107, p=0.03)
- rolling_beta (ICIR=0.096, p=0.08 — not quite significant)
- mom_252_756, mom_126_252 (ICIR ≈ 0.09, p=0.16 — not significant)

**Tier 3 — No signal:**
- skew_252, pv_interaction_252, mom_21_252, idiosyncratic

**Negative IC (contrarian signal):**
- vol_surprise_63 (significant, p=0.046)
- mom_1_21, dollar_volume_21, dollar_volume_63 (significant, negative)
- dist_252_high, return_zscore_21/63/252 (negative, reversal effects)

---

## 5. Bugs Found and Fixed

| Bug | Location | Fix |
|---|---|---|
| `Mkt-RF + RF` gives raw market return, not excess | notebook | Use `factors['Mkt-RF']` only |
| `vol_{h}` missing `shift(1)` before rolling std | feature_engineer.py | Add `.shift(1)` |
| `ls_spread` computed on all dates, not subsampled | utils.py | Filter `pred_df_sampled` before both `ic_series` AND `ls_spread` |
| `rolling_beta` look-ahead: not shifted after OLS | feature_engineer.py | `.groupby('Ticker')['rolling_beta'].shift(1)` after computation |
| Amihud: division by zero when volume=0 | feature_engineer.py | `.replace(0, np.nan)` on dollar_volume before dividing |
| Parkinson: `.apply(np.sqrt)` is slow element-wise | feature_engineer.py | Replace with `** 0.5` |
| `vol_surprise`: global `df["Volume"].shift(1)` contaminates across tickers | feature_engineer.py | `groupby("Ticker").transform(lambda s: s.shift(1))` |
| `mom_21_126` used instead of `mom_126_252` in FEATURE_COLS | notebook | These are completely different features with opposite IC |
| Cross-sectional ranking commented out | notebook | Uncomment `df[FEATURE_COLS] = df.groupby("Date")[FEATURE_COLS].rank(pct=True)` |
| `BETA_WINDOW = 50` | notebook | Too short for stable OLS. Use 252 |
| Amihud log transform ineffective (values ~1e-7) | notebook | `log(1+x) ≈ x` for tiny values — use `log(x)` directly for visualization |
| Quintile plot title says "Q5-Q1" but uses deciles | notebook | Fix label to "Q10-Q1" |

---

## 6. Key Design Decisions and Rationale

**Why cross-sectional ranking?** Removes time-series level effects (bull vs bear market). The model learns relative ordering of stocks, not absolute return levels. Makes features comparable across decades.

**Why Spearman IC, not Pearson?** Features are skewed (Amihud especially). Spearman ranks first, making it robust to outliers. Also naturally aligns with the cross-sectional ranking framework.

**Why Newey-West and not standard t-test?** IC observations are autocorrelated when using overlapping forward returns. Standard t-test gives inflated t-statistics by up to √21.

**Why 21-day subsampling for IC?** Makes IC observations non-overlapping when using 21-day forward returns. Without this, adjacent IC observations share 20/21 of the same return window.

**Why embargo of 21 days?** Prevents the model from using information that would not have been available at prediction time due to settlement delays and computation lags.

**Why beta window = 252?** Rolling OLS beta requires sufficient data to be stable. 50-day window gives noisy, unstable estimates. 252 days (~1 year) is standard in industry.

**Why idio momentum uses shift(21)?** Skips the most recent month to avoid the short-term reversal effect contaminating the medium-term signal.

---

## 7. Quintile Profile Findings

All top 8 features by ICIR show near-perfect monotone staircases (monotonicity ρ = 0.95–1.00). Key observation: **alpha is concentrated in the tails** — Q1 and Q10 drive the spread, while Q3–Q8 hover near 0.50. This means long-short implementations should focus on the extremes.

Q10–Q1 spread ≈ 0.035–0.046 in rank units across all top features.

---

## 8. What Comes Next

- [ ] Run walk-forward backtest with final feature set
- [ ] Compute L/S returns in actual return units (not rank units)
- [ ] Run Fama-French factor regression to check if alpha survives
- [ ] Consider three-model architecture (short/medium/long horizon)
- [ ] ICIR-weighted blending of models
- [ ] Turnover analysis
- [ ] Write README explaining methodology
