"""Documentation query 1 — Table 5 (Markov regime) uses smoothed_marginal_probabilities
(06_regime_analysis.py line: `smooth = res.smoothed_marginal_probabilities`). Smoothed
probabilities condition on the FULL sample (backward + forward pass) and constitute
look-ahead when used to partition returns for an in-sample conditional test. This
script fits the identical model and reports the FILTERED (forward-pass-only,
no-look-ahead) classification alongside, to quantify how much the 61.4% figure and
the high/low split move.
"""
import numpy as np
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"

panel = pd.read_parquet(f"{DATA}/variables_monthly.parquet")
panel["date"] = pd.to_datetime(panel["date"])
T_ts = panel.groupby("date")["T_raw"].first().sort_index()

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
mod = MarkovRegression(T_ts.dropna(), k_regimes=2, switching_variance=True)
res = mod.fit(disp=False, maxiter=500)

smooth = res.smoothed_marginal_probabilities
filt = res.filtered_marginal_probabilities

state_means_sm = [T_ts.dropna()[smooth.iloc[:, i] > 0.5].mean() for i in range(2)]
high_state_sm = int(np.argmax(state_means_sm))
high_prob_sm = smooth.iloc[:, high_state_sm]
assign_sm = (high_prob_sm > 0.5).astype(int)

state_means_f = [T_ts.dropna()[filt.iloc[:, i] > 0.5].mean() for i in range(2)]
high_state_f = int(np.argmax(state_means_f))
high_prob_f = filt.iloc[:, high_state_f]
assign_f = (high_prob_f > 0.5).astype(int)

print(f"N months = {len(T_ts.dropna())}")
print(f"SMOOTHED : high-T months = {assign_sm.sum()} ({assign_sm.mean()*100:.1f}%)")
print(f"FILTERED : high-T months = {assign_f.sum()} ({assign_f.mean()*100:.1f}%)")
agree = (assign_sm.values == assign_f.values).mean()
print(f"Classification agreement (smoothed vs filtered): {agree*100:.1f}% of months")
disagree_idx = T_ts.dropna().index[assign_sm.values != assign_f.values]
print(f"Months where classification differs: {len(disagree_idx)}")
if len(disagree_idx) > 0:
    print(f"  first few: {list(disagree_idx[:10].strftime('%Y-%m'))}")

# average probability difference, useful to see if it's borderline-only churn
prob_diff = (high_prob_sm - high_prob_f).abs()
print(f"Mean |smoothed_prob - filtered_prob| = {prob_diff.mean():.4f}, "
      f"median = {prob_diff.median():.4f}, max = {prob_diff.max():.4f}")
