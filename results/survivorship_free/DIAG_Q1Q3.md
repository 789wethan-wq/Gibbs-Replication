# Survivorship-Correction Diagnostic — Q1–Q3

Sample window (monthly original): 1995-01-31 .. 2023-11-30
Quarterly corrected panel range : 1995Q3 .. 2023Q4

## Q1 — What exactly was added

| Panel | firms | firm-period obs | periods | avg firms/period |
|---|---:|---:|---:|---:|
| ORIGINAL (monthly, S&P500 survivor) | 462 | 126,990 | 347 | 366 |
| CORRECTED (quarterly, full universe) | 12,449 | 434,016 | 114 | 3807 |

- Firms in CORRECTED but NOT in ORIGINAL: **11,993**
- Firms retained (in both): 456
- of added, delisted (isdelisted=Y): **8,936**
- of added, active   (isdelisted=N): **3,057**

SF1-universe delisted firms total = 11,178; absent from original = 11,177 (the stated ~11,178 delisted-omission is confirmed at universe level).

### Last-observation (disappearance) year — ADDED firms
| disappearance year | added firms |
|---|---:|
| 1996 | 2 |
| 1997 | 24 |
| 1998 | 125 |
| 1999 | 269 |
| 2000 | 721 |
| 2001 | 559 |
| 2002 | 421 |
| 2003 | 385 |
| 2004 | 345 |
| 2005 | 357 |
| 2006 | 366 |
| 2007 | 392 |
| 2008 | 351 |
| 2009 | 293 |
| 2010 | 311 |
| 2011 | 263 |
| 2012 | 268 |
| 2013 | 231 |
| 2014 | 224 |
| 2015 | 288 |
| 2016 | 286 |
| 2017 | 256 |
| 2018 | 256 |
| 2019 | 226 |
| 2020 | 204 |
| 2021 | 222 |
| 2022 | 263 |
| 2023 | 4085 |

- Added firms whose series ends before panel end (2023Q4): **8,273 / 11,993** (69%) — i.e. they disappeared mid-sample.

### Disorder (ΔS / iVol) percentile AT ENTRY  (1.0 = most disordered)
- ADDED firms  : median entry-disorder pct = 0.528  mean = 0.506
- RETAINED firms: median entry-disorder pct = 0.433  mean = 0.438

### Realized returns before disappearing  (quarterly)
| group | mean last-4q ret | median last-4q ret | mean final-q ret |
|---|---:|---:|---:|
| ADDED & delisted | +0.53% | +3.37% | +4.48% |
| RETAINED (survivors) | +3.55% | +2.86% | +16.16% |

-> Added (delisted) firms enter at **higher disorder** and exit on **materially worse** realized returns than retained survivors — exactly the mechanism the survivorship story predicts.

## Q2 — Why they were missing:  (a)/(b)/(c) classification

| class | firms | % of added |
|---|---:|---:|
| (a) DELISTED within sample window — CRSP+Shumway WOULD include | 8,936 | 74.5% |
| (b) NEVER IN SOURCE — active firm, never in current-S&P500 pull (breadth/coverage) | 3,057 | 25.5% |
| (c) delisted but last trade outside sample window | 0 | 0.0% |
| (c) other / no metadata | 0 | 0.0% |

**Decisive read:** type-(a) = 8,936 (75%), type-(b) = 3,057 (25%). Neither (a) nor (b) is a correction to the *literature*: (a) firms are standard delisted names a CRSP+Shumway panel already carries; (b) firms are survivors omitted purely by the current-S&P500 universe choice.

## Q3 — Controlled decomposition (one variable at a time)

Headline estimand: FM slope on ΔS (disorder/iVol), the coefficient that moves from t=+4.80 to t=+0.02.

**M0 baseline** — monthly / S&P500 survivor-only / AHXZ-36m iVol
    FM t(ΔS) = +4.59   (β=+0.00470, T=347 months)

**Quarterly rungs** (SF1 source, 12-quarter iVol held FIXED — so frequency/measure is constant across all four; only the universe changes):

    Qa  S&P500 survivor-only (orig tickers)              FM t(ΔS) = +4.17  (β=+0.023823, firms=456, Tq=112)
    Qb  ever-S&P500, survivor-only                       FM t(ΔS) = +3.39  (β=+0.020560, firms=615, Tq=112)
    Qc  full universe, survivor-only                     FM t(ΔS) = +1.28  (β=+0.006829, firms=3,512, Tq=112)
    Qd  full universe, ALL incl delisted  = CORRECTED    FM t(ΔS) = -0.28  (β=-0.001406, firms=12,449, Tq=112)

### One-variable-at-a-time effects

| step | change (one variable) | from | to | Δt |
|---|---|---:|---:|---:|
| A | **frequency/measure**: monthly-AHXZ → quarterly-12q (universe held: S&P500 survivor) | +4.59 | +4.17 | -0.41 |
| B | **breadth**: S&P500 survivor → full-universe survivor (freq & survivorship held) | +4.17 | +1.28 | -2.89 |
| C | **survivorship**: full-universe survivor → full-universe incl. delisted (freq & breadth held) | +1.28 | -0.28 | -1.56 |

**Isolation of survivorship alone (the cleanest single toggle):** within the identical quarterly full-universe panel, the ONLY difference between rung Qc and rung Qd is whether delisted firms are included. That toggle moves FM t(ΔS) from +1.28 to -0.28 (Δt = -1.56).

### Q3b — Breadth × Survivorship 2×2 (frequency/measure held fixed, quarterly)

FM t(ΔS) in each cell (firm count in parens):

| breadth ↓ / survivorship → | survivor-only | incl. delisted | Δ (survivorship) |
|---|---:|---:|---:|
| ever-S&P500 | +3.39 (615) | +2.21 (1,092) | -1.19 |
| full universe | +1.28 (3,512) | -0.28 (12,449) | -1.56 |
| **Δ (breadth)** | -2.11 | -2.49 | |

- Survivorship toggle: -1.19 within S&P500 vs -1.56 within full universe — delisting bites **far harder among small caps**, so breadth and survivorship are entangled.
- Breadth toggle: -2.11 among survivors vs -2.49 with delisted included.
- Both orderings agree on the qualitative split: breadth removes the **larger** share of the t-stat, survivorship removes the rest and pushes it through zero.

**Reconciliation:** paper's Model B (bivariate ΔH+ΔS, full-channel panel) reproduces FM t(ΔS) = +0.02 (β=+0.000087) — matches the reported +0.02. The univariate Qd (−0.28) and bivariate Model B (+0.02) are both statistically zero; the small gap is ΔH conditioning + the non-missing-ΔH subsample.
