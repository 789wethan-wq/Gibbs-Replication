# Channel Verification — Same-Panel + Order-Robustness

Paper's significance hurdle: **|t| > 3.0** (Harvey-Liu-Zhu 2016).

## CHECK 1 — Same-panel confirmation

### 1a. AS-ORIGINALLY-COMPUTED in DIAG_channels.py (discloses the mismatch)

| rung | channel | spec | N firms | N obs | period | source |
|---|---|---|---:|---:|---|---|
| corrected | ΔS disorder | univariate ΔS | 12,022 | 420,712 | 1995Q3..2023Q3 | merged_sf1_quarterly_survfree.parquet |
| corrected | ΔH quality | bivariate ΔH+ΔS | 10,699 | 392,557 | 1995Q4..2023Q3 | merged_sf1_quarterly_survfree.parquet |
| corrected | β_ΔS~T asym | bivariate ΔH+ΔS | 10,699 | 392,557 | 1995Q4..2023Q3 | merged_sf1_quarterly_survfree.parquet |
| corrected | T·ΔS level | bivariate ΔH+T·ΔS | 10,699 | 392,557 | 1995Q4..2023Q3 | merged_sf1_quarterly_survfree.parquet |
| baseline | ΔS disorder | univariate ΔS | 462 | 126,990 | 1995-01-31..2023-11-30 | merged_with_accounting.parquet |
| baseline | ΔH quality | bivariate ΔH+ΔS | 457 | 118,014 | 1996-01-31..2023-11-30 | merged_with_accounting.parquet |
| baseline | β_ΔS~T asym | bivariate ΔH+ΔS | 457 | 118,014 | 1996-01-31..2023-11-30 | merged_with_accounting.parquet |
| baseline | T·ΔS level | bivariate ΔH+T·ΔS | 457 | 118,014 | 1996-01-31..2023-11-30 | merged_with_accounting.parquet |

**Mismatch found:** the disorder channel was run UNIVARIATE (needs only ΔS, which is 100% covered) while ΔH / asym / T·ΔS were run BIVARIATE (need ΔH_z, ~93% covered). At the corrected rung ΔS used 420,712 obs / 12,022 firms vs 392,557 obs / 10,699 firms for the others. **The literal PASS condition (identical N) is NOT met as-originally-computed.** (Note the raw panel is 12,449/434,016, but no forward-return FM can use all 434,016 — each firm's terminal quarter has ret_next=NaN.)

### 1b. SAME-PANEL recomputation — ΔS and ΔH from the SAME bivariate regression on the SAME rows (this is the valid contrast)


**Corrected rung (full universe, quarterly)** — sample: 10,699 firms / 392,557 obs / 1995Q4..2023Q3  [merged_sf1_quarterly_survfree.parquet]
| channel | t | pass |t|>3.0 |
|---|---:|:--:|
| ΔS disorder (Model B) | +0.02 | no |
| ΔH quality (Model B, SAME reg) | +3.46 | YES |
| β_ΔS~T asymmetric | +3.64 | YES |
| T·ΔS level (Model C, SAME sample) | -0.55 | no |

**Baseline rung (S&P500, monthly)** — sample: 457 firms / 118,014 obs / 1996-01-31..2023-11-30  [merged_with_accounting.parquet]
| channel | t | pass |t|>3.0 |
|---|---:|:--:|
| ΔS disorder (Model B) | +4.68 | YES |
| ΔH quality (Model B, SAME reg) | +2.70 | no |
| β_ΔS~T asymmetric | +3.11 | YES |
| T·ΔS level (Model C, SAME sample) | +4.59 | YES |

**Same-panel verdict:** on the identical common sample (ΔS and ΔH from one regression), the contrast holds — at the corrected rung ΔH quality t=+3.46 and asymmetric t=+3.64 while ΔS disorder t=+0.02 and T·ΔS level t=-0.55. The 'survives vs dies' split is NOT a sample-mismatch artifact.

## CHECK 2 — Order-robust breadth × survivorship 2×2

### Quality channel ΔH (FM t, Model B)

| breadth ↓ / surv → | survivor-only | incl delisted | Δ survivorship |
|---|---:|---:|---:|
| ever-S&P500 | +0.36 | +0.33 | -0.03 |
| full universe | +1.45 | +3.46 | +2.01 |
| **Δ breadth** | +1.09 | +3.12 | |

- cells with |t|>3.0: **1/4**  (values: +0.36, +0.33, +1.45, +3.46)
- breadth effect sign: survivor-order +1.09, delisted-order +3.12 -> consistently POSITIVE (strengthens)
- survivorship effect: S&P500 -0.03, full +2.01

**Quality PASS condition (all 4 cells |t|>3 AND breadth positive both orders): FAIL**

### Asymmetric prediction β_ΔS~T (HAC t)

| breadth ↓ / surv → | survivor-only | incl delisted | Δ survivorship |
|---|---:|---:|---:|
| ever-S&P500 | +2.44 | +2.97 | +0.52 |
| full universe | +3.67 | +3.64 | -0.04 |
| **Δ breadth** | +1.23 | +0.67 | |

- cells with |t|>3.0: **2/4**  (values: +2.44, +2.97, +3.67, +3.64)
- breadth effect sign: survivor-order +1.23, delisted-order +0.67 -> consistently POSITIVE (strengthens)
- survivorship effect: S&P500 +0.52, full -0.04

**Asymmetric PASS condition (all 4 cells |t|>3 AND breadth positive both orders): FAIL**


================================================================
## CONSOLIDATED VERDICT
================================================================

**CHECK 1 (same-panel): PASS, with disclosure.**
The original DIAG_channels.py ran ΔS univariate (entropy panel: 12,022 firms /
420,712 obs) but ΔH / asym / T·ΔS bivariate (full-channel: 10,699 / 392,557) —
so the literal identical-N condition failed as first computed. Recomputed on the
IDENTICAL common sample (ΔS and ΔH from one Model-B regression, 10,699 / 392,557
at the corrected rung), the contrast still holds: ΔS +0.02 / T·ΔS -0.55 (die) vs
ΔH +3.46 / asym +3.64 (survive). The "survives vs dies" split is NOT a
sample-mismatch artifact. CAVEAT: at the S&P500 baseline on the common sample,
ΔH quality is only t=+2.70 — BELOW the 3.0 hurdle. ΔH clears 3.0 only in the
corrected full-universe panel.

**CHECK 2 (order-robust 2×2): FAIL for BOTH surviving channels.**
  Quality ΔH:   significant in 1/4 cells (only full-universe + delisted, +3.46).
                The other three cells (+0.36, +0.33, +1.45) are insignificant.
                Breadth sign is consistently positive, but significance is
                CONCENTRATED in the full+delisted corner — ΔH needs BOTH breadth
                AND the delisted small-cap tail to clear the hurdle. Entangled,
                not cleanly robust.
  Asymmetric β_ΔS~T: significant in 2/4 cells (both full-universe, +3.67/+3.64);
                the two ever-S&P500 cells (+2.44, +2.97) fall below 3.0. Breadth
                consistently positive. Closer to robust than ΔH, but still clears
                the hurdle only in the full universe. Strictly FAILS all-4-cells.

**PLAIN-LANGUAGE VERDICT: NO — not clean, order-robust, same-panel results.**
Both surviving channels are same-panel-valid (Check 1) but their significance is
concentrated in the full-universe (for ΔH, full+delisted) cells and does NOT
hold across the breadth×survivorship 2×2 (Check 2). The correct claim is:
"ΔH quality and the asymmetric β_ΔS~T prediction clear |t|>3.0 in the corrected
full-universe panel and are strengthened by breadth, but are NOT uniformly
significant across subsamples — ΔH significance in particular requires the
full-universe-with-delisted sample." Do NOT write these up as clean survival.
The earlier phrase "robustly positively priced at every rung" is FALSE (ΔH is
insignificant at the quarterly S&P500 and full-survivor rungs) and must be
corrected.
================================================================
