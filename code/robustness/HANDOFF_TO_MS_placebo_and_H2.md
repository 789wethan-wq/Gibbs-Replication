# Handoff to MS/chat instance — placebo-table prose + §2.5 distinctiveness criterion

Source: `FINAL_PAPER_V26_PLOS.docx` (built on V25, NOT canonical — do not merge
structurally, just lift the prose/table below). Underlying data:
`results/revision/D1_placebo_characteristic_test.txt` (full run log, already
on disk, path stable).

Note per the hand-off: reliability numbers cited inside the §4.4 prose below
came from the FIRST D2 pass and have since been corrected — the chat instance
should hold off on citing D2 reliability figures in the manuscript until the
corrected rerun (`results/revision/D2_corrected_split_half.txt`, in progress)
lands. Everything below is D1-only (placebo-characteristic test), which is
unaffected by the D2 correction.

---

## 1. New §4.4 prose (two paragraphs), to insert after the paragraph ending
"...a distinct question we address directly next." and before the existing
measurement-error footnote paragraph:

### Paragraph A (methodology + D1.1 result)

> Placebo-characteristic test. β̂_ΔS,t is a cross-sectional slope on a
> standardized regressor, proportional to cov(r_{i,t+1}, ΔS^z_{i,t}); since
> cross-sectional return dispersion rises with market volatility
> (Corr(σ_cs,t+1, T_t) = +0.39 in-sample, +0.47 survivorship-corrected), any
> characteristic with a non-zero mean loading could produce a second-step
> slope that co-varies with T mechanically, with no conditional-pricing
> content whatsoever. To test whether the entropy-temperature link is
> distinctive, we re-run the identical first-step-then-HAC design (same
> panel, same controls, same months, β̂_char,t regressed on T_t) for four
> characteristics unrelated to disorder: size (log market capitalization),
> book-to-market, momentum (12-1), and market beta. Table 4.4a reports the
> results: three of four placebo characteristics (size, book-to-market,
> momentum) show HAC t > 2 in both panels; the fourth (beta) is significant
> in the S&P 500 panel (HAC t = +2.61) but not in the full-universe panel
> (HAC t = +1.25). Four of four placebo characteristics are significant on T
> in at least one panel.

### Table 4.4a (final numbering pending the Table renumbering pass)

| Characteristic | S&P 500 HAC-t | Full-Universe HAC-t | Note |
|---|---|---|---|
| Size (log mkt cap) | −2.89 | −2.72 | significant, both panels |
| Book-to-market | +2.24 | +2.65 | significant, both panels |
| Momentum (12-1) | −2.15 | −4.00 | significant, both panels |
| Market beta | +2.61 | +1.25 | significant S&P only |
| ΔS (paper's result, raw) | +2.25 | +2.70 | reference row |
| ΔS, dispersion-normalized | +1.39 | +2.57 | S&P drops below significance; full-universe survives |
| Stacked diff test, raw | t=+2.23, p=0.026 | t=+2.26, p=0.024 | date-clustered |
| Stacked diff test, normalized | t=+1.25, p=0.213 | t=+2.21, p=0.027 | date-clustered |

### Paragraph B (D1.2/D1.3 + interpretation)

> Normalizing the entropy loading by the same-period cross-sectional standard
> deviation of the priced return (β̂_ΔS,t / σ_cs,t+1) partially separates the
> mechanical dispersion channel from a genuine conditional-pricing signal. In
> the S&P 500 in-sample panel, the normalized HAC t falls from +2.25 (raw) to
> +1.39, below conventional significance. In the survivorship-corrected
> full-universe panel — the panel this paper's headline asymmetric-prediction
> claim is built on — the normalized HAC t is +2.57 (p = 0.010), close to the
> raw +2.70 and still significant. The stacked date-clustered slope-difference
> test (the direct test of asymmetry introduced above) shows the same pattern
> when re-run on normalized loadings: the full-universe difference remains
> significant (d = +1.65, t = +2.21, p = 0.027) while the S&P 500 difference
> does not (d = +0.74, t = +1.25, p = 0.213). We draw two conclusions. First,
> on the placebo count alone, the asymmetric temperature prediction cannot be
> treated as distinctive evidence for the thermodynamic mapping: loading on T
> is the generic case across the characteristics tested, not a property
> unique to entropy, so we withdraw the earlier claim that this asymmetry is
> the paper's strongest and most credible evidence that the thermodynamic
> mapping captures something structural. Second, the dispersion-normalized
> result in the survivorship-corrected full-universe panel — the sample the
> paper's headline claim is drawn from — remains significant after this
> correction, so there is a residual conditional relationship between the
> entropy loading and market temperature that is not purely mechanical. We
> report this as a directional conditional-pricing observation, not a
> thermodynamically specific one: something about the entropy channel's
> pricing does vary with market temperature beyond what dispersion scaling
> alone predicts, but the placebo battery removes our basis for attributing
> that residual to the Gibbs functional form rather than to some other
> conditional-pricing mechanism operating on idiosyncratic-volatility-sorted
> portfolios generally. See Section 5.1 for the resulting reframing of the
> paper's contribution.

**Construction check (for referee-facing footnote if useful):** the
z-scoring convention used for all four placebo characteristics is
byte-identical to R20's `cs_wz` (the function that builds ΔS_z itself), and
the same first-step regression retains the same month/quarter count (334
months S&P, 111 quarters full-universe) across all four placebos and the ΔS
baseline — same panel, same controls, same months, confirmed not just
asserted.

---

## 2. §2.5 — H2 statement (replaces the existing H2 bullet)

> H2 (T-Scaling): The temperature interaction T·ΔS is significant in
> specifications that identify it — the cluster-robust Wald test, the pooled
> two-way-clustered interaction, and the asymmetric HAC test — and, if the
> Gibbs mapping is thermodynamically specific rather than a generic
> dispersion-scaling channel, this T-covariation should be distinctive to the
> entropy channel and largely absent from unrelated characteristics. Outcomes
> are reported in Sections 4.3, 4.4, and 4.7: the plain significance claim
> holds (Wald p = 0.013/0.017; pooled interaction t = +2.49 both panels;
> asymmetric HAC t = +2.25 in-sample / +2.70 survivorship-corrected). The
> distinctiveness sub-claim does not hold: a placebo test of size,
> book-to-market, momentum, and beta through the identical design
> (Section 4.4) finds comparable temperature-covariation in most of them,
> consistent with a generic dispersion-scaling mechanism rather than a
> channel specific to disorder. (The encompassing-model FM is uninformative
> for T·ΔS due to within-cross-section collinearity, and Model C FM does not
> by itself isolate T-scaling from level pricing of ΔS; see Section 4.3.)

## 3. §2.5 — Falsifiability criterion, third bullet (add to the existing
Strong/Partial falsification bullets)

> Distinctiveness falsification (added post hoc, following referee review):
> if two or more of four placebo characteristics unrelated to disorder
> (size, book-to-market, momentum, beta) show second-step
> temperature-covariation of comparable magnitude and significance to ΔS
> under the identical design, the asymmetric T-scaling result cannot be
> attributed to the Gibbs mapping specifically and must be treated as a
> generic conditional-pricing / dispersion-scaling pattern rather than
> structural confirmation of the thermodynamic form.

## 4. §2.5 — Verdict paragraph (replaces "Neither strong nor partial
falsification applies...")

> Strong falsification does not apply: the Wald test clears p < 0.05 in both
> constructions. Partial falsification applies in a limited sense via the
> post-2009 weakness. Distinctiveness falsification applies: four of four
> placebo characteristics show significant temperature-covariation in at
> least one panel (three of four in both), so the asymmetric T-scaling
> result is downgraded from a structural, thermodynamically-specific claim to
> a directional conditional-pricing observation consistent with, but not
> diagnostic of, the Gibbs functional form. This is the paper's most
> consequential post hoc falsification result and the primary reason for
> reframing the paper's contribution around the survivorship correction
> rather than the thermodynamic mapping (Section 5.1).

---

## Also flagging (not part of the requested handoff, but relevant to V27)

- No `FINAL_PAPER_V27_PLOS.docx` exists in this repo as of this handoff —
  only `FINAL_PAPER_V25_PLOS.docx` and my (non-canonical) `FINAL_PAPER_V26_PLOS.docx`.
  If V27 lives elsewhere, disregard; if it doesn't exist yet, the chat
  instance will need the V25 base plus its own ~25 tracked edits plus this
  handoff to construct it.
- D1's full run log (all four sub-tests, alignment audit) is at
  `results/revision/D1_placebo_characteristic_test.txt` if the chat instance
  wants primary-source numbers beyond what's excerpted above.
