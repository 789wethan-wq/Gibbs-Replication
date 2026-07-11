# Q4 — Honest Contribution Statement

## Verdict: **[B] — with a mandatory amendment**

The evidence supports **[B]** (survivorship resurfaces in a practitioner-grade
non-CRSP dataset, with material effect on the IVOL/disorder premium) — **but not
[B] as currently written in `SUMMARY.txt`.** The controlled Q3 decomposition shows
the +4.80 → +0.02 collapse is **not** "largely a survivorship artifact." It is two
distinct effects, with survivorship the *second* largest.

### The numbers that justify it

FM t(ΔS) decomposed, one variable at a time (monthly baseline +4.59 → corrected −0.28,
total drop = 4.87 t-units):

| driver | Δt | share of collapse | what it is |
|---|---:|---:|---|
| **Breadth** (S&P500 large-cap → full US-common universe) | −2.89 | **59%** | external-validity failure |
| **Survivorship** (add delisted firms) | −1.56 | **32%** | the classic bias |
| **Frequency/measure** (monthly-AHXZ → quarterly-12q) | −0.41 | 8% | immaterial |

- **[A] is ruled out.** 74.5% of added firms are type-(a) delisted-within-window
  names any CRSP+Shumway(1997) panel already carries; 25.5% are type-(b) active
  survivors omitted by the universe choice. Nothing here corrects the *literature's
  methodology* — it corrects an omission specific to the original yfinance /
  current-S&P500 pull (Q0).
- **[B] holds but must be split.** Survivorship is real and decisive at the margin
  (it pushes the t-stat through zero), and it bites harder among small caps
  (−1.56 full-universe vs −1.19 within S&P500). But the *larger* share of the
  collapse is that the premium is a large-cap phenomenon that does not generalize.

### Honest one-liner for the paper

> The +4.80 disorder/IVOL premium is a **large-cap, survivor-only** result. Roughly
> **60%** of its disappearance is failure to generalize beyond the S&P 500, and
> **~30%** is survivorship among the (mostly small-cap) delisted tail; frequency is
> immaterial. Because the quarterly SF1 panel applies no true delisting return, the
> survivorship share is a **lower bound** (R19's synthetic Shumway stress is the
> complementary upper bound).

### Load-bearing caveat

The quarterly SF1 panel has **no Shumway delisting return** — a firm that goes to
zero simply stops appearing and its last quarterly return is winsorized at the 1st
percentile. So Q3's survivorship step (−1.56) is *understated*. The true figure sits
between the Q3 lower bound and the R19 synthetic upper bound. This does not change
the verdict but does mean "survivorship + breadth" is the honest framing, and the
relative shares (~60/30) should be quoted as approximate.
