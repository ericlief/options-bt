# Vol-Scalar Bounding in CTA/Trend-Following Position Sizing

Research date: 2026-06-25

## Question

In volatility-targeting / inverse-volatility position sizing used by systematic trend-following / CTA strategies, is it standard, documented practice to bound/clamp the vol-scalar (the leverage multiplier `vol_target / realized_vol`) to prevent extreme leverage when realized vol is very low, or extreme de-risking when it's very high?

### Context

This system uses:

```
vol_scalar = clamp(vol_target / realized_vol, 0.25, 2.0)
scalar = trend_strength * vol_scalar * regime_discount
fractional_target_notional = budget_constant * scalar
budget_constant = (account_equity * target_portfolio_vol / sqrt(n_effective)) / vol_target
```

`vol_target` cancels out of the final position-sizing formula exactly when `vol_scalar` is unclamped. When the clamp is active, `vol_target` stops canceling and reappears as a free parameter — meaning clamped instruments respond to changes in `vol_target`'s absolute value, while unclamped instruments don't. This asymmetry was identified as a real, narrow side effect of bounding the scalar at all, not necessarily a design flaw.

## (a) Is bounding/clamping the vol-scalar standard, documented practice?

**Yes, common — but not universal.** The bound mechanism and exact numeric values vary across sources.

- **AlphaArchitect**'s worked examples of volatility targeting state leverage is "calculated as target volatility divided by the actual 20-day volatility," with "the maximal leverage we use is capped at two" — a direct match to this system's upper bound of 2.0.
- **Bongaerts, Kang & van Dijk, "Conditional Volatility Targeting"** ([Financial Analysts Journal, 2020](https://repub.eur.nl/pub/130215/Bongaerts-Kang-van-Dijk-Conditional-volatility-targeting-2020-FAJ.pdf)) — **directly verified against the primary text.** Formula (3): `r_scaled_t = I_t × r_t × min(σ_target/σ̂_{t-1}, L_max) + (1-I_t) × r_t`, where `I_t=1` only in months pre-identified as an extreme (high- or low-) volatility quintile, with an *unscaled* exposure in medium-vol months. The `min(...,L_max)` term applies in both extreme states by construction but only ever binds in the low-vol/leverage-up case in practice (in high-vol states the ratio is already small). Verbatim: "We capped the leverage in the low-volatility states to prevent an increase in overall risk and drawdowns (our initial strategy capped the maximum risk exposure at 200%, but the finding is robust to different levels of leverage)." For the momentum factor, their uncapped/always-on "conventional" strategy reached **5.5x maximum leverage** (range 3.7–5.5 across markets) versus **2.0x** for the regime-gated, capped "conditional" version — directly, empirically motivating the cap. They also report the conditional strategy cuts annual turnover roughly in half versus the always-on conventional one (1.2 vs. 2.4 average turnover for momentum, 1.4 vs. 2.1 average across equity markets) — a second, independently verified benefit of *regime-gating* the clamp (only scaling in extreme states) distinct from the leverage-level cap itself.
  - **Methodological caveat, also from this same paper**: Bongaerts et al. critique both Moreira & Muir (2017) and Harvey et al. (2018) — two other sources cited above — for "look-ahead bias," because those papers' scaling constant is fit ex post to hit an exact full-sample volatility target rather than being computable in real time. This doesn't undermine the specific leverage-distribution figures cited from those papers here (a single overall multiplicative constant doesn't change the *relative* spread of period-to-period weights, e.g. Moreira & Muir's P99/P50 ratio), but it's a real, sourced disagreement between these papers on backtest methodology, worth knowing before leaning on their Sharpe-ratio/alpha claims specifically.
- **Robert Carver** (ex-Man AHL PM; [qoppac.blogspot.com](https://qoppac.blogspot.com)) confirms using a "V floor" in his own system for the same reason — abrupt vol-regime shifts (e.g. EUR/CHF pre-2015) producing wildly different position sizes depending on lookback. He handles extremely low-vol instruments by **excluding them from the tradable universe** rather than letting the scalar blow up.
- **Bernardi, Bianchi & Bianco, "Smoothing Volatility Targeting"** ([arXiv:2212.07288](https://arxiv.org/abs/2212.07288)) document the scale of the underlying problem: unbounded vol-targeting of equity strategies produces leverage of "1.8 to 4x for more than 10%" of cases and "3 to 11x for at least 1%" of cases. Verbatim: leverage constraints "in the form of a capped notional exposure... are typically set arbitrarily, absent sounded economic arguments for their optimal setup." They test the weights "capped so that the maximum leverage attainable is 500% (panel A) or 50% (panel B) of the original factor exposure" — one-sided maximum-only caps (5x / 1.5x), no floor.
- **Moreira & Muir, "Volatility-Managed Portfolios"** (*The Journal of Finance*, 2017; also circulated as [NBER Working Paper 22208](https://www.nber.org/papers/w22208)) — **directly verified against the primary text**, not just agent-summarized. They use `fσ_{t+1} = (c/σ̂²_t(f)) f_{t+1}` — inverse **variance**, not inverse vol, a different scaling exponent than this system's `vol_target/realized_vol`. Their uncapped market-portfolio weights reach **6.39x leverage at the 99th percentile** (Table V) — a third independent confirmation, alongside Bongaerts et al.'s 5.5x and Bernardi et al.'s "3 to 11x for at least 1%," that uncapped ratio-based vol targeting produces real, extreme leverage in practice, not just a theoretical tail risk. Their leverage-capped variants (`min(c/RV²,1)` and `min(c/RV²,1.5)`, one-sided max-only) confirm verbatim: "Sharpe ratios do not change with these leverage constraints, but... the leverage-constrained portfolios have lower alphas because risk weights are, on average, lower" — capping is a pure risk/practicality overlay with a real performance cost, not something that interacts with a target-vol parameter.
- **Counter-evidence it's not universal**: AQR's **"Chasing Your Own Tail (Risk), Revisited"** ([2019](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-Chasing-Your-Own-Tail-Risk-Revisited.pdf)) explicitly discloses that its illustrative trend-following backtest "does **not** limit volatility" during extreme periods. AQR's **"Demystifying Managed Futures"** (Journal of Investment Management) is a second, separate confirmation: position sizing is described as having "no free parameters or optimization in choosing the position sizes," targeting constant ex-ante volatility per instrument with no cap or floor mentioned anywhere. The foundational academic paper — **Harvey, Hoyle, Korgaonkar, Rattray, Sargaison & Van Hemert, "The Impact of Volatility Targeting"** ([Journal of Portfolio Management, 2018](https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf)) — presents the core formula `r_scaled = r_t × (σ_target/σ_{t-2}) × k` with no bound anywhere; a full-text search of the paper for "leverage/cap/floor/bound/clamp/limit" turns up only references to the equity *leverage effect* (Black 1976, debt-to-equity capital structure) — an unrelated use of the word "leverage" — confirming this canonical paper has nothing to say about position-sizing guardrails.

## (b) Does any source address the "clamp reintroduces target-vol dependence" mechanism?

**No.** None of the sources surveyed — Harvey et al., AQR's white papers, Man AHL/Carver, Winton, Bongaerts et al., Bernardi/Bianchi/Bianco, Research Affiliates — discuss the specific algebraic point that clamping `vol_target/realized_vol` causes `vol_target`'s absolute value to stop canceling for the instruments/periods where the clamp binds. This is genuinely undocumented territory in the literature, not a settled question either way.

## (c) How do real systems structurally avoid (or sidestep) this?

Several distinct approaches were found, none motivated by preserving the target's cancellation property specifically:

1. **Floor the realized-vol estimate** (the denominator), not the ratio — Carver's approach, and confirmed concretely in his open-source **pysystemtrade** config: `vol_abs_min: 0.0000000001`, an absolute floor on the realized-vol estimate itself, applied before any ratio is formed. This structurally cannot reintroduce target-vol dependence the way clamping the ratio does, since the floor lives entirely in denominator-space.
2. **Cap leverage in a separate, portfolio-level dollar/notional space, decoupled from the per-instrument ratio** — also confirmed concretely in pysystemtrade: a `risk_overlay` block (disabled by default) with keys `max_risk_leverage`, `max_risk_fraction_normal_risk`, `max_risk_fraction_stdev_risk`, `max_risk_limit_sum_abs_risk`. This is a real, named instance of exactly the "different space" mechanism this question is asking about — not hypothetical.
3. **Cap a trading *signal/forecast*** rather than the vol scalar — pysystemtrade's `forecast_cap: 20.0` (with `average_absolute_forecast: 10.0`) bounds the input signal before it ever reaches vol scaling, a third distinct space from either the ratio or the dollar-leverage overlay.
4. **Exclude the instrument from the universe** when its vol is too low to size sensibly (Carver, EUR/CHF pre-2015 peg).
5. **Cap leverage conditionally, in identified extreme states only** (Bongaerts et al.) — note this is mathematically the *same* ratio-clamp structure as this system, just one-sided and regime-gated. It has the identical algebraic property; the paper just never names it.
6. **Tie a separate exposure floor to realized drawdown, not to volatility at all** — AQR's "Chasing Your Own Tail, Revisited" describes a drawdown-control overlay that reduces total portfolio exposure in 14% increments down to a 50% minimum near a -10% drawdown, then rebuilds gradually back to 100% — entirely independent of the vol_target/realized_vol ratio.
7. **Smooth the volatility estimator itself** rather than bound the output ratio (Bernardi/Bianchi/Bianco) — replaces a hard discontinuity with reduced estimator noise.
8. **Don't bound at all, and disclose that explicitly** (AQR's "Chasing Your Own Tail" trend model, and separately "Demystifying Managed Futures").

## (d) Overall verdict

Bounding the vol-scalar is well-precedented, with real published numeric bounds matching or close to this system's `[0.25, 2.0]` (cap-at-2 in AlphaArchitect; 200%/2.0x in Bongaerts et al.). The Bongaerts et al. `min(ratio, L_max)` formula is structurally identical to this system's clamp and shares the identical algebraic property — and that peer-reviewed paper never raises it as an issue.

The specific "clamp reintroduces target-vol dependence" mechanism is not discussed anywhere in the academic or practitioner literature surveyed. Treating it as an accepted, narrow, deliberate side effect of bounding leverage is **consistent with the closest published methodology that exists**, not a rationalization invented to dismiss a real critique — but it is also not something the literature has explicitly debated or resolved. Where real systems do avoid the asymmetry, it's via a structurally different mechanism — most concretely, Carver's pysystemtrade floors the realized-vol *estimate* (`vol_abs_min`) and offers a separate dollar/leverage overlay (`risk_overlay`/`max_risk_leverage`) entirely decoupled from the per-instrument ratio, while Bernardi et al. smooth the estimator itself and AQR ties a drawdown-based exposure floor to realized losses rather than to volatility at all. None of these were motivated by a goal of preserving the target's cancellation property — that's an incidental side effect of choosing a different space to clamp in, not the stated design goal anywhere.

## Sources confirmed (text successfully extracted and quoted)

- Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, Van Hemert, ["The Impact of Volatility Targeting"](https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf), Journal of Portfolio Management, 2018 — full text searched for leverage/cap/floor/bound/clamp/limit; no position-sizing guardrail discussion found
- Bongaerts, Kang, van Dijk, ["Conditional Volatility Targeting"](https://repub.eur.nl/pub/130215/Bongaerts-Kang-van-Dijk-Conditional-volatility-targeting-2020-FAJ.pdf), Financial Analysts Journal, 2020
- Bernardi, Bianchi, Bianco, ["Smoothing Volatility Targeting"](https://arxiv.org/abs/2212.07288), arXiv:2212.07288, 2022
- Moreira, Muir, ["Volatility-Managed Portfolios"](https://www.nber.org/papers/w22208), NBER Working Paper 22208, 2017
- AQR, ["Chasing Your Own Tail (Risk), Revisited"](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-Chasing-Your-Own-Tail-Risk-Revisited.pdf), 2019
- AQR, "Demystifying Managed Futures", Journal of Investment Management
- AQR, ["A Century of Evidence on Trend-Following Investing"](https://www.trendfollowing.com/whitepaper/Century_Evidence_Trend_Following.pdf)
- Robert Carver, [qoppac.blogspot.com](https://qoppac.blogspot.com) — "How much risk should we take?" (2020), "Diversification and small account size" (2016)
- Robert Carver, [pysystemtrade](https://github.com/robcarver17/pysystemtrade) — `defaults.yaml` config: `vol_abs_min`, `forecast_cap`/`average_absolute_forecast`, disabled-by-default `risk_overlay` block (`max_risk_leverage`, `max_risk_fraction_normal_risk`, `max_risk_fraction_stdev_risk`, `max_risk_limit_sum_abs_risk`)
- AlphaArchitect, "Volatility Targeting Improves Risk-Adjusted Returns" (cap-at-2 example)

## Sources attempted but unverifiable

These were located and an attempt was made to extract their content; the attempt failed, so nothing from them is cited above as confirmed.

- Winton Capital Management, ["Systematic risk"](https://www.belmontinvestments.com/cimg/file/articles/39/pdf/160201Wintonsystematicrisk.pdf), working paper, Feb 2016 — **PDF will not extract.** Two independent attempts (one via the research agent's `pdftotext`, one via a direct `WebFetch` re-check while editing this file) both returned only binary/compressed-stream garbage, no readable text. An earlier draft of this report cited a verbatim quote from this paper ("daily changes... limited to 1% of the position") attributed to a different research-agent run that claimed to have read it successfully — that claim could not be reproduced and has been removed. Treat any future citation of this specific paper's content as unconfirmed until a clean copy is found.
- Man Group / Man AHL, "Explains: Volatility Scaling" — page fetched, but it contained only a video description with no transcript; no methodology text was retrieved.
- AQR, "Understanding Managed Futures" / Brian Hurst (2010) — text was successfully extracted, but turned out to be a general trend-following marketing/explainer piece (range-bound markets, drawdown framing) with no position-sizing formula or leverage-cap discussion — checked and ruled out as relevant, not a guess.
