# TSMOM Sizing Terms Cheat Sheet

This cheat sheet explains the live TSMOM sizing fields using the historical
MES row from the $90,000 Goulding/IDM/ERC report generated on 2026-09-10,
plus the `$1,842` post-fix live example discussed below. The historical table
values remain useful for tracing the original rounding issue; current values
change with prices, volatility, and the corrected sizing function.

The central distinction is:

```text
portfolio risk budget
  -> notional budget
  -> signal/volatility-scaled target notional
  -> fractional contracts
  -> whole contracts
  -> realized standalone and portfolio risk
```

## Volatility-scaling correction (2026-09-10)

Goulding supplies a direction only: `+1` or `-1`.  The desired vol-parity
size is therefore the direction times the instrument's volatility scale:

```text
risk_scalar = clamp(vol_target / hv, 0.25, 2.0)
combined_scalar = direction * risk_scalar * intentional discounts
```

The `0.25`--`2.0` range is the low-/high-volatility leverage guardrail.  It
must be the only volatility-scaling clamp: a second final clamp of
`combined_scalar` to `[-1, +1]` would prevent all leverage-up for a
full-strength Goulding signal.  For example, the historical MES row below
had `risk_scalar=1.3035` but was limited to `combined_scalar=1.0`; the
corrected function permits `combined_scalar=1.3035`.

The report computes the two dollar-vol fields by different direct formulas:

```text
pre_scalar_dollar_vol_budget = abs(pre_scalar_notional_budget × vol_target)

fractional_target_dollar_vol = abs(fractional_target_notional × hv)
```

That is why a `$12.5K` `frac_tgt_not` value is not `$12.5K` of risk: it is
cash notional. With the run's annualized `hv` of roughly `14.74%`, for example:

```text
$12,500 fractional target notional × 14.74% hv ≈ $1,842 fractional target dollar-vol
```

The two dollar-vol figures are related, but they are not aliases:

```text
fractional_target_dollar_vol / pre_scalar_dollar_vol_budget
= abs(combined_scalar) × hv / vol_target
```

They match only in the pure full-strength vol-parity case, where the combined
scalar is exactly `vol_target / hv` (and no VIX, confidence, regime, or hard
notional overlay changes it):

```text
fractional_target_notional = direction × pre_scalar_dollar_vol_budget / hv

abs(fractional_target_notional) × hv = pre_scalar_dollar_vol_budget
```

IDM/ERC can intentionally give different `pre_scalar_dollar_vol_budget`
values to different symbols.  That is a portfolio-allocation choice; the
`risk_scalar` still equalizes each symbol to its own assigned dollar-vol
budget, subject to the explicit 0.25--2.0 guardrail and downstream contract,
notional, and portfolio-risk limits.

## MES inputs from the report

| Field | Value | Meaning |
|---|---:|---|
| `acct_equity` | `$90,000` | Account equity used for portfolio sizing. |
| `target_portfolio_vol` | `0.15` | Requested annualized portfolio volatility, or 15%. |
| `idm_multiplier` | `2.0386` | IDM diversification multiplier applied to the total IDM budget. |
| `notional_allocation_weight` | `0.0684` | MES's ERC share of the IDM-adjusted dollar-vol budget. |
| `g_sig` / `signal` | `+1.0000` | Goulding direction; Bull means long. This is not the final position size. |
| `risk_scalar` | `1.3035` | MES volatility scale: `clamp(vol_target / hv, 0.25, 2.0)`. |
| `reg_discount` | `1.0000` | Goulding Bull state receives no correction/rebound discount. |
| `vix_scalar` | `1.0000` | Portfolio-wide VX/VIX adjustment; normal regime in this run. |
| `combined_scalar` | `1.0000` | Historical pre-fix scalar. The final `[-1, +1]` clamp incorrectly discarded MES's permitted 1.3035× low-vol scale; after the correction it would be `1.3035`. |
| `close` | `$7,608.25` | MES futures price used for the contract-notional calculation. |
| `mult` | `5` | MES contract multiplier: `$5` per index point. |
| `hv` | `0.1151` | MES annualized realized volatility used for dollar-vol conversion. |

## Budget and notional terms

| Report field | Better interpretation | Calculation / formula | MES value |
|---|---|---|---:|
| `portfolio_risk_target` | Base dollar-vol target for the entire account | `account_equity × target_portfolio_vol` | `$90,000 × 0.15 = $13,500` |
| `idm_risk_target` | IDM-adjusted total dollar-vol budget before ERC splitting | `portfolio_risk_target × idm_multiplier` | `$13,500 × 2.0386 ≈ $27,521` |
| `pre_scalar_notional_budget` | MES's notional budget before applying the signal/volatility scalar | `(idm_risk_target × notional_allocation_weight) ÷ vol_target` | `($27,521 × 0.0684) ÷ 0.15 ≈ $12,548` |
| `pre_scalar_dollar_vol_budget` | MES's dollar-vol budget before the signal/volatility scalar | `pre_scalar_notional_budget × vol_target` | `$12,548.06 × 0.15 ≈ $1,882` |
| `uncapped_fractional_target_notional` | Uncapped continuous dollar exposure after applying the final scalar | `pre_scalar_notional_budget × combined_scalar` | `$12,548.06 × 1.00 = $12,548` |
| `fractional_target_notional` | Continuous dollar exposure after any optional `max_notional` ceiling | `clamp(uncapped_fractional_target_notional, -max_notional, +max_notional)` | `$12,548` because no ceiling reduced it |

Important: `uncapped_fractional_target_notional` and
`fractional_target_notional` are continuous desired portfolio exposure
amounts.
They are not the cash notional of one MES contract, and they are not
unscaled values. They already incorporate the final signal and volatility
scalar through `combined_scalar`.

## Contract and risk terms

| Report field | Better interpretation | Calculation / formula | MES value |
|---|---|---|---:|
| `one_contract_notional` | Cash notional of one MES contract at the current price | `close × mult` | `$7,608.25 × 5 = $38,041.25` |
| `one_contract_dollar_vol` | Standalone annualized dollar-vol risk of one MES contract | `one_contract_notional × hv` | `$38,041.25 × 0.1151 ≈ $4,379` |
| `fractional_target_dollar_vol` | Dollar-vol risk of the continuous target before integer rounding | `abs(fractional_target_notional) × hv` | `$1,842` when the continuous target notional is about `$12.5K` and `hv` is about `14.74%`. It equals `pre_scalar_dollar_vol_budget` only under pure full-strength vol parity. The original pre-fix snapshot's `$1,444` is retained only as a comparison. |
| `fractional_target_contracts` | Fractional desired contract count | `fractional_target_notional ÷ one_contract_notional` | `$12,548.06 ÷ $38,041.25 = 0.3299` |
| `final_target_contracts` | Final whole-contract position target | `round(fractional_target_contracts)` in independent mode | `round(0.3299) = 0` |
| `standalone_position_dollar_vol` | Standalone dollar-vol risk of the final integer position | `abs(final_target_contracts) × one_contract_dollar_vol` | `0 × $4,379 = $0` |
| `portfolio_risk_contribution` | MES's correlation-aware Euler contribution to realized portfolio risk | Derived from signed exposure vector `x` and correlation matrix `H` | `$0` because MES has zero contracts |
| `rounding_gap` | Distance between fractional and final contract targets | `abs(fractional_target_contracts - final_target_contracts)` | `abs(0.3299 - 0) = 0.3299 contracts` |

## Why MES becomes zero

MES's desired exposure is `$12,548`, but one MES contract represents
approximately `$38,041` of futures notional:

```text
fractional target = $12,548 / $38,041 = 0.3299 contracts
final target      = 0 contracts
```

The one-contract risk hurdle is approximately `$4,379`, while the continuous
target's dollar-vol is only `$1,842` in the current example. The legacy
independent allocator cannot trade a sub-half-contract target, so it rounds to
zero and reports `zero_reason=below_half_contract`.

This is separate from the account-level portfolio target of `$13,500`. The
portfolio target is a risk budget for the whole book; it does not mean every
instrument receives enough allocation to buy one contract. The IDM/ERC split
and each instrument's contract size determine whether an individual symbol
can clear its one-contract hurdle.

## One-line interpretation of the MES row

> Goulding wants to be long MES, and the volatility model assigns MES about
> `$12.5K` of desired futures exposure, but that is only `0.3299` MES
> contracts; one contract is about `$38.0K` notional and `$4.4K` annualized
> dollar-vol risk, so the trade rounds to zero.

## Retired report names

The live target schema and CSV now use the canonical names above. This table
maps historical reports to their replacements:

| Historical name | Canonical name |
|---|---|
| `budg_const` | `pre_scalar_notional_budget` |
| `raw_not` | `uncapped_fractional_target_notional` |
| `targ_not` | `fractional_target_notional` |
| `contract_notional` | `one_contract_notional` |
| `one_contract_risk` | `one_contract_dollar_vol` |
| `target_risk` | `fractional_target_dollar_vol` |
| `contin_con` | `fractional_target_contracts` |
| `target_con` | `final_target_contracts` |
| `pos_risk` | `standalone_position_dollar_vol` |
| `risk_contrib` | `portfolio_risk_contribution` |
| `acct_equity` | `account_equity` |
| `targ_port_vol` | `target_portfolio_vol` |
| `idm_mult` | `idm_multiplier` |
| `not_weight` | `notional_allocation_weight` |
