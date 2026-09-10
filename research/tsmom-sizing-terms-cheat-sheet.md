# TSMOM Sizing Terms Cheat Sheet

This cheat sheet explains the live TSMOM sizing fields using the actual MES
row from the $90,000 Goulding/IDM/ERC report generated on 2026-09-10.

The central distinction is:

```text
portfolio risk budget
  -> notional budget
  -> signal/volatility-scaled target notional
  -> fractional contracts
  -> whole contracts
  -> realized standalone and portfolio risk
```

## MES inputs from the report

| Field | Value | Meaning |
|---|---:|---|
| `acct_equity` | `$90,000` | Account equity used for portfolio sizing. |
| `targ_port_vol` | `0.15` | Requested annualized portfolio volatility, or 15%. |
| `idm_mult` | `2.0386` | IDM diversification multiplier applied to the total IDM budget. |
| `not_weight` | `0.0684` | MES's ERC share of the IDM-adjusted dollar-vol budget. |
| `g_sig` / `signal` | `+1.0000` | Goulding direction; Bull means long. This is not the final position size. |
| `risk_scalar` | `1.3035` | MES volatility-scaling component before the final scalar clamp. |
| `reg_discount` | `1.0000` | Goulding Bull state receives no correction/rebound discount. |
| `vix_scalar` | `1.0000` | Portfolio-wide VX/VIX adjustment; normal regime in this run. |
| `combined_scalar` / `scalar` | `1.0000` | Final sizing scalar after multiplying components and clamping to `[-1, +1]`. |
| `close` | `$7,608.25` | MES futures price used for the contract-notional calculation. |
| `mult` | `5` | MES contract multiplier: `$5` per index point. |
| `hv` | `0.1151` | MES annualized realized volatility used for dollar-vol conversion. |

## Budget and notional terms

| Report field | Better interpretation | Calculation / formula | MES value |
|---|---|---|---:|
| `portfolio_risk_target` | Base dollar-vol target for the entire account | `acct_equity × targ_port_vol` | `$90,000 × 0.15 = $13,500` |
| `idm_risk_target` | IDM-adjusted total dollar-vol budget before ERC splitting | `portfolio_risk_target × idm_mult` | `$13,500 × 2.0386 ≈ $27,521` |
| `budg_const` | MES's notional budget before applying the signal/volatility scalar | `(idm_risk_target × not_weight) ÷ vol_target` | `($27,521 × 0.0684) ÷ 0.15 ≈ $12,548` |
| `symbol_risk_budget` | MES's pre-signal dollar-vol budget | `budg_const × vol_target` | `$12,548.06 × 0.15 ≈ $1,882` |
| `raw_not` | Uncapped desired dollar exposure after applying the final scalar | `budg_const × scalar` | `$12,548.06 × 1.00 = $12,548` |
| `targ_not` | Desired dollar exposure after any optional `max_notional` ceiling | `clamp(raw_not, -max_notional, +max_notional)` | `$12,548` because no ceiling reduced it |

Important: `raw_not` and `targ_not` are desired portfolio exposure amounts.
They are not the cash notional of one MES contract, and they are not
unscaled values. They already incorporate the final signal and volatility
scalar through `scalar`.

## Contract and risk terms

| Report field | Better interpretation | Calculation / formula | MES value |
|---|---|---|---:|
| `contract_notional` | Cash notional of one MES contract at the current price | `close × mult` | `$7,608.25 × 5 = $38,041.25` |
| `one_contract_risk` | Standalone annualized dollar-vol risk of one MES contract | `contract_notional × hv` | `$38,041.25 × 0.1151 ≈ $4,379` |
| `target_risk` | Dollar-vol risk of the fractional target before integer rounding | `abs(targ_not) × hv` | `$12,548.06 × 0.1151 ≈ $1,444` |
| `contin_con` | Fractional desired contract count | `targ_not ÷ contract_notional` | `$12,548.06 ÷ $38,041.25 = 0.3299` |
| `target_con` | Final whole-contract position target | `round(contin_con)` in legacy independent mode | `round(0.3299) = 0` |
| `pos_risk` | Standalone dollar-vol risk of the final integer position | `abs(target_con) × one_contract_risk` | `0 × $4,379 = $0` |
| `risk_contrib` | MES's correlation-aware Euler contribution to realized portfolio risk | Derived from signed exposure vector `x` and correlation matrix `H` | `$0` because MES has zero contracts |
| `rounding_gap` | Distance between fractional and final contract targets | `abs(contin_con - target_con)` | `abs(0.3299 - 0) = 0.3299 contracts` |

## Why MES becomes zero

MES's desired exposure is `$12,548`, but one MES contract represents
approximately `$38,041` of futures notional:

```text
fractional target = $12,548 / $38,041 = 0.3299 contracts
final target      = 0 contracts
```

The one-contract risk hurdle is approximately `$4,379`, while the fractional
target only requests approximately `$1,444` of MES dollar-vol risk. The
legacy independent allocator cannot trade `0.3299` contracts, so it rounds to
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

## Recommended terminology

The current abbreviated names are retained here to match the existing report,
but these names would be clearer in a future schema cleanup:

| Current name | Proposed name |
|---|---|
| `budg_const` | `pre_scalar_notional_budget` |
| `raw_not` | `uncapped_target_notional` |
| `targ_not` | `target_notional` |
| `contract_notional` | `one_contract_notional` |
| `one_contract_risk` | `one_contract_dollar_vol` |
| `target_risk` | `fractional_target_dollar_vol` |
| `contin_con` | `fractional_target_contracts` |
| `target_con` | `final_target_contracts` |
| `pos_risk` | `standalone_position_dollar_vol` |
| `risk_contrib` | `portfolio_risk_contribution` |
