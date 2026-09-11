# Live TSMOM Rebalance: $90K Goulding/IDM/ERC Rounding Analysis

Research date: 2026-09-10.

This note analyzes the dry-run report produced with a $90,000 account and:

```text
--target-portfolio-vol 0.15
--signal-weighting goulding
--risk-budget-mode idm
--notional-weighting erc
--use-idm
--data-source database
```

The conclusion is that the MES/MNQ zero targets are expected whole-contract
granularity effects, not a Goulding-direction or contract-multiplier bug. The
run does expose a broader portfolio-sizing limitation: the continuous IDM/ERC
allocation does not guarantee that the rounded, tradeable portfolio will
realize the requested 15% volatility.

## Observed result

The cluster report shows:

```text
standalone position dollar-vol       $13,958  (15.5% of equity)
portfolio risk contribution          $ 9,402  (10.4% of equity)
```

`standalone_position_dollar_vol` is the sum of each final position's
standalone dollar-vol risk, with direction removed. It is not portfolio
volatility. `portfolio_risk_contribution` is the Euler decomposition of the
realized signed portfolio exposure through the correlation matrix. Its total,
$9,402, is the relevant realized portfolio-risk figure for this report.

Six clusters are active: equity, energy, grain, metal, FX, and rates. In IDM
mode, `n_effective_clusters=6` is informational; it is not used as a six-way equal budget
divisor. All twelve instruments have live directional signals, but six of them
round to zero contracts: MES, MNQ, MGC, SIL, J7, and 6M.

## IDM/ERC budget construction

The portfolio-level base risk target is:

```text
$90,000 × 0.15 = $13,500
```

The report's IDM multiplier is `2.0386`, so the correlation-aware dollar-vol
budget before per-symbol ERC splitting is approximately:

```text
$13,500 × 2.0386 = $27,521
```

For each active symbol:

```text
allocated dollar-vol budget = $27,521 × ERC weight
pre_scalar_notional_budget  = allocated dollar-vol budget / 0.15
fractional_target_notional  = pre_scalar_notional_budget × combined_scalar
```

The ERC weights sum to one. Consequently, `pre_scalar_notional_budget` is a
pre-scalar notional budget; it is not an actual position notional and is not
the shared `cluster_dollar_vol_budget` used by cluster mode. The blank
`cluster_dollar_vol_budget` column in this IDM run is therefore expected,
although the report would be clearer if it exposed the per-symbol dollar-vol
budget explicitly.

The implementation performs this per-symbol budget construction in
[`compute_symbol_notional_budget`](../src/derivatives_bt_engine/domain/allocation.py#L868),
and the live path selects it under `risk_budget_mode='idm'` in
[`compute_rebalance_targets`](../src/derivatives_bt_engine/live/tsmom_rebalance.py#L1374).

## MES and MNQ sizing

The two equity rows are:

| Symbol | `pre_scalar_notional_budget` | `combined_scalar` | `fractional_target_notional` | One-contract notional | Fractional contracts | Final contracts |
|---|---:|---:|---:|---:|---:|---:|
| MES | $12,548 | 1.0000 | $12,548 | $38,041 | 0.3299 | 0 |
| MNQ | $12,287 | 0.7644 | $9,392 | $58,456 | 0.1607 | 0 |

The conversion is:

```text
fractional_target_contracts = fractional_target_notional / (close × multiplier)
```

The final target must be an integer. The live sizing pass rounds the
continuous value to a whole contract even when the cluster cap itself is
disabled. Therefore:

```text
MES: 0.3299 -> 0
MNQ: 0.1607 -> 0
```

One MES would require at least approximately `0.5 × $38,041 = $19,021` of
target notional to clear the rounding threshold. One MNQ would require about
`0.5 × $58,456 = $29,228`.

This is not caused by `max_contracts`, `max_cluster_risk_pct`, or the Goulding
signal. `apply_cluster_cap` was omitted and defaults to false; disabling that
cap disables cluster redistribution, but it does not permit fractional futures
contracts. The unconditional rounding and `standalone_position_dollar_vol`
calculation are in
[`apply_cluster_risk_cap`](../src/derivatives_bt_engine/domain/allocation.py#L192).

## What Goulding mode means here

This implementation uses Goulding as a direction model and uses the existing
volatility-parity machinery for size:

- Bull resolves to `g_sig=+1`.
- Bear resolves to `g_sig=-1`.
- Correction/Rebound uses the sign of the `a_Co/a_Re` blend of Goulding's
  fast and slow monthly signals.
- Volatility scaling, optional signal confidence, and the portfolio-wide VIX
  scalar determine `combined_scalar`; the configured continuous-mode
  `regime_discount` is ignored in Goulding mode because the Goulding state
  and blend provide that directional mechanism.

Thus `g_sig` being exactly `+1` or `-1` is expected. In Bull/Bear rows,
`g_blend` is blank because the Goulding blend is only used in
Correction/Rebound. The `a_co/a_re` values at or near 0/1 are pooled,
clamped mixing estimates; they do not affect a Bull/Bear row.

Historically, the final sizing scalar was also capped at `[-1, 1]`. In
Goulding mode the direction is already ±1, so that erased allowed low-vol
leverage even when the explicit `risk_scalar` 2.0 ceiling had not bound; MES
was the example. This was corrected on 2026-09-10: direction is bounded
separately, while `risk_scalar`'s `[0.25, 2.0]` range remains the sole
volatility-leverage guardrail. The `scalar_capped` diagnostic now identifies
that real risk-scalar bound instead of the removed final cap.

The code's own signal documentation describes this separation as “Goulding
decides direction, vol-parity decides size”; it is not a literal reproduction
of a continuous Goulding portfolio weight.

## Bug or expected behavior?

### Expected

- MES/MNQ are active signals but have no affordable whole-contract target.
- `cluster_dollar_vol_budget` is blank in IDM mode because the budget is
  stored as a per-symbol `pre_scalar_notional_budget` instead of one shared
  cluster budget.
- `g_blend` is blank in Bull/Bear states.
- `current_contracts` is blank under `data_source='database'`; no IB account
  positions were queried. It does not mean the real account position is zero.
- The cluster cap being disabled does not disable integer contract rounding.

### Design limitation

The continuous allocation has been sized for a hypothetical fractional
portfolio. Integer contracts create a large downward bias at this account
size. The realized 10.4% risk contribution versus the 15% target is therefore
plausible and expected under the current pipeline, but it is not target-vol
tracking in the strict realized sense.

For context, one contract has approximately:

```text
MES  $4,379 standalone dollar-vol risk  (~4.9% of equity)
MNQ $11,469 standalone dollar-vol risk (~12.7% of equity)
```

Forcing one MNQ would be a large discrete jump relative to its assigned ERC
budget. If an equity-cluster representative is required at $90K, MES is the
more gradual choice.

## Recommended improvements

1. Add explicit report fields:

   - `one_contract_notional`
   - `one_contract_dollar_vol`
   - `pre_scalar_dollar_vol_budget = pre_scalar_notional_budget × vol_target`
   - unsigned `fractional_target_dollar_vol = abs(fractional_target_notional) × hv`
   - `rounding_gap`
   - `scalar_capped`
   - `zero_reason` such as `below_half_contract`

2. Add portfolio-level fields for IDM mode:

   - base dollar-vol target: `$13,500`
   - IDM-adjusted dollar-vol target: `$27,521`
   - realized post-rounding dollar-vol: `$9,402`

3. Add lot-aware allocation after the continuous sizing step. A simple
   active-set redistribution can transfer unused allocations from symbols that
   round to zero to survivors, but equal redistribution alone will often keep
   expensive equity, FX, or metal contracts at zero. A better live method is
   an integer portfolio allocator that considers contract risk, correlation,
   cluster representation, and the account-level risk limit together.

4. Reuse the backtest's active-set redistribution and cluster-representation
   ideas as candidate live policies, but adapt them to IDM/ERC budgets rather
   than blindly applying flat equal weights. Any such policy should be
   backtested because it changes breadth, turnover, and realized risk.

5. Clarify the naming of `signal_weighting='goulding'`. A name such as
   `direction_model='goulding'` would better communicate that the current
   implementation uses Goulding for direction and a separate vol-parity model
   for size.

6. The supplied `live.sh` shows a bare `--vol-fast-window` with no integer
   argument. The CLI requires a value for that option. It is ignored by
   Goulding mode, so it should either be removed or given an explicit value.

## Implementation status

Improvements 1–2 are now implemented in the live rebalance reporting path.
Each target row exposes the contract hurdle, symbol risk budget, continuous-
to-integer gap, scalar-cap flag, and zero-target reason. IDM runs also expose
the base portfolio risk target, IDM-adjusted target, and realized post-rounding
portfolio risk in both the CSV rows and the text report. These fields are
diagnostic only and do not change position sizing.

The original MES/MNQ result was internally consistent, but its small-account
rounding trade-off motivated the standard lot-aware policy below: preserve
the complete rounded signal book, then repair only an actual portfolio-risk
breach rather than manufacturing broader cluster representation.

## v1.0 implementation of improvement 3

The standard opt-in lot-aware policy is now implemented behind:

```text
--discrete-allocation lot-aware
--discrete-risk-overrun-pct 0.00
--min-fractional-contracts 0.50
```

Standard lot-aware is the default `--discrete-allocation` policy. The legacy
per-symbol rounding remains available as `--discrete-allocation independent`.
Standard lot-aware first applies the threshold and nearest-integer rounding to
the complete active target book. With the default `0.50` threshold this is
normal round-half-up sizing;
`0.25` is an explicit research choice that promotes a 0.25--0.49 target to
one lot. It then uses each symbol's one-contract dollar-vol risk and the IDM
correlation matrix when available (identity otherwise) only to remove lots
if the rounded book exceeds `account_equity × target_portfolio_vol` plus the
chosen overrun. It neither forces cluster representation nor spends unused
risk capacity, so a weak signal does not become a trade purely for
diversification. `--apply-cluster-cap` is deliberately incompatible with
this policy.

The earlier start-from-flat algorithm is retained for comparison under:

```text
--discrete-allocation flat-cluster-diversified
```

That experimental policy first selects the least-risk feasible representative
from every signaled cluster, then fits continuous targets and may spend spare
risk capacity. It is useful for researching diversification-first small
accounts, but it can promote a sub-half-contract signal and is not the
standard lot-aware behavior.

The account risk limit is hard by default and can be relaxed explicitly with
`--discrete-risk-overrun-pct`. A symbol that has a live signal but cannot fit
one lot after risk repair is retained at zero with
`zero_reason=integer_risk_limit`; a signal below the configured threshold is
`below_min_fractional_contracts`. The chosen policy, threshold, and integer
risk limit are included in the CSV and cluster report so a lot-aware run is
distinguishable from the legacy and experimental paths.
