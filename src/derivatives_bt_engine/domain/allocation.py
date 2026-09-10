"""
Risk allocation -- how much risk/budget each POSITION gets, both at the
single-instrument level (compute_position_scalar: this instrument's own
vol-targeted risk_scalar) and across instruments/clusters relative to each
other (everything else here), given they aren't independent bets.

compute_position_scalar moved here (2026-07) from domain/signal.py, which
had held it since before this module existed -- it's genuinely risk-SIZING
math (vol-targeting, momentum/signal-confidence discounts), not signal
CONSTRUCTION (trend detection), so it belongs in the risk module, not
alongside build_features/continuous_momentum/goulding_monthly. This was the
second half of the same split that originally created this module:
compute_n_effective/compute_desired_risk_budget/apply_cluster_risk_cap moved
here unchanged from the old tsmom_signal.py (see git history for that
mechanical move); _bounded_ewm_correlation_matrix/compute_idm moved here from
derivatives_bt_engine.strats.tsmom_binary_vol_parity_backtest, which
originally built them for its own idm_scaling feature -- this is now the one
canonical implementation, not a duplicate, so any other caller (the live
system's own compute_desired_risk_budget, tsmom_backtester.py) can reuse the
same correlation math rather than re-deriving it.

compute_n_effective/compute_desired_risk_budget assume active clusters are
UNCORRELATED (their 1/sqrt(n_effective) scaling is exactly compute_idm's own
formula at rho=0) -- compute_idm generalizes that to the REAL measured
correlation. Not yet wired together (compute_desired_risk_budget still uses
the zero-correlation assumption as of this move) -- see
research/cta-layer-separation-risk-budgeting.md and this project's own
tsmom_risk_budget_diagnostic.py for the ERC/HRP comparison this was
motivated by.
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import polars as pl
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from derivatives_bt_engine.domain.enums import TrendRegime
from derivatives_bt_engine.domain.signal import DEFAULT_ANNUALIZATION_DAYS

log = logging.getLogger(__name__)


def compute_position_scalar(trend_strength, daily_std_last, vol_target: float,
                             regime: TrendRegime, regime_discount: float = 0.5,
                             signal_confidence: float = 1.0,
                             annualization_days=DEFAULT_ANNUALIZATION_DAYS) -> float:
    """
    Layers 2-4 of the position sizing framework (plus the opt-in layer 5,
    signal_confidence), combined into a single scalar in [-1, +1]:

        scalar = trend_strength * risk_scalar * regime_discount * signal_confidence

    Long-only filtering (signal_scalar = max(0, trend_strength)) is the
    caller's responsibility — pass the already-filtered trend_strength in
    for long-only accounts. This function stays pure w.r.t. direction.

    risk_scalar = vol_target / current_realized_vol, clamped to [0.25, 2.0]
    -- a risk-equalization ratio driven by THIS instrument's own realized
    vol, nothing regime- or market-wide about it.
    current_realized_vol = daily_std_last * sqrt(annualization_days) -- the
    annualization is scaling daily_std_last up to an annual figure; the
    window daily_std_last happened to be estimated over (63-day rolling, in
    this project's callers) is irrelevant to that scaling itself.
    Callers should pass the SAME annualization_days used to compute
    daily_std_last in the first place (instruments.resolve_annualization_days)
    -- this project's confirmed universe splits 252 (CBOT grains) vs. 259
    (everything else checked); the plain 252 default here reproduces prior
    behavior exactly for anyone not passing an instrument-specific value.

    regime_discount is applied only for Correction/Rebound (disagreement
    between the fast and slow momentum signal — lower conviction);
    Bull/Bear/Unknown get a discount factor of 1.0. Despite the similar
    "discount" shape, this is unrelated to vix_scalar (the
    portfolio-wide, VX-driven de-risking lever applied by the caller in
    derivatives_bt_engine.live.tsmom_rebalance, not in here).

    signal_confidence (default 1.0, no-op) is a separate, opt-in, per-
    instrument discount on trust in THIS instrument's signal when its own
    vol_ratio (short-window/long-window realized vol, asset-specific, NOT
    VIX/VX-driven) is unusual relative to its own history -- see
    compute_signal_confidence() in signal.py. Orthogonal to regime_discount
    (which is about fast/slow sign disagreement, not vol) and to
    vix_scalar (which is portfolio-wide, not per-instrument).
    """
    if trend_strength is None or (isinstance(trend_strength, float) and math.isnan(trend_strength)):
        return 0.0

    if daily_std_last is None or (isinstance(daily_std_last, float) and math.isnan(daily_std_last)) or daily_std_last <= 0:
        risk_scalar = 1.0   # insufficient history to size by vol — neutral
    else:
        current_realized_vol = daily_std_last * math.sqrt(annualization_days) # annualized realized vol, estimated from daily_std_last's own trailing window
        risk_scalar = vol_target / current_realized_vol # 0.15/0.60 ~= 0.25, this is not hv_long here, but a param
        risk_scalar = max(0.25, min(2.0, risk_scalar))

    regime_discount = regime_discount if regime in (TrendRegime.CORRECTION, TrendRegime.REBOUND) else 1.0

    scalar = trend_strength * risk_scalar * regime_discount * signal_confidence
    return max(-1.0, min(1.0, scalar))


def compute_n_effective(active_clusters: set) -> int:
    """Count of clusters carrying a live signal -- the portfolio's
    effective number of independent bets, not its raw instrument count
    (e.g. 4 grain micros moving on one shared ag-complex factor are one
    bet, not four)."""
    return len(active_clusters)


def compute_desired_risk_budget(account_equity: float, target_portfolio_vol: float,
                                 n_effective: int) -> float:
    """Per-cluster dollar risk budget, sqrt(N)-scaled so total portfolio
    risk (assuming roughly independent clusters) targets
    account_equity * target_portfolio_vol regardless of how many clusters
    are currently active."""
    if n_effective <= 0:
        return 0.0
    return account_equity * target_portfolio_vol / math.sqrt(n_effective)


def apply_cluster_risk_cap(targets: list[dict], max_cluster_risk_pct: float,
                            total_risk_target: Optional[float], n_active_clusters: int,
                            max_lot_overrun_pct: float = 0.5, apply_cap: bool = True) -> list[dict]:
    """
    Second pass over an already-sized targets list: for any cluster whose
    aggregate dollar-vol risk exceeds its share of a fixed, account-equity-
    derived risk target, allocates that cluster's capped budget by
    conviction priority rather than a uniform haircut, so e.g. 4 grain
    micros that are each individually sized correctly don't collectively
    become one oversized bet on the ag-complex factor they all share.

    The cap is taken against total_risk_target (account_equity *
    target_portfolio_vol, computed by the caller) -- a fixed number, NOT
    the emergent sum of this run's pre-cap position risks. effective_cap_pct
    = max(max_cluster_risk_pct, 1/n_active_clusters) -- the 1/n floor means
    a single active cluster isn't capped below 100% of the target just for
    being the only trade in the book.

    Allocation within an over-budget cluster (replaces a uniform scale
    factor, which produces an all-or-nothing failure mode: scaling every
    instrument down by the same factor can push ALL of them below the 0.5
    rounding threshold at once, even when the top-conviction instrument
    alone would easily survive on the full cap):

      1. Sort the cluster's instruments by priority = abs(combined_scalar),
         descending -- combined_scalar (not raw signal) since it already folds in
         vol-targeting, regime discount, and the long-only filter.
      2. Walk the sorted list with remaining_budget starting at the cap.
         For each instrument: affordable_continuous = remaining_budget /
         single_contract_risk (0 if remaining_budget <= 0); usable =
         min(abs(original_continuous), affordable_continuous); contracts =
         round(usable) if usable >= 0.5 else 0.
      3. Lot-size exception, first instrument only (remaining_budget still
         == cap, i.e. nothing spent yet): if usable < 0.5 (would round to
         zero) but the instrument's own signal genuinely wants a full
         contract (abs(original_continuous) >= 0.5) and its single-
         contract risk isn't wildly over the cap (<= cap * (1 +
         max_lot_overrun_pct)), it still gets exactly 1 contract -- a
         clearly-wanted top-conviction position shouldn't be sacrificed
         purely to discrete contract granularity. remaining_budget can go
         negative after this fires, which is intentional: it guarantees
         every later instrument in the walk gets 0.
      4. remaining_budget -= contracts * single_contract_risk; continue
         down the sorted list.

    Clusters within budget (risk <= cap) are untouched. Mixed-sign
    clusters work without special-casing -- risk is abs()-based, priority
    is abs(combined_scalar), direction is sign(original_continuous).

    infeasible is an OUTCOME-based flag, computed after every cluster's
    walk-down (or no-op) is final: True for every instrument in a cluster
    where ALL instruments end up at final_target_contracts == 0 despite at
    least one having abs(fractional_target_contracts) >= 0.5 pre-cap -- i.e. a real
    signal existed but the cluster genuinely captured no exposure. This is
    NOT decided by a cap-vs-single-contract-risk precomputation (the top-
    priority instrument may still land a contract via the lot exception
    even when that precomputed check would have said "infeasible") --
    only the actual result matters.

    Each target dict must carry 'cluster', 'fractional_target_contracts',
    'combined_scalar',
    'close', 'mult', 'hv' (already computed per-instrument by the
    caller). Targets with an 'error' key, or missing one of those fields,
    are left untouched and excluded from the risk totals. Mutates and
    returns the same list (adds/overwrites 'final_target_contracts' and
    'standalone_position_dollar_vol', and 'infeasible' where applicable).

    `apply_cap` (default True, unchanged prior behavior): whole-contract
    ROUNDING and 'standalone_position_dollar_vol' always happen regardless -- those aren't
    part of "the cap," they're just how a continuous target becomes a
    tradeable integer. False (or total_risk_target being None/<=0) skips
    only the cluster-level cap/redistribution walk-down itself, treating
    every cluster as within budget (the same "round the continuous value
    directly" path already used for any cluster that doesn't exceed the
    cap) -- e.g. for risk_budget_mode='idm', where compute_symbol_
    notional_budget's own IDM/ERC/HRP sizing is already correlation-aware,
    and re-capping against a hand-assigned-cluster, zero-correlation-
    assumption total_risk_target on top of that would re-impose exactly
    the assumption IDM was used to move past (confirmed directly: it can
    silently claw back the majority of IDM's own diversification credit,
    hitting hardest whichever position is the real hedge -- see the
    caller's own docstring)."""
    valid = [
        t for t in targets
        if not t.get('error')
        and t.get('fractional_target_contracts') is not None
        and t.get('cluster') is not None
        and t.get('close') is not None
        and t.get('mult') is not None
        and t.get('hv') is not None
    ]

    cluster_risk: dict[str, float] = {}
    for t in valid:
        position_risk = abs(t['fractional_target_contracts']) * t['close'] * t['mult'] * t['hv']
        cluster_risk[t['cluster']] = cluster_risk.get(t['cluster'], 0.0) + position_risk

    if apply_cap and total_risk_target is not None and total_risk_target > 0:
        effective_cap_pct = max(max_cluster_risk_pct, 1.0 / n_active_clusters) if n_active_clusters > 0 else max_cluster_risk_pct
        cap = effective_cap_pct * total_risk_target
    else:
        cap = None

    over_budget_clusters = {cluster for cluster, risk in cluster_risk.items() if risk > cap} if cap is not None else set()

    for cluster in over_budget_clusters:
        # over_budget_clusters is only ever non-empty when cap is not None
        # (see its own construction above), so cap is guaranteed a real
        # float whenever this loop body runs -- not something a type
        # checker infers across the two separate expressions on its own.
        assert cap is not None
        members = [t for t in valid if t['cluster'] == cluster]
        # Priority order is fixed before any mutation -- read each
        # instrument's original, unscaled continuous_contracts once, up
        # front, so later iterations of this loop can't see a value
        # another instrument's walk step already changed.
        members_sorted = sorted(members, key=lambda t: abs(t.get('combined_scalar') or 0.0), reverse=True)
        original_continuous = {t['symbol']: t['fractional_target_contracts'] for t in members_sorted}

        remaining_budget = cap
        for i, t in enumerate(members_sorted):
            is_first = (i == 0)
            orig = original_continuous[t['symbol']]
            single_contract_risk = t['close'] * t['mult'] * t['hv']

            if remaining_budget <= 0:
                contracts = 0
            else:
                affordable_continuous = remaining_budget / single_contract_risk if single_contract_risk else 0.0
                usable_continuous = min(abs(orig), affordable_continuous)

                if (is_first and usable_continuous < 0.5 and abs(orig) >= 0.5
                        and single_contract_risk <= cap * (1 + max_lot_overrun_pct)):
                    contracts = 1
                else:
                    # math.floor(x + 0.5), not round(): round() ties to even
                    # (round(2.5) == 2), which would silently under-allocate
                    # a genuine 0.5-contract signal half the time.
                    contracts = math.floor(usable_continuous + 0.5) if usable_continuous >= 0.5 else 0

            sign = 1 if orig > 0 else (-1 if orig < 0 else 0)
            t['final_target_contracts'] = sign * contracts
            remaining_budget -= contracts * single_contract_risk

    # Clusters within budget (and any target not part of an over-budget
    # cluster) round their continuous value directly, once -- equivalent
    # to a no-op scale of 1.0 through the same single-rounding rule the
    # walk-down above uses.
    for t in valid:
        if t['cluster'] in over_budget_clusters:
            continue
        scaled = t['fractional_target_contracts']
        sign = 1 if scaled > 0 else (-1 if scaled < 0 else 0)
        magnitude = 0 if abs(scaled) < 0.5 else math.floor(abs(scaled) + 0.5)
        t['final_target_contracts'] = sign * magnitude

    # max_contracts clamp is the true last step, after sizing is otherwise
    # final, then standalone_position_dollar_vol is recomputed from that final value.
    for t in valid:
        max_contracts = t.get('max_contracts')
        if max_contracts is not None:
            t['final_target_contracts'] = max(
                -max_contracts,
                min(max_contracts, t['final_target_contracts']),
            )
        t['standalone_position_dollar_vol'] = (
            abs(t['final_target_contracts']) * t['close'] * t['mult'] * t['hv']
        )

    # Outcome-based infeasibility: every instrument in the cluster ended up
    # at zero despite at least one having a genuine (>=0.5) pre-cap signal.
    for cluster in over_budget_clusters:
        members = [t for t in valid if t['cluster'] == cluster]
        had_real_signal = any(abs(t['fractional_target_contracts']) >= 0.5 for t in members)
        all_zero = all(t['final_target_contracts'] == 0 for t in members)
        if had_real_signal and all_zero:
            log.warning(
                "%s cluster captured zero exposure despite a live signal -- cap ($%.0f) "
                "could not accommodate any instrument in this cluster even with conviction-"
                "priority allocation; consider raising max_cluster_risk_pct, reducing "
                "n_effective's denominator effect, or excluding large-multiplier "
                "instruments from this account",
                cluster, cap,
            )
            for t in members:
                t['infeasible'] = True

    return targets


def allocate_lot_aware_targets(
    targets: list[dict],
    portfolio_risk_target: Optional[float],
    *,
    risk_overrun_pct: float = 0.0,
    H: Optional[np.ndarray] = None,
    active_symbols: Optional[list[str]] = None,
    max_cluster_risk_pct: float = 0.25,
    total_risk_target: Optional[float] = None,
    n_active_clusters: int = 0,
    apply_cluster_cap: bool = False,
) -> list[dict]:
    """Allocate whole contracts against the portfolio risk budget.

    This is the opt-in, deterministic discrete allocator used by live
    rebalancing.  It starts from zero and greedily adds one contract at a
    time, subject to the correlation-aware portfolio-risk limit and the
    optional standalone cluster cap.  The order of decisions is deliberate:

    * give each signaled cluster its least-risk representative when feasible;
    * add contracts that reduce the continuous dollar-risk/exposure gap;
    * use spare risk capacity to move the realized portfolio risk toward the
      requested portfolio target, allowing at most one contract beyond each
      continuous target.

    The existing ``apply_cluster_risk_cap`` remains the default path.  This
    function is intentionally a separate policy so that changing integer
    sizing is explicit and independently auditable.  Correlation is used for
    every feasibility check when ``H`` is supplied; an identity matrix is
    used when no correlation estimate is available.

    Targets with errors or incomplete sizing inputs are left untouched.  The
    remaining valid targets are mutated with ``final_target_contracts``,
    ``standalone_position_dollar_vol``,
    and, where a live signal cannot fit even as one lot, an
    ``integer_zero_reason`` diagnostic.
    """
    if risk_overrun_pct < 0:
        raise ValueError('risk_overrun_pct must be non-negative')

    valid = [
        t for t in targets
        if not t.get('error')
        and t.get('symbol') is not None
        and t.get('fractional_target_contracts') is not None
        and t.get('cluster') is not None
        and t.get('close') is not None
        and t.get('mult') is not None
        and t.get('hv') is not None
    ]
    eligible = [
        t for t in valid
        if t.get('active', True) is not False
        and abs(float(t['fractional_target_contracts'])) > 1e-12
    ]
    if not valid:
        return targets

    # One-contract dollar-vol risk is the common unit for both the
    # correlation-aware portfolio check and the cluster standalone check.
    one_contract_dollar_vol = {
        t['symbol']: abs(float(t['close']) * float(t['mult']) * float(t['hv']))
        for t in eligible
    }
    desired_risk = {
        t['symbol']: (
            abs(float(t['fractional_target_contracts'])) * one_contract_dollar_vol[t['symbol']]
        )
        for t in eligible
    }
    max_contracts = {
        t['symbol']: max(0, int(t['max_contracts'])) if t.get('max_contracts') is not None else 10**9
        for t in eligible
    }
    continuous_contracts = {
        t['symbol']: abs(float(t['fractional_target_contracts'])) for t in eligible
    }
    direction = {
        t['symbol']: 1 if float(t['fractional_target_contracts']) > 0 else -1
        for t in eligible
    }
    clusters = {t['symbol']: t['cluster'] for t in eligible}
    symbols = [t['symbol'] for t in eligible]

    # Keep the correlation matrix's symbol ordering explicit.  A malformed or
    # unavailable estimate falls back to independent risk rather than making
    # the rebalance fail at the final integer-allocation step.
    risk_symbols = list(active_symbols or [])
    corr = None
    if H is not None and risk_symbols:
        candidate = np.asarray(H, dtype=float)
        if candidate.ndim == 2 and candidate.shape == (len(risk_symbols), len(risk_symbols)):
            corr = candidate
    if corr is None:
        risk_symbols = symbols
        corr = np.eye(len(risk_symbols), dtype=float)
    risk_index = {symbol: i for i, symbol in enumerate(risk_symbols)}

    portfolio_limit = None
    if portfolio_risk_target is not None and portfolio_risk_target > 0:
        portfolio_limit = float(portfolio_risk_target) * (1.0 + risk_overrun_pct)

    cluster_limit = None
    if apply_cluster_cap and total_risk_target is not None and total_risk_target > 0:
        effective_cap_pct = (
            max(max_cluster_risk_pct, 1.0 / n_active_clusters)
            if n_active_clusters > 0 else max_cluster_risk_pct
        )
        cluster_limit = effective_cap_pct * float(total_risk_target)

    q = {symbol: 0 for symbol in symbols}

    def portfolio_risk(contract_counts: dict[str, int]) -> float:
        exposure = np.zeros(len(risk_symbols), dtype=float)
        for symbol, count in contract_counts.items():
            index = risk_index.get(symbol)
            if index is not None:
                exposure[index] = count * one_contract_dollar_vol[symbol]
        variance = float(exposure @ corr @ exposure)
        return math.sqrt(max(variance, 0.0))

    def cluster_risks(contract_counts: dict[str, int]) -> dict[str, float]:
        result: dict[str, float] = {}
        for symbol, count in contract_counts.items():
            cluster = clusters[symbol]
            result[cluster] = (
                result.get(cluster, 0.0) + abs(count) * one_contract_dollar_vol[symbol]
            )
        return result

    def feasible(contract_counts: dict[str, int]) -> bool:
        if portfolio_limit is not None and portfolio_risk(contract_counts) > portfolio_limit + 1e-9:
            return False
        if cluster_limit is not None:
            if any(risk > cluster_limit + 1e-9 for risk in cluster_risks(contract_counts).values()):
                return False
        return True

    def distance_from_continuous(contract_counts: dict[str, int]) -> float:
        return sum(
            abs(
                abs(contract_counts[symbol]) * one_contract_dollar_vol[symbol]
                - desired_risk[symbol]
            )
            for symbol in symbols
        )

    def add_one(contract_counts: dict[str, int], symbol: str) -> dict[str, int]:
        candidate = contract_counts.copy()
        candidate[symbol] += direction[symbol]
        return candidate

    # First ensure that each live cluster has a representative where the
    # account-level risk and cluster cap permit one.  The least-risk contract
    # wins, making the choice deterministic for a given target table.
    while True:
        represented = {clusters[symbol] for symbol, count in q.items() if count != 0}
        candidates: list[tuple[float, float, str, dict[str, int]]] = []
        for symbol in symbols:
            if (clusters[symbol] in represented or abs(q[symbol]) >= 1
                    or max_contracts[symbol] < 1):
                continue
            candidate = add_one(q, symbol)
            if feasible(candidate):
                candidates.append((
                    portfolio_risk(candidate), one_contract_dollar_vol[symbol], symbol, candidate
                ))
        if not candidates:
            break
        _, _, _, q = min(candidates, key=lambda item: (item[0], item[1], item[2]))

    # Fit the continuous target without crossing the first integer above it.
    # This makes .33 contracts become zero and 1.17 contracts become one, but
    # lets a one-contract representative continue to two when the continuous
    # target actually calls for two.
    while True:
        current_distance = distance_from_continuous(q)
        candidates = []
        for symbol in symbols:
            desired_ceiling = min(max_contracts[symbol], math.ceil(continuous_contracts[symbol]))
            if desired_ceiling <= 0 or abs(q[symbol]) >= desired_ceiling:
                continue
            candidate = add_one(q, symbol)
            if not feasible(candidate):
                continue
            new_distance = distance_from_continuous(candidate)
            if new_distance < current_distance - 1e-9:
                candidates.append((new_distance, portfolio_risk(candidate), symbol, candidate))
        if not candidates:
            break
        _, _, _, q = min(candidates, key=lambda item: (item[0], item[1], item[2]))

    # Spend unused portfolio capacity only when it moves realized risk closer
    # to the requested target.  One extra lot beyond the continuous target is
    # the bounded utilization allowance; it prevents tiny continuous targets
    # from consuming the entire discrete budget while still avoiding runaway
    # integer over-allocation.
    if portfolio_risk_target is not None and portfolio_risk_target > 0:
        while True:
            current_risk = portfolio_risk(q)
            current_gap = abs(float(portfolio_risk_target) - current_risk)
            candidates = []
            for symbol in symbols:
                desired_ceiling = min(
                    max_contracts[symbol],
                    max(1, math.ceil(continuous_contracts[symbol]) + 1),
                )
                if desired_ceiling <= 0 or abs(q[symbol]) >= desired_ceiling:
                    continue
                candidate = add_one(q, symbol)
                if not feasible(candidate):
                    continue
                new_gap = abs(float(portfolio_risk_target) - portfolio_risk(candidate))
                if new_gap < current_gap - 1e-9:
                    candidates.append((new_gap, distance_from_continuous(candidate), symbol, candidate))
            if not candidates:
                break
            _, _, _, q = min(candidates, key=lambda item: (item[0], item[1], item[2]))

    # Finalize every valid row, including zeroed rows, from the same integer
    # map.  The reason is only set when a single lot itself is infeasible; a
    # feasible sub-half-contract signal remains diagnosable as the ordinary
    # below-half rounding case.
    for target in valid:
        symbol = target['symbol']
        count = q.get(symbol, 0)
        target['final_target_contracts'] = count
        target['standalone_position_dollar_vol'] = abs(count) * one_contract_dollar_vol.get(symbol, 0.0)

        if (symbol in q and count == 0
                and abs(float(target['fractional_target_contracts'])) > 1e-12):
            one_lot = add_one(q, symbol)
            if portfolio_limit is not None and portfolio_risk(one_lot) > portfolio_limit + 1e-9:
                target['integer_zero_reason'] = 'integer_risk_limit'
            elif cluster_limit is not None and any(
                    risk > cluster_limit + 1e-9
                    for risk in cluster_risks(one_lot).values()):
                target['integer_zero_reason'] = 'cluster_risk_limit'

    return targets


# Minimum rows in a bounded EWM correlation window before trusting the
# estimate at all -- see _bounded_ewm_correlation_matrix's own docstring.
MIN_IDM_WINDOW_ROWS = 63

# compute_symbol_notional_budget's notional_weighting choices -- see that
# function's own docstring for what each one does.
NOTIONAL_WEIGHTING_SCHEMES = ('flat', 'erc', 'hrp')

# compute_erc_weights' damped-Newton solve of Spinu's (2013) convex ERC
# reformulation -- see that function's own docstring for the formulation.
ERC_NEWTON_MAX_ITER = 50
# max|gradient| convergence threshold. Confirmed via a 2100-trial stress
# test (n in 2..40, both random well-conditioned correlation matrices and
# adversarial near-singular ones with a highly-correlated cluster, cond(H)
# up to ~4e7) that anything tighter than ~1e-7 is NOT reliably achievable:
# Newton's own quadratic convergence hits a floating-point noise floor
# (compounded rounding through np.linalg.solve on the Hessian) somewhere
# around 1e-8 to 1e-10 depending on conditioning, below which the gradient
# can neither shrink further nor find a backtracking step that improves
# the objective past that noise floor -- a numerical-precision limit, not
# a correctness issue (0/2100 failures at 1e-7 across that same stress
# test). Still far tighter than matters for portfolio weights (a 1e-7
# risk-contribution mismatch is immaterial against real dollar budgets).
ERC_NEWTON_TOL = 1e-7
# Secondary, looser threshold for the specific case where backtracking
# line search stalls (can't find ANY step that both stays positive and
# strictly decreases the objective) before ERC_NEWTON_TOL is reached --
# this happens on adversarial near-singular matrices where the objective
# itself is already at its floating-point-representable minimum. Accept
# the current point as converged if its gradient is at least this small
# (still tight enough to be practically meaningless for real portfolio
# weights); a genuine non-convergence with a materially large gradient
# still falls through to the caller's own flat-weights fallback.
ERC_NEWTON_STALL_TOL = 1e-4
ERC_NEWTON_MIN_STEP = 1e-12  # backtracking line-search floor before giving up on an iteration

# Ceiling on the total notional-budget share collectively given to symbols
# with NO correlation coverage (a live signal, but missing/insufficient
# return history -- see `covered` in _bounded_ewm_correlation_matrix's own
# docstring) under 'erc'/'hrp' notional_weighting: min(this, |uncovered| /
# |active_symbols|), so a single new instrument gets roughly its proportional
# headcount share (not artificially punished), but a wave of simultaneously-
# uncovered instruments can't collectively claim more than this fraction of
# the book regardless of how many there are.
UNCOVERED_BUDGET_CAP_FRACTION = 0.10


def build_returns_wide(price_data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per date (inner-joined across every symbol's own daily
    close -- only dates common to ALL symbols survive), one column per
    symbol, values = that symbol's own simple daily return
    (close.pct_change()). Pure polars -- no pandas, per this project's own
    CLAUDE.md convention (pandas stays scoped to a single library call
    site, e.g. HRPOpt, never leaks into general data-handling code).
    Shared by every caller of _bounded_ewm_correlation_matrix (both TSMOM
    backtesters) -- build ONCE per backtest run and reuse at every
    rebalance date's own bounded-window slice; recomputing the raw return
    series per rebalance would be pure waste, only the EWM calc itself
    genuinely needs to run once per (rebalance date, bounded window)
    pair."""
    wide = None
    for sym, df in price_data.items():
        s = df.sort('ts_event').select('ts_event', pl.col('close').pct_change().alias(sym))
        wide = s if wide is None else wide.join(s, on='ts_event', how='inner')
    return wide.sort('ts_event').drop_nulls()


def _bounded_ewm_correlation_matrix(returns_wide: pl.DataFrame, symbols: list[str], as_of: date,
                                     window_years: float, halflife: float,
                                     min_rows: int = MIN_IDM_WINDOW_ROWS
                                     ) -> tuple[np.ndarray, np.ndarray]:
    """EWM-weighted correlation among `symbols`, computed ONLY from the
    trailing `window_years` slice of returns_wide ending STRICTLY before
    `as_of` (no lookahead) -- a genuinely BOUNDED window, not an unbounded
    full-history EWM. This distinction matters: a plain `.ewm_mean(half_life=hl)`
    applied to the ENTIRE historical series never fully zeroes out old data
    -- it decays toward negligible weight but asymptotically, so a few
    percent of a 2026 correlation estimate could technically still trace
    back to 2010 even at a short halflife. Slicing to a bounded window
    FIRST, then computing the EWM only within that slice, guarantees
    exactly zero weight on anything older than `window_years` -- the EWM
    only supplies the within-window recency emphasis (Carver's "regime"
    weighting), not the outer bound on history.

    Built as a single joint Gram-matrix decomposition, not n*(n+1)/2
    separately-estimated pairwise correlations. Polars' own
    `.ewm_mean(half_life=..., adjust=True)` (the default `adjust`) is, at
    any given row, exactly a normalized static weight vector over the
    rows up to and including it: weight of row t is (1-alpha)^(age of t),
    alpha = 1 - 2**(-1/halflife), normalized to sum to 1 -- so evaluating
    "at the last row of the bounded slice" is equivalent to a single
    static weight vector `w` anchored at that last row. That lets the
    whole n x n covariance matrix be built in one shot: `X` = the bounded
    slice as a plain (T x n) array, `mu = w @ X` the weighted mean,
    `Z = sqrt(w)[:, None] * (X - mu)`, `cov = Z.T @ Z`. This is
    numerically identical (confirmed to ~1e-16, well within float noise)
    to computing each pair's ewm_cov(x, y) = ewm_mean(x*y) - ewm_mean(x) *
    ewm_mean(y) separately, but ~9x faster at n=12 (one BLAS matmul over
    the whole universe instead of O(n^2) individual polars EWM
    evaluations) and PSD *by construction* -- `cov` is a Gram matrix
    (`Z.T @ Z`), so no post-hoc eigenvalue check is needed the way a set
    of independently-estimated pairwise correlations would require, and
    each diagonal entry (a sum of squares) can't land below zero the way
    an independently-estimated ewm_var(x) occasionally does on
    floating-point noise.

    `returns_wide`: one row per date, one column per symbol, simple daily
    returns (a caller-built, synchronized wide frame -- e.g.
    tsmom_binary_vol_parity_backtest.py's own _build_returns_wide).

    Returns (H, covered). H is ALWAYS a dense n x n matrix (n = len(symbols),
    never None -- np.eye(n) is a real, usable "nothing measured" placeholder,
    not an error signal); the canonical output now that this function builds
    one joint matrix rather than independently-estimated pairs (see above).
    compute_idm/compute_erc_weights/compute_hrp_weights/compute_notional_
    split all take H directly too (H is their sole correlation-data input --
    no hand-built corr_pairs dict accepted on those functions' own
    signatures either; a caller with only pairwise correlations writes H
    directly, e.g. test_allocation.py's own fixtures), so there's no
    pairwise dict anywhere in this path at all -- one representation, not
    two that could drift apart.

    covered is a length-n boolean array (True where `symbols[i]` actually
    had return data in this window, `symbols`' own order) -- an EXPLICIT,
    PER-SYMBOL usability signal, not something a caller should infer from H
    itself. This matters because H's identity-default entries (1.0 diag,
    0.0 off-diag) for an uncovered symbol are a computational placeholder,
    not a measurement: H[i, j] = 0 for an uncovered symbol i means "no
    correlation was measured," NOT "this symbol was measured and found to
    be uncorrelated with j." Treating those two as the same thing silently
    manufactures fake diversification credit -- confirmed directly: adding
    one zero-history symbol to an otherwise-real 2-asset correlation matrix
    inflated IDM by 34% and handed that symbol the largest individual ERC
    weight of the three, purely because "unknown" was encoded as "measured
    independent." Every caller that cares about this distinction (currently
    compute_notional_split/compute_symbol_notional_budget, both of which
    accept `covered` and route uncovered symbols to a capped, correlation-
    blind fallback allocation instead of letting them participate in H at
    all -- see UNCOVERED_BUDGET_CAP_FRACTION) must consult `covered`
    directly rather than trusting H's own entries for an uncovered symbol.

    A caller that doesn't care about the covered/uncovered distinction (or
    is calling compute_idm/compute_erc_weights/compute_hrp_weights
    directly, without going through compute_notional_split/compute_symbol_
    notional_budget) should still guard against the DEGENERATE case where
    NOTHING is covered (covered.all() is False and covered.any() is False,
    i.e. `covered.sum() == 0`): H is np.eye(n) there too, and passing it
    straight through would manufacture the same fake-independence credit
    for the WHOLE active set, not just one symbol -- squash H to None in
    that case (`H = None if not covered.any() else H`) so compute_idm's
    etc. own `H is None` fallback fires instead.

    (np.eye(len(symbols)), all-False) if the bounded slice itself has fewer
    than `min_rows` (too little history this early in the backtest to trust
    ANY correlation estimate, regardless of which symbols technically have
    a column) or if fewer than 2 symbols have data in that slice; (H,
    per-symbol coverage) otherwise, where H's covered rows/columns are real
    measurements and its uncovered ones are the identity placeholder."""
    window_start = as_of - timedelta(days=int(window_years * 365.25))
    sl = returns_wide.filter((pl.col('ts_event') >= window_start) & (pl.col('ts_event') < as_of))
    n = len(symbols)
    if sl.height < min_rows:
        return np.eye(n), np.zeros(n, dtype=bool)

    present = [s for s in symbols if s in sl.columns]
    covered = np.array([s in present for s in symbols])
    if len(present) < 2:
        return np.eye(n), covered

    # Static weight vector equivalent to ewm_mean(half_life=..., adjust=True)
    # evaluated at the last row of the slice: row t (0-indexed from the
    # start) gets weight (1-alpha)^((T-1)-t), normalized to sum to 1.
    T = sl.height
    alpha = 1.0 - 2.0 ** (-1.0 / halflife)
    age = (T - 1) - np.arange(T)
    weights = (1.0 - alpha) ** age
    weights /= weights.sum()

    X = sl.select(present).to_numpy()
    mu = weights @ X
    Z = np.sqrt(weights)[:, None] * (X - mu)
    C = Z.T @ Z
    d = np.sqrt(np.diag(C))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr_present = C / np.outer(d, d)
    # nan (0/0, a zero-variance/constant column in-window) defaults to the
    # same 0.0 off-diagonal np.eye already carries for an absent symbol;
    # +-inf can't arise mathematically here (Cauchy-Schwarz bounds |cov_ij|
    # by d_i*d_j, so a zero denominator forces a zero numerator too) but is
    # guarded the same way as a defensive floor against float noise, same
    # as the explicit clip below.
    corr_present = np.nan_to_num(corr_present, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(corr_present, -1.0, 1.0, out=corr_present)
    np.fill_diagonal(corr_present, 1.0)

    # Embed the correlation matrix for symbols with data into the FULL
    # requested symbol universe, preserving `symbols` order. Symbols with
    # no data retain the identity fallback (diag=1, off-diag=0) -- a
    # placeholder for "unmeasured," not evidence of independence; `covered`
    # (built above, before this block) is what tells a caller which is
    # which.
    idx = {s: i for i, s in enumerate(symbols)}
    present_idx = np.array([idx[s] for s in present])
    H = np.eye(n)
    H[np.ix_(present_idx, present_idx)] = corr_present

    return H, covered


def _spinu_erc_newton(H: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Damped-Newton solve of Spinu's (2013) convex risk-budgeting
    reformulation -- minimize_{y > 0} f(y) = (1/2) y'Hy - sum_i b_i*log(y_i),
    then w = y / sum(y) -- see compute_erc_weights' own docstring for why
    this replaces a generic constrained simplex search for b = uniform
    1/n (equal risk contribution). Generalizes to arbitrary b (any
    positive budget vector, not required to sum to 1) -- untested by
    compute_erc_weights itself, which always calls this with b = 1/n, but
    exercised directly in test_allocation.py to validate the solver
    against non-uniform budgets (RC_i proportional to b_i, not just
    equal), since compute_erc_weights' own tests only ever see b = 1/n
    and so can't distinguish "solves ERC" from "coincidentally solves
    ERC because b happens to be uniform."

    Returns None (not a fallback value -- that's the caller's decision) on
    genuine non-convergence within ERC_NEWTON_MAX_ITER (a materially large
    gradient with no way to shrink it further -- e.g. a pathological H).
    Backtracking line search exhausting without finding a step that both
    stays positive and strictly decreases f is NOT automatically treated
    as non-convergence: on adversarial near-singular H (a tight cluster of
    correlations near +-1), the objective itself can already be at its
    floating-point-representable minimum before the gradient reaches
    ERC_NEWTON_TOL, so a stalled line search accepts the current point
    instead when its own gradient is at least below the looser
    ERC_NEWTON_STALL_TOL -- confirmed via a 2100-trial stress test (n in
    2..40, random well-conditioned matrices and adversarial near-singular
    ones up to cond(H)~4e7): 0 failures at these thresholds, vs. real
    (if infrequent -- ~1%) spurious failures on the near-singular cases
    when relying on ERC_NEWTON_TOL alone."""
    n = H.shape[0]
    y = np.ones(n)
    for _ in range(ERC_NEWTON_MAX_ITER):
        Hy = H @ y
        grad = Hy - b / y
        gmax = np.max(np.abs(grad))
        if gmax < ERC_NEWTON_TOL:
            return y / y.sum()
        hess = H + np.diag(b / y ** 2)
        step = np.linalg.solve(hess, -grad)

        # Damped/backtracking line search: positivity alone isn't enough
        # to accept a Newton step (an oversized step can still increase f
        # even while staying positive) -- require an actual objective
        # decrease too, halving the step until both hold.
        f0 = 0.5 * y @ Hy - b @ np.log(y)
        t = 1.0
        y_next = y  # t=1.0 >= ERC_NEWTON_MIN_STEP always holds, so the loop
                     # below runs at least once and always overwrites this --
                     # only here so a static checker can see y_next is bound.
        accepted = False
        while t >= ERC_NEWTON_MIN_STEP:
            y_next = y + t * step
            if np.all(y_next > 0):
                f_next = 0.5 * y_next @ (H @ y_next) - b @ np.log(y_next)
                if f_next < f0:
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            return y / y.sum() if gmax < ERC_NEWTON_STALL_TOL else None
        y = y_next
    return None


def compute_erc_weights(active_symbols: list[str],
                         H: Optional[np.ndarray] = None) -> dict[str, float]:
    """Equal Risk Contribution weights (Maillard, Roncalli & Teiletche,
    2010), solved on the CORRELATION matrix directly rather than a raw
    covariance matrix -- deliberately, not a simplification of convenience:
    every instrument reaching this system's portfolio-level sizing has
    ALREADY been vol-equalized by compute_position_scalar's own
    risk_scalar (vol_target / current_realized_vol), so folding in each
    instrument's raw variance a second time here would double-count that
    normalization. Treating the correlation matrix (1.0 diagonal) as the
    relevant "covariance" is exactly the same simplification compute_idm's
    own W @ H @ W_t already makes.

    Solved via Spinu's (2013) convex reformulation, not a generic
    constrained search over the simplex (the textbook formulation this
    project's own scripts/tsmom_risk_budget_diagnostic.py still uses, via
    scipy.optimize.minimize/SLSQP with numerical gradients -- fine for
    that module's own read-only, manually-run diagnostic, but this
    function runs once per rebalance date across an entire backtest, where
    SLSQP's finite-difference Jacobian showed up as ~16% of a full
    backtest's total runtime under profiling):

        minimize_{y > 0}  f(y) = (1/2) y' H y - sum_i b_i * log(y_i)

    b_i = 1/n (equal budget, hence "Equal" Risk Contribution), then w = y
    / sum(y). At the stationary point, grad f(y) = Hy - b/y = 0, i.e.
    y_i * (Hy)_i = b_i for every i -- exactly the equal-risk-contribution
    condition, and that proportionality survives the normalization to w
    unchanged (RC_i(w) = b_i / (sum(y) * sqrt(y'Hy)), the same constant
    denominator for every i). Unlike the old squared-RC-deviation
    objective, f is PROVABLY CONVEX (H is PSD, -log(y_i) is convex for
    y_i > 0) -- one global minimum, not a black-box search that can fail
    to converge -- with closed-form gradient/Hessian (grad = Hy - b/y,
    hess = H + diag(b/y^2), always strictly positive definite since the
    diagonal term is strictly positive even where H alone is singular/
    near-singular for highly correlated symbols), so it's solved by
    damped Newton's method (typically converges in well under 10
    iterations regardless of n) instead of a derivative-free constrained
    search. Confirmed numerically equivalent to the old SLSQP solve --
    not just in the resulting weights, but in the thing that actually
    matters, each asset's realized risk contribution RC_i = w_i * (H @
    w)_i / sqrt(w' H w) -- and ~65x faster per call at n=12 (see
    tests/domain/test_allocation.py's own SLSQP cross-check, a fresh
    independent oracle, not the retired production implementation).

    Falls back to uniform 1/n (matching compute_idm's own fallback
    convention) when fewer than 2 active_symbols, H is None (no usable
    correlation data), or (rare -- H strictly PD makes this a well-posed
    problem) the damped Newton solve doesn't converge within max_iter
    (logged) -- one bad date's numerical hiccup shouldn't abort an
    otherwise-multi-year backtest run.

    `H` is the sole correlation-data input -- a hand-built corr_pairs dict
    (e.g. a test fixture) is no longer accepted directly; write H itself.
    Keeping two accepted representations of the same data (a dict AND a
    matrix) was a redundancy this whole module's H-primary refactor was
    meant to eliminate -- it just hadn't reached this function's own
    public signature yet."""
    n = len(active_symbols)
    if n < 2 or H is None:
        return {s: 1.0 / n for s in active_symbols} if n > 0 else {}

    w = _spinu_erc_newton(H, np.full(n, 1.0 / n))
    if w is None:
        log.warning("ERC Newton solve failed to converge within %d iterations -- "
                    "falling back to flat 1/n weights", ERC_NEWTON_MAX_ITER)
        return {s: 1.0 / n for s in active_symbols}
    return dict(zip(active_symbols, w))


def _cluster_var(H: np.ndarray, cluster_idx: list[int]) -> float:
    """Inverse-variance-weighted cluster variance (Lopez de Prado's own
    getClusterVar) -- degenerates to a plain average of H's sub-block since
    every diagonal entry of H is 1.0 by construction (see
    compute_hrp_weights' own docstring for why that's the correct
    simplification here, not an oversight). Clamped to a small positive
    floor: H's off-diagonal entries are independently-estimated pairwise
    correlations, not guaranteed to form a jointly positive-semidefinite
    matrix for cluster sizes > 2, so this quadratic form can land a hair
    below zero on a near-degenerate block -- the caller's own alpha
    computation divides by (var_left + var_right), so a floor here matters
    more than it would for a lone diagonal read."""
    sub = H[np.ix_(cluster_idx, cluster_idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return max(float(ivp @ sub @ ivp), 1e-12)


def _hrp_recursive_bisection(H: np.ndarray, order: list[int]) -> np.ndarray:
    """Lopez de Prado's getRecBipart (2016), translated from his own
    pandas-Series reference implementation to a plain numpy array indexed
    by original (pre-reorder) position -- `order` (the dendrogram leaf
    order) determines split structure only; weights are written back at
    each element's original index either way, so the returned array is
    already aligned to the caller's own active_symbols order."""
    w = np.ones(len(order))
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters
                    for i, j in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_var(H, left)
            var_right = _cluster_var(H, right)
            alpha = min(1.0, max(0.0, 1.0 - var_left / (var_left + var_right)))
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    # Guaranteed to sum to ~1 already by the recursive halving/complement
    # property (alpha + (1 - alpha) == 1 at every split); normalizing is
    # just a floating-point-drift correction, not a structural fix.
    return w / w.sum()


def compute_hrp_weights(active_symbols: list[str],
                         H: Optional[np.ndarray] = None) -> dict[str, float]:
    """Hierarchical Risk Parity (Lopez de Prado, 2016), reimplemented here
    in pure numpy/scipy rather than calling PyPortfolioOpt's HRPOpt
    (scripts/tsmom_risk_budget_diagnostic.py's own choice, for a one-off,
    manually-run comparison). Two reasons this needs its own
    implementation instead of reusing that one: HRPOpt requires a pandas
    DataFrame input (this project's CLAUDE.md keeps pandas scoped to a
    single library call site, not a per-rebalance-date hot path across an
    entire backtest), and HRPOpt.optimize() guards its linkage_method
    argument against a private scipy attribute that scipy >= 1.18 removed
    entirely, needing its own compatibility shim. scipy.cluster.hierarchy's
    linkage/leaves_list gives the same dendrogram leaf order directly, with
    no pandas and no plotting dependency (dendrogram() itself is never
    called here -- leaves_list reads the leaf order straight off the
    linkage matrix).

    Like compute_erc_weights, operates on the correlation matrix directly
    rather than a raw covariance matrix, for the same reason: every
    instrument is already vol-equalized upstream by compute_position_
    scalar's own risk_scalar. The one accepted consequence (see
    _cluster_var's own docstring): HRP's usual "size inversely to each
    cluster member's own variance" step degenerates to an equal split
    within a cluster, since every diagonal entry of the correlation matrix
    is 1.0 -- exactly consistent with that vol-equalization having already
    happened, not a lost feature.

    Falls back to uniform 1/n (matching compute_idm's own fallback
    convention) when fewer than 2 active_symbols or H is None (no usable
    correlation data).

    `H` is the sole correlation-data input -- see compute_erc_weights' own
    docstring for why a hand-built corr_pairs dict isn't accepted directly
    here anymore; write H itself."""
    n = len(active_symbols)
    if n < 2 or H is None:
        return {s: 1.0 / n for s in active_symbols} if n > 0 else {}

    distance = np.sqrt(np.clip(0.5 * (1.0 - H), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    link = sch.linkage(condensed, method='single')
    order = sch.leaves_list(link).tolist()

    w = _hrp_recursive_bisection(H, order)
    return dict(zip(active_symbols, w))


def _partial_coverage(active_symbols: list[str], covered: Optional[np.ndarray]) -> bool:
    """True iff `covered` (see _bounded_ewm_correlation_matrix's own
    docstring) marks SOME but not all of active_symbols -- the only
    situation needing special covered/uncovered handling. Fully-covered
    (nothing to protect against) and fully-uncovered (nothing to protect
    -- see _squash_fully_uncovered_H) both degenerate to the ordinary
    single-pool path."""
    if covered is None:
        return False
    n_covered = int(covered.sum())
    return 0 < n_covered < len(active_symbols)


def _squash_fully_uncovered_H(H: Optional[np.ndarray],
                               covered: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """H is _bounded_ewm_correlation_matrix's np.eye(n) placeholder when
    `covered` is present but entirely False (no symbol had usable data at
    all, whether from too few window rows or too few present columns) --
    squash to None so compute_idm/compute_erc_weights/compute_hrp_weights
    fall back to their own flat/1.0 default instead of treating "identity
    matrix" as measured independence for the WHOLE active set."""
    if covered is not None and not covered.any():
        return None
    return H


def _blend_covered_uncovered_split(active_symbols: list[str], covered: np.ndarray,
                                    notional_weighting: str, H: Optional[np.ndarray]) -> dict[str, float]:
    """The 'erc'/'hrp' split when coverage is PARTIAL (see
    _partial_coverage): uncovered symbols never participate in the
    correlation-aware optimization at all (their row/column in H is a
    placeholder, not a measurement -- see _bounded_ewm_correlation_matrix's
    own docstring), so they can't inherit an inflated share the way
    passing the full H straight through would let them. Instead:

    - covered symbols split B_c = 1 - B_u of the total via the ordinary
      compute_erc_weights/compute_hrp_weights call, restricted to just
      their own sub-block of H (H_c) -- i.e. exactly as if the uncovered
      symbols didn't exist for this calculation.
    - uncovered symbols split B_u = min(UNCOVERED_BUDGET_CAP_FRACTION,
      |uncovered| / |active_symbols|) flat among themselves -- a single
      new instrument gets roughly its proportional headcount share, but a
      wave of simultaneously-uncovered instruments can't collectively
      claim more than UNCOVERED_BUDGET_CAP_FRACTION regardless of count."""
    covered_symbols = [s for s, c in zip(active_symbols, covered) if c]
    uncovered_symbols = [s for s, c in zip(active_symbols, covered) if not c]
    covered_idx = [i for i, c in enumerate(covered) if c]
    H_c = H[np.ix_(covered_idx, covered_idx)] if H is not None else None

    budget_uncovered = min(UNCOVERED_BUDGET_CAP_FRACTION, len(uncovered_symbols) / len(active_symbols))
    budget_covered = 1.0 - budget_uncovered

    weight_fn = compute_erc_weights if notional_weighting == 'erc' else compute_hrp_weights
    split_covered = weight_fn(covered_symbols, H_c)

    split = {s: budget_covered * w for s, w in split_covered.items()}
    flat_uncovered = budget_uncovered / len(uncovered_symbols)
    split.update({s: flat_uncovered for s in uncovered_symbols})
    return split


def _coverage_restricted_idm(active_symbols: list[str], H: Optional[np.ndarray],
                              covered: Optional[np.ndarray],
                              weights: Optional[dict[str, float]] = None) -> float:
    """compute_idm, restricted to the covered subset when coverage is
    partial -- IDM answers "how much bigger can the book be, given
    MEASURED diversification," so a symbol with no correlation evidence
    must contribute nothing to that estimate, not even at a small weight
    (compare _blend_covered_uncovered_split, which still gives an
    uncovered symbol a capped NOTIONAL share -- IDM is a different
    question, entirely about what's been measured). Fully covered/
    uncovered both degenerate to the ordinary single-pool compute_idm
    call (via _squash_fully_uncovered_H for the fully-uncovered case)."""
    if _partial_coverage(active_symbols, covered):
        assert covered is not None
        covered_symbols = [s for s, c in zip(active_symbols, covered) if c]
        covered_idx = [i for i, c in enumerate(covered) if c]
        H_c = H[np.ix_(covered_idx, covered_idx)] if H is not None else None
        covered_weights = {s: weights[s] for s in covered_symbols} if weights else None
        return compute_idm(covered_symbols, H=H_c, weights=covered_weights)
    return compute_idm(active_symbols, H=_squash_fully_uncovered_H(H, covered), weights=weights)


def compute_notional_split(active_symbols: list[str], notional_weighting: str,
                            H: Optional[np.ndarray] = None,
                            covered: Optional[np.ndarray] = None) -> dict[str, float]:
    """The 'flat'/'erc'/'hrp' fraction of the total dollar-vol budget each
    active symbol gets -- the same split compute_symbol_notional_budget
    computes internally (and, when use_idm=True, feeds into compute_idm as
    its own weight vector), pulled out into its own function so a caller
    that already has active_symbols/H can inspect the split itself
    directly (e.g. reporting/diagnostics -- what fraction of the book did
    ERC/HRP actually give this symbol), not just the resulting dollar
    figure. 'flat': 1/n each, ALWAYS -- correlation-blind by definition, so
    `H`/`covered` don't apply (there's no fake-diversification credit to
    protect against when nothing depends on correlation). 'erc'/'hrp':
    compute_erc_weights/compute_hrp_weights on H -- see either's own
    docstring for the fallback-to-flat behavior when H is None or n < 2.

    `covered`: _bounded_ewm_correlation_matrix's own per-symbol coverage
    mask (see its docstring). When coverage is PARTIAL under 'erc'/'hrp',
    delegates to _blend_covered_uncovered_split instead of letting an
    uncovered symbol inherit a share of the split from H's identity
    placeholder. Ignored (H used as-is, or squashed to None if NOTHING is
    covered) when coverage is uniform or `covered` is omitted -- e.g. a
    caller with a hand-written H that has no concept of per-symbol
    coverage at all."""
    if notional_weighting not in NOTIONAL_WEIGHTING_SCHEMES:
        raise ValueError(f"notional_weighting must be one of {NOTIONAL_WEIGHTING_SCHEMES}, "
                          f"got {notional_weighting!r}")
    if not active_symbols:
        return {}
    if notional_weighting == 'flat':
        return {s: 1.0 / len(active_symbols) for s in active_symbols}
    if _partial_coverage(active_symbols, covered):
        assert covered is not None
        return _blend_covered_uncovered_split(active_symbols, covered, notional_weighting, H)
    H = _squash_fully_uncovered_H(H, covered)
    if notional_weighting == 'erc':
        return compute_erc_weights(active_symbols, H)
    return compute_hrp_weights(active_symbols, H)


def compute_symbol_notional_budget(active_symbols: list[str], returns_wide: Optional[pl.DataFrame],
                                    as_of: date, capital: float, target_portfolio_vol: float,
                                    vol_target: float, corr_window_years: float,
                                    corr_halflife_days: float,
                                    notional_weighting: str = 'flat',
                                    use_idm: bool = True,
                                    H: Optional[np.ndarray] = None,
                                    covered: Optional[np.ndarray] = None) -> dict[str, float]:
    """IDM-derived per-symbol notional_budget for TsmomBacktestConfig's
    target_portfolio_vol path (tsmom_backtester.py's run_tsmom_backtest) --
    the correlation-aware alternative to sizing off a flat config.max_notional.
    Computes a total target DOLLAR VOL budget (capital * target_portfolio_vol *
    IDM, where IDM captures active_symbols' REAL measured correlation rather
    than assuming independence), splits it across active_symbols per
    `notional_weighting`, then divides by vol_target to convert each
    symbol's own dollar-vol share back into the notional_budget
    _compute_signal_row actually expects.

    The division by vol_target matters: scalar already folds in
    risk_scalar = vol_target / current_realized_vol (compute_position_scalar's
    own per-instrument vol-equalization), so passing the dollar-vol target
    straight through as notional_budget would apply vol_target a SECOND
    time on top of an already-vol-target-derived figure -- confirmed
    directly during this function's extraction from tsmom_backtester.py,
    where an earlier version did exactly that and undershot a 15% target by
    ~24x.

    notional_weighting (NOTIONAL_WEIGHTING_SCHEMES) selects how the total
    dollar-vol budget is split ACROSS active_symbols:
      'flat' (default -- exact prior behavior, unchanged): the total is
        split equally across active_symbols regardless of correlation
        structure -- every symbol gets the same budget.
      'erc' / 'hrp': split via compute_erc_weights/compute_hrp_weights on
        the SAME bounded correlation matrix already built for IDM -- a
        correlated cluster's members collectively end up with roughly one
        undiversified bet's worth of budget, rather than each member
        separately claiming an equal share of the whole book. This is the
        automated, data-driven alternative to Carver's hand-assigned
        subgroups that research/cta-layer-separation-risk-budgeting.md
        flags as this system's least-precedented, unaddressed gap.

    The split is computed FIRST and then fed into compute_idm as `weights`
    (when use_idm=True) -- NOT the flat 1/n vector regardless of
    notional_weighting, as an earlier version of this function did. Using
    mismatched weights would double-count diversification: IDM would credit
    the active set's correlation structure once (sized as if the total were
    about to be split flat), and then 'erc'/'hrp' would credit it again by
    reshaping that same total's distribution to reduce correlated exposure
    further, systematically undershooting target_portfolio_vol beyond the
    imprecision this module's own docstring already documents for 'flat'.
    Feeding the SAME weight vector into both the multiplier and the split
    means IDM answers "how diversified is the portfolio given the weights
    I'm actually about to use," not a hypothetical flat one -- exactly one
    diversification adjustment, self-consistently applied. For 'flat' this
    is numerically identical to the prior flat-vector behavior.

    use_idm: True (default) applies the IDM multiplier as described above.
    False skips it entirely (total_dollar_vol_target = capital *
    target_portfolio_vol, no correlation-based up/down-sizing of the total)
    -- for notional_weighting='erc'/'hrp' this isolates the diversification
    adjustment to the split alone (closer to Option B in
    research/cta-layer-separation-risk-budgeting.md's IDM-vs-allocation
    framing); for 'flat' it removes correlation-awareness altogether.

    Returns {} if active_symbols is empty, or if there's no way to get a
    correlation matrix at all (returns_wide is None AND H wasn't already
    supplied) -- the caller's own probe pass already means every such
    symbol's target is 0 regardless of budget, so there's nobody to size
    for. A too-short bounded correlation window
    (_bounded_ewm_correlation_matrix's own min_rows floor) instead makes
    compute_idm (and, under 'erc'/'hrp', the weighting itself) fall back
    to its own flat/no-adjustment default, not an early return here --
    and neither does passing H directly without returns_wide (the whole
    point of the H/covered params below is to let a caller skip needing
    returns_wide at all once they already have H).

    `H`/`covered`: pass both if a caller already ran
    _bounded_ewm_correlation_matrix for this exact (active_symbols, as_of,
    corr_window_years, corr_halflife_days) -- e.g. the live rebalance report,
    which needs them itself for notional_weight_by_symbol before ever
    calling this function. Skips rerunning the EWM estimation over
    returns_wide a second time. `covered` is what actually protects an
    uncovered symbol (a live signal but no correlation data -- see
    _bounded_ewm_correlation_matrix's own docstring) from inheriting a
    fake diversification-credited share via H's identity placeholder: the
    split routes through compute_notional_split's own covered-aware
    blending, and use_idm's multiplier is computed via
    _coverage_restricted_idm, which excludes uncovered symbols from the
    measurement entirely rather than crediting them at a merely-smaller
    weight. Both recomputed from returns_wide when H is omitted (the
    default)."""
    if notional_weighting not in NOTIONAL_WEIGHTING_SCHEMES:
        raise ValueError(f"notional_weighting must be one of {NOTIONAL_WEIGHTING_SCHEMES}, "
                          f"got {notional_weighting!r}")
    if not active_symbols or (H is None and returns_wide is None):
        return {}
    if H is None:
        # returns_wide is guaranteed non-None here: the guard above only
        # returns early when BOTH H and returns_wide are None, and this
        # branch is H is None -- but that's a fact about the two guards
        # together, not something a type checker narrows across separate
        # if-statements on its own.
        assert returns_wide is not None
        H, covered = _bounded_ewm_correlation_matrix(returns_wide, active_symbols, as_of,
                                                       corr_window_years, corr_halflife_days)
    split = compute_notional_split(active_symbols, notional_weighting, H, covered)
    idm_multiplier = _coverage_restricted_idm(active_symbols, H, covered, weights=split) if use_idm else 1.0
    total_dollar_vol_target = capital * target_portfolio_vol * idm_multiplier

    return {s: (total_dollar_vol_target * split[s]) / vol_target for s in active_symbols}


def compute_idm(active_symbols: list[str], H: Optional[np.ndarray] = None,
                 weights: Optional[dict[str, float]] = None) -> float:
    """Carver's Instrument Diversification Multiplier, exact matrix form:
    IDM = 1 / sqrt(W @ H @ W_t) -- W the weight vector (equal, 1/n each,
    when `weights` is None; otherwise `weights` normalized so
    sum(abs(w)) == 1, preserving each symbol's relative and signed share),
    H the REAL pairwise correlation matrix (1.0 on the diagonal) -- NOT
    the average-correlation algebraic shortcut (1/sqrt(1/N + (1-1/N)*
    avg_corr)), which is only exactly equivalent to this when every
    pairwise correlation happens to be identical AND weights are equal.
    Since _bounded_ewm_correlation_matrix already builds the full matrix,
    using it directly costs nothing extra over averaging its entries down
    to one scalar first.

    Per Carver's own convention, H is floored at 0 here (negative
    pairwise correlations clamped to 0) before computing IDM -- NOT for
    compute_erc_weights/compute_hrp_weights, which use the real signed
    matrix, since a negative correlation is a genuine hedging signal
    those optimizers should weigh into the allocation itself. IDM asks a
    different question -- how much bigger can the total book be, given
    today's measured diversification -- and a negative correlation is
    precisely the estimate least likely to survive a stress regime:
    correlations tend to converge upward exactly when diversification
    would matter most, so IDM shouldn't credit extra leverage for a hedge
    that may not hold.

    Falls back to 1.0 (no diversification adjustment) whenever there's
    nothing meaningful to compute from: fewer than 2 active_symbols, or H
    is None (not enough bounded-window history yet, or fewer than 2
    symbols had synchronized return data).

    `H` is the sole correlation-data input -- see compute_erc_weights' own
    docstring for why a hand-built corr_pairs dict isn't accepted directly
    here anymore; write H itself. The 0-floor above is applied to whatever
    H is passed -- it's an IDM-specific adjustment,
    not a property of the shared matrix itself."""
    n = len(active_symbols)
    if n < 2 or H is None:
        return 1.0
    H = np.clip(H, 0.0, 1.0)
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        raw = np.array([weights.get(s, 0.0) for s in active_symbols])
        total_abs = np.abs(raw).sum()
        w = raw / total_abs if total_abs > 0 else np.full(n, 1.0 / n)
    port_var = w @ H @ w
    if port_var <= 0:
        return 1.0
    return 1.0 / math.sqrt(port_var)


def compute_realized_portfolio_risk(active_symbols: list[str], H: np.ndarray,
                                     dollar_exposure: dict[str, float]) -> dict:
    """Realized (post-sizing) portfolio dollar vol and each symbol's
    marginal dollar risk contribution -- a diagnostic, NOT a sizing input:
    answers "given the positions this rebalance actually ended up with
    (after lot-size rounding and apply_cluster_risk_cap), what is the
    book's true correlation-aware vol," as opposed to compute_idm/
    compute_erc_weights/compute_hrp_weights, which all answer sizing
    questions about a HYPOTHETICAL weight vector before any rounding
    happens.

    `dollar_exposure`: {symbol: SIGNED dollar risk}, positive for a long,
    negative for a short -- e.g. {t['symbol']: math.copysign(t
    ['standalone_position_dollar_vol'], t['final_target_contracts']) for t
    in targets if not t.get('error')}. This is deliberately NOT the same as
    apply_cluster_risk_cap's own `standalone_position_dollar_vol` field
    (abs(final_target_contracts) * close *
    mult * hv, always >= 0), and NOT the same vector as compute_idm/
    compute_notional_split's own `weights` (a pre-sizing, always-
    nonnegative ERC/HRP BUDGET SPLIT, one pipeline stage earlier -- see
    those functions' own docstrings). This function's own vector -- called
    `risk_exposure` (mathematically x) below, not `w`, specifically to
    keep the two apart -- is the REALIZED, signed dollar-vol exposure each
    symbol ended up with, downstream of that budget split, sizing,
    rounding, and apply_cluster_risk_cap: x' H x only nets a short against
    a positively-correlated long (or compounds it against a negatively-
    correlated one) if direction is actually in the vector. Feed this
    function unsigned magnitudes instead and you silently get the
    variance of a hypothetical all-long book -- every short's hedging (or
    anti-hedging) interaction with the rest of the portfolio disappears,
    and any resulting negative portfolio risk contribution reflects H's raw
    correlation sign against an artificially all-positive x, not the
    symbol's actual direction. Missing symbols default to 0.0 (no
    position). H is a CORRELATION matrix (1.0 diagonal, PSD) with no
    notion of any instrument's own vol built in -- the vector multiplied
    through it must already carry that, unlike a raw notional or
    contract-count weight.

    port_var = x' H x, port_vol = sqrt(port_var) (0.0 if port_var <= 0,
    e.g. every exposure is 0). H being PSD guarantees x' H x >= 0 for ANY
    signed x, so port_var can never go negative here regardless of how the
    book is split long/short -- the individual Euler terms below are what
    can be negative, not the total. marginal = H @ x; contribution_i
    = x_i * marginal_i / port_vol -- by construction (Euler's theorem for
    a degree-1-homogeneous function), the contributions sum to
    port_vol exactly, so it's a genuine decomposition of the book's total
    risk across symbols, not an approximation. Because x is signed here,
    a portfolio risk contribution can legitimately be negative for a real reason now:
    a short position that's net diversifying (or a long that's net
    diversifying against the book's shorts) marginally REDUCES total
    portfolio variance -- diversification/hedging showing up exactly as
    it should, not a contradiction of H's own PSD-ness -- not just
    "correlated with a low weight."

    The point of comparing portfolio risk contribution against the SAME
    symbol's UNSIGNED standalone_position_dollar_vol (apply_cluster_risk_cap's
    field, not this function's input): standalone_position_dollar_vol is each
    instrument's UNDIVERSIFIED
    standalone dollar vol (what it would contribute alone, direction
    stripped out since a standalone position has no portfolio to net
    against); portfolio risk contribution is what it ACTUALLY contributes given
    today's measured correlation AND its own direction -- exactly the gap
    compute_idm/ERC/HRP sizing is meant to account for UP FRONT, so this
    is also a live check on whether that sizing is holding up against
    realized (rounded) positions, not just the pre-rounding theoretical
    split. A genuine diversifier's contribution comes in BELOW its own
    standalone position dollar-vol (confirmed directly in test_allocation.py); a
    symbol correlated with the rest of the book AND held in the same net
    direction approaches, but is bounded above by its own standalone position
    dollar-vol when exposures are comparable in size (the sum of contributions
    is no more than the sum of standalone dollar-vol in that same-direction
    case, since correlation
    entries are bounded by 1 -- Cauchy-Schwarz) -- an individual risk_
    contribution exceeding its own standalone position dollar-vol IS possible in general
    (e.g. a small position heavily correlated with a much larger
    cluster), just not in a symmetric, comparably-weighted, same-
    direction case."""
    risk_exposure = np.array([dollar_exposure.get(s, 0.0) for s in active_symbols])
    port_var = risk_exposure @ H @ risk_exposure
    if port_var <= 0:
        return {
            'port_vol': 0.0,
            'portfolio_risk_contribution': {s: 0.0 for s in active_symbols},
        }
    port_vol = math.sqrt(port_var)
    marginal = H @ risk_exposure
    rc = risk_exposure * marginal / port_vol
    return {'port_vol': port_vol, 'portfolio_risk_contribution': dict(zip(active_symbols, rc))}


def group_by_cluster(cluster_by_symbol: dict[str, str], values: dict[str, float]) -> dict[str, float]:
    """Sum a per-symbol dollar figure -- standalone position dollar-vol
    (undiversified) or compute_realized_portfolio_risk's own portfolio risk
    contribution (diversification-aware) are the two this project uses -- into
    per-cluster totals. Generic aggregation, not tied to either one
    specifically: whatever `values` means per-symbol, this sums it by
    cluster the same way. A symbol present in `values` but missing from
    `cluster_by_symbol` falls into an 'other' bucket, matching
    build_instruments' own cluster-default convention."""
    totals: dict[str, float] = {}
    for symbol, value in values.items():
        cluster = cluster_by_symbol.get(symbol, 'other')
        totals[cluster] = totals.get(cluster, 0.0) + value
    return totals
