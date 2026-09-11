"""
Tests for domain.allocation -- risk allocation, both single-instrument
(compute_position_scalar: this instrument's own vol-targeted risk_scalar)
and cross-instrument (compute_n_effective/compute_desired_risk_budget/
apply_cluster_risk_cap/compute_idm: how instruments/clusters get weighted
relative to each other, given they aren't independent bets).

Consolidated (2026-07) from test_signal.py (formerly test_tsmom_signal.py),
once compute_position_scalar itself moved out of domain/signal.py into
domain/allocation.py -- it's risk-SIZING math, not signal CONSTRUCTION, so
it belongs with the rest of this module's risk-allocation functions, not
alongside build_features/continuous_momentum/goulding_monthly.
compute_n_effective/compute_desired_risk_budget/apply_cluster_risk_cap's own
tests had already been left behind in test_signal.py after an earlier,
incomplete module split (only their imports were fixed at the time, not
their physical test-file location) -- moved here now too, completing that
split properly.
"""

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from derivatives_bt_engine.domain.allocation import (
    UNCOVERED_BUDGET_CAP_FRACTION,
    _bounded_ewm_correlation_matrix,
    _coverage_restricted_idm,
    allocate_lot_aware_targets,
    _spinu_erc_newton,
    apply_cluster_risk_cap,
    build_returns_wide,
    compute_desired_risk_budget,
    compute_erc_weights,
    compute_hrp_weights,
    compute_idm,
    compute_n_effective,
    compute_notional_split,
    compute_position_scalar,
    compute_realized_portfolio_risk,
    compute_symbol_notional_budget,
    group_by_cluster,
)
from derivatives_bt_engine.domain.enums import TrendRegime
from derivatives_bt_engine.domain.signal import (
    calculate_trend_strength,
    compute_signal_confidence,
    compute_vol_ratio,
)


def _vol_series(n_calm: int, n_shock: int, calm_vol: float, shock_vol: float,
                drift: float = 0.0005, seed: int = 0) -> pl.DataFrame:
    """A price series that's calm for n_calm bars, then switches to a
    different (shock_vol) volatility for the trailing n_shock bars --
    simulates an instrument-specific vol regime change independent of any
    broad-market state."""
    rng = np.random.default_rng(seed)
    rets = np.concatenate([
        rng.normal(drift, calm_vol, n_calm),
        rng.normal(0.0, shock_vol, n_shock),
    ])
    close = 100 * np.exp(np.cumsum(rets))
    return calculate_trend_strength(pl.DataFrame({'close': close}))


# ── compute_position_scalar ──────────────────────────────────────────────────

def test_position_scalar_sign_follows_trend_strength():
    pos = compute_position_scalar(0.5, 0.01, vol_target=0.15, regime=TrendRegime.BULL)
    neg = compute_position_scalar(-0.5, 0.01, vol_target=0.15, regime=TrendRegime.BEAR)
    assert pos > 0
    assert neg < 0


def test_position_scalar_preserves_permitted_low_vol_leverage():
    # A full-strength directional signal may use the 2x risk_scalar ceiling.
    # The directional input is bounded separately; it must not re-clamp the
    # finished scalar and erase allowed low-vol leverage.
    scalar = compute_position_scalar(1.0, daily_std_last=0.0001, vol_target=0.50, regime=TrendRegime.BULL)
    assert scalar == 2.0


def test_position_scalar_bounds_direction_without_erasing_vol_scale():
    scalar = compute_position_scalar(5.0, daily_std_last=0.0001, vol_target=0.50, regime=TrendRegime.BULL)
    assert scalar == 2.0


def test_position_scalar_vol_scalar_clamp_floor():
    # very high realized vol should clamp vol_scalar to the 0.25 floor, not go to ~0
    scalar = compute_position_scalar(1.0, daily_std_last=1.0, vol_target=0.15, regime=TrendRegime.BULL)
    assert math.isclose(scalar, 0.25, rel_tol=1e-6)


def test_position_scalar_discount_applied_for_disagreement_regimes():
    bull = compute_position_scalar(0.5, 0.02, vol_target=0.15, regime=TrendRegime.BULL, regime_discount=0.5)
    correction = compute_position_scalar(0.5, 0.02, vol_target=0.15, regime=TrendRegime.CORRECTION, regime_discount=0.5)
    assert math.isclose(correction, bull * 0.5, rel_tol=1e-9)


def test_position_scalar_discount_disabled_at_1():
    correction = compute_position_scalar(0.5, 0.02, vol_target=0.15, regime=TrendRegime.CORRECTION, regime_discount=1.0)
    bull = compute_position_scalar(0.5, 0.02, vol_target=0.15, regime=TrendRegime.BULL, regime_discount=1.0)
    assert math.isclose(correction, bull, rel_tol=1e-9)


def test_position_scalar_zero_on_null_trend_strength():
    assert compute_position_scalar(None, 0.01, vol_target=0.15, regime=TrendRegime.BULL) == 0.0
    assert compute_position_scalar(float('nan'), 0.01, vol_target=0.15, regime=TrendRegime.BULL) == 0.0


def test_position_scalar_neutral_vol_scalar_on_missing_std():
    # daily_std_last unusable -> vol_scalar treated as neutral (1.0), not a crash
    scalar = compute_position_scalar(0.4, None, vol_target=0.15, regime=TrendRegime.BULL)
    assert math.isclose(scalar, 0.4, rel_tol=1e-9)


# ── compute_n_effective ──────────────────────────────────────────────────────

def test_n_effective_counts_distinct_clusters():
    assert compute_n_effective({'equity', 'energy', 'grain', 'metal', 'fx'}) == 5
    assert compute_n_effective(set()) == 0
    assert compute_n_effective({'grain'}) == 1


# ── compute_desired_risk_budget ──────────────────────────────────────────────

def test_desired_risk_budget_scales_with_equity_and_vol():
    budget = compute_desired_risk_budget(account_equity=100_000, target_portfolio_vol=0.15, n_effective=5)
    assert math.isclose(budget, 100_000 * 0.15 / math.sqrt(5), rel_tol=1e-9)


def test_desired_risk_budget_zero_n_effective_is_zero_not_error():
    assert compute_desired_risk_budget(100_000, 0.15, 0) == 0.0


def test_desired_risk_budget_fewer_active_clusters_means_bigger_budget_each():
    # sqrt(N) scaling -- fewer active bets means each can be sized bigger,
    # without blowing through the total target_portfolio_vol.
    budget_1_cluster = compute_desired_risk_budget(100_000, 0.15, 1)
    budget_5_clusters = compute_desired_risk_budget(100_000, 0.15, 5)
    assert budget_1_cluster > budget_5_clusters


# ── compute_erc_weights / compute_hrp_weights ────────────────────────────────
# Data-driven alternatives to a flat 1/n notional split -- both solve on the
# same correlation matrix compute_idm already builds, so a cluster of
# correlated instruments collectively earns roughly one undiversified bet's
# worth of budget instead of each member claiming an equal individual share.

_SYMBOLS = ['A', 'B', 'C']
# H is these functions' sole input (see compute_erc_weights' own
# docstring) -- built directly, no dict intermediary.
_CORRELATED_PAIR_H = np.array([
    [1.0, 0.9, 0.0],
    [0.9, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
_EQUAL_CORR_H = np.array([
    [1.0, 0.3, 0.3],
    [0.3, 1.0, 0.3],
    [0.3, 0.3, 1.0],
])


@pytest.mark.parametrize('weight_fn', [compute_erc_weights, compute_hrp_weights])
def test_weights_sum_to_one(weight_fn):
    w = weight_fn(_SYMBOLS, _CORRELATED_PAIR_H)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-6)


@pytest.mark.parametrize('weight_fn', [compute_erc_weights, compute_hrp_weights])
def test_weights_down_weight_correlated_pair_relative_to_diversifier(weight_fn):
    # A/B move together (corr 0.9); C is uncorrelated with either -- C
    # should get a bigger individual share than A or B get individually,
    # since A and B are collectively closer to one bet than two.
    w = weight_fn(_SYMBOLS, _CORRELATED_PAIR_H)
    assert w['C'] > w['A']
    assert w['C'] > w['B']


def test_erc_weights_equal_under_uniform_correlation():
    # ERC is exactly symmetric under a fully symmetric correlation
    # structure (every pair equally correlated) -- unlike HRP (below),
    # which isn't guaranteed tie-free at its clustering step.
    w = compute_erc_weights(_SYMBOLS, _EQUAL_CORR_H)
    assert w['A'] == pytest.approx(1 / 3, abs=1e-6)
    assert w['B'] == pytest.approx(1 / 3, abs=1e-6)
    assert w['C'] == pytest.approx(1 / 3, abs=1e-6)


def test_hrp_weights_sum_to_one_under_uniform_correlation():
    # HRP's clustering step still has to pick some pair to merge first even
    # when every pairwise correlation is tied -- it isn't guaranteed to
    # reproduce ERC's exact 1/3-each symmetry here, just a valid, properly
    # normalized split.
    w = compute_hrp_weights(_SYMBOLS, _EQUAL_CORR_H)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-6)
    assert all(v > 0 for v in w.values())


@pytest.mark.parametrize('weight_fn', [compute_erc_weights, compute_hrp_weights])
def test_weights_fall_back_to_flat_when_h_missing(weight_fn):
    assert weight_fn(_SYMBOLS, None) == {s: pytest.approx(1 / 3) for s in _SYMBOLS}


@pytest.mark.parametrize('weight_fn', [compute_erc_weights, compute_hrp_weights])
def test_weights_fall_back_to_flat_under_two_symbols(weight_fn):
    assert weight_fn(['A'], None) == {'A': 1.0}
    assert weight_fn([], None) == {}


def _random_correlation_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    cov = A @ A.T
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def _risk_contributions(w: np.ndarray, H: np.ndarray) -> np.ndarray:
    return w * (H @ w) / np.sqrt(w @ H @ w)


def _slsqp_erc_oracle(H: np.ndarray) -> np.ndarray:
    """Fresh, independent ERC solve via generic scipy.optimize.minimize/
    SLSQP -- the textbook simplex-search formulation, NOT the retired
    compute_erc_weights implementation -- used only as a cross-check
    oracle for _spinu_erc_newton's convex reformulation."""
    from scipy.optimize import minimize

    n = H.shape[0]

    def objective(w):
        rc = _risk_contributions(w, H)
        return np.sum((rc - rc.mean()) ** 2)

    result = minimize(objective, np.full(n, 1.0 / n), method='SLSQP',
                      bounds=[(1e-6, 1.0)] * n,
                      constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}],
                      options={'maxiter': 1000, 'ftol': 1e-12})
    assert result.success
    return result.x


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_spinu_newton_matches_independent_slsqp_oracle_on_risk_contributions(seed):
    # The thing that actually defines "solved ERC" is equal risk
    # contribution, not any particular weight vector -- two different
    # correlation matrices could in principle have close-but-not-identical
    # weight solutions that both satisfy RC_i ~ equal, so compare RC
    # directly, not just weights (weights are also checked, as a bonus).
    n = 8
    H = _random_correlation_matrix(n, seed)
    b = np.full(n, 1.0 / n)

    w_newton = _spinu_erc_newton(H, b)
    w_slsqp = _slsqp_erc_oracle(H)

    assert w_newton is not None
    np.testing.assert_allclose(w_newton, w_slsqp, atol=1e-5)

    rc_newton = _risk_contributions(w_newton, H)
    rc_slsqp = _risk_contributions(w_slsqp, H)
    np.testing.assert_allclose(rc_newton, rc_slsqp, atol=1e-5)
    # Equal risk contribution: every asset's RC should be ~1/n of total risk.
    sigma_p = math.sqrt(w_newton @ H @ w_newton)
    np.testing.assert_allclose(rc_newton, sigma_p / n, atol=1e-5)


def test_spinu_newton_converges_across_many_matrices():
    # Regression test for a real bug found while tuning ERC_NEWTON_TOL:
    # an initial tolerance of 1e-10 (then 1e-9) caused _spinu_erc_newton
    # to spuriously return None (non-convergence) on a meaningful fraction
    # of random well-conditioned matrices, because Newton's own quadratic
    # convergence hits a floating-point noise floor before the gradient
    # ever gets that tight -- not a correctness issue, but a tolerance set
    # beyond what double-precision arithmetic can reliably achieve. Covers
    # both ordinary random correlation matrices AND adversarial near-
    # singular ones (a tight cluster of near +-1 correlations), which
    # additionally exercises the ERC_NEWTON_STALL_TOL backtracking-stall
    # path.
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 8, 12, 20, 30):
        for seed in range(20):
            H = _random_correlation_matrix(n, seed * 1000 + n)
            b = np.full(n, 1.0 / n)
            w = _spinu_erc_newton(H, b)
            assert w is not None, f'failed to converge: n={n} seed={seed}'
            assert w.sum() == pytest.approx(1.0, abs=1e-6)
            assert np.all(w > 0)

    for n in (2, 5, 12, 20):
        for _ in range(20):
            base_corr = rng.uniform(0.90, 0.999)
            H = np.full((n, n), base_corr)
            noise = rng.normal(0, 0.01, (n, n))
            noise = (noise + noise.T) / 2
            np.fill_diagonal(noise, 0.0)
            H = np.clip(H + noise, -0.999, 0.999)
            np.fill_diagonal(H, 1.0)
            eigvals, eigvecs = np.linalg.eigh(H)
            H = eigvecs @ np.diag(np.clip(eigvals, 1e-6, None)) @ eigvecs.T
            d = np.sqrt(np.diag(H))
            H = H / np.outer(d, d)

            b = np.full(n, 1.0 / n)
            w = _spinu_erc_newton(H, b)
            assert w is not None, f'failed to converge: n={n} cond(H)={np.linalg.cond(H):.2e}'
            rc = _risk_contributions(w, H)
            assert np.allclose(rc, rc.mean(), atol=1e-3)


def test_spinu_newton_solves_non_equal_risk_budgets():
    # compute_erc_weights itself always calls _spinu_erc_newton with a
    # uniform b = 1/n, so its own tests can't distinguish "this solver
    # genuinely solves risk budgeting" from "it coincidentally works
    # because b happens to be uniform" -- exercise the solver directly
    # with a non-uniform budget and confirm RC_i is proportional to b_i,
    # not just equal, which is the actual mathematical claim behind
    # Spinu's reformulation.
    n = 6
    H = _random_correlation_matrix(n, seed=42)
    b = np.array([0.05, 0.10, 0.15, 0.20, 0.20, 0.30])
    assert b.sum() == pytest.approx(1.0)

    w = _spinu_erc_newton(H, b)
    assert w is not None

    rc = _risk_contributions(w, H)
    sigma_p = math.sqrt(w @ H @ w)
    np.testing.assert_allclose(rc, b * sigma_p, atol=1e-6)


# ── compute_symbol_notional_budget (notional_weighting) ──────────────────────

def _corr_price_data(n: int = 300, seed: int = 0) -> dict[str, pl.DataFrame]:
    """Three symbols on a synced date range: A and B share the SAME
    underlying random walk (near-perfectly correlated, plus small
    idiosyncratic noise so they aren't numerically identical), C is an
    independent random walk with the same drift/vol -- the minimal
    "correlated cluster vs. lone diversifier" scenario compute_erc_weights/
    compute_hrp_weights are meant to detect."""
    rng = np.random.default_rng(seed)
    dates = [date(2018, 1, 1) + timedelta(days=i) for i in range(n)]
    shared_rets = rng.normal(0.0005, 0.01, n)
    idio_a = rng.normal(0.0, 0.0005, n)
    idio_b = rng.normal(0.0, 0.0005, n)
    indep_rets = rng.normal(0.0005, 0.01, n)

    def _df(rets):
        return pl.DataFrame({'ts_event': dates, 'close': 100 * np.exp(np.cumsum(rets))})

    return {
        'A': _df(shared_rets + idio_a),
        'B': _df(shared_rets + idio_b),
        'C': _df(indep_rets),
    }


def _notional_budget(notional_weighting: str, use_idm: bool = True) -> dict[str, float]:
    price_data = _corr_price_data()
    returns_wide = build_returns_wide(price_data)
    as_of = price_data['A']['ts_event'][-1]
    return compute_symbol_notional_budget(
        ['A', 'B', 'C'], returns_wide, as_of, capital=100_000, target_portfolio_vol=0.15,
        vol_target=0.15, corr_window_years=3.0, corr_halflife_days=63.0,
        notional_weighting=notional_weighting, use_idm=use_idm,
    )


def test_notional_budget_flat_gives_every_symbol_the_same_budget():
    budget = _notional_budget('flat')
    assert budget['A'] == pytest.approx(budget['B'])
    assert budget['A'] == pytest.approx(budget['C'])


@pytest.mark.parametrize('notional_weighting', ['erc', 'hrp'])
def test_notional_budget_data_driven_schemes_favor_independent_symbol(notional_weighting):
    # Same intuition as compute_erc_weights/compute_hrp_weights' own tests
    # above, now through the full compute_symbol_notional_budget path
    # (bounded EWM correlation matrix -> weighting -> dollar-vol split ->
    # notional_budget): C (uncorrelated with A/B) should end up with a
    # bigger individual budget than A or B under either data-driven scheme.
    budget = _notional_budget(notional_weighting)
    assert budget['C'] > budget['A']
    assert budget['C'] > budget['B']


@pytest.mark.parametrize('notional_weighting', ['erc', 'hrp'])
def test_notional_budget_idm_uses_the_same_split_as_weights(notional_weighting):
    # Regression test for the weight-consistency fix: compute_idm must be
    # called with weights=<the actual notional_weighting split>, not a flat
    # 1/n vector regardless of notional_weighting -- otherwise the total
    # budget is sized for a flat split that never actually gets used,
    # double-counting diversification once via IDM (assuming flat) and
    # again via erc/hrp (reshaping the split). Reconstruct the expected
    # total independently (split first, then IDM on that same split) and
    # confirm compute_symbol_notional_budget's actual output matches it --
    # not the flat-weights IDM value.
    price_data = _corr_price_data()
    returns_wide = build_returns_wide(price_data)
    as_of = price_data['A']['ts_event'][-1]
    symbols = ['A', 'B', 'C']
    H, covered = _bounded_ewm_correlation_matrix(returns_wide, symbols, as_of, 3.0, 63.0)
    assert covered.all()
    weight_fn = compute_erc_weights if notional_weighting == 'erc' else compute_hrp_weights
    split = weight_fn(symbols, H)

    idm_consistent = compute_idm(symbols, H, weights=split)
    idm_flat = compute_idm(symbols, H)
    assert idm_consistent != pytest.approx(idm_flat), \
        "test fixture must produce a non-flat split for this regression check to be meaningful"

    budget = _notional_budget(notional_weighting)
    expected_total = 100_000 * 0.15 * idm_consistent
    actual_total = sum(budget.values()) * 0.15
    assert actual_total == pytest.approx(expected_total, rel=1e-9)


def test_notional_budget_use_idm_false_skips_the_multiplier():
    budget_flat = _notional_budget('flat', use_idm=False)
    # With no IDM adjustment, the total dollar-vol budget is exactly
    # capital * target_portfolio_vol, split flat -- no correlation-based
    # up/down-sizing at all.
    assert sum(budget_flat.values()) * 0.15 == pytest.approx(100_000 * 0.15, rel=1e-9)

    budget_erc = _notional_budget('erc', use_idm=False)
    weight_fn_total = sum(budget_erc.values()) * 0.15
    assert weight_fn_total == pytest.approx(100_000 * 0.15, rel=1e-9)
    # The split itself still favors the independent symbol even with IDM off.
    assert budget_erc['C'] > budget_erc['A']


def test_notional_budget_empty_active_symbols_returns_empty_dict():
    price_data = _corr_price_data()
    returns_wide = build_returns_wide(price_data)
    as_of = price_data['A']['ts_event'][-1]
    assert compute_symbol_notional_budget([], returns_wide, as_of, 100_000, 0.15, 0.15, 3.0, 63.0) == {}
    assert compute_symbol_notional_budget(['A'], None, as_of, 100_000, 0.15, 0.15, 3.0, 63.0) == {}


def test_notional_budget_works_with_precomputed_h_and_no_returns_wide():
    # Regression test: a caller that already ran _bounded_ewm_correlation_
    # matrix itself (e.g. the live rebalance report) shouldn't need
    # returns_wide at all -- that's the documented point of the H/covered
    # params. The early-return guard used to check `returns_wide is None`
    # unconditionally, silently discarding a caller-supplied H and
    # returning {} even when H made returns_wide unnecessary.
    symbols = ['A', 'B', 'C']
    H = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    budget = compute_symbol_notional_budget(
        symbols, None, date(2024, 1, 1), 100_000, 0.15, 0.15, 3.0, 63.0,
        notional_weighting='erc', H=H,
    )
    assert budget != {}
    assert sum(budget.values()) > 0
    # ERC split still favors the independent symbol C over the correlated A/B.
    assert budget['C'] > budget['A']


def test_notional_budget_rejects_unknown_weighting_scheme():
    price_data = _corr_price_data()
    returns_wide = build_returns_wide(price_data)
    as_of = price_data['A']['ts_event'][-1]
    with pytest.raises(ValueError):
        compute_symbol_notional_budget(['A', 'B', 'C'], returns_wide, as_of, 100_000, 0.15, 0.15,
                                        3.0, 63.0, 'bogus')


def test_bounded_ewm_correlation_matches_independent_pandas_ewm():
    # Independent oracle for _bounded_ewm_correlation_matrix's own Gram-
    # matrix construction: pandas' .ewm(halflife=..., adjust=True).cov(),
    # a completely separate implementation of the same "EWM correlation
    # evaluated at the last row" definition. Pandas is scoped to just this
    # one test call site, per this project's CLAUDE.md convention -- not a
    # general dependency.
    import pandas as pd

    price_data = _corr_price_data()
    returns_wide = build_returns_wide(price_data)
    symbols = ['A', 'B', 'C']
    as_of = price_data['A']['ts_event'][-1]

    H, covered = _bounded_ewm_correlation_matrix(returns_wide, symbols, as_of, 3.0, 63.0)
    assert covered.all()

    window_start = as_of - timedelta(days=int(3.0 * 365.25))
    sl = returns_wide.filter((pl.col('ts_event') >= window_start) & (pl.col('ts_event') < as_of))
    pdf = sl.select(symbols).to_pandas()
    cov = pdf.ewm(halflife=63.0, adjust=True).cov().iloc[-len(symbols):]
    cov.index = cov.index.droplevel(0)
    std = np.sqrt(np.diag(cov))
    expected = (cov.to_numpy() / np.outer(std, std))

    np.testing.assert_allclose(H, expected, atol=1e-9)


# ── uncovered-symbol handling (missing correlation data != measured independence) ─

def test_bounded_ewm_correlation_covered_mask_flags_missing_symbol():
    # NEWSYM has no column in returns_wide at all (e.g. a failed price
    # fetch, or an instrument too new to have synchronized data yet) --
    # `covered` must flag it False even though H still returns a usable
    # (identity-placeholder) row/column for it.
    price_data = _corr_price_data()
    returns_wide = build_returns_wide(price_data)
    as_of = price_data['A']['ts_event'][-1]
    symbols = ['A', 'B', 'C', 'NEWSYM']

    H, covered = _bounded_ewm_correlation_matrix(returns_wide, symbols, as_of, 3.0, 63.0)

    assert covered.tolist() == [True, True, True, False]
    assert H[3, 3] == 1.0
    assert H[3, 0] == 0.0 and H[0, 3] == 0.0


def test_coverage_restricted_idm_ignores_uncovered_symbol():
    # Regression test for the bug this session found: without covered-
    # awareness, an uncovered symbol's identity row (0.0 correlation to
    # everything) gets read as MEASURED independence, inflating IDM.
    # _coverage_restricted_idm must produce the exact same IDM as running
    # the calculation on the covered symbols alone -- NEWSYM contributes
    # nothing, not even at a small weight.
    symbols = ['A', 'B', 'NEWSYM']
    H = np.array([
        [1.0, 0.998, 0.0],
        [0.998, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    covered = np.array([True, True, False])

    idm_with_newsym = _coverage_restricted_idm(symbols, H, covered)
    idm_covered_only = compute_idm(['A', 'B'], H[:2, :2])

    assert idm_with_newsym == pytest.approx(idm_covered_only, abs=1e-9)
    # Sanity check on the bug itself: naively passing the full H/weights
    # through (the old behavior) DOES inflate IDM relative to this.
    idm_naive = compute_idm(symbols, H)
    assert idm_naive > idm_with_newsym


def test_notional_split_caps_uncovered_symbol_budget():
    # Same regression, at the notional-split level: NEWSYM must not out-
    # earn every real, measured symbol just because it has no data.
    symbols = ['A', 'B', 'C', 'NEWSYM']
    H = np.eye(4)
    H[:3, :3] = _CORRELATED_PAIR_H
    covered = np.array([True, True, True, False])

    split = compute_notional_split(symbols, 'erc', H, covered)

    assert sum(split.values()) == pytest.approx(1.0, abs=1e-9)
    # |uncovered|/n = 1/4 = 0.25 exceeds the cap, so NEWSYM is held at
    # exactly UNCOVERED_BUDGET_CAP_FRACTION, not a proportional 25% share.
    assert split['NEWSYM'] == pytest.approx(UNCOVERED_BUDGET_CAP_FRACTION, abs=1e-9)
    assert split['NEWSYM'] < split['A']
    assert split['NEWSYM'] < split['B']
    assert split['NEWSYM'] < split['C']
    # Covered symbols still show the ordinary ERC qualitative behavior
    # (the independent diversifier C outweighs the correlated A/B pair)
    # among themselves, just scaled down to fit the remaining budget.
    assert split['C'] > split['A']
    assert split['C'] > split['B']


def test_notional_budget_caps_uncovered_symbol_end_to_end():
    # Full compute_symbol_notional_budget path: NEWSYM has a live signal
    # (it's in active_symbols) but no price data at all (absent from
    # price_data, so absent from returns_wide).
    price_data = _corr_price_data()
    returns_wide = build_returns_wide(price_data)
    as_of = price_data['A']['ts_event'][-1]
    symbols = ['A', 'B', 'C', 'NEWSYM']

    budget = compute_symbol_notional_budget(
        symbols, returns_wide, as_of, capital=100_000, target_portfolio_vol=0.15,
        vol_target=0.15, corr_window_years=3.0, corr_halflife_days=63.0,
        notional_weighting='erc', use_idm=True,
    )

    total = sum(budget.values())
    assert budget['NEWSYM'] / total == pytest.approx(UNCOVERED_BUDGET_CAP_FRACTION, abs=1e-6)
    assert budget['NEWSYM'] < budget['A']
    assert budget['NEWSYM'] < budget['B']
    assert budget['NEWSYM'] < budget['C']


# ── compute_realized_portfolio_risk ──────────────────────────────────────────

def test_realized_portfolio_risk_decomposes_exactly_to_port_vol():
    # Euler's theorem for a degree-1-homogeneous function: sum of marginal
    # risk contributions must equal port_vol EXACTLY, not approximately --
    # this is the actual mathematical claim behind "risk contribution",
    # not just a sanity check.
    symbols = ['A', 'B', 'C']
    H = _CORRELATED_PAIR_H
    position_risk = {'A': 10_000.0, 'B': 8_000.0, 'C': 12_000.0}

    result = compute_realized_portfolio_risk(symbols, H, position_risk)

    assert sum(result['portfolio_risk_contribution'].values()) == pytest.approx(result['port_vol'], rel=1e-9)


def test_realized_portfolio_risk_credits_diversifier_less_than_correlated_pair():
    # A/B correlated at 0.9, C uncorrelated with either, all three sized
    # to the SAME standalone (undiversified) dollar risk -- despite that
    # equal standalone sizing, C's actual risk contribution comes in well
    # below A/B's, and below its own standalone position_risk, since it's
    # genuinely diversifying. (An individual RC exceeding its own
    # position_risk is possible in general -- e.g. a small position
    # heavily correlated with a large cluster -- but not in this
    # symmetric equal-weight case: A/B's RC approaches, but stays below,
    # their own standalone risk as correlation -> 1, confirmed directly
    # rather than assumed.)
    symbols = ['A', 'B', 'C']
    H = _CORRELATED_PAIR_H
    position_risk = {'A': 10_000.0, 'B': 10_000.0, 'C': 10_000.0}

    result = compute_realized_portfolio_risk(symbols, H, position_risk)
    rc = result['portfolio_risk_contribution']

    assert rc['C'] < position_risk['C']
    assert rc['A'] == pytest.approx(rc['B'])
    assert rc['C'] < rc['A']


def test_realized_portfolio_risk_zero_positions_gives_zero_vol():
    symbols = ['A', 'B', 'C']
    result = compute_realized_portfolio_risk(symbols, _CORRELATED_PAIR_H, {})
    assert result['port_vol'] == 0.0
    assert all(v == 0.0 for v in result['portfolio_risk_contribution'].values())


def test_realized_portfolio_risk_missing_symbol_defaults_to_no_position():
    # A symbol absent from position_risk (e.g. it got zero contracts this
    # rebalance) contributes nothing, same as an explicit 0.0.
    symbols = ['A', 'B', 'C']
    result_missing = compute_realized_portfolio_risk(symbols, _CORRELATED_PAIR_H, {'A': 10_000.0, 'B': 8_000.0})
    result_explicit = compute_realized_portfolio_risk(symbols, _CORRELATED_PAIR_H,
                                                        {'A': 10_000.0, 'B': 8_000.0, 'C': 0.0})
    assert result_missing == result_explicit


def test_realized_portfolio_risk_short_nets_against_correlated_long():
    # A/B correlated at 0.9. Same-signed (both long) -- correlation
    # compounds, port_var = wA^2 + wB^2 + 2*wA*wB*rho. Opposite-signed
    # (long A, short B) -- correlation nets, port_var = wA^2 + wB^2 -
    # 2*wA*wB*rho. This is the whole point of feeding SIGNED exposure
    # into w' H w rather than the unsigned position_risk magnitude: a
    # short position that's genuinely hedging a correlated long must
    # produce a materially SMALLER port_vol, not the same one abs(w)
    # would give regardless of direction.
    symbols = ['A', 'B', 'C']
    same_signed = compute_realized_portfolio_risk(symbols, _CORRELATED_PAIR_H, {'A': 10_000.0, 'B': 10_000.0})
    opposite_signed = compute_realized_portfolio_risk(symbols, _CORRELATED_PAIR_H, {'A': 10_000.0, 'B': -10_000.0})

    assert opposite_signed['port_vol'] < same_signed['port_vol']


def test_realized_portfolio_risk_small_hedge_gets_negative_portfolio_risk_contribution():
    # A big long (A) plus a SMALL short (B) in something correlated at 0.9:
    # increasing B's short further, at the margin, still reduces total
    # portfolio variance (B hasn't "used up" its hedging capacity against
    # A's much larger size yet) -- Euler decomposition assigns that a
    # genuine negative portfolio risk contribution. This is the direction-driven
    # negative RC compute_realized_portfolio_risk is meant to surface;
    # it requires SIGNED exposure (an unsigned magnitude can never
    # produce it here, since B is the smaller position -- with unsigned
    # w every marginal contribution in this symmetric-sign H is
    # non-negative).
    symbols = ['A', 'B', 'C']
    result = compute_realized_portfolio_risk(symbols, _CORRELATED_PAIR_H, {'A': 10_000.0, 'B': -2_000.0})

    assert result['portfolio_risk_contribution']['B'] < 0
    assert sum(result['portfolio_risk_contribution'].values()) == pytest.approx(result['port_vol'], rel=1e-9)


# ── apply_cluster_risk_cap ───────────────────────────────────────────────────
#
# The cap is taken against a FIXED total_risk_target (account_equity *
# target_portfolio_vol, supplied by the caller) -- never the emergent sum
# of this run's pre-cap position risks. effective_cap_pct = max(
# max_cluster_risk_pct, 1/n_active_clusters), so a single active cluster
# isn't capped below 100% of the target just for being the only trade in
# the book.
#
# Within an over-budget cluster, allocation is GREEDY BY CONVICTION
# PRIORITY (priority = abs(combined_scalar), descending), not a uniform haircut --
# a uniform scale factor can push every instrument below the 0.5 rounding
# threshold at once, even when the top-conviction instrument alone would
# easily survive on the full cap. A bounded lot-size exception (gated to
# the first instrument only) still grants exactly 1 contract when the true
# continuous math would round to zero but the instrument's own signal
# genuinely wants a full contract and its single-contract risk isn't
# wildly over the cap.
#
# infeasible is OUTCOME-based: True only when every instrument in a
# cluster ends at final_target_contracts == 0 despite at least one having a
# genuine (>=0.5) pre-cap signal -- not a cap-vs-single-contract-risk
# precomputation, since the top-priority instrument may still land a
# contract via the lot exception even when that precomputed check would
# have said "infeasible".

def _target(symbol, cluster, continuous_contracts, close=100.0, multiplier=10.0, hv=0.2,
            max_contracts=None, target_contracts=None, scalar=0.0):
    return {
        'symbol': symbol, 'cluster': cluster,
        'fractional_target_contracts': continuous_contracts,
        'combined_scalar': scalar,
        # final_target_contracts simulates whatever the caller
        # (tsmom_rebalance.py) already computed upstream; rounding and
        # standalone_position_dollar_vol
        # always overwrite it (see apply_cluster_risk_cap's own `apply_cap`
        # docstring: even with the cap fully disabled, only the cluster-level
        # redistribution is skipped, not rounding), so a sentinel value here
        # is only meaningful in a test that explicitly checks it gets
        # overwritten, not one relying on it staying untouched.
        'final_target_contracts': (
            target_contracts if target_contracts is not None else round(continuous_contracts)
        ),
        'close': close, 'mult': multiplier, 'hv': hv, 'max_contracts': max_contracts,
    }


def test_lot_aware_allocator_prefers_feasible_cluster_representative():
    # A's one-lot risk fits the account target while B's does not.  Both
    # belong to the same cluster, so the allocator should represent the
    # cluster with A rather than independently rounding either signal.
    targets = [
        _target('A', 'equity', 0.33, close=100, multiplier=200, hv=0.20),  # one-lot risk = 4,000
        _target('B', 'equity', -0.16, close=100, multiplier=500, hv=0.20),  # one-lot risk = 10,000
    ]
    out = allocate_lot_aware_targets(
        targets,
        portfolio_risk_target=5_000,
        H=np.eye(2),
        active_symbols=['A', 'B'],
    )

    assert out[0]['final_target_contracts'] == 1
    assert out[1]['final_target_contracts'] == 0
    assert out[1]['integer_zero_reason'] == 'integer_risk_limit'
    assert out[0]['standalone_position_dollar_vol'] == pytest.approx(4_000)


def test_lot_aware_allocator_preserves_direction_and_hard_limit():
    targets = [
        _target('LONG', 'equity', 1.2, close=100, multiplier=100, hv=0.10),
        _target('SHORT', 'rates', -1.2, close=100, multiplier=100, hv=0.10),
    ]
    out = allocate_lot_aware_targets(
        targets,
        portfolio_risk_target=15,
        H=np.eye(2),
        active_symbols=['LONG', 'SHORT'],
    )

    assert out[0]['final_target_contracts'] >= 0
    assert out[1]['final_target_contracts'] <= 0
    realized = math.sqrt(sum(t['standalone_position_dollar_vol'] ** 2 for t in out))
    assert realized <= 15 + 1e-9


@pytest.mark.parametrize('n_active_clusters,expected_pct', [
    (1, 1.0), (2, 0.5), (3, 1 / 3), (4, 0.25), (5, 0.25), (6, 0.25),
])
def test_cluster_cap_floor_table(n_active_clusters, expected_pct):
    # max_cluster_risk_pct=0.25 throughout -- the floor (1/n) only matters
    # while it's bigger than 0.25, i.e. n <= 4. Single-instrument cluster,
    # so greedy allocation reduces to the same math as a plain cap check.
    targets = [_target('MES', 'equity', continuous_contracts=10, close=100, multiplier=1, hv=0.1, scalar=0.9)]  # risk=100
    total_risk_target = 1000.0
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=total_risk_target, n_active_clusters=n_active_clusters)
    assert out[0]['final_target_contracts'] == 10

    big_targets = [_target('MES', 'equity', continuous_contracts=100, close=100, multiplier=1, hv=0.1, scalar=0.9)]  # risk=1000
    out_big = apply_cluster_risk_cap(big_targets, max_cluster_risk_pct=0.25,
                                     total_risk_target=total_risk_target, n_active_clusters=n_active_clusters)
    expected_risk = expected_pct * total_risk_target
    assert math.isclose(out_big[0]['standalone_position_dollar_vol'], expected_risk, rel_tol=0.05)


def test_cluster_cap_fixed_denominator_not_inflated_by_correlated_cluster():
    # Four grain instruments that would, under the old (buggy) emergent-sum
    # denominator, have summed to 60% of total_risk -- the bug was that
    # this sum itself became the denominator the cap was measured against.
    # Under the fixed total_risk_target, the cap is compared against the
    # account-level target instead, independent of how much risk this
    # cluster happens to carry this run. (Tied scalars here -- this test
    # is about the fixed denominator, not priority order; see the dedicated
    # walk-down tests for priority behavior.)
    targets = [
        _target('MES', 'equity', continuous_contracts=4, close=100, multiplier=1, hv=0.1, scalar=0.5),
        _target('MZC', 'grain', continuous_contracts=15, close=100, multiplier=1, hv=0.1, scalar=0.5),
        _target('MZS', 'grain', continuous_contracts=15, close=100, multiplier=1, hv=0.1, scalar=0.5),
        _target('MZW', 'grain', continuous_contracts=15, close=100, multiplier=1, hv=0.1, scalar=0.5),
        _target('MZL', 'grain', continuous_contracts=15, close=100, multiplier=1, hv=0.1, scalar=0.5),
    ]
    total_risk_target = 400.0
    n_active_clusters = 2  # equity, grain
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=total_risk_target, n_active_clusters=n_active_clusters)
    by_symbol = {t['symbol']: t for t in out}

    grain_risk = sum(
        by_symbol[s]['standalone_position_dollar_vol'] for s in ('MZC', 'MZS', 'MZW', 'MZL')
    )
    cap = max(0.25, 1 / n_active_clusters) * total_risk_target  # = 200
    assert grain_risk <= cap * 1.1
    assert grain_risk < 600  # rescaled down from the 600 pre-cap sum


def test_cluster_cap_leaves_single_instrument_cluster_dominant_clusters_alone():
    # Equity and energy are each a single-instrument cluster -- with
    # nothing else in their own cluster to share risk with, they aren't
    # clipped just for being individually large relative to other
    # clusters' instrument counts.
    targets = [
        _target('MES', 'equity', continuous_contracts=3, close=100, multiplier=10, hv=0.1, scalar=0.5),  # risk=300
        _target('MCL', 'energy', continuous_contracts=2, close=100, multiplier=10, hv=0.1, scalar=0.5),  # risk=200
        _target('MGC', 'metal', continuous_contracts=1, close=100, multiplier=10, hv=0.1, scalar=0.5),   # risk=100
    ]
    total_risk_target = 600.0
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=total_risk_target, n_active_clusters=3)
    by_symbol = {t['symbol']: t for t in out}
    assert abs(by_symbol['MES']['final_target_contracts']) < 3
    assert by_symbol['MGC']['final_target_contracts'] == 1  # 100 < 200 cap, untouched


def test_cluster_cap_no_op_when_all_clusters_within_budget():
    targets = [
        _target('MES', 'equity', continuous_contracts=1, close=100, multiplier=10, hv=0.1, scalar=0.5),
        _target('MCL', 'energy', continuous_contracts=1, close=100, multiplier=10, hv=0.1, scalar=0.5),
        _target('MGC', 'metal', continuous_contracts=1, close=100, multiplier=10, hv=0.1, scalar=0.5),
        _target('J7', 'fx', continuous_contracts=1, close=100, multiplier=10, hv=0.1, scalar=0.5),
    ]
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=100_000.0, n_active_clusters=4)
    assert all(t['final_target_contracts'] == 1 for t in out)


def test_cluster_cap_within_budget_cluster_untouched():
    # Explicit, isolated within-budget case: continuous=1.3 with a cap so
    # large the cluster never needs the walk-down -- rounds directly,
    # unaffected by any cluster-cap mechanics.
    targets = [_target('A', 'c', continuous_contracts=1.3, close=100, multiplier=1, hv=1.0, scalar=0.9)]
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=1_000_000.0, n_active_clusters=1)
    assert out[0]['final_target_contracts'] == 1
    assert not out[0].get('infeasible')


def test_cluster_cap_skips_error_targets():
    targets = [
        _target('MES', 'equity', continuous_contracts=1, close=100, multiplier=10, hv=0.1, scalar=0.5),
        {'symbol': 'MCL', 'error': 'boom', 'final_target_contracts': None},
    ]
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=1000.0, n_active_clusters=1)
    errored = next(t for t in out if t['symbol'] == 'MCL')
    assert errored['error'] == 'boom'
    assert errored['final_target_contracts'] is None


def test_cluster_cap_zero_total_risk_target_skips_cap_not_rounding():
    # A None/zero total_risk_target disables the CAP, not rounding -- use a
    # sentinel final_target_contracts that would only survive if the function
    # genuinely didn't touch it, to prove it's NOT a true no-op: rounding
    # and standalone position dollar-vol still get recomputed from the
    # fractional target contracts.
    targets = [_target('MES', 'equity', continuous_contracts=5.3, target_contracts=999,
                       close=100, multiplier=10, hv=0.1)]
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=0.0, n_active_clusters=1)
    assert out[0]['final_target_contracts'] == 5
    assert out[0]['standalone_position_dollar_vol'] == pytest.approx(5 * 100 * 10 * 0.1)


def test_cluster_cap_none_total_risk_target_skips_cap_not_rounding():
    targets = [_target('MES', 'equity', continuous_contracts=5.3, target_contracts=999,
                       close=100, multiplier=10, hv=0.1)]
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=None, n_active_clusters=1)
    assert out[0]['final_target_contracts'] == 5
    assert out[0]['standalone_position_dollar_vol'] == pytest.approx(5 * 100 * 10 * 0.1)


def test_cluster_cap_apply_cap_false_skips_cap_not_rounding():
    # Same as the total_risk_target=None/0 cases, but via the explicit
    # apply_cap=False toggle with a genuinely valid, over-budget
    # total_risk_target -- proves apply_cap, not just total_risk_target's
    # own value, controls whether the cap fires. Two MES-cluster instruments
    # whose combined risk would exceed a tiny cap if the cap were active.
    targets = [
        _target('A', 'equity', continuous_contracts=10, target_contracts=999,
               close=100, multiplier=10, hv=0.1, scalar=0.9),   # risk=10,000
        _target('B', 'equity', continuous_contracts=10, target_contracts=999,
               close=100, multiplier=10, hv=0.1, scalar=0.5),   # risk=10,000
    ]
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.01, total_risk_target=100.0,
                                 n_active_clusters=1, apply_cap=False)
    # Would be aggressively capped/reprioritized if apply_cap were True
    # (cap = 0.01*100 = $1) -- instead both round their continuous value
    # directly, untouched by any cluster-level redistribution.
    assert out[0]['final_target_contracts'] == 10
    assert out[1]['final_target_contracts'] == 10
    assert not out[0].get('infeasible')
    assert not out[1].get('infeasible')


def test_group_by_cluster_sums_per_symbol_values():
    cluster_by_symbol = {'MES': 'equity', 'MNQ': 'equity', 'MCL': 'energy'}
    values = {'MES': 100.0, 'MNQ': 50.0, 'MCL': 30.0}
    assert group_by_cluster(cluster_by_symbol, values) == {'equity': 150.0, 'energy': 30.0}


def test_group_by_cluster_unknown_symbol_falls_back_to_other():
    cluster_by_symbol = {'MES': 'equity'}
    values = {'MES': 100.0, 'ZZZ': 25.0}
    assert group_by_cluster(cluster_by_symbol, values) == {'equity': 100.0, 'other': 25.0}


def test_cluster_cap_zero_n_active_clusters_falls_back_to_max_cluster_risk_pct():
    # n_active_clusters=0 shouldn't divide by zero -- falls back to the
    # flat max_cluster_risk_pct with no floor adjustment.
    targets = [_target('MES', 'equity', continuous_contracts=100, close=100, multiplier=1, hv=0.1, scalar=0.9)]  # risk=1000
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=1000.0, n_active_clusters=0)
    assert math.isclose(out[0]['standalone_position_dollar_vol'], 0.25 * 1000.0, rel_tol=0.05)


def test_cluster_cap_max_contracts_clamp_applied_after_cap_not_before():
    # max_contracts is applied as the TRUE last step (after the cluster-
    # cap allocation), not before. Single uncapped-cluster instrument:
    # continuous=10, no cluster pressure (cap is huge), so the final target
    # would be 10 without a clamp -- max_contracts=3 must still bring it
    # down to 3 at the very end.
    targets = [_target('MES', 'equity', continuous_contracts=10, close=100, multiplier=1, hv=0.1,
                       max_contracts=3, scalar=0.9)]
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=1_000_000.0, n_active_clusters=1)
    assert out[0]['final_target_contracts'] == 3


def test_cluster_cap_walk_down_three_instruments_partial_consumption():
    """3+ instrument cluster: the walk-down generalizes past 2 instruments
    -- #1 (highest priority) fully consumes its desired size, #2 gets
    whatever's left, #3 gets nothing once the budget is exhausted."""
    targets = [
        _target('A', 'c', continuous_contracts=2.0, close=100, multiplier=1, hv=3.0, scalar=0.9),  # single=300
        _target('B', 'c', continuous_contracts=1.0, close=100, multiplier=1, hv=3.0, scalar=0.6),  # single=300
        _target('C', 'c', continuous_contracts=1.0, close=100, multiplier=1, hv=3.0, scalar=0.3),  # single=300
    ]
    # cap = 1.0 (n_active=1 floor) * 1000 = 1000
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=1000.0, n_active_clusters=1)
    by_symbol = {t['symbol']: t for t in out}
    # A: affordable=1000/300=3.33, usable=min(2.0,3.33)=2.0 -> 2 contracts, remaining=1000-600=400
    assert by_symbol['A']['final_target_contracts'] == 2
    # B: affordable=400/300=1.33, usable=min(1.0,1.33)=1.0 -> 1 contract, remaining=400-300=100
    assert by_symbol['B']['final_target_contracts'] == 1
    # C: affordable=100/300=0.33<0.5, not first -> 0
    assert by_symbol['C']['final_target_contracts'] == 0
    assert not any(t.get('infeasible') for t in out)  # A got real exposure -- not infeasible


def test_cluster_cap_greedy_diverges_from_old_proportional_haircut():
    """Documents the known, accepted tradeoff of greedy allocation: a
    uniform proportional haircut (the previous algorithm) would have kept
    BOTH instruments here at 1 contract each (0.6875x scale applied to two
    already-integer continuous values, both staying >= 0.5). Greedy gives
    the top-priority instrument (A) everything it wants first, leaving B
    with too little to round to a nonzero contract. This is intentional --
    conviction-priority allocation deliberately sacrifices the lower-
    priority instrument rather than diluting both -- and should NOT be
    "fixed" later to restore the old proportional-survival behavior."""
    targets = [
        _target('A', 'c', continuous_contracts=1.0, close=100, multiplier=1, hv=4.0, scalar=0.6),  # single=400
        _target('B', 'c', continuous_contracts=1.0, close=100, multiplier=1, hv=4.0, scalar=0.5),  # single=400
    ]
    # cluster_risk = 400+400=800; cap=1.0*550=550 (n_active=1 floor).
    # Old proportional: scale=550/800=0.6875; A=1*0.6875=0.6875->1; B=1*0.6875=0.6875->1 (both survive).
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=550.0, n_active_clusters=1)
    by_symbol = {t['symbol']: t for t in out}
    # Greedy: A first, affordable=550/400=1.375, usable=1.0 -> 1, remaining=550-400=150.
    # B: affordable=150/400=0.375<0.5, not first -> 0. This is the accepted tradeoff: B would
    # have survived (1 contract) under the old uniform haircut but gets zero under greedy --
    # not a regression to "fix", a deliberate design choice (top conviction wins the budget).
    assert by_symbol['A']['final_target_contracts'] == 1
    assert by_symbol['B']['final_target_contracts'] == 0


@pytest.mark.parametrize('single_contract_risk,expect_exception', [
    (2499.0, True),   # just below cap * (1 + max_lot_overrun_pct) = 1000 * 2.5 = 2500
    (2501.0, False),  # just above
])
def test_cluster_cap_lot_overrun_boundary(single_contract_risk, expect_exception):
    """affordable_continuous itself falls under 0.5 here (1000/2499=0.40,
    1000/2501=0.40) -- isolating whether the exception's overrun-tolerance
    check (single_contract_risk <= cap * (1 + max_lot_overrun_pct)) is the
    deciding factor at its exact boundary."""
    targets = [_target('A', 'c', continuous_contracts=1.0, close=single_contract_risk,
                       multiplier=1, hv=1.0, scalar=0.9)]
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=1000.0, n_active_clusters=1,
                                 max_lot_overrun_pct=1.5)
    if expect_exception:
        assert out[0]['final_target_contracts'] == 1
    else:
        assert out[0]['final_target_contracts'] == 0
        assert out[0]['infeasible'] is True


def test_cluster_cap_lot_exception_never_applies_past_first_instrument():
    """B's own situation (in isolation) would qualify for the lot
    exception -- abs(original) >= 0.5, single_contract_risk within the
    overrun tolerance of the cap -- but B is evaluated second, after A has
    already spent part of the budget, so remaining_budget != cap and the
    exception must not fire for B."""
    targets = [
        _target('A', 'c', continuous_contracts=1.0, close=400, multiplier=1, hv=1.0, scalar=0.9),    # single=400
        _target('B', 'c', continuous_contracts=1.0, close=2500, multiplier=1, hv=1.0, scalar=0.5),   # single=2500
    ]
    # cap=1000 (n_active=1 floor). max_lot_overrun_pct=2.0 -> threshold=1000*3=3000.
    # B's single_contract_risk (2500) <= 3000, and B's own original (1.0) >= 0.5 -- B
    # WOULD qualify for the exception if it were first (remaining=1000: affordable=1000/2500=0.4<0.5).
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=1000.0, n_active_clusters=1,
                                 max_lot_overrun_pct=2.0)
    by_symbol = {t['symbol']: t for t in out}
    # A (first, scalar=0.9): affordable=1000/400=2.5, usable=min(1.0,2.5)=1.0 -> 1, remaining=600.
    assert by_symbol['A']['final_target_contracts'] == 1
    # B (second): affordable=600/2500=0.24<0.5, not first -> exception gated off -> 0,
    # even though B would have qualified had it been evaluated first.
    assert by_symbol['B']['final_target_contracts'] == 0


def test_cluster_cap_infeasible_when_cap_below_single_contract_risk():
    """Small synthetic equivalent of the $1,000,000/ES+NQ scenario: a
    cluster cap that's smaller than even the cheapest single contract's
    own dollar-vol risk. Outcome-based: the instrument ends at zero AND
    that's flagged infeasible, not because of a cap-vs-single-contract-
    risk precomputation alone."""
    targets = [_target('J7', 'fx', continuous_contracts=1.0, close=0.0066,
                       multiplier=6_250_000, hv=0.08, scalar=0.9)]
    # single_contract_risk = 0.0066*6_250_000*0.08 = 3300; cap=1.0*1000=1000 < 3300.
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=1000.0, n_active_clusters=1)
    assert out[0]['infeasible'] is True
    assert out[0]['final_target_contracts'] == 0  # scale-equivalent (1000/3300=0.303) * 1.0 < 0.5


def test_cluster_cap_not_infeasible_when_walk_down_captures_exposure():
    # Contrast case: even though the cluster needs rescaling, the top-
    # priority instrument still captures real exposure -- not infeasible.
    targets = [
        _target('MES', 'equity', continuous_contracts=5, close=100, multiplier=10, hv=0.1, scalar=0.9),
        _target('MNQ', 'equity', continuous_contracts=5, close=100, multiplier=10, hv=0.1, scalar=0.5),
    ]
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=600.0, n_active_clusters=1)  # cap=600
    assert not any(t.get('infeasible') for t in out)
    assert any(t['final_target_contracts'] != 0 for t in out)


def test_cluster_cap_mes_mnq_80k_scenario():
    """The exact $80K MES/MNQ scenario from this session's live run: MES
    (higher scalar/conviction) gets 1 contract; MNQ (lower scalar, and a
    much more expensive single contract) gets 0 -- the walk-down's top
    priority wins the cluster's budget instead of both being uniformly
    scaled to zero."""
    targets = [
        _target('MES', 'equity', continuous_contracts=1.0133, close=7472.75, multiplier=5, hv=0.1539, scalar=0.8168),
        _target('MNQ', 'equity', continuous_contracts=0.4137, close=30028.25, multiplier=2, hv=0.2381, scalar=0.5301),
    ]
    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=80_000 * 0.15, n_active_clusters=3)
    by_symbol = {t['symbol']: t for t in out}
    assert by_symbol['MES']['final_target_contracts'] == 1
    assert by_symbol['MNQ']['final_target_contracts'] == 0
    assert not by_symbol['MES'].get('infeasible')
    assert not by_symbol['MNQ'].get('infeasible')  # cluster captured exposure via MES -- not infeasible


def test_cluster_cap_es_nq_million_dollar_scenario_recomputed_for_greedy():
    """Reproduces this session's exact $1,000,000 account, ES+NQ-both-
    near-max-conviction scenario under the corrected greedy allocation.
    Recomputed (not assumed): ES (the higher-scalar instrument) wins the
    cluster's entire cap and gets 1 contract; NQ gets 0. Neither is
    flagged infeasible, since the cluster did capture real exposure."""
    es_close, es_mult, es_hv, es_scalar = 7428.25, 50, 0.1524, 0.8302
    nq_close, nq_mult, nq_hv, nq_scalar = 29514.25, 20, 0.238, 0.5379
    es_continuous = 479_325.91 / (es_close * es_mult)   # ~1.2908
    nq_continuous = 310_528.16 / (nq_close * nq_mult)   # ~0.5258

    targets = [
        _target('ES', 'equity', continuous_contracts=es_continuous, close=es_close, multiplier=es_mult,
               hv=es_hv, scalar=es_scalar),
        _target('NQ', 'equity', continuous_contracts=nq_continuous, close=nq_close, multiplier=nq_mult,
               hv=nq_hv, scalar=nq_scalar),
    ]
    total_risk_target = 1_000_000 * 0.15  # 150,000
    n_active_clusters = 3
    cap = max(0.25, 1 / n_active_clusters) * total_risk_target  # 50,000

    es_single_contract_risk = es_close * es_mult * es_hv   # ~56,607.3
    assert cap < es_single_contract_risk  # confirms the scenario's premise: even ES alone exceeds the cap

    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=total_risk_target, n_active_clusters=n_active_clusters)
    by_symbol = {t['symbol']: t for t in out}

    assert by_symbol['ES']['final_target_contracts'] == 1
    assert by_symbol['NQ']['final_target_contracts'] == 0
    # The cluster DID capture exposure (via ES) -- not infeasible, even
    # though a precomputed cap-vs-single-contract-risk check alone would
    # have said it should be.
    assert not by_symbol['ES'].get('infeasible')
    assert not by_symbol['NQ'].get('infeasible')


def test_cluster_cap_live_session_scenario_recomputed_for_greedy_priority():
    """Recomputes the grain/fx/equity scenario from this session under
    conviction-priority allocation: within each over-budget cluster, the
    highest-|scalar| instrument wins the cluster's cap first. Recomputed
    via direct verification, not assumed."""
    targets = [
        _target('MES', 'equity', continuous_contracts=1, close=7437.50, multiplier=5, hv=0.1524, scalar=0.85),
        _target('MZL', 'grain', continuous_contracts=5, close=66.58, multiplier=60, hv=0.2429, scalar=0.70),
        _target('MZC', 'grain', continuous_contracts=-13, close=437.00, multiplier=5, hv=0.1937, scalar=0.80),
        _target('MZS', 'grain', continuous_contracts=-2, close=1142.00, multiplier=5, hv=0.1492, scalar=0.20),
        _target('MZW', 'grain', continuous_contracts=-2, close=597.00, multiplier=5, hv=0.2969, scalar=0.19),
        _target('J7', 'fx', continuous_contracts=1, close=0.0066, multiplier=6_250_000, hv=0.0825, scalar=0.85),
        _target('BRE', 'fx', continuous_contracts=2, close=0.19, multiplier=100_000, hv=0.1143, scalar=0.75),
        _target('6M', 'fx', continuous_contracts=2, close=0.06, multiplier=500_000, hv=0.0825, scalar=0.78),
    ]
    total_risk_target = 100_000 * 0.15
    n_active_clusters = 3

    out = apply_cluster_risk_cap(targets, max_cluster_risk_pct=0.25,
                                 total_risk_target=total_risk_target, n_active_clusters=n_active_clusters)
    by_symbol = {t['symbol']: t for t in out}

    # Equity: only one instrument, no competition.
    assert by_symbol['MES']['final_target_contracts'] == 1
    # Grain: MZC (scalar 0.80, highest) wins the cluster's cap; the others
    # (lower priority) get nothing once MZC's allocation exhausts it.
    assert by_symbol['MZC']['final_target_contracts'] < 0  # short, sign preserved
    assert by_symbol['MZL']['final_target_contracts'] == 0
    assert by_symbol['MZS']['final_target_contracts'] == 0
    assert by_symbol['MZW']['final_target_contracts'] == 0
    # FX: J7 (0.85) and 6M (0.78) both outrank BRE (0.75) and capture
    # exposure; BRE gets whatever's left, which in this case is nothing.
    assert by_symbol['J7']['final_target_contracts'] > 0
    assert by_symbol['6M']['final_target_contracts'] > 0
    assert by_symbol['BRE']['final_target_contracts'] == 0
    # Every over-budget cluster still captured real exposure via its
    # top-priority instrument -- none are infeasible.
    assert not any(t.get('infeasible') for t in out)


# ── compute_position_scalar + signal_confidence wiring ──────────────────────

def test_position_scalar_signal_confidence_defaults_to_noop():
    """signal_confidence must default to 1.0 -- existing callers that don't
    pass it get byte-identical behavior to before Phase 2 existed."""
    with_default = compute_position_scalar(0.6, 0.01, vol_target=0.15, regime=TrendRegime.BULL,
                                            regime_discount=0.5)
    explicit_noop = compute_position_scalar(0.6, 0.01, vol_target=0.15, regime=TrendRegime.BULL,
                                             regime_discount=0.5, signal_confidence=1.0)
    assert with_default == explicit_noop


def test_position_scalar_signal_confidence_discounts_multiplicatively():
    base = compute_position_scalar(0.6, 0.01, vol_target=0.15, regime=TrendRegime.BULL, regime_discount=0.5)
    discounted = compute_position_scalar(0.6, 0.01, vol_target=0.15, regime=TrendRegime.BULL,
                                          regime_discount=0.5, signal_confidence=0.5)
    assert math.isclose(discounted, base * 0.5, rel_tol=1e-9)


def test_signal_confidence_instrument_specific_spike_discounts_only_that_instrument():
    """The core Phase 2 case: an instrument-specific vol spike (high
    hv_short/hv_long) discounts THAT instrument's scalar, while a sibling
    instrument with calm own-history vol is untouched -- even when both
    share the exact same vix_scalar (portfolio-wide VX state).
    This is the JPY-/corn-spike blind spot vix_scalar alone can't
    see, since it only looks at VIX/VX, not at corn's or JPY's own vol."""
    spiking_df = _vol_series(379, 21, calm_vol=0.01, shock_vol=0.08, seed=4)
    calm_df = _vol_series(379, 21, calm_vol=0.01, shock_vol=0.01, seed=5)

    spiking_vol_ratio = compute_vol_ratio(spiking_df).tail(1)['vol_ratio'][0]
    calm_vol_ratio = compute_vol_ratio(calm_df).tail(1)['vol_ratio'][0]

    low_threshold, high_threshold = 0.7, 1.5
    spiking_confidence = compute_signal_confidence(spiking_vol_ratio, low_threshold, high_threshold)
    calm_confidence = compute_signal_confidence(calm_vol_ratio, low_threshold, high_threshold)

    assert spiking_confidence < 1.0   # discounted
    assert calm_confidence == 1.0     # untouched

    # Same trend_strength/daily_std/regime for both instruments -- only
    # signal_confidence differs -- and the SAME vix_scalar
    # applied afterward to both (simulating one portfolio-wide VX state
    # that's calm, i.e. 1.0, so it doesn't mask the per-instrument effect).
    vix_scalar = 1.0
    spiking_scalar = compute_position_scalar(
        0.6, 0.01, vol_target=0.15, regime=TrendRegime.BULL,
        regime_discount=1.0, signal_confidence=spiking_confidence,
    ) * vix_scalar
    calm_scalar = compute_position_scalar(
        0.6, 0.01, vol_target=0.15, regime=TrendRegime.BULL,
        regime_discount=1.0, signal_confidence=calm_confidence,
    ) * vix_scalar

    assert spiking_scalar < calm_scalar
    assert math.isclose(calm_scalar, 0.6 * max(0.25, min(2.0, 0.15 / (0.01 * math.sqrt(252)))), rel_tol=1e-9)
