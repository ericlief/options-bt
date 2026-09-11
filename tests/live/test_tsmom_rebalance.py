"""
Tests for derivatives_bt_engine.live.tsmom_rebalance -- focused on
data_source='database' (no IB dependency at all, monkeypatched
FuturesDataLoader/VIX in place of a real duckdb/parquet read) plus
TsmomLiveConfig validation. The data_source='ib' path makes real IBPySync
calls (contract resolution, live bars/prices) with no local equivalent to
fake out cheaply -- 'database' is what makes this module testable, and is
also the notebook-runnable, no-live-account path the tests below exercise
end to end.
"""

import subprocess
import sys
import textwrap
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from derivatives_bt_engine.live import tsmom_rebalance as tr
from derivatives_bt_engine.live.tsmom_rebalance import TsmomLiveConfig, build_instruments, compute_rebalance_targets


def _trading_dates(start: date, n: int) -> list[date]:
    dates = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _price_df(start: date, n: int, drift: float, vol: float = 0.01, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    dates = _trading_dates(start, n)
    return pl.DataFrame({'ts_event': dates, 'close': close})


def _instrument(symbol: str, cluster: str = 'other', multiplier: float = 50.0) -> dict:
    return {
        'symbol': symbol, 'ib_symbol': symbol, 'signal_symbol': symbol, 'db_symbol': symbol,
        'exchange': 'CME', 'expiry': 'auto', 'multiplier': multiplier, 'cluster': cluster,
        'max_contracts': 50, 'max_notional': None,
    }


def _patch_db(monkeypatch, price_data: dict[str, pl.DataFrame], vx: tuple[float, float] = (15.0, 15.0)):
    """Fakes both data sources compute_rebalance_targets(data_source=
    'database') reads: FuturesDataLoader (per-instrument bars) and
    _vx_spike_ratio_from_db (local spot-VIX gate) -- no duckdb/parquet
    file needed."""
    class _FakeLoader:
        def __init__(self, asset, **kwargs):
            self.asset = asset

        @property
        def daily(self):
            return price_data[self.asset]

    monkeypatch.setattr(tr, 'FuturesDataLoader', _FakeLoader)
    monkeypatch.setattr(tr, 'assert_monotonic_expiration', lambda df, sym: None)
    monkeypatch.setattr(tr, '_vx_spike_ratio_from_db', lambda as_of=None, ma_window_days=63: vx)


# ── No ib_tools dependency for data_source='database' ────────────────────────

def test_module_imports_and_runs_database_mode_without_ib_tools_installed():
    # Regression test for the module-level `from ib_tools.ibpysync import
    # IBPySync` this module used to have -- that made even importing
    # TsmomLiveConfig/compute_rebalance_targets require ib_tools/ib_insync
    # to be installed, defeating the point of a notebook-runnable, no-IB
    # data_source='database' path for an environment that doesn't have
    # them. IBPySync is now imported lazily, only inside the functions that
    # actually touch it (all on the data_source='ib' path) -- verified here
    # in a genuinely fresh subprocess (this test file's own module-level
    # import already pulled ib_tools in for THIS process, so testing it
    # in-process would prove nothing) with ib_tools/ib_insync imports
    # blocked outright.
    script = textwrap.dedent("""
        import builtins
        real_import = builtins.__import__
        def blocked_import(name, *args, **kwargs):
            if name == 'ib_tools' or name.startswith('ib_tools.') or name == 'ib_insync':
                raise ImportError(f'blocked for test: {name}')
            return real_import(name, *args, **kwargs)
        builtins.__import__ = blocked_import

        from derivatives_bt_engine.live.tsmom_rebalance import TsmomLiveConfig, compute_rebalance_targets

        instr = {'symbol': 'X', 'ib_symbol': 'X', 'signal_symbol': 'X', 'db_symbol': 'X',
                 'exchange': 'CME', 'expiry': 'auto', 'multiplier': 1.0, 'cluster': 'other',
                 'max_contracts': 50, 'max_notional': None}
        config = TsmomLiveConfig(account_equity=100_000, data_source='database', vix_gating=False)
        # No FuturesDataLoader available (real duckdb may not be cached for
        # 'X') is fine to fail on -- the point is the IMPORT and the
        # up-front data_source/ib dispatch succeed without ib_tools; a
        # RuntimeError from the instrument-not-found path is an acceptable
        # (expected) outcome here, an ImportError of ib_tools is not.
        try:
            compute_rebalance_targets([instr], config, ib=None)
        except ImportError as exc:
            if 'ib_tools' in str(exc) or 'ib_insync' in str(exc):
                raise
        except Exception:
            pass
        print('OK')
    """)
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


# ── build_instruments ─────────────────────────────────────────────────────

def test_build_instruments_resolves_known_symbols():
    instruments = build_instruments(['MES', 'J7'])
    by_symbol = {i['symbol']: i for i in instruments}
    assert set(by_symbol) == {'MES', 'J7'}
    # MES: db_symbol falls back through signal_symbol to the full-size ES
    # sibling's own duckdb history.
    assert by_symbol['MES']['signal_symbol'] == 'ES'
    assert by_symbol['MES']['db_symbol'] == 'ES'
    # J7: explicit db_symbol divergence (IBKR ticker vs Globex root).
    assert by_symbol['J7']['db_symbol'] == '6J'
    for i in instruments:
        assert i['expiry'] == 'auto'
        assert i['max_notional'] is None
        assert i['max_contracts'] == tr.DEFAULT_MAX_CONTRACTS


def test_build_instruments_rejects_unknown_symbol():
    with pytest.raises(ValueError):
        build_instruments(['NOT_A_REAL_SYMBOL'])


def test_build_instruments_passes_through_max_notional_and_max_contracts():
    instruments = build_instruments(['MES'], max_notional=25_000.0, max_contracts=7)
    assert instruments[0]['max_notional'] == 25_000.0
    assert instruments[0]['max_contracts'] == 7


# ── TsmomLiveConfig validation ───────────────────────────────────────────────

@pytest.mark.parametrize('field,value', [
    ('signal_weighting', 'bogus'),
    ('mixing_pool', 'bogus'),
    ('risk_budget_mode', 'bogus'),
    ('notional_weighting', 'bogus'),
    ('discrete_allocation', 'bogus'),
    ('data_source', 'bogus'),
])
def test_tsmom_live_config_rejects_unknown_values(field, value):
    with pytest.raises(ValueError):
        TsmomLiveConfig(**{field: value})


def test_tsmom_live_config_defaults_match_prior_behavior():
    config = TsmomLiveConfig()
    assert config.signal_weighting == 'continuous'
    assert config.risk_budget_mode == 'cluster'
    assert config.data_source == 'ib'
    assert config.notional_weighting == 'flat'
    assert config.use_idm is True
    assert config.fast_window == 63
    assert config.slow_window == 252
    assert config.vol_fast_window is None
    assert config.vol_slow_window is None
    assert config.discrete_allocation == 'independent'
    assert config.discrete_risk_overrun_pct == 0.0


def test_tsmom_live_config_rejects_negative_discrete_risk_overrun():
    with pytest.raises(ValueError):
        TsmomLiveConfig(discrete_risk_overrun_pct=-0.01)


def test_tsmom_live_config_rejects_bad_windows():
    with pytest.raises(ValueError):
        TsmomLiveConfig(fast_window=0)
    with pytest.raises(ValueError):
        TsmomLiveConfig(fast_window=252, slow_window=63)


def test_fetch_signal_inputs_respects_custom_windows(monkeypatch):
    price_data = {'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)}
    _patch_db(monkeypatch, price_data)

    default_config = TsmomLiveConfig(account_equity=100_000, data_source='database')
    custom_config = TsmomLiveConfig(account_equity=100_000, data_source='database',
                                     fast_window=21, slow_window=100)
    instr = _instrument('X')

    default_raw = tr._fetch_signal_inputs(None, instr, default_config)
    custom_raw = tr._fetch_signal_inputs(None, instr, custom_config)

    # Different windows -> different ts_fast/ts_slow for the same series
    # (would only coincidentally match if the underlying r1d series were
    # perfectly flat, which _price_df's random walk isn't).
    assert default_raw['cm_df'].tail(1)['ts_fast'][0] != custom_raw['cm_df'].tail(1)['ts_fast'][0]


# ── compute_rebalance_targets: ib/database dispatch ──────────────────────────

def test_compute_rebalance_targets_requires_ib_connection_for_ib_data_source():
    config = TsmomLiveConfig(account_equity=100_000, data_source='ib')
    with pytest.raises(ValueError):
        compute_rebalance_targets([_instrument('X')], config, ib=None)


def test_compute_rebalance_targets_database_mode_needs_no_ib(monkeypatch):
    price_data = {'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)}
    _patch_db(monkeypatch, price_data)

    config = TsmomLiveConfig(account_equity=100_000, data_source='database')
    targets = compute_rebalance_targets([_instrument('X')], config, ib=None)

    assert len(targets) == 1
    t = targets[0]
    assert t.get('error') is None
    # No IB connection anywhere in this mode -- current_contracts is
    # unknowable without one, reported as None rather than a misleading 0.
    assert t['current_contracts'] is None
    assert t['final_target_contracts'] is not None


def test_sizing_diagnostics_expose_contract_hurdle_and_targets(monkeypatch):
    price_data = {'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)}
    _patch_db(monkeypatch, price_data)

    config = TsmomLiveConfig(account_equity=100_000, data_source='database')
    target = compute_rebalance_targets([_instrument('X', multiplier=50.0)], config, ib=None)[0]

    assert target['one_contract_notional'] == pytest.approx(target['close'] * target['mult'])
    assert target['one_contract_dollar_vol'] == pytest.approx(target['one_contract_notional'] * target['hv'])
    assert target['pre_scalar_dollar_vol_budget'] == pytest.approx(
        target['pre_scalar_notional_budget'] * config.vol_target
    )
    assert target['fractional_target_dollar_vol'] == pytest.approx(
        abs(target['fractional_target_notional']) * target['hv']
    )
    assert target['rounding_gap'] == pytest.approx(
        abs(target['fractional_target_contracts'] - target['final_target_contracts'])
    )
    assert target['scalar_capped'] in (True, False)
    assert target['portfolio_risk_target'] == pytest.approx(100_000 * config.target_portfolio_vol)
    assert target['idm_risk_target'] is None
    assert target['realized_portfolio_risk'] is None
    assert {
        'pre_scalar_notional_budget',
        'pre_scalar_dollar_vol_budget',
        'uncapped_fractional_target_notional',
        'fractional_target_notional',
        'one_contract_notional',
        'one_contract_dollar_vol',
        'fractional_target_dollar_vol',
        'fractional_target_contracts',
        'final_target_contracts',
        'standalone_position_dollar_vol',
    } <= target.keys()
    assert not {
        'budg_const', 'raw_not', 'targ_not', 'contract_notional',
        'one_contract_risk', 'target_risk', 'allocated_dollar_vol_budget', 'uncapped_target_notional',
        'target_notional', 'contin_con', 'target_con',
        'pos_risk',
    } & target.keys()


def test_compute_rebalance_targets_database_mode_respects_as_of(monkeypatch):
    price_data = {'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)}
    _patch_db(monkeypatch, price_data)

    early_as_of = price_data['X']['ts_event'][300]
    config = TsmomLiveConfig(account_equity=100_000, data_source='database', as_of=early_as_of)
    targets = compute_rebalance_targets([_instrument('X')], config, ib=None)

    assert targets[0].get('error') is None
    assert targets[0]['close'] == pytest.approx(price_data['X'].filter(pl.col('ts_event') <= early_as_of)
                                                 .tail(1)['close'][0])


def test_vix_gating_false_skips_the_vx_read_entirely(monkeypatch):
    # No VIX/VX source configured at all (unlike _patch_db's other callers,
    # _vx_spike_ratio_from_db is deliberately left unpatched here) --
    # vix_gating=False must mean compute_rebalance_targets never calls it,
    # not just that it tolerates it failing.
    price_data = {'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)}

    class _FakeLoader:
        def __init__(self, asset, **kwargs):
            self.asset = asset

        @property
        def daily(self):
            return price_data[self.asset]

    monkeypatch.setattr(tr, 'FuturesDataLoader', _FakeLoader)
    monkeypatch.setattr(tr, 'assert_monotonic_expiration', lambda df, sym: None)

    def _unreachable(*args, **kwargs):
        raise AssertionError("_vx_spike_ratio_from_db should not be called when vix_gating=False")

    monkeypatch.setattr(tr, '_vx_spike_ratio_from_db', _unreachable)

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', vix_gating=False)
    targets = compute_rebalance_targets([_instrument('X')], config, ib=None)

    assert targets[0].get('error') is None
    assert targets[0]['vol_regime'] == tr.VolRegime.NORMAL
    assert targets[0]['vx_current'] is None
    assert targets[0]['vx_ma'] is None
    assert targets[0]['vix_scalar'] == 1.0


# ── risk_budget_mode: 'cluster' vs 'idm' ─────────────────────────────────────

def test_risk_budget_mode_cluster_gives_every_active_instrument_the_same_budget(monkeypatch):
    price_data = {
        'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1),
        'Y': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=2),
    }
    _patch_db(monkeypatch, price_data)

    config = TsmomLiveConfig(account_equity=1_000_000, data_source='database', risk_budget_mode='cluster')
    targets = compute_rebalance_targets([_instrument('X'), _instrument('Y')], config, ib=None)

    budgets = {t['symbol']: t['pre_scalar_notional_budget'] for t in targets if t.get('error') is None}
    assert len(budgets) == 2
    assert budgets['X'] == pytest.approx(budgets['Y'])


def test_risk_budget_mode_idm_favors_independent_symbol_over_correlated_pair(monkeypatch):
    # A and B are the SAME series (correlation exactly 1.0), C is
    # independent -- same "correlated cluster vs. lone diversifier" setup
    # as domain.allocation's own compute_erc_weights tests, now exercised
    # end to end through the live rebalance's 'idm' risk_budget_mode.
    same_series = _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)
    price_data = {
        'A': same_series,
        'B': same_series,
        'C': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=2),
    }
    _patch_db(monkeypatch, price_data)

    # as_of pinned to the synthetic data's own last date -- the bounded EWM
    # correlation window (default 3y back from as_of) would otherwise find
    # zero overlap against config.as_of's default of date.today(), miles
    # past this synthetic 2018-19 series, and silently fall back to a flat
    # split (compute_erc_weights' own no-corr-data default) instead of
    # actually exercising the correlation-aware path this test is for.
    as_of = price_data['A']['ts_event'][-1]
    config = TsmomLiveConfig(account_equity=1_000_000, data_source='database', risk_budget_mode='idm',
                              notional_weighting='erc', as_of=as_of)
    targets = compute_rebalance_targets(
        [_instrument('A'), _instrument('B'), _instrument('C')], config, ib=None)

    budgets = {t['symbol']: t['pre_scalar_notional_budget'] for t in targets if t.get('error') is None}
    assert set(budgets) == {'A', 'B', 'C'}
    assert budgets['C'] > budgets['A']
    assert budgets['C'] > budgets['B']


def test_risk_budget_mode_idm_use_idm_false_skips_multiplier(monkeypatch):
    price_data = {
        'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1),
        'Y': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=2),
    }
    _patch_db(monkeypatch, price_data)

    config = TsmomLiveConfig(account_equity=1_000_000, target_portfolio_vol=0.15, vol_target=0.15,
                              data_source='database', risk_budget_mode='idm', notional_weighting='flat',
                              use_idm=False)
    targets = compute_rebalance_targets([_instrument('X'), _instrument('Y')], config, ib=None)

    budgets = [t['pre_scalar_notional_budget'] for t in targets if t.get('error') is None]
    assert len(budgets) == 2
    # No IDM adjustment, flat split -- total dollar-vol budget is exactly
    # account_equity * target_portfolio_vol, split evenly.
    assert sum(budgets) * 0.15 == pytest.approx(1_000_000 * 0.15, rel=1e-6)


def test_apply_cluster_cap_defaults_to_false():
    assert TsmomLiveConfig(account_equity=100_000).apply_cluster_cap is False


def test_apply_cluster_cap_wiring_reaches_apply_cluster_risk_cap(monkeypatch):
    # Spy on the real apply_cluster_risk_cap to confirm config.
    # apply_cluster_cap actually reaches its apply_cap kwarg, not just that
    # the config field exists.
    price_data = {
        'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1),
        'Y': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=2),
    }
    _patch_db(monkeypatch, price_data)

    captured = {}
    original = tr.apply_cluster_risk_cap

    def spy(*args, **kwargs):
        captured['apply_cap'] = kwargs.get('apply_cap')
        return original(*args, **kwargs)

    monkeypatch.setattr(tr, 'apply_cluster_risk_cap', spy)

    config = TsmomLiveConfig(account_equity=1_000_000, data_source='database', risk_budget_mode='idm',
                             notional_weighting='erc', apply_cluster_cap=True)
    compute_rebalance_targets([_instrument('X'), _instrument('Y')], config, ib=None)

    assert captured['apply_cap'] is True


def test_total_risk_target_scaled_by_idm_multiplier_when_cap_enabled(monkeypatch):
    # When apply_cluster_cap=True and risk_budget_mode='idm', the cap's own
    # total_risk_target must be scaled by the same idm_multiplier used to
    # size positions -- otherwise the cap silently claws back IDM's own
    # diversification credit (the bug this whole feature exists to fix).
    same_series = _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)
    price_data = {
        'A': same_series, 'B': same_series,
        'C': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=2),
    }
    _patch_db(monkeypatch, price_data)
    as_of = price_data['A']['ts_event'][-1]

    captured = {}
    original = tr.apply_cluster_risk_cap

    def spy(targets, max_cluster_risk_pct, total_risk_target, n_active_clusters, **kwargs):
        captured['total_risk_target'] = total_risk_target
        return original(targets, max_cluster_risk_pct, total_risk_target, n_active_clusters, **kwargs)

    monkeypatch.setattr(tr, 'apply_cluster_risk_cap', spy)

    config = TsmomLiveConfig(account_equity=1_000_000, data_source='database', risk_budget_mode='idm',
                             notional_weighting='erc', as_of=as_of, apply_cluster_cap=True)
    compute_rebalance_targets([_instrument('A'), _instrument('B'), _instrument('C')], config, ib=None)

    flat_total_risk_target = 1_000_000 * config.target_portfolio_vol
    # Real diversification exists (A/B correlated, C independent) -- the
    # scaled cap must be strictly bigger than the flat, uncredited figure.
    assert captured['total_risk_target'] > flat_total_risk_target


def test_portfolio_risk_contribution_attached_to_targets_under_idm_mode(monkeypatch):
    same_series = _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)
    price_data = {
        'A': same_series, 'B': same_series,
        'C': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=2),
    }
    _patch_db(monkeypatch, price_data)
    as_of = price_data['A']['ts_event'][-1]

    config = TsmomLiveConfig(account_equity=1_000_000, data_source='database', risk_budget_mode='idm',
                             notional_weighting='erc', as_of=as_of)
    targets = compute_rebalance_targets([_instrument('A'), _instrument('B'), _instrument('C')], config, ib=None)

    base_target = 1_000_000 * config.target_portfolio_vol
    for t in targets:
        if not t.get('error'):
            assert 'portfolio_risk_contribution' in t
            assert t['portfolio_risk_target'] == pytest.approx(base_target)
            assert t['idm_risk_target'] == pytest.approx(base_target * t['idm_multiplier'])
            assert t['realized_portfolio_risk'] is not None

    report = tr.print_cluster_risk_report(targets)
    assert 'portfolio_risk_contribution=' in report
    assert 'standalone_position_dollar_vol=' in report
    assert 'portfolio_risk_target=' in report
    assert 'idm_risk_target=' in report
    assert 'realized_portfolio_risk=' in report


def test_portfolio_risk_contribution_absent_under_cluster_mode(monkeypatch):
    # 'cluster' mode never computes H (no correlation-aware sizing at
    # all), so there's nothing for compute_realized_portfolio_risk to use
    # -- print_cluster_risk_report should fall back to standalone-position
    # dollar-vol totals only, with no contribution column.
    price_data = {
        'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1),
        'Y': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=2),
    }
    _patch_db(monkeypatch, price_data)

    config = TsmomLiveConfig(account_equity=1_000_000, data_source='database', risk_budget_mode='cluster')
    targets = compute_rebalance_targets([_instrument('X'), _instrument('Y')], config, ib=None)

    for t in targets:
        assert 'portfolio_risk_contribution' not in t

    report = tr.print_cluster_risk_report(targets)
    assert 'standalone_position_dollar_vol=' in report
    assert 'portfolio_risk_contribution=' not in report


def test_active_field_reflects_min_conviction_and_inactive_symbol_gets_clean_zero(monkeypatch):
    # Regression test: an 'idm'-mode symbol that fails min_conviction used
    # to fall through to budget_constant_by_symbol.get(symbol) is None ->
    # a spurious "account_equity not configured" ValueError, surfacing as
    # an ERROR row even though account_equity WAS configured -- the symbol
    # just wasn't in active_symbols. Must now be a clean target=0,
    # active=False row instead, with no 'error' key at all.
    price_data = {
        'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1),
        'Y': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=2),
    }
    _patch_db(monkeypatch, price_data)

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', risk_budget_mode='idm',
                              notional_weighting='erc', min_conviction=999.0)
    targets = compute_rebalance_targets([_instrument('X'), _instrument('Y')], config, ib=None)

    for t in targets:
        assert t.get('error') is None
        assert t['active'] is False
        assert t['final_target_contracts'] == 0
        assert t['pre_scalar_notional_budget'] is None
        assert t['notional_allocation_weight'] is None


# ── signal_weighting: 'goulding' ─────────────────────────────────────────────

def test_signal_weighting_goulding_populates_audit_fields(monkeypatch):
    price_data = {'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)}
    _patch_db(monkeypatch, price_data)

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', signal_weighting='goulding')
    targets = compute_rebalance_targets([_instrument('X')], config, ib=None)

    assert targets[0].get('error') is None
    assert targets[0]['g_regime'] in ('bull', 'bear', 'correction', 'rebound')
    assert targets[0]['a_co'] is not None
    assert targets[0]['a_re'] is not None


def test_signal_weighting_continuous_leaves_audit_fields_none(monkeypatch):
    price_data = {'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)}
    _patch_db(monkeypatch, price_data)

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', signal_weighting='continuous')
    targets = compute_rebalance_targets([_instrument('X')], config, ib=None)

    assert targets[0]['g_regime'] is None
    assert targets[0]['a_co'] is None
    assert targets[0]['a_re'] is None


# ── splice_live_price: live IB bar spliced onto the DB series' tail ──────────

def _full_price_df(start: date, n: int, drift: float, expiration: date, instrument_id: int = 1,
                    vol: float = 0.01, seed: int = 0) -> pl.DataFrame:
    """Like _price_df but with the full FuturesDataLoader.daily schema
    (open/high/low/volume/instrument_id/expiration) the splice path reads
    from `bars.tail(1)` -- _price_df's bare ts_event/close is enough for
    every OTHER test in this file since splice_live_price defaults False."""
    base = _price_df(start, n, drift, vol=vol, seed=seed)
    return base.with_columns(
        pl.col('close').alias('open'), pl.col('close').alias('high'), pl.col('close').alias('low'),
        pl.lit(100_000).alias('volume'), pl.lit(instrument_id).alias('instrument_id'),
        pl.lit(expiration).alias('expiration'),
    )


class _FakeIB:
    """Fakes the three IBPySync round-trip methods the splice path touches
    (req_contract_details, qualify_contracts, get_historical_bars).
    IBPySync.future(...) itself is called for real -- a pure, network-free
    ib_insync.Future construction (ib_tools is a real editable install in
    this dev env), so only the round-trip calls need faking.

    listed_months: ['YYYYMMDD', ...] -- what req_contract_details returns,
    simulating IB's own real listed-contract enumeration (see
    _list_live_dated_contracts' own docstring for why this replaced
    guessing a bare YYYYMM and letting IB re-resolve it). volumes_by_month/
    bar_by_month: {'YYYYMM': ...} keyed by the requested contract's own
    month (qualify_contracts/get_historical_bars both re-derive this from
    whatever full YYYYMMDD they were actually asked to qualify, so these
    dicts stay 6-char regardless of which exact listed date got picked) --
    bar_by_month's single (ts_event, close) becomes the LATEST (last) row
    of a single-row result; volumes_by_month's value is that latest row's
    own volume, used for the roll-detection comparison regardless of how
    many rows bars_by_month (plural, see below) returns for the same
    month. raise_for: months whose get_historical_bars call raises
    (simulates an IB error). mangle_month: a month whose qualified
    contract gets its lastTradeDateOrContractMonth corrupted, to trigger
    the resolved-date-mismatch guard. ambiguous_without_multiplier: months
    whose BLANK-multiplier request returns zero bars (simulating real IB's
    observed behavior for a genuinely ambiguous root like bare 'SI' -- no
    exception, just an empty result) but succeeds once ANY multiplier is
    set, to test _qualify_and_pull_recent_bar_with_fallback's retry.
    bars_by_month (plural): {'YYYYMM': [(ts_event, close), ...]} -- when
    set for a month, returns ALL of these rows (ascending ts_event, volume
    taken from volumes_by_month for every row) instead of the single-row
    bar_by_month/default -- for testing multi-day backfill."""

    def __init__(self, volumes_by_month, listed_months=None, bar_by_month=None, raise_for=frozenset(),
                 mangle_month=None, ambiguous_without_multiplier=frozenset(), bars_by_month=None):
        self.volumes_by_month = volumes_by_month
        self.listed_months = listed_months or []
        self.bar_by_month = bar_by_month or {}
        self.raise_for = raise_for
        self.mangle_month = mangle_month
        self.ambiguous_without_multiplier = ambiguous_without_multiplier
        self.bars_by_month = bars_by_month or {}
        self.calls: list[tuple] = []
        self.multipliers_seen: list[str] = []
        self.symbols_seen: list[str] = []

    def req_contract_details(self, contract):
        self.calls.append(('contract_details', contract.symbol))
        return [SimpleNamespace(contract=SimpleNamespace(lastTradeDateOrContractMonth=d))
                for d in self.listed_months]

    def qualify_contracts(self, contract):
        month = contract.lastTradeDateOrContractMonth[:6]
        self.calls.append(('qualify', month))
        self.multipliers_seen.append(contract.multiplier)
        self.symbols_seen.append(contract.symbol)
        if month == self.mangle_month:
            contract.lastTradeDateOrContractMonth = '20991231'

    def get_historical_bars(self, contract, end_date='', duration='2 D', bar_size='1 day',
                             what_to_show='TRADES', use_rth=True):
        month = contract.lastTradeDateOrContractMonth[:6]
        self.calls.append(('bars', month))
        if month in self.raise_for:
            raise RuntimeError('simulated IB error')
        if month in self.ambiguous_without_multiplier and not contract.multiplier:
            return pl.DataFrame()
        vol = self.volumes_by_month.get(month)
        if vol is None:
            return pl.DataFrame()
        if month in self.bars_by_month:
            rows = self.bars_by_month[month]
            return pl.DataFrame({'date': [r[0] for r in rows], 'open': [r[1] for r in rows],
                                 'high': [r[1] for r in rows], 'low': [r[1] for r in rows],
                                 'close': [r[1] for r in rows], 'volume': [vol] * len(rows)})
        ts_event, close = self.bar_by_month.get(month, (date(2026, 8, 15), 100.0))
        return pl.DataFrame({'date': [ts_event], 'open': [close], 'high': [close],
                             'low': [close], 'close': [close], 'volume': [vol]})


_TEST_CYCLE = ['H', 'M', 'U', 'Z']  # Mar/Jun/Sep/Dec
_ALL_MONTHS = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']  # CL's "no restriction" case


def _cycle_listed_months(cycle_letters: list[str], day: int = 20, years=None) -> list[str]:
    """Full YYYYMMDD strings for every cycle_letters month across `years`
    (default: today's year - 1 through +3) -- what a real IB
    req_contract_details response would return for a symbol whose listed
    months follow `cycle_letters`. Used to build _FakeIB's listed_months."""
    if years is None:
        y0 = date.today().year
        years = range(y0 - 1, y0 + 4)
    months = sorted(tr.CME_MONTH_LETTERS[letter] for letter in cycle_letters)
    return [f'{y}{m:02d}{day:02d}' for y in years for m in months]


def _next_cycle_date(cycle_letters: list[str], d: date, day: int = 20) -> date:
    """Test-local helper (independent of production code) -- the next
    cycle_letters month strictly after d's own month, wrapping to next
    year past the cycle's last month. Only used to build expected/fixture
    dates in these tests, not to re-derive the mechanism under test."""
    months = sorted(tr.CME_MONTH_LETTERS[letter] for letter in cycle_letters)
    idx = months.index(d.month)
    if idx == len(months) - 1:
        return date(d.year + 1, months[0], day)
    return date(d.year, months[idx + 1], day)


def _nearest_cycle_date_from_today(cycle_letters: list[str], day: int = 20) -> date:
    """Test-local helper -- the cycle_letters date _list_live_dated_
    contracts would ACTUALLY pick as the nearest live candidate right now
    (candidate selection is driven entirely by real IB listings filtered
    against date.today(), not by any date baked into a test fixture -- see
    test_splice_ignores_db_own_stale_expiration_column). Tests anchor
    their volumes_by_month/bar_by_month keys off this, not off an
    arbitrary fixed-future date, so they stay correct regardless of when
    they actually run."""
    today = date.today()
    months = sorted(tr.CME_MONTH_LETTERS[letter] for letter in cycle_letters)
    for m in months:
        candidate = date(today.year, m, day)
        if candidate >= today:
            return candidate
    return date(today.year + 1, months[0], day)


def _not_stale_expiration() -> date:
    """A _TEST_CYCLE month safely >= date.today() regardless of when tests
    actually run (December of next year). Use for "DB is current, not
    stale" test scenarios."""
    return date(date.today().year + 1, 12, 20)


def _stale_expiration() -> date:
    """A _TEST_CYCLE month safely IN THE PAST relative to date.today()
    regardless of when tests actually run (March two years ago). Use for
    "DB's own recorded expiration is very stale" test scenarios -- the
    splice no longer reads this column for candidate selection at all
    (see test_splice_ignores_db_own_stale_expiration_column below), so
    this now exists to prove exactly that, not to test a walk-forward
    mechanism (there isn't one anymore -- candidates come from IB's own
    live listing, not from extrapolating off the DB's last-known
    contract)."""
    return date(date.today().year - 2, 3, 20)


def test_list_live_dated_contracts_sorts_ascending_by_expiry(monkeypatch):
    fake_ib = _FakeIB(volumes_by_month={}, listed_months=['20260920', '20260620', '20261220'])
    result = tr._list_live_dated_contracts(fake_ib, 'CL', 'NYMEX')
    assert result == [(date(2026, 6, 20), '20260620'), (date(2026, 9, 20), '20260920'),
                       (date(2026, 12, 20), '20261220')]


def test_list_live_dated_contracts_skips_blank_entries():
    fake_ib = _FakeIB(volumes_by_month={}, listed_months=['20260920', '', '20261220'])
    result = tr._list_live_dated_contracts(fake_ib, 'CL', 'NYMEX')
    assert result == [(date(2026, 9, 20), '20260920'), (date(2026, 12, 20), '20261220')]


def test_splice_live_price_requires_ib_even_in_database_mode():
    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    with pytest.raises(ValueError):
        compute_rebalance_targets([_instrument('X')], config, ib=None)


def test_splice_disabled_by_default_makes_no_ib_calls(monkeypatch):
    price_data = {'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)}
    _patch_db(monkeypatch, price_data)
    calls = []
    monkeypatch.setattr(tr, '_splice_live_front_month_bar', lambda *a, **k: calls.append(1) or a[-1])

    config = TsmomLiveConfig(account_equity=100_000, data_source='database')  # splice_live_price defaults False
    targets = compute_rebalance_targets([_instrument('X')], config, ib=None)

    assert calls == []
    assert targets[0].get('error') is None


def test_splice_enabled_db_contract_still_front_appends_new_row(monkeypatch):
    current = _nearest_cycle_date_from_today(_TEST_CYCLE)
    current_month = current.strftime('%Y%m')
    next_month_date = _next_cycle_date(_TEST_CYCLE, current)
    next_month = next_month_date.strftime('%Y%m')
    price_data = {'X': _full_price_df(date(2018, 1, 1), 500, drift=0.0015,
                                       expiration=_not_stale_expiration(), seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'CME'})
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000, next_month: 500},
                      listed_months=_cycle_listed_months(_TEST_CYCLE),
                      bar_by_month={current_month: (date.today(), 111.0)})

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    targets = compute_rebalance_targets([_instrument('X')], config, ib=fake_ib)

    assert targets[0].get('error') is None
    assert targets[0]['close'] == pytest.approx(111.0)
    assert ('qualify', next_month) in fake_ib.calls  # next contract WAS checked...
    assert ('bars', next_month) in fake_ib.calls      # ...just lost the volume comparison


def test_splice_uses_blank_multiplier_not_instrument_or_db_symbol_own(monkeypatch):
    # Regression test: this project's internal `multiplier` field is a
    # $-per-point P&L-scaling convention, not necessarily IB's own
    # contract multiplier -- passing EITHER the calling instrument's own
    # value (e.g. MES's 5) or even db_symbol's own registry value (ES's
    # 50) caused IB to reject "No security definition has been found" for
    # several real products this session (grains, silver, JPY, where our
    # internal convention diverges from IB's). Confirmed live: dropping
    # multiplier entirely (symbol + exchange + explicit date) is what
    # actually works, matching _resolve_contract's own proven pattern for
    # a canonical (non ticker-renamed) root.
    current = _nearest_cycle_date_from_today(_TEST_CYCLE)
    current_month = current.strftime('%Y%m')
    next_month_date = _next_cycle_date(_TEST_CYCLE, current)
    next_month = next_month_date.strftime('%Y%m')
    third_month = _next_cycle_date(_TEST_CYCLE, next_month_date).strftime('%Y%m')
    price_data = {'ES': _full_price_df(date(2018, 1, 1), 500, drift=0.0015,
                                        expiration=_not_stale_expiration(), seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'CME', 'multiplier': 50})
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000, next_month: 500, third_month: 100},
                      listed_months=_cycle_listed_months(_TEST_CYCLE))

    mes = _instrument('MES', multiplier=5.0)
    mes['db_symbol'] = 'ES'
    mes['signal_symbol'] = 'ES'
    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    targets = compute_rebalance_targets([mes], config, ib=fake_ib)

    assert targets[0].get('error') is None
    assert all(m == '' for m in fake_ib.multipliers_seen)


def test_splice_retries_with_registry_multiplier_when_blank_is_ambiguous(monkeypatch):
    # Regression test: confirmed live this session, bare 'SI' resolves to
    # BOTH full-size silver (multiplier=5000) and the SIL micro
    # (multiplier=1000) on IB -- an ambiguous-contract case that returns
    # ZERO bars (not an exception) on the blank-multiplier attempt, so it
    # can't be told apart from genuine "no data" ahead of time. The retry
    # with db_symbol's own registry multiplier is what actually
    # disambiguates it.
    expiration = _not_stale_expiration()
    current_month = expiration.strftime('%Y%m')
    next_month = _next_cycle_date(_TEST_CYCLE, expiration).strftime('%Y%m')
    price_data = {'SI': _full_price_df(date(2018, 1, 1), 500, drift=0.0015, expiration=expiration, seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'COMEX', 'multiplier': 5000})
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000, next_month: 500},
                      listed_months=_cycle_listed_months(_TEST_CYCLE),
                      ambiguous_without_multiplier={current_month, next_month})

    sil = _instrument('SIL', multiplier=1000.0)
    sil['db_symbol'] = 'SI'
    sil['signal_symbol'] = 'SI'
    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    targets = compute_rebalance_targets([sil], config, ib=fake_ib)

    assert targets[0].get('error') is None
    # Both attempts happened for the picked contract: blank first, then
    # the registry multiplier that actually resolved it.
    assert '' in fake_ib.multipliers_seen
    assert '5000' in fake_ib.multipliers_seen


def test_splice_translates_fx_db_symbol_to_ib_facing_ticker(monkeypatch):
    # Regression test: confirmed live this session, the DB stores JPY's
    # continuous series under the raw Globex root '6J', but IB's own API
    # only resolves it under 'JPY' (instruments._FX_TICKER_TO_KEY) --
    # passing '6J' straight through got "No security definition has been
    # found" for every candidate month.
    expiration = _not_stale_expiration()
    current_month = expiration.strftime('%Y%m')
    next_month = _next_cycle_date(_TEST_CYCLE, expiration).strftime('%Y%m')
    price_data = {'6J': _full_price_df(date(2018, 1, 1), 500, drift=0.0015, expiration=expiration, seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'CME', 'multiplier': 12_500_000})
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000, next_month: 500},
                      listed_months=_cycle_listed_months(_TEST_CYCLE))

    j7 = _instrument('J7', multiplier=6_250_000.0)
    j7['db_symbol'] = '6J'
    j7['signal_symbol'] = 'JPY'
    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    targets = compute_rebalance_targets([j7], config, ib=fake_ib)

    assert targets[0].get('error') is None
    assert fake_ib.symbols_seen == ['JPY'] * len(fake_ib.symbols_seen)
    assert '6J' not in fake_ib.symbols_seen


def test_splice_enabled_roll_detected_uses_next_contract(monkeypatch, caplog):
    current = _nearest_cycle_date_from_today(_TEST_CYCLE)
    current_month = current.strftime('%Y%m')
    next_month = _next_cycle_date(_TEST_CYCLE, current).strftime('%Y%m')
    price_data = {'X': _full_price_df(date(2018, 1, 1), 500, drift=0.0015,
                                       expiration=_not_stale_expiration(), seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'CME'})
    fake_ib = _FakeIB(volumes_by_month={current_month: 500, next_month: 2000},
                      listed_months=_cycle_listed_months(_TEST_CYCLE),
                      bar_by_month={next_month: (date.today(), 222.0)})

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    with caplog.at_level('WARNING'):
        targets = compute_rebalance_targets([_instrument('X')], config, ib=fake_ib)

    assert targets[0].get('error') is None
    assert targets[0]['close'] == pytest.approx(222.0)
    assert any('roll detected' in r.message for r in caplog.records)


def test_splice_skipped_when_active_months_unconfirmed(monkeypatch):
    price_data = {'X': _full_price_df(date(2018, 1, 1), 500, drift=0.0015,
                                       expiration=_not_stale_expiration(), seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: None)
    fake_ib = _FakeIB(volumes_by_month={})

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    targets = compute_rebalance_targets([_instrument('X')], config, ib=fake_ib)

    assert targets[0].get('error') is None
    assert fake_ib.calls == []


def test_splice_no_restriction_symbol_like_cl_now_gets_spliced(monkeypatch):
    # Regression test: CL (crude oil) has NO restricted active_months
    # cycle -- confirmed empirically to trade all 12 CME months in
    # comparable volume, unlike every other product surveyed. It used to
    # get skipped entirely (resolve_active_months returned None, read as
    # "unconfirmed"), even though "no restriction" is itself a confirmed
    # finding, not an unknown. instruments.py now sets CL's active_months
    # to the full 12-month list specifically so it isn't treated the same
    # as a genuinely-unconfirmed symbol (e.g. BRE) -- this proves the
    # fix: a full-12-month cycle symbol actually gets spliced now.
    current_month = _nearest_cycle_date_from_today(_ALL_MONTHS).strftime('%Y%m')
    price_data = {'CL': _full_price_df(date(2018, 1, 1), 500, drift=0.0015,
                                        expiration=_not_stale_expiration(), seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _ALL_MONTHS)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'NYMEX'})
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000}, listed_months=_cycle_listed_months(_ALL_MONTHS),
                      bar_by_month={current_month: (date.today(), 55.0)})

    mcl = _instrument('MCL', multiplier=100.0)
    mcl['db_symbol'] = 'CL'
    mcl['signal_symbol'] = 'CL'
    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    targets = compute_rebalance_targets([mcl], config, ib=fake_ib)

    assert targets[0].get('error') is None
    assert fake_ib.calls != []  # actually attempted, not skipped like the unconfirmed case above
    assert targets[0]['close'] == pytest.approx(55.0)  # and actually succeeded, not just attempted


def test_splice_widened_search_finds_winner_two_months_out(monkeypatch, caplog):
    # The concrete scenario this widened search (3 candidates, not 1) was
    # built for: a no-restriction product's true volume leader can be
    # further out than just "the next month" -- confirmed live can't be
    # exercised here (needs real IB volume data), but this proves the
    # mechanism: the 3rd candidate (2 months out) wins the comparison,
    # and a narrower search would have missed it entirely.
    current = _nearest_cycle_date_from_today(_ALL_MONTHS)
    current_month = current.strftime('%Y%m')
    next_month_date = _next_cycle_date(_ALL_MONTHS, current)
    next_month = next_month_date.strftime('%Y%m')
    third_month = _next_cycle_date(_ALL_MONTHS, next_month_date).strftime('%Y%m')
    price_data = {'CL': _full_price_df(date(2018, 1, 1), 500, drift=0.0015,
                                        expiration=_not_stale_expiration(), seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _ALL_MONTHS)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'NYMEX'})
    fake_ib = _FakeIB(volumes_by_month={current_month: 500, next_month: 700, third_month: 5000},
                      listed_months=_cycle_listed_months(_ALL_MONTHS),
                      bar_by_month={third_month: (date.today(), 77.0)})

    mcl = _instrument('MCL', multiplier=100.0)
    mcl['db_symbol'] = 'CL'
    mcl['signal_symbol'] = 'CL'
    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    with caplog.at_level('WARNING'):
        targets = compute_rebalance_targets([mcl], config, ib=fake_ib)

    assert targets[0].get('error') is None
    assert targets[0]['close'] == pytest.approx(77.0)
    assert any('roll detected' in r.message for r in caplog.records)


def test_splice_falls_back_gracefully_on_ib_error(monkeypatch, caplog):
    expiration = _not_stale_expiration()
    current_month = expiration.strftime('%Y%m')
    next_month = _next_cycle_date(_TEST_CYCLE, expiration).strftime('%Y%m')
    price_data = {'X': _full_price_df(date(2018, 1, 1), 500, drift=0.0015, expiration=expiration, seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'CME'})
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000, next_month: 500},
                      listed_months=_cycle_listed_months(_TEST_CYCLE), raise_for={current_month})

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    with caplog.at_level('WARNING'):
        targets = compute_rebalance_targets([_instrument('X')], config, ib=fake_ib)

    # No exception propagates -- the whole rebalance still completes
    # normally, and the still-live next candidate is used instead of the
    # one that errored.
    assert targets[0].get('error') is None
    assert targets[0]['final_target_contracts'] is not None
    assert any('candidate contract' in r.message and 'unavailable' in r.message for r in caplog.records)


def test_splice_all_candidates_unavailable_falls_back_gracefully(monkeypatch, caplog):
    expiration = _not_stale_expiration()
    current_month = expiration.strftime('%Y%m')
    next_month_date = _next_cycle_date(_TEST_CYCLE, expiration)
    next_month = next_month_date.strftime('%Y%m')
    third_month = _next_cycle_date(_TEST_CYCLE, next_month_date).strftime('%Y%m')
    price_data = {'X': _full_price_df(date(2018, 1, 1), 500, drift=0.0015, expiration=expiration, seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'CME'})
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000, next_month: 500},
                      listed_months=_cycle_listed_months(_TEST_CYCLE),
                      raise_for={current_month, next_month, third_month})

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    with caplog.at_level('WARNING'):
        targets = compute_rebalance_targets([_instrument('X')], config, ib=fake_ib)

    assert targets[0].get('error') is None
    assert any('live price splice failed' in r.message for r in caplog.records)


def test_splice_ignores_db_own_stale_expiration_column(monkeypatch):
    # Confirms candidate selection is now driven entirely by IB's own live
    # listing (_list_live_dated_contracts), NOT by extrapolating off the
    # DB's own `expiration` column -- which can be arbitrarily stale
    # (confirmed live this session: a DB cache stale by more than one roll
    # used to feed a now-delisted month straight into IB and abort the
    # whole splice when it was unconditionally rejected). Here the DB
    # fixture's own recorded expiration is 2 years stale; the splice must
    # still succeed correctly using only real, current IB listings.
    stale_expiration = _stale_expiration()
    current_month = _nearest_cycle_date_from_today(_TEST_CYCLE).strftime('%Y%m')
    price_data = {'X': _full_price_df(date(2018, 1, 1), 500, drift=0.0015, expiration=stale_expiration, seed=1)}
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'CME'})
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000}, listed_months=_cycle_listed_months(_TEST_CYCLE),
                      bar_by_month={current_month: (date.today(), 444.0)})

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    targets = compute_rebalance_targets([_instrument('X')], config, ib=fake_ib)

    assert targets[0].get('error') is None
    assert targets[0]['close'] == pytest.approx(444.0)
    # The DB's own (2-years-stale) recorded month was never even queried.
    assert stale_expiration.strftime('%Y%m') not in [month for _, month in fake_ib.calls]


def test_splice_replaces_existing_same_day_row_not_duplicates(monkeypatch):
    current = _nearest_cycle_date_from_today(_TEST_CYCLE)
    current_month = current.strftime('%Y%m')
    next_month = _next_cycle_date(_TEST_CYCLE, current).strftime('%Y%m')
    price_data = {'X': _full_price_df(date(2024, 1, 1), 500, drift=0.0015,
                                       expiration=_not_stale_expiration(), seed=1)}
    original_height = price_data['X'].height
    same_day = price_data['X'].tail(1)['ts_event'][0]
    _patch_db(monkeypatch, price_data)
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'CME'})
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000, next_month: 500},
                      listed_months=_cycle_listed_months(_TEST_CYCLE),
                      bar_by_month={current_month: (same_day, 333.0)})

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True)
    targets = compute_rebalance_targets([_instrument('X')], config, ib=fake_ib)

    assert targets[0].get('error') is None
    assert targets[0]['close'] == pytest.approx(333.0)
    # Replaced, not appended -- row count unchanged.
    assert price_data['X'].height == original_height


def test_splice_sets_expiration_on_spliced_rows(monkeypatch):
    # Spliced rows used to leave `expiration` null (diagonal_relaxed fills
    # missing columns with nulls) -- now set explicitly from the picked
    # candidate's own confirmed date, so a notebook stacking cm_df/closes
    # across symbols doesn't lose it for exactly the freshest rows.
    current = _nearest_cycle_date_from_today(_TEST_CYCLE)
    current_month = current.strftime('%Y%m')
    price_data = {'X': _full_price_df(date(2018, 1, 1), 500, drift=0.0015,
                                       expiration=_not_stale_expiration(), seed=1)}
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'CME'})
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000}, listed_months=_cycle_listed_months(_TEST_CYCLE),
                      bar_by_month={current_month: (date.today(), 111.0)})

    merged = tr._splice_live_front_month_bar(fake_ib, _instrument('X'), 'X', price_data['X'])

    spliced_row = merged.filter(pl.col('ts_event') == date.today())
    assert spliced_row['expiration'][0] == current


def _trading_dates_ending(end: date, n: int) -> list[date]:
    dates = []
    d = end
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    return list(reversed(dates))


def test_splice_backfills_entire_gap_not_just_latest_bar(monkeypatch):
    # Regression test for the actual point of this feature: a DB cache
    # stale by MORE than a day or two needs every missing trading day
    # backfilled, not just today's -- continuous_momentum's rolling
    # windows are row-count-based, not calendar-aware, so splicing only
    # the latest bar while N days are missing silently compresses those N
    # days out of the window instead of actually filling the gap. Calls
    # _splice_live_front_month_bar directly (not through
    # compute_rebalance_targets) to inspect the merged frame row-by-row --
    # the public pipeline only ever exposes the LATEST computed values.
    expiration = _not_stale_expiration()  # DB fixture placeholder only -- unread by candidate selection
    current_month = _nearest_cycle_date_from_today(_TEST_CYCLE).strftime('%Y%m')
    gap_start = date.today() - timedelta(days=12)  # DB stale by ~12 calendar days
    hist_dates = _trading_dates_ending(gap_start, 100)
    rng = np.random.default_rng(1)
    hist_close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(hist_dates))))
    db_bars = pl.DataFrame({
        'ts_event': hist_dates, 'open': hist_close, 'high': hist_close, 'low': hist_close,
        'close': hist_close, 'volume': [100_000] * len(hist_dates),
        'instrument_id': [1] * len(hist_dates), 'expiration': [expiration] * len(hist_dates),
    })
    monkeypatch.setattr(tr, 'resolve_active_months', lambda symbol: _TEST_CYCLE)
    monkeypatch.setattr(tr, 'get_spec', lambda symbol: {'exchange': 'CME'})

    # IB has every trading day from the gap start through today -- more
    # than what's actually missing (gap_start itself is already in the
    # DB), so this also exercises the >= same-day-refresh overlap.
    ib_dates = _trading_dates_ending(date.today(), 20)
    ib_dates = [d for d in ib_dates if d >= gap_start]
    new_days = [d for d in ib_dates if d > gap_start]
    assert len(new_days) > 1  # sanity: this test is actually exercising a multi-day gap
    fake_ib = _FakeIB(volumes_by_month={current_month: 1000}, listed_months=_cycle_listed_months(_TEST_CYCLE),
                      bars_by_month={current_month: [(d, 200.0 + i) for i, d in enumerate(ib_dates)]})

    merged = tr._splice_live_front_month_bar(fake_ib, _instrument('X'), 'X', db_bars)

    assert merged.height == db_bars.height + len(new_days)
    assert set(merged['ts_event'].to_list()) == set(hist_dates) | set(new_days)
    for i, d in enumerate(ib_dates):
        assert merged.filter(pl.col('ts_event') == d)['close'][0] == pytest.approx(200.0 + i)


def test_splice_skipped_when_as_of_is_set(monkeypatch):
    price_data = {'X': _price_df(date(2018, 1, 1), 500, drift=0.0015, vol=0.005, seed=1)}
    _patch_db(monkeypatch, price_data)
    calls = []
    monkeypatch.setattr(tr, '_splice_live_front_month_bar', lambda *a, **k: calls.append(1) or a[-1])
    early_as_of = price_data['X']['ts_event'][300]

    config = TsmomLiveConfig(account_equity=100_000, data_source='database', splice_live_price=True,
                              as_of=early_as_of)
    targets = compute_rebalance_targets([_instrument('X')], config, ib=_FakeIB(volumes_by_month={}))

    assert calls == []
    assert targets[0].get('error') is None
