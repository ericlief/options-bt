"""
Multi-symbol TSMOM backtest: monthly rebalance to a vol-targeted contract
count per symbol, with a spot-VIX regime gate, in a simple form.

Deliberately separate from Backtester/TradeManager/FuturesPosition: those
are built around discrete "open position, hold until roll/expiry, then
close" trades, shared with the still-pandas option backtest path. TSMOM's
lifecycle (continuously-sized monthly rebalance toward a target contract
count, no roll/expiry-driven open-close cycle) doesn't fit that model, and
retrofitting it would risk regressing the shared option path. This module
reuses the existing pure signal math (signal.py) and FuturesDataLoader
but implements its own portfolio loop.

No VX futures (CFE) history is available locally (the Globex MDP3.0 duckdb
is CME-only) -- the regime gate below uses spot VIX vs its own trailing
63-day MA as the closest available analog to the live system's VX
front-month / VX-63d-MA ratio (see derivatives_bt_engine.live.tsmom_rebalance).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

import duckdb
import polars as pl

from derivatives_bt_engine.domain.allocation import (
    NOTIONAL_WEIGHTING_SCHEMES,
    build_returns_wide,
    compute_position_scalar,
    compute_symbol_notional_budget,
)
from derivatives_bt_engine.domain.enums import VolRegime
from derivatives_bt_engine.domain.futures_dataloader import FuturesDataLoader, assert_monotonic_expiration
from derivatives_bt_engine.domain.instruments import (
    CME_MONTH_NUM_TO_LETTER, get_spec, resolve_active_months, resolve_annualization_days, resolve_price_symbol,
)
from derivatives_bt_engine.domain.signal import (
    DEFAULT_FAST_WINDOW,
    DEFAULT_SLOW_WINDOW,
    SignalSpec,
    build_features,
    build_monthly_state_return_history,
    continuous_momentum,
    estimate_mixing_params,
    goulding_monthly,
    resolve_trend_direction,
)
from derivatives_bt_engine.utils.logger import setup_logger

logger = setup_logger()

# Same VIX_PATH convention as naked_futures.py/the options strategies -- a
# directory resolves to {dir}/processed/vix.parquet (see
# BaseDataLoader._resolve_source_paths). The old hardcoded
# .../VIX/historical/vix.parquet path no longer exists (stale, pre-dates a
# data-directory reorg) and would raise FileNotFoundError.
VIX_FILE_PATH = os.path.join(
    os.path.expanduser(os.getenv('VIX_PATH', '~/data/fin/market/index/VIX/eod')),
    'processed', 'vix.parquet',
)

# Same band thresholds as derivatives_bt_engine.live.tsmom_rebalance's VX-futures gate,
# applied to spot-VIX-current / spot-VIX-63d-MA instead.
VIX_ELEVATED_RATIO = 1.3
VIX_SPIKE_RATIO = 1.5
VIX_EXTREME_RATIO = 2.0
VIX_ELEVATED_SCALE = 0.6
# Trailing window (calendar days) for the spot-VIX moving average
# vix_ratio is computed against -- see TsmomBacktestConfig.vix_ma_window_days.
# The bands above were calibrated against this default; changing it changes
# what "elevated"/"spike"/"extreme" actually mean.
DEFAULT_VIX_MA_WINDOW_DAYS = 63
# Decimal places for genuinely PRICE-scale fields (entry_price/exit_price/
# transaction price/close/peak) -- NOT dollar amounts (fees/pnl/capital,
# which stay at 2dp) or ratios/percentages (ts_fast/vix_ratio/etc., which
# keep their own existing precision). 2dp was fine for equity-index-scale
# instruments (MES ~3700) but silently collapsed FX futures like J7 (quoted
# ~0.0097, USD per JPY) to a flat 0.01 for every single row -- confirmed
# directly: a real -0.0001315 price move (a genuine, correctly-computed
# -$824.60 PnL on full-precision internal math) displayed as
# entry_price == exit_price == 0.01, making a real loss look like a
# flat/impossible trade. 6dp keeps equity/metal/grain instruments perfectly
# readable (just trailing zeros) while actually distinguishing consecutive
# FX-scale price observations from each other.
_PRICE_ROUND_NDIGITS = 6


@dataclass
class TsmomBacktestConfig:
    symbols: list[str]
    initial_capital: float = 100_000.0
    vol_target: float = 0.15
    max_contracts: int = 5
    max_notional: float = 25_000.0
    long_only: bool = False
    regime_discount: float = 0.5
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    # Signal-based entry/exit gate -- same mechanism as FuturesStrategyConfig/
    # TradeManager's ts_exit_threshold/ts_entry_threshold/exit_on_ts_crossover
    # (domain/trade_manager.py), adapted here for TSMOM's variable-direction
    # sizing: direction is derived from whichever side actually matters (the
    # currently-held position for exit, the newly-proposed target for entry)
    # rather than a fixed config field, since a TSMOM symbol can go long or
    # short depending on the sign of its own signal. signal_gate_mode picks
    # the cadence at which the gate is checked -- 'monthly' only at the
    # existing rebalance points (near-zero extra cost, reuses signal/ts_fast/
    # ts_slow _compute_signal_row already computed that day); 'daily'
    # additionally checks both entry AND exit every day in between,
    # off-cycle from the monthly resize, so a flat symbol can open the day
    # its entry gate first clears (not just at month-end) and a held
    # symbol can flatten the day its exit gate fires. Resizing (magnitude
    # changes to an already-open position) stays strictly monthly-only in
    # BOTH modes either way -- 'daily' only adds off-cycle open/flatten
    # transitions, never off-cycle resizing.
    ts_exit_threshold: Optional[float] = None
    ts_entry_threshold: Optional[float] = None
    exit_on_ts_crossover: bool = False
    signal_gate_mode: str = 'off'  # 'off' | 'monthly' | 'daily'
    # Opt-in "no rebalancing" mode: a positional list of fixed contract
    # counts, one per symbol (index-aligned with `symbols`, e.g.
    # symbols=['ES','GC','CL'], fixed_quantities=[4,3,2] means ES always
    # trades in units of 4, GC in units of 3, CL in units of 2). When set,
    # _compute_target skips vol-targeted/notional-scaled sizing entirely --
    # direction still comes from the sign of that symbol's own raw signal
    # (there's no other principled way to know when to go short without
    # it), but magnitude is just this fixed count (times vix_scalar
    # during an elevated-VIX regime, rounded, same as the vol-targeted
    # path's own elevated-VIX scaling) instead of vol_target/max_notional
    # math. Entry/exit gates (ts_exit_threshold/ts_entry_threshold/
    # exit_on_ts_crossover) and the VIX spike/extreme hold-or-halve
    # override still apply exactly as before -- this only replaces the
    # continuous vol-targeted scalar with a constant, it doesn't touch
    # the rest of the day loop.
    fixed_quantities: Optional[list[int]] = None
    # Portfolio-wide VIX regime gate (spot-VIX-vs-63d-MA spike/extreme
    # hold-or-halve override, elevated -> vix_scalar de-risking)
    # -- on by default (matches all prior behavior). Toggle off to isolate
    # the effect of ts_exit_threshold/ts_entry_threshold/exit_on_ts_crossover
    # alone, without VIX-driven interference -- particularly relevant for
    # signal_gate_mode='daily', which is already a departure from
    # traditional monthly-only TSMOM, and where mixing in a second,
    # differently-cadenced portfolio-wide gate makes it harder to isolate
    # what's actually being tested.
    vix_gating: bool = True
    # Trailing window (calendar days) for the spot-VIX moving average
    # vix_ratio is computed against -- see DEFAULT_VIX_MA_WINDOW_DAYS above
    # and derivatives_bt_engine.live.tsmom_rebalance's own
    # TsmomLiveConfig.vx_ma_window_days, which this mirrors so a live run
    # and a backtest comparison use the same window when you want them to.
    vix_ma_window_days: int = DEFAULT_VIX_MA_WINDOW_DAYS
    # Correlation-aware sizing -- None (default) preserves this module's
    # original behaviour exactly: every symbol independently sized to
    # config.max_notional * scalar / one_contract_notional, with NOTHING
    # scaling the book down for holding multiple symbols at once. Confirmed
    # directly (2026-07) that this has no diversification correction of any
    # kind anywhere -- no n_effective, no sqrt(N), no correlation term --
    # which is exactly why a correlated multi-symbol backtest run here
    # overstated realized vol 82-90% against a 15% target (see
    # derivatives_bt_engine.strats.tsmom_binary_vol_parity_backtest's own
    # module docstring, which documents this as the reason that script was
    # built with a deliberately simpler sizing scheme instead of reusing
    # this one).
    #
    # When set (e.g. 0.15), run_tsmom_backtest instead derives EACH
    # rebalance's own per-symbol notional_budget from
    # domain.allocation.compute_idm/build_returns_wide/
    # _bounded_ewm_correlation_matrix: total_budget = current capital *
    # target_portfolio_vol * IDM (IDM computed from that rebalance's own
    # signal-active symbols' REAL correlation, over a bounded trailing EWM
    # window -- corr_window_years/corr_halflife_days below), split across
    # those active symbols per notional_weighting below (flat by default).
    # At the degenerate zero-correlation case
    # this reduces exactly to
    # account_equity * target_portfolio_vol / sqrt(n_effective) -- the same
    # formula the live system's own compute_desired_risk_budget already
    # uses (see derivatives_bt_engine.domain.allocation) -- so this is a
    # strict generalization of that existing, live-validated formula to
    # the REAL measured correlation, not a novel scheme invented here.
    #
    # Scope: only affects the standard monthly-rebalance vol-targeted path
    # (the branch this module's documented overstated-vol finding was
    # actually measured on). Deliberately NOT wired into the pre-start_date
    # seed rebalance or the signal_gate_mode='daily' off-cycle path -- both
    # are separate, less-used code paths; extending this into them is a
    # deliberate follow-up, not an oversight, kept out of this change to
    # stay scoped to the diagnosed bug. Has no effect when fixed_quantities
    # is set (that mode never reads max_notional/notional_budget at all).
    #
    # Confirmed directly this FIXES the unbounded-overstatement direction of
    # the bug (a 12-symbol/$500k-notional run that overstated realized vol
    # at 25.37% against a 15% target came down to 7.16% with this on) but is
    # NOT precisely calibrated to target_portfolio_vol -- it undershot by
    # roughly 2x in that same test. Same root cause as the single-shot
    # calibration imprecision already documented in
    # tsmom_binary_vol_parity_backtest.py's own target_portfolio_vol
    # feature: a point-in-time bounded-window IDM/correlation estimate,
    # re-estimated at each rebalance, won't exactly match whatever
    # correlation structure the FULL backtest period actually realizes.
    # Treat this as "no longer structurally broken," not "hits its target
    # precisely" -- tightening that (e.g. an iterated rescale, matching the
    # sibling script's own calibration pattern) is a deliberate follow-up,
    # not implemented here.
    target_portfolio_vol: Optional[float] = None
    corr_window_years: float = 3.0
    corr_halflife_days: float = 63.0
    # How the total IDM-derived dollar-vol budget is split ACROSS active
    # symbols -- only matters when target_portfolio_vol is set. 'flat'
    # (default, unchanged prior behavior): every active symbol gets the
    # same budget, regardless of correlation structure. 'erc'/'hrp':
    # data-driven alternatives (allocation.compute_erc_weights/
    # compute_hrp_weights) that split by each symbol's OWN measured
    # correlation to the rest of the active set, so a correlated cluster
    # collectively earns roughly one undiversified bet's worth of budget
    # instead of each member separately claiming an equal share -- see
    # compute_symbol_notional_budget's own docstring for the full
    # derivation. When use_idm=True (below), this SAME split is also fed
    # into IDM as its weight vector -- not a flat 1/n vector regardless of
    # this choice, which would double-count diversification (IDM crediting
    # the active set's correlation once for the total, 'erc'/'hrp'
    # crediting it again for the split) -- see compute_symbol_notional_
    # budget's own docstring for the full argument.
    notional_weighting: str = 'flat'
    # Whether the total dollar-vol budget (capital * target_portfolio_vol)
    # gets scaled by IDM at all before being split per notional_weighting
    # above. True (default): total_budget = capital * target_portfolio_vol
    # * IDM, IDM computed with weights=<the notional_weighting split
    # itself> (see notional_weighting's own comment and compute_symbol_
    # notional_budget's docstring for why the weights must match). False:
    # total_budget = capital * target_portfolio_vol, no correlation-based
    # up/down-sizing of the total -- for 'erc'/'hrp' this isolates
    # diversification to the split alone instead of applying it twice at
    # two different levels; for 'flat' it removes correlation-awareness
    # from sizing altogether. Only matters when target_portfolio_vol is
    # set. See research/cta-layer-separation-risk-budgeting.md for the
    # IDM-vs-allocation-level framing this toggle exists to let you compare.
    use_idm: bool = True
    # Signal DIRECTION source -- 'continuous' (default, unchanged prior
    # behaviour): continuous_momentum's daily, vol-normalized trend_strength
    # + classify_regime(ts_fast, ts_slow) + a flat regime_discount in
    # Correction/Rebound. 'goulding': Goulding/Harvey/Mazzoleni (2023)'s own
    # monthly Bull/Correction/Bear/Rebound classification (goulding_monthly)
    # with a_Co/a_Re mixing weights re-estimated at EVERY rebalance from all
    # prior pooled history (domain.signal's build_monthly_state_return_
    # history/estimate_mixing_params, no lookahead) blending the slow/fast
    # direction in Correction/Rebound instead of a flat discount --
    # regime_discount is ignored in this mode (the a_Co/a_Re blend IS the
    # discount mechanism; applying a second flat one on top would double-
    # discount). Position SIZE/vol-targeting is unaffected either way --
    # this only changes which model decides the +-1/0 direction, mirroring
    # tsmom_binary_vol_parity_backtest.py's own weighting_mode='dynamic'
    # ("Goulding decides direction, vol-parity decides size"), now shared
    # via domain/signal.py instead of being that script's own local
    # implementation.
    signal_weighting: str = 'continuous'
    # Only matters when signal_weighting == 'goulding'. 'cluster' (default):
    # a_Co/a_Re re-estimated separately per instruments.py cluster (each
    # symbol using only its own cluster's pooled Correction/Rebound
    # history -- pooling across unrelated clusters would blend one asset
    # class's behavior into another's). 'global': one shared a_Co/a_Re
    # pooled across every symbol regardless of cluster, kept for direct
    # comparison. See estimate_mixing_params's own docstring.
    mixing_pool: str = 'cluster'
    # continuous_momentum's own return-horizon windows -- only matters
    # when signal_weighting == 'continuous' (goulding_monthly ignores
    # these entirely, using fast_months/slow_months instead, hardcoded
    # elsewhere -- not plumbed through this dataclass since nothing here
    # currently overrides them). vol_fast_window/vol_slow_window default
    # to None -> horizon-matched to fast_window/slow_window (see
    # continuous_momentum's own docstring on why that pairing matters:
    # ts_fast/ts_slow are horizon Sharpe-like statistics, and an
    # unmatched vol window breaks that clean n-day-return-over-n-day-vol
    # correspondence) -- pass them explicitly only to deliberately
    # decouple the two.
    fast_window: int = DEFAULT_FAST_WINDOW
    slow_window: int = DEFAULT_SLOW_WINDOW
    vol_fast_window: Optional[int] = None
    vol_slow_window: Optional[int] = None

    def __post_init__(self):
        if self.signal_gate_mode not in ('off', 'monthly', 'daily'):
            raise ValueError(f"signal_gate_mode must be 'off', 'monthly', or 'daily', got {self.signal_gate_mode!r}")
        if self.signal_weighting not in ('continuous', 'goulding'):
            raise ValueError(f"signal_weighting must be 'continuous' or 'goulding', got {self.signal_weighting!r}")
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("fast_window/slow_window must be positive")
        if self.fast_window >= self.slow_window:
            raise ValueError(f"fast_window ({self.fast_window}) must be < slow_window ({self.slow_window})")
        if self.mixing_pool not in ('cluster', 'global'):
            raise ValueError(f"mixing_pool must be 'cluster' or 'global', got {self.mixing_pool!r}")
        if self.notional_weighting not in NOTIONAL_WEIGHTING_SCHEMES:
            raise ValueError(f"notional_weighting must be one of {NOTIONAL_WEIGHTING_SCHEMES}, "
                              f"got {self.notional_weighting!r}")
        if self.fixed_quantities is not None and len(self.fixed_quantities) != len(self.symbols):
            raise ValueError(
                f"fixed_quantities must have exactly one entry per symbol (positional, same order): "
                f"got {len(self.fixed_quantities)} quantities for {len(self.symbols)} symbols "
                f"({self.symbols})"
            )


def check_vol_regime(vix_ratio: Optional[float]) -> VolRegime:
    """Normal | Elevated | Spike | Extreme from vix_current / vix_ma.

    Deliberately one-sided (mirrors derivatives_bt_engine.live.tsmom_rebalance's
    version): every threshold checks vix_ratio being HIGH -- no symmetric
    low-vix_ratio bucket, not an oversight. This is a portfolio-wide risk-
    management gate (feeds vix_scalar / the spike-extreme hold-
    or-halve bypass), not a regime-confidence detector. Per-instrument,
    asset-specific vol state (including a low-vol bucket) is a separate
    mechanism -- see SignalConfidenceRegime in signal.py."""
    if vix_ratio is None:
        return VolRegime.NORMAL
    if vix_ratio > VIX_EXTREME_RATIO:
        return VolRegime.EXTREME
    if vix_ratio > VIX_SPIKE_RATIO:
        return VolRegime.SPIKE
    if vix_ratio > VIX_ELEVATED_RATIO:
        return VolRegime.ELEVATED
    return VolRegime.NORMAL


def load_portfolio_data(symbols: list[str]) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Loads each symbol's continuous front-month OHLCV (via the existing
    FuturesDataLoader, parquet-cached) plus one shared spot-VIX series,
    read directly as polars (covers 1990-present, unlike the older
    pandas/CSV vix_file BaseDataLoader.vix_data still uses for the option
    path, which is stale past 2024-12-31).

    Some micros (MES, MNQ, MTN, ...) have no db history under their own
    symbol -- resolve_price_symbol borrows the full-size sibling's (ES,
    NQ, ZN, ...) via instruments.py's db_symbol field, so the cache/query
    below runs against the resolved symbol while price_data stays keyed by
    the raw traded symbol (matching futures_types/windowed elsewhere in
    this module, which always use the traded symbol for margin/commission/
    mult -- MES stays sized as MES, never as ES)."""
    cache_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.cache', 'futures'))
    os.makedirs(cache_dir, exist_ok=True)
    price_symbols = {s: resolve_price_symbol(s) for s in symbols}
    _validate_symbols_exist(set(price_symbols.values()), cache_dir)
    price_data = {s: FuturesDataLoader(asset=price_symbols[s], data_dir=cache_dir, use_preprocessed=True, save_preprocessed=True).daily
                  for s in symbols}
    # Defense-in-depth: FuturesDataLoader.daily already validates this at the
    # source (keyed by the resolved price symbol), but re-check here keyed by
    # the raw traded symbol (MES, not ES) so a failure is unambiguous about
    # which of this multi-symbol portfolio's series is the problem.
    for s, df in price_data.items():
        assert_monotonic_expiration(df, s)
    vix = pl.read_parquet(VIX_FILE_PATH).select(['date', 'close']).rename({'close': 'vix_close'}).sort('date')
    return price_data, vix


def _validate_symbols_exist(price_symbols, cache_dir: str) -> None:
    """The continuous-front-month query (FuturesDataLoader.daily) has no
    early-exit for a non-matching asset -- it's an unindexed full-table scan
    that takes minutes either way, so a typo'd or IB-only symbol (e.g. the
    live rebalance's IBKR ticker 'JPY'/'BRE' rather than this db's real
    CME/Globex root '6J'/'6L') would otherwise silently come back as an
    empty frame after minutes of waiting. Check the cheap `DISTINCT asset`
    list up front instead, skipping symbols that are already parquet-cached
    (no need to hit duckdb at all for those). Takes already-resolved price
    symbols (see load_portfolio_data), not raw traded symbols -- a micro
    like MES is expected to be absent from `daily` and shouldn't raise."""
    uncached = [s for s in price_symbols
                if not os.path.exists(os.path.join(cache_dir, f'{s}_daily.parquet'))]
    if not uncached:
        return
    # FuturesDataLoader.db_path is a dataclass field with a default_factory,
    # so it isn't readable as a class attribute (FuturesDataLoader.db_path
    # raises AttributeError) -- call the same factory instances use, so this
    # doesn't duplicate the GLOBEX_DB_PATH env-var/default logic.
    db_path = FuturesDataLoader.__dataclass_fields__['db_path'].default_factory()
    con = duckdb.connect(db_path, read_only=True)
    try:
        known = set(con.sql('SELECT DISTINCT asset FROM daily').pl()['asset'].to_list())
    finally:
        con.close()
    missing = [s for s in uncached if s not in known]
    if missing:
        raise ValueError(
            f'No data found for symbol(s) {missing} in the futures db -- '
            f'note this must be the real CME/Globex ticker (e.g. 6J, 6L, '
            f'6M), not whatever symbol IBKR uses for live contract resolution.'
        )


def _compute_vix_regime_series(vix: pl.DataFrame,
                                window_days: int = DEFAULT_VIX_MA_WINDOW_DAYS) -> pl.DataFrame:
    """Adds vix_ma / vix_ratio / vol_regime columns to a raw VIX frame --
    the column stays named vix_ma regardless of window_days (matches this
    project's live counterpart, derivatives_bt_engine.live.tsmom_rebalance's
    own vx_current/vx_ma fields, which keep their names the same way); only
    the rolling window LENGTH used to compute it varies."""
    vix = vix.with_columns(vix_ma=pl.col('vix_close').rolling_mean(window_days))
    vix = vix.with_columns(
        vix_ratio=pl.when(pl.col('vix_ma') > 0)
        .then(pl.col('vix_close') / pl.col('vix_ma'))
        .otherwise(None)
    )
    regimes = [check_vol_regime(r) for r in vix['vix_ratio'].to_list()]
    return vix.with_columns(vol_regime=pl.Series(regimes))


def _round(x: Optional[float], ndigits: int) -> Optional[float]:
    """round() that passes None through, for optional diagnostic fields
    that may not have been computable (e.g. gated/skipped rebalances)."""
    return None if x is None else round(x, ndigits)


def _signal_gate_reason(sig_val, ts_fast_val, ts_slow_val, is_long: bool, threshold: Optional[float],
                         exit_on_ts_crossover: bool) -> Optional[str]:
    """Same shape as TradeManager's per-position gate check
    (domain/trade_manager.py's _signal_gate_reason), standalone here since
    TSMOM has no per-position `self.config` to close over and this module
    is deliberately kept separate from TradeManager. Bails out (never
    gates) if either ts_fast or ts_slow is still null -- continuous_momentum
    only requires ts_fast to be non-null to emit a `signal` value at all, so
    early in any backtest window `signal` can look like a real number
    while being an unreliable ts_fast-only estimate."""
    if ts_fast_val is None or ts_slow_val is None:
        return None
    if threshold is not None and sig_val is not None:
        if is_long and sig_val < threshold:
            return 'signal_ts_threshold'
        if not is_long and sig_val > -threshold:
            return 'signal_ts_threshold'
    if exit_on_ts_crossover:
        if is_long and ts_fast_val < ts_slow_val:
            return 'signal_crossover'
        if not is_long and ts_fast_val > ts_slow_val:
            return 'signal_crossover'
    return None


def _apply_signal_gate(prior_contracts: int, proposed_target: int, result: dict,
                        config: TsmomBacktestConfig) -> tuple[int, Optional[str]]:
    """Overrides `proposed_target` to 0 if the signal gate fires. Direction
    is derived from whichever side actually matters: the CURRENTLY-HELD
    position's sign for the exit check (an existing long/short that's
    weakened), the PROPOSED target's sign for the entry check (a new
    position sizing wants to open, in that direction) -- not a fixed
    config field, since a TSMOM symbol's direction comes from its own
    signal sign, unlike a naked single-direction FuturesStrategyConfig.
    Returns (final_target, gate_reason)."""
    if config.signal_gate_mode == 'off':
        return proposed_target, None
    sig_val, ts_fast_val, ts_slow_val = result.get('signal'), result.get('ts_fast'), result.get('ts_slow')

    if prior_contracts != 0:
        is_long = prior_contracts > 0
        reason = _signal_gate_reason(sig_val, ts_fast_val, ts_slow_val, is_long,
                                      config.ts_exit_threshold, config.exit_on_ts_crossover)
        if reason is not None:
            return 0, reason
        if config.fixed_quantities is not None:
            # fixed_quantities has no "resize" concept -- magnitude is a
            # constant, so if the exit gate didn't fire, ANY difference
            # between prior_contracts and proposed_target here can only be
            # an implicit sign flip driven by the raw composite signal's
            # own sign changing independently of whatever specific exit
            # condition is configured (e.g. exit_on_ts_crossover checks
            # ts_fast-vs-ts_slow directly, which isn't guaranteed to coincide
            # with tanh(0.4*ts_fast+0.6*ts_slow) crossing zero) -- stay held,
            # unchanged; a flip must go through the gate, never happen
            # silently just because the vol-targeted path's own "let the
            # freshly computed target through" fallthrough doesn't
            # distinguish resizing from flipping.
            return prior_contracts, None

    if prior_contracts == 0 and proposed_target != 0:
        is_long = proposed_target > 0
        blocked = _signal_gate_reason(sig_val, ts_fast_val, ts_slow_val, is_long,
                                       config.ts_entry_threshold, config.exit_on_ts_crossover) is not None
        if blocked:
            return 0, 'signal_entry_blocked'

    return proposed_target, None


def _month_end_dates(price_data: dict[str, pl.DataFrame]) -> set[date]:
    """Last trading day of each calendar month, across the union of every
    loaded symbol's dates."""
    all_dates = sorted(set().union(*(set(df['ts_event'].to_list()) for df in price_data.values())))
    dates_df = pl.DataFrame({'ts_event': all_dates}).with_columns(
        ym=pl.col('ts_event').dt.strftime('%Y-%m')
    )
    month_ends = dates_df.group_by('ym').agg(pl.col('ts_event').max().alias('month_end'))
    return set(month_ends['month_end'].to_list())


def _detect_roll_dates(df: pl.DataFrame, start: date, end: date,
                        active_months: Optional[list[str]], symbol: str) -> list[date]:
    """Real per-symbol roll dates: every date within [start, end] where the
    continuous front-month series' own selected contract's `expiration`
    changes from the prior row -- an actual volume-driven front-month
    crossover in FuturesDataLoader.daily's sticky/volume-ranked query (see
    futures_dataloader.py's _CONTINUOUS_FRONT_MONTH_SQL), not a fixed
    calendar assumption applied uniformly regardless of a symbol's real
    roll cadence. Replaces the previous fixed-quarterly schedule
    (FuturesSignalGenerator._get_quarterly_roll_dates, still used
    unchanged by the naked single-symbol path), which was empirically
    confirmed wrong for a meaningful chunk of this project's universe: GC,
    SI, and the four grains each roll roughly monthly among their own 4-5
    real active months, not quarterly, and none of their active-month sets
    is a subset of Mar/Jun/Sep/Dec (see
    research/research_futures_roll_logic_and_active_months.md §2, §4.2).

    `df` must be the symbol's own unbounded (not date-windowed) continuous
    series -- the first row of any slice always looks like "a change" (its
    own prior row is unavailable), which would register a false roll right
    at whatever date happens to start the slice, if `df` had already been
    windowed before this runs.

    `active_months` (instruments.resolve_active_months(symbol), when
    confirmed) is used only as a validation guard on the DETECTED dates,
    not to generate them: each crossover's target contract's month-letter
    is checked against it, and a warning (not a raised error -- a hard
    failure here risks reintroducing the "stuck-forever roll" class of bug
    this project already hit once) is logged if a detected roll lands
    outside the confirmed active set. That mismatch is exactly the shape
    of a single spurious volume-spike hijacking the sticky series (the
    still-open BRE/6L bug, research doc §1.2) -- surfaced here rather than
    silently trusted, for every symbol this guard is available for, not
    just BRE.

    No-ops (returns an empty list, i.e. "no detected rolls") when `df` has
    no `expiration` column at all -- mirrors assert_monotonic_expiration's
    own defensive no-op for the same case (futures_dataloader.py), e.g. a
    hand-built or synthetic price series (some of this module's own test
    fixtures) that never carried contract-level metadata to begin with."""
    if 'expiration' not in df.columns:
        return []
    d = df.sort('ts_event')
    changed = d.filter(pl.col('expiration') != pl.col('expiration').shift(1))
    changed = changed.filter((pl.col('ts_event') >= start) & (pl.col('ts_event') <= end))
    if active_months:
        for row in changed.iter_rows(named=True):
            letter = CME_MONTH_NUM_TO_LETTER.get(row['expiration'].month)
            if letter is not None and letter not in active_months:
                logger.warning(
                    "%s: detected roll on %s into a contract expiring %s (month %s) -- outside "
                    "this symbol's confirmed active_months %s; possible spurious volume-spike "
                    "crossover rather than a genuine roll.",
                    symbol, row['ts_event'], row['expiration'], letter, active_months,
                )
    return changed['ts_event'].to_list()


def _vix_regime_at(vix: pl.DataFrame, d: date) -> tuple[VolRegime, Optional[float], Optional[float]]:
    """(vol_regime, vix_close, vix_ratio) as of the latest available VIX
    row at or before `d`. (Normal, None, None) if no VIX data is
    available yet."""
    row = vix.filter(pl.col('date') <= d).tail(1)
    if row.height == 0:
        return VolRegime.NORMAL, None, None
    return VolRegime(row['vol_regime'][0]), row['vix_close'][0], row['vix_ratio'][0]


def _compute_signal_row(symbol: str, precomputed: dict[str, pl.DataFrame], d: date,
                         futures_types: dict[str, dict], config: TsmomBacktestConfig,
                         vix_scalar: float, annualization_days: int,
                         notional_budget: Optional[float] = None,
                         g_regime_val: Optional[str] = None, g_fast_val: Optional[float] = None,
                         g_slow_val: Optional[float] = None, a_co: Optional[float] = None,
                         a_re: Optional[float] = None) -> Optional[dict]:
    """Signal + vol-targeted (or fixed-quantity) sizing for one symbol as of
    date `d`, reading from `precomputed` -- each symbol's full
    continuous_momentum output, computed ONCE for the whole unbounded
    history (see run_tsmom_backtest). continuous_momentum's rolling/
    diff functions are strictly backward-looking, so a given date's row is
    identical whether computed from the full series or from a series
    truncated to that date -- precomputing once and looking up by date is
    exactly equivalent to (and far cheaper than) this function's old
    per-call recompute, which used to run fresh at every rebalance and
    would otherwise need to run for every symbol on every calendar day to
    support daily entry/exit checking. None if there isn't yet enough
    history for a signal at all (continuous_momentum's own `signal`
    column is null until ts_fast has 63 bars, goulding_monthly's `g_regime`
    is null until fast/slow have enough completed months).

    notional_budget: None (default) uses config.max_notional, exactly as
    before. A caller doing correlation-aware sizing (see
    run_tsmom_backtest's own target_portfolio_vol handling) passes an
    explicit per-rebalance, IDM-derived override instead -- only affects
    the non-fixed_quantities branch below (fixed_quantities' own sizing
    never reads max_notional/notional_budget at all).

    g_regime_val/g_fast_val/g_slow_val/a_co/a_re: only read when
    config.signal_weighting == 'goulding' -- the caller resolves these from
    its own precomputed, forward-matched goulding_monthly output and that
    rebalance date's own (per-cluster or global) estimate_mixing_params
    result, since a_Co/a_Re is shared across every symbol in a cluster and
    only needs estimating once per rebalance date, not once per symbol
    call. g_fast_val/g_slow_val here are goulding_monthly's own `fast`/
    `slow` -- Goulding's lagged trailing-average momentum signals (already
    shift(1)'d, no lookahead), NOT the realized return of the period about
    to be traded; see _goulding_direction's own docstring for why that
    distinction matters."""
    row = precomputed[symbol].filter(pl.col('ts_event') == d)
    if row.height == 0:
        return None

    def _col(name):
        return row[name][0] if name in row.columns else None

    # Sizing inputs -- daily_std_last/hv/risk_scalar/close/dd -- ALWAYS
    # come from continuous_momentum's own output regardless of
    # signal_weighting: "Goulding decides direction, vol-parity decides
    # size" (mirrors tsmom_binary_vol_parity_backtest.py's own
    # weighting_mode='dynamic' design), not a literal end-to-end
    # reproduction of the paper's own portfolio construction (which
    # doesn't size positions at all -- it studies raw dynamic-blend
    # RETURNS as a standalone series). ts_fast/ts_slow/avg_r_fast/
    # avg_r_slow are read here unconditionally too (not just in
    # 'continuous' mode) so the returned diagnostic dict always has both
    # models' readings side by side for comparison, matching that same
    # script's own "both models computed unconditionally" convention.
    ts_fast, ts_slow = _col('ts_fast'), _col('ts_slow')
    daily_std_last = _col('std_fast')
    last_close = _col('close')
    # `dd` is already a (close - peak) / peak fraction from
    # continuous_momentum -- express as a percentage to match
    # `stats`' own drawdown_pct convention.
    dd_raw = _col('dd')
    dd_pct = dd_raw * 100 if dd_raw is not None else None
    hv = daily_std_last * math.sqrt(annualization_days) if daily_std_last and daily_std_last > 0 else None
    risk_scalar = max(0.25, min(2.0, config.vol_target / hv)) if hv else 1.0

    # resolve_trend_direction (domain.signal) -- shared with
    # live.tsmom_rebalance's own per-instrument signal computation, so both
    # modules implement this continuous-vs-goulding branch exactly once.
    resolved = resolve_trend_direction(config.signal_weighting, _col('signal'), ts_fast, ts_slow,
                                        config.regime_discount, g_regime_val, g_fast_val, g_slow_val,
                                        a_co, a_re)
    if resolved is None:
        return None
    trend_strength, regime, regime_discount, g_blend = resolved

    signal_for_scalar = trend_strength
    if config.long_only and signal_for_scalar is not None and not (
        isinstance(signal_for_scalar, float) and math.isnan(signal_for_scalar)
    ):
        signal_for_scalar = max(0.0, signal_for_scalar)

    scalar = compute_position_scalar(
        signal_for_scalar, daily_std_last, config.vol_target, regime,
        regime_discount=regime_discount, annualization_days=annualization_days,
    ) * vix_scalar

    mult = futures_types[symbol]['multiplier']
    if config.fixed_quantities is not None:
        # No-rebalancing mode: direction is still signal-driven (there's no
        # other principled way to know when to go short without it), but
        # magnitude is this symbol's own fixed contract count -- scaled by
        # vix_scalar (the same elevated-VIX de-risking the vol-
        # targeted path applies) and rounded, not derived from vol_target/
        # max_notional at all. max_contracts still clamps as a sanity
        # backstop (raise it if it's below your configured fixed_quantities
        # -- it silently truncates otherwise, same as the vol-targeted path).
        fixed_qty = config.fixed_quantities[config.symbols.index(symbol)]
        if signal_for_scalar is None or (isinstance(signal_for_scalar, float) and math.isnan(signal_for_scalar)) or signal_for_scalar == 0:
            target = 0
        else:
            direction = 1 if signal_for_scalar > 0 else -1
            target = direction * round(fixed_qty * vix_scalar)
    else:
        budget = notional_budget if notional_budget is not None else config.max_notional
        one_contract_notional = last_close * mult
        target = round((budget * scalar) / one_contract_notional) if one_contract_notional else 0
    target = max(-config.max_contracts, min(config.max_contracts, target))

    return {
        'target': target, 'signal': trend_strength, 'regime': regime,
        # scalar itself (pre-notional-conversion, post-vix_scalar) --
        # not printed/logged anywhere before this, needed by
        # run_tsmom_backtest's target_portfolio_vol handling to decide which
        # symbols are genuinely signal-active (scalar != 0) independent of
        # any particular notional_budget's own rounding, since a symbol that
        # rounds to 0 contracts at one budget can still be "active" at a
        # bigger one (see run_tsmom_backtest's own two-pass comment).
        'scalar': scalar,
        'hv': hv, 'risk_scalar': risk_scalar * vix_scalar, 'regime_discount': regime_discount,
        'close': last_close, 'dd_pct': dd_pct,
        # Raw signal-row fields, straight from continuous_momentum, purely
        # for debugging/sanity-checking the sizing math end to end.
        # fast_return/slow_return named r_fast/r_slow in continuous_momentum's
        # own output -- kept under their old dict keys here since downstream
        # consumers (e.g. line ~655's _round(s.get(...))) already expect them.
        'peak': _col('peak'), 'avg_r_fast': _col('avg_r_fast'), 'avg_r_slow': _col('avg_r_slow'),
        'fast_return': _col('r_fast'), 'slow_return': _col('r_slow'), 'ts_fast': ts_fast, 'ts_slow': ts_slow,
        'r1y_pct': _col('r1y_pct'),
        # Goulding audit fields -- None in 'continuous' mode (nothing to
        # report), populated in 'goulding' mode so a saved trend_signals
        # CSV shows exactly what drove that rebalance's direction: this
        # rebalance's cluster's own a_Co/a_Re as of this date, the raw
        # g_fast/g_slow/g_regime inputs _goulding_direction blended, and
        # g_blend itself -- the raw pre-sign eq. 7 value (resolve_trend_
        # direction's own 4th return, via _goulding_direction), None in
        # Bull/Bear (eq. 7 doesn't apply there) even though trend_strength
        # still resolves in that case.
        'g_regime': g_regime_val, 'g_fast': g_fast_val, 'g_slow': g_slow_val,
        'g_blend': g_blend, 'a_co': a_co, 'a_re': a_re,
    }


def _mixing_params_for_date(config: TsmomBacktestConfig, monthly_history: Optional[pl.DataFrame],
                             d: date) -> dict[str, tuple[float, float]]:
    """{cluster: (a_co, a_re)} as of rebalance date `d` -- estimated once
    per rebalance date (shared across every symbol in that cluster), not
    once per symbol. Empty dict (never read) outside 'goulding' mode."""
    if config.signal_weighting != 'goulding':
        return {}
    clusters_needed = {get_spec(s)['cluster'] for s in config.symbols}
    if config.mixing_pool == 'cluster':
        return {c: estimate_mixing_params(monthly_history, d, c) for c in clusters_needed}
    global_params = estimate_mixing_params(monthly_history, d, None)
    return {c: global_params for c in clusters_needed}


def _goulding_kwargs_for(config: TsmomBacktestConfig, rebal_monthly: dict[str, pl.DataFrame],
                         symbol: str, d: date, mixing_params_by_cluster: dict[str, tuple[float, float]]) -> dict:
    """This symbol's own g_regime_val/g_fast_val/g_slow_val/a_co/a_re as of
    `d`, ready to **-unpack straight into _compute_signal_row. Empty dict
    outside 'goulding' mode -- _compute_signal_row's own defaults (all
    None) then apply, matching its 'continuous'-mode behaviour exactly."""
    if config.signal_weighting != 'goulding':
        return {}
    g_row = rebal_monthly[symbol].filter(pl.col('ts_event') == d)
    g_regime_val = g_row['g_regime'][0] if g_row.height else None
    g_fast_val = g_row['g_fast'][0] if g_row.height else None
    g_slow_val = g_row['g_slow'][0] if g_row.height else None
    a_co, a_re = mixing_params_by_cluster.get(get_spec(symbol)['cluster'], (0.5, 0.5))
    return {'g_regime_val': g_regime_val, 'g_fast_val': g_fast_val, 'g_slow_val': g_slow_val,
            'a_co': a_co, 'a_re': a_re}


class _PortfolioLedger:
    """Owns every piece of run_tsmom_backtest's own mutable portfolio
    state (capital, held contracts, last-seen close, open trade spans,
    and the events/transactions/trades logs) plus every mutation of it
    (mark-to-market, rebalance, roll, close). Previously these were five
    separate closures (_close_trade/_rebalance_to/_process_roll plus the
    plain dicts/lists they closed over via `nonlocal`) defined inline
    inside run_tsmom_backtest, ahead of the day loop -- readable in
    isolation but, taken together with the goulding-mixing helpers, over
    200 lines a reader had to scroll past before reaching the loop itself.
    Pulling them out here lets the day loop read as a sequence of named
    steps against `ledger` instead."""

    def __init__(self, symbols: list[str], initial_capital: float, futures_types: dict[str, dict]):
        self.futures_types = futures_types
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.held_contracts: dict[str, int] = {s: 0 for s in symbols}
        self.prior_close: dict[str, Optional[float]] = {s: None for s in symbols}
        # Per-symbol open trade accumulator: None when flat, else a dict
        # tracking the currently-open span's entry info plus running
        # realized MTM pnl/fees, closed out (appended to `trades`) on a
        # return to flat, a direct sign flip, or a final force-close after
        # the day loop ends.
        self.open_trade: dict[str, Optional[dict]] = {s: None for s in symbols}
        self.events: list[dict] = []
        self.transactions: list[dict] = []
        self.trades: list[dict] = []

    def mark_to_market(self, symbol: str, close: float) -> None:
        """Today's close vs yesterday's, the same diff-based daily MTM
        approach Backtester.calculate_futures_mtm_drawdown already uses
        for single-symbol. No-op on capital/open_trade the first time a
        symbol is seen (prior_close still None) -- prior_close is still
        recorded so the NEXT call has something to diff against."""
        if self.prior_close[symbol] is not None and self.held_contracts[symbol] != 0:
            day_pnl = (self.held_contracts[symbol] * (close - self.prior_close[symbol])
                       * self.futures_types[symbol]['multiplier'])
            self.capital += day_pnl
            ot = self.open_trade[symbol]
            if ot is not None:
                ot['mtm_pnl'] += day_pnl
        self.prior_close[symbol] = close

    def close_trade(self, symbol: str, exit_date: date, exit_price: Optional[float]) -> None:
        ot = self.open_trade[symbol]
        if ot is None:
            return
        net_pnl = round(ot['mtm_pnl'] - ot['fees'], 2)
        self.trades.append({
            'symbol': symbol, 'direction': ot['direction'],
            'entry_date': ot['entry_date'], 'entry_price': _round(ot['entry_price'], _PRICE_ROUND_NDIGITS),
            'exit_date': exit_date, 'exit_price': _round(exit_price, _PRICE_ROUND_NDIGITS),
            'days_held': (exit_date - ot['entry_date']).days,
            'max_contracts': ot['max_contracts'],
            # Total contracts shed via MID-TRADE resizes (same-direction
            # downsizes), NOT counting the final close itself -- 0 whenever
            # the position was held at a constant size its whole life.
            # Exists specifically so entry_price/exit_price/max_contracts
            # alone don't invite a naive (exit-entry)*max_contracts*mult
            # sanity check that silently overstates PnL whenever the
            # position was actually smaller for part of its life (confirmed
            # directly: MZW opened at 7, resized down to 1 partway through,
            # and the naive full-max_contracts calc overstated the real PnL
            # by exactly the exposure lost in that resize) -- a nonzero
            # value here is a direct signal that max_contracts wasn't held
            # the entire time, so a manual check needs transactions.csv's
            # own resize-by-resize history, not this summary row alone.
            'lots_closed_pre_exit': ot['lots_closed_pre_exit'],
            'fees': round(ot['fees'], 2),
            'pnl': net_pnl, 'close_reason': ot['close_reason'],
        })
        self.open_trade[symbol] = None

    def rebalance_to(self, symbol: str, target: int, rebalance_date: date, vol_regime,
                      vix_close=None, vix_ratio=None, signal: Optional[dict] = None, is_seed=False) -> None:
        """`signal` is _compute_signal_row's full result dict (or None on a
        spike/extreme-gated event, where signal computation is skipped
        entirely -- every signal-derived field below is then None, same
        as before)."""
        s = signal or {}
        prior = self.held_contracts[symbol]
        fee = 0.0
        if target != prior:
            # Commission is charged only on the quantity actually closed out
            # -- opening a position, or adding to one, is free (matches
            # FuturesPosition.calculate_pnl / _process_roll's convention:
            # fees = commission * 2 * quantity, charged entirely at close).
            # A flip closes the *entire* prior side (the new opposite-
            # direction open that follows is then free, same as any other
            # open); a same-direction resize only charges for the portion
            # that shrinks back toward zero, not the portion added.
            if prior == 0:
                closed_qty = 0
            elif target == 0 or (prior > 0) != (target > 0):
                closed_qty = abs(prior)
            else:
                closed_qty = max(0, abs(prior) - abs(target))
            fee = self.futures_types[symbol]['commission'] * 2 * closed_qty
            self.capital -= fee
            price = s.get('close')
            self.transactions.append({
                'symbol': symbol, 'date': rebalance_date,
                'action': 'buy' if target > prior else 'sell',
                'quantity': abs(target - prior), 'price': _round(price, _PRICE_ROUND_NDIGITS),
                'fee': round(fee, 2), 'prior_contracts': prior, 'target_contracts': target,
                'gate_reason': s.get('gate_reason'), 'is_seed': is_seed,
            })

            flipped = prior != 0 and target != 0 and (prior > 0) != (target > 0)
            if flipped or target == 0:
                # This transaction's fee is the closing leg's cost alone (the
                # whole prior side on a flip -- the new opposite-direction
                # open that follows is free, same as any other open) --
                # must be folded in before close_trade reads ot['fees'], or
                # it's silently dropped from the trade's own total (portfolio
                # capital stays correct regardless, since that deduction
                # already happened above; only the per-trade fees/pnl
                # fields were at risk of undercounting).
                ot = self.open_trade[symbol]
                if ot is not None:
                    ot['fees'] += fee
                self.close_trade(symbol, rebalance_date, price)
            ot = self.open_trade[symbol]  # re-read post-close: close_trade above may have just cleared it
            if target != 0 and ot is None:
                self.open_trade[symbol] = {
                    'entry_date': rebalance_date, 'entry_price': price,
                    'direction': 'long' if target > 0 else 'short',
                    'max_contracts': abs(target), 'mtm_pnl': 0.0,
                    'fees': 0.0 if flipped else fee,
                    'close_reason': None, 'lots_closed_pre_exit': 0,
                }
            elif target != 0 and ot is not None:
                # Resize within the same direction -- extend the existing
                # span rather than starting a new trade; fold in this
                # resize's own fee (zero unless this shrank toward zero),
                # track the largest size held, and accumulate any quantity
                # shed by a downsize (closed_qty, computed above) into
                # lots_closed_pre_exit -- see close_trade's own comment for
                # why this is tracked separately from max_contracts.
                ot['fees'] += fee
                ot['max_contracts'] = max(ot['max_contracts'], abs(target))
                ot['lots_closed_pre_exit'] += closed_qty
            if (flipped or target == 0) and s.get('gate_reason'):
                # close_trade already ran above and cleared open_trade;
                # the reason belongs on the trade that just closed, so
                # patch the just-appended row rather than re-opening it.
                self.trades[-1]['close_reason'] = s.get('gate_reason')

        self.held_contracts[symbol] = target
        self.events.append({
            'date': rebalance_date, 'symbol': symbol,
            'close': _round(s.get('close'), _PRICE_ROUND_NDIGITS), 'peak': _round(s.get('peak'), _PRICE_ROUND_NDIGITS),
            'dd_pct': _round(s.get('dd_pct'), 2),
            'avg_r_fast': _round(s.get('avg_r_fast'), 4), 'avg_r_slow': _round(s.get('avg_r_slow'), 4),
            'fast_return': _round(s.get('fast_return'), 4), 'slow_return': _round(s.get('slow_return'), 4),
            'ts_fast': _round(s.get('ts_fast'), 4), 'ts_slow': _round(s.get('ts_slow'), 4),
            'signal': _round(s.get('signal'), 4), 'r1y_pct': _round(s.get('r1y_pct'), 2),
            'regime': s.get('regime'), 'vix_close': _round(vix_close, 2), 'vix_ratio': _round(vix_ratio, 4),
            'vol_regime': vol_regime, 'hv': _round(s.get('hv'), 4), 'risk_scalar': _round(s.get('risk_scalar'), 4),
            'regime_discount': _round(s.get('regime_discount'), 2),
            'prior_contracts': prior, 'target_contracts': target, 'is_seed': is_seed,
            'gate_reason': s.get('gate_reason'),
            # Goulding audit fields -- None in 'continuous' mode (nothing to
            # report; _compute_signal_row's own result dict already carries
            # None for all of these there), populated in 'goulding' mode.
            # Previously computed by _compute_signal_row but never actually
            # threaded through to this event dict, so a saved trend_signals
            # CSV in goulding mode silently never showed what drove a given
            # rebalance's direction, despite _compute_signal_row's own
            # docstring promising exactly that.
            'g_regime': s.get('g_regime'), 'g_fast': _round(s.get('g_fast'), 4),
            'g_slow': _round(s.get('g_slow'), 4),
            'a_co': _round(s.get('a_co'), 4), 'a_re': _round(s.get('a_re'), 4),
            # Portfolio-level capital snapshot as of this event (after
            # today's mark-to-market and this event's own commission fee,
            # both already applied above) -- previously only available in
            # the separate daily `stats` table (keyed by date only, not
            # per-symbol), so reading an event required cross-referencing
            # a different table by date to see its $ context.
            'capital': round(self.capital, 2),
            'cum_pnl': round(self.capital - self.initial_capital, 2),
        })

    def process_roll(self, symbol: str, roll_date: date) -> None:
        """Mandatory quarterly contract roll for a currently-held symbol:
        close the expiring contract (full round-trip commission on its own
        quantity, close_reason='roll') and immediately reopen the identical
        size under the new contract at the same price -- net zero PnL/size
        effect, cost is the fee alone. Mirrors FuturesPosition.close()
        (position.py:1785-1877): a roll is a mechanical consequence of the
        contract's own expiration, not a signal decision, so it fires
        regardless of signal_gate_mode/fixed_quantities and doesn't touch
        held_contracts or go through rebalance_to (which would incorrectly
        charge a second commission for the "reopen" leg -- opening a
        position is free in this fee model, only closing charges, exactly
        as in FuturesPosition.calculate_pnl)."""
        prior = self.held_contracts[symbol]
        if prior == 0:
            return
        price = self.prior_close[symbol]
        fee = self.futures_types[symbol]['commission'] * 2 * abs(prior)
        self.capital -= fee
        self.transactions.append({
            'symbol': symbol, 'date': roll_date, 'action': 'roll',
            'quantity': abs(prior), 'price': _round(price, _PRICE_ROUND_NDIGITS),
            'fee': round(fee, 2), 'prior_contracts': prior, 'target_contracts': prior,
            'gate_reason': None, 'is_seed': False,
        })
        ot = self.open_trade[symbol]
        if ot is not None:
            ot['fees'] += fee
            ot['close_reason'] = 'roll'
            self.close_trade(symbol, roll_date, price)
        self.open_trade[symbol] = {
            'entry_date': roll_date, 'entry_price': price,
            'direction': 'long' if prior > 0 else 'short',
            'max_contracts': abs(prior), 'mtm_pnl': 0.0, 'lots_closed_pre_exit': 0,
            'fees': 0.0, 'close_reason': None,
        }


def run_tsmom_backtest(config: TsmomBacktestConfig) -> dict:
    """Runs the monthly-rebalance TSMOM backtest. Returns a dict with
    'daily_mtm' (daily portfolio capital/drawdown, polars DataFrame -- same
    key name as the naked single-position path's Backtester.run() result),
    'trend_signals' (per-rebalance trend/signal diagnostic log: ts_fast, ts_slow,
    regime, risk_scalar, regime_discount, gate_reason, etc., list of
    dicts), 'transactions' (one row per
    rebalance that actually changed a symbol's contract count -- what was
    bought/sold, when, at what price/fee), and 'trades' (reconstructed
    round-trips: TSMOM has no discrete open/close lifecycle like
    FuturesPosition -- a symbol's exposure is continuously resized, not
    "opened then closed" -- so a trade here is defined as one continuous
    span of nonzero exposure in a single direction: 0->nonzero opens it,
    nonzero->0 or a direct sign flip closes it; resizing within the same
    direction extends the same trade rather than starting a new one. The
    quarterly contract roll is the one exception forced regardless of
    exposure direction: a held span is always closed and immediately
    reopened at each scheduled roll date (close_reason='roll'), same as
    FuturesPosition's own roll_date handling, so 'trades' never reports a
    holding period spanning more than one actual futures contract even
    when the signal itself never triggers a close)."""
    # `full_price_data` stays unbounded -- continuous_momentum's 252-day
    # lookback needs real history before config.start_date, not just
    # whatever falls inside the requested window. Only the iterated date
    # range (and what counts as a rebalance/MTM date) is bounded.
    full_price_data, vix = load_portfolio_data(config.symbols)
    vix = _compute_vix_regime_series(vix, config.vix_ma_window_days)
    # Real trading-days/year per symbol (instruments.resolve_annualization_days)
    # -- this project's confirmed universe splits 252 (CBOT grains) vs. 259
    # (everything else checked, post Sunday-session-merge fix); anything
    # unconfirmed falls back to 252 (DEFAULT_ANNUALIZATION_DAYS), unchanged
    # from this module's own prior universal-252 behavior.
    annualization_by_symbol = {s: resolve_annualization_days(s) for s in config.symbols}
    # Precomputed once per symbol, unconditionally (not just for
    # signal_gate_mode == 'daily') -- see _compute_signal_row's own
    # docstring for why this is exactly equivalent to (and much cheaper
    # than) recomputing continuous_momentum fresh at every rebalance.
    precomputed = {
        s: continuous_momentum(build_features(full_price_data[s].sort('ts_event')),
                                fast_window=config.fast_window, slow_window=config.slow_window,
                                vol_fast_window=config.vol_fast_window, vol_slow_window=config.vol_slow_window,
                                annualization_days=annualization_by_symbol[s])
        for s in config.symbols
    }
    futures_types = {s: get_spec(s) for s in config.symbols}
    # Built once (reusing full_price_data already loaded above), reused at
    # every rebalance date's own bounded-window slice -- only when
    # target_portfolio_vol is actually set, since this is an extra
    # inner-join + pct_change pass over every symbol's full history that
    # the module's original (default) sizing has no use for.
    returns_wide = build_returns_wide(full_price_data) if config.target_portfolio_vol is not None else None

    windowed = {}
    for symbol, df in full_price_data.items():
        if config.start_date:
            df = df.filter(pl.col('ts_event') >= config.start_date)
        if config.end_date:
            df = df.filter(pl.col('ts_event') <= config.end_date)
        windowed[symbol] = df.sort('ts_event')

    all_dates = sorted(set().union(*(set(df['ts_event'].to_list()) for df in windowed.values())))
    # A position opened on the very last date in the window has zero
    # subsequent days to mark to market -- it shows up as a pure commission
    # cost with no chance of P&L, which is misleading rather than meaningful.
    # Mirrors Backtester's own bounded-window handling (backtester.py:152-160)
    # in spirit: don't let the backtest take an action it can't show the
    # result of within the requested range.
    rebalance_dates = _month_end_dates(windowed) - ({all_dates[-1]} if all_dates else set())
    # Real per-symbol roll dates -- every date THIS symbol's own continuous
    # series actually switches contracts (volume-ranked/sticky crossover,
    # see _detect_roll_dates), not a single fixed calendar schedule shared
    # uniformly across every symbol regardless of its real roll cadence.
    # Computed from full_price_data (unbounded history), not windowed, so
    # the window's own start date can't masquerade as a false "roll" -- see
    # _detect_roll_dates' own docstring. active_months (when confirmed) is
    # consulted only as a validation guard on the detected dates here, not
    # to generate them.
    roll_dates_by_symbol: dict[str, set[date]] = {
        s: set(_detect_roll_dates(full_price_data[s], all_dates[0], all_dates[-1],
                                   resolve_active_months(s), s))
        for s in config.symbols
    } if all_dates else {s: set() for s in config.symbols}

    # Precomputed once, only when signal_weighting == 'goulding' -- Goulding's
    # own genuine calendar-month Bull/Correction/Bear/Rebound classification
    # (goulding_monthly), forward-matched to each rebalance date (a rebalance
    # on month-end date `d` decides what to hold GOING FORWARD, i.e. during
    # the NEXT month, so it needs the NEXT month's own bucket -- computed
    # from the just-completed month's data -- not the bucket already in
    # effect on `d` itself; strategy='forward' finds the first monthly label
    # >= d, which is always exactly that next month's bucket since a rebal
    # date always falls strictly inside its own month), plus the pooled,
    # expanding-window a_Co/a_Re estimation history built from every
    # symbol's forward-matched buckets. Mirrors
    # tsmom_binary_vol_parity_backtest.py's own construction (see that
    # script's run() for the fuller rationale), reusing domain/signal.py's
    # shared build_monthly_state_return_history/estimate_mixing_params
    # instead of a duplicate implementation.
    rebal_monthly: dict[str, pl.DataFrame] = {}
    monthly_history: Optional[pl.DataFrame] = None
    if config.signal_weighting == 'goulding':
        rebal_dates_sorted = sorted(rebalance_dates)
        rebal_dates_df = pl.DataFrame({'ts_event': rebal_dates_sorted}).sort('ts_event')
        for s in config.symbols:
            feat = build_features(full_price_data[s].sort('ts_event'))
            monthly = goulding_monthly(feat, **SignalSpec.goulding().goulding_kwargs())
            monthly = monthly.rename({'fast': 'g_fast', 'slow': 'g_slow', 'regime': 'g_regime'})
            monthly = monthly.select(['ts_event', 'ret', 'g_fast', 'g_slow', 'g_regime']).sort('ts_event')
            rebal_monthly[s] = rebal_dates_df.join_asof(monthly, on='ts_event', strategy='forward')
        cluster_by_symbol = {s: get_spec(s)['cluster'] for s in config.symbols}
        monthly_history = build_monthly_state_return_history(rebal_monthly, rebal_dates_sorted, cluster_by_symbol)

    ledger = _PortfolioLedger(config.symbols, config.initial_capital, futures_types)
    daily_rows = []

    # Seed the position from the last completed month-end *before*
    # start_date, using full unbounded history -- otherwise the backtest
    # starts flat and wastes its entire first calendar month sitting in
    # cash even when a perfectly valid prior-month signal already called
    # for a position, only entering at the window's first in-range
    # month-end. This makes start_date behave like "continuing an
    # already-running strategy," not "day one of trading."
    #
    # Deliberately NOT wired up for signal_weighting == 'goulding' (no
    # _goulding_kwargs_for call below, same scope limitation
    # target_portfolio_vol's own docstring already documents for this
    # block) -- seed_date falls strictly before config.start_date, outside
    # rebal_monthly's own forward-matched date range (built only over the
    # windowed rebalance_dates), so g_regime_val would always be missing
    # here regardless. Rather than extend the goulding precompute to cover
    # a date range it doesn't otherwise need, 'goulding' mode simply starts
    # flat (no pre-existing seed position) and takes its first real
    # position at the window's first in-range monthly rebalance instead --
    # a real gap, not a crash, and confirmed not to affect anything past
    # the first month or two of any reasonably long backtest.
    if config.start_date:
        prior_month_ends = [d for d in _month_end_dates(full_price_data) if d < config.start_date]
        if prior_month_ends:
            seed_date = max(prior_month_ends)
            vol_regime, vix_close, vix_ratio = _vix_regime_at(vix, seed_date)
            if not config.vix_gating:
                vol_regime = VolRegime.NORMAL  # vix_close/vix_ratio still logged, just not acted on
            if vol_regime not in (VolRegime.SPIKE, VolRegime.EXTREME):  # held_contracts are all 0 here -- hold/halve would be a no-op anyway
                vix_scalar = VIX_ELEVATED_SCALE if vol_regime == VolRegime.ELEVATED else 1.0
                for symbol in config.symbols:
                    result = _compute_signal_row(symbol, precomputed, seed_date, futures_types, config,
                                                  vix_scalar, annualization_by_symbol[symbol])
                    if result is None:
                        continue
                    target, gate_reason = _apply_signal_gate(ledger.held_contracts[symbol], result['target'], result, config)
                    ledger.rebalance_to(symbol, target, seed_date, vol_regime, vix_close=vix_close,
                                        vix_ratio=vix_ratio, signal={**result, 'gate_reason': gate_reason}, is_seed=True)
                    if ledger.held_contracts[symbol] != 0:
                        # Confirmed bug fix (2026-07): without this,
                        # prior_close[symbol] stays None going into the day
                        # loop below, whose own "1. Mark existing holdings"
                        # step skips day 1's mark-to-market entirely when
                        # prior_close is None and only sets prior_close
                        # AFTER that skip, to the first IN-WINDOW day's own
                        # close -- silently dropping the ENTIRE
                        # seed_date -> first-in-window-day price move from
                        # PnL, while the trade record's own entry_price
                        # still (correctly) shows the seed's real price.
                        # Confirmed directly: a seeded MES long recorded
                        # entry 3748.75 -> exit 3696.5 (a real
                        # -52.25pt/-$783.75 loss on 3 contracts) but
                        # reported pnl=+$45.09 -- exactly the number that
                        # results from silently substituting the first
                        # trading day's close as the effective entry price
                        # instead of the seed's own. result['close'] here is
                        # the SAME value rebalance_to just used as this
                        # trade's entry_price (signal={**result, ...} ->
                        # s.get('close')), so this guarantees they agree.
                        ledger.prior_close[symbol] = result['close']

    for d in all_dates:
        # 1. Mark existing holdings to market: today's close vs yesterday's,
        # the same diff-based daily MTM approach Backtester.
        # calculate_futures_mtm_drawdown already uses for single-symbol.
        for symbol in config.symbols:
            row = windowed[symbol].filter(pl.col('ts_event') == d)
            if row.height == 0:
                continue
            ledger.mark_to_market(symbol, row['close'][0])

        # 1.25. Mandatory per-symbol contract roll -- unconditional (not
        # gated by signal_gate_mode/fixed_quantities), since it's a
        # mechanical consequence of the contract's own expiration, exactly
        # like the naked path's FuturesPosition.roll_date. Each symbol rolls
        # on its OWN detected real crossover dates (roll_dates_by_symbol --
        # see _detect_roll_dates), not a single calendar schedule shared
        # across every symbol: a plain set-membership check is enough here
        # (no monotonic pointer needed) since each symbol's own set is
        # already restricted to real dates present in its own continuous
        # series.
        for symbol in config.symbols:
            if d in roll_dates_by_symbol[symbol]:
                ledger.process_roll(symbol, d)

        # 1.5. Daily signal-gate check (signal_gate_mode == 'daily' only),
        # off-cycle from the monthly resize below -- BOTH entry and exit,
        # not exit-only: a currently-held symbol can only be flattened here
        # (its own resize/magnitude stays monthly-only in both modes, so a
        # weakening vol-targeted position doesn't get continuously
        # rebalanced mid-month just because this loop now also checks
        # daily); a currently-flat symbol can open the very day its entry
        # gate first clears, instead of waiting for month-end. Skipped on
        # rebalance_dates themselves since the monthly block below already
        # re-evaluates the same gate that day (avoids a duplicate event).
        # VIX spike/extreme hold-or-halve intentionally stays a monthly-
        # only mechanism -- not extended to off-cycle days here.
        if config.signal_gate_mode == 'daily' and d not in rebalance_dates:
            vol_regime_d, vix_close_d, vix_ratio_d = _vix_regime_at(vix, d)
            if not config.vix_gating:
                vol_regime_d = VolRegime.NORMAL
            vix_scalar_d = VIX_ELEVATED_SCALE if vol_regime_d == VolRegime.ELEVATED else 1.0
            mixing_params_by_cluster_d = _mixing_params_for_date(config, monthly_history, d)

            for symbol in config.symbols:
                prior = ledger.held_contracts[symbol]
                result = _compute_signal_row(symbol, precomputed, d, futures_types, config,
                                              vix_scalar_d, annualization_by_symbol[symbol],
                                              **_goulding_kwargs_for(config, rebal_monthly, symbol, d, mixing_params_by_cluster_d))
                if result is None:
                    continue

                if prior != 0:
                    is_long = prior > 0
                    reason = _signal_gate_reason(result['signal'], result['ts_fast'], result['ts_slow'], is_long,
                                                  config.ts_exit_threshold, config.exit_on_ts_crossover)
                    if reason is not None:
                        ledger.rebalance_to(symbol, 0, d, vol_regime_d, vix_close=vix_close_d, vix_ratio=vix_ratio_d,
                                            signal={**result, 'gate_reason': reason})
                elif result['target'] != 0:
                    target, gate_reason = _apply_signal_gate(0, result['target'], result, config)
                    if target != 0:
                        ledger.rebalance_to(symbol, target, d, vol_regime_d, vix_close=vix_close_d,
                                            vix_ratio=vix_ratio_d, signal={**result, 'gate_reason': gate_reason})

        # 2. On rebalance dates, resize toward the vol-targeted signal,
        # gated by the spot-VIX regime (mirrors
        # tsmom_rebalance.compute_rebalance_targets' early-return shape).
        if d in rebalance_dates:
            vol_regime, vix_close, vix_ratio = _vix_regime_at(vix, d)
            if not config.vix_gating:
                vol_regime = VolRegime.NORMAL  # vix_close/vix_ratio still logged, just not acted on

            if vol_regime in (VolRegime.SPIKE, VolRegime.EXTREME):
                for symbol in config.symbols:
                    prior = ledger.held_contracts[symbol]
                    target = round(prior / 2) if vol_regime == VolRegime.EXTREME else prior
                    close_row = full_price_data[symbol].filter(pl.col('ts_event') <= d).tail(1)
                    close = float(close_row['close'][0]) if close_row.height > 0 else None
                    ledger.rebalance_to(symbol, target, d, vol_regime, vix_close=vix_close, vix_ratio=vix_ratio,
                                        signal={'close': close})
            else:
                vix_scalar = VIX_ELEVATED_SCALE if vol_regime == VolRegime.ELEVATED else 1.0
                # Computed once per rebalance date, shared by both branches
                # below -- a_Co/a_Re only needs estimating once per date,
                # not once per symbol or per branch.
                mixing_params_by_cluster = _mixing_params_for_date(config, monthly_history, d)

                if config.target_portfolio_vol is not None and config.fixed_quantities is None:
                    # Correlation-aware sizing -- see TsmomBacktestConfig.
                    # target_portfolio_vol's own docstring for the full
                    # derivation. Two passes are needed because "which
                    # symbols are active" (and therefore n_effective/the
                    # correlation matrix/IDM) can only be known AFTER
                    # computing everyone's own scalar, but the FINAL target
                    # for those active symbols depends on the IDM-derived
                    # budget computed FROM that same active set -- a single
                    # pass can't do both in one order.
                    #
                    # Pass 1 (probe): config.max_notional stands in as a
                    # placeholder budget purely to get each symbol's own
                    # `scalar` (and other signal fields) -- never used for
                    # the FINAL target of an active symbol, only to decide
                    # who's active (scalar != 0). An inactive symbol's
                    # target is 0 regardless of budget (0 * anything == 0),
                    # so its probe result is reused as final directly, no
                    # second call needed.
                    probe_results = {}
                    for symbol in config.symbols:
                        result = _compute_signal_row(symbol, precomputed, d, futures_types, config,
                                                      vix_scalar, annualization_by_symbol[symbol],
                                                      **_goulding_kwargs_for(config, rebal_monthly, symbol, d, mixing_params_by_cluster))
                        if result is not None:
                            probe_results[symbol] = result

                    active_symbols = [s for s, r in probe_results.items() if r['scalar'] != 0]

                    # IDM-derived, correlation-aware per-symbol budget -- see
                    # compute_symbol_notional_budget's own docstring for the
                    # full derivation (capital * target_portfolio_vol * IDM,
                    # split across active_symbols per config.notional_weighting,
                    # converted back to a notional_budget by dividing out
                    # config.vol_target). One entry per active symbol -- NOT
                    # a single shared float -- since 'erc'/'hrp' give
                    # different symbols different budgets ('flat' is the
                    # only scheme where every symbol happens to get the same
                    # value). Returns {} if there are no active symbols, or
                    # too little history yet for a bounded-window correlation
                    # estimate -- nobody trades this month regardless (every
                    # probe result's own target is already 0 in that case).
                    per_symbol_budget = compute_symbol_notional_budget(
                        active_symbols, returns_wide, d, ledger.capital, config.target_portfolio_vol,
                        config.vol_target, config.corr_window_years, config.corr_halflife_days,
                        config.notional_weighting, config.use_idm)

                    for symbol in config.symbols:
                        result = probe_results.get(symbol)
                        if result is None:
                            continue
                        if symbol in active_symbols:
                            # Recompute with the REAL, IDM-derived budget --
                            # the probe pass's own target (implicitly sized
                            # off config.max_notional) is discarded here.
                            result = _compute_signal_row(symbol, precomputed, d, futures_types, config,
                                                          vix_scalar, annualization_by_symbol[symbol],
                                                          notional_budget=per_symbol_budget[symbol],
                                                          **_goulding_kwargs_for(config, rebal_monthly, symbol, d, mixing_params_by_cluster))
                        target, gate_reason = _apply_signal_gate(ledger.held_contracts[symbol], result['target'], result, config)
                        ledger.rebalance_to(symbol, target, d, vol_regime, vix_close=vix_close,
                                            vix_ratio=vix_ratio, signal={**result, 'gate_reason': gate_reason})
                else:
                    for symbol in config.symbols:
                        result = _compute_signal_row(symbol, precomputed, d, futures_types, config,
                                                      vix_scalar, annualization_by_symbol[symbol],
                                                      **_goulding_kwargs_for(config, rebal_monthly, symbol, d, mixing_params_by_cluster))
                        if result is None:
                            continue
                        target, gate_reason = _apply_signal_gate(ledger.held_contracts[symbol], result['target'], result, config)
                        ledger.rebalance_to(symbol, target, d, vol_regime, vix_close=vix_close,
                                            vix_ratio=vix_ratio, signal={**result, 'gate_reason': gate_reason})

        daily_rows.append({'date': d, 'capital': round(ledger.capital, 2)})

    # Force-close any position still open at the end of the window -- same
    # spirit as Backtester's close_all sweep: a trade that's still running
    # when the backtest ends isn't abandoned, it's marked at the last
    # available price so `trades` doesn't silently drop it.
    if all_dates:
        last_date = all_dates[-1]
        for symbol in config.symbols:
            ot = ledger.open_trade[symbol]
            if ot is not None:
                close_row = full_price_data[symbol].filter(pl.col('ts_event') <= last_date).tail(1)
                last_price = float(close_row['close'][0]) if close_row.height > 0 else None
                ot['close_reason'] = 'end_of_backtest'
                ledger.close_trade(symbol, last_date, last_price)

    stats = pl.DataFrame(daily_rows)
    stats = stats.with_columns(running_max=pl.col('capital').cum_max())
    stats = stats.with_columns(
        cum_pnl=(pl.col('capital') - config.initial_capital).round(2),
        drawdown_usd=(pl.col('capital') - pl.col('running_max')).round(2),
    )
    stats = stats.with_columns(
        drawdown_pct=pl.when(pl.col('running_max') > 0)
        .then((pl.col('drawdown_usd') / pl.col('running_max') * 100).round(2))
        .otherwise(0.0)
    )

    # Summary stats -- same shape/naming as
    # tsmom_binary_vol_parity_backtest.py's own run() return dict
    # (ann_ret_pct/ann_vol_pct/sharpe/max_dd_pct/total_fees). This module
    # previously had no equivalent anywhere: main() printed final
    # capital/cum_pnl/max_dd_usd but never computed an actual Sharpe ratio,
    # and nothing was ever saved to a summary CSV -- confirmed there was no
    # way to compare runs' risk-adjusted performance without recomputing
    # this by hand from daily_mtm every time.
    daily_ret = stats.with_columns(
        ret=pl.col('capital') / pl.col('capital').shift(1) - 1
    ).drop_nulls('ret')
    mean_ret, std_ret = daily_ret['ret'].mean(), daily_ret['ret'].std()
    ann_ret = (mean_ret or 0.0) * 252
    ann_vol = (std_ret or 0.0) * (252 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol else None
    total_fees = sum(t['fee'] for t in ledger.transactions)

    return {
        'daily_mtm': stats, 'trend_signals': ledger.events,
        'transactions': pl.DataFrame(ledger.transactions),
        'trades': pl.DataFrame(ledger.trades).sort('entry_date') if ledger.trades else pl.DataFrame(ledger.trades),
        'n_days': stats.height,
        'ann_ret_pct': round(ann_ret * 100, 2),
        'ann_vol_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 2) if sharpe else None,
        'max_dd_pct': round(stats['drawdown_pct'].min(), 2) if stats.height else None,
        'total_fees': round(total_fees, 2),
    }
