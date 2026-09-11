"""
TSMOM monthly rebalancing orchestrator -- has an IB dependency (contract
resolution, historical bars, live VX/VIX fallback, current positions), via
the ib_tools.ibpysync connectivity layer, but ONLY on the data_source='ib'
code path (see TsmomLiveConfig). ib_tools is imported lazily, inside the
functions that actually touch IBPySync, rather than at module level -- this
module (TsmomLiveConfig, compute_rebalance_targets, ...) is importable and
fully usable with data_source='database' in an environment that has no
ib_tools/ib_insync installed at all, which is what makes it runnable in a
plain notebook kernel with zero IB dependency.

Pure signal math lives in derivatives_bt_engine.domain.signal (used by both this
live orchestrator and the duckdb-backed backtest); cross-instrument risk
allocation lives in derivatives_bt_engine.domain.allocation. This module wires
that signal up to IBPySync (data_source='ib' only), applies the VX/VIX
vol-spike gate, and turns the result into a per-instrument rebalance plan
(contract counts), without placing any orders itself.

The VX/expiry-resolution helpers below intentionally mirror the equivalent
logic in ib_tools' combined_monitor.py (live VX front-month via CFE, fall
back to VIX spot's last RTH close when VX is unavailable, e.g. weekend
close), kept local so this module only depends on ib_tools.ibpysync, not on
any of ib_tools' monitor-specific scripts.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Mapping, Optional
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

if TYPE_CHECKING:
    # Type-hint only -- from __future__ import annotations above means this
    # is never evaluated at runtime, so it doesn't force ib_tools to be
    # installed just to import this module. Actual runtime usages of
    # IBPySync each do their own local `from ib_tools.ibpysync import
    # IBPySync`, scoped to the data_source='ib' functions that need it.
    from ib_tools.ibpysync import IBPySync

from derivatives_bt_engine.domain.enums import TrendRegime, VolRegime
from derivatives_bt_engine.domain.instruments import (
    CME_MONTH_LETTERS,
    CME_MONTH_NUM_TO_LETTER,
    INSTRUMENTS,
    _FX_TICKER_TO_KEY,
    get_spec,
    resolve_active_months,
    resolve_annualization_days,
    resolve_signal_symbol,
)
from derivatives_bt_engine.domain.allocation import (
    NOTIONAL_WEIGHTING_SCHEMES,
    _bounded_ewm_correlation_matrix,
    _coverage_restricted_idm,
    allocate_flat_cluster_diversified_targets,
    allocate_lot_aware_targets,
    apply_cluster_risk_cap,
    build_returns_wide,
    compute_desired_risk_budget,
    compute_n_effective,
    compute_notional_split,
    compute_position_scalar,
    compute_realized_portfolio_risk,
    compute_symbol_notional_budget,
    group_by_cluster,
)
from derivatives_bt_engine.domain.futures_dataloader import FuturesDataLoader, assert_monotonic_expiration
from derivatives_bt_engine.domain.signal import (
    DEFAULT_FAST_WINDOW,
    DEFAULT_SLOW_WINDOW,
    build_features,
    classify_signal_confidence,
    compute_signal_confidence,
    compute_vol_ratio,
    continuous_momentum,
    estimate_mixing_params_diagnostics,
    goulding_monthly,
    resolve_trend_direction,
)
# VIX_FILE_PATH: the same local spot-VIX parquet the backtest reads (see
# that module's own docstring for why -- no VX futures/CFE history is
# available locally, so spot-VIX-vs-its-own-63d-MA is the closest available
# analog to this module's live VX-front-month/VX-63d-MA ratio). Reused here
# (not duplicated) so data_source='database' stays byte-for-byte consistent
# with what the backtest itself would compute for the same date.
from derivatives_bt_engine.domain.tsmom_backtester import VIX_FILE_PATH

log = logging.getLogger(__name__)

ET = ZoneInfo('America/New_York')

# VX spike ratio bands (vx_current / vx_ma) -> regime
VX_ELEVATED_RATIO = 1.3
VX_SPIKE_RATIO    = 1.5
VX_EXTREME_RATIO  = 2.0
VX_ELEVATED_SCALE = 0.6   # reduce all positions to this fraction of target when 'elevated'

# ── Tunable defaults ─────────────────────────────────────────────────────
DEFAULT_BAR_YEARS = 3.0
# Trailing window for the VX/VIX moving average the spike gate compares
# vx_current against (vx_ratio = vx_current / vx_ma) -- same window used by
# domain.tsmom_backtester's own vix_ma_window_days (see that module's
# TsmomBacktestConfig field), which this project's VX_ELEVATED_RATIO/
# VX_SPIKE_RATIO/VX_EXTREME_RATIO bands above were calibrated against.
DEFAULT_VX_MA_WINDOW_DAYS = 63

DEFAULT_MAX_CONTRACTS = 15

# ── Config validation ────────────────────────────────────────────────────
SIGNAL_WEIGHTINGS = ('continuous', 'goulding')
MIXING_POOLS = ('cluster', 'global')
RISK_BUDGET_MODES = ('cluster', 'idm')
DISCRETE_ALLOCATIONS = ('independent', 'lot-aware', 'flat-cluster-diversified')
DATA_SOURCES = ('ib', 'database')

# ── Infrastructure ───────────────────────────────────────────────────────
# Same futures-bar parquet cache the backtest uses (domain.tsmom_backtester's
# own load_portfolio_data) -- sharing it means a symbol already cached by a
# backtest run doesn't need re-fetching here, and vice versa.
_DB_CACHE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.cache', 'futures'))


@dataclass
class TsmomLiveConfig:
    """Portfolio-level config for compute_rebalance_targets.

    Deliberately named/shaped after domain.tsmom_backtester.TsmomBacktestConfig
    -- signal_weighting/mixing_pool/notional_weighting/use_idm/
    corr_window_years/corr_halflife_days/vol_target/target_portfolio_vol share
    both name AND meaning with that dataclass, so the same mental model
    (and the same config values) transfer directly between a backtest run
    and a live rebalance. Fields with no backtest equivalent
    (account_equity, max_cluster_risk_pct, min_conviction,
    max_lot_overrun_pct, signal_confidence_*, vx_expiry, data_source,
    risk_budget_mode, as_of) are live-specific portfolio-risk-management or
    data-sourcing concerns the backtest has no need of.

    Per-instrument metadata (exchange, expiry, multiplier, cluster,
    max_contracts, max_notional, ib_symbol/signal_symbol/db_symbol) stays
    in the separate `instruments` list compute_rebalance_targets also
    takes -- unlike vol_target/target_portfolio_vol/etc., those are
    genuinely per-instrument, not portfolio-level, so folding them into
    this dataclass would just mean threading a list through it anyway."""
    vol_target: float = 0.15
    long_only: bool = False
    regime_discount: float = 0.5
    max_contracts: int = 15
    account_equity: Optional[float] = None
    target_portfolio_vol: float = 0.15
    max_cluster_risk_pct: float = 0.25
    min_conviction: float = 0.05
    max_lot_overrun_pct: float = 0.5
    enable_signal_confidence: bool = False
    signal_confidence_low_threshold: float = 0.7
    signal_confidence_high_threshold: float = 1.5
    signal_confidence_high_vol: float = 0.5
    signal_confidence_low_vol: float = 1.0
    # Portfolio-wide VX/VIX spike gate -- on by default (matches all prior
    # behavior). Toggle off when no VX/VIX data source is available at all
    # (data_source='database' needs a local spot-VIX parquet that may not
    # exist in every environment -- VX/VIX aren't in the CME futures
    # duckdb FuturesDataLoader reads for everything else, so there's no
    # way to derive this gate from that db; data_source='ib' needs a live
    # CFE/CBOE subscription). Unlike TsmomBacktestConfig.vix_gating (which
    # still READS vix data and just doesn't act on it -- cheap when that
    # data is already being loaded for the whole backtest anyway), False
    # here SKIPS the read entirely (_get_vx_spike_ratio is never called) --
    # compute_rebalance_targets proceeds as vol_regime=Normal,
    # vx_current/vx_ma=None, vix_scalar=1.0, no VX/VIX dependency of any
    # kind. Caching VX/VIX from IB into a local file to work around a
    # missing data source was considered and rejected -- a cached snapshot
    # goes stale the moment it's reused for a later as_of/rebalance date,
    # silently gating on old data instead of not gating at all.
    vix_gating: bool = True
    # Signal DIRECTION source -- see domain.signal.resolve_trend_direction
    # (shared with TsmomBacktestConfig.signal_weighting, same semantics):
    # 'continuous' (default): continuous_momentum's daily trend_strength +
    # classify_regime + regime_discount. 'goulding': Goulding/Harvey/
    # Mazzoleni (2023)'s own monthly Bull/Correction/Bear/Rebound
    # classification with a_Co/a_Re mixing weights re-estimated from all
    # available prior history (mixing_pool below) -- "Goulding decides
    # direction, vol-parity decides size"; regime_discount is ignored in
    # this mode (a_Co/a_Re IS the discount mechanism).
    signal_weighting: str = 'continuous'
    # Only used when signal_weighting == 'goulding'. 'cluster' (default):
    # a_Co/a_Re estimated separately per instrument cluster (pooled across
    # only the instruments passed to this rebalance, not the full
    # instruments.py universe). 'global': one shared estimate pooled
    # across every instrument regardless of cluster.
    mixing_pool: str = 'cluster'
    # continuous_momentum's own return-horizon windows -- only matters
    # when signal_weighting == 'continuous' (goulding_monthly ignores
    # these, using its own fast_months/slow_months, not exposed here).
    # vol_fast_window/vol_slow_window default to None -> horizon-matched
    # to fast_window/slow_window (see continuous_momentum's own docstring
    # on why that pairing matters: ts_fast/ts_slow are horizon Sharpe-like
    # statistics, and an unmatched vol window breaks that clean n-day-
    # return-over-n-day-vol correspondence) -- pass them explicitly only
    # to deliberately decouple the two. Shared with TsmomBacktestConfig's
    # own identically-named fields, same semantics/defaults.
    fast_window: int = DEFAULT_FAST_WINDOW
    slow_window: int = DEFAULT_SLOW_WINDOW
    vol_fast_window: Optional[int] = None
    vol_slow_window: Optional[int] = None
    # 'cluster' (default, unchanged prior behavior): compute_n_effective/
    # compute_desired_risk_budget -- one shared risk budget per ACTIVE
    # CLUSTER (zero-correlation assumption), replicated across every
    # instrument in that cluster. 'idm': domain.allocation.
    # compute_symbol_notional_budget -- one risk budget PER ACTIVE SYMBOL,
    # correlation-aware (bounded trailing EWM correlation matrix over the
    # active set, IDM-scaled total split via notional_weighting/use_idm
    # below) -- the same machinery TsmomBacktestConfig.target_portfolio_vol
    # drives in the backtest, now also available live.
    risk_budget_mode: str = 'cluster'
    # Only used when risk_budget_mode == 'idm' -- see
    # domain.allocation.compute_symbol_notional_budget's own docstring for
    # the full derivation of both (including why the split is fed into IDM
    # as its own weight vector, not a flat one, to avoid double-counting
    # diversification across the two steps).
    notional_weighting: str = 'flat'
    use_idm: bool = True
    corr_window_years: float = 3.0
    corr_halflife_days: float = 63.0
    # Whether apply_cluster_risk_cap's cluster-level cap/redistribution
    # runs at all (whole-contract rounding always happens regardless --
    # see that function's own `apply_cap` docstring). Default OFF: under
    # risk_budget_mode='idm', that cap re-imposes a hand-assigned-cluster,
    # zero-correlation assumption on top of sizing that's already
    # correlation-aware, and can silently claw back most of IDM's own
    # diversification credit -- confirmed directly in a live check where
    # it cut a genuine hedge position to 21% of its IDM-intended size.
    # When True and risk_budget_mode='idm', the cap's own total_risk_target
    # is scaled by the SAME idm_multiplier compute_symbol_notional_budget
    # used to size positions, so the cap stays a consistency backstop
    # (e.g. against a stale/broken correlation estimate) rather than
    # silently re-imposing the assumption idm mode exists to replace.
    apply_cluster_cap: bool = False
    # Whole-contract conversion policy. 'independent' preserves the existing
    # per-symbol rounding behavior. 'lot-aware' rounds the complete target
    # book then uses correlation-aware risk repair only if required; it has no
    # cluster rule. 'flat-cluster-diversified' retains the prior experimental
    # strategy that starts flat and forces one representative per cluster.
    discrete_allocation: str = 'lot-aware'
    # Permitted realized portfolio-risk overrun for lot-aware sizing, as a
    # fraction of account_equity * target_portfolio_vol. Zero is a hard cap.
    discrete_risk_overrun_pct: float = 0.0
    # Minimum absolute continuous contracts that may survive lot-aware's
    # initial round. The default .50 is ordinary nearest-integer rounding;
    # .25 deliberately makes .25-.49 target contracts one lot for research.
    min_fractional_contracts: float = 0.5
    # 'ib' (default, unchanged prior behavior): live IB historical bars +
    # live VX/VIX spike gate, via IBPySync -- requires an active connection
    # (compute_rebalance_targets' own `ib` argument). 'database': the same
    # local futures duckdb (FuturesDataLoader) and VIX parquet the backtest
    # reads from, no IB connection anywhere -- notebook-runnable, for
    # inspecting signals/regimes without a live account. current_contracts
    # is always None in this mode (no position source without IB) --
    # compute_rebalance_targets still reports what final_target_contracts WOULD
    # be, just not a delta against a real position.
    #
    # Staleness caveat specific to 'database' used for something CLOSE TO
    # live (not a historical backtest as_of): the local futures duckdb and
    # VIX parquet are snapshots as of whenever they were last refreshed --
    # fine for a backtest, which always names an explicit historical as_of
    # and never claims to be "now". Using 'database' with as_of=None (or
    # an as_of near today) implicitly assumes those snapshots are also
    # near-current; if they haven't been refreshed recently, the resulting
    # signals/regimes are stale without anything here telling you so. Keep
    # those sources refreshed yourself if you're using 'database' this
    # way, or set vix_gating=False below and treat 'database' output as
    # what-if analysis on whatever the snapshot happens to hold, not a
    # live signal.
    data_source: str = 'ib'
    vx_expiry: str = 'auto'  # only used when data_source == 'ib'
    min_days: int = 7        # only used when data_source == 'ib' (expiry-resolution margin)
    bar_years: float = DEFAULT_BAR_YEARS  # historical window: IB request duration / database lookback
    # Trailing window (calendar days) for the VX/VIX moving average the
    # spike gate compares vx_current against -- used in BOTH data_source
    # modes (fetch_vx_spike_ratio's IB bars / _vx_spike_ratio_from_db's
    # local VIX parquet). VX_ELEVATED_RATIO/VX_SPIKE_RATIO/VX_EXTREME_RATIO
    # above were calibrated against the 63-day default; changing this
    # changes what those bands actually mean.
    vx_ma_window_days: int = DEFAULT_VX_MA_WINDOW_DAYS
    as_of: Optional[date] = None  # only used when data_source == 'database'; None = latest available bar
    # Only meaningful when data_source == 'database' and as_of is None --
    # backfills EVERY fresh IB daily bar strictly newer than the DB's own
    # last cached row (a plain DATED Future, e.g. today's actual ZCZ6, NOT
    # ContFuture -- see research/research_ib_continuous_futures_data_
    # reliability.md for why ContFuture's back-adjustment is unusable
    # here) onto the tail of the DB-sourced series for each symbol with
    # confirmed active_months (instruments.resolve_active_months), closing
    # exactly the staleness gap the caveat above describes -- the WHOLE
    # gap, not just today, since continuous_momentum's rolling windows are
    # row-count-based, not calendar-aware (see _splice_live_front_month_
    # bar's own docstring for why a single latest-bar splice silently
    # corrupts them instead of actually fixing the staleness). Requires a
    # live `ib` connection even though data_source='database' (compute_
    # rebalance_targets raises if splice_live_price=True and ib is None).
    # Default OFF: costs up to 7 extra IB round-trips per spliced symbol
    # per run (1 req_contract_details to discover real listed months, up
    # to 3 candidate contracts x qualify+historical-bars, to replicate
    # the DB's own volume-based front-month rule live) and any
    # splice failure for a symbol silently falls back to that symbol's
    # plain (possibly stale) DB bars rather than failing the run -- see
    # _splice_live_front_month_bar's own docstring. Never touches H/IDM's
    # correlation machinery directly; H incidentally sees the fresher
    # close too, since it's built from the same per-symbol `bars` this
    # splices.
    splice_live_price: bool = False

    def __post_init__(self):
        if self.signal_weighting not in SIGNAL_WEIGHTINGS:
            raise ValueError(f"signal_weighting must be one of {SIGNAL_WEIGHTINGS}, got {self.signal_weighting!r}")
        if self.mixing_pool not in MIXING_POOLS:
            raise ValueError(f"mixing_pool must be one of {MIXING_POOLS}, got {self.mixing_pool!r}")
        if self.risk_budget_mode not in RISK_BUDGET_MODES:
            raise ValueError(f"risk_budget_mode must be one of {RISK_BUDGET_MODES}, got {self.risk_budget_mode!r}")
        if self.notional_weighting not in NOTIONAL_WEIGHTING_SCHEMES:
            raise ValueError(f"notional_weighting must be one of {NOTIONAL_WEIGHTING_SCHEMES}, "
                              f"got {self.notional_weighting!r}")
        if self.discrete_allocation not in DISCRETE_ALLOCATIONS:
            raise ValueError(f"discrete_allocation must be one of {DISCRETE_ALLOCATIONS}, "
                              f"got {self.discrete_allocation!r}")
        if self.discrete_risk_overrun_pct < 0:
            raise ValueError("discrete_risk_overrun_pct must be non-negative")
        if not 0 < self.min_fractional_contracts <= 0.5:
            raise ValueError("min_fractional_contracts must be in (0, 0.5]")
        if self.discrete_allocation == 'lot-aware' and self.apply_cluster_cap:
            raise ValueError("apply_cluster_cap is incompatible with lot-aware; use "
                             "flat-cluster-diversified for the experimental cluster policy")
        if self.data_source not in DATA_SOURCES:
            raise ValueError(f"data_source must be one of {DATA_SOURCES}, got {self.data_source!r}")
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("fast_window/slow_window must be positive")
        if self.fast_window >= self.slow_window:
            raise ValueError(f"fast_window ({self.fast_window}) must be < slow_window ({self.slow_window})")


def build_instruments(symbols: list[str], max_notional: Optional[float] = None,
                       max_contracts: int = DEFAULT_MAX_CONTRACTS) -> list[dict]:
    """The `instruments` list compute_rebalance_targets expects, built from
    a plain symbol list against domain.instruments.INSTRUMENTS -- the
    equivalent of domain.tsmom_backtester.load_portfolio_data(symbols) for
    this module's own instrument-dict shape, not a price/VIX frame.
    INSTRUMENTS.get(s) alone isn't enough: this also resolves each known
    spec's ib_symbol/signal_symbol/db_symbol fallback chain and fills in
    expiry/max_contracts/max_notional, which compute_rebalance_targets'
    per-instrument fields (_resolve_contract, _fetch_signal_inputs, ...)
    all read directly. No IB dependency -- domain.instruments has none.

    Raises ValueError on any symbol not in INSTRUMENTS -- for anything
    outside that known universe, build the dict(s) yourself instead (same
    fields this function produces: symbol, ib_symbol, signal_symbol,
    db_symbol, exchange, expiry, multiplier, cluster, max_contracts,
    max_notional)."""
    instruments = []
    for symbol in (s.strip().upper() for s in symbols if s.strip()):
        if symbol not in INSTRUMENTS:
            raise ValueError(f'Unknown symbol {symbol!r} -- not in domain.instruments.INSTRUMENTS '
                              f'({sorted(INSTRUMENTS)}); build its dict manually instead')
        known = INSTRUMENTS[symbol]
        ib_symbol = known.get('ib_symbol') or symbol
        signal_symbol = known.get('signal_symbol') or ib_symbol
        # db_symbol: Globex root symbol in the duckdb (daily.asset).
        # Explicit when IB and Globex names diverge (e.g. J7->6J, BRE->6L);
        # falls back to signal_symbol (thin contracts borrow their full-size
        # sibling's duckdb data too) then ib_symbol.
        db_symbol = known.get('db_symbol') or signal_symbol
        instruments.append({
            'symbol': symbol,
            'ib_symbol': ib_symbol,
            'signal_symbol': signal_symbol,
            'db_symbol': db_symbol,
            'exchange': known['exchange'],
            'expiry': 'auto',
            'multiplier': known['multiplier'],
            'cluster': known.get('cluster', 'other'),
            'max_contracts': max_contracts,
            # max_notional is an optional hard per-instrument ceiling, not
            # the main sizing lever (see account_equity) -- None unless
            # explicitly passed.
            'max_notional': max_notional,
        })
    return instruments


# ------------------------------------------------------------------
# VX / expiry resolution
# ------------------------------------------------------------------

def _vx_is_stale() -> bool:
    """VX futures (CFE) trade Sun 6pm - Fri 4:15pm ET with no daily break."""
    now = datetime.now(ET)
    weekday = now.weekday()
    t = now.time()
    if weekday == 5:
        return True
    if weekday == 6 and t.hour < 18:
        return True
    if weekday == 4 and (t.hour, t.minute) >= (16, 15):
        return True
    return False


def _get_nearest_vx_expiry(ib: IBPySync, min_days: int = 3) -> str:
    from ib_tools.ibpysync import IBPySync
    vx = IBPySync.future('VIX', exchange='CFE')
    vx.tradingClass = 'VX'
    details = ib.req_contract_details(vx)
    cutoff = date.today() + timedelta(days=min_days)
    expiries = sorted(
        d.contract.lastTradeDateOrContractMonth
        for d in details
        if d.contract.tradingClass == 'VX'
        and d.contract.lastTradeDateOrContractMonth
        and datetime.strptime(
            d.contract.lastTradeDateOrContractMonth[:8], '%Y%m%d'
        ).date() >= cutoff
    )
    if not expiries:
        raise RuntimeError('No VX monthly contracts found beyond min_days cutoff')
    nearest = expiries[0][:6]
    log.info('Auto-resolved VX expiry: %s (min_days=%d)', nearest, min_days)
    return nearest


def _get_vx_future(ib: IBPySync, expiry: str):
    from ib_tools.ibpysync import IBPySync
    vx = IBPySync.future('VIX', exchange='CFE', expiration=expiry)
    vx.tradingClass = 'VX'
    ib.qualify_contracts(vx)
    log.info('VX contract: %s', vx.localSymbol)
    return vx


def get_nearest_quarterly_expiry(ib: IBPySync, symbol: str, exchange: str, min_days: int = 7,
                                  multiplier: str = '') -> str:
    """Nearest expiry with at least min_days remaining, as the full
    YYYYMMDD IB already gave us in req_contract_details -- truncating to
    YYYYMM and letting IB re-resolve from the partial month was observed to
    fail qualify_contracts for some symbols (MCL/MZC/MZW) even though the
    exact same month was a real, listed contract a moment earlier; passing
    back the untruncated date IB itself returned avoids that re-resolution
    step entirely."""
    from ib_tools.ibpysync import IBPySync
    c = IBPySync.future(symbol, exchange=exchange, multiplier=multiplier)
    details = ib.req_contract_details(c)
    log.debug(
        'req_contract_details(%s, %s) returned %d contract(s): %s',
        symbol, exchange, len(details),
        [(d.contract.lastTradeDateOrContractMonth, d.contract.localSymbol,
          d.contract.tradingClass, d.contract.multiplier) for d in details],
    )
    cutoff = date.today() + timedelta(days=min_days)
    expiries = sorted(
        d.contract.lastTradeDateOrContractMonth
        for d in details
        if d.contract.lastTradeDateOrContractMonth
        and datetime.strptime(
            d.contract.lastTradeDateOrContractMonth[:8], '%Y%m%d'
        ).date() >= cutoff
    )
    if not expiries:
        raise RuntimeError(f'No {symbol} ({exchange}) contracts found beyond {min_days}d')
    nearest = expiries[0]
    log.info('Auto-resolved %s (%s) expiry: %s (all candidates: %s)', symbol, exchange, nearest, expiries)
    return nearest


# ------------------------------------------------------------------
# VX vol-spike gate
# ------------------------------------------------------------------

def fetch_vx_spike_ratio(ib: IBPySync, vx_expiry: str = 'auto', min_days: int = 3,
                          ma_window_days: int = DEFAULT_VX_MA_WINDOW_DAYS) -> tuple[float, float]:
    """
    Returns (vx_current, vx_ma). Raises RuntimeError if no usable VX/VIX
    data is available at all.

    vx_ma always comes from VX historical daily bars (historical data is
    available even when the market/contract has no live price, e.g.
    weekends). vx_current prefers the live VX front-month price; if that is
    unavailable (stale/weekend close) it falls back to VIX spot's last
    close, then as a last resort to the most recent VX historical close.
    """
    from ib_tools.ibpysync import IBPySync
    expiry = _get_nearest_vx_expiry(ib, min_days) if vx_expiry == 'auto' else vx_expiry
    vx = _get_vx_future(ib, expiry)

    # +27 calendar-day buffer above ma_window_days (matches the prior fixed
    # 90d-for-a-63d-window margin) -- covers non-trading days plus the
    # small extra history the tail(ma_window_days + 7) slice below reads
    # from for the last-resort vx_current fallback.
    bars = ib.get_historical_bars(vx, duration=f'{ma_window_days + 27} d', bar_size='1 day')
    if bars is None or bars.height == 0:
        raise RuntimeError('No VX historical bars available — cannot compute vx_ma')

    closes = bars['close'].tail(ma_window_days + 7)
    vx_ma = closes.tail(ma_window_days).mean()
    if vx_ma is None or math.isnan(vx_ma) or vx_ma <= 0:
        raise RuntimeError(f'Insufficient VX history to compute a {ma_window_days}-day MA')

    # Delayed data type, not live: this account has no CFE/CBOE real-time
    # subscription, so reqMktData on VX/VIX would otherwise hit error 10168
    # and burn ~100s per get_price call before falling through (see
    # combined_monitor.py, which sidesteps the same issue the same way).
    ib.set_market_data_type(3)

    vx_current = None
    if not _vx_is_stale():
        try:
            vx_current = ib.get_price(vx)
        except Exception as exc:
            log.warning('VX live price unavailable (%s) — falling back to VIX spot close', exc)

    if vx_current is None:
        try:
            vix = IBPySync.index('VIX', exchange='CBOE')
            ib.qualify_contracts(vix)
            vx_current = ib.get_price(vix)
            log.info('Using VIX spot last close as vx_current fallback: %.2f', vx_current)
        except Exception as exc:
            log.warning('VIX spot also unavailable (%s) — using last VX historical close', exc)
            vx_current = float(closes[-1])

    return float(vx_current), float(vx_ma)


def _vx_spike_ratio_from_db(as_of: Optional[date] = None,
                             ma_window_days: int = DEFAULT_VX_MA_WINDOW_DAYS) -> tuple[float, float]:
    """Local spot-VIX analog to fetch_vx_spike_ratio, for
    TsmomLiveConfig(data_source='database') -- no VX futures (CFE) history
    is available locally (same reason domain.tsmom_backtester's own module
    docstring gives), so this reads the same VIX spot parquet the backtest
    uses instead: current close vs its own trailing ma_window_days MA.
    Filtered to <= as_of when given (no lookahead); None uses the full
    available series, i.e. "as of today". Raises RuntimeError if there
    isn't yet a full ma_window_days window."""
    vix = pl.read_parquet(VIX_FILE_PATH).select(['date', 'close']).rename({'close': 'vix_close'}).sort('date')
    if as_of is not None:
        vix = vix.filter(pl.col('date') <= as_of)
    vix = vix.with_columns(vix_ma=pl.col('vix_close').rolling_mean(ma_window_days))
    last = vix.tail(1)
    if last.height == 0 or last['vix_ma'][0] is None:
        raise RuntimeError(f'Insufficient local VIX history to compute a {ma_window_days}-day MA')
    return float(last['vix_close'][0]), float(last['vix_ma'][0])


def _get_vx_spike_ratio(ib: Optional[IBPySync], config: TsmomLiveConfig) -> tuple[float, float]:
    """Dispatches to the live VX-futures gate or the local spot-VIX analog
    per config.data_source -- same (vx_current, vx_ma) shape either way,
    so check_vol_regime below is unaware of which source produced it."""
    if config.data_source == 'ib':
        return fetch_vx_spike_ratio(ib, config.vx_expiry, ma_window_days=config.vx_ma_window_days)
    return _vx_spike_ratio_from_db(config.as_of, config.vx_ma_window_days)


def check_vol_regime(vx_ratio: float) -> VolRegime:
    """Normal | Elevated | Spike | Extreme from vx_current / vx_ma.

    Deliberately one-sided: every threshold here checks vx_ratio being
    HIGH (1.3/1.5/2.0) -- there is no symmetric low-vx_ratio bucket, and
    that's not an oversight. This function's job is a portfolio-wide risk-
    management gate (feeds vix_scalar, and the spike/extreme
    hold-or-halve bypass), not a regime-confidence detector -- "the market
    looks unusually calm" isn't a risk to manage the same way "the market
    looks dangerous" is, so nothing here classifies it. (Per-instrument,
    asset-specific vol state -- including a low-vol-ratio bucket -- is a
    different, independent mechanism: see SignalConfidenceRegime /
    classify_signal_confidence in signal.py.)"""
    if vx_ratio > VX_EXTREME_RATIO:
        return VolRegime.EXTREME
    if vx_ratio > VX_SPIKE_RATIO:
        return VolRegime.SPIKE
    if vx_ratio > VX_ELEVATED_RATIO:
        return VolRegime.ELEVATED
    return VolRegime.NORMAL


# ------------------------------------------------------------------
# Positions
# ------------------------------------------------------------------

def _current_contracts(ib: IBPySync, contract) -> int:
    """Net signed position size for `contract`'s conId across the account."""
    try:
        positions = ib.ib.positions()
    except Exception as exc:
        log.warning('Could not fetch current positions (%s) — assuming 0', exc)
        return 0
    total = 0.0
    for p in positions:
        if p.contract.conId == contract.conId:
            total += p.position
    return int(round(total))


def _scalar_was_capped(target: dict) -> Optional[bool]:
    """Whether the volatility scale hit its explicit [0.25, 2.0] bound.

    The finished position scalar is intentionally not capped at +/-1: that
    would erase permitted low-vol leverage.  `risk_scalar` is the actual
    guardrail, reconstructed here from the audit fields so the diagnostic
    identifies either of its bounds rather than a non-existent final cap.
    """
    hv = target.get('hv')
    vol_target = target.get('vol_target')
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (hv, vol_target)):
        return None
    if float(hv) <= 0:
        return None
    raw_risk_scalar = float(vol_target) / float(hv)
    return raw_risk_scalar <= 0.25 + 1e-12 or raw_risk_scalar >= 2.0 - 1e-12


def _attach_sizing_diagnostics(targets: list[dict], *,
                               portfolio_risk_target: Optional[float],
                               idm_risk_target: Optional[float],
                               realized_portfolio_risk: Optional[float],
                               discrete_allocation: str = 'independent',
                               discrete_risk_overrun_pct: float = 0.0,
                               min_fractional_contracts: float = 0.5) -> None:
    """Attach per-symbol and run-level sizing diagnostics to target rows.

    These fields are deliberately informational.  They describe the
    continuous target, the one-contract hurdle, and the final integer result;
    they do not feed back into sizing.

    `fractional_target_dollar_vol`, `one_contract_notional`, and
    `one_contract_dollar_vol` are unsigned
    dollar values.  `rounding_gap` is measured in contracts and includes any
    final max-contract clamp, so it is more precisely the continuous-to-final
    discrete sizing gap.
    """
    for target in targets:
        close = target.get('close')
        multiplier = target.get('mult')
        hv = target.get('hv')
        one_contract_notional = None
        one_contract_dollar_vol = None
        if close is not None and multiplier is not None:
            one_contract_notional = abs(float(close) * float(multiplier))
            if hv is not None and not (isinstance(hv, float) and math.isnan(hv)):
                one_contract_dollar_vol = one_contract_notional * abs(float(hv))

        pre_scalar_notional_budget = target.get('pre_scalar_notional_budget')
        vol_target = target.get('vol_target')
        pre_scalar_dollar_vol_budget = None
        if pre_scalar_notional_budget is not None and vol_target is not None:
            pre_scalar_dollar_vol_budget = abs(float(pre_scalar_notional_budget) * float(vol_target))

        fractional_target_notional = target.get('fractional_target_notional')
        fractional_target_dollar_vol = None
        if fractional_target_notional is not None and hv is not None and not (
                isinstance(hv, float) and math.isnan(hv)):
            fractional_target_dollar_vol = abs(float(fractional_target_notional) * float(hv))

        fractional_target_contracts = target.get('fractional_target_contracts')
        final_target_contracts = target.get('final_target_contracts')
        rounding_gap = None
        if fractional_target_contracts is not None and final_target_contracts is not None:
            rounding_gap = abs(float(fractional_target_contracts) - float(final_target_contracts))

        zero_reason = None
        if target.get('error'):
            zero_reason = 'error'
        elif final_target_contracts is None:
            if target.get('vol_regime') in (VolRegime.SPIKE, VolRegime.EXTREME):
                zero_reason = 'vol_regime_hold'
        elif final_target_contracts == 0:
            if target.get('vol_regime') in (VolRegime.SPIKE, VolRegime.EXTREME):
                zero_reason = 'vol_regime_hold'
            elif target.get('active') is False:
                zero_reason = 'inactive_signal'
            elif fractional_target_contracts is None:
                zero_reason = 'no_fractional_target'
            elif target.get('integer_zero_reason'):
                zero_reason = target['integer_zero_reason']
            elif abs(float(fractional_target_contracts)) < 0.5:
                zero_reason = 'below_half_contract'
            else:
                zero_reason = 'cluster_cap_or_limit'

        target.update({
            'one_contract_notional': one_contract_notional,
            'one_contract_dollar_vol': one_contract_dollar_vol,
            'pre_scalar_dollar_vol_budget': pre_scalar_dollar_vol_budget,
            'fractional_target_dollar_vol': fractional_target_dollar_vol,
            'rounding_gap': rounding_gap,
            'scalar_capped': _scalar_was_capped(target),
            'zero_reason': zero_reason,
            'portfolio_risk_target': portfolio_risk_target,
            'idm_risk_target': idm_risk_target,
            'realized_portfolio_risk': realized_portfolio_risk,
            'discrete_allocation': discrete_allocation,
            'discrete_risk_overrun_pct': discrete_risk_overrun_pct,
            'min_fractional_contracts': min_fractional_contracts,
            'integer_risk_limit': (
                portfolio_risk_target * (1.0 + discrete_risk_overrun_pct)
                if discrete_allocation in ('lot-aware', 'flat-cluster-diversified')
                and portfolio_risk_target is not None
                else None
            ),
        })


# ------------------------------------------------------------------
# Main orchestration
# ------------------------------------------------------------------

def _list_live_dated_contracts(ib: IBPySync, ib_symbol: str, exchange: str, multiplier: str = '') -> list:
    """Every currently-listed dated Future for ib_symbol/exchange, as
    (expiry_date, full YYYYMMDD) tuples sorted ascending by expiry --
    mirrors get_nearest_quarterly_expiry's own already-proven pattern
    (this module) of enumerating req_contract_details rather than
    guessing a bare YYYYMM and letting IB re-resolve it from the partial
    month: that function's own docstring documents this exact failure
    mode (MCL/MZC/MZW), and it recurred here too -- confirmed live this
    session against CL specifically, where a guessed '202608' correctly
    got rejected (WTI's real last-trade date is roughly a MONTH BEFORE
    its own named delivery month, an early-expiry convention no other
    product in this registry has, so an already-expired August contract
    doesn't look expired under a same-calendar-month heuristic) but
    '202609'/'202610' silently mis-resolved back to August's own contract
    instead of erroring. Passing IB's own untruncated date straight back
    to IBPySync.future avoids the re-resolution step that causes both.
    Includes every returned listing regardless of expiry -- callers
    filter for "not yet expired" against their own `today`."""
    from ib_tools.ibpysync import IBPySync
    c = IBPySync.future(ib_symbol, exchange=exchange, multiplier=multiplier)
    details = ib.req_contract_details(c)
    out = []
    for d in details:
        raw = d.contract.lastTradeDateOrContractMonth
        if not raw:
            continue
        out.append((datetime.strptime(raw[:8], '%Y%m%d').date(), raw[:8]))
    return sorted(out)


def _qualify_and_pull_recent_bar(ib: IBPySync, ib_symbol: str, exchange: str, multiplier: str,
                                  expiration_yyyymmdd: str, duration: str = '2 D') -> tuple:
    """Qualifies a plain DATED Future (IBPySync.future -- NOT cont_future/
    ContFuture, see _splice_live_front_month_bar's own docstring for why)
    for ib_symbol/expiration_yyyymmdd and pulls `duration` worth of
    trailing daily bars (default '2 D' -- just enough for a same-day
    volume comparison; _splice_live_front_month_bar passes a gap-sized
    duration when it needs more than the single latest bar).
    `expiration_yyyymmdd` must be a FULL, untruncated date IB itself
    already confirmed exists (via _list_live_dated_contracts) -- NOT a
    bare YYYYMM guess, which was observed to unreliably mis-resolve or
    outright fail qualification (see _list_live_dated_contracts' own
    docstring). `ib_symbol` must be IB's OWN facing ticker, not
    necessarily the DB's `db_symbol` -- they diverge for the 3 FX
    contracts instruments._FX_TICKER_TO_KEY exists to translate (e.g. the
    DB stores JPY's continuous series under the raw Globex root '6J', but
    IB's own API only resolves it under 'JPY'). Returns (qualified_
    contract, bars) with bars renamed 'date'->'ts_event' (same convention
    as this module's 'ib' data_source branch). Raises on any
    qualification/data failure, a resolved date mismatch (catches a
    wrong contract), or an empty bar result -- makes no fallback
    decisions itself; the caller owns that."""
    from ib_tools.ibpysync import IBPySync
    contract = IBPySync.future(ib_symbol, exchange=exchange, expiration=expiration_yyyymmdd, multiplier=multiplier)
    ib.qualify_contracts(contract)
    resolved = contract.lastTradeDateOrContractMonth[:8]
    if resolved != expiration_yyyymmdd:
        raise RuntimeError(f"{ib_symbol}: qualified contract's own expiry {resolved} "
                            f"!= requested {expiration_yyyymmdd}")
    bars = ib.get_historical_bars(contract, duration=duration, bar_size='1 day', what_to_show='TRADES', use_rth=True)
    if bars is None or bars.height == 0:
        raise RuntimeError(f"{ib_symbol} ({expiration_yyyymmdd}): no historical bars returned")
    return contract, bars.rename({'date': 'ts_event'})


def _qualify_and_pull_recent_bar_with_fallback(ib: IBPySync, ib_symbol: str, exchange: str,
                                                fallback_multiplier: str, expiration_yyyymmdd: str,
                                                duration: str = '2 D') -> tuple:
    """Tries _qualify_and_pull_recent_bar with a BLANK multiplier first
    (correct for the overwhelming majority of symbols -- see
    _splice_live_front_month_bar's own docstring on why multiplier is
    dropped by default), and only on failure retries once with
    `fallback_multiplier` (db_symbol's own registry value). Handles the
    genuine exception: some IB-listed roots are truly ambiguous without
    one (confirmed live this session -- 'SI' alone resolves to BOTH
    full-size silver (multiplier=5000, tradingClass='SI') AND micro silver
    (multiplier=1000, tradingClass='SIL') under the SAME bare symbol,
    and IB refuses to pick one; the blank attempt there returns zero bars,
    not an outright reject, so this can't be distinguished from a genuine
    "no data" case ahead of time -- the retry is what actually
    disambiguates it). Re-raises the SECOND attempt's exception if that
    also fails."""
    try:
        return _qualify_and_pull_recent_bar(ib, ib_symbol, exchange, '', expiration_yyyymmdd, duration=duration)
    except Exception:
        return _qualify_and_pull_recent_bar(ib, ib_symbol, exchange, fallback_multiplier, expiration_yyyymmdd,
                                             duration=duration)


def _splice_live_front_month_bar(ib: Optional[IBPySync], instr: dict, db_symbol: str,
                                  bars: pl.DataFrame) -> pl.DataFrame:
    """Splices EVERY fresh IB daily bar strictly newer than `bars`' own
    last cached row onto its tail (the DB-sourced continuous front-month
    series), in memory only -- never written back to FuturesDataLoader's
    cached parquet. Closes the staleness gap TsmomLiveConfig.
    splice_live_price's own docstring describes: the local duckdb cache
    isn't refreshed every day, so a symbol's trend_strength/hv/regime can
    be computed off a stale close without anything flagging it.

    Backfills the WHOLE gap, not just the single latest bar: continuous_
    momentum's rolling windows (ts_fast/ts_slow/daily_std/hv,
    signal.py's rolling_mean/rolling_std) are plain ROW-COUNT windows,
    calendar-agnostic -- splicing on only today's bar while the DB is
    stale by N days doesn't "skip" those N missing days, it silently
    compresses them out of the window entirely (the newest computed
    signal ends up a window of "62 stale rows + 1 fresh row" for a
    63-row window, not a real 63-trading-day lookback). A DB cache stale
    by weeks (confirmed this session) needs those weeks' worth of real
    bars actually present, not approximated by one point.

    Approximation/tradeoff to flag: every backfilled day in the gap uses
    the SAME single contract this function ultimately picks as current
    front-month (via the volume comparison below) -- it does NOT
    re-derive which contract was genuinely front on each individual day
    the way _CONTINUOUS_FRONT_MONTH_SQL does (that would mean replicating
    the full day-by-day sticky-crossover rule live across every candidate
    contract, a much larger undertaking). If a real roll happened partway
    through the gap, the days before that roll get backfilled with the
    NEW contract's prices rather than the (technically correct for that
    date) old one -- a small, usually minor discontinuity right at the
    roll boundary, traded off against the much larger error of leaving
    those days out of the rolling windows entirely.

    Deliberately uses a plain DATED Future (IBPySync.future, e.g. today's
    actual ZCZ6), never ContFuture/cont_future: research/research_ib_
    continuous_futures_data_reliability.md found ContFuture/reqHistoricalData
    pulls are back-adjusted with a compounding, non-reconstructable
    adjustment that diverges from this project's raw/unadjusted DB series by
    up to 23% on real dates, and IB's own two continuous surfaces don't even
    agree with each other. A single DATED contract's own raw print needs no
    adjustment to match the DB's own raw series.

    Replicates the DB's own volume-based sticky-front-month rule
    (_CONTINUOUS_FRONT_MONTH_SQL's `ORDER BY volume DESC`) live, over up
    to 3 candidates drawn from _list_live_dated_contracts' own REAL,
    IB-confirmed listing (not a guessed/walked month -- see that
    function's own docstring for why guessing is unreliable), filtered to
    not-yet-expired and, for a symbol with a genuinely restricted cycle
    (active_months != all 12 CME months, e.g. grains/metals/financials),
    to months whose letter is actually in that confirmed cycle -- a
    symbol like CL with NO restriction (active_months deliberately set to
    all 12) skips that filter entirely, just takes the 3 nearest live
    listings. 3 candidates (not 1): matters most for CL specifically,
    where "the next listed month" isn't necessarily the true volume
    leader the way it reliably is for a sparse, well-separated cycle --
    WTI's real front-month convention can roll multiple months out ahead
    of nominal expiration. Whichever candidate has the highest fresh IB
    volume wins, exactly this project's own roll rule, just evaluated
    live instead of from the DB's own bar history. Each candidate is
    pulled independently and fault-tolerantly (one dead/
    delisted candidate is skipped with a warning, not allowed to abort the
    whole splice); ties/nulls favor the nearer candidate (mirrors
    _CONTINUOUS_FRONT_MONTH_SQL's own `expiration ASC` tie-break, since
    only a STRICTLY greater volume ever overwrites an earlier pick).

    Deliberately never raises: any failure anywhere (no active_months
    confirmed for this symbol, every candidate contract unavailable, a
    missing/ambiguous volume) logs and falls back to returning `bars`
    UNCHANGED -- matches this module's existing soft-fallback philosophy
    elsewhere (e.g. fetch_vx_spike_ratio's vx_current fallback chain) of
    flagging loudly without failing a whole rebalance over one enrichment
    step."""
    symbol = instr['symbol']
    if ib is None:
        log.warning('%s: splice_live_price requested but no IB connection available -- '
                    'skipping live price splice, using DB bars as-is', symbol)
        return bars

    active_months = resolve_active_months(symbol)
    if active_months is None:
        log.info('%s: no confirmed active_months -- skipping live price splice, using DB bars as-is', symbol)
        return bars

    try:
        last = bars.tail(1)
        last_ts_event = last['ts_event'][0]
        if last_ts_event is None:
            raise ValueError("DB bars' tail row has no ts_event")
        # Enough trailing days to cover the ENTIRE gap since the DB's own
        # last cached row, not just today -- see this function's own
        # docstring for why a single latest-bar splice isn't enough. +5
        # calendar days of margin for weekends/holidays inside the gap;
        # capped at 400 to keep a pathologically stale cache from
        # generating an enormous IB request (a gap that large means the
        # cache needs a real refresh, not a live patch).
        gap_days = max(2, (date.today() - last_ts_event).days + 5)
        duration = f'{min(gap_days, 400)} D'
        # db_symbol's OWN exchange, not instr's -- a micro/mini (e.g. MES)
        # borrowing its full-size sibling's history (ES) needs THAT
        # sibling's own exchange to qualify against IB (get_spec is the
        # same lookup point used elsewhere in this codebase for this).
        # db_symbol is the DB-side (Databento/Globex) root, which for 3 FX
        # contracts diverges from IB's own facing ticker (instruments.
        # _FX_TICKER_TO_KEY -- e.g. the DB stores JPY's series under raw
        # Globex root '6J', but IB's API only resolves it under 'JPY';
        # confirmed live this session, passing '6J' straight through got
        # "No security definition has been found" for every candidate
        # month). Translate before any IB call; db_symbol itself stays
        # unchanged for the FuturesDataLoader/get_spec lookups above,
        # which already key off the DB-side root correctly.
        ib_symbol = _FX_TICKER_TO_KEY.get(db_symbol.upper(), db_symbol.upper())
        db_spec = get_spec(db_symbol)
        exchange = db_spec.get('exchange', 'CME')
        # Blank multiplier is tried FIRST (correct for the overwhelming
        # majority -- this project's internal $-per-point convention
        # doesn't universally match IB's own contract multiplier, see
        # _qualify_and_pull_recent_bar's own docstring), db_symbol's own
        # registry multiplier only as a fallback for the genuinely
        # ambiguous case (confirmed live: bare 'SI' resolves to BOTH
        # full-size silver and the SIL micro under IB without one).
        registry_multiplier = str(db_spec.get('multiplier', '') or '')

        # REAL, IB-confirmed listings -- not a guessed/walked month (see
        # _list_live_dated_contracts' own docstring for why guessing is
        # unreliable, confirmed live against CL this session). Filtered
        # to not-yet-expired; further filtered to the confirmed
        # active_months cycle UNLESS this symbol has none (CL-style "no
        # restriction", active_months == all 12 CME months), in which
        # case every live listing is a genuine candidate.
        listed = _list_live_dated_contracts(ib, ib_symbol, exchange)
        today = date.today()
        live_listed = [(exp, yyyymmdd) for exp, yyyymmdd in listed if exp >= today]
        if set(active_months) != set(CME_MONTH_LETTERS):
            live_listed = [(exp, yyyymmdd) for exp, yyyymmdd in live_listed
                           if CME_MONTH_NUM_TO_LETTER.get(exp.month) in active_months]
        if not live_listed:
            raise RuntimeError(f"no live listed contracts found for {ib_symbol} ({exchange})")
        candidate_dates = [yyyymmdd for _, yyyymmdd in live_listed[:3]]

        picked_date = picked_bars = picked_vol = None
        for expiration_yyyymmdd in candidate_dates:
            try:
                _, month_bars = _qualify_and_pull_recent_bar_with_fallback(ib, ib_symbol, exchange,
                                                                            registry_multiplier,
                                                                            expiration_yyyymmdd,
                                                                            duration=duration)
            except Exception as exc:
                log.warning('%s: candidate contract %s unavailable (%s) -- skipping', symbol,
                            expiration_yyyymmdd, exc)
                continue
            vol = month_bars.tail(1)['volume'][0]
            if picked_vol is None or (vol is not None and vol > picked_vol):
                picked_date, picked_bars, picked_vol = expiration_yyyymmdd, month_bars, vol

        if picked_bars is None:
            raise RuntimeError(f"no live candidate contract available among {candidate_dates}")
        assert picked_date is not None  # picked alongside picked_bars, always set together
        if picked_date != candidate_dates[0]:
            # Every candidate STRICTLY nearer than the one actually
            # picked -- not candidate_dates[:-1] (wrong once there are
            # 3+ candidates and the pick isn't the very last one).
            nearer = candidate_dates[:candidate_dates.index(picked_date)]
            log.warning('%s: live roll detected since DB was last refreshed -- picked %s (vol=%s) '
                        'over nearer candidate(s) %s', symbol, picked_date, picked_vol, nearer)

        # >= (not >): also picks up a same-day refresh of the DB's own
        # last cached row (e.g. IB has a more current intraday close for
        # a date the DB already has a stale print for), same as this
        # function's original single-bar behavior -- while still
        # backfilling everything strictly newer via the same filter.
        picked_expiration = datetime.strptime(picked_date, '%Y%m%d').date()
        new_rows = (picked_bars.filter(pl.col('ts_event') >= last_ts_event)
                    .select('ts_event', 'open', 'high', 'low', 'close', 'volume')
                    .with_columns(pl.lit(picked_expiration).alias('expiration')))
        if new_rows.height == 0:
            raise RuntimeError(f"no bars at or after {last_ts_event} in picked contract {picked_date} "
                                f"(duration={duration})")
        new_ts_list = new_rows['ts_event'].to_list()
        replaced = bars.filter(pl.col('ts_event').is_in(new_ts_list)).height
        appended = new_rows.height - replaced
        merged = pl.concat(
            [bars.filter(~pl.col('ts_event').is_in(new_ts_list)), new_rows],
            how='diagonal_relaxed',
        ).sort('ts_event')
        latest = new_rows.tail(1)
        log.info('%s: backfilled %d day(s) from %s (gap since %s; %d replaced, %d appended) -- '
                 'latest %s close=%s volume=%s -- DB series %d -> %d rows',
                 symbol, new_rows.height, picked_date, last_ts_event, replaced, appended,
                 latest['ts_event'][0], latest['close'][0], latest['volume'][0], bars.height, merged.height)
        return merged
    except Exception as exc:
        log.warning('%s: live price splice failed (%s) -- falling back to unspliced DB bars', symbol, exc)
        return bars


def _fetch_signal_inputs(ib: Optional[IBPySync], instr: dict, config: TsmomLiveConfig) -> dict:
    """Stage 1a: resolve the contract (IB mode only) and fetch this
    instrument's own OHLCV history, sourced per config.data_source --
    everything downstream (continuous_momentum, goulding_monthly,
    returns_wide for risk_budget_mode='idm') is pure polars either way, so
    only bar-sourcing itself differs between 'ib' and 'database'. Returns
    the FULL continuous_momentum/goulding_monthly frames (not just the
    latest row) -- compute_vol_ratio needs trailing history for
    signal_confidence, and goulding's own mixing-param estimation needs
    every prior (state, monthly_return) pair, not just the current one.

    'ib': continuous front-month historical bars via IBPySync
    (config.bar_years back) -- NOT the dated/expiry-specific contract,
    which only has bars back to its own listing (well under a year),
    silently starving the 252-day (ts_slow) momentum calc. signal_symbol
    lets a recently-listed thin contract (e.g. the CBOT micro grains)
    borrow its full-size sibling's much longer history instead -- same
    quote scale, just a different multiplier -- while sizing/orders still
    use the actually-traded contract (`contract` below).

    'database': the local futures duckdb via FuturesDataLoader, keyed by
    instr['db_symbol'] (already resolved by the caller -- either
    _build_instruments' own KNOWN_INSTRUMENTS lookup or an explicit JSON
    instrument config) -- no IB connection needed at all, UNLESS
    config.splice_live_price is also set (see its own docstring), in which
    case one fresh dated-contract bar is spliced onto this series' tail via
    _splice_live_front_month_bar -- an `ib` connection becomes required in
    that case even though data_source == 'database'. Filtered to
    config.as_of when given (no lookahead); None uses the full available
    history, i.e. "as of today" (splice_live_price is only ever attempted
    when as_of is None -- splicing "today" onto a run pinned to a past
    as_of would be a lookahead bug)."""
    if config.data_source == 'ib':
        from ib_tools.ibpysync import IBPySync
        if ib is None:
            raise ValueError("data_source='ib' requires an IBPySync connection "
                              "(compute_rebalance_targets' own ib= argument)")
        contract = _resolve_contract(ib, instr, config.min_days)
        signal_symbol = resolve_signal_symbol(instr)
        cont = IBPySync.cont_future(signal_symbol, exchange=instr.get('exchange', 'CME'))
        ib.qualify_contracts(cont)
        bars = ib.get_historical_bars(cont, duration=f'{int(config.bar_years)} y', bar_size='1 day')
        if bars is None or bars.height < 64:
            raise RuntimeError(f"Insufficient bar history for {instr['symbol']} "
                                f"({bars.height if bars is not None else 0} rows)")
        # ib_tools' get_historical_bars returns ib_insync BarData's own field
        # names verbatim (a 'date' column, not 'ts_event') -- build_features
        # requires 'ts_event' (it sorts by it).
        bars = bars.rename({'date': 'ts_event'})
    else:
        contract = None
        db_symbol = instr.get('db_symbol') or instr.get('signal_symbol') or instr.get('ib_symbol') or instr['symbol']
        bars = FuturesDataLoader(asset=db_symbol, data_dir=_DB_CACHE_DIR, use_preprocessed=True,
                                 save_preprocessed=True).daily
        assert_monotonic_expiration(bars, instr['symbol'])
        if config.as_of is not None:
            bars = bars.filter(pl.col('ts_event') <= config.as_of)
        bars = bars.sort('ts_event')
        if bars.height < 64:
            raise RuntimeError(f"Insufficient bar history for {instr['symbol']} ({bars.height} rows)")
        if config.splice_live_price and config.as_of is None:
            bars = _splice_live_front_month_bar(ib, instr, db_symbol, bars)

    # Real trading-days/year for this symbol's own continuous series
    # (instruments.resolve_annualization_days) -- this project's confirmed
    # universe splits 252 (CBOT grains) vs. 259 (everything else checked,
    # post Sunday-session-merge fix); falls back to 252 for anything
    # unconfirmed, unchanged from this module's prior universal-252 behavior.
    annualization_days = resolve_annualization_days(instr['symbol'])
    feat = build_features(bars)
    cm_df = continuous_momentum(feat, fast_window=config.fast_window, slow_window=config.slow_window,
                                 vol_fast_window=config.vol_fast_window, vol_slow_window=config.vol_slow_window,
                                 annualization_days=annualization_days)
    g_df = goulding_monthly(feat) if config.signal_weighting == 'goulding' else None

    return {
        'contract': contract,
        'annualization_days': annualization_days,
        'cm_df': cm_df,
        'g_df': g_df,
        # ts_event/close only -- the minimal columns build_returns_wide
        # needs for risk_budget_mode='idm's correlation matrix.
        'closes': bars.select('ts_event', 'close'),
    }


def _goulding_history_frame(cluster: str, g_df: pl.DataFrame) -> pl.DataFrame:
    """All-but-the-most-recent row of goulding_monthly's own output, in the
    {date, cluster, state, monthly_return} schema estimate_mixing_params
    expects -- the live equivalent of domain.tsmom_backtester's
    build_monthly_state_return_history, simplified for a single as-of
    snapshot: there's no separate backtest-style rebalance-date calendar
    to forward-match against here, so goulding_monthly's own 'ts_event'
    (month-start, already the bucket a `regime`/`ret` pair applies to) IS
    the date. The most recent row is excluded -- it's the row
    _finalize_signal itself reads as "now" (still using its own regime for
    THIS rebalance's direction); using it as its own training example
    would be a lookahead."""
    hist = g_df[:-1] if g_df.height > 0 else g_df
    hist = hist.filter(pl.col('regime').is_not_null() & pl.col('ret').is_not_null())
    return hist.select(
        pl.col('ts_event').alias('date'),
        pl.lit(cluster).alias('cluster'),
        pl.col('regime').str.to_lowercase().alias('state'),
        pl.col('ret').alias('monthly_return'),
    )


def _finalize_signal(instr: dict, raw: dict, config: TsmomLiveConfig, vix_scalar: float,
                      mixing_params_by_cluster: dict[str, tuple[float, float]],
                      signal_confidence_cfg: dict) -> dict:
    """Stage 1c: resolves this instrument's final trend_strength/regime
    (continuous vs goulding, via domain.signal.resolve_trend_direction --
    shared with the backtest's own _compute_signal_row) and every
    reporting/sizing field downstream stages need. Takes `raw` from
    _fetch_signal_inputs and, when signal_weighting == 'goulding',
    mixing_params_by_cluster (from _mixing_params_for_instruments) --
    empty in 'continuous' mode, never read.

    signal_confidence_cfg: {'enabled': bool, 'low_threshold': float,
    'high_threshold': float, 'high_vol': float, 'low_vol': float} -- see
    compute_signal_confidence(). When disabled (the default), this stage
    still returns signal_confidence=1.0 (no-op) and vol_ratio=None."""
    last = raw['cm_df'].tail(1)
    ts_fast = last['ts_fast'][0]
    ts_slow = last['ts_slow'][0]
    # continuous_momentum's own weighted ts_fast/ts_slow combination (pre
    # regime-discount) and its post-discount 'signal' -- ALWAYS computed
    # (continuous_momentum runs regardless of config.signal_weighting), so
    # these are available as a continuous-mode audit trail even in
    # 'goulding' mode, where trend_strength itself comes from goulding_
    # monthly instead and is pinned to +-1.
    ts = last['ts'][0] if 'ts' in last.columns else None
    contin_signal = last['signal'][0] if 'signal' in last.columns else None
    # continuous_momentum's OWN regime classification -- sign(ts_fast)/
    # sign(ts_slow) only, computed independently of whatever signal_
    # weighting mode this run actually uses. This is the regime that
    # gates contin_signal's correction/rebound discount (signal = ts *
    # discount) -- NOT the same thing as the 'regime' key returned below,
    # which under 'goulding' is goulding_monthly's own regime instead. The
    # two can disagree (e.g. goulding's monthly fast/slow says Correction
    # while ts_fast/ts_slow are both >=0, so continuous_momentum's own
    # regime is Bull and contin_signal comes out un-discounted).
    ts_regime = last['regime'][0] if 'regime' in last.columns else None
    daily_std_last = last['std_fast'][0] if 'std_fast' in last.columns else None
    last_close = float(last['close'][0])
    dd_raw = last['dd'][0] if 'dd' in last.columns else None
    dd_pct = dd_raw * 100 if dd_raw is not None else None

    g_regime_val = g_fast_val = g_slow_val = None
    cluster = instr.get('cluster', 'other')
    a_co, a_re = mixing_params_by_cluster.get(cluster, (0.5, 0.5))
    if config.signal_weighting == 'goulding' and raw['g_df'] is not None and raw['g_df'].height > 0:
        g_last = raw['g_df'].tail(1)
        g_regime_val = g_last['regime'][0]
        g_fast_val = g_last['fast'][0]
        g_slow_val = g_last['slow'][0]

    resolved = resolve_trend_direction(config.signal_weighting, contin_signal,
                                        ts_fast, ts_slow, config.regime_discount,
                                        g_regime_val, g_fast_val, g_slow_val, a_co, a_re)
    if resolved is not None:
        trend_strength, regime, regime_discount, g_blend = resolved
    else:
        trend_strength, regime, regime_discount, g_blend = None, TrendRegime.UNKNOWN, 1.0, None

    signal_for_scalar = trend_strength
    if config.long_only and signal_for_scalar is not None and not (
        isinstance(signal_for_scalar, float) and math.isnan(signal_for_scalar)
    ):
        signal_for_scalar = max(0.0, signal_for_scalar)

    # hv/risk_scalar recomputed here (mirrors compute_position_scalar's own
    # internal math) purely for reporting -- so the printed report shows
    # *why* a given trend_strength did or didn't turn into a trade.
    hv = daily_std_last * math.sqrt(raw['annualization_days']) if daily_std_last and daily_std_last > 0 else None
    risk_scalar = max(0.25, min(2.0, config.vol_target / hv)) if hv else 1.0

    # signal_confidence: opt-in, per-instrument discount on trust in THIS
    # instrument's signal when ITS OWN vol_ratio (short/long realized vol,
    # asset-specific) is unusual -- not VIX/VX-driven, orthogonal to
    # vix_scalar (portfolio-wide) and regime_discount (fast/slow sign
    # disagreement). Computed off the same bars already fetched in stage
    # 1a, no extra IB calls.
    vol_ratio = None
    signal_confidence_regime = None
    signal_confidence = 1.0
    if signal_confidence_cfg.get('enabled'):
        conf_df = compute_vol_ratio(raw['cm_df'])
        vol_ratio = conf_df.tail(1)['vol_ratio'][0]
        signal_confidence_regime = classify_signal_confidence(
            vol_ratio, signal_confidence_cfg['low_threshold'], signal_confidence_cfg['high_threshold'],
        )
        signal_confidence = compute_signal_confidence(
            vol_ratio, signal_confidence_cfg['low_threshold'], signal_confidence_cfg['high_threshold'],
            high_vol_discount=signal_confidence_cfg['high_vol'], low_vol_discount=signal_confidence_cfg['low_vol'],
        )

    return {
        'contract': raw['contract'],
        'signal': trend_strength,
        'signal_for_scalar': signal_for_scalar,
        'ts_fast': ts_fast,
        'ts_slow': ts_slow,
        'ts': ts,
        'contin_signal': contin_signal,
        'ts_regime': ts_regime,
        'daily_std': daily_std_last,
        'hv': hv,
        'risk_scalar': risk_scalar,
        'regime_discount': regime_discount,
        'vol_ratio': vol_ratio,
        'signal_confidence_regime': signal_confidence_regime,
        'signal_confidence': signal_confidence,
        'close': last_close,
        'dd_pct': dd_pct,
        'regime': regime,
        'cluster': cluster,
        'multiplier': instr.get('multiplier'),
        'annualization_days': raw['annualization_days'],
        # Goulding audit fields -- None in 'continuous' mode (nothing to
        # report), populated in 'goulding' mode so a saved report shows
        # exactly what drove that rebalance's direction: this instrument's
        # cluster's own a_Co/a_Re as of now, the raw g_fast/g_slow/
        # g_regime inputs resolve_trend_direction blended, and g_blend
        # itself (the raw pre-sign eq. 7 value -- None in Bull/Bear, where
        # eq. 7 doesn't apply, even though trend_strength still resolves).
        'g_regime': g_regime_val, 'g_fast': g_fast_val, 'g_slow': g_slow_val, 'g_blend': g_blend,
        'a_co': a_co if config.signal_weighting == 'goulding' else None,
        'a_re': a_re if config.signal_weighting == 'goulding' else None,
    }


def _log_mixing_params_diag(key: str, diag: dict) -> None:
    log.info('mixing_params[%s]: n_bull=%d n_bear=%d n_correction=%d n_rebound=%d  '
             'C=%s inv_C=%s  a_co_raw=%s a_re_raw=%s -> a_co=%.4f a_re=%.4f  fallback=%s',
             key, diag['n_bull'], diag['n_bear'], diag['n_correction'], diag['n_rebound'],
             diag['C'], diag['inv_C'], diag['a_co_raw'], diag['a_re_raw'],
             diag['a_co'], diag['a_re'], diag['fallback_reason'])


def _mixing_params_for_instruments(config: TsmomLiveConfig, raw_by_symbol: dict[str, dict],
                                    instruments: list[dict],
                                    diagnostics: Optional[dict] = None) -> dict[str, tuple[float, float]]:
    """{cluster: (a_co, a_re)} pooled ONLY across the instruments passed to
    THIS rebalance (not the full instruments.py universe) -- estimated
    once, shared across every symbol in that cluster, not once per symbol.
    Empty dict (never read) outside 'goulding' mode.

    `diagnostics`, when given a dict, gets populated {key: diag} -- one
    entry per cluster under config.mixing_pool='cluster', a single
    'global' entry under 'global' -- from estimate_mixing_params_
    diagnostics: every intermediate value (C, 1/C, per-state avg_r/
    avg_r2, the RAW pre-clamp a_co/a_re, which fallback if any) behind
    each returned (a_co, a_re), logged here too (once per key, per live
    rebalance -- NOT inside estimate_mixing_params_diagnostics itself,
    since that function is also called once per rebalance-MONTH per
    cluster from the backtest's own monthly loop, where logging every
    call would flood a multi-year run's log). None (default) skips both
    the population and the logging -- existing callers that don't pass it
    see no behavior change."""
    if config.signal_weighting != 'goulding':
        return {}
    instr_by_symbol = {i['symbol']: i for i in instruments}
    frames = [
        _goulding_history_frame(instr_by_symbol[sym].get('cluster', 'other'), raw['g_df'])
        for sym, raw in raw_by_symbol.items() if raw['g_df'] is not None
    ]
    monthly_history = (
        pl.concat(frames, how='vertical') if frames
        else pl.DataFrame(schema={'date': pl.Date, 'cluster': pl.Utf8, 'state': pl.Utf8, 'monthly_return': pl.Float64})
    )
    as_of = config.as_of or date.today()
    clusters_needed = {i.get('cluster', 'other') for i in instruments}
    if config.mixing_pool == 'cluster':
        params = {}
        for c in clusters_needed:
            diag = estimate_mixing_params_diagnostics(monthly_history, as_of, c)
            params[c] = (diag['a_co'], diag['a_re'])
            if diagnostics is not None:
                diagnostics[c] = diag
                _log_mixing_params_diag(c, diag)
        return params
    diag = estimate_mixing_params_diagnostics(monthly_history, as_of, None)
    if diagnostics is not None:
        diagnostics['global'] = diag
        _log_mixing_params_diag('global', diag)
    global_params = (diag['a_co'], diag['a_re'])
    return {c: global_params for c in clusters_needed}


def compute_rebalance_targets(instruments: list[dict], config: TsmomLiveConfig,
                               ib: Optional[IBPySync] = None,
                               mixing_diagnostics: Optional[dict] = None) -> list[dict]:
    """
    Runs the VX spike gate first. If a spike/extreme regime is detected,
    returns early with final_target_contracts == current_contracts (held
    unchanged), halved on 'extreme', and skips signal computation entirely.

    `mixing_diagnostics`, when given a dict, gets passed straight through
    to _mixing_params_for_instruments -- see that function's own docstring
    -- to audit/verify a 'goulding'-mode run's per-cluster a_co/a_re
    estimate (C, 1/C, per-state avg_r/avg_r2, the raw pre-clamp value,
    which fallback if any). None (default, and the only option outside
    'goulding' mode, where it's never populated) skips this entirely.

    Otherwise this runs in three stages:
      1. Signal for every instrument, no sizing yet -- _fetch_signal_inputs
         (bars, sourced per config.data_source) then, in 'goulding' mode,
         _mixing_params_for_instruments (needs every instrument's bars
         first) then _finalize_signal (trend_strength/regime/hv per
         instrument, via domain.signal.resolve_trend_direction -- shared
         with the backtest's own _compute_signal_row).
      2. Derive the risk budget, per config.risk_budget_mode:
         'cluster' (default, unchanged prior behavior): which CLUSTERS have
         a live signal (abs(signal_for_scalar) above min_conviction) ->
         n_effective -> desired_risk_budget (account_equity *
         target_portfolio_vol / sqrt(n_effective)) -> ONE shared
         budget_constant for every instrument (zero-correlation
         assumption). 'idm': which SYMBOLS have a live signal ->
         domain.allocation.compute_symbol_notional_budget over that active
         set (correlation-aware, via a bounded trailing EWM correlation
         matrix built from the same config.data_source's own price
         history) -> a budget_constant PER ACTIVE SYMBOL instead of one
         shared figure.
      3. Per instrument: combined_scalar -> fractional_target_notional (the
         pre_scalar_notional_budget times combined_scalar, optionally capped
         by instr['max_notional'] as a hard ceiling) ->
         fractional_target_contracts -> final_target_contracts, clamped to
         max_contracts as a sanity backstop. Whole-contract conversion and
         standalone_position_dollar_vol then use the selected
         config.discrete_allocation policy: 'independent' applies the legacy
         per-symbol rounding via apply_cluster_risk_cap; 'lot-aware' rounds
         the complete target book and removes lots only when its measured
         portfolio risk breaches the optional
         config.discrete_risk_overrun_pct allowance. It has no cluster cap
         or cluster-coverage rule. 'flat-cluster-diversified' retains the
         earlier experimental flat-book, cluster-representative allocator.
         (see that function's own `apply_cap` docstring); the cluster-level
         cap/redistribution itself (rescaling any cluster whose aggregate
         dollar-vol risk exceeds max_cluster_risk_pct of total portfolio
         risk -- e.g. 4 grain micros that each individually look fine can
         still collectively be one oversized bet on the shared ag-complex
         factor) only runs when config.apply_cluster_cap is True (default
         False -- see that field's own docstring for why: under
         risk_budget_mode='idm', this cap re-imposes a hand-assigned-
         cluster, zero-correlation assumption directly on top of sizing
         that already measured real correlation, and can silently claw
         back most of that credit). When True and risk_budget_mode='idm',
         the cap's own total_risk_target is scaled by the same
         idm_multiplier stage 2 already used, so it stays a consistency
         backstop rather than reversing IDM's own credit. Finally, when
         risk_budget_mode='idm', compute_realized_portfolio_risk computes
         each symbol's ACTUAL (post-rounding, post-cap-or-not) risk
         contribution -- attached to each target as
         'portfolio_risk_contribution',
         informational, unconditional on config.apply_cluster_cap -- see
         print_cluster_risk_report for the per-cluster view of it.

    ib: required (and used) only when config.data_source == 'ib'. None is
    valid and expected for config.data_source == 'database' -- that mode
    makes no IB calls anywhere, which is what makes this notebook-runnable
    for signal/regime inspection with no live account at all.
    current_contracts is always None in 'database' mode (no position
    source without IB); final_target_contracts/fractional_target_notional/etc. are still
    computed and reported."""
    if config.data_source == 'ib' and ib is None:
        raise ValueError("config.data_source == 'ib' requires an IBPySync connection (pass ib=...) "
                          "-- use data_source='database' for a no-IB, notebook-runnable signal inspection")
    if config.splice_live_price and config.data_source == 'database' and ib is None:
        raise ValueError("config.splice_live_price=True requires an IBPySync connection (pass ib=...) "
                          "even though data_source='database' -- disable splice_live_price for a pure "
                          "no-IB notebook run")

    signal_confidence_cfg = {
        'enabled': config.enable_signal_confidence,
        'low_threshold': config.signal_confidence_low_threshold,
        'high_threshold': config.signal_confidence_high_threshold,
        'high_vol': config.signal_confidence_high_vol,
        'low_vol': config.signal_confidence_low_vol,
    }

    if config.vix_gating:
        vx_current, vx_ma = _get_vx_spike_ratio(ib, config)
        vx_ratio = vx_current / vx_ma
        vol_regime = check_vol_regime(vx_ratio)
        log.info('VX spike gate — vx_current=%.2f  vx_ma=%.2f  ratio=%.3f  regime=%s',
                 vx_current, vx_ma, vx_ratio, vol_regime.value)
    else:
        # No read at all -- not even an attempt against a VX/VIX source
        # that may not exist in this environment (see TsmomLiveConfig.
        # vix_gating's own docstring).
        vx_current, vx_ma, vx_ratio = None, None, 1.0
        vol_regime = VolRegime.NORMAL
        log.info('VX spike gate disabled (config.vix_gating=False) — proceeding as vol_regime=Normal')

    if vol_regime in (VolRegime.SPIKE, VolRegime.EXTREME):
        log.warning('VX %s detected (ratio=%.3f) — holding existing positions, skipping rebalance',
                     vol_regime.value, vx_ratio)
        targets = []
        for instr in instruments:
            current = None
            if config.data_source == 'ib':
                try:
                    contract = _resolve_contract(ib, instr, config.min_days)
                    current = _current_contracts(ib, contract)
                except Exception as exc:
                    log.warning('Could not resolve %s during VX %s (%s) — reporting current=0',
                                instr['symbol'], vol_regime.value, exc)
                    current = 0
            target = None
            if current is not None:
                target = round(current / 2) if vol_regime == VolRegime.EXTREME else current
            targets.append({
                'symbol': instr['symbol'],
                'final_target_contracts': target,
                'current_contracts': current,
                'signal': None,
                'vx_current': vx_current,
                'vx_ma': vx_ma,
                'vx_ratio': vx_ratio,
                'vol_regime': vol_regime,
                # n_effective_clusters/cluster_dollar_vol_budget/
                # pre_scalar_notional_budget aren't computed on this
                # early-return path (signal computation, which they depend
                # on, is skipped entirely during a spike/extreme) -- only
                # what's already in scope from config is available.
                'account_equity': config.account_equity,
                'vol_target': config.vol_target,
                'target_portfolio_vol': config.target_portfolio_vol,
                'max_cluster_risk_pct': config.max_cluster_risk_pct,
                'max_lot_overrun_pct': config.max_lot_overrun_pct,
            })
        portfolio_risk_target = (
            config.account_equity * config.target_portfolio_vol
            if config.account_equity else None
        )
        _attach_sizing_diagnostics(
            targets,
            portfolio_risk_target=portfolio_risk_target,
            idm_risk_target=None,
            realized_portfolio_risk=None,
            discrete_allocation=config.discrete_allocation,
            discrete_risk_overrun_pct=config.discrete_risk_overrun_pct,
            min_fractional_contracts=config.min_fractional_contracts,
        )
        return targets

    vix_scalar = VX_ELEVATED_SCALE if vol_regime == VolRegime.ELEVATED else 1.0

    # Stage 1a: raw bars/features for every instrument, no signal resolved
    # yet -- goulding's mixing-param estimation (1b) needs every
    # instrument's own goulding_monthly output first.
    raw_by_symbol: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for instr in instruments:
        symbol = instr['symbol']
        try:
            raw_by_symbol[symbol] = _fetch_signal_inputs(ib, instr, config)
        except Exception as exc:
            log.error('Failed to fetch signal inputs for %s: %s', symbol, exc)
            errors[symbol] = str(exc)

    # Stage 1b (goulding only): a_Co/a_Re, pooled per mixing_params_for_instruments.
    mixing_params_by_cluster = _mixing_params_for_instruments(config, raw_by_symbol, instruments,
                                                                diagnostics=mixing_diagnostics)

    # Stage 1c: resolve each instrument's final trend_strength/regime/hv.
    instr_by_symbol = {i['symbol']: i for i in instruments}
    signals: dict[str, dict] = {}
    for symbol, raw in raw_by_symbol.items():
        try:
            signals[symbol] = _finalize_signal(instr_by_symbol[symbol], raw, config, vix_scalar,
                                               mixing_params_by_cluster, signal_confidence_cfg)
        except Exception as exc:
            log.error('Failed to finalize signal for %s: %s', symbol, exc)
            errors[symbol] = str(exc)

    def _is_active(sig: dict) -> bool:
        v = sig['signal_for_scalar']
        return v is not None and not (isinstance(v, float) and math.isnan(v)) and abs(v) > config.min_conviction

    # Stage 2: derive the risk budget, per config.risk_budget_mode.
    n_effective = None
    desired_risk_budget = None
    # Mapping, not dict: 'cluster' assigns a dict[str, Optional[float]]
    # (shared_budget_constant can be None with no account_equity), 'idm'
    # assigns compute_symbol_notional_budget's dict[str, float] -- dict is
    # invariant in its value type so a dict[str, float] isn't assignable
    # to a dict[str, Optional[float]]-typed variable even though every
    # float is a valid Optional[float]; Mapping is covariant and this
    # variable is only ever read (.get()/.items()) after assignment, never
    # mutated in place, so covariance is sound here.
    budget_constant_by_symbol: Mapping[str, Optional[float]] = {}
    # Only populated under risk_budget_mode='idm' -- the notional_weighting
    # SPLIT fraction each active symbol got of the total dollar-vol budget
    # (compute_notional_split -- the same split compute_symbol_
    # notional_budget computes internally, exposed here so a caller can
    # see the actual ERC/HRP weights, not just the resulting dollar
    # figure). Empty under 'cluster' -- that mode has no such split at all
    # (every instrument in a cluster shares one flat budget_constant).
    notional_weight_by_symbol: dict[str, float] = {}
    # Only populated under risk_budget_mode='idm' with account_equity and at
    # least one active symbol -- used below both to scale total_risk_target
    # (when config.apply_cluster_cap) and to compute realized per-symbol/
    # per-cluster risk contributions for the report, after apply_cluster_
    # risk_cap has finalized final_target_contracts and standalone position
    # dollar-vol. idm_multiplier
    # stays 1.0 under 'cluster' mode or when IDM has nothing to compute from
    # -- both correctly leave total_risk_target unscaled below.
    idm_multiplier = 1.0
    active_symbols: list[str] = []
    H: Optional[np.ndarray] = None
    covered: Optional[np.ndarray] = None
    if config.risk_budget_mode == 'cluster':
        active_clusters = {s['cluster'] for s in signals.values() if _is_active(s)}
        n_effective = compute_n_effective(active_clusters)
        account_equity = config.account_equity
        if account_equity:
            desired_risk_budget = compute_desired_risk_budget(account_equity, config.target_portfolio_vol,
                                                               n_effective)
            shared_budget_constant = desired_risk_budget / config.vol_target if config.vol_target else 0.0
        else:
            shared_budget_constant = None
        budget_constant_by_symbol = {s: shared_budget_constant for s in signals}
        log.info('Risk budget (cluster) — active_clusters=%s  n_effective=%d  desired_risk_budget=%s  budget_constant=%s',
                 sorted(active_clusters), n_effective,
                 f'{desired_risk_budget:.0f}' if desired_risk_budget is not None else 'N/A (no account_equity)',
                 f'{shared_budget_constant:.0f}' if shared_budget_constant is not None else 'N/A')
    else:  # 'idm'
        active_symbols = [s for s, sig in signals.items() if _is_active(sig)]
        n_effective = compute_n_effective({signals[s]['cluster'] for s in active_symbols})
        account_equity = config.account_equity
        if account_equity and active_symbols:
            returns_wide = build_returns_wide({s: raw['closes'] for s, raw in raw_by_symbol.items()})
            as_of = config.as_of or date.today()
            H, covered = _bounded_ewm_correlation_matrix(returns_wide, active_symbols, as_of,
                                                          config.corr_window_years, config.corr_halflife_days)
            notional_weight_by_symbol = compute_notional_split(active_symbols, config.notional_weighting,
                                                                H, covered)
            # H/covered passed straight through -- compute_symbol_notional_budget
            # would otherwise rerun the EWM estimation over returns_wide a
            # second time for the exact same (active_symbols, as_of, window)
            # this function just computed above.
            budget_constant_by_symbol = compute_symbol_notional_budget(
                active_symbols, returns_wide, as_of, account_equity, config.target_portfolio_vol,
                config.vol_target, config.corr_window_years, config.corr_halflife_days,
                config.notional_weighting, config.use_idm,
                H=H, covered=covered,
            )
            # Same (active_symbols, H, covered, notional_weight_by_symbol)
            # compute_symbol_notional_budget already used internally to size
            # positions -- recomputed here (cheap: a handful of matrix ops on
            # an n<=a few dozen matrix, not the expensive EWM estimation
            # above) so total_risk_target below can be scaled consistently
            # with it, if config.apply_cluster_cap actually uses it.
            if config.use_idm:
                idm_multiplier = _coverage_restricted_idm(active_symbols, H, covered,
                                                           weights=notional_weight_by_symbol)
        log.info('Risk budget (idm) — active_symbols=%s  n_effective=%d  notional_weighting=%s  use_idm=%s  '
                 'idm_multiplier=%.3f  budget_constant=%s',
                 sorted(active_symbols), n_effective, config.notional_weighting, config.use_idm, idm_multiplier,
                 {s: round(v, 0) for s, v in budget_constant_by_symbol.items()} or 'N/A (no account_equity or no active symbols)')

    # Stage 3: per-instrument sizing off the derived budget, then the
    # cluster risk cap as a second pass.
    targets = []
    for instr in instruments:
        symbol = instr['symbol']
        if symbol in errors:
            targets.append({
                'symbol': symbol,
                'final_target_contracts': None,
                'current_contracts': None,
                'signal': None,
                'vx_ratio': vx_ratio,
                'vol_regime': vol_regime,
                'error': errors[symbol],
                'account_equity': config.account_equity,
                'n_effective_clusters': n_effective,
                'cluster_dollar_vol_budget': desired_risk_budget,
                'vol_target': config.vol_target,
                'target_portfolio_vol': config.target_portfolio_vol,
                'pre_scalar_notional_budget': budget_constant_by_symbol.get(symbol),
                'max_cluster_risk_pct': config.max_cluster_risk_pct,
                'max_lot_overrun_pct': config.max_lot_overrun_pct,
            })
            continue

        s = signals[symbol]
        active = _is_active(s)
        try:
            multiplier = s['multiplier']
            max_contracts = instr.get('max_contracts', config.max_contracts)
            max_notional_ceiling = instr.get('max_notional')
            budget_constant = budget_constant_by_symbol.get(symbol)

            if multiplier is None:
                raise ValueError(f'{symbol}: instrument config missing multiplier')
            if budget_constant is None:
                if config.account_equity is None:
                    raise ValueError(f'{symbol}: account_equity not configured — cannot derive a risk budget')
                # risk_budget_mode='idm' and this symbol simply isn't in
                # active_symbols (below min_conviction) -- not a config
                # error, just nothing to size against. Report a clean
                # target=0 rather than raising -- mirrors
                # risk_budget_mode='cluster', where an inactive symbol
                # still gets sized (possibly to 0 on its own merits) off
                # the shared cluster budget instead of being excluded
                # from the report outright.
                targets.append({
                    'symbol': symbol, 'final_target_contracts': 0,
                    'fractional_target_contracts': 0.0,
                    'max_contracts': max_contracts,
                    'current_contracts': (_current_contracts(ib, s['contract'])
                                    if config.data_source == 'ib' else None),
                    'active': active,
                    'signal': s['signal'], 'combined_scalar': None,
                    'ts_fast': s['ts_fast'], 'ts_slow': s['ts_slow'],
                    'ts': s['ts'], 'contin_sig': s['contin_signal'], 'ts_regime': s['ts_regime'],
                    'daily_std': s['daily_std'], 'hv': s['hv'], 'risk_scalar': s['risk_scalar'],
                    'reg_discount': s['regime_discount'], 'vol_ratio': s['vol_ratio'],
                    'sig_confid_reg': s['signal_confidence_regime'],
                    'sig_confid': s['signal_confidence'], 'vix_scalar': vix_scalar,
                    'close': s['close'], 'mult': multiplier,
                    'uncapped_fractional_target_notional': None, 'fractional_target_notional': None,
                    'cluster': s['cluster'], 'dd_pct': s['dd_pct'],
                    'vx_current': vx_current, 'vx_ma': vx_ma, 'vx_ratio': vx_ratio, 'vol_regime': vol_regime,
                    'g_regime': s['g_regime'], 'g_fast': s['g_fast'], 'g_slow': s['g_slow'],
                    'g_blend': s['g_blend'], 'a_co': s['a_co'], 'a_re': s['a_re'],
                    'account_equity': config.account_equity,
                    'n_effective_clusters': n_effective,
                    'cluster_dollar_vol_budget': desired_risk_budget,
                    'vol_target': config.vol_target,
                    'target_portfolio_vol': config.target_portfolio_vol,
                    'pre_scalar_notional_budget': None,
                    'notional_allocation_weight': None,
                    'idm_multiplier': idm_multiplier if config.risk_budget_mode == 'idm' else None,
                    'risk_budget_mode': config.risk_budget_mode,
                    'notional_weighting': config.notional_weighting,
                    'use_idm': config.use_idm,
                    'max_cluster_risk_pct': config.max_cluster_risk_pct,
                    'max_lot_overrun_pct': config.max_lot_overrun_pct,
                })
                continue

            combined_scalar = compute_position_scalar(
                s['signal_for_scalar'], s['daily_std'], config.vol_target, s['regime'],
                regime_discount=s['regime_discount'], signal_confidence=s['signal_confidence'],
                annualization_days=s['annualization_days'],
            )
            combined_scalar *= vix_scalar

            # uncapped_fractional_target_notional is budget_constant * combined_scalar
            # before the optional per-instrument max_notional ceiling clamp;
            # fractional_target_notional is what drives fractional_target_contracts
            # below. They only differ when max_notional_ceiling clips the
            # uncapped target.
            uncapped_fractional_target_notional = budget_constant * combined_scalar
            fractional_target_notional = uncapped_fractional_target_notional
            if max_notional_ceiling is not None:
                fractional_target_notional = max(
                    -max_notional_ceiling,
                    min(max_notional_ceiling, fractional_target_notional),
                )

            one_contract_notional = s['close'] * multiplier
            # fractional_target_contracts is the unrounded, unclamped value the
            # cluster cap operates on -- rescaling and rounding an already-
            # rounded-and-clamped integer (the provisional final target below)
            # double-rounds, which can zero out large-multiplier instruments
            # (full-size ES/NQ/JPY/etc) that would survive on the true
            # continuous math. The provisional final target is still computed the same
            # way here for any caller that wants a pre-cluster-cap integer
            # (e.g. granularity-tracking instrumentation) -- apply_cluster_
            # risk_cap is what now does the real, single round+clamp.
            fractional_target_contracts = (
                fractional_target_notional / one_contract_notional if one_contract_notional else 0.0
            )
            final_target_contracts = (
                round(fractional_target_notional / one_contract_notional) if one_contract_notional else 0
            )
            final_target_contracts = max(-max_contracts, min(max_contracts, final_target_contracts))

            # No IB connection in 'database' mode -- current_contracts is
            # unknowable without one, reported as None rather than a
            # misleading 0 (this is pure signal inspection, not a claim
            # about any real position).
            current_contracts = _current_contracts(ib, s['contract']) if config.data_source == 'ib' else None

            targets.append({
                'symbol': symbol,
                'final_target_contracts': final_target_contracts,
                'fractional_target_contracts': fractional_target_contracts,
                'max_contracts': max_contracts,
                'current_contracts': current_contracts,
                'active': active,
                'signal': s['signal'],
                'combined_scalar': combined_scalar,
                'ts_fast': s['ts_fast'],
                'ts_slow': s['ts_slow'],
                'ts': s['ts'],
                'contin_sig': s['contin_signal'],
                'ts_regime': s['ts_regime'],
                'daily_std': s['daily_std'],
                'hv': s['hv'],
                'risk_scalar': s['risk_scalar'],
                'reg_discount': s['regime_discount'],
                'vol_ratio': s['vol_ratio'],
                'sig_confid_reg': s['signal_confidence_regime'],
                'sig_confid': s['signal_confidence'],
                'vix_scalar': vix_scalar,
                'close': s['close'],
                'mult': multiplier,
                'uncapped_fractional_target_notional': uncapped_fractional_target_notional,
                'fractional_target_notional': fractional_target_notional,
                'cluster': s['cluster'],
                'dd_pct': s['dd_pct'],
                'vx_current': vx_current,
                'vx_ma': vx_ma,
                'vx_ratio': vx_ratio,
                'vol_regime': vol_regime,
                # Goulding audit fields -- None in 'continuous' mode.
                'g_regime': s['g_regime'], 'g_fast': s['g_fast'], 'g_slow': s['g_slow'],
                'g_blend': s['g_blend'], 'a_co': s['a_co'], 'a_re': s['a_re'],
                # Portfolio-level context -- identical across every
                # instrument under risk_budget_mode='cluster', per-symbol
                # under 'idm' -- included per-row so each CSV row is
                # self-contained (no need to cross-reference the log for
                # what budget/equity this run used).
                'account_equity': config.account_equity,
                'n_effective_clusters': n_effective,
                'cluster_dollar_vol_budget': desired_risk_budget,
                'vol_target': config.vol_target,
                'target_portfolio_vol': config.target_portfolio_vol,
                'pre_scalar_notional_budget': budget_constant,
                'notional_allocation_weight': notional_weight_by_symbol.get(symbol),
                'idm_multiplier': idm_multiplier if config.risk_budget_mode == 'idm' else None,
                'risk_budget_mode': config.risk_budget_mode,
                'notional_weighting': config.notional_weighting if config.risk_budget_mode == 'idm' else None,
                'use_idm': config.use_idm if config.risk_budget_mode == 'idm' else None,
                'max_cluster_risk_pct': config.max_cluster_risk_pct,
                'max_lot_overrun_pct': config.max_lot_overrun_pct,
            })
        except Exception as exc:
            log.error('Failed to compute rebalance target for %s: %s', symbol, exc)
            targets.append({
                'symbol': symbol,
                'final_target_contracts': None,
                'current_contracts': None,
                'signal': None,
                'vx_ratio': vx_ratio,
                'vol_regime': vol_regime,
                'error': str(exc),
            })

    portfolio_risk_target = (
        config.account_equity * config.target_portfolio_vol
        if config.account_equity else None
    )
    idm_risk_target = None
    total_risk_target = portfolio_risk_target
    if config.risk_budget_mode == 'idm' and total_risk_target is not None:
        # Keep the cap consistent with the diversification credit
        # compute_symbol_notional_budget already used to size these
        # positions -- see TsmomLiveConfig.apply_cluster_cap's own
        # docstring for why leaving this unscaled silently claws that
        # credit back.
        total_risk_target *= idm_multiplier
        idm_risk_target = total_risk_target
    if config.discrete_allocation == 'lot-aware':
        allocate_lot_aware_targets(
            targets,
            portfolio_risk_target,
            risk_overrun_pct=config.discrete_risk_overrun_pct,
            H=H,
            active_symbols=active_symbols,
            min_fractional_contracts=config.min_fractional_contracts,
        )
    elif config.discrete_allocation == 'flat-cluster-diversified':
        allocate_flat_cluster_diversified_targets(
            targets,
            portfolio_risk_target,
            risk_overrun_pct=config.discrete_risk_overrun_pct,
            H=H,
            active_symbols=active_symbols,
            max_cluster_risk_pct=config.max_cluster_risk_pct,
            total_risk_target=total_risk_target,
            n_active_clusters=n_effective,
            apply_cluster_cap=config.apply_cluster_cap,
        )
    else:
        apply_cluster_risk_cap(targets, config.max_cluster_risk_pct, total_risk_target, n_effective,
                              max_lot_overrun_pct=config.max_lot_overrun_pct,
                              apply_cap=config.apply_cluster_cap)

    # Realized per-symbol risk contribution -- informational, computed from
    # whatever targets actually ended up being (capped or not, per
    # config.apply_cluster_cap above), not gated by it. Only possible under
    # risk_budget_mode='idm', where H is the real measured correlation
    # matrix active_symbols was built from ('cluster' mode never computes
    # H at all, so there's nothing to feed compute_realized_portfolio_risk
    # here -- H stays None, this is skipped, print_cluster_risk_report
    # falls back to standalone position dollar-vol totals only). See compute_realized_
    # portfolio_risk's own docstring: this is a different question from
    # idm_multiplier above -- not "how much can the book be scaled up",
    # but "given the ACTUAL rounded positions, what does each symbol
    # really contribute to total portfolio risk."
    realized_portfolio_risk = None
    if H is not None and active_symbols:
        # Signed by final_target_contracts' direction -- see compute_realized_
        # portfolio_risk's own docstring for why an unsigned magnitude
        # here would silently drop every short's netting (or compounding)
        # interaction with the rest of the book.
        dollar_exposure = {
                           t['symbol']: math.copysign(
                               t['standalone_position_dollar_vol'],
                               t['final_target_contracts'],
                           )
                           for t in targets
                           if not t.get('error')
                           and t.get('standalone_position_dollar_vol') is not None
                           and t.get('final_target_contracts') is not None}
        realized = compute_realized_portfolio_risk(active_symbols, H, dollar_exposure)
        realized_portfolio_risk = realized['port_vol']
        for t in targets:
            if t['symbol'] in realized['portfolio_risk_contribution']:
                t['portfolio_risk_contribution'] = (
                    realized['portfolio_risk_contribution'][t['symbol']]
                )

    _attach_sizing_diagnostics(
        targets,
        portfolio_risk_target=portfolio_risk_target,
        idm_risk_target=idm_risk_target,
        realized_portfolio_risk=realized_portfolio_risk,
        discrete_allocation=config.discrete_allocation,
        discrete_risk_overrun_pct=config.discrete_risk_overrun_pct,
        min_fractional_contracts=config.min_fractional_contracts,
    )

    return targets


def _resolve_contract(ib: IBPySync, instr: dict, min_days: int):
    from ib_tools.ibpysync import IBPySync
    ib_symbol = instr.get('ib_symbol') or instr['symbol']
    # Only pass multiplier when ib_symbol diverges from our local symbol
    # (i.e. a genuine same-ticker collision like SI/SIL) -- passing it
    # unconditionally risks breaking already-working contracts if our
    # multiplier's string formatting doesn't exactly match what IB has on
    # file (e.g. "0.5" vs "0.50").
    multiplier = str(instr.get('multiplier', '') or '') if ib_symbol != instr['symbol'] else ''
    expiry = instr.get('expiry', 'auto')
    if expiry == 'auto':
        expiry = get_nearest_quarterly_expiry(ib, ib_symbol, instr.get('exchange', 'CME'), min_days,
                                               multiplier=multiplier)
    contract = IBPySync.future(ib_symbol, exchange=instr.get('exchange', 'CME'), expiration=expiry,
                               multiplier=multiplier)
    ib.qualify_contracts(contract)
    return contract


def _fmt(v, spec='+.3f'):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'N/A'
    return format(v, spec)


def print_rebalance_report(targets: list[dict]) -> str:
    """Pretty-print (and return as a string) the rebalancing plan, grouped
    as a left-to-right narrative: identity/result -> the goulding block
    (regime -> raw g_fast/g_slow -> mixing weights a_co/a_re -> blend ->
    g_sig, the resolved direction; None under signal_weighting=
    'continuous') -> a fully separate continuous-momentum block (ALWAYS
    populated regardless of signal_weighting) -> VX context -> the four
    multiplicative sizing components -> their product -> notional sizing
    -> the contract-count conversion.

    Label disambiguation (the same-sounding fields this grouping/labelling
    exists to untangle):
      g_sig (targets' 'signal' field): the raw direction/magnitude call --
        compute_position_scalar's first input, NOT the final sizing
        multiplier. Under signal_weighting='goulding' this is ALWAYS
        exactly +-1 by construction ("Goulding decides direction,
        vol-parity decides size" -- resolve_trend_direction's own
        docstring): direction and size are deliberately separate decisions
        in that mode, so seeing g_sig pinned to +-1 run after run is
        expected, not a sign anything's broken. Under 'continuous' it
        instead carries continuous_momentum's own signal magnitude --
        named to match contin_sig below, not because it's exclusively a
        goulding-mode value.
      risk_scalar / reg_discount / sig_confid / vix_scalar: the four
        independent multiplicative components -- vol-equalization,
        Correction/Rebound conviction discount, per-instrument signal
        trust discount, and portfolio-wide VX de-risking, respectively.
        None of these alone is "the" scalar; each is one ingredient.
      combined_scalar: EXACTLY the product
        g_sig * risk_scalar * reg_discount * sig_confid * vix_scalar --
        entirely reconstructable from the fields already printed to its
        left, kept here as a convenience total rather than making every
        reader do that multiplication by hand. `risk_scalar`'s explicit
        [0.25, 2.0] bound is the volatility-leverage guardrail; the finished
        product is intentionally not re-clamped to [-1, 1].
      ts / contin_sig: continuous_momentum's own ts_fast/ts_slow
        combination -- ts is the tanh-squashed weighted blend BEFORE the
        correction/rebound discount, contin_sig is that same blend AFTER
        it (continuous_momentum's own 'ts'/'signal' columns). ALWAYS
        computed regardless of signal_weighting, so under 'goulding' these
        are a continuous-mode audit trail (what g_sig WOULD read if
        signal_weighting='continuous' were selected instead) -- kept in
        their own block, entirely apart from the goulding block above.
      ts_regime: continuous_momentum's OWN regime classification --
        sign(ts_fast)/sign(ts_slow) only (Bull/Bear/Correction/Rebound),
        computed independently of signal_weighting. THIS is what gates
        contin_sig's correction/rebound discount (contin_sig = ts * 0.5
        when ts_regime is Correction or Rebound, unchanged otherwise).
        There used to be a separate top-level regime= field here too, but
        it was always exactly redundant -- under 'goulding' it was a
        verbatim copy of g_regime (resolve_trend_direction just passes
        g_regime_val through), and under 'continuous' it was
        classify_regime(ts_fast, ts_slow), the exact same formula
        ts_regime already computes. Dropped rather than kept as a third
        near-duplicate name; g_regime is the goulding-mode regime,
        ts_regime is the continuous-mode one, and under 'goulding' they
        can (and do) disagree -- that's real information, not
        redundancy."""
    lines = ['TSMOM Rebalance Report', '=' * 60]
    for t in targets:
        if t.get('error'):
            lines.append(f"{t['symbol']:6s}  ERROR: {t['error']}")
            continue
        lines.append(
            f"{t['symbol']:6s}  final_target_contracts={t['final_target_contracts']!s:>4}  "
            f"current_contracts={t['current_contracts']!s:>4}  "
            f"active={str(t.get('active')):>5}"
            + (f"  |  g_regime={t['g_regime']}  g_fast={_fmt(t.get('g_fast'), '.4f')}  "
               f"g_slow={_fmt(t.get('g_slow'), '.4f')}  a_co={_fmt(t.get('a_co'), '.2f')}  "
               f"a_re={_fmt(t.get('a_re'), '.2f')}  g_blend={_fmt(t.get('g_blend'), '.4f')}  "
               f"g_sig={_fmt(t.get('signal')):>7}"
               if t.get('a_co') is not None else f"  g_sig={_fmt(t.get('signal')):>7}")
            + f"  |  ts_fast={_fmt(t.get('ts_fast')):>7}  ts_slow={_fmt(t.get('ts_slow')):>7}  "
              f"ts={_fmt(t.get('ts')):>7}  contin_sig={_fmt(t.get('contin_sig')):>7}  "
              f"ts_regime={t['ts_regime'].capitalize() if t.get('ts_regime') else 'N/A':<10}  "
              f"dd_pct={_fmt(t.get('dd_pct'), '.2f'):>7}  "
              f"daily_std={_fmt(t.get('daily_std'), '.4f'):>7}  hv={_fmt(t.get('hv'), '.3f'):>6}  "
              f"vx_current={_fmt(t.get('vx_current'), '.2f'):>6}  vx_ma={_fmt(t.get('vx_ma'), '.2f'):>6}  "
              f"vx_ratio={t['vx_ratio']:.3f}  vol_regime={t['vol_regime'].capitalize()}  "
              f"risk_scalar={_fmt(t.get('risk_scalar'), '.3f'):>6}  "
              f"reg_discount={_fmt(t.get('reg_discount'), '.2f'):>5}  "
              f"sig_confid={_fmt(t.get('sig_confid'), '.2f'):>5}  "
              f"vix_scalar={_fmt(t.get('vix_scalar'), '.2f')}  "
              f"combined_scalar={_fmt(t.get('combined_scalar'), '.3f'):>6}  "
              f"pre_scalar_notional_budget={_fmt(t.get('pre_scalar_notional_budget'), '.0f')}"
            + (f"  notional_allocation_weight={_fmt(t.get('notional_allocation_weight'), '.3f')}"
               if t.get('notional_allocation_weight') is not None else "")
            + (f"  idm_multiplier={_fmt(t.get('idm_multiplier'), '.3f')}"
               if t.get('idm_multiplier') is not None else "")
            + f"  pre_scalar_dollar_vol_budget={_fmt(t.get('pre_scalar_dollar_vol_budget'), '.0f')}  "
              f"fractional_target_dollar_vol={_fmt(t.get('fractional_target_dollar_vol'), '.0f')}  "
              f"fractional_target_notional={_fmt(t.get('fractional_target_notional'), '.0f')}  "
              f"close={_fmt(t.get('close'), '.2f'):>9}  "
              f"one_contract_notional={_fmt(t.get('one_contract_notional'), '.0f')}  "
              f"one_contract_dollar_vol={_fmt(t.get('one_contract_dollar_vol'), '.0f')}  "
              f"fractional_target_contracts={_fmt(t.get('fractional_target_contracts'), '.3f'):>7}  "
              f"rounding_gap={_fmt(t.get('rounding_gap'), '.3f')}  "
              f"scalar_capped={t.get('scalar_capped')}"
            + (f"  zero_reason={t.get('zero_reason')}" if t.get('zero_reason') else "")
            + ("  INFEASIBLE (cluster cap < min contract risk in this cluster)" if t.get('infeasible') else "")
        )
    report = '\n'.join(lines)
    print(report)
    return report


def print_cluster_risk_report(targets: list[dict], account_equity: Optional[float] = None) -> str:
    """Pretty-print (and return as a string) per-cluster totals: standalone
    position dollar-vol
    (each symbol's own undiversified, standalone dollar risk, summed by
    cluster) alongside portfolio risk contribution (diversification-aware -- what
    each symbol/cluster ACTUALLY contributes to total portfolio risk, per
    compute_realized_portfolio_risk) when available. Portfolio risk contribution is
    only ever populated under risk_budget_mode='idm' (compute_
    rebalance_targets' own docstring) -- 'cluster' mode's report falls back
    to standalone position dollar-vol totals alone, with no contribution column.

    Each cluster header is followed by its member instruments (sorted by
    symbol), so the totals can be traced back to what's actually driving
    them, then a per-cluster subtotal line.

    account_equity (optional): when given, the TOTAL line also shows each
    figure as a % of equity. Only portfolio risk contribution's pct is a true
    portfolio-vol read (Euler's theorem: its sum equals realized portfolio
    risk) -- standalone position dollar-vol's pct sums per-symbol risk
    ignoring correlation, so it overstates actual vol for any correlated
    book and should not be read as "the" vol figure.

    Computed from whatever `targets` actually ended up being -- capped or
    not, per TsmomLiveConfig.apply_cluster_cap -- not itself gated by that
    flag; this is informational either way, exactly the comparison that
    surfaced apply_cluster_risk_cap silently reversing IDM's own
    diversification credit in the first place."""
    cluster_by_symbol = {t['symbol']: t['cluster'] for t in targets if t.get('cluster')}
    standalone_position_dollar_vol = {
        t['symbol']: t['standalone_position_dollar_vol']
        for t in targets if t.get('standalone_position_dollar_vol') is not None
    }
    portfolio_risk_contribution = {
        t['symbol']: t['portfolio_risk_contribution']
        for t in targets if t.get('portfolio_risk_contribution') is not None
    }

    cluster_standalone_dollar_vol = group_by_cluster(cluster_by_symbol, standalone_position_dollar_vol)
    cluster_portfolio_risk_contribution = (
        group_by_cluster(cluster_by_symbol, portfolio_risk_contribution)
        if portfolio_risk_contribution else {}
    )

    symbols_by_cluster: dict[str, list[str]] = {}
    for symbol, cluster in cluster_by_symbol.items():
        symbols_by_cluster.setdefault(cluster, []).append(symbol)

    lines = ['TSMOM Cluster Risk Report', '=' * 60]
    for cluster in sorted(set(cluster_standalone_dollar_vol) | set(cluster_portfolio_risk_contribution)):
        lines.append(f"{cluster}:")
        for symbol in sorted(symbols_by_cluster.get(cluster, [])):
            standalone = standalone_position_dollar_vol.get(symbol, 0.0)
            line = f"  {symbol:10s}  standalone_position_dollar_vol={standalone:>12,.0f}"
            if portfolio_risk_contribution:
                contribution = portfolio_risk_contribution.get(symbol, 0.0)
                line += f"  portfolio_risk_contribution={contribution:>12,.0f}"
            lines.append(line)
        standalone = cluster_standalone_dollar_vol.get(cluster, 0.0)
        line = f"  {'subtotal':10s}  standalone_position_dollar_vol={standalone:>12,.0f}"
        if cluster_portfolio_risk_contribution:
            contribution = cluster_portfolio_risk_contribution.get(cluster, 0.0)
            line += f"  portfolio_risk_contribution={contribution:>12,.0f}"
        lines.append(line)
    lines.append('-' * 60)
    total_standalone_dollar_vol = sum(cluster_standalone_dollar_vol.values())
    total_line = (
        f"{'TOTAL':12s}  standalone_position_dollar_vol={total_standalone_dollar_vol:>12,.0f}"
    )
    if account_equity:
        total_line += f" ({total_standalone_dollar_vol / account_equity:>5.1%})"
    if cluster_portfolio_risk_contribution:
        total_portfolio_risk_contribution = sum(cluster_portfolio_risk_contribution.values())
        total_line += f"  portfolio_risk_contribution={total_portfolio_risk_contribution:>12,.0f}"
        if account_equity:
            total_line += f" ({total_portfolio_risk_contribution / account_equity:>5.1%})"
    lines.append(total_line)
    context = next((t for t in targets if t.get('portfolio_risk_target') is not None), None)
    if context is not None:
        lines.append(
            f"TARGETS       portfolio_risk_target={context['portfolio_risk_target']:,.0f}"
            + (f"  discrete_allocation={context.get('discrete_allocation')}"
               if context.get('discrete_allocation') is not None else '')
            + (f"  min_fractional_contracts={context['min_fractional_contracts']:.2f}"
               if context.get('discrete_allocation') == 'lot-aware'
               and context.get('min_fractional_contracts') is not None else '')
            + (f"  integer_risk_limit={context['integer_risk_limit']:,.0f}"
               if context.get('integer_risk_limit') is not None else '')
            + (f"  idm_risk_target={context['idm_risk_target']:,.0f}"
               if context.get('idm_risk_target') is not None else '')
            + (f"  realized_portfolio_risk={context['realized_portfolio_risk']:,.0f}"
               if context.get('realized_portfolio_risk') is not None else '')
        )
    report = '\n'.join(lines)
    print(report)
    return report
