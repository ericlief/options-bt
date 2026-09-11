"""
CLI entry point for the TSMOM monthly rebalance (live, via IB).

Run (dry-run, default — no orders placed):
    python -m derivatives_bt_engine.live.run_tsmom_rebalance --instruments MES,MNQ

Live (places orders after a typed confirmation):
    python -m derivatives_bt_engine.live.run_tsmom_rebalance --instruments MES,MNQ --live

--instruments accepts either a comma-separated symbol list (defaults below
fill in exchange/multiplier/max_notional) or a path to a JSON file with the
full instrument config list (see tsmom_rebalance.py docstring for the
format: symbol, exchange, expiry, multiplier, max_contracts, max_notional).

This depends only on ib_tools.ibpysync / ib_tools.alerts for IB
connectivity — all strategy/signal logic lives in derivatives_bt_engine.
"""

import argparse
import atexit
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from ib_insync import Order

from ib_tools.alerts import send_telegram
from ib_tools.ibpysync import IBPySync

from derivatives_bt_engine.live.tsmom_rebalance import (
    DATA_SOURCES,
    DISCRETE_ALLOCATIONS,
    MIXING_POOLS,
    RISK_BUDGET_MODES,
    SIGNAL_WEIGHTINGS,
    TsmomLiveConfig,
    build_instruments,
    compute_rebalance_targets,
    print_cluster_risk_report,
    print_rebalance_report,
    _resolve_contract,
)
from derivatives_bt_engine.domain.allocation import NOTIONAL_WEIGHTING_SCHEMES
from derivatives_bt_engine.utils.logger import setup_logger

load_dotenv()

# This file is also supported through ``python -m``. In that form its module
# name becomes ``__main__``, which is outside the package logger hierarchy;
# use the canonical package name so the shared file handler receives it either
# way.
log = logging.getLogger('derivatives_bt_engine.live.run_tsmom_rebalance')

DEFAULT_MAX_NOTIONAL = float(os.getenv('TSMOM_DEFAULT_MAX_NOTIONAL', '0')) or None

# CSV-column labels only -- target dictionaries deliberately keep descriptive
# canonical names for code/API use, while the saved, wide rebalance CSV uses
# compact names that remain specific about sizing stage and units. Keeping
# this at the output boundary avoids reviving ambiguous internal names such as
# `raw_not` or `budg_const`.
_CSV_COLUMN_LABEL = {
    'symbol': 'symbol',
    'current_contracts': 'cur_con',
    'final_target_contracts': 'tgt_con',
    'fractional_target_contracts': 'frac_con',
    'max_contracts': 'max_con',
    'infeasible': 'infeas',
    'active': 'active',
    'cluster': 'cluster',
    'close': 'close',
    'mult': 'mult',
    'daily_std': 'day_std',
    'hv': 'hv',
    'dd_pct': 'dd_pct',
    'g_regime': 'g_regime',
    'g_fast': 'g_fast',
    'g_slow': 'g_slow',
    'a_co': 'a_co',
    'a_re': 'a_re',
    'g_blend': 'g_blend',
    'signal': 'g_sig',
    'vol_regime': 'vol_reg',
    'ts_fast': 'ts_fast',
    'ts_slow': 'ts_slow',
    'ts': 'ts',
    'contin_sig': 'contin_sig',
    'ts_regime': 'ts_reg',
    'risk_scalar': 'risk_sc',
    'reg_discount': 'reg_disc',
    'sig_confid_reg': 'sig_conf_reg',
    'sig_confid': 'sig_conf',
    'vol_ratio': 'vol_ratio',
    'vix_scalar': 'vx_sc',
    'combined_scalar': 'comb_sc',
    'idm_multiplier': 'idm_mult',
    'account_equity': 'acct_eq',
    'n_effective_clusters': 'n_eff',
    'portfolio_risk_target': 'port_risk_tgt',
    'idm_risk_target': 'idm_risk_tgt',
    'realized_portfolio_risk': 'real_port_risk',
    'discrete_allocation': 'disc_alloc',
    'discrete_risk_overrun_pct': 'disc_over_pct',
    'min_fractional_contracts': 'min_frac_con',
    'integer_risk_limit': 'int_risk_lim',
    'integer_zero_reason': 'int_zero_why',
    'cluster_dollar_vol_budget': 'clust_dvol_bud',
    'vol_target': 'vol_tgt',
    'target_portfolio_vol': 'port_vol_tgt',
    'pre_scalar_notional_budget': 'pre_sc_not_bud',
    'notional_allocation_weight': 'not_alloc_w',
    'pre_scalar_dollar_vol_budget': 'pre_sc_dvol_bud',
    'fractional_target_dollar_vol': 'frac_tgt_dvol',
    'standalone_position_dollar_vol': 'pos_dvol',
    'portfolio_risk_contribution': 'port_risk_con',
    'uncapped_fractional_target_notional': 'uncap_frac_tgt_not',
    'fractional_target_notional': 'frac_tgt_not',
    'one_contract_notional': 'one_con_not',
    'one_contract_dollar_vol': 'one_con_dvol',
    'rounding_gap': 'rnd_gap',
    'scalar_capped': 'risk_sc_bound',
    'zero_reason': 'zero_why',
    'max_cluster_risk_pct': 'max_clust_risk_pct',
    'max_lot_overrun_pct': 'max_lot_over_pct',
    'risk_budget_mode': 'risk_budg_mode',
    'notional_weighting': 'not_weighting',
    'use_idm': 'use_idm',
    'vx_current': 'vx_cur',
    'vx_ma': 'vx_ma',
    'vx_ratio': 'vx_ratio',
    'error': 'error',
}


def _csv_label(key: str) -> str:
    """Compact CSV label for a descriptive target-dictionary field name."""
    return _CSV_COLUMN_LABEL.get(key, key)


def configure_logging() -> str:
    """Use the package-wide file handler shared by live and backtest modules."""
    package_logger = setup_logger()
    log_path = next(
        (handler.baseFilename for handler in package_logger.handlers
         if handler.get_name() == 'derivatives_bt_engine_file'),
        None,
    )
    return log_path or 'unavailable'


def connect_with_retry(ib: IBPySync, host, ports, client_id, interval=30):
    """Try each port in order (IBG then TWS). Loops until one accepts."""
    while True:
        for port in ports:
            try:
                ib.connect(host, port, client_id)
                log.info('Connected at %s:%s client_id=%s', host, port, client_id)
                return
            except Exception as exc:
                log.warning('Cannot connect to %s:%s (%s)', host, port, exc)
        log.info('No gateway available — retrying in %ss', interval)
        time.sleep(interval)


def _build_instruments(spec: str, max_notional: float, max_contracts: int) -> list[dict]:
    """CLI-only wrapper: --instruments accepts either a comma-separated
    symbol list OR a JSON config path, unlike tsmom_rebalance.py's own
    build_instruments (symbol list only, notebook-facing, no CLI/JSON
    concerns). Delegates the symbol-list case to that shared function
    instead of duplicating its KNOWN_INSTRUMENTS lookup/fallback logic."""
    path = Path(spec)
    if path.exists() and path.suffix == '.json':
        with open(path) as f:
            return json.load(f)
    return build_instruments(spec.split(','), max_notional=max_notional, max_contracts=max_contracts)


def _save_report(cluster_report: str, targets: list[dict], mixing_diagnostics: Optional[dict] = None) -> None:
    """Persists each run's cluster risk report (plain text, matches stdout)
    and targets (CSV, one row per instrument) to results/ at the project
    root, timestamped -- mirrors tsmom.py's results dir so live and
    backtest output live in the same place. Previously this only ever
    printed to stdout/Telegram and was lost the moment the terminal
    scrolled.

    Only print_cluster_risk_report's output goes to the .txt file --
    print_rebalance_report's own per-instrument dump is redundant with the
    CSV (every field it prints, plus more, is a CSV column) and duplicated
    every field the cluster report doesn't already summarize; portfolio risk
    contribution (when present -- only under risk_budget_mode='idm', see
    compute_rebalance_targets' own docstring) already flows into the CSV
    automatically via all_keys below, no separate handling needed there.

    The lot-aware iterative decision trail belongs exclusively in the shared
    run log under ``logs/``. Keeping it out of this text report avoids
    duplicating the audit while preserving this compact cluster-risk summary.

    `mixing_diagnostics` (compute_rebalance_targets' own out-param, {cluster
    or 'global': diag}, populated only under signal_weighting='goulding')
    -- when non-empty, ALSO saves a small tsmom_mixing_params_{ts}.csv, one
    row per cluster (or a single 'global' row under mixing_pool='global'),
    with every intermediate value behind that cluster's a_co/a_re (C, 1/C,
    per-state avg_r/avg_r2/kelly, the RAW pre-clamp a_co_raw/a_re_raw,
    which fallback if any) -- see estimate_mixing_params_diagnostics' own
    docstring. Lets a saturated a_co/a_re (== exactly 0.0 or 1.0 in the
    main CSV) be traced back to the small/noisy pooled sample that
    produced it, instead of looking identical to a genuine interior
    calibration. None/empty (e.g. 'continuous' mode, or a caller that
    doesn't pass mixing_diagnostics at all) skips this file entirely."""
    results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    txt_path = os.path.join(results_dir, f'tsmom_live_rebalance_{ts}.txt')
    with open(txt_path, 'w') as f:
        f.write(cluster_report)

    if mixing_diagnostics:
        mixing_fieldnames = ['cluster', 'fallback_reason', 'a_co', 'a_re', 'a_co_raw', 'a_re_raw', 'C', 'inv_C',
                              'n_bull', 'n_bear', 'n_correction', 'n_rebound',
                              'avg_r_bull', 'avg_r2_bull', 'kelly_bull',
                              'avg_r_bear', 'avg_r2_bear', 'kelly_bear',
                              'avg_r_correction', 'avg_r2_correction', 'kelly_correction',
                              'avg_r_rebound', 'avg_r2_rebound', 'kelly_rebound']
        mixing_rows = [
            {'cluster': key, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in diag.items()
                                 if k != 'cluster'}}
            for key, diag in sorted(mixing_diagnostics.items())
        ]
        mixing_csv_path = os.path.join(results_dir, f'tsmom_mixing_params_{ts}.csv')
        with open(mixing_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=mixing_fieldnames)
            writer.writeheader()
            writer.writerows(mixing_rows)
        log.info('Saved per-cluster mixing-params diagnostics to %s', mixing_csv_path)

    # symbol -> current/target position first (the columns you actually
    # scan a rebalance report for), THEN one dedicated goulding-decision
    # block (regime -> raw g_fast/g_slow -> mixing weights a_co/a_re ->
    # blend -> the resolved direction, g_sig -- None under
    # signal_weighting='continuous'), THEN a fully separate continuous-
    # momentum-only block (ts_fast/ts_slow/ts/contin_sig are ALWAYS
    # computed regardless of signal_weighting, so this block is populated
    # even in 'goulding' mode -- kept apart from the goulding block above
    # so the two models' columns never interleave), and finally the
    # sizing-math trail / run-level (CLI-echoed) params, in the order
    # you'd actually want to follow that calculation.
    priority = ['symbol', 'current_contracts', 'final_target_contracts',
                'fractional_target_contracts', 'max_contracts', 'infeasible',
                'g_regime', 'g_fast', 'g_slow', 'a_co', 'a_re', 'g_blend', 'g_sig',
                'vol_regime', 'ts_fast', 'ts_slow', 'ts', 'contin_sig', 'ts_regime',
                'risk_scalar', 'reg_discount',
                'sig_confid_reg', 'sig_confid', 'vol_ratio', 'vix_scalar',
                'combined_scalar', 'idm_multiplier',
                'account_equity', 'n_effective_clusters',
                'portfolio_risk_target', 'idm_risk_target', 'realized_portfolio_risk',
                'discrete_allocation', 'discrete_risk_overrun_pct', 'min_fractional_contracts',
                'integer_risk_limit',
                'integer_zero_reason',
                'cluster_dollar_vol_budget', 'vol_target', 'target_portfolio_vol',
                'pre_scalar_notional_budget', 'notional_allocation_weight',
                'pre_scalar_dollar_vol_budget', 'fractional_target_dollar_vol',
                'standalone_position_dollar_vol', 'portfolio_risk_contribution',
                'uncapped_fractional_target_notional', 'fractional_target_notional', 'one_contract_notional',
                'one_contract_dollar_vol',
                'rounding_gap', 'scalar_capped', 'zero_reason',
                'max_cluster_risk_pct', 'max_lot_overrun_pct']
    rounded_rows = [
        {_csv_label(k): (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v)
         for k, v in t.items()}
        for t in targets
    ]
    all_keys = {key for row in rounded_rows for key in row}
    compact_priority = [_csv_label(key) for key in priority]
    fieldnames = [key for key in compact_priority if key in all_keys] + sorted(all_keys - set(compact_priority))
    csv_path = os.path.join(results_dir, f'tsmom_live_rebalance_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rounded_rows)

    log.info('Saved rebalance report to %s, %s', txt_path, csv_path)


def _execute_rebalance_order(ib: IBPySync, contract, delta_contracts: int):
    """Simple limit-at-mid (falls back to market) order for the size delta
    needed to reach final_target_contracts from current_contracts."""
    action = 'BUY' if delta_contracts > 0 else 'SELL'
    qty = abs(delta_contracts)
    ticker = ib.req_mkt_data(contract)
    ib.sleep(3)
    bid = ticker.bid if ticker.bid and not math.isnan(ticker.bid) else None
    ask = ticker.ask if ticker.ask and not math.isnan(ticker.ask) else None
    if bid and ask:
        mid = round((bid + ask) / 2, 2)
        log.info('%s %d %s LMT %.2f (bid=%.2f ask=%.2f)', action, qty, contract.symbol, mid, bid, ask)
        order = Order(action=action, orderType='LMT', totalQuantity=qty, lmtPrice=mid, tif='DAY')
    else:
        log.warning('No quote for %s — placing MKT order', contract.symbol)
        order = Order(action=action, orderType='MKT', totalQuantity=qty)
    trade = ib.place_order(contract, order)
    ib.wait_fill(trade, 60)
    ib.cancel_mkt_data(contract)
    return trade


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--instruments', required=True,
                   help='Comma-separated symbols (e.g. MES,MNQ) or path to a JSON instrument config')
    p.add_argument('--vol-target', type=float, default=0.15,
                   help='Annualized vol target (default: %(default)s = 15%%)')
    p.add_argument('--account-equity', type=float, required=True,
                   help='Account equity USD -- the primary sizing input: drives the derived '
                        'per-cluster risk budget (account_equity * target_portfolio_vol / sqrt(n_effective))')
    p.add_argument('--target-portfolio-vol', type=float, default=0.15,
                   help='Target total portfolio vol, used to derive the risk budget (default: %(default)s = 15%%)')
    p.add_argument('--max-cluster-risk-pct', type=float, default=0.25,
                   help='Max share of total portfolio dollar-vol risk any one cluster '
                        '(e.g. grains, FX) may carry before being rescaled down (default: %(default)s)')
    p.add_argument('--min-conviction', type=float, default=0.05,
                   help='Min abs(trend_strength) for a cluster to count as "active" '
                        'when deriving n_effective (default: %(default)s)')
    p.add_argument('--max-lot-overrun-pct', type=float, default=0.5,
                   help='Lot-size exception tolerance: the top-priority instrument in an '
                        'over-budget cluster still gets 1 contract (instead of 0) if its own '
                        'single-contract risk is within this fraction over the cluster cap '
                        '(default: %(default)s = 50%%)')
    p.add_argument('--max-contracts', type=int, default=15,
                   help='Per-instrument sanity backstop (not the primary sizing lever -- '
                        'that is the derived risk budget + cluster cap), used when '
                        '--instruments is a symbol list (default: %(default)s)')
    p.add_argument('--max-notional', type=float, default=DEFAULT_MAX_NOTIONAL,
                   help='Optional hard per-instrument notional USD ceiling, used when '
                        '--instruments is a symbol list (default: %(default)s, i.e. no ceiling '
                        'beyond the derived risk budget)')
    p.add_argument('--vx-expiry', default='auto',
                   help='VX futures expiry YYYYMM or "auto" for nearest >=3d (default: %(default)s)')
    p.add_argument('--disable-vix-gating', action='store_true',
                   help="Turn off the portfolio-wide VX/VIX spike gate entirely -- no read of any "
                        "VX/VIX source is attempted (proceeds as vol_regime=Normal). Use this when no "
                        "VX/VIX data source is available at all (default: gating on, matching all "
                        "prior behavior)")
    p.add_argument('--long-only', action='store_true',
                   help='Disable short positions (signal_scalar = max(0, trend_strength))')
    p.add_argument('--regime-discount', type=float, default=0.5,
                   help='Position discount for Correction/Rebound regimes (fast/slow trend sign '
                        'disagreement); 1.0 disables (default: %(default)s)')
    p.add_argument('--enable-signal-confidence', action='store_true',
                   help='Opt in to signal_confidence: a per-instrument discount on trust in that '
                        "instrument's own trend signal when ITS OWN vol_ratio (short/long realized "
                        'vol, asset-specific, NOT VIX/VX-driven) is unusual relative to its own '
                        'history. Off by default -- existing behavior is unchanged unless set.')
    p.add_argument('--signal-confidence-low-threshold', type=float, default=0.7,
                   help='vol_ratio (hv_short/hv_long) at or below this is classified "low" '
                        '(default: %(default)s)')
    p.add_argument('--signal-confidence-high-threshold', type=float, default=1.5,
                   help='vol_ratio (hv_short/hv_long) at or above this is classified "high" '
                        '(default: %(default)s)')
    p.add_argument('--signal-confidence-high-vol', type=float, default=0.5,
                   help='Discount factor applied when vol_ratio is "high" -- vol spikes specifically '
                        'damage momentum reliability (default: %(default)s)')
    p.add_argument('--signal-confidence-low-vol', type=float, default=1.0,
                   help='Discount factor applied when vol_ratio is "low" -- no settled answer for '
                        'whether low vol should discount trend confidence, so this defaults to a '
                        'no-op (default: %(default)s)')
    p.add_argument('--signal-weighting', choices=SIGNAL_WEIGHTINGS, default='continuous',
                   help="Signal DIRECTION source (default: %(default)s). 'continuous': "
                        "continuous_momentum's daily trend_strength + --regime-discount. "
                        "'goulding': monthly Bull/Correction/Bear/Rebound classification with "
                        "a_Co/a_Re mixing weights re-estimated from all available prior history "
                        "(--mixing-pool) -- --regime-discount is ignored in this mode. Position "
                        "size/vol-targeting is unaffected either way; see "
                        "TsmomLiveConfig.signal_weighting's own docstring")
    p.add_argument('--mixing-pool', choices=MIXING_POOLS, default='cluster',
                   help="Only used with --signal-weighting goulding (default: %(default)s). 'cluster': "
                        "a_Co/a_Re estimated separately per instrument cluster. 'global': one shared "
                        "estimate pooled across every --instruments regardless of cluster")
    p.add_argument('--fast-window', type=int, default=63,
                   help="Only used with --signal-weighting continuous: continuous_momentum's fast "
                        "return-horizon window in trading days (default: %(default)s)")
    p.add_argument('--slow-window', type=int, default=252,
                   help="Only used with --signal-weighting continuous: continuous_momentum's slow "
                        "return-horizon window in trading days (default: %(default)s)")
    p.add_argument('--vol-fast-window', type=int, default=None,
                   help="Only used with --signal-weighting continuous: vol-normalization window for "
                        "the fast leg (default: None -> horizon-matched to --fast-window). Pass "
                        "explicitly only to deliberately decouple the return horizon from the vol "
                        "estimation horizon")
    p.add_argument('--vol-slow-window', type=int, default=None,
                   help="Only used with --signal-weighting continuous: vol-normalization window for "
                        "the slow leg (default: None -> horizon-matched to --slow-window)")
    p.add_argument('--risk-budget-mode', choices=RISK_BUDGET_MODES, default='cluster',
                   help="How the risk budget is derived (default: %(default)s). 'cluster': "
                        "compute_n_effective/compute_desired_risk_budget -- one shared budget per "
                        "active cluster (zero-correlation assumption). 'idm': "
                        "compute_symbol_notional_budget -- one correlation-aware budget PER ACTIVE "
                        "SYMBOL, via a bounded trailing EWM correlation matrix (see "
                        "--notional-weighting/--use-idm/--corr-window-years/--corr-halflife-days)")
    p.add_argument('--notional-weighting', choices=NOTIONAL_WEIGHTING_SCHEMES, default='flat',
                   help="Only used with --risk-budget-mode idm (default: %(default)s). How the "
                        "IDM-derived total is split across active symbols -- 'flat': equal split. "
                        "'erc'/'hrp': data-driven, correlation-aware splits -- see "
                        "domain.allocation.compute_symbol_notional_budget's own docstring")
    p.add_argument('--use-idm', action=argparse.BooleanOptionalAction, default=True,
                   help="Only used with --risk-budget-mode idm (default: %(default)s). Whether the "
                        "total budget is scaled by IDM before being split, or left as account_equity "
                        "* --target-portfolio-vol with no correlation-based up/down-sizing")
    p.add_argument('--apply-cluster-cap', action=argparse.BooleanOptionalAction, default=False,
                   help="Whether apply_cluster_risk_cap's cluster-level cap/redistribution runs at all "
                        "(default: %(default)s; whole-contract rounding always happens regardless). "
                        "Default off because under --risk-budget-mode idm, that cap re-imposes a "
                        "hand-assigned-cluster, zero-correlation assumption on top of sizing that's "
                        "already correlation-aware, and can silently claw back most of IDM's own "
                        "diversification credit -- see TsmomLiveConfig.apply_cluster_cap's own "
                        "docstring. When on with --risk-budget-mode idm, the cap's own total risk "
                        "target is scaled by the same idm_multiplier used to size positions, so it "
                        "stays a consistency backstop rather than reversing that credit. Cannot be "
                        "combined with standard --discrete-allocation lot-aware")
    p.add_argument('--discrete-allocation', choices=DISCRETE_ALLOCATIONS, default='lot-aware',
                   help="Whole-contract sizing policy (default: %(default)s). 'independent' keeps "
                        "the existing per-symbol rounding. 'lot-aware' rounds all targets at the "
                        "minimum threshold then only removes lots if measured portfolio risk exceeds "
                        "its limit; it has no cluster rule. 'flat-cluster-diversified' is the "
                        "experimental former flat-book policy that forces a representative per cluster")
    p.add_argument('--discrete-risk-overrun-pct', type=float, default=0.0,
                   help="Only used with correlation-aware discrete policies: permitted realized "
                        "portfolio-risk overrun as a fraction of account_equity * "
                        "--target-portfolio-vol (default: %(default)s = hard cap)")
    p.add_argument('--min-fractional-contracts', type=float, default=0.5,
                   help="Only used with --discrete-allocation lot-aware: minimum absolute "
                        "fractional target allowed to become an initial lot (default: %(default)s; "
                        ".50 is ordinary nearest-integer rounding, .25 promotes .25-.49 to one lot)")
    p.add_argument('--corr-window-years', type=float, default=3.0,
                   help='Only used with --risk-budget-mode idm: bounded trailing window for the EWM '
                        'correlation estimate (default: %(default)s)')
    p.add_argument('--corr-halflife-days', type=float, default=63.0,
                   help='Only used with --risk-budget-mode idm: EWM halflife within the bounded '
                        'window (default: %(default)s)')
    p.add_argument('--data-source', choices=DATA_SOURCES, default='ib',
                   help="Where signal/correlation/mixing-param history comes from (default: "
                        "%(default)s). 'ib': live IB historical bars + the live VX/VIX spike gate -- "
                        "requires a connection. 'database': the same local futures duckdb/VIX parquet "
                        "the backtest reads from, no IB connection anywhere -- runnable in a notebook "
                        "for signal/regime inspection with no live account (current_contracts is "
                        "always None in this mode; --live/order placement requires 'ib')")
    p.add_argument('--as-of', default=None,
                   help='Only used with --data-source database: YYYY-MM-DD to compute signals as of '
                        '(no lookahead past this date) -- default: latest available bar')
    p.add_argument('--splice-live-price', action='store_true', default=False,
                   help="Only meaningful with --data-source database and no --as-of: backfills every "
                        "fresh IB daily bar since the DB series' own last cached row (a plain dated "
                        "Future, not ContFuture -- see TsmomLiveConfig.splice_live_price's own "
                        "docstring), not just today's, for symbols with confirmed active_months -- "
                        "closes the staleness gap when the local db cache hasn't been refreshed in a "
                        "while, without corrupting the rolling-window signal calcs a single-bar patch "
                        "would leave silently distorted. Requires a live IB connection even though "
                        "--data-source database (opens one automatically when this flag is set). "
                        "Default off -- costs extra IB round-trips per spliced symbol per run, and any "
                        "per-symbol splice failure silently falls back to that symbol's plain DB bars.")
    p.add_argument('--bar-years', type=float, default=3.0,
                   help='Historical window: IB request duration / database lookback (default: %(default)s)')
    p.add_argument('--vx-ma-window-days', type=int, default=63,
                   help="Trailing window for the VX/VIX moving average the spike gate compares "
                        "vx_current against, in BOTH --data-source modes (default: %(default)s) -- "
                        "the VX_ELEVATED_RATIO/VX_SPIKE_RATIO/VX_EXTREME_RATIO bands were calibrated "
                        "against this default; changing it changes what those bands mean")
    p.add_argument('--dry-run', action='store_true', default=True,
                   help='Print targets only, no orders (default — this is the safe default)')
    p.add_argument('--no-save', action='store_true',
                   help='Skip saving the report/targets to options-bt/results/ (saved by default)')
    p.add_argument('--paper',    action='store_true')
    p.add_argument('--live', action='store_true',
                   help='Place real orders to reach target_contracts, after a typed confirmation')
    conn = p.add_argument_group('Connection')
    conn.add_argument('--host', default=os.getenv('IB_HOST', '127.0.0.1'))
    conn.add_argument('--port', default=int(os.getenv('IB_PORT', '7497')), type=int)
    conn.add_argument('--client-id', default=int(os.getenv('IB_CLIENT_ID', '15')), type=int)
    return p.parse_args()


def main():
    args = parse_args()
    dry_run = not args.live

    if args.live and args.data_source != 'ib':
        sys.exit(f"--live requires --data-source ib (real orders need a live account) — got "
                 f"--data-source {args.data_source!r}")

    log_path = configure_logging()
    print(f'Logging to {log_path}')

    instruments = _build_instruments(args.instruments, args.max_notional, args.max_contracts)
    as_of = datetime.strptime(args.as_of, '%Y-%m-%d').date() if args.as_of else None
    config = TsmomLiveConfig(
        vol_target=args.vol_target,
        max_contracts=args.max_contracts,
        vx_expiry=args.vx_expiry,
        vix_gating=not args.disable_vix_gating,
        long_only=args.long_only,
        regime_discount=args.regime_discount,
        account_equity=args.account_equity,
        target_portfolio_vol=args.target_portfolio_vol,
        max_cluster_risk_pct=args.max_cluster_risk_pct,
        min_conviction=args.min_conviction,
        max_lot_overrun_pct=args.max_lot_overrun_pct,
        enable_signal_confidence=args.enable_signal_confidence,
        signal_confidence_low_threshold=args.signal_confidence_low_threshold,
        signal_confidence_high_threshold=args.signal_confidence_high_threshold,
        signal_confidence_high_vol=args.signal_confidence_high_vol,
        signal_confidence_low_vol=args.signal_confidence_low_vol,
        signal_weighting=args.signal_weighting,
        mixing_pool=args.mixing_pool,
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        vol_fast_window=args.vol_fast_window,
        vol_slow_window=args.vol_slow_window,
        risk_budget_mode=args.risk_budget_mode,
        notional_weighting=args.notional_weighting,
        use_idm=args.use_idm,
        apply_cluster_cap=args.apply_cluster_cap,
        discrete_allocation=args.discrete_allocation,
        discrete_risk_overrun_pct=args.discrete_risk_overrun_pct,
        min_fractional_contracts=args.min_fractional_contracts,
        corr_window_years=args.corr_window_years,
        corr_halflife_days=args.corr_halflife_days,
        data_source=args.data_source,
        bar_years=args.bar_years,
        vx_ma_window_days=args.vx_ma_window_days,
        as_of=as_of,
        splice_live_price=args.splice_live_price,
    )

    if args.live:
        print('=' * 60)
        print('  LIVE REBALANCE — REAL ORDERS WILL BE PLACED')
        print('=' * 60)
        if sys.stdin.isatty():
            confirm = input("Type 'YES REBALANCE LIVE' to confirm: ")
            if confirm != 'YES REBALANCE LIVE':
                sys.exit('Live rebalance not confirmed — exiting.')
        else:
            log.warning('LIVE rebalance — non-interactive, skipping confirmation prompt')
    else:
        print('DRY RUN — targets only, no orders will be placed')

    ib = None
    if config.data_source == 'ib' or config.splice_live_price:
        ib = IBPySync()
        host      = '127.0.0.1' if args.paper else args.host
        ports     = [4002, 7497] if args.paper else [7496, 4001]
        client_id = 2            if args.paper else args.client_id

        connect_with_retry(ib, host, ports, client_id)
        atexit.register(ib.disconnect)
    else:
        log.info('data_source=database, splice_live_price=False — no IB connection made')

    mixing_diagnostics: dict = {}
    targets = compute_rebalance_targets(instruments, config, ib=ib, mixing_diagnostics=mixing_diagnostics)
    report = print_rebalance_report(targets)
    cluster_report = print_cluster_risk_report(targets, account_equity=args.account_equity)

    if not args.no_save:
        _save_report(cluster_report, targets, mixing_diagnostics)

    if not dry_run:
        send_telegram(f'TSMOM Rebalance\n{report}')

        instr_by_symbol = {i['symbol']: i for i in instruments}
        for t in targets:
            if (t.get('error') or t['final_target_contracts'] is None
                    or t['current_contracts'] is None):
                log.warning('Skipping %s — no valid target', t['symbol'])
                continue
            delta = t['final_target_contracts'] - t['current_contracts']
            if delta == 0:
                log.info('%s already at target (%d) — no order', t['symbol'], t['final_target_contracts'])
                continue
            instr = instr_by_symbol[t['symbol']]
            contract = _resolve_contract(ib, instr, min_days=7)
            log.info('%s: current=%d target=%d delta=%+d', t['symbol'],
                     t['current_contracts'], t['final_target_contracts'], delta)
            trade = _execute_rebalance_order(ib, contract, delta)
            status = trade.orderStatus.status
            log.info('%s order status: %s', t['symbol'], status)

    if ib is not None:
        ib.disconnect()


if __name__ == '__main__':
    main()
