"""
Central futures contract registry: multiplier, margin, commission, and the
symbol-resolution metadata (IBKR/Globex ticker divergence, thin-history
substitution) needed across both the live TSMOM system and the general
backtest path.

Each `INSTRUMENTS` entry carries:

  exchange       -- IB exchange for contract resolution
  multiplier     -- contract notional multiplier (USD per point/cent/etc.)
  cluster        -- risk factor bucket for cluster-cap allocation (live
                    TSMOM only -- unused by the general backtest path)
  initial_margin -- CME SPAN maintenance margin, USD. Moves with volatility/
                    exchange resets -- only ES/MES were directly given;
                    every other value below is a rough estimate, scaled
                    from typical CME margin levels for that product. Verify
                    against current CME/broker figures before relying on
                    them for sizing. Not every symbol has this field --
                    only ones with a confirmed or estimated figure; missing
                    means no data exists yet, not zero.
  active_months  -- standard CME month-letter codes (F,G,H,J,K,M,N,Q,U,V,X,Z
                    for Jan-Dec) this product's underlying REALLY trades
                    actively, confirmed empirically (per-date max-volume
                    winner over 2015-2026, see
                    research/research_futures_roll_logic_and_active_months.md
                    §2) rather than assumed from a generic quarterly
                    calendar. Set directly only on a "full-size" entry (GC,
                    ZC, ES, ...) that was actually queried -- a micro/mini
                    that borrows its full-size sibling's price series
                    (MGC->GC, J7->JPY, ...) does NOT get its own duplicate
                    copy; call resolve_active_months(symbol) instead, which
                    follows the same db_symbol/signal_symbol/ib_symbol
                    fallback chain resolve_price_symbol already uses, so
                    the two lists can never drift out of sync. Missing means
                    either "not yet empirically confirmed" or "confirmed to
                    have no restriction" (e.g. CL trades all 12 months) --
                    see the comment on that specific entry for which; either
                    way, a consumer should treat missing as "don't filter,"
                    the same conservative no-op get_nearest_quarterly_expiry
                    already falls back to today. BRE is deliberately left
                    unset -- its own continuous-series construction has a
                    known, unresolved sticky-anchor bug (see the research
                    doc's §1.2) that make its real active-month pattern
                    unreliable to read off the data as-is; excluded from the
                    live/backtest default universe for now, not deleted from
                    this registry.
  annualization_days -- real trading-days/year for this product's own
                    continuous series, confirmed empirically (median/mean
                    distinct-date count per calendar year, 2011-2025, post
                    Sunday-session-merge fix -- see the research doc's
                    annualization-factor discussion) rather than assumed
                    uniformly as 252 everywhere. Consumed only by
                    tsmom_signal.py's genuine annual-scaling terms (`hv`,
                    `avg_r_fast`/`avg_r_slow`, compute_position_scalar's
                    current_realized_vol) -- deliberately NOT used to derive
                    ts_fast/ts_slow's own window lengths (63/252 trading days,
                    e.g. "3 months"/"1 year" at ~21 trading days/month),
                    which stay fixed, literal day-counts independent of this
                    field: window length (how much history to look back
                    over) and annualization (how to rescale a per-day stat
                    to a per-year unit) are orthogonal design choices that
                    happen to share some of the same underlying numbers by
                    convention, not because one derives from the other.
                    Same missing-means-fallback convention as active_months
                    -- resolve_annualization_days(symbol) returns
                    DEFAULT_ANNUALIZATION_DAYS (252) for anything unset,
                    following the same db_symbol/signal_symbol/ib_symbol
                    chain (a micro/mini inherits its full-size sibling's
                    confirmed figure, never a separate copy). Confirmed
                    values: 252 for the CBOT grains (essentially exact,
                    already the industry-standard default); 259 for the
                    rest of this project's confirmed universe (equity,
                    rates, energy where checked, metals, the non-BRE FX
                    symbols) -- a real, structural difference (CME's ag
                    session hours vs. the near-24h financial complex), not
                    noise. BRE left unset for the same reason as
                    active_months.
  commission     -- per contract, per side (i.e. half of round trip); the
                    backtester's calculate_pnl() doubles it for the full
                    round trip. Reuses per-contract tiers (standard vs
                    micro) across sibling products. Same missing-means-
                    no-data convention as initial_margin.
  ib_symbol      -- IBKR contract ticker; only set when it differs from the
                    dict key (e.g. SIL trades under IBKR root 'SI',
                    disambiguated by multiplier)
  signal_symbol  -- IB continuous-front-month ticker for LIVE signal/
                    covariance history; only set when the traded contract's
                    own history is too short/thin for a reliable estimate
                    (e.g. J7→JPY, MZC→ZC for CBOT micro grains launched
                    ~2025, MES/MNQ/MTN/MCL→ES/NQ/ZN/CL -- CME Micro
                    products launched 2019-2021, comparably new to J7).
                    Read by resolve_signal_symbol() (this module) for the
                    live IB continuous-bars fetch -- shared by
                    derivatives_bt_engine.live.tsmom_rebalance and
                    scripts.tsmom_risk_budget_diagnostic's covariance
                    analysis, so setting this gives both the full-size
                    contract's longer history, not just live rebalancing.
  db_symbol      -- Globex root symbol in daily.asset (Databento CME
                    MDP3.0 feed); only set when it differs from the key AND
                    signal_symbol doesn't already resolve it (resolve_price_
                    symbol falls through db_symbol > signal_symbol >
                    ib_symbol, so MES etc. don't need an explicit db_symbol
                    once signal_symbol is set). Needed on its own only for a
                    pure IBKR/Globex ticker naming divergence with NO
                    history problem -- J7→6J, BRE→6L, JPY→6J (these
                    symbols' own IB history is genuinely fine; the local
                    duckdb just happens to store them under a different
                    ticker).
                    Used by the duckdb continuous-front-month queries in
                    the backtester, diagnostic, and data-quality scripts.

`BACKTEST_ONLY_SPECS` holds multiplier/margin/commission for contracts that
exist only on the general single/multi-symbol backtest path (naked_futures.py,
tsmom_backtester.py) and have never been part of the live TSMOM instrument
universe -- see its own docstring for why this must stay a separate dict
from INSTRUMENTS.

get_spec(symbol) is the single lookup point for a contract's multiplier/
margin/commission -- there is no per-instrument enum (deliberately: an
underlying instrument isn't a distinct *type*, any more than an option's
underlying is -- OptionsType is just CALL/PUT, not one member per
underlying). FuturesStrategy (LONG_FUTURES/SHORT_FUTURES) remains the
correct futures analog to OptionsStrategy: it describes position *shape*,
orthogonal to the traded symbol.

Imported by:
  derivatives_bt_engine.domain.position            -- FuturesPosition margin/mult/
                                            commission (via get_spec)
  derivatives_bt_engine.domain.strategy_config     -- FuturesStrategyConfig validation
                                            (via known_futures_symbols)
  derivatives_bt_engine.domain.futures_signal_generator -- signal margin (via get_spec)
  derivatives_bt_engine.live.run_tsmom_rebalance   -- live rebalancing + order execution
  derivatives_bt_engine.domain.tsmom_backtester    -- historical backtest (via
                                            get_spec, resolve_price_symbol)
  derivatives_bt_engine.strats.naked_futures       -- single-symbol backtest CLI (via
                                            resolve_price_symbol)
  scripts.tsmom_risk_budget_diagnostic  -- ERC/HRP vs cluster-cap diagnostic
  scripts.futures_data_quality          -- data quality checks
"""

from typing import Optional

# ── Tunable defaults ────────────────────────────────────────────────────────
DEFAULT_DB_PATH = '/home/dev/fin/db/globex_mdp_3.0.duckdb'
# Fallback trading-days/year used by resolve_annualization_days() for any
# symbol without an explicit `annualization_days` entry -- the standard
# convention this project's signal math (tsmom_signal.py) already assumed
# uniformly everywhere before this field existed, and (per the Sunday-
# session-merge fix, 2026-07) now the empirically-exact figure for the
# CBOT grains specifically, not just a round-number default.
DEFAULT_ANNUALIZATION_DAYS = 252

# ── Instrument universe ─────────────────────────────────────────────────────
INSTRUMENTS: dict[str, dict] = {
    # ── Equity index ────────────────────────────────────────────────────────
    # MES/MNQ (launched May 2019, ~6y of history) borrow ES/NQ's via
    # signal_symbol -- both for backtesting (resolve_price_symbol falls
    # through to signal_symbol) and for live/diagnostic IB covariance
    # history (resolve_signal_symbol), matching the J7/MZC pattern.
    # active_months on ES/NQ confirmed empirically as the standard financial
    # quarterly cycle -- see research doc §2.2.
    'ES':  {'exchange': 'CME',   'multiplier': 50,         'cluster': 'equity',
            'initial_margin': 34068.38, 'commission': 2.24, 'active_months': ['H', 'M', 'U', 'Z'],
            'annualization_days': 259},
    'MES': {'exchange': 'CME',   'multiplier': 5,          'cluster': 'equity',
            'initial_margin': 3429.11, 'commission': 0.61, 'signal_symbol': 'ES'},
    'NQ':  {'exchange': 'CME',   'multiplier': 20,         'cluster': 'equity',
            'initial_margin': 67582.55, 'commission': 2.24, 'active_months': ['H', 'M', 'U', 'Z'],
            'annualization_days': 259},
    'MNQ': {'exchange': 'CME',   'multiplier': 2,          'cluster': 'equity',
            'initial_margin': 6607.19, 'commission': 0.61, 'signal_symbol': 'NQ'},

    # ── Energy ──────────────────────────────────────────────────────────────
    # CL/MCL clear on NYMEX (crude oil), not COMEX (metals only). MCL
    # (launched 2021) borrows CL's via signal_symbol, same reasoning as
    # MES/MNQ above.
    # CL: confirmed empirically (research doc §2.2) to have NO restriction
    # at all -- all 12 CME month codes win the daily volume race in
    # comparable numbers, unlike every other cluster surveyed. Unlike
    # every other confirmed active_months entry, this doesn't narrow the
    # live splice's candidate search (splice_live_price) -- it's set to
    # the full 12-month list specifically so resolve_active_months
    # doesn't return None (which the splice reads as "unconfirmed, skip
    # this symbol entirely," the right call for a genuinely-unconfirmed
    # case like BRE but wrong here, where "no restriction" IS the
    # confirmed finding); tsmom_rebalance._splice_live_front_month_bar
    # detects the full-12-month case and skips filtering its (REAL,
    # IB-confirmed, not guessed) candidate listing by month letter
    # entirely, taking the 3 nearest live contracts regardless of which
    # ones they are -- see that function's own docstring for why WTI
    # specifically needs 3 candidates, not the usual fewer: there's no
    # sparse, well-separated cycle to anchor "the next listed month" as
    # the presumed volume leader, and WTI's real front-month convention
    # can roll multiple months out ahead of nominal expiration.
    'CL':  {'exchange': 'NYMEX', 'multiplier': 1000,       'cluster': 'energy',
            'initial_margin': 16602.40, 'commission': 2.36,
            'active_months': ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'],
            'annualization_days': 259},
    'MCL': {'exchange': 'NYMEX', 'multiplier': 100,        'cluster': 'energy',
            'initial_margin': 1660.24, 'commission': 0.76, 'signal_symbol': 'CL'},

    # ── Metals ──────────────────────────────────────────────────────────────
    # SIL: IBKR has no separate Micro Silver ticker -- it trades under the
    # same root 'SI' as full-size silver, disambiguated by multiplier.
    # resolve_price_symbol already falls through to ib_symbol for SIL, so no
    # separate db_symbol/signal_symbol is needed. No margin/commission for
    # SIL yet (not sourced).
    # active_months on GC/SI confirmed empirically -- research doc §2.2
    # (GC: Feb/Apr/Jun/Aug/Dec; SI: Mar/May/Jul/Sep/Dec). Neither follows the
    # financial quarterly cycle at all.
    'GC':  {'exchange': 'COMEX', 'multiplier': 100,        'cluster': 'metal',
            'initial_margin': 40701.95, 'commission': 2.24, 'active_months': ['G', 'J', 'M', 'Q', 'Z'],
            'annualization_days': 259},
    'MGC': {'exchange': 'COMEX', 'multiplier': 10,         'cluster': 'metal',
            'initial_margin': 4084.79, 'commission': 0.96, 'signal_symbol': 'GC'},
    'SI':  {'exchange': 'COMEX', 'multiplier': 5000,       'cluster': 'metal',
            'initial_margin': 74299.37, 'commission': 1.70, 'active_months': ['H', 'K', 'N', 'U', 'Z'],
            'annualization_days': 259},
    'SIL': {'exchange': 'COMEX', 'multiplier': 1000,       'cluster': 'metal',
             'initial_margin': 13038.92, 'commission': 1.36, 'ib_symbol': 'SI'},

    # ── Rates ───────────────────────────────────────────────────────────────
    # MTN borrows ZN's via signal_symbol (the standard 10-Year, not TN the
    # Ultra 10-Year -- a different duration/contract, not MTN's full-size
    # sibling), same reasoning as MES/MNQ above.
    # active_months on ZN/ZT confirmed empirically as the standard financial
    # quarterly cycle -- research doc §2.2. TN (Ultra 10-Year) not
    # separately queried -- left unset (unconfirmed, not "no restriction").
    'ZN':  {'exchange': 'CBOT',  'multiplier': 1000,       'cluster': 'rates', # notional ~= 109K
            'initial_margin': 2156.25, 'commission': 1.66, 'active_months': ['H', 'M', 'U', 'Z'],
            'annualization_days': 259},
    'TN':  {'exchange': 'CBOT',  'multiplier': 1000,       'cluster': 'rates',
            'initial_margin': 2932.50, 'commission': 1.66},
    'MTN': {'exchange': 'CBOT',  'multiplier': 100,        'cluster': 'rates', # notional ~= 11K
            'initial_margin': 769.16, 'commission': 0.56, 'signal_symbol': 'ZN'},
    'ZT':  {'exchange': 'CBOT',  'multiplier': 2000,       'cluster': 'rates', # notional ~= 205K
            'initial_margin': 1380.00, 'commission': 1.51, 'active_months': ['H', 'M', 'U', 'Z'],
            'annualization_days': 259},

    # ── Grains ──────────────────────────────────────────────────────────────
    # CBOT micro grains (MZL/MZC/MZS/MZW) launched ~Feb 2025 -- too short a
    # history for the 252-day TSMOM lookback. signal_symbol borrows the
    # full-size contract's continuous IB bars; db_symbol falls back to
    # signal_symbol so the duckdb path uses the same full-size asset.
    # active_months confirmed empirically for all four grains -- research
    # doc §2.2 -- none follows the financial quarterly cycle; each has its
    # own standard CBOT ag listing calendar (wheat: Mar/May/Jul/Sep/Dec;
    # soybeans/soy oil: Jan/Mar/May/Jul/Nov, roughly). Corn nominally
    # shares wheat's Mar/May/Jul/Sep/Dec listing calendar too, but a full
    # re-derivation of the DB's own sticky-crossover series (research doc
    # §6.3, this project's own duckdb, full ~2010-2026 history) found
    # September has never once actually won corn's front-month crossover
    # -- genuinely liquid in its own right (91.7M total volume, same order
    # of magnitude as March/May/July), just never the single highest-
    # volume day -- unlike wheat, where September DOES win. ZC's own
    # active_months excludes it accordingly; ZW's below does not.
    # The multiplier below is USD P&L for a 1.00 move in the quoted price,
    # not the physical contract quantity. ZC/ZS/ZW are quoted in cents/bu:
    # 5,000 bu * $0.01 = $50 (full size), while the 500-bu Micro contracts
    # are $5. ZL is quoted in cents/lb: 60,000 lb * $0.01 = $600, while
    # Micro Soybean Oil's 6,000 lb contract is $60.
    'ZL':  {'exchange': 'CBOT',  'multiplier': 600,        'cluster': 'grain', # notional ~= 43K
            'initial_margin': 5102.79, 'commission': 3.01, 'active_months': ['F', 'H', 'K', 'N', 'Z'],
            'annualization_days': 252},
    'MZL': {'exchange': 'CBOT',  'multiplier': 60,         'cluster': 'grain',
            'initial_margin': 525.16, 'commission': 0.76, 'signal_symbol': 'ZL'},
    'ZC':  {'exchange': 'CBOT',  'multiplier': 50,         'cluster': 'grain',
            'initial_margin': 1855.76, 'commission': 3.01, 'active_months': ['H', 'K', 'N', 'Z'],
            'annualization_days': 252},
    'MZC': {'exchange': 'CBOT',  'multiplier': 5,          'cluster': 'grain',
            'initial_margin': 166.51, 'commission': 0.76, 'signal_symbol': 'ZC'},
    'ZS':  {'exchange': 'CBOT',  'multiplier': 50,         'cluster': 'grain',
            'initial_margin': 4038.46, 'commission': 3.01, 'active_months': ['F', 'H', 'K', 'N', 'X'],
            'annualization_days': 252},
    'MZS': {'exchange': 'CBOT',  'multiplier': 5,          'cluster': 'grain',
            'initial_margin': 382.95, 'commission':  0.76, 'signal_symbol': 'ZS'},
    'ZW':  {'exchange': 'CBOT',  'multiplier': 50,         'cluster': 'grain',
            'initial_margin': 3321.39, 'commission': 3.01, 'active_months': ['H', 'K', 'N', 'U', 'Z'],
            'annualization_days': 252},
    'MZW': {'exchange': 'CBOT',  'multiplier': 5,          'cluster': 'grain',
             'initial_margin': 332.14, 'commission': 0.76, 'signal_symbol': 'ZW'},

    # ── International equity ─────────────────────────────────────────────────
    # Nikkei: its own factor (Japan equity, JPY-adjacent), not lumped with
    # US equity. NKD/MNK (dollar-denominated) are a DIFFERENT contract from
    # BACKTEST_ONLY_SPECS' NIY (yen-denominated).
    'NKD': {'exchange': 'CME',   'multiplier': 5,          'cluster': 'intl_equity',
            'initial_margin': 43347.93, 'commission': 3.01},
    'MNK': {'exchange': 'CME',   'multiplier': 0.5,        'cluster': 'intl_equity',
             'initial_margin': 4334.76, 'commission': 0.69, 'signal_symbol': 'NKD'},

    # ── FX ──────────────────────────────────────────────────────────────────
    # IBKR tickers differ from Globex root symbols in the duckdb:
    #   JPY / J7  → Globex '6J'  (J7 is the mini; same underlying, ÷2 size)
    #   BRE       → Globex '6L'  (Brazilian Real)
    #   6M        → Globex '6M'  (Mexican Peso; name matches, no mapping)
    # signal_symbol on J7: borrows full-size JPY continuous IB history for
    # signal/covariance (same MZC→ZC pattern -- mini listed ~2019, too thin
    # for a reliable 252-day signal on its own).
    # active_months on JPY/6M confirmed empirically as the standard financial
    # quarterly cycle -- research doc §2.2 (checked via db_symbol '6J'/'6M').
    'JPY': {'exchange': 'CME',   'multiplier': 12_500_000, 'cluster': 'fx', 'db_symbol': '6J',
            'initial_margin': 3051.83, 'commission': 2.46, 'active_months': ['H', 'M', 'U', 'Z'],
            'annualization_days': 259},
    'J7':  {'exchange': 'CME',   'multiplier': 6_250_000,  'cluster': 'fx', 'db_symbol': '6J', # notional ~= 39K
            'initial_margin': 1525.92, 'commission': 1.36, 'signal_symbol': 'JPY'},
    # BRE: active_months deliberately NOT set. Checked (research doc §2.2)
    # and its own continuous series is the one confirmed exception to
    # "volume-ranking + stickiness works" (§1.2's still-open sticky-anchor
    # hijack bug -- only 71.5% of 6L's dates survive the sticky join, vs
    # 100% for every other symbol here) -- its raw per-day "which month
    # actually won" data can't be trusted the same way the rest of this
    # survey could, so no active_months figure is asserted for it.
    # Excluded from the default live/backtest universe for now (see
    # scripts' DEFAULT_SYMBOLS/DEFAULT_INSTRUMENTS) -- 6M (below) is used in
    # its place: correlated FX-EM exposure, cleaner data, no known bug.
    'BRE': {'exchange': 'CME',   'multiplier': 100_000,    'cluster': 'fx', 'db_symbol': '6L', # notional ~= 19K
            'initial_margin': 5023.68, 'commission': 2.46},
    '6M':  {'exchange': 'CME',   'multiplier': 500_000,    'cluster': 'fx', # notional ~= 28K
            'initial_margin': 1962.36, 'commission': 2.46, 'active_months': ['H', 'M', 'U', 'Z'],
            'annualization_days': 259},
}

# ── Backtest-only contracts ─────────────────────────────────────────────────
# Multiplier/margin/commission for contracts with NO live TSMOM
# counterpart -- never part of INSTRUMENTS, never traded/analyzed by the live
# system. Used only by ad-hoc single-symbol backtests (naked_futures.py,
# profile_bt.py) via get_spec().
#
# MUST NEVER be merged into INSTRUMENTS or unioned into any
# "default to every known instrument" fallback (e.g. run_tsmom_rebalance.py's
# `args.instruments or ','.join(sorted(KNOWN_INSTRUMENTS))`, mirrored in
# tsmom_risk_budget_diagnostic.py/futures_data_quality.py) -- doing so would
# silently expand what those live/diagnostic tools trade or analyze by
# default to symbols that were never vetted for the live TSMOM strategy.
#
# db data status (confirmed by direct query): NIY has real duckdb history
# (asset='NIY'), genuinely backtestable via naked_futures.py --symbol NIY.
# MYM/YM/M2K/RTY have none -- neither the micro nor its full-size sibling
# (YM/RTY) appears in this Databento feed at all, so these four entries are
# inert until/unless that data exists; no signal_symbol/db_symbol
# substitution can fix a family with zero data on either side.
BACKTEST_ONLY_SPECS: dict[str, dict] = {
    'MYM': {'multiplier': 0.5, 'initial_margin': 1727.20, 'commission': 1.24},  # Micro E-mini Dow -- margin estimated, verify; no db data
    'YM':  {'multiplier': 5,   'initial_margin': 12503.0, 'commission': 1.70},  # E-mini Dow -- margin estimated, verify; no db data
    'M2K': {'multiplier': 5,   'initial_margin': 1250.05, 'commission': 1.24},  # Micro E-mini Russell 2000 -- margin estimated, verify; no db data
    'RTY': {'multiplier': 50,  'initial_margin': 12593.0, 'commission': 1.70},  # E-mini Russell 2000 -- margin estimated, verify; no db data
    'NIY': {'multiplier': 500, 'initial_margin': 10000.0, 'commission': 1.70},  # Nikkei 225 Yen-denominated (CME) -- margin estimated, verify.
                                                                                 # NOTE: contract is JPY-denominated (Y500/point); this codebase's
                                                                                 # PnL math has no FX conversion, so PnL comes out in JPY, not USD,
                                                                                 # unless that's added separately. Has real duckdb history.
    # SOX = {...} # Not added: genuinely unsure of this contract's point
    # value/margin (possibly a Small Exchange product, not a standard CME
    # index future I have reliable specs for) -- ask before adding rather
    # than guess.
}


def resolve_price_symbol(symbol: str) -> str:
    """Symbol whose price history (duckdb `daily` table) should be used for
    `symbol`'s signal/backtest data -- either its own (default) or a
    fuller-history sibling's, via the same db_symbol > signal_symbol >
    ib_symbol > symbol fallback chain run_tsmom_rebalance.py's
    _build_instruments uses to compute db_symbol -- generalized to work
    from a bare traded symbol (not an already-built instrument dict) so the
    general Backtester (naked_futures.py) and tsmom_backtester.py's
    multi-symbol backtests can reuse it too, not just live rebalancing.

    Distinct from the instrument's own contract specs (multiplier/margin/
    commission), which always come from `symbol` itself via get_spec(symbol)
    -- e.g. MES borrows ES's price series but stays sized/margined as MES,
    never as ES.

    Returns `symbol` unchanged for anything not in INSTRUMENTS (including
    BACKTEST_ONLY_SPECS members, which have no db-coverage gap to resolve).
    """
    instr = INSTRUMENTS.get(symbol, {})
    return instr.get('db_symbol') or instr.get('signal_symbol') or instr.get('ib_symbol') or symbol


def resolve_signal_symbol(instr: dict) -> str:
    """Ticker whose continuous IB front-month history should back `instr`'s
    live TSMOM signal/covariance calculation -- either its own traded
    ticker (default) or a fuller-history sibling's, via signal_symbol >
    ib_symbol > symbol. Deliberately excludes db_symbol (unlike
    resolve_price_symbol above) -- db_symbol only resolves a pure IBKR/
    Globex ticker-naming divergence for the local duckdb path (J7->6J,
    BRE->6L) and isn't necessarily a valid/intended ticker for the live IB
    continuous-bars fetch this function backs.

    Takes an already-built instrument dict (as produced by
    run_tsmom_rebalance._build_instruments, or a hand-written JSON
    instrument config -- see _build_instruments' own docstring), not a
    bare traded symbol, since every call site already has one in hand.
    `instr['symbol']` is required; `signal_symbol`/`ib_symbol` are
    optional, same as the INSTRUMENTS entries themselves.

    Shared by derivatives_bt_engine.live.tsmom_rebalance (the live IB
    continuous-bars fetch) and scripts.tsmom_risk_budget_diagnostic (the
    same fetch, for covariance) -- previously duplicated in both places."""
    return instr.get('signal_symbol') or instr.get('ib_symbol') or instr['symbol']


# get_spec()'s Globex/db ticker -> INSTRUMENTS dict key, for the 3 FX
# symbols where they diverge (INSTRUMENTS keys by IBKR-facing ticker, not
# the raw Globex root -- see this module's docstring, db_symbol field).
_FX_TICKER_TO_KEY = {'6J': 'JPY', '6L': 'BRE', '6M': '6M'}


def get_spec(symbol: str) -> dict:
    """Full contract spec (multiplier/initial_margin/commission, plus
    exchange/cluster/etc. if present) for `symbol` -- the single lookup
    point for consumers that need a contract's multiplier/margin/
    commission (FuturesPosition, FuturesStrategyConfig,
    FuturesSignalGenerator, tsmom_backtester.py, naked_futures.py).
    `symbol` may be the real exchange/Globex ticker (e.g. '6J') for the 3
    FX contracts where that diverges from INSTRUMENTS' IBKR-facing key.

    Raises KeyError (with the full known-symbol list) if not found --
    callers wanting a friendlier error should validate against
    known_futures_symbols() first."""
    key = _FX_TICKER_TO_KEY.get(symbol.upper(), symbol.upper())
    info = INSTRUMENTS.get(key) or BACKTEST_ONLY_SPECS.get(key)
    if info is None:
        raise KeyError(f"Unknown futures symbol {symbol!r}. Known: {sorted(known_futures_symbols())}")
    return info


def known_futures_symbols() -> set[str]:
    """Every symbol get_spec() accepts -- INSTRUMENTS/BACKTEST_ONLY_SPECS
    keys plus the FX Globex tickers that map to a different INSTRUMENTS
    key (6J, 6L; 6M's Globex ticker matches its INSTRUMENTS key already)."""
    return set(INSTRUMENTS) | set(BACKTEST_ONLY_SPECS) | set(_FX_TICKER_TO_KEY)


# Standard CME month-letter codes, Jan-Dec. Shared by resolve_active_months()
# below and by any consumer translating a contract's `expiration` month back
# to the letter convention active_months is expressed in.
CME_MONTH_LETTERS: dict[str, int] = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12,
}
CME_MONTH_NUM_TO_LETTER: dict[int, str] = {v: k for k, v in CME_MONTH_LETTERS.items()}


def resolve_active_months(symbol: str) -> Optional[list[str]]:
    """Confirmed CME month-letter codes `symbol`'s underlying product
    actually trades actively (see the `active_months` field docstring
    above), resolved via the same db_symbol > signal_symbol > ib_symbol >
    symbol fallback chain as resolve_price_symbol -- so a micro/mini that
    borrows its full-size sibling's price series (MGC->GC, J7->JPY, ...)
    automatically inherits that sibling's confirmed list rather than
    needing (and risking drifting out of sync with) its own duplicate copy.

    Returns None when unset -- either "not yet empirically confirmed" or
    "confirmed to have no restriction" (see the specific INSTRUMENTS entry's
    own comment for which). Callers should treat None as "don't filter,"
    not as "assume quarterly" -- the same conservative default this
    project's live contract-resolution path already falls back to when
    no better information is available. Also returns None (rather than
    raising, unlike get_spec) for a symbol this registry doesn't know at
    all -- this function is an advisory guard, not a hard validation gate,
    and a placeholder/test-only symbol name (e.g. tsmom_backtester's own
    test suite uses 'X') should degrade to "no info available," the same
    as a real but not-yet-confirmed one, rather than crash a caller that
    never asked for strict validation."""
    try:
        return get_spec(resolve_price_symbol(symbol)).get('active_months')
    except KeyError:
        return None


def resolve_annualization_days(symbol: str) -> int:
    """Real trading-days/year for `symbol`'s own continuous series (see the
    `annualization_days` field docstring above), resolved via the same
    db_symbol > signal_symbol > ib_symbol > symbol fallback chain as
    resolve_price_symbol/resolve_active_months -- a micro/mini inherits its
    full-size sibling's confirmed figure automatically.

    Unlike resolve_active_months, always returns a concrete int, never
    None -- annualization needs a real number to multiply/sqrt by, so an
    unset or unresolvable symbol falls back to DEFAULT_ANNUALIZATION_DAYS
    (252, this project's pre-existing universal assumption and now the
    empirically-exact figure for the CBOT grains) rather than propagating
    an absence a caller would have to special-case."""
    try:
        info = get_spec(resolve_price_symbol(symbol))
    except KeyError:
        return DEFAULT_ANNUALIZATION_DAYS
    return info.get('annualization_days', DEFAULT_ANNUALIZATION_DAYS)
