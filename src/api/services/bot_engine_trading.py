from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtester.strategies.Asset_Preprocessing import get_pivot_price


def has_yesterday_positions(
    holdings_path: Path,
    trade_date: date,
    parse_ymd: Callable[[str], date],
) -> tuple[bool, list[str]]:
    if not holdings_path.exists():
        return False, []

    try:
        payload = json.loads(holdings_path.read_text(encoding="utf-8"))
        positions = payload.get("positions", [])
    except Exception:
        return False, []

    yesterday = trade_date - timedelta(days=1)
    alive: list[str] = []
    for pos in positions:
        code = str(pos.get("code", "")).strip()
        if not code:
            continue
        entry_raw = pos.get("entry_date")
        if not entry_raw:
            continue
        entry_date = parse_ymd(str(entry_raw))
        exit_raw = pos.get("exit_date")
        exit_date = parse_ymd(str(exit_raw)) if exit_raw else None
        if entry_date <= yesterday and (exit_date is None or exit_date > yesterday):
            alive.append(code)
    return len(alive) > 0, sorted(set(alive))


def build_trade_candidates(
    final_codes: list[str],
    signal_date: date,
    trade_date: date,
    price_data_dir: Path,
    price_source: str,
    account_equity: float,
    risk_per_trade: float,
    max_position_ratio: float,
    stop_atr_multiple: float,
    take_profit_r: float,
    pivot_window: int,
    load_price_slice: Callable[[str, date, Path, str], pd.DataFrame | None],
) -> list[dict[str, Any]]:
    signal_ts = pd.Timestamp(signal_date)
    trade_ts = pd.Timestamp(trade_date)
    plans: list[dict[str, Any]] = []

    for code in final_codes:
        df = load_price_slice(code, trade_date, price_data_dir, price_source)
        if df is None or signal_ts not in df.index:
            plans.append(
                {
                    "code": code,
                    "status": "no_price_data",
                    "signal_date": signal_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                }
            )
            continue

        pivot_price = get_pivot_price(df.loc[:signal_ts], window=pivot_window)
        if pivot_price is None:
            plans.append(
                {
                    "code": code,
                    "status": "no_pivot",
                    "signal_date": signal_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                }
            )
            continue

        if trade_ts not in df.index:
            plans.append(
                {
                    "code": code,
                    "status": "await_trade_day_open",
                    "signal_date": signal_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                    "pivot_price": float(pivot_price),
                }
            )
            continue

        day_row = df.loc[trade_ts]
        open_price = float(day_row.get("Open", day_row.get("Close", 0)))
        if open_price < pivot_price:
            plans.append(
                {
                    "code": code,
                    "status": "open_below_pivot",
                    "signal_date": signal_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                    "pivot_price": float(pivot_price),
                    "open_price": open_price,
                }
            )
            continue

        prev_days = df.index[df.index < trade_ts]
        if len(prev_days) == 0:
            plans.append(
                {
                    "code": code,
                    "status": "no_prev_day_for_atr",
                    "signal_date": signal_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                }
            )
            continue

        atr = float(df.loc[prev_days[-1], "ATR_14"])
        if pd.isna(atr) or atr <= 0:
            plans.append(
                {
                    "code": code,
                    "status": "invalid_atr",
                    "signal_date": signal_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                }
            )
            continue

        entry_price = open_price
        stop_price = entry_price - (atr * stop_atr_multiple)
        per_share_risk = entry_price - stop_price
        if per_share_risk <= 0:
            plans.append(
                {
                    "code": code,
                    "status": "invalid_risk",
                    "signal_date": signal_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                }
            )
            continue

        shares_by_risk = int((account_equity * risk_per_trade) // per_share_risk)
        shares_by_value = int((account_equity * max_position_ratio) // entry_price)
        shares = min(shares_by_risk, shares_by_value)
        if shares <= 0:
            plans.append(
                {
                    "code": code,
                    "status": "position_size_zero",
                    "signal_date": signal_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                    "pivot_price": float(pivot_price),
                    "open_price": open_price,
                }
            )
            continue

        take_profit_price = entry_price + (per_share_risk * take_profit_r)
        plans.append(
            {
                "code": code,
                "status": "ready",
                "signal_date": signal_date.isoformat(),
                "trade_date": trade_date.isoformat(),
                "pivot_price": float(pivot_price),
                "open_price": open_price,
                "entry_price": entry_price,
                "stop_price": float(stop_price),
                "take_profit_price": float(take_profit_price),
                "shares": int(shares),
            }
        )

    return plans
