import pandas as pd
from pathlib import Path

# =========================================================
# [CONFIG] 백테스팅 하이퍼파라미터 설정
# =========================================================
CONFIG = {
    "INITIAL_CASH": 100000000,
    "RISK_PER_TRADE": 0.01,
    "TOTAL_RISK_CAP": 0.15,
    "SIGNAL_WINDOW_MODE": "next_day",  # "next_day" or "until_breakout"
    "RANKING_VALUE_LOOKBACK": 20,
    "USE_REGIME_FILTER": False,
    "REGIME_TICKERS": ["KS11", "KQ11", "252650", "277650", "226980"],
    "REGIME_REQUIREMENT": "any",
    "REGIME_SELECTED_TICKERS": ["226980"],
    "REGIME_SELECTED_REQUIREMENT": "any",
    "REGIME_OUTPUT_SUFFIX": "REGIME",
    "COMPARE_REGIMES": False,
    "COST_RATE": 0.003,
    "START_DATE": "2016-01-01",
    "END_DATE": "2025-12-31",
    "PRELOAD_USED_ONLY": True,
    "REGIME_SELECTED_TICKER": "226980",
    "STOP_ATR_MULTIPLE": 2.5,
    "TAKE_PROFIT_R": 2.0,
    "CHANDELIER_ATR_MULTIPLE": 2.5,
    "ATR_FLOOR_RATIO": 0.01,
    "ATR_COLUMN": "ATR_14",
    "MAX_POSITION_RATIO": 0.15,
    "ENTRY_VALUE_LOOKBACK": 25,
    "ENTRY_VALUE_MULTIPLE": 1.2,
}

# ---------------------------------------------------------
# 데이터 경로 설정 (사용자 원본 경로 유지)
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
UNIVERSE_PATH = BASE_DIR / "data" / "universe" / "intersection_80_minervini4.csv"
PARQUET_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "src" / "backtester" / "strategies"


def load_from_parquet(code, data_dir, columns=None, compute_indicators=True):
    file_path = f"{data_dir}{code}.parquet"
    try:
        df = pd.read_parquet(file_path, columns=columns)
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index)

        if compute_indicators:
            if "MA50" not in df.columns and "Close" in df.columns:
                df["MA50"] = df["Close"].rolling(window=50).mean()

            if "MA200" not in df.columns and "Close" in df.columns:
                df["MA200"] = df["Close"].rolling(window=200).mean()

            if "ATR_14" not in df.columns:
                if all(col in df.columns for col in ["High", "Low", "Close"]):
                    tr = pd.concat(
                        [
                            df["High"] - df["Low"],
                            (df["High"] - df["Close"].shift(1)).abs(),
                            (df["Low"] - df["Close"].shift(1)).abs(),
                        ],
                        axis=1,
                    ).max(axis=1)
                    df["ATR_14"] = tr.rolling(window=14).mean()

        return df
    except Exception:
        return None


def preload_price_cache(universe_df, columns=None):
    cache = {}
    unique_codes = universe_df["Code"].unique()
    data_dir_str = str(PARQUET_DIR) + "/"

    for code in unique_codes:
        df = load_from_parquet(code, data_dir_str, columns=columns)
        if df is not None:
            cache[code] = df
    return cache


def load_regime_cache(tickers=None):
    cache = {}
    data_dir_str = str(PARQUET_DIR) + "\\"
    symbols = tickers if tickers is not None else CONFIG.get("REGIME_TICKERS", [])
    for symbol in symbols:
        try:
            df = load_from_parquet(
                symbol,
                data_dir_str,
                columns=["Date", "Close", "MA200"],
                compute_indicators=False,
            )
            if df is not None:
                cache[symbol] = df
        except Exception:
            continue
    return cache


def is_regime_on(regime_cache, date, requirement=None):
    if not regime_cache:
        return False

    checks = []
    for _, df in regime_cache.items():
        df_slice = df.loc[:date]
        if df_slice.empty:
            checks.append(False)
            continue
        ma200 = df_slice["MA200"]
        if pd.isna(ma200.iloc[-1]):
            checks.append(False)
            continue
        checks.append(df_slice["Close"].iloc[-1] > ma200.iloc[-1])

    requirement = (
        str(requirement).lower()
        if requirement is not None
        else str(CONFIG.get("REGIME_REQUIREMENT", "any")).lower()
    )
    if requirement == "all":
        return all(checks)
    return any(checks)


def get_pivot_price(df_slice, window=10):
    # 신호 발생일(today_ts)까지 포함해서 10일 고가 계산
    if len(df_slice) < window:
        return None
    high_col = "High" if "High" in df_slice.columns else "Close"
    pivot = df_slice[high_col].iloc[-window:].max()
    if pd.isna(pivot) or pivot <= 0:
        return None
    return float(pivot)


def run_backtest(
    output_suffix=None,
    save_outputs=True,
    silent=False,
    price_cache=None,
    regime_cache=None,
    use_regime_filter=None,
    regime_requirement=None,
    regime_tickers=None,
    early_mdd_pct=None,
    early_mdd_check_ratio=None,
    early_recovery_progress=None,
    early_recovery_cagr=None,
):
    # --- 1. 데이터 로드 및 전처리 ---
    if not silent:
        print("- 유니버스 데이터를 읽는 중...")
    if not UNIVERSE_PATH.exists():
        print(f"오류: 유니버스 파일을 찾을 수 없습니다. ({UNIVERSE_PATH})")
        return

    universe = pd.read_csv(UNIVERSE_PATH, dtype={"Code": str})
    universe["Date"] = pd.to_datetime(universe["Date"])
    universe = universe[
        (universe["Date"] >= CONFIG["START_DATE"])
        & (universe["Date"] <= CONFIG["END_DATE"])
    ]
    all_dates = sorted(universe["Date"].unique())

    if price_cache is None:
        if CONFIG.get("PRELOAD_USED_ONLY"):
            price_cache = preload_price_cache(
                universe,
                columns=[
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    CONFIG["ATR_COLUMN"],
                    "MA50",
                ],
            )
        else:
            price_cache = {}

    # --- 2. 시뮬레이션 변수 초기화 ---
    cash = CONFIG["INITIAL_CASH"]
    portfolio = {}  # {티커: 정보}
    history = []  # 일별 기록
    # price_cache는 preload_data_to_ram에서 초기화
    trade_log = []  # 매매 기록
    sell_reasons = {}  # 매도 사유 통계
    pending_sells = {}  # {exit_date: [주문정보]}
    watchlist = {}  # {ticker: {"start_date": Timestamp, "universe_date": Timestamp}}

    if use_regime_filter is None:
        use_regime_filter = CONFIG.get("USE_REGIME_FILTER")
    if use_regime_filter and regime_cache is None:
        regime_cache = load_regime_cache(regime_tickers)

    if not silent:
        print(f"- 백테스팅 시작: {CONFIG['START_DATE']} ~ {CONFIG['END_DATE']}")
        if use_regime_filter:
            print("- 레짐 필터 적용: MA200 기준")

    total_days = len(all_dates)
    total_span_days = (
        pd.to_datetime(CONFIG["END_DATE"]) - pd.to_datetime(CONFIG["START_DATE"])
    ).days
    total_years = total_span_days / 365 if total_span_days > 0 else 0.0
    early_check_index = None
    early_check_done = False
    if early_mdd_pct is not None and early_mdd_check_ratio:
        early_check_index = max(0, int(total_days * early_mdd_check_ratio) - 1)

    def _return_early_cut(current_equity, today_ts):
        result_df = pd.DataFrame(history)
        trade_log_df = pd.DataFrame(trade_log)
        if result_df.empty:
            final_val = int(current_equity)
        else:
            final_val = int(result_df["자산총액"].iloc[-1])
        total_ret = (final_val / CONFIG["INITIAL_CASH"] - 1) * 100
        days = (pd.to_datetime(today_ts) - pd.to_datetime(CONFIG["START_DATE"])).days
        cagr = ((final_val / CONFIG["INITIAL_CASH"]) ** (365 / max(days, 1)) - 1) * 100

        result_df["Peak"] = result_df["자산총액"].cummax()
        result_df["DD"] = (result_df["자산총액"] - result_df["Peak"]) / result_df[
            "Peak"
        ]
        mdd = result_df["DD"].min() * 100

        win_rate = 0.0
        profit_factor = float("inf")
        if not trade_log_df.empty:
            win_trades = trade_log_df[trade_log_df["Profit"] > 0]
            loss_trades = trade_log_df[trade_log_df["Profit"] <= 0]
            win_rate = (len(win_trades) / len(trade_log_df)) * 100
            avg_prof = win_trades["Profit"].mean() * 100 if not win_trades.empty else 0
            avg_loss = (
                loss_trades["Profit"].mean() * 100 if not loss_trades.empty else 0
            )
            profit_factor = abs(avg_prof / avg_loss) if avg_loss != 0 else float("inf")

        metrics = {
            "FinalAsset": int(final_val),
            "TotalReturnPct": round(total_ret, 2),
            "CAGR": round(cagr, 2),
            "MDD": round(mdd, 2),
            "WinRate": round(win_rate, 2),
            "ProfitFactor": (
                round(profit_factor, 2)
                if profit_factor != float("inf")
                else profit_factor
            ),
            "Trades": int(len(trade_log_df)),
            "EarlyCut": True,
        }

        return result_df, trade_log_df, metrics

    peak_equity = float(cash)
    min_dd = 0.0

    for day_idx, today in enumerate(all_dates):
        today_ts = pd.Timestamp(today)

        # -----------------------------------------------------
        # PHASE 0: [청산] 예약 매도 (Chandelier 종가 하락)
        # -----------------------------------------------------
        if today_ts in pending_sells:
            for order in pending_sells.pop(today_ts):
                ticker = order["ticker"]
                if ticker not in portfolio:
                    continue
                if ticker not in price_cache:
                    price_cache[ticker] = load_from_parquet(
                        ticker,
                        str(PARQUET_DIR) + "\\",
                        columns=[
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            CONFIG["ATR_COLUMN"],
                            "MA50",
                        ],
                    )
                df = price_cache[ticker]
                if df is None or today_ts not in df.index:
                    continue

                day_row = df.loc[today_ts]
                exit_price = (
                    float(day_row["Open"])
                    if "Open" in day_row
                    else float(day_row["Close"])
                )
                if exit_price <= 0:
                    continue

                equity_before_exit = float(cash)
                for t, p_info in portfolio.items():
                    if t == ticker:
                        mark_price = exit_price
                    elif t in price_cache and today_ts in price_cache[t].index:
                        mark_price = price_cache[t].loc[today_ts, "Close"]
                    else:
                        mark_price = p_info.get("last_price", p_info["entry_price"])
                    equity_before_exit += p_info["shares"] * float(mark_price)

                pnl_amount = (
                    exit_price - portfolio[ticker]["entry_price"]
                ) * portfolio[ticker]["shares"]
                pnl_pct = (
                    (pnl_amount / equity_before_exit) * 100
                    if equity_before_exit > 0
                    else 0.0
                )
                realized_ret = (
                    exit_price - portfolio[ticker]["entry_price"]
                ) / portfolio[ticker]["entry_price"]
                trade_log.append(
                    {
                        "Ticker": ticker,
                        "Profit": round(realized_ret, 4),
                        "EquityPct": round(pnl_pct, 4),
                        "Reason": order.get("reason", "Chandelier_Exit"),
                        "Entry_Date": portfolio[ticker]["entry_date"],
                        "Exit_Date": today_ts.date(),
                    }
                )
                sell_reasons[order.get("reason", "Chandelier_Exit")] = (
                    sell_reasons.get(order.get("reason", "Chandelier_Exit"), 0) + 1
                )
                cash += (
                    portfolio[ticker]["shares"] * exit_price * (1 - CONFIG["COST_RATE"])
                )
                del portfolio[ticker]

        # -----------------------------------------------------
        # PHASE 1: [청산] 매도 조건 체크 (상장폐지 대응 포함)
        # -----------------------------------------------------
        for ticker in list(portfolio.keys()):
            if ticker not in price_cache:
                price_cache[ticker] = load_from_parquet(
                    ticker,
                    str(PARQUET_DIR) + "\\",
                    columns=["Open", "High", "Low", "Close", CONFIG["ATR_COLUMN"]],
                )

            df = price_cache[ticker]
            if df is None or len(df) == 0:
                continue

            sell_reason = None
            exit_price = 0.0

            # [A] 상장폐지 및 데이터 종료 체크
            if today_ts >= df.index[-1]:
                exit_price = float(df.loc[df.index[-1], "Close"])
                sell_reason = "Delisting_Force_Exit"

            # [B] 일반 매도 조건 체크 (데이터가 있는 경우)
            elif today_ts in df.index:
                curr_data = df.loc[today_ts]
                curr_price = float(curr_data["Close"])
                curr_open = (
                    float(curr_data["Open"]) if "Open" in curr_data else curr_price
                )
                curr_low = float(curr_data["Low"]) if "Low" in curr_data else curr_price
                curr_high = (
                    float(curr_data["High"]) if "High" in curr_data else curr_price
                )
                info = portfolio[ticker]

                if info.get("entry_date") == today_ts:
                    continue

                # 정보 업데이트
                info["days_held"] += 1

                highest_high = info.get("highest_high", info["entry_price"])
                if curr_high > highest_high:
                    highest_high = curr_high
                info["highest_high"] = highest_high

                entry_p = info["entry_price"]
                stop_price = info["stop_price"]
                r_per_share = info["r_per_share"]
                take_profit_price = entry_p + (r_per_share * CONFIG["TAKE_PROFIT_R"])
                curr_ret = (curr_price - entry_p) / entry_p

                hit_stop = curr_low <= stop_price
                hit_tp = curr_high >= take_profit_price

                # 조건 판별 (봉 내 체결 가정: 저가/고가에 닿으면 해당 가격 체결)
                if pd.isna(curr_open) or curr_open <= 0:
                    sell_reason = "Delisting_ZeroPrice"
                    exit_price = info.get("last_price", curr_price)
                elif curr_open < stop_price:
                    sell_reason = "Gap_Down"
                    exit_price = curr_open
                elif hit_stop:
                    sell_reason = "Stop_Loss"
                    exit_price = stop_price
                elif hit_tp and not info.get("partial_exit_done"):
                    sell_shares = info["shares"] // 2
                    if sell_shares > 0:
                        equity_before_exit = float(cash)
                        for t, p_info in portfolio.items():
                            if t == ticker:
                                mark_price = take_profit_price
                            elif t in price_cache and today_ts in price_cache[t].index:
                                mark_price = price_cache[t].loc[today_ts, "Close"]
                            else:
                                mark_price = p_info.get(
                                    "last_price", p_info["entry_price"]
                                )
                            equity_before_exit += p_info["shares"] * float(mark_price)

                        pnl_amount = (
                            take_profit_price - info["entry_price"]
                        ) * sell_shares
                        pnl_pct = (
                            (pnl_amount / equity_before_exit) * 100
                            if equity_before_exit > 0
                            else 0.0
                        )
                        realized_ret = (take_profit_price - info["entry_price"]) / info[
                            "entry_price"
                        ]
                        trade_log.append(
                            {
                                "Ticker": ticker,
                                "Profit": round(realized_ret, 4),
                                "EquityPct": round(pnl_pct, 4),
                                "Reason": f"Take_Profit_{CONFIG['TAKE_PROFIT_R']}R_Half",
                                "Entry_Date": info["entry_date"],
                                "Exit_Date": today_ts.date(),
                            }
                        )
                        sell_reasons[f"Take_Profit_{CONFIG['TAKE_PROFIT_R']}R_Half"] = (
                            sell_reasons.get(
                                f"Take_Profit_{CONFIG['TAKE_PROFIT_R']}R_Half", 0
                            )
                            + 1
                        )
                        cash += (
                            sell_shares * take_profit_price * (1 - CONFIG["COST_RATE"])
                        )
                        info["shares"] -= sell_shares
                        info["partial_exit_done"] = True
                        info["last_price"] = curr_price

                if not sell_reason and info.get("shares", 0) > 0:
                    # Chandelier Exit는 partial_exit_done(부분익절) 이후에만 작동
                    if info.get("partial_exit_done"):
                        atr_value = (
                            float(curr_data.get(CONFIG["ATR_COLUMN"]))
                            if CONFIG["ATR_COLUMN"] in curr_data
                            else None
                        )
                        if (
                            atr_value is not None
                            and not pd.isna(atr_value)
                            and atr_value > 0
                        ):
                            atr_floor = curr_price * CONFIG["ATR_FLOOR_RATIO"]
                            atr_effective = max(atr_value, atr_floor)
                            chandelier_stop = info["highest_high"] - (
                                atr_effective * CONFIG["CHANDELIER_ATR_MULTIPLE"]
                            )
                            if curr_price < chandelier_stop and not info.get(
                                "pending_chandelier_exit"
                            ):
                                next_days = df.index[df.index > today_ts]
                                if len(next_days) > 0:
                                    exit_ts = next_days[0]
                                    pending_for_day = pending_sells.get(exit_ts, [])
                                    pending_for_day.append(
                                        {
                                            "ticker": ticker,
                                            "reason": "Chandelier_Exit",
                                        }
                                    )
                                    pending_sells[exit_ts] = pending_for_day
                                    info["pending_chandelier_exit"] = True

            # 매도 확정 시 처리
            if sell_reason:
                equity_before_exit = float(cash)
                for t, p_info in portfolio.items():
                    if t == ticker:
                        mark_price = exit_price
                    elif t in price_cache and today_ts in price_cache[t].index:
                        mark_price = price_cache[t].loc[today_ts, "Close"]
                    else:
                        mark_price = p_info.get("last_price", p_info["entry_price"])
                    equity_before_exit += p_info["shares"] * float(mark_price)

                pnl_amount = (
                    exit_price - portfolio[ticker]["entry_price"]
                ) * portfolio[ticker]["shares"]
                pnl_pct = (
                    (pnl_amount / equity_before_exit) * 100
                    if equity_before_exit > 0
                    else 0.0
                )
                realized_ret = (
                    exit_price - portfolio[ticker]["entry_price"]
                ) / portfolio[ticker]["entry_price"]
                trade_log.append(
                    {
                        "Ticker": ticker,
                        "Profit": round(realized_ret, 4),
                        "EquityPct": round(pnl_pct, 4),
                        "Reason": sell_reason,
                        "Entry_Date": portfolio[ticker]["entry_date"],
                        "Exit_Date": today_ts.date(),
                    }
                )
                sell_reasons[sell_reason] = sell_reasons.get(sell_reason, 0) + 1
                cash += (
                    portfolio[ticker]["shares"] * exit_price * (1 - CONFIG["COST_RATE"])
                )
                del portfolio[ticker]

        # -----------------------------------------------------
        # PHASE 2: [진입] 시그널 추적 및 당일 체결
        # -----------------------------------------------------
        if watchlist:
            equity_before_entry = float(cash)
            for t, p_info in portfolio.items():
                equity_before_entry += p_info["shares"] * p_info.get(
                    "last_price", p_info["entry_price"]
                )

            current_risk_used = sum(
                p["r_per_share"] * p["shares"]
                for p in portfolio.values()
                if "r_per_share" in p and "shares" in p
            )
            remaining_risk = (
                equity_before_entry * CONFIG["TOTAL_RISK_CAP"]
            ) - current_risk_used

            signal_mode = CONFIG.get("SIGNAL_WINDOW_MODE", "next_day")

            def _roll_or_drop(ticker):
                if signal_mode == "until_breakout":
                    watchlist[ticker]["universe_date"] = today_ts
                else:
                    if ticker in watchlist:
                        del watchlist[ticker]

            ordered_watchlist = []
            for ticker, watch in watchlist.items():
                signal_date = watch.get("universe_date")
                if signal_date is None:
                    continue
                rs_rank = None
                rs_slice = universe[universe["Date"] == signal_date]
                if not rs_slice.empty and "RS" in rs_slice.columns:
                    rs_rank = rs_slice.set_index("Code")["RS"].to_dict().get(ticker)

                avg_value = 0.0
                if ticker not in price_cache:
                    price_cache[ticker] = load_from_parquet(
                        ticker,
                        str(PARQUET_DIR) + "\\",
                        columns=["Close", "Volume"],
                    )
                df_rank = price_cache.get(ticker)
                if df_rank is not None and signal_date in df_rank.index:
                    df_rank_slice = df_rank.loc[:signal_date]
                    value_series = df_rank_slice["Close"] * df_rank_slice["Volume"]
                    rank_lookback = int(CONFIG.get("RANKING_VALUE_LOOKBACK", 20))
                    if len(value_series) >= rank_lookback:
                        avg_value = float(value_series.tail(rank_lookback).mean())
                ordered_watchlist.append((ticker, watch, rs_rank, avg_value))

            ordered_watchlist.sort(
                key=lambda item: (
                    item[2] if item[2] is not None else float("-inf"),
                    item[3],
                ),
                reverse=True,
            )

            for ticker, watch, _, _ in ordered_watchlist:
                if remaining_risk <= 0:
                    break

                if ticker in portfolio:
                    del watchlist[ticker]
                    continue

                signal_date = watch.get("universe_date")
                if signal_date is None:
                    del watchlist[ticker]
                    continue

                if today_ts <= signal_date:
                    continue

                if ticker not in price_cache:
                    price_cache[ticker] = load_from_parquet(
                        ticker,
                        str(PARQUET_DIR) + "\\",
                        columns=[
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "Volume",
                            CONFIG["ATR_COLUMN"],
                        ],
                    )
                df = price_cache[ticker]
                if df is None:
                    _roll_or_drop(ticker)
                    continue
                if today_ts not in df.index:
                    if signal_mode == "next_day":
                        _roll_or_drop(ticker)
                    continue

                if signal_date not in df.index:
                    _roll_or_drop(ticker)
                    continue

                next_days = df.index[df.index > signal_date]
                if len(next_days) == 0:
                    _roll_or_drop(ticker)
                    continue
                first_entry_day = next_days[0]

                if today_ts < first_entry_day:
                    continue
                if today_ts > first_entry_day:
                    _roll_or_drop(ticker)
                    continue

                signal_row = df.loc[signal_date]
                if "Volume" not in signal_row or pd.isna(signal_row["Volume"]):
                    _roll_or_drop(ticker)
                    continue

                entry_value = float(signal_row["Close"]) * float(signal_row["Volume"])
                if entry_value <= 0:
                    _roll_or_drop(ticker)
                    continue

                value_lookback = int(CONFIG["ENTRY_VALUE_LOOKBACK"])
                value_multiple = float(CONFIG["ENTRY_VALUE_MULTIPLE"])
                prev_value_series = (
                    df.loc[:signal_date, "Close"] * df.loc[:signal_date, "Volume"]
                    if "Volume" in df.columns
                    else None
                )
                if prev_value_series is None or len(prev_value_series) < value_lookback:
                    _roll_or_drop(ticker)
                    continue

                prev_avg_value = float(prev_value_series.tail(value_lookback).mean())
                if prev_avg_value <= 0 or entry_value < (
                    prev_avg_value * value_multiple
                ):
                    _roll_or_drop(ticker)
                    continue

                pivot_price = get_pivot_price(df.loc[:signal_date], window=10)
                if pivot_price is None:
                    _roll_or_drop(ticker)
                    continue

                day_row = df.loc[today_ts]
                curr_high = (
                    float(day_row["High"])
                    if "High" in day_row
                    else float(day_row["Close"])
                )
                curr_open = (
                    float(day_row["Open"])
                    if "Open" in day_row
                    else float(day_row["Close"])
                )
                if curr_high < pivot_price:
                    _roll_or_drop(ticker)
                    continue

                buy_price = curr_open if curr_open > pivot_price else pivot_price
                buy_close = float(day_row["Close"])

                prev_days = df.index[df.index < today_ts]
                if len(prev_days) == 0:
                    _roll_or_drop(ticker)
                    continue
                prev_ts = prev_days[-1]
                atr_raw = (
                    float(df.loc[prev_ts].get(CONFIG["ATR_COLUMN"]))
                    if CONFIG["ATR_COLUMN"] in df.columns
                    else None
                )
                if pd.isna(atr_raw) or atr_raw is None or atr_raw <= 0:
                    _roll_or_drop(ticker)
                    continue

                atr_floor = buy_price * CONFIG["ATR_FLOOR_RATIO"]
                atr_effective = max(atr_raw, atr_floor)
                stop_price = buy_price - (atr_effective * CONFIG["STOP_ATR_MULTIPLE"])
                per_share_risk = buy_price - stop_price
                if stop_price <= 0 or per_share_risk <= 0:
                    _roll_or_drop(ticker)
                    continue

                risk_amount = min(
                    equity_before_entry * CONFIG["RISK_PER_TRADE"], remaining_risk
                )
                target_shares = int(risk_amount // per_share_risk)
                if target_shares <= 0:
                    _roll_or_drop(ticker)
                    continue

                max_value_shares = int(
                    (equity_before_entry * CONFIG["MAX_POSITION_RATIO"]) // buy_price
                )
                if max_value_shares <= 0:
                    _roll_or_drop(ticker)
                    continue

                target_shares = min(target_shares, max_value_shares)
                if target_shares <= 0:
                    _roll_or_drop(ticker)
                    continue

                required_cash = target_shares * buy_price
                if cash < required_cash:
                    _roll_or_drop(ticker)
                    continue

                portfolio[ticker] = {
                    "shares": target_shares,
                    "entry_price": buy_price,
                    "entry_date": today_ts,
                    "stop_price": stop_price,
                    "r_per_share": per_share_risk,
                    "pivot_price": pivot_price,
                    "partial_exit_done": False,
                    "pending_chandelier_exit": False,
                    "days_held": 0,
                    "highest_high": curr_high,
                    "last_price": buy_close,
                }
                if buy_close < stop_price:
                    next_days = df.index[df.index > today_ts]
                    if len(next_days) > 0:
                        exit_ts = next_days[0]
                        pending_for_day = pending_sells.get(exit_ts, [])
                        pending_for_day.append(
                            {
                                "ticker": ticker,
                                "reason": "Stop_Close_Below",
                            }
                        )
                        pending_sells[exit_ts] = pending_for_day
                cash -= target_shares * buy_price
                remaining_risk -= target_shares * per_share_risk
                del watchlist[ticker]

        # -----------------------------------------------------
        # PHASE 3: [자산] 자산 가치 계산
        # -----------------------------------------------------
        current_equity = float(cash)
        for t, p_info in portfolio.items():
            # 자산 총액 계산 시 결측치 보정 로직 통합
            if t not in price_cache:
                price_cache[t] = load_from_parquet(
                    t,
                    str(PARQUET_DIR) + "\\",
                    columns=["Open", "High", "Low", "Close", CONFIG["ATR_COLUMN"]],
                )
            if price_cache[t] is not None and today_ts in price_cache[t].index:
                curr_close = price_cache[t].loc[today_ts, "Close"]
                p_info["last_price"] = curr_close
            else:
                curr_close = p_info.get("last_price", p_info["entry_price"])

            current_equity += p_info["shares"] * curr_close

        if current_equity > peak_equity:
            peak_equity = current_equity
        if peak_equity > 0:
            current_dd = (current_equity - peak_equity) / peak_equity
            if current_dd < min_dd:
                min_dd = current_dd

        # -----------------------------------------------------
        # PHASE 4: [진입 후보] 시그널 추적 대상 등록
        # -----------------------------------------------------
        regime_ok = True
        if use_regime_filter:
            regime_ok = is_regime_on(regime_cache, today_ts, regime_requirement)

        if regime_ok:
            candidates = universe[universe["Date"] == today].sort_values(
                "RS", ascending=False
            )
            for _, row in candidates.iterrows():
                ticker = row["Code"]
                if ticker in portfolio or ticker in watchlist:
                    continue

                if ticker not in price_cache:
                    price_cache[ticker] = load_from_parquet(
                        ticker,
                        str(PARQUET_DIR) + "\\",
                        columns=[
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "Volume",
                            CONFIG["ATR_COLUMN"],
                        ],
                    )
                df = price_cache[ticker]
                if df is None or today_ts not in df.index:
                    continue

                df_slice = df.loc[:today_ts]
                next_days = df.index[df.index > today_ts]
                if len(next_days) == 0:
                    continue
                entry_ts = next_days[0]
                watchlist[ticker] = {
                    "start_date": entry_ts,
                    "universe_date": today_ts,
                }

        # -----------------------------------------------------
        # PHASE 5: [기록] 일별 자산 상세 기록
        # -----------------------------------------------------
        asset_details = []
        for t, p_info in portfolio.items():
            curr_close = p_info["last_price"]
            val = p_info["shares"] * curr_close
            stock_ret = (curr_close - p_info["entry_price"]) / p_info["entry_price"]
            asset_details.append((t, val, stock_ret))

        cash_ratio = (cash / current_equity * 100) if current_equity > 0 else 0.0
        risk_amount = current_equity * CONFIG["RISK_PER_TRADE"]
        record = {
            "DATE": today_ts.date(),
            "자산총액": int(current_equity),
            "현금총액": int(cash),
            "현금비율(%)": round(cash_ratio, 2),
            "1회리스크금액": int(risk_amount),
        }
        for i, (ticker, total_val, s_ret) in enumerate(asset_details):
            record[f"자산{i+1}_티커"] = ticker
            record[f"자산{i+1}_수익률"] = round(s_ret * 100, 2)
            weight_pct = (
                (total_val / current_equity * 100) if current_equity > 0 else 0.0
            )
            record[f"자산{i+1}_비중(%)"] = round(weight_pct, 2)
        history.append(record)

        if (
            early_check_index is not None
            and not early_check_done
            and day_idx >= early_check_index
        ):
            early_check_done = True
            if (min_dd * 100) < float(early_mdd_pct):
                return _return_early_cut(current_equity, today_ts)

        if (
            early_recovery_progress is not None
            and early_recovery_cagr is not None
            and total_years > 0
        ):
            progress = day_idx / max(total_days - 1, 1)
            if progress >= float(early_recovery_progress):
                remaining_years = total_years * (1 - progress)
                if remaining_years > 0 and current_equity > 0:
                    required_cagr = (CONFIG["INITIAL_CASH"] / current_equity) ** (
                        1 / remaining_years
                    ) - 1
                    if required_cagr > float(early_recovery_cagr):
                        return _return_early_cut(current_equity, today_ts)

    # -----------------------------------------------------
    # 4. 결과 분석 및 파일 저장
    # -----------------------------------------------------
    result_df = pd.DataFrame(history)
    trade_log_df = pd.DataFrame(trade_log)

    if save_outputs:
        suffix = f"_{output_suffix}" if output_suffix else ""
        result_df.to_csv(
            OUTPUT_PATH / f"Asset_List_Final{suffix}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        trade_log_df.to_csv(
            OUTPUT_PATH / f"Trade_Log{suffix}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    final_val = result_df["자산총액"].iloc[-1]
    total_ret = (final_val / CONFIG["INITIAL_CASH"] - 1) * 100
    days = (
        pd.to_datetime(CONFIG["END_DATE"]) - pd.to_datetime(CONFIG["START_DATE"])
    ).days
    cagr = ((final_val / CONFIG["INITIAL_CASH"]) ** (365 / max(days, 1)) - 1) * 100

    result_df["Peak"] = result_df["자산총액"].cummax()
    result_df["DD"] = (result_df["자산총액"] - result_df["Peak"]) / result_df["Peak"]
    mdd = result_df["DD"].min() * 100

    win_rate = 0.0
    profit_factor = float("inf")
    if not trade_log_df.empty:
        win_trades = trade_log_df[trade_log_df["Profit"] > 0]
        loss_trades = trade_log_df[trade_log_df["Profit"] <= 0]
        win_rate = (len(win_trades) / len(trade_log_df)) * 100
        avg_prof = win_trades["Profit"].mean() * 100 if not win_trades.empty else 0
        avg_loss = loss_trades["Profit"].mean() * 100 if not loss_trades.empty else 0
        profit_factor = abs(avg_prof / avg_loss) if avg_loss != 0 else float("inf")

    metrics = {
        "FinalAsset": int(final_val),
        "TotalReturnPct": round(total_ret, 2),
        "CAGR": round(cagr, 2),
        "MDD": round(mdd, 2),
        "WinRate": round(win_rate, 2),
        "ProfitFactor": (
            round(profit_factor, 2) if profit_factor != float("inf") else profit_factor
        ),
        "Trades": int(len(trade_log_df)),
        "EarlyCut": False,
    }

    if not silent:
        print("\n" + "=" * 50)
        print("      백테스팅 성과 요약 리포트 ")
        print("=" * 50)
        print(f"- 최종 자산: {final_val:,}원")
        print(f"- 누적 수익률: {total_ret:.2f}% / CAGR: {cagr:.2f}%")
        print(f"- 최대 낙폭(MDD): {mdd:.2f}%")
        if not trade_log_df.empty:
            print(f"- 승률: {win_rate:.2f}% / 손익비: {profit_factor:.2f}")
        print("-" * 50)
        print("- 매도 사유별 통계:")
        for reason, count in sell_reasons.items():
            print(f"   - {reason}: {count}회")
        print("=" * 50)

    return result_df, trade_log_df, metrics


def run_regime_comparison():
    results = []
    price_cache = {}

    _, _, base_metrics = run_backtest(
        output_suffix="NO_REGIME",
        save_outputs=False,
        silent=True,
        price_cache=price_cache,
        use_regime_filter=False,
    )
    base_metrics["Regime"] = "NO_REGIME"
    results.append(base_metrics)

    for ticker in CONFIG.get("REGIME_TICKERS", []):
        regime_cache = load_regime_cache([ticker])
        _, _, metrics = run_backtest(
            output_suffix=ticker,
            save_outputs=False,
            silent=True,
            regime_cache=regime_cache,
            price_cache=price_cache,
            use_regime_filter=True,
            regime_requirement="any",
        )
        metrics["Regime"] = ticker
        results.append(metrics)

    if len(CONFIG.get("REGIME_TICKERS", [])) > 1:
        all_cache = load_regime_cache(CONFIG.get("REGIME_TICKERS", []))
        _, _, metrics_all = run_backtest(
            output_suffix="ALL_REGIMES",
            save_outputs=False,
            silent=True,
            regime_cache=all_cache,
            price_cache=price_cache,
            use_regime_filter=True,
            regime_requirement="all",
        )
        metrics_all["Regime"] = "ALL"
        results.append(metrics_all)

    result_df = pd.DataFrame(results)
    result_df = result_df[
        [
            "Regime",
            "FinalAsset",
            "TotalReturnPct",
            "CAGR",
            "MDD",
            "WinRate",
            "ProfitFactor",
            "Trades",
        ]
    ]
    result_df.to_csv(
        OUTPUT_PATH / "Regime_Comparison.csv", index=False, encoding="utf-8-sig"
    )

    best_row = result_df.sort_values("FinalAsset", ascending=False).iloc[0]
    print("\n[REGIME COMPARISON] 결과 저장 완료: Regime_Comparison.csv")
    print(
        f"[REGIME COMPARISON] 최고 성과: {best_row['Regime']} / 최종자산 {int(best_row['FinalAsset']):,}원"
    )


if __name__ == "__main__":
    if CONFIG.get("COMPARE_REGIMES", False):
        run_regime_comparison()
    else:
        use_regime = CONFIG.get("USE_REGIME_FILTER")
        selected_tickers = CONFIG.get("REGIME_SELECTED_TICKERS")
        output_suffix = None
        regime_cache = None
        if use_regime and selected_tickers:
            output_suffix = CONFIG.get("REGIME_OUTPUT_SUFFIX")
            regime_cache = load_regime_cache(selected_tickers)

        run_backtest(
            output_suffix=output_suffix,
            use_regime_filter=use_regime,
            regime_cache=regime_cache,
            regime_requirement=CONFIG.get("REGIME_SELECTED_REQUIREMENT"),
            regime_tickers=selected_tickers,
        )
