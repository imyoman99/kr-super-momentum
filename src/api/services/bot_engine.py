from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.api.services.financial_universe import (
    FINANCIAL_UNIVERSE_CONFIG,
    build_financial_universe,
    map_date_to_fiscal_period,
)
from src.api.services.bot_engine_trading import (
    build_trade_candidates as _build_trade_candidates,
    has_yesterday_positions as _has_yesterday_positions,
)
from src.backtester.strategies.minervini_filter import (
    check_minervini_from_df,
    evaluate_minervini_from_df,
)
from src.api.services.bot_engine_common import (
    find_latest_stage1_file as _find_latest_stage1_file,
    log_progress_percent as _log_progress_percent,
    next_business_day as _next_business_day,
    parse_hhmm as _parse_hhmm,
    parse_ymd as _parse_ymd,
    previous_business_day as _previous_business_day,
    push_log as _push_log,
    read_stage_codes as _read_stage_codes,
    resolve_path as _resolve_path,
    resolve_runtime_phase as _resolve_runtime_phase,
    resolve_signal_and_trade_date_by_runtime as _resolve_signal_and_trade_date_by_runtime,
    stage_file as _stage_file,
)
from src.api.utils.kis_client import KisClient
from src.api.utils.slack_notifier import create_slack_notifier


_KIS_CLIENT: KisClient | None = None
_PRICE_SLICE_CACHE: dict[tuple[str, str, date], pd.DataFrame | None] = {}
_MARKET_METRIC_CACHE: dict[str, dict[str, Any] | None] = {}
_BENCHMARK_RET_CACHE: dict[
    tuple[str, date, tuple[tuple[str, int], ...]], dict[str, float]
] = {}
_EXTERNAL_BENCH_SYMBOL_MAP: dict[str, str] = {
    "KS11": "^KS11",
    "KQ11": "^KQ11",
}


BOT_ENGINE_CONFIG: dict[str, Any] = {
    "price_data_dir": "data",
    "price_source": "fdr",
    "stage2_max_workers": 12,
    "stage_output_dir": "data/universe",
    "holdings_path": "data/runtime/holdings.json",
    "market_open_hhmm": "09:00",
    "market_close_hhmm": "15:30",
    "use_runtime_signal_date": True,
    "pivot_window": 10,
    "ranking": {
        "min_rs_rating": 80,
        "min_market_cap": 100_000_000_000,
        "trading_value_lookback_days": 20,
        "min_avg_daily_trading_value": 50_000_000_000,
        "rs_benchmark_code": "KS11",
        "rs_windows_days": {
            "3m": 63,
            "6m": 126,
            "9m": 189,
            "12m": 252,
        },
        "rs_weights": {
            "3m": 0.4,
            "6m": 0.3,
            "9m": 0.2,
            "12m": 0.1,
        },
    },
    "trading": {
        "account_equity": 100000000.0,
        "risk_per_trade": 0.01,
        "max_position_ratio": 0.15,
        "stop_atr_multiple": 2.5,
        "take_profit_r": 2.0,
    },
}


def _get_kis_client() -> KisClient:
    global _KIS_CLIENT
    if _KIS_CLIENT is None:
        _KIS_CLIENT = KisClient()
    return _KIS_CLIENT


def _get_market_metrics(code: str) -> dict[str, Any] | None:
    if code in _MARKET_METRIC_CACHE:
        return _MARKET_METRIC_CACHE[code]
    client = _get_kis_client()
    metrics = client.get_market_metrics(code)
    _MARKET_METRIC_CACHE[code] = metrics
    return metrics


def _load_price_slice(
    code: str,
    end_date: date,
    price_dir: Path,
    price_source: str = "auto",
) -> pd.DataFrame | None:
    del price_dir
    normalized_source = str(price_source or "auto").strip().lower()
    cache_key = (normalized_source, code, end_date)
    if cache_key in _PRICE_SLICE_CACHE:
        return _PRICE_SLICE_CACHE[cache_key]

    def _finalize_df(raw_df: pd.DataFrame | None) -> pd.DataFrame | None:
        if raw_df is None or raw_df.empty:
            return None

        df = raw_df.copy()
        if "Date" not in df.columns:
            df = df.reset_index()
            if "Date" not in df.columns:
                df = df.rename(columns={df.columns[0]: "Date"})

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        df = df.loc[: pd.Timestamp(end_date)].copy()
        if df.empty:
            return None

        if "Open" not in df.columns and "Close" in df.columns:
            df["Open"] = df["Close"]
        if "High" not in df.columns and "Close" in df.columns:
            df["High"] = df["Close"]
        if "Low" not in df.columns and "Close" in df.columns:
            df["Low"] = df["Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 0.0

        if "TradingValue" not in df.columns:
            df["TradingValue"] = pd.to_numeric(
                df["Close"], errors="coerce"
            ) * pd.to_numeric(df["Volume"], errors="coerce")

        if "MA50" not in df.columns:
            df["MA50"] = (
                pd.to_numeric(df["Close"], errors="coerce").rolling(window=50).mean()
            )
        if "MA150" not in df.columns:
            df["MA150"] = (
                pd.to_numeric(df["Close"], errors="coerce").rolling(window=150).mean()
            )
        if "MA200" not in df.columns:
            df["MA200"] = (
                pd.to_numeric(df["Close"], errors="coerce").rolling(window=200).mean()
            )
        if "ATR_14" not in df.columns:
            tr = pd.concat(
                [
                    pd.to_numeric(df["High"], errors="coerce")
                    - pd.to_numeric(df["Low"], errors="coerce"),
                    (
                        pd.to_numeric(df["High"], errors="coerce")
                        - pd.to_numeric(df["Close"], errors="coerce").shift(1)
                    ).abs(),
                    (
                        pd.to_numeric(df["Low"], errors="coerce")
                        - pd.to_numeric(df["Close"], errors="coerce").shift(1)
                    ).abs(),
                ],
                axis=1,
            ).max(axis=1)
            df["ATR_14"] = tr.rolling(window=14).mean()

        needed = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "TradingValue",
            "MarketCap",
            "MA50",
            "MA150",
            "MA200",
            "ATR_14",
        ]
        return df[[c for c in needed if c in df.columns]].copy()

    if normalized_source in {"fdr", "auto"}:
        try:
            import FinanceDataReader as fdr  # type: ignore

            start_text = (end_date - timedelta(days=450)).isoformat()
            end_text = end_date.isoformat()
            fdr_df = fdr.DataReader(code, start_text, end_text)
            finalized = _finalize_df(fdr_df)
            if finalized is not None:
                _PRICE_SLICE_CACHE[cache_key] = finalized
                return finalized
        except Exception:
            if normalized_source == "fdr":
                _PRICE_SLICE_CACHE[cache_key] = None
                return None

    if normalized_source in {"kis", "auto"}:
        try:
            client = _get_kis_client()
            api_df = client.get_daily_ohlcv(code)
            finalized = _finalize_df(api_df)
            _PRICE_SLICE_CACHE[cache_key] = finalized
            return finalized
        except Exception:
            _PRICE_SLICE_CACHE[cache_key] = None
            return None

    _PRICE_SLICE_CACHE[cache_key] = None
    return None


def _load_external_benchmark_slice(
    code: str,
    end_date: date,
    logs: list[str],
    log_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame | None:
    cache_key = ("external", code, end_date)
    cached = _PRICE_SLICE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    start_text = (end_date - timedelta(days=450)).isoformat()
    end_text = end_date.isoformat()

    try:
        import FinanceDataReader as fdr  # type: ignore

        raw = fdr.DataReader(code, start_text, end_text)
        if raw is not None and not raw.empty and "Close" in raw.columns:
            df = raw.reset_index()
            if "Date" not in df.columns:
                df = df.rename(columns={df.columns[0]: "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
            df = df.loc[: pd.Timestamp(end_date)].copy()
            if not df.empty:
                if "Open" not in df.columns:
                    df["Open"] = df["Close"]
                if "High" not in df.columns:
                    df["High"] = df["Close"]
                if "Low" not in df.columns:
                    df["Low"] = df["Close"]
                if "Volume" not in df.columns:
                    df["Volume"] = 0.0
                _push_log(
                    logs,
                    f"2차 RS 벤치마크 외부 폴백(FDR) 사용: {code}",
                    log_callback,
                )
                _PRICE_SLICE_CACHE[cache_key] = df
                return df
    except Exception:
        pass

    try:
        import yfinance as yf  # type: ignore

        symbol = _EXTERNAL_BENCH_SYMBOL_MAP.get(code.upper(), code)
        raw = yf.download(symbol, start=start_text, end=end_text, progress=False)
        if raw is not None and not raw.empty and "Close" in raw.columns:
            df = raw.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()
            df = df.loc[: pd.Timestamp(end_date)].copy()
            if not df.empty:
                if "Open" not in df.columns:
                    df["Open"] = df["Close"]
                if "High" not in df.columns:
                    df["High"] = df["Close"]
                if "Low" not in df.columns:
                    df["Low"] = df["Close"]
                if "Volume" not in df.columns:
                    df["Volume"] = 0.0
                _push_log(
                    logs,
                    f"2차 RS 벤치마크 외부 폴백(yfinance) 사용: {symbol}",
                    log_callback,
                )
                _PRICE_SLICE_CACHE[cache_key] = df
                return df
    except Exception:
        pass

    return None


def _get_close_on_or_before(
    price_df: pd.DataFrame | None, target_date: date
) -> tuple[float | None, date | None]:
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return None, None

    close = pd.to_numeric(price_df["Close"], errors="coerce").dropna()
    if close.empty:
        return None, None

    close.index = pd.to_datetime(close.index, errors="coerce")
    close = close[~close.index.isna()].sort_index()
    if close.empty:
        return None, None

    target_ts = pd.Timestamp(target_date)

    exact = close[close.index == target_ts]
    if not exact.empty:
        return float(exact.iloc[-1]), target_date

    past = close[close.index < target_ts]
    if not past.empty:
        ts = past.index[-1]
        return float(past.iloc[-1]), ts.date()

    return None, None


def _ensure_stage1(
    stage_dir: Path,
    signal_date: date,
    financial_data_path: str,
    financial_cache_dir: str | None,
    financial_api_options: dict[str, Any],
    extra_financial_conditions: list[dict[str, Any]],
    force_rebuild: bool,
    logs: list[str],
    log_callback: Callable[[str], None] | None = None,
) -> list[str]:
    stage1_path = _stage_file(stage_dir, "stage1_financial", signal_date)
    _push_log(logs, f"1차 유니버스 파일 확인 중: {stage1_path.name}", log_callback)

    if stage1_path.exists() and not force_rebuild:
        _push_log(
            logs, f"1차 유니버스 기존 파일 사용: {stage1_path.name}", log_callback
        )
        return _read_stage_codes(stage1_path)

    if not force_rebuild:
        latest_path, latest_date = _find_latest_stage1_file(stage_dir, signal_date)
        if latest_path is not None and latest_date is not None:
            current_period = map_date_to_fiscal_period(signal_date)
            latest_period = map_date_to_fiscal_period(latest_date)
            if current_period == latest_period:
                loaded = pd.read_csv(latest_path, dtype={"Code": str})
                codes = loaded["Code"].dropna().astype(str).unique().tolist()
                _push_log(
                    logs,
                    f"1차 유니버스 동일 분기 재사용: {latest_path.name} (기간 {latest_period[0]}-{latest_period[1]})",
                    log_callback,
                )
                return codes
            _push_log(
                logs,
                f"1차 유니버스 분기 변경 감지: {latest_period[0]}-{latest_period[1]} -> {current_period[0]}-{current_period[1]}, 재생성",
                log_callback,
            )

    _push_log(logs, "1차 유니버스 생성 중...", log_callback)

    result = build_financial_universe(
        {
            "use_today": False,
            "reference_date": signal_date.isoformat(),
            "financial_data_path": financial_data_path,
            "financial_cache_dir": financial_cache_dir
            or FINANCIAL_UNIVERSE_CONFIG.get("financial_cache_dir"),
            "financial_api": financial_api_options,
            "financial_source": "api",
            "extra_conditions": extra_financial_conditions,
        },
        progress_callback=(
            (lambda msg: _push_log(logs, msg, log_callback))
            if log_callback is not None
            else None
        ),
    )
    codes = result["codes"]
    stage1_rows = result.get("stage1_rows")
    if isinstance(stage1_rows, list) and stage1_rows:
        stage1_df = pd.DataFrame(stage1_rows)
        if "Code" in stage1_df.columns:
            stage1_df["Code"] = stage1_df["Code"].astype(str)
        stage1_df.to_csv(stage1_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame({"Code": codes}).to_csv(
            stage1_path, index=False, encoding="utf-8-sig"
        )
    _push_log(logs, f"1차 유니버스 생성 완료: {len(codes)}개", log_callback)
    return codes


def _ensure_stage2(
    stage_dir: Path,
    signal_date: date,
    stage1_codes: list[str],
    price_data_dir: Path,
    min_rs_rating: float,
    min_market_cap: float,
    trading_value_lookback_days: int,
    min_avg_daily_trading_value: float,
    rs_benchmark_code: str,
    rs_windows_days: dict[str, int],
    rs_weights: dict[str, float],
    price_source: str,
    stage2_max_workers: int,
    force_rebuild: bool,
    logs: list[str],
    log_callback: Callable[[str], None] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    stage2_path = _stage_file(stage_dir, "stage2_ranked", signal_date)
    stage2_diag_path = (
        stage_dir / f"{signal_date.strftime('%Y%m%d')}_2차_RS시총거래대금_진단.csv"
    )
    _push_log(logs, f"2차 유니버스 파일 확인 중: {stage2_path.name}", log_callback)

    if stage2_path.exists() and not force_rebuild:
        loaded = pd.read_csv(stage2_path, dtype={"Code": str})
        loaded_codes = (
            loaded["Code"].dropna().astype(str).unique().tolist()
            if "Code" in loaded.columns
            else []
        )
        if stage1_codes and not loaded_codes:
            _push_log(
                logs,
                f"2차 유니버스 기존 파일이 비어 있어 재생성: {stage2_path.name}",
                log_callback,
            )
        else:
            _push_log(
                logs, f"2차 유니버스 기존 파일 사용: {stage2_path.name}", log_callback
            )
            return loaded_codes, loaded

    _push_log(logs, "2차 유니버스 생성 중...", log_callback)

    day_df = pd.DataFrame({"Code": sorted(set(stage1_codes))})
    if day_df.empty:
        day_df.to_csv(stage2_path, index=False, encoding="utf-8-sig")
        _push_log(
            logs,
            "2차 유니버스 생성 완료(RS Rating/시총/20일평균거래대금): 0개",
            log_callback,
        )
        return [], day_df

    benchmark_ret: dict[str, float] = {}
    benchmark_cache_key = (
        rs_benchmark_code,
        signal_date,
        tuple(sorted((str(k), int(v)) for k, v in rs_windows_days.items())),
    )
    cached_benchmark_ret = _BENCHMARK_RET_CACHE.get(benchmark_cache_key)
    if cached_benchmark_ret is not None:
        benchmark_ret = dict(cached_benchmark_ret)
        _push_log(
            logs,
            f"2차 RS 벤치마크 캐시 재사용: {rs_benchmark_code} ({signal_date.isoformat()})",
            log_callback,
        )

    benchmark_df = _load_price_slice(
        rs_benchmark_code,
        signal_date,
        price_data_dir,
        price_source=price_source,
    )
    if benchmark_df is None:
        benchmark_df = _load_external_benchmark_slice(
            rs_benchmark_code,
            signal_date,
            logs,
            log_callback,
        )
    benchmark_now_close: float | None = None
    if benchmark_df is None:
        _push_log(
            logs,
            f"2차 RS 계산 실패: 코스피 벤치마크({rs_benchmark_code}) 데이터 없음",
            log_callback,
        )
    elif not benchmark_ret:
        bench_close_series = pd.to_numeric(benchmark_df["Close"], errors="coerce")
        bench_today = bench_close_series[
            bench_close_series.index == pd.Timestamp(signal_date)
        ]
        if not bench_today.empty:
            benchmark_now_close = float(bench_today.iloc[-1])
        else:
            _push_log(
                logs,
                f"2차 RS 계산 실패: 벤치마크({rs_benchmark_code}) 기준일({signal_date.isoformat()}) 종가 없음",
                log_callback,
            )

        for key, lb in rs_windows_days.items():
            if benchmark_now_close is None:
                continue

            lookback_date = signal_date - timedelta(days=int(lb))
            b_prev, _ = _get_close_on_or_before(benchmark_df, lookback_date)
            if b_prev is None:
                continue

            if b_prev > 0:
                benchmark_ret[key] = (benchmark_now_close / b_prev) - 1

        _BENCHMARK_RET_CACHE[benchmark_cache_key] = dict(benchmark_ret)

    rs_raw_map: dict[str, float] = {}
    avg_value_map: dict[str, float] = {}
    marcap_map: dict[str, float] = {}
    daily_value_map: dict[str, float] = {}
    stage2_targets = day_df["Code"].dropna().astype(str).unique().tolist()
    stage2_total = len(stage2_targets)
    stage2_progress_bucket = -1
    price_map: dict[str, pd.DataFrame | None] = {}

    if stage2_total > 0:
        max_workers = max(1, int(stage2_max_workers))
        _push_log(
            logs,
            f"2차 가격 데이터 병렬 수집 시작: {stage2_total}개 (워커 {max_workers})",
            log_callback,
        )
        prefetch_progress_bucket = -1
        prefetch_completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _load_price_slice,
                    code,
                    signal_date,
                    price_data_dir,
                    price_source,
                ): code
                for code in stage2_targets
            }
            for future in as_completed(future_map):
                code = future_map[future]
                try:
                    price_map[code] = future.result()
                except Exception:
                    price_map[code] = None
                prefetch_completed += 1
                prefetch_progress_bucket = _log_progress_percent(
                    logs,
                    "2차 가격수집",
                    prefetch_completed,
                    stage2_total,
                    prefetch_progress_bucket,
                    log_callback,
                )
        _push_log(logs, "2차 가격 데이터 병렬 수집 완료", log_callback)

    for idx, code in enumerate(stage2_targets, start=1):
        code_df = price_map.get(code)
        if code_df is None:
            stage2_progress_bucket = _log_progress_percent(
                logs,
                "2차 유니버스",
                idx,
                stage2_total,
                stage2_progress_bucket,
                log_callback,
            )
            continue

        if "MarketCap" in code_df.columns:
            mcap_series = pd.to_numeric(code_df["MarketCap"], errors="coerce").dropna()
            if not mcap_series.empty:
                marcap_map[code] = float(mcap_series.iloc[-1])

        metrics = _get_market_metrics(code)
        if metrics is not None:
            daily_value = metrics.get("daily_trading_value")
            if daily_value is not None:
                daily_value_map[code] = float(daily_value)

        stock_close_series = pd.to_numeric(code_df["Close"], errors="coerce")
        stock_today = stock_close_series[
            stock_close_series.index == pd.Timestamp(signal_date)
        ]
        if stock_today.empty:
            continue
        stock_now_close = float(stock_today.iloc[-1])

        if code not in marcap_map and metrics is not None:
            listed_shares = metrics.get("listed_shares")
            if listed_shares is not None and float(listed_shares) > 0:
                marcap_map[code] = float(listed_shares) * stock_now_close
            else:
                market_cap = metrics.get("market_cap")
                if market_cap is not None:
                    marcap_map[code] = float(market_cap)

        if len(code_df) >= trading_value_lookback_days:
            if "TradingValue" in code_df.columns:
                v_series = pd.to_numeric(code_df["TradingValue"], errors="coerce").tail(
                    trading_value_lookback_days
                )
            else:
                v_series = (
                    pd.to_numeric(code_df["Close"], errors="coerce")
                    * pd.to_numeric(code_df["Volume"], errors="coerce")
                ).tail(trading_value_lookback_days)
            avg_value_map[code] = float(v_series.mean())

        weighted_score = 0.0
        valid_term = 0
        for key, weight in rs_weights.items():
            lb = rs_windows_days.get(key)
            if lb is None:
                continue

            lookback_date = signal_date - timedelta(days=int(lb))
            s_prev, _ = _get_close_on_or_before(code_df, lookback_date)
            if s_prev is None:
                continue

            if s_prev <= 0:
                continue

            stock_ret = (stock_now_close / s_prev) - 1
            benchmark_component = benchmark_ret.get(key)
            if benchmark_component is None:
                excess_ret = stock_ret
            else:
                excess_ret = stock_ret - benchmark_component
            weighted_score += excess_ret * float(weight)
            valid_term += 1

        if valid_term > 0:
            rs_raw_map[code] = weighted_score

        stage2_progress_bucket = _log_progress_percent(
            logs,
            "2차 유니버스",
            idx,
            stage2_total,
            stage2_progress_bucket,
            log_callback,
        )

    day_df["RS_RAW"] = day_df["Code"].map(rs_raw_map)
    day_df["avg_trading_value_20d"] = day_df["Code"].map(avg_value_map)
    day_df["marcap"] = day_df["Code"].map(marcap_map)
    day_df["daily_trading_value"] = day_df["Code"].map(daily_value_map)
    day_df["avg_trading_value_20d"] = day_df["avg_trading_value_20d"].fillna(
        day_df["daily_trading_value"]
    )

    raw_non_null = day_df["RS_RAW"].dropna()
    if len(raw_non_null) > 0:
        rank = raw_non_null.rank(method="min", ascending=True)
        rating = ((rank - 1) / len(raw_non_null)) * 100.0
        day_df.loc[raw_non_null.index, "RS"] = rating
    else:
        day_df["RS"] = pd.NA

    day_df["RS"] = pd.to_numeric(day_df["RS"], errors="coerce")
    day_df["marcap"] = pd.to_numeric(day_df["marcap"], errors="coerce")
    day_df["avg_trading_value_20d"] = pd.to_numeric(
        day_df["avg_trading_value_20d"], errors="coerce"
    )

    rs_mask = day_df["RS"] >= min_rs_rating
    marcap_mask = (day_df["marcap"] >= min_market_cap) | day_df["marcap"].isna()
    value_mask = (
        day_df["avg_trading_value_20d"] >= min_avg_daily_trading_value
    ) | day_df["avg_trading_value_20d"].isna()
    combined_mask = rs_mask & marcap_mask & value_mask

    _push_log(
        logs,
        (
            "2차 필터 진단: "
            f"RS통과={int(rs_mask.sum())}/{len(day_df)}, "
            f"시총통과={int(marcap_mask.sum())}/{len(day_df)}, "
            f"거래대금통과={int(value_mask.sum())}/{len(day_df)}, "
            f"최종통과={int(combined_mask.sum())}/{len(day_df)}"
        ),
        log_callback,
    )

    marcap_missing = int(day_df["marcap"].isna().sum())
    value_missing = int(day_df["avg_trading_value_20d"].isna().sum())
    if marcap_missing > 0:
        _push_log(
            logs,
            f"2차 API 시총 누락: {marcap_missing}개 (누락 종목은 시총 필터 예외 적용)",
            log_callback,
        )
    if value_missing > 0:
        _push_log(
            logs,
            f"2차 API 거래대금 누락: {value_missing}개 (누락 종목은 거래대금 필터 예외 적용)",
            log_callback,
        )

    passed_df = day_df[combined_mask].copy()
    passed_df = passed_df.sort_values(
        ["RS", "RS_RAW", "avg_trading_value_20d", "marcap"], ascending=False
    )

    if not passed_df.empty:
        rs_series = pd.to_numeric(passed_df["RS"], errors="coerce").dropna()
        if not rs_series.empty:
            _push_log(
                logs,
                (
                    "2차 RS 분포(TopN 적용 전): "
                    f"min={rs_series.min():.3f}, max={rs_series.max():.3f}, "
                    f"count={len(rs_series)}"
                ),
                log_callback,
            )

    passed_df.to_csv(stage2_diag_path, index=False, encoding="utf-8-sig")
    _push_log(
        logs,
        f"2차 진단 저장(TopN 전): {stage2_diag_path.name}",
        log_callback,
    )

    day_df = passed_df
    _push_log(
        logs,
        f"2차 TopN 미적용: 기준 통과 종목 전체 사용 ({len(day_df)}개)",
        log_callback,
    )

    day_df.to_csv(stage2_path, index=False, encoding="utf-8-sig")
    codes = day_df["Code"].dropna().astype(str).unique().tolist()
    _push_log(
        logs,
        f"2차 유니버스 생성 완료(RS Rating/시총/20일평균거래대금): {len(codes)}개",
        log_callback,
    )
    return codes, day_df


def _ensure_stage3(
    stage_dir: Path,
    signal_date: date,
    stage2_codes: list[str],
    price_data_dir: Path,
    price_source: str,
    force_rebuild: bool,
    logs: list[str],
    log_callback: Callable[[str], None] | None = None,
) -> list[str]:
    stage3_path = _stage_file(stage_dir, "stage3_minervini", signal_date)
    stage3_diag_path = (
        stage_dir / f"{signal_date.strftime('%Y%m%d')}_3차_미너비니_진단.csv"
    )
    _push_log(logs, f"3차 유니버스 파일 확인 중: {stage3_path.name}", log_callback)

    if stage3_path.exists() and not force_rebuild:
        loaded = pd.read_csv(stage3_path, dtype={"Code": str})
        loaded_codes = (
            loaded["Code"].dropna().astype(str).unique().tolist()
            if "Code" in loaded.columns
            else []
        )
        if stage2_codes and not loaded_codes:
            _push_log(
                logs,
                f"최종 유니버스 기존 파일이 비어 있어 재생성: {stage3_path.name}",
                log_callback,
            )
        else:
            _push_log(
                logs, f"최종 유니버스 기존 파일 사용: {stage3_path.name}", log_callback
            )
            return loaded_codes

    _push_log(logs, "3차 미너비니 필터 생성 중...", log_callback)

    final_codes: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    signal_ts = pd.Timestamp(signal_date)
    stage3_total = len(stage2_codes)
    stage3_progress_bucket = -1
    for idx, code in enumerate(stage2_codes, start=1):
        df = _load_price_slice(code, signal_date, price_data_dir, price_source)
        if df is None:
            diagnostics.append(
                {
                    "Code": code,
                    "stage3_pass": 0,
                    "reason": "no_price_data",
                    "price_data_ok": 0,
                    "price_above_mas": 0,
                    "ma_alignment": 0,
                    "ma200_up": 0,
                    "ma50_up": 0,
                    "near_6m_high": 0,
                    "ma150_up": 0,
                    "volatility_contraction": 0,
                    "value_contraction": 0,
                    "contraction_ok": 0,
                }
            )
            stage3_progress_bucket = _log_progress_percent(
                logs,
                "3차 유니버스",
                idx,
                stage3_total,
                stage3_progress_bucket,
                log_callback,
            )
            continue

        detail = evaluate_minervini_from_df(df, signal_ts)
        diagnostics.append(
            {
                "Code": code,
                "stage3_pass": 1 if bool(detail.get("pass", False)) else 0,
                "reason": str(detail.get("reason", "unknown")),
                "price_data_ok": 1 if bool(detail.get("price_data_ok", False)) else 0,
                "price_above_mas": (
                    1 if bool(detail.get("price_above_mas", False)) else 0
                ),
                "ma_alignment": 1 if bool(detail.get("ma_alignment", False)) else 0,
                "ma200_up": 1 if bool(detail.get("ma200_up", False)) else 0,
                "ma50_up": 1 if bool(detail.get("ma50_up", False)) else 0,
                "near_6m_high": 1 if bool(detail.get("near_6m_high", False)) else 0,
                "ma150_up": 1 if bool(detail.get("ma150_up", False)) else 0,
                "volatility_contraction": (
                    1 if bool(detail.get("volatility_contraction", False)) else 0
                ),
                "value_contraction": (
                    1 if bool(detail.get("value_contraction", False)) else 0
                ),
                "contraction_ok": 1 if bool(detail.get("contraction_ok", False)) else 0,
            }
        )
        if check_minervini_from_df(df, signal_ts):
            final_codes.append(code)

        stage3_progress_bucket = _log_progress_percent(
            logs,
            "3차 유니버스",
            idx,
            stage3_total,
            stage3_progress_bucket,
            log_callback,
        )

    pd.DataFrame({"Code": sorted(set(final_codes))}).to_csv(
        stage3_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(diagnostics).to_csv(
        stage3_diag_path, index=False, encoding="utf-8-sig"
    )
    _push_log(logs, f"3차 미너비니 진단 저장: {stage3_diag_path.name}", log_callback)
    _push_log(logs, f"3차 미너비니 필터 완료: {len(final_codes)}개", log_callback)
    return sorted(set(final_codes))


def run_bot_pipeline(
    params: dict[str, Any],
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    config = {**BOT_ENGINE_CONFIG, **params}
    base_log_callback = log_callback
    slack_notifier = create_slack_notifier(
        config.get("slack"),
        prefix="[BOT] ",
    )

    def _log_callback(message: str) -> None:
        if base_log_callback is not None:
            try:
                base_log_callback(message)
            except Exception:
                pass
        slack_notifier.send(message)

    log_callback = _log_callback

    mode = str(config.get("mode", "universe")).lower()
    if mode not in {"universe", "trading"}:
        raise ValueError("mode는 universe 또는 trading 이어야 합니다.")

    reference_date = _parse_ymd(str(config.get("reference_date")))
    if reference_date >= date.today():
        raise ValueError("기준일은 과거 날짜만 허용됩니다. (reference_date < today)")

    logs: list[str] = []
    _push_log(
        logs,
        f"봇 실행 시작: mode={mode}, reference_date={reference_date.isoformat()}",
        log_callback,
    )

    stage_dir = _resolve_path(config["stage_output_dir"])
    stage_dir.mkdir(parents=True, exist_ok=True)

    price_data_dir = _resolve_path(config["price_data_dir"])
    holdings_path = _resolve_path(config["holdings_path"])
    financial_data_path = (
        config.get("financial_data_path")
        or FINANCIAL_UNIVERSE_CONFIG["financial_data_path"]
    )
    financial_data_path = str(_resolve_path(financial_data_path))
    financial_cache_dir = config.get(
        "financial_cache_dir"
    ) or FINANCIAL_UNIVERSE_CONFIG.get("financial_cache_dir")
    financial_api_options = {
        **FINANCIAL_UNIVERSE_CONFIG.get("financial_api", {}),
        **config.get("financial_api", {}),
    }

    runtime_phase = "n/a"
    now_dt = datetime.now()

    if mode == "trading":
        market_open_hhmm = str(
            config.get("market_open_hhmm", BOT_ENGINE_CONFIG["market_open_hhmm"])
        )
        market_close_hhmm = str(
            config.get("market_close_hhmm", BOT_ENGINE_CONFIG["market_close_hhmm"])
        )
        runtime_phase = _resolve_runtime_phase(
            now_dt=now_dt,
            market_open_hhmm=market_open_hhmm,
            market_close_hhmm=market_close_hhmm,
        )

        if bool(config.get("use_runtime_signal_date", True)):
            signal_date, trade_date, signal_reason = (
                _resolve_signal_and_trade_date_by_runtime(
                    now_dt=now_dt,
                    market_open_hhmm=market_open_hhmm,
                    market_close_hhmm=market_close_hhmm,
                )
            )
            _push_log(logs, f"신호일 자동결정: {signal_reason}", log_callback)
            _push_log(logs, f"런타임 장상태: {runtime_phase}", log_callback)
        else:
            signal_date = _previous_business_day(reference_date)
            trade_date = reference_date
    else:
        signal_date = reference_date
        trade_date = reference_date

    _push_log(
        logs,
        f"신호기준일(signal_date)={signal_date.isoformat()}, 거래기준일(trade_date)={trade_date.isoformat()}",
        log_callback,
    )

    ranking = {**BOT_ENGINE_CONFIG["ranking"], **config.get("ranking", {})}
    trading = {**BOT_ENGINE_CONFIG["trading"], **config.get("trading", {})}
    force_rebuild = bool(config.get("force_rebuild", False))
    price_source = str(config.get("price_source", BOT_ENGINE_CONFIG["price_source"]))
    stage2_max_workers = int(
        config.get("stage2_max_workers", BOT_ENGINE_CONFIG["stage2_max_workers"])
    )
    _push_log(
        logs,
        f"2차 가격소스={price_source}, 병렬 워커={stage2_max_workers}",
        log_callback,
    )

    _push_log(logs, "1차 유니버스 단계 시작", log_callback)
    stage1_codes = _ensure_stage1(
        stage_dir=stage_dir,
        signal_date=signal_date,
        financial_data_path=financial_data_path,
        financial_cache_dir=financial_cache_dir,
        financial_api_options=financial_api_options,
        extra_financial_conditions=config.get("extra_financial_conditions", []),
        force_rebuild=force_rebuild,
        logs=logs,
        log_callback=log_callback,
    )

    stage1_path = _stage_file(stage_dir, "stage1_financial", signal_date)
    if stage1_path.exists():
        stage1_codes = _read_stage_codes(stage1_path)
    else:
        latest_path, latest_date = _find_latest_stage1_file(stage_dir, signal_date)
        if latest_path is None or latest_date is None:
            raise FileNotFoundError(
                f"1차 유니버스 파일이 없습니다: {stage1_path}. 1차 생성 후 2차를 진행하세요."
            )
        current_period = map_date_to_fiscal_period(signal_date)
        latest_period = map_date_to_fiscal_period(latest_date)
        if current_period != latest_period:
            raise FileNotFoundError(
                f"1차 유니버스 파일이 없습니다: {stage1_path}. 1차 생성 후 2차를 진행하세요."
            )
        stage1_codes = _read_stage_codes(latest_path)
        _push_log(
            logs,
            f"2차 진행 시 1차 동일 분기 파일 사용: {latest_path.name}",
            log_callback,
        )

    _push_log(logs, "2차 유니버스 단계 시작", log_callback)
    stage2_codes, stage2_df = _ensure_stage2(
        stage_dir=stage_dir,
        signal_date=signal_date,
        stage1_codes=stage1_codes,
        price_data_dir=price_data_dir,
        min_rs_rating=float(ranking["min_rs_rating"]),
        min_market_cap=float(ranking["min_market_cap"]),
        trading_value_lookback_days=int(ranking["trading_value_lookback_days"]),
        min_avg_daily_trading_value=float(ranking["min_avg_daily_trading_value"]),
        rs_benchmark_code=str(ranking.get("rs_benchmark_code", "KS11")),
        rs_windows_days=dict(ranking.get("rs_windows_days", {})),
        rs_weights=dict(ranking.get("rs_weights", {})),
        price_source=price_source,
        stage2_max_workers=stage2_max_workers,
        force_rebuild=force_rebuild,
        logs=logs,
        log_callback=log_callback,
    )

    stage2_path = _stage_file(stage_dir, "stage2_ranked", signal_date)
    if stage2_path.exists():
        stage2_codes = _read_stage_codes(stage2_path)
    else:
        raise FileNotFoundError(
            f"2차 유니버스 파일이 없습니다: {stage2_path}. 2차 생성 후 3차를 진행하세요."
        )

    _push_log(logs, "3차 유니버스 단계 시작", log_callback)
    final_codes = _ensure_stage3(
        stage_dir=stage_dir,
        signal_date=signal_date,
        stage2_codes=stage2_codes,
        price_data_dir=price_data_dir,
        price_source=price_source,
        force_rebuild=force_rebuild,
        logs=logs,
        log_callback=log_callback,
    )
    _push_log(logs, f"최종 유니버스 생성 완료: {len(final_codes)}개", log_callback)

    has_yesterday_positions, yesterday_codes = _has_yesterday_positions(
        holdings_path=holdings_path,
        trade_date=trade_date,
        parse_ymd=_parse_ymd,
    )
    if has_yesterday_positions:
        _push_log(logs, f"어제 보유 종목 확인: {len(yesterday_codes)}개", log_callback)
    else:
        _push_log(logs, "어제 보유 종목 없음", log_callback)

    trade_candidates: list[dict[str, Any]] = []
    if mode == "trading" and final_codes:
        trade_candidates = _build_trade_candidates(
            final_codes=final_codes,
            signal_date=signal_date,
            trade_date=trade_date,
            price_data_dir=price_data_dir,
            price_source=price_source,
            account_equity=float(trading["account_equity"]),
            risk_per_trade=float(trading["risk_per_trade"]),
            max_position_ratio=float(trading["max_position_ratio"]),
            stop_atr_multiple=float(trading["stop_atr_multiple"]),
            take_profit_r=float(trading["take_profit_r"]),
            pivot_window=int(
                config.get("pivot_window", BOT_ENGINE_CONFIG["pivot_window"])
            ),
            load_price_slice=_load_price_slice,
        )

    ready_count = sum(1 for item in trade_candidates if item.get("status") == "ready")
    await_open_count = sum(
        1 for item in trade_candidates if item.get("status") == "await_trade_day_open"
    )

    status_counts: dict[str, int] = {}
    for item in trade_candidates:
        status = str(item.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    if mode == "trading" and trade_candidates:
        for item in trade_candidates:
            pivot_price = item.get("pivot_price")
            if pivot_price is None:
                continue
            open_price = item.get("open_price")
            msg = (
                f"매수후보 code={item.get('code')}, status={item.get('status')}, "
                f"pivot={float(pivot_price):.2f}"
            )
            if open_price is not None:
                msg += f", open={float(open_price):.2f}"
            _push_log(logs, msg, log_callback)

    if mode == "trading":
        if not final_codes:
            _push_log(logs, "오늘 매수할 종목이 없습니다", log_callback)
        elif ready_count > 0:
            _push_log(logs, f"오늘 진입 가능 종목: {ready_count}개", log_callback)
        elif await_open_count > 0:
            if trade_date > now_dt.date() or runtime_phase == "after_close":
                _push_log(
                    logs,
                    f"다음 거래일 매수 후보: {await_open_count}개 (장시작 후 피벗 돌파 확인)",
                    log_callback,
                )
            else:
                _push_log(
                    logs,
                    f"오늘 매수 후보: {await_open_count}개 (장시작 후 시가/피벗 확인)",
                    log_callback,
                )
        else:
            _push_log(
                logs,
                f"오늘 매수할 종목이 없습니다 (사유별: {status_counts})",
                log_callback,
            )
    else:
        if not final_codes:
            _push_log(logs, "오늘 매수할 종목이 없습니다", log_callback)

    return {
        "mode": mode,
        "reference_date": reference_date.isoformat(),
        "signal_date": signal_date.isoformat(),
        "trade_date": trade_date.isoformat(),
        "has_yesterday_positions": has_yesterday_positions,
        "yesterday_positions": yesterday_codes,
        "stage_counts": {
            "stage1_financial": len(stage1_codes),
            "stage2_ranked": len(stage2_codes),
            "stage3_minervini": len(final_codes),
            "trade_ready": ready_count,
        },
        "final_codes": final_codes,
        "ranked_snapshot": stage2_df.to_dict(orient="records"),
        "trade_candidates": trade_candidates,
        "logs": logs,
    }
