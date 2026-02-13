import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# [MINERVINI_CONFIG] 기술적 지표 및 임계값 설정
MINERVINI_CONFIG = {
    "MA_MEDIUM": 50,  # 중기 이동평균선
    "MA_LONG_15": 150,  # 장기 이동평균선 1
    "MA_LONG_20": 200,  # 장기 이동평균선 2
    "LOOKBACK_200MA": 20,  # 200일선 추세 확인 기간
    "LOOKBACK_150MA": 10,  # 150일선 추세 확인 기간
    "LOOKBACK_50MA": 5,  # 5일선 추세 확인 기간
    "MAX_DIST_6M_HIGH": 0.15,  # 6개월 고점 대비 최대 이격 (15% 이내)
    "SIX_MONTH_WINDOW": 120,  # 6개월(거래일 기준 약 120일)
    "VOL_LOOKBACK": 10,  # 거래량 돌파 확인 기간
    "VCP_WINDOW": 20,  # 변동성/거래량 수축 확인 윈도우
    "ATR_WINDOW": 14,  # ATR 계산 기간
}


def load_from_parquet(code, data_dir="./data/"):
    """Parquet 파일 로드 및 기술적 지표(MA, TR, ATR) 계산"""
    file_path = f"{data_dir}/{code}.parquet"
    try:
        df = pd.read_parquet(file_path)
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index)

        # 이동평균선 계산
        for ma in [
            MINERVINI_CONFIG["MA_MEDIUM"],
            MINERVINI_CONFIG["MA_LONG_15"],
            MINERVINI_CONFIG["MA_LONG_20"],
        ]:
            col = f"MA{ma}"
            if col not in df.columns:
                df[col] = df["Close"].rolling(window=ma).mean()

        # ATR 계산
        if "ATR_14" not in df.columns:
            tr = pd.concat(
                [
                    df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift(1)).abs(),
                    (df["Low"] - df["Close"].shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            df["ATR_14"] = tr.rolling(window=MINERVINI_CONFIG["ATR_WINDOW"]).mean()

        return df
    except:
        return None


# ---------------------------------------------------------
# 개별 필터 로직
# ---------------------------------------------------------


def check_price_above_mas(close, ma50, ma150, ma200):
    """현재가 > MA50 AND 현재가 > MA150 AND 현재가 > MA200"""
    return (close > ma50) and (close > ma150) and (close > ma200)


def check_ma_alignment(ma50, ma150, ma200):
    """MA50 > MA150 > MA200 (정배열 상태)"""
    return (ma50 > ma150) and (ma150 > ma200)


def check_200ma_up(ma200_series):
    """200일선 상승 추세 확인 (20일 전 대비)"""
    lb = MINERVINI_CONFIG["LOOKBACK_200MA"]
    return (
        ma200_series.iloc[-1] > ma200_series.iloc[-(lb + 1)]
        if len(ma200_series) > lb
        else False
    )


def check_50ma_up(ma50_series):
    """50일선 상승 추세 확인 (5일 전 대비)"""
    lb = MINERVINI_CONFIG["LOOKBACK_50MA"]
    return (
        ma50_series.iloc[-1] > ma50_series.iloc[-(lb + 1)]
        if len(ma50_series) > lb
        else False
    )


def check_near_6m_high(close_series, max_distance, window):
    """현재가가 최근 6개월 고점 대비 max_distance 이내 위치"""
    window = min(window, len(close_series))
    if window <= 0:
        return False
    high_6m = close_series.iloc[-window:].max()
    return (
        ((high_6m - close_series.iloc[-1]) / high_6m) <= max_distance
        if high_6m > 0
        else False
    )


def check_150ma_up(ma150_series):
    """150일선 상승 추세 확인 (10일 전 대비)"""
    lb = MINERVINI_CONFIG["LOOKBACK_150MA"]
    return (
        ma150_series.iloc[-1] > ma150_series.iloc[-(lb + 1)]
        if len(ma150_series) > lb
        else False
    )


def check_volatility_contraction(close_series, atr_series):
    """VCP 패턴: 최근 ATR 평균과 최대값이 이전 기간보다 수축"""
    win = MINERVINI_CONFIG["VCP_WINDOW"]
    if len(atr_series) < win * 2:
        return False

    recent = atr_series.iloc[-win:]
    prev = atr_series.iloc[-(win * 2) : -win]

    return (recent.mean() < prev.mean()) and (recent.max() < prev.max())


def check_value_contraction(volume_series, close_series):
    """거래대금 수축: 최근 평균 거래대금이 이전 기간보다 충분히 감소"""
    win = MINERVINI_CONFIG["VCP_WINDOW"]

    if len(volume_series) < win * 2:
        return False

    value = volume_series * close_series

    recent_value = value.iloc[-win:].mean()
    prev_value = value.iloc[-(win * 2) : -win].mean()

    # 1) 거래대금 수축 강도
    if recent_value > prev_value * 0.7:
        return False

    # 2) 가격 유지 조건
    recent_price = close_series.iloc[-win:]
    price_range = recent_price.max() - recent_price.min()

    if price_range / recent_price.mean() > 0.15:
        return False

    return True


# ---------------------------------------------------------
# 통합 실행 엔진
# ---------------------------------------------------------


def evaluate_minervini_from_df(
    df: pd.DataFrame | None, date: pd.Timestamp
) -> dict[str, object]:
    result: dict[str, object] = {
        "pass": False,
        "reason": "unknown",
        "price_data_ok": False,
        "price_above_mas": False,
        "ma_alignment": False,
        "ma200_up": False,
        "ma50_up": False,
        "near_6m_high": False,
        "ma150_up": False,
        "volatility_contraction": False,
        "value_contraction": False,
        "contraction_ok": False,
    }

    if df is None or df.empty:
        result["reason"] = "no_price_data"
        return result

    if date not in df.index:
        result["reason"] = "missing_signal_date"
        return result

    result["price_data_ok"] = True
    df_slice = df.loc[:date]
    if df_slice.empty:
        result["reason"] = "empty_slice"
        return result

    if "MA200" not in df_slice.columns or pd.isna(df_slice["MA200"].iloc[-1]):
        result["reason"] = "missing_ma200"
        return result

    c, v = df_slice["Close"], df_slice["Volume"]
    m50, m150, m200 = (
        df_slice["MA50"],
        df_slice["MA150"],
        df_slice["MA200"],
    )
    atr = df_slice.get("ATR_14")

    result["price_above_mas"] = check_price_above_mas(
        c.iloc[-1], m50.iloc[-1], m150.iloc[-1], m200.iloc[-1]
    )
    result["ma_alignment"] = check_ma_alignment(
        m50.iloc[-1], m150.iloc[-1], m200.iloc[-1]
    )
    result["ma200_up"] = check_200ma_up(m200)
    result["ma50_up"] = check_50ma_up(m50)
    result["near_6m_high"] = check_near_6m_high(
        c,
        MINERVINI_CONFIG["MAX_DIST_6M_HIGH"],
        MINERVINI_CONFIG["SIX_MONTH_WINDOW"],
    )
    result["ma150_up"] = check_150ma_up(m150)

    vol_ok = False
    if atr is not None:
        vol_ok = check_volatility_contraction(c, atr)
    val_ok = check_value_contraction(v, c)
    result["volatility_contraction"] = vol_ok
    result["value_contraction"] = val_ok
    result["contraction_ok"] = bool(vol_ok or val_ok)

    checks = [
        "price_above_mas",
        "ma_alignment",
        "ma200_up",
        "ma50_up",
        "near_6m_high",
        "ma150_up",
        "contraction_ok",
    ]
    failed = [key for key in checks if not bool(result[key])]

    if not failed:
        result["pass"] = True
        result["reason"] = "pass"
    else:
        result["reason"] = "|".join(failed)

    return result


def check_minervini_from_df(df: pd.DataFrame | None, date: pd.Timestamp) -> bool:
    """메모리에 로드된 데이터프레임을 대상으로 모든 미너비니 필터 조건 검증"""
    detail = evaluate_minervini_from_df(df, date)
    return bool(detail.get("pass", False))
