import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# [MINERVINI_CONFIG] 기술적 지표 및 임계값 설정
MINERVINI_CONFIG = {
    'MA_SHORT': 20,            # 단기 이동평균선
    'MA_MEDIUM': 50,           # 중기 이동평균선
    'MA_LONG_15': 150,         # 장기 이동평균선 1
    'MA_LONG_20': 200,         # 장기 이동평균선 2
    
    'LOOKBACK_200MA': 20,      # 200일선 추세 확인 기간
    'LOOKBACK_150MA': 10,      # 150일선 추세 확인 기간
    'LOOKBACK_50MA': 5,        # 5일선 추세 확인 기간
    
    'MAX_DIST_52W_HIGH': 0.1,  # 52주 신고가 대비 최대 이격 (10% 이내)
    'MIN_DIST_52W_LOW': 0.30,  # 52주 신저가 대비 최소 상승폭 (30% 이상)
    
    'VOL_LOOKBACK': 10,        # 거래량 돌파 확인 기간
    'VCP_WINDOW': 20,          # 변동성/거래량 수축 확인 윈도우
    'ATR_WINDOW': 14,          # ATR 계산 기간
}

def load_from_parquet(code, data_dir='./data/'):
    """Parquet 파일 로드 및 기술적 지표(MA, TR, ATR) 계산"""
    file_path = f"{data_dir}/{code}.parquet"
    try:
        df = pd.read_parquet(file_path)
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)

        # 이동평균선 계산
        for ma in [MINERVINI_CONFIG['MA_SHORT'], MINERVINI_CONFIG['MA_MEDIUM'], 
                   MINERVINI_CONFIG['MA_LONG_15'], MINERVINI_CONFIG['MA_LONG_20']]:
            col = f'MA{ma}'
            if col not in df.columns:
                df[col] = df['Close'].rolling(window=ma).mean()

        # ATR 계산
        if 'ATR_14' not in df.columns:
            tr = pd.concat([
                df['High'] - df['Low'],
                (df['High'] - df['Close'].shift(1)).abs(),
                (df['Low'] - df['Close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            df['ATR_14'] = tr.rolling(window=MINERVINI_CONFIG['ATR_WINDOW']).mean()
        
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
    lb = MINERVINI_CONFIG['LOOKBACK_200MA']
    return ma200_series.iloc[-1] > ma200_series.iloc[-(lb + 1)] if len(ma200_series) > lb else False

def check_50ma_up(ma50_series):
    """50일선 상승 추세 확인 (5일 전 대비)"""
    lb = MINERVINI_CONFIG['LOOKBACK_50MA']
    return ma50_series.iloc[-1] > ma50_series.iloc[-(lb + 1)] if len(ma50_series) > lb else False

def check_within_52w_high(close_series, max_distance):
    """현재가가 52주 최고가 대비 max_distance 이내 위치"""
    window = min(252, len(close_series))
    high_52w = close_series.iloc[-window:].max()
    return ((high_52w - close_series.iloc[-1]) / high_52w) <= max_distance if high_52w > 0 else False

def check_above_52w_low(close_series, min_distance):
    """현재가가 52주 최저가 대비 min_distance 이상 상승"""
    window = min(252, len(close_series))
    low_52w = close_series.iloc[-window:].min()
    return ((close_series.iloc[-1] - low_52w) / low_52w) >= min_distance if low_52w > 0 else False

def check_price_above_10ma(close, ma20):
    """현재가 > MA20 (단기 추세 확인)"""
    return close > ma20

def check_150ma_up(ma150_series):
    """150일선 상승 추세 확인 (10일 전 대비)"""
    lb = MINERVINI_CONFIG['LOOKBACK_150MA']
    return ma150_series.iloc[-1] > ma150_series.iloc[-(lb + 1)] if len(ma150_series) > lb else False

def check_sufficient_volume(volume_series):
    """당일 거래량이 최근 10일 중 최대치 (거래량 돌파)"""
    win = MINERVINI_CONFIG['VOL_LOOKBACK']
    return volume_series.iloc[-1] == volume_series.iloc[-win:].max() if len(volume_series) >= win else False

def check_volatility_contraction(close_series, atr_series=None):
    """VCP 패턴: 최근 변동성(ATR 또는 표준편차)이 이전 기간보다 수축"""
    win = MINERVINI_CONFIG['VCP_WINDOW']
    if len(close_series) < win * 2: return False
    if atr_series is not None:
        return atr_series.iloc[-win:].mean() < atr_series.iloc[-(win*2):-win].mean()
    return close_series.iloc[-win:].std() < close_series.iloc[-(win*2):-win].std()

def check_volume_contraction(volume_series):
    """거래량 수축: 최근 평균 거래량이 이전 기간보다 감소"""
    win = MINERVINI_CONFIG['VCP_WINDOW']
    return volume_series.iloc[-win:].mean() < volume_series.iloc[-(win*2):-win].mean() if len(volume_series) >= win * 2 else False

# ---------------------------------------------------------
# 통합 실행 엔진
# ---------------------------------------------------------

def check_minervini_from_df(df, date):
    """메모리에 로드된 데이터프레임을 대상으로 모든 미너비니 필터 조건 검증"""
    if df is None or df.empty or date not in df.index:
        return False
        
    df_slice = df.loc[:date]
    if df_slice.empty or pd.isna(df_slice['MA200'].iloc[-1]):
        return False
    
    c, v = df_slice['Close'], df_slice['Volume']
    m20, m50, m150, m200 = df_slice['MA20'], df_slice['MA50'], df_slice['MA150'], df_slice['MA200']
    atr = df_slice.get('ATR_14')

    # 필터 조건 집합
    filters = [
        check_price_above_mas(c.iloc[-1], m50.iloc[-1], m150.iloc[-1], m200.iloc[-1]),
        check_ma_alignment(m50.iloc[-1], m150.iloc[-1], m200.iloc[-1]),
        check_200ma_up(m200),
        check_50ma_up(m50),
        check_within_52w_high(c, MINERVINI_CONFIG['MAX_DIST_52W_HIGH']),
        check_above_52w_low(c, MINERVINI_CONFIG['MIN_DIST_52W_LOW']),
        check_price_above_10ma(c.iloc[-1], m20.iloc[-1]),
        check_150ma_up(m150),
        check_sufficient_volume(v),
        check_volatility_contraction(c, atr),
        check_volume_contraction(v)
    ]

    return all(filters)