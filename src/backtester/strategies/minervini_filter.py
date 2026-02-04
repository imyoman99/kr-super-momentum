import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# =========================================================
# [MINERVINI_CONFIG] 필터 수치 설정
# =========================================================
MINERVINI_CONFIG = {
    # 1. 이동평균선(MA) 설정
    'MA_SHORT': 20,          # 단기 추세선 (원본 코드 MA10 대신 MA20 사용)
    'MA_MEDIUM': 50,         # 중기 추세선
    'MA_LONG_15': 150,       # 장기 추세선 1
    'MA_LONG_20': 200,       # 장기 추세선 2
    
    # 2. 이평선 추세 확인 기간 (Lookback)
    'LOOKBACK_200MA': 20,    # 200일선 상승 확인 (20일 전 데이터와 비교)
    'LOOKBACK_150MA': 10,    # 150일선 상승 확인 (10일 전 데이터와 비교)
    'LOOKBACK_50MA': 5,      # 5일선 상승 확인 (5일 전 데이터와 비교)
    
    # 3. 52주 고가/저가 기준
    'MAX_DIST_52W_HIGH': 0.1,  # 52주 최고가 대비 이격도 (25% 이내)0.25
    'MIN_DIST_52W_LOW': 0.30,   # 52주 최저가 대비 상승폭 (30% 이상)
    
    # 4. 거래량 및 VCP(변동성 수축) 설정
    'VOL_LOOKBACK': 10,      # 거래량 돌파 확인 기간
    'VCP_WINDOW': 10,        # VCP 수축 확인 윈도우 (최근 10일 vs 이전 10일)
    'ATR_WINDOW': 14,        # 변동성 지표(ATR) 계산 기간
}

def load_from_parquet(code, data_dir='./data/'):
    """
    특정 종목의 Parquet 데이터를 로드하고 필요한 이동평균선 및 ATR을 계산합니다.
    
    매개변수:
    - code: 종목 코드
    - data_dir: 데이터 파일 경로
    """
    file_path = f"{data_dir}{code}.parquet"
    try:
        df = pd.read_parquet(file_path)
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)

        # 설정된 주기에 맞춰 이동평균선(MA) 계산
        ma_list = [
            MINERVINI_CONFIG['MA_SHORT'], 
            MINERVINI_CONFIG['MA_MEDIUM'], 
            MINERVINI_CONFIG['MA_LONG_15'], 
            MINERVINI_CONFIG['MA_LONG_20']
        ]
        
        for ma in ma_list:
            col = f'MA{ma}'
            if col not in df.columns and 'Close' in df.columns:
                df[col] = df['Close'].rolling(window=ma).mean()

        # ATR(Average True Range) 계산: 변동성 수축 확인 지표
        if 'ATR_14' not in df.columns and all(k in df.columns for k in ['High', 'Low', 'Close']):
            df['TR'] = pd.concat([
                df['High'] - df['Low'],
                (df['High'] - df['Close'].shift(1)).abs(),
                (df['Low'] - df['Close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            df['ATR_14'] = df['TR'].rolling(window=MINERVINI_CONFIG['ATR_WINDOW']).mean()
        
        return df
    except Exception:
        return None

# ---------------------------------------------------------
# 개별 기술적 필터 함수 (Minervini's Trend Template)
# ---------------------------------------------------------

def check_price_above_mas(close, ma50, ma150, ma200):
    """현재가가 중/장기 이평선(50, 150, 200일선) 위에 있는지 확인"""
    try:
        return (close > ma50) and (close > ma150) and (close > ma200)
    except:
        return False

def check_ma_alignment(ma50, ma150, ma200):
    """이평선이 정배열(50 > 150 > 200) 상태인지 확인"""
    try:
        return (ma50 > ma150) and (ma150 > ma200)
    except:
        return False

def check_200ma_up(ma200_series):
    """200일 이동평균선이 최근 상승 추세인지 확인"""
    lookback = MINERVINI_CONFIG['LOOKBACK_200MA']
    try:
        if len(ma200_series) >= lookback + 1:
            return ma200_series.iloc[-1] > ma200_series.iloc[-(lookback + 1)]
        return False
    except:
        return False

def check_50ma_up(ma50_series):
    """50일 이동평균선이 최근 상승 추세인지 확인"""
    lookback = MINERVINI_CONFIG['LOOKBACK_50MA']
    try:
        if len(ma50_series) >= lookback + 1:
            return ma50_series.iloc[-1] > ma50_series.iloc[-(lookback + 1)]
        return False
    except:
        return False

def check_within_52w_high(close_series, max_distance):
    """현재가가 52주 최고가 부근(설정값 이내)인지 확인"""
    try:
        window = 252 if len(close_series) >= 252 else len(close_series)
        week_52_high = close_series.iloc[-window:].max()
        distance = (week_52_high - close_series.iloc[-1]) / week_52_high if week_52_high != 0 else 1.0
        return distance <= max_distance
    except:
        return False

def check_above_52w_low(close_series, min_distance):
    """현재가가 52주 최저가 대비 충분히(설정값 이상) 반등했는지 확인"""
    try:
        window = 252 if len(close_series) >= 252 else len(close_series)
        week_52_low = close_series.iloc[-window:].min()
        distance = (close_series.iloc[-1] - week_52_low) / week_52_low if week_52_low != 0 else 0.0
        return distance >= min_distance
    except:
        return False

def check_price_above_10ma(close, ma20):
    """현재가가 단기 추세선(20일선) 위에 있는지 확인"""
    try:
        return close > ma20
    except:
        return False

def check_150ma_up(ma150_series):
    """150일 이동평균선이 최근 상승 추세인지 확인"""
    lookback = MINERVINI_CONFIG['LOOKBACK_150MA']
    try:
        if len(ma150_series) >= lookback + 1:
            return ma150_series.iloc[-1] > ma150_series.iloc[-(lookback + 1)]
        return False
    except:
        return False

def check_sufficient_volume(volume_series):
    """오늘 거래량이 최근 10일 중 최대 거래량인지 확인 (돌파 여부)"""
    win = MINERVINI_CONFIG['VOL_LOOKBACK']
    try:
        if len(volume_series) >= win:
            return volume_series.iloc[-1] == volume_series.iloc[-win:].max()
        return False
    except:
        return False

def check_volatility_contraction(close_series, atr_series=None):
    """변동성 수축(VCP) 확인: 최근 ATR/변동성이 이전보다 낮아졌는지 확인"""
    win = MINERVINI_CONFIG['VCP_WINDOW']
    try:
        if atr_series is not None and len(atr_series) >= win * 2:
            return atr_series.iloc[-win:].mean() < atr_series.iloc[-(win*2):-win].mean()
        elif len(close_series) >= win * 2:
            return close_series.iloc[-win:].std() < close_series.iloc[-(win*2):-win].std()
        return False
    except:
        return False

def check_volume_contraction(volume_series):
    """거래량 수축 확인: 최근 거래량 평균이 이전보다 낮아졌는지 확인"""
    win = MINERVINI_CONFIG['VCP_WINDOW']
    try:
        if len(volume_series) >= win * 2:
            return volume_series.iloc[-win:].mean() < volume_series.iloc[-(win*2):-win].mean()
        return False
    except:
        return False

# ---------------------------------------------------------
# 통합 필터 실행 엔진
# ---------------------------------------------------------

def check_all_minervini_filters(code, date, data_dir='./data/'):
    """
    모든 미너비니 기술적 조건을 검사하여 매수 적격 여부를 반환합니다.
    """
    df = load_from_parquet(code, data_dir)
    if df is None or df.empty:
        return False

    # 분석 시점(date)까지의 데이터로 제한
    df_slice = df.loc[:date]
    if df_slice.empty or pd.isna(df_slice['MA200'].iloc[-1]):
        return False
    
    # 필요한 지표 시리즈 추출
    close, volume = df_slice['Close'], df_slice['Volume']
    ma20, ma50 = df_slice['MA20'], df_slice['MA50']
    ma150, ma200 = df_slice['MA150'], df_slice['MA200']
    atr = df_slice.get('ATR_14')

    # 모든 개별 조건 리스트
    filters = [
        check_price_above_mas(close.iloc[-1], ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1]),
        check_ma_alignment(ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1]),
        check_200ma_up(ma200),
        check_50ma_up(ma50),
        check_within_52w_high(close, MINERVINI_CONFIG['MAX_DIST_52W_HIGH']),
        check_above_52w_low(close, MINERVINI_CONFIG['MIN_DIST_52W_LOW']),
        check_price_above_10ma(close.iloc[-1], ma20.iloc[-1]),
        check_150ma_up(ma150),
        check_sufficient_volume(volume),
        check_volatility_contraction(close, atr),
        check_volume_contraction(volume)
    ]

    # 모든 조건이 충족(True)될 때만 True 반환
    return all(filters)