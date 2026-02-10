import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# 경고 무시 (백테스트 반복 시 발생하는 불필요한 메시지 제거)
warnings.filterwarnings('ignore')

# =========================================================
# 1. 환경 설정 및 데이터 로드 (전역 변수로 한 번만 로드)
# =========================================================
# 기존 Asset_Preprocessing_v3.py의 로직을 그대로 활용
BASE_DIR = Path(".").resolve()
UNIVERSE_PATH = BASE_DIR / "data" / "universe" / "Pivot_Signals.csv"
PARQUET_DIR = BASE_DIR / "data"

def load_price_data(code, data_dir):
    try:
        code_str = str(code).zfill(6).strip()
        file_path = data_dir / f"{code_str}.parquet"
        if not file_path.exists(): return None
        df = pd.read_parquet(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
            df.set_index('Date', inplace=True)
        return df[~df.index.duplicated(keep='first')].sort_index()
    except: return None

def preload_data(signal_df):
    cache = {}
    last_valid_dates = {} 
    unique_codes = signal_df['Code'].unique()
    for code in unique_codes:
        df = load_price_data(code, PARQUET_DIR)
        if df is not None and not df.empty:
            valid_df = df[df['Close'] > 0]
            if not valid_df.empty:
                cache[code] = df
                last_valid_dates[code] = valid_df.index[-1]
    return cache, last_valid_dates

# 데이터 사전 로드 (Optuna 실행 전 1회 수행)
print("[System] Pre-loading data for optimization...")
try:
    signals_raw = pd.read_csv(UNIVERSE_PATH)
    signals_raw['Code'] = signals_raw['Code'].astype(str).str.zfill(6).str.strip()
    signals_raw['Date'] = pd.to_datetime(signals_raw['Date']).dt.normalize()
    DATA_CACHE, TICKER_LAST_DATES = preload_data(signals_raw)
    VALID_MARKET_DATES = sorted(list(set().union(*(df.index for df in DATA_CACHE.values()))))
    NEXT_DAY_MAP = {VALID_MARKET_DATES[i]: VALID_MARKET_DATES[i+1] for i in range(len(VALID_MARKET_DATES)-1)}
except Exception as e:
    print(f"[Error] 데이터 로드 실패: {e}")
    # 실제 환경에서 파일이 없을 경우 예외 처리가 필요합니다.

# =========================================================
# 2. Optuna Objective 함수 (최적화 대상)
# =========================================================
def objective(trial):
    # [A] 하이퍼파라미터 제안 (탐색 범위 설정)
    params = {
        'STOP_LOSS_PCT': trial.suggest_float('STOP_LOSS_PCT', 0.03, 0.07, step=0.01),
        'TRAIL_TRIGGER_PCT': trial.suggest_float('TRAIL_TRIGGER_PCT', 0.26, 0.30, step=0.01), # 상향 조정
        'TRAIL_STOP_PCT': trial.suggest_float('TRAIL_STOP_PCT', 0.00, 0.03, step=0.01),     # 상향 조정
        'TIME_STOP_DAYS': trial.suggest_float('TIME_STOP_DAYS', 11, 15, step=1),                        # 범위 최적화
        
        # 신규 최적화 변수 추가
        'BREAK_EVEN_TRIGGER': trial.suggest_float('BREAK_EVEN_TRIGGER', 0.10, 0.14, step=0.01),
        'BREAK_EVEN_BUFFER': trial.suggest_float('BREAK_EVEN_BUFFER', 0.012, 0.16, step=0.001),
        'TIME_STOP_ROI': 0.02,
        
        # 고정 환경 설정
        'INITIAL_CASH': 100_000_000,
        'MAX_POSITIONS': 3,
        'POS_WEIGHT': 0.33,
        'FEE': 0.003,
        'START_DATE': '2016-01-01',
        'END_DATE': '2025-8-29'
    }

    # [B] 백테스트 실행 (run_backtest 핵심 로직 요약)
    signals = signals_raw.copy()
    
    # 피벗 계산
    pivot_data = []
    for idx, row in signals.iterrows():
        df = DATA_CACHE.get(row['Code'])
        if df is not None and row['Date'] in df.index:
            hist = df.loc[:row['Date']].tail(10)
            if len(hist) >= 10:
                row['Pivot_Price'] = hist['High'].max()
                pivot_data.append(row)
    signals = pd.DataFrame(pivot_data)
    signals['Date'] = signals['Date'].map(NEXT_DAY_MAP)
    signals = signals.dropna(subset=['Date'])

    cash = params['INITIAL_CASH']
    portfolio = {}
    history = []
    all_dates = pd.date_range(start=params['START_DATE'], end=params['END_DATE'], freq='B')

    for today in all_dates:
        today_ts = pd.Timestamp(today).normalize()
        if today_ts not in VALID_MARKET_DATES: continue
        
        sold_codes = []
        # 매도 로직
        for code in list(portfolio.keys()):
            pos = portfolio[code]
            df = DATA_CACHE.get(code)
            curr_data = df.loc[today_ts] if today_ts in df.index else None
            
            if (today_ts == TICKER_LAST_DATES.get(code)) or curr_data is None or curr_data['Close'] <= 0 or curr_data['Open'] <= 0:
                exit_price = max(pos['last_close'], pos['stop_price'])
                cash += pos['shares'] * exit_price * (1 - params['FEE'])
                sold_codes.append(code); del portfolio[code]; continue

            curr_open, curr_high, curr_low, curr_close = curr_data['Open'], curr_data['High'], curr_data['Low'], curr_data['Close']
            pos['last_close'] = curr_close
            p_rate = (pos['highest_price'] / pos['entry_price'] - 1)
            
            if p_rate >= params['BREAK_EVEN_TRIGGER']:
                pos['stop_price'] = max(pos['stop_price'], pos['entry_price'] * (1 + params['BREAK_EVEN_BUFFER']))
            if p_rate >= params['TRAIL_TRIGGER_PCT']:
                pos['stop_price'] = max(pos['stop_price'], pos['highest_price'] * (1 - params['TRAIL_STOP_PCT']))
            
            exit_reason = None
            if curr_low <= pos['stop_price']:
                exit_price = min(pos['stop_price'], curr_open)
                exit_reason = "SL/TS"
            elif (today_ts - pos['entry_date']).days >= params['TIME_STOP_DAYS']:
                if (curr_close / pos['entry_price'] - 1) <= params['TIME_STOP_ROI']:
                    exit_price, exit_reason = curr_close, "Time"
            
            if curr_high > pos['highest_price']: pos['highest_price'] = curr_high
            if exit_reason:
                cash += pos['shares'] * exit_price * (1 - params['FEE'])
                sold_codes.append(code); del portfolio[code]

        # 매수 로직
        daily_signals = signals[signals['Date'] == today_ts]
        for _, row in daily_signals.iterrows():
            if len(portfolio) >= params['MAX_POSITIONS']: break
            code, pivot = row['Code'], row['Pivot_Price']
            if code in portfolio or code in sold_codes: continue
            
            day_data = DATA_CACHE.get(code).loc[today_ts]
            if day_data['Open'] >= pivot:
                entry_price = day_data['Open']
                current_equity = cash + sum(p['shares'] * p['last_close'] for p in portfolio.values())
                invest_amt = min(current_equity * params['POS_WEIGHT'], cash)
                shares = int(invest_amt / entry_price)
                if shares > 0:
                    cost = shares * entry_price * (1 + params['FEE'])
                    if cost <= cash:
                        cash -= cost
                        portfolio[code] = {
                            'entry_date': today_ts, 'entry_price': entry_price, 'shares': shares, 
                            'highest_price': entry_price, 'stop_price': entry_price * (1 - params['STOP_LOSS_PCT']), 
                            'last_close': day_data['Close']
                        }
        
        curr_equity = cash + sum(info['shares'] * info['last_close'] for info in portfolio.values())
        history.append(curr_equity)

    # [C] 성과 지표 계산
    res = pd.Series(history)
    final_return = (res.iloc[-1] / params['INITIAL_CASH']) - 1
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    cagr = ((1 + final_return) ** (1 / years)) - 1
    
    peak = res.cummax()
    dd = (res - peak) / peak
    mdd = abs(dd.min())

    # [D] 목표 함수 스코어링 (MDD 30% 제약 조건 적용)
    # MDD가 30%를 초과하면 큰 패널티를 부여하여 CAGR이 아무리 높아도 선택되지 않게 함
    if mdd > 0.30:
        score = cagr - (mdd - 0.30) * 10  # 패널티 부여
    else:
        score = cagr  # MDD 30% 이하면 CAGR 그대로 반환

    return score

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=300) # 여기서 횟수 조절 가능

    # [추가] 모든 최적화 결과 기록을 데이터프레임으로 변환
    df_results = study.trials_dataframe()
    
    # [추가] CSV 파일로 저장 (한글 깨짐 방지를 위해 utf-8-sig 사용)
    df_results.to_csv("optuna_optimization_results.csv", index=False, encoding='utf-8-sig')

    print("\n" + "="*40)
    print("Optimization Complete! Results saved to 'optuna_optimization_results.csv'")
    print(f"Best Score (CAGR): {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*40)