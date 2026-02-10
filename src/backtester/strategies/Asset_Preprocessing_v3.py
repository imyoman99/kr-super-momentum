import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =========================================================
# 1. 전략 설정 (Configuration)
# =========================================================
CONFIG = {
    'START_DATE': '2023-11-01',     # 백테스트 시작일
    'END_DATE': '2025-12-31',       # 백테스트 종료일
    'INITIAL_CASH': 100_000_000,    # 초기 투자 자본금 (1억 원)
    'MAX_POSITIONS': 3,             # 동시 보유 최대 종목 수 (집중 투자 전략)
    'POS_WEIGHT': 0.33,             # 종목당 할당 비중 (자산의 약 33%)
    
    # [리스크 관리 및 수익 보존]
    'STOP_LOSS_PCT': 0.04,          # 하락 손절 라인 (-5%)
    'TRAIL_TRIGGER_PCT': 0.27,      # 트레일링 스탑 활성화 기준 수익률 (15% 수익 시 작동)
    'TRAIL_STOP_PCT': 0.01,         # 고점 대비 허용 하락폭 (고점 대비 -5% 시 익절)
    
    'BREAK_EVEN_TRIGGER': 0.14,     # 본전 보호 활성화 수익률 (5% 수익 시 손절가를 매수가 위로 상향)
    'BREAK_EVEN_BUFFER': 0.098,     # 본전 보호 시 최소 마진 (0.5%)

    # [매매 효율화]
    'TIME_STOP_DAYS': 14,           # 시간 제한 손절 (진입 후 10일간 보유)
    'TIME_STOP_ROI': 0.02,          # 10일 후에도 수익이 2% 미만이면 기회비용 차원에서 정리
    'FEE': 0.003,                   # 거래 비용 (수수료 및 세금 0.3% 적용)
}

# =========================================================
# 2. 데이터 처리 및 성과 분석 유틸리티
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path(".").resolve()

# 경로 설정: 유니버스(신호), 가격 데이터(Parquet), 결과 저장 경로
UNIVERSE_PATH = BASE_DIR / "data" / "universe" / "Pivot_Signals.csv"
PARQUET_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "src" / "backtester" / "strategies"

def load_price_data(code, data_dir):
    """개별 종목의 Parquet 파일을 로드하여 시계열 데이터프레임으로 변환"""
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
    """백테스트 속도 향상을 위해 신호가 발생한 종목들의 데이터를 메모리에 사전 로드"""
    cache = {}
    last_valid_dates = {} 
    unique_codes = signal_df['Code'].unique()
    print(f"[Info] Loading data for {len(unique_codes)} tickers...")
    for code in tqdm(unique_codes):
        df = load_price_data(code, PARQUET_DIR)
        if df is not None and not df.empty:
            valid_df = df[df['Close'] > 0]
            if not valid_df.empty:
                cache[code] = df
                last_valid_dates[code] = valid_df.index[-1] # 상장폐지 체크용 마지막 거래일 저장
    return cache, last_valid_dates

def save_results(history, trade_log, initial_cash):
    """백테스트 최종 성과 지표(CAGR, MDD, Win Rate, PF) 계산 및 리포팅"""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    res_df = pd.DataFrame(history)
    res_df.to_csv(OUTPUT_PATH / "Asset_List_Final.csv", index=False, encoding='utf-8-sig')
    
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        cols = ['Ticker', 'Profit_Pct', 'Net_PnL', 'Reason', 'Entry_Date', 'Exit_Date']
        existing_cols = [c for c in cols if c in log_df.columns]
        log_df = log_df[existing_cols]
        log_df.to_csv(OUTPUT_PATH / "Trade_Log.csv", index=False, encoding='utf-8-sig')
        
        # [수익성 지표 계산]
        wins = log_df[log_df['Net_PnL'] > 0]
        loses = log_df[log_df['Net_PnL'] <= 0]
        
        win_rate = len(wins) / len(log_df) * 100 if not log_df.empty else 0
        avg_win = wins['Net_PnL'].mean() if not wins.empty else 0
        avg_loss = abs(loses['Net_PnL'].mean()) if not loses.empty else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # [리스크 지표 계산 - MDD]
        res_df['Peak'] = res_df['자산총액'].cummax()
        res_df['Drawdown'] = (res_df['자산총액'] - res_df['Peak']) / res_df['Peak']
        mdd = res_df['Drawdown'].min()

        final_asset = res_df.iloc[-1]['자산총액']
        total_return = (final_asset / initial_cash - 1) * 100

        print("\n" + "="*40)
        print(f"Final Asset  : {int(final_asset):,} KRW")
        print(f"Total Return : {total_return:.2f}%")
        print(f"MDD          : {mdd*100:.2f}%")
        print(f"Win Rate     : {win_rate:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print("="*40)
    else:
        print("\n[Info] No trades occurred.")

# =========================================================
# 3. 메인 백테스트 엔진 (Main Engine)
# =========================================================
def run_backtest():
    if not UNIVERSE_PATH.exists(): 
        print("[Error] Signal file not found.")
        return
    
    # 신호 데이터 로드 및 형식 최적화
    signals = pd.read_csv(UNIVERSE_PATH)
    signals['Code'] = signals['Code'].astype(str).str.zfill(6).str.strip()
    signals['Date'] = pd.to_datetime(signals['Date']).dt.normalize()
    
    data_cache, ticker_last_dates = preload_data(signals)
    
    # 1. [피벗 포인트 계산] 신호 발생 시점 기준 '과거 10일 최고가'를 돌파 기준으로 설정
    print("[Info] Calculating Pivot Points...")
    pivot_data = []
    for idx, row in tqdm(signals.iterrows(), total=len(signals)):
        df = data_cache.get(row['Code'])
        if df is not None and row['Date'] in df.index:
            hist = df.loc[:row['Date']].tail(10) # 신호일까지의 10봉 데이터
            if len(hist) >= 10:
                row['Pivot_Price'] = hist['High'].max() # 10일 신고가를 피벗으로 설정
                pivot_data.append(row)
    signals = pd.DataFrame(pivot_data)
    
    # 2. [미래 참조 오류 방지] 신호 발생일(D)을 확인하고 실제 매매는 익일(D+1) 시가에 집행
    valid_market_dates = sorted(list(set().union(*(df.index for df in data_cache.values()))))
    next_day_map = {valid_market_dates[i]: valid_market_dates[i+1] for i in range(len(valid_market_dates)-1)}
    signals['Date'] = signals['Date'].map(next_day_map)
    signals = signals.dropna(subset=['Date'])
    
    # 변수 초기화
    cash = CONFIG['INITIAL_CASH']
    portfolio, history, trade_log = {}, [], []
    all_dates = pd.date_range(start=CONFIG['START_DATE'], end=CONFIG['END_DATE'], freq='B')
    
    print(f"[Info] Simulating from {CONFIG['START_DATE']} to {CONFIG['END_DATE']}...")
    
    # 일자별 시뮬레이션 메인 루프
    for today in tqdm(all_dates):
        today_ts = pd.Timestamp(today).normalize()
        if today_ts not in valid_market_dates: continue
            
        sold_codes = [] # 당일 매도한 종목은 당일 재매수 금지
        
        # ---------------------------------------------------------
        # [A] 매도 프로세스 (Exit Management)
        # ---------------------------------------------------------
        for code in list(portfolio.keys()):
            pos = portfolio[code]
            df = data_cache.get(code)
            is_last_day = (today_ts == ticker_last_dates.get(code)) # 상장폐지일 여부
            curr_data = df.loc[today_ts] if today_ts in df.index else None
            
            # [A-1] 비정상적 데이터 혹은 상장폐지 직전일 경우 강제 청산
            if is_last_day or curr_data is None or curr_data['Close'] <= 0:
                exit_price = max(pos['last_close'], pos['stop_price']) 
                revenue = pos['shares'] * exit_price * (1 - CONFIG['FEE'])
                cash += revenue
                trade_log.append({
                    'Ticker': code, 'Profit_Pct': round((exit_price/pos['entry_price']-1)*100, 2), 
                    'Net_PnL': int(revenue - (pos['shares']*pos['entry_price'])), 
                    'Reason': 'Safe_Exit', 'Entry_Date': pos['entry_date'].date(), 'Exit_Date': today_ts.date()
                })
                sold_codes.append(code); del portfolio[code]; continue

            # 가격 정보 업데이트
            curr_open, curr_high, curr_low, curr_close = curr_data['Open'], curr_data['High'], curr_data['Low'], curr_data['Close']
            pos['last_close'] = curr_close
            
            p_rate = (pos['highest_price'] / pos['entry_price'] - 1)
            
            # [매도 조건 1] 본전 보호 (수익 발생 후 하락 시 원금 사수)
            if p_rate >= CONFIG['BREAK_EVEN_TRIGGER']:
                pos['stop_price'] = max(pos['stop_price'], pos['entry_price'] * (1 + CONFIG['BREAK_EVEN_BUFFER']))
            
            # [매도 조건 2] 트레일링 스탑 (수익 가속 시 익절가 상향)
            if p_rate >= CONFIG['TRAIL_TRIGGER_PCT']:
                pos['stop_price'] = max(pos['stop_price'], pos['highest_price'] * (1 - CONFIG['TRAIL_STOP_PCT']))
            
            exit_reason = None
            # [판정] 손절가 터치 시 매도
            if curr_low <= pos['stop_price']:
                exit_price = pos['stop_price']
                exit_reason = "Trailing_Win" if exit_price > pos['entry_price'] else "Stop_Loss"
            
            # [매도 조건 3] 타임 스탑 (지정 기간 내 수익 부진 시 정리)
            elif (today_ts - pos['entry_date']).days >= CONFIG['TIME_STOP_DAYS']:
                if (curr_close / pos['entry_price'] - 1) <= CONFIG['TIME_STOP_ROI']:
                    exit_price, exit_reason = curr_close, "Time_Stop"
            
            if curr_high > pos['highest_price']: pos['highest_price'] = curr_high
            if exit_reason:
                revenue = pos['shares'] * exit_price * (1 - CONFIG['FEE'])
                cash += revenue
                trade_log.append({
                    'Ticker': code, 'Profit_Pct': round((exit_price/pos['entry_price']-1)*100, 2), 
                    'Net_PnL': int(revenue - (pos['shares']*pos['entry_price'])), 
                    'Reason': exit_reason, 'Entry_Date': pos['entry_date'].date(), 'Exit_Date': today_ts.date()
                })
                sold_codes.append(code); del portfolio[code]

        # ---------------------------------------------------------
        # [B] 신규 진입 프로세스 (Entry Management)
        # ---------------------------------------------------------
        daily_signals = signals[signals['Date'] == today_ts]
        for _, row in daily_signals.iterrows():
            if len(portfolio) >= CONFIG['MAX_POSITIONS']: break # 최대 보유 종목 수 제한
            
            code, pivot = row['Code'], row['Pivot_Price']
            if code in portfolio or code in sold_codes: continue # 이미 보유 중이거나 오늘 판 종목 제외
            
            df = data_cache.get(code)
            if df is None or today_ts not in df.index: continue
            
            day_data = df.loc[today_ts]
            if day_data['Volume'] <= 0 or day_data['Open'] <= 0: continue
            
            curr_open = day_data['Open']
            # [핵심 진입 조건] 당일 시가가 피벗(10일 고가) 위에서 시작할 때만 매수 (갭 상승 돌파)
            if curr_open >= pivot:
                entry_price = curr_open
                
                # 자본 관리에 따른 매수 수량 계산
                current_equity = cash + sum(p['shares'] * p['last_close'] for p in portfolio.values())
                invest_amt = min(current_equity * CONFIG['POS_WEIGHT'], cash)
                shares = int(invest_amt / entry_price)
                
                if shares > 0:
                    cost = shares * entry_price * (1 + CONFIG['FEE'])
                    if cost <= cash:
                        cash -= cost
                        # 포트폴리오 추가 및 초기 리스크 라인 설정
                        portfolio[code] = {
                            'entry_date': today_ts, 'entry_price': entry_price, 'shares': shares, 
                            'highest_price': entry_price, 
                            'stop_price': entry_price * (1 - CONFIG['STOP_LOSS_PCT']), 
                            'last_close': day_data['Close']
                        }

        # ---------------------------------------------------------
        # [C] 일일 성과 기록 (Daily Snapshot)
        # ---------------------------------------------------------
        curr_equity = cash + sum(info['shares'] * info['last_close'] for info in portfolio.values())
        record = {'DATE': today_ts.date(), '자산총액': int(curr_equity), '현금비율': round(cash/curr_equity*100, 2)}
        
        # 보유 종목 상세 정보 기록 (비중, 수익률 등)
        holdings = sorted([{'ticker': k, 'weight': (v['shares']*v['last_close']/curr_equity*100), 
                           'return': (v['last_close']/v['entry_price']-1)*100} for k, v in portfolio.items()], 
                          key=lambda x: x['weight'], reverse=True)
        
        for i in range(CONFIG['MAX_POSITIONS']):
            record[f'자산{i+1}_티커'] = holdings[i]['ticker'] if i < len(holdings) else None
            record[f'자산{i+1}_비율'] = round(holdings[i]['weight'], 2) if i < len(holdings) else 0.0
            record[f'자산{i+1}_수익률'] = round(holdings[i]['return'], 2) if i < len(holdings) else 0.0
        history.append(record)

    # 결과 데이터 저장 및 최종 리포트 출력
    save_results(history, trade_log, CONFIG['INITIAL_CASH'])

if __name__ == "__main__":
    run_backtest()