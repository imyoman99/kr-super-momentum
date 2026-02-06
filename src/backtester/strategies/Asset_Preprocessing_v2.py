import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from minervini_filter_v2 import check_minervini_from_df, load_from_parquet

# [CONFIG] 전략 및 리스크 관리 설정
CONFIG = {
    'INITIAL_CASH': 100_000_000,      # 초기 자본금
    'MAX_SLOTS': 4,                   # 최대 보유 종목 수
    'MAX_POS_WEIGHT': 0.25,           # 종목당 최대 투자 비중

    'HARD_STOP_PCT': 0.70,            # 하드 스탑 (구매가 대비 손실폭)
    'TS_ACTIVATION_PCT': 0.40,        # 트레일링 스탑 활성화 수익률
    'TRAILING_STOP_PCT': 0.20,        # 트레일링 스탑 폭
    
    'TIME_STOP_DAYS': 250,            # 시간 손절 기준일
    'TIME_STOP_MIN_RET': -0.30,       # 시간 손절 적용 최소 수익률

    'COST_RATE': 0.0040,              # 거래 비용 (세금 및 수수료)
    'RISK_PER_TRADE_PCT': 0.02,       # 총 자산 대비 1회 매매 리스크 (2%)
    'ATR_STOP_MULTIPLIER': 6,         # ATR 기반 스탑 거리 배수
    
    'START_DATE': "2016-01-01",
    'END_DATE': "2025-12-31",
}

BASE_DIR = Path(__file__).resolve().parents[3]
UNIVERSE_PATH = BASE_DIR / "data" / "universe" / "intersection_80.csv"
PARQUET_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "src" / "backtester" / "strategies"

def preload_data_to_ram(universe_df):
    """지정된 유니버스의 데이터를 메모리에 적재"""
    cache = {}
    unique_codes = universe_df['Code'].unique()
    data_dir_str = str(PARQUET_DIR)

    for code in tqdm(unique_codes, desc="Data Loading"):
        try:
            df = load_from_parquet(code, data_dir_str)
            if df is not None:
                if 'ATR_14' not in df.columns:
                    df['ATR_14'] = 0.0
                cache[code] = df
        except: continue
    return cache

def run_backtest():
    """메인 백테스팅 엔진 실행"""
    if not UNIVERSE_PATH.exists():
        print(f"Error: {UNIVERSE_PATH} not found.")
        return
    
    universe = pd.read_csv(UNIVERSE_PATH, dtype={'Code': str})
    universe['Date'] = pd.to_datetime(universe['Date'])
    universe = universe[(universe['Date'] >= CONFIG['START_DATE']) & (universe['Date'] <= CONFIG['END_DATE'])]
    all_dates = sorted(universe['Date'].unique())

    data_cache = preload_data_to_ram(universe)
    if not data_cache:
        print("Error: No data loaded.")
        return

    cash = float(CONFIG['INITIAL_CASH'])
    portfolio = {}      
    history = []
    trade_log = []
    
    for today in tqdm(all_dates, desc="Backtesting"):
        today_ts = pd.Timestamp(today)
        
        # 1. 포트폴리오 관리 (매도 로직)
        for ticker in list(portfolio.keys()):
            df = data_cache.get(ticker)
            
            # 데이터 누락 시 청산 처리
            if df is None or len(df) == 0 or today_ts not in df.index:
                info = portfolio[ticker]
                last_close = info['last_close'] if info['last_close'] != 0 else float(df.iloc[-1]['Close'])
                exit_val = float(info['shares']) * last_close
                pnl = exit_val - (float(info['entry_price']) * float(info['shares']))
                cash += exit_val * (1 - CONFIG['COST_RATE'])
                
                trade_log.append({
                    'Ticker': ticker, 'Profit_Pct': -99.9, 'Net_PnL': int(pnl),
                    'Reason': 'Data_Ended', 'Entry_Date': info['entry_date'], 'Exit_Date': today_ts.date()
                })
                del portfolio[ticker]
                continue

            curr_data = df.loc[today_ts]
            curr_open, curr_high, curr_low, curr_close = map(float, [curr_data['Open'], curr_data['High'], curr_data['Low'], curr_data['Close']])
            info = portfolio[ticker]
            
            sell_signal = False; sell_price = 0.0; sell_reason = ""

            # 매도 조건 체크: 상장폐지(가격 0), 갭하락, 스탑로스, 시간 손절
            if curr_open <= 0 or pd.isna(curr_open):
                sell_signal = True; sell_price = info['last_close']; sell_reason = "Delisting_ZeroPrice"
            elif curr_open < info['stop_price']:
                sell_signal = True; sell_price = curr_open; sell_reason = "Gap_Down"
            elif curr_low <= info['stop_price']:
                sell_signal = True; sell_price = info['stop_price']
                sell_reason = "Trailing_Profit" if sell_price > info['entry_price'] else "Stop_Loss"
            elif info['days_held'] >= CONFIG['TIME_STOP_DAYS']:
                if (curr_close - info['entry_price']) / info['entry_price'] < CONFIG['TIME_STOP_MIN_RET']:
                    sell_signal = True; sell_price = curr_close; sell_reason = "Time_Stop"

            if sell_signal:
                shares = float(info['shares'])
                exit_val = sell_price * shares
                pnl = exit_val - (info['entry_price'] * shares) - (exit_val * CONFIG['COST_RATE'])
                ret_pct = (sell_price - info['entry_price']) / info['entry_price'] * 100
                cash += (exit_val * (1 - CONFIG['COST_RATE']))
                
                trade_log.append({
                    'Ticker': ticker, 'Profit_Pct': round(ret_pct, 2), 'Net_PnL': int(pnl),
                    'Reason': sell_reason, 'Entry_Date': info['entry_date'], 'Exit_Date': today_ts.date()
                })
                del portfolio[ticker]
            else:
                # 홀딩 및 트레일링 스탑 업데이트
                info['days_held'] += 1
                info['last_close'] = curr_close
                if curr_high > info['highest_price']: info['highest_price'] = curr_high
                
                peak_ret = (info['highest_price'] - info['entry_price']) / info['entry_price']
                trailing_stop = info['highest_price'] * (1 - CONFIG['TRAILING_STOP_PCT']) if peak_ret >= CONFIG['TS_ACTIVATION_PCT'] else 0.0
                info['stop_price'] = max(info['stop_price'], info['fixed_hard_stop'], trailing_stop)

        # 2. 신규 진입 관리
        curr_equity = cash + sum(info['shares'] * info['last_close'] for info in portfolio.values())
        if len(portfolio) < CONFIG['MAX_SLOTS']:
            candidates = universe[universe['Date'] == today].sort_values('RS', ascending=False)
            
            for _, row in candidates.iterrows():
                if len(portfolio) >= CONFIG['MAX_SLOTS']: break
                ticker = row['Code']
                if ticker in portfolio: continue
                
                df = data_cache.get(ticker)
                if df is not None and check_minervini_from_df(df, today_ts):
                    signal_close = float(df.loc[today_ts, 'Close'])
                    atr = df.loc[today_ts].get('ATR_14', 0)
                    
                    if atr == 0 or pd.isna(atr) or signal_close < 500: continue

                    try:
                        curr_loc = df.index.get_loc(today_ts)
                        next_data = df.iloc[curr_loc + 1]
                        buy_price = float(next_data['Open'])
                        if (buy_price - signal_close) / signal_close <= -0.10: continue
                    except: continue

                    # 리스크 기반 포지션 사이징
                    real_stop_dist = max(min(atr * CONFIG['ATR_STOP_MULTIPLIER'], buy_price * CONFIG['HARD_STOP_PCT']), buy_price * 0.03)
                    shares = int(min(curr_equity * CONFIG['RISK_PER_TRADE_PCT'] // real_stop_dist, 
                                     curr_equity * CONFIG['MAX_POS_WEIGHT'] // buy_price, 
                                     cash // buy_price))
                    
                    if shares > 0:
                        cash -= (shares * buy_price)
                        initial_stop = buy_price - real_stop_dist
                        portfolio[ticker] = {
                            'shares': shares, 'entry_price': buy_price, 'entry_date': next_data.name,
                            'stop_price': initial_stop, 'fixed_hard_stop': initial_stop,
                            'highest_price': buy_price, 'last_close': buy_price, 'days_held': 0
                        }

        # 3. 일일 기록 저장
        record = {'DATE': today_ts.date(), '자산총액': int(curr_equity), '현금비율': (cash/curr_equity*100) if curr_equity > 0 else 0}
        p_items = list(portfolio.items())
        for i in range(CONFIG['MAX_SLOTS']):
            if i < len(p_items):
                t, info = p_items[i]
                record.update({f'자산{i+1}_티커': t, f'자산{i+1}_비율': (info['shares']*info['last_close']/curr_equity*100), 
                               f'자산{i+1}_수익률': round((info['last_close']/info['entry_price']-1)*100, 2)})
            else:
                record.update({f'자산{i+1}_티커': "None", f'자산{i+1}_비율': 0, f'자산{i+1}_수익률': 0.0})
        history.append(record)

    # 결과물 출력 및 저장
    save_results(history, trade_log, CONFIG['INITIAL_CASH'])

def save_results(history, trade_log, initial_cash):
    """결과 데이터프레임 저장 및 통계 출력"""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(history)
    res_df.to_csv(OUTPUT_PATH / "Asset_List_Final.csv", index=False, encoding='utf-8-sig')
    
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        log_df.to_csv(OUTPUT_PATH / "Trade_Log.csv", index=False, encoding='utf-8-sig')
        
        wins = log_df[log_df['Net_PnL'] > 0]
        loses = log_df[log_df['Net_PnL'] <= 0]
        win_rate = len(wins) / len(log_df) * 100
        avg_win = wins['Net_PnL'].mean() if not wins.empty else 0
        avg_loss = loses['Net_PnL'].mean() if not loses.empty else 1
        
        print("\n" + "="*40)
        print(f"Final Asset: {int(res_df.iloc[-1]['자산총액']):,} KRW")
        print(f"Total Return: {(res_df.iloc[-1]['자산총액']/initial_cash-1)*100:.2f}%")
        print(f"Win Rate: {win_rate:.2f}% / PF: {abs(avg_win/avg_loss):.2f}")
        print("="*40)

if __name__ == "__main__":
    run_backtest()