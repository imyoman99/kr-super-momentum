import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# [중요] minervini_filter_v2.py 파일에서 함수 임포트
# 파일이 같은 디렉토리에 있다고 가정합니다.
try:
    from minervini_filter_v2 import check_sufficient_volume
except ImportError:
    print("[Error] 'minervini_filter_v2.py' 파일을 찾을 수 없습니다.")
    print("같은 폴더에 필터 파일이 있는지 확인해주세요.")
    sys.exit()

# =========================================================
# 1. 설정 (Configuration) - 최적화된 값 유지
# =========================================================
CONFIG = {
    'START_DATE': '2016-01-01',     
    'END_DATE': '2025-12-31',       

    # [자금 관리]
    'INITIAL_CASH': 100_000_000,    
    'MAX_POSITIONS': 3,             
    'POS_WEIGHT': 0.33,             
    
    # [리스크 관리] 
    'STOP_LOSS_PCT': 0.05,          # -5% 손절
    'TRAIL_TRIGGER_PCT': 0.08,      # +5% 도달 시 트레일링 가동
    'TRAIL_STOP_PCT': 0.03,         # 고점 대비 -2% 익절
    
    # [본전 보존]
    'BREAK_EVEN_TRIGGER': 0.03,     # +3% 수익 시
    'BREAK_EVEN_BUFFER': 0.005,     # 본전+0.5%로 손절가 이동

    # [시간 제한] 
    'TIME_STOP_DAYS': 5,            
    'TIME_STOP_ROI': 0.01,          
    
    # [비용] 
    'FEE': 0.003,                   
}

# =========================================================
# 2. 경로 및 환경 설정
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path(".").resolve()

UNIVERSE_PATH = BASE_DIR / "data" / "universe" / "Pivot_Signals.csv"
PARQUET_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "src" / "backtester" / "strategies"

def load_price_data(code, data_dir):
    try:
        code_str = str(code).zfill(6).strip()
        file_path = data_dir / f"{code_str}.parquet"
        
        if not file_path.exists():
            return None
            
        df = pd.read_parquet(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
            df.set_index('Date', inplace=True)
        
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return None

def preload_data(signal_df):
    cache = {}
    unique_codes = signal_df['Code'].unique()
    print(f"[Info] Loading data for {len(unique_codes)} tickers...")
    
    for code in tqdm(unique_codes):
        df = load_price_data(code, PARQUET_DIR)
        if df is not None:
            cache[code] = df
    return cache

def save_results(history, trade_log, initial_cash):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    res_df = pd.DataFrame(history)
    res_df.to_csv(OUTPUT_PATH / "Asset_List_Final.csv", index=False, encoding='utf-8-sig')
    
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        cols = ['Ticker', 'Profit_Pct', 'Net_PnL', 'Reason', 'Entry_Date', 'Exit_Date']
        existing_cols = [c for c in cols if c in log_df.columns]
        log_df = log_df[existing_cols]
        log_df.to_csv(OUTPUT_PATH / "Trade_Log.csv", index=False, encoding='utf-8-sig')
        
        wins = log_df[log_df['Net_PnL'] > 0]
        loses = log_df[log_df['Net_PnL'] <= 0]
        
        win_rate = len(wins) / len(log_df) * 100 if not log_df.empty else 0
        avg_win = wins['Net_PnL'].mean() if not wins.empty else 0
        avg_loss = abs(loses['Net_PnL'].mean()) if not loses.empty else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        res_df['Peak'] = res_df['자산총액'].cummax()
        res_df['Drawdown'] = (res_df['자산총액'] - res_df['Peak']) / res_df['Peak']
        mdd = res_df['Drawdown'].min()

        final_asset = res_df.iloc[-1]['자산총액']
        total_return = (final_asset / initial_cash - 1) * 100

        print("\n" + "="*40)
        print(f"Final Asset : {int(final_asset):,} KRW")
        print(f"Total Return: {total_return:.2f}%")
        print(f"MDD         : {mdd*100:.2f}%")
        print(f"Win Rate    : {win_rate:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print("="*40)
    else:
        print("\n[Info] No trades occurred.")

def run_backtest():
    # --- [1] 시그널 로드 ---
    if not UNIVERSE_PATH.exists():
        print(f"[Error] Signal file not found: {UNIVERSE_PATH}")
        return

    signals = pd.read_csv(UNIVERSE_PATH)
    signals['Code'] = signals['Code'].astype(str).str.zfill(6).str.strip()
    signals['Date'] = pd.to_datetime(signals['Date']).dt.normalize()
    
    if CONFIG['START_DATE']:
        start_dt = pd.Timestamp(CONFIG['START_DATE']).normalize()
    else:
        start_dt = signals['Date'].min()

    if CONFIG['END_DATE']:
        end_dt = pd.Timestamp(CONFIG['END_DATE']).normalize()
    else:
        end_dt = signals['Date'].max() + pd.Timedelta(days=365)
    
    signals = signals[(signals['Date'] >= start_dt) & (signals['Date'] <= end_dt)]
    signals = signals.drop_duplicates(subset=['Date', 'Code'])
    
    # --- [2] 데이터 로딩 및 날짜 Shift ---
    data_cache = preload_data(signals)
    
    print("[Info] Calculating valid market dates and shifting signals...")
    valid_market_dates = set()
    for df in data_cache.values():
        valid_market_dates.update(df.index)
    
    # 시장 거래일 정렬
    sorted_dates = sorted(list(valid_market_dates))
    
    # {오늘 : 내일} 날짜 매핑 (오늘 뜬 신호 -> 내일 매수)
    next_day_map = {sorted_dates[i]: sorted_dates[i+1] for i in range(len(sorted_dates)-1)}
    
    # ★ 시그널 날짜를 '매수해야 할 날짜(D+1)'로 변경 ★
    signals['Date'] = signals['Date'].map(next_day_map)
    signals = signals.dropna(subset=['Date'])
    
    # --- [3] 시뮬레이션 ---
    cash = CONFIG['INITIAL_CASH']
    portfolio = {} 
    history = []
    trade_log = []
    
    all_dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
    
    print(f"[Info] Simulating trades from {start_dt.date()} to {end_dt.date()}...")
    
    for today in tqdm(all_dates):
        today_ts = pd.Timestamp(today).normalize()
        
        if today_ts not in valid_market_dates:
            continue
            
        sold_codes = [] 
        
        # ---------------------------------------------------------
        # [A] 보유 종목 관리 (매도 로직)
        # ---------------------------------------------------------
        for code in list(portfolio.keys()):
            pos = portfolio[code]
            df = data_cache.get(code)
            
            # 1. 상장 폐지/데이터 없음 체크
            if df is None or today_ts not in df.index:
                exit_reason = "Delisted"
                exit_price = pos['last_close']
                revenue = pos['shares'] * exit_price * (1 - CONFIG['FEE'])
                cash += revenue
                pnl_val = revenue - (pos['shares'] * pos['entry_price'])
                pnl_pct = (exit_price / pos['entry_price']) - 1
                trade_log.append({
                    'Ticker': code, 'Profit_Pct': round(pnl_pct * 100, 2),
                    'Net_PnL': int(pnl_val), 'Reason': exit_reason,
                    'Entry_Date': pos['entry_date'].date(), 'Exit_Date': today_ts.date()
                })
                sold_codes.append(code)
                del portfolio[code]
                continue

            # 2. 정상 데이터 조회
            day_data = df.loc[today_ts]
            curr_open = day_data['Open']
            curr_high = day_data['High']
            curr_low = day_data['Low']
            curr_close = day_data['Close']
            
            pos['last_close'] = curr_close
            if curr_open == 0 or curr_close == 0: continue

            # 3. 고점 갱신 및 트레일링/본전 로직
            if curr_high > pos['highest_price']:
                pos['highest_price'] = curr_high
            
            profit_rate_high = (pos['highest_price'] - pos['entry_price']) / pos['entry_price']

            # (A) 본전 보존 (+3% 시 손절가 상향)
            if profit_rate_high >= CONFIG['BREAK_EVEN_TRIGGER']:
                break_even_price = pos['entry_price'] * (1 + CONFIG['BREAK_EVEN_BUFFER'])
                pos['stop_price'] = max(pos['stop_price'], break_even_price)

            # (B) 트레일링 스탑 (+5% 시 익절선 추격)
            if profit_rate_high >= CONFIG['TRAIL_TRIGGER_PCT']:
                new_trail_stop = pos['highest_price'] * (1 - CONFIG['TRAIL_STOP_PCT'])
                pos['stop_price'] = max(pos['stop_price'], new_trail_stop)
            
            # 4. 매도 실행
            exit_reason = None
            exit_price = curr_close
            
            # (1) 손절/익절
            if curr_low <= pos['stop_price']:
                exit_price = curr_open if curr_open < pos['stop_price'] else pos['stop_price']
                if exit_price > pos['entry_price']:
                    exit_reason = "Trailing_Win"
                else:
                    exit_reason = "Stop_Loss"
            
            # (2) 시간 제한
            elif (today_ts - pos['entry_date']).days >= CONFIG['TIME_STOP_DAYS']:
                current_pnl = (curr_close - pos['entry_price']) / pos['entry_price']
                if current_pnl <= CONFIG['TIME_STOP_ROI']:
                    exit_reason = "Time_Stop"
                    exit_price = curr_close
            
            if exit_reason:
                revenue = pos['shares'] * exit_price * (1 - CONFIG['FEE'])
                cash += revenue
                pnl_val = revenue - (pos['shares'] * pos['entry_price'])
                pnl_pct = (exit_price / pos['entry_price']) - 1
                trade_log.append({
                    'Ticker': code, 'Profit_Pct': round(pnl_pct * 100, 2),
                    'Net_PnL': int(pnl_val), 'Reason': exit_reason,
                    'Entry_Date': pos['entry_date'].date(), 'Exit_Date': today_ts.date()
                })
                sold_codes.append(code)
                del portfolio[code]

        # ---------------------------------------------------------
        # [B] 신규 진입 (매수 로직)
        # ---------------------------------------------------------
        # 1. 시그널은 이미 '오늘(매수일)' 날짜로 Shift 되어 있음
        daily_signals = signals[signals['Date'] == today_ts]
        
        for _, row in daily_signals.iterrows():
            if len(portfolio) >= CONFIG['MAX_POSITIONS']: break
            
            code = row['Code']
            if code in portfolio or code in sold_codes: continue
            
            df = data_cache.get(code)
            if df is None or today_ts not in df.index: continue
            
            # =========================================================
            # [핵심] 거래량 폭발 확인 (Look-ahead Check)
            # =========================================================
            # 오늘(매수일)의 거래량이 최근 10일 중 최고인지 확인
            hist_data = df.loc[:today_ts] # 오늘 데이터까지 포함
            volume_series = hist_data['Volume']
            
            # 외부 함수(minervini_filter_v2) 사용
            # check_sufficient_volume 함수는 True/False를 반환함
            if not check_sufficient_volume(volume_series):
                continue # 거래량 조건 만족 안 하면 매수 패스 (관망)
            
            # =========================================================
            # [진입] 시가 매수
            # =========================================================
            curr_open = df.loc[today_ts]['Open'] 
            curr_close = df.loc[today_ts]['Close']
            
            if curr_open <= 0: continue
            
            current_equity_for_calc = cash
            for p_val in portfolio.values():
                current_equity_for_calc += p_val['shares'] * p_val['last_close']
            
            target_amt = current_equity_for_calc * CONFIG['POS_WEIGHT']
            invest_amt = min(target_amt, cash)
            
            shares = int(invest_amt / curr_open) # 시가 기준

            if shares > 0:
                cost = shares * curr_open * (1 + CONFIG['FEE'])
                if cost > cash: 
                    shares = int(cash / (curr_open * (1 + CONFIG['FEE'])))
                    cost = shares * curr_open * (1 + CONFIG['FEE'])
                
                if shares > 0:
                    cash -= cost
                    portfolio[code] = {
                        'entry_date': today_ts,
                        'entry_price': curr_open,
                        'shares': shares,
                        'highest_price': curr_open,
                        'stop_price': curr_open * (1 - CONFIG['STOP_LOSS_PCT']),
                        'last_close': curr_close
                    }

        # ---------------------------------------------------------
        # [C] 일별 자산 평가 ... (동일)
        # ---------------------------------------------------------
        curr_equity = cash
        for info in portfolio.values():
            curr_equity += info['shares'] * info['last_close']
            
        cash_ratio = (cash / curr_equity * 100) if curr_equity > 0 else 0
        
        record = {
            'DATE': today_ts.date(), 
            '자산총액': int(curr_equity), 
            '현금비율': round(cash_ratio, 2)
        }

        holdings = []
        for ticker, info in portfolio.items():
            curr_val = info['shares'] * info['last_close']
            weight = (curr_val / curr_equity * 100) if curr_equity > 0 else 0
            ret = (info['last_close'] - info['entry_price']) / info['entry_price'] * 100
            holdings.append({'ticker': ticker, 'weight': weight, 'return': ret})
        
        holdings.sort(key=lambda x: x['weight'], reverse=True)

        for i in range(CONFIG['MAX_POSITIONS']):
            key_ticker = f'자산{i+1}_티커'
            key_weight = f'자산{i+1}_비율'
            key_return = f'자산{i+1}_수익률'
            
            if i < len(holdings):
                record[key_ticker] = holdings[i]['ticker']
                record[key_weight] = round(holdings[i]['weight'], 2)
                record[key_return] = round(holdings[i]['return'], 2)
            else:
                record[key_ticker] = None
                record[key_weight] = 0.0
                record[key_return] = 0.0

        history.append(record)

    save_results(history, trade_log, CONFIG['INITIAL_CASH'])

if __name__ == "__main__":
    run_backtest()