import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from minervini_filter_v2 import check_minervini_from_df, load_from_parquet

# [CONFIG] 최종 전략: 1% 리스크 + MA20 + 2.0 ATR 트레일링 + 갭하락 방어
CONFIG = {
    'INITIAL_CASH': 100_000_000,
    'MAX_POS_WEIGHT': 0.25,           # 비중 캡 25%
    'RISK_PER_TRADE_PCT': 0.01,       # 리스크 1% (방어력 강화)
    'TOTAL_RISK_CAP': 0.12,           # 리스크 총량 12%
    
    'ATR_STOP_MULTIPLIER': 2.5,       # 초기 손절폭 (1.75 ATR)
    'ATR_TRAIL_MULTIPLIER': 3.0,      # 트레일링 스탑 2.0 ATR (수익 보전)
    'ATR_FLOOR_RATIO': 0.01,          # ATR 최소값 보정

    'TIME_STOP_DAYS': 25,             # 15일 내 승부 안 나면 교체
    'TIME_STOP_MIN_RET': 0.00,       
    'TECH_EXIT_MA': 'MA20',           # 추세 기준 MA20

    'COST_RATE': 0.0040,              # 수수료+세금
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
    for code in tqdm(unique_codes, desc="Data Loading"):
        try:
            df = load_from_parquet(code, str(PARQUET_DIR))
            if df is not None: cache[code] = df
        except: continue
    return cache

def run_backtest():
    """메인 백테스팅 엔진"""
    if not UNIVERSE_PATH.exists(): return
    
    universe = pd.read_csv(UNIVERSE_PATH, dtype={'Code': str})
    universe['Date'] = pd.to_datetime(universe['Date'])
    universe = universe[(universe['Date'] >= CONFIG['START_DATE']) & (universe['Date'] <= CONFIG['END_DATE'])]
    all_dates = sorted(universe['Date'].unique())

    data_cache = preload_data_to_ram(universe)
    cash = float(CONFIG['INITIAL_CASH'])
    portfolio = {}      
    history = []; trade_log = []
    
    # CSV 출력용 슬롯 개수
    MAX_DISPLAY_SLOTS = 20 
    
    for today in tqdm(all_dates, desc="Backtesting"):
        today_ts = pd.Timestamp(today)
        curr_equity = cash + sum(info['shares'] * info['last_close'] for info in portfolio.values())
        
        # -----------------------------------------------------
        # 1. 포트폴리오 관리 (청산 로직)
        # -----------------------------------------------------
        for ticker in list(portfolio.keys()):
            df = data_cache.get(ticker)
            if df is None or today_ts not in df.index: continue 

            curr_data = df.loc[today_ts]
            curr_open, curr_high, curr_low, curr_close = map(float, [curr_data['Open'], curr_data['High'], curr_data['Low'], curr_data['Close']])
            info = portfolio[ticker]
            
            # [트레일링 스탑 로직]
            # 1. 역대 최고가 갱신
            if curr_high > info['highest_price']:
                info['highest_price'] = curr_high
            
            # 2. 샹들리에 스탑 계산 (최고가 - 2.0 * ATR)
            atr_val = info['R_val'] / CONFIG['ATR_STOP_MULTIPLIER'] 
            chandelier_stop = info['highest_price'] - (atr_val * CONFIG['ATR_TRAIL_MULTIPLIER'])
            
            # 3. 스탑 라인은 위로만 이동 (Ratchet)
            if chandelier_stop > info['trailing_stop']:
                info['trailing_stop'] = chandelier_stop
            
            # 4. 현재 유효 스탑 = MAX(초기손절가, 트레일링스탑)
            active_stop_price = max(info['fixed_stop'], info['trailing_stop'])
            
            sell_signal = False; sell_price = 0.0; sell_reason = ""

            # (1) 갭하락 스탑 (Gap Down)
            if curr_open < active_stop_price:
                sell_signal = True; sell_price = curr_open; sell_reason = "Gap_Down_Stop"
            
            # (2) 장중 스탑 (손절 OR 익절)
            elif curr_low <= active_stop_price:
                sell_signal = True; sell_price = active_stop_price
                if active_stop_price > info['entry_price']:
                    sell_reason = "Trailing_Profit" # 익절
                else:
                    sell_reason = "Stop_Loss" # 손절
            """
            # (3) 기술적 매도 (MA20 이탈)
            elif curr_close < curr_data.get(CONFIG['TECH_EXIT_MA'], 0):
                sell_signal = True; sell_price = curr_close; sell_reason = f"{CONFIG['TECH_EXIT_MA']}_Break"

            # (4) 시간 손절 (15일)
            elif info['days_held'] >= CONFIG['TIME_STOP_DAYS'] and (curr_close/info['entry_price']-1) < CONFIG['TIME_STOP_MIN_RET']:
                sell_signal = True; sell_price = curr_close; sell_reason = "Time_Stop"
            """
            # 매도 실행
            if sell_signal:
                shares = float(info['shares'])
                exit_val = sell_price * shares
                cost = exit_val * CONFIG['COST_RATE']
                
                # 순손익 및 수익률 계산
                net_pnl = exit_val - (info['entry_price'] * shares) - cost
                ret_pct = (sell_price - info['entry_price']) / info['entry_price'] * 100
                
                cash += (exit_val - cost)
                
                trade_log.append({
                    'Ticker': ticker, 
                    'Profit_Pct': round(ret_pct, 2),  # 수익률 기록
                    'Net_PnL': int(net_pnl), 
                    'Reason': sell_reason, 
                    'Entry_Date': info['entry_date'], 
                    'Exit_Date': today_ts.date()
                })
                del portfolio[ticker]
            else:
                info['days_held'] += 1
                info['last_close'] = curr_close

        # -----------------------------------------------------
        # 2. 신규 진입 관리 (갭하락 방어 포함)
        # -----------------------------------------------------
        current_open_risk = sum((info['entry_price'] - info['fixed_stop']) * info['shares'] for info in portfolio.values())
        risk_cap_amount = curr_equity * CONFIG['TOTAL_RISK_CAP']
        
        if current_open_risk < risk_cap_amount:
            candidates = universe[universe['Date'] == today].sort_values('RS', ascending=False)
            for _, row in candidates.iterrows():
                ticker = row['Code']
                if ticker in portfolio: continue
                
                df = data_cache.get(ticker)
                if df is not None and check_minervini_from_df(df, today_ts):
                    try:
                        sig_data = df.loc[today_ts]
                        signal_close = float(sig_data['Close']) # 신호 발생일 종가
                        
                        # 내일 시가 진입 가정
                        buy_price = float(df.iloc[df.index.get_loc(today_ts) + 1]['Open'])
                        raw_atr = float(sig_data.get('ATR_14', 0))
                        
                        # 갭상승 10% 이상 시 진입 무효 (이미 오른 종목 사지 않기)
                        if (buy_price - signal_close) / signal_close <= +0.10: continue
                        
                        if raw_atr == 0 or buy_price < 500: continue
                        
                        effective_atr = max(raw_atr, buy_price * CONFIG['ATR_FLOOR_RATIO'])
                        stop_dist = effective_atr * CONFIG['ATR_STOP_MULTIPLIER']
                        stop_price = buy_price - stop_dist
                        per_share_risk = stop_dist
                        
                        shares_by_risk = (curr_equity * CONFIG['RISK_PER_TRADE_PCT']) // per_share_risk
                        shares_by_cap = (curr_equity * CONFIG['MAX_POS_WEIGHT']) // buy_price
                        shares_by_cash = cash // buy_price
                        
                        shares = int(min(shares_by_risk, shares_by_cap, shares_by_cash))
                        new_trade_risk = shares * per_share_risk
                        
                        if shares > 0 and (current_open_risk + new_trade_risk) <= risk_cap_amount:
                            cash -= (shares * buy_price)
                            portfolio[ticker] = {
                                'shares': shares, 
                                'entry_price': buy_price, 
                                'entry_date': today_ts.date(),
                                'stop_price': stop_price,   # 현재 유효 스탑
                                'fixed_stop': stop_price,   # 초기 고정 스탑
                                'trailing_stop': 0.0,       # 트레일링 스탑
                                'highest_price': buy_price, # 최고가
                                'R_val': per_share_risk,
                                'last_close': buy_price, 
                                'days_held': 0
                            }
                            current_open_risk += new_trade_risk
                    except: continue

        # -----------------------------------------------------
        # 3. 상세 기록 저장
        # -----------------------------------------------------
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

        for i in range(MAX_DISPLAY_SLOTS):
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

def save_results(history, trade_log, initial_cash):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(history)
    res_df.to_csv(OUTPUT_PATH / "Asset_List_Final.csv", index=False, encoding='utf-8-sig')
    
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        # 컬럼 순서 재정렬
        cols = ['Ticker', 'Profit_Pct', 'Net_PnL', 'Reason', 'Entry_Date', 'Exit_Date']
        # 혹시 모를 에러 방지
        existing_cols = [c for c in cols if c in log_df.columns]
        log_df = log_df[existing_cols]
        log_df.to_csv(OUTPUT_PATH / "Trade_Log.csv", index=False, encoding='utf-8-sig')
        
        wins = log_df[log_df['Net_PnL'] > 0]
        loses = log_df[log_df['Net_PnL'] <= 0]
        
        win_rate = len(wins) / len(log_df) * 100
        avg_win = wins['Net_PnL'].mean() if not wins.empty else 0
        avg_loss = abs(loses['Net_PnL'].mean()) if not loses.empty else 1
        
        res_df['Peak'] = res_df['자산총액'].cummax()
        res_df['Drawdown'] = (res_df['자산총액'] - res_df['Peak']) / res_df['Peak']
        mdd = res_df['Drawdown'].min()

        print("\n" + "="*40)
        print(f"Final Asset: {int(res_df.iloc[-1]['자산총액']):,} KRW")
        print(f"Total Return: {(res_df.iloc[-1]['자산총액']/initial_cash-1)*100:.2f}%")
        print(f"MDD: {mdd*100:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Profit Factor: {avg_win/avg_loss:.2f}")
        print("="*40)

if __name__ == "__main__":
    run_backtest()