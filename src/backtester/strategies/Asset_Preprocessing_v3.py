import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =========================================================
# 1. 설정 (Configuration)
# =========================================================
CONFIG = {
    # [기간 설정] 분석하고자 하는 시작일과 종료일
    'START_DATE': '2016-01-01',     
    'END_DATE': '2025-12-31',       

    # [자금 관리] 초기 자본 및 포트폴리오 비중 설정
    'INITIAL_CASH': 100_000_000,    # 초기 자본금 (1억 원)
    'MAX_POSITIONS': 4,             # 최대 보유 가능 종목 수
    'POS_WEIGHT': 0.25,             # 종목당 투자 비중 (25%)
    
    # [리스크 관리] 손절매 및 트레일링 스탑 설정
    'STOP_LOSS_PCT': 0.07,          # 초기 손절매 기준 (진입가 대비 -7%)
    'TRAIL_TRIGGER_PCT': 0.10,      # 트레일링 스탑 발동 조건 (수익률 +10% 도달 시)
    'TRAIL_STOP_PCT': 0.07,         # 트레일링 스탑 간격 (고점 대비 -7%)
    
    # [시간 제한] 자금 회전율을 위한 타임 스탑 설정
    'TIME_STOP_DAYS': 75,           # 최대 보유 기간 (75일)
    'TIME_STOP_ROI': 0.00,          # 기간 내 도달해야 하는 최소 수익률 (0% 이하 시 청산)
    
    # [비용] 거래 비용 설정
    'FEE': 0.004,                   # 매매 수수료 및 세금 합계 (0.4% 적용)
}

# =========================================================
# 2. 경로 및 환경 설정
# =========================================================
try:
    # 현재 실행 파일의 위치를 기준으로 상위 디렉토리를 탐색하여 경로 설정
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    # 대화형 환경(Jupyter 등)을 위한 예외 처리
    BASE_DIR = Path(".").resolve()

# 데이터 및 결과 파일 경로 정의
UNIVERSE_PATH = BASE_DIR / "data" / "universe" / "Pivot_Signals.csv"
PARQUET_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "src" / "backtester" / "strategies"

def load_price_data(code, data_dir):
    """
    개별 종목의 가격 데이터(Parquet)를 로드하고 전처리합니다.
    - 날짜 정규화 (시간 정보 제거)
    - 중복 데이터 제거 및 정렬
    """
    try:
        code_str = str(code).zfill(6).strip()
        file_path = data_dir / f"{code_str}.parquet"
        
        if not file_path.exists():
            return None
            
        df = pd.read_parquet(file_path)
        
        if 'Date' in df.columns:
            # 시간(Time) 정보를 제거하고 날짜(Date)만 남김
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
            df.set_index('Date', inplace=True)
        
        # 중복된 인덱스 제거 (첫 번째 값 유지) 및 날짜순 정렬
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return None

def preload_data(signal_df):
    """
    백테스팅 속도 향상을 위해 시그널에 존재하는 종목들의 데이터를 메모리에 미리 로드합니다.
    """
    cache = {}
    unique_codes = signal_df['Code'].unique()
    print(f"[Info] Loading data for {len(unique_codes)} tickers...")
    
    for code in tqdm(unique_codes):
        df = load_price_data(code, PARQUET_DIR)
        if df is not None:
            cache[code] = df
    return cache

def save_results(history, trade_log, initial_cash):
    """
    백테스팅 결과를 CSV 파일로 저장하고, 주요 성과 지표(MDD, 수익률 등)를 출력합니다.
    """
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # 1. 일별 자산 내역 저장 (Asset_List_Final.csv)
    res_df = pd.DataFrame(history)
    res_df.to_csv(OUTPUT_PATH / "Asset_List_Final.csv", index=False, encoding='utf-8-sig')
    
    # 2. 매매 기록 저장 (Trade_Log.csv) 및 성과 분석
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        
        # 컬럼 순서 재정렬
        cols = ['Ticker', 'Profit_Pct', 'Net_PnL', 'Reason', 'Entry_Date', 'Exit_Date']
        existing_cols = [c for c in cols if c in log_df.columns]
        log_df = log_df[existing_cols]
        log_df.to_csv(OUTPUT_PATH / "Trade_Log.csv", index=False, encoding='utf-8-sig')
        
        # 승률 및 손익비 계산
        wins = log_df[log_df['Net_PnL'] > 0]
        loses = log_df[log_df['Net_PnL'] <= 0]
        
        win_rate = len(wins) / len(log_df) * 100 if not log_df.empty else 0
        avg_win = wins['Net_PnL'].mean() if not wins.empty else 0
        avg_loss = abs(loses['Net_PnL'].mean()) if not loses.empty else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # MDD (Maximum Drawdown) 계산
        res_df['Peak'] = res_df['자산총액'].cummax()
        res_df['Drawdown'] = (res_df['자산총액'] - res_df['Peak']) / res_df['Peak']
        mdd = res_df['Drawdown'].min()

        final_asset = res_df.iloc[-1]['자산총액']
        total_return = (final_asset / initial_cash - 1) * 100

        # 결과 요약 출력
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
    """
    백테스팅 메인 로직을 실행합니다.
    1. 시그널 로드 및 기간 필터링
    2. 데이터 프리로딩 및 유효 거래일 계산
    3. 일별 시뮬레이션 (매도 -> 매수 -> 자산 평가)
    4. 결과 저장
    """
    
    # --- [1] 시그널 로드 및 전처리 ---
    if not UNIVERSE_PATH.exists():
        print(f"[Error] Signal file not found: {UNIVERSE_PATH}")
        return

    signals = pd.read_csv(UNIVERSE_PATH)
    signals['Code'] = signals['Code'].astype(str).str.zfill(6).str.strip()
    signals['Date'] = pd.to_datetime(signals['Date']).dt.normalize()
    
    # 기간 필터링 설정
    if CONFIG['START_DATE']:
        start_dt = pd.Timestamp(CONFIG['START_DATE']).normalize()
    else:
        start_dt = signals['Date'].min()

    if CONFIG['END_DATE']:
        end_dt = pd.Timestamp(CONFIG['END_DATE']).normalize()
    else:
        end_dt = signals['Date'].max() + pd.Timedelta(days=365)
    
    # 설정된 기간 내의 시그널만 추출 및 중복 제거
    signals = signals[(signals['Date'] >= start_dt) & (signals['Date'] <= end_dt)]
    signals = signals.drop_duplicates(subset=['Date', 'Code'])
    
    # --- [2] 데이터 로딩 및 환경 설정 ---
    data_cache = preload_data(signals)
    
    # 유효한 시장 거래일 계산 (휴일 및 비거래일 필터링용)
    # 데이터가 존재하는 모든 날짜의 합집합을 구함
    print("[Info] Calculating valid market dates...")
    valid_market_dates = set()
    for df in data_cache.values():
        valid_market_dates.update(df.index)
    
    # --- [3] 시뮬레이션 루프 초기화 ---
    cash = CONFIG['INITIAL_CASH']
    portfolio = {} 
    history = []
    trade_log = []
    
    # 전체 시뮬레이션 기간 생성 (평일 기준)
    all_dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
    
    print(f"[Info] Simulating from {start_dt.date()} to {end_dt.date()}...")
    
    for today in tqdm(all_dates):
        today_ts = pd.Timestamp(today).normalize()
        
        # 공휴일 체크: 실제 시장 데이터가 없는 날은 건너뜀
        if today_ts not in valid_market_dates:
            continue
            
        sold_codes = [] # 당일 매도한 종목 (당일 재매수 방지용)
        
        # ---------------------------------------------------------
        # [A] 보유 종목 관리 (매도 로직)
        # ---------------------------------------------------------
        for code in list(portfolio.keys()):
            pos = portfolio[code]
            df = data_cache.get(code)
            
            # 1. 상장 폐지 및 거래 정지 체크
            # 시장은 열렸으나 해당 종목의 데이터가 없는 경우 -> 전일 종가로 강제 청산
            if df is None or today_ts not in df.index:
                exit_reason = "Delisted" # 상장폐지/거래정지
                exit_price = pos['last_close'] # 마지막 관측 종가 사용
                
                revenue = pos['shares'] * exit_price * (1 - CONFIG['FEE'])
                cash += revenue
                pnl_val = revenue - (pos['shares'] * pos['entry_price'])
                pnl_pct = (exit_price / pos['entry_price']) - 1
                
                trade_log.append({
                    'Ticker': code,
                    'Profit_Pct': round(pnl_pct * 100, 2),
                    'Net_PnL': int(pnl_val),
                    'Reason': exit_reason,
                    'Entry_Date': pos['entry_date'].date(),
                    'Exit_Date': today_ts.date()
                })
                sold_codes.append(code)
                del portfolio[code]
                continue # 다음 종목으로 넘어감

            # 2. 정상 거래일 데이터 조회
            day_data = df.loc[today_ts]
            curr_open = day_data['Open']
            curr_high = day_data['High']
            curr_low = day_data['Low']
            curr_close = day_data['Close']
            
            pos['last_close'] = curr_close # 자산 평가를 위해 종가 업데이트

            # 데이터 오류 방어 (가격이 0인 경우 패스)
            if curr_open == 0 or curr_close == 0: continue

            # 3. 트레일링 스탑 기준가 업데이트
            if curr_high > pos['highest_price']:
                pos['highest_price'] = curr_high
            
            # 수익률이 특정 수준(Trigger)을 넘었을 때만 트레일링 스탑 가격 상향
            profit_rate_high = (pos['highest_price'] - pos['entry_price']) / pos['entry_price']
            if profit_rate_high >= CONFIG['TRAIL_TRIGGER_PCT']:
                new_trail_stop = pos['highest_price'] * (1 - CONFIG['TRAIL_STOP_PCT'])
                pos['stop_price'] = max(pos['stop_price'], new_trail_stop)
            
            # 4. 매도 조건 실행 (우선순위: 손절/트레일링 -> 타임컷)
            exit_reason = None
            exit_price = curr_close
            
            # (1) 손절매 또는 트레일링 스탑
            if curr_low <= pos['stop_price']:
                exit_reason = "Stop_Loss"
                # 시가가 이미 스탑가보다 낮게 시작(갭락)했다면 시가 매도, 아니면 스탑가 매도
                exit_price = curr_open if curr_open < pos['stop_price'] else pos['stop_price']
            
            # (2) 시간 제한 (Time Stop)
            elif (today_ts - pos['entry_date']).days >= CONFIG['TIME_STOP_DAYS']:
                current_pnl = (curr_close - pos['entry_price']) / pos['entry_price']
                # 보유 기간이 지났음에도 수익률이 기준치 이하라면 매도
                if current_pnl <= CONFIG['TIME_STOP_ROI']:
                    exit_reason = "Time_Stop"
                    exit_price = curr_close
            
            # 매도 확정 시 처리
            if exit_reason:
                revenue = pos['shares'] * exit_price * (1 - CONFIG['FEE'])
                cash += revenue
                
                pnl_val = revenue - (pos['shares'] * pos['entry_price'])
                pnl_pct = (exit_price / pos['entry_price']) - 1
                
                trade_log.append({
                    'Ticker': code,
                    'Profit_Pct': round(pnl_pct * 100, 2),
                    'Net_PnL': int(pnl_val),
                    'Reason': exit_reason,
                    'Entry_Date': pos['entry_date'].date(),
                    'Exit_Date': today_ts.date()
                })
                sold_codes.append(code)
                del portfolio[code]

        # ---------------------------------------------------------
        # [B] 신규 진입 (매수 로직)
        # ---------------------------------------------------------
        daily_signals = signals[signals['Date'] == today_ts]
        
        for _, row in daily_signals.iterrows():
            # 최대 보유 종목 수 제한 확인
            if len(portfolio) >= CONFIG['MAX_POSITIONS']: break
            
            code = row['Code']
            # 이미 보유 중이거나 당일 매도한 종목은 재진입 금지
            if code in portfolio or code in sold_codes: continue
            
            df = data_cache.get(code)
            if df is None or today_ts not in df.index: continue
            
            curr_open = df.loc[today_ts]['Open']
            curr_close = df.loc[today_ts]['Close']
            
            if curr_open <= 0: continue
            
            # 자금 관리: 현재 총 자산 가치 기준 비중 계산
            current_equity_for_calc = cash
            for p_val in portfolio.values():
                current_equity_for_calc += p_val['shares'] * p_val['last_close']
            
            target_amt = current_equity_for_calc * CONFIG['POS_WEIGHT']
            invest_amt = min(target_amt, cash) # 가용 현금 내에서만 매수
            shares = int(invest_amt / curr_open)
            
            if shares > 0:
                cost = shares * curr_open
                cash -= cost
                # 포트폴리오 편입
                portfolio[code] = {
                    'entry_date': today_ts,
                    'entry_price': curr_open,
                    'shares': shares,
                    'highest_price': curr_open,
                    'stop_price': curr_open * (1 - CONFIG['STOP_LOSS_PCT']),
                    'last_close': curr_close # 매수 당일 종가 초기화
                }

        # ---------------------------------------------------------
        # [C] 일별 자산 평가 및 기록
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

        # 보유 종목 상세 내역 기록 (비중 순 정렬)
        holdings = []
        for ticker, info in portfolio.items():
            curr_val = info['shares'] * info['last_close']
            weight = (curr_val / curr_equity * 100) if curr_equity > 0 else 0
            ret = (info['last_close'] - info['entry_price']) / info['entry_price'] * 100
            holdings.append({'ticker': ticker, 'weight': weight, 'return': ret})
        
        holdings.sort(key=lambda x: x['weight'], reverse=True)

        # CSV 컬럼에 맞춰 최대 N개 종목 정보 기입
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

    # --- [4] 최종 결과 저장 ---
    save_results(history, trade_log, CONFIG['INITIAL_CASH'])

if __name__ == "__main__":
    run_backtest()