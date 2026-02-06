import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 마크 미너비니의 '트렌드 템플릿' 및 'VCP 패턴' 필터 함수 임포트
from minervini_filter import check_all_minervini_filters, load_from_parquet

# =========================================================
# [CONFIG] 백테스팅 하이퍼파라미터 설정
# =========================================================
CONFIG = {
    'INITIAL_CASH': 100_000_000,
    'MAX_SLOTS': 4,
    'MAX_POS_WEIGHT': 0.25,

    'STOP_LOSS_PCT': 0.07,          # 매수가 대비 손절선 (-7%)
    'TRAILING_STOP_PCT': 0.10,      # 최고가 대비 하락 시 익절선 (-10%)

    'TIME_STOP_DAYS': 10,           # 시간 제한 매도 기준일
    'TIME_STOP_MIN_RET': 0.01,      # 시간 제한 내 최소 수익률
    'TECH_EXIT_MA': 'MA20',         # 추세 이탈 기준 이평선

    'COST_RATE': 0.0040,            # 거래 비용 비율 (0.4%)

    'RISK_PER_TRADE_PCT': 0.02,     # 1회 매매 시 자산의 2% 리스크
    'ATR_STOP_MULTIPLIER': 2.0,     # ATR의 몇 배를 손절폭으로 잡을 것인가 
    'MAX_POS_WEIGHT': 0.25,         # 아무리 ATR이 작아도 자산의 25%까지만 매수 (안전장치)

    'START_DATE': "2016-01-01",
    'END_DATE': "2025-12-31",
}

# ---------------------------------------------------------
# 데이터 경로 설정 (사용자 원본 경로 유지)
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
UNIVERSE_PATH = BASE_DIR / "data" / "universe" / "intersection_80.csv"
PARQUET_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "src" / "backtester" / "strategies"

def run_backtest():
    # --- 1. 데이터 로드 및 전처리 ---
    print("- 유니버스 데이터를 읽는 중...")
    if not UNIVERSE_PATH.exists():
        print(f"오류: 유니버스 파일을 찾을 수 없습니다. ({UNIVERSE_PATH})")
        return

    universe = pd.read_csv(UNIVERSE_PATH, dtype={'Code': str})
    universe['Date'] = pd.to_datetime(universe['Date'])
    universe = universe[(universe['Date'] >= CONFIG['START_DATE']) & (universe['Date'] <= CONFIG['END_DATE'])]
    all_dates = sorted(universe['Date'].unique())
    
    # --- 2. 시뮬레이션 변수 초기화 ---
    cash = CONFIG['INITIAL_CASH']
    portfolio = {}      # {티커: 정보}
    history = []        # 일별 기록
    price_cache = {}    # 데이터 캐싱
    trade_log = []      # 매매 기록
    sell_reasons = {}   # 매도 사유 통계

    print(f"- 백테스팅 시작: {CONFIG['START_DATE']} ~ {CONFIG['END_DATE']}")
    
    for today in tqdm(all_dates):
        today_ts = pd.Timestamp(today)
        
        # -----------------------------------------------------
        # PHASE 1: [청산] 매도 조건 체크 (상장폐지 대응 포함)
        # -----------------------------------------------------
        for ticker in list(portfolio.keys()):
            if ticker not in price_cache:
                price_cache[ticker] = load_from_parquet(ticker, str(PARQUET_DIR) + "\\")
            
            df = price_cache[ticker]
            if df is None or len(df) == 0: continue
            
            sell_reason = None
            exit_price = 0.0

            # [A] 상장폐지 및 데이터 종료 체크
            if today_ts >= df.index[-1]:
                exit_price = float(df.loc[df.index[-1], 'Close'])
                sell_reason = "Delisting_Force_Exit"
            
            # [B] 일반 매도 조건 체크 (데이터가 있는 경우)
            elif today_ts in df.index:
                curr_data = df.loc[today_ts]
                curr_price = float(curr_data['Close'])
                info = portfolio[ticker]
                
                # 정보 업데이트
                info['days_held'] += 1
                info['highest_price'] = max(info['highest_price'], float(curr_data['High']))
                
                entry_p = info['entry_price']
                highest_p = info['highest_price']
                curr_ret = (curr_price - entry_p) / entry_p
                
                # 매도가 계산 (손절선 vs 트레일링 스탑)
                info = portfolio[ticker]
                # 진입 시점에 ATR로 계산해서 저장해둔 '고정 손절가' 사용
                hard_stop = info['stop_loss_price'] 

                # 트레일링 스탑과 비교하여 더 높은 가격을 최종 매도가로 설정
                trailing_stop = highest_p * (1 - CONFIG['TRAILING_STOP_PCT'])
                final_stop = max(hard_stop, trailing_stop)
                info['stop_loss_price'] = final_stop
                # 조건 판별
                if curr_price <= final_stop:
                    sell_reason = "Initial_Stop_Loss" if final_stop == hard_stop else "Trailing_Stop_Profit"
                elif info['days_held'] >= CONFIG['TIME_STOP_DAYS'] and curr_ret < CONFIG['TIME_STOP_MIN_RET']:
                    sell_reason = "Time_Stop"
                elif curr_price < curr_data.get(CONFIG['TECH_EXIT_MA'], 0):
                    sell_reason = f"{CONFIG['TECH_EXIT_MA']}_Break"
                
                exit_price = curr_price
            
            # 매도 확정 시 처리
            if sell_reason:
                realized_ret = (exit_price - portfolio[ticker]['entry_price']) / portfolio[ticker]['entry_price']
                trade_log.append({
                    'Ticker': ticker, 
                    'Profit': round(realized_ret, 4), 
                    'Reason': sell_reason, 
                    'Entry_Date': portfolio[ticker]['entry_date'],
                    'Exit_Date': today_ts.date()
                })
                sell_reasons[sell_reason] = sell_reasons.get(sell_reason, 0) + 1
                cash += portfolio[ticker]['shares'] * exit_price * (1 - CONFIG['COST_RATE'])
                del portfolio[ticker]

        # -----------------------------------------------------
        # PHASE 2: [진입] 매수 신호 탐색 및 자산 가치 계산
        # -----------------------------------------------------
        current_equity = float(cash)
        for t, p_info in portfolio.items():
            # 자산 총액 계산 시 결측치 보정 로직 통합
            if today_ts in price_cache[t].index:
                curr_close = price_cache[t].loc[today_ts, 'Close']
                p_info['last_price'] = curr_close 
            else:
                curr_close = p_info.get('last_price', p_info['entry_price'])
            
            current_equity += float(p_info['shares']) * float(curr_close)

        # 슬롯 여유가 있을 때만 매수 탐색
        if len(portfolio) < CONFIG['MAX_SLOTS']:
            candidates = universe[universe['Date'] == today].sort_values('RS', ascending=False)
            for _, row in candidates.iterrows():
                if len(portfolio) >= CONFIG['MAX_SLOTS']: break
                ticker = row['Code']
                if ticker in portfolio: continue
                
                if check_all_minervini_filters(ticker, today_ts, data_dir=str(PARQUET_DIR) + "\\"):
                    if ticker not in price_cache:
                        price_cache[ticker] = load_from_parquet(ticker, str(PARQUET_DIR) + "\\")
                    df = price_cache[ticker]
                    
                    if df is not None and today_ts in df.index:
                        curr_data = df.loc[today_ts]
                        buy_price = float(curr_data['Close'])
                        atr_value = float(curr_data['ATR_14']) # minervini_filter에서 계산된 ATR 사용

                        # 1. 내가 감당할 리스크 금액 (예: 1억 자산의 2% = 200만원)
                        risk_amount = current_equity * CONFIG['RISK_PER_TRADE_PCT']

                        # 2. 1주당 손절폭 (예: ATR 500원 * 2배 = 1,000원)
                        stop_distance = atr_value * CONFIG['ATR_STOP_MULTIPLIER']
    
                        # 3. ATR이 너무 작을 경우를 대비한 최소 손절폭 (예: 주가의 최소 2%)
                        stop_distance = max(stop_distance, buy_price * 0.02)

                        # 4. 수량 계산: 리스크 금액 / 주당 손절폭
                        shares = int(risk_amount // stop_distance)

                        # 5. 비중 캡핑: 아무리 안전해 보여도 총 자산의 25%를 넘지 않게 함
                        max_shares = int((current_equity * CONFIG['MAX_POS_WEIGHT']) // buy_price)
                        shares = min(shares, max_shares)
    
                        # 6. 실제 매수 가능 금액(잔고) 체크
                        if shares * buy_price > cash:
                            shares = int(cash // buy_price)

                        if shares > 0:
                            portfolio[ticker] = {
                                'shares': shares, 
                                'entry_price': buy_price, 
                                'stop_loss_price': buy_price - stop_distance,
                                'entry_date': today_ts, 
                                'highest_price': buy_price, 
                                'days_held': 0,
                                'last_price': buy_price 
                            }
                            cash -= shares * buy_price

        # -----------------------------------------------------
        # PHASE 3: [기록] 일별 자산 상세 기록
        # -----------------------------------------------------
        asset_details = []
        for t, p_info in portfolio.items():
            curr_close = p_info['last_price']
            val = float(p_info['shares']) * float(curr_close)
            stock_ret = (curr_close - p_info['entry_price']) / p_info['entry_price']
            asset_details.append((t, val, stock_ret))
        
        record = {'DATE': today_ts.date(), '자산총액': int(current_equity), '현금총액': int(cash)}
        for i in range(CONFIG['MAX_SLOTS']):
            if i < len(asset_details):
                ticker, total_val, s_ret = asset_details[i]
                record[f'자산{i+1}_티커'] = ticker
                record[f'자산{i+1}_총액'] = int(total_val)
                record[f'자산{i+1}_수익률'] = round(s_ret * 100, 2)
            else:
                record[f'자산{i+1}_티커'], record[f'자산{i+1}_총액'], record[f'자산{i+1}_수익률'] = "None", 0, 0.0
        history.append(record)

    # -----------------------------------------------------
    # 4. 결과 분석 및 파일 저장
    # -----------------------------------------------------
    result_df = pd.DataFrame(history)
    result_df.to_csv(OUTPUT_PATH / "Asset_List_Final.csv", index=False, encoding='utf-8-sig')
    
    trade_log_df = pd.DataFrame(trade_log)
    trade_log_df.to_csv(OUTPUT_PATH / "Trade_Log.csv", index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("      백테스팅 성과 요약 리포트 ")
    print("="*50)
    
    final_val = result_df['자산총액'].iloc[-1]
    total_ret = (final_val / CONFIG['INITIAL_CASH'] - 1) * 100
    days = (pd.to_datetime(CONFIG['END_DATE']) - pd.to_datetime(CONFIG['START_DATE'])).days
    cagr = ((final_val / CONFIG['INITIAL_CASH']) ** (365/max(days, 1)) - 1) * 100
    
    result_df['Peak'] = result_df['자산총액'].cummax()
    result_df['DD'] = (result_df['자산총액'] - result_df['Peak']) / result_df['Peak']
    mdd = result_df['DD'].min() * 100

    print(f"- 최종 자산: {final_val:,}원")
    print(f"- 누적 수익률: {total_ret:.2f}% / CAGR: {cagr:.2f}%")
    print(f"- 최대 낙폭(MDD): {mdd:.2f}%")

    if trade_log:
        t_df = pd.DataFrame(trade_log)
        win_trades = t_df[t_df['Profit'] > 0]
        loss_trades = t_df[t_df['Profit'] <= 0]
        
        win_rate = (len(win_trades) / len(t_df)) * 100
        avg_prof = win_trades['Profit'].mean() * 100 if not win_trades.empty else 0
        avg_loss = loss_trades['Profit'].mean() * 100 if not loss_trades.empty else 0
        
        profit_factor = abs(avg_prof / avg_loss) if avg_loss != 0 else float('inf')
        print(f"- 승률: {win_rate:.2f}% / 손익비: {profit_factor:.2f}")
        
    print("-" * 50)
    print("- 매도 사유별 통계:")
    for reason, count in sell_reasons.items():
        print(f"   - {reason}: {count}회")
    print("="*50)

if __name__ == "__main__":
    run_backtest()