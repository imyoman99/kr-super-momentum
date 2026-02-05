import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 마크 미너비니의 '트렌드 템플릿' 및 'VCP 패턴' 필터 함수 임포트
# (같은 폴더에 minervini_filter.py가 있어야 합니다)
from minervini_filter import check_all_minervini_filters, load_from_parquet

# =========================================================
# [CONFIG] 백테스팅 하이퍼파라미터 설정
# =========================================================
CONFIG = {
    'INITIAL_CASH': 100_000_000,
    'MAX_SLOTS': 4,
    'MAX_POS_WEIGHT': 0.25,

    'STOP_LOSS_PCT': 0.07,          # 매수가 대비 손절선 (-7%)
    'TRAILING_STOP_PCT': 0.10,      # 고가 대비 트레일링 스탑 (-10%)

    'TIME_STOP_DAYS': 10,           # 시간 제한 매도 기준일
    'TIME_STOP_MIN_RET': 0.01,      # 시간 제한 내 최소 수익률
    'TECH_EXIT_MA': 'MA50',         # 추세 이탈 기준 이평선
    
    'COST_RATE': 0.0040,            # [변경] 왕복 거래비용 0.40% (매수/매도 수수료+세금 포함)

    'START_DATE': "2016-01-01",
    'END_DATE': "2025-12-31",
}

# ---------------------------------------------------------
# 데이터 경로 설정
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
UNIVERSE_PATH = BASE_DIR / "data" / "universe" / "intersection_80.csv"
PARQUET_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "src" / "backtester" / "strategies"

# 경로 생성
if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

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
    portfolio = {}      # {티커: {shares, entry_price, highest_price, ...}}
    history = []        # 일별 기록
    price_cache = {}    # 데이터 캐싱
    trade_log = []      # 매매 기록
    sell_reasons = {}   # 매도 사유 통계

    print(f"- 백테스팅 시작: {CONFIG['START_DATE']} ~ {CONFIG['END_DATE']}")
    
    for today in tqdm(all_dates):
        today_ts = pd.Timestamp(today)
        
        # -----------------------------------------------------
        # PHASE 1: [청산] Low 기준 장중 대응 및 상폐 처리
        # -----------------------------------------------------
        for ticker in list(portfolio.keys()):
            # 캐싱된 데이터 가져오기
            df = price_cache.get(ticker)
            if df is None: continue 
            
            sell_reason = None
            exit_price = 0.0
            
            # [A] 상장폐지 방어 (데이터 마지막 날짜 도달 시 강제 청산)
            # 백테스트 종료일(END_DATE)이 아닌데 데이터가 끝났다면 상폐/거래정지로 간주
            is_sim_end = (today_ts == pd.Timestamp(CONFIG['END_DATE']))
            if today_ts >= df.index[-1] and not is_sim_end:
                exit_price = float(df.loc[df.index[-1], 'Close'])
                sell_reason = "Delisting_Force_Exit"
            
            # [B] 일반 매도 조건 (데이터가 존재할 때)
            elif today_ts in df.index:
                curr_data = df.loc[today_ts]
                curr_open = float(curr_data['Open'])
                curr_high = float(curr_data['High'])
                curr_low = float(curr_data['Low'])
                curr_close = float(curr_data['Close'])
                
                info = portfolio[ticker]
                
                # 보유일수 증가 및 최고가 갱신
                info['days_held'] += 1
                info['highest_price'] = max(info['highest_price'], curr_high)
                
                entry_p = info['entry_price']
                highest_p = info['highest_price']
                
                # 손절/익절 기준가 계산
                hard_stop = entry_p * (1 - CONFIG['STOP_LOSS_PCT'])
                trailing_stop = highest_p * (1 - CONFIG['TRAILING_STOP_PCT'])
                
                # 최종 스탑 라인 (둘 중 더 높은 가격이 유효한 스탑)
                # (단, 로직 분리를 위해 아래에서 각각 체크)
                
                # 1. 하드 스탑 (장중 저점이 손절선 건드리면 손절선 가격에 매도)
                if curr_low <= hard_stop:
                    sell_reason = "Initial_Stop_Loss"
                    exit_price = hard_stop  # 요청사항: 저점이 더 낮아도 손절선 가격에 매도
                    
                # 2. 트레일링 스탑 (장중 저점이 익절선 건드리면 익절선 가격에 매도)
                elif curr_low <= trailing_stop:
                    sell_reason = "Trailing_Stop_Profit"
                    exit_price = trailing_stop # 요청사항: 익절선 가격에 매도
                
                # 3. 시간 제한 (종가 기준)
                elif info['days_held'] >= CONFIG['TIME_STOP_DAYS']:
                    curr_ret = (curr_close - entry_p) / entry_p
                    if curr_ret < CONFIG['TIME_STOP_MIN_RET']:
                        sell_reason = "Time_Stop"
                        exit_price = curr_close
                
                # 4. 기술적 이평선 이탈 (종가 기준)
                elif curr_close < curr_data.get(CONFIG['TECH_EXIT_MA'], 0):
                    sell_reason = f"{CONFIG['TECH_EXIT_MA']}_Break"
                    exit_price = curr_close
                
                # 가격 업데이트 (평가금액 계산용)
                info['last_price'] = curr_close

            # 매도 실행 로직
            if sell_reason:
                shares = portfolio[ticker]['shares']
                sell_amt = shares * exit_price
                buy_amt = shares * portfolio[ticker]['entry_price']
                
                # [수수료 적용] 수익률 계산 시 비용 차감 (단순 수익률 - 0.4%)
                raw_ret = (exit_price - portfolio[ticker]['entry_price']) / portfolio[ticker]['entry_price']
                realized_ret = raw_ret - CONFIG['COST_RATE'] 
                
                # 현금 반환 (실제 현금은 단순히 매도금액 더함, 비용은 수익률 통계에만 반영하거나, 
                # 현금에서도 제하고 싶다면 아래처럼 차감)
                # 여기서는 현금 시뮬레이션의 정확성을 위해 현금에서도 비용 차감
                transaction_cost = sell_amt * CONFIG['COST_RATE'] # 약식으로 매도금액 기준 전체 비용 계산
                cash += (sell_amt - transaction_cost)
                
                trade_log.append({
                    'Ticker': ticker, 
                    'Profit': round(realized_ret, 4), 
                    'Reason': sell_reason, 
                    'Entry_Date': portfolio[ticker]['entry_date'],
                    'Exit_Date': today_ts.date()
                })
                sell_reasons[sell_reason] = sell_reasons.get(sell_reason, 0) + 1
                del portfolio[ticker]

        # -----------------------------------------------------
        # PHASE 2: [진입] '내일 시가' 매수 (미래 참조 방지)
        # -----------------------------------------------------
        # 현재 총 자산 가치 계산
        current_equity = float(cash)
        for t, p_info in portfolio.items():
            current_equity += p_info['shares'] * p_info['last_price']

        # 슬롯 여유가 있을 때만 탐색
        if len(portfolio) < CONFIG['MAX_SLOTS']:
            candidates = universe[universe['Date'] == today].sort_values('RS', ascending=False)
            
            for _, row in candidates.iterrows():
                if len(portfolio) >= CONFIG['MAX_SLOTS']: break
                ticker = row['Code']
                if ticker in portfolio: continue
                
                # 미너비니 필터 통과 확인
                if check_all_minervini_filters(ticker, today_ts, data_dir=str(PARQUET_DIR) + "\\"):
                    
                    # 데이터 로드
                    if ticker not in price_cache:
                        price_cache[ticker] = load_from_parquet(ticker, str(PARQUET_DIR) + "\\")
                    df = price_cache[ticker]
                    
                    if df is not None:
                        # [핵심] 오늘이 아니라 '내일 시가'를 확인
                        # 현재 날짜의 인덱스를 찾고 +1
                        try:
                            curr_loc = df.index.get_loc(today_ts)
                            next_loc = curr_loc + 1
                            
                            # 내일 데이터가 존재해야 매수 가능
                            if next_loc < len(df):
                                next_day_data = df.iloc[next_loc]
                                buy_price = float(next_day_data['Open']) # 내일 시가
                                buy_date = df.index[next_loc]
                                
                                if buy_price > 0:
                                    target_amt = current_equity * CONFIG['MAX_POS_WEIGHT']
                                    actual_amt = min(target_amt, cash)
                                    shares = int(actual_amt // buy_price)
                                    
                                    if shares > 0:
                                        portfolio[ticker] = {
                                            'shares': shares, 
                                            'entry_price': buy_price, 
                                            'entry_date': buy_date, # 진입일은 내일로 기록
                                            'highest_price': buy_price, 
                                            'days_held': 0,
                                            'last_price': buy_price 
                                        }
                                        # 현금 차감 (내일 아침에 산다고 가정하고 미리 차감)
                                        cash -= shares * buy_price
                        except KeyError:
                            # 인덱스 에러 등의 경우 스킵
                            continue

        # -----------------------------------------------------
        # PHASE 3: [기록] 일별 자산 상세 기록
        # -----------------------------------------------------
        asset_details = []
        for t, p_info in portfolio.items():
            curr_close = p_info['last_price']
            val = p_info['shares'] * curr_close
            stock_ret = (curr_close - p_info['entry_price']) / p_info['entry_price']
            asset_details.append((t, val, stock_ret))
        
        record = {'DATE': today_ts.date(), '자산총액': int(current_equity), '현금총액': int(cash)}
        
        # 슬롯별 기록
        for i in range(CONFIG['MAX_SLOTS']):
            if i < len(asset_details):
                ticker, total_val, s_ret = asset_details[i]
                record[f'자산{i+1}_티커'] = ticker
                record[f'자산{i+1}_총액'] = int(total_val)
                record[f'자산{i+1}_수익률'] = round(s_ret * 100, 2)
            else:
                record[f'자산{i+1}_티커'] = "None"
                record[f'자산{i+1}_총액'] = 0
                record[f'자산{i+1}_수익률'] = 0.0
                
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
    
    if len(result_df) > 0:
        final_val = result_df['자산총액'].iloc[-1]
        total_ret = (final_val / CONFIG['INITIAL_CASH'] - 1) * 100
        days = (pd.to_datetime(CONFIG['END_DATE']) - pd.to_datetime(CONFIG['START_DATE'])).days
        cagr = ((final_val / CONFIG['INITIAL_CASH']) ** (365/max(days, 1)) - 1) * 100
        
        result_df['Peak'] = result_df['자산총액'].cummax()
        result_df['DD'] = (result_df['자산총액'] - result_df['Peak']) / result_df['Peak']
        mdd = result_df['DD'].min() * 100

        print(f"- 최종 자산: {final_val:,.0f}원")
        print(f"- 누적 수익률: {total_ret:.2f}% / CAGR: {cagr:.2f}%")
        print(f"- 최대 낙폭(MDD): {mdd:.2f}%")
    else:
        print("결과 데이터가 없습니다.")

    if trade_log:
        t_df = pd.DataFrame(trade_log)
        win_trades = t_df[t_df['Profit'] > 0]
        loss_trades = t_df[t_df['Profit'] <= 0]
        
        win_rate = (len(win_trades) / len(t_df)) * 100
        avg_prof = win_trades['Profit'].mean() * 100 if not win_trades.empty else 0
        avg_loss = loss_trades['Profit'].mean() * 100 if not loss_trades.empty else 0
        
        profit_factor = abs(avg_prof / avg_loss) if avg_loss != 0 else float('inf')
        print(f"- 총 거래 횟수: {len(t_df)}회")
        print(f"- 승률: {win_rate:.2f}% / 손익비: {profit_factor:.2f}")
        
    print("-" * 50)
    print("- 매도 사유별 통계:")
    for reason, count in sell_reasons.items():
        print(f"   - {reason}: {count}회")
    print("="*50)

if __name__ == "__main__":
    run_backtest()