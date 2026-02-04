import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
# 마크 미너비니의 '트렌드 템플릿' 및 'VCP 패턴' 필터 함수 임포트
from minervini_filter import check_all_minervini_filters, load_from_parquet

# =========================================================
# [CONFIG] 백테스팅 하이퍼파라미터 설정
# =========================================================
CONFIG = {
    # 1. 투자 자본 및 포트폴리오 구성
    'INITIAL_CASH': 100_000_000,    # 초기 자본금 (원)
    'MAX_SLOTS': 4,                 # 최대 보유 종목 수 (집중 투자 원칙)
    'MAX_POS_WEIGHT': 0.25,         # 종목당 최대 진입 비중 (자산의 25%)

    # 2. 수비적 청산 (Defensive Exit) - 자본 보호 및 수익 보존
    'STOP_LOSS_PCT': 0.07,          # 초기 손절 폭 (매수가 대비 -7%)
    'BACKSTOP_1_TRIGGER': 0.10,     # 1단계 백스탑: 최고수익 10% 도달 시 손절가를 본전으로 상향
    'BACKSTOP_2_TRIGGER': 0.20,     # 2단계 백스탑: 최고수익 20% 도달 시 손절가를 수익 구간으로 상향
    'BACKSTOP_2_LOCK_PCT': 0.10,    # 2단계 백스탑: 상향 시 확보할 최소 이익 수준 (매수가 대비 +10%)

    # 3. 공격적 청산 (Offensive Exit) - 오버슈팅 대응 및 수익 실현
    'PARTIAL_EXIT_TRIGGER': 0.20,   # 부분 익절 수익률 (20% 도달 시 보유량 일부 매도)
    'PARTIAL_EXIT_RATIO': 0.5,      # 부분 익절 비중 (보유 수량의 50%)
    'CLIMAX_RUN_DAYS': 10,          # 클라이맥스 런 감지 기간 (매수 후 10일 이내)
    'CLIMAX_RUN_TRIGGER': 0.40,     # 클라이맥스 런 폭등 수익률 (40% 이상 폭등 시 전량 매도)

    # 4. 시간 및 기술적 청산 (Efficiency) - 기회비용 및 추세 관리
    'TIME_STOP_DAYS': 10,           # 타임 스탑 기준일 (매수 후 10일 대기)
    'TIME_STOP_MIN_RET': 0.01,      # 타임 스탑 기준 수익률 (10일간 1% 미만 수익 시 교체)
    'TECH_EXIT_MA': 'MA50',         # 추세 이탈 기준 이동평균선 (기본 50일선)

    # 5. 시뮬레이션 기간 설정
    'START_DATE': "2016-01-01",
    'END_DATE': "2025-12-31",
}

# ---------------------------------------------------------
# 데이터 경로 및 출력 설정
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = BASE_DIR / "strategies" / "intersection_80.csv"   # RS 점수 포함된 종목 리스트
PARQUET_DIR = BASE_DIR / "20160101_20251231_parquet"              # 개별 종목 시세 데이터 (OHLCV)
OUTPUT_PATH = BASE_DIR / "strategies" / "Asset_List_Final.csv"    # 최종 자산 변동 기록 파일

def run_backtest():
    # --- 1. 유니버스 데이터 로드 및 기간 필터링 ---
    print("- 유니버스 데이터를 읽는 중...")
    universe = pd.read_csv(UNIVERSE_PATH, dtype={'Code': str})
    universe['Date'] = pd.to_datetime(universe['Date'])
    universe = universe[(universe['Date'] >= CONFIG['START_DATE']) & (universe['Date'] <= CONFIG['END_DATE'])]
    all_dates = sorted(universe['Date'].unique())
    
    # --- 2. 시뮬레이션 변수 초기화 ---
    cash = CONFIG['INITIAL_CASH']
    portfolio = {}      # 현재 보유 포지션 관리용 {티커: 정보}
    history = []        # 일별 자산 변화 기록용
    price_cache = {}    # 입출력 속도 향상을 위한 데이터 캐싱
    trade_log = []      # 종료된 모든 매매의 통계 기록
    sell_reasons = {}   # 매도 사유별 횟수 통계

    print(f"- 백테스팅 시작: {CONFIG['START_DATE']} ~ {CONFIG['END_DATE']}")
    
    # 설정된 기간 동안 일별로 시뮬레이션 진행
    for today in tqdm(all_dates):
        today_ts = pd.Timestamp(today)
        
        # -----------------------------------------------------
        # PHASE 1: [청산] 보유 종목 매도 조건 체크
        # -----------------------------------------------------
        for ticker in list(portfolio.keys()):
            # 시세 데이터 로드 (캐시 우선 확인)
            if ticker not in price_cache:
                price_cache[ticker] = load_from_parquet(ticker, str(PARQUET_DIR) + "\\")
            
            df = price_cache[ticker]
            if today_ts not in df.index: continue
            
            curr_data = df.loc[today_ts]
            curr_price = float(curr_data['Close'])
            info = portfolio[ticker]
            
            # 보유 정보 업데이트
            info['days_held'] += 1
            info['highest_price'] = max(info['highest_price'], float(curr_data['High']))
            
            entry_p = info['entry_price']
            curr_ret = (curr_price - entry_p) / entry_p         # 현재 수익률
            max_ret = (info['highest_price'] - entry_p) / entry_p # 매수 이후 기록한 최고 수익률
            
            # === [수비 로직] 유동적 손절가(Stop Price) 계산 ===
            # 초기 하방 방어선 설정
            stop_price = entry_p * (1 - CONFIG['STOP_LOSS_PCT']) 
            
            # 1단계 백스탑: 수익이 일정 수준 이상이면 매수가로 손절가 상향 (본전 보존)
            if max_ret >= CONFIG['BACKSTOP_1_TRIGGER']:
                stop_price = max(stop_price, entry_p)
            
            # 2단계 백스탑: 수익이 더 커지면 수익의 일부를 보전하도록 손절가 상향 (이익 잠금)
            if max_ret >= CONFIG['BACKSTOP_2_TRIGGER']:
                stop_price = max(stop_price, entry_p * (1 + CONFIG['BACKSTOP_2_LOCK_PCT']))
            
            sell_reason = None
            
            # === [매도 조건 판정] 우선순위별 체크 ===
            # 1. 가격이 손절가 또는 백스탑 라인 이탈 시 (수비)
            if curr_price <= stop_price:
                sell_reason = "Stop_Loss_or_Backstop"
            # 2. 매수 직후 단기 급등 발생 시 (클라이맥스 런)
            elif info['days_held'] <= CONFIG['CLIMAX_RUN_DAYS'] and curr_ret >= CONFIG['CLIMAX_RUN_TRIGGER']:
                sell_reason = "Climax_Run"
            # 3. 매수 후 일정 기간 동안 반응이 없을 시 (타임 스탑)
            elif info['days_held'] >= CONFIG['TIME_STOP_DAYS'] and curr_ret < CONFIG['TIME_STOP_MIN_RET']:
                sell_reason = "Time_Stop"
            # 4. 중기 추세선(예: 50일선) 이탈 시 (기술적 청산)
            elif curr_price < curr_data.get(CONFIG['TECH_EXIT_MA'], 0):
                sell_reason = f"{CONFIG['TECH_EXIT_MA']}_Break"
            
            # 매도 확정 시 계좌 정보 갱신 및 포트폴리오 제외
            if sell_reason:
                # 부분 익절 여부에 따른 최종 실현 수익률 산출
                p_ratio = CONFIG['PARTIAL_EXIT_RATIO']
                realized_ret = (CONFIG['PARTIAL_EXIT_TRIGGER'] * p_ratio + curr_ret * (1 - p_ratio)) if info['half_sold'] else curr_ret
                
                trade_log.append({'Ticker': ticker, 'Profit': realized_ret, 'Reason': sell_reason, 'Entry_Date': info['entry_date']})
                sell_reasons[sell_reason] = sell_reasons.get(sell_reason, 0) + 1
                
                cash += info['shares'] * curr_price # 전량 매도 및 현금화
                del portfolio[ticker]
                continue
            
            # === [공격 로직] 수익 목표 도달 시 분할 매도 (Partial Profit Taking) ===
            if not info['half_sold'] and curr_ret >= CONFIG['PARTIAL_EXIT_TRIGGER']:
                sell_shares = int(info['shares'] * CONFIG['PARTIAL_EXIT_RATIO'])
                if sell_shares > 0:
                    cash += sell_shares * curr_price
                    info['shares'] -= sell_shares
                    info['half_sold'] = True

        # -----------------------------------------------------
        # PHASE 2: [진입] 매수 신호 탐색 및 포트폴리오 구축
        # -----------------------------------------------------
        # 현재 총 평가 자산 계산 (현금 + 보유 주식 가치)
        current_equity = float(cash)
        for t, p_info in portfolio.items():
            if today_ts in price_cache[t].index:
                current_equity += p_info['shares'] * price_cache[t].loc[today_ts, 'Close']

        # 빈 슬롯이 있는 경우에만 신규 매수 탐색
        if len(portfolio) < CONFIG['MAX_SLOTS']:
            # 상대강도(RS)가 높은 주도주 순으로 후보 정렬
            candidates = universe[universe['Date'] == today].sort_values('RS', ascending=False)
            for _, row in candidates.iterrows():
                if len(portfolio) >= CONFIG['MAX_SLOTS']: break
                ticker = row['Code']
                if ticker in portfolio: continue
                
                # 미너비니의 기술적 필터(8가지 조건 및 VCP) 통과 확인
                if check_all_minervini_filters(ticker, today_ts, data_dir=str(PARQUET_DIR) + "\\"):
                    if ticker not in price_cache:
                        price_cache[ticker] = load_from_parquet(ticker, str(PARQUET_DIR) + "\\")
                    df = price_cache[ticker]
                    if today_ts in df.index:
                        buy_price = float(df.loc[today_ts, 'Close'])
                        if buy_price <= 0: continue

                        # 집중 투자를 위한 비중 계산 (자산의 25%)
                        target_amt = current_equity * CONFIG['MAX_POS_WEIGHT']
                        actual_amt = min(target_amt, cash)
                        shares = int(actual_amt // buy_price)
                        
                        if shares > 0:
                            portfolio[ticker] = {
                                'shares': shares, 
                                'entry_price': buy_price, 
                                'entry_date': today_ts,
                                'highest_price': buy_price, 
                                'days_held': 0, 
                                'half_sold': False
                            }
                            cash -= shares * buy_price

        # -----------------------------------------------------
        # PHASE 3: [기록] 일별 성과 데이터 저장
        # -----------------------------------------------------
        asset_details = []
        for t, p_info in portfolio.items():
            val = p_info['shares'] * price_cache[t].loc[today_ts, 'Close']
            asset_details.append((t, val))
        
        # 일별 자산 리포트 생성
        record = {'DATE': today_ts.date(), '자산총액': int(current_equity), '현금총액': int(cash)}
        for i in range(CONFIG['MAX_SLOTS']):
            if i < len(asset_details):
                record[f'자산{i+1}_티커'], record[f'자산{i+1}_총액'] = asset_details[i]
            else:
                record[f'자산{i+1}_티커'], record[f'자산{i+1}_총액'] = "None", 0
        history.append(record)

    # -----------------------------------------------------
    # 4. 결과 분석 및 성과 요약 리포트 출력
    # -----------------------------------------------------
    result_df = pd.DataFrame(history)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("      백테스팅 성과 요약 리포트 ")
    print("="*50)
    
    final_val = result_df['자산총액'].iloc[-1]
    total_ret = (final_val / CONFIG['INITIAL_CASH'] - 1) * 100
    days = (pd.to_datetime(CONFIG['END_DATE']) - pd.to_datetime(CONFIG['START_DATE'])).days
    cagr = ((final_val / CONFIG['INITIAL_CASH']) ** (365/days) - 1) * 100
    
    # 최대 낙폭(MDD) 계산
    result_df['Peak'] = result_df['자산총액'].cummax()
    result_df['DD'] = (result_df['자산총액'] - result_df['Peak']) / result_df['Peak']
    mdd = result_df['DD'].min() * 100

    print(f"- 최종 자산: {final_val:,}원")
    print(f"- 누적 수익률: {total_ret:.2f}% / CAGR: {cagr:.2f}%")
    print(f"- 최대 낙폭(MDD): {mdd:.2f}%")
    print("-" * 50)

    # 매매 통계 데이터 분석 (승률 및 손익비)
    if trade_log:
        t_df = pd.DataFrame(trade_log)
        win_rate = (len(t_df[t_df['Profit'] > 0]) / len(t_df)) * 100
        avg_prof = t_df[t_df['Profit'] > 0]['Profit'].mean() * 100
        avg_loss = t_df[t_df['Profit'] <= 0]['Profit'].mean() * 100
        print(f"- 승률: {win_rate:.2f}% / 손익비: {abs(avg_prof/avg_loss):.2f}")
    
    print("-" * 50)
    print("- 매도 사유별 통계:")
    for reason, count in sell_reasons.items():
        print(f"   - {reason}: {count}회")
    print("="*50)
    # 종료된 매매 내역을 별도 CSV로 저장
    trade_log_df = pd.DataFrame(trade_log)
    trade_log_df.to_csv(BASE_DIR / "strategies" / "Trade_Log.csv", index=False)

if __name__ == "__main__":
    run_backtest()