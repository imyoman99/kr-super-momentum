import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from minervini_filter_v2 import check_minervini_from_df, load_from_parquet

# [설정]
BASE_DIR = Path(__file__).resolve().parents[3] # 경로 상황에 맞게 조정 필요
DATA_DIR = BASE_DIR / "data"
UNIVERSE_PATH = DATA_DIR / "universe" / "intersection_80.csv"
OUTPUT_FILE = "Pivot_Signals.csv"

# 피벗 재진입 쿨타임 (일): 한 번 신호가 뜨고 나서 N일 동안은 또 뜨더라도 무시 (중복 방지)
PIVOT_COOLDOWN_DAYS = 20 

def main():
    if not UNIVERSE_PATH.exists():
        print(f"[Error] Universe file not found: {UNIVERSE_PATH}")
        return

    # 1. 유니버스 로딩 및 정렬
    print("Loading universe...")
    universe = pd.read_csv(UNIVERSE_PATH, dtype={'Code': str})
    universe['Date'] = pd.to_datetime(universe['Date'])
    
    # 날짜순 -> 종목순 정렬 (순차 처리를 위해)
    universe = universe.sort_values(['Date', 'Code'])
    
    # 데이터를 미리 로드하여 캐싱 (I/O 최적화)
    # 메모리가 부족하면 이 부분을 루프 안에서 로드하는 방식으로 변경해야 함
    print("Pre-loading stock data to memory...")
    unique_codes = universe['Code'].unique()
    data_cache = {}
    
    for code in tqdm(unique_codes, desc="Loading Parquet"):
        try:
            df = load_from_parquet(code, str(DATA_DIR))
            if df is not None:
                data_cache[code] = df
        except Exception as e:
            continue

    pivot_signals = []
    last_signal_date = {} # {ticker: last_pivot_date}

    # 2. 날짜별로 순회하며 피벗 체크
    # 유니버스는 이미 필터링된 후보군이지만, '기술적 타점'은 매일 다를 수 있음
    # 여기서는 유니버스에 등재된 날짜를 기준으로 검사
    
    # 그룹화하여 처리 속도 향상
    grouped = universe.groupby('Date')
    
    print("Scanning for Pivot Points...")
    for date, group in tqdm(grouped, desc="Filtering Pivots"):
        date_ts = pd.Timestamp(date)
        
        for _, row in group.iterrows():
            ticker = row['Code']
            rs_score = row.get('RS', 0) # RS 점수 보존
            
            # 데이터 존재 확인
            if ticker not in data_cache:
                continue
            
            df = data_cache[ticker]
            
            # (1) 쿨타임 체크: 최근에 이미 신호가 떴었다면 패스 (추격 매수 방지 핵심)
            if ticker in last_signal_date:
                days_since = (date_ts - last_signal_date[ticker]).days
                if days_since < PIVOT_COOLDOWN_DAYS:
                    continue

            # (2) 미너비니 조건 정밀 검사
            # check_minervini_from_df는 이동평균선 정배열, 신고가 근접 등을 체크함
            if check_minervini_from_df(df, date_ts):
                
                # (3) 추가 필터: 거래량 폭발 여부 (선택 사항)
                # Pivot은 보통 거래량을 동반해야 신뢰도가 높음
                try:
                    vol_today = df.loc[date_ts, 'Volume']
                    vol_avg = df.loc[:date_ts]['Volume'].iloc[-20:-1].mean() # 직전 20일 평균
                    
                    # 평균 거래량 대비 1.5배 이상 터졌을 때만 인정 (옵션)
                    # if vol_today < vol_avg * 1.5: continue 
                except:
                    pass

                # 신호 포착!
                pivot_signals.append({
                    'Date': date_ts.date(),
                    'Code': ticker,
                    'RS': rs_score,
                    'Close': df.loc[date_ts, 'Close']
                })
                
                # 마지막 신호 날짜 갱신 (쿨타임 시작)
                last_signal_date[ticker] = date_ts

    # 3. 결과 저장
    if pivot_signals:
        result_df = pd.DataFrame(pivot_signals)
        result_df = result_df.sort_values(['Date', 'RS'], ascending=[True, False])
        result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print(f"\n[Success] Process completed.")
        print(f"Total Pivot Signals Found: {len(result_df)}")
        print(f"Saved to: {OUTPUT_FILE}")
        
        # 샘플 출력
        print("\n[Sample Signals]")
        print(result_df.head())
    else:
        print("\n[Result] No pivot signals found matching the criteria.")

if __name__ == "__main__":
    main()