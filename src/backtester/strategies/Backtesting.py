import pandas as pd
import sys
import os
import shutil
import json
import datetime
import uuid
from pathlib import Path

# =========================================================
# [수정 1] 사용자 요청에 맞춰 설정 파일 Import 변경
# =========================================================
try:
    from Asset_Preprocessing_v3 import CONFIG
except ImportError:
    print("[WARNING] Asset_Preprocessing_v3.py를 찾을 수 없습니다. 빈 설정을 사용합니다.")
    CONFIG = {}

try:
    from minervini_filter_v2 import MINERVINI_CONFIG
except ImportError:
    # v2가 없으면 혹시 모를 구버전 시도 혹은 빈 값
    try:
        from minervini_filter import MINERVINI_CONFIG
    except ImportError:
        print("[WARNING] minervini_filter_v2.py 설정을 찾을 수 없습니다.")
        MINERVINI_CONFIG = {}

# =========================================================
# 1. 경로 및 환경 설정
# =========================================================
# 백테스터 엔진 및 결과 데이터가 위치한 기본 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent

# 백테스팅 실행 결과 파일 경로 (입력 데이터)
RESULT_PATH = BASE_DIR / "strategies" / "Asset_List_Final.csv"

# 성과 분석 및 시각화 모듈이 위치한 경로
ANALYTICS_DIR = BASE_DIR / "analytics"

# 분석 결과물(차트, 리포트)을 순차적으로 저장할 루트 폴더
RESULTS_BASE_DIR = BASE_DIR.parent.parent / "results"

# 외부 분석 모듈 참조를 위한 시스템 경로 추가
sys.path.append(str(ANALYTICS_DIR))

# 사용자 정의 분석 모듈 임포트
try:
    from performance import PerformanceAnalyzer
    from visualizer import Visualizer
    # minervini_filter_v2에 load_from_parquet가 있다고 가정 (없으면 수정 필요)
    try:
        from minervini_filter_v2 import load_from_parquet
    except ImportError:
        from minervini_filter import load_from_parquet
except ImportError:
    print("[ERROR] analytics 폴더 내 performance.py 또는 visualizer.py를 찾을 수 없습니다.")
    sys.exit()

def get_next_test_dir(base_path):
    """
    날짜와 랜덤 해시를 조합하여 고유한 결과 폴더명을 생성합니다.
    예: 20260204_a9f3c2
    """
    if not base_path.exists():
        base_path.mkdir(parents=True)
    
    # 1. 현재 날짜 (YYYYMMDD)
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    
    # 2. 랜덤 해시 생성 (6자리)
    random_hash = uuid.uuid4().hex[:6]
    
    folder_name = f"{date_str}_{random_hash}"
    new_dir = base_path / folder_name
    
    # 3. 중복 방지 안전장치
    while new_dir.exists():
        random_hash = uuid.uuid4().hex[:6]
        new_dir = base_path / f"{date_str}_{random_hash}"
        
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir

def main():
    """
    백테스팅 결과 파일을 로드하여 지표 계산, 리포트 생성, 차트 시각화를 수행하고
    이를 별도의 테스트 폴더에 통합 저장합니다.
    """
    
    # --- 단계 1: 데이터 로드 및 환경 준비 ---
    if not RESULT_PATH.exists():
        print(f"[ERROR] 분석할 결과 파일이 존재하지 않습니다: {RESULT_PATH}")
        return

    # 새로운 실험 결과 저장을 위한 자동 넘버링 폴더 생성
    save_dir = get_next_test_dir(RESULTS_BASE_DIR)
    print(f"[INFO] 신규 결과 폴더 생성 완료: {save_dir}")

    print(f"[INFO] {RESULT_PATH.name} 데이터 로딩 및 전처리 중...")
    df = pd.read_csv(RESULT_PATH)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    
    # 원본 데이터 복사
    shutil.copy(RESULT_PATH, save_dir / "Source_Data.csv")

    # --- 단계 2: 성과 지표 분석 ---
    # PerformanceAnalyzer 인스턴스 생성 (자산총액 컬럼 기준)
    analyzer = PerformanceAnalyzer(df, price_col='자산총액')
    metrics = analyzer.calculate_metrics()
    
    # 분석된 지표들을 텍스트 파일(report.txt)로 기록
    with open(save_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("=== BACKTESTING PERFORMANCE REPORT ===\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    # 콘솔에 요약 리포트 출력
    analyzer.print_report(metrics)

    # --- 단계 3: 데이터 시각화 및 차트 저장 ---
    print("\n[INFO] 시각화 차트 생성 및 저장 중...")
    vis = Visualizer(df, price_col='자산총액')
    
    try:
        # 각 항목별 성과 차트 생성 및 경로 지정 저장
        vis.plot_equity_curve(str(save_dir / "1_equity_curve.png"))       # 자산 곡선
        vis.plot_drawdown(str(save_dir / "2_drawdown.png"))               # 최대 낙폭(MDD)
        vis.plot_monthly_heatmap(str(save_dir / "3_monthly_heatmap.png")) # 월별 수익률 히트맵
        vis.plot_seasonality(str(save_dir / "4_seasonality.png"))         # 월별 평균 수익률(계절성)
        vis.plot_yearly_return(str(save_dir / "5_yearly_return.png"))     # 연도별 수익률 요약
        vis.plot_return_dist(str(save_dir / "6_return_dist.png"))         # 일일 수익률 분포도
        print("[SUCCESS] 모든 시각화 차트가 폴더 내 저장되었습니다.")
    except Exception as e:
        print(f"[WARNING] 시각화 과정 중 일부 항목에서 오류가 발생했습니다: {e}")

    # --- 단계 4: 매매 사례 분석 ---
    print("\n[INFO] 주요 매매 사례 분석 차트 생성 중...")
    trade_log_path = BASE_DIR / "strategies" / "Trade_Log.csv"
    if trade_log_path.exists():
        trades = pd.read_csv(trade_log_path)
        # 수익률 높은 순으로 상위 3개 추출
        top_trades = trades.sort_values('Net_PnL', ascending=False).head(3)
        
        for idx, row in top_trades.iterrows():
            ticker = str(row['Ticker']).zfill(6)
            # Entry_Date가 있는지 확인
            if 'Entry_Date' in row:
                entry_date = pd.to_datetime(row['Entry_Date'])
            else:
                continue

            # 개별 종목 데이터 로드 (경로 주의: 기존 코드의 경로 로직 유지)
            # 주의: parquet 폴더 경로가 사용자 환경에 맞는지 확인 필요
            parquet_folder = BASE_DIR.parent.parent / "data" # data 폴더 추정
            
            df_ticker = load_from_parquet(ticker, str(parquet_folder) + "\\")
            
            if df_ticker is not None:
                save_name = save_dir / f"analysis_{ticker}_{idx}.png"
                vis.plot_trade_analysis(ticker, entry_date, df_ticker, str(save_name))

    print("[SUCCESS] 매매 사례 분석 차트가 저장되었습니다.")

    # --- 단계 5: 설정값 저장 (수정된 부분) ---
    configs_to_save = {
        "BACKTEST_CONFIG": CONFIG,
        "MINERVINI_FILTER_CONFIG": MINERVINI_CONFIG,
        "GENERATED_AT": str(datetime.datetime.now())
    }
    
    with open(save_dir / "used_configs.json", "w", encoding="utf-8") as f:
        # [핵심 수정] default=str 옵션을 넣어 WindowsPath 객체를 문자열로 자동 변환
        json.dump(configs_to_save, f, indent=4, ensure_ascii=False, default=str)
        
    print(f"[INFO] 설정값 저장 완료: used_configs.json")

    print(f"\n[FINISH] 모든 데이터 분석 절차가 완료되었습니다.")
    print(f"[PATH] 결과물 확인: {save_dir}")

if __name__ == "__main__":
    main()