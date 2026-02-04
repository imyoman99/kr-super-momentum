import pandas as pd
import sys
import os
import shutil
from pathlib import Path

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
RESULTS_BASE_DIR = BASE_DIR / "results"

# 외부 분석 모듈 참조를 위한 시스템 경로 추가
sys.path.append(str(ANALYTICS_DIR))

# 사용자 정의 분석 모듈 임포트
try:
    from performance import PerformanceAnalyzer
    from visualizer import Visualizer
    from minervini_filter import load_from_parquet
except ImportError:
    print("[ERROR] analytics 폴더 내 performance.py 또는 visualizer.py를 찾을 수 없습니다.")
    sys.exit()

def get_next_test_dir(base_path):
    """
    기존 결과 폴더들을 조사하여 중복되지 않는 다음 번호의 폴더명을 생성합니다.
    예: Test 1, Test 2가 존재하면 'Test 3' 폴더를 생성하고 경로를 반환합니다.
    
    매개변수:
    - base_path: 결과 폴더들이 생성될 루트 경로
    
    반환값:
    - new_dir: 새로 생성된 테스트 폴더의 Path 객체
    """
    if not base_path.exists():
        base_path.mkdir(parents=True)
    
    i = 1
    while (base_path / f"Test {i}").exists():
        i += 1
    
    new_dir = base_path / f"Test {i}"
    new_dir.mkdir()
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
    
    # 차후 분석 결과와 대조를 위해 사용된 원본 데이터를 해당 폴더에 복사본으로 저장
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
        vis.plot_yearly_monthly_bar(str(save_dir / "6_yearly_monthly_bar.png")) # 연도별 월간 비교 바 차트
        vis.plot_return_dist(str(save_dir / "7_return_dist.png"))         # 일일 수익률 분포도
        print("[SUCCESS] 모든 시각화 차트가 폴더 내 저장되었습니다.")
    except Exception as e:
        # Visualizer 클래스 내 일부 함수가 구현되지 않았거나 데이터 부족 시 오류 예외 처리
        print(f"[WARNING] 시각화 과정 중 일부 항목에서 오류가 발생했습니다: {e}")

    # Backtesting.py 내 main() 함수 중간에 추가

    print("\n[INFO] 주요 매매 사례 분석 차트 생성 중...")
    trade_log_path = BASE_DIR / "strategies" / "Trade_Log.csv"
    if trade_log_path.exists():
        trades = pd.read_csv(trade_log_path)
        # 수익률 높은 순으로 상위 3개 추출
        top_trades = trades.sort_values('Profit', ascending=False).head(3)
        
        for idx, row in top_trades.iterrows():
            ticker = row['Ticker']
            entry_date = pd.to_datetime(row.get('Entry_Date')) # 매수날짜가 저장되어 있다고 가정
            
            # 개별 종목 데이터 로드 (minervini_filter의 함수 재사용)
            df_ticker = load_from_parquet(ticker, str(BASE_DIR / "20160101_20251231_parquet") + "\\")
            
            if df_ticker is not None:
                save_name = save_dir / f"analysis_{ticker}_{idx}.png"
                vis.plot_trade_analysis(ticker, entry_date, df_ticker, str(save_name))

    print("[SUCCESS] 매매 사례 분석 차트가 저장되었습니다.")
    print(f"\n[FINISH] 모든 데이터 분석 절차가 완료되었습니다.")
    print(f"[PATH] 결과물 확인: {save_dir}")

if __name__ == "__main__":
    main()