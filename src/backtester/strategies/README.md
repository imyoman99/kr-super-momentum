# KR Super Momentum Project

## Overview
KR Super Momentum 프로젝트는 Minervini 트렌드 템플릿을 기반으로 한 슈퍼 모멘텀 전략을 구현하는 프로젝트입니다. 펀더멘털 분석으로 선정된 종목들을 대상으로 기술적 필터를 적용하여 고성장 잠재력을 가진 종목을 선별합니다.

## Data Usage
- `growth_strategy_result.csv`: 펀더멘털 분석으로 선정된 종목들의 데이터
- `./data/Parquet 파일`: 선정된 종목들의 종가(Close), 거래량(Volume), 이동평균(MA20, MA50, MA150, MA200), ATR(ATR_14) 등의 기술적 지표 데이터를 사용하여 분석 및 필터링 수행

## Files and Functions

### minervini_filter.py
Minervini 트렌드 템플릿의 기술적 필터를 구현한 모듈입니다.

- `load_from_parquet(code, data_dir='./data/')`: 특정 종목 코드의 Parquet 파일을 로드하여 데이터프레임을 반환합니다. Date를 인덱스로 설정하고, 필요한 MA 컬럼(MA20, MA50, MA150, MA200)과 ATR_14가 없으면 계산하여 추가합니다.
- `check_price_above_mas(close, ma50, ma150, ma200)`: 조건 1: 가격이 50일/150일/200일 이동평균선 위에 있음
- `check_ma_alignment(ma50, ma150, ma200)`: 조건 2: 50일 이동평균선 > 150일 이동평균선 > 200일 이동평균선 정렬
- `check_200ma_up(ma200_series)`: 조건 3: 200일 이동평균선이 20일 전보다 상승 중
- `check_50ma_up(ma50_series)`: 조건 4: 50일 이동평균선이 5일 전보다 상승 중
- `check_within_52w_high(close_series, max_distance_from_high)`: 조건 5: 52주 최고가의 25% 이내에 위치
- `check_above_52w_low(close_series, min_distance_from_low)`: 조건 6: 52주 최저가의 30% 이상 상승
- `check_price_above_10ma(close, ma20)`: 조건 7: 가격이 20일 이동평균선 위에 있음 (MA10 대신 MA20 사용)
- `check_150ma_up(ma150_series)`: 조건 8: 150일 이동평균선이 10일 전보다 상승 중
- `check_sufficient_volume(volume_series)`: 조건 9: 현재 거래량이 최근 10일 거래량 중 가장 큰 값인지 확인
- `check_volatility_contraction(close_series, atr_series=None)`: 조건 10: VCP: 변동성 수축 (ATR 기반 또는 std 기반). ATR이 있으면 최근 10일 ATR 평균 < 이전 10일 ATR 평균, 없으면 종가의 표준편차로 대체.
- `check_volume_contraction(volume_series)`: 조건 11: VCP: 거래량 수축 (최근 10일 평균 거래량 < 이전 10일 평균 거래량)
- `check_all_minervini_filters(code, date, ...)`: 특정 종목과 날짜에 대해 모든 Minervini 필터를 체크합니다. 모든 조건이 만족되면 True를 반환합니다.

### minervini_vis.py
Minervini 필터 결과를 시각화하는 모듈입니다.

- `MinerviniVisualizer.__init__(data_dir, start_date, end_date)`: 시각화 클래스를 초기화합니다.
- `visualize_stocks_at_date(target_date, passed_stocks)`: 특정 날짜에 필터를 통과한 종목들의 가격과 거래량 차트를 표시합니다.
- `run_visualization(target_date, passed_stocks)`: 시각화를 실행합니다.
- `visualize_stock_entry_points(stock_code, entry_dates)`: 특정 종목의 전체 기간 차트에 진입 시점을 표시합니다.

### Minervini_tec_part.ipynb
Minervini 전략의 기술적 분석 부분을 다루는 Jupyter 노트북입니다. 필터 적용과 결과 분석을 포함합니다.

### CSV Files
- `growth_strategy_result.csv`: 펀더멘털 분석 결과
- `minervini_filter_results.csv`: Minervini 필터 적용 결과

## Logic Flow
1. `growth_strategy_result.csv`에서 펀더멘털로 선정된 종목 리스트를 가져옵니다.
2. 각 종목에 대해 `minervini_filter.py`의 `check_all_minervini_filters` 함수를 사용하여 기술적 필터를 적용합니다.
3. 필터를 통과한 종목들을 `minervini_filter_results.csv`에 저장합니다.
4. `minervini_vis.py`의 시각화 함수들을 사용하여 결과 차트를 생성하고 진입 시점을 표시합니다.
5. `Minervini_tec_part.ipynb`에서 전체 분석 과정을 노트북으로 실행하고 결과를 검토합니다.
