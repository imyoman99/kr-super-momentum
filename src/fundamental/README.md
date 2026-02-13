# Fundamental Filter

## 역할 (Role)
- 재무 지표 기반 종목 및 유동성 필터링 로직 정의 및 코드 구현

## 담당 (Owner)
- 임동환 (Quant/Dev)

## 주요 내용 (What)
1. 주요 기능 (Features)
멀티 팩터 스코어링 (Multi-Factor Scoring)

RS Score (50%): 시장 대비 상대적 주가 강세 (모멘텀)

Revenue Growth (30%): 전년 동기 대비 매출 성장률 (성장성)

Operating Margin (20%): 영업이익률 (수익성)

각 팩터는 일별 Z-Score 정규화 후 가중 합산되어 0~100점의 total_score로 산출됩니다.

스마트 필터링 (Smart Filtering)

유동성: 시가총액 500억 이상 & 일 거래대금 5억 이상

재무 건전성: 영업이익률 > 0 (흑자 기업)

구간별 유연한 기준 적용:

2016년 ~ 2020년: 시장 침체기를 반영하여 '매출 성장률 > 0' 조건 면제 (영업이익 흑자만 충족하면 통과)

2021년 ~ 현재: '매출 성장률 > 0' 조건 필수 적용 (고성장 종목 선별 강화)

한국형 분기 대응 (Quarterly Handling)

한국 주식 시장의 실적 발표 마감일(5/15, 8/14, 11/14, 3/31)을 기준으로 1Q, 2Q, 3Q, 4Q 라벨링을 자동 적용하여 시점 편향(Look-ahead Bias)을 방지합니다.

2. 프로젝트 구조 (Directory Structure)
bash
PROJECT_ROOT/
├── data/
│   ├── *.parquet              # 일별 주가 및 재무 데이터 (종목별 파일)
│   └── intersection_rs.csv    # 사전 계산된 RS 점수 파일
├── src/
│   └── fundamental/
│       └── screener.py        # 메인 스크리너 코드 (본 파일)
└── README.md
3. 설치 및 실행 (Installation & Usage)
요구 사항 (Requirements)

Python 3.8+

pandas, numpy, tqdm, pyarrow

실행 방법
터미널에서 screener.py가 있는 경로로 이동 후 실행합니다.

bash
python src/fundamental/screener.py
실행 결과

스크리닝 결과는 src/fundamental/scored_strategy_result_daily.csv 경로에 저장됩니다.

주요 컬럼: date, ticker, name, total_score, rs_score, rev_yoy, op_margin 등

4. 데이터 로직 상세 (Logic Details)
데이터 로딩: data/ 폴더 내의 모든 .parquet 파일을 병렬로 로드합니다.

안전 장치: 파일명에서 티커 추출, 컬럼명 표준화(소문자), 필수 컬럼(date, name) 자동 보정 기능 포함.

RS 병합: intersection_rs.csv 파일과 날짜/티커 기준으로 Inner Join 합니다.

지표 계산:

op_margin: 영업이익 / 매출액

rev_yoy: 250일 전 대비 매출액 변화율 (Trailing YoY)

조건부 필터링:

공통: op_margin > 0, marcap >= 500억, trading_value >= 5억

기간별 분기:

2016~2020: 매출 성장 조건 Pass (시장 상황 반영)

2021~: rev_yoy > 0 필수
