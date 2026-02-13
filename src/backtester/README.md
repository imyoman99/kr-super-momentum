# 📈 Backtester Module: KR-Super-Momentum

본 디렉토리는 마크 미너비니(Mark Minervini)의 **SEPA(Specific Entry Point Analysis) 전략**을 한국 주식 시장(KOSPI, KOSDAQ)의 특성에 맞춰 구현한 백테스팅 핵심 엔진입니다. 수많은 시행착오 끝에 **미래 참조 편향(Look-ahead Bias)**을 완전히 제거한 `v3` 엔진을 통해 가장 현실적인 시뮬레이션 결과를 제공합니다.

---

## 📂 폴더 구조 및 파일 역할 (Directory Structure)

```text
backtester/
├── analytics/              # [분석] 성과 지표 계산 및 시각화 모듈
│   ├── performance.py      # CAGR, MDD, Sharpe Ratio, Profit Factor 산출
│   └── visualizer.py       # Equity Curve, Drawdown, 월별 히트맵 생성
├── strategies/             # [핵심] 전략 로직 및 시뮬레이션 엔진
│   ├── minervini_filter_v2.py      # Stage 2 상승 국면 및 VCP 필터 엔진
│   ├── Filter_Pivot.py             # 유니버스 선별 및 10일 고가 돌파 피벗 탐지
│   ├── Asset_Preprocessing_v3.py   # [Main] 편향 제거 및 4단계 리스크 관리 엔진
│   ├── Asset_Prepro_v3_future_reference.py # 개발/검증용 대조군 (미래참조 포함 버전)
│   ├── Backtesting.py              # 전체 프로세스 통합 실행 및 리포팅 스크립트
│   ├── Optuna.py                   # 베이지안 최적화 기반 전략 파라미터 튜닝
│   ├── Trade_Log.csv               # 개별 거래의 진입/청산/사유 상세 기록
│   └── Asset_List_Final.csv        # 일자별 총자산 변화 및 포트폴리오 상태
└── portfolio.py            # 자산 배분 및 종목 교체 운용 로직
```

---

## 🛠 주요 모듈 상세 로직

### 1. 기술적 필터 엔진 (`minervini_filter_v2.py`)
마크 미너비니의 **Trend Template**을 한국 시장 수치에 맞게 정량화했습니다.
* **이동평균 정배열**: 현재 주가가 50일, 150일, 200일 이평선 위에 위치하며, MA50 > MA150 > MA200 조건을 검증합니다.
* **가격 위치**: 52주 신고가 대비 10% 이내에 위치하고, 저가 대비 최소 30% 이상 반등한 종목을 선별합니다.
* **VCP 수축**: 최근 20일간의 변동성(ATR)과 거래량이 직전 구간 대비 수축했는지 체크합니다.

### 2. 피벗 탐지 로직 (`Filter_Pivot.py`)
상승 2단계 종목 중 폭발적인 시세가 시작되는 지점을 포착합니다.
* **Pivot Price**: 신호 발생 시점을 기준으로 과거 10일간의 최고가를 돌파 기준으로 설정합니다.
* **중복 방지**: `PIVOT_COOLDOWN_DAYS`를 통해 동일 종목에서 단기간 내 반복 신호가 발생하는 것을 제어합니다.

### 3. 최종 백테스팅 엔진 (`Asset_Preprocessing_v3.py`)
가장 완성도 높은 엔진으로, 실전 매매와 동일한 메커니즘을 가집니다.
* **Look-ahead Bias 제거**: 당일(D) 발생한 신호를 바탕으로 **익일(D+1) 시가**에 매수를 집행하여 데이터 왜곡을 방지했습니다.
* **자산 배분**: 최대 3종목 집중 투자 전략을 사용하며, 종목당 자산의 약 33%(0.33)를 고정 배분합니다.
* **4단계 리스크 관리 (Absolute Rules)**:
    1. **Stop Loss**: 진입가 대비 **-4%** 하락 시 무조건 손절합니다.
    2. **Break-even**: 수익률 **14%** 달성 시 손절가를 매수가 위(+9.8%)로 올려 원금을 보호합니다.
    3. **Trailing Stop**: 수익률 **27%** 도달 시 활성화되며, 고점에서 **1%**만 하락해도 익절하여 수익을 보존합니다.
    4. **Time Stop**: 보유 **14일** 경과 후 수익률이 2% 미만일 경우 기회비용 차원에서 청산합니다.

### 4. 리포팅 및 분석 (`Backtesting.py`)
백테스트 결과물을 시각화하고 상세 리포트를 생성합니다.
* **매매 사례 분석**: 수익률이 높았던 거래들을 선별하여 개별 차트 분석 이미지를 자동 생성합니다.
* **설정 기록**: 테스트에 사용된 `CONFIG` 및 `MINERVINI_CONFIG` 값을 JSON 형태로 저장하여 실험의 재현성을 보장합니다.

---

## 🚀 실행 가이드 (Quick Start)

1. **유니버스 및 피벗 신호 생성**:
   ```bash
   python src/backtester/strategies/Filter_Pivot.py
   ```
2. **백테스트 시뮬레이션 (v3) 실행**:
   ```bash
   python src/backtester/strategies/Asset_Preprocessing_v3.py
   ```
3. **성과 분석 및 대시보드 출력**:
   ```bash
   python src/backtester/strategies/Backtesting.py
   ```

---

## ⚠️ 주의 사항
* `Asset_Prepro_v3_future_reference.py`는 개발 중 발생했던 미래 참조 오류를 확인하기 위한 용도이므로, 실제 성과 분석에는 **v3** 버전을 사용하시기 바랍니다.