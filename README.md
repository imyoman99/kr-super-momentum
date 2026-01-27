# K-초수익 모멘텀

**한국형 SEPA(마크 미너비니) 모멘텀 전략 최적화 프로젝트**

---

## 프로젝트 개요

본 프로젝트는 미국 시장에서 검증된 **마크 미너비니(Mark Minervini)의 SEPA 전략**을
한국 주식시장(KOSPI, KOSDAQ)의 구조적 특성에 맞게 재해석·최적화하여
**한국형 모멘텀 인디케이터 및 투자 전략을 개발하고 성과를 검증**하는 것을 목표로 합니다.

단순한 전략 복제가 아닌,
**박스권 장세·짧은 모멘텀 지속성·테마 순환이 잦은 한국 시장** 환경에서도
실질적인 초과수익(Alpha)을 낼 수 있는 전략 구조를 설계합니다.

---

## 핵심 목표

* 한국 시장에 적합한 **SEPA 기반 모멘텀 전략 재설계**
* 펀더멘털 + 기술적 분석을 결합한 종목 선별
* 현실적인 가정 기반의 백테스팅 및 리스크 분석
* 재현 가능한 **모듈형 퀀트 리서치 구조 구축**

---

## 전략 개요

### 1. 펀더멘털 필터

* 수익성, 성장성, 안정성, 밸류에이션 지표 활용
* 조건 필터링 또는 팩터 점수화 방식 적용

### 2. 기술적 분석

* 이동평균, 추세 돌파, 상대강도(RS) 등 모멘텀 지표
* VCP(Volatility Contraction Pattern) 기반 진입 조건

### 3. 포트폴리오 & 리스크 관리

* 종목 수 제한 및 비중 규칙
* 리밸런싱 주기 설정
* 거래비용 및 룩어헤드 바이어스 반영

---

## 프로젝트 구조

```text
.
├── cli/            # 실행 진입점 (데이터 수집, 백테스트 실행)
├── scripts/        # 데이터 다운로드 및 전처리 스크립트
├── src/
│   ├── api/        # 증권사 API 연동
│   ├── backtester/ # 백테스팅 엔진 및 전략 로직
│   ├── indicators/ # 기술적 지표 모듈
│   ├── fundamental/# 기본적 분석 및 필터
│   ├── utils/      # 공용 유틸리티
├── config/         # 전략/모멘텀/리스크 설정 파일
├── tests/          # 테스트 코드
└── README.md
```

---

## 기술 스택

* **Language**: Python
* **Data**: KRX, OpenDART
* **Analysis**: Pandas, NumPy
* **Backtesting & Visualization**: Custom Backtester, Matplotlib

---

## 최종 산출물

* 한국형 SEPA 모멘텀 인디케이터 (Python)
* 전략 백테스팅 성과 리포트
* 종목 필터링 및 매매 시그널 탐지 로직
* 프로젝트 발표 자료 및 시연 영상

---

## 팀 구성 및 역할

* **PM / Dev**: 프로젝트 총괄, API 설계, Git 관리
* **Quant**: 전략 로직, 리스크 관리, 포트폴리오 설계
* **Data**: 데이터 수집 및 전처리
* **Dev / Visualization**: 백테스터 및 성과 분석 시각화

---

## 참고

* Mark Minervini, *Trade Like a Stock Market Wizard*
* SEPA (Specific Entry Point Analysis)

---