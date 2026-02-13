
# src/api/ 폴더 구조 및 역할

이 디렉토리는 전체 파이프라인의 API 서버, 유니버스 생성, 실거래/모의투자, 슬랙 알림 등 핵심 자동화 로직을 담당합니다.

## 전체 구조

- app.py : FastAPI 기반 메인 엔트리포인트. REST API 서버 구동, 라우터 등록, 오케스트레이션 담당
- routers/
	- bot_controller.py : 주요 API 엔드포인트(유니버스 생성, 봇 실행, 라이브 엔진 등) 구현
- schemas/
	- bot.py : API 요청/응답 데이터 모델(pydantic 기반) 정의
- services/
	- bot_engine.py : 전체 파이프라인 실행(유니버스 생성, 랭킹, 미너비니 필터, 트레이딩 후보 산출 등)
	- bot_engine_common.py : 날짜/경로/로그/스테이지 파일 등 공통 유틸리티
	- bot_engine_trading.py : 트레이딩 후보 생성, 포지션 체크 등 트레이딩 로직 분리
	- financial_universe.py : 1차 재무 유니버스 생성, 재무 API/CSV 처리, 분기 매핑 등
	- live_trading_engine.py : 실시간/모의 라이브 엔진(포지션 관리, 장중 체크, 실거래 연동)
- utils/
	- config_loader.py : .env 및 kis_live.yaml 환경설정 로더
	- kis_client.py : 키움/증권사 API 연동, 인증/주문/계좌조회 등
	- progress_logger.py : 표준 로그 및 콜백 유틸
	- slack_notifier.py : 슬랙 웹훅 알림 유틸
- __init__.py : 패키지 초기화

## 각 파일/폴더별 상세 역할

### app.py
- FastAPI 서버 구동, 라우터 등록, 전체 파이프라인 오케스트레이션
- REST 엔드포인트: /health, /app/run-all 등

### routers/
- bot_controller.py: 주요 API 엔드포인트(1차/2차/3차 유니버스 생성, 봇 실행, 라이브 엔진 제어 등)

### schemas/
- bot.py: API 요청/응답 데이터 모델 정의 (pydantic 기반, 유니버스/트레이딩/라이브 관련)

### services/
- bot_engine.py: 전체 파이프라인 실행(유니버스 생성, 랭킹, 미너비니 필터, 트레이딩 후보 산출 등)
- bot_engine_common.py: 날짜/경로/로그/스테이지 파일 등 공통 유틸리티
- bot_engine_trading.py: 트레이딩 후보 생성, 포지션 체크 등 트레이딩 로직 분리
- financial_universe.py: 1차 재무 유니버스 생성, 재무 API/CSV 처리, 분기 매핑 등
- live_trading_engine.py: 실시간/모의 라이브 엔진(포지션 관리, 장중 체크, 실거래 연동)

### utils/
- config_loader.py: .env 및 kis_live.yaml 환경설정 로더
- kis_client.py: 키움/증권사 API 연동, 인증/주문/계좌조회 등
- progress_logger.py: 표준 로그 및 콜백 유틸
- slack_notifier.py: 슬랙 웹훅 알림 유틸

---

## 주요 API/파이프라인 요약

- 1차 재무 유니버스: POST /universe/financial (영업이익 등 재무 필터)
- 전체 파이프라인 실행: POST /bot/run (유니버스→랭킹→미너비니→트레이딩 후보)
- 실시간/모의 라이브 엔진: holdings.json 기반 포지션 관리, 장중 매매 체크
- 슬랙 알림: 파이프라인/트레이딩 로그 실시간 전송

---

문의: 유영민 (PM/Dev)
