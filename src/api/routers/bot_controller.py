from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas.bot import (
    BotRunRequest,
    BotRunResponse,
    FinancialUniverseRequest,
    FinancialUniverseResponse,
    LiveActionResponse,
    LivePositionDeleteRequest,
    LivePositionsUpsertRequest,
    LiveStartRequest,
    LiveStatusResponse,
)
from src.api.services.bot_engine import run_bot_pipeline
from src.api.services.financial_universe import (
    FINANCIAL_UNIVERSE_CONFIG,
    build_financial_universe,
)
from src.api.services.live_trading_engine import ENGINE
from src.api.utils.progress_logger import build_progress_callback, get_progress_logger


router = APIRouter(tags=["bot"])
ROUTER_LOGGER = get_progress_logger("kr_super_momentum.router")


@router.post("/universe/financial", response_model=FinancialUniverseResponse)
def create_financial_universe(
    payload: FinancialUniverseRequest,
) -> FinancialUniverseResponse:
    try:
        config = {
            "use_today": payload.use_today,
            "reference_date": payload.reference_date,
            "financial_data_path": payload.financial_data_path
            or FINANCIAL_UNIVERSE_CONFIG["financial_data_path"],
            "financial_cache_dir": payload.financial_cache_dir
            or FINANCIAL_UNIVERSE_CONFIG.get("financial_cache_dir"),
            "financial_api": {
                "enabled": payload.financial_api_enabled,
                "url": payload.financial_api_url or "",
                "method": payload.financial_api_method,
                "data_key": payload.financial_api_data_key,
                "headers": payload.financial_api_headers,
                "params": payload.financial_api_params,
            },
            "extra_conditions": [
                cond.model_dump() for cond in payload.extra_conditions
            ],
        }
        result = build_financial_universe(config)
        return FinancialUniverseResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/bot/run", response_model=BotRunResponse)
def run_bot(payload: BotRunRequest) -> BotRunResponse:
    try:
        progress_callback = build_progress_callback(ROUTER_LOGGER, prefix="[BOT-RUN]")
        progress_callback("/bot/run 요청 수신")
        result = run_bot_pipeline(
            {
                "mode": payload.mode.value,
                "reference_date": payload.reference_date,
                "financial_data_path": payload.financial_data_path,
                "financial_api": {
                    "enabled": payload.financial_api_enabled,
                    "url": payload.financial_api_url or "",
                    "method": payload.financial_api_method,
                    "data_key": payload.financial_api_data_key,
                    "headers": payload.financial_api_headers,
                    "params": payload.financial_api_params,
                },
                "financial_cache_dir": payload.financial_cache_dir,
                "universe_snapshot_path": payload.universe_snapshot_path,
                "force_rebuild": payload.force_rebuild,
                "extra_financial_conditions": [
                    cond.model_dump() for cond in payload.extra_financial_conditions
                ],
                "ranking": {
                    "min_rs_rating": payload.min_rs_rating,
                    "min_market_cap": payload.min_market_cap,
                    "trading_value_lookback_days": payload.trading_value_lookback_days,
                    "min_avg_daily_trading_value": payload.min_avg_daily_trading_value,
                    "top_n": payload.top_n_stage2,
                    "rs_benchmark_code": payload.rs_benchmark_code,
                },
                "trading": {
                    "account_equity": payload.account_equity,
                    "risk_per_trade": payload.risk_per_trade,
                    "max_position_ratio": payload.max_position_ratio,
                    "stop_atr_multiple": payload.stop_atr_multiple,
                    "take_profit_r": payload.take_profit_r,
                },
                "pivot_window": payload.pivot_window,
            },
            log_callback=progress_callback,
        )
        progress_callback("/bot/run 실행 완료")
        return BotRunResponse(**result)
    except Exception as exc:
        ROUTER_LOGGER.exception("[BOT-RUN] 실행 실패: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/live/start", response_model=LiveStatusResponse)
def start_live_trading(payload: LiveStartRequest) -> LiveStatusResponse:
    try:
        ROUTER_LOGGER.info("[LIVE-API] /live/start 요청 수신")
        result = ENGINE.start(payload.model_dump())
        ROUTER_LOGGER.info("[LIVE-API] /live/start 처리 완료")
        return LiveStatusResponse(**result)
    except Exception as exc:
        ROUTER_LOGGER.exception("[LIVE-API] /live/start 실패: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/live/stop", response_model=LiveStatusResponse)
def stop_live_trading() -> LiveStatusResponse:
    try:
        ROUTER_LOGGER.info("[LIVE-API] /live/stop 요청 수신")
        result = ENGINE.stop()
        ROUTER_LOGGER.info("[LIVE-API] /live/stop 처리 완료")
        return LiveStatusResponse(**result)
    except Exception as exc:
        ROUTER_LOGGER.exception("[LIVE-API] /live/stop 실패: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/live/status", response_model=LiveStatusResponse)
def get_live_status() -> LiveStatusResponse:
    try:
        result = ENGINE.status()
        return LiveStatusResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/live/check/preopen", response_model=dict)
def run_live_preopen_check() -> dict:
    try:
        ROUTER_LOGGER.info("[LIVE-API] /live/check/preopen 요청 수신")
        result = ENGINE.run_preopen_check_once()
        ROUTER_LOGGER.info("[LIVE-API] /live/check/preopen 처리 완료")
        return result
    except Exception as exc:
        ROUTER_LOGGER.exception("[LIVE-API] /live/check/preopen 실패: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/live/check/open", response_model=LiveActionResponse)
def run_live_open_check() -> LiveActionResponse:
    try:
        ROUTER_LOGGER.info("[LIVE-API] /live/check/open 요청 수신")
        result = ENGINE.run_open_check_once()
        ROUTER_LOGGER.info("[LIVE-API] /live/check/open 처리 완료")
        return LiveActionResponse(**result)
    except Exception as exc:
        ROUTER_LOGGER.exception("[LIVE-API] /live/check/open 실패: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/live/check/monitor", response_model=LiveActionResponse)
def run_live_monitor_check() -> LiveActionResponse:
    try:
        ROUTER_LOGGER.info("[LIVE-API] /live/check/monitor 요청 수신")
        result = ENGINE.run_monitor_once()
        ROUTER_LOGGER.info("[LIVE-API] /live/check/monitor 처리 완료")
        return LiveActionResponse(**result)
    except Exception as exc:
        ROUTER_LOGGER.exception("[LIVE-API] /live/check/monitor 실패: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/live/positions", response_model=list[dict])
def get_live_positions() -> list[dict]:
    try:
        return ENGINE.get_positions()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/live/positions/upsert", response_model=list[dict])
def upsert_live_positions(payload: LivePositionsUpsertRequest) -> list[dict]:
    try:
        return ENGINE.upsert_positions(
            [item.model_dump() for item in payload.positions]
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/live/positions/delete", response_model=list[dict])
def delete_live_position(payload: LivePositionDeleteRequest) -> list[dict]:
    try:
        return ENGINE.remove_position(payload.code)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
