from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.routers.bot_controller import router as financial_universe_router
from src.api.schemas.bot import BotRunRequest
from src.api.services.bot_engine import run_bot_pipeline
from src.api.services.live_trading_engine import ENGINE
from src.api.utils.progress_logger import build_progress_callback, get_progress_logger


app = FastAPI(title="KR Super Momentum API")
PIPELINE_LOGGER = get_progress_logger("kr_super_momentum.app")

DEFAULT_RUN_PAYLOAD: dict[str, Any] = {
    "start_live": False,
    "upsert_from_candidates": True,
    "bot": {
        "mode": "trading",
        "reference_date": "2026-02-12",
        "force_rebuild": False,
    },
    "live": {
        "holdings_path": "data/runtime/holdings.json",
        "loop_interval_sec": 2.0,
        "market_open_hhmm": "09:00",
        "market_close_hhmm": "15:30",
    },
}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


def _build_bot_config(payload: BotRunRequest) -> dict[str, Any]:
    dart_api_key = os.getenv("OPEN_DART_API_KEY", "") or os.getenv("DART_API_KEY", "")
    config: dict[str, Any] = {
        "mode": payload.mode.value,
        "reference_date": payload.reference_date,
        "financial_api": {
            "enabled": payload.financial_api_enabled,
            "url": payload.financial_api_url or "",
            "method": payload.financial_api_method,
            "data_key": payload.financial_api_data_key,
            "headers": payload.financial_api_headers,
            "params": payload.financial_api_params,
            "provider": "opendart" if dart_api_key else "",
            "api_key": dart_api_key,
        },
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
    }

    if payload.financial_data_path is not None:
        config["financial_data_path"] = payload.financial_data_path
    if payload.financial_cache_dir is not None:
        config["financial_cache_dir"] = payload.financial_cache_dir
    if payload.universe_snapshot_path is not None:
        config["universe_snapshot_path"] = payload.universe_snapshot_path
    else:
        default_snapshot = (
            PROJECT_ROOT / "data" / "universe" / "intersection_80_minervini4.csv"
        )
        if not default_snapshot.exists():
            universe_dir = PROJECT_ROOT / "data" / "universe"
            candidates = (
                sorted(universe_dir.glob("intersection*.csv"))
                if universe_dir.exists()
                else []
            )
            if candidates:
                config["universe_snapshot_path"] = str(
                    candidates[-1].relative_to(PROJECT_ROOT)
                ).replace("\\", "/")

    return config


@app.post("/app/run-all")
def run_all(payload: dict[str, Any]) -> dict[str, Any]:
    """
    app.py 단일 진입점 오케스트레이션:
    1) 봇 파이프라인 실행
    2) ready 종목을 holdings에 업서트
    3) 장전 점검 1회 실행
    4) 옵션으로 라이브 엔진 시작
    """
    try:
        progress_callback = build_progress_callback(
            PIPELINE_LOGGER, prefix="[PIPELINE]"
        )
        progress_callback("run_all 오케스트레이션 시작")

        raw_bot = dict(payload.get("bot", {}))
        if "financial_api_enabled" not in raw_bot:
            raw_bot["financial_api_enabled"] = True
        if not raw_bot.get("financial_api_url"):
            raw_bot["financial_api_url"] = os.getenv("FINANCIAL_API_URL", "")
        if not raw_bot.get("financial_api_method"):
            raw_bot["financial_api_method"] = os.getenv("FINANCIAL_API_METHOD", "GET")
        if not raw_bot.get("financial_api_data_key"):
            raw_bot["financial_api_data_key"] = os.getenv(
                "FINANCIAL_API_DATA_KEY", "data"
            )

        bot_payload = BotRunRequest(**raw_bot)
        bot_config = _build_bot_config(bot_payload)

        try:
            bot_result = run_bot_pipeline(bot_config, log_callback=progress_callback)
        except Exception as exc:
            msg = str(exc)
            if "재무 데이터 파일이 없습니다" in msg:
                progress_callback("재무 CSV 미존재 감지, API 모드로 재시도 준비")
                retry_config = {**bot_config}
                api_conf = {**retry_config.get("financial_api", {})}
                api_conf["enabled"] = True
                api_conf["url"] = api_conf.get("url") or os.getenv(
                    "FINANCIAL_API_URL", ""
                )
                if not api_conf.get("url"):
                    dart_key = os.getenv("OPEN_DART_API_KEY", "") or os.getenv(
                        "DART_API_KEY", ""
                    )
                    if dart_key:
                        api_conf["provider"] = "opendart"
                        api_conf["api_key"] = dart_key
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "재무 CSV가 없고 API URL도 없습니다. FINANCIAL_API_URL 또는 OPEN_DART_API_KEY를 설정해 주세요."
                            ),
                        ) from exc
                elif not api_conf.get("provider"):
                    api_conf["provider"] = ""

                if not api_conf.get("url") and api_conf.get("provider") != "opendart":
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "재무 CSV가 없어서 API 재시도를 시도했지만 FINANCIAL_API_URL이 비어 있습니다. "
                            "환경변수 FINANCIAL_API_URL을 설정해 주세요."
                        ),
                    ) from exc
                retry_config["financial_api"] = api_conf
                bot_result = run_bot_pipeline(
                    retry_config,
                    log_callback=progress_callback,
                )
            else:
                raise

        upsert_from_candidates = bool(payload.get("upsert_from_candidates", True))
        upserted_positions: list[dict[str, Any]] = []

        if upsert_from_candidates:
            progress_callback("ready 후보를 holdings로 업서트하는 단계 시작")
            candidates = bot_result.get("trade_candidates", [])
            positions: list[dict[str, Any]] = []
            for item in candidates:
                if str(item.get("status")) != "ready":
                    continue
                code = str(item.get("code", "")).strip()
                qty = int(item.get("shares") or 0)
                if not code or qty <= 0:
                    continue
                positions.append(
                    {
                        "code": code,
                        "qty": qty,
                        "entry_price": float(item.get("entry_price") or 0.0),
                        "pivot_price": float(item.get("pivot_price") or 0.0),
                        "stop_price": float(item.get("stop_price") or 0.0),
                        "take_profit_price": float(
                            item.get("take_profit_price") or 0.0
                        ),
                        "status": "holding",
                        "entry_date": str(bot_result.get("trade_date")),
                    }
                )

            if positions:
                upserted_positions = ENGINE.upsert_positions(positions)
                progress_callback(f"holdings 업서트 완료: {len(upserted_positions)}개")
            else:
                progress_callback("업서트할 ready 후보가 없음")

        progress_callback("장전 점검 실행")
        preopen_summary = ENGINE.run_preopen_check_once()
        progress_callback("장전 점검 완료")

        live_payload = payload.get("live", {})
        start_live = bool(payload.get("start_live", False))
        live_status = ENGINE.status()
        if start_live:
            progress_callback("라이브 엔진 시작")
            live_status = ENGINE.start(
                {
                    "holdings_path": live_payload.get(
                        "holdings_path", "data/runtime/holdings.json"
                    ),
                    "loop_interval_sec": float(
                        live_payload.get("loop_interval_sec", 2.0)
                    ),
                    "market_open_hhmm": str(
                        live_payload.get("market_open_hhmm", "09:00")
                    ),
                    "market_close_hhmm": str(
                        live_payload.get("market_close_hhmm", "15:30")
                    ),
                }
            )
            if bool(live_status.get("running")):
                progress_callback("라이브 엔진 시작 완료")
            else:
                progress_callback(
                    f"라이브 엔진 시작 차단: {live_status.get('last_error') or '사유 없음'}"
                )

        progress_callback("run_all 오케스트레이션 완료")

        return {
            "bot": bot_result,
            "preopen_summary": preopen_summary,
            "upserted_positions_count": len(upserted_positions),
            "live_status": live_status,
        }
    except Exception as exc:
        PIPELINE_LOGGER.exception("[PIPELINE] run_all 실패: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.include_router(financial_universe_router)


def run_once(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return run_all(payload or DEFAULT_RUN_PAYLOAD)


if __name__ == "__main__":
    if "--server" in sys.argv:
        import uvicorn

        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
    else:
        try:
            result = run_once()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(0)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            raise SystemExit(1)
