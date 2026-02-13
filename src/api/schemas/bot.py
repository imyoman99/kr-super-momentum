from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FinancialCondition(BaseModel):
    column: str = Field(..., description="필터 대상 컬럼명")
    operator: str = Field(
        ">=",
        description="비교 연산자(>=, >, <=, <, ==, !=)",
    )
    value: Any = Field(..., description="비교 기준값")


class FinancialUniverseRequest(BaseModel):
    use_today: bool = True
    reference_date: str | None = None
    financial_data_path: str | None = None
    financial_api_enabled: bool = False
    financial_api_url: str | None = None
    financial_api_method: str = "GET"
    financial_api_data_key: str = "data"
    financial_api_headers: dict[str, str] = Field(default_factory=dict)
    financial_api_params: dict[str, Any] = Field(default_factory=dict)
    financial_cache_dir: str | None = None
    extra_conditions: list[FinancialCondition] = Field(default_factory=list)


class FinancialUniverseResponse(BaseModel):
    base_date: str
    fiscal_year: int
    fiscal_quarter: str
    count: int
    codes: list[str]


class BotMode(str, Enum):
    universe = "universe"
    trading = "trading"


class BotRunRequest(BaseModel):
    mode: BotMode = BotMode.universe
    reference_date: str = Field(..., description="과거 기준일(YYYY-MM-DD)")
    financial_data_path: str | None = None
    financial_api_enabled: bool = False
    financial_api_url: str | None = None
    financial_api_method: str = "GET"
    financial_api_data_key: str = "data"
    financial_api_headers: dict[str, str] = Field(default_factory=dict)
    financial_api_params: dict[str, Any] = Field(default_factory=dict)
    financial_cache_dir: str | None = None
    universe_snapshot_path: str | None = None
    force_rebuild: bool = False

    extra_financial_conditions: list[FinancialCondition] = Field(default_factory=list)

    min_rs_rating: float = 80
    min_market_cap: float = 100_000_000_000
    trading_value_lookback_days: int = 20
    min_avg_daily_trading_value: float = 50_000_000_000
    top_n_stage2: int = 30
    rs_benchmark_code: str = "KS11"

    account_equity: float = 100000000
    risk_per_trade: float = 0.01
    max_position_ratio: float = 0.15
    stop_atr_multiple: float = 2.5
    take_profit_r: float = 2.0
    pivot_window: int = 10


class TradeCandidate(BaseModel):
    code: str
    status: str
    signal_date: str
    trade_date: str
    pivot_price: float | None = None
    open_price: float | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    shares: int | None = None


class BotRunResponse(BaseModel):
    mode: str
    reference_date: str
    signal_date: str
    trade_date: str
    has_yesterday_positions: bool
    yesterday_positions: list[str]
    stage_counts: dict[str, int]
    final_codes: list[str]
    ranked_snapshot: list[dict[str, Any]]
    trade_candidates: list[TradeCandidate]
    logs: list[str]


class LiveStartRequest(BaseModel):
    holdings_path: str = "data/runtime/holdings.json"
    loop_interval_sec: float = 2.0
    market_open_hhmm: str = "09:00"
    market_close_hhmm: str = "15:30"


class LivePosition(BaseModel):
    code: str
    qty: int
    entry_price: float
    pivot_price: float
    stop_price: float
    take_profit_price: float
    status: str = "holding"
    entry_date: str | None = None


class LivePositionsUpsertRequest(BaseModel):
    positions: list[LivePosition]


class LivePositionDeleteRequest(BaseModel):
    code: str


class LiveStatusResponse(BaseModel):
    running: bool
    started_at: str | None = None
    last_loop_at: str | None = None
    last_error: str | None = None
    last_preopen_check_date: str | None = None
    last_open_check_date: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    positions: list[dict[str, Any]] = Field(default_factory=list)


class LiveActionResponse(BaseModel):
    actions: list[dict[str, Any]] = Field(default_factory=list)
