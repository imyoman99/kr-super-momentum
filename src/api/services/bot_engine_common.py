from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def parse_ymd(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_hhmm(value: str) -> tuple[int, int]:
    hour_text, minute_text = str(value).split(":")
    return int(hour_text), int(minute_text)


def stage_file(stage_dir: Path, stage_name: str, target_date: date) -> Path:
    date_text = target_date.strftime("%Y%m%d")
    name_map = {
        "stage1_financial": "1차_재무유니버스",
        "stage2_ranked": "2차_RS시총거래대금유니버스",
        "stage3_minervini": "3차_마크미너비니_최종유니버스",
    }
    label = name_map.get(stage_name, stage_name)
    return stage_dir / f"{date_text}_{label}.csv"


def extract_stage_date(file_path: Path) -> date | None:
    stem = file_path.stem
    if len(stem) < 8:
        return None
    prefix = stem[:8]
    if not prefix.isdigit():
        return None
    try:
        return datetime.strptime(prefix, "%Y%m%d").date()
    except ValueError:
        return None


def read_stage_codes(stage_path: Path) -> list[str]:
    if not stage_path.exists():
        return []
    df = pd.read_csv(stage_path, dtype={"Code": str})
    if "Code" not in df.columns:
        return []
    return df["Code"].dropna().astype(str).unique().tolist()


def push_log(
    logs: list[str],
    message: str,
    log_callback: Callable[[str], None] | None = None,
) -> None:
    logs.append(message)
    if log_callback is not None:
        try:
            log_callback(message)
        except Exception:
            pass


def log_progress_percent(
    logs: list[str],
    stage_label: str,
    current: int,
    total: int,
    last_bucket: int,
    log_callback: Callable[[str], None] | None = None,
    bucket_size: int = 1,
) -> int:
    if total <= 0:
        return last_bucket

    percent = int((current * 100) / total)
    bucket = min(100, (percent // bucket_size) * bucket_size)
    if bucket > last_bucket:
        push_log(
            logs,
            f"{stage_label} 생성 진행률: {bucket}% ({current}/{total})",
            log_callback,
        )
        return bucket
    return last_bucket


def find_latest_stage1_file(
    stage_dir: Path, up_to_date: date
) -> tuple[Path | None, date | None]:
    candidates = sorted(stage_dir.glob("*_1차_재무유니버스.csv"), reverse=True)
    for path in candidates:
        file_date = extract_stage_date(path)
        if file_date is None:
            continue
        if file_date <= up_to_date:
            return path, file_date
    return None, None


def previous_business_day(base_date: date) -> date:
    candidate = base_date - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def next_business_day(base_date: date) -> date:
    candidate = base_date + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def resolve_signal_and_trade_date_by_runtime(
    now_dt: datetime,
    market_open_hhmm: str,
    market_close_hhmm: str,
) -> tuple[date, date, str]:
    today = now_dt.date()
    if today.weekday() >= 5:
        signal_date = previous_business_day(today)
        trade_date = next_business_day(today)
        return signal_date, trade_date, "주말 실행: 직전 거래일 기준"

    open_h, open_m = parse_hhmm(market_open_hhmm)
    close_h, close_m = parse_hhmm(market_close_hhmm)
    open_dt = now_dt.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    close_dt = now_dt.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

    if now_dt < open_dt or (open_dt <= now_dt <= close_dt):
        signal_date = previous_business_day(today)
        trade_date = today
        return signal_date, trade_date, "장전/장중 실행: 직전 거래일 기준"

    signal_date = today
    trade_date = next_business_day(today)
    return signal_date, trade_date, "장마감 후 실행: 당일 포함 기준"


def resolve_runtime_phase(
    now_dt: datetime,
    market_open_hhmm: str,
    market_close_hhmm: str,
) -> str:
    today = now_dt.date()
    if today.weekday() >= 5:
        return "weekend"

    open_h, open_m = parse_hhmm(market_open_hhmm)
    close_h, close_m = parse_hhmm(market_close_hhmm)
    open_dt = now_dt.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    close_dt = now_dt.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

    if now_dt < open_dt:
        return "preopen"
    if open_dt <= now_dt <= close_dt:
        return "market_open"
    return "after_close"
