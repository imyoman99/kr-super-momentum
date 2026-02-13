from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any

from src.api.utils.kis_client import KisClient
from src.api.utils.progress_logger import get_progress_logger
from src.api.utils.slack_notifier import create_slack_notifier

try:
    import holidays as py_holidays
except Exception:
    py_holidays = None


@dataclass
class LiveEngineConfig:
    holdings_path: str = "data/runtime/holdings.json"
    loop_interval_sec: float = 2.0
    market_open_hhmm: str = "09:00"
    market_close_hhmm: str = "15:30"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(str(value).replace(",", "").strip())
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return default


def _parse_hhmm(value: str) -> dt_time:
    hour_text, minute_text = str(value).split(":")
    return dt_time(hour=int(hour_text), minute=int(minute_text))


def _extract_stage_date_from_name(file_path: Path) -> date | None:
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


class LiveTradingEngine:
    def __init__(self) -> None:
        self._logger = get_progress_logger("kr_super_momentum.live")
        self._slack_notifier = create_slack_notifier(prefix="[LIVE] ")
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._client: KisClient | None = None
        self._config = LiveEngineConfig()
        self._last_loop_phase: str | None = None
        self._holiday_cache: dict[int, set[date]] = {}
        self._state: dict[str, Any] = {
            "running": False,
            "started_at": None,
            "last_loop_at": None,
            "last_error": None,
            "last_preopen_check_date": None,
            "last_open_check_date": None,
            "summary": {},
            "actions": [],
        }

    def _log(self, message: str) -> None:
        self._logger.info("[LIVE] %s", message)
        self._slack_notifier.send(message)

    def _is_market_open_now(self) -> bool:
        now_time = datetime.now().time()
        open_t = _parse_hhmm(self._config.market_open_hhmm)
        close_t = _parse_hhmm(self._config.market_close_hhmm)
        return open_t <= now_time <= close_t

    def _find_latest_stage_file(
        self,
        stage_dir: Path,
        pattern: str,
    ) -> tuple[Path | None, date | None]:
        candidates = sorted(stage_dir.glob(pattern), reverse=True)
        for path in candidates:
            file_date = _extract_stage_date_from_name(path)
            if file_date is not None:
                return path, file_date
        return None, None

    def _is_kr_holiday(self, target_day: date) -> bool:
        if py_holidays is None:
            return False
        year = target_day.year
        if year not in self._holiday_cache:
            try:
                holiday_map = py_holidays.country_holidays("KR", years=[year])
                self._holiday_cache[year] = set(holiday_map.keys())
            except Exception:
                self._holiday_cache[year] = set()
        return target_day in self._holiday_cache[year]

    def _previous_business_day(self, base_date: date) -> date:
        candidate = base_date - timedelta(days=1)
        while candidate.weekday() >= 5 or self._is_kr_holiday(candidate):
            candidate -= timedelta(days=1)
        return candidate

    def _expected_stage_date_for_live(self, stage_dir: Path) -> date:
        del stage_dir
        today = datetime.now().date()
        expected = self._previous_business_day(today)
        if py_holidays is None:
            self._log("holidays 패키지가 없어 주말 기준으로 직전 거래일 계산")
        self._log(f"실행일 기준 직전 거래일 사용: {expected.isoformat()}")
        return expected

    def _validate_stage_files_for_live(self) -> tuple[bool, str]:
        root = Path(__file__).resolve().parents[3]
        stage_dir = root / "data" / "universe"
        expected_date = self._expected_stage_date_for_live(stage_dir)

        if not stage_dir.exists():
            return (
                False,
                "2차 3차 업데이트가 필요합니다: data/universe 폴더가 없습니다.",
            )

        stage2_path, stage2_date = self._find_latest_stage_file(
            stage_dir,
            "*_2차_RS시총거래대금유니버스.csv",
        )
        stage3_path, stage3_date = self._find_latest_stage_file(
            stage_dir,
            "*_3차_마크미너비니_최종유니버스.csv",
        )

        if stage2_path is None or stage2_date is None:
            return False, "2차 3차 업데이트가 필요합니다: 2차 유니버스 파일이 없습니다."
        if stage3_path is None or stage3_date is None:
            return False, "2차 3차 업데이트가 필요합니다: 3차 유니버스 파일이 없습니다."

        if stage2_date != expected_date or stage3_date != expected_date:
            return (
                False,
                (
                    "2차 3차 업데이트가 필요합니다: "
                    f"2차={stage2_path.name}({stage2_date.isoformat()}), "
                    f"3차={stage3_path.name}({stage3_date.isoformat()}), "
                    f"필요날짜(직전 거래일)={expected_date.isoformat()}"
                ),
            )

        self._log(
            "2차/3차 최신 파일 검증 완료: %s, %s" % (stage2_path.name, stage3_path.name)
        )
        return True, ""

    def start(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if config:
                self._config = LiveEngineConfig(
                    holdings_path=str(
                        config.get("holdings_path", self._config.holdings_path)
                    ),
                    loop_interval_sec=float(
                        config.get("loop_interval_sec", self._config.loop_interval_sec)
                    ),
                    market_open_hhmm=str(
                        config.get("market_open_hhmm", self._config.market_open_hhmm)
                    ),
                    market_close_hhmm=str(
                        config.get("market_close_hhmm", self._config.market_close_hhmm)
                    ),
                )
                self._log(
                    "시작 설정 반영: holdings=%s, interval=%.1fs, market=%s~%s"
                    % (
                        self._config.holdings_path,
                        self._config.loop_interval_sec,
                        self._config.market_open_hhmm,
                        self._config.market_close_hhmm,
                    )
                )

            if self._state["running"]:
                self._log("이미 실행 중이라 현재 상태 반환")
                return self.status()

            if not self._is_market_open_now():
                message = "장중 시간이 아닙니다. 라이브매매를 종료합니다."
                self._state["running"] = False
                self._state["last_error"] = message
                self._log(message)
                return self.status()

            valid_stage_files, stage_message = self._validate_stage_files_for_live()
            if not valid_stage_files:
                self._state["running"] = False
                self._state["last_error"] = stage_message
                self._log(stage_message)
                return self.status()

            self._stop_event.clear()
            self._state["running"] = True
            self._state["started_at"] = datetime.now().isoformat()
            self._state["last_error"] = None
            self._last_loop_phase = None
            self._log("라이브 엔진 시작")

            self._thread = threading.Thread(
                target=self._run_loop, name="live-trading-engine", daemon=True
            )
            self._thread.start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._state["running"]:
                self._log("중지 요청: 이미 중지 상태")
                return self.status()
            self._stop_event.set()
            self._log("라이브 엔진 중지 요청 수신")

        if self._thread is not None:
            self._thread.join(timeout=5)

        with self._lock:
            self._state["running"] = False
            self._log("라이브 엔진 중지 완료")
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = deepcopy(self._state)
            result["config"] = {
                "holdings_path": self._config.holdings_path,
                "loop_interval_sec": self._config.loop_interval_sec,
                "market_open_hhmm": self._config.market_open_hhmm,
                "market_close_hhmm": self._config.market_close_hhmm,
            }
            result["positions"] = self.get_positions()
            return result

    def get_positions(self) -> list[dict[str, Any]]:
        payload = self._read_holdings_payload()
        positions = payload.get("positions", [])
        if not isinstance(positions, list):
            return []
        return positions

    def upsert_positions(self, positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = self._read_holdings_payload()
        current = payload.get("positions", [])
        if not isinstance(current, list):
            current = []

        by_code: dict[str, dict[str, Any]] = {}
        for item in current:
            code = str(item.get("code", "")).strip()
            if code:
                by_code[code] = item

        for item in positions:
            code = str(item.get("code", "")).strip()
            if not code:
                continue
            merged = {**by_code.get(code, {}), **item}
            merged["code"] = code
            merged.setdefault("status", "holding")
            merged["qty"] = _to_int(merged.get("qty"), 0)
            merged["entry_price"] = _to_float(merged.get("entry_price"), 0.0)
            merged["pivot_price"] = _to_float(merged.get("pivot_price"), 0.0)
            merged["stop_price"] = _to_float(merged.get("stop_price"), 0.0)
            merged["take_profit_price"] = _to_float(
                merged.get("take_profit_price"), 0.0
            )
            by_code[code] = merged

        payload["positions"] = sorted(
            by_code.values(), key=lambda x: str(x.get("code", ""))
        )
        self._write_holdings_payload(payload)
        return payload["positions"]

    def remove_position(self, code: str) -> list[dict[str, Any]]:
        target = str(code).strip()
        payload = self._read_holdings_payload()
        current = payload.get("positions", [])
        if not isinstance(current, list):
            current = []
        payload["positions"] = [
            item for item in current if str(item.get("code", "")).strip() != target
        ]
        self._write_holdings_payload(payload)
        return payload["positions"]

    def run_preopen_check_once(self) -> dict[str, Any]:
        self._log("장전 점검 시작")
        self._ensure_client()
        summary = self._build_account_summary()
        now = datetime.now().date().isoformat()
        with self._lock:
            self._state["summary"] = summary
            self._state["last_preopen_check_date"] = now
        self._log(
            "장전 점검 완료: 보유=%d, 현금=%.0f, 총자산=%.0f"
            % (
                len(summary.get("positions", [])),
                _to_float(summary.get("cash"), 0.0),
                _to_float(summary.get("total_asset"), 0.0),
            )
        )
        return summary

    def run_open_check_once(self) -> dict[str, Any]:
        self._log("장시작 점검 시작(gap stop)")
        self._ensure_client()
        actions = self._check_gap_stop_and_send_sell()
        with self._lock:
            self._state["last_open_check_date"] = datetime.now().date().isoformat()
            self._append_actions(actions)
        self._log(f"장시작 점검 완료: 액션 {len(actions)}건")
        return {"actions": actions}

    def run_monitor_once(self) -> dict[str, Any]:
        self._log("장중 모니터링 시작")
        self._ensure_client()
        self._reconcile_positions_with_balance()
        actions = self._monitor_positions_and_exit()
        with self._lock:
            self._append_actions(actions)
        self._log(f"장중 모니터링 완료: 액션 {len(actions)}건")
        return {"actions": actions}

    def _run_loop(self) -> None:
        self._log("루프 스레드 시작")
        while not self._stop_event.is_set():
            try:
                self._ensure_client()
                now = datetime.now()
                open_t = _parse_hhmm(self._config.market_open_hhmm)
                close_t = _parse_hhmm(self._config.market_close_hhmm)

                phase = "closed"
                if now.time() < open_t:
                    phase = "preopen"
                elif open_t <= now.time() <= close_t:
                    phase = "market_open"

                if phase != self._last_loop_phase:
                    self._last_loop_phase = phase
                    self._log(f"시장 단계 전환: {phase}")

                if now.time() < open_t:
                    last = self._state.get("last_preopen_check_date")
                    if last != now.date().isoformat():
                        self._log("일일 장전 점검 트리거")
                        self.run_preopen_check_once()

                if open_t <= now.time() <= close_t:
                    last_open = self._state.get("last_open_check_date")
                    if last_open != now.date().isoformat():
                        self._log("일일 장시작 점검 트리거")
                        self.run_open_check_once()
                    self._log("장중 모니터링 트리거")
                    self.run_monitor_once()
                else:
                    message = "장중 시간이 아닙니다. 라이브매매를 종료합니다."
                    self._log(message)
                    with self._lock:
                        self._state["last_error"] = message
                    self._stop_event.set()
                    break

                with self._lock:
                    self._state["last_loop_at"] = now.isoformat()
                    self._state["last_error"] = None
            except Exception as exc:
                with self._lock:
                    self._state["last_error"] = str(exc)
                self._logger.exception("[LIVE] 루프 에러: %s", exc)

            time.sleep(max(0.5, float(self._config.loop_interval_sec)))

        with self._lock:
            self._state["running"] = False
        self._log("루프 스레드 종료")

    def _append_actions(self, actions: list[dict[str, Any]]) -> None:
        if not actions:
            return
        for action in actions:
            self._log(
                "액션 기록: type=%s, code=%s, qty=%s, order_no=%s"
                % (
                    str(action.get("type", "")),
                    str(action.get("code", "")),
                    str(action.get("qty", "")),
                    str(action.get("order_no", "")),
                )
            )
        current = self._state.get("actions", [])
        current.extend(actions)
        self._state["actions"] = current[-200:]

    def _ensure_client(self) -> None:
        if self._client is None:
            self._log("KIS 클라이언트 초기화")
            self._client = KisClient()

    def _holdings_path(self) -> Path:
        path = Path(self._config.holdings_path)
        if path.is_absolute():
            return path
        root = Path(__file__).resolve().parents[3]
        return root / path

    def _read_holdings_payload(self) -> dict[str, Any]:
        file_path = self._holdings_path()
        if not file_path.exists():
            return {"positions": []}
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("positions", [])
                return payload
            return {"positions": []}
        except Exception:
            return {"positions": []}

    def _write_holdings_payload(self, payload: dict[str, Any]) -> None:
        file_path = self._holdings_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _build_account_summary(self) -> dict[str, Any]:
        payload = self._read_holdings_payload()
        positions = payload.get("positions", [])
        if not isinstance(positions, list):
            positions = []

        balance = self._client.get_balance() if self._client is not None else None
        cash = _to_float((balance or {}).get("deposit"), 0.0)

        stock_asset = 0.0
        details: list[dict[str, Any]] = []
        for pos in positions:
            if str(pos.get("status", "holding")) == "exited":
                continue
            code = str(pos.get("code", "")).strip()
            qty = _to_int(pos.get("qty"), 0)
            if not code or qty <= 0:
                continue
            quote = self._client.get_quote(code) if self._client is not None else None
            current_price = _to_float(
                (quote or {}).get("current_price"),
                _to_float(pos.get("entry_price"), 0.0),
            )
            value = current_price * qty
            stock_asset += value
            details.append(
                {
                    "code": code,
                    "qty": qty,
                    "current_price": current_price,
                    "market_value": value,
                    "stop_price": _to_float(pos.get("stop_price"), 0.0),
                    "take_profit_price": _to_float(pos.get("take_profit_price"), 0.0),
                }
            )

        self._log("계좌 요약 계산: holding_positions=%d" % len(details))

        total_asset = cash + stock_asset
        cash_ratio = (cash / total_asset) if total_asset > 0 else 0.0

        return {
            "checked_at": datetime.now().isoformat(),
            "cash": cash,
            "stock_asset": stock_asset,
            "total_asset": total_asset,
            "cash_ratio": cash_ratio,
            "positions": details,
        }

    def _check_gap_stop_and_send_sell(self) -> list[dict[str, Any]]:
        payload = self._read_holdings_payload()
        positions = payload.get("positions", [])
        if not isinstance(positions, list):
            return []

        actions: list[dict[str, Any]] = []
        changed = False

        for pos in positions:
            if str(pos.get("status", "holding")) != "holding":
                continue

            code = str(pos.get("code", "")).strip()
            qty = _to_int(pos.get("qty"), 0)
            stop_price = _to_float(pos.get("stop_price"), 0.0)
            if not code or qty <= 0 or stop_price <= 0:
                continue

            open_price = (
                self._client.get_open_price(code) if self._client is not None else None
            )
            if open_price is None:
                self._log(f"장시작 체크 스킵: 시가 조회 실패 code={code}")
                continue

            if float(open_price) <= stop_price:
                self._log(
                    f"장시작 손절 조건 충족: code={code}, open={float(open_price):.2f}, stop={stop_price:.2f}"
                )
                order_no = (
                    self._client.send_order(code, "SELL", qty, price=0)
                    if self._client is not None
                    else None
                )
                changed = True
                pos["status"] = "exit_pending"
                pos["exit_reason"] = "gap_stop_open"
                pos["exit_order_id"] = order_no
                pos["exit_signal_price"] = float(open_price)
                pos["exit_requested_at"] = datetime.now().isoformat()
                actions.append(
                    {
                        "type": "gap_stop_open",
                        "code": code,
                        "qty": qty,
                        "open_price": float(open_price),
                        "stop_price": stop_price,
                        "order_no": order_no,
                    }
                )

        if changed:
            payload["positions"] = positions
            self._write_holdings_payload(payload)
            self._log("장시작 체크 결과 holdings 저장 완료")

        return actions

    def _monitor_positions_and_exit(self) -> list[dict[str, Any]]:
        payload = self._read_holdings_payload()
        positions = payload.get("positions", [])
        if not isinstance(positions, list):
            return []

        actions: list[dict[str, Any]] = []
        changed = False

        for pos in positions:
            if str(pos.get("status", "holding")) != "holding":
                continue

            code = str(pos.get("code", "")).strip()
            qty = _to_int(pos.get("qty"), 0)
            stop_price = _to_float(pos.get("stop_price"), 0.0)
            take_profit = _to_float(pos.get("take_profit_price"), 0.0)
            if not code or qty <= 0:
                continue

            quote = self._client.get_quote(code) if self._client is not None else None
            current_price = _to_float((quote or {}).get("current_price"), 0.0)
            if current_price <= 0:
                self._log(f"장중 모니터링 스킵: 현재가 조회 실패 code={code}")
                continue

            pos["last_price"] = current_price
            pos["last_check_at"] = datetime.now().isoformat()

            exit_reason = None
            if stop_price > 0 and current_price <= stop_price:
                exit_reason = "stop_loss"
            elif take_profit > 0 and current_price >= take_profit:
                exit_reason = "take_profit"

            if exit_reason is not None:
                self._log(
                    f"청산 조건 충족: code={code}, reason={exit_reason}, price={current_price:.2f}"
                )
                order_no = (
                    self._client.send_order(code, "SELL", qty, price=0)
                    if self._client is not None
                    else None
                )
                changed = True
                pos["status"] = "exit_pending"
                pos["exit_reason"] = exit_reason
                pos["exit_order_id"] = order_no
                pos["exit_signal_price"] = current_price
                pos["exit_requested_at"] = datetime.now().isoformat()
                actions.append(
                    {
                        "type": exit_reason,
                        "code": code,
                        "qty": qty,
                        "current_price": current_price,
                        "stop_price": stop_price,
                        "take_profit_price": take_profit,
                        "order_no": order_no,
                    }
                )

        if changed:
            payload["positions"] = positions
            self._write_holdings_payload(payload)
            self._log("장중 모니터링 결과 holdings 저장 완료")

        return actions

    def _reconcile_positions_with_balance(self) -> None:
        if self._client is None:
            return
        self._log("브로커 잔고 동기화 시작")
        balance = self._client.get_balance()
        if not balance:
            self._log("브로커 잔고 동기화 스킵: 잔고 응답 없음")
            return

        stocks = balance.get("stocks", [])
        if not isinstance(stocks, list):
            stocks = []

        broker_qty: dict[str, int] = {}
        for stock in stocks:
            code = str(stock.get("pdno") or stock.get("mksc_shrn_iscd") or "").strip()
            qty = _to_int(
                stock.get("hldg_qty") or stock.get("hold_qty") or stock.get("qty"), 0
            )
            if code:
                broker_qty[code] = qty

        payload = self._read_holdings_payload()
        positions = payload.get("positions", [])
        if not isinstance(positions, list):
            return

        changed = False
        now_text = datetime.now().isoformat()
        today_text = datetime.now().date().isoformat()

        for pos in positions:
            code = str(pos.get("code", "")).strip()
            if not code:
                continue
            qty = broker_qty.get(code, 0)
            prev_qty = _to_int(pos.get("qty"), 0)
            if prev_qty != qty:
                pos["qty"] = qty
                changed = True
                self._log(f"보유수량 동기화: code={code}, {prev_qty} -> {qty}")

            if qty <= 0 and str(pos.get("status", "holding")) in {
                "holding",
                "exit_pending",
            }:
                pos["status"] = "exited"
                pos["exit_date"] = today_text
                pos["updated_at"] = now_text
                changed = True

        if changed:
            payload["positions"] = positions
            self._write_holdings_payload(payload)
            self._log("브로커 잔고 동기화 결과 holdings 저장 완료")
        else:
            self._log("브로커 잔고 동기화 완료: 변경 없음")


ENGINE = LiveTradingEngine()
