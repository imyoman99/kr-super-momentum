from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
import io
import json
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_DIR = Path(__file__).resolve().parents[3]


FINANCIAL_UNIVERSE_CONFIG: dict[str, Any] = {
    "use_today": True,
    "reference_date": None,
    "financial_source": "api_cache_first",
    "financial_data_path": "data/fundamental/financial_statements.csv",
    "financial_cache_dir": "data/cache/financial",
    "financial_api": {
        "enabled": False,
        "url": "",
        "method": "GET",
        "headers": {},
        "params": {},
        "body": {},
        "timeout": 20,
        "data_key": "data",
        "target_markets": ["KOSPI", "KOSDAQ"],
        "enforce_market_filter": True,
        "open_dart_max_workers": 15,
    },
    "code_column_candidates": ["Code", "code", "stock_code", "ticker", "종목코드"],
    "year_column_candidates": ["fiscal_year", "year", "사업연도", "회계연도"],
    "quarter_column_candidates": ["fiscal_quarter", "quarter", "분기", "reprt_code"],
    "operating_income_candidates": [
        "operating_income",
        "op_income",
        "영업이익",
        "영업이익(손실)",
    ],
    "quarter_map": {
        "Q1": ["Q1", "1Q", "1", 1, "11013"],
        "Q2": ["Q2", "2Q", "2", 2, "11012"],
        "Q3": ["Q3", "3Q", "3", 3, "11014"],
        "Q4": ["Q4", "4Q", "4", 4, "11011"],
    },
    "base_conditions": [
        {"column": "operating_income", "operator": ">", "value": 0},
    ],
    "extra_conditions": [],
}


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _deep_get(data: Any, key_path: str | None) -> Any:
    if key_path is None or key_path == "":
        return data

    current = data
    for key in key_path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def _fetch_financial_df_from_api(
    conf: dict[str, Any],
    fiscal_year: int,
    fiscal_quarter: str,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    api_conf = conf.get("financial_api", {})
    if not api_conf.get("enabled", False):
        raise ValueError("financial_api.enabled=False")

    provider = str(api_conf.get("provider", "")).strip().lower()
    if provider == "opendart":
        return _fetch_financial_df_from_open_dart(
            conf,
            fiscal_year,
            fiscal_quarter,
            progress_callback=progress_callback,
        )

    url = str(api_conf.get("url", "")).strip()
    if not url:
        raise ValueError("financial_api.url 이 비어 있습니다.")

    method = str(api_conf.get("method", "GET")).upper()
    headers = dict(api_conf.get("headers", {}))
    params = dict(api_conf.get("params", {}))
    body = dict(api_conf.get("body", {}))
    timeout = int(api_conf.get("timeout", 20))
    data_key = api_conf.get("data_key", "data")

    params["fiscal_year"] = fiscal_year
    params["fiscal_quarter"] = fiscal_quarter
    body["fiscal_year"] = fiscal_year
    body["fiscal_quarter"] = fiscal_quarter

    if method == "POST":
        response = requests.post(
            url, headers=headers, params=params, json=body, timeout=timeout
        )
    else:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)

    response.raise_for_status()

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise ValueError("재무 API 응답이 JSON 형식이 아닙니다.") from exc

    rows = _deep_get(payload, data_key if isinstance(data_key, str) else None)
    if rows is None:
        rows = payload

    if isinstance(rows, dict):
        if "items" in rows and isinstance(rows["items"], list):
            rows = rows["items"]
        else:
            rows = [rows]

    if not isinstance(rows, list):
        raise ValueError("재무 API 응답에서 행 리스트를 찾지 못했습니다.")

    return pd.DataFrame(rows)


def _reprt_code_from_quarter(fiscal_quarter: str) -> str:
    mapping = {
        "Q1": "11013",
        "Q2": "11012",
        "Q3": "11014",
        "Q4": "11011",
    }
    return mapping.get(str(fiscal_quarter).upper(), "11011")


def _extract_dart_api_key(api_conf: dict[str, Any]) -> str:
    return (
        str(api_conf.get("api_key") or "").strip()
        or str(os.getenv("OPEN_DART_API_KEY") or "").strip()
        or str(os.getenv("DART_API_KEY") or "").strip()
    )


def _load_krx_symbols_by_market(target_markets: list[str]) -> set[str] | None:
    normalized_targets = {
        str(m).strip().upper() for m in target_markets if str(m).strip()
    }
    if not normalized_targets:
        return None

    try:
        import FinanceDataReader as fdr  # type: ignore

        listing = fdr.StockListing("KRX")
        if listing is not None and not listing.empty:
            if "Symbol" in listing.columns and "Market" in listing.columns:
                market_series = listing["Market"].astype(str).str.strip().str.upper()
                symbol_series = listing["Symbol"].astype(str).str.strip().str.zfill(6)
                mask = market_series.isin(normalized_targets)
                symbols = set(symbol_series[mask])
                symbols = {s for s in symbols if s}
                if symbols:
                    return symbols
    except Exception:
        pass

    try:
        from pykrx import stock  # type: ignore

        today_text = datetime.now().strftime("%Y%m%d")
        symbols: set[str] = set()
        if "KOSPI" in normalized_targets:
            symbols.update(
                stock.get_market_ticker_list(date=today_text, market="KOSPI")
            )
        if "KOSDAQ" in normalized_targets:
            symbols.update(
                stock.get_market_ticker_list(date=today_text, market="KOSDAQ")
            )
        normalized = {
            str(code).strip().zfill(6) for code in symbols if str(code).strip()
        }
        return normalized if normalized else None
    except Exception:
        return None


def _fetch_financial_df_from_open_dart(
    conf: dict[str, Any],
    fiscal_year: int,
    fiscal_quarter: str,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    api_conf = conf.get("financial_api", {})
    api_key = _extract_dart_api_key(api_conf)
    if not api_key:
        raise ValueError(
            "OpenDART API 키가 없습니다. financial_api.api_key 또는 OPEN_DART_API_KEY를 설정하세요."
        )

    base_url = str(api_conf.get("base_url") or "https://opendart.fss.or.kr/api").rstrip(
        "/"
    )
    timeout = int(api_conf.get("timeout", 20))
    fs_div = str(api_conf.get("fs_div") or "CFS")
    reprt_code = _reprt_code_from_quarter(fiscal_quarter)
    max_companies = int(api_conf.get("max_companies", 0) or 0)
    open_dart_max_workers = max(1, int(api_conf.get("open_dart_max_workers", 15)))

    target_markets_raw = api_conf.get("target_markets", ["KOSPI", "KOSDAQ"])
    if isinstance(target_markets_raw, list):
        target_markets = [
            str(m).strip().upper() for m in target_markets_raw if str(m).strip()
        ]
    else:
        target_markets = (
            [str(target_markets_raw).strip().upper()]
            if str(target_markets_raw).strip()
            else []
        )
    enforce_market_filter = bool(api_conf.get("enforce_market_filter", True))

    corp_url = f"{base_url}/corpCode.xml"
    corp_res = requests.get(corp_url, params={"crtfc_key": api_key}, timeout=timeout)
    corp_res.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(corp_res.content))
    xml_name = zf.namelist()[0]
    xml_bytes = zf.read(xml_name)
    root = ET.fromstring(xml_bytes)

    corp_pairs: list[tuple[str, str]] = []
    for item in root.findall("list"):
        corp_code = (item.findtext("corp_code") or "").strip()
        stock_code = (item.findtext("stock_code") or "").strip()
        if corp_code and stock_code:
            corp_pairs.append((corp_code, stock_code))

    if target_markets:
        target_symbol_set = _load_krx_symbols_by_market(target_markets)
        if target_symbol_set is None:
            if enforce_market_filter:
                raise ValueError(
                    "KRX 상장목록 조회 실패로 코스피/코스닥 전종목 필터를 적용할 수 없습니다."
                )
        else:
            corp_pairs = [
                (corp_code, stock_code)
                for corp_code, stock_code in corp_pairs
                if stock_code in target_symbol_set
            ]

    if max_companies > 0:
        corp_pairs = corp_pairs[:max_companies]

    total_pairs = len(corp_pairs)
    market_text = "/".join(target_markets) if target_markets else "ALL"
    if progress_callback is not None:
        progress_callback(
            f"1차 유니버스 생성 진행률: 0% (OpenDART {market_text} 대상 {total_pairs}개)"
        )

    rows: list[dict[str, Any]] = []
    account_url = f"{base_url}/fnlttSinglAcntAll.json"
    progress_bucket = -1

    def _fetch_one(corp_code: str, stock_code: str) -> dict[str, Any] | None:
        params = {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bsns_year": str(fiscal_year),
            "reprt_code": reprt_code,
            "fs_div": fs_div,
        }
        try:
            res = requests.get(account_url, params=params, timeout=timeout)
            res.raise_for_status()
            payload = res.json()
        except Exception:
            return None

        if str(payload.get("status")) != "000":
            return None

        items = payload.get("list", [])
        if not isinstance(items, list):
            return None

        op_income = None
        for line in items:
            account_nm = str(line.get("account_nm") or "")
            if "영업이익" not in account_nm:
                continue
            amount_text = str(
                line.get("thstrm_amount")
                or line.get("thstrm_add_amount")
                or line.get("frmtrm_amount")
                or ""
            ).replace(",", "")
            if amount_text in {"", "-", "N/A"}:
                continue
            try:
                op_income = float(amount_text)
                break
            except Exception:
                continue

        if op_income is None:
            return None

        return {
            "Code": stock_code,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "operating_income": op_income,
        }

    with ThreadPoolExecutor(max_workers=open_dart_max_workers) as executor:
        futures = [
            executor.submit(_fetch_one, corp_code, stock_code)
            for corp_code, stock_code in corp_pairs
        ]
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is not None:
                rows.append(result)
            if progress_callback is not None and total_pairs > 0:
                scaled_percent = int((completed * 59) / total_pairs)
                bucket = min(59, scaled_percent)
                if bucket > progress_bucket:
                    progress_bucket = bucket
                    progress_callback(
                        f"1차 유니버스 생성 진행률: {bucket}% ({completed}/{total_pairs})"
                    )

    if progress_callback is not None:
        progress_callback(
            f"1차 유니버스 생성 진행률: 59% (재무행 {len(rows)}건 수집 완료)"
        )

    return pd.DataFrame(rows)


def _cache_filename(base_date: date, fiscal_year: int, fiscal_quarter: str) -> str:
    return (
        f"{base_date.strftime('%Y%m%d')}_financial_{fiscal_year}_{fiscal_quarter}.csv"
    )


def _load_financial_df(
    conf: dict[str, Any],
    base_date: date,
    fiscal_year: int,
    fiscal_quarter: str,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    cache_dir = _resolve_path(
        str(conf.get("financial_cache_dir", "data/cache/financial"))
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_filename(base_date, fiscal_year, fiscal_quarter)

    if cache_path.exists():
        if progress_callback is not None:
            progress_callback("1차 유니버스 생성 진행률: 30% (재무 캐시 로드 완료)")
        return pd.read_csv(cache_path, dtype=str)

    source_mode = str(conf.get("financial_source", "api_cache_first")).lower()

    if source_mode in {"api", "api_cache_first"}:
        try:
            api_df = _fetch_financial_df_from_api(
                conf,
                fiscal_year,
                fiscal_quarter,
                progress_callback=progress_callback,
            )
            if api_df.empty:
                raise ValueError("재무 API 결과가 비어 있습니다.")
            api_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            if progress_callback is not None:
                progress_callback("1차 유니버스 생성 진행률: 60% (재무 API 로드 완료)")
            return api_df.astype(str)
        except Exception:
            if source_mode == "api":
                raise

    csv_path = _resolve_path(str(conf["financial_data_path"]))
    if not csv_path.exists():
        raise FileNotFoundError(
            f"재무 데이터 파일이 없습니다: {csv_path}. financial_api 설정 또는 파일 경로를 확인하세요."
        )

    csv_df = pd.read_csv(csv_path, dtype=str)
    csv_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    if progress_callback is not None:
        progress_callback("1차 유니버스 생성 진행률: 60% (재무 CSV 로드 완료)")
    return csv_df


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def get_financial_base_date(config: dict[str, Any] | None = None) -> date:
    conf = config or FINANCIAL_UNIVERSE_CONFIG
    if conf.get("use_today", True):
        return date.today()

    ref = conf.get("reference_date")
    if not ref:
        raise ValueError(
            "reference_date가 없습니다. use_today=False이면 기준일을 넣어야 합니다."
        )
    return _parse_date(ref)


def map_date_to_fiscal_period(base_date: date) -> tuple[int, str]:
    y = base_date.year
    md = (base_date.month, base_date.day)

    if (1, 1) <= md <= (3, 31):
        return y - 1, "Q3"
    if (4, 1) <= md <= (5, 15):
        return y - 1, "Q4"
    if (5, 16) <= md <= (8, 15):
        return y, "Q1"
    if (8, 16) <= md <= (11, 15):
        return y, "Q2"
    return y, "Q3"


def _find_first_existing_column(
    df: pd.DataFrame, candidates: list[str], logical_name: str
) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"{logical_name} 컬럼을 찾을 수 없습니다. candidates={candidates}")


def _normalize_quarter(value: Any, quarter_map: dict[str, list[Any]]) -> str | None:
    text = str(value).strip().upper()
    for standard, aliases in quarter_map.items():
        for alias in aliases:
            if str(alias).strip().upper() == text:
                return standard
    return None


def _apply_condition(df: pd.DataFrame, cond: dict[str, Any]) -> pd.Series:
    column = cond["column"]
    operator = cond.get("operator", ">=")
    value = cond.get("value")

    if column not in df.columns:
        raise KeyError(f"조건 컬럼이 데이터에 없습니다: {column}")

    series = df[column]
    if operator == ">=":
        return series >= value
    if operator == ">":
        return series > value
    if operator == "<=":
        return series <= value
    if operator == "<":
        return series < value
    if operator == "==":
        return series == value
    if operator == "!=":
        return series != value
    raise ValueError(f"지원하지 않는 연산자입니다: {operator}")


def build_financial_universe(
    config: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    conf = {**FINANCIAL_UNIVERSE_CONFIG, **(config or {})}
    conf["financial_api"] = {
        **FINANCIAL_UNIVERSE_CONFIG.get("financial_api", {}),
        **dict((config or {}).get("financial_api", {})),
    }

    base_date = get_financial_base_date(conf)
    fiscal_year, fiscal_quarter = map_date_to_fiscal_period(base_date)

    df = _load_financial_df(
        conf=conf,
        base_date=base_date,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        progress_callback=progress_callback,
    )

    if progress_callback is not None:
        progress_callback("1차 유니버스 생성 진행률: 75% (재무 데이터 정규화 중)")

    code_col = _find_first_existing_column(
        df, conf["code_column_candidates"], "종목코드"
    )
    year_col = _find_first_existing_column(
        df, conf["year_column_candidates"], "사업연도"
    )
    quarter_col = _find_first_existing_column(
        df, conf["quarter_column_candidates"], "분기"
    )
    op_col = _find_first_existing_column(
        df,
        conf["operating_income_candidates"],
        "영업이익",
    )

    work = df.copy()
    work = work.rename(
        columns={
            code_col: "code",
            year_col: "fiscal_year",
            quarter_col: "fiscal_quarter",
            op_col: "operating_income",
        }
    )

    work["fiscal_year"] = pd.to_numeric(work["fiscal_year"], errors="coerce")
    work["fiscal_quarter"] = work["fiscal_quarter"].apply(
        lambda v: _normalize_quarter(v, conf["quarter_map"])
    )
    work["operating_income"] = pd.to_numeric(work["operating_income"], errors="coerce")

    filtered = work[
        (work["fiscal_year"] == fiscal_year)
        & (work["fiscal_quarter"] == fiscal_quarter)
    ].copy()

    all_conditions = list(conf.get("base_conditions", [])) + list(
        conf.get("extra_conditions", [])
    )

    if all_conditions:
        mask = pd.Series(True, index=filtered.index)
        for cond in all_conditions:
            mask &= _apply_condition(filtered, cond)
        filtered = filtered[mask]

    if progress_callback is not None:
        progress_callback("1차 유니버스 생성 진행률: 100% (필터링 완료)")

    universe_codes = sorted(filtered["code"].dropna().astype(str).unique().tolist())

    stage1_detail = (
        filtered[["code", "operating_income"]].dropna(subset=["code"]).copy()
    )
    stage1_detail["code"] = stage1_detail["code"].astype(str)
    stage1_detail["operating_income"] = pd.to_numeric(
        stage1_detail["operating_income"],
        errors="coerce",
    )
    stage1_detail = stage1_detail.sort_values(["code", "operating_income"])
    stage1_detail = stage1_detail.drop_duplicates(subset=["code"], keep="last")
    stage1_detail["operating_income_positive"] = stage1_detail["operating_income"] > 0

    stage1_rows = [
        {
            "Code": row["code"],
            "영업이익": (
                None
                if pd.isna(row["operating_income"])
                else float(row["operating_income"])
            ),
            "영업이익_0초과": bool(row["operating_income_positive"]),
        }
        for _, row in stage1_detail.iterrows()
    ]

    return {
        "base_date": str(base_date),
        "fiscal_year": int(fiscal_year),
        "fiscal_quarter": fiscal_quarter,
        "count": len(universe_codes),
        "codes": universe_codes,
        "stage1_rows": stage1_rows,
    }
