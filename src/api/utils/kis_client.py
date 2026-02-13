import requests
import json
import time
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.api.utils.config_loader import settings


class KisClient:
    INDEX_CODE_MAP = {
        "KS11": "0001",  # 코스피 종합
        "KQ11": "1001",  # 코스닥 종합
    }

    def __init__(self):
        # 1. 설정 로드
        self.base_url = settings["kis"]["base_url"]
        self.app_key = settings["kis"]["app_key"]
        self.app_secret = settings["kis"]["app_secret"]

        # [수정] 계좌번호 앞뒤 공백 제거 (안전장치)
        # .env에 ' 12345678 ' 처럼 공백이 들어가면 에러 나므로 .strip() 필수
        self.acc_no = str(settings["kis"]["account_number"]).strip()
        self.acc_code = str(settings["kis"]["product_code"]).strip()

        # 2. 토큰 파일 경로 설정
        self.mode = settings["mode"]
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.token_file_path = root_dir / f"token_{self.mode}.json"

        self.access_token = None
        self.token_expired_at = 0

        # 3. 시작하자마자 인증
        self._auth()

    # ===========================================================
    # [인증] 토큰 관리
    # ===========================================================
    def _auth(self):
        if self._load_token_from_file():
            return

        print(f"[INFO] [{self.mode}] 토큰 신규 발급 요청 중...")
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        res = requests.post(url, headers=headers, data=json.dumps(body))

        if res.status_code == 200:
            data = res.json()
            self.access_token = data["access_token"]
            self.token_expired_at = time.time() + int(data["expires_in"]) - 60
            print(f"[INFO] 토큰 발급 성공 (유효기간: {data['expires_in']}초)")
            self._save_token_to_file()
        else:
            raise Exception(f"토큰 발급 실패: {res.text}")

    def _save_token_to_file(self):
        data = {"access_token": self.access_token, "expired_at": self.token_expired_at}
        with open(self.token_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _load_token_from_file(self):
        if not os.path.exists(self.token_file_path):
            return False
        try:
            with open(self.token_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if time.time() < data.get("expired_at", 0):
                self.access_token = data.get("access_token")
                self.token_expired_at = data.get("expired_at")
                print(f"[INFO] [{self.mode}] 저장된 토큰 로드 성공 (유효함)")
                return True
            else:
                return False
        except:
            return False

    def get_header(self, tr_id):
        """API 요청용 헤더 생성"""
        if time.time() >= self.token_expired_at:
            self._auth()

        return {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    # ===========================================================
    # [기능] 잔고 조회 & 주문
    # ===========================================================
    def get_balance(self):
        """주식 잔고 조회 (헤더 문제 해결 버전)"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = "VTTC8434R" if self.mode == "MOCK" else "TTTC8434R"

        # [핵심 수정]
        # get_header로 받아온 기본 헤더만 사용하고,
        # tr_cont(연속조회 여부)는 아예 넣지 않습니다. (API가 알아서 판단하게 함)
        # 잘못된 값을 넣느니 안 넣는 게 낫습니다.
        headers = self.get_header(tr_id)

        params = {
            "CANO": self.acc_no,
            "ACNT_PRDT_CD": self.acc_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",  # 모의투자: 00 권장
            "CTX_AREA_FK100": "",  # 공백
            "CTX_AREA_NK100": "",  # 공백
        }

        res = requests.get(url, headers=headers, params=params)
        print(res.status_code)
        print(res.text)
        if res.status_code == 200:
            data = res.json()
            if data["rt_cd"] != "0":
                print(f"잔고 조회 실패: {data['msg1']} (Code: {data['msg_cd']})")
                return None
            deposit = int(data["output2"][0]["dnca_tot_amt"])
            print(f"예수금 조회 성공: {deposit:,}원")
            return {"deposit": deposit, "stocks": data["output1"]}
        else:
            print(f"통신 에러: {res.text}")
            return None

    def send_order(self, ticker, order_type, quantity, price=0):
        """주문 전송"""
        path = "/uapi/domestic-stock/v1/trading/order-cash"
        url = f"{self.base_url}{path}"

        if self.mode == "MOCK":
            tr_id = "VTTC0802U" if order_type == "BUY" else "VTTC0801U"
        else:
            tr_id = "TTTC0802U" if order_type == "BUY" else "TTTC0801U"

        headers = self.get_header(tr_id)
        ord_dvsn = "01" if price == 0 else "00"

        body = {
            "CANO": self.acc_no,
            "ACNT_PRDT_CD": self.acc_code,
            "PDNO": ticker,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
        }

        res = requests.post(url, headers=headers, data=json.dumps(body))

        if res.status_code == 200:
            data = res.json()
            if data["rt_cd"] == "0":
                print(
                    f"[{order_type}] 주문 전송 성공 (주문번호: {data['output']['ODNO']})"
                )
                return data["output"]["ODNO"]
            else:
                print(f"주문 거부됨: {data['msg1']}")
                return None
        else:
            print(f"주문 통신 에러: {res.text}")
            return None

    def get_quote(self, ticker):
        """현재가/시가 조회"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self.get_header("FHKST01010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": str(ticker),
        }

        res = requests.get(url, headers=headers, params=params)
        if res.status_code != 200:
            return None

        data = res.json()
        if data.get("rt_cd") != "0":
            return None

        output = data.get("output", {})

        def _to_float(value):
            text = str(value).replace(",", "").strip()
            if text == "":
                return None
            try:
                return float(text)
            except Exception:
                return None

        return {
            "ticker": str(ticker),
            "current_price": _to_float(output.get("stck_prpr")),
            "open_price": _to_float(output.get("stck_oprc")),
            "high_price": _to_float(output.get("stck_hgpr")),
            "low_price": _to_float(output.get("stck_lwpr")),
            "raw": output,
        }

    def get_open_price(self, ticker):
        quote = self.get_quote(ticker)
        if not quote:
            return None
        return quote.get("open_price")

    def _to_float(self, value):
        text = str(value).replace(",", "").strip()
        if text in {"", "-", "None"}:
            return None
        try:
            return float(text)
        except Exception:
            return None

    def _parse_ohlcv_rows(self, rows, date_key_candidates, field_map):
        parsed: list[dict] = []
        for row in rows:
            date_text = ""
            for key in date_key_candidates:
                value = str(row.get(key) or "").strip()
                if value:
                    date_text = value
                    break
            if len(date_text) != 8:
                continue

            try:
                dt = datetime.strptime(date_text, "%Y%m%d")
            except Exception:
                continue

            close_price = self._to_float(row.get(field_map["close"]))
            open_price = self._to_float(row.get(field_map["open"]))
            high_price = self._to_float(row.get(field_map["high"]))
            low_price = self._to_float(row.get(field_map["low"]))
            volume = self._to_float(row.get(field_map["volume"]) or 0.0)
            trading_value = self._to_float(row.get(field_map["trading_value"]) or 0.0)
            market_cap = self._to_float(row.get(field_map.get("market_cap", "") or ""))

            if close_price is None:
                continue

            parsed.append(
                {
                    "Date": dt,
                    "Open": open_price if open_price is not None else close_price,
                    "High": high_price if high_price is not None else close_price,
                    "Low": low_price if low_price is not None else close_price,
                    "Close": close_price,
                    "Volume": volume if volume is not None else 0.0,
                    "TradingValue": trading_value,
                    "MarketCap": market_cap,
                }
            )
        return parsed

    def get_index_daily_ohlcv(self, ticker, period_div="D"):
        index_code = self.INDEX_CODE_MAP.get(str(ticker).upper())
        if not index_code:
            return None

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-index-daily-price"
        headers = self.get_header("FHKUP03500100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "U",
            "FID_INPUT_ISCD": index_code,
            "FID_PERIOD_DIV_CODE": period_div,
        }

        try:
            res = requests.get(url, headers=headers, params=params, timeout=20)
            if res.status_code != 200:
                return None
            payload = res.json()
            if payload.get("rt_cd") != "0":
                return None

            rows = payload.get("output") or payload.get("output2") or []
            if not isinstance(rows, list) or len(rows) == 0:
                return None

            parsed = self._parse_ohlcv_rows(
                rows,
                date_key_candidates=["stck_bsop_date", "bstp_bsop_date", "bas_dt"],
                field_map={
                    "open": "bstp_nmix_oprc",
                    "high": "bstp_nmix_hgpr",
                    "low": "bstp_nmix_lwpr",
                    "close": "bstp_nmix_prpr",
                    "volume": "acml_vol",
                    "trading_value": "acml_tr_pbmn",
                },
            )
            if not parsed:
                return None

            return pd.DataFrame(parsed).sort_values("Date")
        except Exception:
            return None

    def get_daily_ohlcv(self, ticker, period_div="D", adjusted="1"):
        """종목 일봉 OHLCV 조회 (KIS API 직접)"""
        ticker_text = str(ticker).upper()

        if ticker_text in self.INDEX_CODE_MAP:
            index_df = self.get_index_daily_ohlcv(ticker_text, period_div=period_div)
            if index_df is not None and not index_df.empty:
                return index_df
            print(f"[WARN] 지수 {ticker_text} 조회 실패 (대체지수 미사용)")
            return None

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        headers = self.get_header("FHKST01010400")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": str(ticker),
            "FID_PERIOD_DIV_CODE": period_div,
            "FID_ORG_ADJ_PRC": str(adjusted),
        }

        try:
            res = requests.get(url, headers=headers, params=params, timeout=20)
        except Exception:
            return None

        if res.status_code != 200:
            return None

        payload = res.json()
        if payload.get("rt_cd") != "0":
            return None

        rows = payload.get("output") or payload.get("output2") or []
        if not isinstance(rows, list) or len(rows) == 0:
            return None

        parsed = self._parse_ohlcv_rows(
            rows,
            date_key_candidates=["stck_bsop_date", "bas_dt"],
            field_map={
                "open": "stck_oprc",
                "high": "stck_hgpr",
                "low": "stck_lwpr",
                "close": "stck_clpr",
                "volume": "acml_vol",
                "trading_value": "acml_tr_pbmn",
                "market_cap": "hts_avls",
            },
        )

        if not parsed:
            return None

        df = pd.DataFrame(parsed).sort_values("Date")
        return df

    def get_market_metrics(self, ticker):
        """시총/당일거래대금/현재가 조회"""
        quote = self.get_quote(ticker)
        if not quote:
            return None

        raw = quote.get("raw", {})
        current_price = quote.get("current_price")
        listed_shares = self._to_float(raw.get("lstn_stcn"))

        market_cap = self._to_float(
            raw.get("hts_avls")
            or raw.get("mrkt_tot_amt")
            or raw.get("stck_avls")
            or raw.get("lstn_stcn")
        )
        trading_value = self._to_float(
            raw.get("acml_tr_pbmn") or raw.get("acml_tr_pbmn1")
        )

        if market_cap is not None and market_cap < 1e10:
            market_cap = market_cap * 100_000_000.0

        if market_cap is None or trading_value is None:
            daily = self.get_daily_ohlcv(ticker)
            if daily is not None and not daily.empty:
                last = daily.iloc[-1]
                if market_cap is None:
                    mcap = self._to_float(last.get("MarketCap"))
                    if mcap is not None:
                        market_cap = mcap
                if trading_value is None:
                    tval = self._to_float(last.get("TradingValue"))
                    if tval is None:
                        close = self._to_float(last.get("Close"))
                        vol = self._to_float(last.get("Volume"))
                        if close is not None and vol is not None:
                            tval = close * vol
                    trading_value = tval

        return {
            "market_cap": market_cap,
            "listed_shares": listed_shares,
            "daily_trading_value": trading_value,
            "current_price": current_price,
        }
