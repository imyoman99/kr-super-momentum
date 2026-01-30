import requests
import json
import time
import os
from pathlib import Path
from src.api.utils.config_loader import settings


class KisClient:
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

        print(f"🔄 [{self.mode}] 토큰 신규 발급 요청 중...")
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
            print(f"✅ 토큰 발급 성공! (유효기간: {data['expires_in']}초)")
            self._save_token_to_file()
        else:
            raise Exception(f"❌ 토큰 발급 실패: {res.text}")

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
                print(f"📂 [{self.mode}] 저장된 토큰 로드 성공 (유효함)")
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
                print(f"❌ 잔고 조회 실패: {data['msg1']} (Code: {data['msg_cd']})")
                return None
            deposit = int(data["output2"][0]["dnca_tot_amt"])
            print(f"💰 예수금 조회 성공: {deposit:,}원")
            return {"deposit": deposit, "stocks": data["output1"]}
        else:
            print(f"❌ 통신 에러: {res.text}")
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
                    f"🚀 [{order_type}] 주문 전송 성공! (주문번호: {data['output']['ODNO']})"
                )
                return data["output"]["ODNO"]
            else:
                print(f"❌ 주문 거부됨: {data['msg1']}")
                return None
        else:
            print(f"❌ 주문 통신 에러: {res.text}")
            return None
