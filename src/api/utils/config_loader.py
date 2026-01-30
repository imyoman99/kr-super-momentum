import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. .env 파일 로드 (환경변수 메모리에 등록)
load_dotenv()

# 경로 설정 (src/configs/kis_live.yaml로 고정)
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "kis_live.yaml"
)


def load_config():
    # 1. YAML 파일 읽기 (기본 설정 및 URL 정보)
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"설정 파일이 없습니다: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        yaml_conf = yaml.safe_load(f)

    # 2. 현재 모드 확인 (MOCK vs REAL)
    mode = yaml_conf.get("mode", "MOCK").upper()
    print(f"현재 실행 모드: {mode}")

    # 3. [핵심] 모드에 따라 .env에서 가져올 키 결정 & 통합 설정 생성
    # 우리는 프로그램 안에서 항상 settings['kis']['app_key']라고 부르고 싶으므로,
    # 여기서 매핑(Mapping)을 해줍니다.

    config = {
        "mode": mode,
        "kis": {},
        "paths": yaml_conf.get("paths", {}),  # 기존 경로 설정 유지
    }

    # 공통 정보 (.env 우선, 없으면 yaml 값)
    config["kis"]["hts_id"] = os.getenv("KIS_HTS_ID", yaml_conf.get("my_htsid"))
    config["kis"]["user_agent"] = os.getenv("KIS_USER_AGENT", yaml_conf.get("my_agent"))
    config["kis"]["product_code"] = yaml_conf.get("my_prod", "01")  # 계좌 뒷자리

    if mode == "MOCK":
        # --- 모의투자 세팅 ---
        config["kis"]["app_key"] = os.getenv("KIS_MOCK_APP_KEY")
        config["kis"]["app_secret"] = os.getenv("KIS_MOCK_APP_SECRET")
        config["kis"]["account_number"] = os.getenv("KIS_MOCK_ACC_STOCK")

        # URL (YAML에 적힌 vps, vops 사용)
        config["kis"]["base_url"] = yaml_conf.get("vps")
        config["kis"]["ws_url"] = yaml_conf.get("vops")

    else:
        # --- 실전투자 세팅 ---
        config["kis"]["app_key"] = os.getenv("KIS_REAL_APP_KEY")
        config["kis"]["app_secret"] = os.getenv("KIS_REAL_APP_SECRET")
        config["kis"]["account_number"] = os.getenv("KIS_REAL_ACC_STOCK")

        # URL (YAML에 적힌 prod, ops 사용)
        config["kis"]["base_url"] = yaml_conf.get("prod")
        config["kis"]["ws_url"] = yaml_conf.get("ops")

    # 필수값 체크 (누락되면 실행 불가)
    if not config["kis"]["app_key"] or not config["kis"]["app_secret"]:
        print("경고: .env 파일에서 APP KEY를 찾을 수 없습니다.")

    return config


# 전역 변수로 로드
settings = load_config()
