import sys
import os

# [핵심] 프로젝트 루트 경로를 강제로 추가하는 코드
# 1. 현재 파일(test_order.py)의 위치
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 부모 폴더(tests)의 부모 폴더(kr_super_momentum) = 프로젝트 루트
project_root = os.path.dirname(current_dir)
# 3. 파이썬이 찾을 경로 리스트(sys.path)에 루트 추가
sys.path.append(project_root)

# -------------------------------------------------------
# 위 코드가 실행된 후에 import를 해야 에러가 안 납니다.
from src.api.utils.kis_client import KisClient
# -------------------------------------------------------

def main():
    print(f"--- ⚡ 한투 모의투자 연결 테스트 ---")
    print(f"📍 프로젝트 루트 경로 인식: {project_root}")
    
    # 1. 클라이언트 생성 (토큰 자동 로드)
    # 이 과정에서 token_MOCK.json을 읽거나 새로 만듭니다.
    client = KisClient()
    
    # 2. 내 잔고 확인 (예수금)
    balance = client.get_balance()
    if balance:
        # 예수금 천 단위 콤마 찍어서 출력
        print(f"💰 현재 예수금: {balance['deposit']:,}원")
        print(f"📦 보유 종목 수: {len(balance['stocks'])}개")

    # 3. 매수 주문 테스트 (삼성전자 1주 시장가)
    print("\n--- [주의] 장 운영 시간(09:00~15:30)에만 체결됩니다 ---")
    user_input = input("삼성전자(005930) 1주 시장가 매수 테스트를 하시겠습니까? (y/n): ")
    
    if user_input.strip().lower() == 'y':
        # 시장가(0원)로 1주 매수
        order_no = client.send_order("005930", "BUY", quantity=1, price=0)
        
        if order_no:
            print(f"\n✅ 테스트 성공! 주문번호: {order_no}")
            print("한투 MTS/HTS에서 체결 내역을 확인해보세요.")

if __name__ == "__main__":
    main()