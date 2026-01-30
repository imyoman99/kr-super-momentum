# scripts/test_token.py
import sys
import os

# src 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.utils.kis_client import KisClient

def main():
    print("--- 🔑 토큰 발급 테스트 시작 ---")
    
    try:
        # 객체를 만드는 순간 내부적으로 _auth()가 실행됨
        client = KisClient()
        
        print("\n[결과 확인]")
        if client.access_token:
            print(f"Token 값(앞 20자리): {client.access_token[:20]}...")
            print("🎉 테스트 성공! API 사용할 준비 완료.")
        else:
            print("❌ 토큰 발급 실패: access_token이 None입니다.")
            
    except Exception as e:
        print(f"💀 에러 발생: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()