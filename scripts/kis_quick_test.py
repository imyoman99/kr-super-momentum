import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.utils.kis_client import KisClient


def parse_args():
    parser = argparse.ArgumentParser(description="KIS 토큰/잔고/주문 테스트 스크립트")
    parser.add_argument("--balance", action="store_true", help="잔고 조회")
    parser.add_argument("--buy", nargs=2, metavar=("TICKER", "QTY"), help="매수 주문")
    parser.add_argument("--sell", nargs=2, metavar=("TICKER", "QTY"), help="매도 주문")
    parser.add_argument(
        "--price", type=int, default=0, help="지정가 가격(기본: 0=시장가)"
    )
    parser.add_argument("--confirm", action="store_true", help="실주문 실행 확인")
    return parser.parse_args()


def main():
    args = parse_args()
    client = KisClient()

    if args.balance:
        client.get_balance()

    def place_order(order_type, ticker, qty):
        if not args.confirm:
            print(
                f"⚠️ 확인 필요: {order_type} {ticker} {qty}주 (price={args.price})\n"
                "--confirm 옵션을 주면 실주문 전송됩니다."
            )
            return
        client.send_order(ticker, order_type, qty, price=args.price)

    if args.buy:
        ticker, qty = args.buy
        place_order("BUY", ticker, int(qty))

    if args.sell:
        ticker, qty = args.sell
        place_order("SELL", ticker, int(qty))

    if not (args.balance or args.buy or args.sell):
        print("실행 옵션이 없습니다. --balance 또는 --buy/--sell을 사용하세요.")


if __name__ == "__main__":
    main()
